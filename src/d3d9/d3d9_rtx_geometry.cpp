#include <vector>
#include "d3d9_device.h"
#include "d3d9_rtx.h"
#include "d3d9_rtx_utils.h"
#include "d3d9_state.h"
#include "../dxvk/dxvk_buffer.h"
#include "../dxvk/rtx_render/rtx_hashing.h"
#include "../util/util_fastops.h"

namespace dxvk {
  // Geometry indices should never be signed.  Using this to handle the non-indexed case for templates.
  typedef int NoIndices;

  namespace VertexRegions {
    enum Type : uint32_t {
      Position = 0,
      Texcoord,
      Count
    };
  }

  // NOTE: Intentionally leaving the legacy hashes out of here, because they are special (REMIX-656)
  // (Was a std::map looked up twice per component per draw on the geometry workers.)
  inline bool componentToRegion(const HashComponents component, VertexRegions::Type& regionOut) {
    switch (component) {
    case HashComponents::VertexPosition: regionOut = VertexRegions::Position; return true;
    case HashComponents::VertexTexcoord: regionOut = VertexRegions::Texcoord; return true;
    default: return false;
    }
  }

  bool getVertexRegion(const RasterBuffer& buffer, const size_t vertexCount, HashQuery& outResult) {
    ScopedCpuProfileZone();

    if (!buffer.defined())
      return false;

    outResult.pBase = (uint8_t*) buffer.mapPtr(buffer.offsetFromSlice());
    outResult.elementSize = imageFormatInfo(buffer.vertexFormat())->elementSize;
    outResult.stride = buffer.stride();
    outResult.size = outResult.stride * vertexCount;
    // Make sure we hold on to this reference while the hashing is in flight
    outResult.ref = buffer.buffer().ptr();
    assert(outResult.ref);
    return true;
  }

  // Sorts and deduplicates a set of integers, storing the result in a vector
  template<typename T>
  void deduplicateSortIndices(const void* pIndexData, const size_t indexCount, const uint32_t maxIndexValue, std::vector<T>& uniqueIndicesOut) {
    // TODO (REMIX-657): Implement optimized variant of this function
    // We know there will be at most, this many unique indices
    const uint32_t indexRange = maxIndexValue + 1;

    // Initialize all to 0
    uniqueIndicesOut.resize(indexRange, (T)0);

    // Use memory as a bin table for index data
    for (uint32_t i = 0; i < indexCount; i++) {
      const T& index = ((T*) pIndexData)[i];
      assert(index <= maxIndexValue);
      uniqueIndicesOut[index] = 1;
    }

    // Repopulate the bins with contiguous index values
    uint32_t uniqueIndexCount = 0;
    for (uint32_t i = 0; i < indexRange; i++) {
      if (uniqueIndicesOut[i])
        uniqueIndicesOut[uniqueIndexCount++] = i;
    }

    // Remove any unused entries
    uniqueIndicesOut.resize(uniqueIndexCount);
  }

  template<typename T>
  void hashGeometryData(const size_t indexCount, const uint32_t maxIndexValue, const void* pIndexData,
                        DxvkBuffer* indexBufferRef, const HashQuery vertexRegions[VertexRegions::Count], GeometryHashes& hashesOut) {
    ScopedCpuProfileZone();

    const HashRule& globalHashRule = RtxOptions::geometryHashGenerationRule();

    // REMIX-658: one vector per geometry worker thread, reused across draws (was a heap allocation
    // sized maxIndex + 1 per draw). clear() + resize(n, 0) zero-fills exactly as before.
    thread_local std::vector<T> tlsUniqueIndices;
    std::vector<T>& uniqueIndices = tlsUniqueIndices;
    uniqueIndices.clear();
    if constexpr (!std::is_same<T, NoIndices>::value) {
      assert((indexCount > 0 && indexBufferRef));
      deduplicateSortIndices(pIndexData, indexCount, maxIndexValue, uniqueIndices);

      if (globalHashRule.test(HashComponents::Indices)) {
        hashesOut[HashComponents::Indices] = hashContiguousMemory(pIndexData, indexCount * sizeof(T));
      }

      // TODO (REMIX-656): Remove this once we can transition content to new hash
      if (globalHashRule.test(HashComponents::LegacyIndices)) {
        hashesOut[HashComponents::LegacyIndices] = hashIndicesLegacy<T>(pIndexData, indexCount);
      }

      // Release this memory back to the staging allocator
      indexBufferRef->release(DxvkAccess::Read);
      indexBufferRef->decRef();
    }

    // Do vertex based rules
    for (uint32_t i = 0; i < (uint32_t) HashComponents::Count; i++) {
      const HashComponents& component = (HashComponents) i;

      VertexRegions::Type region = VertexRegions::Position;
      if (globalHashRule.test(component) && componentToRegion(component, region)) {
        hashesOut[component] = hashVertexRegionIndexed(vertexRegions[(uint32_t)region], uniqueIndices);
      }
    }

    // Don't keep a huge bin table alive on the worker after an unusually large draw.
    if (uniqueIndices.capacity() > (size_t(1) << 22)) {
      std::vector<T>().swap(uniqueIndices);
    }

    // TODO (REMIX-656): Remove this once we can transition content to new hash
    if (globalHashRule.test(HashComponents::LegacyPositions0) || globalHashRule.test(HashComponents::LegacyPositions1)) {
      hashRegionLegacy(vertexRegions[VertexRegions::Position], hashesOut[HashComponents::LegacyPositions0], hashesOut[HashComponents::LegacyPositions1]);
    }

    // Release this memory back to the staging allocator
    for (uint32_t i = 0; i < VertexRegions::Count; i++) {
      const HashQuery& region = vertexRegions[i];
      if (region.size == 0)
        continue;

      if (region.ref) {
        region.ref->release(DxvkAccess::Read);
        region.ref->decRef();
      }
    }
  }

  // A draw call is recognised across frames by its shader, the buffer ranges it reads and where its
  // world transform places it. Instances of one mesh share the shader and buffers, and a game is free
  // to submit them in a different order every frame, so their placement is what tells them apart;
  // the submission order only separates instances that share a placement too. The placement is
  // coarsened so that a drifting transform keeps its identity.
  XXH64_hash_t D3D9Rtx::vertexShaderDrawKey(const RasterGeometry& geoData) {
    constexpr float kPlacementCellsPerUnit = 4.f;
    const Matrix4& world = d3d9State().transforms[GetTransformIndex(D3DTS_WORLD)];
    struct DrawIdentity {
      const void* shader;
      const void* vertexBuffer;
      size_t vertexOffset;
      const void* indexBuffer;
      size_t indexOffset;
      uint32_t vertexCount;
      uint32_t indexCount;
      int32_t placement[3];
    } identity {
      d3d9State().vertexShader->GetCommonShader(),
      geoData.positionBuffer.buffer().ptr(), geoData.positionBuffer.offset() + geoData.positionBuffer.offsetFromSlice(),
      geoData.indexBuffer.buffer().ptr(), geoData.indexBuffer.offset(),
      geoData.vertexCount, geoData.indexCount,
      { static_cast<int32_t>(std::floor(world[3][0] * kPlacementCellsPerUnit)),
        static_cast<int32_t>(std::floor(world[3][1] * kPlacementCellsPerUnit)),
        static_cast<int32_t>(std::floor(world[3][2] * kPlacementCellsPerUnit)) }
    };
    const XXH64_hash_t identityHash = XXH3_64bits(&identity, sizeof(identity));
    const uint32_t occurrence = m_vsDrawOccurrencesThisFrame[identityHash]++;
    return XXH3_64bits_withSeed(&occurrence, sizeof(occurrence), identityHash);
  }

  // Compares a draw call's constants with its own on the previous frame.
  void D3D9Rtx::noteVertexShaderConstants(XXH64_hash_t key, uint32_t usedConstants) {
    const uint32_t count = std::min(usedConstants, caps::MaxFloatConstantsVS);
    const Vector4* current = &d3d9State().vsConsts.fConsts[0];

    // Bounded: ring-buffered geometry gets a new identity every frame and would otherwise accumulate.
    constexpr size_t kMaxTrackedDraws = 16384;
    if (m_previousVsConstants.size() >= kMaxTrackedDraws) {
      m_previousVsConstants.clear();
    }

    auto [entry, inserted] = m_previousVsConstants.try_emplace(key);
    std::vector<Vector4>& previous = entry->second;
    if (!inserted && previous.size() == count) {
      ++m_vsConstantDrawsCompared;

      // A constant spanning several registers (a matrix, an array) counts once per draw. Registers
      // the constant table does not name are reported by register.
      const std::vector<std::string>& names = d3d9State().vertexShader->GetCommonShader()->GetFloatConstantNames();
      std::unordered_map<std::string, bool> changedThisDraw;
      for (uint32_t i = 0; i < count; ++i) {
        const std::string name = i < names.size() && !names[i].empty() ? names[i] : str::format("c", i);
        bool& changed = changedThisDraw[name];
        const bool registerChanged = std::memcmp(&previous[i], &current[i], sizeof(Vector4)) != 0;
        changed = changed || registerChanged;
        if (registerChanged && !logChangingVertexShaderConstantsTextures().empty()) {
          std::string& sample = m_vsConstantChangeSamples[name];
          if (sample.empty()) {
            sample = str::format("c", i, " ", previous[i], " -> ", current[i]);
          }
        }
      }
      for (const auto& [name, changed] : changedThisDraw) {
        ConstantChangeCount& counts = m_vsConstantChanges[name];
        ++counts.draws;
        counts.changed += changed ? 1 : 0;
      }
    }
    previous.assign(current, current + count);
  }

  void D3D9Rtx::logVertexShaderConstantChanges() {
    constexpr uint32_t kSummaryFrames = 600;
    if (m_d3d9FrameIndex % kSummaryFrames != 0 || m_vsConstantDrawsCompared == 0) {
      return;
    }

    std::vector<std::pair<std::string, ConstantChangeCount>> changing;
    for (const auto& entry : m_vsConstantChanges) {
      if (entry.second.changed * 100 >= entry.second.draws) {
        changing.push_back(entry);
      }
    }
    std::sort(changing.begin(), changing.end(), [](const auto& a, const auto& b) { return a.second.changed > b.second.changed; });

    std::string summary = str::format("[RTX VS constants] ", m_vsConstantDrawsCompared,
                                      " draw comparisons; constants changed since the draw's previous frame (changed/read):");
    for (const auto& [name, counts] : changing) {
      summary += str::format(" ", name, "=", counts.changed, "/", counts.draws);
      const auto sample = m_vsConstantChangeSamples.find(name);
      if (sample != m_vsConstantChangeSamples.end()) {
        summary += str::format(" [", sample->second, "]");
      }
    }
    Logger::info(summary);

    m_vsConstantChanges.clear();
    m_vsConstantChangeSamples.clear();
    m_vsConstantDrawsCompared = 0;
  }

  // rtx.vertexCaptureConstantTolerance. A shader-captured draw's geometry hash covers the float
  // constants its shader reads, so a mesh whose constants creep by a hair each frame -- a skinned
  // prop whose bones wobble by a millimetre -- is re-captured, re-interleaved and has its BLAS
  // rebuilt every frame for no visible change. While every hashed constant stays within the
  // tolerance of the values the draw's current hash was taken from, that hash is kept. The
  // comparison is against those values, not the previous frame's, so a slow drift still adds up
  // to a new capture instead of being ignored forever. `watched` is the draw's geometry when its
  // albedo is listed in rtx.logChangingVertexShaderConstantsTextures, and makes a fresh hash explain itself.
  XXH64_hash_t D3D9Rtx::settleVertexShaderConstants(XXH64_hash_t key, const D3D9CommonShader& shader,
                                                    uint32_t usedConstants, XXH64_hash_t hash, const RasterGeometry* watched) {
    const float tolerance = vertexCaptureConstantTolerance();
    const Vector4* current = &d3d9State().vsConsts.fConsts[0];
    const uint32_t count = std::min(usedConstants, caps::MaxFloatConstantsVS);

    // The first hashed register beyond the tolerance, or `count` when every one is within it. Only
    // the registers the hash covers: an ignored constant changes every frame by design.
    const auto firstRegisterBeyondTolerance = [&](const std::vector<Vector4>& settled) {
      for (const auto& [begin, end] : shader.GetHashedFloatConstants()) {
        for (uint32_t i = begin; i < std::min(end, count); ++i) {
          for (uint32_t c = 0; c < 4; ++c) {
            // Negated so a NaN on either side counts as a change
            if (!(std::abs(settled[i][c] - current[i][c]) <= tolerance)) {
              return i;
            }
          }
        }
      }
      return count;
    };

    constexpr uint32_t kMaxExplainedPerSummary = 60;
    const auto explain = [&](const std::string& reason) {
      if (watched == nullptr || m_settledVsConstantsExplained >= kMaxExplainedPerSummary) {
        return;
      }
      ++m_settledVsConstantsExplained;
      const Matrix4& world = d3d9State().transforms[GetTransformIndex(D3DTS_WORLD)];
      Logger::info(str::format("[RTX VS constants] frame ", m_d3d9FrameIndex, " draw ", std::hex, key, std::dec, " (",
                               watched->vertexCount, " vertices, ", watched->indexCount, " indices, world ", world[3],
                               ") hash taken fresh: ", reason));
    };

    const auto previous = m_settledVsConstantsLastFrame.find(key);
    if (previous == m_settledVsConstantsLastFrame.end()) {
      explain("no draw with this shader, buffers and placement last frame");
    } else if (previous->second.values.size() != count) {
      explain("the shader's constant count changed");
    } else {
      const uint32_t beyond = firstRegisterBeyondTolerance(previous->second.values);
      if (beyond == count) {
        ++m_settledVsConstantsKept;
        return m_settledVsConstantsThisFrame.insert_or_assign(key, std::move(previous->second)).first->second.hash;
      }
      const std::vector<std::string>& names = shader.GetFloatConstantNames();
      explain(str::format(beyond < names.size() && !names[beyond].empty() ? names[beyond] : std::string("unnamed"), " c", beyond,
                          " ", previous->second.values[beyond], " -> ", current[beyond]));
    }
    ++m_settledVsConstantsTaken;
    m_settledVsConstantsThisFrame.insert_or_assign(key, SettledVsConstants { std::vector<Vector4>(current, current + count), hash });
    return hash;
  }

  void D3D9Rtx::logSettledVertexShaderConstants() {
    constexpr uint32_t kSummaryFrames = 600;
    if (m_d3d9FrameIndex % kSummaryFrames != 0 || m_settledVsConstantsKept + m_settledVsConstantsTaken == 0) {
      return;
    }
    Logger::info(str::format("[RTX VS constants] tolerance ", vertexCaptureConstantTolerance(), " over ", kSummaryFrames,
                             " frames: geometry hash kept for ", m_settledVsConstantsKept, " shader-captured draws, taken fresh for ",
                             m_settledVsConstantsTaken));
    m_settledVsConstantsKept = 0;
    m_settledVsConstantsTaken = 0;
    m_settledVsConstantsExplained = 0;
  }

  Future<GeometryHashes> D3D9Rtx::computeHash(RasterGeometry& geoData, const uint32_t maxIndexValue, const XXH64_hash_t memoizationKey) {
    ScopedCpuProfileZone();

    // rtx.geometryHashMemoSelfCheckFrames: on a self-check frame a memo hit is re-hashed in full, and
    // the worker compares the result with the memoized entry before replacing it.
    bool selfCheck = false;

    if (memoizationKey != 0) {
      std::lock_guard<dxvk::mutex> lock(m_geometryHashCacheMutex);
      auto it = m_geometryHashCache.find(memoizationKey);
      if (it != m_geometryHashCache.end()) {
        const uint32_t selfCheckFrames = geometryHashMemoSelfCheckFrames();
        selfCheck = selfCheckFrames != 0 && (m_d3d9FrameIndex % selfCheckFrames) == 0;
        if (!selfCheck) {
          ++m_geometryHashCacheHits;

          if (geometryHashMemoInline()) {
            // Served inline: no worker round trip. finalizeGeometryHashes() takes `hashes` as is.
            geoData.hashes = it->second;
            geoData.hashesPrecomputed = true;
            return Future<GeometryHashes>();
          }

          // Future is single-consumption - get() clears its task pointer - so a stored Future cannot
          // be handed out twice. Schedule a task that simply returns the memoized value instead: the
          // dispatch is kept, but the buffer acquisition and the content hashing are both skipped.
          const GeometryHashes cachedHashes = it->second;
          return m_pGeometryWorkers->Schedule([cachedHashes]() -> GeometryHashes {
            return cachedHashes;
          });
        }
      } else {
        ++m_geometryHashCacheMisses;
      }
    }

    const uint32_t indexCount = geoData.indexCount;
    const uint32_t vertexCount = geoData.vertexCount;

    HashQuery vertexRegions[VertexRegions::Count];
    memset(&vertexRegions[0], 0, sizeof(vertexRegions));

    if (!getVertexRegion(geoData.positionBuffer, vertexCount, vertexRegions[VertexRegions::Position]))
      return Future<GeometryHashes>(); //invalid

    // Acquire prevents the staging allocator from re-using this memory
    vertexRegions[VertexRegions::Position].ref->acquire(DxvkAccess::Read);
    vertexRegions[VertexRegions::Position].ref->incRef();

    if (getVertexRegion(geoData.texcoordBuffer, vertexCount, vertexRegions[VertexRegions::Texcoord])) {
      vertexRegions[VertexRegions::Texcoord].ref->acquire(DxvkAccess::Read);
      vertexRegions[VertexRegions::Texcoord].ref->incRef();
    }

    // Make sure we hold a ref to the index buffer while hashing.
    const Rc<DxvkBuffer> indexBufferRef = geoData.indexBuffer.buffer();
    if (indexBufferRef.ptr()) {
      indexBufferRef->acquire(DxvkAccess::Read);
      indexBufferRef->incRef();
    }
    const void* pIndexData = geoData.indexBuffer.defined() ? geoData.indexBuffer.mapPtr(0) : nullptr;
    const size_t indexStride = geoData.indexBuffer.stride();
    const size_t indexDataSize = indexCount * indexStride;

    // Assume the GPU changed the data via shaders, include the constant buffer data in hash
    XXH64_hash_t vertexShaderHash = kEmptyHash;
    if (m_parent->UseProgrammableVS() && useVertexCapture()) {
      if (RtxOptions::geometryHashGenerationRule().test(HashComponents::GeometryDescriptor)) {
        const D3D9ConstantSets& cb = m_parent->m_consts[DxsoProgramTypes::VertexShader];
        const D3D9CommonShader* pVertexShader = d3d9State().vertexShader->GetCommonShader();
        if (cacheShaderBytecodeHash()) {
          // Hashed once when the shader was created: same value, no per-draw pass over the bytecode.
          vertexShaderHash = pVertexShader->GetBytecodeHash();
        } else {
          auto& shaderByteCode = pVertexShader->GetBytecode();
          vertexShaderHash = XXH3_64bits(shaderByteCode.data(), shaderByteCode.size());
        }
        // Hashed as the shader's hashed register runs, clamped to the ones it reads. With nothing
        // left out this is the single pass over every register, bit for bit.
        const uint32_t usedConstants = cb.meta.maxConstIndexF;
        for (const auto& [begin, end] : pVertexShader->GetHashedFloatConstants()) {
          if (begin >= usedConstants) {
            break;
          }
          vertexShaderHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.fConsts[begin], (std::min(end, usedConstants) - begin) * sizeof(float) * 4, vertexShaderHash);
        }

        const bool watched = !logChangingVertexShaderConstantsTextures().empty() &&
          lookupHash(logChangingVertexShaderConstantsTextures(), m_activeDrawCallState.materialData.getColorTexture().getImageHash());
        const bool noteConstants = logChangingVertexShaderConstants() &&
          (logChangingVertexShaderConstantsTextures().empty() || watched);
        const bool settleConstants = vertexCaptureConstantTolerance() > 0.f;
        if (noteConstants || settleConstants) {
          const XXH64_hash_t drawKey = vertexShaderDrawKey(geoData);
          if (noteConstants) {
            noteVertexShaderConstants(drawKey, usedConstants);
          }
          if (settleConstants) {
            vertexShaderHash = settleVertexShaderConstants(drawKey, *pVertexShader, usedConstants, vertexShaderHash,
                                                           watched ? &geoData : nullptr);
          }
        }
        vertexShaderHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.iConsts[0], cb.meta.maxConstIndexI * sizeof(int) * 4, vertexShaderHash);
        vertexShaderHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.bConsts[0], cb.meta.maxConstIndexB * sizeof(uint32_t)/32, vertexShaderHash);
      }
    }

    // Calculate this based on the RasterGeometry input data
    XXH64_hash_t geometryDescriptorHash = kEmptyHash;
    if (RtxOptions::geometryHashGenerationRule().test(HashComponents::GeometryDescriptor)) {
      geometryDescriptorHash = hashGeometryDescriptor(geoData.indexCount, 
                                                      geoData.vertexCount, 
                                                      geoData.indexBuffer.indexType(), 
                                                      geoData.topology);
    }

    // Calculate this based on the RasterGeometry input data
    XXH64_hash_t vertexLayoutHash = kEmptyHash;
    if (RtxOptions::geometryHashGenerationRule().test(HashComponents::VertexLayout)) {
      vertexLayoutHash = hashVertexLayout(geoData);
    }

    return m_pGeometryWorkers->Schedule([this, memoizationKey, vertexRegions, indexBufferRef = indexBufferRef.ptr(),
                                 pIndexData, indexStride, indexDataSize, indexCount,
                                 maxIndexValue, vertexShaderHash, geometryDescriptorHash,
                                 vertexLayoutHash, selfCheck]() -> GeometryHashes {
      ScopedCpuProfileZone();

      GeometryHashes hashes;

      // Finalize the descriptor hash
      hashes[HashComponents::GeometryDescriptor] = geometryDescriptorHash;
      hashes[HashComponents::VertexLayout] = vertexLayoutHash;
      hashes[HashComponents::VertexShader] = vertexShaderHash;

      // Index hash
      switch (indexStride) {
      case 2:
        hashGeometryData<uint16_t>(indexCount, maxIndexValue, pIndexData, indexBufferRef, vertexRegions, hashes);
        break;
      case 4:
        hashGeometryData<uint32_t>(indexCount, maxIndexValue, pIndexData, indexBufferRef, vertexRegions, hashes);
        break;
      default:
        hashGeometryData<NoIndices>(indexCount, maxIndexValue, pIndexData, indexBufferRef, vertexRegions, hashes);
        break;
      }

      assert(hashes[HashComponents::VertexPosition] != kEmptyHash);

      hashes.precombine();

      if (memoizationKey != 0) {
        std::lock_guard<dxvk::mutex> lock(m_geometryHashCacheMutex);
        // Bounded so a scene that streams unique geometry indefinitely cannot grow this without
        // limit. Clearing wholesale is acceptable because a miss only costs the hash we were
        // computing anyway.
        if (selfCheck) {
          // The fresh hash must equal the memoized one; a mismatch means a buffer write that did not
          // bump remixContentVersion (the memo would have served a stale hash). The entry is replaced below.
          // Compare the hash components only: GeometryHashes::precombined[3/4] are left uninitialized when
          // the legacy components are empty, so a whole-struct memcmp reports false mismatches.
          auto it = m_geometryHashCache.find(memoizationKey);
          if (it != m_geometryHashCache.end()) {
            std::string differing;
            for (uint32_t i = 0; i < (uint32_t) HashComponents::Count; i++) {
              const HashComponents component = (HashComponents) i;
              if (it->second[component] != hashes[component]) {
                differing += str::format(differing.empty() ? "" : ", ", getHashComponentName(component), " ",
                                         std::hex, it->second[component], " -> ", hashes[component]);
              }
            }
            if (!differing.empty()) {
              static std::atomic<uint32_t> s_mismatchLogs { 0 };
              if (s_mismatchLogs.fetch_add(1, std::memory_order_relaxed) < 20) {
                Logger::warn(str::format("[GeometryHashMemoCheck] memoized geometry hash is stale: key ", std::hex, memoizationKey,
                                         " | ", differing, " (first 20 mismatches are logged)"));
              }
            }
          }
        }
        if (m_geometryHashCache.size() >= kMaxGeometryHashCacheEntries) {
          m_geometryHashCache.clear();
        }
        m_geometryHashCache[memoizationKey] = hashes;
      }

      return hashes;
    });
  }

  Future<AxisAlignedBoundingBox> D3D9Rtx::computeAxisAlignedBoundingBox(const RasterGeometry& geoData) {
    ScopedCpuProfileZone();

    if (!RtxOptions::needsMeshBoundingBox()) {
      return Future<AxisAlignedBoundingBox>();
    }

    const void* pVertexData = geoData.positionBuffer.mapPtr((size_t)geoData.positionBuffer.offsetFromSlice());
    const uint32_t vertexCount = geoData.vertexCount;
    const size_t vertexStride = geoData.positionBuffer.stride();

    if (pVertexData == nullptr) {
      return Future<AxisAlignedBoundingBox>();
    }

    auto vertexBuffer = geoData.positionBuffer.buffer().ptr();
    vertexBuffer->incRef();

    return m_pGeometryWorkers->Schedule([pVertexData, vertexCount, vertexStride, vertexBuffer]()->AxisAlignedBoundingBox {
      ScopedCpuProfileZone();

#if defined(_M_ARM64) || defined(_M_ARM64EC)
      float32x4_t minPos = vdupq_n_f32(FLT_MAX);
      float32x4_t maxPos = vdupq_n_f32(-FLT_MAX);

      const uint8_t* pVertex = static_cast<const uint8_t*>(pVertexData);
      for (uint32_t vertexIdx = 0; vertexIdx < vertexCount; ++vertexIdx) {
        const Vector3* const pVertexPos = reinterpret_cast<const Vector3* const>(pVertex);
        float32x4_t vertexPos;
        vertexPos.n128_f32[0] = pVertexPos->x;
        vertexPos.n128_f32[1] = pVertexPos->y;
        vertexPos.n128_f32[2] = pVertexPos->z;

        minPos = vminq_f32(minPos, vertexPos);
        maxPos = vmaxq_f32(maxPos, vertexPos);

        pVertex += vertexStride;
      }

      AxisAlignedBoundingBox boundingBox {
        Vector3{ vgetq_lane_f32(minPos, 0), vgetq_lane_f32(minPos, 1), vgetq_lane_f32(minPos, 2) },
        Vector3{ vgetq_lane_f32(maxPos, 0), vgetq_lane_f32(maxPos, 1), vgetq_lane_f32(maxPos, 2) }
      };
#else
      __m128 minPos = _mm_set_ps1(FLT_MAX);
      __m128 maxPos = _mm_set_ps1(-FLT_MAX);

      const uint8_t* pVertex = static_cast<const uint8_t*>(pVertexData);
      for (uint32_t vertexIdx = 0; vertexIdx < vertexCount; ++vertexIdx) {
        const Vector3* const pVertexPos = reinterpret_cast<const Vector3* const>(pVertex);
        __m128 vertexPos = _mm_set_ps(0.0f, pVertexPos->z, pVertexPos->y, pVertexPos->x);
        minPos = _mm_min_ps(minPos, vertexPos);
        maxPos = _mm_max_ps(maxPos, vertexPos);

        pVertex += vertexStride;
      }

      AxisAlignedBoundingBox boundingBox{
        Vector3{ minPos.m128_f32[0], minPos.m128_f32[1], minPos.m128_f32[2] },
        Vector3{ maxPos.m128_f32[0], maxPos.m128_f32[1], maxPos.m128_f32[2] }
      };
#endif

      vertexBuffer->decRef();

      return boundingBox;
    });
  }
}
