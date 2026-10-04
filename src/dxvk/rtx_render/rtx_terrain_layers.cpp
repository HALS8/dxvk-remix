#include "rtx_terrain_layers.h"

#include <algorithm>
#include <cmath>

#include "rtx_asset_replacer.h"
#include "rtx_context.h"
#include "rtx_texture.h"
#include "rtx_scene_manager.h"
#include "rtx_imgui.h"

namespace dxvk {

  namespace {
    constexpr uint32_t kFramesBetweenLogs = 600;
    constexpr uint32_t kTableHeight = 256;
    // A footprint is shrunk by this fraction of a cell before it is put in the grid, so chunks
    // that tile exactly do not spill into their neighbours' cells.
    constexpr float kFootprintInset = 1e-3f;

    XXH64_hash_t transformKey(const Matrix4& objectToWorld) {
      return XXH3_64bits(&objectToWorld, sizeof(objectToWorld));
    }

    // The world axis a direction mostly points along.
    uint32_t dominantAxis(const Vector4& direction) {
      uint32_t axis = 0;
      for (uint32_t i = 1; i < 3; i++) {
        if (std::abs(direction.data[i]) > std::abs(direction.data[axis])) {
          axis = i;
        }
      }
      return axis;
    }

    // The extent of a chunk's mask square on two world axes.
    struct Footprint {
      float min[2];
      float max[2];
    };

    Footprint footprintOf(const TerrainLayers::Chunk& chunk, const uint32_t axes[2]) {
      const float extent = 1.f / chunk.layers.front().draw.maskScale;
      const Matrix4& m = chunk.objectToWorld;

      Footprint footprint;
      for (uint32_t k = 0; k < 2; k++) {
        const float origin = m[3].data[axes[k]];
        const float alongX = m[0].data[axes[k]] * extent;
        const float alongZ = m[2].data[axes[k]] * extent;
        footprint.min[k] = origin + std::min(0.f, alongX) + std::min(0.f, alongZ);
        footprint.max[k] = origin + std::max(0.f, alongX) + std::max(0.f, alongZ);
      }
      return footprint;
    }
  }

  TerrainLayers::Chunk* TerrainLayers::ChunkSet::findOrAdd(const Matrix4& objectToWorld, size_t maxChunks) {
    const XXH64_hash_t key = transformKey(objectToWorld);

    const auto found = indexByTransform.find(key);
    if (found != indexByTransform.end()) {
      return &chunks[found->second];
    }

    if (chunks.size() == maxChunks) {
      return nullptr;
    }

    indexByTransform.emplace(key, chunks.size());
    Chunk& chunk = chunks.emplace_back();
    chunk.objectToWorld = objectToWorld;
    return &chunk;
  }

  void TerrainLayers::ChunkSet::clear() {
    chunks.clear();
    indexByTransform.clear();
    layerCount = 0;
  }

  void TerrainLayers::trackLayerTextures(SceneManager& sceneManager, Layer& layer) {
    const auto track = [&sceneManager](const TextureRef& texture, uint32_t& index) {
      if (texture.isValid()) {
        sceneManager.trackTexture(texture, index, true);
      }
    };

    track(layer.draw.maskTexture, layer.maskTextureIndex);
    layer.colorSamplerIndex = sceneManager.trackSampler(layer.draw.colorSampler);

    const auto samplerLinearises = [](const TextureRef& texture) {
      const DxvkImageView* view = texture.getImageView();
      return RtxOptions::linearizeSrgbTextures() && view != nullptr && TextureUtils::isSRGB(view->info().format);
    };

    layer.colorIsLinear = samplerLinearises(layer.draw.colorTexture);
    layer.roughnessConstant = LegacyMaterialDefaults::roughnessConstant();
    layer.metallicConstant = LegacyMaterialDefaults::metallicConstant();

    if (layer.replacement == nullptr) {
      track(layer.draw.colorTexture, layer.colorTextureIndex);
      return;
    }

    OpaqueMaterialData& material = layer.replacement->getOpaqueMaterialData();
    layer.roughnessConstant = material.getRoughnessConstant();
    layer.metallicConstant = material.getMetallicConstant();
    track(material.getAlbedoOpacityTexture(), layer.colorTextureIndex);
    if (layer.colorTextureIndex != kSurfaceMaterialInvalidTextureIndex) {
      layer.colorIsLinear = samplerLinearises(material.getAlbedoOpacityTexture());
    }
    track(material.getNormalTexture(), layer.normalTextureIndex);
    track(material.getRoughnessTexture(), layer.roughnessTextureIndex);
    track(material.getMetallicTexture(), layer.metallicTextureIndex);
    track(material.getEmissiveColorTexture(), layer.emissiveTextureIndex);
    track(material.getHeightTexture(), layer.heightTextureIndex);

    // A replacement without an albedo texture keeps the game's.
    if (layer.colorTextureIndex == kSurfaceMaterialInvalidTextureIndex) {
      track(layer.draw.colorTexture, layer.colorTextureIndex);
    }
  }

  void TerrainLayers::resolveLayer(SceneManager& sceneManager, Layer& layer) {
    // A legacy material is identified by its colour texture.
    std::shared_ptr<MaterialData> replacement =
      sceneManager.getAssetReplacer()->getReplacementMaterial(layer.draw.colorTexture.getImageHash());
    layer.replacement = replacement != nullptr && replacement->getType() == MaterialDataType::Opaque
      ? std::move(replacement) : nullptr;

    // Texture indices are the frame's, so a layer kept from an earlier frame starts over.
    layer.colorTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    layer.maskTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    layer.normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    layer.roughnessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    layer.metallicTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    layer.emissiveTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    layer.heightTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    trackLayerTextures(sceneManager, layer);
  }

  void TerrainLayers::countLayer(const Chunk& chunk, const Layer& layer) {
    m_statistics.maxLayersInChunk = std::max(m_statistics.maxLayersInChunk, static_cast<uint32_t>(chunk.layers.size()));
    m_statistics.maskedLayers += layer.draw.maskTexture.isValid();
    m_statistics.replacedLayers += layer.replacement != nullptr;
    m_statistics.sideProjectedLayers += layer.draw.projection != TerrainLayerDraw::Projection::XZ;
  }

  void TerrainLayers::addLayerDraw(RtxContext& ctx, const DrawCallState& drawCallState) {
    if (drawCallState.getTerrainLayer().enabled) {
      addLayer(ctx, drawCallState.getTerrainLayer(), drawCallState.getTransformData().objectToWorld);
    }
  }

  void TerrainLayers::addLayer(RtxContext& ctx, const TerrainLayerDraw& layerDraw, const Matrix4& objectToWorld) {
    if (!enable()) {
      return;
    }

    // Masks span their chunk once and are magnified, so one clamped bilinear sampler serves all.
    if (m_maskSampler == nullptr) {
      m_maskSampler = ctx.getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }

    const bool retained = layerDraw.retained;
    ChunkSet& set = retained ? m_retained : m_frame;

    // The layer limit is checked first, so a chunk is never added without a layer.
    Chunk* chunk = set.layerCount < (retained ? kMaxRetainedLayers : kMaxFrameLayers)
      ? set.findOrAdd(objectToWorld, retained ? kMaxRetainedChunks : kMaxFrameChunks)
      : nullptr;
    if (chunk == nullptr) {
      m_statistics.droppedLayers++;
      return;
    }
    set.layerCount++;

    Layer& layer = chunk->layers.emplace_back();
    layer.draw = layerDraw;

    // A retained layer is resolved every frame it is in the table, in prepareSceneData.
    if (!retained) {
      resolveLayer(ctx.getSceneManager(), layer);
      m_statistics.layers++;
      countLayer(*chunk, layer);
    }
  }

  void TerrainLayers::releaseChunk(const Matrix4& objectToWorld) {
    const auto found = m_retained.indexByTransform.find(transformKey(objectToWorld));
    if (found == m_retained.indexByTransform.end()) {
      return;
    }

    std::vector<Chunk>& chunks = m_retained.chunks;
    const size_t index = found->second;
    m_retained.layerCount -= static_cast<uint32_t>(chunks[index].layers.size());
    m_retained.indexByTransform.erase(found);

    if (index != chunks.size() - 1) {
      chunks[index] = std::move(chunks.back());
      m_retained.indexByTransform[transformKey(chunks[index].objectToWorld)] = index;
    }
    chunks.pop_back();
  }

  std::vector<const TerrainLayers::Chunk*> TerrainLayers::gatherChunks() const {
    std::vector<const Chunk*> chunks;
    const auto usable = [](const Chunk& chunk) {
      return !chunk.layers.empty() && chunk.layers.front().draw.maskScale > 0.f;
    };

    for (const Chunk& chunk : m_frame.chunks) {
      if (usable(chunk)) {
        chunks.push_back(&chunk);
      }
    }

    if (useRetainedChunks()) {
      for (const auto& [key, index] : m_retained.indexByTransform) {
        const Chunk& chunk = m_retained.chunks[index];
        if (usable(chunk) && m_frame.indexByTransform.count(key) == 0) {
          chunks.push_back(&chunk);
        }
      }
    }

    return chunks;
  }

  bool TerrainLayers::writeGrid(const std::vector<const Chunk*>& chunks) {
    // The grid lies in the plane the chunks' masks span, taken from the first chunk.
    const Matrix4& reference = chunks.front()->objectToWorld;
    uint32_t axes[2] = { dominantAxis(reference[0]), dominantAxis(reference[2]) };
    if (axes[0] == axes[1]) {
      axes[1] = (axes[0] + 1) % 3;
    }

    std::vector<Footprint> footprints;
    footprints.reserve(chunks.size());
    Footprint bounds = { { FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX } };
    float cellSize = FLT_MAX;
    for (const Chunk* chunk : chunks) {
      const Footprint& footprint = footprints.emplace_back(footprintOf(*chunk, axes));
      for (uint32_t k = 0; k < 2; k++) {
        bounds.min[k] = std::min(bounds.min[k], footprint.min[k]);
        bounds.max[k] = std::max(bounds.max[k], footprint.max[k]);
      }
      cellSize = std::min(cellSize, std::max(footprint.max[0] - footprint.min[0], footprint.max[1] - footprint.min[1]));
    }

    const float span = std::max(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1]);
    if (!(cellSize > 0.f) || !std::isfinite(span)) {
      return false;
    }
    cellSize = std::max(cellSize, span / static_cast<float>(kMaxGridSide));

    const auto cellsAlong = [&](uint32_t k) {
      const float cells = std::ceil((bounds.max[k] - bounds.min[k]) / cellSize - kFootprintInset);
      return std::clamp(static_cast<uint32_t>(std::max(cells, 1.f)), 1u, kMaxGridSide);
    };
    const uint32_t width = cellsAlong(0);
    const uint32_t height = cellsAlong(1);

    m_table.assign(static_cast<size_t>(width) * height, Vector4(0.f));

    const uint32_t size[2] = { width, height };
    const auto cellOf = [&](float position, uint32_t k) {
      const float cell = std::floor((position - bounds.min[k]) / cellSize);
      return static_cast<uint32_t>(std::clamp(cell, 0.f, static_cast<float>(size[k] - 1)));
    };

    const float inset = cellSize * kFootprintInset;
    for (size_t i = 0; i < chunks.size(); i++) {
      const Footprint& footprint = footprints[i];
      for (uint32_t y = cellOf(footprint.min[1] + inset, 1); y <= cellOf(footprint.max[1] - inset, 1); y++) {
        for (uint32_t x = cellOf(footprint.min[0] + inset, 0); x <= cellOf(footprint.max[0] - inset, 0); x++) {
          Vector4& cell = m_table[static_cast<size_t>(y) * width + x];
          uint32_t slot = 0;
          while (slot < kMaxTerrainLayerGridCandidates && cell.data[slot] != 0.f) {
            slot++;
          }
          if (slot == kMaxTerrainLayerGridCandidates) {
            m_statistics.crowdedCells++;
            continue;
          }
          // A chunk number plus one; zero is an empty slot.
          cell.data[slot] = static_cast<float>(i + 1);
        }
      }
    }

    m_args.gridOrigin = vec2 { bounds.min[0], bounds.min[1] };
    m_args.gridCellSize = cellSize;
    m_args.gridAxes = axes[0] | (axes[1] << 2);
    m_args.gridWidth = width;
    m_args.gridHeight = height;
    return true;
  }

  void TerrainLayers::writeChunksAndLayers(const std::vector<const Chunk*>& chunks) {
    const auto unorm8 = [](float value) {
      return static_cast<uint32_t>(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
    };
    const auto number = [](uint32_t value) {
      return static_cast<float>(value);
    };

    const size_t chunkBase = m_table.size();
    m_table.resize(chunkBase + chunks.size() * kTerrainLayerChunkTexels);
    const size_t layerBase = m_table.size();
    m_args.chunkOffset = static_cast<uint32_t>(chunkBase);
    m_args.layerOffset = static_cast<uint32_t>(layerBase);

    uint32_t layerNumber = 0;
    for (size_t i = 0; i < chunks.size(); i++) {
      const Chunk& chunk = *chunks[i];
      const TerrainLayerDraw& first = chunk.layers.front().draw;
      const Matrix4 worldToObject = inverse(chunk.objectToWorld);

      Vector4* texels = &m_table[chunkBase + i * kTerrainLayerChunkTexels];
      for (uint32_t row = 0; row < 3; row++) {
        texels[row] = Vector4(worldToObject[0].data[row], worldToObject[1].data[row],
                              worldToObject[2].data[row], worldToObject[3].data[row]);
      }
      texels[3] = Vector4(first.projectionOrigin.x, first.projectionOrigin.y, first.projectionOrigin.z, first.maskScale);
      texels[4] = Vector4(number(layerNumber), number(static_cast<uint32_t>(chunk.layers.size())), 0.f, 0.f);

      for (const Layer& layer : chunk.layers) {
        // A layer whose mask is not resident yet is left covering nothing rather than everything.
        const bool hasMask = layer.draw.maskTexture.isValid();
        const bool maskMissing = hasMask && layer.maskTextureIndex == kSurfaceMaterialInvalidTextureIndex;
        const uint32_t flags =
          (maskMissing ? 0xFFu : unorm8(layer.draw.alphaReference)) |
          (unorm8(layer.roughnessConstant) << TERRAIN_LAYER_ROUGHNESS_SHIFT) |
          (unorm8(layer.metallicConstant) << TERRAIN_LAYER_METALLIC_SHIFT) |
          (static_cast<uint32_t>(layer.draw.projection) << TERRAIN_LAYER_PROJECTION_SHIFT) |
          (hasMask ? TERRAIN_LAYER_FLAG_HAS_MASK : 0u) |
          (layer.draw.colorAlphaInCoverage ? TERRAIN_LAYER_FLAG_COLOR_ALPHA_IN_COVERAGE : 0u) |
          (layer.colorIsLinear ? TERRAIN_LAYER_FLAG_COLOR_IS_LINEAR : 0u);

        m_table.emplace_back(layer.draw.texcoordU.x, layer.draw.texcoordU.y, layer.draw.texcoordU.z, number(layer.colorTextureIndex));
        m_table.emplace_back(layer.draw.texcoordV.x, layer.draw.texcoordV.y, layer.draw.texcoordV.z, number(layer.normalTextureIndex));
        m_table.emplace_back(number(layer.roughnessTextureIndex), number(layer.metallicTextureIndex),
                             number(layer.maskTextureIndex), number(layer.colorSamplerIndex));
        m_table.emplace_back(number(flags & 0xFFFFFFu), number(flags >> 24), 0.f, 0.f);
        layerNumber++;
      }
    }
  }

  void TerrainLayers::prepareSceneData(Rc<RtxContext> ctx) {
    m_args = {};
    if (!enable() || !evaluateAtHit()) {
      return;
    }

    SceneManager& sceneManager = ctx->getSceneManager();

    if (useRetainedChunks()) {
      for (Chunk& chunk : m_retained.chunks) {
        for (Layer& layer : chunk.layers) {
          resolveLayer(sceneManager, layer);
          countLayer(chunk, layer);
        }
      }
      m_statistics.retainedChunks = static_cast<uint32_t>(m_retained.chunks.size());
      m_statistics.retainedLayers = m_retained.layerCount;
    }

    const std::vector<const Chunk*> chunks = gatherChunks();
    if (chunks.empty() || !writeGrid(chunks)) {
      return;
    }
    writeChunksAndLayers(chunks);

    // Whole rows are uploaded.
    const uint32_t rows = static_cast<uint32_t>((m_table.size() + kTerrainLayerTableWidth - 1) / kTerrainLayerTableWidth);
    if (rows > kTableHeight) {
      ONCE(Logger::warn("[RTX Terrain Layers] The layer table does not fit its texture; terrain is read from the cascade alone."));
      return;
    }
    m_table.resize(static_cast<size_t>(rows) * kTerrainLayerTableWidth, Vector4(0.f));

    if (!m_tableImage.isValid()) {
      Rc<DxvkContext> dxvkContext = ctx.ptr();
      m_tableImage = Resources::createImageResource(dxvkContext, "RTX Terrain Layers - Table",
                                                    { kTerrainLayerTableWidth, kTableHeight, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
    }

    VkImageSubresourceLayers subresources = {};
    subresources.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresources.mipLevel = 0;
    subresources.baseArrayLayer = 0;
    subresources.layerCount = 1;
    ctx->updateImage(m_tableImage.image, subresources, { 0, 0, 0 }, { kTerrainLayerTableWidth, rows, 1 }, m_table.data(),
                     kTerrainLayerTableWidth * sizeof(Vector4), m_table.size() * sizeof(Vector4));

    m_tableTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    sceneManager.trackTexture(TextureRef(m_tableImage.view), m_tableTextureIndex, true, false);
    if (m_tableTextureIndex == kSurfaceMaterialInvalidTextureIndex) {
      return;
    }

    m_maskSamplerIndex = sceneManager.trackSampler(m_maskSampler);

    m_args.chunkCount = static_cast<uint32_t>(chunks.size());
    m_args.maskSamplerIndex = m_maskSamplerIndex;
    m_args.minBlendWeight = minBlendWeight();
    m_args.tableTextureIndex = m_tableTextureIndex;
  }

  TerrainLayerArgs TerrainLayers::getTerrainLayerArgs() const {
    return m_args;
  }

  void TerrainLayers::onFrameEnd() {
    m_statistics.chunks = static_cast<uint32_t>(m_frame.chunks.size());
    m_shownStatistics = m_statistics;

    if (logSummary() && ++m_framesSinceLog >= kFramesBetweenLogs) {
      m_framesSinceLog = 0;
      const Statistics& s = m_shownStatistics;
      Logger::info(str::format("[RTX Terrain Layers] ", s.chunks, " chunks, ", s.layers, " layers from this frame's draws; ",
                               s.retainedChunks, " chunks, ", s.retainedLayers, " layers retained (most in one chunk: ",
                               s.maxLayersInChunk, "); ", s.maskedLayers, " masked, ", s.sideProjectedLayers,
                               " side projected, ", s.replacedLayers, " with a replacement material; ",
                               s.droppedLayers, " left out over the table limits, ", s.crowdedCells,
                               " left out of a full grid cell. Grid ", m_args.gridWidth, " x ", m_args.gridHeight,
                               " cells of ", m_args.gridCellSize, "."));

      for (const Chunk& chunk : m_frame.chunks) {
        const Layer& first = chunk.layers.front();
        Logger::info(str::format("[RTX Terrain Layers]   chunk at (", chunk.objectToWorld[3][0], ", ", chunk.objectToWorld[3][1],
                                 ", ", chunk.objectToWorld[3][2], "): ", chunk.layers.size(), " layers, projection origin (",
                                 first.draw.projectionOrigin.x, ", ", first.draw.projectionOrigin.y, ", ",
                                 first.draw.projectionOrigin.z, "), mask scale ", first.draw.maskScale));
      }
    }

    m_frame.clear();
    m_statistics = {};
  }

  void TerrainLayers::showImguiSettings() const {
    RemixGui::Checkbox("Collect Terrain Layers", &enableObject());
    RemixGui::Checkbox("Evaluate Terrain Layers at Hit", &evaluateAtHitObject());
    RemixGui::Checkbox("Use Retained Terrain Chunks", &useRetainedChunksObject());
    RemixGui::Checkbox("Log Terrain Layer Summary", &logSummaryObject());

    const Statistics& s = m_shownStatistics;
    ImGui::Text("This frame: %u chunks, %u layers. Retained: %u chunks, %u layers", s.chunks, s.layers, s.retainedChunks, s.retainedLayers);
    ImGui::Text("Most layers in one chunk: %u. Masked: %u, side projected: %u, replaced: %u",
                s.maxLayersInChunk, s.maskedLayers, s.sideProjectedLayers, s.replacedLayers);
    ImGui::Text("Left out: %u over the limits, %u in full grid cells", s.droppedLayers, s.crowdedCells);
  }

}
