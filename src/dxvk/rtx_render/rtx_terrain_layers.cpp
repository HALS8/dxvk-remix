#include "rtx_terrain_layers.h"

#include "rtx_asset_replacer.h"
#include "rtx_context.h"
#include "rtx_texture.h"
#include "rtx_scene_manager.h"
#include "rtx_imgui.h"

namespace dxvk {

  namespace {
    constexpr uint32_t kFramesBetweenLogs = 600;
  }

  TerrainLayers::Chunk* TerrainLayers::findOrAddChunk(const Matrix4& objectToWorld) {
    const XXH64_hash_t key = XXH3_64bits(&objectToWorld, sizeof(objectToWorld));

    const auto found = m_chunkIndexByTransform.find(key);
    if (found != m_chunkIndexByTransform.end()) {
      return &m_chunks[found->second];
    }

    if (m_chunks.size() == kMaxChunks) {
      return nullptr;
    }

    m_chunkIndexByTransform.emplace(key, m_chunks.size());
    Chunk& chunk = m_chunks.emplace_back();
    chunk.objectToWorld = objectToWorld;
    return &chunk;
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

  void TerrainLayers::addLayerDraw(RtxContext& ctx, const DrawCallState& drawCallState) {
    if (drawCallState.getTerrainLayer().enabled) {
      addLayer(ctx, drawCallState.getTerrainLayer(), drawCallState.getTransformData().objectToWorld);
    }
  }

  void TerrainLayers::addLayer(RtxContext& ctx, const TerrainLayerDraw& layerDraw, const Matrix4& objectToWorld) {
    if (!enable()) {
      return;
    }

    SceneManager& sceneManager = ctx.getSceneManager();

    // The layer limit is checked first, so a chunk is never added without a layer.
    Chunk* chunk = m_layerCount < kMaxLayers ? findOrAddChunk(objectToWorld) : nullptr;
    if (chunk == nullptr) {
      m_statistics.droppedLayers++;
      return;
    }
    m_layerCount++;

    // Masks span their chunk once and are magnified, so one clamped bilinear sampler serves all.
    if (m_maskSampler == nullptr) {
      m_maskSampler = ctx.getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }
    m_maskSamplerIndex = sceneManager.trackSampler(m_maskSampler);

    Layer& layer = chunk->layers.emplace_back();
    layer.draw = layerDraw;

    // A legacy material is identified by its colour texture.
    std::shared_ptr<MaterialData> replacement =
      sceneManager.getAssetReplacer()->getReplacementMaterial(layerDraw.colorTexture.getImageHash());
    if (replacement != nullptr && replacement->getType() == MaterialDataType::Opaque) {
      layer.replacement = std::move(replacement);
    }

    trackLayerTextures(sceneManager, layer);

    m_statistics.layers++;
    m_statistics.maxLayersInChunk = std::max(m_statistics.maxLayersInChunk, static_cast<uint32_t>(chunk->layers.size()));
    m_statistics.maskedLayers += layer.draw.maskTexture.isValid();
    m_statistics.replacedLayers += layer.replacement != nullptr;
    m_statistics.sideProjectedLayers += layer.draw.projection != TerrainLayerDraw::Projection::XZ;
  }

  TerrainLayerArgs TerrainLayers::getTerrainLayerArgs() const {
    TerrainLayerArgs args = {};
    if (!enable() || !evaluateAtHit()) {
      return args;
    }

    const auto pack16 = [](uint32_t low, uint32_t high) {
      return (low & 0xFFFFu) | (high << 16);
    };
    const auto unorm8 = [](float value) {
      return static_cast<uint32_t>(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
    };

    args.maskSamplerIndex = m_maskSamplerIndex;
    args.minBlendWeight = minBlendWeight();

    uint32_t layerCount = 0;
    for (const Chunk& chunk : m_chunks) {
      const TerrainLayerDraw& first = chunk.layers.front().draw;

      TerrainLayerChunk& gpuChunk = args.chunks[args.chunkCount++];
      gpuChunk.worldToObject = inverse(chunk.objectToWorld);
      gpuChunk.projectionOrigin = first.projectionOrigin;
      gpuChunk.maskScale = first.maskScale;
      gpuChunk.firstLayer = layerCount;
      gpuChunk.layerCount = static_cast<uint32_t>(chunk.layers.size());

      for (const Layer& layer : chunk.layers) {
        TerrainLayer& gpuLayer = args.layers[layerCount++];
        gpuLayer.texcoordU = layer.draw.texcoordU;
        gpuLayer.texcoordV = layer.draw.texcoordV;
        gpuLayer.colorAndNormalTextureIndex = pack16(layer.colorTextureIndex, layer.normalTextureIndex);
        gpuLayer.roughnessAndMetallicTextureIndex = pack16(layer.roughnessTextureIndex, layer.metallicTextureIndex);
        gpuLayer.maskTextureAndSamplerIndex = pack16(layer.maskTextureIndex, layer.colorSamplerIndex);

        // A layer whose mask is not resident yet is left covering nothing rather than everything.
        const bool hasMask = layer.draw.maskTexture.isValid();
        const bool maskMissing = hasMask && layer.maskTextureIndex == kSurfaceMaterialInvalidTextureIndex;
        gpuLayer.flags =
          (maskMissing ? 0xFFu : unorm8(layer.draw.alphaReference)) |
          (unorm8(layer.roughnessConstant) << TERRAIN_LAYER_ROUGHNESS_SHIFT) |
          (unorm8(layer.metallicConstant) << TERRAIN_LAYER_METALLIC_SHIFT) |
          (static_cast<uint32_t>(layer.draw.projection) << TERRAIN_LAYER_PROJECTION_SHIFT) |
          (hasMask ? TERRAIN_LAYER_FLAG_HAS_MASK : 0u) |
          (layer.draw.colorAlphaInCoverage ? TERRAIN_LAYER_FLAG_COLOR_ALPHA_IN_COVERAGE : 0u) |
          (layer.colorIsLinear ? TERRAIN_LAYER_FLAG_COLOR_IS_LINEAR : 0u);
      }
    }

    return args;
  }

  void TerrainLayers::onFrameEnd() {
    m_statistics.chunks = static_cast<uint32_t>(m_chunks.size());
    m_shownStatistics = m_statistics;

    if (logSummary() && ++m_framesSinceLog >= kFramesBetweenLogs) {
      m_framesSinceLog = 0;
      const Statistics& s = m_shownStatistics;
      Logger::info(str::format("[RTX Terrain Layers] ", s.chunks, " chunks, ", s.layers, " layers (most in one chunk: ",
                               s.maxLayersInChunk, "); ", s.maskedLayers, " masked, ", s.sideProjectedLayers,
                               " side projected, ", s.replacedLayers, " with a replacement material; ",
                               s.droppedLayers, " left out over the table limits."));

      for (const Chunk& chunk : m_chunks) {
        const Layer& first = chunk.layers.front();
        Logger::info(str::format("[RTX Terrain Layers]   chunk at (", chunk.objectToWorld[3][0], ", ", chunk.objectToWorld[3][1],
                                 ", ", chunk.objectToWorld[3][2], "): ", chunk.layers.size(), " layers, projection origin (",
                                 first.draw.projectionOrigin.x, ", ", first.draw.projectionOrigin.y, ", ",
                                 first.draw.projectionOrigin.z, "), mask scale ", first.draw.maskScale));
      }
    }

    m_chunks.clear();
    m_chunkIndexByTransform.clear();
    m_layerCount = 0;
    m_statistics = {};
  }

  void TerrainLayers::showImguiSettings() const {
    RemixGui::Checkbox("Collect Terrain Layers", &enableObject());
    RemixGui::Checkbox("Evaluate Terrain Layers at Hit", &evaluateAtHitObject());
    RemixGui::Checkbox("Log Terrain Layer Summary", &logSummaryObject());

    const Statistics& s = m_shownStatistics;
    ImGui::Text("Chunks: %u, layers: %u (most in one chunk: %u)", s.chunks, s.layers, s.maxLayersInChunk);
    ImGui::Text("Masked: %u, side projected: %u, replaced: %u, left out: %u",
                s.maskedLayers, s.sideProjectedLayers, s.replacedLayers, s.droppedLayers);
  }

}
