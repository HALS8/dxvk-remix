#include "rtx_terrain_layers.h"

#include "rtx_asset_replacer.h"
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

    if (layer.replacement == nullptr) {
      track(layer.draw.colorTexture, layer.colorTextureIndex);
      return;
    }

    OpaqueMaterialData& material = layer.replacement->getOpaqueMaterialData();
    track(material.getAlbedoOpacityTexture(), layer.colorTextureIndex);
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

  void TerrainLayers::addLayerDraw(SceneManager& sceneManager, const DrawCallState& drawCallState) {
    if (!enable() || !drawCallState.getTerrainLayer().enabled) {
      return;
    }

    Chunk* chunk = findOrAddChunk(drawCallState.getTransformData().objectToWorld);
    if (chunk == nullptr || chunk->layers.size() == kMaxLayersPerChunk) {
      m_statistics.droppedLayers++;
      return;
    }

    Layer& layer = chunk->layers.emplace_back();
    layer.draw = drawCallState.getTerrainLayer();

    std::shared_ptr<MaterialData> replacement =
      sceneManager.getAssetReplacer()->getReplacementMaterial(drawCallState.getMaterialData().getHash());
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
    m_statistics = {};
  }

  void TerrainLayers::showImguiSettings() const {
    RemixGui::Checkbox("Collect Terrain Layers", &enableObject());
    RemixGui::Checkbox("Log Terrain Layer Summary", &logSummaryObject());

    const Statistics& s = m_shownStatistics;
    ImGui::Text("Chunks: %u, layers: %u (most in one chunk: %u)", s.chunks, s.layers, s.maxLayersInChunk);
    ImGui::Text("Masked: %u, side projected: %u, replaced: %u, left out: %u",
                s.maskedLayers, s.sideProjectedLayers, s.replacedLayers, s.droppedLayers);
  }

}
