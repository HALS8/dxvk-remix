#pragma once

// rtx_terrain_layers.h -- layered terrain: the table of terrain chunks and their layers, built each
// frame from the draws a game describes through REMIXAPI_D3D9_RS_TERRAIN_LAYER (remix_c.h).
//
// A game that draws terrain as a stack of blended layers over shared chunk geometry gives every
// layer its own tiling texture and coverage mask. The terrain baker resamples that stack into
// cascade textures; this table keeps the stack itself -- per chunk, the layers in submission
// order, each with the textures and projection needed to evaluate it at a ray hit.

#include <memory>
#include <unordered_map>
#include <vector>

#include "rtx_option.h"
#include "rtx_types.h"
#include "rtx/pass/raytrace_args.h"

namespace dxvk {

  class RtxContext;
  class SceneManager;

  class TerrainLayers {
  public:
    RTX_OPTION("rtx.terrainLayers", bool, enable, false,
               "Collects the draws a game describes as terrain layers (REMIXAPI_D3D9_RS_TERRAIN_LAYER) into a per-chunk\n"
               "layer table each frame, and keeps the layers' textures resident.");
    RTX_OPTION("rtx.terrainLayers", bool, evaluateAtHit, true,
               "Terrain over a collected chunk takes albedo, normal, roughness and metalness from the chunk's layers,\n"
               "each sampled at its own resolution through its own projection and blended by the game's coverage masks.\n"
               "Off, terrain reads the baked cascade alone. Requires rtx.terrainLayers.enable and the terrain baker.");
    RTX_OPTION("rtx.terrainLayers", float, minBlendWeight, 0.02f,
               "Layers whose composited weight at a hit is below this are not sampled.");
    RTX_OPTION("rtx.terrainLayers", bool, logSummary, false,
               "Logs, every 600 frames, the size of the terrain layer table and what was left out of it.");

    // The textures and constants one layer is evaluated from. Texture indices are bindless
    // indices, kSurfaceMaterialInvalidTextureIndex where the layer has no such texture.
    struct Layer {
      TerrainLayerDraw draw;
      // Null for a layer evaluated from the game's own colour texture.
      std::shared_ptr<MaterialData> replacement;

      bool colorIsLinear = false;
      float roughnessConstant = 0.f;
      float metallicConstant = 0.f;

      uint32_t colorTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t colorSamplerIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t maskTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t roughnessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t metallicTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t emissiveTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t heightTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    };

    // The draws sharing one object-to-world transform, in submission order: the order the game
    // composites them in.
    struct Chunk {
      Matrix4 objectToWorld;
      std::vector<Layer> layers;
    };

    static constexpr size_t kMaxChunks = kMaxTerrainLayerChunks;
    static constexpr size_t kMaxLayers = kMaxTerrainLayers;

    /**
      * \brief: Appends a draw to its chunk's layer list and tracks its textures for this frame.
      *         Does nothing for a draw that is not described as a terrain layer.
      *
      * \param [in] ctx: tracks the layer's textures and finds its replacement material
      * \param [in] drawCallState: the draw, before the terrain baker overrides its material
      */
    void addLayerDraw(RtxContext& ctx, const DrawCallState& drawCallState);

    /**
      * \brief: The table as the shaders read it. Empty unless the layers are evaluated at a hit.
      */
    TerrainLayerArgs getTerrainLayerArgs() const;

    void onFrameEnd();

    const std::vector<Chunk>& getChunks() const {
      return m_chunks;
    }

    void showImguiSettings() const;

  private:
    struct Statistics {
      uint32_t chunks = 0;
      uint32_t layers = 0;
      uint32_t maxLayersInChunk = 0;
      uint32_t maskedLayers = 0;
      uint32_t replacedLayers = 0;
      uint32_t sideProjectedLayers = 0;
      // Left out because a limit above was reached.
      uint32_t droppedLayers = 0;
    };

    Chunk* findOrAddChunk(const Matrix4& objectToWorld);
    void trackLayerTextures(SceneManager& sceneManager, Layer& layer);

    std::vector<Chunk> m_chunks;
    std::unordered_map<XXH64_hash_t, size_t> m_chunkIndexByTransform;
    Statistics m_statistics;
    // The last completed frame's, for display.
    Statistics m_shownStatistics;
    uint32_t m_layerCount = 0;
    Rc<DxvkSampler> m_maskSampler;
    uint32_t m_maskSamplerIndex = kSurfaceMaterialInvalidTextureIndex;
    uint32_t m_framesSinceLog = 0;
  };

}
