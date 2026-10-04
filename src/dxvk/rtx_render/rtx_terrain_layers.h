#pragma once

// rtx_terrain_layers.h -- layered terrain: the table of terrain chunks and their layers, built
// from the draws a game describes through REMIXAPI_D3D9_RS_TERRAIN_LAYER (remix_c.h).
//
// A game that draws terrain as a stack of blended layers over shared chunk geometry gives every
// layer its own tiling texture and coverage mask. The terrain baker resamples that stack into
// cascade textures; this table keeps the stack itself -- per chunk, the layers in submission
// order, each with the textures and projection needed to evaluate it at a ray hit.
//
// Chunks come two ways: described by the frame's own layer draws and gone at its end, or
// described once as retained and kept until released -- for terrain the game is not drawing in
// layers itself, such as the ground under a distant level of detail.
//
// The shaders read the table from a float texture: a grid over the chunks' footprints, so a hit
// finds its chunk without walking them all, then the chunks, then the layers.

#include <memory>
#include <unordered_map>
#include <vector>

#include "rtx_option.h"
#include "rtx_resources.h"
#include "rtx_types.h"
#include "rtx/pass/raytrace_args.h"

namespace dxvk {

  class RtxContext;
  class SceneManager;

  class TerrainLayers {
  public:
    RTX_OPTION("rtx.terrainLayers", bool, enable, false,
               "Collects the draws a game describes as terrain layers (REMIXAPI_D3D9_RS_TERRAIN_LAYER) into a per-chunk\n"
               "layer table, and keeps the layers' textures resident.");
    RTX_OPTION("rtx.terrainLayers", bool, evaluateAtHit, true,
               "Terrain over a collected chunk takes albedo, normal, roughness and metalness from the chunk's layers,\n"
               "each sampled at its own resolution through its own projection and blended by the game's coverage masks.\n"
               "Off, terrain reads the baked cascade alone. Requires rtx.terrainLayers.enable and the terrain baker.");
    RTX_OPTION("rtx.terrainLayers", bool, useRetainedChunks, true,
               "Includes the chunks a game described as retained (REMIXAPI_D3D9_TERRAIN_LAYER_RETAINED) in the table.\n"
               "Off, only the chunks described by the current frame's draws are evaluated.");
    RTX_OPTION("rtx.terrainLayers", float, fadeStartDistance, 0.f,
               "Distance from the camera, in world units, at which terrain starts to fade from its layers back to the\n"
               "surface's own material. For a game whose layers are only described out to some range.");
    RTX_OPTION("rtx.terrainLayers", float, fadeEndDistance, 0.f,
               "Distance from the camera at which terrain has faded to the surface's own material entirely.\n"
               "No fade unless this is greater than rtx.terrainLayers.fadeStartDistance.");
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

    // The layers sharing one object-to-world transform, in submission order: the order the game
    // composites them in.
    struct Chunk {
      Matrix4 objectToWorld;
      std::vector<Layer> layers;
    };

    static constexpr size_t kMaxFrameChunks = 64;
    static constexpr size_t kMaxFrameLayers = 1024;
    static constexpr size_t kMaxRetainedChunks = 2048;
    static constexpr size_t kMaxRetainedLayers = 32768;
    // The grid is at most this many cells a side; its cells grow to fit.
    static constexpr uint32_t kMaxGridSide = 128;

    /**
      * \brief: Appends a draw to its chunk's layer list and tracks its textures for this frame.
      *         Does nothing for a draw that is not described as a terrain layer.
      *
      * \param [in] ctx: tracks the layer's textures and finds its replacement material
      * \param [in] drawCallState: the draw, before the terrain baker overrides its material
      */
    void addLayerDraw(RtxContext& ctx, const DrawCallState& drawCallState);

    /**
      * \brief: Appends a layer to the chunk with the given transform: this frame's chunk, with
      *         its textures tracked for the frame, or the retained chunk when the layer is
      *         described as retained.
      *
      * \param [in] ctx: tracks the layer's textures and finds its replacement material
      * \param [in] layer: the layer as the game described it
      * \param [in] objectToWorld: the chunk's transform
      */
    void addLayer(RtxContext& ctx, const TerrainLayerDraw& layer, const Matrix4& objectToWorld);

    /**
      * \brief: Drops the retained chunk with the given transform, if there is one.
      */
    void releaseChunk(const Matrix4& objectToWorld);

    /**
      * \brief: Tracks the retained layers' textures for this frame and writes the table the
      *         shaders read. Call once the frame's draws are in, before the scene's textures
      *         are bound.
      */
    void prepareSceneData(Rc<RtxContext> ctx);

    /**
      * \brief: The table as the shaders find it. Empty unless the layers are evaluated at a hit.
      */
    TerrainLayerArgs getTerrainLayerArgs() const;

    void onFrameEnd();

    void showImguiSettings() const;

  private:
    struct Statistics {
      uint32_t chunks = 0;
      uint32_t layers = 0;
      uint32_t retainedChunks = 0;
      uint32_t retainedLayers = 0;
      uint32_t maxLayersInChunk = 0;
      uint32_t maskedLayers = 0;
      uint32_t replacedLayers = 0;
      uint32_t sideProjectedLayers = 0;
      // Left out because a limit above was reached.
      uint32_t droppedLayers = 0;
      // Chunks left out of a grid cell that already held kMaxTerrainLayerGridCandidates.
      uint32_t crowdedCells = 0;
    };

    // Chunks keyed by their transform.
    struct ChunkSet {
      std::vector<Chunk> chunks;
      std::unordered_map<XXH64_hash_t, size_t> indexByTransform;
      uint32_t layerCount = 0;

      Chunk* findOrAdd(const Matrix4& objectToWorld, size_t maxChunks);
      void clear();
    };

    void resolveLayer(SceneManager& sceneManager, Layer& layer);
    void trackLayerTextures(SceneManager& sceneManager, Layer& layer);
    void countLayer(const Chunk& chunk, const Layer& layer);
    // The chunks the table holds this frame: the frame's own first, so they are found first.
    std::vector<const Chunk*> gatherChunks() const;
    bool writeGrid(const std::vector<const Chunk*>& chunks);
    void writeChunksAndLayers(const std::vector<const Chunk*>& chunks);

    ChunkSet m_frame;
    ChunkSet m_retained;

    // The table, one texel per element, and where the shaders find it.
    std::vector<Vector4> m_table;
    Resources::Resource m_tableImage;
    uint32_t m_tableTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    TerrainLayerArgs m_args = {};

    Statistics m_statistics;
    // The last completed frame's, for display.
    Statistics m_shownStatistics;
    Rc<DxvkSampler> m_maskSampler;
    uint32_t m_maskSamplerIndex = kSurfaceMaterialInvalidTextureIndex;
    uint32_t m_framesSinceLog = 0;
  };

}
