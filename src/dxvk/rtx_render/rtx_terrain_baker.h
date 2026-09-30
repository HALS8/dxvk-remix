/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

#include "rtx_context.h"
#include "rtx_geometry_utils.h"
#include "rtx_resources.h"
#include "rtx_mipmap.h"
#include "../util/util_struct_hash.h"

#include <unordered_map>
#include <unordered_set>

namespace dxvk {

  struct D3D9FixedFunctionVS;
  struct D3D9SharedPS;

  // Which faces a side projection view keeps. The view only holds surfaces facing it, so the
  // floor under a ceiling, or the far side of a ridge, cannot hide the surface a hit samples.
  // Which winding counts as facing depends on the game's convention, hence the choice.
  enum class TerrainSideProjectionCulling : uint32_t {
    None = 0,
    Back,
    Front
  };

  class TerrainBaker {
  public:
    TerrainBaker() { }
    ~TerrainBaker() { }

    bool bakeDrawCall(Rc<RtxContext> rtxContext, const DxvkContextState& dxvkCtxState,
                      DxvkRaytracingInstanceState& rtState, const DrawParameters& params,
                      const DrawCallState& drawCallState, OpaqueMaterialData* replacementMaterial,
                      Matrix4& textureTransformOut);
    TerrainArgs getTerrainArgs() const;

    void onFrameEnd(Rc<DxvkContext> ctx);
    void prepareSceneData(Rc<RtxContext> ctx);

    const RtxMipmap::Resource& getTerrainTexture(ReplacementMaterialTextureType::Enum textureType) const;
    const MaterialData* getMaterialData() const;
    const Rc<DxvkSampler>& getTerrainSampler() const;

    // True if any cascade image was created or resized during the current frame.
    // Used by SceneManager to keep terrain draws on the dynamic path for one
    // frame after the cascade set changes so the per-frame surface-material
    // pre-creation cache picks up the new cascade indices instead of reusing
    // a stale entry that was built when the cascade set was incomplete.
    bool cascadeCompositionChangedThisFrame() const {
      return m_cascadeCompositionChangedThisFrame;
    }

    void showImguiSettings() const;

    //
    // RTX OPTIONS
    
    friend class ImGUI; // <-- we want to modify these values directly.

    // Note: call needsTerrainBaking() to check if baking is enabled
    RTX_OPTION("rtx.terrainBaker", bool, enableBaking, true, "[Experimental] Enables runtime baking of blended terrains from top down (i.e. in an opposite direction of \"rtx.zUp\").\n"
                                                             "It bakes multiple blended albedo terrain textures into a single texture sampled during ray tracing. The system requires \"Terrain Textures\" to contain hashes of the terrain textures to apply.\n"
                                                              "Only use this system if the game renders terrain surfaces with multiple blended surfaces on top of each other (i.e. sand mixed with dirt, grass, snow, etc.).\n"
                                                              "Requirement: the baked terrain surfaces must not be placed vertically in the game world. Horizontal surfaces will have the best image quality. Requires \"rtx.zUp\" to be set properly.");

    RTX_OPTION("rtx.terrainBaker", bool, clearTerrainBeforeBaking, false, "Clears the terrain textures before the cascade map is baked from scratch, i.e. whenever it is recentered or its layout changes.");
    RTX_OPTION("rtx.terrainBaker", bool, logLayoutChanges, false, "Logs, every 600 frames, how often the cascade map was discarded and which part of its layout changed.");
    RTX_OPTION("rtx.terrainBaker", bool, debugDisableBaking , false, "Force disables rebaking every frame. Used for debugging only.")
    RTX_OPTION("rtx.terrainBaker", bool, logSurfaceOrientation, false,
               "Diagnostic. Logs how each baked terrain draw call is oriented, once per distinct geometry.\n"
               "The cascade map is a top-down projection, so it can only represent a surface that faces roughly up:\n"
               "a near-vertical face projects to almost no area, writes almost nothing, and then samples the column\n"
               "of texels above it - which is what smears one texture down a cliff, a cave wall or a ceiling.\n"
               "Each line reports the material hash (the same hash rtx.terrainTextures is matched against), the\n"
               "triangle count, and how those triangles are distributed by the angle between their face normal and\n"
               "the up axis. A draw whose triangles are all flat or all vertical can be handled per draw call; one\n"
               "that mixes both cannot, and needs the decision made per pixel instead.");
    RTX_OPTION("rtx.terrainBaker", bool, debugDisableBinding, false, "Force disables binding of the baked terrain texture to the terrain meshes. Used for debugging only.");
    RTX_OPTION("rtx.terrainBaker", bool, disableBackFaceCulling, false, "Disables back-face culling for baked terrain instances. When enabled, all terrain will render as double-sided.");

    // Returns shared enablement composed of multiple enablement inputs
    static bool needsTerrainBaking();

    struct Material {
      friend class ImGUI; // <-- we want to modify these values directly.
      friend class TerrainBaker; // <-- we want to modify these values directly.

      RTX_OPTION("rtx.terrainBaker.material", bool, replacementSupportInPS, true, 
                 "Enables reading of secondary PBR replacement textures in pixel shaders when supported.\n"
                 "Current support is limitted to fixed function pipelines and programmable shaders with Shader Model 1.0.\n"
                 "When set to false or unsupported, an extra compute shader is used to preproces the secondary textures to make them compatible at an expense of performance and quality instead.\n"
                 "Requires \"rtx.terrainBaker.material.replacementSupportInPS_fixedFunction = True\" to apply for draw calls with fixed function graphics pipeline.\n"
                 "Requires \"rtx.terrainBaker.material.replacementSupportInPS_programmableShaders = True\" to apply for draw calls with programmable graphics pipeline.");
      RTX_OPTION("rtx.terrainBaker.material", bool, replacementSupportInPS_fixedFunction, true, 
                 "Enables reading of secondary PBR replacement textures in pixel shaders for games with fixed function graphics pipelines.\n"
                 "When set to false, an extra compute shader is used to preproces the secondary textures to make them compatible at an expense of performance and quality instead.\n"
                 "This parameter must be set at launch to apply.");
      RTX_OPTION_ENV("rtx.terrainBaker.material", bool, replacementSupportInPS_programmableShaders, true, "RTX_TERRAIN_BAKER_PS_REPLACEMENT_SUPPORT_IN_PROGRAMMABLE_SHADERS",
                 "[Experimental] Enables reading of secondary PBR replacement textures in pixel shaders for games with programmable graphics pipelines.\""
                 "When set to false, an extra compute shader is used to preproces the secondary textures to make them compatible at an expense of performance and quality instead.\n"
                 "This parameter must be set at launch to apply. The current support for this is limitted to draw calls with programmable shaders with Shader Model 1.0 only.\n"
                 "Draw calls with Shader Model 2.0+ will use the preprocessing compute pass.");
      RTX_OPTION("rtx.terrainBaker.material", bool, bakeReplacementMaterials, true, "Enables baking of replacement materials when they are present.");
      RTX_OPTION("rtx.terrainBaker.material", bool, bakeMaterialConstants, true,
                 "Bakes a replacement material's roughness or metallic constant where that material has no texture for it.\n"
                 "Without this, a terrain surface carrying a constant contributes nothing to that texture's cascade, so the\n"
                 "texels it covers keep the cascade's clear value - one value for every such surface in the world.\n"
                 "A material with no normal map bakes a flat normal the same way, so that it covers the normals of the\n"
                 "layers beneath it rather than letting them show through.\n"
                 "A constant costs the draw call an extra baking pass, so turning this off trades per-material accuracy for\n"
                 "baking cost.");
      // ToDo disable by default
      RTX_OPTION_ENV("rtx.terrainBaker.material", bool, bakeSecondaryPBRTextures, true, "RTX_TERRAIN_BAKER_BAKE_SECONDARY_PBR_TEXTURES", 
                     "Enables baking of secondary textures in replacement materials when they are present.\n"
                     "Secondary textures are all PBR textures except for albedoOpacity. So that includes normal, roughness, etc.");
      RTX_OPTION("rtx.terrainBaker.material", uint32_t, maxResolutionToUseForReplacementMaterials, 8192, 
                 "Max resolution to use for preprocessing and baking of input replacement material textures other than color opacity which is used as is.\n"
                 "Applies only to a case when a preprocessing compute shader is used to support baking of secondary PBR materials.\n"
                 "Replacement materials need to be preprocessed prior to baking them and limitting the max resolution allows to balance the quality vs performance cost.");

      struct Properties {
        friend class ImGUI;
        friend class TerrainBaker; // <-- we want to modify these values directly.

        RTX_OPTION("rtx.terrainBaker.material.properties", float, roughnessAnisotropy, 0.f, "Roughness anisotropy. Valid range is <-1, 1>, where 0 is isotropic.");
        RTX_OPTION("rtx.terrainBaker.material.properties", float, emissiveIntensity, 0.f, "Emissive intensity.");
        RTX_OPTION("rtx.terrainBaker.material.properties", float, roughnessConstant, 0.7f, "Perceptual roughness constant. Valid range is <0, 1>.");
        RTX_OPTION("rtx.terrainBaker.material.properties", float, metallicConstant, 0.1f, "Metallic constant. Valid range is <0, 1>.");
        RTX_OPTION("rtx.terrainBaker.material.properties", Vector3, emissiveColorConstant, Vector3(0.0f, 0.0f, 0.0f), "Emissive color constant. Should be a color in sRGB colorspace with gamma encoding.");
        RTX_OPTION("rtx.terrainBaker.material.properties", bool, enableEmission, false, "A flag to determine if emission is enabled.");
        RTX_OPTION("rtx.terrainBaker.material.properties", float, displaceInFactor, 1.f,
                   "The max depth and height the baked terrain can support will be larger than the max \n"
                   "of any incoming draw call, which results in a loss of detail. When this is \n"
                   "too low, the displacement will lack detail. When it is too high, the lowest \n"
                   "and highest parts of the POM will flatten out.  This affects both displaceIn \n"
                   "and displaceOut, despite the name.");
      };
    };

    static struct CascadeMap {
      friend class ImGUI; // <-- we want to modify these values directly.
      friend class TerrainBaker; // <-- we want to modify these values directly.

      RTX_OPTION("rtx.terrainBaker.cascadeMap", bool, useTerrainBBOX, true, "Uses terrain's bounding box to calculate the cascade map's scene footprint.");
      RTX_OPTION("rtx.terrainBaker.cascadeMap", float, defaultHalfWidth, 1000.f, "Cascade map square's default half width around the camera [meters]. Used when the terrain's BBOX couldn't be estimated.");
      RTX_OPTION("rtx.terrainBaker.cascadeMap", float, defaultHeight, 1000.f, "Cascade map baker's camera default height above the in-game camera [meters]. Used when the terrain's BBOX couldn't be estimated.");
      RTX_OPTION("rtx.terrainBaker.cascadeMap", float, levelHalfWidth, 10.f, "First cascade level square's half width around the camera [meters].");
      RTX_OPTION_ARGS("rtx.terrainBaker.cascadeMap", uint32_t, maxLevels, 8, "Max number of cascade levels.",
                      args.minValue = 1,
                      args.maxValue = 16,
                      args.environment = "RTX_TERRAIN_BAKER_MAX_CASCADE_LEVELS");
      RTX_OPTION_ARGS("rtx.terrainBaker.cascadeMap", uint32_t, levelResolution, 4096, "Texture resolution per cascade level.",
                      args.minValue = 1,
                      args.maxValue = 32 * 1024,
                      args.environment = "RTX_TERRAIN_BAKER_LEVEL_RESOLUTION");
      RTX_OPTION("rtx.terrainBaker.cascadeMap", float, recenterDistance, 5.f,
                 "How far the camera may move from the cascade map's center before the map is recentered on it [meters].\n"
                 "Baked texels stay valid for as long as the map keeps its center and layout, so a terrain draw call is baked\n"
                 "once and then reused until the map is recentered or the draw's inputs change. Recentering rebakes every\n"
                 "terrain draw call. Must stay below the first cascade level's half width, since the camera can sit this far\n"
                 "from the center of the level that holds the most detail. 0 recenters whenever the camera moves.");
      RTX_OPTION_ARGS("rtx.terrainBaker.cascadeMap", uint32_t, sideProjectionLevels, 0,
                      "Number of cascade levels that are also baked from the four horizontal directions and from below.\n"
                      "The cascade map is a top-down projection, so a near-vertical face gets one column of texels smeared\n"
                      "down its height, and a surface with another above it (a floor under a ceiling) shares its texels\n"
                      "with it. Side projections hold those surfaces at their own resolution, and each ray hit picks the\n"
                      "projection its surface faces most. A projection is only used where it saw the hit surface itself,\n"
                      "checked against its depth, so an occluded surface falls back to the top-down projection.\n"
                      "The projections are baked at half the level resolution, only for draw calls that have faces\n"
                      "steep enough to need them, and only created once such a draw call is seen. 0 disables them.",
                      args.minValue = 0,
                      args.maxValue = kMaxTerrainSideProjectionLevels);
      RTX_OPTION("rtx.terrainBaker.cascadeMap", TerrainSideProjectionCulling, sideProjectionCulling, TerrainSideProjectionCulling::Back,
                 "Faces a side projection culls, in terms of the terrain draw call's own front face. Back keeps the\n"
                 "surfaces facing each projection. Switch to Front if the Terrain Cascade Map debug view shows the side\n"
                 "projections holding the faces turned away from them instead.");
      RTX_OPTION("rtx.terrainBaker.cascadeMap", float, sideProjectionDepthTolerance, 0.05f,
                 "How far a hit may be from the surface a side projection baked at its texel and still sample it [meters].\n"
                 "Added to an allowance for the surface's slope across the texel.");
      RTX_OPTION("rtx.terrainBaker.cascadeMap", bool, expandLastCascade, true,
                 "Expands the last cascade's footprint to cover the whole cascade map.\n"
                 "This ensures whole terrain surface has valid baked texture data to sample from\n"
                 "even if there isn't enough cascades generated (due to the current settings or limitations).");
    } cascadeMap;
    
    // RTX OPTIONS

  private:
    struct BakingParameters {
      uint32_t numCascades;
      uint32_t numSideProjectionLevels;
      // Side projection views are packed four to a tile, in the first tiles of the map, so that
      // the depth buffer they are baked with only needs to cover the map's first row.
      uint32_t numSideProjectionTiles;
      uint2 cascadeMapSize;
      VkExtent2D cascadeLevelResolution;
      VkExtent2D cascadeMapResolution;

      Matrix4 sceneView;  // View matrix for a camera looking along scene's forward axis
      std::vector<Matrix4> bakingCameraOrthoProjection;  // Ortho projections to bake for all cascades
      // Every side projection view, indexed by level * kNumTerrainSideProjectionViews + view
      std::vector<Matrix4> sideProjectionView;
      std::vector<Matrix4> sideProjectionOrthoProjection;
      Vector3 sideProjectionViewDirection[kNumTerrainSideProjectionViews];  // Towards each view's camera
      Matrix4 viewToCascade0TextureSpace; // Matrix transforming viwe coordinates to 1st cascade texture space
      float zNear;
      float zFar;
      float lastCascadeScale; // Scale applied on last cascade's size to expand it to cover the whole cascade map span
      uint32_t frameIndex = kInvalidFrameIndex;  // Frame index for which the parameters have been calculated
    };
    TextureRef* getConstantTexture(Rc<DxvkContext>& ctx, float value);
    void reportSurfaceOrientation(const DrawCallState& drawCallState);
    void keepReplacementTexturesResident(Rc<RtxContext> ctx, const DrawCallState& drawCallState, OpaqueMaterialData& replacementMaterial);
    bool gatherAndPreprocessReplacementTextures(Rc<RtxContext> ctx, const DrawCallState& drawCallState, OpaqueMaterialData* replacementMaterial, std::vector<RtxGeometryUtils::TextureConversionInfo>& replacementTextures);
    void updateMaterialData(Rc<RtxContext> ctx);
    void onFrameBegin(Rc<RtxContext> ctx, const DxvkContextState& dxvkCtxState);
    // Sets up this frame's baking state on the first terrain draw that may bake, and
    // returns whether baking can proceed. False means the main camera for this frame has
    // not been established yet, so the cascade parameters cannot be calculated and the
    // caller must leave the draw unbaked.
    bool registerTerrainMesh(Rc<RtxContext> ctx, const DxvkContextState& dxvkCtxState, const DrawCallState& drawCallState);
    void calculateTerrainBBOX(const uint32_t currentFrameIndex);
    void calculateBakingParameters(Rc<RtxContext> ctx, const DxvkContextState& dxvkCtxState);
    void updateCascadeCenter(const RtCamera& camera);
    struct LayoutPart {
      enum Enum : uint32_t { SceneView, TopDownProjection, Resolution, SideProjections, Displacement, MaterialOptions, Count };
    };
    using LayoutHashes = std::array<XXH64_hash_t, LayoutPart::Count>;
    LayoutHashes calculateBakedLayoutHashes() const;
    void reportLayoutChanges(const LayoutHashes& layout);
    void flushLayoutChangeReport();
    void beginBakedContent(Rc<RtxContext> ctx);
    static XXH64_hash_t calculateDrawKey(const DrawCallState& drawCallState, const OpaqueMaterialData* replacementMaterial,
                                         const Matrix4& world, const D3D9FixedFunctionVS* fixedFunctionVS,
                                         const D3D9SharedPS& sharedPS);
    void accountDisplacement(const OpaqueMaterialData& replacementMaterial);

    // Triangle counts of a terrain geometry by which projection represents them best. Upward and
    // downward follow the geometry's winding, whose sense is only known across all terrain.
    struct SurfaceOrientation {
      uint32_t upward = 0;
      uint32_t downward = 0;
      uint32_t steep = 0;
    };
    const SurfaceOrientation& getSurfaceOrientation(const DrawCallState& drawCallState);
    bool needsHorizontalSideProjections(const SurfaceOrientation& orientation) const;
    bool needsSideProjectionFromBelow(const SurfaceOrientation& orientation) const;
    void calculateSideProjectionParameters(const Vector3& cascadeCenter, float zNear);
    VkRect2D getTopDownViewRect(uint32_t level) const;
    VkRect2D getSideProjectionViewRect(uint32_t sideView) const;
    void updateSideProjectionDepth(Rc<DxvkContext> ctx);
    void releaseSideProjectionDepth(RtxTextureManager& textureManager);
    void updateTextureFormat(const DxvkContextState& dxvkCtxState);
    void calculateCascadeMapResolution(const Rc<DxvkDevice>& device);
    const RtxMipmap::Resource& getTerrainTexture(Rc<DxvkContext> ctx, RtxTextureManager& textureManager, ReplacementMaterialTextureType::Enum textureType, uint32_t width, uint32_t height);
    void clearMaterialTexture(Rc<DxvkContext> ctx, ReplacementMaterialTextureType::Enum textureType);
    static bool isPSReplacementSupportEnabled(const DrawCallState& drawCallState);
    VkClearColorValue getClearColor(ReplacementMaterialTextureType::Enum textureType);

    BakingParameters m_bakingParams;

    // The cascade map's center. It is held while the camera stays within recenterDistance of it,
    // which is what keeps the baked texels valid from one frame to the next.
    std::optional<Vector3> m_cascadeCenter;

    // The cascade layout the baked texels were rendered for, and the draw calls already baked
    // into it. A draw call whose key is in the set has nothing new to write and is not baked
    // again. Both reset whenever the layout changes, which rebakes everything.
    LayoutHashes m_bakedLayout {};
    std::array<uint32_t, LayoutPart::Count> m_layoutChangeCounts {};
    uint32_t m_layoutChanges = 0;
    uint32_t m_framesSinceLayoutReport = 0;
    std::unordered_set<XXH64_hash_t> m_bakedDraws;

    // A newly created terrain texture holds only its clear value, including where draw calls
    // that were reused rather than baked this frame should have written. Forces a full rebake.
    bool m_bakedContentLost = false;

    // Latched once a baked draw call has faces the top-down projection cannot represent. The side
    // projections cost memory and baking time, so a scene without such faces never creates them.
    bool m_sideProjectionsNeeded = false;
    std::unordered_map<XXH64_hash_t, SurfaceOrientation> m_surfaceOrientations;
    // Upward minus downward triangles over all terrain seen. Terrain faces up far more than down, so
    // its sign says which winding faces up.
    int64_t m_upwardTriangleBalance = 0;

    // Depth the side projections are baked with, and read back from to tell whether a hit is the
    // surface a projection holds at that texel. Covers the map's first row of tiles only.
    Resources::Resource m_sideProjectionDepth;
    Rc<DxvkImageView> m_sideProjectionDepthTarget;
    uint32_t m_sideProjectionDepthTextureIndex = kSurfaceMaterialInvalidTextureIndex;

    uint32_t m_numDrawsBakedThisFrame = 0;
    uint32_t m_numDrawsReusedThisFrame = 0;
    uint32_t m_numDrawsBakedLastFrame = 0;
    uint32_t m_numDrawsReusedLastFrame = 0;

    struct TextureKey {
      uint16_t width;
      uint16_t height;
      uint16_t /*ReplacementMaterialTextureType::Enum*/ textureType;
      
      XXH64_hash_t calculateHash() const {
        return hashStructByMemory<TextureKey,
            &TextureKey::width,
            &TextureKey::height,
            &TextureKey::textureType>(*this);
      }
    };

    // Replacement textures converted for baking, each with a full mip chain: the bake draws them at the
    // cascade's texel density, typically far below their own, and without mips they would alias.
    fast_unordered_cache<RtxMipmap::Resource> m_stagingTextureCache;

    // A terrain material may carry a constant where another carries a texture -- a roughness
    // number rather than a roughness map. Materialising that constant as a 1x1 texture lets it
    // bake through exactly the path a real texture takes, so the draw writes its own value over
    // its own footprint with the blend weight the albedo pass used. Keyed on the quantised
    // value, so the handful of distinct constants in a scene share one image.
    struct ConstantTexture {
      Resources::Resource resource;
      TextureRef texture;
    };
    fast_unordered_cache<ConstantTexture> m_constantTextureCache;

    // Geometries already reported by logSurfaceOrientation. The same terrain patch is redrawn
    // every frame, so without this the log is one line per draw call per frame.
    std::unordered_set<XXH64_hash_t> m_loggedOrientations;

    VkFormat m_terrainRtColorFormat = VK_FORMAT_UNDEFINED;

    class AxisAlignedBoundingBoxLink {
    public:
      AxisAlignedBoundingBoxLink(const DrawCallState& drawCallState);
      AxisAlignedBoundingBox calculateAABBInWorldSpace();

    private:
      const AxisAlignedBoundingBox aabbObjectSpace;
      const Matrix4 objectToWorld;
    };

    std::list<AxisAlignedBoundingBoxLink> m_terrainMeshBBOXes;

    // Terrain BBOX found during previous frame
    AxisAlignedBoundingBox m_bakedTerrainBBOX = {
      -Vector3(cascadeMap.defaultHalfWidth()),
      Vector3(cascadeMap.defaultHalfWidth()) };          

    // Frame index for which the terrain BBOX has been calculated
    // -1 to avoid aliasing on frame 0, since this value is checked for a match a frame later
    uint32_t m_terrainBBOXFrameIndex = kInvalidFrameIndex - 1;

    // Since blending of anything but the material textures is not supported 
    // baked terrain uses the first material data properties it sees in a frame
    bool m_hasInitializedMaterialDataThisFrame = false;

    // Made optional since it requires valid explicit parameters specified on construction
    std::optional<MaterialData> m_materialData;

    float m_currFrameMaxDisplaceIn = 0.f;
    float m_prevFrameMaxDisplaceIn = 0.f;

    float m_currFrameMaxDisplaceOut = 0.f;
    float m_prevFrameMaxDisplaceOut = 0.f;

    // Set to true when m_materialData needs to be updated to reflect latest changes.
    bool m_needsMaterialDataUpdate = false;

    // Tracks per-frame cascade image creation/resizing. Cleared in onFrameEnd.
    bool m_cascadeCompositionChangedThisFrame = false;
    
    // Set to true when a button is clicked in the GUI, tracks all incoming uv densities for a frame and changes displaceInFactor to the resulting value.
    bool m_calculatingDisplaceInFactor = false;
    float m_calculatedDisplaceInFactor = 1.0f;
    // UI button click may come in mid frame, so set a transient bool, then start actually calculating it next frame.
    mutable bool m_calculateDisplaceInFactorNextFrame = false;

    struct BakedTexture {
      // Baked texture needs to be retained for at least a frame after it was baked so that 
      // a bound terrain material is consistent (i.e. bounds same texture types) across all draw calls
      // even if a draw does not bake into all PBR textures that all draws combined in a frame do.
      // Note the consistency is only ensured for a frame in which no new texture types have been added.
      // 1: means current frame only
      const uint8_t kNumFramesToRetainBakedTexture = 2;
      
      RtxMipmap::Resource texture;
      uint8_t numFramesToRetain = 0;

      bool isBaked() {
        return numFramesToRetain > 0;
      }

      void markAsBaked() {
        numFramesToRetain = kNumFramesToRetainBakedTexture;
      }

      void onFrameEnd(Rc<DxvkContext> ctx);

    };

    BakedTexture m_materialTextures[ReplacementMaterialTextureType::Count];

    Rc<DxvkSampler> m_terrainSampler;
  };
}
