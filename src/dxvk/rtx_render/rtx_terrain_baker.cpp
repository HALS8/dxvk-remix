/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_terrain_baker.h"

#include "dxvk_device.h"
#include "../tracy/Tracy.hpp"
#include "dxvk_scoped_annotation.h"
#include "rtx_imgui.h"
#include "rtx_option.h"
#include "rtx_texture.h"
#include "rtx_texture_manager.h"

#include "../d3d9/d3d9_state.h"
#include "../d3d9/d3d9_spec_constants.h"
#include "../dxso/dxso_util.h"
#include "../../d3d9/d3d9_rtx.h"
#include "../../dxso/dxso_util.h"
#include "../../d3d9/d3d9_caps.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iomanip>

namespace {
  // By default, a value of 1.f will have 0 displacement.
  const float kDefaultNeutralHeight = 1.f;

  // Faces whose normal is further than this from vertical (as a cosine) may pick a horizontal side
  // projection, allowing for the band over which a hit dithers between projections.
  const float kSideProjectionSteepness = 0.8f;
}

namespace dxvk {

  uint32_t getMipLevels(ReplacementMaterialTextureType::Enum textureType, const VkExtent3D& extent) {
    switch (textureType) {
    case ReplacementMaterialTextureType::Height:
      // height maps need to be created with mip levels for quad tree POM support.
      return static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height))));
      break;

    default:
      return 1;
      break;
    }
  }

  VkFormat getTextureFormat(ReplacementMaterialTextureType::Enum textureType) {
    switch (textureType) {
    case ReplacementMaterialTextureType::Normal:
    case ReplacementMaterialTextureType::Tangent:
      return VK_FORMAT_R8G8B8A8_SNORM;
      break;

    case ReplacementMaterialTextureType::AlbedoOpacity:
    case ReplacementMaterialTextureType::Emissive:
      return VK_FORMAT_R8G8B8A8_UNORM;
      break;

      // R16
    case ReplacementMaterialTextureType::Height:
    case ReplacementMaterialTextureType::Roughness:
    case ReplacementMaterialTextureType::Metallic:
      return VK_FORMAT_R8_UNORM;
      break;

    default:
      assert(0);
      return VK_FORMAT_UNDEFINED;
      break;
    }
  }

  VkClearColorValue TerrainBaker::getClearColor(ReplacementMaterialTextureType::Enum textureType) {
    const float prevFrameTotalHeight = m_prevFrameMaxDisplaceIn + m_prevFrameMaxDisplaceOut;
    float neutral_height = prevFrameTotalHeight != 0.f ? m_prevFrameMaxDisplaceIn / prevFrameTotalHeight : kDefaultNeutralHeight;
    switch (textureType) {
    case ReplacementMaterialTextureType::Height:
      // height maps should be cleared to neutral_height, which keeps the displaced surface identical to the original surface.
      // The height texture should be single channel, so only the first value actually matters.
      return { neutral_height, neutral_height, neutral_height, neutral_height };
      break;

    case ReplacementMaterialTextureType::Roughness:
      // A terrain material without a roughness texture bakes nothing into this cascade, and a
      // texel no draw has written reads back as the cleared value. Cleared to zero that is a
      // perfect mirror wherever the terrain happens to have no roughness map, so the cascade
      // starts at the constant the material would have used had it carried no texture at all.
      return { Material::Properties::roughnessConstant(), Material::Properties::roughnessConstant(),
               Material::Properties::roughnessConstant(), Material::Properties::roughnessConstant() };
      break;

    case ReplacementMaterialTextureType::Metallic:
      // Same reasoning as roughness: unwritten metallic must read as the material constant.
      return { Material::Properties::metallicConstant(), Material::Properties::metallicConstant(),
               Material::Properties::metallicConstant(), Material::Properties::metallicConstant() };
      break;

    default:
      return { 0.0f, 0.0f, 0.0f, 0.0f };
      break;
    }
  }

  TextureRef* TerrainBaker::getConstantTexture(Rc<DxvkContext>& ctx, float value) {
    // Eight bits is the precision the baked cascade keeps anyway, and quantising here is what
    // lets materials that authored the same value share one image.
    const uint32_t quantized = static_cast<uint32_t>(fclamp(value, 0.f, 1.f) * 255.f + 0.5f);

    auto entry = m_constantTextureCache.find(quantized);
    if (entry == m_constantTextureCache.end()) {
      const float level = quantized / 255.f;
      const VkClearColorValue clearValue = { level, level, level, level };
      Resources::Resource resource = Resources::createImageResource(
        ctx, "terrain baking: material constant", VkExtent3D { 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        1, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 0, VK_IMAGE_USAGE_STORAGE_BIT, clearValue);

      if (resource.view == nullptr) {
        ONCE(Logger::err("[RTX Terrain Baker] Failed to create a texture for a material constant. "
                         "Materials with no texture for an input keep the cascade's clear value."));
        return nullptr;
      }

      entry = m_constantTextureCache.emplace(quantized, ConstantTexture()).first;
      entry->second.resource = std::move(resource);
      entry->second.texture = TextureRef(entry->second.resource.view);
    }
    return &entry->second.texture;
  }

  bool TerrainBaker::isPSReplacementSupportEnabled(const DrawCallState& drawCallState) {
    if (drawCallState.usesPixelShader) {
      return Material::replacementSupportInPS() && 
             Material::replacementSupportInPS_programmableShaders() &&
             drawCallState.programmablePixelShaderInfo.majorVersion() <= 1;
    } else {
      return Material::replacementSupportInPS() && Material::replacementSupportInPS_fixedFunction();
    }
  }

  // Keeps the replacement textures a draw call bakes from in video memory. Tracking must precede
  // any check for a valid view, since a texture gets its views only once it is promoted. A draw
  // call reusing its baked texels still tracks them: an untracked texture is demoted, and the new
  // view would change the draw's key and force a rebake at lower detail.
  void TerrainBaker::keepReplacementTexturesResident(Rc<RtxContext> ctx, const DrawCallState& drawCallState,
                                                     OpaqueMaterialData& replacementMaterial) {
    SceneManager& sceneManager = ctx->getSceneManager();
    const bool hasTexcoords = drawCallState.hasTextureCoordinates();

    auto track = [&](TextureRef& texture) {
      if (texture.isValid()) {
        uint32_t unusedTextureIndex;
        sceneManager.trackTexture(texture, unusedTextureIndex, hasTexcoords);
      }
    };

    track(replacementMaterial.getAlbedoOpacityTexture());

    if (Material::bakeSecondaryPBRTextures()) {
      track(replacementMaterial.getNormalTexture());
      track(replacementMaterial.getTangentTexture());
      track(replacementMaterial.getHeightTexture());
      track(replacementMaterial.getRoughnessTexture());
      track(replacementMaterial.getMetallicTexture());
      track(replacementMaterial.getEmissiveColorTexture());
    }
  }

  // Gathers available textures from a replacement material and
  // runs a compute shader to convert them into a compatible format for baking
  bool TerrainBaker::gatherAndPreprocessReplacementTextures(Rc<RtxContext> ctx,
                                                            const DrawCallState& drawCallState,
                                                            OpaqueMaterialData* replacementMaterial,
                                                            std::vector<RtxGeometryUtils::TextureConversionInfo>& replacementTextures) {
    if (!replacementMaterial) {
      return false;
    }

    Resources& resourceManager = ctx->getResourceManager();
    // We're going to use this to create a modified sampler for textures.
    DxvkSampler* pOriginalSampler = drawCallState.getMaterialData().getSampler().ptr();
    Rc<DxvkContext> dxvkCtx = ctx;

    // Opacity texture is currently required for blending to work. 
    // Scenarios where blending does not require a colorOpacity texture or 
    // replacement material is using a colorOpacity constant are not currently supported
    if (!replacementMaterial->getAlbedoOpacityTexture().isValid()) {
      ONCE(Logger::warn(str::format("[RTX Texture Baker] Replacement material for ", drawCallState.getMaterialData().getHash(), " does not have a color opacity texture.",
                                    " This scenario is not currently supported by the texture baker. Ignoring the replacement material.")));
      return false;
    }

    if (!drawCallState.getMaterialData().getColorTexture2().isValid()) {
      ONCE(Logger::warn(str::format("[RTX Texture Baker] Legacy material for ", drawCallState.getMaterialData().getHash(), " has a second color texture.",
                                    "Only single texture legacy materials are supported. Ignoring the second color texture.")));
    }

    keepReplacementTexturesResident(ctx, drawCallState, *replacementMaterial);

    const DxvkImageCreateInfo& aoImageInfo = replacementMaterial->getAlbedoOpacityTexture().getImageView()->imageInfo();

    // Returns a scaled down the extent that fits within the max resolution constraint preserving the aspect ratio (barring float to integer conversion errors)
    auto calculateScaledResolution2D = [&](VkExtent3D extent, const uint32_t maxResolutionPerDimension) {
      const float scalingFactor = 
        std::min(
          1.f,     // Don't scale up the input dimensions
          1 / std::max(
            extent.width / static_cast<float>(maxResolutionPerDimension),
            extent.height / static_cast<float>(maxResolutionPerDimension)));

      extent.width = static_cast<uint32_t>(extent.width * scalingFactor);
      extent.height = static_cast<uint32_t>(extent.height * scalingFactor);

      return extent;
    };

    // Staging textures written by the conversion below, whose mip chains are generated once it has run
    std::vector<std::pair<size_t, RtxMipmap::Resource>> stagedTextures;

    auto addValidTexture = [&](TextureRef& texture, ReplacementMaterialTextureType::Enum textureType) {

      if (!texture.isValid() || !texture.getImageView()) {
        return;
      }

      RtxGeometryUtils::TextureConversionInfo& conversionInfo = replacementTextures.emplace_back();
      conversionInfo.type = textureType;
      conversionInfo.sourceTexture = &texture;
      
      if (textureType == ReplacementMaterialTextureType::Height) {
        // Normalize the displaceIn and displaceOut to the previous frame's displacement range.
        const float prevFrameTotalHeight = m_prevFrameMaxDisplaceIn + m_prevFrameMaxDisplaceOut;
        const float materialTotalHeight = replacementMaterial->getDisplaceIn() + replacementMaterial->getDisplaceOut();

        conversionInfo.scale = prevFrameTotalHeight <= 0.f ? 0.f : materialTotalHeight / prevFrameTotalHeight;
        // We want to subtract the original neutral displacement, then scale the values, then add the new neutral displacement.
        conversionInfo.offset = -1.f * (materialTotalHeight == 0.f ? kDefaultNeutralHeight : (replacementMaterial->getDisplaceIn() / materialTotalHeight));
      }

      if (isPSReplacementSupportEnabled(drawCallState)) {
        conversionInfo.targetTexture = TextureRef(texture.getImageView());
      } else {
        const DxvkImageCreateInfo& imageInfo = texture.getImageView()->imageInfo();
        const VkExtent3D& extent = imageInfo.extent;

        const VkExtent3D adjustedExtent = calculateScaledResolution2D(extent, Material::maxResolutionToUseForReplacementMaterials());

        TextureKey textureKey;
        textureKey.width = adjustedExtent.width;
        textureKey.height = adjustedExtent.height;
        textureKey.textureType = textureType;
        XXH64_hash_t textureKeyHash = textureKey.calculateHash();

        auto textureIter = m_stagingTextureCache.find(textureKeyHash);

        // Staging texture must be 4 channel as the 4th channel will contain opacity
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

        if (textureType == ReplacementMaterialTextureType::Normal ||
            textureType == ReplacementMaterialTextureType::Tangent) {
          format = VK_FORMAT_R8G8B8A8_SNORM;
        }

        // No matching cached texture found, create a new one
        if (textureIter == m_stagingTextureCache.end()) {
          const uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(adjustedExtent.width, adjustedExtent.height)))) + 1;
          textureIter =
            m_stagingTextureCache.emplace(
              textureKeyHash,
              RtxMipmap::createResource(dxvkCtx, "terrain baking: staging replacement texture", adjustedExtent,
                                        format, 0, VkClearColorValue {}, mipLevels)).first;
        }

        // The conversion writes the top level; the bake samples the whole chain once it is generated
        const RtxMipmap::Resource& staged = textureIter->second;
        conversionInfo.targetTexture = TextureRef(staged.views.empty() ? staged.view : staged.views[0]);
        stagedTextures.emplace_back(replacementTextures.size() - 1, staged);

        // Track lifetime of the resource now since targetTexture object is about to get destroyed
        ctx->getCommandList()->trackResource<DxvkAccess::Write>(textureIter->second.image);
      }
    };

    // A material with no texture for an input still has a value for it. Baking that value over the
    // draw's own footprint is what keeps one surface's roughness off another's, which the
    // cascade's clear cannot do: it holds a single value for every surface that writes nothing.
    auto addTextureOrConstant = [&](TextureRef& texture, ReplacementMaterialTextureType::Enum textureType,
                                    float constant) {
      if (texture.isValid()) {
        addValidTexture(texture, textureType);
        return;
      }
      if (!Material::bakeMaterialConstants()) {
        return;
      }
      if (TextureRef* constantTexture = getConstantTexture(dxvkCtx, constant)) {
        addValidTexture(*constantTexture, textureType);
      }
    };

    // Gather all replacement textures that need to be preprocessed
    replacementTextures.reserve(ReplacementMaterialTextureType::Count);

    if (Material::bakeSecondaryPBRTextures()) {
      // A layer without a normal map is flat, and must cover the normals of the layers beneath it
      // as its albedo covers theirs: (0.5, 0.5) is the octahedral encoding of the unperturbed normal.
      constexpr float kFlatOctahedralNormal = 0.5f;
      addTextureOrConstant(replacementMaterial->getNormalTexture(), ReplacementMaterialTextureType::Normal,
                           kFlatOctahedralNormal);
      addValidTexture(replacementMaterial->getTangentTexture(), ReplacementMaterialTextureType::Tangent);
      addValidTexture(replacementMaterial->getHeightTexture(), ReplacementMaterialTextureType::Height);
      addTextureOrConstant(replacementMaterial->getRoughnessTexture(), ReplacementMaterialTextureType::Roughness,
                           replacementMaterial->getRoughnessConstant());
      addTextureOrConstant(replacementMaterial->getMetallicTexture(), ReplacementMaterialTextureType::Metallic,
                           replacementMaterial->getMetallicConstant());
      addValidTexture(replacementMaterial->getEmissiveColorTexture(), ReplacementMaterialTextureType::Emissive);


      if (!isPSReplacementSupportEnabled(drawCallState)) {
        // Pre-process textures to be compatible with baking
        ctx->getCommonObjects()->metaGeometryUtils().decodeAndAddOpacity(ctx, replacementMaterial->getAlbedoOpacityTexture(), replacementTextures);

        for (auto& [index, staged] : stagedTextures) {
          RtxMipmap::updateMipmap(ctx, staged, MipmapMethod::Simple);
          replacementTextures[index].targetTexture = TextureRef(staged.view);
        }
      }
    }

    // Add the remaining albedo opacity which does not needed to be preprocessed to the texture list for baking.
    RtxGeometryUtils::TextureConversionInfo& conversionInfo = replacementTextures.emplace_back();
    conversionInfo.type = ReplacementMaterialTextureType::AlbedoOpacity;
    conversionInfo.sourceTexture = nullptr;
    conversionInfo.targetTexture = replacementMaterial->getAlbedoOpacityTexture();

    // Move albedo opacity to the front of the baking queue as the baking aborts if baking of albedo opacity texture fails
    if (replacementTextures.size() > 1) {
      std::swap(replacementTextures.front(), replacementTextures.back());
    }

    return true;
  }

  // Every draw call that would bake a height map widens the displacement range the height cascade
  // is normalized to, whether it bakes this frame or reuses what it baked earlier. Counting only
  // the draws that bake would shrink the range as soon as they are reused, and the changed range
  // would invalidate the cascade on the next frame.
  void TerrainBaker::accountDisplacement(const OpaqueMaterialData& replacementMaterial) {
    if (!Material::bakeSecondaryPBRTextures() ||
        !replacementMaterial.getAlbedoOpacityTexture().isValid() ||
        replacementMaterial.getHeightTexture().getImageView() == nullptr) {
      return;
    }
    m_currFrameMaxDisplaceIn = std::max(m_currFrameMaxDisplaceIn, replacementMaterial.getDisplaceIn());
    m_currFrameMaxDisplaceOut = std::max(m_currFrameMaxDisplaceOut, replacementMaterial.getDisplaceOut());
  }

  // Identifies everything a draw call writes into the cascade map, so that a draw call whose key
  // was already baked into the current cascade layout can reuse those texels. Replacement textures
  // are keyed by their current image view rather than their content: a streamed texture changes
  // view when it gains mip levels, and the draw call should then bake again at the new detail.
  XXH64_hash_t TerrainBaker::calculateDrawKey(const DrawCallState& drawCallState,
                                              const OpaqueMaterialData* replacementMaterial,
                                              const Matrix4& world,
                                              const D3D9FixedFunctionVS* fixedFunctionVS,
                                              const D3D9SharedPS& sharedPS) {
    auto combine = [](XXH64_hash_t hash, const void* data, size_t size) {
      return XXH64(data, size, hash);
    };

    XXH64_hash_t key = drawCallState.getGeometryData().getHashForRule<rules::FullGeometryHash>();
    key = combine(key, &world, sizeof(world));

    const XXH64_hash_t legacyMaterial = drawCallState.getMaterialData().computeIdentityHash();
    key = combine(key, &legacyMaterial, sizeof(legacyMaterial));

    if (fixedFunctionVS != nullptr) {
      key = combine(key, &fixedFunctionVS->Material, sizeof(fixedFunctionVS->Material));
      key = combine(key, fixedFunctionVS->TexcoordMatrices.data(), sizeof(fixedFunctionVS->TexcoordMatrices));
    }

    for (const D3D9SharedPS::Stage& stage : sharedPS.Stages) {
      key = combine(key, stage.Constant, sizeof(stage.Constant));
      key = combine(key, stage.BumpEnvMat, sizeof(stage.BumpEnvMat));
      key = combine(key, &stage.BumpEnvLScale, sizeof(stage.BumpEnvLScale));
      key = combine(key, &stage.BumpEnvLOffset, sizeof(stage.BumpEnvLOffset));
    }

    if (replacementMaterial != nullptr) {
      const TextureRef* textures[] = {
        &replacementMaterial->getAlbedoOpacityTexture(), &replacementMaterial->getNormalTexture(),
        &replacementMaterial->getTangentTexture(), &replacementMaterial->getHeightTexture(),
        &replacementMaterial->getRoughnessTexture(), &replacementMaterial->getMetallicTexture(),
        &replacementMaterial->getEmissiveColorTexture() };
      for (const TextureRef* texture : textures) {
        const DxvkImageView* view = texture->getImageView();
        key = combine(key, &view, sizeof(view));
      }
      const float constants[] = { replacementMaterial->getRoughnessConstant(), replacementMaterial->getMetallicConstant() };
      key = combine(key, constants, sizeof(constants));
    }

    return key;
  }

  // Calls visit(up) for every non-degenerate triangle of the draw call, with the cosine of the angle
  // between its world space face normal and the scene's up axis. Its sign follows the winding. Returns
  // false when the geometry has no CPU-side positions to measure.
  template<typename Visitor>
  static bool visitTriangleUpComponents(const DrawCallState& drawCallState, uint32_t& degenerate, Visitor&& visit) {
    const RasterGeometry& geometry = drawCallState.getGeometryData();
    const GeometryBufferData buffers(geometry);
    if (buffers.positionData == nullptr) {
      return false;
    }

    const Matrix4 objectToWorld = drawCallState.getTransformData().objectToWorld;
    const Vector3 upAxis = SceneManager::getSceneUp();

    const bool indexed = buffers.indexData != nullptr && geometry.indexCount >= 3;
    const uint32_t cornerCount = indexed ? geometry.indexCount : geometry.vertexCount;

    const auto worldPosition = [&](uint32_t index) {
      const float* p = &buffers.positionData[index * buffers.positionStride];
      const Vector4 world = objectToWorld * Vector4(p[0], p[1], p[2], 1.f);
      return Vector3(world.x, world.y, world.z);
    };

    for (uint32_t corner = 0; corner + 2 < cornerCount; corner += 3) {
      const uint32_t i0 = indexed ? buffers.indexData[(corner + 0) * buffers.indexStride] : corner + 0;
      const uint32_t i1 = indexed ? buffers.indexData[(corner + 1) * buffers.indexStride] : corner + 1;
      const uint32_t i2 = indexed ? buffers.indexData[(corner + 2) * buffers.indexStride] : corner + 2;

      if (i0 >= geometry.vertexCount || i1 >= geometry.vertexCount || i2 >= geometry.vertexCount) {
        ++degenerate;
        continue;
      }

      const Vector3 p0 = worldPosition(i0);
      const Vector3 faceNormal = cross(worldPosition(i1) - p0, worldPosition(i2) - p0);
      const float faceNormalLength = length(faceNormal);
      if (faceNormalLength <= 1e-8f) {
        ++degenerate;
        continue;
      }

      visit(dot(faceNormal, upAxis) / faceNormalLength);
    }

    return true;
  }

  void TerrainBaker::reportSurfaceOrientation(const DrawCallState& drawCallState) {
    const RasterGeometry& geometry = drawCallState.getGeometryData();

    // The same terrain patch is redrawn every frame; one line per distinct geometry is enough.
    if (!m_loggedOrientations.insert(geometry.hashes[HashComponents::VertexPosition]).second) {
      return;
    }

    // Bands are on |n . up|, so a consistent winding is not required to answer the question that
    // matters: whether one draw call mixes surfaces the cascade can represent with surfaces it
    // cannot. The signed count is reported separately and does depend on winding.
    uint32_t flat = 0, gentle = 0, steep = 0, vertical = 0, facingDown = 0, degenerate = 0;
    float minAbsUp = FLT_MAX;
    float maxAbsUp = -FLT_MAX;
    double sumAbsUp = 0.0;
    uint32_t triangles = 0;

    const bool measured = visitTriangleUpComponents(drawCallState, degenerate, [&](float up) {
      const float absUp = std::abs(up);

      ++triangles;
      sumAbsUp += absUp;
      minAbsUp = std::min(minAbsUp, absUp);
      maxAbsUp = std::max(maxAbsUp, absUp);

      if (absUp > 0.85f)      ++flat;
      else if (absUp > 0.50f) ++gentle;
      else if (absUp > 0.15f) ++steep;
      else                    ++vertical;

      if (up < -0.15f) {
        ++facingDown;
      }
    });

    if (!measured) {
      ONCE(Logger::warn("[RTX Terrain Baker] orientation: no CPU-side positions to measure."));
      return;
    }

    if (triangles == 0) {
      return;
    }

    const auto percent = [&](uint32_t count) { return 100.f * count / triangles; };

    Logger::info(str::format(
      "[RTX Terrain Baker] orientation: material 0x", std::hex, drawCallState.getMaterialData().getHash(), std::dec,
      " tris ", triangles,
      " | flat ", flat, " (", std::setprecision(3), percent(flat), "%)",
      " gentle ", gentle, " (", percent(gentle), "%)",
      " steep ", steep, " (", percent(steep), "%)",
      " vertical ", vertical, " (", percent(vertical), "%)",
      " | facingDown ", facingDown,
      " | absUp min ", minAbsUp, " mean ", static_cast<float>(sumAbsUp / triangles), " max ", maxAbsUp,
      " | degenerate ", degenerate,
      " cullMode ", static_cast<uint32_t>(geometry.cullMode),
      " frontFace ", static_cast<uint32_t>(geometry.frontFace)));
  }

  const TerrainBaker::SurfaceOrientation& TerrainBaker::getSurfaceOrientation(const DrawCallState& drawCallState) {
    const Matrix4& objectToWorld = drawCallState.getTransformData().objectToWorld;
    const XXH64_hash_t key = XXH64(&objectToWorld, sizeof(objectToWorld),
                                   drawCallState.getGeometryData().hashes[HashComponents::VertexPosition]);

    auto [entry, isNew] = m_surfaceOrientations.try_emplace(key);
    SurfaceOrientation& orientation = entry->second;
    if (!isNew) {
      return orientation;
    }

    // Classified the way a hit picks its projection: by the axis its normal is closest to
    uint32_t degenerate = 0;
    const bool measured = visitTriangleUpComponents(drawCallState, degenerate, [&](float up) {
      if (std::abs(up) < kSideProjectionSteepness) {
        ++orientation.steep;
      } else if (up > 0.f) {
        ++orientation.upward;
      } else {
        ++orientation.downward;
      }
    });

    // Without positions there is no telling, so the geometry gets every projection
    if (!measured) {
      orientation = SurfaceOrientation { 1, 1, 1 };
    }

    m_upwardTriangleBalance += static_cast<int64_t>(orientation.upward) - static_cast<int64_t>(orientation.downward);
    return orientation;
  }

  bool TerrainBaker::needsHorizontalSideProjections(const SurfaceOrientation& orientation) const {
    return orientation.steep > 0;
  }

  bool TerrainBaker::needsSideProjectionFromBelow(const SurfaceOrientation& orientation) const {
    const uint32_t facingDown = m_upwardTriangleBalance >= 0 ? orientation.downward : orientation.upward;
    return facingDown > 0;
  }

  bool TerrainBaker::bakeDrawCall(Rc<RtxContext> ctx,
                                  const DxvkContextState& dxvkCtxState,
                                  DxvkRaytracingInstanceState& rtState,
                                  const DrawParameters& drawParams,
                                  const DrawCallState& drawCallState,
                                  OpaqueMaterialData* replacementMaterial,
                                  Matrix4& textureTransformOut) {

    SceneManager& sceneManager = ctx->getSceneManager();
    Resources& resourceManager = ctx->getResourceManager();
    RtxTextureManager& textureManger = ctx->getCommonObjects()->getTextureManager();
    const RtCamera& camera = sceneManager.getCamera();

    if (drawCallState.usesVertexShader && !D3D9Rtx::useVertexCapture()) {
      ONCE(Logger::warn(str::format("[RTX Terrain Baker] Terrain texture corresponds to a draw call with programmable Vertex Shader usage. Vertex capture must be enabled to support baking of such draw calls. Ignoring the draw call.")));
      return false;
    }

    if (logSurfaceOrientation()) {
      reportSurfaceOrientation(drawCallState);
    }

    if (!Material::bakeReplacementMaterials()) {
      replacementMaterial = nullptr;
    }

    // Register mesh and preprocess state for baking for this frame
    if (!registerTerrainMesh(ctx, dxvkCtxState, drawCallState)) {
      return false;
    }

    if (!debugDisableBinding()) {
      textureTransformOut = m_bakingParams.viewToCascade0TextureSpace;
    }

    if (debugDisableBaking()) {
      const bool isBaked =
        (debugDisableBinding() ? false : true) &&
        getTerrainTexture(ReplacementMaterialTextureType::AlbedoOpacity).view != nullptr;

      // Recreate material data as it will be needed and textures are available even though baking is currently disabled
      if (isBaked) {
        updateMaterialData(ctx);
      }
      
      return isBaked;
    }

    if (m_calculatingDisplaceInFactor && replacementMaterial != nullptr && (replacementMaterial->getDisplaceIn() > 0.f || replacementMaterial->getDisplaceOut() > 0.f)) {
      const float maxUvTileSize = RtxGeometryUtils::computeMaxUVTileSize(drawCallState.getGeometryData(), drawCallState.getTransformData().objectToWorld);
      // This is the deepest any part of this mesh can go.
      const float maxInputDepth = maxUvTileSize * replacementMaterial->getDisplaceIn();
      // Ths is the highest any part of the mesh can go
      const float maxInputHeight = maxUvTileSize * replacementMaterial->getDisplaceOut();

      const float maxInputDisplacement = maxInputDepth + maxInputHeight;

      // The deepest the baked terrain can go.
      const float maxBakedDepth = 2 * RtxOptions::getMeterToWorldUnitScale() * cascadeMap.levelHalfWidth() * m_prevFrameMaxDisplaceIn;
      // The highest the baked terrain can go.
      const float maxBakedHeight = 2 * RtxOptions::getMeterToWorldUnitScale() * cascadeMap.levelHalfWidth() * m_prevFrameMaxDisplaceOut;

      const float maxBakedDisplacement = maxBakedDepth + maxBakedHeight;

      // Optimal displaceInFactor for this mesh (multiply the pixel value, divide the baked drawcall's displaceIn)
      const float displaceInFactor = maxBakedDisplacement / maxInputDisplacement;

      // Need the largest value from any of the meshes, or else the bottom
      m_calculatedDisplaceInFactor = std::max(m_calculatedDisplaceInFactor, displaceInFactor);
    }

    // The constants buffers are fairly large, and their use is mutually exclusive, so use a union to save memory.
    union UnifiedCB {
      D3D9RtxVertexCaptureData programmablePipeline;
      D3D9FixedFunctionVS fixedFunction;

      UnifiedCB() { }
    };

    UnifiedCB prevCB;

    if (drawCallState.usesVertexShader) {
      prevCB.programmablePipeline = *static_cast<D3D9RtxVertexCaptureData*>(rtState.vertexCaptureCB->mapPtr(0));
    } else {
      prevCB.fixedFunction = *static_cast<D3D9FixedFunctionVS*>(rtState.vsFixedFunctionCB->mapPtr(0));
    }
    D3D9SharedPS prevSharedState = *static_cast<D3D9SharedPS*>(rtState.psSharedStateCB->mapPtr(0));

    if (replacementMaterial != nullptr) {
      accountDisplacement(*replacementMaterial);
    }

    const Matrix4& world = drawCallState.usesVertexShader ? prevCB.programmablePipeline.normalTransform : prevCB.fixedFunction.World;
    const XXH64_hash_t drawKey = calculateDrawKey(drawCallState, replacementMaterial, world,
                                                  drawCallState.usesVertexShader ? nullptr : &prevCB.fixedFunction,
                                                  prevSharedState);

    // Already in the cascade map: the texels this draw call would write are there from an earlier frame.
    if (m_bakedDraws.count(drawKey) != 0) {
      if (replacementMaterial != nullptr) {
        keepReplacementTexturesResident(ctx, drawCallState, *replacementMaterial);
      }
      ++m_numDrawsReusedThisFrame;
      updateMaterialData(ctx);
      return true;
    }

    ScopedGpuProfileZone(ctx, "Terrain Baker: Bake Draw Call");

    bool bakeHorizontalSideProjections = false;
    bool bakeSideProjectionFromBelow = false;
    if (cascadeMap.sideProjectionLevels() > 0) {
      const SurfaceOrientation& orientation = getSurfaceOrientation(drawCallState);
      bakeHorizontalSideProjections = needsHorizontalSideProjections(orientation);
      bakeSideProjectionFromBelow = needsSideProjectionFromBelow(orientation);
      // Takes effect from the next frame's layout, which rebakes everything with the side projections
      m_sideProjectionsNeeded |= bakeHorizontalSideProjections || bakeSideProjectionFromBelow;
    }
    const bool bakeSideProjections =
      m_bakingParams.numSideProjectionLevels > 0 && m_sideProjectionDepthTarget != nullptr &&
      (bakeHorizontalSideProjections || bakeSideProjectionFromBelow);

    // Side projections bake with their own depth test and culling, and restore the draw call's afterwards
    const DxvkRsInfo& rsInfo = dxvkCtxState.gp.state.rs;
    const DxvkRasterizerState prevRasterizerState = {
      rsInfo.polygonMode(), rsInfo.cullMode(), rsInfo.frontFace(), rsInfo.depthClipEnable(),
      rsInfo.depthBiasEnable(), rsInfo.conservativeMode(), rsInfo.sampleCount() };
    const DxvkDsInfo& dsInfo = dxvkCtxState.gp.state.ds;
    const DxvkDepthStencilState prevDepthStencilState = {
      dsInfo.enableDepthTest(), dsInfo.enableDepthWrite(), dsInfo.enableStencilTest(), dsInfo.depthCompareOp(),
      dxvkCtxState.gp.state.dsFront.state(), dxvkCtxState.gp.state.dsBack.state() };

    // Save viewports
    const uint32_t prevViewportCount = dxvkCtxState.gp.state.rs.viewportCount();
    const DxvkViewportState prevViewportState = dxvkCtxState.vp;

    // Save previous render targets
    DxvkRenderTargets prevRenderTargets = dxvkCtxState.om.renderTargets; 
    Rc<DxvkSampler> prevSecondaryResourceSlotSampler;   // Initialized when overriden

    // Gather replacement textures, if available, to be used for baking
    std::vector<RtxGeometryUtils::TextureConversionInfo> replacementTextures;
    bool bakeReplacementTextures = gatherAndPreprocessReplacementTextures(ctx, drawCallState, replacementMaterial, replacementTextures);

    const uint32_t numTexturesToBake = bakeReplacementTextures ? replacementTextures.size() : 1;

    // Lookup texture slots to bind replacement textures at
    uint32_t colorTextureSlot = kInvalidResourceSlot;
    uint32_t secondaryTextureSlot = kInvalidResourceSlot;

    if (bakeReplacementTextures) {
      colorTextureSlot = drawCallState.getMaterialData().getColorTextureSlot(0);

      // Check that the slot for secondary textures is available
      const uint32_t textureSlot = drawCallState.getMaterialData().getColorTextureSlot(kTerrainBakerSecondaryTextureStage);

      if (textureSlot == kInvalidResourceSlot) {
        auto shaderSampler = RemapStateSamplerShader(static_cast<uint8_t>(kTerrainBakerSecondaryTextureStage));
        const uint32_t bindingIndex = shaderSampler.second;
        secondaryTextureSlot = computeResourceSlotId(DxsoProgramType::PixelShader, DxsoBindingType::Image, bindingIndex);
      }
    }

    // Update spec constants
    DxvkScInfo prevSpecConstantsInfo = ctx->getSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS);
    {
      // Disable fog

      ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::FogEnabled, false);
      ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::VertexFogMode, D3DFOG_NONE);
      ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::PixelFogMode, D3DFOG_NONE);

      if (drawCallState.usesVertexShader) {
        ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::CustomVertexTransformEnabled, true);
      }
    }

    bool bakingResult = false;

    ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ReplacementTextureCategory,
                             packReplacementTextureSpecConstant(ReplacementMaterialTextureCategory::AlbedoOpacity, drawCallState.getMaterialData().colorTextureStage));
    
    // The height value that corresponds to the original surface height.
    const float prevFrameTotalHeight = m_prevFrameMaxDisplaceIn + m_prevFrameMaxDisplaceOut;
    const float neutralDisplacement = prevFrameTotalHeight != 0.f ? m_prevFrameMaxDisplaceIn / prevFrameTotalHeight : kDefaultNeutralHeight;


    // Bake all material textures
    for (uint32_t iTexture = 0; iTexture < numTexturesToBake; iTexture++) {
      
      ReplacementMaterialTextureType::Enum textureType = ReplacementMaterialTextureType::AlbedoOpacity;
      float texturePreOffset = 0.f;
      float textureScale = 1.f;

      // Bind a source replacement texture to bake, if available.
      // Otherwise the legacy albedoOpacity texture that's already bound will be baked
      if (bakeReplacementTextures) {
        TextureRef& replacementTexture = replacementTextures[iTexture].targetTexture;
        textureType = replacementTextures[iTexture].type;
        textureScale = replacementTextures[iTexture].scale;
        texturePreOffset = replacementTextures[iTexture].offset;

        ctx->bindResourceView(colorTextureSlot, replacementTexture.getImageView(), nullptr);

        if (isPSReplacementSupportEnabled(drawCallState)) {

          if (drawCallState.usesPixelShader) {
            if (textureType != ReplacementMaterialTextureType::Enum::AlbedoOpacity &&
                drawCallState.programmablePixelShaderInfo.majorVersion() >= 2) {
              // Unsupported right now - REMIX-2223 
              ONCE(Logger::err("[RTX Terrain Baker] Draw call associated with a terrain texture uses a shader model version 2 or higher. This is currently not supported when baking replacement PBR material textures other than albedoOpacity. Skipping baking of the replacement texture of all but albedoOpacity."));
              continue;
            }
          }

          // Set texture category in a specconst
          switch (textureType) {
          case ReplacementMaterialTextureType::Enum::AlbedoOpacity:
          default:
            ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ReplacementTextureCategory,
                             packReplacementTextureSpecConstant(ReplacementMaterialTextureCategory::AlbedoOpacity, drawCallState.getMaterialData().colorTextureStage));
            break;

          case ReplacementMaterialTextureType::Enum::Normal:
          case ReplacementMaterialTextureType::Enum::Tangent:
            ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ReplacementTextureCategory,
                             packReplacementTextureSpecConstant(ReplacementMaterialTextureCategory::SecondaryOctahedralEncoded, drawCallState.getMaterialData().colorTextureStage));
            break;

          case ReplacementMaterialTextureType::Enum::Roughness:
          case ReplacementMaterialTextureType::Enum::Metallic:
          case ReplacementMaterialTextureType::Enum::Emissive:
            ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ReplacementTextureCategory,
                             packReplacementTextureSpecConstant(ReplacementMaterialTextureCategory::SecondaryRaw, drawCallState.getMaterialData().colorTextureStage));
            break;
          case ReplacementMaterialTextureType::Enum::Height:
            ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ReplacementTextureCategory,
                             packReplacementTextureSpecConstant(ReplacementMaterialTextureCategory::SecondaryScaled, drawCallState.getMaterialData().colorTextureStage));
            break;
          }


          // Finalize bindings when baking a secondary non-albedo opacity texture
          if (textureType != ReplacementMaterialTextureType::AlbedoOpacity) {
            if (secondaryTextureSlot == kInvalidResourceSlot) {
              ONCE(Logger::err("[RTX Terrain Baker] Failed to retrieve a valid secondary texture slot required for baking of secondary replacement textures. Possibly due to it being used by the terrain draw call itself. Skipping baking for all but the AlbedoOpacity replacement texture."));
              continue;
            }

            // Bind the albedo opacity texture as a secondary texture when baking non-albedo opacity replacement textures
            TextureRef& albedoOpacityReplacementTexture = replacementTextures[ReplacementMaterialTextureType::AlbedoOpacity].targetTexture;
            ctx->bindResourceView(secondaryTextureSlot, albedoOpacityReplacementTexture.getImageView(), nullptr);

            // Bind a sampler for the secondary texture
            prevSecondaryResourceSlotSampler = ctx->getShaderResourceSlot(secondaryTextureSlot).sampler;
            ctx->bindResourceSampler(secondaryTextureSlot, ctx->getShaderResourceSlot(colorTextureSlot).sampler);
          }
        }
      }

      // Bind terrain texture as render target 
      const RtxMipmap::Resource& terrainResource = getTerrainTexture(ctx, textureManger, textureType, m_bakingParams.cascadeMapResolution.width,
                          m_bakingParams.cascadeMapResolution.height);
      const Rc<DxvkImageView>& terrainTextureView = terrainResource.views.empty() ? terrainResource.view : terrainResource.views[0];
      if (terrainTextureView == nullptr) {
        if (textureType == ReplacementMaterialTextureType::AlbedoOpacity) {
          ONCE(Logger::err(str::format("[RTX Terrain Baker] Failed to retrieve a terrain texture of type albedo opacity. This texture is required for baking of any replacement texture. Skipping baking of the material for this draw call.")));
          break;
        } else {
          ONCE(Logger::err(str::format("[RTX Terrain Baker] Failed to retrieve a terrain texture of type ", static_cast<uint32_t>(textureType), ". Skipping baking of the texture.")));
          continue;
        }
      }

      // Bind the target terrain texture as render target
      DxvkRenderTargets terrainRt;
      terrainRt.color[0].view = terrainTextureView;
      terrainRt.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;
      ctx->bindRenderTargets(terrainRt);
    
      m_materialTextures[textureType].markAsBaked();

      // Account for the difference in UV density between the input terrain material and the baked terrain.
      // This part is just pre-multiplying the "multiply by output uv density".  The input UV density is accounted for in `postprocessTextureReadForTerrainBaking`
      const float cascadeUvDensity = Material::Properties::displaceInFactor() / std::max(m_bakingParams.cascadeMapResolution.width, m_bakingParams.cascadeMapResolution.height);

      const auto bakeView = [&](const VkRect2D& rect, const Matrix4& view, const Matrix4& projection) {
        // Maps clip space <-1, 1> to the rect, accounting for the inverted y coordinate in Vulkan
        const VkViewport viewport {
          static_cast<float>(rect.offset.x),
          static_cast<float>(rect.offset.y) + static_cast<float>(rect.extent.height),
          static_cast<float>(rect.extent.width),
          -static_cast<float>(rect.extent.height),
          0.f, 1.f
        };
        ctx->setViewports(1, &viewport, &rect);

        const Matrix4 worldView = view * world;

        // Update constant buffers
        //
        D3D9SharedPS& sharedState = ctx->allocAndMapPSSharedStateConstantBuffer();
        for (int i = 0; i < caps::TextureStageCount; ++i) {
          sharedState.Stages[i] = prevSharedState.Stages[i];
        }
        // The neutral height value of the input image and the output map don't match.
        // To account, first subtract the input's neutral value from the pixel,
        // then apply all scale operations, then add the output neutral value.
        sharedState.Stages[kTerrainBakerSecondaryTextureStage].texturePreOffset = texturePreOffset;
        sharedState.Stages[kTerrainBakerSecondaryTextureStage].textureScale = textureScale * cascadeUvDensity;
        sharedState.Stages[kTerrainBakerSecondaryTextureStage].texturePostOffset = neutralDisplacement;

        // Programmable VS path
        if (drawCallState.usesVertexShader) {
          D3D9RtxVertexCaptureData& cbData = ctx->allocAndMapVertexCaptureConstantBuffer();
          cbData = prevCB.programmablePipeline;
          cbData.customWorldToProjection = projection * worldView;
        }
        else { // Fixed function path
          D3D9FixedFunctionVS& cbData = ctx->allocAndMapFixedFunctionVSConstantBuffer();
          cbData = prevCB.fixedFunction;

          cbData.InverseView = inverse(view);
          cbData.View = view;
          cbData.WorldView = worldView;
          cbData.Projection = projection;

          // Disable lighting
          for (auto& light : cbData.Lights) {
            light.Diffuse = Vector4(0.f);
            light.Specular = Vector4(0.f);
            light.Ambient = Vector4(1.f);
          }
        }

        if (drawParams.indexCount == 0) {
          ctx->DxvkContext::draw(drawParams.vertexCount, drawParams.instanceCount, drawParams.vertexOffset, 0);
        } else {
          ctx->DxvkContext::drawIndexed(drawParams.indexCount, drawParams.instanceCount, drawParams.firstIndex, drawParams.vertexOffset, 0);
        }
      };

      // Render into all cascade levels
      for (uint32_t iCascade = 0; iCascade < m_bakingParams.numCascades; iCascade++) {
        bakeView(getTopDownViewRect(iCascade), m_bakingParams.sceneView, m_bakingParams.bakingCameraOrthoProjection[iCascade]);
      }

      if (bakeSideProjections) {
        DxvkRenderTargets sideProjectionRt;
        sideProjectionRt.color[0].view = terrainTextureView;
        sideProjectionRt.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;
        sideProjectionRt.depth.view = m_sideProjectionDepthTarget;
        sideProjectionRt.depth.layout = VK_IMAGE_LAYOUT_GENERAL;
        ctx->bindRenderTargets(sideProjectionRt);

        // Every layer of a surface is drawn from the same vertices through the same matrices, so its
        // overlays land at exactly the depth of its base and pass a less-or-equal test.
        DxvkDepthStencilState sideProjectionDepthStencilState = prevDepthStencilState;
        sideProjectionDepthStencilState.enableDepthTest = VK_TRUE;
        sideProjectionDepthStencilState.enableDepthWrite = VK_TRUE;
        sideProjectionDepthStencilState.enableStencilTest = VK_FALSE;
        sideProjectionDepthStencilState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        ctx->setDepthStencilState(sideProjectionDepthStencilState);

        DxvkRasterizerState sideProjectionRasterizerState = prevRasterizerState;
        sideProjectionRasterizerState.polygonMode = VK_POLYGON_MODE_FILL;
        sideProjectionRasterizerState.depthClipEnable = VK_TRUE;
        sideProjectionRasterizerState.depthBiasEnable = VK_FALSE;
        switch (cascadeMap.sideProjectionCulling()) {
        case TerrainSideProjectionCulling::None:  sideProjectionRasterizerState.cullMode = VK_CULL_MODE_NONE; break;
        case TerrainSideProjectionCulling::Back:  sideProjectionRasterizerState.cullMode = VK_CULL_MODE_BACK_BIT; break;
        case TerrainSideProjectionCulling::Front: sideProjectionRasterizerState.cullMode = VK_CULL_MODE_FRONT_BIT; break;
        }
        ctx->setRasterizerState(sideProjectionRasterizerState);

        const uint32_t numSideViews = m_bakingParams.numSideProjectionLevels * kNumTerrainSideProjectionViews;
        for (uint32_t sideView = 0; sideView < numSideViews; sideView++) {
          const bool isViewFromBelow = sideView % kNumTerrainSideProjectionViews == kTerrainSideProjectionViewBelow;
          if (isViewFromBelow ? !bakeSideProjectionFromBelow : !bakeHorizontalSideProjections) {
            continue;
          }
          bakeView(getSideProjectionViewRect(sideView),
                   m_bakingParams.sideProjectionView[sideView], m_bakingParams.sideProjectionOrthoProjection[sideView]);
        }

        ctx->setDepthStencilState(prevDepthStencilState);
        ctx->setRasterizerState(prevRasterizerState);
      }

      if (textureType == ReplacementMaterialTextureType::AlbedoOpacity) {
        bakingResult = true;
      }
    }

    // Restore prev state
    {
      ctx->setViewports(prevViewportCount, prevViewportState.viewports.data(), prevViewportState.scissorRects.data());
      ctx->bindRenderTargets(prevRenderTargets);
      ctx->setSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS, prevSpecConstantsInfo);

      ctx->allocAndMapPSSharedStateConstantBuffer() = prevSharedState;
      if (drawCallState.usesVertexShader) {
        ctx->allocAndMapVertexCaptureConstantBuffer() = prevCB.programmablePipeline;
      } else {
        ctx->allocAndMapFixedFunctionVSConstantBuffer() = prevCB.fixedFunction;
      }

      if (secondaryTextureSlot != kInvalidResourceSlot) {
        // Secondary texture slot wasn't used prior to baking, so set it to a null view
        ctx->bindResourceView(secondaryTextureSlot, nullptr, nullptr);

        if (prevSecondaryResourceSlotSampler.ptr()) {
          ctx->bindResourceSampler(secondaryTextureSlot, prevSecondaryResourceSlotSampler);
        }
      }

      // Input color texture will be restored in RtxContext::bakeTerrain
    }

    if (bakingResult) {
      m_bakedDraws.insert(drawKey);
      ++m_numDrawsBakedThisFrame;
    }

    updateMaterialData(ctx);

    return bakingResult;
  }

  void TerrainBaker::updateMaterialData(Rc<RtxContext> ctx) {
    if (m_hasInitializedMaterialDataThisFrame && !m_needsMaterialDataUpdate) {
      return;
    }

    // We're going to use this to create a modified sampler for terrain textures.
    // Terrain textures have only mip 0, so use nearest for mip filtering
    if (!m_terrainSampler.ptr()) {
      Resources& resourceManager = ctx->getResourceManager();
      m_terrainSampler = resourceManager.getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }
    
    auto createTextureRef = [&](ReplacementMaterialTextureType::Enum textureType) {
      return m_materialTextures[textureType].isBaked()
        ? TextureRef(m_materialTextures[textureType].texture.view)
        : TextureRef();
    };

    // Create a material with the baked material textures
    m_materialData.emplace(OpaqueMaterialData(
      createTextureRef(ReplacementMaterialTextureType::AlbedoOpacity),
      createTextureRef(ReplacementMaterialTextureType::Normal),
      createTextureRef(ReplacementMaterialTextureType::Tangent),
      createTextureRef(ReplacementMaterialTextureType::Height),
      createTextureRef(ReplacementMaterialTextureType::Roughness),
      createTextureRef(ReplacementMaterialTextureType::Metallic),
      createTextureRef(ReplacementMaterialTextureType::Emissive),
      TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), // SSS textures
      Material::Properties::roughnessAnisotropy(),
      Material::Properties::emissiveIntensity(),
      Vector3(1, 1, 1), // AlbedoConstant - unused since the AlbedoOpacity texture must be always present for baking
      1.f, // OpacityConstant - unused since the AlbedoOpacity texture must be always present for baking
      Material::Properties::roughnessConstant(),
      Material::Properties::metallicConstant(),
      Material::Properties::emissiveColorConstant(),
      Material::Properties::enableEmission(),
      false, // fork (2026-07-26): OpaqueMaterial::SkyLitParticle - terrain is not a particle
      // Setting expected constant values. Baked terrain should not need to have other values for the below material parameters set
      1, 1, 0, /* spriteSheet* */
      false, // LegacyMaterialDefaults::enableThinFilm(),
      false, // LegacyMaterialDefaults::alphaIsThinFilmThickness(),
      0.f,
      false, // Set to false for now, otherwise the baked terrain is not fully opaque - opaqueMaterialDefaults.UseLegacyAlphaState
      false, // OpaqueMaterialDefaults::BlendEnabled,
      BlendType::kAlpha,
      false, // OpaqueMaterialDefaults::InvertedBlend,
      AlphaTestType::kAlways,
      0,//OpaqueMaterialDefaults::AlphaReferenceValue;
      // Using the previous frame's displaceIn/Out because all current frame draw calls are normalized to the previous frame's max.
      m_prevFrameMaxDisplaceIn / Material::Properties::displaceInFactor(),  // OpaqueMaterialDefaults::DisplaceIn
      m_prevFrameMaxDisplaceOut / Material::Properties::displaceInFactor(),  // OpaqueMaterialDefaults::DisplaceOut
      Vector3(),  // OpaqueMaterialDefaults::subsurfaceTransmittanceColor
      0.0f,  // OpaqueMaterialDefaults::subsurfaceMeasurementDistance
      Vector3(),  // OpaqueMaterialDefaults::subsurfaceSingleScatteringAlbedo
      0.0f, // OpaqueMaterialDefaults::subsurfaceVolumetricAnisotropy
      false, // OpaqueMaterialDefaults::subsurfaceDiffusionProfile
      Vector3(),  // OpaqueMaterialDefaults::subsurfaceRadius
      0.0f, // OpaqueMaterialDefaults::subsurfaceRadiusScale
      0.0f, // OpaqueMaterialDefaults::subsurfaceMaxSampleRadius
      // NOTE: The terrain defines it's own sampler, and these are the modes it uses.
      lss::Mdl::Filter::Linear,
      lss::Mdl::WrapMode::Clamp, // U
      lss::Mdl::WrapMode::Clamp,  // V
      true,
      1.f,
      0.4f,
      0.8f
    ));

    m_hasInitializedMaterialDataThisFrame = true;
    m_needsMaterialDataUpdate = false;
  }

  const RtxMipmap::Resource& TerrainBaker::getTerrainTexture(ReplacementMaterialTextureType::Enum textureType) const {
    return m_materialTextures[textureType].texture;
  }

  const RtxMipmap::Resource& TerrainBaker::getTerrainTexture(
    Rc<DxvkContext> ctx, 
    RtxTextureManager& textureManager, 
    ReplacementMaterialTextureType::Enum textureType, 
    uint32_t width, 
    uint32_t height) {
    VkExtent3D resolution = { width, height, 1 };

    RtxMipmap::Resource& texture = m_materialTextures[static_cast<uint32_t>(textureType)].texture;

    // Recreate the texture
    if (!texture.isValid() ||
        texture.image->info().extent != resolution) {

      // WAR (REMIX-1557) to force release previous terrain texture reference from texture cache since it doesn't do it automatically resulting in a leak
      if (texture.isValid()) {
        TextureRef textureRef = TextureRef(texture.view);
        textureManager.releaseTexture(textureRef);

        if (texture.views.size() > 0) {
          for (Rc<DxvkImageView>& view : texture.views) {
            auto viewRef = TextureRef(view);
            textureManager.releaseTexture(viewRef);
          }
        }
      }

      texture = RtxMipmap::createResource(
        ctx, "baked terrain texture", resolution, getTextureFormat(textureType), VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, getClearColor(textureType), getMipLevels(textureType, resolution));

      m_needsMaterialDataUpdate = true;
      m_bakedContentLost = true;
      // Cascade image just appeared or changed size, so the cascade set the
      // override OpaqueMaterialData describes is different from any prior frame.
      // SceneManager reads this to keep terrain draws on the dynamic path for
      // this frame and rebuild their cached RtSurfaceMaterial with the new
      // cascade texture indices.
      m_cascadeCompositionChangedThisFrame = true;
    }

    return texture;
  }

  const Rc<DxvkSampler>& TerrainBaker::getTerrainSampler() const {
    return m_terrainSampler;
  }

  const MaterialData* TerrainBaker::getMaterialData() const {
    if (m_materialData.has_value()) {
      return &(*m_materialData);
    }

    return nullptr;
  }

  TerrainArgs TerrainBaker::getTerrainArgs() const {

    TerrainArgs args;

    args.cascadeMapSize = m_bakingParams.cascadeMapSize;
    args.rcpCascadeMapSize.x = 1.f / args.cascadeMapSize.x;
    args.rcpCascadeMapSize.y = 1.f / args.cascadeMapSize.y;

    args.maxCascadeLevel = m_bakingParams.numCascades - 1;
    if (m_materialData.has_value()) {
      args.displaceIn = m_materialData->getOpaqueMaterialData().getDisplaceIn();
    } else {
      args.displaceIn = 0.0f;
    }
    args.lastCascadeScale = m_bakingParams.lastCascadeScale;

    // Both in the first cascade's texture space, where the level spans <0, 1> across twice its half width
    args.recenterMargin = cascadeMap.recenterDistance() / (2.f * cascadeMap.levelHalfWidth());
    const Vector4 cameraTexcoord = m_bakingParams.viewToCascade0TextureSpace * Vector4(0.f, 0.f, 0.f, 1.f);
    args.cameraOffsetFromCenter = float2 { cameraTexcoord.x - 0.5f, cameraTexcoord.y - 0.5f };
    args.firstLevelTile = m_bakingParams.numSideProjectionTiles;

    const float metersToWorldUnitScale = RtxOptions::getMeterToWorldUnitScale();
    const float levelHalfWidth = metersToWorldUnitScale * cascadeMap.levelHalfWidth();
    const bool hasSideProjections =
      m_bakingParams.numSideProjectionLevels > 0 && m_sideProjectionDepthTextureIndex != kSurfaceMaterialInvalidTextureIndex;

    args.numSideProjectionLevels = hasSideProjections ? m_bakingParams.numSideProjectionLevels : 0;
    args.sideProjectionDepthTextureIndex = m_sideProjectionDepthTextureIndex;
    args.sideProjectionDepthTolerance = metersToWorldUnitScale * cascadeMap.sideProjectionDepthTolerance();
    args.sideProjectionLevelTexelSize = 2.f * levelHalfWidth / std::max(m_bakingParams.cascadeLevelResolution.width / 2, 1u);
    args.sideProjectionCenter = m_cascadeCenter.value_or(Vector3(0.f));
    args.sideProjectionLevelHalfWidth = levelHalfWidth;
    args.recenterDistance = metersToWorldUnitScale * cascadeMap.recenterDistance();
    args.sideProjectionLevelDepthRange = 2.f * levelHalfWidth;
    args.cascadeMapResolution = float2 {
      static_cast<float>(m_bakingParams.cascadeMapResolution.width),
      static_cast<float>(m_bakingParams.cascadeMapResolution.height) };
    args.sceneUp = SceneManager::getSceneUp();

    for (uint32_t view = 0; view < kNumTerrainSideProjectionViews; view++) {
      args.sideProjectionViewDirection[view] = Vector4(m_bakingParams.sideProjectionViewDirection[view], 0.f);
    }

    // Converts from clip space <-1, 1> to <0, 1> and flips y for Vulkan, as for the top-down levels
    const Matrix4 textureOffset = Matrix4(Vector4(.5f, 0, 0, 0),
                                          Vector4(0, -.5f, 0, 0),
                                          Vector4(0, 0, 1, 0),
                                          Vector4(.5f, .5f, 0, 1));
    for (uint32_t sideView = 0; sideView < args.numSideProjectionLevels * kNumTerrainSideProjectionViews; sideView++) {
      args.sideProjectionWorldToTexture[sideView] =
        textureOffset * m_bakingParams.sideProjectionOrthoProjection[sideView] * m_bakingParams.sideProjectionView[sideView];
    }

    return args;
  }

  static RemixGui::ComboWithKey<TerrainSideProjectionCulling> sideProjectionCullingCombo {
    "Side Projection Culling",
    RemixGui::ComboWithKey<TerrainSideProjectionCulling>::ComboEntries { {
        {TerrainSideProjectionCulling::None, "None"},
        {TerrainSideProjectionCulling::Back, "Back"},
        {TerrainSideProjectionCulling::Front, "Front"},
    } } };

  void TerrainBaker::showImguiSettings() const {

    constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;

    {
      RemixGui::Checkbox("Use Terrain Bounding Box", &cascadeMap.useTerrainBBOXObject());
      RemixGui::Checkbox("Clear Terrain Textures Before Terrain Baking", &clearTerrainBeforeBakingObject());

      if (RemixGui::CollapsingHeader("Material")) {
        ImGui::Indent();

        const bool isPSReplacementSupportEnabled = Material::replacementSupportInPS_fixedFunction() || Material::replacementSupportInPS_programmableShaders();
        ImGui::BeginDisabled(!isPSReplacementSupportEnabled);
        RemixGui::Checkbox("Replacements Support in PS", &Material::replacementSupportInPSObject());
        ImGui::EndDisabled();

        RemixGui::Checkbox("Bake Replacement Materials", &Material::bakeReplacementMaterialsObject());
        RemixGui::Checkbox("Bake Secondary PBR Textures", &Material::bakeSecondaryPBRTexturesObject());
        RemixGui::DragInt("Max Resolution (except for colorOpacity)", &Material::maxResolutionToUseForReplacementMaterialsObject(), 1.f, 1, 16384);

        if (RemixGui::CollapsingHeader("Properties")) {
          ImGui::Indent();

          RemixGui::ColorEdit3("Emissive Color", &Material::Properties::emissiveColorConstantObject());
          RemixGui::Checkbox("Enable Emission", &Material::Properties::enableEmissionObject());
          RemixGui::DragFloat("Emissive Intensity", &Material::Properties::emissiveIntensityObject(), 0.01f, 0.f, FLT_MAX, "%.3f", sliderFlags);
          RemixGui::DragFloat("Roughness", &Material::Properties::roughnessConstantObject(), 0.01f, 0.f, 1.f, "%.3f", sliderFlags);
          RemixGui::DragFloat("Metallic", &Material::Properties::metallicConstantObject(), 0.01f, 0.f, 1.f, "%.3f", sliderFlags);
          RemixGui::DragFloat("Anisotropy", &Material::Properties::roughnessAnisotropyObject(), 0.01f, -1.0f, 1.f, "%.3f", sliderFlags);
          
          ImGui::Text("\nDisplacement Settings");
          RemixGui::DragFloat("Displacement Factor", &Material::Properties::displaceInFactorObject(), 0.01f, 0.01f, 100.f, "%.3f", sliderFlags);
          ImGui::Text("Calculate the lowest safe Displacement Factor for the current \n"
                      "scene.  Mod creators should run this in scenes across the \n"
                      "game, and use the highest returned value.  See Displacement \n"
                      "Factor's tooltip for more info.");
          if (ImGui::Button("Calculate Scene's Optimal Displacement Factor")) {
            m_calculateDisplaceInFactorNextFrame = true;
          }

          ImGui::Unindent();
        }
        ImGui::Unindent();
      }

      if (RemixGui::CollapsingHeader("Cascade Map")) {
        ImGui::Indent();

        RemixGui::DragFloat("Cascade Map's Default Half Width [meters]", &cascadeMap.defaultHalfWidthObject(), 1.f, 0.1f, 10000.f);
        RemixGui::DragFloat("Cascade Map's Default Height [meters]", &cascadeMap.defaultHeightObject(), 1.f, 0.1f, 10000.f);
        RemixGui::DragFloat("First Cascade Level's Half Width [meters]", &cascadeMap.levelHalfWidthObject(), 1.f, 0.1f, 10000.f);

        RemixGui::DragInt("Max Cascade Levels", &cascadeMap.maxLevelsObject(), 1.f, 1, 16);
        RemixGui::DragInt("Texture Resolution Per Cascade Level", &cascadeMap.levelResolutionObject(), 8.f, 1, 32 * 1024);
        RemixGui::Checkbox("Expand Last Cascade Level", &cascadeMap.expandLastCascadeObject());
        RemixGui::DragFloat("Recenter Distance [meters]", &cascadeMap.recenterDistanceObject(), 0.1f, 0.f, 10000.f, "%.1f", sliderFlags);
        RemixGui::DragInt("Side Projection Levels", &cascadeMap.sideProjectionLevelsObject(), 0.1f, 0, kMaxTerrainSideProjectionLevels);
        sideProjectionCullingCombo.getKey(&cascadeMap.sideProjectionCullingObject());
        RemixGui::DragFloat("Side Projection Depth Tolerance [meters]", &cascadeMap.sideProjectionDepthToleranceObject(), 0.01f, 0.f, 10.f, "%.2f", sliderFlags);

        if (RemixGui::CollapsingHeader("Statistics")) {
          ImGui::Indent();
        
          ImGui::Text("Cascade Levels: %u", m_bakingParams.numCascades);
          ImGui::Text("Cascade Level Resolution: %u, %u", m_bakingParams.cascadeLevelResolution.width, m_bakingParams.cascadeLevelResolution.height);
          ImGui::Text("Cascade Map Resolution: %u, %u", m_bakingParams.cascadeMapResolution.width, m_bakingParams.cascadeMapResolution.height);
          ImGui::Text("Draw Calls Baked / Reused Last Frame: %u / %u", m_numDrawsBakedLastFrame, m_numDrawsReusedLastFrame);
          ImGui::Text("Side Projection Levels: %u%s", m_bakingParams.numSideProjectionLevels,
                      m_sideProjectionsNeeded ? "" : " (no terrain has needed them yet)");
          ImGui::Text("Side Projection Depth Texture Index: %u", m_sideProjectionDepthTextureIndex);
        
          ImGui::Unindent();
        }

        ImGui::Unindent();
      }

      RemixGui::Checkbox("Debug: Disable Baking", &debugDisableBakingObject());
      RemixGui::Checkbox("Debug: Disable Binding", &debugDisableBindingObject());
    }
  }

  void TerrainBaker::calculateTerrainBBOX(const uint32_t currentFrameIndex) {
    m_bakedTerrainBBOX.invalidate();

    // Find the union of all terrain mesh BBOXes
    if (m_terrainMeshBBOXes.size() > 0) {
      for (auto& meshBBOX : m_terrainMeshBBOXes) {
        m_bakedTerrainBBOX.unionWith(meshBBOX.calculateAABBInWorldSpace());
      }
      m_terrainMeshBBOXes.clear();
      m_terrainBBOXFrameIndex = currentFrameIndex;
    }
  }

  void TerrainBaker::onFrameEnd(Rc<DxvkContext> ctx) {
    RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();
    const uint32_t currentFrameIndex = ctx->getDevice()->getCurrentFrameId();

    if (TerrainBaker::needsTerrainBaking()) {
      // Expects the mesh BBOXes to be calculated by this point
      calculateTerrainBBOX(currentFrameIndex);
    }

    m_hasInitializedMaterialDataThisFrame = false;
    m_cascadeCompositionChangedThisFrame = false;

    m_numDrawsBakedLastFrame = m_numDrawsBakedThisFrame;
    m_numDrawsReusedLastFrame = m_numDrawsReusedThisFrame;
    m_numDrawsBakedThisFrame = 0;
    m_numDrawsReusedThisFrame = 0;

    for (BakedTexture& texture : m_materialTextures) {
      texture.onFrameEnd(ctx);
    }

    if (m_calculatingDisplaceInFactor) {
      m_calculatingDisplaceInFactor = false;
      Material::Properties::displaceInFactor.setDeferred(m_calculatedDisplaceInFactor);
    }
    m_calculatingDisplaceInFactor = m_calculateDisplaceInFactorNextFrame;
    m_calculateDisplaceInFactorNextFrame = false;

    m_prevFrameMaxDisplaceIn = m_currFrameMaxDisplaceIn;
    m_currFrameMaxDisplaceIn = 0.f;

    m_prevFrameMaxDisplaceOut = m_currFrameMaxDisplaceOut;
    m_currFrameMaxDisplaceOut = 0.f;

    m_stagingTextureCache.clear();

    // Destroy material data every frame so as not keep texture references around.
    // Material data gets recreated every frame on baking
    m_materialData.reset();
  }

  void TerrainBaker::prepareSceneData(Rc<RtxContext> ctx) {
    if (TerrainBaker::needsTerrainBaking()) {
      // The mip chain only changes when texels do, i.e. on frames that baked something
      if (m_numDrawsBakedThisFrame > 0 && m_materialTextures[ReplacementMaterialTextureType::Height].texture.isValid()) {
        ScopedGpuProfileZone(ctx, "Terrain Height Mip Map");
        RtxMipmap::updateMipmap(ctx, m_materialTextures[ReplacementMaterialTextureType::Height].texture, MipmapMethod::Maximum);
      }

      m_sideProjectionDepthTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      if (m_sideProjectionDepth.view != nullptr) {
        ctx->getSceneManager().trackTexture(TextureRef(m_sideProjectionDepth.view), m_sideProjectionDepthTextureIndex, true, false);
      }
    }
  }

  void TerrainBaker::BakedTexture::onFrameEnd(Rc<DxvkContext> ctx) {

    auto releaseTexture = [&](RtxMipmap::Resource& texture) {
      if (!texture.isValid()) {
        return;
      }

      RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();

      // WAR (REMIX-1557) to force release terrain texture reference from texture cache since it doesn't do it automatically resulting in a leak
      TextureRef textureRef = TextureRef(texture.view);
      textureManager.releaseTexture(textureRef);
      
      if (texture.views.size() > 0) {
        for (Rc<DxvkImageView>& view : texture.views) {
          auto viewRef = TextureRef(view);
          textureManager.releaseTexture(viewRef);
        }
      }
      texture.reset();
    };

    // Retain textures when baking is disabled as they are not being refreshed and can still be used
    if (!debugDisableBaking()) {
      if (numFramesToRetain > 0) {
        numFramesToRetain--;
      }
    }

    // Release the texture if it has not been baked to recently
    if (numFramesToRetain == 0) {
      releaseTexture(texture);
    }
  }

  void TerrainBaker::updateTextureFormat(const DxvkContextState& dxvkCtxState) {
    DxvkRenderTargets currentRenderTargets = dxvkCtxState.om.renderTargets;

    VkFormat terrainRtColorFormat = currentRenderTargets.color[0].view->image()->info().format;
    VkFormat terrainSrgbColorFormat = TextureUtils::toSRGB(terrainRtColorFormat);

    // RT shaders expect the textures in sRGB format but but as linear targets
    if (terrainRtColorFormat == terrainSrgbColorFormat) {
      ONCE(Logger::warn(str::format("[RTX Terrain Baker] Terrain render target is of sRGB format ", terrainRtColorFormat, ". Instead, it is expected to be of linear format.")));
    }
  }

  void TerrainBaker::clearMaterialTexture(Rc<DxvkContext> ctx, ReplacementMaterialTextureType::Enum textureType) {
    Resources::Resource& texture = m_materialTextures[textureType].texture;

    VkImageSubresourceRange subRange = {};
    subRange.layerCount = 1;
    subRange.levelCount = 1;
    subRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    ctx->clearColorImage(texture.image, getClearColor(textureType), subRange);
  }

  TerrainBaker::AxisAlignedBoundingBoxLink::AxisAlignedBoundingBoxLink(const DrawCallState& drawCallState)
    : aabbObjectSpace(drawCallState.getGeometryData().boundingBox)
    , objectToWorld(drawCallState.getTransformData().objectToWorld) {
  }

  AxisAlignedBoundingBox TerrainBaker::AxisAlignedBoundingBoxLink::calculateAABBInWorldSpace() {
    AxisAlignedBoundingBox aabb;
    aabb.minPos = (objectToWorld * Vector4(aabbObjectSpace.minPos, 1.f)).xyz();
    aabb.maxPos = (objectToWorld * Vector4(aabbObjectSpace.maxPos, 1.f)).xyz();
    return aabb;
  }

  bool TerrainBaker::needsTerrainBaking() {
    return enableBaking() && RtxOptions::terrainTextures().size() > 0;
  }

  void TerrainBaker::onFrameBegin(Rc<RtxContext> ctx, const DxvkContextState& dxvkCtxState) {
    Resources& resourceManager = ctx->getResourceManager();
    RtxTextureManager& textureManger = ctx->getCommonObjects()->getTextureManager();

    // Force material data update every frame to pick up any material parameter changes
    m_needsMaterialDataUpdate = true;

    updateTextureFormat(dxvkCtxState);
    calculateBakingParameters(ctx, dxvkCtxState);
    updateSideProjectionDepth(ctx);
    beginBakedContent(ctx);
  }

  // Everything that decides where a world position lands in the cascade map, and what a baked
  // height value means. Texels baked under one layout are meaningless under another.
  XXH64_hash_t TerrainBaker::calculateBakedLayoutHash() const {
    auto combine = [](XXH64_hash_t hash, const void* data, size_t size) {
      return XXH64(data, size, hash);
    };

    XXH64_hash_t hash = XXH64(&m_bakingParams.sceneView, sizeof(m_bakingParams.sceneView), 0);
    hash = combine(hash, m_bakingParams.bakingCameraOrthoProjection.data(),
                   m_bakingParams.bakingCameraOrthoProjection.size() * sizeof(Matrix4));
    hash = combine(hash, &m_bakingParams.cascadeMapResolution, sizeof(m_bakingParams.cascadeMapResolution));
    hash = combine(hash, &m_bakingParams.cascadeLevelResolution, sizeof(m_bakingParams.cascadeLevelResolution));
    hash = combine(hash, m_bakingParams.sideProjectionView.data(), m_bakingParams.sideProjectionView.size() * sizeof(Matrix4));
    hash = combine(hash, m_bakingParams.sideProjectionOrthoProjection.data(),
                   m_bakingParams.sideProjectionOrthoProjection.size() * sizeof(Matrix4));
    const TerrainSideProjectionCulling sideProjectionCulling = cascadeMap.sideProjectionCulling();
    hash = combine(hash, &sideProjectionCulling, sizeof(sideProjectionCulling));

    const float displacement[] = { m_prevFrameMaxDisplaceIn, m_prevFrameMaxDisplaceOut, Material::Properties::displaceInFactor() };
    hash = combine(hash, displacement, sizeof(displacement));

    const bool materialOptions[] = {
      Material::bakeReplacementMaterials(), Material::bakeSecondaryPBRTextures(), Material::bakeMaterialConstants(),
      Material::replacementSupportInPS() };
    hash = combine(hash, materialOptions, sizeof(materialOptions));

    return hash;
  }

  // Starts the frame's baking: either from scratch, when the layout changed or the textures lost
  // their contents, or on top of what earlier frames baked.
  void TerrainBaker::beginBakedContent(Rc<RtxContext> ctx) {
    const XXH64_hash_t layoutHash = calculateBakedLayoutHash();
    const bool texturesLost =
      m_bakedContentLost || !m_materialTextures[ReplacementMaterialTextureType::AlbedoOpacity].texture.isValid();

    if (layoutHash == m_bakedLayoutHash && !texturesLost) {
      // Draw calls reusing their texels do not rebake, so they would not mark the textures they
      // wrote to as baked. Every texture holding valid texels stays bound instead.
      for (BakedTexture& texture : m_materialTextures) {
        if (texture.texture.isValid()) {
          texture.markAsBaked();
        }
      }
      return;
    }

    m_bakedLayoutHash = layoutHash;
    m_bakedDraws.clear();
    m_bakedContentLost = false;

    // Unlike the colour it guards, stale depth would reject the surfaces now being baked
    if (m_sideProjectionDepth.image != nullptr) {
      VkImageSubresourceRange subRange = {};
      subRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      subRange.levelCount = 1;
      subRange.layerCount = 1;
      ctx->clearDepthStencilImage(m_sideProjectionDepth.image, VkClearDepthStencilValue { 1.f, 0 }, subRange);
    }

    if (clearTerrainBeforeBaking() && !debugDisableBaking()) {
      for (uint32_t i = 0; i < ReplacementMaterialTextureType::Count; i++) {
        if (m_materialTextures[i].texture.isValid()) {
          clearMaterialTexture(ctx, static_cast<ReplacementMaterialTextureType::Enum>(i));
        }
      }
    }
  }

  void TerrainBaker::updateCascadeCenter(const RtCamera& camera) {
    const Vector3& position = camera.getPosition();
    const float recenterDistance = RtxOptions::getMeterToWorldUnitScale() * cascadeMap.recenterDistance();

    if (!m_cascadeCenter.has_value() || length(position - *m_cascadeCenter) > recenterDistance) {
      m_cascadeCenter = position;
    }
  }

  bool TerrainBaker::registerTerrainMesh(Rc<RtxContext> ctx, const DxvkContextState& dxvkCtxState, const DrawCallState& drawCallState) {
    const uint32_t currentFrameIndex = ctx->getDevice()->getCurrentFrameId();

    // This is the first call in a frame, set up baking state for the new frame
    if (m_bakingParams.frameIndex != currentFrameIndex) {
      // The cascade is centered on the main camera and its parameters are frozen for the
      // rest of the frame. RtCamera::update ignores every call after the first one in a
      // frame, so a terrain draw submitted before the main camera has been established
      // would center the cascade on the previous frame's camera and leave every baked
      // texel lagging the geometry it is sampled by. Wait for the camera instead, so that
      // which draw call happens to arrive first cannot move the cascade.
      if (!ctx->getSceneManager().getCamera().isValid(currentFrameIndex)) {
        ONCE(Logger::info("[RTX Terrain Baker] Terrain was submitted before the main camera was established. Those draw calls are left unbaked until the camera is known for the frame."));
        return false;
      }

      onFrameBegin(ctx, dxvkCtxState);
    }

    if (cascadeMap.useTerrainBBOX()) { 
      m_terrainMeshBBOXes.emplace_back(AxisAlignedBoundingBoxLink(drawCallState));
    }

    return true;
  }

  void TerrainBaker::calculateCascadeMapResolution(const Rc<DxvkDevice>& device) {
    // ToDo: switch to using vkGetPhysicalDeviceImageFormatProperties which may allow larger dimensions for a given image config
    const VkPhysicalDeviceLimits& limits = device->adapter()->deviceProperties().limits;
    const uint32_t maxDimension = limits.maxImageDimension2D;

    m_bakingParams.cascadeLevelResolution = VkExtent2D { cascadeMap.levelResolution(), cascadeMap.levelResolution() };

    // Calculate cascade map resolution
    m_bakingParams.cascadeMapResolution.width = m_bakingParams.cascadeMapSize.x * m_bakingParams.cascadeLevelResolution.width;
    m_bakingParams.cascadeMapResolution.height = m_bakingParams.cascadeMapSize.y * m_bakingParams.cascadeLevelResolution.height;

    // Ensure the texture resolution fits within device limits
    if (m_bakingParams.cascadeMapResolution.width > maxDimension || m_bakingParams.cascadeMapResolution.height > maxDimension) {
      const float2 downscale = {
        static_cast<float>(maxDimension) / m_bakingParams.cascadeMapResolution.width,
        static_cast<float>(maxDimension) / m_bakingParams.cascadeMapResolution.height };

      VkExtent2D prevCascadeMapResolution = m_bakingParams.cascadeMapResolution;

      m_bakingParams.cascadeLevelResolution.width = static_cast<uint32_t>(floor(downscale.x * m_bakingParams.cascadeMapResolution.width) / m_bakingParams.cascadeMapSize.x);
      m_bakingParams.cascadeLevelResolution.height = static_cast<uint32_t>(floor(downscale.y * m_bakingParams.cascadeMapResolution.height) / m_bakingParams.cascadeMapSize.y);

      m_bakingParams.cascadeMapResolution.width = m_bakingParams.cascadeLevelResolution.width * m_bakingParams.cascadeMapSize.x;
      m_bakingParams.cascadeMapResolution.height = m_bakingParams.cascadeLevelResolution.height * m_bakingParams.cascadeMapSize.y;

      ONCE(Logger::warn(str::format("[RTX Terrain Baker] Requested terrain cascade map resolution {", prevCascadeMapResolution.width, ", ", prevCascadeMapResolution.height, "} is outside the device limits {", maxDimension, ", ", maxDimension, "}. Reducing the cascade map resolution to {", m_bakingParams.cascadeMapResolution.width, ", ", m_bakingParams.cascadeMapResolution.height, "}.")));
    }
  }

  void TerrainBaker::calculateBakingParameters(Rc<RtxContext> ctx, const DxvkContextState& dxvkCtxState) {

    SceneManager& sceneManager = ctx->getSceneManager();
    Resources& resourceManager = ctx->getResourceManager();
    const RtCamera& camera = sceneManager.getCamera();
    const uint32_t currentFrameIndex = ctx->getDevice()->getCurrentFrameId();
    const float metersToWorldUnitScale = RtxOptions::getMeterToWorldUnitScale();

    m_bakingParams.frameIndex = currentFrameIndex;

    updateCascadeCenter(camera);
    const Vector3& cascadeCenter = *m_cascadeCenter;

    const bool terrainBBOXIsValid = m_bakedTerrainBBOX.isValid();
    const float epsilon = 0.01f;      // Epsilon to ensure distances are greater or equal

    const float terrainHeight =
      terrainBBOXIsValid
      ? SceneManager::worldToSceneOrientedVector(m_bakedTerrainBBOX.maxPos - m_bakedTerrainBBOX.minPos).z
      : metersToWorldUnitScale * cascadeMap.defaultHeight();

    const float cameraRelativeTerrainHeight =
      terrainBBOXIsValid
      ? SceneManager::worldToSceneOrientedVector(m_bakedTerrainBBOX.maxPos - cascadeCenter).z
      : metersToWorldUnitScale * cascadeMap.defaultHeight() / 2; // Assume the center is in the middle of terrain's height span

    // Constants set to what makes generally should make sense
    // Offset zFar by zNear to match the baking camera position being offset by it.
    // Offset by 1.f in case terrainHeight is zero (i.e. it's planar)
    const float zNear = 0.01f;
    const float zFar = (terrainHeight + 1.f) * (1 + epsilon) + zNear;

    // Compute the relative half width of the cascade map around the camera
    float cascadeMapHalfWidth = metersToWorldUnitScale * cascadeMap.defaultHalfWidth();
    if (terrainBBOXIsValid) {
      // Add offset for all terrain samples to be within the baked terrain texture
      const float halfTexelOffset = 10;       // ToDo: calculate an exact value

      // Compute bbox relative to the cascade center
      AxisAlignedBoundingBox cameraRelativeTerrainBBOX = {
        m_bakedTerrainBBOX.minPos - cascadeCenter - Vector3{ halfTexelOffset },
        m_bakedTerrainBBOX.maxPos - cascadeCenter + Vector3{ halfTexelOffset }
      };

      // Convert the bbox to scene space
      cameraRelativeTerrainBBOX.minPos = SceneManager::worldToSceneOrientedVector(cameraRelativeTerrainBBOX.minPos);
      cameraRelativeTerrainBBOX.maxPos = SceneManager::worldToSceneOrientedVector(cameraRelativeTerrainBBOX.maxPos);

      // Calculate a half width of a cascade map around camera that covers the terrain's BBOX
      cascadeMapHalfWidth =
        std::max(std::max(std::abs(cameraRelativeTerrainBBOX.maxPos.x), std::abs(cameraRelativeTerrainBBOX.minPos.x)),
                 std::max(std::abs(cameraRelativeTerrainBBOX.maxPos.y), std::abs(cameraRelativeTerrainBBOX.minPos.y)));
    }

    // Construct a scene oriented view
    Matrix4 sceneView;
    {
      const Vector3 up = SceneManager::getSceneUp();
      const Vector3 forward = SceneManager::getSceneForward();
      const Vector3 right = SceneManager::calculateSceneRight();

      // Set baking camera position just above the terrain
      // Offset by zNear so that zNear doesn't clip the terrain
      // Offset by epsilon so that it doesn't clip top of the terrain
      const Vector3 bakingCameraPosition = cameraRelativeTerrainHeight >= 0.f
        ? cascadeCenter + (cameraRelativeTerrainHeight * (1 + epsilon) + zNear) * up
        : cascadeCenter + (cameraRelativeTerrainHeight * (1 - epsilon) - zNear) * up;

      const Vector3 translation = Vector3(
        dot(right, -bakingCameraPosition),
        dot(forward, -bakingCameraPosition),
        dot(up, -bakingCameraPosition));

      sceneView[0] = Vector4(right.x, forward.x, up.x, 0.f);
      sceneView[1] = Vector4(right.y, forward.y, up.y, 0.f);
      sceneView[2] = Vector4(right.z, forward.z, up.z, 0.f);
      sceneView[3] = Vector4(translation.x, translation.y, translation.z, 1.f);
    }

    m_bakingParams.sceneView = sceneView;

    // Number of cascades required to cover the whole bbox
    const uint32_t numRequiredCascades = 
      1 + static_cast<uint32_t>(ceil(log2(std::max(1.f, cascadeMapHalfWidth / (metersToWorldUnitScale * cascadeMap.levelHalfWidth())))));

    // Number of cascades actually used
    m_bakingParams.numCascades = std::min(cascadeMap.maxLevels(), numRequiredCascades);

    // If there isn't enough cascades to cover the terrain radius, expand the last cascade to cover the cascade map's span
    const bool isLastCascadeExpanded = m_bakingParams.numCascades != numRequiredCascades;
    m_bakingParams.lastCascadeScale = 1.f;

    m_bakingParams.numSideProjectionLevels =
      m_sideProjectionsNeeded ? std::min(cascadeMap.sideProjectionLevels(), m_bakingParams.numCascades) : 0;
    m_bakingParams.numSideProjectionTiles = (m_bakingParams.numSideProjectionLevels * kNumTerrainSideProjectionViews + 3) / 4;

    // The side projection tiles must all fit in the first row, which is all their depth buffer covers
    const uint32_t numTiles = m_bakingParams.numSideProjectionTiles + m_bakingParams.numCascades;
    m_bakingParams.cascadeMapSize.x = std::max(static_cast<uint32_t>(ceilf(sqrtf(static_cast<float>(numTiles)))),
                                               m_bakingParams.numSideProjectionTiles);
    m_bakingParams.cascadeMapSize.y = static_cast<uint32_t>(ceilf(static_cast<float>(numTiles) / m_bakingParams.cascadeMapSize.x));

    m_bakingParams.bakingCameraOrthoProjection.resize(m_bakingParams.numCascades);

    // Calculate cascade map resolution
    calculateCascadeMapResolution(ctx->getDevice());

    // Calculate params for each cascade level
    for (uint32_t iCascade = 0; iCascade < m_bakingParams.numCascades; iCascade++) {
      // Half width of the cascade level
      float halfWidth = metersToWorldUnitScale * cascadeMap.levelHalfWidth() * pow(2, iCascade);

      // Expand the last cascade level if necessary
      const bool isLastCascade = iCascade == m_bakingParams.numCascades - 1;
      if (isLastCascade && isLastCascadeExpanded && cascadeMap.expandLastCascade()) {
        // Note: 1st cascade is naturally expanded by matching the projection to the expanded range, rather than applying 
        // expansion scale if it is to be expanded. But for pedantic purposes we set the scale to 1 here anyway
        m_bakingParams.lastCascadeScale = iCascade > 0 ? cascadeMapHalfWidth / halfWidth : 1.f;
        halfWidth = cascadeMapHalfWidth;
      }

      // Setup orthographic projection top-down that maps <-halfWidth, halfWidth> around camera to <0, 1> in clip space
      float4x4& newProjection = *reinterpret_cast<float4x4*>(&m_bakingParams.bakingCameraOrthoProjection[iCascade]);
      newProjection.SetupByOrthoProjection(-halfWidth, halfWidth, -halfWidth, halfWidth, zNear, zFar);

      if (iCascade == 0) {
        // Convert from clip space <-1, 1> to <0, 1> and flip y coordinate for Vulkan
        const Matrix4 textureOffset = Matrix4(Vector4(.5f, 0, 0, 0),
                                              Vector4(0, -.5f, 0, 0),
                                              Vector4(0, 0, 1, 0),
                                              Vector4(.5f, .5f, 0, 1));

        m_bakingParams.viewToCascade0TextureSpace = textureOffset * m_bakingParams.bakingCameraOrthoProjection[iCascade] * sceneView * camera.getViewToWorld();
      }
    }

    calculateSideProjectionParameters(cascadeCenter, zNear);
  }

  // Each level's side projection views look at a cube around the map's centre, as wide as the level:
  // from each horizontal direction with the scene's up as the texture's vertical axis, and from below.
  // A view's depth range spans the cube, so only surfaces within the level can occlude one another.
  void TerrainBaker::calculateSideProjectionParameters(const Vector3& cascadeCenter, float zNear) {
    const uint32_t numSideViews = m_bakingParams.numSideProjectionLevels * kNumTerrainSideProjectionViews;
    m_bakingParams.sideProjectionView.resize(numSideViews);
    m_bakingParams.sideProjectionOrthoProjection.resize(numSideViews);

    const Vector3 up = SceneManager::getSceneUp();
    const Vector3 forward = SceneManager::getSceneForward();
    const Vector3 right = SceneManager::calculateSceneRight();

    // The side views keep the top-down view's handedness, so that the draw call's winding means the same in all of them
    const float handedness = dot(cross(right, forward), up);

    const Vector3 directions[kNumTerrainSideProjectionViews] = { right, -right, forward, -forward, -up };
    std::copy(std::begin(directions), std::end(directions), std::begin(m_bakingParams.sideProjectionViewDirection));

    for (uint32_t level = 0; level < m_bakingParams.numSideProjectionLevels; level++) {
      const float halfWidth = RtxOptions::getMeterToWorldUnitScale() * cascadeMap.levelHalfWidth() * static_cast<float>(1u << level);

      for (uint32_t view = 0; view < kNumTerrainSideProjectionViews; view++) {
        const uint32_t sideView = level * kNumTerrainSideProjectionViews + view;

        const Vector3& towardsCamera = directions[view];
        const Vector3 textureUp = view == kTerrainSideProjectionViewBelow ? forward : up;
        Vector3 textureRight = cross(textureUp, towardsCamera);
        if (dot(cross(textureRight, textureUp), towardsCamera) * handedness < 0.f) {
          textureRight = -textureRight;
        }

        const Vector3 position = cascadeCenter + (halfWidth + zNear) * towardsCamera;
        const Vector3 translation = Vector3(
          dot(textureRight, -position),
          dot(textureUp, -position),
          dot(towardsCamera, -position));

        Matrix4& viewMatrix = m_bakingParams.sideProjectionView[sideView];
        viewMatrix[0] = Vector4(textureRight.x, textureUp.x, towardsCamera.x, 0.f);
        viewMatrix[1] = Vector4(textureRight.y, textureUp.y, towardsCamera.y, 0.f);
        viewMatrix[2] = Vector4(textureRight.z, textureUp.z, towardsCamera.z, 0.f);
        viewMatrix[3] = Vector4(translation.x, translation.y, translation.z, 1.f);

        float4x4& projection = *reinterpret_cast<float4x4*>(&m_bakingParams.sideProjectionOrthoProjection[sideView]);
        projection.SetupByOrthoProjection(-halfWidth, halfWidth, -halfWidth, halfWidth, zNear, zNear + 2.f * halfWidth);
      }
    }
  }

  // Top-down levels follow the side projection tiles, left to right, top to bottom
  VkRect2D TerrainBaker::getTopDownViewRect(uint32_t level) const {
    const uint32_t tile = m_bakingParams.numSideProjectionTiles + level;
    const VkExtent2D& resolution = m_bakingParams.cascadeLevelResolution;
    return VkRect2D {
      VkOffset2D {
        static_cast<int32_t>((tile % m_bakingParams.cascadeMapSize.x) * resolution.width),
        static_cast<int32_t>((tile / m_bakingParams.cascadeMapSize.x) * resolution.height) },
      resolution };
  }

  // Side projection views take a quarter of a tile each, four to a tile, in the first row
  VkRect2D TerrainBaker::getSideProjectionViewRect(uint32_t sideView) const {
    const uint32_t tile = sideView / 4;
    const uint32_t quarter = sideView % 4;
    const VkExtent2D quarterResolution = {
      m_bakingParams.cascadeLevelResolution.width / 2, m_bakingParams.cascadeLevelResolution.height / 2 };
    return VkRect2D {
      VkOffset2D {
        static_cast<int32_t>(tile * m_bakingParams.cascadeLevelResolution.width + (quarter % 2) * quarterResolution.width),
        static_cast<int32_t>((quarter / 2) * quarterResolution.height) },
      quarterResolution };
  }

  void TerrainBaker::updateSideProjectionDepth(Rc<DxvkContext> ctx) {
    RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();

    if (m_bakingParams.numSideProjectionTiles == 0) {
      releaseSideProjectionDepth(textureManager);
      return;
    }

    const VkExtent3D extent = {
      m_bakingParams.numSideProjectionTiles * m_bakingParams.cascadeLevelResolution.width,
      m_bakingParams.cascadeLevelResolution.height,
      1 };
    if (m_sideProjectionDepth.image != nullptr && m_sideProjectionDepth.image->info().extent == extent) {
      return;
    }

    releaseSideProjectionDepth(textureManager);

    const VkFormat format = VK_FORMAT_D16_UNORM;

    DxvkImageCreateInfo desc;
    desc.type = VK_IMAGE_TYPE_2D;
    desc.format = format;
    desc.flags = 0;
    desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    desc.extent = extent;
    desc.numLayers = 1;
    desc.mipLevels = 1;
    desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    desc.stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                  VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    desc.access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    desc.tiling = VK_IMAGE_TILING_OPTIMAL;
    desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    m_sideProjectionDepth.image = ctx->getDevice()->createImage(
      desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "terrain side projection depth");

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 1;

    viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    m_sideProjectionDepth.view = ctx->getDevice()->createImageView(m_sideProjectionDepth.image, viewInfo);
    viewInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    m_sideProjectionDepthTarget = ctx->getDevice()->createImageView(m_sideProjectionDepth.image, viewInfo);

    ctx->changeImageLayout(m_sideProjectionDepth.image, VK_IMAGE_LAYOUT_GENERAL);

    // Cleared by beginBakedContent, which rebakes everything into it
    m_bakedContentLost = true;
  }

  void TerrainBaker::releaseSideProjectionDepth(RtxTextureManager& textureManager) {
    if (m_sideProjectionDepth.view != nullptr) {
      // WAR (REMIX-1557) to force release the texture reference from the texture cache, as for the terrain textures
      TextureRef textureRef = TextureRef(m_sideProjectionDepth.view);
      textureManager.releaseTexture(textureRef);
    }
    m_sideProjectionDepth.reset();
    m_sideProjectionDepthTarget = nullptr;
    m_sideProjectionDepthTextureIndex = kSurfaceMaterialInvalidTextureIndex;
  }
}
