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

namespace ReplacementMaterialTextureType {
  enum Enum {
    AlbedoOpacity = 0,
    Normal,
    Tangent,
    Height,
    Roughness,
    Metallic,
    Emissive,

    Count
  };
}

namespace ReplacementMaterialTextureCategory {
  enum Enum {
    AlbedoOpacity = 0,
    SecondaryRaw,                // Roughness, Metallic, Emissive
    SecondaryScaled,             // Height
    SecondaryOctahedralEncoded   // Normal, Tangent
  };
}

#ifdef __cplusplus
// Texture stage used for secondary texture during terrain baking.
// This could be made bound to a dynamic slot or to a different resource should it 
// conflict in any game in the future.
static const unsigned int kTerrainBakerSecondaryTextureStage = 7;

// The terrain baking spec constant carries two things: which category of replacement texture is
// being baked, and which D3D9 texture stage the baker swapped it in at. They share one constant
// because D3D9SpecConstantId is already at DxvkLimits::MaxNumSpecConstants, and a new one would
// grow the pipeline state key for every pipeline in the renderer.
//
// The stage matters because only that one stage holds a replacement texture. Every other stage
// keeps sampling the game's own -- a blend mask, a lightmap -- so the baking postprocess must be
// applied to the swapped stage alone. Applied to all of them, a mask gets decoded as if it were
// an octahedral normal and its alpha replaced by the albedo's, which makes a multi-stage terrain
// draw bake its secondary textures with the wrong blend weight.
static const unsigned int kReplacementTextureStageShift = 8;
static const unsigned int kReplacementTextureCategoryMask = 0xFF;

static inline unsigned int packReplacementTextureSpecConstant(unsigned int category, unsigned int stage) {
  return (category & kReplacementTextureCategoryMask) | (stage << kReplacementTextureStageShift);
}
#endif