#pragma once

// rtx_fork_d3d9_alpha.h -- GPU/BLAS work for alpha-tested/blended content and replacement assets on the standard D3D9
// path (fork, d3d9/alpha-gpu; aimed at Half-Life 2 RTX).
//
// Every behaviour here is off by default and has a control in the dev menu (Rendering, "D3D9 Alpha / Replacements",
// next to "Opacity Micromap"):
// - opacityMicromap.bindSharedBlas: per-geometry BLASes shared by several instances bind the opacity micromap their
//   instances all request (OpacityMicromapManager::checkSharedBlasOmm). Before, only single-instance ones could.
// - trackerMaterialFallback: the draw-call tracker keeps an instance whose material changed at the same transform
//   (animated textures) instead of creating a new one. Can be visible (motion vectors, denoiser history).
// - particleSkipZeroAlbedoLighting: the particle lighting approximation skips texels with zero albedo.

#include <cstdint>

#include "rtx_option.h"

struct RaytraceArgs;

namespace dxvk {

  class D3d9Alpha {
  public:
    RTX_OPTION("rtx.opacityMicromap", bool, bindSharedBlas, true,
               "Per-geometry BLASes shared by several instances (meshes above rtx.minPrimsInDynamicBLAS drawn more than once,\n"
               "e.g. repeated replacement foliage and fences) bind an opacity micromap when every instance requests the same\n"
               "4-state one: same OMM source hash (material, alpha state, texture stage ops, texture transform, geometry) and\n"
               "the same tFactor and colour flags. Before, such BLASes never got one, although the micromaps were baked.\n"
               "Same result as for single-instance geometry: unknown micro-triangles keep the exact alpha test.");
    RTX_OPTION("rtx", bool, trackerMaterialFallback, false,
               "Draw-call tracker: when no previous-frame instance with the same material is found, reuse one at exactly the\n"
               "same transform with the same vertex positions whose material changed (animated textures, texture flipbooks)\n"
               "instead of creating a new instance. Keeps the instance's history: motion vectors and denoiser history of such\n"
               "surfaces change, so the image can differ slightly. Off = original behaviour.");
    RTX_OPTION("rtx", bool, particleSkipZeroAlbedoLighting, false,
               "Particle lighting approximation (unordered resolve): skip the froxel radiance and sky lookups for texels whose\n"
               "albedo is zero, which they would multiply by zero. Same result.");

    static void setRaytraceArgs(RaytraceArgs& args);
    static void showImguiSettings();
  };

} // namespace dxvk
