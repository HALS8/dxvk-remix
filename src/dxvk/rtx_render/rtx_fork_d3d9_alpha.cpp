// rtx_fork_d3d9_alpha.cpp -- options, shader constants and dev-menu controls for the D3D9 alpha / replacement GPU
// work (fork, d3d9/alpha-gpu). See rtx_fork_d3d9_alpha.h.

#include "rtx_fork_d3d9_alpha.h"

#include "rtx_imgui.h"
#include "rtx/pass/raytrace_args.h"

namespace dxvk {

  void D3d9Alpha::setRaytraceArgs(RaytraceArgs& args) {
    args.particleSkipZeroAlbedoLighting = particleSkipZeroAlbedoLighting() ? 1u : 0u;
  }

  void D3d9Alpha::showImguiSettings() {
    if (RemixGui::CollapsingHeader("D3D9 Alpha / Replacements", ImGuiTreeNodeFlags_CollapsingHeader)) {
      ImGui::Indent();
      ImGui::PushID("D3d9Alpha");

      RemixGui::Checkbox("OMM on Shared Per-Geometry BLAS", &bindSharedBlasObject());
      RemixGui::Checkbox("Tracker: Keep Instance on Material Change (may be visible)", &trackerMaterialFallbackObject());
      RemixGui::Checkbox("Particles: Skip Lighting of Zero-Albedo Texels", &particleSkipZeroAlbedoLightingObject());

      ImGui::PopID();
      ImGui::Unindent();
    }
  }

} // namespace dxvk
