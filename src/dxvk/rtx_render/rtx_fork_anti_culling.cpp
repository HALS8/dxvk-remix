// rtx_fork_anti_culling.cpp -- options, distance test, statistics and dev-menu controls for distance-bounded
// anti-culling retention (fork, anti-culling/retention). See rtx_fork_anti_culling.h.

#include "rtx_fork_anti_culling.h"

#include <algorithm>
#include <cmath>

#include "rtx_imgui.h"
#include "rtx_options.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  namespace {
    constexpr uint32_t kFramesPerReport = 600;

    uint32_t average(uint64_t sum, uint32_t frames) {
      return frames == 0 ? 0u : static_cast<uint32_t>(sum / frames);
    }
  }

  float AntiCullingRetention::objectDistance() {
    return std::max(0.0f, Object::retentionDistanceMeters()) * RtxOptions::getMeterToWorldUnitScale();
  }

  float AntiCullingRetention::lightDistance() {
    return std::max(0.0f, Light::retentionDistanceMeters()) * RtxOptions::getMeterToWorldUnitScale();
  }

  float AntiCullingRetention::distanceToBounds(
      const Vector3& point, const AxisAlignedBoundingBox& bounds, const Matrix4& objectToWorld) {
    const Vector3 centre = bounds.getTransformedCentroid(objectToWorld);
    const Vector3 halfExtent = (bounds.maxPos - bounds.minPos) * 0.5f;

    float distanceSqr = 0.0f;
    for (uint32_t axis = 0; axis < 3; axis++) {
      const float worldHalfExtent =
          std::abs(objectToWorld[0][axis]) * halfExtent.x +
          std::abs(objectToWorld[1][axis]) * halfExtent.y +
          std::abs(objectToWorld[2][axis]) * halfExtent.z;
      const float outside = std::max(0.0f, std::abs(point[axis] - centre[axis]) - worldHalfExtent);
      distanceSqr += outside * outside;
    }
    return std::sqrt(distanceSqr);
  }

  void AntiCullingRetention::countInstanceRelease(Release reason, uint32_t count) {
    s_window.instanceReleases[static_cast<size_t>(reason)] += count;
    if (reason == Release::OverCap) {
      s_window.capEvents++;
    }
  }

  void AntiCullingRetention::countLightRelease(Release reason) {
    s_window.lightReleases[static_cast<size_t>(reason)]++;
  }

  void AntiCullingRetention::countInstances(uint32_t total, uint32_t keptObjects, uint32_t keptLights) {
    s_window.instancesSum += total;
    s_window.instancesPeak = std::max(s_window.instancesPeak, total);
    s_window.keptObjectsSum += keptObjects;
    s_window.keptObjectsPeak = std::max(s_window.keptObjectsPeak, keptObjects);
    s_window.keptInstanceLightsSum += keptLights;
    s_window.keptInstanceLightsPeak = std::max(s_window.keptInstanceLightsPeak, keptLights);
  }

  void AntiCullingRetention::countLights(uint32_t total, uint32_t kept) {
    s_window.lightsSum += total;
    s_window.lightsPeak = std::max(s_window.lightsPeak, total);
    s_window.keptLightsSum += kept;
    s_window.keptLightsPeak = std::max(s_window.keptLightsPeak, kept);
  }

  void AntiCullingRetention::endFrame(uint32_t frameId) {
    if (s_window.frames == 0) {
      s_window.firstFrame = frameId;
    }
    if (++s_window.frames < kFramesPerReport) {
      return;
    }
    if (logStatistics()) {
      log(s_window, frameId);
    }
    s_window = Window {};
  }

  void AntiCullingRetention::log(const Window& window, uint32_t lastFrame) {
    const auto instances = [&](Release reason) { return window.instanceReleases[static_cast<size_t>(reason)]; };
    const auto lights = [&](Release reason) { return window.lightReleases[static_cast<size_t>(reason)]; };

    Logger::info(str::format(
        "[AntiCulling] frames ", window.firstFrame, "-", lastFrame,
        " | object rule: ", RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? "on" : "off",
        ", distance ", Object::retentionDistanceMeters(), " m, cap ", RtxOptions::AntiCulling::Object::numObjectsToKeep(),
        " | light rule: ", RtxOptions::AntiCulling::isLightAntiCullingEnabled() ? "on" : "off",
        ", distance ", Light::retentionDistanceMeters(), " m, lifetime ",
        RtxOptions::AntiCulling::Light::numFramesToExtendLightLifetime(), " frames"));
    Logger::info(str::format(
        "[AntiCulling]   instances: average ", average(window.instancesSum, window.frames), ", peak ", window.instancesPeak,
        " | kept objects: average ", average(window.keptObjectsSum, window.frames), ", peak ", window.keptObjectsPeak,
        " | kept mesh lights: average ", average(window.keptInstanceLightsSum, window.frames),
        ", peak ", window.keptInstanceLightsPeak));
    Logger::info(str::format(
        "[AntiCulling]   instances released: inside frustum ", instances(Release::InsideFrustum),
        ", beyond distance ", instances(Release::BeyondDistance),
        ", over cap ", instances(Release::OverCap), " in ", window.capEvents, " passes",
        ", light lifetime ", instances(Release::LifetimeExpired),
        ", not covered ", instances(Release::NotCovered),
        " | exempt: skinned ", instances(Release::Skinned),
        ", player model ", instances(Release::PlayerModel),
        ", ignore category ", instances(Release::IgnoreCategory),
        ", moving ", instances(Release::Moving),
        ", never matched ", instances(Release::NeverMatched),
        ", camera cut ", instances(Release::CameraCut)));
    Logger::info(str::format(
        "[AntiCulling]   game lights: average ", average(window.lightsSum, window.frames), ", peak ", window.lightsPeak,
        " | kept: average ", average(window.keptLightsSum, window.frames), ", peak ", window.keptLightsPeak,
        " | released: inside frustum ", lights(Release::InsideFrustum),
        ", beyond distance ", lights(Release::BeyondDistance),
        ", lifetime ", lights(Release::LifetimeExpired),
        ", not covered ", lights(Release::NotCovered)));
  }

  void AntiCullingRetention::showImguiSettings() {
    RemixGui::Separator();
    RemixGui::DragFloat("Object Retention Distance (m, 0 = unlimited)", &Object::retentionDistanceMetersObject(), 1.0f, 0.0f, 5000.0f, "%.0f");
    RemixGui::DragFloat("Light Retention Distance (m, 0 = frame limit)", &Light::retentionDistanceMetersObject(), 1.0f, 0.0f, 5000.0f, "%.0f");
    RemixGui::Checkbox("Log Anti-Culling Statistics", &logStatisticsObject());
  }

} // namespace dxvk
