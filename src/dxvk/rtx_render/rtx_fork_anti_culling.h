#pragma once

// rtx_fork_anti_culling.h -- distance-bounded anti-culling retention and its statistics (fork, anti-culling/retention).
//
// Upstream keeps an object the game stopped drawing for as long as it stays outside the camera frustum, and a light for
// a fixed number of frames. Under a path tracer what matters is whether the thing is still near enough to cast a shadow
// or light the view, so both can instead be bounded by distance from the camera:
// - rtx.antiCulling.object.retentionDistanceMeters: an object outside the frustum is kept only while it is within this
//   distance. 0 keeps upstream's rule (no distance limit).
// - rtx.antiCulling.light.retentionDistanceMeters: a light outside the frustum is kept while it is within this distance,
//   however long ago it was last submitted. 0 keeps upstream's rule (rtx.antiCulling.light.numFramesToExtendLightLifetime).
// When the instance count passes rtx.antiCulling.object.numObjectsToKeep, the kept instances farthest from the camera
// are released first, and only as many as the excess.
// rtx.antiCulling.logStatistics reports what the garbage collector kept and released, and why.

#include <array>
#include <cstdint>

#include "rtx_option.h"
#include "../../util/util_bounding_box.h"
#include "../../util/util_matrix.h"
#include "../../util/util_vector.h"

namespace dxvk {

  class AntiCullingRetention {
  public:
    struct Object {
      RTX_OPTION("rtx.antiCulling.object", float, retentionDistanceMeters, 0.0f,
                 "Object anti-culling keeps an object outside the camera frustum only while its bounds are within this\n"
                 "distance of the camera, in metres. 0 = no distance limit: such objects are kept until\n"
                 "rtx.antiCulling.object.numObjectsToKeep is reached.");
    };
    struct Light {
      RTX_OPTION("rtx.antiCulling.light", float, retentionDistanceMeters, 0.0f,
                 "Light anti-culling keeps a light outside the camera frustum while it is within this distance of the\n"
                 "camera, in metres, with no time limit. 0 = keep it for\n"
                 "rtx.antiCulling.light.numFramesToExtendLightLifetime frames instead, at any distance.");
    };
    RTX_OPTION("rtx.antiCulling", bool, logStatistics, false,
               "Logs, every 600 frames, how many instances and lights anti-culling kept alive and how many the garbage\n"
               "collector released for each reason.");

    // Why the garbage collector released something the game stopped submitting.
    enum class Release : uint32_t {
      CameraCut,
      NeverMatched,        // created and never submitted again
      Skinned,
      PlayerModel,
      IgnoreCategory,      // tagged rtx.antiCulling.antiCullingTextures
      Moving,              // its transform changed on its last submission
      NotCovered,          // anti-culling off or unsupported, or nothing to bound it by
      InsideFrustum,
      BeyondDistance,
      LifetimeExpired,
      OverCap,

      Count
    };

    // Retention distances in world units. 0 = no distance rule.
    static float objectDistance();
    static float lightDistance();

    // Distance from a world-space point to the world-space box enclosing an object-space box. 0 inside it.
    static float distanceToBounds(const Vector3& point, const AxisAlignedBoundingBox& bounds, const Matrix4& objectToWorld);

    static void countInstanceRelease(Release reason, uint32_t count = 1);
    static void countLightRelease(Release reason);
    static void countInstances(uint32_t total, uint32_t keptObjects, uint32_t keptLights);
    static void countLights(uint32_t total, uint32_t kept);
    // Call once per frame after every garbage-collection pass.
    static void endFrame(uint32_t frameId);

    static void showImguiSettings();

  private:
    struct Window {
      uint32_t frames = 0;
      uint32_t firstFrame = 0;

      uint64_t instancesSum = 0;
      uint32_t instancesPeak = 0;
      uint64_t keptObjectsSum = 0;
      uint32_t keptObjectsPeak = 0;
      uint64_t keptInstanceLightsSum = 0;
      uint32_t keptInstanceLightsPeak = 0;
      std::array<uint32_t, static_cast<size_t>(Release::Count)> instanceReleases {};
      uint32_t capEvents = 0;

      uint64_t lightsSum = 0;
      uint32_t lightsPeak = 0;
      uint64_t keptLightsSum = 0;
      uint32_t keptLightsPeak = 0;
      std::array<uint32_t, static_cast<size_t>(Release::Count)> lightReleases {};
    };

    static void log(const Window& window, uint32_t lastFrame);

    inline static Window s_window;
  };

} // namespace dxvk
