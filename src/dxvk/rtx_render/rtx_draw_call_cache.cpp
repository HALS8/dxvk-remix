/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_draw_call_cache.h"
#include "../d3d9/d3d9_state.h"
#include "rtx_instance_manager.h"
#include "rtx_options.h"

namespace dxvk
{

namespace {
  bool isSameSkyClass(const DrawCallState& drawCall, const BlasEntry& blas) {
    return (drawCall.cameraType == CameraType::Sky) == (blas.input.cameraType == CameraType::Sky);
  }

  bool exactMatch(const DrawCallState& drawCall, BlasEntry& blas) {
    return isSameSkyClass(drawCall, blas)
        && drawCall.getMaterialData().getHash() == blas.input.getMaterialData().getHash()
        && drawCall.getGeometryData().getHashForRule<rules::FullGeometryHash>() == blas.input.getGeometryData().getHashForRule<rules::FullGeometryHash>()
        && drawCall.getSkinningState().boneHash == blas.input.getSkinningState().boneHash;
  }

  bool isLinkedToOtherLiveInstance(const BlasEntry& blas, const RtInstance* owner) {
    for (const RtInstance* linked : blas.getLinkedInstances()) {
      if (linked != owner && !linked->isMarkedForGC()) {
        return true;
      }
    }
    return false;
  }

  // Whether an entry may be reused for a draw that is not an exact match. Different meshes of one
  // topology share a bucket, so the entry must agree on what the draw is drawn with and from: the
  // same sky class and vertex layout, and either the same material or the same pre-shader positions
  // and bones.
  bool isCompatible(const DrawCallState& drawCall, const BlasEntry& blas) {
    if (!isSameSkyClass(drawCall, blas)) {
      return false;
    }
    const RasterGeometry& geometry = drawCall.getGeometryData();
    if (blas.modifiedGeometryData.hashes[HashComponents::VertexLayout] != geometry.hashes[HashComponents::VertexLayout]) {
      return false;
    }
    const bool sameMaterial = blas.input.getMaterialData().getHash() == drawCall.getMaterialData().getHash();
    const bool sameSourceMesh = blas.modifiedGeometryData.hashes[HashComponents::VertexPosition] == geometry.hashes[HashComponents::VertexPosition]
                             && blas.input.getSkinningState().boneHash == drawCall.getSkinningState().boneHash;
    return sameMaterial || sameSourceMesh;
  }

  // Measured to the instance that last drew the entry, not to every instance linked to it.
  float worldDistanceSqr(const BlasEntry& blas, const Vector3& newWorldPosition) {
    const Matrix4 oldTransform = blas.input.getTransformData().objectToWorld;
    return lengthSqr(newWorldPosition - blas.input.getGeometryData().boundingBox.getTransformedCentroid(oldTransform));
  }
}

DrawCallCache::DrawCallCache(DxvkDevice* device) : CommonDeviceObject(device) {
  m_entries.reserve(1024);
}
DrawCallCache::~DrawCallCache() {}

DrawCallCache::CacheState DrawCallCache::get(const DrawCallState& drawCall, BlasEntry** out, const RtInstance* owner) {
  // First, find the right bucket:
  const XXH64_hash_t hash = drawCall.getGeometryData().getHashForRule<rules::TopologicalHash>();
  auto range = m_entries.equal_range(hash);
  if (range.first == m_entries.end()) {
    // New bucket
    *out = allocateEntry(hash, drawCall);
    return CacheState::kNew;
  }

  const bool strict = strictCachePairing();
  const uint32_t currentFrame = m_device->getCurrentFrameId();
  const Matrix4 newTransform = drawCall.getTransformData().objectToWorld;
  const Vector3 newWorldPosition = drawCall.getGeometryData().boundingBox.getTransformedCentroid(newTransform);

  // The tracker has already chosen which instance this draw updates and bounded its distance, so
  // that instance's own entry is the right one whenever it still describes the same mesh.
  if (strict && owner != nullptr && !owner->isMarkedForGC()) {
    BlasEntry* own = owner->getBlas();
    if (own != nullptr && own->input.getGeometryData().getHashForRule<rules::TopologicalHash>() == hash) {
      if (exactMatch(drawCall, *own)) {
        *out = own;
        return CacheState::kExact;
      }
      if (own->frameLastTouched != currentFrame && isCompatible(drawCall, *own)) {
        *out = own;
        return CacheState::kOwn;
      }
    }
  }

  // Handle buckets with 1 entry:
  auto iter = range.first;
  iter++;
  if (iter == range.second) {
    // Only 1 element
    BlasEntry& entry = range.first->second;

    const bool updatedThisFrame = entry.frameLastTouched == currentFrame;
    const bool vertexDataMatches = entry.input.getGeometryData().getHashForRule<rules::VertexDataHash>() == drawCall.getGeometryData().getHashForRule<rules::VertexDataHash>();
    const bool boneHashesMatch = entry.input.getSkinningState().boneHash == drawCall.getSkinningState().boneHash;
    const bool materialHashesMatch = entry.input.getMaterialData().getHash() == drawCall.getMaterialData().getHash();

    const bool similarEnough = strict
      ? !isLinkedToOtherLiveInstance(entry, owner) && isCompatible(drawCall, entry)
          && worldDistanceSqr(entry, newWorldPosition) <= RtxOptions::getUniqueObjectDistanceSqr()
      : vertexDataMatches && boneHashesMatch || materialHashesMatch;

    if (exactMatch(drawCall, entry)) {
      *out = &entry;
      return CacheState::kExact;
    } else if (!updatedThisFrame && similarEnough) {
      // Something that hasn't been updated this frame and is similar enough.
      // Matching the logic in the multi-element loop below.
      *out = &entry;
      return CacheState::kSimilar;
    } else {
      // First frame of having two mismatching instances, and the first instance has already
      // been paired with the existing BlasEntry.
      *out = allocateEntry(hash, drawCall);
      return CacheState::kNew;
    }
  }

  // Bucket has multiple BlasEntries

  // Unless pairing is strict the initial value is the smallest positive float, which makes a score
  // above zero the acceptance threshold.
  float bestScore = strict ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::min();

  for (auto bucketIter = range.first; bucketIter != range.second; bucketIter++) {
    BlasEntry& blas  = bucketIter->second;
    if (exactMatch(drawCall, blas)) {
      *out = &blas;
      return CacheState::kExact;
    }
    if (blas.frameLastTouched == currentFrame) {
      continue;
    }
    if (strict && (isLinkedToOtherLiveInstance(blas, owner) || !isCompatible(drawCall, blas))) {
      continue;
    }
    const float distanceSqr = worldDistanceSqr(blas, newWorldPosition);
    if (strict && distanceSqr > RtxOptions::getUniqueObjectDistanceSqr()) {
      continue;
    }
    // TODO these heuristics could use more refinement.
    float score = 0;
    if (blas.modifiedGeometryData.hashes[HashComponents::VertexPosition] == drawCall.getGeometryData().hashes[HashComponents::VertexPosition] &&
        blas.input.getSkinningState().boneHash == drawCall.getSkinningState().boneHash) {
      score += 1000.f;
    }
    if (blas.modifiedGeometryData.hashes[HashComponents::VertexTexcoord] == drawCall.getGeometryData().hashes[HashComponents::VertexTexcoord]) {
      score += 1000.f;
    }
    if (blas.input.getMaterialData().getHash() == drawCall.getMaterialData().getHash()) {
      score += 1000.f;
    }
    // TODO the distance does not include the portal logic from InstanceManager.
    score -= distanceSqr;
    if (score > bestScore) {
      bestScore = score;
      *out = &blas;
    }
  }
  if (*out == nullptr) {
    // Failed to find similar blas, so allocate a new one
    *out = allocateEntry(hash, drawCall);
    return CacheState::kNew;
  }
  return CacheState::kSimilar;

}

BlasEntry* DrawCallCache::allocateEntry(XXH64_hash_t hash, const DrawCallState& drawCall) {
  auto iter = m_entries.emplace(hash, drawCall);
  BlasEntry* result = &iter->second;
  result->frameCreated = m_device->getCurrentFrameId();
  return result;
}

}  // namespace nvvk
