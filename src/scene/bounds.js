// World-space bounds for renderables.
//
// Lives in the scene layer, not the renderer, because it reads and writes only
// scene columns. Culling is its biggest consumer but not its only one: picking
// needs the same bounds, and a raycast that silently used last frame's would be
// the "you forgot to call update()" failure this engine does not allow.

import { aabbTransform } from '../core/math/aabb.js';

/**
 * Recompute world-space bounds for every renderable whose transform moved.
 *
 * World bounds are always derived from LOCAL bounds, never from the previous
 * world bounds: an AABB grows every time it is rotated, so re-transforming an
 * already-transformed box inflates it a little more each frame until it never
 * culls anything.
 *
 * Bounds columns are 3 floats per renderable; `dirty` is indexed by matrix
 * slot, and null updates everything.
 */
export function updateWorldBounds(
  count, localMin, localMax, worldMin, worldMax, matrices, matrixSlot, dirty = null,
) {
  let updated = 0;
  for (let i = 0; i < count; i++) {
    const slot = matrixSlot[i];
    if (dirty !== null && dirty[slot] === 0) continue;

    const o = i * 3;
    // Offsets rather than subarray views: four views per renderable per frame
    // would be four allocations per renderable per frame.
    aabbTransform(worldMin, worldMax, localMin, localMax, matrices, slot * 16, o, o);
    updated++;
  }
  return updated;
}

/**
 * Union of every renderable's world bounds, into `outMin`/`outMax`.
 *
 * A full pass, not an incremental one: a renderable that MOVES can shrink the
 * union as easily as grow it, so there is nothing to update in place. The
 * caller decides how often to pay for it -- the renderer only does when
 * something actually moved, which on a settled scene is never.
 *
 * Returns false and leaves the outputs alone for an empty scene, because there
 * is no box that means "nothing" which a caller would not have to special-case
 * anyway.
 */
export function unionWorldBounds(count, worldMin, worldMax, outMin, outMax) {
  if (count === 0) return false;

  outMin[0] = worldMin[0]; outMin[1] = worldMin[1]; outMin[2] = worldMin[2];
  outMax[0] = worldMax[0]; outMax[1] = worldMax[1]; outMax[2] = worldMax[2];

  for (let i = 1; i < count; i++) {
    const o = i * 3;
    if (worldMin[o] < outMin[0]) outMin[0] = worldMin[o];
    if (worldMin[o + 1] < outMin[1]) outMin[1] = worldMin[o + 1];
    if (worldMin[o + 2] < outMin[2]) outMin[2] = worldMin[o + 2];
    if (worldMax[o] > outMax[0]) outMax[0] = worldMax[o];
    if (worldMax[o + 1] > outMax[1]) outMax[1] = worldMax[o + 1];
    if (worldMax[o + 2] > outMax[2]) outMax[2] = worldMax[o + 2];
  }
  return true;
}

/**
 * The farthest view-space depth any corner of a world box reaches.
 *
 * Row 2 of the view matrix takes a world point to its view z, which is negative
 * in front of the camera; depth is its negation. All eight corners, because a
 * box behind the camera on one axis can still have a far corner in front.
 */
export function farthestViewDepth(view, min, max) {
  let farthest = -Infinity;
  for (let c = 0; c < 8; c++) {
    const x = (c & 1) ? max[0] : min[0];
    const y = (c & 2) ? max[1] : min[1];
    const z = (c & 4) ? max[2] : min[2];
    const depth = -(view[2] * x + view[6] * y + view[10] * z + view[14]);
    if (depth > farthest) farthest = depth;
  }
  return farthest;
}
