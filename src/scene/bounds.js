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
