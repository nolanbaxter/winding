// The transform composition loop, as a standalone function over shared columns.
//
// Extracted from TransformStore.update() so the serial path and the parallel
// path run the SAME code. A job system whose parallel version is a second
// implementation of the algorithm is a job system that will eventually disagree
// with itself, and the disagreement will look like a physics bug.
//
// WHY THIS PARALLELISES AT ALL is the payoff for storing nodes in depth order.
// Every node at depth d depends only on
// nodes at depths below d. One depth level is therefore embarrassingly
// parallel: no two threads in a level touch the same entity, and every parent
// they read was finished before the level began.
//
// The barrier between levels is the job system's own completion counter. A
// worker's plain writes to `world` and `recomputed` happen before its
// Atomics.add on that counter, and the dispatcher's Atomics.load pairs with it
// -- so by the time the next level starts, everything the previous one wrote is
// visible. Release/acquire, without anything here having to say so.

import { mat4Copy, mat4FromQuatPosScale, mat4MultiplyAffine } from '../core/math/mat4.js';

export const NO_PARENT = -1;

/**
 * Compose world matrices for order[base + start] .. order[base + end).
 *
 * @param columns the TransformStore's arrays, shared or not
 * @param base    where this depth level starts in the order array
 */
export function composeRange(columns, base, start, end) {
  const { order, parent, dirty, recomputed, moved } = columns;
  const { local, world, position, rotation, scale } = columns;

  for (let k = start; k < end; k++) {
    const e = order[base + k];
    const p = parent[e];

    // Safe because the level containing p finished before this one began.
    const parentMoved = p !== NO_PARENT && recomputed[p] === 1;

    if (dirty[e] === 0 && !parentMoved) {
      recomputed[e] = 0;
      continue;
    }

    if (dirty[e] === 1) {
      mat4FromQuatPosScale(local, rotation, position, scale, e * 16, e * 4, e * 3, e * 3);
      dirty[e] = 0;
    }

    if (p === NO_PARENT) {
      mat4Copy(world, local, e * 16, e * 16);
    } else {
      // Affine: both are built from translation, rotation and scale.
      mat4MultiplyAffine(world, world, local, e * 16, p * 16, e * 16);
    }

    recomputed[e] = 1;
    moved[e] = 1;
  }
}

/** Rebuild the column views a worker needs from the raw shared buffers. */
export function columnsFromBuffers(buffers) {
  return {
    position: new Float32Array(buffers.position),
    rotation: new Float32Array(buffers.rotation),
    scale: new Float32Array(buffers.scale),
    local: new Float32Array(buffers.local),
    world: new Float32Array(buffers.world),
    parent: new Int32Array(buffers.parent),
    dirty: new Uint8Array(buffers.dirty),
    recomputed: new Uint8Array(buffers.recomputed),
    moved: new Uint8Array(buffers.moved),
    order: new Uint32Array(buffers.order),
  };
}

/** The raw buffers to hand a worker. Nothing is copied; these are shared. */
export function buffersFromColumns(columns) {
  return {
    position: columns.position.buffer,
    rotation: columns.rotation.buffer,
    scale: columns.scale.buffer,
    local: columns.local.buffer,
    world: columns.world.buffer,
    parent: columns.parent.buffer,
    dirty: columns.dirty.buffer,
    recomputed: columns.recomputed.buffer,
    moved: columns.moved.buffer,
    order: columns.order.buffer,
  };
}
