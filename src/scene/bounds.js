// World-space bounds for renderables.
//
// Lives in the scene layer, not the renderer, because it reads and writes only
// scene columns. Culling is its biggest consumer but not its only one: picking
// needs the same bounds, and a raycast that silently used last frame's would be
// the "you forgot to call update()" failure this engine does not allow.

import { aabbTransform } from '../core/math/aabb.js';
import { handleIndex } from '../core/handle.js';

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

/**
 * World bounds for every skinned instance, from where its joints are now.
 *
 * The reason this exists at all: a skinned mesh's vertices move without its
 * model matrix moving. Transforming its bind-pose box by that matrix -- which
 * is what every other renderable does -- gives the box the character had when
 * it was authored, so raising an arm puts geometry outside its own bounds. It
 * then gets frustum-culled with the arm on screen, or occlusion-culled behind
 * something it now reaches past. Nothing errors; geometry just goes missing at
 * angles nobody tested.
 *
 * A skinned vertex is a weighted average of its per-joint positions, so it
 * lies inside their convex hull, so a union of spheres -- one per joint, at
 * that joint's current world position, sized by how far its influence reached
 * in bind pose -- contains every vertex the skin can produce. Conservative by
 * construction, and the cost is per JOINT rather than per vertex.
 *
 * Computed once per skin instance, not per renderable: every primitive a
 * character is made of shares its skeleton and therefore its box.
 */
export function updateSkinBounds(skins, world) {
  for (const skin of skins) {
    const { joints, jointRadii, boundsMin, boundsMax } = skin;
    let started = false;

    for (let j = 0; j < joints.length; j++) {
      const radius = jointRadii[j];
      // Zero means no vertex is influenced by this joint, so where it is says
      // nothing about where the mesh is.
      if (!(radius > 0)) continue;

      const t = handleIndex(joints[j]) * 16 + 12;
      const x = world[t], y = world[t + 1], z = world[t + 2];

      if (!started) {
        boundsMin[0] = x - radius; boundsMin[1] = y - radius; boundsMin[2] = z - radius;
        boundsMax[0] = x + radius; boundsMax[1] = y + radius; boundsMax[2] = z + radius;
        started = true;
        continue;
      }
      if (x - radius < boundsMin[0]) boundsMin[0] = x - radius;
      if (y - radius < boundsMin[1]) boundsMin[1] = y - radius;
      if (z - radius < boundsMin[2]) boundsMin[2] = z - radius;
      if (x + radius > boundsMax[0]) boundsMax[0] = x + radius;
      if (y + radius > boundsMax[1]) boundsMax[1] = y + radius;
      if (z + radius > boundsMax[2]) boundsMax[2] = z + radius;
    }
    skin.hasBounds = started;
  }
}

/**
 * Copy each skinned renderable's bounds from the skin that drives it.
 *
 * Runs after updateWorldBounds, overwriting what it wrote: that pass has no
 * way to know a renderable is skinned, and giving it one would put skinning
 * into a file whose whole point is that it reads only scene columns.
 */
export function applySkinBounds(count, renderableSkin, skins, worldMin, worldMax) {
  let applied = 0;
  for (let i = 0; i < count; i++) {
    const s = renderableSkin[i];
    if (s < 0) continue;
    const skin = skins[s];
    if (!skin || !skin.hasBounds) continue;

    const o = i * 3;
    worldMin[o] = skin.boundsMin[0];
    worldMin[o + 1] = skin.boundsMin[1];
    worldMin[o + 2] = skin.boundsMin[2];
    worldMax[o] = skin.boundsMax[0];
    worldMax[o + 1] = skin.boundsMax[1];
    worldMax[o + 2] = skin.boundsMax[2];
    applied++;
  }
  return applied;
}

/**
 * How far a morphed instance's vertices travel from the undeformed mesh.
 *
 * `extent[t]` is the farthest any vertex moves under target t at weight 1, so
 * the weighted sum is the farthest any vertex can move under all of them at
 * once. Conservative: it assumes every target pulls the SAME vertex the same
 * way, which no real target set does, and being loose is the only direction a
 * cull bound may err.
 *
 * The absolute value is not decoration. glTF does not bound weights to [0,1] --
 * a negative weight is the legitimate way to author "the opposite of this
 * expression", and an overshoot past 1 is how exaggeration is animated. A sum
 * without it would go NEGATIVE and shrink the box.
 */
export function morphPadding(weights, extent) {
  let pad = 0;
  for (let t = 0; t < extent.length; t++) pad += Math.abs(weights[t]) * extent[t];
  return pad;
}

/**
 * World bounds for every morphed renderable, at its current weights.
 *
 * The same problem skinning has and a different shape of answer: vertices move
 * without the model matrix moving, so a box built from the authored one is the
 * mesh's resting shape rather than its current one.
 *
 * Runs AFTER updateWorldBounds and after applySkinBounds. It is idempotent:
 * every case below rebuilds its box from inputs rather than growing what is
 * already there, which is what lets it run unconditionally each frame while
 * updateWorldBounds runs only for things that moved.
 *
 * `lastPad` remembers each renderable's padding so the return value can be
 * how many bounds actually CHANGED. Weights move vertices without moving a
 * transform, so the scene's "did anything move" answer would otherwise miss
 * them -- and the alternative, recomputing the scene union whenever a morph
 * exists, pays a full pass every frame for a face that is holding still.
 *
 * The two cases differ:
 *
 *   Unskinned. The padded LOCAL box is transformed, rather than the world box
 *   grown. Growing the world box would need the largest singular value of the
 *   model matrix to stay conservative, and a hierarchy of non-uniform scales
 *   composes to a matrix whose columns do not give it. Transforming a padded
 *   local box needs no such argument -- it is exactly what a static mesh with
 *   that box would produce.
 *
 *   Skinned. There is no local box to pad: applySkinBounds built a world box
 *   from where the joints are. The delta is applied in mesh space and then by
 *   a joint matrix, which is rigid, so it reaches at most `pad` in world space
 *   too. Rigid is the same assumption the joint radii and the skinned normals
 *   already make.
 */
export function applyMorphBounds(
  count, renderableMorph, renderableSkin, morphs, extents,
  localMin, localMax, worldMin, worldMax, matrices, matrixSlot, lastPad,
) {
  let changed = 0;
  for (let i = 0; i < count; i++) {
    const m = renderableMorph[i];
    if (m < 0) continue;
    const extent = extents[i];
    if (extent === null || extent === undefined) continue;

    const pad = morphPadding(morphs[m].weights, extent);
    if (pad !== lastPad[i]) {
      lastPad[i] = pad;
      changed++;
    }
    // Every weight at zero is the undeformed mesh, which is what the pass
    // before this one already computed. The resting state of every morphed
    // mesh in the scene, so it is worth not touching.
    if (pad === 0) continue;

    const o = i * 3;
    if (renderableSkin[i] >= 0) {
      worldMin[o] -= pad; worldMin[o + 1] -= pad; worldMin[o + 2] -= pad;
      worldMax[o] += pad; worldMax[o + 1] += pad; worldMax[o + 2] += pad;
    } else {
      PADDED_MIN[0] = localMin[o] - pad;
      PADDED_MIN[1] = localMin[o + 1] - pad;
      PADDED_MIN[2] = localMin[o + 2] - pad;
      PADDED_MAX[0] = localMax[o] + pad;
      PADDED_MAX[1] = localMax[o + 1] + pad;
      PADDED_MAX[2] = localMax[o + 2] + pad;
      aabbTransform(
        worldMin, worldMax, PADDED_MIN, PADDED_MAX, matrices, matrixSlot[i] * 16, o, 0,
      );
    }
  }
  return changed;
}

/** Scratch for the padded local box. One per module, never nested. */
const PADDED_MIN = new Float32Array(3);
const PADDED_MAX = new Float32Array(3);
