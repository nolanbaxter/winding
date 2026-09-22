// Draw lists and sort keys.
//
// Culling produces a flat list of (key, payload) pairs -- integers, not
// objects. The payload is opaque here: the renderer decides what it indexes.
// Keeping this file ignorant of meshes and pipelines is what lets the sort be
// a radix sort over raw integers instead of a comparator over objects.
//
// Sorting is by a packed key whose field ORDER encodes the priorities, so the
// sort itself needs no policy:
//
//   OPAQUE        [ pipeline:10 | material:12 | depth:10 ]
//   TRANSPARENT   [ depth:16    | pipeline:8  | material:8 ]
//
// Opaque puts state first because state changes are the expensive thing, and
// sorts near-to-far within a state bucket so the depth test rejects shaded
// pixels early. Transparent CANNOT reorder by state -- blending is
// order-dependent -- so depth takes the high bits and state efficiency gets
// whatever is left. That difference is forced by the hardware, not chosen.

import { DEBUG, assert } from '../core/assert.js';
import { grownCapacity, growArray } from '../core/grow.js';

export const OPAQUE_PIPELINE_BITS = 10;    // 1024 distinct pipelines
export const OPAQUE_MATERIAL_BITS = 12;    // 4096 distinct materials
export const OPAQUE_DEPTH_BITS = 10;       // 1024 depth buckets

// The transparent key carries the same three fields in a different order, and
// its material field must be the same WIDTH as the opaque one. It was 8 bits.
//
// That made 256 a second, tighter material ceiling that no mint site enforced:
// MaterialRegistry.register guards unconditionally against 4096 -- derived from
// OPAQUE_MATERIAL_BITS -- and its comment claims that is the trust boundary
// because "drawlist.js only DEBUG-checks". True of the opaque key and false of
// this one. A blended material with id 300 ORed a bit into the pipeline field
// and silently reordered the whole frame's blending, in release, with no
// visual signature beyond the artifact itself.
//
// The width came back from the pipeline field, which never needed it. Pipeline
// ids are a dense index over variantKey(alphaMode, doubleSided), and that has
// at most SIX distinct values, so 8 bits were holding 3 bits of information
// right next to a field that was 4 bits short.
export const TRANSPARENT_DEPTH_BITS = 16;  // fine ordering: blending needs it
export const TRANSPARENT_PIPELINE_BITS = 4;
export const TRANSPARENT_MATERIAL_BITS = OPAQUE_MATERIAL_BITS;

assert(OPAQUE_PIPELINE_BITS + OPAQUE_MATERIAL_BITS + OPAQUE_DEPTH_BITS === 32,
  "opaque sort key does not fill exactly 32 bits");
assert(TRANSPARENT_DEPTH_BITS + TRANSPARENT_PIPELINE_BITS + TRANSPARENT_MATERIAL_BITS === 32,
  "transparent sort key does not fill exactly 32 bits");
// Both keys address materials, so one registry ceiling has to cover both.
assert(TRANSPARENT_MATERIAL_BITS === OPAQUE_MATERIAL_BITS,
  "the two sort keys disagree on how many materials exist");

const OPAQUE_DEPTH_MAX = (1 << OPAQUE_DEPTH_BITS) - 1;
const TRANSPARENT_DEPTH_MAX = (1 << TRANSPARENT_DEPTH_BITS) - 1;

/**
 * Pack an opaque key. Ascending sort order gives: grouped by pipeline, then by
 * material, then near-to-far.
 *
 * ponytail: 32-bit keys, so the field widths above are hard ceilings. They are
 * DEBUG-checked here rather than unconditionally, because the real trust
 * boundary is pipeline/material REGISTRATION -- that is where an unconditional
 * check belongs and where an id that cannot fit should be refused. Widening
 * means a second key word and a radix sort that walks both.
 */
export function opaqueSortKey(pipelineId, materialId, depthBucket) {
  if (DEBUG) {
    assert(pipelineId >>> 0 === pipelineId && pipelineId < (1 << OPAQUE_PIPELINE_BITS),
      `opaqueSortKey: pipeline id ${pipelineId} does not fit in ${OPAQUE_PIPELINE_BITS} bits`);
    assert(materialId >>> 0 === materialId && materialId < (1 << OPAQUE_MATERIAL_BITS),
      `opaqueSortKey: material id ${materialId} does not fit in ${OPAQUE_MATERIAL_BITS} bits`);
    assert(depthBucket >= 0 && depthBucket <= OPAQUE_DEPTH_MAX, 'depth bucket out of range');
  }
  // >>> 0 because the top bit of a 32-bit shift makes a signed negative, and a
  // negative key would sort before everything.
  return ((pipelineId << (OPAQUE_MATERIAL_BITS + OPAQUE_DEPTH_BITS))
    | (materialId << OPAQUE_DEPTH_BITS)
    | depthBucket) >>> 0;
}

/** Ascending sort order gives far-to-near, which is the order blending needs. */
export function transparentSortKey(pipelineId, materialId, depthBucket) {
  if (DEBUG) {
    assert(pipelineId < (1 << TRANSPARENT_PIPELINE_BITS),
      `transparentSortKey: pipeline id ${pipelineId} does not fit in ${TRANSPARENT_PIPELINE_BITS} bits`);
    assert(materialId < (1 << TRANSPARENT_MATERIAL_BITS),
      `transparentSortKey: material id ${materialId} does not fit in ${TRANSPARENT_MATERIAL_BITS} bits`);
    assert(depthBucket >= 0 && depthBucket <= TRANSPARENT_DEPTH_MAX, 'depth bucket out of range');
  }
  return ((depthBucket << (TRANSPARENT_PIPELINE_BITS + TRANSPARENT_MATERIAL_BITS))
    | (pipelineId << TRANSPARENT_MATERIAL_BITS)
    | materialId) >>> 0;
}

/**
 * Quantize view distance into a sort bucket.
 *
 * Reuses the projection's own depth mapping, near/distance, rather than
 * inventing a range to normalize against. That matters for two reasons: there
 * IS no far plane to normalize against, and the reciprocal
 * curve already puts most of its resolution close to the camera, which is
 * exactly where draw order matters most. No arbitrary constant.
 *
 * `near` is the camera's near distance; `distance` is view-space depth.
 */
export function opaqueDepthBucket(near, distance) {
  // near -> 0 (drawn first), infinity -> max.
  return Math.round((1 - reverseZDepth(near, distance)) * OPAQUE_DEPTH_MAX);
}

export function transparentDepthBucket(near, distance) {
  // far -> 0 (drawn first), near -> max.
  return Math.round(reverseZDepth(near, distance) * TRANSPARENT_DEPTH_MAX);
}

function reverseZDepth(near, distance) {
  if (!(distance > near)) return 1;          // at or inside the near plane, and NaN-safe
  return near / distance;
}

/**
 * A flat list of sortable draw references.
 *
 * `payload` is whatever 32-bit handle the renderer wants back after sorting --
 * an index into its own SoA columns, typically. This class never dereferences
 * it, which is the whole reason the sort can be a radix sort.
 */
export class DrawList {
  constructor(capacity) {
    this.capacity = capacity;
    this.keys = new Uint32Array(capacity);
    this.payloads = new Uint32Array(capacity);
    this.count = 0;

    // Ping-pong targets for the radix passes, allocated once.
    this._keysAlt = new Uint32Array(capacity);
    this._payloadsAlt = new Uint32Array(capacity);
    this._counts = new Uint32Array(256);
  }

  clear() {
    this.count = 0;
  }

  _grow(needed) {
    const capacity = grownCapacity(this.capacity, needed);
    this.keys = growArray(this.keys, capacity);
    this.payloads = growArray(this.payloads, capacity);
    // The ping-pong targets are written before they are read on every pass, so
    // they only need to be the right SIZE, not to carry anything across.
    this._keysAlt = new Uint32Array(capacity);
    this._payloadsAlt = new Uint32Array(capacity);
    this.capacity = capacity;
  }

  push(key, payload) {
    // Grows rather than throwing: a dropped draw is an invisible object, and
    // the capacity that would have been exceeded was never anything but a guess.
    if (this.count >= this.capacity) this._grow(this.count + 1);
    const i = this.count++;
    this.keys[i] = key;
    this.payloads[i] = payload;
    return i;
  }

  /**
   * Least-significant-digit radix sort: four counting-sort passes over one
   * byte each. O(n) rather than O(n log n), no comparator call per element, and
   * no allocation -- which is why the key is an integer in the first place.
   *
   * Passes whose byte is constant across the whole list are skipped. That is
   * the common case for the high byte, and it means a small scene usually
   * sorts in two passes instead of four.
   */
  sort() {
    const n = this.count;
    if (n < 2) return;

    let keys = this.keys;
    let payloads = this.payloads;
    let keysAlt = this._keysAlt;
    let payloadsAlt = this._payloadsAlt;
    const counts = this._counts;

    for (let pass = 0; pass < 4; pass++) {
      const shift = pass * 8;
      counts.fill(0);
      for (let i = 0; i < n; i++) counts[(keys[i] >>> shift) & 0xff]++;

      // Every element shares this byte, so the pass would be an exact copy.
      if (counts[(keys[0] >>> shift) & 0xff] === n) continue;

      let total = 0;
      for (let b = 0; b < 256; b++) {
        const c = counts[b];
        counts[b] = total;
        total += c;
      }

      for (let i = 0; i < n; i++) {
        const destination = counts[(keys[i] >>> shift) & 0xff]++;
        keysAlt[destination] = keys[i];
        payloadsAlt[destination] = payloads[i];
      }

      // Swap the roles rather than copying back.
      let t = keys; keys = keysAlt; keysAlt = t;
      t = payloads; payloads = payloadsAlt; payloadsAlt = t;
    }

    // Skipped passes can leave the sorted data in the scratch buffers. Point
    // the public arrays at whichever pair actually holds it; they are the same
    // two buffers either way, so nothing is copied.
    this.keys = keys;
    this.payloads = payloads;
    this._keysAlt = keysAlt;
    this._payloadsAlt = payloadsAlt;
  }
}
