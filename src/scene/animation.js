// Animation playback.
//
// Deliberately knows nothing about glTF: it takes clips of flat typed arrays
// (see gltf/animation.js) and writes TRS into a TransformStore. Everything
// downstream -- dirty propagation, composition, the GPU upload -- then happens
// exactly as it does for a node someone moved by hand, because it IS that.
//
// Sampling writes through the TransformStore setters rather than into its
// columns directly. Those setters are what mark a node dirty, and an animation
// that moved a node without marking it would compose to a stale matrix.

import { quatSlerp, quatNormalize } from '../core/math/quat.js';

const STEP = 'STEP';
const CUBICSPLINE = 'CUBICSPLINE';

/**
 * Scratch for one keyframe's worth of output.
 *
 * Four floats covers TRS and a quaternion. A `weights` channel is as wide as
 * its mesh has morph targets, which has no upper bound worth picking, so this
 * grows to fit and then stops growing -- one allocation per widest clip ever
 * played, not one per sample.
 */
let SAMPLE = new Float32Array(4);
const QUAT_A = new Float32Array(4);
const QUAT_B = new Float32Array(4);

/**
 * Index of the last keyframe at or before `time`.
 *
 * Binary search rather than a remembered cursor: a cursor is faster for
 * monotonic playback and wrong the moment anything seeks, loops backwards, or
 * plays at a negative speed. Keyframe counts are in the hundreds, so log2 of
 * that is under ten comparisons and not worth the correctness risk.
 */
function keyframeBefore(times, time) {
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= time) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/**
 * Sample one channel at `time` into SAMPLE, returning how many components.
 *
 * Times outside the channel's range clamp to its first or last value. glTF
 * requires this: a clip's duration is the longest of its channels, so a short
 * channel is simply held at its final value for the rest of the clip.
 */
function sampleChannel(channel, time) {
  const { times, values, components, interpolation } = channel;
  const count = times.length;

  if (count === 0) return 0;
  if (SAMPLE.length < components) SAMPLE = new Float32Array(components);

  // Whether the four floats are a ROTATION, asked of the path rather than of
  // the width. They were the same question until morph targets: a mesh with
  // four targets has a four-component weights channel, and treating that as a
  // quaternion would slerp four independent sliders through a sphere.
  const rotation = channel.path === 'rotation';

  const cubic = interpolation === CUBICSPLINE;
  // With CUBICSPLINE the value sits between its two tangents.
  const valueAt = (k) => (cubic ? (k * 3 + 1) : k) * components;

  if (count === 1 || time <= times[0]) {
    for (let c = 0; c < components; c++) SAMPLE[c] = values[valueAt(0) + c];
    return components;
  }
  if (time >= times[count - 1]) {
    for (let c = 0; c < components; c++) SAMPLE[c] = values[valueAt(count - 1) + c];
    return components;
  }

  const k = keyframeBefore(times, time);
  const span = times[k + 1] - times[k];
  // Coincident keyframes are legal and would divide by zero.
  const t = span > 0 ? (time - times[k]) / span : 0;

  if (interpolation === STEP) {
    for (let c = 0; c < components; c++) SAMPLE[c] = values[valueAt(k) + c];
    return components;
  }

  if (cubic) {
    // Hermite. The tangents are stored per key and scaled by the span, which is
    // what makes the curve independent of how the timeline is sampled.
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;

    const p0 = (k * 3 + 1) * components;
    const m0 = (k * 3 + 2) * components;          // out-tangent of k
    const p1 = ((k + 1) * 3 + 1) * components;
    const m1 = ((k + 1) * 3) * components;        // in-tangent of k+1

    for (let c = 0; c < components; c++) {
      SAMPLE[c] = h00 * values[p0 + c] + h10 * span * values[m0 + c]
        + h01 * values[p1 + c] + h11 * span * values[m1 + c];
    }
    if (rotation) quatNormalize(SAMPLE, SAMPLE);
    return components;
  }

  // LINEAR. Rotations slerp: a component-wise lerp of two quaternions is not a
  // rotation, and normalizing it afterwards still sweeps at the wrong rate.
  const a = valueAt(k);
  const b = valueAt(k + 1);
  if (rotation) {
    for (let c = 0; c < 4; c++) { QUAT_A[c] = values[a + c]; QUAT_B[c] = values[b + c]; }
    quatSlerp(SAMPLE, QUAT_A, QUAT_B, t);
  } else {
    for (let c = 0; c < components; c++) {
      SAMPLE[c] = values[a + c] + (values[b + c] - values[a + c]) * t;
    }
  }
  return components;
}

/**
 * Apply every channel of a clip at `time`.
 *
 * `entityOf` maps a clip's node indices onto the entities of one instance --
 * which is what lets two copies of the same asset play the same clip at
 * different times without sharing a frame of state. `entities` is the scene's
 * allocator, and it is the only thing that can answer whether a handle is
 * still the one it was.
 *
 * `weightsOf` is the same shape for morph weights: node index -> the weight
 * array of the instance driven by that node, or undefined. Indexed rather than
 * looked up, because it is answering the same question `entityOf` is and a map
 * per channel per frame would be the only allocation in this loop.
 */
export function sampleClip(clip, time, transforms, entityOf, entities, weightsOf = null) {
  for (const channel of clip.channels) {
    // One question, asked of the one object that knows the answer. A channel
    // can point at nothing in three ways -- a node index this instance has no
    // slot for (undefined), a node outside the asset's default scene (the map
    // is pre-filled with NULL_HANDLE, which is 0), or a child that has since
    // been removed. alive() rejects all three, because it compares the
    // GENERATION and not just the slot.
    //
    // Testing the slot instead, as this did, misses the case that actually
    // happens: handles are recycled last-in-first-out, so the very next alloc()
    // reuses the freed slot and marks it live again. The stale handle then
    // passed, and the clip drove whatever object had taken its place.
    if (!entities.alive(entityOf[channel.node])) continue;
    const entity = entityOf[channel.node];

    const components = sampleChannel(channel, time);
    if (components === 0) continue;

    switch (channel.path) {
      case 'translation':
        transforms.setPosition(entity, SAMPLE[0], SAMPLE[1], SAMPLE[2]);
        break;
      case 'scale':
        transforms.setScale(entity, SAMPLE[0], SAMPLE[1], SAMPLE[2]);
        break;
      case 'rotation':
        transforms.setRotation(entity, SAMPLE);
        break;
      case 'weights': {
        // Straight into the instance's array. No dirty flag to set: the
        // weights ARE the state the renderer uploads, where a transform is an
        // input to a hierarchy that has to recompose.
        const weights = weightsOf === null ? undefined : weightsOf[channel.node];
        if (weights === undefined) break;
        // A clip authored against a different mesh than the one instanced here
        // would write past the end. The shorter of the two is the part both
        // agree on.
        const n = Math.min(components, weights.length);
        for (let c = 0; c < n; c++) weights[c] = SAMPLE[c];
        break;
      }
      default:
        break;
    }
  }
}

/**
 * Playback state for one instance of an asset.
 *
 * One clip at a time. Blending between two clips needs a weight per channel and
 * somewhere to accumulate partial results, which is a different data structure
 * than this -- doing it badly here would be worse than not doing it.
 */
export class AnimationPlayer {
  constructor(clips, entityOf, entities, weightsOf = null) {
    this.clips = clips;
    this.entityOf = entityOf;
    this.entities = entities;
    /** Node index -> that instance's morph weights. Null when it has none. */
    this.weightsOf = weightsOf;
    this.clip = null;
    this.time = 0;
    this.speed = 1;
    this.loop = true;
    /** True once a non-looping clip has reached its end. */
    this.finished = false;
  }

  get names() {
    return this.clips.map((clip) => clip.name);
  }

  /**
   * Start a clip by name or index. Returns false if there is no such clip.
   *
   * Sets state only. The pose does not change until the next advance(), so a
   * player left at speed 0 holds whatever pose the asset loaded in.
   */
  play(nameOrIndex, { loop = true, speed = 1, time = 0 } = {}) {
    const clip = typeof nameOrIndex === 'number'
      ? this.clips[nameOrIndex]
      : this.clips.find((c) => c.name === nameOrIndex);
    if (!clip) return false;

    this.clip = clip;
    this.time = time;
    this.loop = loop;
    this.speed = speed;
    this.finished = false;
    return true;
  }

  /** Stop, leaving the pose where it is. */
  stop() {
    this.clip = null;
    return this;
  }

  advance(dt, transforms) {
    const clip = this.clip;
    if (!clip || this.finished) return false;

    this.time += dt * this.speed;

    const duration = clip.duration;
    if (duration > 0) {
      if (this.loop) {
        // Modulo rather than subtraction, so a large dt or a high speed cannot
        // leave the time outside the clip.
        this.time %= duration;
        if (this.time < 0) this.time += duration;
      } else if (this.time >= duration) {
        this.time = duration;
        this.finished = true;
      } else if (this.time < 0) {
        this.time = 0;
      }
    }

    sampleClip(clip, this.time, transforms, this.entityOf, this.entities, this.weightsOf);
    return true;
  }
}
