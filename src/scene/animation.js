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
 * Add one clip's pose at `time`, scaled by `weight`, into an accumulator.
 *
 * The blending half of sampleClip: the same channels and the same sampling,
 * summed rather than written. Rotations are summed with their signs aligned to
 * the first one added -- q and -q are one rotation, and adding opposite signs
 * would cancel it -- and normalized when the sum is applied, which is nlerp.
 */
function accumulateClip(clip, time, weight, acc, entityOf, entities, weightsOf) {
  for (const channel of clip.channels) {
    if (!entities.alive(entityOf[channel.node])) continue;
    const components = sampleChannel(channel, time);
    if (components === 0) continue;
    const n = channel.node;

    switch (channel.path) {
      case 'translation':
      case 'scale': {
        const sum = channel.path === 'translation' ? acc.position : acc.scale;
        for (let c = 0; c < 3; c++) sum[n * 3 + c] += SAMPLE[c] * weight;
        (channel.path === 'translation' ? acc.positionWeight : acc.scaleWeight)[n] += weight;
        break;
      }
      case 'rotation': {
        const o = n * 4;
        const r = acc.rotation;
        const sign = acc.rotationWeight[n] > 0
          && r[o] * SAMPLE[0] + r[o + 1] * SAMPLE[1] + r[o + 2] * SAMPLE[2] + r[o + 3] * SAMPLE[3] < 0 ? -1 : 1;
        for (let c = 0; c < 4; c++) r[o + c] += SAMPLE[c] * weight * sign;
        acc.rotationWeight[n] += weight;
        break;
      }
      case 'weights': {
        const target = weightsOf === null ? undefined : weightsOf[n];
        if (target === undefined) break;
        let sum = acc.morph[n];
        if (sum === undefined || sum.length !== target.length) sum = acc.morph[n] = new Float32Array(target.length);
        const count = Math.min(components, target.length);
        for (let c = 0; c < count; c++) sum[c] += SAMPLE[c] * weight;
        acc.morphWeight[n] += weight;
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
 * One clip plays at a time, or several blend while one fades into another.
 * play() with a `fade` ramps the new clip's weight up and every playing one's
 * down over that many seconds; without one, the new clip simply replaces the
 * rest. A single clip is sampled straight into the transforms, exactly as it
 * always was -- the blending path costs nothing until a fade is under way.
 *
 * A node only some of the blending clips animate takes the value of those that
 * do, normalized by their weight. It does not blend toward its rest pose: the
 * clip that leaves it alone has not said where it should be.
 *
 * `clip`, `time`, `speed`, `loop` and `finished` describe the clip last
 * played, which is the one a fade is heading to.
 */
export class AnimationPlayer {
  constructor(clips, entityOf, entities, weightsOf = null) {
    this.clips = clips;
    this.entityOf = entityOf;
    this.entities = entities;
    /** Node index -> that instance's morph weights. Null when it has none. */
    this.weightsOf = weightsOf;
    /** Playing clips, the newest last, each with its own time and weight. */
    this.tracks = [];
    this._acc = null;
  }

  get names() {
    return this.clips.map((clip) => clip.name);
  }

  get _current() { return this.tracks[this.tracks.length - 1] ?? null; }
  get clip() { return this._current?.clip ?? null; }
  get time() { return this._current?.time ?? 0; }
  set time(value) { if (this._current) this._current.time = value; }
  get speed() { return this._current?.speed ?? 1; }
  set speed(value) { if (this._current) this._current.speed = value; }
  get loop() { return this._current?.loop ?? true; }
  set loop(value) { if (this._current) this._current.loop = value; }
  /** True once a non-looping clip has reached its end. */
  get finished() { return this._current?.finished ?? false; }

  /**
   * Start a clip by name or index. Returns false if there is no such clip.
   *
   * `fade`, in seconds, cross-fades from whatever is playing. Sets state only:
   * the pose does not change until the next advance(), so a player left at
   * speed 0 holds whatever pose the asset loaded in.
   */
  play(nameOrIndex, { loop = true, speed = 1, time = 0, fade = 0 } = {}) {
    const clip = typeof nameOrIndex === 'number'
      ? this.clips[nameOrIndex]
      : this.clips.find((c) => c.name === nameOrIndex);
    if (!clip) return false;

    const track = { clip, time, loop, speed, finished: false, weight: 1, target: 1, rate: 0 };
    if (fade > 0 && this.tracks.length > 0) {
      // Every playing clip heads to zero at the pace that gets it there in
      // `fade` from wherever it is now, and the new one rises to one.
      for (const old of this.tracks) {
        old.target = 0;
        old.rate = old.weight / fade;
      }
      track.weight = 0;
      track.rate = 1 / fade;
      this.tracks.push(track);
    } else {
      this.tracks = [track];
    }
    return true;
  }

  /** Stop, leaving the pose where it is. */
  stop() {
    this.tracks = [];
    return this;
  }

  advance(dt, transforms) {
    const tracks = this.tracks;
    if (tracks.length === 0) return false;
    if (tracks.length === 1 && tracks[0].finished) return false;

    // Time and weight for every track; then drop the ones faded to nothing.
    let kept = 0;
    for (const track of tracks) {
      if (!track.finished) advanceTrack(track, dt);
      if (track.weight !== track.target) {
        const step = track.rate * dt;
        track.weight = track.weight < track.target
          ? Math.min(track.target, track.weight + step)
          : Math.max(track.target, track.weight - step);
      }
      if (!(track.target === 0 && track.weight <= 0)) tracks[kept++] = track;
    }
    tracks.length = kept;
    if (kept === 0) return false;

    if (kept === 1) {
      const track = tracks[0];
      sampleClip(track.clip, track.time, transforms, this.entityOf, this.entities, this.weightsOf);
      return true;
    }

    const acc = this._accumulator();
    for (const track of tracks) {
      if (track.weight > 0) {
        accumulateClip(track.clip, track.time, track.weight, acc, this.entityOf, this.entities, this.weightsOf);
      }
    }
    this._apply(acc, transforms);
    return true;
  }

  /** Per-node sums, allocated once for this instance and cleared per use. */
  _accumulator() {
    const n = this.entityOf.length;
    let acc = this._acc;
    if (acc === null) {
      acc = this._acc = {
        position: new Float32Array(n * 3), positionWeight: new Float32Array(n),
        rotation: new Float32Array(n * 4), rotationWeight: new Float32Array(n),
        scale: new Float32Array(n * 3), scaleWeight: new Float32Array(n),
        morph: new Array(n), morphWeight: new Float32Array(n),
      };
    } else {
      acc.position.fill(0); acc.positionWeight.fill(0);
      acc.rotation.fill(0); acc.rotationWeight.fill(0);
      acc.scale.fill(0); acc.scaleWeight.fill(0);
      acc.morphWeight.fill(0);
      for (const sum of acc.morph) sum?.fill(0);
    }
    return acc;
  }

  _apply(acc, transforms) {
    for (let n = 0; n < this.entityOf.length; n++) {
      const entity = this.entityOf[n];
      if (!this.entities.alive(entity)) continue;
      let w = acc.positionWeight[n];
      if (w > 0) transforms.setPosition(entity, acc.position[n * 3] / w, acc.position[n * 3 + 1] / w, acc.position[n * 3 + 2] / w);
      w = acc.scaleWeight[n];
      if (w > 0) transforms.setScale(entity, acc.scale[n * 3] / w, acc.scale[n * 3 + 1] / w, acc.scale[n * 3 + 2] / w);
      if (acc.rotationWeight[n] > 0) {
        const q = acc.rotation.subarray(n * 4, n * 4 + 4);
        transforms.setRotation(entity, quatNormalize(QUAT_A, q));
      }
      w = acc.morphWeight[n];
      if (w > 0) {
        const target = this.weightsOf[n];
        const sum = acc.morph[n];
        for (let c = 0; c < target.length; c++) target[c] = sum[c] / w;
      }
    }
  }
}

/** One track's clock: loop, clamp at an end, or finish. */
function advanceTrack(track, dt) {
  track.time += dt * track.speed;
  const duration = track.clip.duration;
  if (duration > 0) {
    if (track.loop) {
      // Modulo rather than subtraction, so a large dt or a high speed cannot
      // leave the time outside the clip.
      track.time %= duration;
      if (track.time < 0) track.time += duration;
    } else if (track.time >= duration) {
      track.time = duration;
      track.finished = true;
    } else if (track.time < 0) {
      // Backwards, the end is the start.
      track.time = 0;
      track.finished = true;
    }
  }
}
