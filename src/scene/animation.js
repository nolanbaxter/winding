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

import { quatSlerp, quatNormalize, quatMultiply, quatConjugate } from '../core/math/quat.js';
import { handleIndex } from '../core/handle.js';

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
 *
 * `properties` maps a property channel's key -- a light's intensity, a
 * material's colour -- onto what this instance has to write it to. A key it
 * does not have is a light or camera outside the instance's default scene.
 */
export function sampleClip(clip, time, transforms, entityOf, entities, weightsOf = null, properties = null) {
  for (const channel of clip.channels) {
    if (channel.path === 'property') {
      const property = properties?.get(channel.key);
      if (property !== undefined && sampleChannel(channel, time) > 0) property.write(SAMPLE);
      continue;
    }
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
        const offset = channel.offset ?? 0;
        const n = Math.min(components, weights.length - offset);
        for (let c = 0; c < n; c++) weights[offset + c] = SAMPLE[c];
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
 *
 * `additive` sums each channel's change from its own first keyframe instead of
 * its value.
 */
function accumulateClip(clip, time, weight, acc, entityOf, entities, weightsOf, properties, additive = false) {
  for (const channel of clip.channels) {
    if (channel.path === 'property') {
      const property = properties?.get(channel.key);
      if (property === undefined || sampleChannel(channel, time) === 0) continue;
      if (additive) changeFromFirstKey(channel, property.components);
      for (let c = 0; c < property.components; c++) acc.property[property.slot * 4 + c] += SAMPLE[c] * weight;
      acc.propertyWeight[property.slot] += weight;
      continue;
    }
    if (!entities.alive(entityOf[channel.node])) continue;
    const components = sampleChannel(channel, time);
    if (components === 0) continue;
    if (additive) changeFromFirstKey(channel, components);
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
        if (sum === undefined || sum.length !== target.length) {
          sum = acc.morph[n] = new Float32Array(target.length);
          acc.morphWeight[n] = new Float32Array(target.length);
        }
        // Weighted per weight, not per node: a clip that drives one weight of
        // the list by pointer says nothing about the others.
        const offset = channel.offset ?? 0;
        const count = Math.min(components, target.length - offset);
        for (let c = 0; c < count; c++) {
          sum[offset + c] += SAMPLE[c] * weight;
          acc.morphWeight[n][offset + c] += weight;
        }
        break;
      }
      default:
        break;
    }
  }
}

/**
 * Turn SAMPLE into its change from the channel's first keyframe: a difference
 * for positions and weights, a ratio for scale, first^-1 * sample for rotation.
 * Each is what the matching composition in composeLayer undoes, so a layer at
 * full weight over a pose equal to the first key reproduces the clip exactly.
 */
function changeFromFirstKey(channel, components) {
  const { values, path } = channel;
  const first = channel.interpolation === CUBICSPLINE ? components : 0;
  if (path === 'rotation') {
    for (let c = 0; c < 4; c++) QUAT_A[c] = values[first + c];
    quatConjugate(QUAT_A, quatNormalize(QUAT_A, QUAT_A));
    quatMultiply(SAMPLE, QUAT_A, SAMPLE);
  } else if (path === 'scale') {
    // A zero scale has no ratio to anything; that axis is left alone.
    for (let c = 0; c < 3; c++) SAMPLE[c] = values[first + c] !== 0 ? SAMPLE[c] / values[first + c] : 1;
  } else {
    for (let c = 0; c < components; c++) SAMPLE[c] -= values[first + c];
  }
}

/**
 * Playback state for one instance of an asset.
 *
 * Clips play on LAYERS, applied in the order they were made, over the pose the
 * asset loaded in. The first is `base` and always exists; `layer()` adds more.
 *
 * Within a layer, clips blend by their weights. play() with a `fade` ramps the
 * new clip up and the others down over that many seconds; without one, the new
 * clip replaces them; with `add`, it joins them and nothing else changes, and
 * setWeight() then moves any one of them -- a walk and a run weighted by how
 * fast the character is going. Weights are normalized per node, so a node only
 * some of a layer's clips animate takes the value of those that do: the clip
 * that leaves it alone has not said where it should be.
 *
 * A layer above the base covers the pose beneath it as far as its own clips
 * are weighted in -- fading a clip in on an empty layer fades the layer in --
 * and only on the nodes its mask names. An ADDITIVE layer adds each clip's
 * change from its own first keyframe instead: a lean, a breath, a flinch, on
 * top of whatever the layers beneath are doing.
 *
 * Clips played with `sync` share one clock per layer, measured in cycles
 * rather than seconds, which runs at the weighted average of their lengths. A
 * 1.2 s walk and a 0.8 s run then put their feet down together at every blend,
 * which is what stops the feet sliding. A synced clip loops.
 *
 * One clip on the base layer and nothing else, the common case, is sampled
 * straight into the transforms as it always was: the layered path costs
 * nothing until something uses it.
 *
 * `clip`, `time`, `speed`, `loop` and `finished` describe the base layer's
 * newest clip, which is the one a fade is heading to.
 */
export class AnimationPlayer {
  /**
   * `nodes` is the asset's node table: names and children for masks, and each
   * node's loaded TRS, which is the pose the layers are applied over.
   */
  constructor(clips, entityOf, entities, weightsOf = null, nodes = null, properties = null) {
    this.clips = clips;
    this.entityOf = entityOf;
    this.entities = entities;
    /** Node index -> that instance's morph weights. Null when it has none. */
    this.weightsOf = weightsOf;
    this.nodes = nodes;
    /**
     * Property key -> { slot, components, rest, write(values) }: the lights,
     * cameras and materials this instance's clips drive by pointer. Null when
     * they drive none.
     */
    this.properties = properties;
    this.layers = [newLayer('base')];
    this._acc = null;
    this._pose = null;
    // The rest pose, taken now: the morph weights are about to be animated.
    this._rest = restPose(entityOf.length, nodes, weightsOf, properties);
    // Set by anything that changes the pose without advancing a clock, so a
    // player whose clips have all finished still samples once more.
    this._changed = false;
    /** The entity the scene placed this instance by: what root motion moves. */
    this.instance = null;
    /**
     * What root motion moved the instance by in the last advance, in the
     * instance's parent space: a distance and a turn about +Y, in radians.
     * Applied already unless rootMotion() was told `apply: false`.
     */
    this.motion = { position: new Float32Array(3), yaw: 0 };
    this._root = null;
    this._parents = null;
  }

  get names() {
    return this.clips.map((clip) => clip.name);
  }

  /** The base layer's clips, the newest last. */
  get tracks() { return this.layers[0].tracks; }

  get _current() { return this.tracks[this.tracks.length - 1] ?? null; }
  get clip() { return this._current?.clip ?? null; }
  get time() { return this._current?.time ?? 0; }
  set time(value) { if (this._current) { this._current.time = value; this._changed = true; } }
  get speed() { return this._current?.speed ?? 1; }
  set speed(value) { if (this._current) this._current.speed = value; }
  get loop() { return this._current?.loop ?? true; }
  set loop(value) { if (this._current) this._current.loop = value; }
  /** True once a non-looping clip has reached its end. */
  get finished() { return this._current?.finished ?? false; }

  /**
   * Make a layer, or change one. Layers apply in the order they were made.
   *
   * `mask` is a node name or a list of them; each names that node and
   * everything under it, and the layer touches nothing else. `null` clears
   * it. `weight` scales the whole layer. `additive` adds its clips' changes
   * instead of covering the pose beneath.
   */
  layer(name, { mask, weight, additive } = {}) {
    let layer = this.layers.find((l) => l.name === name);
    if (layer === undefined) this.layers.push(layer = newLayer(name));
    if (additive !== undefined) {
      if (additive && layer === this.layers[0]) {
        throw new Error('AnimationPlayer: the base layer cannot be additive; there is no pose under it to add to');
      }
      layer.additive = additive === true;
    }
    if (weight !== undefined) layer.weight = checkWeight(weight, 'layer weight');
    if (mask !== undefined) layer.mask = mask === null ? null : this._mask(mask);
    this._changed = true;
    return this;
  }

  /**
   * Start a clip by name or index. Returns false if there is no such clip.
   *
   * `fade`, in seconds, cross-fades from whatever that layer is playing, or
   * fades the layer in when it is playing nothing. `add` joins the clips
   * already playing instead of replacing them. `weight` is where the clip's
   * weight ends up. Sets state only: the pose does not change until the next
   * advance(), so a player left at speed 0 holds whatever pose the asset
   * loaded in.
   */
  play(nameOrIndex, {
    loop = true, speed = 1, time = 0, fade = 0, layer = 'base', weight = 1, add = false, sync = false,
  } = {}) {
    const clip = this._clip(nameOrIndex);
    if (!clip) return false;
    const target = this._layer(layer);
    weight = checkWeight(weight, 'weight');

    const track = {
      clip, time, loop, speed, finished: false, weight, target: weight, rate: 0, leaving: false, sync,
    };
    if (sync) {
      // Joining a group takes the group's place in the cycle; starting one
      // sets it.
      if (target.tracks.some((t) => t.sync)) track.time = target.phase * clip.duration;
      else target.phase = clip.duration > 0 ? time / clip.duration : 0;
    }
    // The base layer needs something to fade FROM. Any other layer fades from
    // the pose beneath it, so it always can.
    const ramp = fade > 0 && (target.tracks.length > 0 || target !== this.layers[0]);
    if (ramp) {
      track.weight = 0;
      track.rate = weight / fade;
    }
    if (add || ramp) {
      if (!add) for (const old of target.tracks) leave(old, fade);
      target.tracks.push(track);
    } else {
      target.tracks = [track];
    }
    this._changed = true;
    return true;
  }

  /**
   * Move a playing clip's weight, at once or over `fade` seconds. Returns
   * false if that clip is not playing on that layer. A weight of zero keeps
   * the clip playing, silent, so it can be weighted back in.
   */
  setWeight(nameOrIndex, weight, { layer = 'base', fade = 0 } = {}) {
    const clip = this._clip(nameOrIndex);
    weight = checkWeight(weight, 'weight');
    let found = false;
    for (const track of this._layer(layer).tracks) {
      if (track.clip !== clip) continue;
      found = true;
      track.leaving = false;
      track.target = weight;
      if (fade > 0) track.rate = Math.abs(weight - track.weight) / fade;
      else track.weight = weight;
    }
    this._changed = true;
    return found;
  }

  /**
   * Stop every layer, or one. Without a fade the pose stays where it is;
   * with one, the clips fade out, and a layer above the base hands the nodes
   * back to the layers beneath.
   */
  stop({ layer, fade = 0 } = {}) {
    for (const l of layer === undefined ? this.layers : [this._layer(layer)]) {
      if (fade > 0) for (const track of l.tracks) leave(track, fade);
      else l.tracks = [];
    }
    return this;
  }

  advance(dt, transforms) {
    const layers = this.layers;
    let any = false;
    let moving = this._changed;
    for (const layer of layers) {
      for (const track of layer.tracks) {
        any = true;
        if (!track.finished || track.weight !== track.target) moving = true;
      }
    }
    // Nothing playing, or everything run out and settled: the pose is what
    // the last advance left.
    if (!any || !moving) return false;
    this._changed = false;
    const root = this._root;
    this.motion.position.fill(0);
    this.motion.yaw = 0;

    let playing = 0;
    for (const layer of layers) playing += stepLayer(layer, dt);
    if (playing === 0) return false;
    if (root !== null) this._measureMotion(root, transforms);

    // Root motion rewrites the pose before it is written, which the direct
    // path has no pose to do to.
    const base = layers[0];
    if (root === null && playing === 1 && base.tracks.length === 1 && base.mask === null && base.weight === 1
      && base.tracks[0].weight > 0) {
      const track = base.tracks[0];
      sampleClip(track.clip, track.time, transforms, this.entityOf, this.entities, this.weightsOf, this.properties);
      return true;
    }

    const pose = this._restPose();
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      if (layer.tracks.length === 0) continue;
      const acc = this._accumulator();
      for (const track of layer.tracks) {
        if (track.weight > 0) {
          accumulateClip(track.clip, track.time, track.weight, acc,
            this.entityOf, this.entities, this.weightsOf, this.properties, layer.additive);
        }
      }
      composeLayer(pose, acc, layer, i === 0);
    }
    if (root !== null) this._keepRootInPlace(root, pose, transforms);
    this._write(pose, transforms);
    if (root !== null && root.apply) this._applyMotion(transforms);
    return true;
  }

  /**
   * Root motion: move the instance by what the base layer's clips move one
   * node by, and hold that node in place, so a clip that walks forward walks
   * the character forward instead of away from it and back at every loop.
   *
   * `node` names the node that carries the motion. Without it, it is the
   * highest node below the instance that any clip translates -- the hips, in
   * most rigs -- and two at the same height is an error that asks for the
   * name. What moves the instance is that node's travel across the ground
   * (`vertical: true` takes its height too) and its turn about the vertical;
   * the rest of its movement stays in the pose. Both are measured in the
   * instance's parent space, through whatever rotations and scales lie between.
   *
   * With `apply: false` the instance is left alone and `motion` says where it
   * would have gone, for a controller that decides where it can. `null`
   * turns root motion off.
   */
  rootMotion(options = {}) {
    if (options === null) {
      this._root = null;
      return this;
    }
    const { node, vertical = false, apply = true } = options;
    if (this.nodes === null || this.instance === null) {
      throw new Error('AnimationPlayer: root motion needs the scene\'s instance and the asset\'s node table');
    }
    const parents = this._parentsOf();
    const index = node === undefined ? this._movingNode(parents) : this._nodeNamed(node);
    if (this.entityOf[index] === this.instance) {
      throw new Error(`AnimationPlayer: "${this.nodes[index].name}" is the instance itself; the node that carries root motion has to be below what it moves`);
    }
    // The nodes between it and the instance, nearest first, and the instance.
    const chain = [];
    let top = index;
    for (let p = parents[index]; p !== -1; p = parents[p]) {
      chain.push(this.entityOf[p]);
      top = p;
    }
    if (this.entityOf[top] !== this.instance) chain.push(this.instance);
    this._root = {
      node: index, vertical: vertical === true, apply: apply !== false, chain,
      rotation: new Float32Array(4), channels: new Map(),
    };
    this._changed = true;
    return this;
  }

  _parentsOf() {
    if (this._parents === null) {
      this._parents = new Int32Array(this.nodes.length).fill(-1);
      this.nodes.forEach((node, n) => { for (const child of node.children ?? []) this._parents[child] = n; });
    }
    return this._parents;
  }

  _nodeNamed(name) {
    const found = [];
    this.nodes.forEach((node, n) => { if (node.name === name) found.push(n); });
    if (found.length !== 1) {
      throw new Error(`AnimationPlayer: ${found.length === 0 ? 'no' : found.length} nodes are named ${JSON.stringify(name)}`);
    }
    return found[0];
  }

  /** The highest node below the instance that some clip translates. */
  _movingNode(parents) {
    let best = [];
    let bestDepth = Infinity;
    for (const clip of this.clips) {
      for (const channel of clip.channels) {
        const n = channel.node;
        if (channel.path !== 'translation' || this.entityOf[n] === this.instance) continue;
        let depth = 0;
        for (let p = parents[n]; p !== -1; p = parents[p]) depth++;
        if (depth < bestDepth) { best = [n]; bestDepth = depth; } else if (depth === bestDepth && !best.includes(n)) best.push(n);
      }
    }
    if (best.length === 0) {
      throw new Error('AnimationPlayer: no clip moves a node below the instance; name the node that carries root motion with { node }');
    }
    if (best.length > 1) {
      throw new Error(`AnimationPlayer: clips move ${best.map((n) => JSON.stringify(this.nodes[n].name)).join(' and ')} equally high up; name the one that carries root motion with { node }`);
    }
    return best[0];
  }

  /**
   * This frame's motion, from each base-layer clip's own travel: its value at
   * the time it reached less its value at the time it left, plus one whole
   * cycle's travel for every loop in between -- which is what turns a wrap
   * into a step forward instead of a jump back. Clips blend by weight.
   */
  _measureMotion(root, transforms) {
    chainRotation(root.rotation, root.chain, transforms);
    MOVE.fill(0);
    let moveWeight = 0, yaw = 0, yawWeight = 0;
    for (const track of this.layers[0].tracks) {
      const w = track.weight;
      const duration = track.clip.duration;
      if (!(w > 0 && duration > 0)) continue;
      let channels = root.channels.get(track.clip);
      if (channels === undefined) {
        const of = (path) => track.clip.channels.find((c) => c.node === root.node && c.path === path) ?? null;
        root.channels.set(track.clip, channels = { translation: of('translation'), rotation: of('rotation') });
      }
      const cycles = Math.floor(track.to / duration) - Math.floor(track.from / duration);
      const from = track.from - Math.floor(track.from / duration) * duration;
      const to = track.to - Math.floor(track.to / duration) * duration;
      if (channels.translation !== null) {
        const c = channels.translation;
        sampleChannel(c, to); for (let i = 0; i < 3; i++) STEP3[i] = SAMPLE[i];
        sampleChannel(c, from); for (let i = 0; i < 3; i++) STEP3[i] -= SAMPLE[i];
        if (cycles !== 0) {
          sampleChannel(c, duration); for (let i = 0; i < 3; i++) STEP3[i] += cycles * SAMPLE[i];
          sampleChannel(c, 0); for (let i = 0; i < 3; i++) STEP3[i] -= cycles * SAMPLE[i];
        }
        for (let i = 0; i < 3; i++) MOVE[i] += w * STEP3[i];
        moveWeight += w;
      }
      if (channels.rotation !== null) {
        const c = channels.rotation;
        const yawAt = (t) => { sampleChannel(c, t); return yawOf(root.rotation, SAMPLE); };
        let turn = wrapAngle(yawAt(to) - yawAt(from));
        if (cycles !== 0) turn += cycles * wrapAngle(yawAt(duration) - yawAt(0));
        yaw += w * turn;
        yawWeight += w;
      }
    }
    const motion = this.motion;
    if (moveWeight > 0) {
      for (let i = 0; i < 3; i++) MOVE[i] /= moveWeight;
      upChain(MOVE, root.chain, transforms);
      if (!root.vertical) MOVE[1] = 0;
      motion.position.set(MOVE);
    }
    if (yawWeight > 0) motion.yaw = yaw / yawWeight;
  }

  /**
   * Take out of the pose what the instance now carries: the node's travel from
   * where it rests, across the ground, and its turn about the vertical.
   */
  _keepRootInPlace(root, pose, transforms) {
    const n = root.node;
    if (pose.touched[n] & POSITION) {
      for (let i = 0; i < 3; i++) MOVE[i] = pose.position[n * 3 + i] - this._rest.position[n * 3 + i];
      upChain(MOVE, root.chain, transforms);
      if (!root.vertical) MOVE[1] = 0;
      downChain(MOVE, root.chain, transforms);
      for (let i = 0; i < 3; i++) pose.position[n * 3 + i] -= MOVE[i];
    }
    if (pose.touched[n] & ROTATION) {
      const q = QUAT_B;
      for (let c = 0; c < 4; c++) q[c] = pose.rotation[n * 4 + c];
      for (let c = 0; c < 4; c++) QUAT_A[c] = this._rest.rotation[n * 4 + c];
      const turn = wrapAngle(yawOf(root.rotation, q) - yawOf(root.rotation, QUAT_A));
      // q' = C^-1 * yaw(-turn) * C * q: the same rotation, turned back about
      // the vertical of the space the turn was measured in.
      const C = root.rotation;
      quatMultiply(q, C, q);
      quatMultiply(q, yawQuat(QUAT_A, -turn), q);
      quatMultiply(q, quatConjugate(QUAT_A, C), q);
      quatNormalize(q, q);
      for (let c = 0; c < 4; c++) pose.rotation[n * 4 + c] = q[c];
    }
  }

  _applyMotion(transforms) {
    const { position, yaw } = this.motion;
    if (position[0] === 0 && position[1] === 0 && position[2] === 0 && yaw === 0) return;
    if (!this.entities.alive(this.instance)) return;
    const i = handleIndex(this.instance);
    transforms.setPosition(this.instance,
      transforms.position[i * 3] + position[0],
      transforms.position[i * 3 + 1] + position[1],
      transforms.position[i * 3 + 2] + position[2]);
    if (yaw !== 0) {
      for (let c = 0; c < 4; c++) QUAT_B[c] = transforms.rotation[i * 4 + c];
      quatMultiply(QUAT_B, yawQuat(QUAT_A, yaw), QUAT_B);
      transforms.setRotation(this.instance, quatNormalize(QUAT_B, QUAT_B));
    }
  }

  _clip(nameOrIndex) {
    return typeof nameOrIndex === 'number'
      ? this.clips[nameOrIndex]
      : this.clips.find((c) => c.name === nameOrIndex);
  }

  _layer(name) {
    const layer = this.layers.find((l) => l.name === name);
    // A typo would otherwise make a new, unmasked layer over the whole body.
    if (layer === undefined) throw new Error(`AnimationPlayer: no layer named ${JSON.stringify(name)}; make it with layer() first`);
    return layer;
  }

  /** Names -> 1 on each named node and everything under it, 0 elsewhere. */
  _mask(names) {
    const nodes = this.nodes;
    if (nodes === null) throw new Error('AnimationPlayer: masks need the asset\'s node table');
    const mask = new Float32Array(this.entityOf.length);
    for (const name of typeof names === 'string' ? [names] : names) {
      let found = false;
      for (let n = 0; n < nodes.length; n++) {
        if (nodes[n].name !== name) continue;
        found = true;
        const stack = [n];
        while (stack.length > 0) {
          const i = stack.pop();
          mask[i] = 1;
          for (const child of nodes[i].children ?? []) stack.push(child);
        }
      }
      if (!found) throw new Error(`AnimationPlayer: no node is named ${JSON.stringify(name)}`);
    }
    return mask;
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
        morph: new Array(n), morphWeight: new Array(n),
        property: new Float32Array(this._rest.property.length),
        propertyWeight: new Float32Array(this._rest.property.length / 4),
      };
    } else {
      acc.position.fill(0); acc.positionWeight.fill(0);
      acc.rotation.fill(0); acc.rotationWeight.fill(0);
      acc.scale.fill(0); acc.scaleWeight.fill(0);
      for (const sum of acc.morph) sum?.fill(0);
      for (const sum of acc.morphWeight) sum?.fill(0);
      acc.property.fill(0); acc.propertyWeight.fill(0);
    }
    return acc;
  }

  /** The pose the layers build on, reset to rest, allocated once. */
  _restPose() {
    const rest = this._rest;
    let pose = this._pose;
    if (pose === null) {
      pose = this._pose = {
        position: new Float32Array(rest.position.length),
        rotation: new Float32Array(rest.rotation.length),
        scale: new Float32Array(rest.scale.length),
        morph: rest.morph.map((weights) => weights && new Float32Array(weights.length)),
        touched: new Uint8Array(this.entityOf.length),
        property: new Float32Array(rest.property.length),
        propertyTouched: new Uint8Array(rest.property.length / 4),
      };
    }
    pose.position.set(rest.position);
    pose.rotation.set(rest.rotation);
    pose.scale.set(rest.scale);
    for (let n = 0; n < rest.morph.length; n++) if (rest.morph[n]) pose.morph[n].set(rest.morph[n]);
    pose.touched.fill(0);
    pose.property.set(rest.property);
    pose.propertyTouched.fill(0);
    return pose;
  }

  /** Write every property some layer touched, through the usual setters. */
  _write(pose, transforms) {
    for (let n = 0; n < this.entityOf.length; n++) {
      const touched = pose.touched[n];
      if (touched === 0) continue;
      const entity = this.entityOf[n];
      if (!this.entities.alive(entity)) continue;
      const p = pose.position, s = pose.scale, o = n * 3;
      if (touched & POSITION) transforms.setPosition(entity, p[o], p[o + 1], p[o + 2]);
      if (touched & SCALE) transforms.setScale(entity, s[o], s[o + 1], s[o + 2]);
      if (touched & ROTATION) {
        // Through scratch rather than a subarray view: a view per joint per
        // frame is garbage the collector then has to find.
        for (let c = 0; c < 4; c++) QUAT_A[c] = pose.rotation[n * 4 + c];
        transforms.setRotation(entity, QUAT_A);
      }
      if (touched & MORPH) this.weightsOf[n].set(pose.morph[n]);
    }
    if (this.properties === null) return;
    for (const property of this.properties.values()) {
      if (pose.propertyTouched[property.slot] === 0) continue;
      for (let c = 0; c < property.components; c++) SAMPLE[c] = pose.property[property.slot * 4 + c];
      property.write(SAMPLE);
    }
  }
}

const POSITION = 1, ROTATION = 2, SCALE = 4, MORPH = 8;
const MOVE = new Float32Array(3);
const STEP3 = new Float32Array(3);
const VEC_T = new Float32Array(3);

/** A turn of `angle` about +Y. */
function yawQuat(out, angle) {
  out[0] = 0; out[1] = Math.sin(angle / 2); out[2] = 0; out[3] = Math.cos(angle / 2);
  return out;
}

/** An angle brought into (-pi, pi]. */
function wrapAngle(angle) {
  return angle - 2 * Math.PI * Math.round(angle / (2 * Math.PI));
}

/**
 * How far `C * q` is turned about +Y: the twist of its swing-twist split.
 * Degenerate only for a half turn about a horizontal axis, which no root
 * stands in.
 */
function yawOf(C, q) {
  const cx = C[0], cy = C[1], cz = C[2], cw = C[3];
  const y = cy * q[3] + cw * q[1] + cz * q[0] - cx * q[2];
  const w = cw * q[3] - cx * q[0] - cy * q[1] - cz * q[2];
  return 2 * Math.atan2(y, w);
}

/** v rotated by the unit quaternion q, in place. */
function rotateVec(v, q, conjugate = false) {
  const qx = conjugate ? -q[0] : q[0], qy = conjugate ? -q[1] : q[1], qz = conjugate ? -q[2] : q[2], qw = q[3];
  const tx = 2 * (qy * v[2] - qz * v[1]);
  const ty = 2 * (qz * v[0] - qx * v[2]);
  const tz = 2 * (qx * v[1] - qy * v[0]);
  VEC_T[0] = v[0] + qw * tx + (qy * tz - qz * ty);
  VEC_T[1] = v[1] + qw * ty + (qz * tx - qx * tz);
  VEC_T[2] = v[2] + qw * tz + (qx * ty - qy * tx);
  v.set(VEC_T);
}

// The chain between the root-motion node and the instance, read from the
// transforms as they stand. ponytail: a node in the chain that is itself
// animated is read as of the last frame; rigs keep motion out of those nodes.

const QUAT_R = new Float32Array(4);
/** An entity's rotation, copied out: a view per call would be garbage. */
function rotationAt(transforms, i) {
  for (let c = 0; c < 4; c++) QUAT_R[c] = transforms.rotation[i * 4 + c];
  return QUAT_R;
}

/** A direction in the node's parent space, into the instance's parent space. */
function upChain(v, chain, transforms) {
  for (const entity of chain) {
    const i = handleIndex(entity);
    for (let c = 0; c < 3; c++) v[c] *= transforms.scale[i * 3 + c];
    rotateVec(v, rotationAt(transforms, i));
  }
}

/** The way back down. A zero scale has no inverse; that axis is left at 0. */
function downChain(v, chain, transforms) {
  for (let k = chain.length - 1; k >= 0; k--) {
    const i = handleIndex(chain[k]);
    rotateVec(v, rotationAt(transforms, i), true);
    for (let c = 0; c < 3; c++) {
      const s = transforms.scale[i * 3 + c];
      v[c] = s !== 0 ? v[c] / s : 0;
    }
  }
}

/** The chain's rotations, composed: node's parent space -> instance's parent space. */
function chainRotation(out, chain, transforms) {
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
  for (const entity of chain) {
    const i = handleIndex(entity);
    quatMultiply(out, rotationAt(transforms, i), out);
  }
  return out;
}
const IDENTITY = new Float32Array([0, 0, 0, 1]);

function newLayer(name) {
  return { name, tracks: [], mask: null, weight: 1, additive: false, phase: 0 };
}

function checkWeight(weight, what) {
  if (!(Number.isFinite(weight) && weight >= 0)) {
    throw new RangeError(`AnimationPlayer: ${what} must be a finite number at least 0, got ${weight}`);
  }
  return weight;
}

/** Head for zero at the pace that gets there in `fade`, then be dropped. */
function leave(track, fade) {
  track.target = 0;
  track.leaving = true;
  track.rate = track.weight / fade;
}

/** Each node's loaded TRS and morph weights, which the layers build on. */
function restPose(count, nodes, weightsOf, properties) {
  const rest = {
    position: new Float32Array(count * 3),
    rotation: new Float32Array(count * 4),
    scale: new Float32Array(count * 3).fill(1),
    morph: new Array(count).fill(null),
    // Four floats a property, the widest (a colour with alpha).
    property: new Float32Array((properties?.size ?? 0) * 4),
  };
  for (const property of properties?.values() ?? []) rest.property.set(property.rest, property.slot * 4);
  for (let n = 0; n < count; n++) {
    rest.rotation[n * 4 + 3] = 1;
    const node = nodes?.[n];
    if (node?.position) rest.position.set(node.position, n * 3);
    if (node?.rotation) rest.rotation.set(node.rotation, n * 4);
    if (node?.scale) rest.scale.set(node.scale, n * 3);
    if (weightsOf?.[n]) rest.morph[n] = Float32Array.from(weightsOf[n]);
  }
  return rest;
}

/**
 * One layer's clocks and weights for this frame. Returns how many clips it
 * still has.
 */
function stepLayer(layer, dt) {
  const tracks = layer.tracks;
  let kept = 0;
  for (const track of tracks) {
    if (!track.sync && !track.finished) advanceTrack(track, dt);
    else if (!track.sync) track.from = track.to = track.time;
    if (track.weight !== track.target) {
      const step = track.rate * dt;
      track.weight = track.weight < track.target
        ? Math.min(track.target, track.weight + step)
        : Math.max(track.target, track.weight - step);
    }
    // Only a clip on its way out is dropped at zero; one weighted to zero by
    // setWeight is still playing, waiting to be weighted back in.
    if (!(track.leaving && track.weight <= 0)) tracks[kept++] = track;
  }
  tracks.length = kept;
  syncLayer(layer, dt);
  return kept;
}

/**
 * Advance a layer's synced clips as one: a phase in cycles, at the weighted
 * average of their speeds over the weighted average of their lengths.
 */
function syncLayer(layer, dt) {
  let count = 0, weight = 0, duration = 0, speed = 0, plainDuration = 0, plainSpeed = 0;
  for (const track of layer.tracks) {
    if (!track.sync) continue;
    count++;
    weight += track.weight;
    duration += track.weight * track.clip.duration;
    speed += track.weight * track.speed;
    plainDuration += track.clip.duration;
    plainSpeed += track.speed;
  }
  if (count === 0) return;
  // All weighted out, the group still keeps time, evenly.
  if (weight > 0) { duration /= weight; speed /= weight; } else { duration = plainDuration / count; speed = plainSpeed / count; }
  const step = duration > 0 ? dt * speed / duration : 0;
  layer.phase = (layer.phase + step) % 1;
  if (layer.phase < 0) layer.phase += 1;
  for (const track of layer.tracks) {
    if (!track.sync) continue;
    track.from = track.time;
    track.to = track.from + step * track.clip.duration;
    track.time = layer.phase * track.clip.duration;
  }
}

/**
 * Lay one layer's accumulated clips over the pose.
 *
 * `reach` is how far this layer covers a node: its mask, its weight, and --
 * above the base -- how far its clips are weighted in there. The base has
 * nothing under it but the rest pose, so its clips cover their nodes fully,
 * however they are weighted, exactly as a lone clip does.
 */
function composeLayer(pose, acc, layer, base) {
  const { mask, additive } = layer;
  const count = pose.touched.length;
  for (let n = 0; n < count; n++) {
    const layerReach = (mask === null ? 1 : mask[n]) * layer.weight;
    if (layerReach <= 0) continue;

    let w = acc.positionWeight[n];
    if (w > 0) {
      const a = layerReach * (base ? 1 : Math.min(1, w));
      blend3(pose.position, acc.position, n * 3, w, a, additive ? ADD : OVER);
      pose.touched[n] |= POSITION;
    }
    w = acc.scaleWeight[n];
    if (w > 0) {
      const a = layerReach * (base ? 1 : Math.min(1, w));
      blend3(pose.scale, acc.scale, n * 3, w, a, additive ? MULTIPLY : OVER);
      pose.touched[n] |= SCALE;
    }
    w = acc.rotationWeight[n];
    if (w > 0) {
      const a = layerReach * (base ? 1 : Math.min(1, w));
      const o = n * 4;
      for (let c = 0; c < 4; c++) QUAT_B[c] = acc.rotation[o + c];
      quatNormalize(QUAT_B, QUAT_B);
      for (let c = 0; c < 4; c++) QUAT_A[c] = pose.rotation[o + c];
      if (additive) {
        // Scaled from no change toward the full change, then applied in the
        // node's own frame: rest * (first key^-1 * key) is the key itself.
        quatSlerp(QUAT_B, IDENTITY, QUAT_B, a);
        quatMultiply(QUAT_A, QUAT_A, QUAT_B);
      } else if (a >= 1) {
        for (let c = 0; c < 4; c++) QUAT_A[c] = QUAT_B[c];
      } else {
        quatSlerp(QUAT_A, QUAT_A, QUAT_B, a);
      }
      quatNormalize(QUAT_A, QUAT_A);
      for (let c = 0; c < 4; c++) pose.rotation[o + c] = QUAT_A[c];
      pose.touched[n] |= ROTATION;
    }
    const weights = acc.morphWeight[n];
    if (weights !== undefined && pose.morph[n]) {
      const target = pose.morph[n];
      const sum = acc.morph[n];
      const m = Math.min(target.length, sum.length);
      for (let c = 0; c < m; c++) {
        w = weights[c];
        if (!(w > 0)) continue;
        const a = layerReach * (base ? 1 : Math.min(1, w));
        const value = sum[c] / w;
        target[c] = additive ? target[c] + a * value : a >= 1 ? value : target[c] + (value - target[c]) * a;
        pose.touched[n] |= MORPH;
      }
    }
  }

  // Lights, cameras and materials. A mask names nodes, and these are not
  // nodes, so only the layer's weight reaches them.
  const slots = pose.propertyTouched.length;
  for (let slot = 0; slot < slots; slot++) {
    const w = acc.propertyWeight[slot];
    if (!(w > 0) || layer.weight <= 0) continue;
    const a = layer.weight * (base ? 1 : Math.min(1, w));
    for (let c = 0; c < 4; c++) {
      const o = slot * 4 + c;
      const value = acc.property[o] / w;
      pose.property[o] = additive ? pose.property[o] + a * value : a >= 1 ? value : pose.property[o] + (value - pose.property[o]) * a;
    }
    pose.propertyTouched[slot] = 1;
  }
}

const OVER = 0, ADD = 1, MULTIPLY = 2;

/** Three floats of the pose, covered, added to, or scaled by a layer. */
function blend3(pose, sum, o, w, a, mode) {
  for (let c = 0; c < 3; c++) {
    const value = sum[o + c] / w;
    if (mode === ADD) pose[o + c] += a * value;
    else if (mode === MULTIPLY) pose[o + c] *= 1 + a * (value - 1);
    else pose[o + c] = a >= 1 ? value : pose[o + c] + (value - pose[o + c]) * a;
  }
}

/** One track's clock: loop, clamp at an end, or finish. */
function advanceTrack(track, dt) {
  track.from = track.time;
  track.time += dt * track.speed;
  // Before any wrap: root motion needs how far the clip really went, and a
  // wrap is where it would otherwise see the character jump back.
  track.to = track.time;
  const duration = track.clip.duration;
  if (duration > 0) {
    if (track.loop) {
      // Modulo rather than subtraction, so a large dt or a high speed cannot
      // leave the time outside the clip.
      track.time %= duration;
      if (track.time < 0) track.time += duration;
    } else if (track.time >= duration) {
      track.time = track.to = duration;
      track.finished = true;
    } else if (track.time < 0) {
      // Backwards, the end is the start.
      track.time = track.to = 0;
      track.finished = true;
    }
  }
}
