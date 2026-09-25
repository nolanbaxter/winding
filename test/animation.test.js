// Animation sampling and playback. Run: node test/animation.test.js
//
// No glTF here on purpose: clips are flat typed arrays by the time they reach
// the sampler, so these build them directly. The glTF side is in gltf.test.js.

import assert from 'node:assert/strict';

import { AnimationPlayer, sampleClip } from '../src/scene/animation.js';
import { Scene } from '../src/scene/scene.js';
import { NULL_HANDLE, HandleAllocator, handleIndex } from '../src/core/handle.js';
import { quatMultiply } from '../src/core/math/quat.js';
import { unboundedLightRadius } from '../src/scene/gltf/parse.js';

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`  ok  ${name}`);
}

const EPS = 1e-5;
function close(a, b, eps = EPS, what = '') {
  assert.ok(Math.abs(a - b) <= eps, `${what} expected ${b}, got ${a}`);
}

/** A one-channel clip driving node 0. */
function clip(path, times, values, interpolation = 'LINEAR', name = 'clip') {
  const components = path === 'rotation' ? 4 : 3;
  return {
    name,
    duration: times[times.length - 1],
    channels: [{
      node: 0,
      path,
      times: Float32Array.from(times),
      values: Float32Array.from(values),
      interpolation,
      components,
    }],
  };
}

/** A TransformStore stand-in that just records the last write. */
function recorder() {
  return {
    position: null, rotation: null, scale: null,
    setPosition(e, x, y, z) { this.position = [x, y, z]; },
    setScale(e, x, y, z) { this.scale = [x, y, z]; },
    setRotation(e, q) { this.rotation = [q[0], q[1], q[2], q[3]]; },
  };
}

// A REAL allocator, not a stub that models liveness. The bug this file now
// pins was invisible to a stub, because the stub modelled the slot and the
// allocator is the only thing that knows the generation.
const ENTITIES = new HandleAllocator(64);
const LIVE = ENTITIES.alloc();
const ENTITY_OF = [LIVE];

// ------------------------------------------------------------- interpolation

console.log('\ninterpolation');

test('LINEAR interpolates between the surrounding keyframes', () => {
  const c = clip('translation', [0, 2], [0, 0, 0, 10, 20, 30]);
  const t = recorder();
  sampleClip(c, 0.5, t, ENTITY_OF, ENTITIES);
  assert.deepEqual(t.position, [2.5, 5, 7.5]);
});

test('STEP holds the earlier keyframe until the next one arrives', () => {
  const c = clip('translation', [0, 2], [0, 0, 0, 10, 20, 30], 'STEP');
  const t = recorder();
  sampleClip(c, 1.999, t, ENTITY_OF, ENTITIES);
  assert.deepEqual(t.position, [0, 0, 0], 'still on the first key');
  sampleClip(c, 2, t, ENTITY_OF, ENTITIES);
  assert.deepEqual(t.position, [10, 20, 30]);
});

test('exactly on a middle key, STEP is on that key', () => {
  // The test above lands on the LAST key, which the clamp answers before the
  // search runs. A middle key is what the search's <= decides.
  const c = clip('translation', [0, 1, 2], [0, 0, 0, 10, 20, 30, 40, 50, 60], 'STEP');
  const t = recorder();
  sampleClip(c, 1, t, ENTITY_OF, ENTITIES);
  assert.deepEqual(t.position, [10, 20, 30]);
});

test('a time before the first key clamps rather than extrapolating', () => {
  const c = clip('translation', [1, 2], [5, 5, 5, 9, 9, 9]);
  const t = recorder();
  sampleClip(c, 0, t, ENTITY_OF, ENTITIES);
  assert.deepEqual(t.position, [5, 5, 5]);
});

test('a time past the last key holds the final value', () => {
  // A short channel in a long clip: glTF says hold, not loop and not snap back.
  const c = clip('translation', [0, 1], [0, 0, 0, 9, 9, 9]);
  const t = recorder();
  sampleClip(c, 100, t, ENTITY_OF, ENTITIES);
  assert.deepEqual(t.position, [9, 9, 9]);
});

test('coincident keyframes do not divide by zero', () => {
  const c = clip('translation', [1, 1], [0, 0, 0, 4, 4, 4]);
  const t = recorder();
  sampleClip(c, 1, t, ENTITY_OF, ENTITIES);
  assert.ok(t.position.every(Number.isFinite), `got ${t.position}`);
});

test('a single-keyframe channel is a constant', () => {
  const c = clip('scale', [0], [2, 3, 4]);
  const t = recorder();
  sampleClip(c, 12.5, t, ENTITY_OF, ENTITIES);
  assert.deepEqual(t.scale, [2, 3, 4]);
});

test('rotation LINEAR slerps, and does not lerp-and-normalize', () => {
  // Identity to 90 degrees about +Y. At t = 0.25 a real slerp is at exactly
  // 22.5 degrees; a normalized component-wise lerp is NOT -- it sweeps the arc
  // at the wrong rate. (They agree at t = 0.5, which is why testing the
  // midpoint would pass either way and catch nothing.)
  const half = Math.PI / 4;                       // half of 90 degrees
  const c = clip('rotation', [0, 1], [0, 0, 0, 1, 0, Math.sin(half), 0, Math.cos(half)]);
  const t = recorder();
  sampleClip(c, 0.25, t, ENTITY_OF, ENTITIES);

  const angle = 2 * Math.acos(t.rotation[3]);
  close(angle, (Math.PI / 2) * 0.25, 1e-4, 'swept angle');

  const nlerpW = (1 - 0.25) * 1 + 0.25 * Math.cos(half);
  const nlerpAngle = 2 * Math.acos(nlerpW / Math.hypot(0.25 * Math.sin(half), nlerpW));
  assert.ok(Math.abs(angle - nlerpAngle) > 1e-3,
    'the test is not discriminating: slerp and nlerp agree here');
});

test('CUBICSPLINE follows the Hermite curve, not a straight line', () => {
  // Zero tangents either side. Hermite then reduces to smoothstep, which at
  // t = 0.25 gives 0.15625 where a linear ramp would give 0.25.
  const values = [
    0, 0, 0, /* in */ 0, 0, 0, /* value */ 0, 0, 0, /* out */
    0, 0, 0, /* in */ 1, 1, 1, /* value */ 0, 0, 0, /* out */
  ];
  const c = clip('translation', [0, 1], values, 'CUBICSPLINE');
  const t = recorder();
  sampleClip(c, 0.25, t, ENTITY_OF, ENTITIES);
  close(t.position[0], 0.15625, 1e-6, 'hermite value');
});

test('CUBICSPLINE tangents actually bend the curve', () => {
  // Same endpoints, non-zero out-tangent on the first key: the result must
  // differ from the zero-tangent case above, or the tangents are being ignored.
  const values = [
    0, 0, 0, 0, 0, 0, 4, 4, 4,
    0, 0, 0, 1, 1, 1, 0, 0, 0,
  ];
  const c = clip('translation', [0, 1], values, 'CUBICSPLINE');
  const t = recorder();
  sampleClip(c, 0.25, t, ENTITY_OF, ENTITIES);
  assert.ok(Math.abs(t.position[0] - 0.15625) > 1e-3, `tangent ignored: ${t.position[0]}`);
});

test('a channel targeting a node this instance does not have is skipped', () => {
  const c = clip('translation', [0, 1], [0, 0, 0, 1, 1, 1]);
  c.channels[0].node = 7;
  const t = recorder();
  sampleClip(c, 0.5, t, ENTITY_OF, ENTITIES);   // no index 7
  assert.equal(t.position, null);
});

test('a channel whose node was never instantiated does not drive entity 0', () => {
  // The node->entity map is pre-filled with NULL_HANDLE, which IS 0. A guard
  // testing for undefined lets an unreached node write to whatever entity was
  // added first.
  const c = clip('translation', [0, 1], [0, 0, 0, 100, 200, 300]);
  c.channels[0].node = 1;
  const t = recorder();
  sampleClip(c, 1, t, [LIVE, NULL_HANDLE], ENTITIES);   // node 1 was never reached
  assert.equal(t.position, null, 'wrote into the null entity');
});

test('a channel pointing at a freed entity is skipped', () => {
  const c = clip('translation', [0, 1], [0, 0, 0, 1, 1, 1]);
  const entities = new HandleAllocator(8);
  const doomed = entities.alloc();
  const t = recorder();

  sampleClip(c, 1, t, [doomed], entities);
  assert.deepEqual(t.position, [1, 1, 1], 'samples while it is alive');

  t.position = null;
  entities.free(doomed);
  sampleClip(c, 1, t, [doomed], entities);
  assert.equal(t.position, null, 'and not once it is freed');
});

test('a channel pointing at a RECYCLED slot is skipped', () => {
  // The case a slot test cannot see, and the one that actually happens:
  // handles are recycled last-in-first-out, so the very next alloc() takes the
  // slot back and marks it live. Only the generation distinguishes the stale
  // handle from the new occupant.
  const c = clip('translation', [0, 1], [0, 0, 0, 1, 1, 1]);
  const entities = new HandleAllocator(8);
  const doomed = entities.alloc();
  entities.free(doomed);
  const reused = entities.alloc();

  assert.equal(handleIndex(doomed), handleIndex(reused), 'the slot really was reused');
  assert.notEqual(doomed, reused, 'but the handle is a different one');

  const t = recorder();
  sampleClip(c, 1, t, [doomed], entities);
  assert.equal(t.position, null, 'the stale handle must not drive its replacement');

  sampleClip(c, 1, t, [reused], entities);
  assert.deepEqual(t.position, [1, 1, 1], 'the live one still works');
});

// ------------------------------------------------------------------ playback

console.log('\nplayback');

test('play selects by name or by index, and reports an unknown clip', () => {
  const player = new AnimationPlayer([clip('translation', [0, 1], [0, 0, 0, 1, 1, 1], 'LINEAR', 'walk')], ENTITY_OF, ENTITIES);
  assert.equal(player.play('walk'), true);
  assert.equal(player.play(0), true);
  assert.equal(player.play('nope'), false);
  assert.deepEqual(player.names, ['walk']);
});

test('a looping clip wraps instead of running off the end', () => {
  const player = new AnimationPlayer([clip('translation', [0, 2], [0, 0, 0, 10, 0, 0])], ENTITY_OF, ENTITIES);
  const t = recorder();
  player.play(0, { loop: true });

  player.advance(3, t);
  close(player.time, 1, EPS, 'wrapped time');
  close(t.position[0], 5, EPS);

  // A single step longer than several loops must still land inside the clip.
  player.advance(100, t);
  assert.ok(player.time >= 0 && player.time < 2, `time escaped the clip: ${player.time}`);
});

test('a non-looping clip stops at the end and stays there', () => {
  const player = new AnimationPlayer([clip('translation', [0, 2], [0, 0, 0, 10, 0, 0])], ENTITY_OF, ENTITIES);
  const t = recorder();
  player.play(0, { loop: false });

  player.advance(5, t);
  close(player.time, 2, EPS);
  close(t.position[0], 10, EPS, 'held on the final pose');
  assert.equal(player.finished, true);

  // Further advances do nothing rather than drifting past the end.
  assert.equal(player.advance(5, t), false);
  close(player.time, 2, EPS);
});

test('a non-looping clip that lands exactly on its end is finished', () => {
  const player = new AnimationPlayer([clip('translation', [0, 2], [0, 0, 0, 10, 0, 0])], ENTITY_OF, ENTITIES);
  player.play(0, { loop: false });
  player.advance(2, recorder());
  assert.equal(player.finished, true);
});

test('a fade blends from one clip to the next, then leaves only the new one', () => {
  const player = new AnimationPlayer([
    clip('translation', [0, 1], [0, 0, 0, 0, 0, 0], 'LINEAR', 'idle'),
    clip('translation', [0, 1], [10, 0, 0, 10, 0, 0], 'LINEAR', 'walk'),
  ], ENTITY_OF, ENTITIES);
  const t = recorder();
  player.play('idle');
  player.advance(0.1, t);
  player.play('walk', { fade: 1 });

  player.advance(0.5, t);
  close(t.position[0], 5, EPS, 'halfway through the fade');
  assert.equal(player.tracks.length, 2);
  assert.equal(player.clip.name, 'walk', 'the player reports the clip it is heading to');

  player.advance(0.5, t);
  close(t.position[0], 10, EPS, 'all the way');
  assert.equal(player.tracks.length, 1, 'the faded clip is dropped');
});

test('rotations blend along the short way, whatever sign a clip stores', () => {
  // q and -q are one rotation. Summed as stored, a quarter turn given as -q
  // would pull the blend the long way round.
  const s = Math.sin(Math.PI / 4), c = Math.cos(Math.PI / 4);
  const player = new AnimationPlayer([
    clip('rotation', [0, 1], [0, 0, 0, 1, 0, 0, 0, 1], 'LINEAR', 'a'),
    clip('rotation', [0, 1], [0, -s, 0, -c, 0, -s, 0, -c], 'LINEAR', 'b'),
  ], ENTITY_OF, ENTITIES);
  const t = recorder();
  player.play('a');
  player.advance(0, t);
  player.play('b', { fade: 1 });
  player.advance(0.5, t);
  // Halfway between none and a quarter turn about Y is an eighth of a turn.
  const [x, y, z, w] = t.rotation;
  close(Math.abs(y), Math.sin(Math.PI / 8), 1e-5, 'y');
  close(Math.abs(w), Math.cos(Math.PI / 8), 1e-5, 'w');
  close(x, 0, EPS); close(z, 0, EPS);
});

test('a node only the new clip animates takes its value during a fade', () => {
  // Blending toward the rest pose would be inventing an opinion the clip
  // that leaves this node alone never gave.
  const scaleOnly = clip('scale', [0, 1], [2, 2, 2, 2, 2, 2], 'LINEAR', 'grow');
  const player = new AnimationPlayer([
    clip('translation', [0, 1], [0, 0, 0, 0, 0, 0], 'LINEAR', 'still'), scaleOnly,
  ], ENTITY_OF, ENTITIES);
  const t = recorder();
  player.play('still');
  player.advance(0, t);
  player.play('grow', { fade: 1 });
  player.advance(0.25, t);
  assert.deepEqual(t.scale, [2, 2, 2]);
});

test('without a fade, play replaces whatever was playing', () => {
  const player = new AnimationPlayer([
    clip('translation', [0, 1], [0, 0, 0, 0, 0, 0], 'LINEAR', 'a'),
    clip('translation', [0, 1], [4, 0, 0, 4, 0, 0], 'LINEAR', 'b'),
  ], ENTITY_OF, ENTITIES);
  const t = recorder();
  player.play('a');
  player.play('b');
  assert.equal(player.tracks.length, 1);
  player.advance(0.1, t);
  close(t.position[0], 4);
});

test('speed scales time, and a negative speed wraps backwards', () => {
  const player = new AnimationPlayer([clip('translation', [0, 2], [0, 0, 0, 10, 0, 0])], ENTITY_OF, ENTITIES);
  const t = recorder();

  player.play(0, { speed: 2 });
  player.advance(0.5, t);
  close(player.time, 1, EPS, 'double speed');

  player.play(0, { speed: -1 });
  player.advance(0.5, t);
  close(player.time, 1.5, EPS, 'wrapped to the end of the clip');
});

test('a clip played backwards without looping finishes at its start', () => {
  // It clamped to 0 and kept reporting itself as playing, forever.
  const player = new AnimationPlayer([clip('translation', [0, 2], [0, 0, 0, 10, 0, 0])], ENTITY_OF, ENTITIES);
  const t = recorder();
  player.play(0, { loop: false, speed: -1, time: 1 });
  player.advance(0.5, t);
  assert.equal(player.finished, false, 'still going');
  player.advance(1, t);
  close(player.time, 0, EPS, 'clamped at the start');
  assert.equal(player.finished, true);
  assert.equal(player.advance(0.1, t), false, 'and reports itself done');
});

test('stop leaves the pose alone and halts further sampling', () => {
  const player = new AnimationPlayer([clip('translation', [0, 2], [0, 0, 0, 10, 0, 0])], ENTITY_OF, ENTITIES);
  const t = recorder();
  player.play(0);
  player.advance(1, t);
  const pose = [...t.position];

  player.stop();
  assert.equal(player.advance(1, t), false);
  assert.deepEqual(t.position, pose);
});

// --------------------------------------------------------------- integration

console.log('\nscene integration');

/** The minimum asset shape scene.add() consumes. */
function animatedAsset() {
  return {
    nodes: [{ name: 'root', position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: -1 }],
    meshes: [],
    roots: [0],
    animations: [clip('translation', [0, 2], [0, 0, 0, 10, 0, 0], 'LINEAR', 'slide')],
  };
}

test('two instances of one asset animate independently', () => {
  // The reason the node->entity map is kept per instance. Sharing it would make
  // the second instance overwrite the first every frame.
  const scene = new Scene({ capacity: 32 });
  const asset = animatedAsset();
  const a = scene.add(asset);
  const b = scene.add(asset);

  a.play('slide');
  b.play('slide');
  scene.advanceAnimations(0.5);
  scene.advanceAnimations(0.5);        // a and b are now both at t = 1.0
  b.animation.time = 0;
  scene.advanceAnimations(0);

  scene.update();
  const x = (node) => scene.transforms.world[scene.transforms.worldOffset(node.entity) + 12];
  close(x(a), 5, EPS, 'instance a is halfway');
  close(x(b), 0, EPS, 'instance b was rewound on its own');
});

test('an asset with no clips registers no player', () => {
  const scene = new Scene({ capacity: 32 });
  const asset = animatedAsset();
  asset.animations = [];
  const node = scene.add(asset);

  assert.equal(node.animation, null);
  assert.deepEqual(node.animations, []);
  node.play('slide');                  // must not throw
  assert.equal(scene.advanceAnimations(0.1), 0);
});

test('removing an instance removes its player', () => {
  const scene = new Scene({ capacity: 32 });
  const node = scene.add(animatedAsset());
  node.play('slide');
  assert.equal(scene.advanceAnimations(0.1), 1);

  scene.remove(node);
  assert.equal(scene.advanceAnimations(0.1), 0, 'a removed instance must not still be sampled');
});

test('animation marks transforms dirty, so composition picks it up', () => {
  // Sampling writes through the TransformStore setters for exactly this reason.
  const scene = new Scene({ capacity: 32 });
  const node = scene.add(animatedAsset());
  scene.update();                      // settle

  node.play('slide');
  scene.advanceAnimations(1);
  assert.equal(scene.update(), 1, 'the animated node must recompose');
});

// ------------------------------------------------------------ morph weights

console.log('\nmorph weight channels');

/** A one-channel clip driving node 0's morph weights. */
function weightClip(times, values, components, interpolation = 'LINEAR') {
  return {
    name: 'expression',
    duration: times[times.length - 1],
    channels: [{
      node: 0,
      path: 'weights',
      times: Float32Array.from(times),
      values: Float32Array.from(values),
      interpolation,
      components,
    }],
  };
}

test('a weights channel writes the instance array, not a transform', () => {
  const weights = new Float32Array(2);
  const t = recorder();
  // Two targets, two keys: [0,1] at t=0 and [1,0] at t=1.
  sampleClip(weightClip([0, 1], [0, 1, 1, 0], 2), 0.25, t, ENTITY_OF, ENTITIES, [weights]);

  close(weights[0], 0.25, EPS, 'target 0');
  close(weights[1], 0.75, EPS, 'target 1');
  assert.equal(t.position, null, 'nothing was written to the transform');
});

test('four morph targets are not mistaken for a quaternion', () => {
  // The trap this exists to catch. A four-target weights channel has exactly
  // the shape of a rotation channel, and the sampler used to decide between
  // lerp and SLERP by the component count. Slerping four independent sliders
  // sweeps them through a sphere and normalizes them to unit length -- every
  // number wrong, nothing thrown.
  const weights = new Float32Array(4);
  sampleClip(
    weightClip([0, 1], [0, 0, 0, 0, 1, 1, 1, 1], 4),
    0.5, recorder(), ENTITY_OF, ENTITIES, [weights],
  );

  for (let i = 0; i < 4; i++) close(weights[i], 0.5, EPS, `target ${i}`);
  // A slerp would have produced a unit-length result. Halfway between two
  // sliders is 0.5 each, whose length is 1.0 only by coincidence of four
  // components -- so check the values, not the length.
  close(weights[0] + weights[1] + weights[2] + weights[3], 2, EPS, 'sum');
});

test('a weights channel holds outside its range like any other', () => {
  const weights = new Float32Array(2);
  const clip = weightClip([1, 2], [0, 0, 1, 1], 2);
  sampleClip(clip, 0, recorder(), ENTITY_OF, ENTITIES, [weights]);
  close(weights[0], 0, EPS, 'before the first key');
  sampleClip(clip, 100, recorder(), ENTITY_OF, ENTITIES, [weights]);
  close(weights[0], 1, EPS, 'after the last key');
});

test('a STEP weights channel does not interpolate', () => {
  const weights = new Float32Array(2);
  sampleClip(
    weightClip([0, 1], [0, 1, 1, 0], 2, 'STEP'),
    0.9, recorder(), ENTITY_OF, ENTITIES, [weights],
  );
  close(weights[0], 0, EPS, 'held at the earlier key');
});

test('a weights channel with no instance array to write is skipped', () => {
  // A clip that names a node this instance did not morph. Not an error: the
  // same non-event as a channel naming a node the instance does not have.
  sampleClip(weightClip([0, 1], [0, 1], 1), 0.5, recorder(), ENTITY_OF, ENTITIES, null);
  sampleClip(weightClip([0, 1], [0, 1], 1), 0.5, recorder(), ENTITY_OF, ENTITIES, []);
});

test('a clip wider than the mesh writes only what both agree on', () => {
  // Three targets in the clip, two on the instance. Writing the third would
  // run off the end of the array.
  const weights = new Float32Array(2);
  sampleClip(
    weightClip([0, 1], [0, 0, 0, 1, 1, 1], 3),
    1, recorder(), ENTITY_OF, ENTITIES, [weights],
  );
  close(weights[0], 1, EPS, 'target 0');
  close(weights[1], 1, EPS, 'target 1');
});

test('the player drives weights through a whole scene', () => {
  // End to end: an asset with a weights channel, added to a scene, advanced.
  // The wiring between add() and the player is where skinning broke twice.
  const scene = new Scene({ capacity: 16 });
  const node = scene.add({
    nodes: [{
      name: 'head',
      position: Float32Array.from([0, 0, 0]),
      rotation: Float32Array.from([0, 0, 0, 1]),
      scale: Float32Array.from([1, 1, 1]),
      children: [],
      mesh: 0,
      skin: -1,
      weights: Float32Array.from([0, 0]),
    }],
    meshes: [{
      name: 'face',
      targetCount: 2,
      primitives: [{
        indexCount: 6,
        materialId: 0,
        bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
        morphExtent: Float32Array.from([1, 1]),
      }],
    }],
    roots: [0],
    animations: [weightClip([0, 1], [0, 1, 1, 0], 2)],
  });

  assert.ok(node.play(0, { loop: false }), 'the clip is there');
  scene.advanceAnimations(0.25);
  close(node.weights[0], 0.25, EPS, 'target 0 after a quarter second');
  close(node.weights[1], 0.75, EPS, 'target 1 after a quarter second');
});

// ------------------------------------------------------------ layers and weights

console.log('\nlayers, masks and blend weights');

// Two nodes: a body at the root and an arm under it, each with its own entity.
const BODY = ENTITIES.alloc();
const ARM = ENTITIES.alloc();
const RIG = [BODY, ARM];
const RIG_NODES = [
  { name: 'body', position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [1] },
  { name: 'arm', position: [4, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [] },
];

/** A clip holding one path of each named node at a fixed value, then another. */
function rigClip(name, path, perNode, times = [0, 1]) {
  const components = path === 'rotation' ? 4 : 3;
  return {
    name,
    duration: times[times.length - 1],
    channels: Object.entries(perNode).map(([node, values]) => ({
      node: Number(node), path, components, interpolation: 'LINEAR',
      times: Float32Array.from(times), values: Float32Array.from(values),
    })),
  };
}
/** Hold x at `x` on both keys. */
const holdX = (x) => [x, 0, 0, x, 0, 0];

/** A TransformStore stand-in that records the last write per entity. */
function rigRecorder() {
  const at = new Map();
  const slot = (e) => at.get(e) ?? at.set(e, {}).get(e);
  return {
    at,
    x: (e) => slot(e).position?.[0],
    setPosition(e, x, y, z) { slot(e).position = [x, y, z]; },
    setScale(e, x, y, z) { slot(e).scale = [x, y, z]; },
    setRotation(e, q) { slot(e).rotation = [q[0], q[1], q[2], q[3]]; },
  };
}

function rigPlayer(clips) {
  return new AnimationPlayer(clips, RIG, ENTITIES, null, RIG_NODES);
}

test('a masked layer drives only the nodes under its mask', () => {
  const player = rigPlayer([
    rigClip('walk', 'translation', { 0: holdX(1), 1: holdX(1) }),
    rigClip('wave', 'translation', { 0: holdX(9), 1: holdX(5) }),
  ]);
  const t = rigRecorder();
  player.layer('upper', { mask: 'arm' });
  player.play('walk');
  player.play('wave', { layer: 'upper' });
  player.advance(0.1, t);
  close(t.x(BODY), 1, EPS, 'the body keeps the walk');
  close(t.x(ARM), 5, EPS, 'the arm waves');
});

test('a mask names a node and everything under it', () => {
  const player = rigPlayer([
    rigClip('walk', 'translation', { 0: holdX(1), 1: holdX(1) }),
    rigClip('wave', 'translation', { 0: holdX(9), 1: holdX(5) }),
  ]);
  const t = rigRecorder();
  player.layer('upper', { mask: ['body'] });
  player.play('walk');
  player.play('wave', { layer: 'upper' });
  player.advance(0.1, t);
  close(t.x(BODY), 9);
  close(t.x(ARM), 5, EPS, 'the arm is under the body');
});

test('fading a clip in on an empty layer fades the layer in', () => {
  const player = rigPlayer([
    rigClip('walk', 'translation', { 1: holdX(1) }),
    rigClip('wave', 'translation', { 1: holdX(5) }),
  ]);
  const t = rigRecorder();
  player.layer('upper');
  player.play('walk');
  player.play('wave', { layer: 'upper', fade: 1 });
  player.advance(0.5, t);
  close(t.x(ARM), 3, EPS, 'halfway between the walk and the wave');
  player.advance(0.5, t);
  close(t.x(ARM), 5, EPS, 'all the way');
});

test('stopping a layer with a fade hands its nodes back to the layer beneath', () => {
  const player = rigPlayer([
    rigClip('walk', 'translation', { 1: holdX(1) }),
    rigClip('wave', 'translation', { 1: holdX(5) }),
  ]);
  const t = rigRecorder();
  player.layer('upper');
  player.play('walk');
  player.play('wave', { layer: 'upper' });
  player.advance(0.1, t);
  player.stop({ layer: 'upper', fade: 1 });
  player.advance(0.5, t);
  close(t.x(ARM), 3);
  player.advance(0.5, t);
  close(t.x(ARM), 1);
  assert.equal(player.layers[1].tracks.length, 0, 'the faded clip is dropped');
  assert.equal(player.tracks.length, 1, 'the base layer plays on');
});

test('a layer\'s weight scales how far it covers', () => {
  const player = rigPlayer([
    rigClip('walk', 'translation', { 1: holdX(1) }),
    rigClip('wave', 'translation', { 1: holdX(5) }),
  ]);
  const t = rigRecorder();
  player.layer('upper', { weight: 0.25 });
  player.play('walk');
  player.play('wave', { layer: 'upper' });
  player.advance(0.1, t);
  close(t.x(ARM), 2);
});

test('a node only an upper layer animates blends from its loaded pose', () => {
  // Not from wherever the last frame left it, which would ease toward the
  // target a little more every frame instead of holding a blend.
  const player = rigPlayer([rigClip('wave', 'translation', { 1: holdX(8) })]);
  const t = rigRecorder();
  player.layer('upper', { weight: 0.5 });
  player.play('wave', { layer: 'upper' });
  player.advance(0.1, t);
  close(t.x(ARM), 6, EPS, 'halfway from the loaded 4');
  player.advance(0.1, t);
  close(t.x(ARM), 6, EPS, 'and still there');
});

test('an additive layer adds each clip\'s change from its first key', () => {
  const player = rigPlayer([
    rigClip('walk', 'translation', { 1: holdX(2) }),
    // From 10 to 13: the change is what counts, not where it starts.
    rigClip('lean', 'translation', { 1: [10, 0, 0, 13, 0, 0] }),
  ]);
  const t = rigRecorder();
  player.layer('lean', { additive: true });
  player.play('walk');
  player.play('lean', { layer: 'lean', loop: false });
  player.advance(0.5, t);
  close(t.x(ARM), 3.5, EPS, '2 + half of 3');
  player.layer('lean', { weight: 2 });
  player.advance(0.5, t);
  close(t.x(ARM), 8, EPS, 'weighted past one, it exaggerates: 2 + 2 * 3');
});

test('an additive rotation turns the node in its own frame, as far as it is weighted', () => {
  const turn = (axis, angle) => {
    const q = [0, 0, 0, Math.cos(angle / 2)];
    q['xyz'.indexOf(axis)] = Math.sin(angle / 2);
    return q;
  };
  const quarter = Math.PI / 2;
  // The change is a quarter turn about y, stored from a first key that does
  // not commute with it, so first^-1 * key and key * first^-1 differ.
  const first = turn('z', quarter);
  const last = quatMultiply([0, 0, 0, 1], first, turn('y', quarter));
  const player = rigPlayer([
    rigClip('face', 'rotation', { 1: [...turn('x', quarter), ...turn('x', quarter)] }),
    rigClip('turn', 'rotation', { 1: [...first, ...last] }),
  ]);
  const t = rigRecorder();
  const check = (expected) => {
    const q = t.at.get(ARM).rotation;
    const sign = Math.sign(q[3] * expected[3]) || 1;
    for (let c = 0; c < 4; c++) close(q[c] * sign, expected[c], 1e-5, `component ${c}`);
  };
  player.layer('turn', { additive: true });
  player.play('face');
  player.play('turn', { layer: 'turn', loop: false });
  player.advance(1, t);
  // face * quarter-y: x-quarter, then y-quarter in the node's own frame.
  check(quatMultiply([0, 0, 0, 1], turn('x', quarter), turn('y', quarter)));

  player.layer('turn', { weight: 0.5 });
  player.advance(0, t);
  check(quatMultiply([0, 0, 0, 1], turn('x', quarter), turn('y', quarter / 2)));
});

test('an additive CUBICSPLINE clip measures from its first value, not its tangent', () => {
  // Each key is in-tangent, value, out-tangent; the tangents here are wild.
  const lean = {
    name: 'lean',
    duration: 1,
    channels: [{
      node: 1, path: 'translation', components: 3, interpolation: 'CUBICSPLINE',
      times: Float32Array.from([0, 1]),
      values: Float32Array.from([99, 99, 99, 10, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 99, 99, 99]),
    }],
  };
  const player = rigPlayer([rigClip('walk', 'translation', { 1: holdX(2) }), lean]);
  const t = rigRecorder();
  player.layer('lean', { additive: true });
  player.play('walk');
  player.play('lean', { layer: 'lean', loop: false });
  player.advance(1, t);
  close(t.x(ARM), 5, EPS, '2 + (13 - 10)');
});

test('a node a layer stops animating holds where it was', () => {
  // The layered path starts every frame from the rest pose; only what some
  // layer touched THIS frame may be written back.
  const player = rigPlayer([
    rigClip('walk', 'translation', { 0: holdX(1) }),
    rigClip('wave', 'translation', { 1: holdX(5) }),
    rigClip('nod', 'translation', { 0: holdX(3) }),
  ]);
  const t = rigRecorder();
  player.layer('upper');
  player.play('walk');
  player.play('wave', { layer: 'upper' });
  player.advance(0.1, t);
  close(t.x(ARM), 5);
  player.play('nod', { layer: 'upper' });
  player.advance(0.1, t);
  close(t.x(ARM), 5, EPS, 'not put back to its loaded 4');
});

test('an additive scale multiplies', () => {
  const player = rigPlayer([
    rigClip('big', 'scale', { 1: [2, 2, 2, 2, 2, 2] }),
    rigClip('breathe', 'scale', { 1: [1, 1, 1, 1.5, 1, 1] }),
  ]);
  const t = rigRecorder();
  player.layer('breath', { additive: true });
  player.play('big');
  player.play('breathe', { layer: 'breath', loop: false });
  player.advance(1, t);
  assert.deepEqual(t.at.get(ARM).scale, [3, 2, 2]);
});

test('an additive layer adds to morph weights', () => {
  const weights = Float32Array.from([0.25, 0]);
  const player = new AnimationPlayer([
    { ...weightClip([0, 1], [0, 0, 0.5, 0.5], 2), name: 'smile' },
  ], RIG, ENTITIES, [weights], RIG_NODES);
  player.layer('face', { additive: true });
  player.play('smile', { layer: 'face', loop: false });
  player.advance(1, rigRecorder());
  close(weights[0], 0.75, EPS, 'the loaded 0.25 plus 0.5');
  close(weights[1], 0.5);
});

test('the base layer cannot be additive, and a layer or mask name must exist', () => {
  const player = rigPlayer([rigClip('walk', 'translation', { 1: holdX(1) })]);
  assert.throws(() => player.layer('base', { additive: true }), /base layer cannot be additive/);
  assert.throws(() => player.play('walk', { layer: 'upperr' }), /no layer named "upperr"/);
  assert.throws(() => player.layer('upper', { mask: 'amr' }), /no node is named "amr"/);
  assert.throws(() => player.play('walk', { weight: -1 }), /weight must be a finite number/);
  assert.throws(() => player.layer('upper', { weight: NaN }), /layer weight must be/);
});

test('setWeight blends clips by a parameter, and zero keeps a clip playing', () => {
  const player = rigPlayer([
    rigClip('walk', 'translation', { 1: holdX(0) }),
    rigClip('run', 'translation', { 1: holdX(10) }),
  ]);
  const t = rigRecorder();
  player.play('walk');
  player.play('run', { add: true, weight: 0 });
  player.advance(0.1, t);
  close(t.x(ARM), 0, EPS, 'run weighted out');

  player.setWeight('walk', 0.3);
  player.setWeight('run', 0.7);
  player.advance(0.1, t);
  close(t.x(ARM), 7);

  player.setWeight('run', 0);
  player.advance(0.1, t);
  assert.equal(player.tracks.length, 2, 'a clip weighted to zero is not dropped');
  player.setWeight('run', 1, { fade: 1 });
  player.advance(0.5, t);
  close(t.x(ARM), 10 * 0.5 / 0.8, EPS, 'run at 0.5 against walk at 0.3');
  assert.equal(player.setWeight('jump', 1), false, 'not playing');
});

test('synced clips share a cycle at the weighted average length', () => {
  const player = rigPlayer([
    rigClip('walk', 'translation', { 1: holdX(0) }, [0, 1]),
    rigClip('run', 'translation', { 1: holdX(0) }, [0, 0.5]),
  ]);
  const t = rigRecorder();
  player.play('walk', { sync: true });
  player.advance(0.25, t);
  close(player.tracks[0].time, 0.25);

  player.play('run', { sync: true, add: true });
  close(player.tracks[1].time, 0.125, EPS, 'the run joins a quarter of the way through its cycle');

  player.setWeight('walk', 0.75);
  player.setWeight('run', 0.25);
  // The group's cycle is 0.75 * 1 + 0.25 * 0.5 = 0.875 s: 0.4375 s is half.
  player.advance(0.4375, t);
  close(player.tracks[0].time, 0.75, EPS, 'walk three quarters through');
  close(player.tracks[1].time, 0.375, EPS, 'and run three quarters through');

  player.advance(0.4375, t);
  close(player.tracks[0].time, 0.25, EPS, 'the cycle loops');
});

test('a synced clip loops even when asked not to', () => {
  const player = rigPlayer([rigClip('walk', 'translation', { 1: holdX(0) }, [0, 1])]);
  player.play('walk', { sync: true, loop: false });
  player.advance(1.5, rigRecorder());
  assert.equal(player.finished, false);
  close(player.time, 0.5);
});

test('setWeight catches a clip on its way out, even at zero', () => {
  const player = rigPlayer([
    rigClip('walk', 'translation', { 1: holdX(0) }),
    rigClip('run', 'translation', { 1: holdX(10) }),
  ]);
  player.play('walk');
  player.play('run', { fade: 1 });
  player.advance(0.5, rigRecorder());
  player.setWeight('walk', 0);
  player.advance(0.5, rigRecorder());
  assert.equal(player.tracks.length, 2, 'walk is no longer leaving');
});

test('a player whose clips have finished samples again when a weight changes', () => {
  const player = rigPlayer([
    rigClip('pose', 'translation', { 1: holdX(10) }),
    rigClip('aim', 'translation', { 1: holdX(20) }),
  ]);
  const t = rigRecorder();
  player.layer('aim');
  player.play('pose', { loop: false });
  player.play('aim', { layer: 'aim', loop: false });
  player.advance(2, t);
  close(t.x(ARM), 20);
  assert.equal(player.advance(0.1, t), false, 'settled');

  player.layer('aim', { weight: 0.5 });
  assert.equal(player.advance(0.1, t), true);
  close(t.x(ARM), 15);
});

test('through scene.add, a mask finds the asset\'s nodes by name', () => {
  const scene = new Scene({ capacity: 32 });
  const node = scene.add(animatedAsset());
  node.animation.layer('upper', { mask: 'root' });
  assert.throws(() => node.animation.layer('lower', { mask: 'legs' }), /no node is named "legs"/);
  node.play('slide', { layer: 'upper', fade: 1, loop: false });
  scene.advanceAnimations(1);
  scene.advanceAnimations(1);
  scene.update();
  close(scene.transforms.world[scene.transforms.worldOffset(node.entity) + 12], 10);
  node.stop({ layer: 'upper', fade: 0.5 });
  assert.equal(node.animation.layers[1].tracks[0].leaving, true);
});

// ------------------------------------------------------------- pointer channels

console.log('\nlights, cameras and materials by pointer');

/** A property channel, as the glTF reader makes one from a pointer. */
function propertyChannel(kind, index, field, times, values, components = 1, interpolation = 'LINEAR') {
  return {
    node: -1, path: 'property', kind, index, field, key: `${kind}/${index}/${field}`, components, interpolation,
    times: Float32Array.from(times), values: Float32Array.from(values),
  };
}

/** One lamp node, one camera node, one material, and whatever clips. */
function propsAsset(clips, { range = 5 } = {}) {
  const node = (name, extra) => ({
    name, position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: -1, ...extra,
  });
  return {
    nodes: [node('lamp', { light: 0 }), node('eye', { camera: 0 })],
    meshes: [],
    roots: [0, 1],
    lights: [{ type: 'spot', color: [1, 1, 1], intensity: 10, radius: range ?? 42, range, innerAngle: 0.1, outerAngle: 0.5 }],
    cameras: [{ orthographic: false, fovY: 0.8, near: 0.1 }],
    materials: [{
      name: 'glow', baseColorFactor: Float32Array.of(1, 1, 1, 1),
      emissive: Float32Array.of(1, 1, 1), emissiveFactor: Float32Array.of(1, 1, 1), emissiveStrength: 1,
    }],
    materialIds: [7],
    animations: clips,
  };
}

const clipOf = (name, ...channels) => ({
  name, duration: Math.max(...channels.map((c) => c.times[c.times.length - 1])), channels,
});
/** The lamp: the only packed light, since the default sun is directional. */
const LIGHT = (scene) => { assert.equal(scene.lightCount, 1); return scene.lights.subarray(0, 16); };

test('a clip drives a light\'s intensity and colour on its node', () => {
  const scene = new Scene({ capacity: 32 });
  const asset = propsAsset([clipOf('flicker',
    propertyChannel('light', 0, 'intensity', [0, 1], [10, 20]),
    propertyChannel('light', 0, 'color', [0, 1], [1, 1, 1, 1, 0, 0], 3),
  )]);
  const root = scene.add(asset);
  root.play('flicker', { loop: false });
  scene.advanceAnimations(0.5);
  const light = LIGHT(scene);
  close(light[7], 15, EPS, 'intensity');
  assert.deepEqual([...light.subarray(4, 7)], [1, 0.5, 0.5]);
  assert.equal(light[3], 5, 'a light with a range keeps it');
});

test('a light the file gave no range re-derives its reach as its intensity moves', () => {
  const scene = new Scene({ capacity: 32 });
  const root = scene.add(propsAsset([clipOf('up', propertyChannel('light', 0, 'intensity', [0, 1], [10, 40]))], { range: null }));
  root.play('up', { loop: false });
  scene.advanceAnimations(1);
  close(LIGHT(scene)[3], unboundedLightRadius(40, [1, 1, 1]), 1e-4);
});

test('a light value the importer would refuse is not written', () => {
  // A CUBICSPLINE overshoot below zero, and an inner cone past the outer.
  const scene = new Scene({ capacity: 32 });
  const root = scene.add(propsAsset([clipOf('bad',
    propertyChannel('light', 0, 'intensity', [0, 1], [10, -5]),
    propertyChannel('light', 0, 'innerAngle', [0, 1], [0.1, 0.9]),
  )]));
  root.play('bad', { loop: false });
  scene.advanceAnimations(0.5);
  close(LIGHT(scene)[7], 2.5, EPS, 'still positive at the midpoint');
  scene.advanceAnimations(0.5);
  close(LIGHT(scene)[7], 2.5, EPS, 'held at the last it would accept');
  close(scene._lightCone[0], 0.1, 1e-6, 'inner never reached the outer, so it stayed');
});

test('a clip drives a camera\'s field of view, but never to one the importer would refuse', () => {
  const scene = new Scene({ capacity: 32 });
  const root = scene.add(propsAsset([
    clipOf('zoom', propertyChannel('camera', 0, 'fovY', [0, 1], [0.8, 0.4])),
    clipOf('flip', propertyChannel('camera', 0, 'fovY', [0, 1], [4, 4])),
  ]));
  root.play('zoom', { loop: false });
  scene.advanceAnimations(1);
  close(scene.cameras[0].fovY, 0.4);
  root.play('flip');
  scene.advanceAnimations(0.1);
  close(scene.cameras[0].fovY, 0.4, EPS, 'past pi is not a field of view');
});

test('a clip changes an asset\'s material and queues it for upload', () => {
  const scene = new Scene({ capacity: 32 });
  const asset = propsAsset([clipOf('pulse',
    propertyChannel('material', 0, 'baseColorFactor', [0, 1], [1, 1, 1, 1, 0, 0.5, 1, 0.5], 4),
    propertyChannel('material', 0, 'emissiveStrength', [0, 1], [1, 3]),
  )]);
  const root = scene.add(asset);
  root.play('pulse', { loop: false });
  scene.advanceAnimations(1);
  const [record] = asset.materials;
  assert.deepEqual([...record.baseColorFactor], [0, 0.5, 1, 0.5]);
  assert.deepEqual([...record.emissive], [3, 3, 3], 'emissive is factor times strength');
  assert.equal(scene.changedMaterials.get(7), record, 'queued under its id');
});

test('a clip scrolls and turns a texture through its transform', () => {
  // A conveyor belt: the base colour texture's offset animated along u, and
  // the emissive texture turned. Each rebuilds only its own slot's rows.
  const scene = new Scene({ capacity: 32 });
  const asset = propsAsset([clipOf('belt',
    propertyChannel('material', 0, 'uv0.offset', [0, 1], [0, 0, 1, 0], 2),
    propertyChannel('material', 0, 'uv4.rotation', [0, 1], [0, Math.PI / 2]),
  )]);
  const parts = () => ({ offset: [0, 0], rotation: 0, scale: [1, 1] });
  asset.materials[0].uvTransformParts = [parts(), parts(), parts(), parts(), parts()];
  asset.materials[0].uvTransforms = Float32Array.from({ length: 30 }, (_, i) => (i % 6 === 0 || i % 6 === 4 ? 1 : 0));
  const root = scene.add(asset);
  root.play('belt', { loop: false });
  scene.advanceAnimations(0.5);
  const t = asset.materials[0].uvTransforms;
  [1, 0, 0.5, 0, 1, 0].forEach((v, i) => close(t[i], v, 1e-6, `base colour moved half along u, value ${i}`));
  const [c, s] = [Math.cos(Math.PI / 4), Math.sin(Math.PI / 4)];
  [c, -s, 0, s, c, 0].forEach((v, i) => close(t[24 + i], v, 1e-6, `emissive row value ${i}`));
  [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0].forEach((v, i) => close(t[6 + i], v, 0, `the other slots untouched, value ${i}`));
  assert.equal(scene.changedMaterials.get(7), asset.materials[0], 'queued for upload');
});


test('property channels cross-fade and add like any other', () => {
  const scene = new Scene({ capacity: 32 });
  const root = scene.add(propsAsset([
    clipOf('dim', propertyChannel('camera', 0, 'fovY', [0, 1], [0.2, 0.2])),
    clipOf('wide', propertyChannel('camera', 0, 'fovY', [0, 1], [1.0, 1.0])),
    clipOf('nudge', propertyChannel('camera', 0, 'fovY', [0, 1], [0.5, 0.6])),
  ]));
  root.play('dim');
  scene.advanceAnimations(0.1);
  root.play('wide', { fade: 1 });
  scene.advanceAnimations(0.5);
  close(scene.cameras[0].fovY, 0.6, 1e-6, 'halfway through the fade');

  // An upper layer fading in starts from the loaded 0.8, not from zero.
  const fresh = scene.add(propsAsset([clipOf('wide', propertyChannel('camera', 0, 'fovY', [0, 1], [1.0, 1.0]))]));
  fresh.animation.layer('upper');
  fresh.play('wide', { layer: 'upper', fade: 1 });
  scene.advanceAnimations(0.25);
  close(scene.cameras[1].fovY, 0.85, 1e-6, 'a quarter of the way from the loaded value');
  scene.remove(fresh);

  root.animation.layer('nudge', { additive: true });
  root.play('nudge', { layer: 'nudge', loop: false });
  scene.advanceAnimations(0.5);
  close(scene.cameras[0].fovY, 1.0 + 0.05, 1e-6, 'the change from 0.5 to 0.55 on top');
});

test('one weight by pointer blends alone, not as a share of the whole list', () => {
  const weights = Float32Array.from([0, 0]);
  const player = new AnimationPlayer([
    { ...weightClip([0, 1], [0.2, 0.2, 0.2, 0.2], 2), name: 'both' },
    {
      name: 'second',
      duration: 1,
      channels: [{
        node: 0, path: 'weights', offset: 1, components: 1, interpolation: 'LINEAR',
        times: Float32Array.of(0, 1), values: Float32Array.of(1, 1),
      }],
    },
  ], ENTITY_OF, ENTITIES, [weights]);
  player.play('second');
  player.advance(0.1, recorder());
  assert.deepEqual([...weights], [0, 1], 'the lone pointer writes only its own weight');

  player.play('both');
  player.play('second', { add: true });
  player.advance(0.1, recorder());
  close(weights[0], 0.2, EPS, 'only one clip drives the first weight, so it has all of it');
  close(weights[1], 0.6, EPS, 'two drive the second, evenly');
});

// ------------------------------------------------------------------ root motion

console.log('\nroot motion');

function vecClose3(a, b, what = '') {
  for (let i = 0; i < 3; i++) close(a[i], b[i], 1e-5, `${what} [${i}]`);
}

/** Quaternions, either sign. */
function vecClose4(a, b, what = '') {
  const sign = Math.sign(a[3] * b[3]) || 1;
  for (let i = 0; i < 4; i++) close(a[i] * sign, b[i], 1e-5, `${what} [${i}]`);
}

/** One translation or rotation channel on node `node`. */
function nodeChannel(node, path, times, values) {
  return {
    node, path, components: path === 'rotation' ? 4 : 3, interpolation: 'LINEAR',
    times: Float32Array.from(times), values: Float32Array.from(values),
  };
}

/**
 * A body with hips under it, optionally through an armature node rotated a
 * quarter turn about -X, as a Z-up rig exports. Clips drive the hips.
 */
function walkerAsset(clips, { zUp = false } = {}) {
  const node = (name, children, extra = {}) => ({
    name, position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children, mesh: -1, ...extra,
  });
  const nodes = zUp
    ? [node('body', [1]), node('armature', [2], { rotation: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2] }), node('hips', [], { position: [0, 0, 1] })]
    : [node('body', [1]), node('hips', [], { position: [0, 1, 0] })];
  return { nodes, meshes: [], roots: [0], animations: clips };
}

/** Hips from rest (0,1,0) forward 2 along +Z and up 0.5 over one second. */
const forward = (name = 'walk', distance = 2) => ({
  name, duration: 1, channels: [nodeChannel(1, 'translation', [0, 1], [0, 1, 0, 0, 1.5, distance])],
});

function walker(clips, options) {
  const scene = new Scene({ capacity: 32 });
  const model = scene.add(walkerAsset(clips, options));
  const t = scene.transforms;
  const at = (entity) => [...t.position.subarray(handleIndex(entity) * 3, handleIndex(entity) * 3 + 3)];
  const hipsNode = options?.zUp ? 2 : 1;
  return {
    scene, model, player: model.animation,
    body: () => at(model.entity),
    hips: () => at(model.animation.entityOf[hipsNode]),
    hipsRotation: () => [...t.rotation.subarray(handleIndex(model.animation.entityOf[hipsNode]) * 4, handleIndex(model.animation.entityOf[hipsNode]) * 4 + 4)],
  };
}

test('a walk moves the character, and the hips stay over it', () => {
  const w = walker([forward()]);
  w.player.rootMotion();
  w.model.play('walk');
  for (let i = 0; i < 3; i++) w.scene.advanceAnimations(0.25);
  vecClose3(w.body(), [0, 0, 1.5], 'the body walked 1.5');
  vecClose3(w.hips(), [0, 1.375, 0], 'the hips kept their height and stayed over it');
});

test('a loop is a step forward, not a jump back', () => {
  const w = walker([forward()]);
  w.player.rootMotion();
  w.model.play('walk');
  w.scene.advanceAnimations(0.75);
  w.scene.advanceAnimations(0.5);      // across the wrap
  vecClose3(w.body(), [0, 0, 2.5], 'two per second, through the loop');
  w.scene.advanceAnimations(2.25);     // several whole cycles in one step
  vecClose3(w.body(), [0, 0, 7], 'and through several');
});

test('played backwards, the character walks backwards', () => {
  const w = walker([forward()]);
  w.player.rootMotion();
  w.model.play('walk', { speed: -1 });
  w.scene.advanceAnimations(0.5);
  w.scene.advanceAnimations(0.75);
  vecClose3(w.body(), [0, 0, -2.5]);
});

test('a clip that stops moves the character exactly to its end', () => {
  const w = walker([forward()]);
  w.player.rootMotion();
  w.model.play('walk', { loop: false });
  w.scene.advanceAnimations(0.75);
  w.scene.advanceAnimations(5);
  vecClose3(w.body(), [0, 0, 2]);
});

test('motion follows the character\'s own facing and scale', () => {
  const w = walker([forward()]);
  w.model.setRotation([0, Math.SQRT1_2, 0, Math.SQRT1_2]);   // a quarter turn left: +Z becomes +X
  w.model.setScale(2, 2, 2);
  w.player.rootMotion();
  w.model.play('walk');
  w.scene.advanceAnimations(0.5);
  vecClose3(w.body(), [2, 0, 0], 'half a second at two, doubled, along +X');
});

test('a Z-up rig still walks across the ground, not into the sky', () => {
  // Under the armature the hips' local +Y is the world's -Z and local +Z is up,
  // as in the Fox. Motion is measured in the instance's parent space.
  const w = walker([{
    name: 'walk', duration: 1,
    channels: [nodeChannel(2, 'translation', [0, 1], [0, 0, 1, 0, 2, 1.5])],
  }], { zUp: true });
  w.player.rootMotion();
  w.model.play('walk');
  w.scene.advanceAnimations(0.5);
  vecClose3(w.body(), [0, 0, -1], 'forward along -Z, level');
  vecClose3(w.hips(), [0, 0, 1.25], 'the hips keep their height, in their own axes');
});

test('a turn turns the character, and the hips face the way they rested', () => {
  const turn = (a) => [0, Math.sin(a / 2), 0, Math.cos(a / 2)];
  const w = walker([{
    name: 'turn', duration: 1,
    channels: [nodeChannel(1, 'rotation', [0, 1], [...turn(0), ...turn(Math.PI / 2)])],
  }]);
  // Named: a clip that only turns moves no node, so there is none to find.
  w.player.rootMotion({ node: 'hips' });
  w.model.play('turn', { loop: false });
  w.scene.advanceAnimations(0.5);
  const body = w.scene.transforms.rotation.subarray(handleIndex(w.model.entity) * 4, handleIndex(w.model.entity) * 4 + 4);
  vecClose4([...body], turn(Math.PI / 4), 'an eighth of a turn');
  vecClose4(w.hipsRotation(), [0, 0, 0, 1], 'the hips hand the turn over');
  assert.equal(w.player.motion.yaw.toFixed(6), (Math.PI / 4).toFixed(6));
});

test('blended clips move the character at their weighted pace', () => {
  const w = walker([forward('walk', 2), forward('run', 4)]);
  w.player.rootMotion();
  w.model.play('walk', { sync: true });
  w.model.play('run', { sync: true, add: true, weight: 1 });
  w.scene.advanceAnimations(0.5);
  vecClose3(w.body(), [0, 0, 1.5], 'three a second, the average');
});

test('with apply: false, the motion is reported and the character left alone', () => {
  const w = walker([forward()]);
  w.player.rootMotion({ apply: false });
  w.model.play('walk');
  w.scene.advanceAnimations(0.5);
  vecClose3(w.body(), [0, 0, 0]);
  vecClose3([...w.player.motion.position], [0, 0, 1]);
  vecClose3(w.hips(), [0, 1.25, 0], 'the hips still stay in place: the controller moves the body');
});

test('vertical: true hands the height over too', () => {
  const w = walker([forward()]);
  w.player.rootMotion({ vertical: true });
  w.model.play('walk');
  w.scene.advanceAnimations(0.5);
  vecClose3(w.body(), [0, 0.25, 1]);
  vecClose3(w.hips(), [0, 1, 0]);
});

test('turned off, the hips walk away again', () => {
  const w = walker([forward()]);
  w.player.rootMotion();
  w.model.play('walk');
  w.scene.advanceAnimations(0.5);
  w.player.rootMotion(null);
  w.scene.advanceAnimations(0.25);
  vecClose3(w.body(), [0, 0, 1], 'no further motion');
  vecClose3([...w.player.motion.position], [0, 0, 0], 'and none reported');
  vecClose3(w.hips(), [0, 1.375, 1.5], 'the clip drives the hips directly');
});

test('the node that carries the motion is found, or asked for by name', () => {
  const hipsAndTail = walkerAsset([forward()]);
  hipsAndTail.nodes[0].children.push(2);
  hipsAndTail.nodes.push({ name: 'tail', position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: -1 });
  hipsAndTail.animations[0].channels.push(nodeChannel(2, 'translation', [0, 1], [0, 0, 0, 0, 0, 1]));
  const scene = new Scene({ capacity: 32 });
  const model = scene.add(hipsAndTail);
  assert.throws(() => model.animation.rootMotion(), /"hips" and "tail" equally high up; name the one/);
  model.animation.rootMotion({ node: 'hips' });
  assert.throws(() => model.animation.rootMotion({ node: 'body' }), /"body" is the instance itself/);
  assert.throws(() => model.animation.rootMotion({ node: 'neck' }), /no nodes are named "neck"/);

  const still = walker([{ name: 'spin', duration: 1, channels: [nodeChannel(1, 'rotation', [0, 1], [0, 0, 0, 1, 0, 0, 0, 1])] }]);
  assert.throws(() => still.player.rootMotion(), /no clip moves a node below the instance/);
});

test('on a Z-up rig, a turn about the rig\'s up axis is a turn about the world\'s', () => {
  // The armature maps the hips' local +Z onto world +Y, so a clip turning
  // the hips about local Z turns the character about the vertical.
  const aboutZ = (a) => [0, 0, Math.sin(a / 2), Math.cos(a / 2)];
  const w = walker([{
    name: 'turn', duration: 1,
    channels: [nodeChannel(2, 'rotation', [0, 1], [...aboutZ(0), ...aboutZ(Math.PI / 2)])],
  }], { zUp: true });
  w.player.rootMotion({ node: 'hips' });
  w.model.play('turn', { loop: false });
  w.scene.advanceAnimations(0.25);
  w.scene.advanceAnimations(0.25);
  close(w.player.motion.yaw, Math.PI / 8, 1e-5, 'the second eighth of the quarter turn');
  vecClose4(w.hipsRotation(), [0, 0, 0, 1], 'the hips face the way they rested');
});


test('a looping turn keeps turning through the wrap', () => {
  const turn = (a) => [0, Math.sin(a / 2), 0, Math.cos(a / 2)];
  const w = walker([{
    name: 'circle', duration: 1,
    channels: [nodeChannel(1, 'rotation', [0, 1], [...turn(0), ...turn(Math.PI / 2)])],
  }]);
  w.player.rootMotion({ node: 'hips' });
  w.model.play('circle');
  w.scene.advanceAnimations(0.75);
  w.scene.advanceAnimations(0.5);
  close(w.player.motion.yaw, Math.PI / 4, 1e-5, 'half a second of a quarter turn a second, across the wrap');
});

test('synced clips move the character through their shared wrap', () => {
  const w = walker([forward('walk', 2), forward('run', 4)]);
  w.player.rootMotion();
  w.model.play('walk', { sync: true });
  w.model.play('run', { sync: true, add: true, weight: 1 });
  w.scene.advanceAnimations(0.75);
  w.scene.advanceAnimations(0.5);
  vecClose3(w.body(), [0, 0, 3.75], 'three a second for 1.25 s');
});

test('a clip that has finished adds no more motion to a blend', () => {
  const w = walker([forward('walk', 2), forward('run', 4)]);
  w.player.rootMotion();
  w.model.play('walk', { loop: false });
  w.scene.advanceAnimations(1.5);
  vecClose3(w.body(), [0, 0, 2], 'the walk ran out at 2');
  w.model.play('run', { add: true });
  w.scene.advanceAnimations(0.5);
  // The run moves 2 in that half second; the finished walk moves nothing and
  // has an equal weight, so the character moves 1.
  vecClose3(w.body(), [0, 0, 3]);
});

test('an asset with several roots moves by its wrapper, facing included', () => {
  // The Fox's shape: the scene puts a wrapper over the roots, and that is
  // what the player moves and what its facing comes from.
  const asset = walkerAsset([forward()]);
  asset.nodes.push({ name: 'prop', position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: -1 });
  asset.roots = [0, 2];
  const scene = new Scene({ capacity: 32 });
  const model = scene.add(asset);
  assert.notEqual(model.entity, model.animation.entityOf[0], 'a wrapper, not the body');
  model.setRotation([0, Math.SQRT1_2, 0, Math.SQRT1_2]);
  model.animation.rootMotion();
  model.play('walk');
  scene.advanceAnimations(0.5);
  const i = handleIndex(model.entity);
  vecClose3([...scene.transforms.position.subarray(i * 3, i * 3 + 3)], [1, 0, 0], 'forward is the wrapper\'s +X');
});

test('the instance\'s own node is never the one found, even when a clip moves it', () => {
  const clip = forward();
  clip.channels.push(nodeChannel(0, 'translation', [0, 1], [0, 0, 0, 5, 0, 0]));
  const w = walker([clip]);
  w.player.rootMotion();
  w.model.play('walk');
  w.scene.advanceAnimations(0.5);
  assert.equal(w.player.motion.position[2].toFixed(5), '1.00000', 'the hips carry it');
});


console.log(`\n${passed} checks passed\n`);
