// Animation sampling and playback. Run: node test/animation.test.js
//
// No glTF here on purpose: clips are flat typed arrays by the time they reach
// the sampler, so these build them directly. The glTF side is in gltf.test.js.

import assert from 'node:assert/strict';

import { AnimationPlayer, sampleClip } from '../src/scene/animation.js';
import { Scene } from '../src/scene/scene.js';
import { NULL_HANDLE, HandleAllocator, handleIndex } from '../src/core/handle.js';

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

console.log(`\n${passed} checks passed\n`);
