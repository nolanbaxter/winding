// Cascaded shadow map math. Run: node test/shadow.test.js
//
// All of this is geometry, and all of it is testable without a GPU. The three
// properties below are the ones that, when wrong, produce the three classic
// shadow artefacts: a bad fit wastes resolution, a non-invariant fit makes
// shadows crawl when the camera turns, and an unsnapped box makes them crawl
// when it moves.

import assert from 'node:assert/strict';

import { mat4Create, mat4OrthographicReverseZ } from '../src/core/math/mat4.js';
import { Camera } from '../src/scene/camera.js';
import { cascadeSplits, frustumSliceSphere, MAX_CASCADES } from '../src/render/shadows.js';

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

/** Project a view-space point through an orthographic matrix. */
function orthoNdc(m, x, y, z) {
  return {
    x: m[0] * x + m[12],
    y: m[5] * y + m[13],
    z: m[10] * z + m[14],
    w: m[15],
  };
}

// --------------------------------------------------- orthographic reverse-Z

console.log('\northographic reverse-Z');

test('near maps to 1 and far maps to 0', () => {
  const m = mat4OrthographicReverseZ(mat4Create(), -10, 10, -10, 10, 1, 100);
  // The camera looks down -Z, so distance d sits at view-space z = -d.
  close(orthoNdc(m, 0, 0, -1).z, 1, 1e-6, 'near');
  close(orthoNdc(m, 0, 0, -100).z, 0, 1e-6, 'far');
  close(orthoNdc(m, 0, 0, -50.5).z, 0.5, 1e-6, 'midpoint is linear');
});

test('depth runs the same way as the perspective projection', () => {
  // The whole reason this is reversed: one depthCompare, one clear value, no
  // pass where a reader has to work out which way this buffer runs.
  const m = mat4OrthographicReverseZ(mat4Create(), -1, 1, -1, 1, 1, 10);
  assert.ok(orthoNdc(m, 0, 0, -2).z > orthoNdc(m, 0, 0, -9).z, 'nearer is larger');
});

test('x and y map the box corners to the clip cube', () => {
  const m = mat4OrthographicReverseZ(mat4Create(), -4, 12, -2, 6, 1, 10);
  close(orthoNdc(m, -4, -2, -5).x, -1, 1e-6, 'left edge');
  close(orthoNdc(m, 12, 6, -5).x, 1, 1e-6, 'right edge');
  close(orthoNdc(m, -4, -2, -5).y, -1, 1e-6, 'bottom edge');
  close(orthoNdc(m, 12, 6, -5).y, 1, 1e-6, 'top edge');
});

test('there is no perspective divide', () => {
  const m = mat4OrthographicReverseZ(mat4Create(), -1, 1, -1, 1, 1, 10);
  assert.equal(m[11], 0, 'no w from z');
  assert.equal(m[15], 1, 'w stays 1');
});

test('a degenerate box is rejected', () => {
  assert.throws(() => mat4OrthographicReverseZ(mat4Create(), 5, 5, -1, 1, 1, 10), /degenerate/);
  assert.throws(() => mat4OrthographicReverseZ(mat4Create(), -1, 1, -1, 1, 10, 1), /far must be/);
});

// ------------------------------------------------------------------ splits

console.log('\ncascade splits');

test('splits increase and end exactly at the shadow distance', () => {
  const splits = cascadeSplits(0.1, 60, 4, 0.7);
  assert.equal(splits.length, 4);
  for (let i = 1; i < splits.length; i++) {
    assert.ok(splits[i] > splits[i - 1], `not increasing at ${i}`);
  }
  close(splits[3], 60, 1e-3, 'last split is the shadow distance');
});

test('lambda 0 is a uniform split', () => {
  const splits = cascadeSplits(1, 100, 4, 0);
  close(splits[0], 25.75, 1e-4);
  close(splits[1], 50.5, 1e-4);
  close(splits[2], 75.25, 1e-4);
  close(splits[3], 100, 1e-4);
});

test('a zero near plane throws instead of producing NaN cascades', () => {
  // near * (far/near)^p is 0 * Infinity when near is 0 -- NaN even with
  // lambda 0, which would fill every cascade matrix with NaN and break every
  // shadow lookup without a single error.
  assert.throws(() => cascadeSplits(0, 100, 4, 0), /near must be positive/);
  assert.throws(() => cascadeSplits(-1, 100, 4), /near must be positive/);
});

test('lambda 1 is logarithmic, so near cascades get far more resolution', () => {
  const logarithmic = cascadeSplits(1, 100, 4, 1);
  const uniform = cascadeSplits(1, 100, 4, 0);
  // The first slice ends much sooner, which is the point: detail follows the
  // camera instead of being spread evenly over a range nobody looks at.
  assert.ok(logarithmic[0] < uniform[0] / 4, `${logarithmic[0]} vs ${uniform[0]}`);
  close(logarithmic[3], 100, 1e-3, 'still ends at the shadow distance');
});

test('a bad cascade count is rejected', () => {
  assert.throws(() => cascadeSplits(0.1, 60, 0), /out of range/);
  assert.throws(() => cascadeSplits(0.1, 60, MAX_CASCADES + 1), /out of range/);
  assert.throws(() => cascadeSplits(10, 5, 2), /shadow distance/);
});

// ------------------------------------------------------------ slice sphere

console.log('\nfrustum slice sphere');

function cameraLookingAt(target, position = [0, 0, 0]) {
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  camera.position.set(position);
  camera.target.set(target);
  camera.update(16 / 9);
  return camera;
}

test('the sphere contains every corner of the slice', () => {
  const camera = cameraLookingAt([0, 0, -1]);
  const sphere = frustumSliceSphere(new Float32Array(4), camera, 2, 20);

  const tanHalf = Math.tan(camera.fovY * 0.5);
  for (const distance of [2, 20]) {
    const h = tanHalf * distance;
    const w = h * camera.aspect;
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        // Camera at the origin looking down -Z, so corners are easy to write.
        const d = Math.hypot(w * sx - sphere[0], h * sy - sphere[1], -distance - sphere[2]);
        assert.ok(d <= sphere[3] + 1e-4, `corner at ${distance} is outside by ${d - sphere[3]}`);
      }
    }
  }
});

test('the radius does not change when the camera rotates', () => {
  // THE reason a sphere is used instead of the frustum corners. If the radius
  // moved with the camera's heading, the shadow box would resize every frame
  // and the shadows would visibly crawl.
  const directions = [
    [0, 0, -1], [1, 0, 0], [0.5, 0.5, -0.7], [-0.3, -0.9, 0.2], [0, 1, 0.001],
  ];
  const radii = directions.map((direction) => {
    const camera = cameraLookingAt(direction, [5, 3, -2]);
    return frustumSliceSphere(new Float32Array(4), camera, 1, 25)[3];
  });

  for (const radius of radii) {
    close(radius, radii[0], 1e-4, 'radius changed with heading');
  }
});

test('the radius does not change when the camera moves', () => {
  const a = frustumSliceSphere(new Float32Array(4), cameraLookingAt([0, 0, -1], [0, 0, 0]), 1, 30);
  const b = frustumSliceSphere(new Float32Array(4), cameraLookingAt([100, 0, -1], [100, 0, 0]), 1, 30);
  close(a[3], b[3], 1e-4, 'radius is translation invariant too');
});

test('the centre sits on the view axis, between the slice planes', () => {
  const camera = cameraLookingAt([0, 0, -1], [0, 0, 0]);
  const sphere = frustumSliceSphere(new Float32Array(4), camera, 4, 12);
  close(sphere[0], 0, 1e-5, 'x on the axis');
  close(sphere[1], 0, 1e-5, 'y on the axis');
  close(sphere[2], -8, 1e-5, 'midway between 4 and 12');
});

test('a wider field of view needs a bigger sphere', () => {
  const narrow = new Camera({ fovY: Math.PI / 6, near: 0.1 });
  narrow.position.set([0, 0, 0]); narrow.target.set([0, 0, -1]); narrow.update(1);

  const wide = new Camera({ fovY: Math.PI / 2, near: 0.1 });
  wide.position.set([0, 0, 0]); wide.target.set([0, 0, -1]); wide.update(1);

  const a = frustumSliceSphere(new Float32Array(4), narrow, 1, 20)[3];
  const b = frustumSliceSphere(new Float32Array(4), wide, 1, 20)[3];
  assert.ok(b > a, `wide ${b} should exceed narrow ${a}`);
});

// ---------------------------------------------------------- texel snapping

console.log('\ntexel snapping');

test('snapping quantizes the box origin to whole texels', () => {
  // The transform shadows.js applies before building each cascade's ortho box.
  // Without it the box slides by fractions of a texel every frame, every texel
  // samples a slightly different patch of world, and edges crawl even in a
  // completely static scene.
  const radius = 10;
  const size = 2048;
  const texelSize = (2 * radius) / size;

  const snap = (v) => Math.floor(v / texelSize) * texelSize;

  // The guarantee is NOT that small moves leave the box alone -- a move of 0.4
  // texels can still cross a boundary. It is that the box only ever moves by
  // WHOLE texels, so a texel never covers a fraction of a different world
  // patch than it did last frame.
  const samples = [3.14159, 3.14159 + texelSize * 0.4, 10.5, -7.2, 0];
  for (const a of samples) {
    for (const b of samples) {
      const steps = (snap(a) - snap(b)) / texelSize;
      close(steps, Math.round(steps), 1e-6, `${a} to ${b} moved a fraction of a texel`);
    }
  }

  // Idempotent: snapping an already-snapped value changes nothing.
  for (const v of samples) close(snap(snap(v)), snap(v), 1e-9, 'not idempotent');

  // One texel of camera movement moves the box exactly one texel.
  for (const v of samples) {
    close(snap(v + texelSize) - snap(v), texelSize, 1e-9, 'a whole texel step');
  }
});

console.log(`\n${passed} checks passed\n`);
