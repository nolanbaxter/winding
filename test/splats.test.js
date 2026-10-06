// Gaussian splat captures: reading .ply and .splat, and placing them in a
// scene. Run: node test/splats.test.js

import assert from 'node:assert/strict';

import { parseSplats } from '../src/scene/splats.js';
import { Scene } from '../src/scene/scene.js';

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`  ok  ${name}`);
}

function close(a, b, eps = 1e-4, what = '') {
  assert.ok(Math.abs(a - b) <= eps, `${what} expected ${b}, got ${a}`);
}

/** One .splat record: position, scale, rgba bytes, rotation bytes (w x y z, 128 is zero). */
function splatRecord([x, y, z], [sx, sy, sz], rgba, rot) {
  const bytes = new Uint8Array(32);
  const view = new DataView(bytes.buffer);
  [x, y, z, sx, sy, sz].forEach((v, i) => view.setFloat32(i * 4, v, true));
  bytes.set(rgba, 24);
  bytes.set(rot, 28);
  return bytes;
}

/** A binary .ply of splats with the given properties, every one a float. */
function ply(properties, rows) {
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${rows.length}\n`
    + properties.map((p) => `property float ${p}\n`).join('') + 'end_header\n';
  const head = new TextEncoder().encode(header);
  const bytes = new Uint8Array(head.length + rows.length * properties.length * 4);
  bytes.set(head);
  const view = new DataView(bytes.buffer, head.length);
  rows.forEach((row, i) => properties.forEach((p, k) => view.setFloat32((i * properties.length + k) * 4, row[p] ?? 0, true)));
  return bytes;
}

const colourOf = (splats, i) => {
  const bits = new Uint32Array(splats.centers.buffer)[i * 4 + 3];
  return [bits & 255, (bits >>> 8) & 255, (bits >>> 16) & 255, bits >>> 24];
};

console.log('\nsplats');

test('a .splat record: centre, colour, and a covariance of its scales squared when unturned', () => {
  const s = parseSplats(splatRecord([1, 2, 3], [0.5, 2, 1], [10, 20, 30, 200], [255, 128, 128, 128]));
  assert.equal(s.count, 1);
  assert.deepEqual([...s.centers.subarray(0, 3)], [1, 2, 3]);
  assert.deepEqual(colourOf(s, 0), [10, 20, 30, 200]);
  const [xx, xy, xz, yy, yz, zz] = s.covariances;
  close(xx, 0.25); close(yy, 4); close(zz, 1);
  close(xy, 0); close(xz, 0); close(yz, 0);
  assert.deepEqual(s.min, [1, 2, 3]);
  assert.deepEqual(s.max, [1, 2, 3]);
});

test('turned a quarter about z, its long axis turns from y to x', () => {
  // w = z = cos 45: a quarter turn about z.
  const b = 128 + Math.round(Math.SQRT1_2 * 128);
  const s = parseSplats(splatRecord([0, 0, 0], [0.5, 2, 1], [0, 0, 0, 255], [b, 128, 128, b]));
  const [xx, xy, , yy] = s.covariances;
  close(xx, 4, 1e-2, 'xx');
  close(yy, 0.25, 1e-2, 'yy');
  close(xy, 0, 1e-2, 'xy');
});

test('a .ply as training writes it: harmonic colour, sigmoid opacity, log scales, extra properties skipped', () => {
  const properties = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
    'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  const SH_C0 = 0.28209479177387814;
  const s = parseSplats(ply(properties, [
    { x: -1, y: 0, z: 4, f_dc_0: (1 - 0.5) / SH_C0, f_dc_1: 0, f_dc_2: (0 - 0.5) / SH_C0,
      opacity: 0, scale_0: Math.log(2), scale_1: Math.log(3), scale_2: 0, rot_0: 2 },
    { x: 5, y: -2, z: 0, rot_0: 1 },
  ]));
  assert.equal(s.count, 2);
  assert.equal(s.degree, 0);
  assert.deepEqual(colourOf(s, 0), [255, 128, 0, 128]);
  close(s.covariances[0], 4); close(s.covariances[3], 9); close(s.covariances[5], 1);
  assert.deepEqual(s.min, [-1, -2, 0]);
  assert.deepEqual(s.max, [5, 0, 4]);
});

/** A half float's value, from its bits. */
function fromHalf(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 31;
  const mantissa = h & 1023;
  return sign * (exponent === 0 ? mantissa / 1024 * 2 ** -14 : (1 + mantissa / 1024) * 2 ** (exponent - 15));
}

test('a .ply\'s higher harmonics: the degree from how many, kept as half floats, red green and blue together', () => {
  const base = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  const rest = Array.from({ length: 9 }, (_, k) => `f_rest_${k}`);
  // f_rest_k for red is k, green 10 + k, blue -(k + 1): red's three first, then green's, then blue's.
  const row = { rot_0: 1 };
  for (let k = 0; k < 3; k++) { row[`f_rest_${k}`] = k; row[`f_rest_${k + 3}`] = 10 + k; row[`f_rest_${k + 6}`] = -(k + 1); }
  const s = parseSplats(ply([...base, ...rest], [row, row]));
  assert.equal(s.degree, 1);
  // Three coefficients of three channels: nine halves, in five words, a splat; and one word over.
  assert.equal(s.sh.length, 2 * 5 + 1);
  const halves = new Uint16Array(s.sh.buffer);
  for (const i of [0, 1]) {
    for (let k = 0; k < 3; k++) {
      const at = i * 10 + k * 3;
      assert.deepEqual([halves[at], halves[at + 1], halves[at + 2]].map(fromHalf), [k, 10 + k, -(k + 1)]);
    }
  }
  const odd = ply([...base, 'f_rest_0', 'f_rest_1'], [{}]);
  assert.throws(() => parseSplats(odd), /2 f_rest_ properties; harmonics of degree 1, 2 or 3 have 9, 24 or 45/);
});

test('what is not a splat capture is refused, by what is wrong with it', () => {
  const ascii = new TextEncoder().encode('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n');
  assert.throws(() => parseSplats(ascii), /only binary_little_endian/);
  assert.throws(() => parseSplats(ply(['x', 'y', 'z'], [{}])), /not a splat capture: no f_dc_0/);
  const whole = ply(['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'], [{}, {}]);
  assert.throws(() => parseSplats(whole.subarray(0, whole.length - 4)), /cut short/);
  assert.throws(() => parseSplats(new Uint8Array(33)), /not a whole number of 32-byte/);
  const nan = splatRecord([NaN, 0, 0], [1, 1, 1], [0, 0, 0, 0], [255, 128, 128, 128]);
  assert.throws(() => parseSplats(nan), /splat 0 has a position that is not a number/);
  // Formats a .splat could be mistaken for, sized as one would be.
  const zip = new Uint8Array(64); zip.set([0x50, 0x4b, 0x03, 0x04]);
  assert.throws(() => parseSplats(zip), /this is a zip/);
  const gz = new Uint8Array(64); gz.set([0x1f, 0x8b]);
  assert.throws(() => parseSplats(gz), /this is gzipped/);
  const compressed = new TextEncoder().encode('ply\nformat binary_little_endian 1.0\nelement chunk 1\nproperty float min_x\n'
    + 'element vertex 1\nproperty uint packed_position\nend_header\n');
  assert.throws(() => parseSplats(compressed), /a compressed .ply \(from SuperSplat\) is not read/);
});

test('a scene places splats as a node: in its bounds where the node puts them, and gone when removed', () => {
  const scene = new Scene({ capacity: 8 });
  const splats = { count: 2, centers: new Float32Array(8), min: [-1, -1, -1], max: [1, 1, 1] };
  const node = scene.addSplats({ splats, position: [10, 0, 0] });
  node.setScale(2, 2, 2);
  assert.equal(scene.splatsOf(node).splats, splats);
  const min = new Float32Array(3), max = new Float32Array(3);
  assert.ok(scene.bounds(min, max));
  assert.deepEqual([...min], [8, -2, -2]);
  assert.deepEqual([...max], [12, 2, 2]);
  scene.remove(node);
  assert.equal(scene.splats.size, 0);
  assert.equal(scene.bounds(min, max), false);
  assert.throws(() => scene.addSplats({ splats: {} }), /what engine.loadSplats returned/);
});

console.log(`\n${passed} checks passed\n`);
