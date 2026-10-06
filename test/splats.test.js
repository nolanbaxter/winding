// Gaussian splat captures: reading .ply, .splat and .spz, and placing them in a
// scene. Run: node test/splats.test.js

import assert from 'node:assert/strict';

import { gzipSync, gunzipSync } from 'node:zlib';

import { parseSplats, readSplats } from '../src/scene/splats.js';
import { Scene } from '../src/scene/scene.js';

let passed = 0;
async function test(name, fn) {
  await fn();
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

await test('a .splat record: centre, colour, and a covariance of its scales squared when unturned', () => {
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

await test('turned a quarter about z, its long axis turns from y to x', () => {
  // w = z = cos 45: a quarter turn about z.
  const b = 128 + Math.round(Math.SQRT1_2 * 128);
  const s = parseSplats(splatRecord([0, 0, 0], [0.5, 2, 1], [0, 0, 0, 255], [b, 128, 128, b]));
  const [xx, xy, , yy] = s.covariances;
  close(xx, 4, 1e-2, 'xx');
  close(yy, 0.25, 1e-2, 'yy');
  close(xy, 0, 1e-2, 'xy');
});

await test('a .ply as training writes it: harmonic colour, sigmoid opacity, log scales, extra properties skipped', () => {
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

await test('a .ply\'s higher harmonics: the degree from how many, kept as half floats, red green and blue together', () => {
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

/**
 * A .spz, packed as Niantic's writer packs one (load-spz.cc, splat-utils.h):
 * from a splat as a .ply has it, turned to RUB, quantised, gzipped.
 */
function spz(version, splats, { degree = 0, fraction = 12 } = {}) {
  const n = splats.length;
  const dim = [0, 3, 8, 15][degree];
  const positionBytes = version === 1 ? 6 : 9;
  const rotationBytes = version >= 3 ? 4 : 3;
  const bytes = new Uint8Array(16 + n * (positionBytes + 1 + 3 + 3 + rotationBytes + dim * 3));
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x5053474e, true);
  view.setUint32(4, version, true);
  view.setUint32(8, n, true);
  bytes[12] = degree;
  bytes[13] = fraction;
  const u8 = (v) => Math.max(0, Math.min(255, Math.round(v)));
  const flipSh = [-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1];
  let at = 16;
  const section = (size, write) => { splats.forEach((s, i) => write(s, at + i * size)); at += n * size; };
  section(positionBytes, ({ position: [x, y, z] }, o) => [x, -y, -z].forEach((v, a) => {
    if (version === 1) { view.setUint16(o + a * 2, halfOf(v), true); return; }
    const fixed = Math.round(v * (1 << fraction)) & 0xffffff;
    bytes.set([fixed & 255, (fixed >> 8) & 255, fixed >> 16], o + a * 3);
  }));
  section(1, ({ opacity }, o) => { bytes[o] = u8(255 / (1 + Math.exp(-opacity))); });
  section(3, ({ dc }, o) => dc.forEach((v, k) => { bytes[o + k] = u8(v * (0.15 * 255) + 0.5 * 255); }));
  section(3, ({ logScale }, o) => logScale.forEach((v, k) => { bytes[o + k] = u8((v + 10) * 16); }));
  section(rotationBytes, ({ rotation: [w, x, y, z] }, o) => {
    const length = Math.hypot(w, x, y, z);
    const q = [x / length, -y / length, -z / length, w / length];   // xyzw, turned to RUB
    if (rotationBytes === 3) {
      const sign = q[3] < 0 ? -127.5 : 127.5;
      for (let k = 0; k < 3; k++) bytes[o + k] = u8(q[k] * sign + 127.5);
      return;
    }
    let largest = 0;
    for (let k = 1; k < 4; k++) if (Math.abs(q[k]) > Math.abs(q[largest])) largest = k;
    const negate = q[largest] < 0 ? 1 : 0;
    let comp = largest;
    for (let k = 0; k < 4; k++) {
      if (k === largest) continue;
      const negbit = (q[k] < 0 ? 1 : 0) ^ negate;
      const mag = Math.floor(511 * (Math.abs(q[k]) / Math.SQRT1_2) + 0.5);
      comp = ((comp << 10) | (negbit << 9) | mag) >>> 0;
    }
    view.setUint32(o, comp, true);
  });
  section(dim * 3, ({ rest }, o) => {
    for (let k = 0; k < dim; k++) for (let c = 0; c < 3; c++) bytes[o + k * 3 + c] = u8(rest[c][k] * flipSh[k] * 128 + 128);
  });
  return gzipSync(bytes);
}

/** A half float's bits, for version 1's positions. */
function halfOf(v) {
  const f = new Float32Array([v]);
  const bits = new Uint32Array(f.buffer)[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 255) - 127 + 15;
  return sign | (exponent << 10) | Math.round((bits & 0x7fffff) / 8192);
}

await test('a .spz of versions 1 to 3 reads as the .ply it was packed from: turned back from RUB, harmonics and all', async () => {
  const SH_C0 = 0.28209479177387814;
  const splat = {
    position: [1.5, -2.25, 0.75], logScale: [-1, 0, 0.5], rotation: [0.9, 0.3, -0.2, 0.25], opacity: 1.4,
    dc: [(0.7 - 0.5) / SH_C0, (0.4 - 0.5) / SH_C0, (0.2 - 0.5) / SH_C0],
    rest: [[0.25, -0.5, 0.125], [0, 0.375, 0], [-0.25, 0, 0.5]],
  };
  const row = {
    x: 1.5, y: -2.25, z: 0.75, scale_0: -1, scale_1: 0, scale_2: 0.5, rot_0: 0.9, rot_1: 0.3, rot_2: -0.2, rot_3: 0.25,
    opacity: 1.4, f_dc_0: splat.dc[0], f_dc_1: splat.dc[1], f_dc_2: splat.dc[2],
  };
  for (let k = 0; k < 9; k++) row[`f_rest_${k}`] = splat.rest[Math.floor(k / 3)][k % 3];
  const plyNames = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3', ...Array.from({ length: 9 }, (_, k) => `f_rest_${k}`)];
  const want = parseSplats(ply(plyNames, [row]));
  for (const version of [1, 2, 3]) {
    const got = await readSplats(spz(version, [splat, splat], { degree: 1 }));
    const what = `version ${version}`;
    assert.equal(got.count, 2, what);
    assert.equal(got.degree, 1, what);
    for (let a = 0; a < 3; a++) close(got.centers[4 + a], want.centers[a], version === 1 ? 2e-3 : 1e-3, `${what} position ${a}`);
    const [r, g, b, alpha] = colourOf(got, 1);
    const [wr, wg, wb, walpha] = colourOf(want, 0);
    for (const [x, y] of [[r, wr], [g, wg], [b, wb], [alpha, walpha]]) close(x, y, 2, `${what} colour`);
    // The rotation, through the covariance it makes: quantised to a byte, so near.
    for (let k = 0; k < 6; k++) close(got.covariances[6 + k], want.covariances[k], 0.03, `${what} covariance ${k}`);
    const halves = new Uint16Array(got.sh.buffer);
    const wantHalves = new Uint16Array(want.sh.buffer);
    for (let h = 0; h < 9; h++) close(fromHalf(halves[10 + h]), fromHalf(wantHalves[h]), 1 / 128, `${what} harmonic ${h}`);
  }
  const v4 = new Uint8Array(64);
  new DataView(v4.buffer).setUint32(0, 0x5053474e, true);
  new DataView(v4.buffer).setUint32(4, 4, true);
  await assert.rejects(readSplats(v4), /version 4, compressed with zstd/);
  await assert.rejects(readSplats(gzipSync(new Uint8Array(32))), /gzipped, but not a .spz/);
  const short = spz(3, [splat]);
  await assert.rejects(readSplats(gzipSync(gunzipSync(short).subarray(0, 30))), /the .spz is cut short/);
});

/** A zip of the given files, stored, as a .sog bundles them. */
function zip(files) {
  const encoder = new TextEncoder();
  const parts = [];
  const directory = [];
  let offset = 0;
  for (const [name, data] of Object.entries(files)) {
    const nameBytes = encoder.encode(name);
    const local = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint32(18, data.length, true);
    lv.setUint32(22, data.length, true);
    lv.setUint16(26, nameBytes.length, true);
    local.set(nameBytes, 30);
    const central = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(central.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint32(20, data.length, true);
    cv.setUint32(24, data.length, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint32(42, offset, true);
    central.set(nameBytes, 46);
    parts.push(local, data);
    directory.push(central);
    offset += local.length + data.length;
  }
  const size = directory.reduce((n, d) => n + d.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(10, directory.length, true);
  ev.setUint32(12, size, true);
  ev.setUint32(16, offset, true);
  return Buffer.concat([...parts, ...directory, end]);
}

await test('a .sog reads as the .ply it was made from: log positions, smallest-three rotation, codebooks and a palette', async () => {
  // Codebooks: scales from -6 by 0.05, colours from -2 by 1/64, harmonics from -1 by 1/128.
  const scaleBook = Array.from({ length: 256 }, (_, k) => -6 + k * 0.05);
  const colourBook = Array.from({ length: 256 }, (_, k) => -2 + k / 64);
  const shBook = Array.from({ length: 256 }, (_, k) => -1 + k / 128);
  const mins = [-1, -2, -1], maxs = [1, 2, 1];
  const log = (x) => Math.sign(x) * Math.log(Math.abs(x) + 1);
  const position = [1.5, -2.25, 0.75];
  const stored = position.map((x, a) => Math.round((log(x) - mins[a]) / (maxs[a] - mins[a]) * 65535));
  const rotation = [0.9, 0.3, -0.2, 0.25].map((v, _, q) => v / Math.hypot(...q));   // w x y z: w is largest
  const quatBytes = rotation.slice(1).map((c) => Math.round((c * Math.SQRT2 + 1) / 2 * 255));
  const scaleIndex = [100, 120, 130];
  const colourIndex = [140, 100, 70];
  // One splat, two palette entries; the splat uses the second.
  const pixel = (r, g, b, a) => new Uint8Array([r, g, b, a]);
  const images = {
    'means_l.webp': pixel(stored[0] & 255, stored[1] & 255, stored[2] & 255, 255),
    'means_u.webp': pixel(stored[0] >> 8, stored[1] >> 8, stored[2] >> 8, 255),
    'quats.webp': pixel(...quatBytes, 252),
    'scales.webp': pixel(...scaleIndex, 255),
    'sh0.webp': pixel(...colourIndex, 200),
    'shN_labels.webp': pixel(1, 0, 0, 255),
  };
  // Palette entry 1 is columns 3..5 of the 192-wide row: red, green and blue indices per coefficient.
  const centroids = new Uint8Array(192 * 4);
  const restIndex = [[160, 64, 128], [128, 176, 128], [96, 128, 192]];   // [channel][coefficient]
  for (let k = 0; k < 3; k++) centroids.set([restIndex[0][k], restIndex[1][k], restIndex[2][k], 255], (3 + k) * 4);
  const meta = {
    version: 2, count: 1,
    means: { mins, maxs, files: ['means_l.webp', 'means_u.webp'] },
    scales: { codebook: scaleBook, files: ['scales.webp'] },
    quats: { files: ['quats.webp'] },
    sh0: { codebook: colourBook, files: ['sh0.webp'] },
    shN: { count: 2, bands: 1, codebook: shBook, files: ['shN_centroids.webp', 'shN_labels.webp'] },
  };
  // The "images" are their pixels as JSON, which this decodeImage reads.
  const asFile = (rgba, width) => new TextEncoder().encode(JSON.stringify({ width, height: rgba.length / 4 / width, rgba: [...rgba] }));
  const bytes = zip({
    'meta.json': new TextEncoder().encode(JSON.stringify(meta)),
    ...Object.fromEntries(Object.entries(images).map(([name, rgba]) => [name, asFile(rgba, 1)])),
    'shN_centroids.webp': asFile(centroids, 192),
  });
  const decodeImage = async (file) => {
    const { width, height, rgba } = JSON.parse(new TextDecoder().decode(file));
    return { width, height, rgba: Uint8Array.from(rgba) };
  };
  const got = await readSplats(bytes, { decodeImage });

  const row = {
    x: position[0], y: position[1], z: position[2],
    scale_0: scaleBook[100], scale_1: scaleBook[120], scale_2: scaleBook[130],
    rot_0: rotation[0], rot_1: rotation[1], rot_2: rotation[2], rot_3: rotation[3],
    opacity: Math.log(200 / 55), f_dc_0: colourBook[140], f_dc_1: colourBook[100], f_dc_2: colourBook[70],
  };
  for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) row[`f_rest_${c * 3 + k}`] = shBook[restIndex[c][k]];
  const names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3', ...Array.from({ length: 9 }, (_, k) => `f_rest_${k}`)];
  const want = parseSplats(ply(names, [row]));
  assert.equal(got.count, 1);
  assert.equal(got.degree, 1);
  for (let a = 0; a < 3; a++) close(got.centers[a], want.centers[a], 1e-3, `position ${a}`);
  assert.deepEqual(colourOf(got, 0), colourOf(want, 0));
  // The rotation, through the covariance it makes: three bytes, so near.
  for (let k = 0; k < 6; k++) close(got.covariances[k], want.covariances[k], 0.03, `covariance ${k}`);
  assert.deepEqual([...new Uint16Array(got.sh.buffer).subarray(0, 9)], [...new Uint16Array(want.sh.buffer).subarray(0, 9)]);

  await assert.rejects(readSplats(bytes), /a .sog needs decodeImage/);
  const old = zip({ 'meta.json': new TextEncoder().encode(JSON.stringify({ means: {} })) });
  await assert.rejects(readSplats(old, { decodeImage }), /a .sog of version 1; version 2 is read/);
  const missing = zip({ 'meta.json': new TextEncoder().encode(JSON.stringify(meta)) });
  await assert.rejects(readSplats(missing, { decodeImage }), /the .sog has no means_l.webp/);
});

await test('what is not a splat capture is refused, by what is wrong with it', () => {
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

await test('a scene places splats as a node: in its bounds where the node puts them, and gone when removed', () => {
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
