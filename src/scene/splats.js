// Gaussian splats: reading a capture.
//
// A splat capture (3D Gaussian Splatting, Kerbl et al. 2023) is a cloud of
// soft ellipsoids, each a position, a size along three axes, a rotation, an
// opacity and a colour, fitted to photographs until drawing them back to
// front reproduces the photographs. Two files carry them:
//
//   .ply    what training writes: a binary PLY whose vertices hold x, y, z,
//           f_dc_0..2 (colour as the constant term of a spherical harmonic),
//           opacity (before a sigmoid), scale_0..2 (logarithms) and rot_0..3
//           (a quaternion, w first), and, for colour that changes with the
//           view, f_rest_*: the higher harmonics, up to degree 3, red's
//           coefficients first, then green's, then blue's.
//   .splat  the compact web format: 32 bytes a splat, position and scale as
//           floats, colour and opacity as four bytes, rotation as four bytes.
//
// Produces plain CPU data, as the glTF importer does, for render/splats.js
// to upload: each splat's centre and colour, and its covariance -- the
// ellipsoid as a symmetric 3x3, R S S^T R^T, six numbers -- which is what the
// renderer projects. Worked out once here rather than every frame there. The
// higher harmonics are kept as half floats, as PlayCanvas keeps them: half
// the memory, and no difference to see.

import { halfBits } from '../render/hdr.js';

/** The constant spherical harmonic, by which f_dc scales into a colour. */
const SH_C0 = 0.28209479177387814;
const SPLAT_BYTES = 32;

/** Coefficients a colour channel has past the constant one, by degree. */
export const SH_COEFFICIENTS = [0, 3, 8, 15];

/** 32-bit words a splat's harmonics take: three channels of half floats, packed. */
export const shWords = (degree) => Math.ceil(SH_COEFFICIENTS[degree] * 3 / 2);

/** A half float's bits, sign and all. */
function signedHalf(v) {
  return v < 0 ? 0x8000 | halfBits(-v) : halfBits(v);
}

/**
 * A capture's splats, from the bytes of a .ply or .splat file:
 * { count, centers, covariances, min, max }. `centers` is four floats a
 * splat, x, y, z and its colour and opacity as four sRGB bytes in the fourth
 * float's bits; `covariances` six, xx xy xz yy yz zz.
 */
export function parseSplats(bytes) {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const isPly = data.length >= 4 && data[0] === 0x70 && data[1] === 0x6c && data[2] === 0x79 && (data[3] === 0x0a || data[3] === 0x0d);
  if (isPly) return parsePly(data);
  // A .splat has no header to check, so the formats it could be mistaken
  // for are refused by theirs: one whose size happened to divide by 32 would
  // otherwise load as noise.
  if (data[0] === 0x50 && data[1] === 0x4b && data[2] === 0x03 && data[3] === 0x04) {
    throw new Error('splats: this is a zip -- a .sog, perhaps -- which is not read; export a .ply or .splat');
  }
  if (data[0] === 0x1f && data[1] === 0x8b) {
    throw new Error('splats: this is gzipped -- a .spz, perhaps -- which is not read; export a .ply or .splat');
  }
  return parseSplatFile(data);
}

/**
 * Room for `count` splats, with harmonics to `degree`. `sh` is shWords(degree)
 * words a splat, coefficient by coefficient, red, green and blue in each, and
 * one word over, so the shader's two-word read of the last never falls off.
 */
function output(count, degree = 0) {
  const centers = new Float32Array(count * 4);
  const sh = new Uint32Array(degree > 0 ? count * shWords(degree) + 1 : 1);
  return {
    count,
    degree,
    centers,
    colours: new Uint32Array(centers.buffer),
    covariances: new Float32Array(count * 6),
    sh,
    halves: new Uint16Array(sh.buffer),
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
  };
}

/** Splat i's coefficient k (0 is the first past the constant one), as red, green and blue. */
function putSH(out, i, k, r, g, b) {
  const h = i * shWords(out.degree) * 2 + k * 3;
  out.halves[h] = signedHalf(r);
  out.halves[h + 1] = signedHalf(g);
  out.halves[h + 2] = signedHalf(b);
}

/** One splat into the output: its centre, colour bytes, and covariance from scale and rotation. */
function put(out, i, x, y, z, r, g, b, a, sx, sy, sz, qw, qx, qy, qz) {
  if (!(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z))) {
    throw new Error(`splats: splat ${i} has a position that is not a number`);
  }
  out.centers[i * 4] = x;
  out.centers[i * 4 + 1] = y;
  out.centers[i * 4 + 2] = z;
  out.colours[i * 4 + 3] = (r | (g << 8) | (b << 16) | (a << 24)) >>> 0;
  if (x < out.min[0]) out.min[0] = x;
  if (y < out.min[1]) out.min[1] = y;
  if (z < out.min[2]) out.min[2] = z;
  if (x > out.max[0]) out.max[0] = x;
  if (y > out.max[1]) out.max[1] = y;
  if (z > out.max[2]) out.max[2] = z;

  // Rotation as a matrix, from the quaternion made unit length: training
  // leaves them as they come out of the optimiser.
  const length = Math.hypot(qw, qx, qy, qz) || 1;
  const w = qw / length, u = qx / length, v = qy / length, t = qz / length;
  const r00 = 1 - 2 * (v * v + t * t), r01 = 2 * (u * v - w * t), r02 = 2 * (u * t + w * v);
  const r10 = 2 * (u * v + w * t), r11 = 1 - 2 * (u * u + t * t), r12 = 2 * (v * t - w * u);
  const r20 = 2 * (u * t - w * v), r21 = 2 * (v * t + w * u), r22 = 1 - 2 * (u * u + v * v);
  // M = R S, and the covariance M M^T.
  const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
  const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
  const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;
  const c = out.covariances;
  const o = i * 6;
  c[o] = m00 * m00 + m01 * m01 + m02 * m02;
  c[o + 1] = m00 * m10 + m01 * m11 + m02 * m12;
  c[o + 2] = m00 * m20 + m01 * m21 + m02 * m22;
  c[o + 3] = m10 * m10 + m11 * m11 + m12 * m12;
  c[o + 4] = m10 * m20 + m11 * m21 + m12 * m22;
  c[o + 5] = m20 * m20 + m21 * m21 + m22 * m22;
}

const byte = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));

function parseSplatFile(data) {
  if (data.length === 0 || data.length % SPLAT_BYTES !== 0) {
    throw new Error(`splats: not a .ply, and ${data.length} bytes is not a whole number of 32-byte .splat records`);
  }
  const count = data.length / SPLAT_BYTES;
  const out = output(count);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  for (let i = 0; i < count; i++) {
    const o = i * SPLAT_BYTES;
    const f = (k) => view.getFloat32(o + k * 4, true);
    const q = (k) => (data[o + 28 + k] - 128) / 128;
    put(out, i, f(0), f(1), f(2), data[o + 24], data[o + 25], data[o + 26], data[o + 27],
      f(3), f(4), f(5), q(0), q(1), q(2), q(3));
  }
  return out;
}

const PLY_TYPES = {
  char: ['getInt8', 1], int8: ['getInt8', 1], uchar: ['getUint8', 1], uint8: ['getUint8', 1],
  short: ['getInt16', 2], int16: ['getInt16', 2], ushort: ['getUint16', 2], uint16: ['getUint16', 2],
  int: ['getInt32', 4], int32: ['getInt32', 4], uint: ['getUint32', 4], uint32: ['getUint32', 4],
  float: ['getFloat32', 4], float32: ['getFloat32', 4], double: ['getFloat64', 8], float64: ['getFloat64', 8],
};
const PLY_NEEDS = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];

function parsePly(data) {
  // The header is text, up to and including "end_header" and its newline.
  const marker = 'end_header';
  let end = -1;
  for (let i = 0; i < Math.min(data.length, 65536) - marker.length; i++) {
    let match = true;
    for (let k = 0; k < marker.length && match; k++) match = data[i + k] === marker.charCodeAt(k);
    if (match) { end = i + marker.length; break; }
  }
  if (end < 0) throw new Error('splats: a .ply with no end_header');
  if (data[end] === 0x0d) end++;
  end++;   // the newline
  const header = new TextDecoder().decode(data.subarray(0, end)).split(/\r?\n/);

  let format = null;
  let vertices = null;
  for (const line of header) {
    const words = line.trim().split(/\s+/);
    if (words[0] === 'format') format = words[1];
    else if (words[0] === 'element') {
      // SuperSplat's compressed .ply leads with its chunks of quantised ranges.
      if (vertices === null && words[1] === 'chunk') throw new Error('splats: a compressed .ply (from SuperSplat) is not read; export an uncompressed .ply or a .splat');
      if (vertices === null && words[1] !== 'vertex') throw new Error(`splats: a .ply whose first element is '${words[1]}', not 'vertex'`);
      if (vertices !== null) break;   // elements after the splats are not needed
      vertices = { count: Number(words[2]), properties: [], stride: 0 };
    } else if (words[0] === 'property' && vertices !== null) {
      if (words[1] === 'list') throw new Error('splats: a .ply with list properties on its vertices is not a splat capture');
      const type = PLY_TYPES[words[1]];
      if (type === undefined) throw new Error(`splats: unknown .ply property type '${words[1]}'`);
      vertices.properties.push({ name: words[2], get: type[0], offset: vertices.stride });
      vertices.stride += type[1];
    }
  }
  if (vertices === null) throw new Error('splats: a .ply with no vertices');
  if (format !== 'binary_little_endian') throw new Error(`splats: a .ply in '${format}' format; only binary_little_endian is read`);
  if (!(Number.isInteger(vertices.count) && vertices.count > 0)) throw new Error(`splats: a .ply with ${vertices.count} vertices`);
  const at = Object.fromEntries(vertices.properties.map((p) => [p.name, p]));
  const missing = PLY_NEEDS.filter((name) => at[name] === undefined);
  if (missing.length > 0) throw new Error(`splats: this .ply is not a splat capture: no ${missing.join(', ')}`);
  if (end + vertices.count * vertices.stride > data.length) {
    throw new Error(`splats: the .ply is cut short: ${vertices.count} vertices need ${vertices.count * vertices.stride} bytes, it has ${data.length - end}`);
  }

  const view = new DataView(data.buffer, data.byteOffset + end, vertices.count * vertices.stride);
  const reader = (name) => {
    const { get, offset } = at[name];
    return (base) => view[get](base + offset, true);
  };
  const [x, y, z, dc0, dc1, dc2, opacity, s0, s1, s2, q0, q1, q2, q3] = PLY_NEEDS.map(reader);
  // The higher harmonics: as many f_rest_ as there are, which says the degree.
  let rest = 0;
  while (at[`f_rest_${rest}`] !== undefined) rest++;
  const degree = SH_COEFFICIENTS.indexOf(rest / 3);
  if (rest > 0 && degree < 1) {
    throw new Error(`splats: a .ply with ${rest} f_rest_ properties; harmonics of degree 1, 2 or 3 have 9, 24 or 45`);
  }
  const restReaders = Array.from({ length: rest }, (_, k) => reader(`f_rest_${k}`));
  const per = rest / 3;
  const out = output(vertices.count, Math.max(degree, 0));
  const sigmoid = (v) => 1 / (1 + Math.exp(-v));
  for (let i = 0; i < vertices.count; i++) {
    const b = i * vertices.stride;
    put(out, i, x(b), y(b), z(b),
      byte(0.5 + SH_C0 * dc0(b)), byte(0.5 + SH_C0 * dc1(b)), byte(0.5 + SH_C0 * dc2(b)), byte(sigmoid(opacity(b))),
      Math.exp(s0(b)), Math.exp(s1(b)), Math.exp(s2(b)), q0(b), q1(b), q2(b), q3(b));
    for (let k = 0; k < per; k++) {
      putSH(out, i, k, restReaders[k](b), restReaders[k + per](b), restReaders[k + 2 * per](b));
    }
  }
  return out;
}
