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
//   .spz    Niantic's: gzip around a 16-byte header and the splats' fields one
//           after another, quantised -- positions in 24-bit fixed point,
//           everything else a byte -- and stored y up, z back (RUB), where a
//           .ply is y down, z forward (RDF): turned back here, harmonics and
//           all, so a capture lands as its .ply would. Versions 1 to 3, as
//           Niantic's own reader (MIT) reads them; version 4 is zstd, which no
//           browser unpacks.
//   .sog    PlayCanvas's: a zip of meta.json and lossless WebP images, a pixel
//           a splat -- positions as two bytes of a log-scaled range, rotation
//           as three components and which one was left out, scales and colour
//           as indices into 256-entry codebooks, the harmonics as a palette.
//           Already in the .ply's axes. Version 2, as PlayCanvas's own
//           splat-transform (MIT) reads it.
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
  if (isSpz4(data)) {
    throw new Error('splats: a .spz of version 4, compressed with zstd, which browsers cannot unpack; '
      + 'save it as version 3, or as a .ply or .splat');
  }
  const isPly = data.length >= 4 && data[0] === 0x70 && data[1] === 0x6c && data[2] === 0x79 && (data[3] === 0x0a || data[3] === 0x0d);
  if (isPly) return parsePly(data);
  // A .splat has no header to check, so the formats it could be mistaken
  // for are refused by theirs: one whose size happened to divide by 32 would
  // otherwise load as noise.
  if (isZip(data)) throw new Error('splats: this is a zip: unpack it with readSplats, which reads a .sog');
  if (data[0] === 0x1f && data[1] === 0x8b) {
    throw new Error('splats: this is gzipped: unpack it with readSplats, which reads a .spz');
  }
  return parseSplatFile(data);
}

/**
 * As parseSplats, for every format engine.loadSplats takes, the compressed
 * ones too: a .spz is gunzipped, by the browser's own DecompressionStream,
 * before it is read; a .sog is unzipped, and its images decoded by
 * `decodeImage(bytes)` -> { width, height, rgba }, which must hand back the
 * bytes exactly as stored: never premultiplied, never colour-managed.
 */
export async function readSplats(bytes, { decodeImage } = {}) {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (isZip(data)) return readSog(data, decodeImage);
  if (data[0] === 0x1f && data[1] === 0x8b) {
    const unpacked = new Uint8Array(await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    return parseSpz(unpacked);
  }
  return parseSplats(data);
}

const isZip = (data) => data[0] === 0x50 && data[1] === 0x4b && data[2] === 0x03 && data[3] === 0x04;

/** A zip's files, by name, each a function that gives its bytes. Stored and deflated entries; not zip64. */
function unzip(data) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  // The end record is the last thing in the file, before a comment of at most 65535 bytes.
  let end = -1;
  for (let i = data.length - 22; i >= Math.max(0, data.length - 22 - 65535); i--) {
    if (view.getUint32(i, true) === 0x06054b50) { end = i; break; }
  }
  if (end < 0) throw new Error('splats: a zip with no end record: cut short?');
  const files = new Map();
  let at = view.getUint32(end + 16, true);
  for (let n = view.getUint16(end + 10, true); n > 0; n--) {
    if (at + 46 > data.length || view.getUint32(at, true) !== 0x02014b50) throw new Error('splats: a zip whose directory is damaged');
    const method = view.getUint16(at + 10, true);
    const size = view.getUint32(at + 20, true);
    const nameLength = view.getUint16(at + 28, true);
    const local = view.getUint32(at + 42, true);
    const name = new TextDecoder().decode(data.subarray(at + 46, at + 46 + nameLength));
    if (size === 0xffffffff || local === 0xffffffff) throw new Error('splats: a zip64 .sog is not read');
    at += 46 + nameLength + view.getUint16(at + 30, true) + view.getUint16(at + 32, true);
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    if (start + size > data.length) throw new Error(`splats: the zip is cut short in ${name}`);
    const stored = data.subarray(start, start + size);
    if (method !== 0 && method !== 8) throw new Error(`splats: ${name} is compressed with method ${method}; stored and deflate are read`);
    files.set(name, () => (method === 0 ? Promise.resolve(stored)
      : new Response(new Blob([stored]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer().then((b) => new Uint8Array(b))));
  }
  return files;
}

/** A .sog: meta.json, then each image a splat a pixel, decoded as splat-transform's readSog decodes them. */
async function readSog(data, decodeImage) {
  if (typeof decodeImage !== 'function') throw new Error('splats: a .sog needs decodeImage, for its WebP images');
  const files = unzip(data);
  const file = async (name) => {
    const get = files.get(name);
    if (get === undefined) throw new Error(`splats: the .sog has no ${name}`);
    return get();
  };
  const meta = JSON.parse(new TextDecoder().decode(await file('meta.json')));
  if (meta.version !== 2) {
    throw new Error(`splats: a .sog of version ${meta.version ?? 1}; version 2 is read (re-save it with splat-transform)`);
  }
  const count = meta.count;
  if (!(Number.isInteger(count) && count > 0)) throw new Error(`splats: a .sog of ${count} splats`);
  const image = async (name) => {
    const decoded = await decodeImage(await file(name));
    if (decoded.width * decoded.height < count) throw new Error(`splats: ${name} has ${decoded.width * decoded.height} pixels, for ${count} splats`);
    return decoded;
  };
  const [low, high, quats, scales, sh0] = await Promise.all([
    image(meta.means.files[0]), image(meta.means.files[1]), image(meta.quats.files[0]),
    image(meta.scales.files[0]), image(meta.sh0.files[0]),
  ]);
  const bands = meta.shN ? meta.shN.bands : 0;
  const degree = SH_COEFFICIENTS[bands] > 0 ? bands : 0;
  const per = SH_COEFFICIENTS[degree];
  let centroids = null;
  let labels = null;
  if (degree > 0) {
    [centroids, labels] = await Promise.all([decodeImage(await file(meta.shN.files[0])), image(meta.shN.files[1])]);
    if (centroids.width !== 64 * per) throw new Error(`splats: the .sog's harmonic palette is ${centroids.width} wide; ${64 * per} for degree ${degree}`);
  }

  const out = output(count, degree);
  const { mins, maxs } = meta.means;
  // Positions were stored as sign(x) ln(|x| + 1), to spend the bits near the middle.
  const position = (a, v) => {
    const n = mins[a] + ((maxs[a] - mins[a]) || 1) * v / 65535;
    const e = Math.exp(Math.abs(n)) - 1;
    return n < 0 ? -e : e;
  };
  const scaleBook = meta.scales.codebook;
  const colourBook = meta.sh0.codebook;
  const shBook = degree > 0 ? meta.shN.codebook : null;
  const q = [0, 0, 0, 0];
  for (let i = 0; i < count; i++) {
    const o = i * 4;
    const m = (a) => position(a, low.rgba[o + a] | (high.rgba[o + a] << 8));
    // Three components, w x y z less the largest, which the tag (252 + its
    // index) names and unit length gives back.
    const tag = quats.rgba[o + 3];
    if (tag < 252) {
      q[0] = 1; q[1] = 0; q[2] = 0; q[3] = 0;
    } else {
      const largest = tag - 252;
      let k = 0;
      let sum = 0;
      for (let c = 0; c < 4; c++) {
        if (c === largest) continue;
        q[c] = ((quats.rgba[o + k++] / 255) * 2 - 1) / Math.SQRT2;
        sum += q[c] * q[c];
      }
      q[largest] = Math.sqrt(Math.max(0, 1 - sum));
    }
    const colour = (k) => byte(0.5 + SH_C0 * colourBook[sh0.rgba[o + k]]);
    const scale = (k) => Math.exp(scaleBook[scales.rgba[o + k]]);
    put(out, i, m(0), m(1), m(2), colour(0), colour(1), colour(2), sh0.rgba[o + 3],
      scale(0), scale(1), scale(2), q[0], q[1], q[2], q[3]);
    if (degree > 0) {
      const label = labels.rgba[o] | (labels.rgba[o + 1] << 8);
      if (label >= meta.shN.count) continue;
      const row = Math.floor(label / 64) * centroids.width + (label % 64) * per;
      for (let k = 0; k < per; k++) {
        const c = (row + k) * 4;
        if (c + 2 >= centroids.rgba.length) break;
        putSH(out, i, k, shBook[centroids.rgba[c]], shBook[centroids.rgba[c + 1]], shBook[centroids.rgba[c + 2]]);
      }
    }
  }
  return out;
}

const SPZ_MAGIC = 0x5053474e;   // 'NGSP'
const SPZ_COLOUR_SCALE = 0.15;
/** Turning RUB to RDF: x stays, y and z turn over. Each harmonic's sign, in 3DGS order (Niantic's table). */
const SPZ_SH_FLIP = [-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1];

const isSpz4 = (data) => data.length >= 8 && data[0] === 0x4e && data[1] === 0x47 && data[2] === 0x53 && data[3] === 0x50
  && new DataView(data.buffer, data.byteOffset, 8).getUint32(4, true) >= 4;

/** A half float's value, from its bits. */
function fromHalf(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 31;
  const mantissa = h & 1023;
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * (exponent === 0 ? mantissa * 2 ** -24 : (1 + mantissa / 1024) * 2 ** (exponent - 15));
}

/** A gunzipped .spz: the header, then positions, alphas, colours, scales, rotations and harmonics, each for every splat. */
function parseSpz(data) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (data.length < 16 || view.getUint32(0, true) !== SPZ_MAGIC) throw new Error('splats: gzipped, but not a .spz');
  const version = view.getUint32(4, true);
  if (version < 1 || version > 3) throw new Error(`splats: a .spz of version ${version}; versions 1 to 3 are read`);
  const count = view.getUint32(8, true);
  const fileDegree = data[12];
  const fraction = data[13];
  if (fileDegree > 4) throw new Error(`splats: a .spz with harmonics of degree ${fileDegree}`);
  // Degree 4's band, which nothing here evaluates, is left out.
  const degree = Math.min(fileDegree, 3);
  const fileCoefficients = [0, 3, 8, 15, 24][fileDegree];
  const positionBytes = version === 1 ? 6 : 9;
  const rotationBytes = version >= 3 ? 4 : 3;
  const offsets = {};
  let at = 16;
  for (const [name, size] of [['positions', positionBytes], ['alphas', 1], ['colours', 3], ['scales', 3],
    ['rotations', rotationBytes], ['sh', fileCoefficients * 3]]) {
    offsets[name] = at;
    at += count * size;
  }
  if (count === 0) throw new Error('splats: a .spz with no splats');
  if (at > data.length) throw new Error(`splats: the .spz is cut short: ${count} splats need ${at} bytes, it has ${data.length}`);

  const out = output(count, degree);
  const fixed = 1 / (1 << fraction);
  const q = [0, 0, 0, 0];
  for (let i = 0; i < count; i++) {
    // Positions, then y and z turned over.
    const position = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      if (version === 1) {
        position[a] = fromHalf(view.getUint16(offsets.positions + i * 6 + a * 2, true));
      } else {
        const o = offsets.positions + i * 9 + a * 3;
        position[a] = ((data[o] | (data[o + 1] << 8) | (data[o + 2] << 16)) << 8 >> 8) * fixed;
      }
    }
    // Rotation as x, y, z, w.
    const r = offsets.rotations + i * rotationBytes;
    if (rotationBytes === 3) {
      for (let a = 0; a < 3; a++) q[a] = data[r + a] / 127.5 - 1;
      q[3] = Math.sqrt(Math.max(0, 1 - (q[0] * q[0] + q[1] * q[1] + q[2] * q[2])));
    } else {
      // Smallest three: the largest component's index in the top two bits,
      // the others as a sign and nine bits of magnitude each, last first.
      let packed = view.getUint32(r, true);
      const largest = packed >>> 30;
      let sum = 0;
      for (let a = 3; a >= 0; a--) {
        if (a === largest) continue;
        const magnitude = packed & 511;
        const negative = (packed >>> 9) & 1;
        packed >>>= 10;
        q[a] = Math.SQRT1_2 * magnitude / 511 * (negative ? -1 : 1);
        sum += q[a] * q[a];
      }
      q[largest] = Math.sqrt(Math.max(0, 1 - sum));
    }
    const c = offsets.colours + i * 3;
    const colour = (k) => byte(0.5 + SH_C0 * ((data[c + k] / 255 - 0.5) / SPZ_COLOUR_SCALE));
    const s = offsets.scales + i * 3;
    const scale = (k) => Math.exp(data[s + k] / 16 - 10);
    // RUB to RDF, turning half round about x: a position's y and z negate,
    // and so do a rotation's y and z (Niantic's flipQ: yz, xz, xy).
    put(out, i, position[0], -position[1], -position[2],
      colour(0), colour(1), colour(2), data[offsets.alphas + i],
      scale(0), scale(1), scale(2), q[3], q[0], -q[1], -q[2]);
    const h = offsets.sh + i * fileCoefficients * 3;
    for (let k = 0; k < SH_COEFFICIENTS[degree]; k++) {
      const value = (channel) => (data[h + k * 3 + channel] - 128) / 128 * SPZ_SH_FLIP[k];
      putSH(out, i, k, value(0), value(1), value(2));
    }
  }
  return out;
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
