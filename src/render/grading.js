// Colour grading: white balance, contrast and saturation in linear light
// before the tonemap, and a 3D LUT after it.
//
//   whiteBalance  the colour temperature, in kelvin, that comes out white:
//                 3200 undoes a tungsten light's orange, 10000 an overcast
//                 blue. Adapted with the Bradford transform, from that
//                 illuminant to D65, sRGB's white.
//   contrast      about middle grey, 0.18, in stops: 1 leaves it, 1.2 pushes
//                 a stop above grey to 1.2 stops
//   saturation    0 is grey, 1 leaves it
//   lut           an Adobe .cube file (engine.loadLUT): what colourists
//                 export from Resolve, Photoshop and the rest. Applied to
//                 the display's encoded values, which is what a .cube expects.
//
// Leave any out and it does nothing.

import { createTexture } from '../rhi/texture.js';
import { halfBits } from './hdr.js';

/**
 * Parse an Adobe .cube 3D LUT: LUT_3D_SIZE N, optional DOMAIN_MIN/MAX, then
 * N^3 rows of r g b with red varying fastest. Returns { size, data } with
 * data as N^3 * 3 floats, the domain folded in: every entry is the output
 * for an input of its grid position over [0, 1].
 */
export function parseCube(text) {
  let size = 0;
  let min = [0, 0, 0];
  let max = [1, 1, 1];
  const rows = [];
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.trim();
    if (line === '' || line.startsWith('#')) continue;
    const parts = line.split(/\s+/);
    const key = parts[0].toUpperCase();
    if (key === 'TITLE') continue;
    if (key === 'LUT_1D_SIZE') throw new Error('parseCube: a 1D LUT; only 3D LUTs are supported');
    if (key === 'LUT_3D_SIZE') { size = Number(parts[1]); continue; }
    if (key === 'DOMAIN_MIN') { min = parts.slice(1, 4).map(Number); continue; }
    if (key === 'DOMAIN_MAX') { max = parts.slice(1, 4).map(Number); continue; }
    const values = parts.slice(0, 3).map(Number);
    if (values.length === 3 && values.every(Number.isFinite)) rows.push(values);
    else throw new Error(`parseCube: cannot read the line '${line}'`);
  }
  if (!(Number.isInteger(size) && size >= 2)) throw new Error(`parseCube: LUT_3D_SIZE must be 2 or more, got ${size}`);
  if (rows.length !== size ** 3) throw new Error(`parseCube: a size ${size} LUT has ${size ** 3} entries, and this has ${rows.length}`);
  if (![0, 1, 2].every((c) => max[c] > min[c])) throw new Error('parseCube: DOMAIN_MAX must be above DOMAIN_MIN');
  // A domain other than [0, 1] means the grid covers [min, max]: the shader
  // looks a value up at (value - min) / (max - min).
  return { size, data: Float32Array.from(rows.flat()), domainMin: min, domainMax: max };
}

// sRGB primaries (D65) to XYZ, and back; Bradford's cone response.
const SRGB_TO_XYZ = [
  [0.4124564, 0.3575761, 0.1804375],
  [0.2126729, 0.7151522, 0.0721750],
  [0.0193339, 0.1191920, 0.9503041],
];
const XYZ_TO_SRGB = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.9692660, 1.8760108, 0.0415560],
  [0.0556434, -0.2040259, 1.0572252],
];
const BRADFORD = [
  [0.8951, 0.2664, -0.1614],
  [-0.7502, 1.7135, 0.0367],
  [0.0389, -0.0685, 1.0296],
];
const BRADFORD_INVERSE = [
  [0.9869929, -0.1470543, 0.1599627],
  [0.4323053, 0.5183603, 0.0492912],
  [-0.0085287, 0.0400428, 0.9684867],
];

const multiply = (a, b) => a.map((row) => b[0].map((_, j) => row.reduce((sum, v, k) => sum + v * b[k][j], 0)));
const apply = (m, v) => m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);

/**
 * A blackbody's chromaticity at T kelvin: the Planckian locus as Kim et al.
 * fit it (the cubic splines behind most colour tools), 1667 K to 25000 K.
 */
export function planckianXY(T) {
  if (!(T >= 1667 && T <= 25000)) throw new Error(`whiteBalance: a colour temperature from 1667 K to 25000 K, got ${T}`);
  const t = 1e3 / T;
  const x = T <= 4000
    ? -0.2661239 * t ** 3 - 0.2343589 * t ** 2 + 0.8776956 * t + 0.179910
    : -3.0258469 * t ** 3 + 2.1070379 * t ** 2 + 0.2226347 * t + 0.240390;
  const y = T <= 2222
    ? -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    : T <= 4000
      ? -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
      : 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483;
  return [x, y];
}

/**
 * The linear-sRGB matrix that makes the light of a T kelvin blackbody
 * white: Bradford chromatic adaptation from its white to D65's. As rows.
 */
export function whiteBalanceMatrix(T) {
  const [x, y] = planckianXY(T);
  const source = [x / y, 1, (1 - x - y) / y];
  // D65, exactly as sRGB defines it: the white SRGB_TO_XYZ sends (1, 1, 1) to.
  const target = apply(SRGB_TO_XYZ, [1, 1, 1]);
  const s = apply(BRADFORD, source);
  const d = apply(BRADFORD, target);
  const scale = [[d[0] / s[0], 0, 0], [0, d[1] / s[1], 0], [0, 0, d[2] / s[2]]];
  return multiply(XYZ_TO_SRGB, multiply(BRADFORD_INVERSE, multiply(scale, multiply(BRADFORD, SRGB_TO_XYZ))));
}

/**
 * The grading uniform: the white balance rows, their w contrast, saturation
 * and the LUT's size (0 for none), then the LUT's domain. 80 bytes.
 */
export function packGrading(out, grading) {
  const rows = grading?.whiteBalance === undefined ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]] : whiteBalanceMatrix(grading.whiteBalance);
  const contrast = grading?.contrast ?? 1;
  const saturation = grading?.saturation ?? 1;
  if (!(contrast > 0 && Number.isFinite(contrast))) throw new Error(`grading: contrast must be positive, got ${contrast}`);
  if (!(saturation >= 0 && Number.isFinite(saturation))) throw new Error(`grading: saturation must be 0 or more, got ${saturation}`);
  out.set([...rows[0], contrast, ...rows[1], saturation, ...rows[2], grading?.lut?.size ?? 0], 0);
  out.set([...(grading?.lut?.domainMin ?? [0, 0, 0]), 0, ...(grading?.lut?.domainMax ?? [1, 1, 1]), 0], 12);
  return out;
}

/** WGSL for the tonemap pass. */
export const GRADING_WGSL = /* wgsl */ `
struct Grading {
  balance0  : vec4<f32>,   // white balance, row by row; w = contrast
  balance1  : vec4<f32>,   // w = saturation
  balance2  : vec4<f32>,   // w = the LUT's size, 0 for none
  domainMin : vec4<f32>,   // the LUT's input range
  domainMax : vec4<f32>,
};

/** Linear grading, before the tonemap: white balance, contrast, saturation. */
fn gradeLinear(g : Grading, c : vec3<f32>) -> vec3<f32> {
  var colour = vec3<f32>(dot(g.balance0.xyz, c), dot(g.balance1.xyz, c), dot(g.balance2.xyz, c));
  // Contrast in stops about middle grey: grey stays, a stop over grows by it.
  colour = 0.18 * pow(max(colour, vec3<f32>(0.0)) / 0.18, vec3<f32>(g.balance0.w));
  let grey = dot(colour, vec3<f32>(0.2126, 0.7152, 0.0722));
  return max(mix(vec3<f32>(grey), colour, g.balance1.w), vec3<f32>(0.0));
}

fn encodeSRGB(c : vec3<f32>) -> vec3<f32> {
  return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
}

fn decodeSRGB(c : vec3<f32>) -> vec3<f32> {
  return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}
`;

/** A parsed .cube on the GPU: a 3D texture, half floats, sampled trilinearly. */
export function uploadLUT(rhi, cube) {
  const { size, data } = cube;
  const texture = createTexture(rhi, {
    label: 'lut', size: [size, size, size], dimension: '3d', format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const halves = new Uint16Array(size ** 3 * 4);
  for (let i = 0; i < size ** 3; i++) {
    halves[i * 4] = halfBits(data[i * 3]);
    halves[i * 4 + 1] = halfBits(data[i * 3 + 1]);
    halves[i * 4 + 2] = halfBits(data[i * 3 + 2]);
    halves[i * 4 + 3] = halfBits(1);
  }
  rhi.queue.writeTexture({ texture }, halves, { bytesPerRow: size * 8, rowsPerImage: size }, [size, size, size]);
  return { ...cube, texture, view: texture.createView({ dimension: '3d' }) };
}
