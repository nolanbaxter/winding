// Drawing the 3D view at fewer pixels than the canvas has, and bringing it
// back up: renderer.resolution.
//
// Most of a frame's GPU time is spent per pixel, so a view drawn at 0.75 of
// the canvas's width and height shades 44% fewer of them. What decides
// whether that is worth it is how the picture comes back up. A plain
// bilinear stretch blurs every edge by the scale; this is NVIDIA Image
// Scaling 1.0.3 (NVScaler, MIT -- the notice is at the end of this file),
// ported to WGSL:
//
//   - a 6-tap filter, one of 64 phases chosen by where the output pixel falls
//     between source pixels, so it stays sharp at any scale between 0.5 and 1;
//   - an edge map from the source's luma, which blends in filters running
//     along edges at 0, 45, 90 and 135 degrees, so a diagonal is interpolated
//     along itself rather than stair-stepped across;
//   - and an unsharp mask on luma, held back in bright areas and where the
//     two sides of a pixel differ sharply in contrast, so it sharpens texture
//     without ringing at hard edges.
//
// It filters luma only and moves each pixel's bilinear colour by the
// difference, as NVIDIA's does: colour detail is what eyes resolve least.
//
// Where it sits: after the tonemap and FXAA, on the display's encoded values,
// which is where NIS is tuned to work -- it reads the view's texture through
// its plain (not sRGB) view, so it sees exactly the bytes the screen would.
// Before the HUD, which is drawn at the canvas's own resolution and stays
// crisp. NIS is a compute shader, and the canvas is not a storage texture, so
// it writes an intermediate a copy pass then puts on the screen.
//
// One workgroup makes a block of 32x24 output pixels. It first loads the
// block's source luma, with three pixels of border, into workgroup memory;
// then derives the edge map there; then each of the 256 threads filters
// three pixels of the block from those, so every source pixel is read from
// the texture once per block rather than 36 times per output pixel.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { clampSampler } from '../rhi/texture.js';

/** The lowest resolution NIS is made for: past half, its filters run out of phases. */
export const MIN_RESOLUTION = 0.5;
/** NIS's sharpness, 0 to 1: its own default. */
const SHARPNESS = 0.5;
const BLOCK_WIDTH = 32;
const BLOCK_HEIGHT = 24;
/** scale, srcNorm, srcSize, dstSize */
const PARAMS_BYTES = 32;
const OUTPUT_FORMAT = 'rgba8unorm';

/** NVScalerUpdateConfig's SDR settings for a sharpness, as WGSL constants. */
function settings(sharpness) {
  const slider = sharpness - 0.5;
  const maxScale = slider >= 0 ? 1.25 : 1.75;
  const minScale = slider >= 0 ? 1.25 : 1.0;
  const limitScale = slider >= 0 ? 1.25 : 1.0;
  const strengthMin = Math.max(0, 0.4 + slider * minScale * 1.2);
  const strengthMax = 1.6 + slider * maxScale * 1.8;
  const limitMin = Math.max(0.1, 0.14 + slider * limitScale * 0.32);
  const limitMax = 0.5 + slider * limitScale * 0.6;
  const minContrast = 2.0;
  const maxContrast = 10.0;
  const sharpStartY = 0.45;
  const sharpEndY = 0.9;
  const values = {
    DETECT_RATIO: 2 * 1127 / 1024,
    DETECT_THRES: 64 / 1024,
    MIN_CONTRAST_RATIO: minContrast,
    RATIO_NORM: 1 / (maxContrast - minContrast),
    CONTRAST_BOOST: 1,
    EPS: 1 / 255,
    SHARP_START_Y: sharpStartY,
    SHARP_SCALE_Y: 1 / (sharpEndY - sharpStartY),
    SHARP_STRENGTH_MIN: strengthMin,
    SHARP_STRENGTH_SCALE: strengthMax - strengthMin,
    SHARP_LIMIT_MIN: limitMin,
    SHARP_LIMIT_SCALE: limitMax - limitMin,
  };
  return Object.entries(values).map(([name, v]) => `const ${name} = ${float(v)};`).join('\n');
}

const float = (v) => (Number.isInteger(v) ? `${v}.0` : String(v));

function scalerShader() {
  return /* wgsl */ `
${settings(SHARPNESS)}

struct Params {
  scale   : vec2<f32>,   // source size over output size
  srcNorm : vec2<f32>,   // 1 / source size
  srcSize : vec2<i32>,
  dstSize : vec2<i32>,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var source : texture_2d<f32>;
@group(0) @binding(2) var samp : sampler;
@group(0) @binding(3) var output : texture_storage_2d<${OUTPUT_FORMAT}, write>;

const BLOCK_W = ${BLOCK_WIDTH};
const BLOCK_H = ${BLOCK_HEIGHT};
const GROUP = 256;
const PHASES = 64;
// The tile: the block's source luma, plus the 6-tap support's border. Sized
// for a scale of 1, where it is largest.
const TILE_PITCH = BLOCK_W + 6;
const TILE_SIZE = TILE_PITCH * (BLOCK_H + 6);
// The edge map: the block's source pixels, plus one of border.
const EDGE_PITCH = BLOCK_W + 2;
const EDGE_SIZE = EDGE_PITCH * (BLOCK_H + 2);

// The filter banks, 64 phases of 6 taps: the scaler, then its unsharp mask.
// A buffer, not WGSL constants: a constant array indexed at run time may be
// built afresh, all of it, wherever it is read.
@group(0) @binding(4) var<storage, read> banks : array<f32, ${SCALE.length + USM.length}>;

var<workgroup> tileY : array<f32, TILE_SIZE>;
var<workgroup> coefScale : array<f32, ${SCALE.length}>;
var<workgroup> coefUsm : array<f32, ${USM.length}>;
// Four weights a pixel, as half floats: four bytes of each eight would put
// the workgroup over the 16 KB WebGPU guarantees.
var<workgroup> edges : array<vec2<u32>, EDGE_SIZE>;

fn luma(c : vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn lumaAt(at : vec2<i32>) -> f32 {
  return luma(textureLoad(source, clamp(at, vec2<i32>(0), params.srcSize - 1), 0).rgb);
}

/**
 * Which way the 3x3 at (i, j) runs: weights for the filters along 0, 90, 45
 * and 135 degrees, all zero where it is flat or runs no one way.
 */
fn edgeMap(luma4 : array<array<f32, 4>, 4>, i : i32, j : i32) -> vec4<f32> {
  var p = luma4;
  let g0 = abs(p[i][j] + p[i][j + 1] + p[i][j + 2] - p[i + 2][j] - p[i + 2][j + 1] - p[i + 2][j + 2]);
  let g45 = abs(p[i + 1][j] + p[i][j] + p[i][j + 1] - p[i + 2][j + 1] - p[i + 2][j + 2] - p[i + 1][j + 2]);
  let g90 = abs(p[i][j] + p[i + 1][j] + p[i + 2][j] - p[i][j + 2] - p[i + 1][j + 2] - p[i + 2][j + 2]);
  let g135 = abs(p[i + 1][j] + p[i + 2][j] + p[i + 2][j + 1] - p[i][j + 1] - p[i][j + 2] - p[i + 1][j + 2]);

  let max0_90 = max(g0, g90);
  let min0_90 = min(g0, g90);
  let max45_135 = max(g45, g135);
  let min45_135 = min(g45, g135);
  if (max0_90 + max45_135 == 0.0) { return vec4<f32>(0.0); }

  let e0_90 = min(max0_90 / (max0_90 + max45_135), 1.0);
  let e45_135 = 1.0 - e0_90;
  let c0_90 = max0_90 > min0_90 * DETECT_RATIO && max0_90 > DETECT_THRES && max0_90 > min45_135;
  let c45_135 = max45_135 > min45_135 * DETECT_RATIO && max45_135 > DETECT_THRES && max45_135 > min0_90;
  let is0 = max0_90 == g0;
  let is45 = max45_135 == g45;
  let both = c0_90 && c45_135;
  let f0_90 = select(1.0, e0_90, both);
  let f45_135 = select(1.0, e45_135, both);
  return vec4<f32>(
    select(0.0, f0_90, c0_90 && is0),
    select(0.0, f0_90, c0_90 && !is0),
    select(0.0, f45_135, c45_135 && is45),
    select(0.0, f45_135, c45_135 && !is45),
  );
}

/** Ringing control: 1 where both sides of the tap have like contrast, toward 0 where one side is much flatter. */
fn lti(p : array<f32, 6>, phase : i32) -> f32 {
  let near = phase <= PHASES / 2;
  let aMin = min(min(p[1], p[2]), select(p[3], p[0], near));
  let aMax = max(max(p[1], p[2]), select(p[3], p[0], near));
  let bMin = min(min(p[3], p[4]), select(p[5], p[2], near));
  let bMax = max(max(p[3], p[4]), select(p[5], p[2], near));
  let a = aMax - aMin;
  let b = bMax - bMin;
  let ratio = max(a, b) / (min(a, b) + EPS);
  return (1.0 - saturate((ratio - MIN_CONTRAST_RATIO) * RATIO_NORM)) * CONTRAST_BOOST;
}

/** Six taps along one direction, filtered and sharpened. */
fn poly6(taps : array<f32, 6>, phase : i32) -> f32 {
  var p = taps;
  var y = 0.0;
  var usm = 0.0;
  for (var i = 0; i < 6; i++) {
    y += coefScale[phase * 6 + i] * p[i];
    usm += coefUsm[phase * 6 + i] * p[i];
  }
  // Sharpened less, and limited more, the brighter it is.
  let yScale = 1.0 - saturate((y - SHARP_START_Y) * SHARP_SCALE_Y);
  usm *= yScale * SHARP_STRENGTH_SCALE + SHARP_STRENGTH_MIN;
  let limit = (yScale * SHARP_LIMIT_SCALE + SHARP_LIMIT_MIN) * y;
  usm = clamp(usm, -limit, limit);
  return y + usm * lti(p, phase);
}

/** The plain separable 6x6 filter, unsharpened. */
fn filterNormal(support : array<array<f32, 6>, 6>, phaseX : i32, phaseY : i32) -> f32 {
  var p = support;
  var h = 0.0;
  for (var j = 0; j < 6; j++) {
    var v = 0.0;
    for (var i = 0; i < 6; i++) { v += p[i][j] * coefScale[phaseY * 6 + i]; }
    h += v * coefScale[phaseX * 6 + j];
  }
  return h;
}

/** Six taps from seven along a diagonal, starting one later past its middle. */
fn diagonal(t : array<f32, 7>, s : f32) -> f32 {
  var q = t;
  let shift = select(0, 1, s >= 1.0);
  let phase = i32((s - f32(shift)) * f32(PHASES));
  var taps : array<f32, 6>;
  for (var i = 0; i < 6; i++) { taps[i] = q[i + shift]; }
  return poly6(taps, phase);
}

/** The filters along edges, each by its weight. */
fn directional(support : array<array<f32, 6>, 6>, fx : f32, fy : f32, phaseX : i32, phaseY : i32, w : vec4<f32>) -> f32 {
  var p = support;
  var f = 0.0;
  var taps : array<f32, 6>;
  if (w.x > 0.0) {   // 0 degrees
    for (var i = 0; i < 6; i++) { taps[i] = mix(p[i][2], p[i][3], fx); }
    f += poly6(taps, phaseY) * w.x;
  }
  if (w.y > 0.0) {   // 90 degrees
    for (var i = 0; i < 6; i++) { taps[i] = mix(p[2][i], p[3][i], fy); }
    f += poly6(taps, phaseX) * w.y;
  }
  if (w.z > 0.0) {   // 45 degrees
    var b = 0.5 + 0.5 * (fx - fy);
    var t : array<f32, 7>;
    t[1] = mix(p[2][1], p[1][2], b);
    t[3] = mix(p[3][2], p[2][3], b);
    t[5] = mix(p[4][3], p[3][4], b);
    b -= 0.5;
    let up = b >= 0.0;
    t[0] = mix(p[1][1], select(p[2][0], p[0][2], up), abs(b));
    t[2] = mix(p[2][2], select(p[3][1], p[1][3], up), abs(b));
    t[4] = mix(p[3][3], select(p[4][2], p[2][4], up), abs(b));
    t[6] = mix(p[4][4], select(p[5][3], p[3][5], up), abs(b));
    f += diagonal(t, fx + fy) * w.z;
  }
  if (w.w > 0.0) {   // 135 degrees
    var b = 0.5 * (fx + fy);
    var t : array<f32, 7>;
    t[1] = mix(p[3][1], p[4][2], b);
    t[3] = mix(p[2][2], p[3][3], b);
    t[5] = mix(p[1][3], p[2][4], b);
    b -= 0.5;
    let up = b >= 0.0;
    t[0] = mix(p[4][1], select(p[3][0], p[5][2], up), abs(b));
    t[2] = mix(p[3][2], select(p[2][1], p[4][3], up), abs(b));
    t[4] = mix(p[2][3], select(p[1][2], p[3][4], up), abs(b));
    t[6] = mix(p[1][4], select(p[0][3], p[2][5], up), abs(b));
    f += diagonal(t, 1.0 + fx - fy) * w.w;
  }
  return f;
}

fn packEdge(e : vec4<f32>) -> vec2<u32> {
  return vec2<u32>(pack2x16float(e.xy), pack2x16float(e.zw));
}

fn unpackEdge(e : vec2<u32>) -> vec4<f32> {
  return vec4<f32>(unpack2x16float(e.x), unpack2x16float(e.y));
}

@compute @workgroup_size(GROUP)
fn scale(@builtin(workgroup_id) block : vec3<u32>, @builtin(local_invocation_index) thread : u32) {
  let dstBlock = vec2<i32>(BLOCK_W * i32(block.x), BLOCK_H * i32(block.y));
  // The source pixels the block's output falls between.
  let srcStart = vec2<i32>(floor((vec2<f32>(dstBlock) + 0.5) * params.scale - 0.5));
  let srcEnd = vec2<i32>(ceil((vec2<f32>(dstBlock + vec2<i32>(BLOCK_W, BLOCK_H)) + 0.5) * params.scale - 0.5));
  // Plus the support, rounded up to even: it is loaded in 2x2s.
  var tile = srcEnd - srcStart + 5;
  tile += tile & vec2<i32>(1);
  let tilePixels = tile.x * tile.y;
  let edge = tile - 4;
  let edgePixels = edge.x * edge.y;
  let i0 = i32(thread) * 2;

  // Source luma, from two pixels up and left of the block's first.
  for (var i = i0; i < (tilePixels >> 1); i += GROUP * 2) {
    let at = vec2<i32>(i % tile.x, (i / tile.x) * 2);
    let origin = srcStart + at - 2;
    let index = at.y * TILE_PITCH + at.x;
    tileY[index] = lumaAt(origin);
    tileY[index + 1] = lumaAt(origin + vec2<i32>(1, 0));
    tileY[index + TILE_PITCH] = lumaAt(origin + vec2<i32>(0, 1));
    tileY[index + TILE_PITCH + 1] = lumaAt(origin + vec2<i32>(1, 1));
  }
  workgroupBarrier();

  // The edge map, in 2x2s from the 4x4 of luma around them.
  for (var i = i0; i < (edgePixels >> 1); i += GROUP * 2) {
    let at = vec2<i32>(i % edge.x, (i / edge.x) * 2);
    let index = at.y * EDGE_PITCH + at.x;
    let corner = (at.y + 1) * TILE_PITCH + at.x + 1;
    var p : array<array<f32, 4>, 4>;
    for (var r = 0; r < 4; r++) {
      for (var c = 0; c < 4; c++) { p[r][c] = tileY[corner + r * TILE_PITCH + c]; }
    }
    edges[index] = packEdge(edgeMap(p, 0, 0));
    edges[index + 1] = packEdge(edgeMap(p, 0, 1));
    edges[index + EDGE_PITCH] = packEdge(edgeMap(p, 1, 0));
    edges[index + EDGE_PITCH + 1] = packEdge(edgeMap(p, 1, 1));
  }
  for (var i = i32(thread); i < ${SCALE.length}; i += GROUP) {
    coefScale[i] = banks[i];
    coefUsm[i] = banks[i + ${SCALE.length}];
  }
  workgroupBarrier();

  // Three output pixels a thread, a column apart by eight rows.
  let pos = vec2<i32>(i32(thread % u32(BLOCK_W)), i32(thread / u32(BLOCK_W)));
  let dstX = dstBlock.x + pos.x;
  let srcX = (0.5 + f32(dstX)) * params.scale.x - 0.5;
  let px = i32(floor(srcX)) - srcStart.x;
  let fx = srcX - floor(srcX);
  let phaseX = i32(fx * f32(PHASES));
  for (var k = 0; k < BLOCK_W * BLOCK_H / GROUP; k++) {
    let dstY = dstBlock.y + pos.y + k * (GROUP / BLOCK_W);
    // Not stored: a store out of bounds may land on some other pixel.
    if (dstX >= params.dstSize.x || dstY >= params.dstSize.y) { continue; }
    let srcY = (0.5 + f32(dstY)) * params.scale.y - 0.5;
    let py = i32(floor(srcY)) - srcStart.y;
    let fy = srcY - floor(srcY);
    let phaseY = i32(fy * f32(PHASES));

    // The edge weights here, interpolated between the four around it.
    let e = py * EDGE_PITCH + px;
    let w = mix(
      mix(unpackEdge(edges[e]), unpackEdge(edges[e + 1]), fx),
      mix(unpackEdge(edges[e + EDGE_PITCH]), unpackEdge(edges[e + EDGE_PITCH + 1]), fx),
      fy,
    );

    // The 6x6 support, from two pixels up and left of this one.
    var p : array<array<f32, 6>, 6>;
    let t = py * TILE_PITCH + px;
    for (var r = 0; r < 6; r++) {
      for (var c = 0; c < 6; c++) { p[r][c] = tileY[t + r * TILE_PITCH + c]; }
    }
    let y = filterNormal(p, phaseX, phaseY) * (1.0 - w.x - w.y - w.z - w.w)
      + directional(p, fx, fy, phaseX, phaseY, w);

    // The bilinear colour, moved to the filtered luma.
    let colour = textureSampleLevel(source, samp, (vec2<f32>(srcX, srcY) + 0.5) * params.srcNorm, 0.0).rgb;
    textureStore(output, vec2<i32>(dstX, dstY), vec4<f32>(colour + (y - luma(colour)), 1.0));
  }
}
`;
}

// Onto the canvas: the scaler's output as it is, or -- while the scaler's
// pipeline builds -- the view stretched bilinearly.
const COPY_SHADER = /* wgsl */ `
@group(0) @binding(0) var source : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0)       uv       : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32) -> VertexOut {
  var out : VertexOut;
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  out.uv = vec2<f32>(x, y);
  out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

@fragment
fn fsCopy(v : VertexOut) -> @location(0) vec4<f32> {
  return textureLoad(source, vec2<i32>(v.position.xy), 0);
}

@fragment
fn fsStretch(v : VertexOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(source, samp, v.uv, 0.0);
}
`;

export class Upscaler {
  static async create(rhi, pipelines) {
    const upscaler = new Upscaler(rhi, pipelines);
    await upscaler._init();
    return upscaler;
  }

  constructor(rhi, pipelines) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    this.sampler = clampSampler(rhi);
    this.params = createBuffer(rhi, { label: 'upscale-params', size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._paramsData = new ArrayBuffer(PARAMS_BYTES);
    this._f32 = new Float32Array(this._paramsData);
    this._i32 = new Int32Array(this._paramsData);
    /** Whether the last frame was scaled by NIS, rather than stretched while it built. */
    this.scaled = false;
    this._groups = new Map();
    this._scaleExecute = (pass) => this._encodeScale(pass);
    this._copyExecute = (pass) => this._encodeCopy(pass);
  }

  async _init() {
    const device = this.rhi.device;
    const shader = await compileShader(device, COPY_SHADER, 'upscale-copy.wgsl');
    this.copyLayout = device.createBindGroupLayout({
      label: 'upscale-copy',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const common = {
      layout: createPipelineLayout(device, { 0: this.copyLayout }, 'upscale-copy'),
      shader, primitive: { topology: 'triangle-list', cullMode: 'none' }, depth: null,
      targets: [{ format: this.rhi.surfaceFormat }],
    };
    this.copyDescriptor = { ...common, label: 'upscale-copy', fragmentEntry: 'fsCopy' };
    this.stretchDescriptor = { ...common, label: 'upscale-stretch', fragmentEntry: 'fsStretch' };
    await this.pipelines.warm([this.copyDescriptor, this.stretchDescriptor]);
    this.copyPipeline = this.pipelines.get(this.copyDescriptor);
    this.stretchPipeline = this.pipelines.get(this.stretchDescriptor);
  }

  /**
   * The scaler's pipeline, started the first time a frame is scaled: an
   * engine that never lowers its resolution never compiles it. Frames are
   * stretched until it is ready -- or for good, if it fails, which is said
   * once rather than every frame.
   */
  _scalerReady() {
    this._building ??= (async () => {
      const device = this.rhi.device;
      const shader = await compileShader(device, scalerShader(), 'nis.wgsl');
      // Destroyed while compiling: nothing is made that nothing would free.
      if (this._destroyed) return;
      this.scaleLayout = device.createBindGroupLayout({
        label: 'nis',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: {} },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: {} },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: OUTPUT_FORMAT } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
      });
      const layout = createPipelineLayout(device, { 0: this.scaleLayout }, 'nis');
      const banks = Float32Array.from([...SCALE, ...USM]);
      this.banks = createBuffer(this.rhi, { label: 'nis-banks', size: banks.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.rhi.queue.writeBuffer(this.banks, 0, banks);
      const pipeline = await device.createComputePipelineAsync({
        label: 'nis', layout: layout.gpu, compute: { module: shader.module, entryPoint: 'scale' },
      });
      if (this._destroyed) {
        this.banks.destroy();
        return;
      }
      this.scalePipeline = pipeline;
    })().catch((error) => {
      this.failed = true;
      console.error(error);
    });
    return this._building;
  }

  /** Whether the scaler is still building: a frame drawn now is stretched, and a later one will not be. */
  get pending() {
    return this._building !== undefined && this.scalePipeline === undefined && this.failed !== true && !this._destroyed;
  }

  /**
   * The view, `source` -- a target drawn at the lower resolution, whose
   * graph resource is `sourceResource` -- brought up to `surface`, the
   * canvas's plain view, `width` x `height`.
   */
  addPasses(graph, { source, sourceResource, surface, width, height }) {
    if (this.scalePipeline === undefined) this._scalerReady();
    this.scaled = this.scalePipeline !== undefined;
    this._source = source;
    this._graph = graph;
    this._width = width;
    this._height = height;
    if (!this.scaled) {
      this._copyFrom = null;
      graph.addPass({ name: 'upscale', reads: [sourceResource], color: [{ resource: surface, clear: BLACK }], execute: this._copyExecute });
      return;
    }
    const f = this._f32, i = this._i32;
    f[0] = source.width / width;
    f[1] = source.height / height;
    f[2] = 1 / source.width;
    f[3] = 1 / source.height;
    i[4] = source.width;
    i[5] = source.height;
    i[6] = width;
    i[7] = height;
    this.rhi.queue.writeBuffer(this.params, 0, this._paramsData);
    this._output = graph.createTexture('upscaled', {
      width, height, format: OUTPUT_FORMAT, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    graph.addPass({ name: 'nis', type: 'compute', reads: [sourceResource], writes: [this._output], execute: this._scaleExecute });
    this._copyFrom = this._output;
    graph.addPass({ name: 'upscale', reads: [this._output], color: [{ resource: surface, clear: BLACK }], execute: this._copyExecute });
  }

  _encodeScale(pass) {
    const output = this._graph.viewOf(this._output);
    pass.setPipeline(this.scalePipeline);
    pass.setBindGroup(0, this._group('nis', this.scaleLayout, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: this._source.plainView },
      { binding: 2, resource: this.sampler },
      { binding: 3, resource: output },
      { binding: 4, resource: { buffer: this.banks } },
    ], this._source.plainView, output));
    pass.dispatchWorkgroups(Math.ceil(this._width / BLOCK_WIDTH), Math.ceil(this._height / BLOCK_HEIGHT));
  }

  _encodeCopy(pass) {
    const view = this._copyFrom === null ? this._source.plainView : this._graph.viewOf(this._copyFrom);
    pass.setPipeline(this._copyFrom === null ? this.stretchPipeline : this.copyPipeline);
    pass.setBindGroup(0, this._group('copy', this.copyLayout, [
      { binding: 0, resource: view },
      { binding: 1, resource: this.sampler },
    ], view, null));
    pass.draw(3);
  }

  /**
   * A bind group over these views, kept while frames keep asking for it.
   * Only the last frame's are kept: there is one canvas, and a resize or a
   * new resolution replaces the views for good.
   */
  _group(kind, layout, entries, a, b) {
    const last = this._groups.get(kind);
    if (last !== undefined && last.a === a && last.b === b) return last.group;
    const group = this.rhi.device.createBindGroup({ label: `upscale-${kind}`, layout, entries });
    this._groups.set(kind, { group, a, b });
    return group;
  }

  destroy() {
    this._destroyed = true;
    this.params.destroy();
    this.banks?.destroy();
    this._groups.clear();
  }
}

const BLACK = { r: 0, g: 0, b: 0, a: 1 };

// NVIDIA Image Scaling's filter banks (NIS_Config.h, coef_scale and coef_usm),
// the six taps of each of 64 phases that the shader uses of eight.
const SCALE = [
  0, 0, 1, 0, 0, 0,
  0.0029, -0.0127, 1, 0.0132, -0.0034, 0,
  0.0063, -0.0249, 0.9985, 0.0269, -0.0068, 0,
  0.0088, -0.0361, 0.9956, 0.0415, -0.0103, 0.0005,
  0.0117, -0.0474, 0.9932, 0.0562, -0.0142, 0.0005,
  0.0142, -0.0576, 0.9897, 0.0713, -0.0181, 0.0005,
  0.0166, -0.0674, 0.9844, 0.0874, -0.022, 0.001,
  0.0186, -0.0762, 0.9785, 0.104, -0.0264, 0.0015,
  0.0205, -0.085, 0.9727, 0.1206, -0.0308, 0.002,
  0.0225, -0.0928, 0.9648, 0.1382, -0.0352, 0.0024,
  0.0239, -0.1006, 0.9575, 0.1558, -0.0396, 0.0029,
  0.0254, -0.1074, 0.9487, 0.1738, -0.0439, 0.0034,
  0.0264, -0.1138, 0.939, 0.1929, -0.0488, 0.0044,
  0.0278, -0.1191, 0.9282, 0.2119, -0.0537, 0.0049,
  0.0288, -0.1245, 0.917, 0.231, -0.0581, 0.0059,
  0.0293, -0.1294, 0.9058, 0.251, -0.063, 0.0063,
  0.0303, -0.1333, 0.8926, 0.271, -0.0679, 0.0073,
  0.0308, -0.1367, 0.8789, 0.2915, -0.0728, 0.0083,
  0.0308, -0.1401, 0.8657, 0.312, -0.0776, 0.0093,
  0.0313, -0.1426, 0.8506, 0.333, -0.0825, 0.0103,
  0.0313, -0.1445, 0.8354, 0.354, -0.0874, 0.0112,
  0.0313, -0.146, 0.8193, 0.3755, -0.0923, 0.0122,
  0.0313, -0.147, 0.8022, 0.3965, -0.0967, 0.0137,
  0.0308, -0.1479, 0.7856, 0.4185, -0.1016, 0.0146,
  0.0303, -0.1479, 0.7681, 0.4399, -0.106, 0.0156,
  0.0298, -0.1479, 0.7505, 0.4614, -0.1104, 0.0166,
  0.0293, -0.147, 0.7314, 0.4829, -0.1147, 0.0181,
  0.0288, -0.146, 0.7119, 0.5049, -0.1187, 0.019,
  0.0278, -0.1445, 0.6929, 0.5264, -0.1226, 0.02,
  0.0273, -0.1431, 0.6724, 0.5479, -0.126, 0.0215,
  0.0264, -0.1411, 0.6528, 0.5693, -0.1299, 0.0225,
  0.0254, -0.1387, 0.6323, 0.5903, -0.1328, 0.0234,
  0.0244, -0.1357, 0.6113, 0.6113, -0.1357, 0.0244,
  0.0234, -0.1328, 0.5903, 0.6323, -0.1387, 0.0254,
  0.0225, -0.1299, 0.5693, 0.6528, -0.1411, 0.0264,
  0.0215, -0.126, 0.5479, 0.6724, -0.1431, 0.0273,
  0.02, -0.1226, 0.5264, 0.6929, -0.1445, 0.0278,
  0.019, -0.1187, 0.5049, 0.7119, -0.146, 0.0288,
  0.0181, -0.1147, 0.4829, 0.7314, -0.147, 0.0293,
  0.0166, -0.1104, 0.4614, 0.7505, -0.1479, 0.0298,
  0.0156, -0.106, 0.4399, 0.7681, -0.1479, 0.0303,
  0.0146, -0.1016, 0.4185, 0.7856, -0.1479, 0.0308,
  0.0137, -0.0967, 0.3965, 0.8022, -0.147, 0.0313,
  0.0122, -0.0923, 0.3755, 0.8193, -0.146, 0.0313,
  0.0112, -0.0874, 0.354, 0.8354, -0.1445, 0.0313,
  0.0103, -0.0825, 0.333, 0.8506, -0.1426, 0.0313,
  0.0093, -0.0776, 0.312, 0.8657, -0.1401, 0.0308,
  0.0083, -0.0728, 0.2915, 0.8789, -0.1367, 0.0308,
  0.0073, -0.0679, 0.271, 0.8926, -0.1333, 0.0303,
  0.0063, -0.063, 0.251, 0.9058, -0.1294, 0.0293,
  0.0059, -0.0581, 0.231, 0.917, -0.1245, 0.0288,
  0.0049, -0.0537, 0.2119, 0.9282, -0.1191, 0.0278,
  0.0044, -0.0488, 0.1929, 0.939, -0.1138, 0.0264,
  0.0034, -0.0439, 0.1738, 0.9487, -0.1074, 0.0254,
  0.0029, -0.0396, 0.1558, 0.9575, -0.1006, 0.0239,
  0.0024, -0.0352, 0.1382, 0.9648, -0.0928, 0.0225,
  0.002, -0.0308, 0.1206, 0.9727, -0.085, 0.0205,
  0.0015, -0.0264, 0.104, 0.9785, -0.0762, 0.0186,
  0.001, -0.022, 0.0874, 0.9844, -0.0674, 0.0166,
  0.0005, -0.0181, 0.0713, 0.9897, -0.0576, 0.0142,
  0.0005, -0.0142, 0.0562, 0.9932, -0.0474, 0.0117,
  0.0005, -0.0103, 0.0415, 0.9956, -0.0361, 0.0088,
  0, -0.0068, 0.0269, 0.9985, -0.0249, 0.0063,
  0, -0.0034, 0.0132, 1, -0.0127, 0.0029,
];
const USM = [
  0, -0.6001, 1.2002, -0.6001, 0, 0,
  0.0029, -0.6084, 1.1987, -0.5903, -0.0029, 0,
  0.0049, -0.6147, 1.1958, -0.5791, -0.0068, 0.0005,
  0.0073, -0.6196, 1.189, -0.5659, -0.0103, 0,
  0.0093, -0.6235, 1.1802, -0.5513, -0.0151, 0,
  0.0112, -0.6265, 1.1699, -0.5352, -0.0195, 0.0005,
  0.0122, -0.627, 1.1582, -0.5181, -0.0259, 0.0005,
  0.0142, -0.6284, 1.1455, -0.5005, -0.0317, 0.0005,
  0.0156, -0.6265, 1.1274, -0.479, -0.0386, 0.0005,
  0.0166, -0.6235, 1.1089, -0.457, -0.0454, 0.001,
  0.0176, -0.6187, 1.0879, -0.4346, -0.0532, 0.001,
  0.0181, -0.6138, 1.0659, -0.4102, -0.0615, 0.0015,
  0.019, -0.6069, 1.0405, -0.3843, -0.0698, 0.0015,
  0.0195, -0.6006, 1.0161, -0.3574, -0.0796, 0.002,
  0.02, -0.5928, 0.9893, -0.3286, -0.0898, 0.0024,
  0.02, -0.582, 0.958, -0.2988, -0.1001, 0.0029,
  0.02, -0.5728, 0.9292, -0.269, -0.1104, 0.0034,
  0.02, -0.562, 0.8975, -0.2368, -0.1226, 0.0039,
  0.0205, -0.5498, 0.8643, -0.2046, -0.1343, 0.0044,
  0.02, -0.5371, 0.8301, -0.1709, -0.1465, 0.0049,
  0.0195, -0.5239, 0.7944, -0.1367, -0.1587, 0.0054,
  0.0195, -0.5107, 0.7598, -0.1021, -0.1724, 0.0059,
  0.019, -0.4966, 0.7231, -0.0649, -0.1865, 0.0063,
  0.0186, -0.4819, 0.6846, -0.0288, -0.1997, 0.0068,
  0.0186, -0.4668, 0.646, 0.0093, -0.2144, 0.0073,
  0.0176, -0.4507, 0.6055, 0.0479, -0.229, 0.0083,
  0.0171, -0.437, 0.5693, 0.0859, -0.2446, 0.0088,
  0.0161, -0.4199, 0.5283, 0.1255, -0.2598, 0.0098,
  0.0161, -0.4048, 0.4883, 0.1655, -0.2754, 0.0103,
  0.0151, -0.3887, 0.4497, 0.2041, -0.291, 0.0107,
  0.0142, -0.3711, 0.4072, 0.2446, -0.3066, 0.0117,
  0.0137, -0.3555, 0.3672, 0.2852, -0.3228, 0.0122,
  0.0132, -0.3394, 0.3262, 0.3262, -0.3394, 0.0132,
  0.0122, -0.3228, 0.2852, 0.3672, -0.3555, 0.0137,
  0.0117, -0.3066, 0.2446, 0.4072, -0.3711, 0.0142,
  0.0107, -0.291, 0.2041, 0.4497, -0.3887, 0.0151,
  0.0103, -0.2754, 0.1655, 0.4883, -0.4048, 0.0161,
  0.0098, -0.2598, 0.1255, 0.5283, -0.4199, 0.0161,
  0.0088, -0.2446, 0.0859, 0.5693, -0.437, 0.0171,
  0.0083, -0.229, 0.0479, 0.6055, -0.4507, 0.0176,
  0.0073, -0.2144, 0.0093, 0.646, -0.4668, 0.0186,
  0.0068, -0.1997, -0.0288, 0.6846, -0.4819, 0.0186,
  0.0063, -0.1865, -0.0649, 0.7231, -0.4966, 0.019,
  0.0059, -0.1724, -0.1021, 0.7598, -0.5107, 0.0195,
  0.0054, -0.1587, -0.1367, 0.7944, -0.5239, 0.0195,
  0.0049, -0.1465, -0.1709, 0.8301, -0.5371, 0.02,
  0.0044, -0.1343, -0.2046, 0.8643, -0.5498, 0.0205,
  0.0039, -0.1226, -0.2368, 0.8975, -0.562, 0.02,
  0.0034, -0.1104, -0.269, 0.9292, -0.5728, 0.02,
  0.0029, -0.1001, -0.2988, 0.958, -0.582, 0.02,
  0.0024, -0.0898, -0.3286, 0.9893, -0.5928, 0.02,
  0.002, -0.0796, -0.3574, 1.0161, -0.6006, 0.0195,
  0.0015, -0.0698, -0.3843, 1.0405, -0.6069, 0.019,
  0.0015, -0.0615, -0.4102, 1.0659, -0.6138, 0.0181,
  0.001, -0.0532, -0.4346, 1.0879, -0.6187, 0.0176,
  0.001, -0.0454, -0.457, 1.1089, -0.6235, 0.0166,
  0.0005, -0.0386, -0.479, 1.1274, -0.6265, 0.0156,
  0.0005, -0.0317, -0.5005, 1.1455, -0.6284, 0.0142,
  0.0005, -0.0259, -0.5181, 1.1582, -0.627, 0.0122,
  0.0005, -0.0195, -0.5352, 1.1699, -0.6265, 0.0112,
  0, -0.0151, -0.5513, 1.1802, -0.6235, 0.0093,
  0, -0.0103, -0.5659, 1.189, -0.6196, 0.0073,
  0.0005, -0.0068, -0.5791, 1.1958, -0.6147, 0.0049,
  0, -0.0029, -0.5903, 1.1987, -0.6084, 0.0029,
];

// The NVScaler port above follows NIS_Scaler.h from the NVIDIA Image Scaling
// SDK 1.0.3 (github.com/NVIDIAGameWorks/NVIDIAImageScaling), under this licence:
//
// The MIT License(MIT)
//
// Copyright(c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files(the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
