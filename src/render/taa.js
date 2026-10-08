// Temporal antialiasing: many samples a pixel, spread over frames.
//
// A frame samples each pixel at one point, so an edge is in or out of it, and
// a thin or shiny thing that moves a little flips from one to the other --
// it crawls. Here the camera's projection is moved a fraction of a pixel each
// frame (renderer.js, _jitterCamera), a different fraction from a Halton
// sequence, so over sixteen frames each pixel has been sampled at sixteen
// points inside it; and each frame is blended with what the frames before it
// made, the history, so the picture is their average.
//
// The history is where this pixel's surface was last frame, not where this
// pixel was: worked out from the depth buffer and last frame's camera, as far
// as anything that does not move by itself goes -- the camera moving, the
// world still. A thing that moves by itself has no motion of its own here;
// what keeps it from smearing is that the history is clipped to the colours
// around the pixel in this frame (Salvi's variance clipping, in YCoCg), so a
// colour that is no longer there cannot stay.
//
// One pass, full resolution, in HDR before depth of field and bloom:
//   - the nearest depth of the 3x3 around the pixel, so an edge reprojects
//     with the thing in front of it, not the background beside it
//   - the history at that place, by a 5-tap Catmull-Rom (Jimenez), which
//     keeps it sharp; bilinear softened it a little more every frame
//   - clipped, then blended: a tenth of this frame, each side weighted by
//     1 / (1 + luma) (Karis), so one bright pixel does not flicker
// It writes the new history, which is also what the rest of the frame reads.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { HDR_FORMAT } from './post.js';

/** Inverse view-projection, this frame's and last frame's view-projection, size, reset, feedback. */
const PARAMS_BYTES = 224;
/** Frames in the jitter's cycle: the Halton (2, 3) points. */
export const TAA_SAMPLES = 16;
/**
 * How much sharper a mip the material textures take under TAA. Its blend of
 * jittered frames is a filter a pixel wide, which a texture's own mip already
 * is: without this, surfaces came out softer than a supersampled frame.
 *
 * Measured on Sponza at 720p (Iris Xe), against 16x supersampling: -0.5 gets
 * nearly all that -1 does (edges 4.9 against 4.7, surfaces 2.5 against 2.4)
 * for a third of its cost (+1.2 ms a frame against +3.3 -- a sharper mip is
 * more texels for the forward pass to read). Sharpening the frame afterwards
 * instead (AMD's CAS) was cheaper and worse: it sharpened the noise too.
 */
export const TAA_MIP_BIAS = -0.5;
/** Frames of a still view after which the picture has settled: run() may rest. */
export const TAA_SETTLE = 24;

const SHADER = /* wgsl */ `
struct Params {
  inverseViewProjection  : mat4x4<f32>,   // this frame's, jittered: which surface the pixel shows
  viewProjection         : mat4x4<f32>,   // this frame's, without its jitter
  previousViewProjection : mat4x4<f32>,   // last frame's, without its jitter
  size                   : vec2<f32>,
  reset                  : f32,           // 1: no history to use
  feedback               : f32,           // the share of this frame
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var current : texture_2d<f32>;
@group(0) @binding(2) var depthMap : texture_depth_2d;
@group(0) @binding(3) var history : texture_2d<f32>;
@group(0) @binding(4) var blend : sampler;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32) -> VertexOut {
  var out : VertexOut;
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

fn toYCoCg(c : vec3<f32>) -> vec3<f32> {
  return vec3<f32>(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

fn fromYCoCg(c : vec3<f32>) -> vec3<f32> {
  return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

fn luma(c : vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/** The history at uv, by Catmull-Rom from five bilinear taps. */
fn historyAt(uv : vec2<f32>) -> vec3<f32> {
  let at = uv * params.size;
  let centre = floor(at - 0.5) + 0.5;
  let f = at - centre;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let p0 = (centre - 1.0) / params.size;
  let p3 = (centre + 2.0) / params.size;
  let p12 = (centre + w2 / w12) / params.size;
  var sum = textureSampleLevel(history, blend, vec2<f32>(p12.x, p0.y), 0.0).rgb * (w12.x * w0.y);
  sum += textureSampleLevel(history, blend, vec2<f32>(p0.x, p12.y), 0.0).rgb * (w0.x * w12.y);
  sum += textureSampleLevel(history, blend, p12, 0.0).rgb * (w12.x * w12.y);
  sum += textureSampleLevel(history, blend, vec2<f32>(p3.x, p12.y), 0.0).rgb * (w3.x * w12.y);
  sum += textureSampleLevel(history, blend, vec2<f32>(p12.x, p3.y), 0.0).rgb * (w12.x * w3.y);
  let weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return max(sum / weight, vec3<f32>(0.0));
}

/** h, pulled toward the box's centre until it is inside it. */
fn clipToBox(low : vec3<f32>, high : vec3<f32>, h : vec3<f32>) -> vec3<f32> {
  let centre = 0.5 * (low + high);
  let extent = 0.5 * (high - low) + vec3<f32>(1e-5);
  let away = h - centre;
  let units = abs(away / extent);
  let most = max(units.x, max(units.y, units.z));
  return select(h, centre + away / most, most > 1.0);
}

@fragment
fn fs(v : VertexOut) -> @location(0) vec4<f32> {
  let p = vec2<i32>(v.position.xy);
  let last = vec2<i32>(params.size) - vec2<i32>(1);
  let here = textureLoad(current, p, 0);

  // The neighbourhood: this frame's colours around the pixel, as a mean and
  // spread, and the nearest depth among them (reverse-Z: the greatest).
  var m1 = vec3<f32>(0.0);
  var m2 = vec3<f32>(0.0);
  var nearest = vec2<i32>(p);
  var nearestDepth = -1.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = clamp(p + vec2<i32>(x, y), vec2<i32>(0), last);
      let c = toYCoCg(textureLoad(current, q, 0).rgb);
      m1 += c;
      m2 += c * c;
      let d = textureLoad(depthMap, q, 0);
      if (d > nearestDepth) { nearestDepth = d; nearest = q; }
    }
  }
  if (params.reset > 0.5) { return here; }

  // How far that surface moved on screen since last frame: back to the world
  // by this frame's jittered camera, then forward by this frame's and last
  // frame's cameras, both without their jitter -- with it, a still view
  // fetched its history a fraction of a pixel off every frame, and the
  // resampling blurred it a little more each time. Homogeneous all the way,
  // so the sky -- depth 0, a point at infinity -- moves by the camera's turn.
  let at = (vec2<f32>(nearest) + 0.5) / params.size;
  let world = params.inverseViewProjection * vec4<f32>(at.x * 2.0 - 1.0, 1.0 - at.y * 2.0, nearestDepth, 1.0);
  let now = params.viewProjection * world;
  let before = params.previousViewProjection * world;
  if (before.w <= 0.0 || now.w <= 0.0) { return here; }
  let was = vec2<f32>(before.x / before.w * 0.5 + 0.5, 0.5 - before.y / before.w * 0.5);
  let is = vec2<f32>(now.x / now.w * 0.5 + 0.5, 0.5 - now.y / now.w * 0.5);
  let uv = v.position.xy / params.size + (was - is);
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return here; }

  let mean = m1 / 9.0;
  let spread = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
  let held = fromYCoCg(clipToBox(mean - 1.25 * spread, mean + 1.25 * spread, toYCoCg(historyAt(uv))));

  // Moving, trust the history less: every frame it is resampled at a
  // fraction of a pixel, which softens it a little, and less of it softens
  // less. Still, the full tenth: the most samples, the finest edge.
  let speed = length((was - is) * params.size);
  let share = mix(params.feedback, 0.25, clamp(speed / 2.0, 0.0, 1.0));
  let a = share / (1.0 + luma(here.rgb));
  let b = (1.0 - share) / (1.0 + luma(held));
  return vec4<f32>(max((here.rgb * a + held * b) / (a + b), vec3<f32>(0.0)), here.a);
}
`;

/** Point i of the Halton sequence in base b, 0 to 1. */
function halton(i, b) {
  let f = 1;
  let r = 0;
  for (let n = i; n > 0; n = Math.floor(n / b)) {
    f /= b;
    r += f * (n % b);
  }
  return r;
}

/** Frame n's jitter, in pixels from the pixel's centre: -0.5 to 0.5 each way. */
export function jitterOf(n) {
  const i = (n % TAA_SAMPLES) + 1;
  return [halton(i, 2) - 0.5, halton(i, 3) - 0.5];
}

export class TemporalAA {
  static async create(rhi, pipelines) {
    const device = rhi.device;
    const shader = await compileShader(device, SHADER, 'taa.wgsl');
    const fragment = GPUShaderStage.FRAGMENT;
    const layout = device.createBindGroupLayout({
      label: 'taa',
      entries: [
        { binding: 0, visibility: fragment, buffer: { type: 'uniform' } },
        { binding: 1, visibility: fragment, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: fragment, texture: { sampleType: 'depth' } },
        { binding: 3, visibility: fragment, texture: {} },
        { binding: 4, visibility: fragment, sampler: {} },
      ],
    });
    const descriptor = {
      label: 'taa',
      layout: createPipelineLayout(device, { 0: layout }, 'taa'),
      shader, fragmentEntry: 'fs', targets: [{ format: HDR_FORMAT }],
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depth: null,
    };
    await pipelines.warm([descriptor]);
    return new TemporalAA(rhi, layout, pipelines.get(descriptor));
  }

  constructor(rhi, layout, pipeline) {
    this.rhi = rhi;
    this.layout = layout;
    this.pipeline = pipeline;
    this.buffer = createBuffer(rhi, { label: 'taa-params', size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.data = new Float32Array(PARAMS_BYTES / 4);
    this.sampler = rhi.device.createSampler({ label: 'taa', magFilter: 'linear', minFilter: 'linear' });
    /** The two histories, read and written by turns. */
    this.textures = [];
    this._turn = 0;
    this._width = 0;
    this._height = 0;
  }

  /**
   * The resolve, into a new history, which it returns for the rest of the
   * frame to read. `reset` uses no history: the first frame, or one after
   * the view jumped.
   */
  addPass(graph, { color, depth, width, height, inverseViewProjection, viewProjection, previousViewProjection, reset }) {
    if (width !== this._width || height !== this._height) {
      for (const t of this.textures) t.destroy();
      this.textures = [0, 1].map((k) => this.rhi.device.createTexture({
        label: `taa-history${k}`, size: [width, height], format: HDR_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }));
      this.views = this.textures.map((t) => t.createView());
      this._width = width;
      this._height = height;
      reset = true;
    }
    const d = this.data;
    d.set(inverseViewProjection, 0);
    d.set(viewProjection, 16);
    d.set(previousViewProjection, 32);
    d[48] = width;
    d[49] = height;
    d[50] = reset ? 1 : 0;
    // A tenth of this frame. A running average for a still view (a share of
    // 1 / n) was measured too: further from supersampling on Sponza, edges
    // 7.0 against 6.2.
    d[51] = 0.1;
    this.rhi.queue.writeBuffer(this.buffer, 0, d);

    const before = this._turn;
    this._turn ^= 1;
    const previous = graph.importTexture(`taa-history${before}`, this.views[before]);
    const next = graph.importTexture(`taa-history${this._turn}`, this.views[this._turn]);
    graph.addPass({
      name: 'taa',
      reads: [color, depth, previous],
      // Every pixel is written; the clear only tells the graph it starts this frame.
      color: [{ resource: next, clear: { r: 0, g: 0, b: 0, a: 0 } }],
      execute: (pass) => {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.rhi.device.createBindGroup({
          label: 'taa',
          layout: this.layout,
          entries: [
            { binding: 0, resource: { buffer: this.buffer } },
            { binding: 1, resource: graph.viewOf(color) },
            { binding: 2, resource: graph.viewOf(depth) },
            { binding: 3, resource: this.views[before] },
            { binding: 4, resource: this.sampler },
          ],
        }));
        pass.draw(3);
      },
    });
    return next;
  }

  /** Free the histories, while TAA is off; the next addPass makes them again. */
  release() {
    for (const t of this.textures) t.destroy();
    this.textures = [];
    this._width = 0;
    this._height = 0;
  }

  destroy() {
    for (const t of this.textures) t.destroy();
    this.buffer.destroy();
  }
}
