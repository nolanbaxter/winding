// Depth of field, from a lens: how much each pixel blurs is what a camera of
// this field of view would blur it, focused at `focusDistance` and stopped
// down to `fStop`.
//
// A thin lens of focal length f and aperture A = f / N, focused at S, images
// a point at depth d as a disc of diameter
//   c = A f |d - S| / (d (S - f))
// on the sensor -- the circle of confusion. The focal length is the one
// the camera's vertical field of view implies on a sensor of `sensorHeight`
// (24 mm, the full frame that f-stops and focal lengths are quoted against),
// and c over the sensor's height is the disc's share of the screen's.
//
// Three passes, all full-screen: the colour and each pixel's signed disc at
// half resolution; a gather there, over a spiral as wide as the largest disc
// on screen, where a sample counts only if its own disc reaches the pixel --
// so a blurred foreground spills over what is behind it, and a blurred
// background does not spill over a sharp foreground; and a composite that
// mixes the sharp picture with the blurred one by each pixel's disc.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { clampSampler } from '../rhi/texture.js';

const BLACK = { r: 0, g: 0, b: 0, a: 0 };

/** The gather's samples. The cost ceiling: past this a wide disc is sampled more sparsely. */
export const DOF_SAMPLES = 64;

/**
 * The lens, as the shader needs it: scale and bias so that the signed disc
 * in pixels is scale * (1 - S / d) -- positive behind the focus, negative in
 * front -- and the largest disc the far distance reaches, which sizes the
 * gather. Returns { scale, largest }.
 */
export function lensCoefficients({ focusDistance, fStop, sensorHeight = 0.024 }, fovY, screenHeight) {
  if (!(focusDistance > 0 && Number.isFinite(focusDistance))) throw new Error(`dof: focusDistance must be positive, got ${focusDistance}`);
  if (!(fStop > 0 && Number.isFinite(fStop))) throw new Error(`dof: fStop must be positive, got ${fStop}`);
  if (!(sensorHeight > 0 && Number.isFinite(sensorHeight))) throw new Error(`dof: sensorHeight must be positive, got ${sensorHeight}`);
  const focal = sensorHeight / (2 * Math.tan(fovY / 2));
  if (!(focusDistance > focal)) throw new Error(`dof: focusDistance must be past the lens's focal length, ${focal.toFixed(4)} m`);
  const aperture = focal / fStop;
  // c = A f (d - S) / (d (S - f)) = [A f / (S - f)] (1 - S / d), on the sensor.
  const onSensor = aperture * focal / (focusDistance - focal);
  const scale = onSensor / sensorHeight * screenHeight;
  return { scale, largest: Math.abs(scale) };
}

const SHADER = /* wgsl */ `
struct Lens {
  scale     : f32,   // disc in pixels = scale * (1 - focus / depth)
  focus     : f32,
  near      : f32,   // the camera's near plane: depth = near / stored depth, reverse-Z
  largest   : f32,   // the widest disc, full-resolution pixels
  texel     : vec2<f32>,   // 1 / full-resolution size
  pad       : vec2<f32>,
};

@group(0) @binding(0) var<uniform> lens : Lens;
@group(0) @binding(1) var scene  : texture_2d<f32>;
@group(0) @binding(2) var depth  : texture_depth_2d;
@group(0) @binding(3) var halfRes: texture_2d<f32>;
@group(0) @binding(4) var blurred: texture_2d<f32>;
@group(0) @binding(5) var samp   : sampler;

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

/** The signed disc, in full-resolution pixels, at a stored depth. */
fn disc(stored : f32) -> f32 {
  // Reverse-Z with no far plane: 0 is infinitely far, where the disc is scale.
  let d = lens.near / max(stored, 1e-9);
  return clamp(lens.scale * (1.0 - lens.focus / d), -lens.largest, lens.largest);
}

/** Half resolution: the colour, and the disc of the nearest of the four depths. */
@fragment
fn fsPrepare(v : VertexOut) -> @location(0) vec4<f32> {
  let full = vec2<i32>(v.position.xy) * 2;
  var nearest = 0.0;
  for (var k = 0; k < 4; k = k + 1) {
    nearest = max(nearest, textureLoad(depth, full + vec2<i32>(k & 1, k >> 1), 0));
  }
  let colour = textureSampleLevel(scene, samp, v.uv, 0.0).rgb;
  return vec4<f32>(colour, disc(nearest));
}

/**
 * The gather, at half resolution: a golden-angle spiral out to the widest
 * disc. A sample counts where its own disc reaches this pixel; one behind
 * this pixel's depth reaches only as far as this pixel's disc does, so a
 * blurred background cannot spread over a sharp thing in front of it.
 */
@fragment
fn fsGather(v : VertexOut) -> @location(0) vec4<f32> {
  // Unfiltered: a filtered read between a sharp texel and a blurred one would
  // blend their discs too, and a grey half-way sample would count as reaching.
  let size = vec2<i32>(textureDimensions(halfRes));
  let here = vec2<i32>(v.position.xy);
  let centre = textureLoad(halfRes, here, 0);
  // The disc is a DIAMETER in full-resolution pixels: a radius a quarter of it here.
  let reach = lens.largest * 0.25;              // in half-resolution pixels
  var sum = vec4<f32>(centre.rgb, 1.0);
  for (var i = 1; i < ${DOF_SAMPLES}; i = i + 1) {
    let r = reach * sqrt(f32(i) / ${DOF_SAMPLES - 1}.0);
    let angle = f32(i) * 2.39996323;
    let offset = vec2<f32>(cos(angle), sin(angle)) * r;
    let step = round(offset);
    let s = textureLoad(halfRes, clamp(here + vec2<i32>(step), vec2<i32>(0), size - 1), 0);
    // Radii in half-resolution pixels; behind the centre, capped by its disc.
    let own = abs(s.a) * 0.25;
    let spread = select(own, min(own, abs(centre.a) * 0.25), s.a > centre.a);
    // Covered where the texel it was read from lies inside the disc, give or
    // take half a texel: measured to that texel, not to the spiral point.
    let weight = clamp(spread - length(step) + 0.5, 0.0, 1.0);
    sum = sum + vec4<f32>(s.rgb, 1.0) * weight;
  }
  return vec4<f32>(sum.rgb / sum.a, centre.a);
}

/**
 * Sharp where the disc is under a pixel, blurred as it grows past one. The
 * blurred picture is read from the four half-resolution texels around the
 * pixel, each weighted by how near its disc is to this pixel's own as well as
 * by distance: a half texel straddling a sharp edge holds some of both sides,
 * and plain filtering spread it over the blurred background as a halo.
 */
@fragment
fn fsComposite(v : VertexOut) -> @location(0) vec4<f32> {
  let sharp = textureSampleLevel(scene, samp, v.uv, 0.0);
  let at = disc(textureLoad(depth, vec2<i32>(v.position.xy), 0));
  let size = vec2<i32>(textureDimensions(blurred));
  let position = v.position.xy * 0.5 - 0.5;
  let base = vec2<i32>(floor(position));
  let f = fract(position);
  var soft = vec3<f32>(0.0);
  var total = 0.0;
  var widest = 0.0;
  for (var k = 0; k < 4; k = k + 1) {
    let o = vec2<i32>(k & 1, k >> 1);
    let texel = textureLoad(blurred, clamp(base + o, vec2<i32>(0), size - 1), 0);
    let bilinear = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    // Squared: a neighbour on the far side of a sharp edge differs by the whole
    // disc and must count for nothing; one down a sloping floor differs by
    // under a pixel and should count nearly in full.
    let difference = texel.a - at;
    let w = bilinear / (1.0 + difference * difference);
    soft = soft + texel.rgb * w;
    total = total + w;
    // Only a nearer neighbour's blur spills over this pixel; a farther one's stays behind it.
    if (texel.a < at) { widest = max(widest, abs(texel.a) * bilinear); }
  }
  soft = soft / max(total, 1e-6);
  let t = clamp(max(abs(at), widest) - 1.0, 0.0, 1.0);
  return vec4<f32>(mix(sharp.rgb, soft, t), sharp.a);
}
`;

export class DepthOfField {
  static async create(rhi, pipelines, format) {
    const shader = await compileShader(rhi.device, SHADER, 'dof.wgsl');
    const layout = rhi.device.createBindGroupLayout({
      label: 'dof',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const common = {
      layout: createPipelineLayout(rhi.device, { 0: layout }, 'dof'),
      shader, targets: [{ format }], primitive: { topology: 'triangle-list', cullMode: 'none' }, depth: null,
    };
    const descriptors = {
      prepare: { ...common, label: 'dof-prepare', fragmentEntry: 'fsPrepare' },
      gather: { ...common, label: 'dof-gather', fragmentEntry: 'fsGather' },
      composite: { ...common, label: 'dof-composite', fragmentEntry: 'fsComposite' },
    };
    await pipelines.warm(Object.values(descriptors));
    return new DepthOfField(rhi, pipelines, descriptors, layout, format);
  }

  constructor(rhi, pipelines, descriptors, layout, format) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    this._descriptors = descriptors;
    this._layout = layout;
    this._format = format;
    this._data = new Float32Array(8);
    this._buffer = createBuffer(rhi, { label: 'dof', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._groups = new Map();
    this._frame = 0;
    this._executes = ['prepare', 'gather', 'composite'].map((name) => (pass) => this._draw(pass, name));
  }

  /**
   * Blur this frame's scene colour as the lens would, and return the result
   * for the post stack to take instead. `dof` is { focusDistance, fStop,
   * sensorHeight }; an orthographic camera has no lens, and is left sharp.
   */
  addPasses(graph, { sceneColor, depth, camera, width, height, dof }) {
    if (camera.orthographic) return sceneColor;
    const { scale, largest } = lensCoefficients(dof, camera.fovY, height);
    this._data.set([scale, dof.focusDistance, camera.near, largest, 1 / width, 1 / height, 0, 0]);
    this.rhi.queue.writeBuffer(this._buffer, 0, this._data);
    this._frame++;
    const halfSize = { width: Math.max(1, width >> 1), height: Math.max(1, height >> 1), format: this._format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING };
    const half = graph.createTexture('dof-half', halfSize);
    const blurred = graph.createTexture('dof-blurred', halfSize);
    const out = graph.createTexture('dof-out', { ...halfSize, width, height });
    this._resources = { sceneColor, depth, half, blurred, graph };
    graph.addPass({ name: 'dof:prepare', reads: [sceneColor, depth], color: [{ resource: half, clear: BLACK }], execute: this._executes[0] });
    graph.addPass({ name: 'dof:gather', reads: [half], color: [{ resource: blurred, clear: BLACK }], execute: this._executes[1] });
    graph.addPass({ name: 'dof:composite', reads: [sceneColor, depth, blurred], color: [{ resource: out, clear: BLACK }], execute: this._executes[2] });
    return out;
  }

  _draw(pass, name) {
    const { sceneColor, depth, half, blurred, graph } = this._resources;
    // A pass may not sample what it writes: each binds only what it reads,
    // the rest standing in with a view it does not touch.
    const scene = graph.viewOf(sceneColor);
    const views = {
      prepare: [scene, graph.viewOf(depth), scene, scene],
      gather: [scene, graph.viewOf(depth), graph.viewOf(half), scene],
      composite: [scene, graph.viewOf(depth), scene, graph.viewOf(blurred)],
    }[name];
    const key = `${name}:${views.map(viewKey).join(':')}`;
    let entry = this._groups.get(key);
    if (!entry) {
      entry = {
        group: this.rhi.device.createBindGroup({
          label: `dof:${name}`,
          layout: this._layout,
          entries: [
            { binding: 0, resource: { buffer: this._buffer } },
            ...views.map((resource, k) => ({ binding: k + 1, resource })),
            { binding: 5, resource: clampSampler(this.rhi) },
          ],
        }),
        frame: 0,
      };
      this._groups.set(key, entry);
    }
    entry.frame = this._frame;
    for (const [k, e] of this._groups) if (e.frame < this._frame - 2) this._groups.delete(k);
    pass.setPipeline(this.pipelines.get(this._descriptors[name]));
    pass.setBindGroup(0, entry.group);
    pass.draw(3);
  }

  destroy() {
    this._buffer.destroy();
  }
}

let nextView = 1;
const viewIds = new WeakMap();
function viewKey(view) {
  let id = viewIds.get(view);
  if (id === undefined) { id = nextView++; viewIds.set(view, id); }
  return id;
}
