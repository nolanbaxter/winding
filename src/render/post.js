// Post-processing: bloom and tonemapping.
//
// What makes the renderer HDR end to end. A shader that tonemaps its own output
// squashes every pixel into [0,1] before anything downstream can look at it, and
// bloom needs to know which pixels were at 60x white, not that they all clipped
// to 1.
//
// So: geometry renders linear HDR into an rgba16float target, bloom reads real
// intensities off it, and ONE pass at the end tonemaps the combination into the
// sRGB swap chain.
//
// Bloom is the progressive chain (Jimenez / Call of Duty), not a Gaussian:
//
//   downsample  half res, then half again, six times, 13 taps each
//   upsample    back up the chain, 3x3 tent, ADDED to the level below
//
// The upsample adds into the same textures the downsample wrote, which is worth
// noticing: those passes declare no clear value, so the render graph derives
// `load` for them because something wrote the texture earlier in the frame. Get
// that wrong by hand and the bloom is simply the top mip with everything below
// it thrown away.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { clampSampler } from '../rhi/texture.js';

export const HDR_FORMAT = 'rgba16float';

/** texelSize(8) + threshold, knee, radius, firstPass, strength, exposure */
const PARAMS_BYTES = 32;
const MAX_BLOOM_LEVELS = 6;

export const POST_SHADER = /* wgsl */ `
struct Params {
  texelSize : vec2<f32>,   // 1 / source size
  threshold : f32,
  knee      : f32,
  radius    : f32,
  firstPass : f32,
  strength  : f32,
  exposure  : f32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var source  : texture_2d<f32>;
@group(0) @binding(2) var bloomTex: texture_2d<f32>;
@group(0) @binding(3) var samp    : sampler;

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

fn luminance(c : vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/**
 * Soft-knee threshold. A hard cutoff makes bloom pop on and off as a highlight
 * crosses it; the knee ramps contribution in over a band instead.
 */
fn prefilter(colour : vec3<f32>) -> vec3<f32> {
  let brightness = max(colour.r, max(colour.g, colour.b));
  let knee = params.threshold * params.knee;
  let soft = clamp(brightness - params.threshold + knee, 0.0, 2.0 * knee);
  let contribution = max(
    soft * soft / (4.0 * knee + 1e-4),
    brightness - params.threshold,
  ) / max(brightness, 1e-4);
  return colour * contribution;
}

/**
 * Karis average: weight each tap by 1/(1+luma) before averaging.
 *
 * Without it a single very bright pixel -- a specular glint, a stray sample --
 * survives every downsample and reappears as a large flickering blob. Only the
 * first downsample needs it; after that the fireflies are already averaged out.
 */
fn karisWeight(c : vec3<f32>) -> f32 {
  return 1.0 / (1.0 + luminance(c));
}

fn tap(uv : vec2<f32>, offset : vec2<f32>) -> vec3<f32> {
  return textureSampleLevel(source, samp, uv + offset * params.texelSize, 0.0).rgb;
}

/**
 * 13-tap downsample. The offset pattern is a 3x3 grid at two texels plus four
 * taps at one texel, which together approximate a wide filter with no aliasing
 * as the chain halves -- a straight 2x2 box picks up shimmering from the
 * frequencies it cannot represent.
 */
@fragment
fn fsDownsample(v : VertexOut) -> @location(0) vec4<f32> {
  let a = tap(v.uv, vec2<f32>(-2.0, -2.0));
  let b = tap(v.uv, vec2<f32>( 0.0, -2.0));
  let c = tap(v.uv, vec2<f32>( 2.0, -2.0));
  let d = tap(v.uv, vec2<f32>(-2.0,  0.0));
  let e = tap(v.uv, vec2<f32>( 0.0,  0.0));
  let f = tap(v.uv, vec2<f32>( 2.0,  0.0));
  let g = tap(v.uv, vec2<f32>(-2.0,  2.0));
  let h = tap(v.uv, vec2<f32>( 0.0,  2.0));
  let i = tap(v.uv, vec2<f32>( 2.0,  2.0));
  let j = tap(v.uv, vec2<f32>(-1.0, -1.0));
  let k = tap(v.uv, vec2<f32>( 1.0, -1.0));
  let l = tap(v.uv, vec2<f32>(-1.0,  1.0));
  let m = tap(v.uv, vec2<f32>( 1.0,  1.0));

  var result : vec3<f32>;
  if (params.firstPass > 0.5) {
    // Group the taps into five boxes, Karis-average each, then combine. Doing
    // it per box rather than per tap is what actually kills the firefly.
    let box0 = (j + k + l + m) * 0.25;
    let box1 = (a + b + d + e) * 0.25;
    let box2 = (b + c + e + f) * 0.25;
    let box3 = (d + e + g + h) * 0.25;
    let box4 = (e + f + h + i) * 0.25;

    let w0 = karisWeight(box0) * 0.5;
    let w1 = karisWeight(box1) * 0.125;
    let w2 = karisWeight(box2) * 0.125;
    let w3 = karisWeight(box3) * 0.125;
    let w4 = karisWeight(box4) * 0.125;

    result = (box0 * w0 + box1 * w1 + box2 * w2 + box3 * w3 + box4 * w4)
           / max(w0 + w1 + w2 + w3 + w4, 1e-4);
    result = prefilter(result);
  } else {
    result = e * 0.125
           + (a + c + g + i) * 0.03125
           + (b + d + f + h) * 0.0625
           + (j + k + l + m) * 0.125;
  }

  return vec4<f32>(result, 1.0);
}

/**
 * 3x3 tent upsample of the smaller level, added to this level's downsample.
 *
 * The addition happens HERE rather than through an additive blend into the
 * downsample texture, and that is a graph constraint rather than a style
 * choice. Blending in place would mean writing a texture that the next
 * downsample already read, and without resource versioning the graph cannot
 * order a write-after-read -- it reports a cycle, correctly. Reading both
 * inputs and writing a third texture has no hazard at all.
 */
@fragment
fn fsUpsample(v : VertexOut) -> @location(0) vec4<f32> {
  let r = params.radius;
  let blurred =
      tap(v.uv, vec2<f32>(-r,  r)) * 1.0 + tap(v.uv, vec2<f32>(0.0,  r)) * 2.0 + tap(v.uv, vec2<f32>(r,  r)) * 1.0
    + tap(v.uv, vec2<f32>(-r, 0.0)) * 2.0 + tap(v.uv, vec2<f32>(0.0, 0.0)) * 4.0 + tap(v.uv, vec2<f32>(r, 0.0)) * 2.0
    + tap(v.uv, vec2<f32>(-r, -r)) * 1.0 + tap(v.uv, vec2<f32>(0.0, -r)) * 2.0 + tap(v.uv, vec2<f32>(r, -r)) * 1.0;

  let thisLevel = textureSampleLevel(bloomTex, samp, v.uv, 0.0).rgb;
  return vec4<f32>(blurred / 16.0 + thisLevel, 1.0);
}

/** Narkowicz's ACES fit. The one place the image leaves HDR. */
fn tonemapACES(x : vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fsTonemap(v : VertexOut) -> @location(0) vec4<f32> {
  let scene = textureSampleLevel(source, samp, v.uv, 0.0).rgb;
  let bloom = textureSampleLevel(bloomTex, samp, v.uv, 0.0).rgb;

  // Bloom is MIXED in rather than added, so raising the strength does not also
  // raise overall brightness -- it moves energy from the source into the halo,
  // which is what a real lens does.
  let combined = mix(scene, bloom, clamp(params.strength, 0.0, 1.0));

  // Exposure applies before the tonemap curve; applying it after would just
  // scale an already-compressed image.
  return vec4<f32>(tonemapACES(combined * params.exposure), 1.0);
}
`;

export class PostStack {
  static async create(rhi, pipelines, options = {}) {
    const post = new PostStack(rhi, options);
    await post._init(pipelines);
    return post;
  }

  constructor(rhi, {
    threshold = 1.2,
    knee = 0.6,
    /** Tent radius in source texels. Larger is a wider, softer halo. */
    filterRadius = 1.0,
    /** 0 is no bloom, 1 is entirely bloom. */
    strength = 0.06,
    levels = 5,
  } = {}) {
    this.rhi = rhi;
    this.threshold = threshold;
    this.knee = knee;
    this.filterRadius = filterRadius;
    this.strength = strength;
    this.requestedLevels = Math.min(levels, MAX_BLOOM_LEVELS);

    this.sampler = clampSampler(rhi);
    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;

    // One slot per pass: downsamples, upsamples, and the tonemap.
    const slots = MAX_BLOOM_LEVELS * 2 + 2;
    this.paramsStaging = new ArrayBuffer(this.alignment * slots);
    this.paramsBuffer = rhi.device.createBuffer({
      label: 'post-params',
      size: this.paramsStaging.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Bind groups are keyed by the VIEWS they reference. Graph transients can
    // be handed a different physical texture when the aliasing assignment
    // changes, so caching by view rather than by pass index is what keeps this
    // correct without rebuilding every frame.
    this._bindGroups = new Map();
    this._frame = 0;
    this._slot = 0;
    this.levels = 0;

    this._downExecutors = [];
    this._upExecutors = [];
    for (let i = 0; i < MAX_BLOOM_LEVELS; i++) {
      this._downExecutors.push((pass) => this._draw(pass, this.downPipeline, i));
      this._upExecutors.push((pass) => this._draw(pass, this.upPipeline, MAX_BLOOM_LEVELS + i));
    }
    this._tonemapExecute = (pass) => this._draw(pass, this.tonemapPipeline, MAX_BLOOM_LEVELS * 2);
  }

  async _init(pipelines) {
    const rhi = this.rhi;
    const shader = await compileShader(rhi.device, POST_SHADER, 'post.wgsl');

    this.layout = rhi.device.createBindGroupLayout({
      label: 'post',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_BYTES },
        },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const layout = createPipelineLayout(rhi.device, { 0: this.layout }, 'post');

    const common = { layout, shader, primitive: { topology: 'triangle-list', cullMode: 'none' }, depth: null };

    this.downDescriptor = {
      ...common, label: 'bloom-down', fragmentEntry: 'fsDownsample',
      targets: [{ format: HDR_FORMAT }],
    };
    // No blend state: the shader adds the two inputs and writes a third
    // texture, which is what keeps the graph free of a write-after-read.
    this.upDescriptor = {
      ...common, label: 'bloom-up', fragmentEntry: 'fsUpsample',
      targets: [{ format: HDR_FORMAT }],
    };
    this.tonemapDescriptor = {
      ...common, label: 'tonemap', fragmentEntry: 'fsTonemap',
      targets: [{ format: rhi.viewFormat }],
    };

    await pipelines.warm([this.downDescriptor, this.upDescriptor, this.tonemapDescriptor]);
    this._pipelines = pipelines;
    this.downPipeline = pipelines.get(this.downDescriptor);
    this.upPipeline = pipelines.get(this.upDescriptor);
    this.tonemapPipeline = pipelines.get(this.tonemapDescriptor);
  }

  /** How many halvings the current resolution supports, capped by the option. */
  levelCountFor(width, height) {
    const possible = Math.floor(Math.log2(Math.max(1, Math.min(width, height)))) - 2;
    return Math.max(1, Math.min(this.requestedLevels, possible));
  }

  _writeParams(slot, texelWidth, texelHeight, { firstPass = 0, radius = 0, strength = 0, exposure = 1 } = {}) {
    const offset = slot * this.alignment;
    const f32 = new Float32Array(this.paramsStaging, offset, PARAMS_BYTES / 4);
    f32[0] = texelWidth;
    f32[1] = texelHeight;
    f32[2] = this.threshold;
    f32[3] = this.knee;
    f32[4] = radius;
    f32[5] = firstPass;
    f32[6] = strength;
    f32[7] = exposure;
    return offset;
  }

  /**
   * Declare the whole chain.
   *
   * `sceneColor` is the HDR target the forward pass wrote; `surface` is the
   * swap chain. Everything between is a graph transient.
   */
  addPasses(graph, { sceneColor, surface, width, height, exposure }) {
    this._frame++;
    this._evictBindGroups();
    const levels = this.levelCountFor(width, height);
    this.levels = levels;
    this._offsets = [];

    // --- allocate the chain -------------------------------------------------
    const chain = [];
    let w = width;
    let h = height;
    for (let i = 0; i < levels; i++) {
      w = Math.max(1, w >> 1);
      h = Math.max(1, h >> 1);
      chain.push({
        resource: graph.createTexture(`bloom${i}`, {
          width: w, height: h, format: HDR_FORMAT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        }),
        width: w,
        height: h,
      });
    }
    this.chain = chain;

    // --- downsample ---------------------------------------------------------
    for (let i = 0; i < levels; i++) {
      const sourceResource = i === 0 ? sceneColor : chain[i - 1].resource;
      const sourceWidth = i === 0 ? width : chain[i - 1].width;
      const sourceHeight = i === 0 ? height : chain[i - 1].height;

      this._offsets[i] = this._writeParams(i, 1 / sourceWidth, 1 / sourceHeight, {
        firstPass: i === 0 ? 1 : 0,
      });

      graph.addPass({
        name: `bloom-down:${i}`,
        reads: [sourceResource],
        color: [{ resource: chain[i].resource, clear: { r: 0, g: 0, b: 0, a: 1 } }],
        execute: this._downExecutors[i],
      });
      this._sourceFor(i, sourceResource);
    }

    // --- upsample -----------------------------------------------------------
    // Walks back up the chain. Each pass reads the smaller level and this
    // level's downsample, and writes a NEW texture -- see fsUpsample for why
    // it cannot accumulate in place.
    //
    // The smallest level needs no upsample pass: it is already its own result.
    let smaller = chain[levels - 1].resource;
    let smallerWidth = chain[levels - 1].width;
    let smallerHeight = chain[levels - 1].height;

    for (let i = levels - 2; i >= 0; i--) {
      const slot = MAX_BLOOM_LEVELS + i;
      this._offsets[slot] = this._writeParams(
        slot, 1 / smallerWidth, 1 / smallerHeight, { radius: this.filterRadius },
      );

      const target = graph.createTexture(`bloom-up${i}`, {
        width: chain[i].width, height: chain[i].height, format: HDR_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });

      graph.addPass({
        name: `bloom-up:${i}`,
        reads: [smaller, chain[i].resource],
        color: [{ resource: target, clear: { r: 0, g: 0, b: 0, a: 1 } }],
        execute: this._upExecutors[i],
      });
      this._sourceFor(slot, smaller, chain[i].resource);

      smaller = target;
      smallerWidth = chain[i].width;
      smallerHeight = chain[i].height;
    }

    // --- tonemap ------------------------------------------------------------
    const bloomResult = smaller;
    const tonemapSlot = MAX_BLOOM_LEVELS * 2;
    this._offsets[tonemapSlot] = this._writeParams(tonemapSlot, 1 / width, 1 / height, {
      strength: this.strength, exposure,
    });

    graph.addPass({
      name: 'tonemap',
      reads: [sceneColor, bloomResult],
      color: [{ resource: surface, clear: { r: 0, g: 0, b: 0, a: 1 } }],
      execute: this._tonemapExecute,
    });
    this._sourceFor(tonemapSlot, sceneColor, bloomResult);

    this.rhi.queue.writeBuffer(this.paramsBuffer, 0, this.paramsStaging);
    this._graph = graph;
    return levels * 2;
  }

  /** Remember which resources each slot samples; resolved to views at execute. */
  _sourceFor(slot, primary, secondary = primary) {
    (this._slotSources ??= [])[slot] = { primary, secondary };
  }

  /**
   * The bind group for one pair of source views, created once and cached.
   *
   * Keyed by the VIEWS rather than by pass index, because a graph transient can
   * be handed a different physical texture when the aliasing assignment
   * changes. Entries are dropped once a frame stops asking for them: the views
   * they name belong to pooled textures the graph now destroys when a frame
   * stops declaring them, so a cache that only grew would both leak and, worse,
   * eventually hand the encoder a bind group over a destroyed texture.
   */
  _bindGroupFor(primaryView, secondaryView) {
    const key = `${viewId(primaryView)}:${viewId(secondaryView)}`;
    let entry = this._bindGroups.get(key);
    if (!entry) {
      entry = {
        bindGroup: this.rhi.device.createBindGroup({
          label: `post:${key}`,
          layout: this.layout,
          entries: [
            { binding: 0, resource: { buffer: this.paramsBuffer, size: PARAMS_BYTES } },
            { binding: 1, resource: primaryView },
            { binding: 2, resource: secondaryView },
            { binding: 3, resource: this.sampler },
          ],
        }),
        lastFrame: 0,
      };
      this._bindGroups.set(key, entry);
    }
    entry.lastFrame = this._frame;
    return entry.bindGroup;
  }

  /** Drop bind groups naming views the last frame did not use. */
  _evictBindGroups() {
    for (const [key, entry] of this._bindGroups) {
      if (entry.lastFrame < this._frame - 2) this._bindGroups.delete(key);
    }
  }

  _draw(pass, pipeline, slot) {
    const sources = this._slotSources[slot];
    const primary = this._graph.viewOf(sources.primary);
    const secondary = this._graph.viewOf(sources.secondary);

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this._bindGroupFor(primary, secondary), [this._offsets[slot]]);
    pass.draw(3);
  }

  destroy() {
    this.paramsBuffer.destroy();
    this._bindGroups.clear();
  }
}

// Views have no identity of their own, so one is stamped on first use. Stable
// because the graph's texture pool hands back the same view objects.
let nextViewId = 1;
function viewId(view) {
  if (!view.__postId) view.__postId = nextViewId++;
  return view.__postId;
}
