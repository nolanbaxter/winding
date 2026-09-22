// Hierarchical Z-buffer, for occlusion culling.
//
// Frustum culling removes what is off screen. Occlusion culling removes what is
// on screen but behind something else -- a city block behind a building, an
// entire room behind a wall. It is the only cull that scales with scene DEPTH
// rather than scene size.
//
// The structure is a mip pyramid over depth where each level stores the most
// CONSERVATIVE depth of its four children, so a single texel at level 5 answers
// "is anything in this 32x32 region closer than X" in one fetch.
//
// Conservative means MINIMUM here, and working out why takes a moment under
// reverse-Z. The depth buffer holds the nearest surface per
// pixel, and nearer is LARGER. An object is hidden only if it is behind the
// nearest surface at every pixel it covers -- so the value that matters over a
// region is the smallest of those, the weakest occluder. Take the max instead
// and distant geometry starts vanishing behind things that do not cover it.
//
// WHY THE PYRAMID IS BUILT MID-FRAME. Culling runs in two phases: whatever was
// visible last frame is drawn first, this pyramid is built from THAT depth,
// and a second cull re-tests everything else against it before a second pass
// draws the newcomers. So the pyramid is never read a frame after it was
// written -- it is produced and consumed inside one command buffer, projected
// with the same matrix that rendered it.
//
// The alternative, and what this was before, is to build it at the end of a
// frame and test against it at the start of the next. Nothing is read back to
// the CPU either way -- a readback would stall the pipeline, which is the cost
// this whole approach exists to avoid -- but the stale version culls an object
// that has just become visible for a frame, which shows as a pop when the
// camera swings past a corner. See gpudriven.js for the phases themselves.

import { DEBUG, assert } from '../core/assert.js';
import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';

/** Single channel float. Not filterable by default, which is fine: every read
 *  is a textureLoad, because a filtered average is not a conservative bound. */
export const HZB_FORMAT = 'r32float';

const PARAMS_BYTES = 16;

const HZB_SHADER = /* wgsl */ `
struct Params {
  srcSize : vec2<u32>,
  dstSize : vec2<u32>,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var depthSource : texture_depth_2d;
@group(0) @binding(2) var mipSource   : texture_2d<f32>;

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

/**
 * Level 0, straight off the depth buffer.
 *
 * The pyramid is the largest power of two that fits the screen, so the scale is
 * in [1, 2) and one output texel covers a source interval shorter than two
 * texels -- which still STRADDLES three of them whenever it is unaligned. At
 * 1920 -> 1024 that is 75% of the row: texel 1 owns [1.875, 3.75), touching
 * source texels 1, 2 and 3.
 *
 * So the gather is 3x3, clamped to the footprint's own last texel rather than
 * to the level. Clamping to that last texel means an oversized gather re-reads
 * one it already has instead of stealing one from the neighbouring output texel,
 * which keeps the reduction exact in both directions.
 *
 * Covering the footprint matters because this is a MIN and a missing texel can
 * only raise the result. Under reverse-Z a larger value is nearer, so the
 * pyramid would claim a nearer weakest-occluder than really exists, and the
 * cull test would delete geometry that is actually visible.
 */
@fragment
fn fsFirst(v : VertexOut) -> @location(0) f32 {
  let dst = vec2<u32>(v.position.xy);
  let scale = vec2<f32>(params.srcSize) / vec2<f32>(params.dstSize);
  let limit = vec2<i32>(params.srcSize) - vec2<i32>(1, 1);

  let lo = vec2<i32>(vec2<f32>(dst) * scale);
  // The last source texel this output texel owns. The interval is half-open,
  // so ceil() lands one past it.
  let hi = min(vec2<i32>(ceil(vec2<f32>(dst + vec2<u32>(1u, 1u)) * scale)) - vec2<i32>(1, 1), limit);

  var farthest = 1e30;
  for (var y = 0; y < 3; y = y + 1) {
    for (var x = 0; x < 3; x = x + 1) {
      let c = min(lo + vec2<i32>(x, y), hi);
      farthest = min(farthest, textureLoad(depthSource, c, 0));
    }
  }
  return farthest;
}

/** Every level after the first: an exact 2x2 min, since the sizes are powers of two. */
@fragment
fn fsReduce(v : VertexOut) -> @location(0) f32 {
  let dst = vec2<u32>(v.position.xy);
  let base = vec2<i32>(dst) * 2;
  let limit = vec2<i32>(params.srcSize) - vec2<i32>(1, 1);

  let a = textureLoad(mipSource, min(base, limit), 0).r;
  let b = textureLoad(mipSource, min(base + vec2<i32>(1, 0), limit), 0).r;
  let c = textureLoad(mipSource, min(base + vec2<i32>(0, 1), limit), 0).r;
  let d = textureLoad(mipSource, min(base + vec2<i32>(1, 1), limit), 0).r;
  return min(min(a, b), min(c, d));
}
`;

export function previousPowerOfTwo(n) {
  if (n < 1) return 1;
  return 2 ** Math.floor(Math.log2(n));
}

/**
 * Which pyramid level answers a screen-space rectangle in one 2x2 fetch.
 *
 * The reference the shader's inline version was written from, and the only
 * place the choice is tested. It does NOT verify the shader -- that is a
 * separate copy, and the two can drift. Treat a change here as a change that
 * must be made twice.
 */
export function mipLevelForExtent(widthTexels, heightTexels, levelCount) {
  const extent = Math.max(widthTexels, heightTexels);
  const level = Math.ceil(Math.log2(Math.max(extent, 1)));
  return Math.min(Math.max(level, 0), levelCount - 1);
}

export class HierarchicalDepth {
  static async create(rhi, pipelines) {
    const hzb = new HierarchicalDepth(rhi);
    await hzb._init(pipelines);
    return hzb;
  }

  constructor(rhi) {
    this.rhi = rhi;
    this.width = 0;
    this.height = 0;
    this.levelCount = 0;
    this.texture = null;
    this.view = null;
    this.levelViews = [];

    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    this.maxLevels = 16;
    this.paramsStaging = new ArrayBuffer(this.alignment * this.maxLevels);
    this.paramsBuffer = rhi.device.createBuffer({
      label: 'hzb-params',
      size: this.paramsStaging.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Level 0 reads the depth buffer and never touches binding 2, but a bind
    // group must still supply every entry its layout declares -- and WebGPU
    // validates usage STATICALLY, so pointing it at the level being rendered
    // into is a conflict even though the shader ignores it. A 1x1 stand-in has
    // no such overlap.
    this.dummy = rhi.device.createTexture({
      label: 'hzb-unused',
      size: [1, 1, 1],
      format: HZB_FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.dummyView = this.dummy.createView();

    this._executors = [];
    for (let i = 0; i < this.maxLevels; i++) {
      this._executors.push((pass) => this._encodeLevel(pass, i));
    }
    this._bindGroups = [];
  }

  async _init(pipelines) {
    const device = this.rhi.device;
    const shader = await compileShader(device, HZB_SHADER, 'hzb.wgsl');

    this.layout = device.createBindGroupLayout({
      label: 'hzb',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_BYTES },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'depth' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          // Unfiltered: r32float is not filterable without an optional feature,
          // and averaging depths would not be conservative anyway.
          texture: { sampleType: 'unfilterable-float' },
        },
      ],
    });
    const layout = createPipelineLayout(device, { 0: this.layout }, 'hzb');
    const common = {
      layout, shader, depth: null,
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      targets: [{ format: HZB_FORMAT }],
    };

    this.firstDescriptor = { ...common, label: 'hzb-first', fragmentEntry: 'fsFirst' };
    this.reduceDescriptor = { ...common, label: 'hzb-reduce', fragmentEntry: 'fsReduce' };
    await pipelines.warm([this.firstDescriptor, this.reduceDescriptor]);

    this._pipelines = pipelines;
    this.firstPipeline = pipelines.get(this.firstDescriptor);
    this.reducePipeline = pipelines.get(this.reduceDescriptor);
  }

  /** Rebuild the pyramid's storage when the surface size changes. */
  resize(screenWidth, screenHeight, depthView) {
    const width = previousPowerOfTwo(screenWidth);
    const height = previousPowerOfTwo(screenHeight);
    if (width === this.width && height === this.height && this._depthView === depthView) return;

    this.texture?.destroy();
    this.width = width;
    this.height = height;
    this.levelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;
    if (DEBUG) assert(this.levelCount <= this.maxLevels, 'HZB has more levels than slots');

    this.texture = this.rhi.device.createTexture({
      label: 'hzb',
      size: [width, height, 1],
      format: HZB_FORMAT,
      mipLevelCount: this.levelCount,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.view = this.texture.createView({ label: 'hzb' });

    this.levelViews = [];
    for (let level = 0; level < this.levelCount; level++) {
      this.levelViews.push(this.texture.createView({
        baseMipLevel: level, mipLevelCount: 1, dimension: '2d',
      }));
    }

    this._depthView = depthView;
    this._bindGroups = [];

    let srcW = screenWidth;
    let srcH = screenHeight;
    for (let level = 0; level < this.levelCount; level++) {
      const dstW = Math.max(1, width >> level);
      const dstH = Math.max(1, height >> level);

      const u32 = new Uint32Array(this.paramsStaging, level * this.alignment, 4);
      u32[0] = srcW; u32[1] = srcH; u32[2] = dstW; u32[3] = dstH;

      this._bindGroups.push(this.rhi.device.createBindGroup({
        label: `hzb-${level}`,
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuffer, size: PARAMS_BYTES } },
          { binding: 1, resource: depthView },
          { binding: 2, resource: level === 0 ? this.dummyView : this.levelViews[level - 1] },
        ],
      }));

      srcW = dstW;
      srcH = dstH;
    }

    this.rhi.queue.writeBuffer(this.paramsBuffer, 0, this.paramsStaging);
  }

  /**
   * Declare one pass per level.
   *
   * Each level writes the mip below it and reads the one above, so the graph
   * chains them without anything here saying so. `depthResource` is only read
   * by level 0, which is what flips the depth buffer's store op from discard to
   * store -- automatically, the moment this pass exists.
   */
  addPasses(graph, depthResource, levelResources) {
    if (DEBUG) assert(levelResources.length === this.levelCount, 'one resource per HZB level');

    for (let level = 0; level < this.levelCount; level++) {
      graph.addPass({
        name: `hzb:${level}`,
        // Each MIP is its own graph resource, even though they share one
        // texture. Declaring them as a single resource would make every level
        // both a reader and a writer of it, which is a write-after-read the
        // graph cannot order -- the same thing that stopped bloom accumulating
        // in place.
        reads: level === 0 ? [depthResource] : [levelResources[level - 1]],
        color: [{ resource: levelResources[level], clear: { r: 0, g: 0, b: 0, a: 0 } }],
        execute: this._executors[level],
      });
    }
  }

  _encodeLevel(pass, level) {
    pass.setPipeline(level === 0 ? this.firstPipeline : this.reducePipeline);
    pass.setBindGroup(0, this._bindGroups[level], [level * this.alignment]);
    pass.draw(3);
  }

  destroy() {
    this.texture?.destroy();
    this.dummy.destroy();
    this.paramsBuffer.destroy();
  }
}
