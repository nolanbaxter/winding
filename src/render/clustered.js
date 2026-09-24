// Clustered lighting (forward+).
//
// A forward renderer with N lights costs N per fragment whether or not any of
// them reach it. Clustering divides the view frustum into a 3D grid of cells --
// froxels, since they are frustum-shaped, not cubes -- assigns each light to
// the cells it can possibly touch, and lets a fragment look up only its own
// cell's list. A thousand lights in the scene becomes maybe four per pixel.
//
// Two compute passes per frame:
//
//   1. cluster bounds   view-space AABB for every cell
//   2. light assignment for each cell, which lights overlap it
//
// The Z slicing is the part with a real design decision in it. Slicing depth
// uniformly wastes almost every cell on the far half of the view, because
// perspective means distant cells are enormous. The standard fix (Olsson, and
// the scheme id Software described for DOOM 2016) slices exponentially:
//
//   slice = log(distance) * scale + bias
//
// which needs a far distance to normalize against -- and this engine famously
// does not have one. So `lightDistance` is a genuine parameter here rather than
// a missing one: it is the range over which clustered lights are RESOLVED, not
// a draw distance. Anything beyond it falls into the last slice and is still
// lit, just by one coarse cell instead of a fitted one.
//
// That last sentence is only true because the last slice is stretched to cover
// it. The exponential mapping ends the final cell exactly at lightDistance, so
// a light past that used to sit outside every cluster box and match nothing,
// while the fragment shader clamped fragments out there INTO that same cell --
// they read a light list the light was never added to and came out unlit. The
// slice's far edge is now the farthest any light actually reaches, computed
// from the light list each frame. See buildClusters.

import { DEBUG, assert } from '../core/assert.js';
import { grownCapacity } from '../core/grow.js';
import { compileShader } from '../rhi/shader.js';

/**
 * Screen tiles in the froxel grid. A budget, not a shape.
 *
 * It sizes the index buffer -- CLUSTER_TILES * CLUSTER_Z * MAX_LIGHTS_PER_CLUSTER
 * -- so it has to be fixed. How those tiles are ARRANGED is not fixed, and used
 * to be: a hardcoded 16 by 9 made froxels square at exactly one aspect ratio
 * and stretched everywhere else. A portrait phone canvas got cells 3x taller
 * than wide, which overlap roughly 3x as many light spheres and reach the
 * per-cluster cap at a third of the light count a 16:9 monitor manages -- and
 * overflow past that cap is dropped silently.
 *
 * 144 is what 16 by 9 was, so the default viewport's grid is unchanged.
 */
export const CLUSTER_TILES = 144;
export const CLUSTER_Z = 24;
export const CLUSTER_COUNT = CLUSTER_TILES * CLUSTER_Z;

/**
 * Split the tile budget to match an aspect ratio, keeping froxels near square.
 *
 * `x * y <= CLUSTER_TILES` is the invariant the index buffer depends on, which
 * is why y floors rather than rounds: rounding both can land above the budget
 * (at 3:2, 15 x 10 is 150).
 */
export function clusterGridFor(aspect) {
  const safe = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const x = Math.max(1, Math.min(CLUSTER_TILES, Math.round(Math.sqrt(CLUSTER_TILES * safe))));
  const y = Math.max(1, Math.floor(CLUSTER_TILES / x));
  return { x, y };
}

/**
 * Lights per cell. Overflow is dropped, which shows as a light winking out in
 * a crowded cell rather than as a crash. Nothing reports it: the count lives
 * only in a GPU buffer, and reading it back would cost the pipeline stall this
 * engine avoids everywhere else.
 */
export const MAX_LIGHTS_PER_CLUSTER = 64;

/** vec4 positionRadius, vec4 colorIntensity, vec4 directionCone, vec4 coneFalloff */
export const LIGHT_BYTES = 64;
/**
 * Starting light capacity, not a ceiling -- the buffer grows past it.
 *
 * Nothing in the shader depends on this: the light list is a runtime-sized
 * storage array, so widening it needs no recompile. Only MAX_LIGHTS_PER_CLUSTER
 * above is baked in, because that one sizes a fixed per-cluster slice.
 */
export const DEFAULT_LIGHT_CAPACITY = 256;

export const CLUSTER_SHADER = /* wgsl */ `
struct Params {
  // Inverse projection, to turn screen tiles back into view-space rays.
  invProjection : mat4x4<f32>,
  view          : mat4x4<f32>,
  grid          : vec4<u32>,   // x, y, z cells; w = light count
  depth         : vec4<f32>,   // x near, y lightDistance, z farthest light reach
  screen        : vec4<f32>,   // x, y = screen size; z, w = tile size in pixels
};

struct Light {
  positionRadius : vec4<f32>,
  colorIntensity : vec4<f32>,
  directionCone  : vec4<f32>,
  coneFalloff    : vec4<f32>,
};

struct Bounds {
  minPoint : vec4<f32>,
  maxPoint : vec4<f32>,
};

@group(0) @binding(0) var<uniform>             params  : Params;
@group(0) @binding(1) var<storage, read_write> bounds  : array<Bounds>;
@group(0) @binding(2) var<storage, read>       lights  : array<Light>;
@group(0) @binding(3) var<storage, read_write> indices : array<u32>;
@group(0) @binding(4) var<storage, read_write> counts  : array<u32>;

/** Screen pixel to a point on the near plane, in view space. */
fn unproject(pixel : vec2<f32>, ndcZ : f32) -> vec3<f32> {
  let ndc = vec4<f32>(
    (pixel.x / params.screen.x) * 2.0 - 1.0,
    1.0 - (pixel.y / params.screen.y) * 2.0,
    ndcZ, 1.0,
  );
  let view = params.invProjection * ndc;
  return view.xyz / view.w;
}

/**
 * The point where the ray through a pixel reaches a view depth.
 *
 * Built from TWO unprojections of the same pixel and a walk along the line
 * between them, which is the correct ray under any projection.
 *
 * This used to scale a single near-plane point by distance/near, commented
 * "the eye is the origin in view space, so the ray is just that point
 * scaled". True for perspective, where every ray passes through the eye. False
 * for orthographic, where rays are PARALLEL -- and not merely loose: an
 * off-centre tile spanning x in [2, 3] got a froxel at [20, 30] ten near-
 * distances out, a box that does not contain the cell at all. Lights inside
 * it were assigned somewhere else and surfaces went unlit, with nothing
 * reported.
 *
 * ndc 1.0 is the near plane (reverse-Z). ndc 0.5 is finite under both
 * projections -- 2 * near for the infinite perspective, midway for an
 * orthographic box -- and never at the same depth as 1.0, so the division
 * below is always defined.
 */
fn pointAtDepth(pixel : vec2<f32>, distance : f32) -> vec3<f32> {
  let a = unproject(pixel, 1.0);
  let b = unproject(pixel, 0.5);
  let t = (-distance - a.z) / (b.z - a.z);
  return a + (b - a) * t;
}

@compute @workgroup_size(4, 4, 4)
fn buildClusters(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= params.grid.x || id.y >= params.grid.y || id.z >= params.grid.z) { return; }
  let cluster = (id.z * params.grid.y + id.y) * params.grid.x + id.x;

  let tileMin = vec2<f32>(f32(id.x), f32(id.y)) * params.screen.zw;
  let tileMax = tileMin + params.screen.zw;

  // Inverse of the exponential slice mapping, giving this slice's depth range.
  let near = params.depth.x;
  let ratio = params.depth.y / near;
  let sliceNear = near * pow(ratio, f32(id.z) / f32(params.grid.z));
  var sliceFar = near * pow(ratio, f32(id.z + 1u) / f32(params.grid.z));

  // The last slice swallows everything past the grid. The fragment shader
  // clamps any depth beyond lightDistance into this same slice, so if the box
  // stopped at lightDistance -- which is exactly what pow() returns here -- a
  // light out there would be tested against a shell it is nowhere near, match
  // nothing, and land in no cluster at all. The fragment would then read a
  // light list built for the wrong band and come out unlit, which made
  // lightDistance a hard cutoff for punctual lights rather than the resolution
  // knob it is documented as.
  //
  // The far edge is the farthest any light actually reaches, computed on the
  // CPU from the light list itself. Nothing exists past it, so extending
  // further would buy nothing, and it is derived rather than chosen.
  if (id.z == params.grid.z - 1u) { sliceFar = max(sliceFar, params.depth.z); }

  // Four points: the tile's two corners projected onto each slice plane. The
  // other four corners of the froxel are covered because the AABB of these
  // already contains them.
  let a = pointAtDepth(tileMin, sliceNear);
  let b = pointAtDepth(tileMax, sliceNear);
  let c = pointAtDepth(tileMin, sliceFar);
  let d = pointAtDepth(tileMax, sliceFar);

  bounds[cluster].minPoint = vec4<f32>(min(min(a, b), min(c, d)), 0.0);
  bounds[cluster].maxPoint = vec4<f32>(max(max(a, b), max(c, d)), 0.0);
}

/** Squared distance from a point to an AABB; zero when inside. */
fn distanceSqToBox(point : vec3<f32>, boxMin : vec3<f32>, boxMax : vec3<f32>) -> f32 {
  let outside = max(boxMin - point, vec3<f32>(0.0)) + max(point - boxMax, vec3<f32>(0.0));
  return dot(outside, outside);
}

@compute @workgroup_size(4, 4, 4)
fn assignLights(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= params.grid.x || id.y >= params.grid.y || id.z >= params.grid.z) { return; }
  let cluster = (id.z * params.grid.y + id.y) * params.grid.x + id.x;

  let boxMin = bounds[cluster].minPoint.xyz;
  let boxMax = bounds[cluster].maxPoint.xyz;

  var found = 0u;
  let base = cluster * ${MAX_LIGHTS_PER_CLUSTER}u;

  for (var i = 0u; i < params.grid.w; i = i + 1u) {
    let light = lights[i];
    // Lights arrive in world space; the cluster grid is view space.
    let centre = (params.view * vec4<f32>(light.positionRadius.xyz, 1.0)).xyz;
    let radius = light.positionRadius.w;

    // A sphere-vs-AABB test. Conservative for spot lights, which are tested by
    // their bounding sphere -- a cone's own test is much more work for a cell
    // count this small.
    if (distanceSqToBox(centre, boxMin, boxMax) <= radius * radius) {
      if (found < ${MAX_LIGHTS_PER_CLUSTER}u) {
        indices[base + found] = i;
        found = found + 1u;
      }
    }
  }

  counts[cluster] = found;
}
`;

export class ClusteredLights {
  /**
   * Async because WGSL compilation errors do not throw -- createShaderModule
   * always returns a module, and a broken one fails later as a confusing
   * pipeline error pointing at the wrong thing. compileShader() surfaces the
   * real diagnostic.
   */
  static async create(rhi) {
    const clusters = new ClusteredLights(rhi);
    await clusters._init();
    return clusters;
  }

  constructor(rhi) {
    this.rhi = rhi;
    const device = rhi.device;

    this.lightCapacity = DEFAULT_LIGHT_CAPACITY;
    /**
     * Tiles across and down. Derived from the viewport in update(); these are
     * the 16:9 split so a caller that reads them before a frame sees the shape
     * the grid has always had rather than zeros.
     */
    ({ x: this.gridX, y: this.gridY } = clusterGridFor(16 / 9));
    this.lightData = new Float32Array(this.lightCapacity * (LIGHT_BYTES / 4));
    this.lightBuffer = device.createBuffer({
      label: 'lights',
      size: this.lightCapacity * LIGHT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    /** Bumped by _grow. The renderer's frame bind group names lightBuffer. */
    this.buffersRevision = 0;

    this.boundsBuffer = device.createBuffer({
      label: 'cluster-bounds',
      size: CLUSTER_COUNT * 32,            // two vec4 per cluster
      usage: GPUBufferUsage.STORAGE,
    });
    this.indexBuffer = device.createBuffer({
      label: 'cluster-light-indices',
      size: CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.countBuffer = device.createBuffer({
      label: 'cluster-light-counts',
      size: CLUSTER_COUNT * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // invProjection(64) + view(64) + grid(16) + depth(16) + screen(16)
    this.paramsData = new ArrayBuffer(176);
    this.paramsF32 = new Float32Array(this.paramsData);
    this.paramsU32 = new Uint32Array(this.paramsData);
    this.paramsBuffer = device.createBuffer({
      label: 'cluster-params',
      size: this.paramsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.layout = device.createBindGroupLayout({
      label: 'clustered',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.bindGroup = this._makeBindGroup();

    this.lightCount = 0;
    this.sliceScale = 0;
    this.sliceBias = 0;
    this.tileSize = new Float32Array(2);

    // Bound once, for the same reason the shadow cascades are.
    this._buildExecute = (pass) => this._dispatch(pass, this.buildPipeline);
    this._assignExecute = (pass) => this._dispatch(pass, this.assignPipeline);
  }

  async _init() {
    const device = this.rhi.device;
    const shader = await compileShader(device, CLUSTER_SHADER, 'clustered.wgsl');
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });

    this.buildPipeline = device.createComputePipeline({
      label: 'cluster-bounds',
      layout: pipelineLayout,
      compute: { module: shader.module, entryPoint: 'buildClusters' },
    });
    this.assignPipeline = device.createComputePipeline({
      label: 'cluster-assign',
      layout: pipelineLayout,
      compute: { module: shader.module, entryPoint: 'assignLights' },
    });
  }

  /** Names lightBuffer, so it is rebuilt whenever that buffer is replaced. */
  _makeBindGroup() {
    return this.rhi.device.createBindGroup({
      label: 'clustered',
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.boundsBuffer } },
        { binding: 2, resource: { buffer: this.lightBuffer } },
        { binding: 3, resource: { buffer: this.indexBuffer } },
        { binding: 4, resource: { buffer: this.countBuffer } },
      ],
    });
  }

  /**
   * Widen the light list.
   *
   * Contents are not copied: update() rewrites every light from the scene on
   * the line after this returns.
   */
  _grow(needed) {
    const capacity = grownCapacity(this.lightCapacity, needed);
    this.lightData = new Float32Array(capacity * (LIGHT_BYTES / 4));

    this.lightBuffer.destroy();
    this.lightBuffer = this.rhi.device.createBuffer({
      label: 'lights',
      size: capacity * LIGHT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.lightCapacity = capacity;

    // This group names the buffer that was just destroyed. So does the
    // renderer's frame group, which is what the revision below is for.
    this.bindGroup = this._makeBindGroup();
    this.buffersRevision++;
  }

  /**
   * Pack the scene's lights and recompute the slice mapping.
   *
   * `lightDistance` is the range clustered lights are resolved over. It is not
   * a cutoff: anything past it falls into the last slice and is still lit.
   */
  update(scene, camera, lightDistance) {
    if (scene.lightCount > this.lightCapacity) this._grow(scene.lightCount);
    const count = scene.lightCount;
    this.lightCount = count;

    this.lightData.set(scene.lights.subarray(0, count * (LIGHT_BYTES / 4)));
    this.rhi.queue.writeBuffer(this.lightBuffer, 0, this.lightData, 0, count * (LIGHT_BYTES / 4));

    // How far the farthest light actually reaches, in view depth. The last
    // cluster slice is stretched to this, because the fragment shader clamps
    // everything past lightDistance into that slice and a light outside its
    // box is in no cluster at all. Derived from the light list, so it is
    // exactly far enough and never further.
    //
    // Row 2 of the view matrix takes a world point to its view z, which is
    // negative in front of the camera; depth is its negation. A light lights
    // out to its radius, so that is what has to be inside the box.
    const view = camera.view;
    let reach = lightDistance;   // published as lightReach below
    for (let i = 0; i < count; i++) {
      const o = i * (LIGHT_BYTES / 4);
      const x = this.lightData[o], y = this.lightData[o + 1], z = this.lightData[o + 2];
      const radius = this.lightData[o + 3];
      const depth = -(view[2] * x + view[6] * y + view[10] * z + view[14]);
      if (depth + radius > reach) reach = depth + radius;
    }
    /** Far edge of the last slice this frame. Never less than lightDistance. */
    this.lightReach = reach;

    // Unconditional, and for the same reason cascadeSplits throws: the slice
    // mapping is log(lightDistance / near), and there is no value of it that
    // fails loudly. A near of 0 gives log(0) = -Infinity and then
    // Infinity/Infinity = NaN; lightDistance === near gives a ratio of 0 and a
    // division by it; lightDistance < near inverts the mapping. Each one ends
    // as NaN slice bounds, every light assigned to no cluster, and a scene that
    // renders perfectly except that nothing is lit.
    if (!(camera.near > 0)) {
      throw new Error(`ClusteredLights: camera near must be positive, got ${camera.near}`);
    }
    if (!(lightDistance > camera.near)) {
      throw new Error(
        `ClusteredLights: lightDistance ${lightDistance} must be beyond the near plane ${camera.near}`,
      );
    }

    // slice = log(d) * scale + bias. Precomputed for the PBR shader, which
    // does two operations instead of a log and a divide per fragment. The
    // CLUSTER shader does not use these -- it needs the inverse mapping and
    // re-derives it with pow() from near and lightDistance.
    const ratio = Math.log(lightDistance / camera.near);
    this.sliceScale = CLUSTER_Z / ratio;
    this.sliceBias = -(CLUSTER_Z * Math.log(camera.near)) / ratio;

    // The grid follows the viewport, so froxels stay near square as it is
    // resized or rotated. Recomputed each frame because it is two integers
    // from one divide -- cheaper than tracking whether the aspect moved.
    const grid = clusterGridFor(this.rhi.width / this.rhi.height);
    this.gridX = grid.x;
    this.gridY = grid.y;
    this.tileSize[0] = this.rhi.width / this.gridX;
    this.tileSize[1] = this.rhi.height / this.gridY;

    const f32 = this.paramsF32;
    const u32 = this.paramsU32;
    f32.set(camera.inverseProjection, 0);
    f32.set(camera.view, 16);
    u32[32] = this.gridX; u32[33] = this.gridY; u32[34] = CLUSTER_Z; u32[35] = count;
    f32[36] = camera.near;
    f32[37] = lightDistance;
    f32[38] = reach;
    f32[39] = 0;
    f32[40] = this.rhi.width;
    f32[41] = this.rhi.height;
    f32[42] = this.tileSize[0];
    f32[43] = this.tileSize[1];

    this.rhi.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
  }

  /**
   * Declare both compute passes.
   *
   * The bounds pass writes what the assignment pass reads, and the assignment
   * pass writes what the forward pass reads -- so the graph orders all three
   * without anything here saying so.
   */
  addPasses(graph, { boundsResource, lightsResource, indicesResource, countsResource }) {
    graph.addPass({
      name: 'cluster-bounds',
      type: 'compute',
      writes: [boundsResource],
      execute: this._buildExecute,
    });
    graph.addPass({
      name: 'cluster-assign',
      type: 'compute',
      reads: [boundsResource, lightsResource],
      writes: [indicesResource, countsResource],
      execute: this._assignExecute,
    });
  }

  _dispatch(pass, pipeline) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup);
    // workgroup_size is (4,4,4), so the grid rounds up to whole workgroups and
    // the shader bounds-checks the leftovers.
    pass.dispatchWorkgroups(
      Math.ceil(this.gridX / 4), Math.ceil(this.gridY / 4), Math.ceil(CLUSTER_Z / 4),
    );
  }

  destroy() {
    this.lightBuffer.destroy();
    this.boundsBuffer.destroy();
    this.indexBuffer.destroy();
    this.countBuffer.destroy();
    this.paramsBuffer.destroy();
  }
}

/**
 * Which cluster a view-space DEPTH falls in -- z along the view axis, not
 * radial distance from the eye. The slices are planes, not shells.
 *
 * The reference the shader's version was written from, and the only place the
 * mapping is tested. It does NOT verify the shader -- that is a separate copy
 * in WGSL, and the two can drift. Changing one means changing both.
 */
export function sliceFor(depth, near, lightDistance, slices = CLUSTER_Z) {
  const ratio = Math.log(lightDistance / near);
  const scale = slices / ratio;
  const bias = -(slices * Math.log(near)) / ratio;
  return Math.min(Math.max(Math.floor(Math.log(depth) * scale + bias), 0), slices - 1);
}
