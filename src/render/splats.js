// Gaussian splats: drawing a capture (scene.addSplats; the file is read by
// scene/splats.js).
//
// Each splat is a 3D Gaussian. Seen from the camera it projects to a 2D one
// -- the EWA splatting of Zwicker et al., as 3D Gaussian Splatting uses it:
// its covariance carried into view space, then through the projection's
// derivative at its centre, J W S W^T J^T -- drawn as a quad over the ellipse
// it covers, whose fragments fade by the Gaussian. Laid over each other back
// to front, blended, they make the picture.
//
// Back to front is the whole difficulty: a capture is hundreds of thousands
// to millions of splats, and their order changes as the camera moves. It is
// sorted on the GPU, every frame, as the rest of the engine culls there:
//
//   keys     a thread a splat: culled against the view, and the survivors
//            appended with a 16-bit key, how far away they are across the
//            cloud's depth, counted into 65536 buckets as they go
//   scan     one workgroup: where each bucket begins, from the counts --
//            which it then zeroes, ready for the next frame
//   scatter  a thread a survivor: placed at its bucket's next slot
//
// A counting sort, in one pass, because a key fits in 16 bits. The order
// within a bucket comes out of atomics, so it varies, but a bucket is a
// 65536th of the cloud's depth, and splats that close are drawn as one. The
// draw is indirect: the survivors' count is what the keys pass appended, and
// never comes back to the CPU.
//
// A splat is lit by nothing: its colour is what the capture saw, decoded from
// sRGB, and tonemapped with everything else. Colour does not change with the
// view (the file's higher harmonics are skipped). Splats are hidden by
// geometry in front of them, and hide nothing: they write no depth, and cast
// no shadows. Two clouds are each sorted on their own and drawn far one
// first, so where two overlap, they do not interleave.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { DEPTH_FORMAT, DEPTH_COMPARE } from '../rhi/device.js';
import { handleIndex } from '../core/handle.js';
import { mat4Multiply } from '../core/math/mat4.js';

const BUCKETS = 65536;
const WORKGROUP = 256;
/** view, projection, model, viewport, depth range, count */
const CLOUD_BYTES = 224;
/** Frames a node's sort buffers outlive the last frame that drew it. */
const KEEP_FRAMES = 120;

const CLOUD_WGSL = /* wgsl */ `
struct Cloud {
  view       : mat4x4<f32>,
  projection : mat4x4<f32>,
  model      : mat4x4<f32>,
  viewport   : vec2<f32>,
  depthMin   : f32,   // the nearest view depth the cloud reaches
  depthScale : f32,   // buckets per unit of depth across it
  count      : u32,
};
`;

const SORT_SHADER = /* wgsl */ `
${CLOUD_WGSL}
struct DrawArgs {
  vertexCount   : u32,
  instanceCount : atomic<u32>,
  firstVertex   : u32,
  firstInstance : u32,
};

const BUCKETS = ${BUCKETS}u;

@group(0) @binding(0) var<uniform> cloud : Cloud;
@group(0) @binding(1) var<storage, read> centers : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> keys : array<u32>;
@group(0) @binding(3) var<storage, read_write> visible : array<u32>;
@group(0) @binding(4) var<storage, read_write> counts : array<atomic<u32>, BUCKETS>;
@group(0) @binding(5) var<storage, read_write> cursors : array<atomic<u32>, BUCKETS>;
@group(0) @binding(6) var<storage, read_write> order : array<u32>;
@group(0) @binding(7) var<storage, read_write> args : DrawArgs;

/** A thread's splat: a grid of workgroups wraps past the 65535 a dimension allows. */
fn splatOf(id : vec3<u32>, groups : vec3<u32>) -> u32 {
  return id.x + id.y * groups.x * ${WORKGROUP}u;
}

@compute @workgroup_size(${WORKGROUP})
fn makeKeys(@builtin(global_invocation_id) id : vec3<u32>, @builtin(num_workgroups) groups : vec3<u32>) {
  let i = splatOf(id, groups);
  if (i >= cloud.count) { return; }
  let view = cloud.view * cloud.model * vec4<f32>(bitcast<vec3<f32>>(centers[i].xyz), 1.0);
  let clip = cloud.projection * view;
  // Behind the camera, or well outside the view: its centre, with a margin
  // for the part of a splat that reaches in from beside it.
  let w = clip.w;
  if (view.z >= 0.0 || abs(clip.x) > 1.3 * w || abs(clip.y) > 1.3 * w) { return; }
  // Far first: the farthest key is the smallest.
  let bucket = u32(clamp((-view.z - cloud.depthMin) * cloud.depthScale, 0.0, f32(BUCKETS - 1u)));
  let key = BUCKETS - 1u - bucket;
  let slot = atomicAdd(&args.instanceCount, 1u);
  keys[slot] = key;
  visible[slot] = i;
  atomicAdd(&counts[key], 1u);
}

var<workgroup> sums : array<u32, ${WORKGROUP}>;

/** Where each bucket starts: a thread sums 256 buckets, the 256 sums are scanned, and each thread writes its own run. */
@compute @workgroup_size(${WORKGROUP})
fn scanCounts(@builtin(local_invocation_index) t : u32) {
  let first = t * ${BUCKETS / WORKGROUP}u;
  var total = 0u;
  for (var b = 0u; b < ${BUCKETS / WORKGROUP}u; b++) { total += atomicLoad(&counts[first + b]); }
  sums[t] = total;
  workgroupBarrier();
  for (var step = 1u; step < ${WORKGROUP}u; step <<= 1u) {
    var add = 0u;
    if (t >= step) { add = sums[t - step]; }
    workgroupBarrier();
    sums[t] += add;
    workgroupBarrier();
  }
  var at = sums[t] - total;
  for (var b = 0u; b < ${BUCKETS / WORKGROUP}u; b++) {
    let n = atomicLoad(&counts[first + b]);
    atomicStore(&cursors[first + b], at);
    atomicStore(&counts[first + b], 0u);   // ready for the next frame's keys
    at += n;
  }
}

@compute @workgroup_size(${WORKGROUP})
fn place(@builtin(global_invocation_id) id : vec3<u32>, @builtin(num_workgroups) groups : vec3<u32>) {
  let i = splatOf(id, groups);
  if (i >= atomicLoad(&args.instanceCount)) { return; }
  order[atomicAdd(&cursors[keys[i]], 1u)] = visible[i];
}
`;

const DRAW_SHADER = /* wgsl */ `
${CLOUD_WGSL}
@group(0) @binding(0) var<uniform> cloud : Cloud;
@group(0) @binding(1) var<storage, read> centers : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> covariances : array<f32>;
@group(0) @binding(3) var<storage, read> order : array<u32>;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  // Where in the quad, in units of the ellipse's axes: the Gaussian is exp(-|offset|^2).
  @location(0) offset : vec2<f32>,
  @location(1) colour : vec4<f32>,
};

fn decodeSRGB(c : vec3<f32>) -> vec3<f32> {
  return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@vertex
fn vs(@builtin(vertex_index) corner : u32, @builtin(instance_index) instance : u32) -> VertexOut {
  var out : VertexOut;
  // Off the screen, for a splat that projects to nothing.
  out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  let i = order[instance];
  let center = centers[i];
  let modelView = cloud.view * cloud.model;
  let view = modelView * vec4<f32>(bitcast<vec3<f32>>(center.xyz), 1.0);
  let clip = cloud.projection * view;

  let o = i * 6u;
  let sigma = mat3x3<f32>(
    covariances[o], covariances[o + 1u], covariances[o + 2u],
    covariances[o + 1u], covariances[o + 3u], covariances[o + 4u],
    covariances[o + 2u], covariances[o + 4u], covariances[o + 5u],
  );
  // The projection's derivative at the centre, in pixels: constant for an
  // orthographic camera; for a perspective one, with the centre held inside
  // a little more than the view, as 3DGS does, so a splat beside it does
  // not stretch without bound.
  let p = cloud.projection;
  let fx = p[0][0] * cloud.viewport.x * 0.5;
  let fy = p[1][1] * cloud.viewport.y * 0.5;
  var J = mat3x3<f32>(vec3<f32>(fx, 0.0, 0.0), vec3<f32>(0.0, fy, 0.0), vec3<f32>(0.0));
  if (p[3][3] == 0.0) {
    let d = -view.z;
    let x = clamp(view.x / d, -1.3 / p[0][0], 1.3 / p[0][0]) * d;
    let y = clamp(view.y / d, -1.3 / p[1][1], 1.3 / p[1][1]) * d;
    J = mat3x3<f32>(vec3<f32>(fx / d, 0.0, 0.0), vec3<f32>(0.0, fy / d, 0.0), vec3<f32>(fx * x / (d * d), fy * y / (d * d), 0.0));
  }
  let T = J * mat3x3<f32>(modelView[0].xyz, modelView[1].xyz, modelView[2].xyz);
  let cov = T * sigma * transpose(T);
  // Plus a third of a pixel each way: a splat smaller than a pixel still
  // covers one, as 3DGS trains them to.
  let a = cov[0][0] + 0.3;
  let b = cov[0][1];
  let c = cov[1][1] + 0.3;

  // The ellipse's axes: the 2D covariance's eigenvectors, each as long as
  // the square root of twice its eigenvalue.
  let mid = 0.5 * (a + c);
  let radius = length(vec2<f32>(0.5 * (a - c), b));
  let major = mid + radius;
  let minor = mid - radius;
  if (minor <= 0.0) { return out; }
  var axis = vec2<f32>(1.0, 0.0);
  if (abs(b) > 1e-12) { axis = normalize(vec2<f32>(b, major - a)); } else if (c > a) { axis = vec2<f32>(0.0, 1.0); }
  let along = min(sqrt(2.0 * major), 1024.0) * axis;
  let across = min(sqrt(2.0 * minor), 1024.0) * vec2<f32>(axis.y, -axis.x);

  // Out to where it fades below 1/255, and no further than e^-4 of its peak:
  // a faint splat covers fewer pixels than an opaque one of the same size.
  let colour = unpack4x8unorm(center.w);
  let reach = log(255.0 * colour.a);
  if (reach <= 0.0) { return out; }
  let q = (vec2<f32>(f32(corner & 1u), f32(corner >> 1u)) * 2.0 - 1.0) * sqrt(min(reach, 4.0));
  let pixels = q.x * along + q.y * across;
  out.position = vec4<f32>(clip.xy + pixels * 2.0 / cloud.viewport * clip.w, clip.z, clip.w);
  out.offset = q;
  out.colour = vec4<f32>(decodeSRGB(colour.rgb), colour.a);
  return out;
}

@fragment
fn fs(v : VertexOut) -> @location(0) vec4<f32> {
  // Premultiplied, and never wholly opaque, as 3DGS draws them.
  let alpha = min(0.99, exp(-dot(v.offset, v.offset)) * v.colour.a);
  if (alpha < 1.0 / 255.0) { discard; }
  return vec4<f32>(v.colour.rgb * alpha, alpha);
}
`;

/**
 * A capture on the GPU: engine.loadSplats makes one, scene.addSplats places
 * it, any number of times. `count` splats; `min` and `max`, the corners of
 * the box around their centres, in the capture's own units.
 */
export class Splats {
  constructor(rhi, data, label = 'splats') {
    this.rhi = rhi;
    this.count = data.count;
    this.min = data.min;
    this.max = data.max;
    this.centers = createBuffer(rhi, { label: `${label}-centers`, data: data.centers, usage: GPUBufferUsage.STORAGE });
    this.covariances = createBuffer(rhi, { label: `${label}-covariances`, data: data.covariances, usage: GPUBufferUsage.STORAGE });
  }

  destroy() {
    this.centers.destroy();
    this.covariances.destroy();
  }
}

export class SplatPass {
  static async create(rhi, pipelines, colorFormat) {
    const device = rhi.device;
    const [sortShader, drawShader] = await Promise.all([
      compileShader(device, SORT_SHADER, 'splats-sort.wgsl'),
      compileShader(device, DRAW_SHADER, 'splats.wgsl'),
    ]);
    const storage = (binding, type = 'storage') => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const sortLayout = device.createBindGroupLayout({
      label: 'splats-sort',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        storage(1, 'read-only-storage'), storage(2), storage(3), storage(4), storage(5), storage(6), storage(7),
      ],
    });
    const drawLayout = device.createBindGroupLayout({
      label: 'splats-draw',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const sort = createPipelineLayout(device, { 0: sortLayout }, 'splats-sort');
    const compute = (entry) => pipelines.compute({ label: `splats-${entry}`, layout: sort, shader: sortShader, entry });
    const drawDescriptor = {
      label: 'splats',
      layout: createPipelineLayout(device, { 0: drawLayout }, 'splats'),
      shader: drawShader,
      buffers: [],
      targets: [{
        format: colorFormat,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
      depth: { format: DEPTH_FORMAT, depthCompare: DEPTH_COMPARE, depthWriteEnabled: false },
    };
    await pipelines.warm([drawDescriptor]);
    return new SplatPass(rhi, {
      sortLayout, drawLayout,
      keys: compute('makeKeys'), scan: compute('scanCounts'), scatter: compute('place'),
      draw: pipelines.get(drawDescriptor),
    });
  }

  constructor(rhi, built) {
    this.rhi = rhi;
    Object.assign(this, built);
    /** Each placed cloud's sort buffers, by its scene record. */
    this._states = new Map();
    this._items = [];
    this._frame = 0;
    this._modelView = new Float32Array(16);
    this._cloud = new ArrayBuffer(CLOUD_BYTES);
    this._cloudF32 = new Float32Array(this._cloud);
    this._cloudU32 = new Uint32Array(this._cloud);
    this._args = new Uint32Array([4, 0, 0, 0]);
    /** Splats in the clouds the last frame drew, before culling. */
    this.count = 0;
    this._sort = (pass) => this._encodeSort(pass);
    this._draw = (pass) => this._encodeDraw(pass);
  }

  /**
   * This frame's clouds: each one's camera, transform and depth range
   * written, and its draw count reset. Returns how many there are.
   */
  prepare(scene, camera, width, height) {
    this._frame++;
    const items = this._items;
    items.length = 0;
    this.count = 0;
    for (const [entity, record] of scene.splats) {
      const splats = record.splats;
      if (splats.unloaded) throw new Error('addSplats: these splats were unloaded; remove the node first');
      const state = this._stateFor(record, splats);
      state.lastFrame = this._frame;
      const m = handleIndex(entity) * 16;
      const world = scene.transforms.world;
      const modelView = this._modelView;
      mat4Multiply(modelView, camera.view, world, 0, 0, m);
      // The depth the cloud spans in view: its box's eight corners.
      let near = Infinity;
      let far = -Infinity;
      for (let k = 0; k < 8; k++) {
        const x = k & 1 ? splats.max[0] : splats.min[0];
        const y = k & 2 ? splats.max[1] : splats.min[1];
        const z = k & 4 ? splats.max[2] : splats.min[2];
        const depth = -(modelView[2] * x + modelView[6] * y + modelView[10] * z + modelView[14]);
        if (depth < near) near = depth;
        if (depth > far) far = depth;
      }
      near = Math.max(near, 0);
      if (far <= near) continue;   // wholly behind the camera
      const f = this._cloudF32;
      f.set(camera.view, 0);
      f.set(camera.projection, 16);
      f.set(world.subarray(m, m + 16), 32);
      f[48] = width;
      f[49] = height;
      f[50] = near;
      f[51] = (BUCKETS - 1) / (far - near);
      this._cloudU32[52] = splats.count;
      this.rhi.queue.writeBuffer(state.cloud, 0, this._cloud);
      this.rhi.queue.writeBuffer(state.args, 0, this._args);
      const cx = (splats.min[0] + splats.max[0]) / 2, cy = (splats.min[1] + splats.max[1]) / 2, cz = (splats.min[2] + splats.max[2]) / 2;
      items.push({ state, count: splats.count, depth: -(modelView[2] * cx + modelView[6] * cy + modelView[10] * cz + modelView[14]) });
      this.count += splats.count;
    }
    items.sort((a, b) => b.depth - a.depth);
    // A cloud no frame has drawn for a while gives its buffers back.
    for (const [record, state] of this._states) {
      if (state.lastFrame < this._frame - KEEP_FRAMES) {
        destroyState(state);
        this._states.delete(record);
      }
    }
    return items.length;
  }

  /** Sorted on the GPU, then drawn over `sceneColor`, behind what `depth` holds. */
  addPasses(graph, { sceneColor, depth }) {
    if (this._items.length === 0) return;
    // One of the sorted orders stands for all of them: it is what puts the draw after the sort.
    const order = graph.importBuffer('splat-order', this._items[0].state.order);
    graph.addPass({ name: 'splats:sort', type: 'compute', writes: [order], execute: this._sort });
    graph.addPass({ name: 'splats', reads: [order], color: [{ resource: sceneColor }], depth: { resource: depth }, execute: this._draw });
  }

  _stateFor(record, splats) {
    let state = this._states.get(record);
    if (state !== undefined && state.splats === splats) return state;
    if (state !== undefined) destroyState(state);
    const rhi = this.rhi;
    const n = Math.max(1, splats.count);
    const make = (label, size, usage = GPUBufferUsage.STORAGE) => createBuffer(rhi, { label: `splats-${label}`, size, usage });
    state = {
      splats,
      lastFrame: 0,
      cloud: make('cloud', CLOUD_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      keys: make('keys', n * 4),
      visible: make('visible', n * 4),
      counts: make('counts', BUCKETS * 4),
      cursors: make('cursors', BUCKETS * 4),
      order: make('order', n * 4),
      args: make('args', 16, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST),
    };
    const device = rhi.device;
    const entries = (list) => list.map((buffer, binding) => ({ binding, resource: { buffer } }));
    state.sortGroup = device.createBindGroup({
      label: 'splats-sort',
      layout: this.sortLayout,
      entries: entries([state.cloud, splats.centers, state.keys, state.visible, state.counts, state.cursors, state.order, state.args]),
    });
    state.drawGroup = device.createBindGroup({
      label: 'splats-draw',
      layout: this.drawLayout,
      entries: entries([state.cloud, splats.centers, splats.covariances, state.order]),
    });
    this._states.set(record, state);
    return state;
  }

  _encodeSort(pass) {
    const limit = this.rhi.limits.maxComputeWorkgroupsPerDimension ?? 65535;
    for (const { state, count } of this._items) {
      const groups = Math.ceil(count / WORKGROUP);
      const x = Math.min(groups, limit);
      const y = Math.ceil(groups / x);
      pass.setBindGroup(0, state.sortGroup);
      pass.setPipeline(this.keys);
      pass.dispatchWorkgroups(x, y);
      pass.setPipeline(this.scan);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.scatter);
      pass.dispatchWorkgroups(x, y);
    }
  }

  _encodeDraw(pass) {
    pass.setPipeline(this.draw);
    for (const { state } of this._items) {
      pass.setBindGroup(0, state.drawGroup);
      pass.drawIndirect(state.args, 0);
    }
  }

  destroy() {
    for (const state of this._states.values()) destroyState(state);
    this._states.clear();
  }
}

function destroyState(state) {
  for (const name of ['cloud', 'keys', 'visible', 'counts', 'cursors', 'order', 'args']) state[name].destroy();
}
