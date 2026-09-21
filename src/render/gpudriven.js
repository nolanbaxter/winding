// GPU-driven rendering: compute culling into indirect draw arguments.
//
// The GPU decides what to draw. A compute shader tests every object's bounds
// against the frustum and writes the survivors into draw arguments the CPU
// never reads.
//
// Three things make that possible:
//
// PER-DRAW DATA MOVES TO A STORAGE BUFFER. Dynamic uniform offsets must be a
// multiple of minUniformBufferOffsetAlignment -- 256 on this hardware -- so the
// 112-byte draw block was occupying 256 bytes. Measured at the time: 69% of
// that buffer was padding. A storage buffer indexed by the shader has no such
// rule, so 112 bytes costs 112 bytes, AND the index can come from the GPU.
//
// DRAWS BECOME INSTANCED BATCHES. Everything sharing a primitive and a material
// is one batch, drawn once with an instance count. That instance count is the
// only thing the indirect path actually needs to write.
//
// THE COUNT IS AN ATOMIC IN THE ARGUMENT BUFFER ITSELF. The cull shader does
// atomicAdd on instanceCount and gets back the slot to write its index into --
// compaction and counting in one operation, with no prefix sum.
//
// What WebGPU does not have yet is multi-draw-indirect, so the CPU still issues
// one drawIndexedIndirect per batch. The batch COUNT is CPU-side; which objects
// are in it, and how many, are not. A fully culled batch costs one call with
// instanceCount 0, which the GPU discards immediately.

import { DEBUG, assert } from '../core/assert.js';
import { compileShader } from '../rhi/shader.js';
import { grownCapacity } from '../core/grow.js';
import { FRUSTUM_PLANE_COUNT } from '../core/math/frustum.js';

/** model mat4 (64) + normal mat3x3 (48). No alignment padding in storage. */
export const DRAW_DATA_BYTES = 112;
/** indexCount, instanceCount, firstIndex, baseVertex, firstInstance */
export const INDIRECT_BYTES = 20;
/** Per-batch uniform: where this batch's slice of the visible list starts. */
export const BATCH_BYTES = 16;

/**
 * itemBatch entry for a renderable that belongs to no batch, which today means
 * blended geometry. Mirrored as NOT_BATCHED in the cull shader below; the two
 * must agree.
 */
export const NOT_BATCHED = 0xffffffff;

const WORKGROUP_SIZE = 64;


const CULL_SHADER = /* wgsl */ `
struct CullParams {
  planes        : array<vec4<f32>, ${FRUSTUM_PLANE_COUNT}>,
  // LAST frame's viewProjection. The pyramid was built from last frame's depth,
  // so the test has to project into the same space that produced it.
  lastViewProj  : mat4x4<f32>,
  count         : u32,
  hzbLevels     : u32,
  hzbWidth      : f32,
  hzbHeight     : f32,
};

struct Bounds {
  minPoint : vec4<f32>,
  maxPoint : vec4<f32>,
};

struct DrawArgs {
  indexCount    : u32,
  instanceCount : atomic<u32>,
  firstIndex    : u32,
  baseVertex    : i32,
  firstInstance : u32,
};

/** itemBatch entry for a renderable that belongs to no batch. */
const NOT_BATCHED : u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform>             params     : CullParams;
@group(0) @binding(1) var<storage, read>       bounds     : array<Bounds>;
@group(0) @binding(2) var<storage, read>       itemBatch  : array<u32>;
@group(0) @binding(3) var<storage, read>       batchFirst : array<u32>;
@group(0) @binding(4) var<storage, read_write> indirect   : array<DrawArgs>;
@group(0) @binding(5) var<storage, read_write> visible    : array<u32>;
@group(0) @binding(6) var                      hzb        : texture_2d<f32>;

/**
 * Is this box hidden behind something drawn last frame?
 *
 * Projects the eight corners, takes the screen rectangle and the NEAREST depth,
 * picks the pyramid level where that rectangle spans about two texels, and
 * compares against the four texels covering it.
 *
 * Returns false whenever it cannot be sure -- behind the camera, straddling the
 * near plane, off the edge of the pyramid. Wrongly keeping an object costs one
 * draw; wrongly culling one leaves a hole.
 */
fn occluded(boxMin : vec3<f32>, boxMax : vec3<f32>) -> bool {
  var minUV = vec2<f32>(1e30, 1e30);
  var maxUV = vec2<f32>(-1e30, -1e30);
  var nearest = 0.0;

  for (var c = 0u; c < 8u; c = c + 1u) {
    let corner = vec3<f32>(
      select(boxMin.x, boxMax.x, (c & 1u) != 0u),
      select(boxMin.y, boxMax.y, (c & 2u) != 0u),
      select(boxMin.z, boxMax.z, (c & 4u) != 0u),
    );
    let clip = params.lastViewProj * vec4<f32>(corner, 1.0);
    // w <= 0 means the box reaches behind the eye; the projection is not
    // meaningful there and the safe answer is "visible".
    if (clip.w <= 0.0) { return false; }

    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    minUV = min(minUV, uv);
    maxUV = max(maxUV, uv);
    // Reverse-Z: nearer is LARGER, so the box's nearest point is the max.
    nearest = max(nearest, ndc.z);
  }

  if (any(maxUV < vec2<f32>(0.0)) || any(minUV > vec2<f32>(1.0))) { return false; }
  minUV = clamp(minUV, vec2<f32>(0.0), vec2<f32>(1.0));
  maxUV = clamp(maxUV, vec2<f32>(0.0), vec2<f32>(1.0));

  let size = (maxUV - minUV) * vec2<f32>(params.hzbWidth, params.hzbHeight);
  let level = clamp(
    i32(ceil(log2(max(max(size.x, size.y), 1.0)))),
    0, i32(params.hzbLevels) - 1,
  );

  let levelSize = vec2<f32>(
    max(params.hzbWidth / f32(1 << u32(level)), 1.0),
    max(params.hzbHeight / f32(1 << u32(level)), 1.0),
  );
  let lo = vec2<i32>(minUV * levelSize);
  let hi = min(vec2<i32>(maxUV * levelSize), vec2<i32>(levelSize) - vec2<i32>(1, 1));

  // The pyramid stores the FARTHEST nearby surface, so this is the weakest
  // occluder in the region. If the box is behind even that, it is behind all.
  var farthest = 1e30;
  farthest = min(farthest, textureLoad(hzb, vec2<i32>(lo.x, lo.y), level).r);
  farthest = min(farthest, textureLoad(hzb, vec2<i32>(hi.x, lo.y), level).r);
  farthest = min(farthest, textureLoad(hzb, vec2<i32>(lo.x, hi.y), level).r);
  farthest = min(farthest, textureLoad(hzb, vec2<i32>(hi.x, hi.y), level).r);

  return nearest < farthest;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn cull(@builtin(global_invocation_id) id : vec3<u32>) {
  let item = id.x;
  if (item >= params.count) { return; }

  let boxMin = bounds[item].minPoint.xyz;
  let boxMax = bounds[item].maxPoint.xyz;

  // The same positive-vertex test the CPU path uses, and the same FIVE planes:
  // the sixth would be the far plane, which does not exist under an infinite
  // reverse-Z projection.
  for (var p = 0u; p < ${FRUSTUM_PLANE_COUNT}u; p = p + 1u) {
    let plane = params.planes[p];
    let corner = vec3<f32>(
      select(boxMin.x, boxMax.x, plane.x >= 0.0),
      select(boxMin.y, boxMax.y, plane.y >= 0.0),
      select(boxMin.z, boxMax.z, plane.z >= 0.0),
    );
    if (dot(plane.xyz, corner) + plane.w < 0.0) { return; }
  }

  if (occluded(boxMin, boxMax)) { return; }

  let batch = itemBatch[item];
  // Blended geometry is not batched and not culled here: its draw order has to
  // be back-to-front, and the atomic below hands out slots in whatever order
  // threads finish. The CPU orders those items instead.
  if (batch == NOT_BATCHED) { return; }

  // Claim a slot and record the count in one operation. The returned value is
  // this thread's index within the batch, so no second compaction pass and no
  // prefix sum is needed -- the atomic IS the allocator.
  let slot = atomicAdd(&indirect[batch].instanceCount, 1u);
  visible[batchFirst[batch] + slot] = item;
}
`;

/**
 * Groups renderables into instanced batches and culls them on the GPU.
 *
 * Batches are rebuilt only when the scene's contents change, which the scene
 * reports through a revision counter -- rebuilding every frame would put an
 * O(n log n) sort back in the hot path that this whole file exists to remove.
 */
export class GpuDriven {
  static async create(rhi, capacity, materials) {
    const gpu = new GpuDriven(rhi, capacity, materials);
    await gpu._init();
    return gpu;
  }

  constructor(rhi, capacity, materials) {
    this.rhi = rhi;
    this.capacity = capacity;
    /** Read for isTransparent() only: which renderables skip the batched path. */
    this.materials = materials;
    this.batchCount = 0;
    this.sceneRevision = -1;

    const device = rhi.device;

    // Per-object data, indexed by the shader rather than bound per draw.
    this.drawData = new Float32Array(capacity * (DRAW_DATA_BYTES / 4));
    this.drawDataBuffer = device.createBuffer({
      label: 'draw-data',
      size: capacity * DRAW_DATA_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.boundsData = new Float32Array(capacity * 8);
    this.boundsBuffer = device.createBuffer({
      label: 'cull-bounds',
      size: capacity * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.itemBatch = new Uint32Array(capacity);
    this.itemBatchBuffer = device.createBuffer({
      label: 'item-batch',
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.batchFirst = new Uint32Array(capacity);
    this.batchFirstBuffer = device.createBuffer({
      label: 'batch-first',
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.indirectData = new Uint32Array(capacity * (INDIRECT_BYTES / 4));
    this.indirectBuffer = device.createBuffer({
      label: 'indirect-args',
      size: capacity * INDIRECT_BYTES,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.visibleBuffer = device.createBuffer({
      label: 'visible-items',
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // The shadow pass draws every caster, so it indexes a static list in batch
    // order rather than the GPU-compacted one.
    this.batchOrder = new Uint32Array(capacity);
    this.batchOrderBuffer = device.createBuffer({
      label: 'batch-order',
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Per-batch uniform: the base index into the visible list. Bound with a
    // dynamic offset, which is the one small thing still done per draw.
    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    // capacity + 1: the extra slot holds a firstVisible of 0, which is what the
    // transparent draws bind. They index the visible list absolutely, by
    // firstInstance, rather than relative to a batch base.
    this.batchStaging = new ArrayBuffer(this.alignment * (capacity + 1));
    this.batchBuffer = device.createBuffer({
      label: 'batch-info',
      size: this.batchStaging.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // planes(5 * 16) + lastViewProj(64) + (count, hzbLevels, hzbWidth, hzbHeight)
    this.cullParams = new ArrayBuffer(FRUSTUM_PLANE_COUNT * 16 + 64 + 16);
    this.lastViewProjection = new Float32Array(16);
    this.lastViewProjection[0] = 1; this.lastViewProjection[5] = 1;
    this.lastViewProjection[10] = 1; this.lastViewProjection[15] = 1;
    this.cullParamsF32 = new Float32Array(this.cullParams);
    this.cullParamsU32 = new Uint32Array(this.cullParams);
    this.cullParamsBuffer = device.createBuffer({
      label: 'cull-params',
      size: this.cullParams.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    /** primitive + materialId per batch, in CPU-side arrays. */
    this.batchPrimitive = [];
    this.batchMaterial = new Uint16Array(capacity);
    this.batchSize = new Uint32Array(capacity);

    // Blended renderables, which never enter a batch. See rebuildBatches.
    this.transparentItems = new Uint32Array(capacity);
    this.transparentCount = 0;
    /** Slots [0, opaqueCount) of the visible list belong to the cull shader. */
    this.opaqueCount = 0;

    this._cullExecute = (pass) => this._dispatch(pass);
    this._needsFullUpload = true;
    /** Bumped by _grow. The renderer's draw bind group names batchBuffer. */
    this.buffersRevision = 0;
    this.stats = { batches: 0, items: 0, uploaded: 0, transparent: 0 };
  }

  async _init() {
    const device = this.rhi.device;
    const shader = await compileShader(device, CULL_SHADER, 'cull.wgsl');

    this.layout = device.createBindGroupLayout({
      label: 'gpu-cull',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      ],
    });

    this.pipeline = device.createComputePipeline({
      label: 'gpu-cull',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module: shader.module, entryPoint: 'cull' },
    });

    this._makeBindGroup = (hzbView) => device.createBindGroup({
      label: 'gpu-cull',
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.cullParamsBuffer } },
        { binding: 1, resource: { buffer: this.boundsBuffer } },
        { binding: 2, resource: { buffer: this.itemBatchBuffer } },
        { binding: 3, resource: { buffer: this.batchFirstBuffer } },
        { binding: 4, resource: { buffer: this.indirectBuffer } },
        { binding: 5, resource: { buffer: this.visibleBuffer } },
        { binding: 6, resource: hzbView },
      ],
    });
  }

  /**
   * Widen every per-renderable buffer.
   *
   * Contents are deliberately NOT copied. Everything here is rewritten from the
   * scene before it is next read: draw data and bounds by update(), the batch
   * tables by the rebuild this is called from, and the visible list by the cull
   * shader. Copying would be work whose result is immediately overwritten.
   */
  _grow(needed) {
    const capacity = grownCapacity(this.capacity, needed);
    const device = this.rhi.device;
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.drawData = new Float32Array(capacity * (DRAW_DATA_BYTES / 4));
    this.boundsData = new Float32Array(capacity * 8);
    this.indirectData = new Uint32Array(capacity * (INDIRECT_BYTES / 4));
    this.itemBatch = new Uint32Array(capacity);
    this.batchFirst = new Uint32Array(capacity);
    this.batchOrder = new Uint32Array(capacity);
    this.batchMaterial = new Uint16Array(capacity);
    this.batchSize = new Uint32Array(capacity);
    this.transparentItems = new Uint32Array(capacity);
    this.batchStaging = new ArrayBuffer(this.alignment * (capacity + 1));

    for (const buffer of [
      this.drawDataBuffer, this.boundsBuffer, this.itemBatchBuffer, this.batchFirstBuffer,
      this.indirectBuffer, this.visibleBuffer, this.batchOrderBuffer, this.batchBuffer,
    ]) buffer.destroy();

    this.drawDataBuffer = device.createBuffer({ label: 'draw-data', size: capacity * DRAW_DATA_BYTES, usage: STORAGE });
    this.boundsBuffer = device.createBuffer({ label: 'cull-bounds', size: capacity * 32, usage: STORAGE });
    this.itemBatchBuffer = device.createBuffer({ label: 'item-batch', size: capacity * 4, usage: STORAGE });
    this.batchFirstBuffer = device.createBuffer({ label: 'batch-first', size: capacity * 4, usage: STORAGE });
    this.indirectBuffer = device.createBuffer({
      label: 'indirect-args', size: capacity * INDIRECT_BYTES,
      usage: GPUBufferUsage.INDIRECT | STORAGE,
    });
    this.visibleBuffer = device.createBuffer({ label: 'visible-items', size: capacity * 4, usage: STORAGE });
    this.batchOrderBuffer = device.createBuffer({ label: 'batch-order', size: capacity * 4, usage: STORAGE });
    this.batchBuffer = device.createBuffer({
      label: 'batch-info', size: this.batchStaging.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.capacity = capacity;
    this._needsFullUpload = true;
    // The cull bind group names the buffers that were just destroyed, and so
    // does the renderer's draw bind group -- which this object does not own.
    // bindHzb builds this if the pyramid has not been bound yet, so growing
    // before the first frame is not an ordering error.
    if (this._hzbView) this.bindGroup = this._makeBindGroup(this._hzbView);
    this.buffersRevision++;
  }

  /** Rebuild the bind group when the pyramid is recreated on resize. */
  bindHzb(hzbView) {
    if (this._hzbView === hzbView) return;
    this._hzbView = hzbView;
    this.bindGroup = this._makeBindGroup(hzbView);
  }

  /**
   * Group renderables sharing a primitive and a material.
   *
   * Only runs when the scene actually changed. `scene.revision` is bumped by
   * add and remove, so a scene that is merely moving never pays for this.
   */
  rebuildBatches(scene) {
    const count = scene.renderableCount;
    // Every buffer here is sized per renderable, so a scene that outgrew its
    // start has to bring these with it. Without this the scene's own growth
    // would just relocate the failure into a writeBuffer validation error.
    if (count > this.capacity) this._grow(count);

    // Key on the primitive object plus material id. The primitive is a stable
    // reference from the asset, so identity is the right comparison.
    const batchOf = new Map();
    this.batchPrimitive.length = 0;
    this.batchCount = 0;
    this.transparentCount = 0;

    for (let i = 0; i < count; i++) {
      const material = scene.renderableMaterial[i];

      // Blended geometry takes the ordered path instead. Batching exists to
      // merge draws that can run in any order, which is exactly what blending
      // is not -- so these are held out here rather than filtered downstream.
      if (this.materials.isTransparent(material)) {
        this.itemBatch[i] = NOT_BATCHED;
        this.transparentItems[this.transparentCount++] = i;
        continue;
      }

      const primitive = scene.renderablePrimitive[i];
      const key = `${primitiveId(primitive)}:${material}`;

      let batch = batchOf.get(key);
      if (batch === undefined) {
        batch = this.batchCount++;
        batchOf.set(key, batch);
        this.batchPrimitive.push(primitive);
        this.batchMaterial[batch] = material;
        this.batchSize[batch] = 0;
      }
      this.itemBatch[i] = batch;
      this.batchSize[batch]++;
    }

    // Prefix sum gives each batch a contiguous slice of the visible list. It
    // covers only the batched items, which leaves the tail of that list --
    // exactly transparentCount slots -- free for the ordered draws.
    let running = 0;
    for (let b = 0; b < this.batchCount; b++) {
      this.batchFirst[b] = running;
      running += this.batchSize[b];
    }
    this.opaqueCount = running;

    // Static draw order for the shadow pass, which culls nothing. Blended
    // geometry is absent from it, so it casts no shadow -- the honest result
    // for glass, and the alternative is an opaque silhouette that is wrong in
    // a more visible way.
    const cursor = new Uint32Array(Math.max(this.batchCount, 1));
    for (let i = 0; i < count; i++) {
      const batch = this.itemBatch[i];
      if (batch === NOT_BATCHED) continue;
      this.batchOrder[this.batchFirst[batch] + cursor[batch]++] = i;
    }

    // Per-batch uniform holding that slice's base index.
    for (let b = 0; b < this.batchCount; b++) {
      new Uint32Array(this.batchStaging, b * this.alignment, 1)[0] = this.batchFirst[b];
    }
    // The transparent draws' base, always 0: they address the visible list
    // through firstInstance, which a direct draw may set freely.
    this.transparentBatchSlot = this.batchCount;
    new Uint32Array(this.batchStaging, this.transparentBatchSlot * this.alignment, 1)[0] = 0;

    const queue = this.rhi.queue;
    queue.writeBuffer(this.itemBatchBuffer, 0, this.itemBatch, 0, count);
    queue.writeBuffer(this.batchFirstBuffer, 0, this.batchFirst, 0, Math.max(this.batchCount, 1));
    queue.writeBuffer(this.batchOrderBuffer, 0, this.batchOrder, 0, Math.max(this.opaqueCount, 1));
    queue.writeBuffer(this.batchBuffer, 0, this.batchStaging, 0, this.alignment * (this.batchCount + 1));

    this._needsFullUpload = true;
    this.sceneRevision = scene.revision;
    this.stats.batches = this.batchCount;
    this.stats.items = count;
    this.stats.transparent = this.transparentCount;
  }

  /** Upload this frame's transforms, bounds and reset argument buffer. */
  update(scene, frustum, hzb, writeDrawData) {
    if (scene.revision !== this.sceneRevision) this.rebuildBatches(scene);

    const count = scene.renderableCount;

    // Only objects whose transform actually recomposed need rewriting. The
    // hierarchy already worked out which ones those are, and rewriting all of
    // them throws that answer away.
    //
    // The touched span is tracked rather than a per-object upload list: one
    // writeBuffer over a range beats hundreds of small ones, and a scene where
    // movers are scattered to both ends degrades to what this did before
    // rather than to something worse.
    const moved = scene.transforms.moved;
    const full = this._needsFullUpload;
    let low = count;
    let high = -1;

    for (let i = 0; i < count; i++) {
      if (!full && moved[scene.renderableMatrixSlot[i]] === 0) continue;

      writeDrawData(this.drawData, i * (DRAW_DATA_BYTES / 4), scene, i);

      const b = i * 8;
      const o = i * 3;
      this.boundsData[b] = scene.worldMin[o];
      this.boundsData[b + 1] = scene.worldMin[o + 1];
      this.boundsData[b + 2] = scene.worldMin[o + 2];
      this.boundsData[b + 4] = scene.worldMax[o];
      this.boundsData[b + 5] = scene.worldMax[o + 1];
      this.boundsData[b + 6] = scene.worldMax[o + 2];

      if (i < low) low = i;
      if (i > high) high = i;
    }
    this._needsFullUpload = false;
    this.stats.uploaded = high >= low ? high - low + 1 : 0;

    // Draw arguments, with instanceCount zeroed. The cull shader raises it with
    // atomicAdd, so resetting here is what makes the frame idempotent -- forget
    // it and counts accumulate until every batch draws the whole scene.
    for (let b = 0; b < this.batchCount; b++) {
      const primitive = this.batchPrimitive[b];
      const o = b * 5;
      this.indirectData[o] = primitive.indexCount;
      this.indirectData[o + 1] = 0;
      this.indirectData[o + 2] = 0;
      this.indirectData[o + 3] = 0;
      this.indirectData[o + 4] = 0;
    }

    const planeFloats = FRUSTUM_PLANE_COUNT * 4;
    this.cullParamsF32.set(frustum, 0);
    this.cullParamsF32.set(this.lastViewProjection, planeFloats);
    this.cullParamsU32[planeFloats + 16] = count;
    this.cullParamsU32[planeFloats + 17] = hzb.levelCount;
    this.cullParamsF32[planeFloats + 18] = hzb.width;
    this.cullParamsF32[planeFloats + 19] = hzb.height;

    const queue = this.rhi.queue;
    if (high >= low) {
      const span = high - low + 1;
      const drawFloats = DRAW_DATA_BYTES / 4;
      queue.writeBuffer(
        this.drawDataBuffer, low * DRAW_DATA_BYTES,
        this.drawData, low * drawFloats, span * drawFloats,
      );
      queue.writeBuffer(
        this.boundsBuffer, low * 32, this.boundsData, low * 8, span * 8,
      );
    }
    queue.writeBuffer(this.indirectBuffer, 0, this.indirectData, 0, Math.max(this.batchCount, 1) * 5);
    queue.writeBuffer(this.cullParamsBuffer, 0, this.cullParams);
    this.itemCount = count;
  }

  addCullPass(graph, { boundsResource, indirectResource, visibleResource }) {
    graph.addPass({
      name: 'gpu-cull',
      type: 'compute',
      reads: [boundsResource],
      writes: [indirectResource, visibleResource],
      execute: this._cullExecute,
    });
  }

  _dispatch(pass) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.itemCount / WORKGROUP_SIZE));
  }

  /** Dynamic offset for the zero-base slot the ordered draws bind. */
  transparentBatchOffset() {
    return this.transparentBatchSlot * this.alignment;
  }

  /**
   * Publish the back-to-front order for this frame's blended draws.
   *
   * They live in the tail of the visible list, past everything the cull shader
   * can reach, so this never races the compute pass for a slot.
   */
  writeTransparentOrder(indices, count) {
    if (count === 0) return;
    this.rhi.queue.writeBuffer(
      this.visibleBuffer, this.opaqueCount * 4, indices, 0, count,
    );
  }

  batchOffset(batch) {
    return batch * this.alignment;
  }

  destroy() {
    for (const buffer of [
      this.drawDataBuffer, this.boundsBuffer, this.itemBatchBuffer, this.batchFirstBuffer,
      this.indirectBuffer, this.visibleBuffer, this.batchOrderBuffer, this.batchBuffer,
      this.cullParamsBuffer,
    ]) buffer.destroy();
  }
}

// Primitives have no identity of their own, so one is stamped on first use.
let nextPrimitiveId = 1;
function primitiveId(primitive) {
  if (!primitive.__batchId) primitive.__batchId = nextPrimitiveId++;
  return primitive.__batchId;
}
