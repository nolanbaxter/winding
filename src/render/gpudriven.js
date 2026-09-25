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
import { sharedPipelines } from '../rhi/pipeline.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { grownCapacity, growArray } from '../core/grow.js';
import { storageCapacity, createBuffer } from '../rhi/buffer.js';
import { FRUSTUM_PLANE_COUNT } from '../core/math/frustum.js';

/**
 * model mat4 (64) + normal mat3x3 (48) + four u32 (16), rounded to the
 * struct's 16-byte alignment. 128.
 */
export const DRAW_DATA_BYTES = 128;
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

/**
 * What one queue.writeBuffer call costs, as the bytes it could have copied in
 * the same time. MEASURED in Chrome: ~0.7 us per call against ~1.3 ns per
 * byte, so about 512. Two runs of changed renderables separated by less than
 * this are sent as one, gap included -- cheaper than a second call.
 */
const WRITE_CALL_BYTES = 512;

/** Bytes one renderable adds to an upload: its draw data and its cull box. */
const UPLOAD_ITEM_BYTES = DRAW_DATA_BYTES + 32;

/**
 * Add item `i` to `runs`, a flat [first, last, first, last, ...] list, in
 * increasing order. It extends the last run when the gap to it is at most
 * `mergeGap` items -- the gap is uploaded too, being cheaper than a call.
 */
export function addToRuns(runs, i, mergeGap) {
  const last = runs.length - 1;
  if (last > 0 && i - runs[last] - 1 <= mergeGap) runs[last] = i;
  else runs.push(i, i);
}


/**
 * Cull phases, each writing its own slice of the indirect arguments and the
 * visible list: early (last frame's visible set), late (against this frame's
 * depth), and shadow (every caster at the level of detail the camera chose,
 * which runs only when the scene has LOD groups).
 */
export const CULL_PHASES = 3;
export const CULL_SHADOW = 2;

/** An item's bounds: min, max (w = coverage range), and its LOD sphere. */
const BOUNDS_FLOATS = 12;

const CULL_SHADER = /* wgsl */ `
struct CullParams {
  planes        : array<vec4<f32>, ${FRUSTUM_PLANE_COUNT}>,
  // THIS frame's viewProjection. The late phase tests against a pyramid built
  // from this frame's own early depth, so there is no lag to compensate for --
  // which is the entire point of culling in two phases.
  viewProj      : mat4x4<f32>,
  count         : u32,
  hzbLevels     : u32,
  hzbWidth      : f32,
  hzbHeight     : f32,
  // 0 = early (redraw last frame's set), 1 = late (test against fresh depth).
  phase         : u32,
  // Where this phase's slice of the indirect and visible lists begins. The two
  // phases share both buffers and never touch each other's half.
  indirectBase  : u32,
  visibleBase   : u32,
  // The projection's y scale: 1 / tan(fovY / 2), or 1 / half-height for an
  // orthographic camera. With clip w it gives a sphere's size on screen.
  projectionScale : f32,
};

/**
 * An item's box, and its level of detail. minPoint.w and maxPoint.w are the
 * screen coverage it draws between; lod is its LOD group's bounding sphere,
 * world centre and radius, radius 0 for an item that is in no group.
 */
struct Bounds {
  minPoint : vec4<f32>,
  maxPoint : vec4<f32>,
  lod      : vec4<f32>,
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

@group(0) @binding(0) var<uniform>             params      : CullParams;
@group(0) @binding(1) var<storage, read>       bounds      : array<Bounds>;
@group(0) @binding(2) var<storage, read>       itemBatch   : array<u32>;
@group(0) @binding(3) var<storage, read>       batchFirst  : array<u32>;
@group(0) @binding(4) var<storage, read_write> indirect    : array<DrawArgs>;
@group(0) @binding(5) var<storage, read_write> visible     : array<u32>;
@group(0) @binding(6) var                      hzb         : texture_2d<f32>;
/**
 * Was this object drawn last frame? Persistent across frames and owned by the
 * late phase, which is the only one that can tell.
 */
@group(0) @binding(7) var<storage, read_write> visibleLast : array<u32>;

/**
 * Is this box hidden behind something already drawn this frame?
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
    let clip = params.viewProj * vec4<f32>(corner, 1.0);
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

/**
 * Whether this is the level of detail its group shows. Coverage is the group
 * sphere's diameter on screen over the screen's height: radius * scale / w,
 * which is exact for a perspective camera and an orthographic one alike --
 * w is the view depth for one and 1 for the other. A group behind the eye
 * counts as filling the screen, so its finest level stands.
 */
fn lodSelected(item : u32) -> bool {
  let sphere = bounds[item].lod;
  if (sphere.w <= 0.0) { return true; }
  let w = (params.viewProj * vec4<f32>(sphere.xyz, 1.0)).w;
  let coverage = select(3.0e38, sphere.w * params.projectionScale / w, w > 0.0);
  return coverage >= bounds[item].minPoint.w && coverage < bounds[item].maxPoint.w;
}

/** The same positive-vertex test the CPU path uses, over the same five planes. */
fn inFrustum(boxMin : vec3<f32>, boxMax : vec3<f32>) -> bool {
  for (var p = 0u; p < ${FRUSTUM_PLANE_COUNT}u; p = p + 1u) {
    let plane = params.planes[p];
    let corner = vec3<f32>(
      select(boxMin.x, boxMax.x, plane.x >= 0.0),
      select(boxMin.y, boxMax.y, plane.y >= 0.0),
      select(boxMin.z, boxMax.z, plane.z >= 0.0),
    );
    if (dot(plane.xyz, corner) + plane.w < 0.0) { return false; }
  }
  return true;
}

/**
 * Claim a slot and record the count in one operation. The returned value is
 * this thread's index within the batch, so no second compaction pass and no
 * prefix sum is needed -- the atomic IS the allocator.
 */
fn emit(batch : u32, item : u32) {
  let slot = atomicAdd(&indirect[params.indirectBase + batch].instanceCount, 1u);
  visible[params.visibleBase + batchFirst[batch] + slot] = item;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn cull(@builtin(global_invocation_id) id : vec3<u32>) {
  // y carries the workgroups past the 65,535 a dimension may hold; see _dispatch.
  let item = id.y * (65535u * ${WORKGROUP_SIZE}u) + id.x;
  if (item >= params.count) { return; }

  let boxMin = bounds[item].minPoint.xyz;
  let boxMax = bounds[item].maxPoint.xyz;
  let batch = itemBatch[item];
  // Blended geometry is not batched and not culled here: its draw order has to
  // be back-to-front, and the atomic hands out slots in thread-completion
  // order. The CPU orders those items instead.
  let batched = batch != NOT_BATCHED;

  if (params.phase == 2u) {
    // SHADOWS. Every caster, on screen or not -- one off to the side still
    // casts onto what is -- but only at the level of detail the camera chose,
    // or a group would cast every one of its levels at once.
    if (batched && lodSelected(item)) { emit(batch, item); }
    return;
  }

  let visibleNow = inFrustum(boxMin, boxMax) && lodSelected(item);

  if (params.phase == 0u) {
    // EARLY. Draw whatever was on screen last frame, with no depth test at all:
    // these are precisely the objects the pyramid is about to be built from, so
    // testing them against last frame's pyramid would be the stale test this
    // whole scheme exists to delete. An object that has since become hidden is
    // drawn once too often; it is never missing.
    if (visibleNow && batched && visibleLast[item] == 1u) { emit(batch, item); }
    return;
  }

  // LATE. The pyramid now holds this frame's early depth, so this test is
  // current rather than a frame behind.
  let survives = visibleNow && !occluded(boxMin, boxMax);
  let drawnEarly = visibleLast[item] == 1u;
  // Read before write: the early phase of the NEXT frame reads what is stored
  // here, and this thread still needs the old value to know what it drew.
  visibleLast[item] = select(0u, 1u, survives);

  if (survives && batched && !drawnEarly) { emit(batch, item); }
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
    /**
     * Capacity of everything indexed by BATCH, which grows on its own.
     * batchCount <= renderableCount is a bound so loose it is the point of
     * batching, so sizing these per renderable wasted most of it.
     */
    this.batchCapacity = Math.min(capacity, 256);
    /** Read for isTransparent() only: which renderables skip the batched path. */
    this.materials = materials;
    this.batchCount = 0;
    // Which scene these batches were built from, and how many changes into
    // it. BOTH, because revision alone does not identify anything: it counts
    // changes within one scene and starts at zero in every scene.
    this.sceneId = -1;
    this.sceneRevision = -1;

    const device = rhi.device;

    // Per-object data, indexed by the shader rather than bound per draw.
    this._allocateDrawData(capacity);
    this.drawDataBuffer = createBuffer(rhi, {
      label: 'draw-data',
      size: capacity * DRAW_DATA_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.boundsData = new Float32Array(capacity * BOUNDS_FLOATS);
    this.boundsBuffer = createBuffer(rhi, {
      label: 'cull-bounds',
      size: capacity * BOUNDS_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.itemBatch = new Uint32Array(capacity);
    this.itemMirrored = new Uint8Array(capacity);
    this.itemBatchBuffer = createBuffer(rhi, {
      label: 'item-batch',
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.batchFirst = new Uint32Array(this.batchCapacity);
    this.batchFirstBuffer = createBuffer(rhi, {
      label: 'batch-first',
      size: this.batchCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.indirectData = new Uint32Array(this.batchCapacity * CULL_PHASES * (INDIRECT_BYTES / 4));
    this.indirectBuffer = createBuffer(rhi, {
      label: 'indirect-args',
      size: this.batchCapacity * CULL_PHASES * INDIRECT_BYTES,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // One slice per cull phase, which write disjoint slices of it. Phase p's
    // batch b lives at `p * capacity + batchFirst[b]`, and the blended tail
    // sits past opaqueCount inside the early slice.
    this.visibleBuffer = createBuffer(rhi, {
      label: 'visible-items',
      size: capacity * CULL_PHASES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // One flag per object: was it drawn last frame? Written only by the late
    // phase, which is the only one that has tested against current depth.
    // Persistent -- this is the entire memory two-phase culling carries between
    // frames, and it replaces trusting a frame-old pyramid.
    this.visibleFlagsBuffer = createBuffer(rhi, {
      label: 'visible-last-frame',
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._zeroFlags = new Uint32Array(capacity);

    // The shadow pass draws every caster, so it indexes a static list in batch
    // order rather than the GPU-compacted one.
    this.batchOrder = new Uint32Array(capacity);
    this.batchOrderBuffer = createBuffer(rhi, {
      label: 'batch-order',
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Per-batch uniform: the base index into the visible list. Bound with a
    // dynamic offset, which is the one small thing still done per draw.
    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    // One slot per (phase, batch), plus two for the blended draws, which
    // index the visible list absolutely, by firstInstance: a base of 0 for the
    // camera's, and the shadow slice's base for the shadow pass's.
    this.batchStaging = new ArrayBuffer(this.alignment * (this.batchCapacity * CULL_PHASES + 2));
    this.batchBuffer = createBuffer(rhi, {
      label: 'batch-info',
      size: this.batchStaging.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // planes(5 * 16) + viewProj(64) + 4 u32 + (phase, indirectBase, visibleBase, projectionScale)
    this.cullParams = new ArrayBuffer(FRUSTUM_PLANE_COUNT * 16 + 64 + 16 + 16);
    this.cullParamsF32 = new Float32Array(this.cullParams);
    this.cullParamsU32 = new Uint32Array(this.cullParams);
    // One buffer, one slot per phase, so a single dispatch pair needs no
    // rewrite between the two.
    this.cullParamsStride = Math.max(this.alignment, this.cullParams.byteLength);
    this.cullParamsBuffer = createBuffer(rhi, {
      label: 'cull-params',
      size: this.cullParamsStride * CULL_PHASES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    /** primitive + materialId per batch, in CPU-side arrays. */
    this.batchPrimitive = [];
    this.batchMaterial = new Uint16Array(this.batchCapacity);
    this.batchMirrored = new Uint8Array(this.batchCapacity);
    this.batchSkinned = new Uint8Array(this.batchCapacity);
    this.batchSize = new Uint32Array(this.batchCapacity);

    // Blended renderables, which never enter a batch. See rebuildBatches.
    this.transparentItems = new Uint32Array(capacity);
    this.transparentCount = 0;
    /** Per blended item, in transparentItems order: what the shadow pass binds. */
    this.blendedCasters = [];
    /** Per blended item, in transparentItems order: what the shadow pass binds. */
    this.blendedCasters = [];
    /** Slots [0, opaqueCount) of the visible list belong to the cull shader. */
    this.opaqueCount = 0;

    this._cullExecute = [0, 1, 2].map((phase) => (pass) => this._dispatch(pass, phase));
    /** Whether any renderable is in an LOD group, which is when the shadow phase runs. */
    this.hasLod = false;
    this._needsFullUpload = true;
    /** Bumped by _grow. The renderer's draw bind group names batchBuffer. */
    this.buffersRevision = 0;
    this.stats = { batches: 0, items: 0, uploaded: 0, transparent: 0 };
    /** This frame's changed runs, as [first, last, first, last, ...]; reused. */
    this._runs = [];
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
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    this.pipeline = sharedPipelines(device).compute({
      label: 'gpu-cull',
      layout: createPipelineLayout(device, { 0: this.layout }, 'gpu-cull'),
      shader,
      entry: 'cull',
    });

    this._makeBindGroup = (hzbView, phase) => device.createBindGroup({
      label: `gpu-cull:${['early', 'late', 'shadow'][phase]}`,
      layout: this.layout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.cullParamsBuffer,
            offset: phase * this.cullParamsStride,
            size: this.cullParams.byteLength,
          },
        },
        { binding: 1, resource: { buffer: this.boundsBuffer } },
        { binding: 2, resource: { buffer: this.itemBatchBuffer } },
        { binding: 3, resource: { buffer: this.batchFirstBuffer } },
        { binding: 4, resource: { buffer: this.indirectBuffer } },
        { binding: 5, resource: { buffer: this.visibleBuffer } },
        { binding: 6, resource: hzbView },
        { binding: 7, resource: { buffer: this.visibleFlagsBuffer } },
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
  /**
   * The draw-data staging array and its integer view, allocated together.
   *
   * Together because they MUST be: half the struct is floats and half is
   * u32s, so both views are written every frame, and a reallocation that
   * replaced one and not the other would leave the integer writes going into
   * a detached array. They would not throw -- an index inside the old length
   * writes to memory nobody uploads, and an index past it writes nowhere at
   * all -- so every u32 field would read as zero on the GPU.
   *
   * That is not hypothetical. This was two statements at two call sites and
   * _grow updated one of them, which pinned paletteOffset at 0 for every
   * skinned instance added after the draw buffer grew: a second character
   * silently wearing the first one's pose.
   */
  _allocateDrawData(capacity) {
    this.drawData = new Float32Array(capacity * (DRAW_DATA_BYTES / 4));
    this.drawDataU32 = new Uint32Array(this.drawData.buffer);
  }

  _grow(needed) {
    // Draw data is the widest per-object buffer, so it sets the ceiling.
    const capacity = grownCapacity(
      this.capacity, needed, storageCapacity(this.rhi, DRAW_DATA_BYTES), 'renderables',
    );
    const device = this.rhi.device;
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this._allocateDrawData(capacity);
    this.boundsData = new Float32Array(capacity * BOUNDS_FLOATS);
    this.itemBatch = new Uint32Array(capacity);
    this.itemMirrored = new Uint8Array(capacity);
    this.batchOrder = new Uint32Array(capacity);
    this.transparentItems = new Uint32Array(capacity);
    this._zeroFlags = new Uint32Array(capacity);

    for (const buffer of [
      this.drawDataBuffer, this.boundsBuffer, this.itemBatchBuffer,
      this.visibleBuffer, this.batchOrderBuffer, this.visibleFlagsBuffer,
    ]) buffer.destroy();

    this.drawDataBuffer = createBuffer(this.rhi, { label: 'draw-data', size: capacity * DRAW_DATA_BYTES, usage: STORAGE });
    this.boundsBuffer = createBuffer(this.rhi, { label: 'cull-bounds', size: capacity * BOUNDS_FLOATS * 4, usage: STORAGE });
    this.itemBatchBuffer = createBuffer(this.rhi, { label: 'item-batch', size: capacity * 4, usage: STORAGE });
    this.visibleBuffer = createBuffer(this.rhi, { label: 'visible-items', size: capacity * CULL_PHASES * 4, usage: STORAGE });
    // Fresh and therefore all zero: after a grow, item indices have moved and
    // last frame's flags describe objects that are no longer at those slots.
    this.visibleFlagsBuffer = createBuffer(this.rhi, { label: 'visible-last-frame', size: capacity * 4, usage: STORAGE });
    this.batchOrderBuffer = createBuffer(this.rhi, { label: 'batch-order', size: capacity * 4, usage: STORAGE });

    this.capacity = capacity;
    this._needsFullUpload = true;
    // The cull bind group names the buffers that were just destroyed, and so
    // does the renderer's draw bind group -- which this object does not own.
    // bindHzb builds this if the pyramid has not been bound yet, so growing
    // before the first frame is not an ordering error.
    if (this._hzbView) this._rebuildBindGroups();
    this.buffersRevision++;
  }

  /**
   * Widen everything indexed by BATCH.
   *
   * Separate from the renderable capacity, and it has to be. These arrays are
   * addressed by batch id, and batchCount <= renderableCount is a bound so
   * loose it is the whole point of batching -- Sponza's 17 renderables make 14
   * batches, but 100,000 instances of one mesh make one. Sizing them per
   * renderable cost 512 bytes of uniform buffer EACH for a per-batch value,
   * and pushed the batch-info buffer past the default maxBufferSize somewhere
   * above half a million renderables. createBuffer does not throw for that; it
   * returns an invalid buffer and the frame goes black.
   *
   * Called after the batching loop, when batchCount is finally known. Nothing
   * has been written to any of these yet -- the prefix sum and the uploads
   * both come after -- so growing here loses nothing.
   */
  _growBatches(needed) {
    const capacity = grownCapacity(this.batchCapacity, needed);
    const device = this.rhi.device;
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    // These three are written as the batching loop discovers batches, so a
    // grow in the middle of it has to carry them over. Everything below is
    // filled after the loop and can start empty.
    this.batchMaterial = growArray(this.batchMaterial, capacity);
    this.batchMirrored = growArray(this.batchMirrored, capacity);
    this.batchSkinned = growArray(this.batchSkinned, capacity);
    this.batchSize = growArray(this.batchSize, capacity);

    this.batchFirst = new Uint32Array(capacity);
    this.indirectData = new Uint32Array(capacity * CULL_PHASES * (INDIRECT_BYTES / 4));
    // One slot per (phase, batch), plus the blended draws' two.
    this.batchStaging = new ArrayBuffer(this.alignment * (capacity * CULL_PHASES + 2));

    this.batchFirstBuffer.destroy();
    this.indirectBuffer.destroy();
    this.batchBuffer.destroy();

    this.batchFirstBuffer = createBuffer(this.rhi, {
      label: 'batch-first', size: capacity * 4, usage: STORAGE,
    });
    this.indirectBuffer = createBuffer(this.rhi, {
      label: 'indirect-args', size: capacity * CULL_PHASES * INDIRECT_BYTES,
      usage: GPUBufferUsage.INDIRECT | STORAGE,
    });
    this.batchBuffer = createBuffer(this.rhi, {
      label: 'batch-info', size: this.batchStaging.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.batchCapacity = capacity;
    if (this._hzbView) this._rebuildBindGroups();
    this.buffersRevision++;
  }

  /** Rebuild both bind groups when the pyramid is recreated on resize. */
  bindHzb(hzbView) {
    if (this._hzbView === hzbView) return;
    this._hzbView = hzbView;
    this._rebuildBindGroups();
  }

  _rebuildBindGroups() {
    this.bindGroups = [0, 1, 2].map((phase) => this._makeBindGroup(this._hzbView, phase));
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
    this.blendedCasters.length = 0;
    this.hasLod = false;

    for (let i = 0; i < count; i++) {
      const material = scene.renderableMaterial[i];
      if (scene.renderableLodSlot[i] >= 0) this.hasLod = true;
      // Recorded for every item, batched or not: the blended draws are issued
      // one at a time and still need to know their winding.
      const mirrored = isMirrored(scene.transforms.world, scene.renderableMatrixSlot[i] * 16);
      this.itemMirrored[i] = mirrored ? 1 : 0;

      // Blended geometry takes the ordered path instead. Batching exists to
      // merge draws that can run in any order, which is exactly what blending
      // is not -- so these are held out here rather than filtered downstream.
      if (this.materials.isTransparent(material)) {
        this.itemBatch[i] = NOT_BATCHED;
        this.transparentItems[this.transparentCount++] = i;
        this.blendedCasters.push({
          primitive: scene.renderablePrimitive[i], material, skinned: scene.renderableSkin[i] >= 0,
        });
        continue;
      }

      const primitive = scene.renderablePrimitive[i];
      // Winding is part of the batch, because it is part of the pipeline. An
      // instance whose world transform mirrors has to be drawn front-face-cw,
      // and one indirect draw has one front face -- so the mirrored copies of
      // a mesh form their own batch even though they share its geometry and
      // material.
      // Skinning is part of the batch for the same reason winding is: it is a
      // different pipeline, and it binds a second vertex buffer besides.
      const skinned = scene.renderableSkin[i] >= 0;
      const key = `${primitiveId(primitive)}:${material}:${mirrored ? 1 : 0}:${skinned ? 1 : 0}`;

      let batch = batchOf.get(key);
      if (batch === undefined) {
        batch = this.batchCount++;
        // Checked here because a batch id is minted here. Nothing batch-indexed
        // is read before the prefix sum below, so widening mid-loop is safe as
        // long as the three arrays this loop writes are carried over.
        if (batch >= this.batchCapacity) this._growBatches(batch + 1);
        batchOf.set(key, batch);
        this.batchPrimitive.push(primitive);
        this.batchMaterial[batch] = material;
        this.batchMirrored[batch] = mirrored ? 1 : 0;
        this.batchSkinned[batch] = skinned ? 1 : 0;
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

    // Static draw order for the shadow pass, which culls nothing: every batch's
    // slice, then every blended caster, in the tail the batches leave free.
    // Blended geometry casts a hashed-alpha shadow -- see shadows.js -- where it
    // used to cast none, which was right for clear glass and wrong for smoke,
    // tinted glass and every leaf drawn with BLEND.
    const cursor = new Uint32Array(Math.max(this.batchCount, 1));
    for (let i = 0; i < count; i++) {
      const batch = this.itemBatch[i];
      if (batch === NOT_BATCHED) continue;
      this.batchOrder[this.batchFirst[batch] + cursor[batch]++] = i;
    }
    for (let k = 0; k < this.transparentCount; k++) {
      this.batchOrder[this.opaqueCount + k] = this.transparentItems[k];
    }

    // Per-batch uniform holding that slice's base index, once per phase. Each
    // phase's slice of the visible list starts a whole capacity along.
    for (let phase = 0; phase < CULL_PHASES; phase++) {
      for (let b = 0; b < this.batchCount; b++) {
        const slot = phase * this.batchCapacity + b;
        // The visible list is still renderable-sized -- only the slot table is
        // per batch -- so the base it stores is a VISIBLE index.
        new Uint32Array(this.batchStaging, slot * this.alignment, 1)[0] =
          phase * this.capacity + this.batchFirst[b];
      }
    }
    // The transparent draws' base, always 0: they address the visible list
    // through firstInstance, which a direct draw may set freely.
    this.transparentBatchSlot = this.batchCapacity * CULL_PHASES;
    new Uint32Array(this.batchStaging, this.transparentBatchSlot * this.alignment, 1)[0] = 0;
    // With LOD, the shadow pass reads the shadow slice, not the static order,
    // so the blended casters are copied into that slice's tail -- past every
    // slot the shadow phase can emit to, as the camera's are.
    new Uint32Array(this.batchStaging, (this.transparentBatchSlot + 1) * this.alignment, 1)[0] =
      CULL_SHADOW * this.capacity;
    /** Per blended item, in transparentItems order: whether its LOD level shows this frame. */
    if (!(this.casterSelected?.length >= this.transparentCount)) this.casterSelected = new Uint8Array(this.capacity);
    this.casterSelected.fill(1);

    const queue = this.rhi.queue;
    queue.writeBuffer(this.itemBatchBuffer, 0, this.itemBatch, 0, count);
    queue.writeBuffer(this.batchFirstBuffer, 0, this.batchFirst, 0, Math.max(this.batchCount, 1));
    queue.writeBuffer(this.batchOrderBuffer, 0, this.batchOrder, 0, Math.max(this.opaqueCount + this.transparentCount, 1));
    if (this.hasLod && this.transparentCount > 0) {
      queue.writeBuffer(this.visibleBuffer, (CULL_SHADOW * this.capacity + this.opaqueCount) * 4,
        this.transparentItems, 0, this.transparentCount);
    }
    queue.writeBuffer(this.batchBuffer, 0, this.batchStaging);
    // Item indices have just been reassigned, so last frame's flags describe
    // whatever used to occupy those slots. Starting from zero costs one frame
    // in which everything is found by the late phase, which is correct.
    queue.writeBuffer(this.visibleFlagsBuffer, 0, this._zeroFlags, 0, Math.max(count, 1));

    this._needsFullUpload = true;
    this.sceneId = scene.id;
    this.sceneRevision = scene.revision;
    this.stats.batches = this.batchCount;
    this.stats.items = count;
    this.stats.transparent = this.transparentCount;
  }

  /** Upload this frame's transforms, bounds and reset argument buffer. */
  update(scene, frustum, hzb, viewProjection, writeDrawData, paletteOffsets, morph, projectionScale = 1) {
    // A DIFFERENT scene needs rebuilding even when its revision happens to
    // match, and it usually does: every scene's first add() takes it to 1. The
    // check used to be on the revision alone, so rendering a second scene
    // silently kept the first one's batches -- its primitives, its materials
    // and therefore its pipelines. Two scenes one add() old each is all it
    // took, which is the shape of a menu behind a game, or a preview beside a
    // main view.
    if (scene.id !== this.sceneId || scene.revision !== this.sceneRevision) {
      this.rebuildBatches(scene);
    }

    const count = scene.renderableCount;

    // Only objects whose transform actually recomposed need rewriting. The
    // hierarchy already worked out which ones those are, and rewriting all of
    // them throws that answer away.
    //
    // Uploaded as RUNS of changed renderables. It was one span from the first
    // to the last, and two movers at opposite ends of 10,000 renderables sent
    // all 10,000 -- 1.6 MB, measured at 3.9 ms a frame against 0.8 ms for the
    // whole frame when still. Runs closer than one call's worth of bytes merge,
    // so scattered movers cannot turn into thousands of calls either.
    const moved = scene.transforms.moved;
    const full = this._needsFullUpload;
    const runs = this._runs;
    runs.length = 0;
    const mergeGap = Math.floor(WRITE_CALL_BYTES / UPLOAD_ITEM_BYTES);

    // Nothing moved and nothing deforms: there is nothing to rewrite, and the
    // scan to find that out was 0.8 ms a frame at 100,000 renderables.
    const scan = full || scene.transforms.movedPending || scene.skins.length > 0 || scene.morphs.length > 0;
    for (let i = 0; scan && i < count; i++) {
      // Skinned and morphed boxes come from joints and weights, which change
      // without the mesh's own node moving -- the usual rig. Gated on the node
      // alone, a character walked off its box and was culled where it stood.
      if (!full && moved[scene.renderableMatrixSlot[i]] === 0
        && scene.renderableSkin[i] < 0 && scene.renderableMorph[i] < 0) continue;

      const drawFloat = i * (DRAW_DATA_BYTES / 4);
      writeDrawData(this.drawData, drawFloat, scene, i);
      // Where this instance's joints begin. Zero for anything unskinned, which
      // the unskinned vertex shader never reads anyway.
      const skin = scene.renderableSkin[i];
      this.drawDataU32[drawFloat + 28] = skin >= 0 ? paletteOffsets[skin] : 0;

      // The three morph words. All static per renderable -- where a
      // primitive's deltas live, where an instance's weights live, and how
      // many of each -- which is why they can ride the same "only what moved"
      // upload as the matrix. The WEIGHTS themselves are a separate buffer,
      // rewritten whole every frame, precisely because they are not.
      const m = scene.renderableMorph[i];
      const primitive = scene.renderablePrimitive[i];
      this.drawDataU32[drawFloat + 29] = m >= 0 ? primitive.morphBase : 0;
      this.drawDataU32[drawFloat + 30] = m >= 0 ? morph.offsets[m] : 0;
      this.drawDataU32[drawFloat + 31] = m >= 0 ? primitive.morphCountStride : 0;

      const b = i * BOUNDS_FLOATS;
      const o = i * 3;
      this.boundsData[b] = scene.worldMin[o];
      this.boundsData[b + 1] = scene.worldMin[o + 1];
      this.boundsData[b + 2] = scene.worldMin[o + 2];
      this.boundsData[b + 3] = scene.renderableCoverage[i * 2];
      this.boundsData[b + 4] = scene.worldMax[o];
      this.boundsData[b + 5] = scene.worldMax[o + 1];
      this.boundsData[b + 6] = scene.worldMax[o + 2];
      this.boundsData[b + 7] = scene.renderableCoverage[i * 2 + 1];
      // The group's sphere, through the group node's world matrix: every
      // level measures the same one, so no distance shows two levels or none.
      const lodSlot = scene.renderableLodSlot[i];
      if (lodSlot >= 0) {
        const w = scene.transforms.world;
        const m = lodSlot * 16;
        const s = scene.renderableLodSphere;
        const x = s[i * 4];
        const y = s[i * 4 + 1];
        const z = s[i * 4 + 2];
        this.boundsData[b + 8] = w[m] * x + w[m + 4] * y + w[m + 8] * z + w[m + 12];
        this.boundsData[b + 9] = w[m + 1] * x + w[m + 5] * y + w[m + 9] * z + w[m + 13];
        this.boundsData[b + 10] = w[m + 2] * x + w[m + 6] * y + w[m + 10] * z + w[m + 14];
        this.boundsData[b + 11] = s[i * 4 + 3] * Math.sqrt(Math.max(
          w[m] * w[m] + w[m + 1] * w[m + 1] + w[m + 2] * w[m + 2],
          w[m + 4] * w[m + 4] + w[m + 5] * w[m + 5] + w[m + 6] * w[m + 6],
          w[m + 8] * w[m + 8] + w[m + 9] * w[m + 9] + w[m + 10] * w[m + 10],
        ));
      } else {
        this.boundsData[b + 11] = 0;
      }

      addToRuns(runs, i, mergeGap);
    }
    this._needsFullUpload = false;
    let uploaded = 0;
    for (let r = 0; r < runs.length; r += 2) uploaded += runs[r + 1] - runs[r] + 1;
    this.stats.uploaded = uploaded;

    // Draw arguments, with instanceCount zeroed, for BOTH phases. The cull
    // shader raises the count with atomicAdd, so resetting here is what makes
    // the frame idempotent -- forget it and counts accumulate until every batch
    // draws the whole scene.
    for (let phase = 0; phase < CULL_PHASES; phase++) {
      for (let b = 0; b < this.batchCount; b++) {
        const primitive = this.batchPrimitive[b];
        const o = (phase * this.batchCapacity + b) * 5;
        this.indirectData[o] = primitive.indexCount;
        this.indirectData[o + 1] = 0;
        this.indirectData[o + 2] = 0;
        this.indirectData[o + 3] = 0;
        this.indirectData[o + 4] = 0;
      }
    }

    const planeFloats = FRUSTUM_PLANE_COUNT * 4;
    this.cullParamsF32.set(frustum, 0);
    this.cullParamsF32.set(viewProjection, planeFloats);
    this.cullParamsU32[planeFloats + 16] = count;
    this.cullParamsU32[planeFloats + 17] = hzb.levelCount;
    this.cullParamsF32[planeFloats + 18] = hzb.width;
    this.cullParamsF32[planeFloats + 19] = hzb.height;
    this.cullParamsF32[planeFloats + 23] = projectionScale;

    const queue = this.rhi.queue;
    const drawFloats = DRAW_DATA_BYTES / 4;
    for (let r = 0; r < runs.length; r += 2) {
      const low = runs[r];
      const span = runs[r + 1] - low + 1;
      queue.writeBuffer(
        this.drawDataBuffer, low * DRAW_DATA_BYTES,
        this.drawData, low * drawFloats, span * drawFloats,
      );
      queue.writeBuffer(
        this.boundsBuffer, low * BOUNDS_FLOATS * 4, this.boundsData, low * BOUNDS_FLOATS, span * BOUNDS_FLOATS,
      );
    }
    // Both halves: phase 1 writes at capacity, so a partial write would leave
    // its counts at whatever the previous frame accumulated.
    queue.writeBuffer(this.indirectBuffer, 0, this.indirectData);

    // One params slot per phase. Only the last four words differ, but writing
    // whole slots keeps the two descriptions independent rather than sharing a
    // prefix that a future field could quietly break.
    for (let phase = 0; phase < CULL_PHASES; phase++) {
      this.cullParamsU32[planeFloats + 20] = phase;
      this.cullParamsU32[planeFloats + 21] = phase * this.batchCapacity;
      this.cullParamsU32[planeFloats + 22] = phase * this.capacity;
      queue.writeBuffer(this.cullParamsBuffer, phase * this.cullParamsStride, this.cullParams);
    }
    this.itemCount = count;
  }

  /**
   * One cull dispatch.
   *
   * The late phase declares a read of the pyramid, which is what orders it
   * after the early draw that produced the depth the pyramid was built from.
   * Without that edge the topological sort is free to run both culls first,
   * and the late one would test against an empty pyramid.
   */
  addCullPass(graph, { phase, boundsResource, indirectResource, visibleResource, hzbResources = [] }) {
    graph.addPass({
      name: `cull:${['early', 'late', 'shadow'][phase]}`,
      type: 'compute',
      reads: [boundsResource, ...hzbResources],
      writes: [indirectResource, visibleResource],
      execute: this._cullExecute[phase],
    });
  }

  _dispatch(pass, phase) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroups[phase]);
    // WebGPU guarantees 65,535 workgroups per dimension, which at 64 items a
    // workgroup is 4.2 million renderables. Past that, y takes the rest -- the
    // draw capacity has no ceiling, so neither may this.
    const groups = Math.ceil(this.itemCount / WORKGROUP_SIZE);
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
  }

  /** Dynamic offset for the zero-base slot the ordered draws bind. */
  transparentBatchOffset() {
    return this.transparentBatchSlot * this.alignment;
  }

  /** The same, for the shadow pass's blended casters when it reads the shadow slice. */
  shadowTransparentBatchOffset() {
    return (this.transparentBatchSlot + 1) * this.alignment;
  }

  /**
   * lodSelected from the cull shader, on the CPU, for the items it never
   * sees: the blended ones, which are ordered here. Reads the bounds as last
   * uploaded, which is this frame's once update() has run.
   */
  lodSelected(item, viewProjection, projectionScale) {
    const b = item * BOUNDS_FLOATS;
    const radius = this.boundsData[b + 11];
    if (radius <= 0) return true;
    const vp = viewProjection;
    const w = vp[3] * this.boundsData[b + 8] + vp[7] * this.boundsData[b + 9] + vp[11] * this.boundsData[b + 10] + vp[15];
    const coverage = w > 0 ? radius * projectionScale / w : 3.0e38;
    return coverage >= this.boundsData[b + 3] && coverage < this.boundsData[b + 7];
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

  /** Byte offset of one batch's draw arguments, in the given phase's half. */
  indirectOffset(batch, phase = 0) {
    return (phase * this.batchCapacity + batch) * INDIRECT_BYTES;
  }

  batchOffset(batch, phase = 0) {
    return (phase * this.batchCapacity + batch) * this.alignment;
  }

  destroy() {
    for (const buffer of [
      this.drawDataBuffer, this.boundsBuffer, this.itemBatchBuffer, this.batchFirstBuffer,
      this.indirectBuffer, this.visibleBuffer, this.batchOrderBuffer, this.batchBuffer,
      this.cullParamsBuffer, this.visibleFlagsBuffer,
    ]) buffer.destroy();
  }
}

// Primitives have no identity of their own, so one is stamped on first use.
let nextPrimitiveId = 1;
/**
 * Does this world matrix flip handedness?
 *
 * The determinant of the upper 3x3. Negative means an odd number of axes were
 * reflected, which reverses triangle winding -- so glTF 3.7.4 requires the
 * front face to reverse with it. Computed from the WORLD matrix rather than
 * the node's own, because a mirroring parent mirrors everything under it, and
 * two mirrors cancel.
 *
 * ponytail: read when a batch is built, which is when the scene's contents
 * change. An instance that flips handedness later -- an animated scale passing
 * through zero -- keeps its old winding until something else rebuilds the
 * batches. Detecting that would cost a determinant per moved object per frame,
 * forever, for a case that requires passing through a degenerate transform.
 */
function isMirrored(world, offset) {
  const m00 = world[offset], m01 = world[offset + 1], m02 = world[offset + 2];
  const m10 = world[offset + 4], m11 = world[offset + 5], m12 = world[offset + 6];
  const m20 = world[offset + 8], m21 = world[offset + 9], m22 = world[offset + 10];
  return m00 * (m11 * m22 - m12 * m21)
    - m10 * (m01 * m22 - m02 * m21)
    + m20 * (m01 * m12 - m02 * m11) < 0;
}

function primitiveId(primitive) {
  if (!primitive.__batchId) primitive.__batchId = nextPrimitiveId++;
  return primitive.__batchId;
}
