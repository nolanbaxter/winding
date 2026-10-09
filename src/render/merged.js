// Merged draws: every one-object batch of a material, drawn in one call.
//
// A batch is a primitive and a material, drawn instanced; a scene of distinct
// meshes -- a level, Sponza, a town of separate buildings -- is a batch per
// mesh, each one object, each its own indirect draw. Measured on Iris Xe, an
// indirect draw costs about 7 us of GPU time whether it draws anything or not:
// nine hundred buildings were 6.2 ms of draws before a pixel was shaded, and
// one draw of the same geometry 0.1 ms (tools/arena-proto.html).
//
// So the batches of one object each are pooled by material and winding into
// GROUPS, and each group is one indexed draw over a list holding every one of
// its objects' triangles. Each object has a fixed place in its group's list
// and a fixed slot in its group's table, set when the batches are built; each
// index is tagged with the slot, (slot << shift) | vertex, and vsMerged in
// pbr.js splits it back, finds the object and its first vertex in the table,
// and reads the vertex out of the geometry arena by hand -- an index is no
// longer a vertex buffer's address.
//
// It is drawn a CHUNK at a time (chunkBoxes in bounds.js: runs of 128
// triangles, each with its own box). A chunk out of view keeps its place,
// filled with its object's tag alone: triangles of one repeated vertex, which
// the GPU discards before anything is rasterized, and which its vertex cache
// shades at most once. Each frame, a thread per chunk tests its box, through
// its object's world matrix, against the view; only one that came into view or
// left it has its place rewritten -- its real indices copied in from the
// arena, or the tag -- by a copy dispatched over exactly that work. A still
// view rewrites nothing, and the half of a wall behind the camera costs
// nothing either.
//
// A first version rebuilt every group's list each phase, from what the cull
// kept: 0.75 ms of Sponza's frame to save 0.6. This tests the view only,
// not the occlusion pyramid -- merged objects are drawn whenever they are in
// view -- which is the right trade for scenes of large meshes, where little
// is ever wholly hidden, and costs nothing when the view holds still.
//
// The split is per group: `shift` is as many bits as the group's largest
// primitive needs for a vertex, the rest number the slots. A group with more
// objects than its slots can number is split in two.
//
// Batches of several objects stay as they are: one draw already, and merging
// them would copy every instance's indices into a list. So do skinned and
// morphed ones, whose vertices leave the chunks' boxes, and any in an LOD
// group, whose shadows cast only the level the camera shows.
//
// SHADOWS draw every caster, in view or not, so each group has a second list
// holding every object's real indices, copied once when the batches are built:
// a cascade, or a strip of one being scrolled, is a draw per group rather than
// a draw per mesh. Nine hundred buildings under a turning sun were 3,600
// shadow draws a frame, half the shadow GPU time and most of the frame's CPU.

import { compileShader } from '../rhi/shader.js';
import { sharedPipelines } from '../rhi/pipeline.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { FRUSTUM_PLANE_COUNT } from '../core/math/frustum.js';
import { CULL_PHASES, BATCH_BYTES, TABLE_WORDS } from './gpudriven.js';
import { CHUNK_TRIANGLES } from '../scene/bounds.js';

/** A chunk's record: eight u32s, then its box in its mesh's space, min and max as vec4s. */
const INFO_WORDS = 16;
const WORKGROUP_SIZE = 64;
/** Indices one copy workgroup takes. */
const SPAN = 1024;
/** count, spans, stateBase, countBase, (unused), listBase, pads. */
const PARAMS_BYTES = 32;

const EXPAND_SHADER = /* wgsl */ `
struct Params {
  count        : u32,
  spans        : u32,
  stateBase    : u32,
  countBase    : u32,
  dispatchBase : u32,
  listBase     : u32,
  pad0         : u32,
  pad1         : u32,
};

/** gpudriven.js's, phase 0's: this frame's view. Only the planes and the LOD inputs are read. */
struct CullParams {
  planes          : array<vec4<f32>, ${FRUSTUM_PLANE_COUNT}>,
  viewProj        : mat4x4<f32>,
  count           : u32,
  hzbLevels       : u32,
  hzbWidth        : f32,
  hzbHeight       : f32,
  phase           : u32,
  indirectBase    : u32,
  visibleBase     : u32,
  projectionScale : f32,
};

/** gpudriven.js's draw data: only the model matrix is read. */
struct DrawData {
  model : mat4x4<f32>,
  rest  : array<vec4<f32>, 4>,
};

/** A merged chunk: its object, its group, its triangles, its place, and its box. */
struct Merged {
  item       : u32,
  group      : u32,
  indexCount : u32,
  firstIndex : u32,
  // Where its run starts in the lists, and its object's slot's tag.
  place      : u32,
  tag        : u32,
  // Its first span: spans of SPAN indices are numbered chunk after chunk.
  spanFirst  : u32,
  spans      : u32,
  // Its box, in its mesh's own space.
  lo         : vec4<f32>,
  hi         : vec4<f32>,
};

@group(0) @binding(0) var<uniform>             params  : Params;
@group(0) @binding(1) var<uniform>             view    : CullParams;
@group(0) @binding(2) var<storage, read>       draws   : array<DrawData>;
@group(0) @binding(3) var<storage, read>       merged  : array<Merged>;
// Each object's state (1 in view), then the work count and the work list of
// spans to rewrite: one flat array, so the count can be the atomic that hands
// out the list.
@group(0) @binding(4) var<storage, read_write> work    : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read>       arena   : array<u32>;
@group(0) @binding(6) var<storage, read_write> indices : array<u32>;
// The copy's dispatch size, bound only for the step that writes it: a buffer
// a dispatch reads its size from may not also be writable in that dispatch.
@group(1) @binding(0) var<storage, read_write> dispatch : array<u32>;

/**
 * The chunk's box, through its object's world matrix, against the view's five
 * planes: centre and half extent, the extent through the matrix's absolute
 * values -- the world box round the moved one, as aabbTransform makes it.
 */
fn inView(m : Merged) -> bool {
  let model = draws[m.item].model;
  let centre = (model * vec4<f32>((m.lo.xyz + m.hi.xyz) * 0.5, 1.0)).xyz;
  let half = (m.hi.xyz - m.lo.xyz) * 0.5;
  let extent = abs(model[0].xyz) * half.x + abs(model[1].xyz) * half.y + abs(model[2].xyz) * half.z;
  for (var p = 0u; p < ${FRUSTUM_PLANE_COUNT}u; p = p + 1u) {
    let plane = view.planes[p];
    if (dot(plane.xyz, centre) + plane.w + dot(abs(plane.xyz), extent) < 0.0) { return false; }
  }
  return true;
}

// A thread per chunk: in view or not, and if that changed, its spans onto the list.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn flip(@builtin(global_invocation_id) id : vec3<u32>) {
  let j = id.y * (65535u * ${WORKGROUP_SIZE}u) + id.x;
  if (j >= params.count) { return; }
  let m = merged[j];
  let now = select(0u, 1u, inView(m));
  if (atomicExchange(&work[params.stateBase + j], now) == now) { return; }
  let at = atomicAdd(&work[params.countBase], m.spans);
  for (var k = 0u; k < m.spans; k = k + 1u) {
    atomicStore(&work[params.listBase + at + k], m.spanFirst + k);
  }
}

// One thread: the copy's dispatch size, from the work count.
@compute @workgroup_size(1)
fn size() {
  let n = atomicLoad(&work[params.countBase]);
  dispatch[0] = min(n, 65535u);
  dispatch[1] = (n + 65534u) / 65535u;
  dispatch[2] = 1u;
}

// Every span, its object shown: the shadow lists, once, when the batches are built.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn copyAll(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_index) t : u32) {
  let s = wg.y * 65535u + wg.x;
  if (s >= params.spans) { return; }
  writeSpan(s, true, t);
}

/** Span s of its object: the object's indices, or its tag alone. */
fn writeSpan(s : u32, shownIfAny : bool, t : u32) {
  var lo = 0u;
  var hi = params.count;
  while (hi - lo > 1u) {
    let mid = (lo + hi) / 2u;
    if (merged[mid].spanFirst <= s) { lo = mid; } else { hi = mid; }
  }
  let m = merged[lo];
  let shown = shownIfAny || atomicLoad(&work[params.stateBase + lo]) == 1u;
  let start = (s - m.spanFirst) * ${SPAN}u;
  let end = min(start + ${SPAN}u, m.indexCount);
  for (var k = start + t; k < end; k = k + ${WORKGROUP_SIZE}u) {
    indices[m.place + k] = select(m.tag, m.tag | arena[m.firstIndex + k], shown);
  }
}

// A workgroup per listed span: its object's indices, or its tag alone.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn copy(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_index) t : u32) {
  let w = wg.y * 65535u + wg.x;
  if (w >= atomicLoad(&work[params.countBase])) { return; }
  writeSpan(atomicLoad(&work[params.listBase + w]), false, t);
}
`;

export class MergedDraws {
  /** `arena`: the geometry arena, whose indices the lists are copied from. */
  static async create(rhi, arena) {
    const merged = new MergedDraws(rhi);
    merged._arena = arena;
    await merged._init();
    return merged;
  }

  constructor(rhi) {
    this.rhi = rhi;
    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    /** Per group: { material, mirrored, shift, indexBase, tableBase, batches }. */
    this.groups = [];
    this.groupCount = 0;
    /** Merged objects, and the indices every group's list holds between them. */
    this.mergedCount = 0;
    this.indexTotal = 0;
    this.spanCount = 0;
    /** Per batch: 1 if a group draws it, so the batch loop does not. */
    this.batchMerged = new Uint8Array(0);
    this._buffers = {};
    this._execute = (pass) => this._dispatch(pass);
    /** Bumped when a buffer is replaced: bind groups naming them rebuild. */
    this.revision = 0;
    this._zeroCount = new Uint32Array(1);
  }

  async _init() {
    const device = this.rhi.device;
    const shader = await compileShader(device, EXPAND_SHADER, 'merged.wgsl');
    const entry = (binding, buffer) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer });
    this.layout = device.createBindGroupLayout({
      label: 'merged',
      entries: [
        entry(0, { type: 'uniform' }), entry(1, { type: 'uniform' }),
        entry(2, { type: 'read-only-storage' }), entry(3, { type: 'read-only-storage' }),
        entry(4, { type: 'storage' }), entry(5, { type: 'read-only-storage' }), entry(6, { type: 'storage' }),
      ],
    });
    this.dispatchLayout = device.createBindGroupLayout({ label: 'merged-dispatch', entries: [entry(0, { type: 'storage' })] });
    const layout = createPipelineLayout(device, { 0: this.layout }, 'merged');
    const sizeLayout = createPipelineLayout(device, { 0: this.layout, 1: this.dispatchLayout }, 'merged-size');
    const make = (name, l) => sharedPipelines(device).compute({ label: `merged-${name}`, layout: l, shader, entry: name });
    this.flipPipeline = make('flip', layout);
    this.sizePipeline = make('size', sizeLayout);
    this.copyPipeline = make('copy', layout);
    this.copyAllPipeline = make('copyAll', layout);
  }

  /** A buffer of at least `bytes`, replaced only when it has to grow. */
  _buffer(name, bytes, usage) {
    const have = this._buffers[name];
    if (have && have.size >= bytes) return have;
    have?.destroy();
    this.revision++;
    // Room to grow into, so a scene that adds one object does not replace them all.
    const size = Math.max(256, 2 ** Math.ceil(Math.log2(Math.max(bytes, 4))));
    return (this._buffers[name] = createBuffer(this.rhi, {
      label: `merged-${name}`, size, usage: usage | GPUBufferUsage.COPY_DST,
    }));
  }

  /**
   * Pool `gpu`'s one-object batches into groups, after its batches were
   * rebuilt -- which is also when it grows, so its capacities are final.
   */
  rebuild(gpu, scene) {
    const byKey = new Map();
    if (this.batchMerged.length < gpu.batchCapacity) this.batchMerged = new Uint8Array(gpu.batchCapacity);
    this.batchMerged.fill(0);
    for (let b = 0; b < gpu.batchCount; b++) {
      if (gpu.batchSize[b] !== 1 || gpu.batchSkinned[b] === 1) continue;
      if (scene.renderableLodSlot[gpu.batchOrder[gpu.batchFirst[b]]] >= 0) continue;
      // Morphed vertices leave the chunks' boxes, which are the rest pose's.
      if (gpu.batchPrimitive[b].morphCountStride !== 0) continue;
      const key = gpu.batchMaterial[b] * 2 + gpu.batchMirrored[b];
      let list = byKey.get(key);
      if (!list) byKey.set(key, (list = []));
      list.push(b);
    }

    this.groups.length = 0;
    let indexTotal = 0, itemTotal = 0;
    for (const [key, list] of byKey) {
      let most = 1;
      for (const b of list) most = Math.max(most, gpu.batchPrimitive[b].vertexCount);
      // Bits for a vertex, the rest for the slot; and as many slots as that leaves.
      const shift = Math.max(1, Math.ceil(Math.log2(most)));
      if (shift >= 31) continue;
      const slots = 2 ** (32 - shift);
      for (let first = 0; first < list.length; first += slots) {
        const batches = list.slice(first, first + slots);
        const group = { material: key >> 1, mirrored: key & 1, shift, indexBase: indexTotal, tableBase: itemTotal, batches };
        for (const b of batches) {
          indexTotal += gpu.batchPrimitive[b].indexCount;
          this.batchMerged[b] = 1;
        }
        itemTotal += batches.length;
        this.groups.push(group);
      }
    }
    this.groupCount = this.groups.length;
    this.mergedCount = itemTotal;
    this.indexTotal = indexTotal;
    if (this.groupCount === 0) return;

    const STORAGE = GPUBufferUsage.STORAGE;
    const queue = this.rhi.queue;
    // A record per chunk: CHUNK_TRIANGLES of a primitive with chunks, the
    // whole of one without, its box its own.
    let records = 0;
    for (const group of this.groups) for (const b of group.batches) records += gpu.batchPrimitive[b].chunks?.count ?? 1;
    const infoBytes = new ArrayBuffer(records * INFO_WORDS * 4);
    const info = new Uint32Array(infoBytes);
    const infoF32 = new Float32Array(infoBytes);
    const table = new Uint32Array(itemTotal * TABLE_WORDS);
    let j = 0, spans = 0, slotTotal = 0;
    this.groups.forEach((group, g) => {
      let place = group.indexBase;
      group.batches.forEach((b, slot) => {
        const p = gpu.batchPrimitive[b];
        const item = gpu.batchOrder[gpu.batchFirst[b]];
        const tag = (slot << group.shift) >>> 0;
        table[slotTotal * TABLE_WORDS] = item;
        table[slotTotal++ * TABLE_WORDS + 1] = p.baseVertex;
        const chunks = p.chunks;
        const count = chunks?.count ?? 1;
        for (let k = 0; k < count; k++) {
          const first = chunks ? k * CHUNK_TRIANGLES * 3 : 0;
          const indices = chunks ? Math.min(CHUNK_TRIANGLES * 3, p.indexCount - first) : p.indexCount;
          const span = Math.ceil(indices / SPAN);
          const o = j * INFO_WORDS;
          info[o] = item; info[o + 1] = g; info[o + 2] = indices; info[o + 3] = p.firstIndex + first;
          info[o + 4] = place + first; info[o + 5] = tag; info[o + 6] = spans; info[o + 7] = span;
          if (chunks) {
            infoF32.set(chunks.min.subarray(k * 3, k * 3 + 3), o + 8);
            infoF32.set(chunks.max.subarray(k * 3, k * 3 + 3), o + 12);
          } else {
            infoF32.set(p.bounds.min, o + 8);
            infoF32.set(p.bounds.max, o + 12);
          }
          spans += span;
          j++;
        }
        place += p.indexCount;
      });
      group.indexCount = place - group.indexBase;
    });
    this.spanCount = spans;
    this.recordCount = records;
    queue.writeBuffer(this._buffer('info', info.byteLength, STORAGE), 0, info);
    // Past every cull phase's slice of the visible list: the groups' tables.
    queue.writeBuffer(gpu.visibleBuffer, gpu.capacity * CULL_PHASES * 4, table);

    // Every object starts out of view, its place all zeros: its group's slot
    // 0, vertex 0, three at a time, nothing drawn. The first frame brings in the rest.
    const indices = this._buffer('indices', indexTotal * 4, GPUBufferUsage.INDEX | STORAGE);
    this._countBase = records;
    this._listBase = records + 1;
    const work = this._buffer('work', (this._listBase + spans) * 4, STORAGE);
    this._buffer('dispatch', 12, GPUBufferUsage.INDIRECT | STORAGE);
    const shadowIndices = this._buffer('shadowIndices', indexTotal * 4, GPUBufferUsage.INDEX | STORAGE);

    // What vsMerged reads through the draw group: where its group's table starts, and the split.
    const uniforms = new ArrayBuffer(this.groupCount * this.alignment);
    this.groups.forEach((group, g) => {
      new Uint32Array(uniforms, g * this.alignment, 2)
        .set([gpu.capacity * CULL_PHASES + group.tableBase * TABLE_WORDS, group.shift]);
    });
    queue.writeBuffer(this._buffer('uniforms', uniforms.byteLength, GPUBufferUsage.UNIFORM), 0, uniforms);
    const params = Uint32Array.of(records, spans, 0, this._countBase, 0, this._listBase, 0, 0);
    queue.writeBuffer(this._buffer('params', PARAMS_BYTES, GPUBufferUsage.UNIFORM), 0, params);

    // Cleared, and the shadow lists filled, now: the writes above go first, in queue order.
    const encoder = this.rhi.device.createCommandEncoder({ label: 'merged-rebuild' });
    encoder.clearBuffer(indices, 0, indexTotal * 4);
    encoder.clearBuffer(work, 0, this._listBase * 4);
    const pass = encoder.beginComputePass({ label: 'merged-shadow-lists' });
    pass.setPipeline(this.copyAllPipeline);
    pass.setBindGroup(0, this._makeBindGroup(gpu.cullParamsBuffer, gpu.cullParams.byteLength, gpu.drawDataBuffer, this._arena.indexBuffer, shadowIndices));
    pass.dispatchWorkgroups(Math.min(spans, 65535), Math.ceil(spans / 65535));
    pass.end();
    queue.submit([encoder.finish()]);
  }

  /** The bind group every step uses, writing into `indices`: the camera's lists, or the shadows'. */
  _makeBindGroup(cullParams, cullBytes, draws, arenaIndices, indices) {
    const b = this._buffers;
    return this.rhi.device.createBindGroup({
      label: 'merged',
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: b.params, size: PARAMS_BYTES } },
        { binding: 1, resource: { buffer: cullParams, offset: 0, size: cullBytes } },
        { binding: 2, resource: { buffer: draws } },
        { binding: 3, resource: { buffer: b.info } },
        { binding: 4, resource: { buffer: b.work } },
        { binding: 5, resource: { buffer: arenaIndices } },
        { binding: 6, resource: { buffer: indices } },
      ],
    });
  }

  /** The draw group vsMerged reads its group's table base and split through, at g * alignment. */
  drawGroup(drawLayout) {
    return (this.drawBindGroup ??= this.rhi.device.createBindGroup({
      label: 'merged-draw', layout: drawLayout,
      entries: [{ binding: 0, resource: { buffer: this._buffers.uniforms, size: BATCH_BYTES } }],
    }));
  }

  /** Every object's real indices, group after group: what a shadow view draws. */
  get shadowIndices() {
    return this._buffers.shadowIndices;
  }

  /** Every frame, before the pass: no work listed yet. */
  reset() {
    if (this.groupCount > 0) this.rhi.queue.writeBuffer(this._buffers.work, this._countBase * 4, this._zeroCount);
  }

  /**
   * The pass that brings the groups' lists up to date with this frame's view,
   * reading `gpu`'s draw data (`drawsResource`) and its cull parameters.
   * Returns the resource the forward pass reads, or null with nothing merged.
   */
  addPass(graph, gpu, arena, drawsResource) {
    if (this.groupCount === 0) return null;
    const key = `${this.revision}:${gpu.buffersRevision}:${arena.revision}`;
    if (this._bindKey !== key) {
      this._bindKey = key;
      const b = this._buffers;
      this._bindGroup = this._makeBindGroup(gpu.cullParamsBuffer, gpu.cullParams.byteLength, gpu.drawDataBuffer, arena.indexBuffer, b.indices);
      this._dispatchGroup = this.rhi.device.createBindGroup({
        label: 'merged-dispatch', layout: this.dispatchLayout, entries: [{ binding: 0, resource: { buffer: b.dispatch } }],
      });
      this.drawBindGroup = undefined;
    }
    const lists = graph.importBuffer('merged', this._buffers.indices);
    graph.addPass({ name: 'merge', type: 'compute', reads: [drawsResource], writes: [lists], execute: this._execute });
    return lists;
  }

  _dispatch(pass) {
    pass.setBindGroup(0, this._bindGroup);
    // Dispatches in one pass see each other's writes: each step reads the last.
    const threads = Math.ceil(this.recordCount / WORKGROUP_SIZE);
    pass.setPipeline(this.flipPipeline);
    pass.dispatchWorkgroups(Math.min(threads, 65535), Math.ceil(threads / 65535));
    pass.setPipeline(this.sizePipeline);
    pass.setBindGroup(1, this._dispatchGroup);
    pass.dispatchWorkgroups(1);
    // The copy's layout has no group 1, so the size it reads is not bound writable here.
    pass.setPipeline(this.copyPipeline);
    pass.dispatchWorkgroupsIndirect(this._buffers.dispatch, 0);
  }

  /**
   * The groups' draws. `pipelineFor(material, mirrored)` gives the merged
   * pipeline, `bindMaterial(material)` sets the material's group.
   */
  encode(pass, drawLayout, groupDraw, pipelineFor, bindMaterial) {
    if (this.groupCount === 0) return;
    const drawGroup = this.drawGroup(drawLayout);
    pass.setIndexBuffer(this._buffers.indices, 'uint32');
    for (let g = 0; g < this.groupCount; g++) {
      const group = this.groups[g];
      const pipeline = pipelineFor(group.material, group.mirrored);
      if (pipeline === null) continue;
      pass.setPipeline(pipeline);
      bindMaterial(group.material);
      pass.setBindGroup(groupDraw, drawGroup, [g * this.alignment]);
      // Direct: a group's count and place never change between rebuilds -- a
      // hidden chunk keeps its place as degenerate triangles -- and a direct
      // draw skips the validation an indirect one gets.
      pass.drawIndexed(group.indexCount, 1, group.indexBase, 0, 0);
    }
  }

  destroy() {
    for (const buffer of Object.values(this._buffers)) buffer.destroy();
  }
}
