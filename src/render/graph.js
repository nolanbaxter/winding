// Render graph.
//
// Passes declare what they READ and what they WRITE. Nothing declares what
// order to run in, when to clear, when to store, or which texture to use for a
// temporary. All four are derived from the dependency graph, which is the whole
// point: those are exactly the things that are tedious to maintain by hand and
// silently wrong when you get them out of step.
//
//   graph.begin();
//   const shadows = graph.importTexture('shadows', shadowView);
//   graph.addPass({ name: 'shadow:0', depth: { resource: shadows, view: layer0, clear: 0 },
//                   execute(pass) { ... } });
//   graph.addPass({ name: 'forward', reads: [shadows], color: [...], depth: {...},
//                   execute(pass) { ... } });
//   graph.compile();
//   graph.execute(encoder);
//
// What it derives:
//
//   ORDER          topological sort over read-after-write and write-after-write
//   LOAD OP        clear if a clear value was given, load if someone wrote it
//                  earlier in the frame
//   STORE OP       store only if something later reads it, or it is imported
//                  and therefore visible outside the frame. Otherwise discard,
//                  which on a tile-based GPU means the buffer is never written
//                  back to memory at all
//   DEAD PASSES    a pass whose output nothing consumes does not execute
//   ALIASING       two transient textures whose lifetimes do not overlap share
//                  one allocation
//
// What it deliberately does NOT do: barriers. WebGPU inserts the hazard
// tracking between passes in a command encoder itself, which removes the single
// most error-prone part of a Vulkan-style graph.
//
// Rebuilding happens every frame, but allocation does not: begin() resets
// counters over pooled pass and resource records, so a steady-state frame adds
// nothing to the heap.

import { DEBUG, assert } from '../core/assert.js';
import { grownCapacity, growArray } from '../core/grow.js';

const DEFAULT_MAX_PASSES = 32;
const DEFAULT_MAX_RESOURCES = 64;

export class RenderGraph {
  constructor(rhi, { maxPasses = DEFAULT_MAX_PASSES, maxResources = DEFAULT_MAX_RESOURCES, profiler = null } = {}) {
    this.rhi = rhi;
    /** Optional GpuProfiler. Every pass gets its timestamp writes from here. */
    this.profiler = profiler;
    this.maxPasses = maxPasses;
    this.maxResources = maxResources;

    // Pooled records, reused across frames. Reset by begin(), never reallocated.
    this._passes = Array.from({ length: maxPasses }, makePassRecord);
    this._resources = Array.from({ length: maxResources }, makeResourceRecord);
    this.passCount = 0;
    this.resourceCount = 0;

    // Physical textures, cached across frames by descriptor. The aliasing pass
    // hands these out; without the cache every frame would recreate them.
    this._pool = new Map();

    this._order = new Uint32Array(maxPasses);
    this._orderCount = 0;
    this._live = new Uint8Array(maxPasses);
    this._indegree = new Uint32Array(maxPasses);
    this._queue = new Uint32Array(maxPasses);
    // Dependency edges, producer -> consumer. Grown on demand, reused forever.
    this._edgeFrom = new Uint32Array(maxPasses * 8);
    this._edgeTo = new Uint32Array(maxPasses * 8);
    this._edgeCount = 0;

    this.stats = { passes: 0, executed: 0, culled: 0, edges: 0, transient: 0, aliased: 0 };
    this._compiled = false;
  }

  /** Start a new frame's declaration. Frees nothing; resets counters. */
  begin() {
    this.profiler?.begin();
    this.passCount = 0;
    this.resourceCount = 0;
    this._compiled = false;
    // _pool maps a descriptor key to an ARRAY of entries, so this needs both
    // loops -- setting inUse on the array itself silently pools nothing and
    // every frame allocates fresh textures.
    for (const entries of this._pool.values()) {
      for (const entry of entries) entry.inUse = 0;
    }
    return this;
  }

  /**
   * A texture the graph does not own: the swap chain image, a shadow map that
   * persists between frames, anything created outside.
   *
   * `external` is a separate question from ownership, and conflating the two is
   * a mistake worth naming. An EXTERNAL resource is visible outside this frame
   * -- the swap chain is presented, a shadow map might be read next frame --
   * so its contents must survive and it is always stored. A resource that is
   * merely imported, like a depth buffer the device allocates once and the
   * renderer scribbles on every frame, is nobody's business after the frame
   * ends, so its store op is derived like any transient's.
   */
  importTexture(name, view, { external = true } = {}) {
    const handle = this._allocResource(name);
    const resource = this._resources[handle];
    resource.imported = true;
    resource.external = external;
    resource.view = view;
    return handle;
  }

  /**
   * A buffer the graph tracks for ordering only.
   *
   * Buffers never alias here: a compute pass's output is usually read next
   * frame as well, and the sizes rarely match anything else. What the graph
   * needs them for is the dependency edge -- without a declared resource, two
   * compute passes that must run in order have nothing linking them and the
   * topological sort is free to swap them.
   */
  importBuffer(name, buffer) {
    const handle = this._allocResource(name);
    const resource = this._resources[handle];
    resource.kind = 'buffer';
    resource.imported = true;
    resource.external = true;
    resource.buffer = buffer;
    return handle;
  }

  /**
   * A texture the graph owns for the duration of the frame. Its memory may be
   * shared with any other transient whose lifetime does not overlap.
   */
  createTexture(name, { width, height, format, usage, sampleCount = 1, depthOrArrayLayers = 1 }) {
    const handle = this._allocResource(name);
    const resource = this._resources[handle];
    resource.imported = false;
    resource.desc = { width, height, format, usage, sampleCount, depthOrArrayLayers };
    resource.key = `${width}x${height}x${depthOrArrayLayers}:${format}:${usage}:${sampleCount}`;
    return handle;
  }

  /**
   * Widen the pass tables. Called when a frame declares more than fits.
   *
   * These were a hard ceiling, and the engine's own frame had already reached
   * it: the pass count is 10 plus one per depth-pyramid mip plus two per bloom
   * level, so it is a function of RESOLUTION. 31 at 1080p, exactly 32 at 1440p
   * and 4K, and 33 at 5K -- where the old limit threw, every frame, on hardware
   * that could otherwise run it. Growing is what the rest of the engine does
   * with a capacity, and it is what the README already claims happens here.
   *
   * Contents are not preserved: begin() resets every counter, so a grow can
   * only happen part-way through a declaration that is about to be rebuilt from
   * scratch next frame anyway. The records themselves are pooled, so the ones
   * already handed out stay valid -- only the flat index arrays are replaced,
   * and nothing reads those until compile().
   */
  _growPasses(needed) {
    const capacity = grownCapacity(this.maxPasses, needed);
    while (this._passes.length < capacity) this._passes.push(makePassRecord());

    this._order = new Uint32Array(capacity);
    this._live = new Uint8Array(capacity);
    this._indegree = new Uint32Array(capacity);
    this._queue = new Uint32Array(capacity);

    const edges = growArray(this._edgeFrom, capacity * 8);
    edges.set(this._edgeFrom);
    this._edgeFrom = edges;
    const edgesTo = growArray(this._edgeTo, capacity * 8);
    edgesTo.set(this._edgeTo);
    this._edgeTo = edgesTo;

    this.maxPasses = capacity;
  }

  _growResources(needed) {
    const capacity = grownCapacity(this.maxResources, needed);
    while (this._resources.length < capacity) this._resources.push(makeResourceRecord());
    this.maxResources = capacity;
  }

  _allocResource(name) {
    if (this.resourceCount >= this.maxResources) this._growResources(this.resourceCount + 1);
    const handle = this.resourceCount++;
    const resource = this._resources[handle];
    resource.name = name;
    resource.kind = 'texture';
    resource.buffer = null;
    resource.imported = false;
    resource.external = false;
    resource.view = null;
    resource.desc = null;
    resource.key = '';
    resource.physical = null;
    resource.firstUse = -1;
    resource.lastUse = -1;
    resource.writerCount = 0;
    resource.readerCount = 0;
    return handle;
  }

  /**
   * @param desc.color   [{ resource, view?, clear? }] colour attachments
   * @param desc.depth   { resource, view?, clear? } depth attachment
   * @param desc.reads   [resource] textures this pass samples
   * @param desc.execute (passEncoder, graph) => void
   */
  addPass(desc) {
    if (this.passCount >= this.maxPasses) this._growPasses(this.passCount + 1);
    const index = this.passCount++;
    const pass = this._passes[index];

    pass.name = desc.name ?? `pass${index}`;
    pass.execute = desc.execute;
    pass.type = desc.type ?? 'render';
    pass.colorCount = 0;
    pass.depth = null;
    pass.readCount = 0;
    pass.writeCount = 0;

    // Writes that are not attachments: storage buffers a compute pass fills.
    // Declared separately because they carry no load or store op, but they are
    // the same thing as far as ordering is concerned.
    for (const resource of desc.writes ?? []) {
      pass.writes[pass.writeCount++] = resource;
      this._recordWrite(resource, index);
    }

    for (const attachment of desc.color ?? []) {
      const slot = pass.color[pass.colorCount++] ??= { resource: 0, view: null, clear: undefined };
      slot.resource = attachment.resource;
      slot.view = attachment.view ?? null;
      slot.clear = attachment.clear;
      this._recordWrite(attachment.resource, index);
    }

    if (desc.depth) {
      pass.depth = pass._depth;
      pass.depth.resource = desc.depth.resource;
      pass.depth.view = desc.depth.view ?? null;
      pass.depth.clear = desc.depth.clear;
      this._recordWrite(desc.depth.resource, index);
    }

    for (const resource of desc.reads ?? []) {
      pass.reads[pass.readCount++] = resource;
      this._recordRead(resource, index);
    }

    return index;
  }

  _recordWrite(handle, passIndex) {
    if (DEBUG) assert(handle >= 0 && handle < this.resourceCount, 'write to an unknown resource');
    const resource = this._resources[handle];
    resource.writers[resource.writerCount++] = passIndex;
  }

  _recordRead(handle, passIndex) {
    if (DEBUG) assert(handle >= 0 && handle < this.resourceCount, 'read of an unknown resource');
    const resource = this._resources[handle];
    resource.readers[resource.readerCount++] = passIndex;
  }

  /** Order the passes, drop the dead ones, derive load/store, assign memory. */
  compile() {
    this._cullDeadPasses();
    this._topologicalSort();
    this._computeLifetimes();
    this._assignPhysicalTextures();
    this._deriveAttachmentOps();

    this.stats.passes = this.passCount;
    this.stats.executed = this._orderCount;
    this.stats.culled = this.passCount - this._orderCount;
    this.stats.edges = this._edgeCount;
    this._compiled = true;
    return this;
  }

  /**
   * A pass is live if it writes something outside the frame can see, or if a
   * live pass reads something it writes. Worked backwards until it settles --
   * one iteration per level of the dependency chain, which for a render frame
   * is single digits.
   */
  _cullDeadPasses() {
    this._live.fill(0, 0, this.passCount);

    for (let p = 0; p < this.passCount; p++) {
      const pass = this._passes[p];
      for (let c = 0; c < pass.colorCount; c++) {
        if (this._resources[pass.color[c].resource].external) this._live[p] = 1;
      }
      for (let w = 0; w < pass.writeCount; w++) {
        if (this._resources[pass.writes[w]].external) this._live[p] = 1;
      }
      if (pass.depth && this._resources[pass.depth.resource].external) this._live[p] = 1;
    }

    let changed = true;
    while (changed) {
      changed = false;
      for (let p = 0; p < this.passCount; p++) {
        if (!this._live[p]) continue;
        const pass = this._passes[p];
        // Everything this live pass reads must be produced, so its producers
        // are live too.
        for (let r = 0; r < pass.readCount; r++) {
          const resource = this._resources[pass.reads[r]];
          for (let w = 0; w < resource.writerCount; w++) {
            const writer = resource.writers[w];
            if (!this._live[writer]) { this._live[writer] = 1; changed = true; }
          }
        }
        // And anything that wrote its attachments earlier, since a load op
        // depends on that content existing.
        for (let c = 0; c < pass.colorCount; c++) {
          changed = this._markEarlierWriters(pass.color[c].resource, p) || changed;
        }
        if (pass.depth) {
          changed = this._markEarlierWriters(pass.depth.resource, p) || changed;
        }
      }
    }
  }

  _markEarlierWriters(handle, before) {
    const resource = this._resources[handle];
    let changed = false;
    for (let w = 0; w < resource.writerCount; w++) {
      const writer = resource.writers[w];
      if (writer < before && !this._live[writer]) { this._live[writer] = 1; changed = true; }
    }
    return changed;
  }

  /** Kahn's algorithm over the live subgraph. _buildEdges defines the edges. */
  _topologicalSort() {
    const n = this.passCount;
    this._indegree.fill(0, 0, n);
    this._buildEdges(n);

    const from = this._edgeFrom;
    const to = this._edgeTo;
    const edges = this._edgeCount;

    let head = 0;
    let tail = 0;
    for (let p = 0; p < n; p++) {
      if (this._live[p] && this._indegree[p] === 0) this._queue[tail++] = p;
    }

    this._orderCount = 0;
    while (head < tail) {
      const p = this._queue[head++];
      this._order[this._orderCount++] = p;
      for (let e = 0; e < edges; e++) {
        if (from[e] === p && --this._indegree[to[e]] === 0) this._queue[tail++] = to[e];
      }
    }

    if (DEBUG) {
      let liveCount = 0;
      for (let p = 0; p < n; p++) if (this._live[p]) liveCount++;
      if (this._orderCount !== liveCount) {
        // Naming the passes that could not be ordered, and what each of them is
        // still waiting on. "Contains a cycle" alone sends you reading the
        // whole frame declaration.
        const stuck = [];
        for (let p = 0; p < n; p++) {
          if (!this._live[p] || this._order.subarray(0, this._orderCount).includes(p)) continue;
          const blockers = [];
          for (let e = 0; e < edges; e++) {
            if (to[e] === p) blockers.push(this._passes[from[e]].name);
          }
          stuck.push(`${this._passes[p].name} <- ${[...new Set(blockers)].join(', ') || 'nothing'}`);
        }
        assert(false, `render graph contains a cycle; unorderable passes:\n  ${stuck.join('\n  ')}`);
      }
    }
  }

  /**
   * Every producer -> consumer edge, read off the resources rather than found
   * by asking each pair of passes whether they are related. The pairwise form
   * was O(passes^2) per sweep and ran two sweeps; this is O(edges), and edges
   * is what the frame actually declared.
   *
   * Parallel edges are fine and deliberately not deduplicated: Kahn's algorithm
   * counts edges, and each one is decremented exactly once when its producer
   * pops, so two passes related through two resources stay balanced.
   *
   * Write-after-read is not an edge. Two resources only share memory when their
   * lifetimes do not overlap, so the hazard it would guard cannot arise.
   */
  _buildEdges(n) {
    this._edgeCount = 0;

    for (let a = 0; a < n; a++) {
      if (!this._live[a]) continue;
      const pass = this._passes[a];

      // Read-after-write, and deliberately NOT restricted to earlier passes
      // while a resource has ONE writer. Declaration order carrying meaning is
      // what this file exists to remove: declaring the consumer first must
      // still run the producer first.
      //
      // A resource written more than once is the exception, and it has to be.
      // Its writes are a SEQUENCE -- the write-after-write rule below already
      // says declaration order is what orders them -- so "the depth buffer" is
      // really several values over the frame, and a read means the one current
      // where the read was declared. Edging from every writer instead would put
      // a reader between two writes after both, which is a cycle the moment
      // the later write depends on that reader. Two-phase culling is exactly
      // that shape: the depth pyramid reads the early depth, and the late pass
      // writes depth again only because the pyramid told it what to draw.
      for (let r = 0; r < pass.readCount; r++) {
        const handle = pass.reads[r];
        const versioned = this._resources[handle].writerCount > 1;
        this._edgesFromWriters(handle, a, versioned ? a : n);
      }

      // Write-after-write. Two passes writing the same target have no data
      // dependency either way, so declaration order is the tie-break -- it is
      // the only thing that says which of four shadow cascades goes first.
      for (let c = 0; c < pass.colorCount; c++) this._edgesFromWriters(pass.color[c].resource, a, a);
      for (let w = 0; w < pass.writeCount; w++) this._edgesFromWriters(pass.writes[w], a, a);
      if (pass.depth) this._edgesFromWriters(pass.depth.resource, a, a);
    }
  }

  /** Edge from every live writer of `handle` below `limit` into pass `a`. */
  _edgesFromWriters(handle, a, limit) {
    const resource = this._resources[handle];
    for (let w = 0; w < resource.writerCount; w++) {
      const b = resource.writers[w];
      if (b === a || b >= limit || !this._live[b]) continue;

      if (this._edgeCount === this._edgeFrom.length) {
        const grown = new Uint32Array(this._edgeCount * 2);
        grown.set(this._edgeFrom); this._edgeFrom = grown;
        const grownTo = new Uint32Array(this._edgeCount * 2);
        grownTo.set(this._edgeTo); this._edgeTo = grownTo;
      }
      this._edgeFrom[this._edgeCount] = b;
      this._edgeTo[this._edgeCount] = a;
      this._edgeCount++;
      this._indegree[a]++;
    }
  }

  _writes(passIndex, handle) {
    const pass = this._passes[passIndex];
    for (let c = 0; c < pass.colorCount; c++) {
      if (pass.color[c].resource === handle) return true;
    }
    for (let w = 0; w < pass.writeCount; w++) {
      if (pass.writes[w] === handle) return true;
    }
    return pass.depth !== null && pass.depth.resource === handle;
  }

  /** First and last position in the EXECUTION order that touches each resource. */
  _computeLifetimes() {
    for (let r = 0; r < this.resourceCount; r++) {
      this._resources[r].firstUse = -1;
      this._resources[r].lastUse = -1;
    }

    for (let step = 0; step < this._orderCount; step++) {
      const pass = this._passes[this._order[step]];
      const touch = (handle) => {
        const resource = this._resources[handle];
        if (resource.firstUse < 0) resource.firstUse = step;
        resource.lastUse = step;
      };
      for (let c = 0; c < pass.colorCount; c++) touch(pass.color[c].resource);
      for (let w = 0; w < pass.writeCount; w++) touch(pass.writes[w]);
      if (pass.depth) touch(pass.depth.resource);
      for (let r = 0; r < pass.readCount; r++) touch(pass.reads[r]);
    }
  }

  /**
   * Hand every transient resource a physical texture, sharing one wherever two
   * lifetimes do not overlap.
   *
   * Resources are visited in first-use order and a texture returns to the free
   * list the moment its last reader is done, so a chain of temporaries -- a
   * bloom downsample ladder, say -- collapses onto a couple of allocations
   * instead of one per step.
   */
  _assignPhysicalTextures() {
    const transient = [];
    for (let r = 0; r < this.resourceCount; r++) {
      const resource = this._resources[r];
      resource.physical = null;
      if (!resource.imported && resource.firstUse >= 0) transient.push(r);
    }
    transient.sort((a, b) => this._resources[a].firstUse - this._resources[b].firstUse);

    this.stats.transient = transient.length;
    this.stats.aliased = 0;

    const freeByKey = new Map();
    const liveUntil = [];   // {handle, lastUse}

    for (const handle of transient) {
      const resource = this._resources[handle];

      // Release anything whose last use is behind this resource's first use.
      for (let i = liveUntil.length - 1; i >= 0; i--) {
        if (liveUntil[i].lastUse >= resource.firstUse) continue;
        const done = this._resources[liveUntil[i].handle];
        const bucket = freeByKey.get(done.key) ?? [];
        bucket.push(done.physical);
        freeByKey.set(done.key, bucket);
        liveUntil.splice(i, 1);
      }

      const bucket = freeByKey.get(resource.key);
      if (bucket && bucket.length > 0) {
        resource.physical = bucket.pop();
        this.stats.aliased++;
      } else {
        resource.physical = this._acquire(resource);
      }
      resource.view = resource.physical.view;
      liveUntil.push({ handle, lastUse: resource.lastUse });
    }
  }

  /** A texture matching this descriptor, from the cross-frame pool or new. */
  _acquire(resource) {
    let entries = this._pool.get(resource.key);
    if (!entries) {
      entries = [];
      this._pool.set(resource.key, entries);
    }
    for (const entry of entries) {
      if (entry.inUse) continue;
      entry.inUse = 1;
      return entry;
    }

    const d = resource.desc;
    const texture = this.rhi.device.createTexture({
      label: `graph:${resource.name}`,
      size: [d.width, d.height, d.depthOrArrayLayers],
      format: d.format,
      usage: d.usage,
      sampleCount: d.sampleCount,
    });
    const entry = { texture, view: texture.createView(), inUse: 1 };
    entries.push(entry);
    return entry;
  }

  /**
   * Work out every attachment's load and store op.
   *
   * This is the derivation that pays for the whole file. A depth buffer nothing
   * reads afterwards gets `discard`, which on a tile-based GPU means it is
   * never written back to main memory -- and it stays correct automatically the
   * day a pass is added that does read it.
   */
  _deriveAttachmentOps() {
    for (let step = 0; step < this._orderCount; step++) {
      const pass = this._passes[this._order[step]];
      for (let c = 0; c < pass.colorCount; c++) this._deriveOps(pass.color[c], step, pass.name);
      if (pass.depth) this._deriveOps(pass.depth, step, pass.name);
    }
  }

  _deriveOps(attachment, step, passName) {
    const resource = this._resources[attachment.resource];

    const writtenEarlier = this._writtenBefore(attachment.resource, step);
    if (attachment.clear !== undefined) {
      attachment.loadOp = 'clear';
    } else if (writtenEarlier) {
      attachment.loadOp = 'load';
    } else {
      // Loading a texture nothing has written yields undefined contents. That
      // is a bug in the declaration, not something to paper over.
      if (DEBUG) {
        assert(false, `${passName}: attachment "${resource.name}" has no clear value and nothing wrote it earlier`);
      }
      attachment.loadOp = 'clear';
      attachment.clear = 0;
    }

    attachment.storeOp = (resource.external || resource.lastUse > step) ? 'store' : 'discard';
  }

  _writtenBefore(handle, step) {
    const resource = this._resources[handle];
    for (let w = 0; w < resource.writerCount; w++) {
      const position = this._positionOf(resource.writers[w]);
      if (position >= 0 && position < step) return true;
    }
    return false;
  }

  _positionOf(passIndex) {
    for (let step = 0; step < this._orderCount; step++) {
      if (this._order[step] === passIndex) return step;
    }
    return -1;
  }

  /** Record every live pass, in dependency order. */
  execute(encoder) {
    if (DEBUG) assert(this._compiled, 'RenderGraph: execute() before compile()');

    for (let step = 0; step < this._orderCount; step++) {
      const pass = this._passes[this._order[step]];

      // Undefined when there is no profiler, which is what a pass descriptor
      // wants for "do not time this".
      const timestampWrites = this.profiler?.writesFor(step, pass.name);

      if (pass.type === 'compute') {
        const computePass = encoder.beginComputePass({ label: pass.name, timestampWrites });
        pass.execute(computePass, this);
        computePass.end();
        continue;
      }

      const colorAttachments = [];
      for (let c = 0; c < pass.colorCount; c++) {
        const attachment = pass.color[c];
        colorAttachments.push({
          view: attachment.view ?? this._resources[attachment.resource].view,
          loadOp: attachment.loadOp,
          storeOp: attachment.storeOp,
          clearValue: attachment.clear ?? undefined,
        });
      }

      let depthStencilAttachment;
      if (pass.depth) {
        depthStencilAttachment = {
          view: pass.depth.view ?? this._resources[pass.depth.resource].view,
          depthLoadOp: pass.depth.loadOp,
          depthStoreOp: pass.depth.storeOp,
          depthClearValue: pass.depth.clear ?? undefined,
        };
      }

      const encoded = encoder.beginRenderPass({
        label: pass.name,
        colorAttachments,
        depthStencilAttachment,
        timestampWrites,
      });
      pass.execute(encoded, this);
      encoded.end();
    }
  }

  /** The view a resource currently resolves to. Valid after compile(). */
  viewOf(handle) {
    return this._resources[handle].view;
  }

  /** Human-readable plan, for when a pass is not running and you need to know why. */
  describe() {
    const lines = [];
    for (let step = 0; step < this._orderCount; step++) {
      const pass = this._passes[this._order[step]];
      const parts = [];
      for (let c = 0; c < pass.colorCount; c++) {
        parts.push(`color ${this._resources[pass.color[c].resource].name}:${pass.color[c].loadOp}/${pass.color[c].storeOp}`);
      }
      if (pass.depth) {
        parts.push(`depth ${this._resources[pass.depth.resource].name}:${pass.depth.loadOp}/${pass.depth.storeOp}`);
      }
      for (let r = 0; r < pass.readCount; r++) {
        parts.push(`reads ${this._resources[pass.reads[r]].name}`);
      }
      lines.push(`${step}. ${pass.name}  [${parts.join(', ')}]`);
    }
    for (let p = 0; p < this.passCount; p++) {
      if (!this._live[p]) lines.push(`-- ${this._passes[p].name}  (culled: nothing consumes it)`);
    }
    return lines.join('\n');
  }

  destroy() {
    for (const entries of this._pool.values()) {
      for (const entry of entries) entry.texture.destroy();
    }
    this._pool.clear();
  }
}

function makePassRecord() {
  return {
    name: '',
    execute: null,
    color: [],
    colorCount: 0,
    depth: null,
    _depth: { resource: 0, view: null, clear: undefined, loadOp: 'clear', storeOp: 'store' },
    reads: [],
    readCount: 0,
    writes: [],
    writeCount: 0,
    type: 'render',
  };
}

function makeResourceRecord() {
  return {
    name: '',
    kind: 'texture',
    buffer: null,
    imported: false,
    external: false,
    view: null,
    desc: null,
    key: '',
    physical: null,
    firstUse: -1,
    lastUse: -1,
    writers: [],
    writerCount: 0,
    readers: [],
    readerCount: 0,
  };
}
