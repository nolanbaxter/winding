// Transform hierarchy.
//
// Every entity has a transform, so the columns are indexed by entity index
// directly -- no sparse set, no map.
//
// The whole design turns on one idea: **evaluate in depth order**. If parents
// are always visited before their children, then composing the hierarchy is a
// flat loop over an array, not a recursive tree walk, and a child can tell
// whether its parent moved by reading a flag the parent just wrote.
//
// That also gives dirty propagation for free. The naive approach marks a whole
// subtree dirty whenever a node moves, which costs O(subtree) per write and
// does it again for every ancestor that moves in the same frame. Here a write
// marks exactly one node, and update() discovers the rest:
//
//     recompute = dirty[self] || recomputed[parent]
//
// O(1) per write, and each node is composed at most once per frame no matter
// how many of its ancestors moved.

import { DEBUG, assert, assertFinite } from '../core/assert.js';
import { mat4Copy } from '../core/math/mat4.js';
import { handleIndex, NULL_HANDLE } from '../core/handle.js';
import {
  sharedFloat32Array, sharedInt32Array, sharedUint32Array, sharedUint8Array,
} from '../core/shared.js';
import { composeRange, buffersFromColumns } from './transformJob.js';
import { grownCapacity, growArray, growShared } from '../core/grow.js';
import { JOB_COMPOSE_TRANSFORMS } from '../core/jobs.js';

export const NO_PARENT = -1;

export class TransformStore {
  constructor(capacity) {
    this.capacity = capacity;

    // One contiguous column per component. Iteration is a linear memory walk;
    // the offsets below (i * 3, i * 4, i * 16) are why mat4 functions take
    // trailing offsets -- nothing is copied into scratch to be composed.
    this.position = sharedFloat32Array(capacity * 3);
    this.rotation = sharedFloat32Array(capacity * 4);
    this.scale = sharedFloat32Array(capacity * 3);
    this.local = sharedFloat32Array(capacity * 16);
    this.world = sharedFloat32Array(capacity * 16);

    this.parent = sharedInt32Array(capacity).fill(NO_PARENT);
    this.used = new Uint8Array(capacity);
    this.dirty = sharedUint8Array(capacity);
    /** Set by update() for nodes it recomputed. Children read their parent's. */
    this.recomputed = sharedUint8Array(capacity);
    /**
     * Sticky "has moved since a consumer last looked", as opposed to
     * `recomputed`, which is only valid within the update that set it.
     *
     * They have to be separate. composeRange CLEARS recomputed for a node that
     * did not move, because the next level reads it to decide whether a parent
     * moved -- so a second update() with nothing dirty wipes the record of the
     * first. Anything outside the compose loop that needs to know what changed,
     * which is the GPU upload and the world bounds, reads this instead and
     * clears it when it has acted. update() is public, so "call it twice and
     * the renderer misses the change" is reachable from ordinary use.
     */
    this.moved = sharedUint8Array(capacity);

    /** Entity indices in depth order: every parent precedes every child. */
    this.order = sharedUint32Array(capacity);
    /** Where each depth level begins in `order`, plus a terminator. Depth is
     *  what makes a level safe to run in parallel. */
    this.levelStart = new Uint32Array(capacity + 1);
    this.levelCount = 0;
    this.orderCount = 0;
    this.orderDirty = false;

    /** How many nodes update() actually composed. Should be ~0 in a still scene. */
    this.lastRecomputedCount = 0;

    // Highest index ever used + 1, so scans cover live data instead of capacity.
    this._high = 0;

    this._depth = new Int32Array(capacity);
    this._chain = new Uint32Array(capacity);
    this._counts = new Int32Array(capacity + 1);

    /** Bumped by _grow. Workers holding older buffers are writing into limbo. */
    this.buffersRevision = 0;
    this._publishedRevision = -1;
  }

  /**
   * Attach a transform to an entity. Omitted fields default to identity, so
   * add(e) alone is a valid, complete transform.
   */
  add(entity, { position, rotation, scale, parent } = {}) {
    const i = handleIndex(entity);
    if (i >= this.capacity) this._grow(i + 1);

    const p3 = i * 3;
    this.position[p3] = position?.[0] ?? 0;
    this.position[p3 + 1] = position?.[1] ?? 0;
    this.position[p3 + 2] = position?.[2] ?? 0;

    const r4 = i * 4;
    this.rotation[r4] = rotation?.[0] ?? 0;
    this.rotation[r4 + 1] = rotation?.[1] ?? 0;
    this.rotation[r4 + 2] = rotation?.[2] ?? 0;
    this.rotation[r4 + 3] = rotation?.[3] ?? 1;

    this.scale[p3] = scale?.[0] ?? 1;
    this.scale[p3 + 1] = scale?.[1] ?? 1;
    this.scale[p3 + 2] = scale?.[2] ?? 1;

    this.parent[i] = NO_PARENT;
    this.used[i] = 1;
    this.dirty[i] = 1;
    this.recomputed[i] = 0;
    this.orderDirty = true;
    if (i >= this._high) this._high = i + 1;

    if (parent !== undefined && parent !== NULL_HANDLE) this.setParent(entity, parent);
    return i;
  }

  /**
   * Detach. Children are re-rooted rather than orphaned: leaving them pointing
   * at a dead slot would silently compose against a stale world matrix, which
   * is precisely the kind of quiet wrongness that must not pass silently.
   *
   * ponytail: O(live nodes) scan for children. A per-node child list would make
   * it O(children); add one if teardown ever shows up in a profile.
   */
  remove(entity) {
    const i = handleIndex(entity);
    if (!this.used[i]) return;

    for (let c = 0; c < this._high; c++) {
      if (this.used[c] && this.parent[c] === i) {
        this.parent[c] = NO_PARENT;
        this.dirty[c] = 1;
      }
    }

    this.used[i] = 0;
    this.parent[i] = NO_PARENT;
    this.dirty[i] = 0;
    this.recomputed[i] = 0;
    this.moved[i] = 0;
    this.orderDirty = true;
  }

  /** Pass NULL_HANDLE to detach to the root. */
  setParent(entity, parentEntity) {
    const i = handleIndex(entity);
    const p = parentEntity === NULL_HANDLE ? NO_PARENT : handleIndex(parentEntity);

    if (DEBUG) {
      assert(this.used[i], 'setParent: entity has no transform');
      assert(p === NO_PARENT || this.used[p], 'setParent: parent has no transform');
    }

    // Unconditional, not a DEBUG assert: a cycle makes _rebuildOrder loop
    // forever. An infinite hang is worse than any exception.
    for (let cur = p, guard = 0; cur !== NO_PARENT; cur = this.parent[cur]) {
      if (cur === i) throw new Error('setParent: would create a cycle in the transform hierarchy');
      if (++guard > this.capacity) throw new Error('setParent: existing hierarchy contains a cycle');
    }

    this.parent[i] = p;
    this.dirty[i] = 1;
    this.orderDirty = true;
  }

  setPosition(entity, x, y, z) {
    const o = handleIndex(entity) * 3;
    this.position[o] = x; this.position[o + 1] = y; this.position[o + 2] = z;
    // Checked HERE rather than in the composition loop: this is where a bad
    // value enters, and the loop runs per node per frame.
    if (DEBUG) assertFinite(this.position, 'setPosition', o, 3);
    this.dirty[handleIndex(entity)] = 1;
  }

  setScale(entity, x, y, z) {
    const o = handleIndex(entity) * 3;
    this.scale[o] = x; this.scale[o + 1] = y; this.scale[o + 2] = z;
    if (DEBUG) assertFinite(this.scale, 'setScale', o, 3);
    this.dirty[handleIndex(entity)] = 1;
  }

  setRotation(entity, q) {
    const o = handleIndex(entity) * 4;
    this.rotation[o] = q[0]; this.rotation[o + 1] = q[1];
    this.rotation[o + 2] = q[2]; this.rotation[o + 3] = q[3];
    if (DEBUG) assertFinite(this.rotation, 'setRotation', o, 4);
    this.dirty[handleIndex(entity)] = 1;
  }

  /** Byte-free accessor: index into `world` for callers that read in place. */
  worldOffset(entity) {
    return handleIndex(entity) * 16;
  }

  worldMatrixInto(out, entity) {
    return mat4Copy(out, this.world, 0, handleIndex(entity) * 16);
  }

  /**
   * Compose every world matrix that changed. One linear pass, no recursion.
   * Returns how many nodes were recomputed.
   */
  update() {
    if (this.orderDirty) this._rebuildOrder();
    composeRange(this, 0, 0, this.orderCount);
    this.lastRecomputedCount = this._countRecomputed();
    return this.lastRecomputedCount;
  }

  /**
   * The same work, one depth level at a time, spread across threads.
   *
   * Each level is a barrier: dispatch() does not return until every item in it
   * is finished, which is exactly the guarantee children at the next depth need.
   */
  updateParallel(jobs) {
    if (this.orderDirty) this._rebuildOrder();

    // Growing replaced the shared columns. Workers are still holding the old
    // ones until this lands, and nothing else in the engine is positioned to
    // notice -- so it is checked here, at the only point that hands them out.
    if (this._publishedRevision !== this.buffersRevision) {
      jobs.setSharedData(this.sharedBuffers());
      this._publishedRevision = this.buffersRevision;
    }

    for (let level = 0; level < this.levelCount; level++) {
      const base = this.levelStart[level];
      const size = this.levelStart[level + 1] - base;
      if (size > 0) jobs.dispatch(JOB_COMPOSE_TRANSFORMS, size, { arg0: base });
    }

    this.lastRecomputedCount = this._countRecomputed();
    return this.lastRecomputedCount;
  }

  _countRecomputed() {
    let count = 0;
    for (let k = 0; k < this.orderCount; k++) {
      if (this.recomputed[this.order[k]] === 1) count++;
    }
    return count;
  }

  /**
   * Widen every column to hold at least `needed` entities.
   *
   * The shared columns become NEW SharedArrayBuffers, which is the part that
   * matters: a worker still holding the old ones would compose into memory
   * nobody reads, and the symptom would be some nodes silently freezing. That
   * is what `buffersRevision` exists for -- updateParallel republishes before
   * it dispatches, so no caller has to know this happened.
   */
  _grow(needed) {
    const capacity = grownCapacity(this.capacity, needed);

    this.position = growShared(this.position, capacity, sharedFloat32Array, 3);
    this.rotation = growShared(this.rotation, capacity, sharedFloat32Array, 4);
    this.scale = growShared(this.scale, capacity, sharedFloat32Array, 3);
    this.local = growShared(this.local, capacity, sharedFloat32Array, 16);
    this.world = growShared(this.world, capacity, sharedFloat32Array, 16);

    // NO_PARENT is -1, not 0, so the new tail has to be filled rather than left
    // zeroed -- a zeroed tail would make every fresh slot a child of entity 0.
    const parent = growShared(this.parent, capacity, sharedInt32Array);
    parent.fill(NO_PARENT, this.capacity);
    this.parent = parent;

    this.used = growArray(this.used, capacity);
    this.dirty = growShared(this.dirty, capacity, sharedUint8Array);
    this.recomputed = growShared(this.recomputed, capacity, sharedUint8Array);
    this.moved = growShared(this.moved, capacity, sharedUint8Array);
    this.order = growShared(this.order, capacity, sharedUint32Array);

    this.levelStart = growArray(this.levelStart, capacity + 1);
    this._depth = growArray(this._depth, capacity);
    this._chain = growArray(this._chain, capacity);
    this._counts = growArray(this._counts, capacity + 1);

    this.capacity = capacity;
    this.buffersRevision++;
  }

  /** The columns a worker needs, as raw shared buffers. */
  sharedBuffers() {
    return buffersFromColumns(this);
  }

  /**
   * Sort live nodes by depth. Counting sort, so it is O(n) rather than
   * O(n log n) -- depth is a small non-negative integer, which is exactly the
   * case a comparison sort is wasteful for.
   */
  _rebuildOrder() {
    const { parent, used, _depth: depth, _chain: chain, _counts: counts } = this;
    const high = this._high;

    depth.fill(-1, 0, high);
    let maxDepth = 0;

    for (let e = 0; e < high; e++) {
      if (!used[e] || depth[e] >= 0) continue;

      // Walk up until a node whose depth is known (or a root), recording the
      // chain, then assign depths back down it. Every node resolves once, so
      // the whole rebuild stays O(n) even for deep hierarchies.
      let n = 0;
      let cur = e;
      while (cur !== NO_PARENT && depth[cur] < 0) {
        chain[n++] = cur;
        cur = parent[cur];
      }

      let d = cur !== NO_PARENT ? depth[cur] : -1;
      while (n > 0) {
        const node = chain[--n];
        depth[node] = ++d;
        if (d > maxDepth) maxDepth = d;
      }
    }

    counts.fill(0, 0, maxDepth + 1);
    for (let e = 0; e < high; e++) if (used[e]) counts[depth[e]]++;

    let total = 0;
    for (let d = 0; d <= maxDepth; d++) {
      const c = counts[d];
      counts[d] = total;
      this.levelStart[d] = total;
      total += c;
    }
    this.levelStart[maxDepth + 1] = total;
    this.levelCount = maxDepth + 1;
    for (let e = 0; e < high; e++) if (used[e]) this.order[counts[depth[e]]++] = e;

    this.orderCount = total;
    this.orderDirty = false;
  }
}
