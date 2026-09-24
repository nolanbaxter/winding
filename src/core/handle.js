// Handles -- u32 references with generation tags.
//
//   31                    8 7        0
//  +------------------------+---------+
//  |       index (24)       | gen (8) |
//  +------------------------+---------+
//
// A handle is an array index plus a generation counter. Freeing a slot bumps
// its generation, so every handle that still points there stops validating.
// A stale reference fails a two-instruction check instead of silently
// resurrecting a dead object or keeping it alive forever.
//
// This is how you get safe references without leaning on the GC, and it is
// what makes structure-of-arrays storage usable at all: the index IS the
// column offset, so component lookup is arr[index(h)] with no indirection.

import { grownCapacity, growArray } from './grow.js';

const GEN_BITS = 8;
const GEN_MASK = 0xff;
const MAX_INDEX = (1 << 24) - 1;   // the largest index, so 16,777,216 slots

/**
 * Generation 0 is never issued, which makes handle 0 permanently invalid and
 * therefore usable as the null handle. Without this reservation, index 0 at
 * generation 0 would be a legitimate handle equal to 0.
 */
export const NULL_HANDLE = 0;

export function handleIndex(h) {
  return h >>> GEN_BITS;
}

export function handleGeneration(h) {
  return h & GEN_MASK;
}

export class HandleAllocator {
  constructor(capacity) {
    if (capacity > MAX_INDEX + 1) {
      throw new Error(`HandleAllocator: capacity ${capacity} exceeds 24-bit index space`);
    }
    this.capacity = capacity;
    /** Current generation per slot. 0 means "never allocated". */
    this.generations = new Uint8Array(capacity);
    /** Stack of reusable indices. Freed slots are reused before fresh ones. */
    this.freeList = new Uint32Array(capacity);
    this.freeCount = 0;
    /** High-water mark: slots [0, next) have been handed out at least once. */
    this.next = 0;
    this.liveCount = 0;
  }

  /**
   * Widen to hold at least `needed` slots.
   *
   * The 24-bit ceiling stays a hard error: it is not a chosen capacity but the
   * width of the index field in a handle, so exceeding it cannot be absorbed by
   * allocating more memory.
   */
  _grow(needed) {
    const capacity = grownCapacity(this.capacity, needed);
    if (capacity > MAX_INDEX + 1) {
      throw new Error(
        `HandleAllocator: ${needed} slots exceeds the 24-bit index space (${MAX_INDEX + 1})`,
      );
    }
    this.generations = growArray(this.generations, capacity);
    this.freeList = growArray(this.freeList, capacity);
    this.capacity = capacity;
  }

  alloc() {
    let index;
    if (this.freeCount > 0) {
      index = this.freeList[--this.freeCount];
    } else {
      if (this.next >= this.capacity) this._grow(this.next + 1);
      index = this.next++;
      this.generations[index] = 1;   // first issue; 0 stays reserved
    }
    this.liveCount++;
    // >>> 0 because JS bitwise ops are signed 32-bit: a high index would
    // otherwise produce a negative handle.
    return ((index << GEN_BITS) | this.generations[index]) >>> 0;
  }

  alive(h) {
    const gen = h & GEN_MASK;
    if (gen === 0) return false;              // NULL_HANDLE, and never-issued slots
    const index = h >>> GEN_BITS;
    return index < this.next && this.generations[index] === gen;
  }

  /**
   * The live handle for a slot: its index with the generation it has now.
   * For turning a slot found in some other column -- a parent link, say --
   * back into the handle that owns it. Meaningful only for a slot in use.
   */
  handleAt(index) {
    return ((index << GEN_BITS) | this.generations[index]) >>> 0;
  }

  /** Double-free and stale-free are rejected, not tolerated. */
  free(h) {
    if (!this.alive(h)) {
      throw new Error(`HandleAllocator: free of dead or invalid handle ${h >>> 0}`);
    }
    const index = h >>> GEN_BITS;
    this.liveCount--;
    // A slot out of generations is retired, never reused. It used to wrap back
    // to 1, and the free list is a stack, so one spawn and despawn a frame
    // cycled the same slot 255 times in four seconds -- after which a Node
    // kept from the first spawn passed alive() and moved whatever owned the
    // slot now. Retiring costs one index per 255 reuses of a slot; generation
    // 0 already means "no handle", so nothing can match it.
    const nextGen = (this.generations[index] + 1) & GEN_MASK;
    this.generations[index] = nextGen;
    if (nextGen !== 0) this.freeList[this.freeCount++] = index;
  }
}
