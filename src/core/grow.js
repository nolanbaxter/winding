// Growing the engine's fixed-size arrays.
//
// Every container here was sized at construction and threw when it filled,
// which made the constructor argument a number you had to guess -- exactly the
// magic number this engine avoids, just relocated into the caller's code. The
// only honest default is one that stops mattering.
//

/**
 * Doubles until `needed` fits, never past `ceiling`.
 *
 * The ceiling is what the device can hold, and going past it is a throw, not
 * a clamp. Doubling regardless made a buffer WebGPU rejects without throwing:
 * the bind group naming it failed, and the frame went black with the cause
 * several calls away.
 */
export function grownCapacity(current, needed, ceiling = Infinity, what = 'items') {
  if (needed > ceiling) {
    throw new RangeError(`${needed} ${what} is past the ${ceiling} this device can hold`);
  }
  let capacity = Math.max(current, 1);
  while (capacity < needed) capacity *= 2;
  return Math.min(capacity, ceiling);
}

/**
 * Ranges of a growable arena: a freed hole first, if one is big enough, else
 * the end. `end` is how far it has been handed out; a caller whose buffer is
 * smaller than that grows it.
 *
 * ponytail: first fit, no compaction -- a hole smaller than every later range
 * stays a hole. Compacting means moving live data and rewriting every offset
 * into it; worth it only if fragmentation shows up.
 */
export class RangeAllocator {
  constructor() {
    this.end = 0;
    /** Freed ranges below `end`, as { base, length }, sorted by base. */
    this._holes = [];
  }

  /** Where `length` items go. */
  alloc(length) {
    for (let h = 0; h < this._holes.length; h++) {
      const hole = this._holes[h];
      if (hole.length < length) continue;
      const base = hole.base;
      hole.base += length;
      hole.length -= length;
      if (hole.length === 0) this._holes.splice(h, 1);
      return base;
    }
    const base = this.end;
    this.end += length;
    return base;
  }

  /** Give back `length` items at `base`. Holes merge; one that reaches the end shortens it. */
  free(base, length) {
    if (length === 0) return;
    let at = 0;
    while (at < this._holes.length && this._holes[at].base < base) at++;
    this._holes.splice(at, 0, { base, length });

    const next = this._holes[at + 1];
    if (next && base + length === next.base) {
      this._holes[at].length += next.length;
      this._holes.splice(at + 1, 1);
    }
    const previous = this._holes[at - 1];
    if (previous && previous.base + previous.length === base) {
      previous.length += this._holes[at].length;
      this._holes.splice(at, 1);
    }

    const last = this._holes[this._holes.length - 1];
    if (last && last.base + last.length === this.end) {
      this.end = last.base;
      this._holes.pop();
    }
  }
}

/** `elementsPerItem` grows a column of packed vec3s or mat4s by item count. */
export function growArray(array, capacity, elementsPerItem = 1) {
  const grown = new array.constructor(capacity * elementsPerItem);
  grown.set(array);
  return grown;
}

/**
 * Same, into shared memory.
 *
 * The result is a NEW SharedArrayBuffer, so a worker still holding the old one
 * writes where nobody reads. Callers that hand columns to workers must
 * republish them.
 */
export function growShared(array, capacity, allocate, elementsPerItem = 1) {
  const grown = allocate(capacity * elementsPerItem);
  grown.set(array);
  return grown;
}
