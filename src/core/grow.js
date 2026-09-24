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
