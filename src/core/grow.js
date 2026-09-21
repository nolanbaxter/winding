// Growing the engine's fixed-size arrays.
//
// Every container here was sized at construction and threw when it filled,
// which made the constructor argument a number you had to guess -- exactly the
// magic number this engine avoids, just relocated into the caller's code. The
// only honest default is one that stops mattering.
//

/** Doubles until `needed` fits. */
export function grownCapacity(current, needed) {
  let capacity = Math.max(current, 1);
  while (capacity < needed) capacity *= 2;
  return capacity;
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
