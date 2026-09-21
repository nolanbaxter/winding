// Axis-aligned bounding boxes.
//
// Stored as two vec3s rather than center+extent: the min/max form is what glTF
// accessors hand you and what the frustum test wants, so anything else would
// mean converting twice.

/**
 * Transform a local-space AABB into world space, producing the tightest
 * axis-aligned box that still contains the rotated one.
 *
 * The obvious implementation transforms all eight corners and takes their
 * bounds: 8 matrix-vector products, 24 multiplies each. Arvo's method gets the
 * same answer in 9 multiplies total, by noticing that each output axis is just
 * the translation plus, for every input axis, whichever of (row * min) and
 * (row * max) is smaller (for the min) or larger (for the max).
 *
 * The result is conservative: rotating a box makes its axis-aligned bounds
 * grow. Re-transforming an already-transformed box repeatedly inflates it,
 * which is why world bounds are always recomputed from LOCAL bounds.
 */
export function aabbTransform(
  outMin, outMax, min, max, m, mOff = 0, outOff = 0, inOff = 0,
) {
  for (let i = 0; i < 3; i++) {
    // Start at the translation component of this axis.
    let lo = m[mOff + 12 + i];
    let hi = lo;

    for (let j = 0; j < 3; j++) {
      // Column j, row i -- the contribution of local axis j to world axis i.
      const e = m[mOff + j * 4 + i];
      const a = e * min[inOff + j];
      const b = e * max[inOff + j];
      if (a < b) { lo += a; hi += b; } else { lo += b; hi += a; }
    }

    outMin[outOff + i] = lo;
    outMax[outOff + i] = hi;
  }
}

/** Bounding sphere of an AABB. Cheaper to test than the box, and looser. */
export function aabbBoundingSphere(outCenter, min, max) {
  outCenter[0] = (min[0] + max[0]) * 0.5;
  outCenter[1] = (min[1] + max[1]) * 0.5;
  outCenter[2] = (min[2] + max[2]) * 0.5;
  return Math.hypot(
    max[0] - outCenter[0],
    max[1] - outCenter[1],
    max[2] - outCenter[2],
  );
}

/** Grow `min`/`max` to contain another box. */
export function aabbUnion(min, max, otherMin, otherMax) {
  for (let i = 0; i < 3; i++) {
    if (otherMin[i] < min[i]) min[i] = otherMin[i];
    if (otherMax[i] > max[i]) max[i] = otherMax[i];
  }
}

export function aabbSetEmpty(min, max) {
  for (let i = 0; i < 3; i++) {
    min[i] = Infinity;
    max[i] = -Infinity;
  }
}

/**
 * Distance along a ray to where it first enters a box, or -1 for a miss.
 *
 * The slab method: the box is the intersection of three pairs of parallel
 * planes, so the ray is inside it exactly on the overlap of the three intervals
 * where it is between each pair. Entry is the latest interval start, exit the
 * earliest end, and they cross if and only if the ray misses.
 *
 * Division by a zero direction component is deliberately NOT special-cased.
 * IEEE gives +/-Infinity there and the comparisons below still order correctly.
 * A component that is exactly zero AND an origin exactly on that slab's plane
 * yields 0 * Infinity = NaN; every comparison against NaN is false, so that axis
 * simply contributes no constraint. That is the right answer -- the ray lies in
 * the plane, so the slab cannot rule it out -- and it makes an exactly grazing
 * ray a hit.
 *
 * What this does NOT do is validate the ray. A non-finite `direction` makes
 * every axis contribute nothing, and the function then reports a hit at 0 on
 * whatever it was handed. Callers taking a ray from outside must check it;
 * Scene.raycast does.
 *
 * An origin already inside the box returns 0, not the far wall: the caller
 * asked where the ray meets the box, and it meets it immediately.
 *
 * `boundsOff` indexes min/max as packed vec3 columns, matching the world bounds
 * the scene keeps.
 */
export function aabbRayDistance(min, max, origin, direction, boundsOff = 0) {
  let enter = 0;
  let exit = Infinity;

  for (let i = 0; i < 3; i++) {
    const inverse = 1 / direction[i];
    let near = (min[boundsOff + i] - origin[i]) * inverse;
    let far = (max[boundsOff + i] - origin[i]) * inverse;
    if (near > far) { const swap = near; near = far; far = swap; }

    if (near > enter) enter = near;
    if (far < exit) exit = far;
    if (enter > exit) return -1;
  }

  return enter;
}
