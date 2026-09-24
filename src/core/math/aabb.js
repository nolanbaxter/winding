// Axis-aligned bounding boxes, and the ray tests that run against them.
//
// Stored as two vec3s rather than center+extent: the min/max form is what glTF
// accessors hand you and what the frustum test wants, so anything else would
// mean converting twice.

import { hypot3 } from './vec3.js';

/**
 * Transform a local-space AABB into world space, producing the tightest
 * axis-aligned box that still contains the rotated one.
 *
 * The obvious implementation transforms all eight corners and takes their
 * bounds: 8 matrix-vector products, 24 multiplies each, so 72. Arvo's method
 * gets the same answer in 18, by noticing that each output axis is just
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
  // An empty box (min above max -- what a primitive with no vertices has)
  // stays empty. Transformed, its infinities met the matrix's zeros and came
  // out NaN, which no cull test ever rejects.
  if (!(min[inOff] <= max[inOff])) {
    for (let i = 0; i < 3; i++) {
      outMin[outOff + i] = Infinity;
      outMax[outOff + i] = -Infinity;
    }
    return;
  }

  // The same loop as before -- for each world axis i, start at the
  // translation and add each local axis j's contribution, smaller end to the
  // minimum -- with the inner three steps written out. V8 did not unroll it
  // itself, and this runs for every moved renderable every frame: twice as
  // fast (62 -> 34 us at 1,700), and the same results to the bit.
  const x0 = min[inOff], y0 = min[inOff + 1], z0 = min[inOff + 2];
  const x1 = max[inOff], y1 = max[inOff + 1], z1 = max[inOff + 2];
  for (let i = 0; i < 3; i++) {
    let lo = m[mOff + 12 + i];
    let hi = lo;

    let e = m[mOff + i];
    let a = e * x0;
    let b = e * x1;
    if (a < b) { lo += a; hi += b; } else { lo += b; hi += a; }

    e = m[mOff + 4 + i];
    a = e * y0;
    b = e * y1;
    if (a < b) { lo += a; hi += b; } else { lo += b; hi += a; }

    e = m[mOff + 8 + i];
    a = e * z0;
    b = e * z1;
    if (a < b) { lo += a; hi += b; } else { lo += b; hi += a; }

    outMin[outOff + i] = lo;
    outMax[outOff + i] = hi;
  }
}

/** Bounding sphere of an AABB. Cheaper to test than the box, and looser. */
export function aabbBoundingSphere(outCenter, min, max) {
  outCenter[0] = (min[0] + max[0]) * 0.5;
  outCenter[1] = (min[1] + max[1]) * 0.5;
  outCenter[2] = (min[2] + max[2]) * 0.5;
  return hypot3(
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
    // + 0 turns -0 into +0. A local-space ray through a mirrored matrix picks
    // up -0 components naturally, and 1 / -0 is -Infinity: the same ray then
    // hit or missed a box depending on the sign of a zero.
    const inverse = 1 / (direction[i] + 0);
    let near = (min[boundsOff + i] - origin[i]) * inverse;
    let far = (max[boundsOff + i] - origin[i]) * inverse;
    if (near > far) { const swap = near; near = far; far = swap; }

    if (near > enter) enter = near;
    if (far < exit) exit = far;
    if (enter > exit) return -1;
  }

  return enter;
}

/**
 * Distance along a ray to where it meets a triangle, or -1 for a miss.
 *
 * Moller-Trumbore, without a backface cull: a pick should find a surface from
 * either side, and glTF materials are double-sided often enough that culling
 * here would make the answer depend on winding.
 *
 * `a`, `b` and `c` index `positions` as flat xyz, so callers pass `index * 3`.
 *
 * Like the slab test above, a degenerate triangle is not special-cased. A zero
 * determinant makes the barycentrics +/-Infinity or NaN, and every path out of
 * those fails one of the range checks, so the miss falls out of the arithmetic.
 *
 * `direction` need not be normalized. The result is in units of its length,
 * which is what lets the caller hand in a ray transformed into local space and
 * compare the answer against world-space distances unchanged.
 */
export function rayTriangleDistance(origin, direction, positions, a, b, c) {
  const e1x = positions[b] - positions[a];
  const e1y = positions[b + 1] - positions[a + 1];
  const e1z = positions[b + 2] - positions[a + 2];
  const e2x = positions[c] - positions[a];
  const e2y = positions[c + 1] - positions[a + 1];
  const e2z = positions[c + 2] - positions[a + 2];

  // p = direction x e2, and det = e1 . p is the scalar triple product: the
  // volume of the parallelepiped the three edges span, zero when the ray is
  // parallel to the triangle's plane.
  const px = direction[1] * e2z - direction[2] * e2y;
  const py = direction[2] * e2x - direction[0] * e2z;
  const pz = direction[0] * e2y - direction[1] * e2x;
  const inverse = 1 / (e1x * px + e1y * py + e1z * pz);

  const tx = origin[0] - positions[a];
  const ty = origin[1] - positions[a + 1];
  const tz = origin[2] - positions[a + 2];

  const u = (tx * px + ty * py + tz * pz) * inverse;
  if (!(u >= 0 && u <= 1)) return -1;

  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;

  const v = (direction[0] * qx + direction[1] * qy + direction[2] * qz) * inverse;
  if (!(v >= 0 && u + v <= 1)) return -1;

  const distance = (e2x * qx + e2y * qy + e2z * qz) * inverse;
  return distance >= 0 ? distance : -1;
}
