// View frustum extraction and tests.
//
// FIVE planes, not six. That is not a shortcut -- it falls out of the
// projection.
//
// The standard Gribb-Hartmann extraction reads each clip-space inequality off
// a combination of the viewProjection matrix's rows. For WebGPU the clip
// volume is -w <= x <= w, -w <= y <= w, 0 <= z <= w, giving:
//
//   left   = row3 + row0        x >= -w
//   right  = row3 - row0        x <=  w
//   bottom = row3 + row1        y >= -w
//   top    = row3 - row1        y <=  w
//   near   = row3 - row2        z <=  w     <- reverse-Z: near is the FAR side
//   far    = row2               z >=  0     <- degenerate, see below
//
// With reverse-Z the depth mapping is z = near/distance, so z >= 0 means
// "distance <= infinity". Always true. That plane's normal is the zero vector,
// and normalizing it divides by zero: every test then returns NaN, NaN >= 0 is
// false, and the renderer silently culls the entire scene. So it is not
// extracted at all.
//
// An ORTHOGRAPHIC camera does have a real far plane, and it is left out there
// too. Skipping a side can only keep more, never cull wrongly: something past
// the far plane is drawn and the rasteriser clips it. Culling it here would buy
// back draws that are beyond a far plane chosen to be generous in the first
// place.
//
// Planes are (a, b, c, d) with a point inside when a*x + b*y + c*z + d >= 0,
// normalized so that expression is the signed distance in world units.

import { hypot3 } from './vec3.js';

export const PLANE_LEFT = 0;
export const PLANE_RIGHT = 1;
export const PLANE_BOTTOM = 2;
export const PLANE_TOP = 3;
export const PLANE_NEAR = 4;
export const FRUSTUM_PLANE_COUNT = 5;

export function frustumCreate() {
  return new Float32Array(FRUSTUM_PLANE_COUNT * 4);
}

/**
 * Extract world-space planes from a viewProjection matrix.
 *
 * Feeding this P*V gives world-space planes; feeding it P alone gives
 * view-space ones. Both are useful; the renderer uses the former so bounds
 * never have to leave world space.
 */
export function frustumFromViewProjection(out, m) {
  // Rows of a column-major matrix: row r is (m[r], m[4+r], m[8+r], m[12+r]).
  const r0x = m[0], r0y = m[4], r0z = m[8], r0w = m[12];
  const r1x = m[1], r1y = m[5], r1z = m[9], r1w = m[13];
  const r2x = m[2], r2y = m[6], r2z = m[10], r2w = m[14];
  const r3x = m[3], r3y = m[7], r3z = m[11], r3w = m[15];

  setPlane(out, PLANE_LEFT, r3x + r0x, r3y + r0y, r3z + r0z, r3w + r0w);
  setPlane(out, PLANE_RIGHT, r3x - r0x, r3y - r0y, r3z - r0z, r3w - r0w);
  setPlane(out, PLANE_BOTTOM, r3x + r1x, r3y + r1y, r3z + r1z, r3w + r1w);
  setPlane(out, PLANE_TOP, r3x - r1x, r3y - r1y, r3z - r1z, r3w - r1w);
  setPlane(out, PLANE_NEAR, r3x - r2x, r3y - r2y, r3z - r2z, r3w - r2w);
  return out;
}

function setPlane(out, index, a, b, c, d) {
  // Normalizing by the normal's length turns the plane equation into a signed
  // distance, which is what the sphere test needs and what makes the box test
  // readable. Unnormalized planes still give the right SIGN, so cheap tests
  // sometimes skip this -- we do not, because it costs one sqrt per frame.
  const length = hypot3(a, b, c);
  const inv = length > 0 ? 1 / length : 0;
  const o = index * 4;
  out[o] = a * inv;
  out[o + 1] = b * inv;
  out[o + 2] = c * inv;
  out[o + 3] = d * inv;
}

/**
 * Is any part of this world-space AABB inside the frustum?
 *
 * Uses the "positive vertex" test: for each plane, only the single box corner
 * furthest along that plane's normal can matter. If even that corner is behind
 * the plane, the whole box is, and nothing else needs checking.
 *
 * False negatives are impossible; false POSITIVES are, for boxes straddling
 * two planes near a corner. That is the accepted trade -- the test is four
 * multiplies and three adds per plane, and a wrongly-kept object costs one
 * wasted draw while
 * a wrongly-culled one is a visible hole.
 */
export function frustumTestAABB(frustum, min, max, boundsOff = 0) {
  for (let p = 0; p < FRUSTUM_PLANE_COUNT; p++) {
    const o = p * 4;
    const a = frustum[o], b = frustum[o + 1], c = frustum[o + 2], d = frustum[o + 3];

    // Pick the corner furthest in the direction of the normal, per axis.
    const x = a >= 0 ? max[boundsOff] : min[boundsOff];
    const y = b >= 0 ? max[boundsOff + 1] : min[boundsOff + 1];
    const z = c >= 0 ? max[boundsOff + 2] : min[boundsOff + 2];

    if (a * x + b * y + c * z + d < 0) return false;
  }
  return true;
}

/** Same test for a sphere: three multiplies per plane, 15 over the five. */
export function frustumTestSphere(frustum, center, radius) {
  for (let p = 0; p < FRUSTUM_PLANE_COUNT; p++) {
    const o = p * 4;
    const distance = frustum[o] * center[0]
      + frustum[o + 1] * center[1]
      + frustum[o + 2] * center[2]
      + frustum[o + 3];
    if (distance < -radius) return false;
  }
  return true;
}
