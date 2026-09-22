// Quaternions -- the only rotation representation the engine stores.
//
// Euler angles are a UI format. Storing them means owning gimbal lock and
// broken interpolation forever, so they are converted at the EDGE and never
// enter the core. quatFromEuler below is that edge and the only one: it exists
// so a caller can hand the engine the angles a control panel produced, and
// nothing downstream of it ever sees an Euler triple again.
//
// Layout is [x, y, z, w] -- vector part first, scalar last. This matches glTF,
// WGSL's vec4, and glMatrix. Some textbooks use [w, x, y, z]; mixing the two
// produces rotations that look almost right, which is the worst kind of bug.

import { DEBUG, assert } from '../assert.js';

/** Startup-only. Allocates. Returns the identity rotation. */
export function quatCreate() {
  const q = new Float32Array(4);
  q[3] = 1;
  return q;
}

export function quatIdentity(out) {
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
  return out;
}

export function quatCopy(out, a) {
  out[0] = a[0]; out[1] = a[1]; out[2] = a[2]; out[3] = a[3];
  return out;
}

/** `axis` must be unit length. `rad` is radians, right-hand rule. */
export function quatSetAxisAngle(out, axis, rad) {
  if (DEBUG) {
    const l = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    assert(Math.abs(l - 1) < 1e-3, 'quatSetAxisAngle: axis must be normalized');
  }
  const half = rad * 0.5;
  const s = Math.sin(half);
  out[0] = axis[0] * s;
  out[1] = axis[1] * s;
  out[2] = axis[2] * s;
  out[3] = Math.cos(half);
  return out;
}

/**
 * out = a * b. Same reading order as matrices: b is applied first, then a.
 * Quaternion multiplication does not commute -- swapping these is a real bug,
 * not a style choice.
 */
export function quatMultiply(out, a, b) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  out[0] = ax * bw + aw * bx + ay * bz - az * by;
  out[1] = ay * bw + aw * by + az * bx - ax * bz;
  out[2] = az * bw + aw * bz + ax * by - ay * bx;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

export function quatDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/**
 * For UNIT quaternions the conjugate is the inverse, and that is the only case
 * the engine produces. Non-unit input silently gives a wrong answer, so DEBUG
 * checks rather than paying for a general inverse everywhere.
 */
export function quatConjugate(out, a) {
  if (DEBUG) {
    const l = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
    assert(Math.abs(l - 1) < 1e-3, 'quatConjugate: input must be unit length');
  }
  out[0] = -a[0]; out[1] = -a[1]; out[2] = -a[2]; out[3] = a[3];
  return out;
}

/**
 * Renormalize. Repeated multiplication accumulates float drift; an animation
 * chain left unnormalized visibly skews geometry after a few hundred frames.
 */
export function quatNormalize(out, a) {
  const x = a[0], y = a[1], z = a[2], w = a[3];
  const lenSq = x * x + y * y + z * z + w * w;
  if (lenSq === 0) {
    if (DEBUG) assert(false, 'quatNormalize: zero quaternion');
    return quatIdentity(out);
  }
  const inv = 1 / Math.sqrt(lenSq);
  out[0] = x * inv; out[1] = y * inv; out[2] = z * inv; out[3] = w * inv;
  return out;
}

/**
 * Euler angles to a quaternion, in YXZ order: roll about Z, then pitch about X,
 * then yaw about Y. q = qYaw * qPitch * qRoll.
 *
 * This is THE edge the file header talks about. Euler angles are how people
 * think and how a UI presents rotation; they are not how the engine stores it.
 * Converting here, once, is what keeps gimbal lock and broken interpolation out
 * of everything downstream -- there is deliberately no inverse function, because
 * a round trip back to Euler is exactly how they leak into storage.
 *
 * YXZ is the camera convention: yaw turns you, pitch looks up and down, and
 * roll tilts the horizon. Any order is defensible; this one keeps yaw and pitch
 * independent, which is what a look-around control needs.
 */
export function quatFromEuler(out, yaw, pitch, roll) {
  const cy = Math.cos(yaw * 0.5), sy = Math.sin(yaw * 0.5);
  const cx = Math.cos(pitch * 0.5), sx = Math.sin(pitch * 0.5);
  const cz = Math.cos(roll * 0.5), sz = Math.sin(roll * 0.5);

  out[0] = cy * sx * cz + sy * cx * sz;
  out[1] = sy * cx * cz - cy * sx * sz;
  out[2] = cy * cx * sz - sy * sx * cz;
  out[3] = cy * cx * cz + sy * sx * sz;
  return out;
}

/**
 * Extract the rotation from the upper 3x3 of a mat4. The matrix must already
 * be orthonormal -- strip scale first (see mat4Decompose).
 *
 * Branches on the largest diagonal term rather than always using the trace.
 * The trace formula divides by sqrt(trace + 1), which approaches zero for
 * rotations near 180 degrees and loses most of its precision there; picking
 * the dominant axis keeps the divisor comfortably large in every case.
 */
export function quatFromMat4(out, m, mOff = 0) {
  const e = (col, row) => m[mOff + col * 4 + row];
  const trace = e(0, 0) + e(1, 1) + e(2, 2);

  if (trace > 0) {
    let root = Math.sqrt(trace + 1);
    out[3] = 0.5 * root;
    root = 0.5 / root;
    out[0] = (e(1, 2) - e(2, 1)) * root;
    out[1] = (e(2, 0) - e(0, 2)) * root;
    out[2] = (e(0, 1) - e(1, 0)) * root;
    return out;
  }

  let i = 0;
  if (e(1, 1) > e(0, 0)) i = 1;
  if (e(2, 2) > e(i, i)) i = 2;
  const j = (i + 1) % 3;
  const k = (i + 2) % 3;

  let root = Math.sqrt(e(i, i) - e(j, j) - e(k, k) + 1);
  out[i] = 0.5 * root;
  root = 0.5 / root;
  out[3] = (e(j, k) - e(k, j)) * root;
  out[j] = (e(j, i) + e(i, j)) * root;
  out[k] = (e(k, i) + e(i, k)) * root;
  return out;
}

// Below this angle, sin(omega) is small enough that the slerp division loses
// precision. Straight lerp is indistinguishable here and cannot divide by ~0.
const SLERP_LINEAR_EPSILON = 1e-6;

/**
 * Spherical linear interpolation -- constant angular velocity, which plain lerp
 * does not give you (lerp speeds up through the middle of a long rotation).
 *
 * Two details that are usually what's wrong when slerp "looks weird":
 *   1. q and -q are the SAME rotation, so we flip b when the dot product is
 *      negative. Without this, a 10-degree turn can interpolate the 350-degree
 *      way around.
 *   2. Near-parallel inputs fall back to lerp to avoid dividing by sin(~0).
 */
export function quatSlerp(out, a, b, t) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];

  let cosom = ax * bx + ay * by + az * bz + aw * bw;

  if (cosom < 0) {           // take the short way around
    cosom = -cosom;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }

  let scale0, scale1;
  if (1 - cosom > SLERP_LINEAR_EPSILON) {
    const omega = Math.acos(cosom);
    const sinom = Math.sin(omega);
    scale0 = Math.sin((1 - t) * omega) / sinom;
    scale1 = Math.sin(t * omega) / sinom;
  } else {
    scale0 = 1 - t;
    scale1 = t;
  }

  out[0] = scale0 * ax + scale1 * bx;
  out[1] = scale0 * ay + scale1 * by;
  out[2] = scale0 * az + scale1 * bz;
  out[3] = scale0 * aw + scale1 * bw;
  return out;
}
