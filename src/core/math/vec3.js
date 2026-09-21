// vec3 -- out-parameter style, zero allocation.
//
// Every function that produces a vector takes `out` first and returns it.
// `create()` is the ONLY allocating function here and is startup-only.
// Nothing in this file allocates otherwise --.
//
//   vec3Add(out, a, b);         // yes
//   const v = a.add(b);         // never, in the loop
//
// Aliasing is safe: every function reads all its inputs into locals before
// writing `out`, so `vec3Cross(a, a, b)` is correct.

import { DEBUG, assert } from '../assert.js';

/** Startup-only. Allocates. */
export function vec3Create(x = 0, y = 0, z = 0) {
  const v = new Float32Array(3);
  v[0] = x; v[1] = y; v[2] = z;
  return v;
}

export function vec3Set(out, x, y, z) {
  out[0] = x; out[1] = y; out[2] = z;
  return out;
}

export function vec3Copy(out, a) {
  out[0] = a[0]; out[1] = a[1]; out[2] = a[2];
  return out;
}

export function vec3Add(out, a, b) {
  out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
  return out;
}

export function vec3Sub(out, a, b) {
  out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
  return out;
}

export function vec3Mul(out, a, b) {
  out[0] = a[0] * b[0]; out[1] = a[1] * b[1]; out[2] = a[2] * b[2];
  return out;
}

export function vec3Scale(out, a, s) {
  out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s;
  return out;
}

/** out = a + b * s. One call instead of a temp + add; the workhorse of integration. */
export function vec3ScaleAndAdd(out, a, b, s) {
  out[0] = a[0] + b[0] * s;
  out[1] = a[1] + b[1] * s;
  out[2] = a[2] + b[2] * s;
  return out;
}

export function vec3Negate(out, a) {
  out[0] = -a[0]; out[1] = -a[1]; out[2] = -a[2];
  return out;
}

export function vec3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vec3Cross(out, a, b) {
  const ax = a[0], ay = a[1], az = a[2];
  const bx = b[0], by = b[1], bz = b[2];
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}

/** Prefer this over vec3Length for comparisons -- skips the sqrt. */
export function vec3LengthSq(a) {
  return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

export function vec3Length(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

export function vec3DistanceSq(a, b) {
  const x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
  return x * x + y * y + z * z;
}

/**
 * Normalize. A zero-length input yields zero rather than NaN -- NaN propagates
 * silently through an entire frame and surfaces as unrelated garbage, so we
 * refuse to produce it. DEBUG builds still complain, because a zero vector
 * reaching normalize is nearly always a bug upstream.
 */
export function vec3Normalize(out, a) {
  const x = a[0], y = a[1], z = a[2];
  const lenSq = x * x + y * y + z * z;
  if (lenSq === 0) {
    if (DEBUG) assert(false, 'vec3Normalize: zero-length input');
    out[0] = 0; out[1] = 0; out[2] = 0;
    return out;
  }
  const inv = 1 / Math.sqrt(lenSq);
  out[0] = x * inv; out[1] = y * inv; out[2] = z * inv;
  return out;
}

export function vec3Lerp(out, a, b, t) {
  out[0] = a[0] + (b[0] - a[0]) * t;
  out[1] = a[1] + (b[1] - a[1]) * t;
  out[2] = a[2] + (b[2] - a[2]) * t;
  return out;
}

export function vec3Min(out, a, b) {
  out[0] = Math.min(a[0], b[0]);
  out[1] = Math.min(a[1], b[1]);
  out[2] = Math.min(a[2], b[2]);
  return out;
}

export function vec3Max(out, a, b) {
  out[0] = Math.max(a[0], b[0]);
  out[1] = Math.max(a[1], b[1]);
  out[2] = Math.max(a[2], b[2]);
  return out;
}

/**
 * Transform as a POINT (implicit w = 1), with perspective divide.
 *
 * For direction vectors use vec3TransformMat4Dir -- translation must not apply.
 * For normals neither is correct under non-uniform scale; that needs the
 * inverse-transpose; mat4NormalMatrix builds it.
 */
export function vec3TransformMat4(out, a, m) {
  const x = a[0], y = a[1], z = a[2];
  // Column-major: m[col * 4 + row].
  let w = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (w === 0) w = 1;
  out[0] = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
  out[1] = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
  out[2] = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;
  return out;
}

/** Transform as a DIRECTION (implicit w = 0) -- upper 3x3 only, no translation. */
export function vec3TransformMat4Dir(out, a, m) {
  const x = a[0], y = a[1], z = a[2];
  out[0] = m[0] * x + m[4] * y + m[8] * z;
  out[1] = m[1] * x + m[5] * y + m[9] * z;
  out[2] = m[2] * x + m[6] * y + m[10] * z;
  return out;
}

/**
 * Rotate by a quaternion. Uses the cross-product form
 *   v' = v + 2w(q x v) + 2(q x (q x v))
 * which is ~15 ops against ~30 for building a matrix first. Worth it because
 * this runs per-vertex-ish in hot paths.
 */
export function vec3TransformQuat(out, a, q) {
  const x = a[0], y = a[1], z = a[2];
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];

  // t = 2 * (q.xyz x v)
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);

  // v' = v + w*t + (q.xyz x t)
  out[0] = x + qw * tx + qy * tz - qz * ty;
  out[1] = y + qw * ty + qz * tx - qx * tz;
  out[2] = z + qw * tz + qx * ty - qy * tx;
  return out;
}
