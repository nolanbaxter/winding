// mat4 -- column-major, column-vector convention.
//
// Storage is m[col * 4 + row], which is what WGSL's mat4x4<f32> and glTF both
// expect. A matrix uploads to a uniform buffer with zero transposition.
//
//   | m0  m4  m8  m12 |     translation lives in m12..m14
//   | m1  m5  m9  m13 |
//   | m2  m6  m10 m14 |
//   | m3  m7  m11 m15 |
//
// Vectors are columns: v' = M * v, so MVP = P * V * M reads right to left
// (model, then view, then projection).

import { DEBUG, assert, assertFinite } from '../assert.js';
import { quatFromMat4 } from './quat.js';
import { hypot3 } from './vec3.js';

/** Startup-only. Allocates. Returns identity. */
export function mat4Create() {
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

export function mat4Identity(out) {
  out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
  out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0;
  out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0;
  out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
  return out;
}

export function mat4Copy(out, a, outOff = 0, aOff = 0) {
  for (let i = 0; i < 16; i++) out[outOff + i] = a[aOff + i];
  return out;
}

export function mat4GetTranslation(out, m) {
  out[0] = m[12]; out[1] = m[13]; out[2] = m[14];
  return out;
}

/**
 * out = a * b. Applied to a vector, b happens first.
 *
 * The trailing offsets address a matrix inside a larger array, which is how
 * the transform hierarchy walks its SoA columns without copying every matrix
 * into a scratch buffer first. They default to 0, so ordinary calls are
 * unchanged and there is still exactly one implementation of the math.
 *
 * Aliasing is safe in both directions: all 16 of `a` are read into locals up
 * front, and `b` is read a column at a time before that column of `out` is
 * written. mat4Multiply(m, m, n) and mat4Multiply(n, m, n) are both correct.
 */
/**
 * mat4Multiply for two AFFINE matrices -- bottom row (0, 0, 0, 1), which is
 * every matrix built from translation, rotation and scale. The product's
 * bottom row is then (0, 0, 0, 1) too, so its twelve terms are not computed,
 * nor the four multiplies by it: 36 multiplies instead of 64. Identical to
 * mat4Multiply to the bit on such input, measured over 160,000 floats; 15%
 * faster composing 10,000 transforms and 26% when only children recompose.
 * Garbage on a projection, which is why it is a separate function.
 */
export function mat4MultiplyAffine(out, a, b, outOff = 0, aOff = 0, bOff = 0) {
  const a00 = a[aOff], a01 = a[aOff + 1], a02 = a[aOff + 2];
  const a10 = a[aOff + 4], a11 = a[aOff + 5], a12 = a[aOff + 6];
  const a20 = a[aOff + 8], a21 = a[aOff + 9], a22 = a[aOff + 10];
  const a30 = a[aOff + 12], a31 = a[aOff + 13], a32 = a[aOff + 14];

  let b0 = b[bOff], b1 = b[bOff + 1], b2 = b[bOff + 2];
  out[outOff] = b0 * a00 + b1 * a10 + b2 * a20;
  out[outOff + 1] = b0 * a01 + b1 * a11 + b2 * a21;
  out[outOff + 2] = b0 * a02 + b1 * a12 + b2 * a22;
  out[outOff + 3] = 0;

  b0 = b[bOff + 4]; b1 = b[bOff + 5]; b2 = b[bOff + 6];
  out[outOff + 4] = b0 * a00 + b1 * a10 + b2 * a20;
  out[outOff + 5] = b0 * a01 + b1 * a11 + b2 * a21;
  out[outOff + 6] = b0 * a02 + b1 * a12 + b2 * a22;
  out[outOff + 7] = 0;

  b0 = b[bOff + 8]; b1 = b[bOff + 9]; b2 = b[bOff + 10];
  out[outOff + 8] = b0 * a00 + b1 * a10 + b2 * a20;
  out[outOff + 9] = b0 * a01 + b1 * a11 + b2 * a21;
  out[outOff + 10] = b0 * a02 + b1 * a12 + b2 * a22;
  out[outOff + 11] = 0;

  b0 = b[bOff + 12]; b1 = b[bOff + 13]; b2 = b[bOff + 14];
  out[outOff + 12] = b0 * a00 + b1 * a10 + b2 * a20 + a30;
  out[outOff + 13] = b0 * a01 + b1 * a11 + b2 * a21 + a31;
  out[outOff + 14] = b0 * a02 + b1 * a12 + b2 * a22 + a32;
  out[outOff + 15] = 1;
  return out;
}

export function mat4Multiply(out, a, b, outOff = 0, aOff = 0, bOff = 0) {
  const a00 = a[aOff], a01 = a[aOff + 1], a02 = a[aOff + 2], a03 = a[aOff + 3];
  const a10 = a[aOff + 4], a11 = a[aOff + 5], a12 = a[aOff + 6], a13 = a[aOff + 7];
  const a20 = a[aOff + 8], a21 = a[aOff + 9], a22 = a[aOff + 10], a23 = a[aOff + 11];
  const a30 = a[aOff + 12], a31 = a[aOff + 13], a32 = a[aOff + 14], a33 = a[aOff + 15];

  let b0 = b[bOff], b1 = b[bOff + 1], b2 = b[bOff + 2], b3 = b[bOff + 3];
  out[outOff] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[outOff + 1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[outOff + 2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[outOff + 3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[bOff + 4]; b1 = b[bOff + 5]; b2 = b[bOff + 6]; b3 = b[bOff + 7];
  out[outOff + 4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[outOff + 5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[outOff + 6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[outOff + 7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[bOff + 8]; b1 = b[bOff + 9]; b2 = b[bOff + 10]; b3 = b[bOff + 11];
  out[outOff + 8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[outOff + 9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[outOff + 10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[outOff + 11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[bOff + 12]; b1 = b[bOff + 13]; b2 = b[bOff + 14]; b3 = b[bOff + 15];
  out[outOff + 12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[outOff + 13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[outOff + 14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[outOff + 15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
  return out;
}

/**
 * Compose a TRS matrix from rotation (quat), translation (vec3), scale (vec3).
 *
 * This is THE hot function of the transform system -- it runs once per dirty
 * entity per frame, so it builds the rotation matrix inline rather than going
 * through quat -> mat4 -> scale -> translate as three separate multiplies.
 *
 * Scale is applied first, then rotation, then translation: M = T * R * S.
 */
export function mat4FromQuatPosScale(
  out, q, pos, scale, outOff = 0, qOff = 0, posOff = 0, scaleOff = 0,
) {
  const x = q[qOff], y = q[qOff + 1], z = q[qOff + 2], w = q[qOff + 3];
  const x2 = x + x, y2 = y + y, z2 = z + z;

  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;

  const sx = scale[scaleOff], sy = scale[scaleOff + 1], sz = scale[scaleOff + 2];

  out[outOff] = (1 - (yy + zz)) * sx;
  out[outOff + 1] = (xy + wz) * sx;
  out[outOff + 2] = (xz - wy) * sx;
  out[outOff + 3] = 0;

  out[outOff + 4] = (xy - wz) * sy;
  out[outOff + 5] = (1 - (xx + zz)) * sy;
  out[outOff + 6] = (yz + wx) * sy;
  out[outOff + 7] = 0;

  out[outOff + 8] = (xz + wy) * sz;
  out[outOff + 9] = (yz - wx) * sz;
  out[outOff + 10] = (1 - (xx + yy)) * sz;
  out[outOff + 11] = 0;

  out[outOff + 12] = pos[posOff];
  out[outOff + 13] = pos[posOff + 1];
  out[outOff + 14] = pos[posOff + 2];
  out[outOff + 15] = 1;
  return out;
}

/**
 * General 4x4 inverse via cofactor expansion. Returns null and leaves `out`
 * untouched when the determinant is exactly zero -- callers must check rather
 * than propagate NaN.
 *
 * Exactly zero is the whole contract. A NEARLY singular matrix passes the test
 * and divides by something tiny, so `out` comes back non-null and full of very
 * large values or Infinity. The assertFinite below catches that in a debug
 * build and not in a release one, so a caller handed a matrix from outside --
 * a glTF node, a user transform -- cannot rely on the null alone.
 *
 * ponytail: general inverse; a rigid-body fast path (transpose the 3x3, negate
 * the translation) is ~4x cheaper and valid for any unscaled transform. Add it
 * if transform inversion ever shows up in a profile.
 */
export function mat4Invert(out, a) {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  // Deliberately not an assert: a singular matrix is a legitimate state, not a
  // bug. An entity with zero scale is a normal way to hide something. The
  // caller checking for null is the non-silent failure
  // by refusing to return NaN, not by refusing to return at all.
  if (det === 0) return null;
  det = 1 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;

  // A near-singular matrix passes the det check and still divides by something
  // tiny enough to overflow. Catching it here beats tracing a NaN camera.
  if (DEBUG) assertFinite(out, 'mat4Invert');
  return out;
}

/**
 * View matrix (world -> view). Right-handed, camera looks down its own -Z.
 *
 * Built directly rather than by composing and inverting a camera transform:
 * the inverse of a rotation is its transpose, so we just write the basis
 * vectors into rows instead of columns and negate the translation.
 */
export function mat4LookAt(out, eye, center, up) {
  // z = backward = normalize(eye - center)
  let zx = eye[0] - center[0];
  let zy = eye[1] - center[1];
  let zz = eye[2] - center[2];
  const zLen = hypot3(zx, zy, zz);
  if (DEBUG) assert(zLen > 1e-6, 'mat4LookAt: eye and center coincide');
  zx /= zLen; zy /= zLen; zz /= zLen;

  // x = right = normalize(up x z)
  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  const xLen = hypot3(xx, xy, xz);
  if (DEBUG) assert(xLen > 1e-6, 'mat4LookAt: up is parallel to the view direction');
  xx /= xLen; xy /= xLen; xz /= xLen;

  // y = true up = z x x  (already unit: both are unit and perpendicular)
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

/**
 * Normal matrix: the inverse transpose of the upper 3x3, written as a WGSL
 * mat3x3 (three columns, each padded to 4 floats -- WGSL aligns matrix columns
 * to 16 bytes, so a "3x3" occupies 48).
 *
 * Normals do not transform like positions. Under non-uniform scale, applying
 * the model matrix to a normal tilts it off the surface: squash a sphere flat
 * and its normals should splay outward, but the model matrix squashes them too.
 * The inverse transpose is the transform that gets it right.
 *
 * Because transpose(inverse(M)) == cofactor(M) / det(M), this computes the
 * cofactor matrix directly -- no separate inverse and transpose step.
 * Returns false for a singular matrix, leaving `out` untouched.
 */
export function mat4NormalMatrix(out, m, outOff = 0, mOff = 0) {
  const m00 = m[mOff], m01 = m[mOff + 1], m02 = m[mOff + 2];
  const m10 = m[mOff + 4], m11 = m[mOff + 5], m12 = m[mOff + 6];
  const m20 = m[mOff + 8], m21 = m[mOff + 9], m22 = m[mOff + 10];

  const c00 = m11 * m22 - m21 * m12;
  const c01 = -(m01 * m22 - m21 * m02);
  const c02 = m01 * m12 - m11 * m02;
  const c10 = -(m10 * m22 - m20 * m12);
  const c11 = m00 * m22 - m20 * m02;
  const c12 = -(m00 * m12 - m10 * m02);
  const c20 = m10 * m21 - m20 * m11;
  const c21 = -(m00 * m21 - m20 * m01);
  const c22 = m00 * m11 - m10 * m01;

  const det = m00 * c00 + m10 * c01 + m20 * c02;
  if (det === 0) return false;
  const inv = 1 / det;

  out[outOff] = c00 * inv; out[outOff + 1] = c10 * inv; out[outOff + 2] = c20 * inv;
  out[outOff + 3] = 0;
  out[outOff + 4] = c01 * inv; out[outOff + 5] = c11 * inv; out[outOff + 6] = c21 * inv;
  out[outOff + 7] = 0;
  out[outOff + 8] = c02 * inv; out[outOff + 9] = c12 * inv; out[outOff + 10] = c22 * inv;
  out[outOff + 11] = 0;
  return true;
}

// Scratch for mat4Decompose. Module-level rather than per-call: decompose runs
// at asset load, but allocating a matrix per node would still be pointless.
const decomposeScratch = new Float32Array(16);

/**
 * Split a transform matrix back into translation, rotation and scale.
 *
 * glTF nodes carry either TRS or a raw 4x4, and TransformStore only stores TRS,
 * so imported matrix-form nodes come through here.
 *
 * The recoverable cases are exactly rotation, translation and axis-aligned
 * scale. Shear cannot be represented as TRS and is silently lost -- glTF
 * forbids it in node matrices ("must be decomposable"), so an asset that hits
 * this has already left the spec.
 *
 * Returns false if the matrix is degenerate (a zero scale axis), in which case
 * the outputs are left untouched.
 */
export function mat4Decompose(outPos, outRot, outScale, m, mOff = 0) {
  // Scale is the length of each basis column.
  let sx = hypot3(m[mOff], m[mOff + 1], m[mOff + 2]);
  const sy = hypot3(m[mOff + 4], m[mOff + 5], m[mOff + 6]);
  const sz = hypot3(m[mOff + 8], m[mOff + 9], m[mOff + 10]);

  // A negative determinant means the matrix mirrors. Mirroring is not a
  // rotation, so it has to live in the scale; by convention it goes on X.
  // Without this the rotation extraction below silently returns garbage.
  if (determinant3(m, mOff) < 0) sx = -sx;

  // Before any output is written. This used to run after the translation had
  // already been copied out, so a caller trusting the documented contract kept
  // its previous rotation and scale and got the degenerate matrix's position --
  // a transform mixed from two different matrices, which is worse than either.
  if (sx === 0 || sy === 0 || sz === 0) return false;

  outPos[0] = m[mOff + 12];
  outPos[1] = m[mOff + 13];
  outPos[2] = m[mOff + 14];

  // Strip scale, leaving a pure rotation for the quaternion extraction.
  const s = decomposeScratch;
  const ix = 1 / sx, iy = 1 / sy, iz = 1 / sz;
  s[0] = m[mOff] * ix; s[1] = m[mOff + 1] * ix; s[2] = m[mOff + 2] * ix; s[3] = 0;
  s[4] = m[mOff + 4] * iy; s[5] = m[mOff + 5] * iy; s[6] = m[mOff + 6] * iy; s[7] = 0;
  s[8] = m[mOff + 8] * iz; s[9] = m[mOff + 9] * iz; s[10] = m[mOff + 10] * iz; s[11] = 0;
  s[12] = 0; s[13] = 0; s[14] = 0; s[15] = 1;

  quatFromMat4(outRot, s);

  outScale[0] = sx;
  outScale[1] = sy;
  outScale[2] = sz;

  if (DEBUG) {
    assertFinite(outPos, 'mat4Decompose position');
    assertFinite(outRot, 'mat4Decompose rotation');
  }
  return true;
}

function determinant3(m, o) {
  const a = m[o], b = m[o + 1], c = m[o + 2];
  const d = m[o + 4], e = m[o + 5], f = m[o + 6];
  const g = m[o + 8], h = m[o + 9], i = m[o + 10];
  return a * (e * i - f * h) - d * (b * i - c * h) + g * (b * f - c * e);
}

/**
 * Orthographic projection, reverse-Z, WebGPU clip space: near -> 1, far -> 0.
 *
 * Reversed for CONSISTENCY, not for precision. The precision argument for
 * reverse-Z does not apply here -- orthographic depth is linear in distance, so float
 * precision is uniform whichever way it runs. But one depth convention across
 * the whole engine means one `depthCompare`, one clear value, and no pass where
 * a reader has to stop and work out which way this particular buffer runs.
 * Two conventions would be exactly the arbitrariness this engine exists to stop.
 *
 * `near` and `far` are positive distances in front of the camera, as usual.
 */
export function mat4OrthographicReverseZ(out, left, right, bottom, top, near, far) {
  if (DEBUG) {
    assert(right !== left && top !== bottom, 'orthographic box is degenerate');
    assert(far > near, 'far must be beyond near');
  }
  const invWidth = 1 / (right - left);
  const invHeight = 1 / (top - bottom);
  const invDepth = 1 / (far - near);

  out[0] = 2 * invWidth; out[1] = 0; out[2] = 0; out[3] = 0;
  out[4] = 0; out[5] = 2 * invHeight; out[6] = 0; out[7] = 0;
  // z maps view-space -near -> 1 and -far -> 0.
  out[8] = 0; out[9] = 0; out[10] = invDepth; out[11] = 0;
  out[12] = -(right + left) * invWidth;
  out[13] = -(top + bottom) * invHeight;
  out[14] = far * invDepth;
  out[15] = 1;
  return out;
}

/**
 * Perspective projection: reverse-Z, infinite far plane, WebGPU clip space.
 *
 *
 * Maps view-space depth to NDC z as `near / distance`:
 *   distance = near  ->  z = 1
 *   distance -> inf  ->  z = 0
 *
 * Standard-Z wastes float precision at the far plane, where it is not needed,
 * and starves the near field, where it is. Reversing the mapping puts the
 * float's dense region near zero exactly where the perspective divide is
 * already compressing hardest, and the two nonlinearities nearly cancel --
 * precision becomes near-uniform across the whole range.
 *
 * The pipeline must match or everything vanishes:
 *   depthCompare:    'greater'
 *   depthClearValue: 0.0
 *   format:          'depth32float'   (the trick is about float distribution;
 *                                      a unorm format gains nothing)
 *
 * There is no far parameter. Nothing is ever clipped for being too distant,
 * which removes an arbitrary number from the API entirely.
 *
 * NOTE: WebGPU clip space is z in [0, 1] (like D3D/Metal). Any projection
 * matrix copied from an OpenGL tutorial assumes [-1, 1] and will be subtly,
 * expensively wrong.
 */
export function mat4PerspectiveReverseZInfinite(out, fovYRadians, aspect, near) {
  if (DEBUG) {
    assert(fovYRadians > 0 && fovYRadians < Math.PI, 'fovY out of range');
    assert(aspect > 0, 'aspect must be positive');
    assert(near > 0, 'near must be positive');
  }
  const f = 1 / Math.tan(fovYRadians * 0.5);

  out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
  out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
  out[8] = 0; out[9] = 0; out[10] = 0; out[11] = -1;
  out[12] = 0; out[13] = 0; out[14] = near; out[15] = 0;
  return out;
}
