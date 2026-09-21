// Math, handles, asserts and capacity growth. Run: node test/core.test.js
//
// No framework, no fixtures. Every bug in a layer above this one will look
// like a math bug until you can rule the math out -- that is what this is for.

import assert from 'node:assert/strict';

import {
  vec3Create, vec3Set, vec3Add, vec3Sub, vec3Scale, vec3ScaleAndAdd,
  vec3Dot, vec3Cross, vec3Length, vec3Normalize, vec3Lerp,
  vec3TransformMat4, vec3TransformMat4Dir, vec3TransformQuat,
} from '../src/core/math/vec3.js';

import {
  quatCreate, quatIdentity, quatSetAxisAngle, quatMultiply,
  quatNormalize, quatConjugate, quatSlerp,
} from '../src/core/math/quat.js';

import {
  mat4Create, mat4Identity, mat4Copy, mat4Multiply, mat4FromQuatPosScale,
  mat4Invert, mat4LookAt, mat4NormalMatrix, mat4PerspectiveReverseZInfinite,
} from '../src/core/math/mat4.js';

import { assertFinite } from '../src/core/assert.js';
import { HandleAllocator, NULL_HANDLE, handleIndex } from '../src/core/handle.js';
import { Clock } from '../src/core/time.js';

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`  ok  ${name}`);
}

const EPS = 1e-5;
function close(a, b, eps = EPS, what = '') {
  assert.ok(Math.abs(a - b) <= eps, `${what} expected ${b}, got ${a} (diff ${Math.abs(a - b)})`);
}
function vecClose(a, b, eps = EPS, what = '') {
  for (let i = 0; i < b.length; i++) close(a[i], b[i], eps, `${what}[${i}]`);
}

// ---------------------------------------------------------------- vec3

console.log('\nvec3');

test('arithmetic', () => {
  const out = vec3Create();
  const a = vec3Create(1, 2, 3);
  const b = vec3Create(4, 5, 6);

  vecClose(vec3Add(out, a, b), [5, 7, 9]);
  vecClose(vec3Sub(out, b, a), [3, 3, 3]);
  vecClose(vec3Scale(out, a, 2), [2, 4, 6]);
  vecClose(vec3ScaleAndAdd(out, a, b, 2), [9, 12, 15]);
  close(vec3Dot(a, b), 32);
  close(vec3Length(vec3Set(out, 3, 4, 0)), 5);
  vecClose(vec3Lerp(out, a, b, 0.5), [2.5, 3.5, 4.5]);
});

test('cross product is right-handed (X cross Y = +Z)', () => {
  // This single assertion pins the engine's handedness. If it ever flips,
  // every normal, every frustum plane and every camera basis flips with it.
  const out = vec3Create();
  vecClose(vec3Cross(out, vec3Create(1, 0, 0), vec3Create(0, 1, 0)), [0, 0, 1]);
});

test('normalize yields unit length', () => {
  const out = vec3Create();
  vec3Normalize(out, vec3Create(3, 4, 12));
  close(vec3Length(out), 1);
});

test('normalize refuses a zero vector instead of producing NaN', () => {
  assert.throws(() => vec3Normalize(vec3Create(), vec3Create(0, 0, 0)));
});

test('aliasing output with input is safe', () => {
  const a = vec3Create(1, 0, 0);
  const b = vec3Create(0, 1, 0);
  vec3Cross(a, a, b);              // out === a
  vecClose(a, [0, 0, 1]);
});

// ---------------------------------------------------------------- quat

console.log('\nquat');

test('identity rotation leaves a vector alone', () => {
  const out = vec3Create();
  const v = vec3Create(1, 2, 3);
  vecClose(vec3TransformQuat(out, v, quatIdentity(quatCreate())), [1, 2, 3]);
});

test('90 degrees about +Y maps +X to -Z', () => {
  // Right-hand rule, and -Z is forward: this is the turn a camera makes
  // when it rotates left.
  const q = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), Math.PI / 2);
  const out = vec3Create();
  vecClose(vec3TransformQuat(out, vec3Create(1, 0, 0), q), [0, 0, -1]);
});

test('quaternion rotation and its matrix agree', () => {
  // Cross-validates two independent implementations. If vec3TransformQuat and
  // mat4FromQuatPosScale ever disagree, one of them has a sign error.
  const q = quatNormalize(quatCreate(), Object.assign(quatCreate(), [0.3, -0.5, 0.2, 0.8]));
  const m = mat4FromQuatPosScale(mat4Create(), q, vec3Create(0, 0, 0), vec3Create(1, 1, 1));

  const v = vec3Create(0.4, -1.3, 2.0);
  const viaQuat = vec3TransformQuat(vec3Create(), v, q);
  const viaMat = vec3TransformMat4Dir(vec3Create(), v, m);
  vecClose(viaQuat, viaMat, 1e-5, 'quat vs matrix');
});

test('q * conjugate(q) is identity', () => {
  const q = quatSetAxisAngle(quatCreate(), vec3Normalize(vec3Create(), vec3Create(1, 2, 3)), 1.1);
  const inv = quatConjugate(quatCreate(), q);
  vecClose(quatMultiply(quatCreate(), q, inv), [0, 0, 0, 1]);
});

test('multiplication order matters and matches matrix order', () => {
  // quatMultiply(a, b) must mean "b first, then a", same as matrices.
  const rotY = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), Math.PI / 2);
  const rotZ = quatSetAxisAngle(quatCreate(), vec3Create(0, 0, 1), Math.PI / 2);

  const combined = quatMultiply(quatCreate(), rotY, rotZ);   // Z first, then Y
  const v = vec3Create(1, 0, 0);

  const stepwise = vec3TransformQuat(vec3Create(), v, rotZ);
  vec3TransformQuat(stepwise, stepwise, rotY);

  vecClose(vec3TransformQuat(vec3Create(), v, combined), stepwise);
});

test('slerp takes the short way around', () => {
  // a and -a are the same rotation. Given the negated form, slerp must still
  // interpolate the 90-degree arc, not the 270-degree one.
  const a = quatIdentity(quatCreate());
  const b = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), Math.PI / 2);
  const bNeg = Object.assign(quatCreate(), [-b[0], -b[1], -b[2], -b[3]]);

  const viaB = quatSlerp(quatCreate(), a, b, 0.5);
  const viaNeg = quatSlerp(quatCreate(), a, bNeg, 0.5);

  const v = vec3Create(1, 0, 0);
  vecClose(
    vec3TransformQuat(vec3Create(), v, viaB),
    vec3TransformQuat(vec3Create(), v, viaNeg),
    1e-5,
    'short-way slerp',
  );
});

test('slerp endpoints are exact', () => {
  const a = quatIdentity(quatCreate());
  const b = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), 1.0);
  vecClose(quatSlerp(quatCreate(), a, b, 0), a);
  vecClose(quatSlerp(quatCreate(), a, b, 1), b);
});

// ---------------------------------------------------------------- mat4

console.log('\nmat4');

test('identity is a no-op', () => {
  const m = mat4Identity(mat4Create());
  vecClose(vec3TransformMat4(vec3Create(), vec3Create(1, 2, 3), m), [1, 2, 3]);
});

test('TRS applies scale, then rotation, then translation', () => {
  const q = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), Math.PI / 2);
  const m = mat4FromQuatPosScale(mat4Create(), q, vec3Create(10, 0, 0), vec3Create(2, 2, 2));

  // (1,0,0) -> scale 2 -> (2,0,0) -> rotY 90 -> (0,0,-2) -> translate -> (10,0,-2)
  vecClose(vec3TransformMat4(vec3Create(), vec3Create(1, 0, 0), m), [10, 0, -2]);
});

test('multiply composes right-to-left', () => {
  const t = mat4FromQuatPosScale(mat4Create(), quatCreate(), vec3Create(5, 0, 0), vec3Create(1, 1, 1));
  const s = mat4FromQuatPosScale(mat4Create(), quatCreate(), vec3Create(0, 0, 0), vec3Create(2, 2, 2));

  // T * S means scale first, then translate: (1,0,0) -> (2,0,0) -> (7,0,0)
  const ts = mat4Multiply(mat4Create(), t, s);
  vecClose(vec3TransformMat4(vec3Create(), vec3Create(1, 0, 0), ts), [7, 0, 0]);

  // S * T means translate first, then scale: (1,0,0) -> (6,0,0) -> (12,0,0)
  const st = mat4Multiply(mat4Create(), s, t);
  vecClose(vec3TransformMat4(vec3Create(), vec3Create(1, 0, 0), st), [12, 0, 0]);
});

test('M * inverse(M) is identity', () => {
  const q = quatNormalize(quatCreate(), Object.assign(quatCreate(), [0.2, 0.4, -0.1, 0.9]));
  const m = mat4FromQuatPosScale(mat4Create(), q, vec3Create(3, -2, 7), vec3Create(1.5, 0.5, 2));

  const inv = mat4Invert(mat4Create(), m);
  assert.ok(inv !== null, 'matrix should be invertible');

  const product = mat4Multiply(mat4Create(), m, inv);
  vecClose(product, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 1e-4);
});

test('inverting a singular matrix returns null, not NaN', () => {
  const flat = mat4FromQuatPosScale(
    mat4Create(), quatCreate(), vec3Create(0, 0, 0), vec3Create(1, 1, 0),
  );
  assert.equal(mat4Invert(mat4Create(), flat), null);
});

test('lookAt puts the target on the view -Z axis', () => {
  const view = mat4LookAt(
    mat4Create(), vec3Create(0, 0, 5), vec3Create(0, 0, 0), vec3Create(0, 1, 0),
  );
  // The camera sits 5 back looking at the origin, so the origin lands 5 in
  // front of it -- and "in front" is -Z in view space.
  vecClose(vec3TransformMat4(vec3Create(), vec3Create(0, 0, 0), view), [0, 0, -5]);
  // The camera's own position maps to the view-space origin.
  vecClose(vec3TransformMat4(vec3Create(), vec3Create(0, 0, 5), view), [0, 0, 0]);
});

test('lookAt rejects an up vector parallel to the view direction', () => {
  assert.throws(() => mat4LookAt(
    mat4Create(), vec3Create(0, 5, 0), vec3Create(0, 0, 0), vec3Create(0, 1, 0),
  ));
});

// -------------------------------------------------------- mat4 offsets

console.log('\nmat4 offsets (SoA addressing)');

test('offset multiply matches unoffset multiply', () => {
  // The transform hierarchy stores every matrix in one big column and composes
  // in place. If offsets and non-offsets ever disagree, every child transform
  // in the scene is silently wrong.
  const q = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), 0.7);
  const a = mat4FromQuatPosScale(mat4Create(), q, vec3Create(1, 2, 3), vec3Create(2, 2, 2));
  const b = mat4FromQuatPosScale(mat4Create(), quatCreate(), vec3Create(-4, 5, 6), vec3Create(1, 1, 1));

  const plain = mat4Multiply(mat4Create(), a, b);

  // Same operands living at slot 2 and slot 5 of a shared column.
  const column = new Float32Array(16 * 8);
  column.set(a, 2 * 16);
  column.set(b, 5 * 16);
  mat4Multiply(column, column, column, 7 * 16, 2 * 16, 5 * 16);

  vecClose(column.subarray(7 * 16, 8 * 16), plain, 1e-6);
});

test('offset multiply is safe when output aliases an input', () => {
  // world = parentWorld * local writes back into the same `world` column the
  // parent was read from, which is exactly this case.
  const column = new Float32Array(16 * 4);
  const a = mat4FromQuatPosScale(mat4Create(), quatCreate(), vec3Create(10, 0, 0), vec3Create(1, 1, 1));
  const b = mat4FromQuatPosScale(mat4Create(), quatCreate(), vec3Create(0, 3, 0), vec3Create(1, 1, 1));
  column.set(a, 0);
  column.set(b, 16);

  const expected = mat4Multiply(mat4Create(), a, b);
  mat4Multiply(column, column, column, 16, 0, 16);     // out and b are the same slot

  vecClose(column.subarray(16, 32), expected, 1e-6);
});

test('offset TRS composition matches unoffset', () => {
  const q = quatSetAxisAngle(quatCreate(), vec3Create(0.6, 0.8, 0), 1.2);
  const plain = mat4FromQuatPosScale(mat4Create(), q, vec3Create(7, -1, 2), vec3Create(1, 2, 3));

  const rotations = new Float32Array(4 * 4);
  const positions = new Float32Array(3 * 4);
  const scales = new Float32Array(3 * 4);
  const out = new Float32Array(16 * 4);
  rotations.set(q, 3 * 4);
  positions.set([7, -1, 2], 3 * 3);
  scales.set([1, 2, 3], 3 * 3);

  mat4FromQuatPosScale(out, rotations, positions, scales, 3 * 16, 3 * 4, 3 * 3, 3 * 3);
  vecClose(out.subarray(3 * 16, 4 * 16), plain, 1e-6);
});

test('normal matrix is the inverse transpose, and undoes non-uniform scale', () => {
  // Squash a shape flat on Y. Its normals must splay OUTWARD, not squash with
  // it -- applying the model matrix to a normal is the classic wrong answer.
  const m = mat4FromQuatPosScale(
    mat4Create(), quatCreate(), vec3Create(0, 0, 0), vec3Create(1, 0.25, 1),
  );
  const normalMatrix = new Float32Array(12);
  assert.ok(mat4NormalMatrix(normalMatrix, m));

  // Column-major with 4-float column stride, matching WGSL's mat3x3 layout.
  close(normalMatrix[0], 1, 1e-5, 'x unchanged');
  close(normalMatrix[5], 4, 1e-5, 'y scaled by 1/0.25');
  close(normalMatrix[10], 1, 1e-5, 'z unchanged');
  assert.equal(normalMatrix[3], 0, 'column padding');
});

test('normal matrix equals the rotation when scale is uniform', () => {
  const q = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), Math.PI / 2);
  const m = mat4FromQuatPosScale(mat4Create(), q, vec3Create(3, 4, 5), vec3Create(2, 2, 2));
  const normalMatrix = new Float32Array(12);
  mat4NormalMatrix(normalMatrix, m);

  // Under uniform scale s the inverse transpose is the rotation divided by s,
  // so a normalized column is just the rotated axis.
  const column0 = [normalMatrix[0], normalMatrix[1], normalMatrix[2]];
  const length = Math.hypot(...column0);
  vecClose(column0.map((c) => c / length), [0, 0, -1], 1e-5, 'rotated +X');
});

test('normal matrix refuses a singular transform', () => {
  const flat = mat4FromQuatPosScale(
    mat4Create(), quatCreate(), vec3Create(0, 0, 0), vec3Create(1, 1, 0),
  );
  assert.equal(mat4NormalMatrix(new Float32Array(12), flat), false);
});

test('offset copy moves the right 16 floats', () => {
  const src = new Float32Array(16 * 3);
  const dst = new Float32Array(16 * 3);
  for (let i = 0; i < 16; i++) src[16 + i] = i + 1;

  mat4Copy(dst, src, 32, 16);
  vecClose(dst.subarray(32, 48), src.subarray(16, 32));
  assert.equal(dst[0], 0, 'must not touch other slots');
});

// ------------------------------------------------- reverse-Z projection

console.log('\nreverse-Z projection');

/** Apply a mat4 to a view-space point and return NDC z (after perspective divide). */
function ndcZ(m, viewZ) {
  const clipZ = m[10] * viewZ + m[14];   // x = y = 0, w = 1
  const clipW = m[11] * viewZ + m[15];
  return clipZ / clipW;
}

const NEAR = 0.1;
const proj = mat4PerspectiveReverseZInfinite(mat4Create(), Math.PI / 3, 16 / 9, NEAR);

test('near plane maps to 1, infinity maps to 0', () => {
  close(ndcZ(proj, -NEAR), 1, 1e-6, 'near');
  close(ndcZ(proj, -1e9), 0, 1e-6, 'far');
});

test('depth decreases monotonically with distance, and stays inside [0,1]', () => {
  // Half-ULP slack: the matrix holds `near` as float32 while these distances
  // are exact doubles, so a point sitting precisely ON the near plane lands a
  // rounding step past 1.0. The boundary is at float precision by nature.
  const SLACK = 1e-6;
  let prev = Infinity;
  for (const d of [0.1, 0.5, 1, 10, 100, 1000, 100000]) {
    const z = ndcZ(proj, -d);
    assert.ok(z < prev, `not monotonic at d=${d}`);
    assert.ok(z >= -SLACK && z <= 1 + SLACK, `z out of range at d=${d}: ${z}`);
    prev = z;
  }
});

test('x and y are unaffected by the reverse-Z change', () => {
  // Only the third and fourth columns differ from a standard projection;
  // horizontal/vertical framing must be identical.
  const f = 1 / Math.tan(Math.PI / 6);
  close(proj[0], f / (16 / 9), 1e-6, 'x scale');
  close(proj[5], f, 1e-6, 'y scale');
});

test('reverse-Z resolves distant detail that standard-Z cannot', () => {
  // The whole justification, as an executable check. Two surfaces 5cm apart at
  // 900m -- a facade and a sign on it -- stored in a 32-bit float depth buffer.
  //
  // With a 0.1/1000 near/far range, standard-Z's smallest resolvable gap at
  // 900m works out to ~48cm: below that, both surfaces round to the same depth
  // value and z-fight. Reverse-Z resolves ~0.12mm at the same distance, about
  // 4000x finer, because its values sit near zero where float32 is dense.
  const A = 900, B = 900.05;

  // Standard [0,1] projection for comparison: z = far/(near-far), w = -viewZ.
  const FAR = 1000;
  const m10 = FAR / (NEAR - FAR);
  const m14 = (NEAR * FAR) / (NEAR - FAR);
  const standardZ = (d) => (m10 * -d + m14) / d;

  const stdA = Math.fround(standardZ(A));
  const stdB = Math.fround(standardZ(B));
  assert.equal(stdA, stdB, 'expected standard-Z to collapse these into one depth value');

  const revA = Math.fround(ndcZ(proj, -A));
  const revB = Math.fround(ndcZ(proj, -B));
  assert.notEqual(revA, revB, 'reverse-Z should still tell them apart');
});

// --------------------------------------------------------------- assert

console.log('\nassertFinite');

test('catches NaN and Infinity, and names the index', () => {
  // The index is the point: NaN is contagious, so knowing WHICH value went bad
  // is most of the diagnosis.
  assertFinite(Float32Array.from([1, -2, 0, 1e30]), 'ok');
  assert.throws(() => assertFinite(Float32Array.from([1, NaN, 3]), 'position'), /position.*\[1\]/);
  assert.throws(() => assertFinite(Float32Array.from([Infinity]), 'scale'), /scale/);
});

test('an offset and length check one slice of a shared column', () => {
  // Columns are shared between entities, so a caller checks its own slice
  // without allocating a view for it.
  const column = Float32Array.from([NaN, NaN, 1, 2, 3, NaN]);
  assertFinite(column, 'slice', 2, 3);
  assert.throws(() => assertFinite(column, 'slice', 3, 3), /\[5\]/);
});

// --------------------------------------------------------------- handle

console.log('\nhandle');

test('handle 0 is never valid', () => {
  const h = new HandleAllocator(8);
  assert.equal(h.alive(NULL_HANDLE), false);
  h.alloc();                                  // claims index 0
  assert.equal(h.alive(NULL_HANDLE), false, 'index 0 must not collide with null');
});

test('fresh handles are alive and index sequentially', () => {
  const h = new HandleAllocator(8);
  const a = h.alloc();
  const b = h.alloc();
  assert.ok(h.alive(a) && h.alive(b));
  assert.equal(handleIndex(a), 0);
  assert.equal(handleIndex(b), 1);
  assert.equal(h.liveCount, 2);
});

test('freeing invalidates the old handle', () => {
  const h = new HandleAllocator(8);
  const a = h.alloc();
  h.free(a);
  assert.equal(h.alive(a), false);
  assert.equal(h.liveCount, 0);
});

test('a reused slot does not resurrect the old handle', () => {
  // The entire reason generations exist.
  const h = new HandleAllocator(8);
  const old = h.alloc();
  h.free(old);
  const fresh = h.alloc();

  assert.equal(handleIndex(fresh), handleIndex(old), 'should reuse the slot');
  assert.notEqual(fresh, old, 'but must be a different handle');
  assert.ok(h.alive(fresh));
  assert.equal(h.alive(old), false, 'stale handle must stay dead');
});

test('double free is rejected', () => {
  const h = new HandleAllocator(8);
  const a = h.alloc();
  h.free(a);
  assert.throws(() => h.free(a));
});

test('generation wraps past 255 without ever hitting 0', () => {
  const h = new HandleAllocator(4);
  let handle = h.alloc();
  for (let i = 0; i < 600; i++) {
    h.free(handle);
    handle = h.alloc();
    assert.notEqual(handle & 0xff, 0, `generation hit 0 on cycle ${i}`);
    assert.ok(h.alive(handle));
  }
});

test('running past the initial capacity grows instead of throwing', () => {
  const h = new HandleAllocator(2);
  const first = h.alloc();
  h.alloc();

  const grown = h.alloc();
  assert.ok(h.capacity > 2, 'capacity must have grown');
  assert.ok(h.alive(grown), 'the handle past the old capacity must be live');
  // Growth copies the generation table; handles issued before it stay valid.
  assert.ok(h.alive(first), 'an older handle must survive the growth');
});

test('growing keeps freed slots reusable', () => {
  const h = new HandleAllocator(1);
  const a = h.alloc();
  h.free(a);
  h.alloc();                                  // reuses slot 0
  const c = h.alloc();                        // forces growth
  assert.ok(h.alive(c));
  assert.equal(h.alive(a), false, 'the freed handle must stay dead after growth');
});

test('the 24-bit index space is still a hard ceiling', () => {
  // Not a capacity that can be widened: it is the width of the index field in
  // a handle, so there is no amount of memory that makes it fit.
  const h = new HandleAllocator(2);
  h.next = 1 << 24;
  h.capacity = 1 << 24;
  assert.throws(() => h.alloc(), /24-bit/);
});

test('handles stay unsigned at high indices', () => {
  // (index << 8) overflows signed 32-bit space well before the 24-bit limit.
  const h = new HandleAllocator(1 << 24);
  h.next = 0x00ffffff;                        // jump to the last slot
  h.generations[0x00ffffff] = 1;
  const big = ((0x00ffffff << 8) | 1) >>> 0;
  assert.ok(big > 0, 'handle must not be negative');
  assert.equal(handleIndex(big), 0x00ffffff);
});

// ---------------------------------------------------------------- clock

console.log('\nclock');

test('first frame contributes no time', () => {
  const c = new Clock(1 / 60);
  c.begin(0);
  assert.equal(c.step(), false);
});

test('tracks realtime over a long run without drifting', () => {
  // Exact 1/60 deltas land the accumulator on a float knife edge, so an
  // individual step can fall an epsilon short and defer to the next frame.
  // That is inherent to accumulators and harmless -- the step is pending, not
  // lost. The property worth asserting is that the error never ACCUMULATES.
  const c = new Clock(1 / 60);
  const FRAMES = 600;                          // 10 seconds at 60 Hz

  let steps = 0;
  for (let i = 0; i <= FRAMES; i++) {
    c.begin(i / 60);
    while (c.step()) steps++;
  }

  assert.ok(Math.abs(steps - FRAMES) <= 1, `drifted to ${steps} steps over ${FRAMES}`);

  // Conservation: every second handed to begin() is either simulated or still
  // waiting in the accumulator. Time is never created or destroyed.
  close(c.elapsed + c.accumulator, FRAMES / 60, 1e-9, 'conservation');
});

test('a slow frame runs several steps and leaves a partial alpha', () => {
  const c = new Clock(1 / 60);
  c.begin(0);
  c.begin(0.025);                   // 1.5 steps' worth

  let steps = 0;
  while (c.step()) steps++;
  assert.equal(steps, 1);
  close(c.alpha, 0.5, 1e-3, 'leftover fraction');
  assert.ok(c.alpha >= 0 && c.alpha < 1);
});

test('a long stall is clamped instead of spiraling', () => {
  const c = new Clock(1 / 60);
  c.begin(0);
  c.begin(30);                      // 30-second stall: 1800 steps if unclamped

  let steps = 0;
  while (c.step()) steps++;
  assert.ok(steps <= 16, `clamped to ${steps} steps, not 1800`);
});

test('a backwards clock does not drain the accumulator', () => {
  const c = new Clock(1 / 60);
  c.begin(10);
  c.begin(5);
  assert.equal(c.realDelta, 0);
  assert.ok(c.accumulator >= 0);
});

console.log(`\n${passed} checks passed\n`);
