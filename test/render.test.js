// Sort keys, draw lists and world bounds. Run: node test/render.test.js

import assert from 'node:assert/strict';

import { aabbTransform, aabbBoundingSphere, aabbUnion, aabbSetEmpty } from '../src/core/math/aabb.js';
import {
  frustumCreate, frustumFromViewProjection, frustumTestAABB, frustumTestSphere,
  FRUSTUM_PLANE_COUNT, PLANE_NEAR, PLANE_LEFT,
} from '../src/core/math/frustum.js';
import {
  mat4Create, mat4LookAt, mat4Multiply, mat4PerspectiveReverseZInfinite, mat4FromQuatPosScale,
} from '../src/core/math/mat4.js';
import { quatCreate, quatSetAxisAngle } from '../src/core/math/quat.js';
import { vec3Create } from '../src/core/math/vec3.js';
import { Camera } from '../src/scene/camera.js';
import { Scene } from '../src/scene/scene.js';
import { GpuDriven } from '../src/render/gpudriven.js';
import {
  DrawList, opaqueSortKey, transparentSortKey,
  transparentDepthBucket,
  OPAQUE_PIPELINE_BITS, OPAQUE_MATERIAL_BITS, OPAQUE_DEPTH_BITS,
  TRANSPARENT_PIPELINE_BITS, TRANSPARENT_MATERIAL_BITS, TRANSPARENT_DEPTH_BITS,
} from '../src/render/drawlist.js';
import { updateWorldBounds } from '../src/scene/bounds.js';
import {
  variantKey, variantPipelineState, ALPHA_OPAQUE, ALPHA_MASK, ALPHA_BLEND,
} from '../src/render/material.js';

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`  ok  ${name}`);
}

const EPS = 1e-5;
function close(a, b, eps = EPS, what = '') {
  assert.ok(Math.abs(a - b) <= eps, `${what} expected ${b}, got ${a}`);
}
function vecClose(a, b, eps = EPS, what = '') {
  for (let i = 0; i < b.length; i++) close(a[i], b[i], eps, `${what}[${i}]`);
}

/** Camera at the origin looking down -Z, 90 degree vertical fov, near = 1. */
function forwardCamera(near = 1) {
  const camera = new Camera({ fovY: Math.PI / 2, near });
  camera.position.set([0, 0, 0]);
  camera.target.set([0, 0, -1]);
  camera.update(1);
  return camera;
}

function boxAt(x, y, z, halfSize = 0.5) {
  return {
    min: Float32Array.from([x - halfSize, y - halfSize, z - halfSize]),
    max: Float32Array.from([x + halfSize, y + halfSize, z + halfSize]),
  };
}

// ------------------------------------------------------------------- aabb

console.log('\naabb');

test('transform applies translation, rotation and scale', () => {
  const q = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), Math.PI / 2);
  const m = mat4FromQuatPosScale(mat4Create(), q, vec3Create(10, 0, 0), vec3Create(2, 2, 2));

  const outMin = new Float32Array(3);
  const outMax = new Float32Array(3);
  aabbTransform(outMin, outMax, Float32Array.from([-1, -1, -1]), Float32Array.from([1, 1, 1]), m);

  // Scaled to +/-2, rotated (a symmetric box is unchanged), moved to x = 10.
  vecClose(outMin, [8, -2, -2], EPS, 'min');
  vecClose(outMax, [12, 2, 2], EPS, 'max');
});

test('rotating a non-cubic box grows its axis-aligned bounds', () => {
  // A long thin box rotated 45 degrees cannot stay the same size when
  // re-expressed on the world axes. That conservatism is the whole reason
  // world bounds are always rebuilt from local ones.
  const q = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), Math.PI / 4);
  const m = mat4FromQuatPosScale(mat4Create(), q, vec3Create(0, 0, 0), vec3Create(1, 1, 1));

  const outMin = new Float32Array(3);
  const outMax = new Float32Array(3);
  aabbTransform(outMin, outMax, Float32Array.from([-4, -1, -1]), Float32Array.from([4, 1, 1]), m);

  const expected = 4 * Math.SQRT1_2 + 1 * Math.SQRT1_2;
  close(outMax[0], expected, 1e-5, 'grew on x');
  close(outMax[2], expected, 1e-5, 'grew on z');
});

test('offsets address into packed bounds columns', () => {
  const localMin = Float32Array.from([0, 0, 0, -1, -1, -1]);
  const localMax = Float32Array.from([0, 0, 0, 1, 1, 1]);
  const worldMin = new Float32Array(6);
  const worldMax = new Float32Array(6);
  const m = mat4FromQuatPosScale(
    mat4Create(), quatCreate(), vec3Create(5, 0, 0), vec3Create(1, 1, 1),
  );

  aabbTransform(worldMin, worldMax, localMin, localMax, m, 0, 3, 3);
  vecClose(worldMin.subarray(3, 6), [4, -1, -1], EPS, 'slot 1 min');
  assert.equal(worldMin[0], 0, 'slot 0 untouched');
});

test('bounding sphere and union', () => {
  const center = new Float32Array(3);
  const radius = aabbBoundingSphere(center, Float32Array.from([-1, -1, -1]), Float32Array.from([1, 1, 1]));
  vecClose(center, [0, 0, 0]);
  close(radius, Math.sqrt(3));

  const min = new Float32Array(3);
  const max = new Float32Array(3);
  aabbSetEmpty(min, max);
  aabbUnion(min, max, Float32Array.from([0, 0, 0]), Float32Array.from([1, 1, 1]));
  aabbUnion(min, max, Float32Array.from([-2, 0, 0]), Float32Array.from([0, 3, 0]));
  vecClose(min, [-2, 0, 0], EPS, 'union min');
  vecClose(max, [1, 3, 1], EPS, 'union max');
});

// ---------------------------------------------------------------- frustum

console.log('\nfrustum');

test('extracts five planes, not six', () => {
  // The sixth would be the far plane, which does not exist under an infinite
  // projection. Extracting it anyway yields a zero-length normal and every
  // test comes back NaN.
  const frustum = frustumCreate();
  assert.equal(frustum.length, FRUSTUM_PLANE_COUNT * 4);
  assert.equal(FRUSTUM_PLANE_COUNT, 5);

  frustumFromViewProjection(frustum, forwardCamera().viewProjection);
  for (let i = 0; i < frustum.length; i++) {
    assert.ok(Number.isFinite(frustum[i]), `plane component ${i} is not finite`);
  }
});

test('plane normals come out unit length', () => {
  const frustum = frustumFromViewProjection(frustumCreate(), forwardCamera().viewProjection);
  for (let p = 0; p < FRUSTUM_PLANE_COUNT; p++) {
    const o = p * 4;
    close(Math.hypot(frustum[o], frustum[o + 1], frustum[o + 2]), 1, 1e-5, `plane ${p}`);
  }
});

test('the near plane sits exactly at the camera near distance', () => {
  const frustum = frustumFromViewProjection(frustumCreate(), forwardCamera(1).viewProjection);
  const o = PLANE_NEAR * 4;

  // Signed distance of a point 5 in front should be 5 - near = 4.
  const distance = frustum[o] * 0 + frustum[o + 1] * 0 + frustum[o + 2] * -5 + frustum[o + 3];
  close(distance, 4, 1e-5);
});

test('the left plane passes through the 90-degree fov edge', () => {
  const frustum = frustumFromViewProjection(frustumCreate(), forwardCamera(1).viewProjection);
  const o = PLANE_LEFT * 4;
  // At 90 degrees and depth 1, the frustum edge is at x = -1.
  const onEdge = frustum[o] * -1 + frustum[o + 2] * -1 + frustum[o + 3];
  close(onEdge, 0, 1e-5, 'point on the edge has zero distance');
});

test('a finite projection is rejected instead of silently losing a plane', () => {
  // Standard-Z projection: near/far both real, so the far row is NOT
  // degenerate and a five-plane frustum would leak distant geometry.
  const finite = mat4Create();
  finite[0] = 1; finite[5] = 1; finite[10] = -1.002; finite[11] = -1; finite[14] = -0.2; finite[15] = 0;
  assert.throws(() => frustumFromViewProjection(frustumCreate(), finite), /infinite reverse-Z/);
});

console.log('\nculling');

test('keeps what is in front and rejects what is behind', () => {
  const frustum = frustumFromViewProjection(frustumCreate(), forwardCamera(1).viewProjection);

  const front = boxAt(0, 0, -5);
  assert.equal(frustumTestAABB(frustum, front.min, front.max), true, 'in front');

  const behind = boxAt(0, 0, 5);
  assert.equal(frustumTestAABB(frustum, behind.min, behind.max), false, 'behind the camera');

  const tooClose = boxAt(0, 0, -0.2, 0.1);
  assert.equal(frustumTestAABB(frustum, tooClose.min, tooClose.max), false, 'inside the near plane');
});

test('rejects geometry outside the side planes', () => {
  const frustum = frustumFromViewProjection(frustumCreate(), forwardCamera(1).viewProjection);

  const wayLeft = boxAt(-50, 0, -5);
  assert.equal(frustumTestAABB(frustum, wayLeft.min, wayLeft.max), false, 'far to the left');

  const wayUp = boxAt(0, 50, -5);
  assert.equal(frustumTestAABB(frustum, wayUp.min, wayUp.max), false, 'far above');
});

test('a box straddling a plane is kept', () => {
  // Partial visibility must survive: a false negative here is a hole in the
  // image, which is much worse than a wasted draw.
  const frustum = frustumFromViewProjection(frustumCreate(), forwardCamera(1).viewProjection);
  const straddling = boxAt(-5, 0, -5, 1.5);
  assert.equal(frustumTestAABB(frustum, straddling.min, straddling.max), true);
});

test('nothing is ever culled for being too far away', () => {
  // The direct consequence of the infinite far plane: no draw distance exists.
  const frustum = frustumFromViewProjection(frustumCreate(), forwardCamera(0.1).viewProjection);
  for (const distance of [100, 10_000, 1e7]) {
    const box = boxAt(0, 0, -distance, 1);
    assert.equal(frustumTestAABB(frustum, box.min, box.max), true, `at ${distance}`);
  }
});

test('sphere test agrees with the box test', () => {
  const frustum = frustumFromViewProjection(frustumCreate(), forwardCamera(1).viewProjection);
  assert.equal(frustumTestSphere(frustum, Float32Array.from([0, 0, -5]), 1), true);
  assert.equal(frustumTestSphere(frustum, Float32Array.from([0, 0, 5]), 1), false);
  // Behind the camera but big enough to reach into view.
  assert.equal(frustumTestSphere(frustum, Float32Array.from([0, 0, 2]), 12), true);
});

test('world bounds update only for dirty transforms', () => {
  const localMin = Float32Array.from([-1, -1, -1, -1, -1, -1]);
  const localMax = Float32Array.from([1, 1, 1, 1, 1, 1]);
  const worldMin = new Float32Array(6);
  const worldMax = new Float32Array(6);

  const matrices = new Float32Array(32);
  mat4FromQuatPosScale(matrices, quatCreate(), vec3Create(5, 0, 0), vec3Create(1, 1, 1), 0);
  mat4FromQuatPosScale(matrices, quatCreate(), vec3Create(0, 9, 0), vec3Create(1, 1, 1), 16);

  const matrixSlot = Uint32Array.from([0, 1]);
  const dirty = Uint8Array.from([1, 0]);

  const updated = updateWorldBounds(2, localMin, localMax, worldMin, worldMax, matrices, matrixSlot, dirty);
  assert.equal(updated, 1, 'only the dirty one');
  vecClose(worldMin.subarray(0, 3), [4, -1, -1], EPS, 'updated');
  vecClose(worldMin.subarray(3, 6), [0, 0, 0], EPS, 'skipped, still zero');
});

test('the opaque key fills exactly 32 bits', () => {
  assert.equal(OPAQUE_PIPELINE_BITS + OPAQUE_MATERIAL_BITS + OPAQUE_DEPTH_BITS, 32);
});

test('opaque order is pipeline, then material, then near-to-far', () => {
  const a = opaqueSortKey(1, 0, 0);
  const b = opaqueSortKey(0, 999, 999);
  assert.ok(b < a, 'pipeline dominates material and depth');

  const c = opaqueSortKey(3, 1, 500);
  const d = opaqueSortKey(3, 2, 0);
  assert.ok(c < d, 'material dominates depth within one pipeline');

  const near = opaqueSortKey(3, 1, 10);
  const far = opaqueSortKey(3, 1, 900);
  assert.ok(near < far, 'nearer sorts first inside one material');
});

test('transparent order is depth first, whatever the state costs', () => {
  // Pipeline ids are a dense index over variantKey, which has at most six
  // distinct values, so 5 is the worst real case rather than 200.
  const farCheapState = transparentSortKey(0, 0, 10);
  const nearExpensiveState = transparentSortKey(5, 4095, 900);
  assert.ok(farCheapState < nearExpensiveState, 'depth outranks state');
});

test('both sort keys address the same number of materials', () => {
  // The transparent material field was 8 bits while the registry guarded 4096
  // from the opaque one, so a blended material past 255 ORed a bit into the
  // pipeline field and silently reordered the frame -- in release, where the
  // DEBUG assert below does not exist.
  assert.equal(TRANSPARENT_MATERIAL_BITS, OPAQUE_MATERIAL_BITS);
  assert.equal(TRANSPARENT_DEPTH_BITS + TRANSPARENT_PIPELINE_BITS + TRANSPARENT_MATERIAL_BITS, 32);

  const highMaterial = transparentSortKey(0, 4095, 0);
  assert.equal(highMaterial, 4095, 'the top material id occupies the material field alone');
});

test('keys stay unsigned at the top of the pipeline field', () => {
  // (pipelineId << 22) sets bit 31, which is negative in signed 32-bit math.
  const key = opaqueSortKey(1023, 4095, 1023);
  assert.ok(key > 0, `key went negative: ${key}`);
  assert.equal(key, 0xffffffff);
});

test('an id too large for its field is caught, not aliased', () => {
  assert.throws(() => opaqueSortKey(1 << OPAQUE_PIPELINE_BITS, 0, 0), /pipeline id/);
  assert.throws(() => opaqueSortKey(0, 1 << OPAQUE_MATERIAL_BITS, 0), /material id/);
});

test('depth buckets derive from the reverse-Z curve, far-first for blending', () => {
  // Only the transparent bucket exists now. The opaque key's depth field is
  // always zero, because a BATCH has no single depth -- so the function that
  // would have filled it had no caller and is gone.
  const near = 0.1;
  let previous = Infinity;
  for (const distance of [0.1, 1, 10, 1000, 1e6]) {
    const bucket = transparentDepthBucket(near, distance);
    assert.ok(bucket <= previous, `not monotonic at ${distance}`);
    assert.ok(bucket >= 0 && bucket <= 65535, `out of range at ${distance}: ${bucket}`);
    previous = bucket;
  }
  assert.equal(transparentDepthBucket(near, near), 65535, 'at the near plane, drawn last');
});

test('transparent buckets run the other way, far-first', () => {
  const near = 0.1;
  assert.ok(transparentDepthBucket(near, 1000) < transparentDepthBucket(near, 1),
    'distant geometry must be drawn before near geometry');
});

test('degenerate distances do not produce NaN buckets', () => {
  for (const distance of [0, -5, NaN]) {
    const bucket = transparentDepthBucket(0.1, distance);
    assert.ok(Number.isInteger(bucket) && bucket >= 0, `distance ${distance} gave ${bucket}`);
  }
});

// -------------------------------------------------------------- draw list

console.log('\ndraw list');

test('push and clear', () => {
  const list = new DrawList(4);
  list.push(5, 100);
  list.push(3, 200);
  assert.equal(list.count, 2);
  list.clear();
  assert.equal(list.count, 0);
});

test('overflow grows rather than dropping draws', () => {
  const list = new DrawList(1);
  list.push(1, 1);
  list.push(2, 2);
  assert.equal(list.count, 2);
  assert.ok(list.capacity >= 2);
  assert.deepEqual([...list.payloads.subarray(0, 2)], [1, 2], 'earlier entries survive');
});

test('a list that grew still sorts correctly', () => {
  // The radix ping-pong buffers are reallocated by growth; if they came back
  // shorter than the list, the sort would read past its own data.
  const list = new DrawList(2);
  const keys = [];
  for (let i = 0; i < 500; i++) {
    const key = (i * 2654435761) >>> 8;       // scattered, deterministic
    keys.push(key);
    list.push(key, i);
  }
  list.sort();

  keys.sort((a, b) => a - b);
  assert.deepEqual([...list.keys.subarray(0, 500)], keys);
  for (let i = 1; i < 500; i++) {
    assert.ok(list.keys[i - 1] <= list.keys[i], 'not ordered after growth');
  }
});

test('radix sort matches a reference sort and keeps payloads attached', () => {
  const list = new DrawList(2000);
  // Deterministic pseudo-random keys spanning the full 32-bit range, including
  // values with the high bit set.
  let seed = 12345;
  const expected = [];
  for (let i = 0; i < 2000; i++) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    list.push(seed, i);
    expected.push({ key: seed, payload: i });
  }
  expected.sort((a, b) => a.key - b.key);

  list.sort();

  for (let i = 0; i < expected.length; i++) {
    assert.equal(list.keys[i], expected[i].key, `key at ${i}`);
    assert.equal(list.payloads[i], expected[i].payload, `payload at ${i}`);
  }
});

test('sorting is stable enough to group identical keys together', () => {
  const list = new DrawList(6);
  for (const [key, payload] of [[7, 0], [3, 1], [7, 2], [3, 3], [7, 4], [1, 5]]) {
    list.push(key, payload);
  }
  list.sort();
  assert.deepEqual([...list.keys.subarray(0, 6)], [1, 3, 3, 7, 7, 7]);
  // Counting sort preserves input order within a bucket, which is what keeps
  // one material's draws contiguous instead of interleaved.
  assert.deepEqual([...list.payloads.subarray(0, 6)], [5, 1, 3, 0, 2, 4]);
});

test('a list whose high bytes are all equal still sorts correctly', () => {
  // Exercises the skipped-pass path: an odd number of executed passes leaves
  // the data in the scratch buffers, and the public arrays must follow it.
  const list = new DrawList(5);
  for (const key of [40, 10, 30, 20, 50]) list.push(key, key);
  list.sort();
  assert.deepEqual([...list.keys.subarray(0, 5)], [10, 20, 30, 40, 50]);
  assert.deepEqual([...list.payloads.subarray(0, 5)], [10, 20, 30, 40, 50]);
});

test('sorting an empty or single-item list is a no-op', () => {
  const empty = new DrawList(4);
  empty.sort();
  assert.equal(empty.count, 0);

  const one = new DrawList(4);
  one.push(9, 9);
  one.sort();
  assert.equal(one.keys[0], 9);
});

test('a realistic list comes out grouped by pipeline then material', () => {
  const list = new DrawList(64);

  // Deliberately pushed in a scrambled order.
  const items = [
    { pipeline: 2, material: 5, distance: 50 },
    { pipeline: 0, material: 9, distance: 3 },
    { pipeline: 2, material: 1, distance: 8 },
    { pipeline: 0, material: 9, distance: 1 },
    { pipeline: 2, material: 5, distance: 2 },
  ];
  items.forEach((item, i) => {
    // Depth is 0 for every batch in the real renderer; the sort under test is
    // the state grouping, so vary it here only to prove it is the weakest field.
    list.push(opaqueSortKey(item.pipeline, item.material, item.distance | 0), i);
  });
  list.sort();

  const order = [...list.payloads.subarray(0, items.length)].map((i) => items[i]);
  assert.deepEqual(order.map((o) => o.pipeline), [0, 0, 2, 2, 2], 'pipelines grouped');
  assert.deepEqual(order.map((o) => o.material), [9, 9, 1, 5, 5], 'materials grouped within pipeline');
  assert.ok(order[0].distance < order[1].distance, 'near before far within a material');
  assert.ok(order[3].distance < order[4].distance, 'near before far within a material');
});

// ------------------------------------------------------- material variants

console.log('\nmaterial variants');

test('variants exist only for state a uniform cannot express', () => {
  // Cull mode, blend state and whether the shader discards. Texture presence
  // is deliberately NOT in here -- absent maps bind a 1x1 default instead of
  // spawning a shader permutation.
  assert.equal(variantKey(ALPHA_OPAQUE, false), 0);
  assert.notEqual(variantKey(ALPHA_OPAQUE, true), variantKey(ALPHA_OPAQUE, false));
  assert.notEqual(variantKey(ALPHA_BLEND, false), variantKey(ALPHA_MASK, false));
});

test('blended materials do not write depth', () => {
  // A blended surface does not occlude what is behind it, and writing depth
  // would make the result depend on draw order in a way no sort can fix.
  const blend = variantPipelineState(variantKey(ALPHA_BLEND, false));
  assert.equal(blend.depth.depthWriteEnabled, false);
  assert.ok(blend.blend, 'blend state present');

  const opaque = variantPipelineState(variantKey(ALPHA_OPAQUE, false));
  assert.equal(opaque.depth.depthWriteEnabled, true);
  assert.equal(opaque.blend, undefined);
});

test('only the masked variant enables the discard', () => {
  // A discard anywhere in a shader disables early-Z for every draw using it,
  // so opaque geometry must compile without it.
  assert.equal(variantPipelineState(variantKey(ALPHA_MASK, false)).constants.USE_ALPHA_MASK, 1);
  assert.equal(variantPipelineState(variantKey(ALPHA_OPAQUE, false)).constants.USE_ALPHA_MASK, 0);
});

test('double-sided materials disable back-face culling', () => {
  assert.equal(variantPipelineState(variantKey(ALPHA_OPAQUE, true)).primitive.cullMode, 'none');
  assert.equal(variantPipelineState(variantKey(ALPHA_OPAQUE, false)).primitive.cullMode, 'back');
});

test('every variant keeps the reverse-Z depth compare', () => {
  for (let variant = 0; variant < 8; variant++) {
    assert.equal(variantPipelineState(variant).depth.depthCompare, 'greater',
      `variant ${variant} broke reverse-Z`);
  }
});

test('variant ids fit the sort key pipeline field', () => {
  // Eight possible variants against a 10-bit field, so the pipeline id minted
  // from a variant can never overflow the key.
  assert.ok(8 <= (1 << OPAQUE_PIPELINE_BITS));
});

// ------------------------------------------------------------------ winding

console.log('\nmirrored instances');

test('a mirrored variant reverses the front face', () => {
  const normal = variantPipelineState(variantKey(ALPHA_OPAQUE, false, false));
  const mirrored = variantPipelineState(variantKey(ALPHA_OPAQUE, false, true));
  assert.equal(normal.primitive.frontFace, 'ccw');
  assert.equal(mirrored.primitive.frontFace, 'cw');
  assert.equal(mirrored.primitive.cullMode, 'back', 'still culls, just the other side');
});

test('the mirrored bit does not disturb the others', () => {
  const v = variantKey(ALPHA_MASK, true, true);
  const state = variantPipelineState(v);
  assert.equal(state.primitive.cullMode, 'none', 'double sided survives');
  assert.equal(state.constants.USE_ALPHA_MASK, 1, 'alpha mode survives');
  assert.equal(state.primitive.frontFace, 'cw');
  assert.ok(v < 16, `variant ${v} must still fit the 4-bit pipeline field`);
});

/** A scene with two unit cubes, whose scales the caller picks. */
function windingScene(scales) {
  const scene = new Scene({ capacity: 16 });
  const primitive = {
    indexCount: 36, materialId: 0,
    bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
  };
  for (const scale of scales) {
    const entity = scene.entities.alloc();
    scene.transforms.add(entity, { position: [0, 0, 0], scale });
    scene._addRenderable(entity, primitive);
  }
  scene.update();
  return scene;
}

/** GpuDriven.rebuildBatches without a GPU. */
function batchesFor(scene) {
  const gpu = Object.create(GpuDriven.prototype);
  gpu.capacity = 16;
  gpu.materials = { isTransparent: () => false };
  gpu.itemBatch = new Uint32Array(16);
  gpu.itemMirrored = new Uint8Array(16);
  gpu.batchFirst = new Uint32Array(16);
  gpu.batchOrder = new Uint32Array(16);
  gpu.batchMaterial = new Uint16Array(16);
  gpu.batchMirrored = new Uint8Array(16);
  gpu.batchSize = new Uint32Array(16);
  gpu.batchPrimitive = [];
  gpu.transparentItems = new Uint32Array(16);
  gpu.alignment = 256;
  gpu.batchStaging = new ArrayBuffer(256 * 33);
  gpu._zeroFlags = new Uint32Array(16);
  gpu.stats = {};
  gpu.rhi = { queue: { writeBuffer() {} } };
  for (const k of ['itemBatchBuffer', 'batchFirstBuffer', 'batchOrderBuffer', 'batchBuffer', 'visibleFlagsBuffer']) gpu[k] = {};
  gpu.rebuildBatches(scene);
  return gpu;
}

test('a mirrored instance is a batch of its own', () => {
  // Same mesh, same material. One indirect draw has one front face, so these
  // cannot share a batch however identical everything else is.
  const gpu = batchesFor(windingScene([[1, 1, 1], [-1, 1, 1]]));
  assert.equal(gpu.batchCount, 2, 'two windings, two batches');
  assert.notEqual(gpu.batchMirrored[gpu.itemBatch[0]], gpu.batchMirrored[gpu.itemBatch[1]]);
});

test('identical windings still share a batch', () => {
  // The negative control: splitting on winding must not split on nothing.
  const gpu = batchesFor(windingScene([[1, 1, 1], [1, 2, 3]]));
  assert.equal(gpu.batchCount, 1);
});

test('two mirrored instances share a batch with each other', () => {
  const gpu = batchesFor(windingScene([[-1, 1, 1], [1, -1, 1]]));
  assert.equal(gpu.batchCount, 1, 'both mirror, so both draw cw');
  assert.equal(gpu.batchMirrored[0], 1);
});

test('mirroring is counted per axis, so two reflections cancel', () => {
  // A determinant test, not a "has a negative number in it" test. Scaling two
  // axes by -1 is a 180 degree rotation and the winding is unchanged.
  const gpu = batchesFor(windingScene([[1, 1, 1], [-1, -1, 1], [-1, -1, -1]]));
  assert.equal(gpu.itemMirrored[0], 0, 'identity');
  assert.equal(gpu.itemMirrored[1], 0, 'two axes flipped cancel');
  assert.equal(gpu.itemMirrored[2], 1, 'three axes flipped do not');
});

test('a mirroring parent mirrors its children', () => {
  // The reason this reads the WORLD matrix and not the node's own.
  const scene = new Scene({ capacity: 16 });
  const primitive = {
    indexCount: 36, materialId: 0,
    bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
  };
  const parent = scene.entities.alloc();
  scene.transforms.add(parent, { scale: [-1, 1, 1] });

  const child = scene.entities.alloc();
  scene.transforms.add(child, { parent, scale: [1, 1, 1] });
  scene._addRenderable(child, primitive);

  const grandchild = scene.entities.alloc();
  scene.transforms.add(grandchild, { parent: child, scale: [-1, 1, 1] });
  scene._addRenderable(grandchild, primitive);

  scene.update();
  const gpu = batchesFor(scene);
  assert.equal(gpu.itemMirrored[0], 1, 'unmirrored child of a mirroring parent IS mirrored');
  assert.equal(gpu.itemMirrored[1], 0, 'and a mirrored child of it is not');
});

console.log(`\n${passed} checks passed\n`);
