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
import { SkinPalette } from '../src/render/skin.js';
import { PBR_SHADER } from '../src/render/shaders/pbr.js';
import { OIT_RESOLVE_SHADER } from '../src/render/shaders/oit.js';
import { HZB_SHADER } from '../src/render/hzb.js';
import { CLUSTER_SHADER } from '../src/render/clustered.js';
import { POST_SHADER } from '../src/render/post.js';
import {
  DrawList, opaqueSortKey, transparentSortKey,
  transparentDepthBucket,
  OPAQUE_PIPELINE_BITS, OPAQUE_MATERIAL_BITS, OPAQUE_DEPTH_BITS,
  TRANSPARENT_PIPELINE_BITS, TRANSPARENT_MATERIAL_BITS, TRANSPARENT_DEPTH_BITS,
} from '../src/render/drawlist.js';
import {
  updateWorldBounds, unionWorldBounds, farthestViewDepth, updateSkinBounds, applySkinBounds,
} from '../src/scene/bounds.js';
import { jointInfluenceRadii } from '../src/scene/gltf/skin.js';
import { handleIndex } from '../src/core/handle.js';
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
  // Derived, not written down: the field width has moved twice.
  const max = (1 << TRANSPARENT_DEPTH_BITS) - 1;
  const near = 0.1;
  let previous = Infinity;
  for (const distance of [0.1, 1, 10, 1000, 1e6]) {
    const bucket = transparentDepthBucket(near, distance);
    assert.ok(bucket <= previous, `not monotonic at ${distance}`);
    assert.ok(bucket >= 0 && bucket <= max, `out of range at ${distance}: ${bucket}`);
    previous = bucket;
  }
  assert.equal(transparentDepthBucket(near, near), max, 'at the near plane, drawn last');
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

// _growBatches names WebGPU usage flags the way every module does, so Node
// needs them to exist. Real values.
globalThis.GPUBufferUsage ??= {
  MAP_READ: 0x0001, COPY_SRC: 0x0004, COPY_DST: 0x0008,
  INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
  INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
};

/** A GpuDriven whose batch tables can actually grow. */
function growableGpu(materials) {
  const gpu = Object.create(GpuDriven.prototype);
  const device = { createBuffer: () => ({ destroy() {} }) };
  gpu.rhi = { device, queue: { writeBuffer() {} } };
  gpu.capacity = 1024;
  gpu.batchCapacity = 256;
  gpu.alignment = 256;
  gpu.materials = materials;
  gpu.itemBatch = new Uint32Array(1024);
  gpu.itemMirrored = new Uint8Array(1024);
  gpu.batchOrder = new Uint32Array(1024);
  gpu.transparentItems = new Uint32Array(1024);
  gpu._zeroFlags = new Uint32Array(1024);
  gpu.batchFirst = new Uint32Array(gpu.batchCapacity);
  gpu.batchMaterial = new Uint16Array(gpu.batchCapacity);
  gpu.batchMirrored = new Uint8Array(gpu.batchCapacity);
  gpu.batchSkinned = new Uint8Array(gpu.batchCapacity);
  gpu.batchSize = new Uint32Array(gpu.batchCapacity);
  gpu.indirectData = new Uint32Array(gpu.batchCapacity * 2 * 5);
  gpu.batchStaging = new ArrayBuffer(gpu.alignment * (gpu.batchCapacity * 2 + 1));
  gpu.batchPrimitive = [];
  gpu.buffersRevision = 0;
  gpu.stats = {};
  for (const k of ['itemBatchBuffer', 'batchFirstBuffer', 'batchOrderBuffer',
    'batchBuffer', 'visibleFlagsBuffer', 'indirectBuffer']) gpu[k] = { destroy() {} };
  return gpu;
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
  gpu.batchSkinned = new Uint8Array(16);
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

test('batch tables grow on batch count, and carry over what the loop wrote', () => {
  // The batch arrays are written AS the loop discovers batches, so a grow in
  // the middle of it has to preserve them. Sizing them per renderable is what
  // this replaces: 512 bytes of uniform buffer each for a per-batch value,
  // which put the batch-info buffer past maxBufferSize above half a million
  // renderables -- where createBuffer returns an invalid buffer rather than
  // throwing, and the frame goes black.
  const scene = new Scene({ capacity: 1024 });
  const materials = { isTransparent: () => false };

  // 400 distinct primitives, so 400 distinct batches, past the 256 start.
  const COUNT = 400;
  for (let i = 0; i < COUNT; i++) {
    const entity = scene.entities.alloc();
    scene.transforms.add(entity, { position: [i, 0, 0] });
    scene._addRenderable(entity, {
      indexCount: 3, materialId: i % 7,
      bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    });
  }
  scene.update();

  const gpu = growableGpu(materials);
  gpu.rebuildBatches(scene);

  assert.equal(gpu.batchCount, COUNT, 'every primitive is its own batch');
  assert.ok(gpu.batchCapacity >= COUNT, `capacity ${gpu.batchCapacity} did not keep up`);
  assert.ok(gpu.batchCapacity < scene.renderableCount * 4, 'and did not overshoot wildly');

  // The three the loop writes must all have survived the grow.
  for (let b = 0; b < COUNT; b++) {
    assert.equal(gpu.batchSize[b], 1, `batch ${b} lost its size`);
  }
  const materialsSeen = new Set();
  for (let b = 0; b < COUNT; b++) materialsSeen.add(gpu.batchMaterial[b]);
  assert.equal(materialsSeen.size, 7, 'material ids survived the grow');

  // And the prefix sum, which runs after it, still partitions the list.
  let running = 0;
  for (let b = 0; b < COUNT; b++) {
    assert.equal(gpu.batchFirst[b], running, `batch ${b} base is wrong after growing`);
    running += gpu.batchSize[b];
  }
  assert.equal(running, COUNT);
});

test('the batch tables do not grow with renderables that share a batch', () => {
  // The negative control, and the whole reason these are sized separately:
  // 500 instances of one mesh are one batch.
  const scene = new Scene({ capacity: 1024 });
  const primitive = { indexCount: 3, materialId: 0, bounds: { min: [-1, -1, -1], max: [1, 1, 1] } };
  for (let i = 0; i < 500; i++) {
    const entity = scene.entities.alloc();
    scene.transforms.add(entity, { position: [i, 0, 0] });
    scene._addRenderable(entity, primitive);
  }
  scene.update();

  const gpu = growableGpu({ isTransparent: () => false });
  gpu.rebuildBatches(scene);
  assert.equal(gpu.batchCount, 1);
  assert.equal(gpu.batchCapacity, 256, 'nothing grew');
});

// -------------------------------------------------------------- scene extent

console.log('\nscene extent');

test('the union covers every renderable, and shrinks when one moves in', () => {
  // Not incremental, and this is why: a renderable that MOVES can shrink the
  // union as easily as grow it, so there is nothing to update in place.
  const worldMin = Float32Array.from([-5, -1, -1, 1, 1, 1]);
  const worldMax = Float32Array.from([-4, 0, 0, 9, 2, 2]);
  const min = new Float32Array(3);
  const max = new Float32Array(3);

  assert.equal(unionWorldBounds(2, worldMin, worldMax, min, max), true);
  vecClose(min, [-5, -1, -1], EPS, 'min');
  vecClose(max, [9, 2, 2], EPS, 'max');

  worldMin.set([0, 0, 0], 3);
  worldMax.set([1, 1, 1], 3);
  unionWorldBounds(2, worldMin, worldMax, min, max);
  // max is now the union of [-4,0,0] and [1,1,1], down from 9 on x.
  vecClose(max, [1, 1, 1], EPS, 'the far object came home, so the union shrank');
});

test('an empty scene has no bounds, rather than a box meaning nothing', () => {
  const min = Float32Array.from([7, 7, 7]);
  const max = Float32Array.from([8, 8, 8]);
  assert.equal(unionWorldBounds(0, new Float32Array(0), new Float32Array(0), min, max), false);
  vecClose(min, [7, 7, 7], EPS, 'outputs untouched');
});

test('scene depth is measured along the view axis, over all eight corners', () => {
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  camera.position.set([0, 0, 0]);
  camera.target.set([0, 0, -1]);
  camera.update(1);

  // A box from z = -2 to z = -10: the far face is 10 deep.
  close(farthestViewDepth(camera.view, [-1, -1, -10], [1, 1, -2]), 10, EPS, 'straight ahead');

  // Straddling the camera. The near corner is BEHIND it, which is a negative
  // depth, and taking the max over corners is what keeps that from winning.
  close(farthestViewDepth(camera.view, [-1, -1, -4], [1, 1, 3]), 4, EPS, 'straddling');

  // Entirely behind: every corner is negative, and the caller floors it.
  assert.ok(farthestViewDepth(camera.view, [-1, -1, 2], [1, 1, 5]) < 0, 'behind the camera');
});

test('a box off to the side is measured by depth, not by distance', () => {
  // The bug class this engine has hit three times. A box 100 units to the RIGHT
  // and 1 unit ahead is 1 deep, not 100 away.
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  camera.position.set([0, 0, 0]);
  camera.target.set([0, 0, -1]);
  camera.update(1);
  close(farthestViewDepth(camera.view, [99, 0, -1], [101, 1, -1]), 1, EPS);
});

// -------------------------------------------------------------- bind pose

console.log('\nskin palette');

/** A scene with one rigged quad whose joints the caller can pose. */
function skinnedScene({ jointPositions = [[0, 0, 0], [0, 0, 0]] } = {}) {
  const scene = new Scene({ capacity: 32 });
  const primitive = {
    indexCount: 6, materialId: 0,
    bounds: { min: [0, 0, 0], max: [1, 1, 0] },
    skinned: true,
    jointIndices: new Uint32Array(16),
    jointWeights: new Float32Array(16),
  };
  const asset = {
    nodes: [
      { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [1, 2], mesh: 0, skin: 0 },
      { position: jointPositions[0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: -1, skin: -1 },
      { position: jointPositions[1], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: -1, skin: -1 },
    ],
    meshes: [{ primitives: [primitive] }],
    // Bind pose: both joints at the origin, so the inverse bind matrices are
    // identity and a palette entry is just the joint's world matrix.
    skins: [{
      name: 'rig',
      joints: Uint32Array.from([1, 2]),
      inverseBind: Float32Array.from([
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
      ]),
    }],
    roots: [0],
    animations: [],
  };
  scene.add(asset);
  scene.update();
  return scene;
}

/** SkinPalette.update with no GPU: the matrix maths is what is under test. */
function paletteFor(scene) {
  const palette = Object.create(SkinPalette.prototype);
  palette.capacity = 256;
  palette.data = new Float32Array(256 * 16);
  palette.offsets = new Uint32Array(64);
  palette.jointCount = 0;
  palette.revision = 0;
  palette.buffer = {};
  palette.rhi = { queue: { writeBuffer() {} } };
  palette.update(scene);
  return palette;
}

test('in bind pose every joint matrix is identity', () => {
  // The whole of step 2 rests on this: a character in bind pose must come out
  // exactly where the unskinned mesh would. Any error in the multiply order,
  // the inverse bind, or the joint-to-entity mapping shows up here as a
  // matrix that is not identity, before any animation exists to confuse it.
  const palette = paletteFor(skinnedScene());
  assert.equal(palette.jointCount, 2);

  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (let j = 0; j < 2; j++) {
    for (let k = 0; k < 16; k++) {
      close(palette.data[j * 16 + k], identity[k], EPS, `joint ${j} element ${k}`);
    }
  }
});

test('moving a joint moves only its own matrix', () => {
  const palette = paletteFor(skinnedScene({ jointPositions: [[0, 0, 0], [5, 0, 0]] }));
  // Column-major: the translation is elements 12..14.
  vecClose(palette.data.subarray(12, 15), [0, 0, 0], EPS, 'joint 0 stayed');
  vecClose(palette.data.subarray(28, 31), [5, 0, 0], EPS, 'joint 1 moved');
});

test('the mesh node transform does not enter the palette', () => {
  // glTF 3.7.3.3: a skinned mesh's own node transform is ignored, because the
  // joints place it entirely. Applying it as well would move the character
  // twice, which looks like a doubled translation rather than an error.
  const scene = skinnedScene();
  const meshEntity = scene.renderableEntity[0];
  scene.transforms.setPosition(meshEntity, 100, 0, 0);
  scene.update();

  const palette = paletteFor(scene);
  // The joints are CHILDREN of the mesh node, so their world matrices do move
  // -- that is inheritance, not the mesh node being applied to the palette.
  vecClose(palette.data.subarray(12, 15), [100, 0, 0], EPS, 'joint inherited the parent');
  // And nothing doubled it.
  assert.ok(palette.data[12] < 150, `translation was applied twice: ${palette.data[12]}`);
});

test('two instances get separate palette slices', () => {
  // The reason paletteOffset is per-instance draw data rather than per batch:
  // two characters in different poses still share a pipeline and a draw call.
  const scene = skinnedScene();
  scene.add({
    nodes: [
      { position: [9, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [1], mesh: 0, skin: 0 },
      { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: -1, skin: -1 },
    ],
    meshes: [{ primitives: [{
      indexCount: 6, materialId: 0, bounds: { min: [0, 0, 0], max: [1, 1, 0] },
      skinned: true,
    jointIndices: new Uint32Array(16), jointWeights: new Float32Array(16),
    }] }],
    skins: [{ name: 'rig2', joints: Uint32Array.from([1]), inverseBind: Float32Array.from([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]) }],
    roots: [0], animations: [],
  });
  scene.update();

  const palette = paletteFor(scene);
  assert.equal(scene.skins.length, 2);
  assert.equal(palette.offsets[0], 0);
  assert.equal(palette.offsets[1], 2, 'the second skin starts after the first two joints');
  assert.equal(palette.jointCount, 3);
  vecClose(palette.data.subarray(2 * 16 + 12, 2 * 16 + 15), [9, 0, 0], EPS, 'second instance');
});

test('no shader template contains a stray backtick', () => {
  // This has bitten twice: a backtick in a WGSL comment terminates the
  // template literal, and the file then fails to parse with an error pointing
  // at whatever word followed it. Cheap to check, invisible to review.
  const shaders = {
    PBR_SHADER, OIT_RESOLVE_SHADER, HZB_SHADER, CLUSTER_SHADER, POST_SHADER,
  };
  for (const [name, source] of Object.entries(shaders)) {
    assert.ok(source && source.length > 0, `${name} is empty, so it was truncated`);
    assert.equal(source.includes('`'), false, `${name} contains a backtick`);
    assert.equal(source.includes('${'), false, `${name} has an uninterpolated placeholder`);
  }
});

// ------------------------------------------------------------ skinned bounds

console.log('\nskinned bounds');

/**
 * Where the skinning actually puts each vertex, computed independently.
 *
 * The reference the bounds are checked against. Deliberately the long way --
 * blend the matrices per vertex and transform -- so it shares no code with the
 * thing under test.
 */
function skinnedVertices(scene, skinIndex, positions, jointIndices, jointWeights, vertexCount) {
  const skin = scene.skins[skinIndex];
  const world = scene.transforms.world;
  const out = [];

  for (let v = 0; v < vertexCount; v++) {
    const p = v * 3;
    const g = v * 4;
    const m = new Float64Array(16);

    for (let k = 0; k < 4; k++) {
      const w = jointWeights[g + k];
      if (w <= 0) continue;
      const j = jointIndices[g + k];
      const joint = new Float32Array(16);
      mat4Multiply(joint, world, skin.inverseBind, 0, handleIndex(skin.joints[j]) * 16, j * 16);
      for (let e = 0; e < 16; e++) m[e] += joint[e] * w;
    }

    const x = positions[p], y = positions[p + 1], z = positions[p + 2];
    out.push([
      m[0] * x + m[4] * y + m[8] * z + m[12],
      m[1] * x + m[5] * y + m[9] * z + m[13],
      m[2] * x + m[6] * y + m[10] * z + m[14],
    ]);
  }
  return out;
}

/** A two-joint quad: bottom edge on joint 0, top edge on joint 1. */
function riggedScene() {
  const positions = Float32Array.from([-1, 0, 0, 1, 0, 0, -1, 2, 0, 1, 2, 0]);
  const jointIndices = Uint32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
  const jointWeights = Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

  const scene = new Scene({ capacity: 32 });
  scene.add({
    nodes: [
      { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [1, 2], mesh: 0, skin: 0 },
      { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: -1, skin: -1 },
      { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: -1, skin: -1 },
    ],
    meshes: [{ primitives: [{
      indexCount: 6, materialId: 0, skinned: true,
      bounds: { min: [-1, 0, 0], max: [1, 2, 0] },
      jointIndices, jointWeights,
    }] }],
    skins: [{
      name: 'rig',
      joints: Uint32Array.from([1, 2]),
      inverseBind: Float32Array.from([...identity, ...identity]),
      jointRadii: jointInfluenceRadii(
        positions, jointIndices, jointWeights, 4,
        Float32Array.from([...identity, ...identity]), 2,
      ),
    }],
    roots: [0], animations: [],
  });
  return { scene, positions, jointIndices, jointWeights };
}

function refreshBounds(scene) {
  scene.update();
  updateWorldBounds(
    scene.renderableCount, scene.localMin, scene.localMax, scene.worldMin, scene.worldMax,
    scene.transforms.world, scene.renderableMatrixSlot, null,
  );
  updateSkinBounds(scene.skins, scene.transforms.world);
  applySkinBounds(
    scene.renderableCount, scene.renderableSkin, scene.skins, scene.worldMin, scene.worldMax,
  );
}

/** Every skinned vertex must be inside the renderable's world box. */
function assertContains(scene, rig, what) {
  const verts = skinnedVertices(scene, 0, rig.positions, rig.jointIndices, rig.jointWeights, 4);
  const min = scene.worldMin.subarray(0, 3);
  const max = scene.worldMax.subarray(0, 3);
  for (const [i, v] of verts.entries()) {
    for (let a = 0; a < 3; a++) {
      assert.ok(v[a] >= min[a] - EPS && v[a] <= max[a] + EPS,
        `${what}: vertex ${i} axis ${a} is ${v[a].toFixed(3)}, outside [${min[a].toFixed(3)}, ${max[a].toFixed(3)}]`);
    }
  }
  return verts;
}

test('the box contains every vertex in bind pose', () => {
  const rig = riggedScene();
  refreshBounds(rig.scene);
  assertContains(rig.scene, rig, 'bind pose');
});

test('the box follows a joint, where a transformed static box would not', () => {
  // THE test for this step. The mesh node never moves, so the model matrix is
  // unchanged and updateWorldBounds writes the same bind-pose box it always
  // does. Only recomputing from the joints catches the arm going up.
  const rig = riggedScene();
  refreshBounds(rig.scene);
  const before = rig.scene.worldMax[1];

  // Raise joint 1 by 4. Nothing about the mesh node changes.
  const joint = rig.scene.skins[0].joints[1];
  rig.scene.transforms.setPosition(joint, 0, 4, 0);
  refreshBounds(rig.scene);

  const verts = assertContains(rig.scene, rig, 'joint raised');
  const highest = Math.max(...verts.map((v) => v[1]));
  assert.ok(highest > before,
    'the test pose must put a vertex outside the old box, or this proves nothing');
  assert.ok(rig.scene.worldMax[1] >= highest - EPS, 'and the box grew to hold it');
});

test('the box shrinks back, rather than only ever growing', () => {
  // An AABB that is re-derived can shrink; one that is re-transformed from its
  // own previous value inflates forever. This is the union of spheres, so it
  // tracks both ways.
  const rig = riggedScene();
  const joint = rig.scene.skins[0].joints[1];
  rig.scene.transforms.setPosition(joint, 0, 20, 0);
  refreshBounds(rig.scene);
  const tall = rig.scene.worldMax[1];

  rig.scene.transforms.setPosition(joint, 0, 0, 0);
  refreshBounds(rig.scene);
  assert.ok(rig.scene.worldMax[1] < tall, `box stayed at ${rig.scene.worldMax[1]}, was ${tall}`);
  assertContains(rig.scene, rig, 'returned to bind pose');
});

test('the box holds up through rotation and a moved root', () => {
  const rig = riggedScene();
  const [hip, chest] = rig.scene.skins[0].joints;
  const root = rig.scene.renderableEntity[0];

  for (const [angle, y, rx] of [[0.3, 1, 0], [1.2, -2, 3], [2.9, 0.5, -4]]) {
    const q = quatCreate();
    quatSetAxisAngle(q, [0, 0, 1], angle);
    rig.scene.transforms.setRotation(chest, q);
    rig.scene.transforms.setPosition(chest, rx, y, 0);
    rig.scene.transforms.setPosition(hip, 0, y * 0.5, 0);
    rig.scene.transforms.setPosition(root, rx * 0.5, y, 0);
    refreshBounds(rig.scene);
    assertContains(rig.scene, rig, `angle ${angle}`);
  }
});

test('picking sees the posed box, not the authored one', () => {
  // raycast refreshes bounds itself, so a click must land on where a character
  // is standing rather than where it was rigged.
  const rig = riggedScene();
  const joint = rig.scene.skins[0].joints[1];
  rig.scene.transforms.setPosition(joint, 0, 30, 0);

  const hit = rig.scene.raycast(vec3Create(0, 31, 10), vec3Create(0, 0, -1));
  assert.ok(hit, 'a ray through the raised geometry must hit it');
});
test('a skinned mesh is picked at its box, not at bind-pose triangles', () => {
  // Combining two features made a gap: step 3 gave skinned renderables a POSED
  // box, while the narrow phase still reaches its triangles by inverting the
  // mesh node's matrix -- which skinned vertices do not follow. Left alone
  // that is not approximate, it misses, so a posed character with retained
  // geometry became unpickable while its box said it was right there.
  const rig = riggedScene();
  // Retain geometry, which is what turns the narrow phase on.
  rig.scene.renderablePrimitive[0].positions = rig.positions;
  rig.scene.renderablePrimitive[0].indices = Uint32Array.from([0, 1, 2, 2, 1, 3]);

  const joint = rig.scene.skins[0].joints[1];
  rig.scene.transforms.setPosition(joint, 0, 30, 0);

  const hit = rig.scene.raycast(vec3Create(0, 31, 10), vec3Create(0, 0, -1));
  assert.ok(hit, 'the posed geometry must still be pickable with retainGeometry on');
});

test('an unskinned mesh still gets the triangle test', () => {
  // The negative control: the skip must key on skinning, not on having
  // retained geometry at all, or it would silently disable the feature.
  const scene = new Scene({ capacity: 8 });
  const entity = scene.entities.alloc();
  scene.transforms.add(entity, { position: [0, 0, -5] });
  scene._addRenderable(entity, {
    indexCount: 3, materialId: 0,
    bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
    positions: new Float32Array([-0.2, -0.2, 0, 0.2, -0.2, 0, 0, 0.2, 0]),
    indices: new Uint32Array([0, 1, 2]),
  });

  assert.ok(scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 0, -1)), 'through the triangle');
  assert.equal(scene.raycast(vec3Create(0.4, 0.4, 0), vec3Create(0, 0, -1)), null,
    'the empty corner of the box still misses');
});


console.log(`\n${passed} checks passed\n`);
