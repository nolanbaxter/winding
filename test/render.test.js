// Sort keys, draw lists and world bounds. Run: node test/render.test.js

import { EXTENSION_TEXTURES } from '../src/scene/gltf/images.js';
import { SHEEN_ALBEDO, SHEEN_TABLE_SIZE, sheenAlbedo, sheenTablePoint } from '../src/render/sheen.js';
import { fogCoefficients, packFog, FOG_WGSL, VISIBILITY_CONTRAST } from '../src/render/fog.js';
import { packProbes, FACE_CAMERAS, PROBE_FLOATS } from '../src/render/probes.js';
import { distanceField, layoutText, SPREAD } from '../src/render/text.js';
import { parseCube, whiteBalanceMatrix, planckianXY, packGrading } from '../src/render/grading.js';
import { lensCoefficients } from '../src/render/dof.js';
import { textRecord } from '../src/scene/scene.js';
import { probeRecord } from '../src/scene/scene.js';
import { parseHDR, halfBits, toHalfRGBA } from '../src/render/hdr.js';
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
import { GpuDriven, DRAW_DATA_BYTES, CULL_PHASES } from '../src/render/gpudriven.js';
import { SkinPalette } from '../src/render/skin.js';
import { MorphStore } from '../src/render/morph.js';
import { PipelineCache } from '../src/rhi/pipeline.js';
import { createBuffer, storageCapacity } from '../src/rhi/buffer.js';
import { grownCapacity } from '../src/core/grow.js';
import { pbrShader } from '../src/render/shaders/pbr.js';

/** The forward shader with every extension texture bound, as a roomy device builds it. */
const PBR_SHADER = pbrShader(EXTENSION_TEXTURES.length);
import { SHADOW_SHADER } from '../src/render/shadows.js';
import { OIT_RESOLVE_SHADER } from '../src/render/shaders/oit.js';
import { HZB_SHADER } from '../src/render/hzb.js';
import { CLUSTER_SHADER } from '../src/render/clustered.js';
import { POST_SHADER } from '../src/render/post.js';
import { Renderer } from '../src/render/renderer.js';
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
  variantKey, variantPipelineState, ALPHA_OPAQUE, ALPHA_MASK, ALPHA_BLEND, MaterialRegistry,
} from '../src/render/material.js';
import { DEFAULT_MATERIAL } from '../src/scene/gltf/parse.js';

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

test('an orthographic box culls by its sides and keeps what is past far', () => {
  // This used to be "a finite projection is rejected", and an orthographic
  // camera is exactly a finite projection. Its far plane is real, but leaving
  // it out only ever KEEPS more: the rasteriser clips what is past it. So the
  // sides must still cull, and far must not.
  const camera = new Camera({ fovY: Math.PI / 2, near: 0.1, orthographic: true, far: 20 });
  camera.position.set([0, 0, 10]);
  camera.update(1);                                     // half extent: 10 * tan(45) = 10
  const frustum = frustumFromViewProjection(frustumCreate(), camera.viewProjection);

  for (let i = 0; i < FRUSTUM_PLANE_COUNT * 4; i++) {
    assert.ok(Number.isFinite(frustum[i]), `plane component ${i} is finite`);
  }
  const keeps = ({ min, max }) => frustumTestAABB(frustum, min, max);
  assert.equal(keeps(boxAt(9, 0, -5)), true, 'inside, near the right edge');
  assert.equal(keeps(boxAt(12, 0, -5)), false, 'past the right edge');
  assert.equal(keeps(boxAt(0, 0, 11)), false, 'behind the camera');
  assert.equal(keeps(boxAt(0, 0, -50)), true, 'past far is kept, not culled');
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
globalThis.GPUShaderStage ??= { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 };
globalThis.GPUTextureUsage ??= {
  COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
};
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
  gpu.indirectData = new Uint32Array(gpu.batchCapacity * CULL_PHASES * 5);
  gpu.batchStaging = new ArrayBuffer(gpu.alignment * (gpu.batchCapacity * CULL_PHASES + 2));
  gpu.batchPrimitive = [];
  gpu.blendedCasters = [];
  gpu.buffersRevision = 0;
  gpu.stats = {};
  for (const k of ['itemBatchBuffer', 'batchFirstBuffer', 'batchOrderBuffer',
    'batchBuffer', 'visibleFlagsBuffer', 'indirectBuffer',
    'drawDataBuffer', 'boundsBuffer', 'visibleBuffer']) gpu[k] = { destroy() {} };
  gpu._allocateDrawData(gpu.capacity);
  gpu.boundsData = new Float32Array(gpu.capacity * 12);
  return gpu;
}

test('growing draw data replaces both views of it', () => {
  // The bug: _grow allocated a new drawData and left drawDataU32 viewing the
  // OLD buffer. Half the struct is u32 -- paletteOffset and the three morph
  // words -- so those writes went into a detached array. Nothing throws: an
  // index inside the old length writes where nothing is uploaded from, and an
  // index past it writes nowhere at all. Every u32 field simply read as zero
  // on the GPU, which is a second skinned character wearing the first one's
  // pose and a morphed mesh that never moves.
  const gpu = growableGpu([]);
  const before = gpu.capacity;
  gpu._grow(before + 1);
  assert.ok(gpu.capacity > before, 'the test needs an actual grow');

  assert.equal(gpu.drawDataU32.buffer, gpu.drawData.buffer,
    'the integer view must address the array that gets uploaded');
  assert.equal(gpu.drawDataU32.length, gpu.drawData.length);

  // And behaviourally: a u32 write at the far end lands in the uploaded array.
  const word = (gpu.capacity - 1) * (DRAW_DATA_BYTES / 4) + 28;
  gpu.drawDataU32[word] = 0x3f800000;                  // 1.0f, seen as bits
  close(gpu.drawData[word], 1, EPS, 'the write reached the float view');
});

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
  gpu.blendedCasters = [];
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

test('the CPU level-of-detail test matches the cull shader: coverage is radius * scale / w', () => {
  const gpu = growableGpu({ isTransparent: () => true });
  // A sphere of radius 2 at the origin, drawn between coverage 0.25 and 0.5.
  // The camera sits `distance` along +z looking down -z, so the sphere's clip
  // w -- its view depth -- is that distance: only w's row matters here.
  const b = gpu.boundsData;
  b.set([0, 0, 0, 0.25], 0);
  b.set([0, 0, 0, 0.5], 4);
  b.set([0, 0, 0, 2], 8);
  const at = (distance, scale = 1) => {
    const viewProjection = new Float32Array(16);
    viewProjection[11] = -1;
    viewProjection[15] = distance;
    return gpu.lodSelected(0, viewProjection, scale);
  };
  assert.equal(at(3), false, 'coverage 0.67: a finer level shows');
  assert.equal(at(5), true, 'coverage 0.4');
  assert.equal(at(4), false, 'coverage 0.5 exactly: the range is [low, high), so the finer level has it');
  assert.equal(at(8), true, 'coverage 0.25 exactly: this level has it');
  assert.equal(at(9), false, 'coverage 0.22: a coarser level');
  assert.equal(at(-1), false, 'behind the eye counts as filling the screen');
  b[11] = 0;
  assert.equal(at(3), true, 'in no group: always');
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

test('every shader that reads DrawData agrees on its layout', () => {
  // The bug this exists for: the shadow shader declared DrawData without
  // paletteOffset, so WGSL sized it at 112 while the CPU wrote a 128-byte
  // stride. Instance 0 landed correctly and every one after it was misaligned
  // -- shadows in the wrong places, no error anywhere, and a GPU suite that
  // only checks for device errors cannot see it.
  const members = (source) => {
    const body = source.match(/struct DrawData \{([\s\S]*?)\n\};/);
    assert.ok(body, 'a shader reading DrawData must declare it');
    return body[1]
      .split('\n')
      .map((line) => line.replace(/\/\/.*$/, '').trim())
      .filter((line) => line.length > 0)
      .map((line) => line.split(':').map((part) => part.trim()).join(':'));
  };

  const forward = members(PBR_SHADER);
  const shadow = members(SHADOW_SHADER);
  assert.deepEqual(shadow, forward,
    'the shadow pass reads the same buffer, so it must see the same struct');

  // And the struct the two agree on must be the stride the CPU writes.
  // WGSL: mat4x4 is 64 with align 16; mat3x3 is 48 with align 16; u32 is 4.
  const SIZES = { 'mat4x4<f32>': [64, 16], 'mat3x3<f32>': [48, 16], u32: [4, 4] };
  let end = 0;
  let structAlign = 1;
  for (const member of forward) {
    const type = member.split(':')[1].replace(/,$/, '');
    const [size, align] = SIZES[type] ?? [0, 0];
    assert.ok(size > 0, `unknown member type ${type}; teach this check about it`);
    end = Math.ceil(end / align) * align + size;
    structAlign = Math.max(structAlign, align);
  }
  assert.equal(Math.ceil(end / structAlign) * structAlign, DRAW_DATA_BYTES,
    'the declared struct does not match the stride the CPU writes');
});

test('the sheen albedo table is the integral of the sheen BRDF', () => {
  // Every entry integrated again. On a mismatch the whole fresh table is
  // printed, ready to paste into render/sheen.js.
  const fresh = [];
  for (let y = 0; y < SHEEN_TABLE_SIZE; y++) {
    for (let x = 0; x < SHEEN_TABLE_SIZE; x++) fresh.push(sheenAlbedo(...sheenTablePoint(x, y)));
  }
  const worst = Math.max(...fresh.map((e, i) => Math.abs(e - SHEEN_ALBEDO[i])));
  if (!(worst <= 6e-5)) {
    const rows = [];
    for (let y = 0; y < SHEEN_TABLE_SIZE; y++) {
      rows.push(`  ${fresh.slice(y * SHEEN_TABLE_SIZE, (y + 1) * SHEEN_TABLE_SIZE).map((e) => e.toFixed(4)).join(', ')},`);
    }
    console.log(rows.join('\n'));
  }
  assert.ok(worst <= 6e-5, `an entry is off by ${worst}; the fresh table is printed above`);
  // And 128 steps a side has converged: four times as many moves nothing
  // past the table's own precision, at the most peaked entry it has.
  assert.ok(Math.abs(sheenAlbedo(0.09375, 0.2) - sheenAlbedo(0.09375, 0.2, 512)) < 1e-3);
});

test('fog visibility is where contrast falls to 2%, and bad options are named', () => {
  const { extinction, inverseScaleHeight } = fogCoefficients({ visibility: 150 });
  assert.ok(Math.abs(Math.exp(-extinction * 150) - VISIBILITY_CONTRAST) < 1e-12);
  assert.equal(inverseScaleHeight, 0, 'no scale height: the same density everywhere');
  assert.equal(fogCoefficients({ visibility: 150, scaleHeight: 20 }).inverseScaleHeight, 1 / 20);
  for (const [options, why] of [
    [{}, /visibility/], [{ visibility: -1 }, /visibility/], [{ visibility: Infinity }, /visibility/],
    [{ visibility: 10, height: NaN }, /height/], [{ visibility: 10, scaleHeight: 0 }, /scaleHeight/],
    [{ visibility: 10, albedo: [1, 1] }, /albedo/], [{ visibility: 10, albedo: [1, -1, 1] }, /albedo/],
  ]) assert.throws(() => fogCoefficients(options), why);
});

test('fog scatters each directional light isotropically, times its albedo', () => {
  // Two lights, eight floats each, colour times intensity at 4..6.
  const directionals = Float32Array.of(0, -1, 0, 0, 4 * Math.PI, 0, 0, 0, 0, -1, 0, 0, 0, 2 * Math.PI, 0, 0);
  const out = new Float32Array(16).fill(9);
  packFog(out, 2, { visibility: 100, albedo: [0.5, 1, 1] }, directionals, 2, 8);
  assert.deepEqual([...out.subarray(10, 13)], [0.5, 0.5, 0], 'albedo x colour / 4 pi, summed');
  assert.equal(out[1], 9, 'nothing written outside its twelve');
  packFog(out, 2, null, directionals, 2, 8);
  assert.deepEqual([...out.subarray(2, 14)], new Array(12).fill(0), 'no fog: zeros, and extinction 0 turns it off');
});

test('a reflection probe is checked, centred by default, and packed smallest box first', () => {
  const room = probeRecord({ min: [-5, 0, -5], max: [5, 4, 5] });
  assert.deepEqual([...room.position], [0, 2, 0], 'the box centre');
  assert.equal(room.blend, 0);
  for (const [options, why] of [
    [{ min: [0, 0, 0], max: [1, 1] }, /max must be three/],
    [{ min: [0, 0, 0], max: [1, 0, 1] }, /above min on every axis/],
    [{ min: [0, 0, 0], max: [1, 1, 1], position: [2, 0, 0] }, /inside the box/],
    [{ min: [0, 0, 0], max: [1, 1, 1], blend: -1 }, /blend/],
  ]) assert.throws(() => probeRecord(options), why);

  const closet = probeRecord({ min: [0, 0, 0], max: [1, 2, 1], blend: 0.25 });
  const hall = probeRecord({ min: [-20, 0, -20], max: [20, 5, 20] });
  const pending = probeRecord({ min: [0, 0, 0], max: [1, 1, 1] });
  room.captured = closet.captured = hall.captured = true;
  const layers = new Map([[room, 0], [closet, 1], [hall, 2], [pending, 3]]);
  const { data, count } = packProbes([hall, room, pending, closet], layers);
  assert.equal(count, 3, 'a probe never captured has nothing to show');
  assert.deepEqual([0, 1, 2].map((k) => data[k * PROBE_FLOATS + 7]), [1, 0, 2], 'closet, room, hall');
  assert.equal(data[3], 0.25, 'blend in min.w');
});

test('each probe face camera is right-handed, so the copy into the cube is one mirror across u', () => {
  // cubeDirection(face, u, v) is what the shader samples. With right = forward
  // x up, the camera image mirrored across u shows forward - u' right - v' up
  // at (u', v') in [-1, 1] -- which must be that direction exactly.
  const cube = [
    (u, v) => [1, -v, -u], (u, v) => [-1, -v, u], (u, v) => [u, 1, v],
    (u, v) => [u, -1, -v], (u, v) => [u, -v, 1], (u, v) => [-u, -v, -1],
  ];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  FACE_CAMERAS.forEach(({ forward, up }, face) => {
    const right = cross(forward, up);
    for (const [u, v] of [[0.5, -0.25], [-1, 1]]) {
      const shown = forward.map((f, a) => f - u * right[a] - v * up[a]);
      assert.deepEqual(shown.map((x) => x + 0), cube[face](u, v).map((x) => x + 0), `face ${face} at ${u}, ${v}`);
    }
  });
});

test('a glyph distance field is the exact distance to its edge, 0.5 on it', () => {
  // A disc of radius 6 in a 24 x 24 raster: texels well inside and outside
  // it hold their true distance to the circle, scaled by SPREAD.
  const W = 24;
  const coverage = new Float32Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) coverage[y * W + x] = Math.hypot(x - 12, y - 12) < 6 ? 1 : 0;
  const field = distanceField(coverage, W, W);
  const at = (x, y) => field[y * W + x] / 255;
  const expected = (x, y) => Math.min(Math.max(0.5 - (Math.hypot(x - 12, y - 12) - 6) / (2 * SPREAD), 0), 1);
  for (const [x, y] of [[12, 12], [12, 9], [12, 16], [12, 20], [4, 12], [15, 15]]) {
    assert.ok(Math.abs(at(x, y) - expected(x, y)) < 0.08, `texel ${x}, ${y}: ${at(x, y).toFixed(3)} against ${expected(x, y).toFixed(3)}`);
  }
  // A half-covered texel sits on the edge.
  coverage[12 * W + 18] = 0.5;
  assert.ok(Math.abs(distanceField(coverage, W, W)[12 * W + 18] / 255 - 0.5) < 0.01);
});

test('text lays out by advance, aligns each line, and puts its anchor at the origin', () => {
  const glyph = (advance) => ({ advance, left: 0, width: advance, height: 1, descent: 0 });
  const metrics = { ascent: 0.8, descent: 0.2, glyphs: new Map([['a', glyph(0.5)], ['b', glyph(1)], [' ', { advance: 0.25, left: 0, width: 0, height: 0, descent: 0 }]]) };
  const one = layoutText('ab a', metrics, { anchor: [0, 0] });
  assert.deepEqual(one.map((g) => g.char), ['a', 'b', 'a'], 'a space advances and draws nothing');
  assert.deepEqual(one.map((g) => g.x), [0, 0.5, 1.75]);
  assert.deepEqual(one.map((g) => +g.y.toFixed(6)), [0.2, 0.2, 0.2], 'on the baseline, above the descent, from the bottom left');
  const two = layoutText('ab\na', metrics, { align: 'right', anchor: [1, 1] });
  assert.equal(two[2].x, -0.5, 'the short line pushed right, and the block anchored at its top right');
  assert.ok(two[2].y < two[0].y, 'the second line below the first');
  const centred = layoutText('b', metrics);
  assert.deepEqual([centred[0].x, +centred[0].y.toFixed(6)], [-0.5, -0.3], 'centred by default');
  assert.throws(() => layoutText('a', metrics, { align: 'justify' }), /align/);
});

test('a text takes a loadFont font, rasterises what it uses, and names a bad option', () => {
  const asked = [];
  const font = { metrics: { ascent: 0.8, descent: 0.2, glyphs: new Map() }, ensure: (t) => asked.push(t) };
  const record = textRecord({ font, text: 'hi', size: 2 });
  assert.deepEqual(asked, ['hi']);
  assert.deepEqual([record.facing, record.pixels, [...record.color].join()], ['camera', false, '1,1,1,1']);
  for (const [options, why] of [
    [{ text: 'x', size: 1 }, /font must be/], [{ font, text: 'x' }, /size must be positive/],
    [{ font, text: 'x', size: 1, facing: 'down' }, /facing/], [{ font, text: 'x', size: 1, anchor: [0] }, /anchor/],
  ]) assert.throws(() => textRecord(options), why);
});

test('a .cube LUT parses red-fastest, keeps its domain, and a malformed one is named', () => {
  const rows = [];
  for (let b = 0; b < 2; b++) for (let g = 0; g < 2; g++) for (let r = 0; r < 2; r++) rows.push(`${r} ${g} ${b}`);
  const cube = parseCube(['# a comment', 'TITLE "identity"', 'LUT_3D_SIZE 2', 'DOMAIN_MAX 2 2 2', ...rows].join('\n'));
  assert.equal(cube.size, 2);
  assert.deepEqual([...cube.data.subarray(3, 6)], [1, 0, 0], 'the second entry is red: red varies fastest');
  assert.deepEqual(cube.domainMax, [2, 2, 2]);
  assert.throws(() => parseCube('LUT_3D_SIZE 2\n0 0 0'), /has 1/);
  assert.throws(() => parseCube('LUT_1D_SIZE 4'), /1D/);
  assert.throws(() => parseCube('LUT_3D_SIZE 2\nnot a row'), /cannot read/);
});

test('white balance makes exactly the named light white, and D65 is left alone', () => {
  const lightRGB = (T) => {
    const [x, y] = planckianXY(T);
    const X = x / y;
    const Z = (1 - x - y) / y;
    return [3.2404542 * X - 1.5371385 - 0.4985314 * Z, -0.9692660 * X + 1.8760108 + 0.0415560 * Z, 0.0556434 * X - 0.2040259 + 1.0572252 * Z];
  };
  for (const T of [2000, 3200, 5000, 10000]) {
    const m = whiteBalanceMatrix(T);
    const out = m.map((row) => row[0] * lightRGB(T)[0] + row[1] * lightRGB(T)[1] + row[2] * lightRGB(T)[2]);
    assert.ok(Math.max(...out) - Math.min(...out) < 1e-3 * Math.max(...out), `${T} K comes out white: ${out}`);
  }
  // 6504 K is on the Planckian locus; D65 is a hair off it. Near, not exact.
  const d65 = whiteBalanceMatrix(6504);
  d65.forEach((row, i) => row.forEach((v, j) => assert.ok(Math.abs(v - (i === j ? 1 : 0)) < 0.04)));
  assert.throws(() => whiteBalanceMatrix(1000), /1667 K to 25000 K/);
  const packed = packGrading(new Float32Array(20), null);
  assert.deepEqual([...packed.subarray(0, 12)], [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0], 'no grading: the identity, and no LUT');
  assert.throws(() => packGrading(new Float32Array(20), { contrast: 0 }), /contrast/);
});

test('the lens: the circle of confusion from focal length, f-stop and focus', () => {
  // 50 mm on full frame is a vertical field of view of 2 atan(12 / 50).
  const fov = 2 * Math.atan(12 / 50);
  const { scale, largest } = lensCoefficients({ focusDistance: 2, fStop: 2 }, fov, 1000);
  // A = 25 mm; c at infinity = A f / (S - f) = 0.025 * 0.05 / 1.95 m on the sensor.
  const expected = (0.025 * 0.05 / 1.95) / 0.024 * 1000;
  assert.ok(Math.abs(scale - expected) < 1e-9);
  assert.equal(largest, Math.abs(scale));
  assert.throws(() => lensCoefficients({ focusDistance: 0.01, fStop: 2 }, fov, 1000), /past the lens/);
  assert.throws(() => lensCoefficients({ focusDistance: 2, fStop: 0 }, fov, 1000), /fStop/);
});

test('no shader template contains a stray backtick', () => {
  // This has bitten twice: a backtick in a WGSL comment terminates the
  // template literal, and the file then fails to parse with an error pointing
  // at whatever word followed it. Cheap to check, invisible to review.
  const shaders = {
    PBR_SHADER, OIT_RESOLVE_SHADER, HZB_SHADER, CLUSTER_SHADER, POST_SHADER, FOG_WGSL,
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
test('a skinned mesh is picked at its posed triangles, not its box or its bind pose', () => {
  // The bind-pose triangles reached through the mesh node's matrix miss a
  // posed character outright, and its box answers for empty space. The
  // triangles are skinned on the click instead, the way the shader skins them.
  const rig = riggedScene();
  // Retain geometry -- the fixture already carries the joint influences.
  rig.scene.renderablePrimitive[0].positions = rig.positions;
  rig.scene.renderablePrimitive[0].indices = Uint32Array.from([0, 1, 2, 2, 1, 3]);

  // Shear the top edge far up and to the side: the quad becomes a thin
  // parallelogram, and its box is mostly empty.
  const joint = rig.scene.skins[0].joints[1];
  rig.scene.transforms.setPosition(joint, 10, 30, 0);

  const hit = rig.scene.raycast(vec3Create(9.5, 31, 10), vec3Create(0, 0, -1));
  assert.ok(hit, 'a ray through the posed geometry hits it');
  close(hit.distance, 10, EPS, 'at the triangle, not the front of a box');
  assert.equal(rig.scene.raycast(vec3Create(-0.5, 31, 10), vec3Create(0, 0, -1)), null,
    'and a ray through the empty corner of its box misses');
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


// ---------------------------------------- material defaults for absent maps

console.log('\ndefault textures for materials with no maps');

/**
 * A device stub that labels every view with the texture it came from, so which
 * default a binding chose is observable without a GPU.
 */
function labellingRhi() {
  const bindGroups = [];
  const device = {
    createTexture: ({ label }) => ({ label, createView: () => ({ label }), destroy() {} }),
    createSampler: () => ({ sampler: true }),
    createBuffer: () => ({ destroy() {} }),
    createBindGroupLayout: () => ({}),
    createBindGroup: (descriptor) => { bindGroups.push(descriptor); return descriptor; },
  };
  return {
    rhi: { device, queue: { writeTexture() {}, writeBuffer() {} }, limits: { minUniformBufferOffsetAlignment: 256, maxSampledTexturesPerShaderStage: 16 } },
    bindGroups,
  };
}

test('only a material the extended shader draws binds the extension slots', () => {
  const { rhi, bindGroups } = labellingRhi();
  const materials = new MaterialRegistry(rhi, { capacity: 4 });
  const plain = materials.register({ ...DEFAULT_MATERIAL, name: 'plain' });
  assert.equal(materials.shadingGroup(plain), materials.bindGroup(plain), 'one group, the core one');
  assert.equal(bindGroups.at(-1).entries.length, 7, 'uniform, five maps and the sampler');
  const coat = materials.register({ ...DEFAULT_MATERIAL, name: 'coat', clearcoat: 1 });
  assert.notEqual(materials.shadingGroup(coat), materials.bindGroup(coat), 'its own pipelines bind a second group');
  assert.equal(bindGroups.at(-1).entries.length, 7 + materials.extensionSlots);
  assert.equal(bindGroups.at(-2).entries.length, 7, 'and the shadow pass still gets the core one');
});

test('a material with no maps binds white everywhere a factor scales it', () => {
  // The bug: emissive fell back to a BLACK 1x1, and the shader multiplies that
  // by emissiveFactor. So a material that asked to glow rendered dark, with
  // nothing reported anywhere. glTF is explicit that an absent texture reads
  // as 1.0 on every channel -- which is the whole premise of this design,
  // since it is what lets the factors in the uniform do the scaling instead of
  // a shader variant per combination of present maps.
  //
  // Emissive WITHOUT a texture is how a simple glowing object is authored, so
  // this was the common case. Nothing caught it because every asset in the
  // repository ships an emissive map.
  const { rhi, bindGroups } = labellingRhi();
  const materials = new MaterialRegistry(rhi, { capacity: 4 });

  const id = materials.register({
    ...DEFAULT_MATERIAL,
    name: 'glowing',
    emissive: Float32Array.from([4, 2, 0]),
  });
  materials.bindGroup(id);

  const entries = bindGroups.at(-1).entries;
  const slot = (binding) => entries.find((e) => e.binding === binding).resource.label;

  assert.equal(slot(1), 'default-white', 'base colour');
  assert.equal(slot(3), 'default-orm', 'occlusion/roughness/metallic packs to 1,1,1');
  assert.equal(slot(4), 'default-white', 'occlusion');
  assert.equal(slot(5), 'default-white', 'emissive must not multiply its own factor away');
  assert.equal(slot(2), 'default-normal', 'and the normal map stays flat, not white');
});

// --------------------------------------------------- blended draw runs

console.log('\nblended draw runs');

/**
 * Just enough renderer for _encodeTransparent: a sorted list of renderables,
 * each a (primitive, material, skinned, mirrored), and a pass that logs calls.
 */
function blendedFrame(items) {
  const primitives = {};
  const primitive = (name) => primitives[name] ??= {
    name, indexCount: 36, vertexBuffer: `vb:${name}`, indexBuffer: `ib:${name}`, skinBuffer: `sb:${name}`,
  };
  const renderer = Object.create(Renderer.prototype);
  renderer._frameScene = {
    renderableMaterial: items.map((it) => it.material),
    renderablePrimitive: items.map((it) => primitive(it.mesh)),
    renderableSkin: items.map((it) => (it.skinned ? 0 : -1)),
  };
  renderer.gpu = {
    opaqueCount: 100,
    itemMirrored: items.map((it) => (it.mirrored ? 1 : 0)),
    transparentBatchOffset: () => 0,
  };
  // Already sorted: payload k is renderable k.
  renderer.transparentList = { count: items.length, payloads: items.map((_, k) => k) };
  renderer.materials = { variants: items.map(() => 0), shadingGroup: (id) => `material:${id}`, isTransmissive: () => false };
  renderer.pipelines = { get: (v) => `pipeline:${v}` };
  renderer.drawBindGroup = 'draws';
  renderer.stats = { transparentDraws: 0 };

  const calls = [];
  const log = (name) => (...args) => calls.push([name, ...args]);
  const pass = {
    setPipeline: log('setPipeline'), setBindGroup: log('setBindGroup'),
    setVertexBuffer: log('setVertexBuffer'), setIndexBuffer: log('setIndexBuffer'),
    drawIndexed: log('drawIndexed'),
  };
  renderer._encodeTransparent(pass, { get: (variant) => variant });
  const draws = calls.filter((c) => c[0] === 'drawIndexed')
    .map(([, , instances, , , first]) => ({ first: first - 100, instances }));
  return { calls, draws, stats: renderer.stats };
}

const glass = { mesh: 'pane', material: 2 };

test('neighbours that draw alike become one instanced call', () => {
  const { draws, stats } = blendedFrame([glass, glass, glass, glass]);
  assert.deepEqual(draws, [{ first: 0, instances: 4 }], 'four panes, one call, from the first slot');
  assert.equal(stats.transparentDraws, 1);
});

test('the sorted order is kept: runs never reach past something different', () => {
  // pane pane BOTTLE pane: the last pane is behind the bottle, so it cannot
  // join the first run without being drawn before the bottle.
  const bottle = { mesh: 'bottle', material: 3 };
  const { draws } = blendedFrame([glass, glass, bottle, glass]);
  assert.deepEqual(draws, [
    { first: 0, instances: 2 },
    { first: 2, instances: 1 },
    { first: 3, instances: 1 },
  ]);
});

test('every slot is drawn exactly once, in order', () => {
  const kinds = [glass, { mesh: 'pane', material: 5 }, { mesh: 'leaf', material: 2 }];
  const items = Array.from({ length: 40 }, (_, k) => kinds[(k * 7) % 5 % 3]);
  const { draws } = blendedFrame(items);
  const slots = draws.flatMap((d) => Array.from({ length: d.instances }, (_, n) => d.first + n));
  assert.deepEqual(slots, items.map((_, k) => k));
});

test('a different material, skinning or mirroring splits a run', () => {
  const { draws } = blendedFrame([
    glass,
    { ...glass, material: 9 },
    { ...glass, skinned: true },
    { ...glass, mirrored: true },
  ]);
  assert.equal(draws.length, 4);
});

test('buffers are bound when the mesh or its skinning changes, not per object', () => {
  const { calls } = blendedFrame([glass, glass, { ...glass, material: 9 }, { ...glass, skinned: true }]);
  const vertexBinds = calls.filter((c) => c[0] === 'setVertexBuffer').map((c) => [c[1], c[2]]);
  assert.deepEqual(vertexBinds, [
    [0, 'vb:pane'],                       // first run
    // the material-9 run reuses them: same mesh, same skinning
    [0, 'vb:pane'], [1, 'sb:pane'],       // skinned needs its skin buffer, so it rebinds
  ]);
});

console.log('\nmorph arena');

test('freed morph ranges are reused, merged, and shrink the arena from the end', () => {
  const rhi = {
    device: {
      createBuffer: () => ({ destroy() {} }),
      createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish() {} }),
    },
    queue: { writeBuffer() {}, submit() {} },
  };
  const store = new MorphStore(rhi, { deltaCapacity: 64 });
  const floats = (n) => new Float32Array(n);

  const a = store.allocate(floats(10));
  const b = store.allocate(floats(10));
  const c = store.allocate(floats(10));
  assert.deepEqual([a, b, c], [0, 10, 20]);

  // The same asset reloaded over and over used to append every time.
  for (let i = 0; i < 100; i++) {
    store.free(b, 10);
    assert.equal(store.allocate(floats(10)), b, `reload ${i} landed in the hole`);
  }
  assert.equal(store.deltaCount, 30, 'and the arena did not grow');

  store.free(a, 10);
  store.free(b, 10);
  assert.equal(store.allocate(floats(20)), 0, 'neighbouring holes merge into one');

  store.free(0, 20);
  store.free(c, 10);
  assert.equal(store.deltaCount, 0, 'freeing the tail gives back every hole touching it');
  assert.equal(store._holes.length, 0);
});

console.log('\ndevice limits');

test('growth stops at what the device can hold, and says so', () => {
  assert.equal(grownCapacity(1024, 1500), 2048, 'no ceiling: plain doubling');
  assert.equal(grownCapacity(1024, 1500, 1800), 1800, 'a ceiling below the double is taken instead');
  assert.throws(() => grownCapacity(1024, 1900, 1800, 'lights'), /1900 lights is past the 1800/);
});

test('a storage array holds what its binding AND its buffer allow, whichever is less', () => {
  const rhi = { limits: { maxStorageBufferBindingSize: 128 << 20, maxBufferSize: 256 << 20 } };
  assert.equal(storageCapacity(rhi, 128), (128 << 20) / 128, 'the binding limit decides');
  rhi.limits.maxBufferSize = 64 << 20;
  assert.equal(storageCapacity(rhi, 128), (64 << 20) / 128, 'and the buffer limit when it is lower');
});

test('a buffer past the device limit throws, rather than returning an invalid one', () => {
  let created = 0;
  const rhi = {
    limits: { maxBufferSize: 1024 },
    device: { createBuffer: () => { created++; return {}; } },
  };
  assert.throws(
    () => createBuffer(rhi, { label: 'mesh.vertices', data: new Float32Array(300), usage: 0 }),
    /mesh.vertices is 1200 bytes, past this device's 1024/,
  );
  assert.equal(created, 0, 'nothing was asked of the device');
});

console.log('\npipeline warming');

await (async () => {
  // Every wanted variant reaches warm(), known or not: one another load is
  // still compiling has to be waited on, and only warm() knows which.
  const handed = [];
  const renderer = Object.assign(Object.create(Renderer.prototype), {
    _variantSets: new Map([[0, { forward: new Map(), oit: new Map(), ready: true }]]),
    pipelines: { warm: async (descs) => handed.push(descs.length) },
    oit: true, pipelineLayout: {}, shader: {},
  });
  const BLEND = 2;
  await renderer.ensureVariants([BLEND]);
  await renderer.ensureVariants([BLEND]);
  assert.deepEqual(handed, [4, 4, 4, 4], 'forward and OIT, both times');
  passed++;
  console.log('  ok  a second load hands warm() the variants the first already asked for');
})();

await (async () => {
  // Probe-reading pipelines only once probes exist, and then for everything:
  // what was loaded before, and what is loaded after.
  const renderer = Object.assign(Object.create(Renderer.prototype), {
    _variantSets: new Map([[0, { forward: new Map(), oit: new Map(), ready: true }]]),
    pipelines: { warm: async () => {} },
    oit: false, pipelineLayout: {}, shader: {}, ao: null,
  });
  await renderer.ensureVariants([0]);
  assert.equal(renderer._variantSets.size, 1, 'no probes, no probe pipelines');
  await renderer._enableFeatures(1);
  const [plain, withProbes] = renderer._variantSets.values();
  assert.equal(withProbes.ready, true);
  assert.equal(renderer._readySet(3), withProbes, 'probes and decals asked for, probes ready: the probes set');
  assert.deepEqual([...withProbes.forward.keys()], [...plain.forward.keys()]);
  assert.ok([...plain.forward.values()].every((d) => d.constants.PROBES === 0 && d.constants.DECALS === 0));
  assert.ok([...withProbes.forward.values()].every((d) => d.constants.PROBES === 1 && d.constants.DECALS === 0));
  await renderer.ensureVariants([4]);
  assert.equal(withProbes.forward.size, plain.forward.size, 'a later load gets both');
  passed++;
  console.log('  ok  probe pipelines are built when probes are, for every variant then and after');
})();

await (async () => {
  // Two loads at once. The second used to find the first one's variants
  // marked as handled and return before they were compiled.
  let release;
  let compiles = 0;
  const device = {
    createRenderPipelineAsync: () => {
      compiles++;
      return new Promise((resolve) => { release = () => resolve({}); });
    },
  };
  const cache = new PipelineCache(device);
  const desc = { label: 'x', layout: 'auto', shader: { module: {}, id: 1 }, targets: [{ format: 'rgba16float' }] };

  const first = cache.warm([desc]);
  let secondDone = false;
  const second = cache.warm([desc]).then(() => { secondDone = true; });
  await Promise.resolve();
  assert.equal(compiles, 1, 'one compile for one pipeline');
  assert.equal(secondDone, false, 'the second warm waits for it rather than returning');
  release();
  await Promise.all([first, second]);
  assert.equal(cache.pipelines.size, 1);
  passed++;
  console.log('  ok  a warm() that finds a compile in flight waits for it');

  // A failed compile is not remembered as done.
  const failing = new PipelineCache({ createRenderPipelineAsync: () => Promise.reject(new Error('bad')) });
  await assert.rejects(() => failing.warm([desc]), /bad/);
  assert.equal(failing._compiling.size, 0, 'and it is forgotten, so the next warm() tries again');
  passed++;
  console.log('  ok  a failed compile is retried, not cached');

  // get() per draw: the same object comes back without building its key
  // again, and an equal descriptor built separately still shares the pipeline.
  let built = 0;
  const syncCache = new PipelineCache({ createRenderPipeline: () => ({ n: ++built }) });
  const one = syncCache.get(desc);
  let keyed = 0;
  const watched = new Proxy(desc, { get(target, name) { if (name === 'constants') keyed++; return target[name]; } });
  syncCache.get(watched);
  assert.equal(keyed, 1, 'a new object is keyed');
  keyed = 0;
  syncCache.get(watched);
  assert.equal(keyed, 0, 'and not again');
  assert.equal(syncCache.get({ ...desc }), one, 'equal descriptors, one pipeline');
  assert.equal(built, 1);
  passed++;
  console.log('  ok  get() keys a descriptor once, and equal ones share a pipeline');
})();

// ------------------------------------------------------------------ .hdr maps

console.log('\n.hdr environment maps');

/** An .hdr file: header lines, a blank line, the size line, then `pixels`. */
function hdrFile(sizeLine, pixels, headers = ['FORMAT=32-bit_rle_rgbe']) {
  const text = ['#?RADIANCE', ...headers, '', sizeLine, ''].join('\n');
  return Uint8Array.from([...Buffer.from(text, 'latin1'), ...pixels]);
}

/** What parseHDR makes of one RGBE pixel. */
const rgbeValue = (m, e) => (m + 0.5) * 2 ** (e - 136);

test('flat pixels decode by Radiance\'s own formula, top row first', () => {
  // 2x1: (128, 64, 0 | 129) and a zero exponent, which is black.
  const { width, height, data } = parseHDR(hdrFile('-Y 1 +X 2', [128, 64, 0, 129, 200, 200, 200, 0]));
  assert.equal(width, 2); assert.equal(height, 1);
  close(data[0], rgbeValue(128, 129), 1e-12);
  close(data[1], rgbeValue(64, 129), 1e-12);
  close(data[2], rgbeValue(0, 129), 1e-12);
  assert.deepEqual([...data.subarray(3)], [0, 0, 0]);
});

test('run-length scanlines decode, one channel at a time', () => {
  // Width 8: the smallest the new-style runs are allowed at.
  const runs = [
    130, 10, 134, 20,          // R: 2 x 10, then 6 x 20
    8, 1, 2, 3, 4, 5, 6, 7, 8, // G: 8 literals
    136, 0,                    // B: 8 x 0
    136, 128,                  // E: 8 x 128
  ];
  const { data } = parseHDR(hdrFile('-Y 1 +X 8', [2, 2, 0, 8, ...runs]));
  const f = 2 ** (128 - 136);
  close(data[0], 10.5 * f, 1e-12, 'R, first run');
  close(data[3 * 5], 20.5 * f, 1e-12, 'R, second run');
  close(data[3 * 7 + 1], 8.5 * f, 1e-12, 'G, last literal');
  close(data[3 * 7 + 2], 0.5 * f, 1e-12, 'B');
});

test('the old repeat marker copies the pixel before it, and +Y rows are bottom first', () => {
  const pixel = [100, 50, 25, 130];
  const { data } = parseHDR(hdrFile('+Y 2 +X 3', [...pixel, 1, 1, 1, 2, ...new Array(12).fill(0)]));
  // The file's first row is the image's BOTTOM row.
  for (let x = 0; x < 3; x++) close(data[(3 + x) * 3], rgbeValue(100, 130), 1e-12, `bottom row ${x}`);
  assert.deepEqual([...data.subarray(0, 9)], new Array(9).fill(0), 'top row, the file\'s last');
});

test('a literal run of exactly 128 is literal, and stacked repeats count in bytes', () => {
  // 128 is the longest literal; one more is the shortest repeat.
  const literal = Array.from({ length: 128 }, (_, i) => i);
  const channels = [128, ...literal, 128, ...literal, 128, ...literal, 255, 128, 129, 128];
  const { data } = parseHDR(hdrFile('-Y 1 +X 128', [2, 2, 0, 128, ...channels]));
  close(data[3 * 127], 127.5 * 2 ** -8, 1e-12, 'the last literal');

  // Two markers in a row: 2, then 1 shifted a byte -- 3 + 256 pixels, then one more.
  const bytes = [9, 9, 9, 130, 1, 1, 1, 2, 1, 1, 1, 1, 5, 5, 5, 130];
  const { data: rows } = parseHDR(hdrFile('-Y 1 +X 260', bytes));
  close(rows[3 * 258], 9.5 * 2 ** -6, 1e-12, 'pixel 258 is still a repeat');
  close(rows[3 * 259], 5.5 * 2 ** -6, 1e-12, 'pixel 259 is the next one');
});

test('EXPOSURE divides out of every pixel', () => {
  const { data } = parseHDR(hdrFile('-Y 1 +X 1', [128, 128, 128, 128], ['EXPOSURE=2', 'EXPOSURE=4']));
  close(data[0], rgbeValue(128, 128) / 8, 1e-12);
});

test('a malformed file says what is wrong, and never reads past its end', () => {
  const cases = [
    [Uint8Array.from(Buffer.from('P6\n1 1\n255\n')), /not a Radiance file/],
    [Uint8Array.from(Buffer.from('#?RADIANCE\nFORMAT=32-bit_rle_rgbe')), /header never ends/],
    [hdrFile('-Y 1 +X 1', [0, 0, 0, 0], ['FORMAT=32-bit_rle_xyze']), /32-bit_rle_xyze is not supported/],
    [hdrFile('+X 1 -Y 1', [0, 0, 0, 0]), /orientations are supported/],
    [hdrFile('-Y 1 +X 2', [1, 2, 3, 4]), /pixel data ends early/],
    [hdrFile('-Y 1 +X 8', [2, 2, 0, 8, 137, 1]), /run is longer than its scanline/],
    [hdrFile('-Y 1 +X 8', [2, 2, 0, 8, 0]), /run is longer than its scanline/],
    [hdrFile('-Y 1 +X 2', [1, 1, 1, 1, 0, 0, 0, 0]), /repeat with no pixel before it/],
    [hdrFile('-Y 1 +X 2', [9, 9, 9, 9, 1, 1, 1, 5]), /run is longer than its scanline/],
    [hdrFile('-Y 1 +X 1', [0, 0, 0, 0], ['EXPOSURE=0']), /EXPOSURE=0 is not a positive number/],
  ];
  for (const [bytes, message] of cases) assert.throws(() => parseHDR(bytes), message);
  assert.throws(() => parseHDR(hdrFile('-Y 2 +X 9000', []), { maxDimension: 8192 }), /9000x2 map is past this device's 8192/);
});

test('half floats round to nearest, hold at the largest finite, and flush the unrepresentable', () => {
  assert.equal(halfBits(1), 0x3c00);
  assert.equal(halfBits(0.5), 0x3800);
  assert.equal(halfBits(2), 0x4000);
  assert.equal(halfBits(65504), 0x7bff);
  assert.equal(halfBits(1e9), 0x7bff, 'a sun past f16 holds at 65504, not infinity');
  assert.equal(halfBits(2 ** -24), 1, 'the smallest subnormal');
  assert.equal(halfBits(2 ** -26), 0);
  assert.equal(halfBits(-3), 0);
  assert.equal(halfBits(NaN), 0);
  assert.equal(halfBits(1 + 1 / 2048 + 1e-9), 0x3c01, 'just past half a step rounds up');
  assert.equal(halfBits(2 - 2 ** -12), 0x4000, 'rounding up past the mantissa carries into the exponent');
  // Every exact power of two in the normal range, where log2 can slip an ulp.
  for (let e = -14; e <= 15; e++) assert.equal(halfBits(2 ** e), (e + 15) << 10, `2^${e}`);
  assert.deepEqual([...toHalfRGBA(Float32Array.of(1, 0.5, 2))], [0x3c00, 0x3800, 0x4000, 0x3c00]);
});


console.log(`\n${passed} checks passed\n`);
