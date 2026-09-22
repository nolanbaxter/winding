// glTF import self-check. Run: node test/gltf.test.js
//
// Every asset here is built in memory. No binary fixtures to go stale, no
// network, and each test states exactly which part of the spec it exercises.

import assert from 'node:assert/strict';

import { parseContainer } from '../src/scene/gltf/glb.js';
import { readAccessorAsFloat32, readAccessorAsUint32 } from '../src/scene/gltf/accessor.js';
import {
  loadGLTF, instantiate, VERTEX_STRIDE_FLOATS, VERTEX_STRIDE_BYTES, VERTEX_BUFFER_LAYOUT,
  VERTEX_COLOR_INDEX,
} from '../src/scene/gltf/parse.js';
import { HandleAllocator, handleIndex } from '../src/core/handle.js';
import {
  textureImageIndex, textureSamplerIndex, samplerDescriptor, materialTextureSlots,
} from '../src/scene/gltf/images.js';
import { TransformStore } from '../src/scene/transform.js';
import { unweldAndComputeFlatNormals } from '../src/scene/gltf/tangents.js';

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`  ok  ${name}`);
}
async function atest(name, fn) {
  await fn();
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

// ------------------------------------------------------------- GLB builder

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

function pad4(bytes, fill) {
  const padded = (bytes.length + 3) & ~3;
  if (padded === bytes.length) return bytes;
  const out = new Uint8Array(padded).fill(fill);
  out.set(bytes);
  return out;
}

function makeGLB(json, binary, { version = 2 } = {}) {
  const jsonChunk = pad4(new TextEncoder().encode(JSON.stringify(json)), 0x20);
  const binChunk = binary ? pad4(binary, 0) : null;

  const total = 12 + 8 + jsonChunk.length + (binChunk ? 8 + binChunk.length : 0);
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);

  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, version, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonChunk.length, true);
  view.setUint32(16, CHUNK_JSON, true);
  out.set(jsonChunk, 20);

  if (binChunk) {
    const o = 20 + jsonChunk.length;
    view.setUint32(o, binChunk.length, true);
    view.setUint32(o + 4, CHUNK_BIN, true);
    out.set(binChunk, o + 8);
  }
  return out;
}

/** Concatenate typed arrays into one buffer, 4-byte aligned, reporting views. */
function packBuffer(arrays) {
  let total = 0;
  const views = arrays.map((a) => {
    const byteOffset = total;
    total += (a.byteLength + 3) & ~3;
    return { byteOffset, byteLength: a.byteLength };
  });

  const bytes = new Uint8Array(total);
  arrays.forEach((a, i) => {
    bytes.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), views[i].byteOffset);
  });
  return { bytes, views };
}

/** A unit quad in the XY plane: 4 vertices, 2 triangles, +Z normals, 0..1 UVs. */
const QUAD = {
  positions: Float32Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
  normals: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
  uvs: Float32Array.from([0, 0, 1, 0, 1, 1, 0, 1]),
  indices: Uint16Array.from([0, 1, 2, 0, 2, 3]),
};

function quadGLB({
  includeNormals = true, includeUVs = true, nodes, scenes,
  uv1s = null, colors = null, colorType = 'VEC4', colorComponentType = 5126,
} = {}) {
  const arrays = [QUAD.positions, QUAD.indices];
  if (includeNormals) arrays.push(QUAD.normals);
  if (includeUVs) arrays.push(QUAD.uvs);
  if (uv1s) arrays.push(uv1s);
  if (colors) arrays.push(colors);

  const { bytes, views } = packBuffer(arrays);
  const accessors = [
    {
      bufferView: 0, componentType: 5126, count: 4, type: 'VEC3',
      min: [0, 0, 0], max: [1, 1, 0],
    },
    { bufferView: 1, componentType: 5123, count: 6, type: 'SCALAR' },
  ];
  const attributes = { POSITION: 0 };
  let next = 2;
  if (includeNormals) {
    accessors.push({ bufferView: next, componentType: 5126, count: 4, type: 'VEC3' });
    attributes.NORMAL = next++;
  }
  if (includeUVs) {
    accessors.push({ bufferView: next, componentType: 5126, count: 4, type: 'VEC2' });
    attributes.TEXCOORD_0 = next++;
  }
  if (uv1s) {
    accessors.push({ bufferView: next, componentType: 5126, count: 4, type: 'VEC2' });
    attributes.TEXCOORD_1 = next++;
  }
  if (colors) {
    accessors.push({
      bufferView: next, componentType: colorComponentType, count: 4, type: colorType,
      normalized: colorComponentType !== 5126,
    });
    attributes.COLOR_0 = next++;
  }

  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors,
    meshes: [{ name: 'quad', primitives: [{ attributes, indices: 1 }] }],
    nodes: nodes ?? [{ name: 'root', mesh: 0 }],
    scenes: scenes ?? [{ nodes: [0] }],
    scene: 0,
  };
  return makeGLB(json, bytes);
}

/**
 * The quad again, rigged to two joints: the bottom edge to joint 0 and the top
 * edge to joint 1, which is the smallest thing that deforms visibly.
 */
function skinnedGLB({
  jointComponentType = 5123,      // UNSIGNED_SHORT
  weightComponentType = 5126,     // FLOAT
  weights = null,
  inverseBind = true,
  joints = null,
  extraJointSet = false,
  omitWeights = false,
} = {}) {
  const jointData = joints ?? Uint16Array.from([
    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
  ]);
  const weightData = weights ?? Float32Array.from([
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
  ]);

  const arrays = [QUAD.positions, QUAD.indices, QUAD.normals, QUAD.uvs, jointData];
  if (!omitWeights) arrays.push(weightData);
  const ibmData = new Float32Array(32);
  ibmData.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 0);
  ibmData.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1], 16);
  if (inverseBind) arrays.push(ibmData);

  const { bytes, views } = packBuffer(arrays);
  const accessors = [
    { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
    { bufferView: 1, componentType: 5123, count: 6, type: 'SCALAR' },
    { bufferView: 2, componentType: 5126, count: 4, type: 'VEC3' },
    { bufferView: 3, componentType: 5126, count: 4, type: 'VEC2' },
    { bufferView: 4, componentType: jointComponentType, count: 4, type: 'VEC4' },
  ];
  const attributes = { POSITION: 0, NORMAL: 2, TEXCOORD_0: 3, JOINTS_0: 4 };
  let next = 5;
  if (!omitWeights) {
    accessors.push({
      bufferView: next, componentType: weightComponentType, count: 4, type: 'VEC4',
      normalized: weightComponentType !== 5126,
    });
    attributes.WEIGHTS_0 = next++;
  }
  if (extraJointSet) attributes.JOINTS_1 = 4;

  const skin = { joints: [1, 2] };
  if (inverseBind) {
    accessors.push({ bufferView: next, componentType: 5126, count: 2, type: 'MAT4' });
    skin.inverseBindMatrices = next++;
  }

  return makeGLB({
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors,
    meshes: [{ name: 'rigged', primitives: [{ attributes, indices: 1 }] }],
    skins: [skin],
    nodes: [
      { name: 'character', mesh: 0, skin: 0 },
      { name: 'hip' },
      { name: 'chest' },
    ],
    scenes: [{ nodes: [0, 1, 2] }],
    scene: 0,
  }, bytes);
}


/**
 * The quad with morph targets.
 *
 * Each entry of `targets` names the deltas it carries. The defaults are the
 * smallest pair that exercises the two things the layout has to get right: a
 * target that moves positions only, beside one that also moves normals, so the
 * stride is decided by the wider of them and the narrower is zero-padded.
 */
const MORPH_T0_POSITION = Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]);
const MORPH_T1_POSITION = Float32Array.from([0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0]);
const MORPH_T1_NORMAL = Float32Array.from([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]);

function morphedGLB({
  targets = [{ POSITION: MORPH_T0_POSITION }, { POSITION: MORPH_T1_POSITION, NORMAL: MORPH_T1_NORMAL }],
  meshWeights,
  nodeWeights,
  includeNormals = true,
  secondPrimitiveTargets,
  animation = null,
  sparsePosition = false,
  targetType = 'VEC3',
  targetCount,
} = {}) {
  const arrays = [QUAD.positions, QUAD.indices, QUAD.normals, QUAD.uvs];
  const accessors = [
    { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
    { bufferView: 1, componentType: 5123, count: 6, type: 'SCALAR' },
    { bufferView: 2, componentType: 5126, count: 4, type: 'VEC3' },
    { bufferView: 3, componentType: 5126, count: 4, type: 'VEC2' },
  ];
  let next = 4;

  // Deferred accessor descriptions: the bufferView index is only known once
  // every array has been queued, so the data goes in first and the JSON after.
  const pending = [];
  const addAccessor = (data, type, count) => {
    arrays.push(data);
    pending.push({ type, count });
    return next++;
  };

  const targetJSON = (targets ?? []).map((target) => {
    const out = {};
    for (const name of ['POSITION', 'NORMAL', 'TANGENT']) {
      if (target[name] === undefined) continue;
      const type = name === 'POSITION' && targetType !== 'VEC3' ? targetType : 'VEC3';
      out[name] = addAccessor(target[name], type, targetCount ?? 4);
    }
    return out;
  });

  const secondTargetJSON = (secondPrimitiveTargets ?? []).map((target) => {
    const out = {};
    for (const name of ['POSITION', 'NORMAL', 'TANGENT']) {
      if (target[name] === undefined) continue;
      out[name] = addAccessor(target[name], 'VEC3', 4);
    }
    return out;
  });

  let animationJSON;
  if (animation) {
    const times = addAccessor(animation.times, 'SCALAR', animation.times.length);
    const valueType = animation.valueType ?? 'SCALAR';
    const wide = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[valueType];
    const values = addAccessor(animation.values, valueType, animation.values.length / wide);
    animationJSON = [{
      samplers: [{ input: times, output: values, interpolation: animation.interpolation ?? 'LINEAR' }],
      channels: [{ sampler: 0, target: { node: animation.node ?? 0, path: animation.path ?? 'weights' } }],
    }];
  }

  const { bytes, views } = packBuffer(arrays);
  for (let i = 0; i < pending.length; i++) {
    accessors.push({
      bufferView: 4 + i, componentType: 5126, count: pending[i].count, type: pending[i].type,
    });
  }

  // A sparse POSITION delta: no bufferView at all, so the base is zeros and
  // the overrides are the only data. The usual shape of a real morph target.
  if (sparsePosition) {
    const indices = Uint32Array.from([2, 3]);
    const values = Float32Array.from([0, 0, 1, 0, 0, 1]);
    const extra = packBuffer([indices, values]);
    const merged = new Uint8Array(bytes.length + extra.bytes.length);
    merged.set(bytes);
    merged.set(extra.bytes, bytes.length);
    const base = views.length;
    views.push(
      { byteOffset: bytes.length + extra.views[0].byteOffset, byteLength: extra.views[0].byteLength },
      { byteOffset: bytes.length + extra.views[1].byteOffset, byteLength: extra.views[1].byteLength },
    );
    const sparseAccessor = accessors.length;
    accessors.push({
      componentType: 5126, count: 4, type: 'VEC3',
      sparse: {
        count: 2,
        indices: { bufferView: base, componentType: 5125 },
        values: { bufferView: base + 1 },
      },
    });
    targetJSON.length = 0;
    targetJSON.push({ POSITION: sparseAccessor });
    return makeGLB(morphJSON(merged, views, accessors, targetJSON, [], {
      meshWeights, nodeWeights, includeNormals, animationJSON,
    }), merged);
  }

  return makeGLB(morphJSON(bytes, views, accessors, targetJSON, secondTargetJSON, {
    meshWeights, nodeWeights, includeNormals, animationJSON,
  }), bytes);
}

function morphJSON(bytes, views, accessors, targetJSON, secondTargetJSON, opts) {
  const attributes = { POSITION: 0, TEXCOORD_0: 3 };
  if (opts.includeNormals) attributes.NORMAL = 2;

  const primitives = [{ attributes, indices: 1, targets: targetJSON }];
  if (secondTargetJSON.length > 0) {
    primitives.push({ attributes, indices: 1, targets: secondTargetJSON });
  }

  const mesh = { name: 'face', primitives };
  if (opts.meshWeights !== undefined) mesh.weights = opts.meshWeights;

  const node = { name: 'head', mesh: 0 };
  if (opts.nodeWeights !== undefined) node.weights = opts.nodeWeights;

  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors,
    meshes: [mesh],
    nodes: [node],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  if (opts.animationJSON) json.animations = opts.animationJSON;
  return json;
}

// --------------------------------------------------------------- container

console.log('\nglb container');

test('parses a well-formed glb', () => {
  const glb = makeGLB({ asset: { version: '2.0' } }, Uint8Array.from([1, 2, 3]));
  const { json, binary } = parseContainer(glb);
  assert.equal(json.asset.version, '2.0');
  assert.deepEqual([...binary.subarray(0, 3)], [1, 2, 3]);
});

test('parses a plain .gltf json document', () => {
  // Same reader, detected by the absence of the magic number.
  const bytes = new TextEncoder().encode(JSON.stringify({ asset: { version: '2.0' } }));
  const { json, binary } = parseContainer(bytes);
  assert.equal(json.asset.version, '2.0');
  assert.equal(binary, null);
});

test('rejects glTF 1.0 by version, not by failing later', () => {
  assert.throws(() => parseContainer(makeGLB({ asset: { version: '1.0' } }, null, { version: 1 })),
    /version 1/);
});

test('rejects a file whose header lies about its length', () => {
  const glb = makeGLB({ asset: { version: '2.0' } }, null);
  new DataView(glb.buffer).setUint32(8, glb.length + 64, true);
  assert.throws(() => parseContainer(glb), /declares/);
});

test('rejects a chunk that runs past the end', () => {
  const glb = makeGLB({ asset: { version: '2.0' } }, null);
  new DataView(glb.buffer).setUint32(12, 10_000, true);
  assert.throws(() => parseContainer(glb), /past the end/);
});

// --------------------------------------------------------------- accessors

console.log('\naccessors');

function accessorDoc(arrays, accessors, bufferViewExtras = []) {
  const { bytes, views } = packBuffer(arrays);
  return {
    json: {
      asset: { version: '2.0' },
      buffers: [{ byteLength: bytes.length }],
      bufferViews: views.map((v, i) => ({ buffer: 0, ...v, ...(bufferViewExtras[i] ?? {}) })),
      accessors,
    },
    buffers: [bytes],
  };
}

test('reads tightly packed floats', () => {
  const { json, buffers } = accessorDoc(
    [Float32Array.from([1, 2, 3, 4, 5, 6])],
    [{ bufferView: 0, componentType: 5126, count: 2, type: 'VEC3' }],
  );
  vecClose(readAccessorAsFloat32(json, buffers, 0), [1, 2, 3, 4, 5, 6]);
});

test('reads interleaved attributes via byteStride', () => {
  // Exporters interleave position and uv into one bufferView; the accessor
  // then has to skip the other attribute's bytes on every element.
  const interleaved = Float32Array.from([
    1, 2, 3, /* uv */ 0.5, 0.25,
    4, 5, 6, /* uv */ 0.75, 0.125,
  ]);
  const { json, buffers } = accessorDoc(
    [interleaved],
    [
      { bufferView: 0, componentType: 5126, count: 2, type: 'VEC3', byteOffset: 0 },
      { bufferView: 0, componentType: 5126, count: 2, type: 'VEC2', byteOffset: 12 },
    ],
    [{ byteStride: 20 }],
  );

  vecClose(readAccessorAsFloat32(json, buffers, 0), [1, 2, 3, 4, 5, 6], EPS, 'positions');
  vecClose(readAccessorAsFloat32(json, buffers, 1), [0.5, 0.25, 0.75, 0.125], EPS, 'uvs');
});

test('dequantizes normalized unsigned shorts', () => {
  const { json, buffers } = accessorDoc(
    [Uint16Array.from([0, 32768, 65535, 65535])],
    [{ bufferView: 0, componentType: 5123, count: 2, type: 'VEC2', normalized: true }],
  );
  const uv = readAccessorAsFloat32(json, buffers, 0);
  close(uv[0], 0, EPS, 'min');
  close(uv[1], 32768 / 65535, EPS, 'mid');
  close(uv[2], 1, EPS, 'max');
});

test('normalized signed bytes clamp at -1', () => {
  // -128 / 127 is -1.0079, which would push a normal outside the unit sphere.
  const { json, buffers } = accessorDoc(
    [Int8Array.from([-128, 0, 127])],
    [{ bufferView: 0, componentType: 5120, count: 3, type: 'SCALAR', normalized: true }],
  );
  const values = readAccessorAsFloat32(json, buffers, 0);
  close(values[0], -1);
  close(values[2], 1);
});

test('widens u16 indices to u32', () => {
  const { json, buffers } = accessorDoc(
    [Uint16Array.from([0, 1, 2, 65535])],
    [{ bufferView: 0, componentType: 5123, count: 4, type: 'SCALAR' }],
  );
  const indices = readAccessorAsUint32(json, buffers, 0);
  assert.ok(indices instanceof Uint32Array);
  assert.deepEqual([...indices], [0, 1, 2, 65535]);
});

test('applies sparse overrides on top of the base data', () => {
  const { bytes, views } = packBuffer([
    Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 0]),   // base: 3 x VEC3
    Uint16Array.from([2]),                            // sparse index
    Float32Array.from([7, 8, 9]),                     // sparse value
  ]);
  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [{
      bufferView: 0, componentType: 5126, count: 3, type: 'VEC3',
      sparse: {
        count: 1,
        indices: { bufferView: 1, componentType: 5123 },
        values: { bufferView: 2 },
      },
    }],
  };
  vecClose(readAccessorAsFloat32(json, [bytes], 0), [0, 0, 0, 0, 0, 0, 7, 8, 9]);
});

test('a sparse accessor with no bufferView starts from zeros', () => {
  const { bytes, views } = packBuffer([
    Uint16Array.from([1]),
    Float32Array.from([5, 5, 5]),
  ]);
  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [{
      componentType: 5126, count: 2, type: 'VEC3',
      sparse: {
        count: 1,
        indices: { bufferView: 0, componentType: 5123 },
        values: { bufferView: 1 },
      },
    }],
  };
  vecClose(readAccessorAsFloat32(json, [bytes], 0), [0, 0, 0, 5, 5, 5]);
});

test('an accessor reading past its bufferView throws', () => {
  const { json, buffers } = accessorDoc(
    [Float32Array.from([1, 2, 3])],
    [{ bufferView: 0, componentType: 5126, count: 99, type: 'VEC3' }],
  );
  assert.throws(() => readAccessorAsFloat32(json, buffers, 0), /bufferView/);
});

// -------------------------------------------------------------- primitives

console.log('\nprimitives');

await atest('interleaves into the shared vertex layout', async () => {
  const model = await loadGLTF(quadGLB());
  const primitive = model.meshes[0].primitives[0];

  assert.equal(primitive.vertexCount, 4);
  assert.equal(primitive.indexCount, 6);
  assert.equal(primitive.vertices.length, 4 * VERTEX_STRIDE_FLOATS);
  assert.equal(VERTEX_STRIDE_BYTES, 60, '12 floats plus a second UV set and a packed colour');
  assert.equal(VERTEX_BUFFER_LAYOUT.arrayStride, VERTEX_STRIDE_BYTES,
    'the layout the pipeline declares must be the layout the importer writes');

  // Vertex 1: position (1,0,0), normal (0,0,1), uv (1,0)
  const v1 = primitive.vertices.subarray(VERTEX_STRIDE_FLOATS, VERTEX_STRIDE_FLOATS * 2);
  vecClose(v1.subarray(0, 3), [1, 0, 0], EPS, 'position');
  vecClose(v1.subarray(3, 6), [0, 0, 1], EPS, 'normal');
  vecClose(v1.subarray(6, 8), [1, 0], EPS, 'uv');

  // No TEXCOORD_1 in this asset, so set 1 falls back to set 0 rather than to
  // zeros: a material naming texCoord 1 then renders as its author meant.
  vecClose(v1.subarray(12, 14), [1, 0], EPS, 'uv1 falls back to uv0');

  // No COLOR_0 either, so every vertex is opaque white, which multiplies to
  // identity in the shader and is why the format needs no variant.
  const packed = new Uint32Array(primitive.vertices.buffer);
  assert.equal(packed[VERTEX_STRIDE_FLOATS + VERTEX_COLOR_INDEX] >>> 0, 0xffffffff,
    'absent COLOR_0 becomes opaque white');
});

await atest('a second UV set is read, not dropped', async () => {
  // A material that puts baked occlusion on UV1 is the standard layout out of
  // Blender and Max. Dropping texCoord sampled it with set 0, which is wrong
  // pixels rather than a missing texture, so nothing reported it.
  const uv1s = Float32Array.from([0.25, 0.75, 0.5, 0.75, 0.25, 0.5, 0.5, 0.5]);
  const model = await loadGLTF(quadGLB({ uv1s }));
  const { vertices } = model.meshes[0].primitives[0];

  for (let v = 0; v < 4; v++) {
    const o = v * VERTEX_STRIDE_FLOATS;
    vecClose(vertices.subarray(o + 12, o + 14), [uv1s[v * 2], uv1s[v * 2 + 1]], EPS, `uv1 ${v}`);
    assert.notDeepEqual(
      [...vertices.subarray(o + 12, o + 14)], [...vertices.subarray(o + 6, o + 8)],
      `uv1 ${v} must differ from uv0, or this proves nothing`,
    );
  }
});

await atest('COLOR_0 is read and packed to unorm8x4', async () => {
  const colors = Float32Array.from([
    1, 0, 0, 1,
    0, 1, 0, 1,
    0, 0, 1, 0.5,
    1, 1, 1, 1,
  ]);
  const model = await loadGLTF(quadGLB({ colors }));
  const packed = new Uint32Array(model.meshes[0].primitives[0].vertices.buffer);

  const at = (v) => packed[v * VERTEX_STRIDE_FLOATS + VERTEX_COLOR_INDEX] >>> 0;
  assert.equal(at(0), 0xff0000ff, 'opaque red');
  assert.equal(at(1), 0xff00ff00, 'opaque green');
  assert.equal(at(2), 0x80ff0000, 'half-alpha blue');
  assert.equal(at(3), 0xffffffff, 'opaque white');
});

await atest('a VEC3 COLOR_0 is opaque, per the spec', async () => {
  const colors = Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]);
  const model = await loadGLTF(quadGLB({ colors, colorType: 'VEC3' }));
  const packed = new Uint32Array(model.meshes[0].primitives[0].vertices.buffer);
  for (let v = 0; v < 4; v++) {
    const alpha = (packed[v * VERTEX_STRIDE_FLOATS + VERTEX_COLOR_INDEX] >>> 24) & 0xff;
    assert.equal(alpha, 255, `vertex ${v} alpha`);
  }
});

await atest('both survive the unweld a mesh without normals goes through', async () => {
  // Flat shading de-indexes the geometry, and every other attribute has to
  // come along or it ends up indexed by the old vertex ids -- which reads as
  // scrambled UVs and colours rather than as an error.
  const uv1s = Float32Array.from([0, 0, 1, 0, 0, 1, 1, 1]);
  const colors = Float32Array.from([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1]);
  const model = await loadGLTF(quadGLB({ includeNormals: false, uv1s, colors }));
  const { vertices, vertexCount } = model.meshes[0].primitives[0];

  assert.equal(vertexCount, 6, 'two triangles, de-indexed');
  const packed = new Uint32Array(vertices.buffer);
  const seen = new Set();
  for (let v = 0; v < vertexCount; v++) {
    seen.add(packed[v * VERTEX_STRIDE_FLOATS + VERTEX_COLOR_INDEX] >>> 0);
  }
  assert.ok(seen.size > 1, 'colours survived; a dropped attribute would leave them uniform');
  assert.ok(!seen.has(0xffffffff), 'and none fell back to the white default');
});

await atest('takes bounds from the accessor min/max the spec requires', async () => {
  const model = await loadGLTF(quadGLB());
  const { bounds } = model.meshes[0].primitives[0];
  vecClose(bounds.min, [0, 0, 0], EPS, 'min');
  vecClose(bounds.max, [1, 1, 0], EPS, 'max');
});

await atest('generates tangents when UVs exist but TANGENT does not', async () => {
  const model = await loadGLTF(quadGLB());
  const v0 = model.meshes[0].primitives[0].vertices;

  // U increases along +X and V along +Y, so the tangent is +X. cross(N,T) is
  // +Y, which matches the bitangent, so handedness is +1.
  vecClose(v0.subarray(8, 12), [1, 0, 0, 1], EPS, 'tangent');
});

await atest('a mesh with no NORMAL is de-indexed and flat shaded', async () => {
  // The spec requires flat normals in this case, which means every triangle
  // needs its own vertices -- 6 instead of 4 for a quad.
  const model = await loadGLTF(quadGLB({ includeNormals: false }));
  const primitive = model.meshes[0].primitives[0];

  assert.equal(primitive.vertexCount, 6, 'two triangles, three vertices each');
  assert.equal(primitive.indexCount, 6);

  for (let v = 0; v < 6; v++) {
    const o = v * VERTEX_STRIDE_FLOATS;
    vecClose(primitive.vertices.subarray(o + 3, o + 6), [0, 0, 1], EPS, `face normal ${v}`);
  }
});

await atest('a mesh with no UVs still produces a finite tangent', async () => {
  const model = await loadGLTF(quadGLB({ includeUVs: false }));
  const v0 = model.meshes[0].primitives[0].vertices;
  for (let i = 8; i < 12; i++) assert.ok(Number.isFinite(v0[i]), 'tangent must not be NaN');
  close(v0[11], 1, EPS, 'handedness');
});

// ------------------------------------------------------------------ nodes

console.log('\nnodes and instantiation');

await atest('decomposes a node matrix back into TRS', async () => {
  // Column-major: scale 2 on every axis, translated to (5, 6, 7).
  const matrix = [
    2, 0, 0, 0,
    0, 2, 0, 0,
    0, 0, 2, 0,
    5, 6, 7, 1,
  ];
  const model = await loadGLTF(quadGLB({ nodes: [{ name: 'm', matrix, mesh: 0 }] }));
  const node = model.nodes[0];

  vecClose(node.position, [5, 6, 7], EPS, 'translation');
  vecClose(node.scale, [2, 2, 2], EPS, 'scale');
  vecClose(node.rotation, [0, 0, 0, 1], EPS, 'rotation');
});

await atest('decomposes a rotated matrix', async () => {
  // 90 degrees about +Y, column-major.
  const matrix = [
    0, 0, -1, 0,
    0, 1, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 1,
  ];
  const model = await loadGLTF(quadGLB({ nodes: [{ matrix }] }));
  const { rotation } = model.nodes[0];

  const halfRoot2 = Math.SQRT1_2;
  vecClose(rotation, [0, halfRoot2, 0, halfRoot2], 1e-4, 'quaternion');
});

await atest('builds the entity hierarchy and composes world transforms', async () => {
  const glb = quadGLB({
    nodes: [
      { name: 'parent', translation: [10, 0, 0], children: [1] },
      { name: 'child', translation: [0, 5, 0], mesh: 0 },
    ],
    scenes: [{ nodes: [0] }],
  });

  const model = await loadGLTF(glb);
  const ids = new HandleAllocator(64);
  const transforms = new TransformStore(64);

  const { entities, renderables } = instantiate(model, ids, transforms);
  transforms.update();

  assert.equal(renderables.length, 1, 'only the child carries a mesh');
  assert.equal(renderables[0].entity, entities[1]);

  const childOffset = transforms.worldOffset(entities[1]);
  vecClose(
    transforms.world.subarray(childOffset + 12, childOffset + 15),
    [10, 5, 0], EPS, 'child world position',
  );
});

await atest('nodes with no declared scene still find their roots', async () => {
  const glb = quadGLB({
    nodes: [{ name: 'a', children: [1] }, { name: 'b' }],
    scenes: undefined,
  });
  const model = await loadGLTF(glb);
  assert.deepEqual(model.roots, [0], 'b is a child, so only a is a root');
});

await atest('a mirrored matrix keeps the flip in the scale, not the rotation', async () => {
  // Negative determinant. Mirroring is not a rotation, so it has to land in
  // the scale; without that split the quaternion extraction returns garbage.
  const matrix = [
    -1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
  const model = await loadGLTF(quadGLB({ nodes: [{ matrix }] }));
  const node = model.nodes[0];

  vecClose(node.scale, [-1, 1, 1], EPS, 'scale carries the mirror');
  vecClose(node.rotation, [0, 0, 0, 1], EPS, 'rotation stays identity');
});

// --------------------------------------------------------------- rejection

console.log('\nmalformed input');

await atest('an unsupported primitive mode names the mode', async () => {
  const { bytes, views } = packBuffer([QUAD.positions]);
  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [{ bufferView: 0, componentType: 5126, count: 4, type: 'VEC3' }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode: 5 }] }],
    nodes: [{ mesh: 0 }],
  };
  await assert.rejects(() => loadGLTF(makeGLB(json, bytes)), /TRIANGLE_STRIP/);
});

await atest('a required extension we cannot honor is refused up front', async () => {
  const json = { asset: { version: '2.0' }, extensionsRequired: ['KHR_draco_mesh_compression'] };
  await assert.rejects(() => loadGLTF(makeGLB(json, null)), /KHR_draco/);
});

await atest('an out-of-range index is caught at load, not at draw', async () => {
  const { bytes, views } = packBuffer([QUAD.positions, Uint16Array.from([0, 1, 99])]);
  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [
      { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    nodes: [{ mesh: 0 }],
  };
  await assert.rejects(() => loadGLTF(makeGLB(json, bytes)), /index 99/);
});

await atest('a mismatched attribute count is refused', async () => {
  const { bytes, views } = packBuffer([QUAD.positions, Float32Array.from([0, 0, 1])]);
  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [
      { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 1, type: 'VEC3' },
    ],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 } }] }],
    nodes: [{ mesh: 0 }],
  };
  await assert.rejects(() => loadGLTF(makeGLB(json, bytes)), /NORMAL has 1 entries/);
});

await atest('a node claimed by two parents is refused rather than recursed into', async () => {
  const glb = quadGLB({
    nodes: [{ children: [2] }, { children: [2] }, { name: 'shared' }],
    scenes: [{ nodes: [0, 1] }],
  });
  const model = await loadGLTF(glb);
  assert.throws(
    () => instantiate(model, new HandleAllocator(16), new TransformStore(16)),
    /more than one parent/,
  );
});

// --------------------------------------------------------------- textures

console.log('\ntexture references and samplers');

test('texture indices resolve to image and sampler indices', () => {
  const json = {
    textures: [{ source: 2, sampler: 1 }, { source: 0 }],
    images: [{}, {}, {}],
    samplers: [{}, {}],
  };
  assert.equal(textureImageIndex(json, 0), 2);
  assert.equal(textureSamplerIndex(json, 0), 1);

  // A texture with no sampler is legal and means "client default".
  assert.equal(textureSamplerIndex(json, 1), -1);

  assert.equal(textureImageIndex(json, -1), -1, 'absent texture');
  assert.equal(textureImageIndex(json, 99), -1, 'out of range');
  assert.equal(textureImageIndex({}, 0), -1, 'no textures array at all');
});

test('glTF filter enums split into WebGPU min and mipmap filters', () => {
  // glTF folds the mip filter into minFilter as six combined values; WebGPU
  // keeps them separate. Collapsing them wrongly gives either no mipmapping or
  // blurry nearest-filtered textures, neither of which errors.
  assert.deepEqual(
    pick(samplerDescriptor({ minFilter: 9987 }), ['minFilter', 'mipmapFilter']),
    { minFilter: 'linear', mipmapFilter: 'linear' },
    'LINEAR_MIPMAP_LINEAR',
  );
  assert.deepEqual(
    pick(samplerDescriptor({ minFilter: 9985 }), ['minFilter', 'mipmapFilter']),
    { minFilter: 'linear', mipmapFilter: 'nearest' },
    'LINEAR_MIPMAP_NEAREST',
  );
  assert.deepEqual(
    pick(samplerDescriptor({ minFilter: 9986 }), ['minFilter', 'mipmapFilter']),
    { minFilter: 'nearest', mipmapFilter: 'linear' },
    'NEAREST_MIPMAP_LINEAR',
  );
});

test('wrap modes map across, defaulting to repeat', () => {
  assert.equal(samplerDescriptor({ wrapS: 33071 }).addressModeU, 'clamp-to-edge');
  assert.equal(samplerDescriptor({ wrapT: 33648 }).addressModeV, 'mirror-repeat');
  assert.equal(samplerDescriptor({}).addressModeU, 'repeat', 'unspecified means repeat');
});

test('anisotropy is only requested when every filter is linear', () => {
  // Asking for anisotropy alongside a nearest filter is a validation error in
  // WebGPU, not a silent downgrade.
  assert.equal(samplerDescriptor({ minFilter: 9987, magFilter: 9729 }).maxAnisotropy, 16);
  assert.equal(samplerDescriptor({ magFilter: 9728 }).maxAnisotropy, 1, 'nearest mag');
  assert.equal(samplerDescriptor({ minFilter: 9984 }).maxAnisotropy, 1, 'nearest min');
});

test('sRGB is decided by the slot, not by the image', () => {
  // The same file used as a base colour map and as a roughness map needs two
  // GPU textures: one that decodes on sample and one that does not.
  const slots = materialTextureSlots({ textures: { baseColor: 0, metallicRoughness: 0, normal: 1 } });
  const bySlot = Object.fromEntries(slots.map((s) => [s.slot, s]));

  assert.equal(bySlot.baseColor.srgb, true);
  assert.equal(bySlot.emissive.srgb, true);
  assert.equal(bySlot.normal.srgb, false);
  assert.equal(bySlot.metallicRoughness.srgb, false);
  assert.equal(bySlot.occlusion.srgb, false);

  assert.equal(bySlot.baseColor.texture, bySlot.metallicRoughness.texture,
    'same source image, different colour space');
});

await atest('normal scale and occlusion strength are read off their texture refs', async () => {
  // glTF hangs both on the TEXTURE reference, not on the material, which is
  // easy to miss and silently leaves them at 1.
  const { bytes, views } = packBuffer([QUAD.positions, QUAD.indices]);
  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [
      { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: 6, type: 'SCALAR' },
    ],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] }],
    materials: [{
      name: 'mapped',
      normalTexture: { index: 0, scale: 0.4 },
      occlusionTexture: { index: 1, strength: 0.25 },
      pbrMetallicRoughness: { metallicRoughnessTexture: { index: 1 } },
    }],
    textures: [{ source: 0 }, { source: 1 }],
    images: [{ uri: 'a.png' }, { uri: 'b.png' }],
    nodes: [{ mesh: 0 }],
  };

  const model = await loadGLTF(makeGLB(json, bytes));
  const material = model.materials[0];

  assert.equal(material.normalScale, 0.4);
  assert.equal(material.occlusionStrength, 0.25);
  assert.equal(material.textures.normal, 0);
  assert.equal(material.textures.occlusion, 1);
  assert.equal(material.textures.metallicRoughness, 1,
    'occlusion and metallic-roughness may share one image');
});

await atest('the raw document is exposed for image decoding', async () => {
  // parse.js stays GPU-free and DOM-free; the image decoder needs the
  // bufferViews it deliberately drops, so they are handed over explicitly.
  const model = await loadGLTF(quadGLB());
  assert.ok(model.source.json.bufferViews, 'json reachable');
  assert.ok(model.source.buffers[0] instanceof Uint8Array, 'resolved buffers reachable');
});

function pick(object, keys) {
  return Object.fromEntries(keys.map((k) => [k, object[k]]));
}

// -------------------------------------------------------------- animations

console.log('\nanimations');

/**
 * A quad document with one animation over `channels`.
 *
 * Accessors are appended after the mesh's, so the indices here start at 4.
 */
function animatedGLB({ path = 'translation', interpolation = 'LINEAR', values, times = [0, 1] } = {}) {
  const components = path === 'rotation' ? 4 : 3;
  const perKey = interpolation === 'CUBICSPLINE' ? 3 : 1;
  const output = values ?? new Array(times.length * components * perKey).fill(1);

  const { bytes, views } = packBuffer([
    QUAD.positions, QUAD.indices, QUAD.normals, QUAD.uvs,
    Float32Array.from(times), Float32Array.from(output),
  ]);

  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [
      { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
      { bufferView: 1, componentType: 5123, count: 6, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: 4, type: 'VEC3' },
      { bufferView: 3, componentType: 5126, count: 4, type: 'VEC2' },
      { bufferView: 4, componentType: 5126, count: times.length, type: 'SCALAR' },
      {
        bufferView: 5, componentType: 5126,
        count: times.length * perKey, type: components === 4 ? 'VEC4' : 'VEC3',
      },
    ],
    meshes: [{ name: 'quad', primitives: [{ attributes: { POSITION: 0, NORMAL: 2, TEXCOORD_0: 3 }, indices: 1 }] }],
    nodes: [{ name: 'root', mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
    animations: [{
      name: 'spin',
      samplers: [{ input: 4, output: 5, interpolation }],
      channels: [{ sampler: 0, target: { node: 0, path } }],
    }],
  };
  return { glb: makeGLB(json, bytes), json };
}

await atest('reads channels, samplers and the clip duration', async () => {
  const model = await loadGLTF(animatedGLB({ times: [0, 2.5] }).glb);
  assert.equal(model.animations.length, 1);

  const clipData = model.animations[0];
  assert.equal(clipData.name, 'spin');
  close(clipData.duration, 2.5, EPS, 'duration is the last keyframe time');
  assert.equal(clipData.channels.length, 1);
  assert.equal(clipData.channels[0].path, 'translation');
  assert.equal(clipData.channels[0].components, 3);
  assert.equal(clipData.channels[0].interpolation, 'LINEAR');
});

await atest('defaults the interpolation to LINEAR when it is absent', async () => {
  const { glb, json } = animatedGLB();
  delete json.animations[0].samplers[0].interpolation;
  const model = await loadGLTF(makeGLB(json, parseContainer(glb).binary));
  assert.equal(model.animations[0].channels[0].interpolation, 'LINEAR');
});

await atest('a document with no animations yields an empty list, not undefined', async () => {
  const model = await loadGLTF(quadGLB());
  assert.deepEqual(model.animations, []);
});

await atest('drops a weights channel instead of failing the load', async () => {
  // Morph targets are not imported, so there is nothing for it to drive. A
  // model that merely contains one must still load.
  const { glb, json } = animatedGLB();
  json.animations[0].channels.push({ sampler: 0, target: { node: 0, path: 'weights' } });
  const model = await loadGLTF(makeGLB(json, parseContainer(glb).binary));
  assert.equal(model.animations[0].channels.length, 1);
});

await atest('rejects an accessor whose type disagrees with the property it drives', async () => {
  // VEC3 values on a rotation channel. Reading them anyway would produce a
  // plausible-looking quaternion assembled from the wrong floats.
  const { glb, json } = animatedGLB();
  json.animations[0].channels[0].target.path = 'rotation';
  await assert.rejects(
    loadGLTF(makeGLB(json, parseContainer(glb).binary)),
    /3-component accessor/,
  );
});

await atest('rejects a sampler whose value count does not match its times', async () => {
  // Both accessors are individually valid and in bounds -- three VEC3 values
  // against two keyframe times. Only comparing the two catches this; neither
  // accessor is wrong on its own.
  const { glb, json } = animatedGLB({ times: [0, 1], values: new Array(9).fill(1) });
  assert.equal(json.accessors[5].count, 2, 'built for two keys');
  json.accessors[5].count = 3;
  await assert.rejects(
    loadGLTF(makeGLB(json, parseContainer(glb).binary)),
    /2 times but 9 values/,
  );
});

// ------------------------------------------------------- degenerate input

console.log('\ndegenerate geometry');

test('a zero-area triangle gets a unit normal, not a zero one', () => {
  // Three collinear points. Exporters emit these at welded seams and collapsed
  // quads, so it is ordinary input rather than a malformed file. A zero normal
  // reaches the shader's normalize(tbn * tangentNormal) and poisons every
  // fragment of that face with NaN.
  const { normals } = unweldAndComputeFlatNormals(
    new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]), new Uint32Array([0, 1, 2]), [],
  );
  for (let v = 0; v < 3; v++) {
    const n = normals.subarray(v * 3, v * 3 + 3);
    close(Math.hypot(n[0], n[1], n[2]), 1, EPS, `vertex ${v} normal is unit length`);
  }
});

test('a real triangle still gets its geometric normal', () => {
  // The negative control: the fallback must not have swallowed the real case.
  const { normals } = unweldAndComputeFlatNormals(
    new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), new Uint32Array([0, 1, 2]), [],
  );
  close(normals[0], 0, EPS, 'x');
  close(normals[1], 0, EPS, 'y');
  close(normals[2], 1, EPS, 'z is the winding normal');
});

// -------------------------------------------------------------- morph targets

console.log('\nmorph targets');

await atest('deltas interleave vertex-major at the widest stride any target needs', async () => {
  const model = await loadGLTF(morphedGLB());
  const morph = model.meshes[0].primitives[0].morph;

  assert.equal(morph.targetCount, 2);
  // One target carries normals, so both are stored six floats wide.
  assert.equal(morph.stride, 6);
  assert.equal(morph.deltas.length, 4 * 2 * 6);

  // Vertex 2, target 0: the position delta it was given, then zero normals --
  // that target does not deform normals, and zero is the identity of addition.
  const v2t0 = (2 * 2 + 0) * 6;
  vecClose(morph.deltas.subarray(v2t0, v2t0 + 6), [0, 0, 1, 0, 0, 0], EPS, 'v2 t0');

  // Vertex 2, target 1: both halves present.
  const v2t1 = (2 * 2 + 1) * 6;
  vecClose(morph.deltas.subarray(v2t1, v2t1 + 6), [0, 2, 0, 1, 0, 0], EPS, 'v2 t1');

  // Vertex 0 moves under neither.
  vecClose(morph.deltas.subarray(0, 12), new Array(12).fill(0), EPS, 'v0');
});

await atest('a positions-only mesh stays three floats wide', async () => {
  const model = await loadGLTF(morphedGLB({ targets: [{ POSITION: MORPH_T0_POSITION }] }));
  const morph = model.meshes[0].primitives[0].morph;
  assert.equal(morph.stride, 3);
  assert.equal(morph.deltas.length, 4 * 1 * 3);
});

await atest('extent is the farthest a vertex travels under one target', async () => {
  const model = await loadGLTF(morphedGLB());
  const { extent } = model.meshes[0].primitives[0].morph;
  // Target 0 moves two vertices one unit along +Z; target 1 moves two by two.
  close(extent[0], 1, EPS, 'extent[0]');
  close(extent[1], 2, EPS, 'extent[1]');
});

await atest('a mesh with no targets has no morph data at all', async () => {
  const model = await loadGLTF(quadGLB());
  assert.equal(model.meshes[0].primitives[0].morph, null);
  assert.equal(model.meshes[0].targetCount, 0);
  assert.equal(model.nodes[0].weights, null);
});

await atest('a sparse delta accessor reads as the dense array it describes', async () => {
  // The common real-world encoding: a target that moves a few vertices stores
  // only those, against a base of zeros.
  const model = await loadGLTF(morphedGLB({ sparsePosition: true }));
  const morph = model.meshes[0].primitives[0].morph;
  assert.equal(morph.stride, 3);
  vecClose(morph.deltas.subarray(0, 6), [0, 0, 0, 0, 0, 0], EPS, 'untouched vertices');
  vecClose(morph.deltas.subarray(6, 12), [0, 0, 1, 0, 0, 1], EPS, 'overridden vertices');
  close(morph.extent[0], 1, EPS, 'extent');
});

// ------------------------------------------------------------- morph weights

await atest('weights default to zero, which is the undeformed mesh', async () => {
  const model = await loadGLTF(morphedGLB());
  vecClose(model.meshes[0].weights, [0, 0], EPS, 'mesh weights');
  vecClose(model.nodes[0].weights, [0, 0], EPS, 'node weights');
});

await atest('the mesh supplies defaults and the node overrides them', async () => {
  const both = await loadGLTF(morphedGLB({ meshWeights: [0.25, 0.5], nodeWeights: [1, 0] }));
  vecClose(both.meshes[0].weights, [0.25, 0.5], EPS, 'mesh');
  vecClose(both.nodes[0].weights, [1, 0], EPS, 'node overrides');

  const inherited = await loadGLTF(morphedGLB({ meshWeights: [0.25, 0.5] }));
  vecClose(inherited.nodes[0].weights, [0.25, 0.5], EPS, 'node inherits');

  // Copied, not shared: two nodes on one mesh animate independently.
  assert.notEqual(inherited.nodes[0].weights, inherited.meshes[0].weights);
});

await atest('a weight list of the wrong length is refused, not padded', async () => {
  await assert.rejects(
    () => loadGLTF(morphedGLB({ meshWeights: [0.5] })),
    /1 morph weights for 2 targets/,
  );
  await assert.rejects(
    () => loadGLTF(morphedGLB({ nodeWeights: [1, 0, 0] })),
    /3 morph weights for 2 targets/,
  );
});

// --------------------------------------------------------- malformed targets

await atest('primitives of one mesh must agree on how many targets there are', async () => {
  await assert.rejects(
    () => loadGLTF(morphedGLB({ secondPrimitiveTargets: [{ POSITION: MORPH_T0_POSITION }] })),
    /different morph target counts/,
  );
});

await atest('a target that deforms nothing is refused', async () => {
  await assert.rejects(
    () => loadGLTF(morphedGLB({ targets: [{}] })),
    /none of which deform/,
  );
});

await atest('a delta accessor of the wrong length is refused', async () => {
  await assert.rejects(
    () => loadGLTF(morphedGLB({
      targets: [{ POSITION: Float32Array.from([0, 0, 1, 0, 0, 1]) }], targetCount: 2,
    })),
    /2 POSITION deltas but the primitive has 4 vertices/,
  );
});

await atest('a non-VEC3 delta is refused', async () => {
  await assert.rejects(
    () => loadGLTF(morphedGLB({
      targets: [{ POSITION: Float32Array.from([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]) }],
      targetType: 'VEC4',
    })),
    /which is not VEC3/,
  );
});

await atest('deltas survive the unweld a mesh without normals goes through', async () => {
  // No NORMAL means flat shading, which de-indexes the mesh. Every attribute
  // has to be reordered with it -- deltas included, or vertex 0 of the new
  // mesh would read the delta of whatever vertex 0 used to be.
  const model = await loadGLTF(morphedGLB({ includeNormals: false }));
  const { morph, vertexCount } = model.meshes[0].primitives[0];

  assert.equal(vertexCount, 6);                       // 2 triangles, unwelded
  assert.equal(morph.deltas.length, 6 * 2 * 6);

  // The quad's indices are [0,1,2, 0,2,3], so unwelded vertices 2 and 4 are
  // both original vertex 2 -- the shared corner, duplicated.
  const v2t0 = (2 * 2 + 0) * 6;
  vecClose(morph.deltas.subarray(v2t0, v2t0 + 3), [0, 0, 1], EPS, 'unwelded v2 t0');
  const v4t0 = (4 * 2 + 0) * 6;
  vecClose(morph.deltas.subarray(v4t0, v4t0 + 3), [0, 0, 1], EPS, 'unwelded v4 t0');
  // Unwelded vertex 3 is original 0, which moves under nothing.
  const v3t0 = (3 * 2 + 0) * 6;
  vecClose(morph.deltas.subarray(v3t0, v3t0 + 3), [0, 0, 0], EPS, 'unwelded v3 t0');

  // Duplicating vertices cannot change how far the target reaches.
  close(morph.extent[0], 1, EPS, 'extent after unweld');
});

// ------------------------------------------------------------ weight channels

await atest('a weights channel is as wide as the mesh has targets', async () => {
  const model = await loadGLTF(morphedGLB({
    animation: { times: Float32Array.from([0, 1]), values: Float32Array.from([0, 0, 1, 0.5]) },
  }));
  assert.equal(model.animations.length, 1);
  const [channel] = model.animations[0].channels;
  assert.equal(channel.path, 'weights');
  // SCALAR in the file; two targets wide here, which is the only reading that
  // makes the run of four values mean anything.
  assert.equal(channel.components, 2);
  assert.equal(channel.times.length, 2);
  vecClose(channel.values, [0, 0, 1, 0.5], EPS, 'values');
  close(model.animations[0].duration, 1, EPS, 'duration');
});

await atest('a weights run of the wrong length is refused', async () => {
  await assert.rejects(
    () => loadGLTF(morphedGLB({
      animation: { times: Float32Array.from([0, 1]), values: Float32Array.from([0, 0, 1]) },
    })),
    /2 times but 3 values/,
  );
});

await atest('a weights channel on a mesh with no targets drives nothing', async () => {
  // Legal, and not worth failing a load over: the same non-event as a channel
  // that names no node.
  const model = await loadGLTF(morphedGLB({
    targets: [],
    animation: { times: Float32Array.from([0, 1]), values: Float32Array.from([0, 1]) },
  }));
  assert.equal(model.meshes[0].targetCount, 0);
  assert.equal(model.animations[0].channels.length, 0);
});

await atest('a TRS channel still checks its accessor type', async () => {
  await assert.rejects(
    () => loadGLTF(morphedGLB({
      animation: {
        path: 'translation', valueType: 'VEC2',
        times: Float32Array.from([0, 1]), values: Float32Array.from([0, 0, 1, 1]),
      },
    })),
    /drives translation \(3 components\) from a 2-component accessor/,
  );
});

// -------------------------------------------------------------------- skins

console.log('\nskins');

await atest('a skin gives its joint nodes and inverse bind matrices', async () => {
  const model = await loadGLTF(skinnedGLB());
  assert.equal(model.skins.length, 1);

  const skin = model.skins[0];
  // joints maps "joint 3" to "node 17". It is the only thing that makes a
  // vertex's integer mean anything, so the ORDER is the contract.
  assert.deepEqual([...skin.joints], [1, 2]);
  assert.equal(skin.inverseBind.length, 2 * 16);
  vecClose(skin.inverseBind.subarray(12, 15), [0, 0, 0], EPS, 'joint 0 bind translation');
  vecClose(skin.inverseBind.subarray(28, 31), [0, -1, 0], EPS, 'joint 1 bind translation');
});

await atest('the node carries the skin, not the mesh', async () => {
  // One mesh can be instanced under two skeletons, so the pairing lives on the
  // node. Everything downstream depends on reading it from there.
  const model = await loadGLTF(skinnedGLB());
  assert.equal(model.nodes[0].skin, 0);
  assert.equal(model.nodes[0].mesh, 0);
  assert.equal(model.nodes[1].skin, -1, 'a joint node is not itself skinned');
});

await atest('influences come through as integers and normalized weights', async () => {
  const model = await loadGLTF(skinnedGLB());
  const primitive = model.meshes[0].primitives[0];

  assert.ok(primitive.jointIndices instanceof Uint32Array, 'indices ADDRESS a palette');
  assert.deepEqual([...primitive.jointIndices.subarray(0, 4)], [0, 0, 0, 0]);
  assert.deepEqual([...primitive.jointIndices.subarray(8, 12)], [1, 0, 0, 0]);
  assert.equal(primitive.jointWeights.length, 16);
});

await atest('unnormalized weights are renormalized, not passed through', async () => {
  // The spec requires the file to normalize and exporters get it wrong often
  // enough that validators check. An unnormalized set does not fail loudly --
  // it scales the vertex toward or away from the origin by whatever the sum
  // is, which reads as a mesh that inflates as it animates.
  const model = await loadGLTF(skinnedGLB({
    weights: Float32Array.from([
      0.5, 0.25, 0, 0,     // sums to 0.75
      2, 2, 0, 0,          // sums to 4
      0, 0, 0, 0,          // sums to 0: no influence at all
      1, 0, 0, 0,
    ]),
  }));
  const w = model.meshes[0].primitives[0].jointWeights;

  for (let v = 0; v < 4; v++) {
    const o = v * 4;
    close(w[o] + w[o + 1] + w[o + 2] + w[o + 3], 1, EPS, `vertex ${v} sums to one`);
  }
  close(w[0], 2 / 3, EPS, 'ratio preserved');
  close(w[1], 1 / 3, EPS, 'ratio preserved');
  // Zero influence pins to the first joint. Left at zero the palette sends the
  // vertex to the origin, which is a spike through the middle of the model.
  close(w[8], 1, EPS, 'a weightless vertex is pinned, not sent to the origin');
});

await atest('an unsigned byte joint accessor reads as the same integers', async () => {
  const model = await loadGLTF(skinnedGLB({
    jointComponentType: 5121,
    joints: Uint8Array.from([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
  }));
  const indices = model.meshes[0].primitives[0].jointIndices;
  assert.deepEqual([...indices.subarray(8, 12)], [1, 0, 0, 0]);
});

await atest('normalized byte weights are dequantized', async () => {
  const model = await loadGLTF(skinnedGLB({
    weightComponentType: 5121,
    weights: Uint8Array.from([
      255, 0, 0, 0, 128, 127, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0,
    ]),
  }));
  const w = model.meshes[0].primitives[0].jointWeights;
  close(w[0], 1, EPS, 'full influence');
  close(w[4] + w[5], 1, EPS, 'split influence still sums to one');
});

await atest('a skin with no inverse bind matrices gets identities', async () => {
  // The spec's meaning for an absent accessor: a skeleton authored already in
  // bind pose. Not an error, and not zeros -- zeros would collapse the mesh.
  const model = await loadGLTF(skinnedGLB({ inverseBind: false }));
  const { inverseBind } = model.skins[0];
  assert.equal(inverseBind.length, 32);
  for (let j = 0; j < 2; j++) {
    const m = inverseBind.subarray(j * 16, j * 16 + 16);
    assert.deepEqual([...m], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], `joint ${j}`);
  }
});

await atest('influences survive the unweld a mesh without normals goes through', async () => {
  // De-indexing per face must carry them, or they come back indexed by the old
  // vertex ids -- which is a mesh rigged to the wrong joints rather than an
  // error.
  const glb = skinnedGLB();
  const model = await loadGLTF(glb);
  const before = model.meshes[0].primitives[0];
  assert.equal(before.vertexCount, 4);

  // Same document with NORMAL removed, which forces the flat-shading unweld.
  const json = JSON.parse(new TextDecoder().decode(
    glb.subarray(20, 20 + new DataView(glb.buffer, glb.byteOffset + 12, 8).getUint32(0, true)),
  ));
  delete json.meshes[0].primitives[0].attributes.NORMAL;
  const unwelded = await loadGLTF(makeGLB(json, glb.subarray(
    28 + new DataView(glb.buffer, glb.byteOffset + 12, 8).getUint32(0, true),
  )));
  const after = unwelded.meshes[0].primitives[0];

  assert.equal(after.vertexCount, 6, 'two triangles, de-indexed');
  assert.equal(after.jointIndices.length, 24);
  const seen = new Set([...after.jointIndices]);
  assert.ok(seen.has(1), 'the second joint survived; a dropped attribute would leave only zeros');
  for (let v = 0; v < after.vertexCount; v++) {
    const o = v * 4;
    const sum = after.jointWeights[o] + after.jointWeights[o + 1]
      + after.jointWeights[o + 2] + after.jointWeights[o + 3];
    close(sum, 1, EPS, `unwelded vertex ${v} still normalized`);
  }
});

await atest('more than four influences is refused, not silently truncated', async () => {
  // Taking the first four and renormalizing is the usual graceful degradation,
  // and it changes how the mesh deforms without saying so. A refusal naming
  // the limit is something a re-export can satisfy.
  await assert.rejects(
    loadGLTF(skinnedGLB({ extraJointSet: true })),
    /four influences/,
  );
});

await atest('JOINTS_0 without WEIGHTS_0 is refused', async () => {
  // Either alone is meaningless: indices with no weights cannot say how much,
  // weights with no indices cannot say of what.
  await assert.rejects(
    loadGLTF(skinnedGLB({ omitWeights: true })),
    /without the other/,
  );
});

await atest('a joint index past the skin is refused at load', async () => {
  // Not survivable downstream: the index reads past the palette, which is a
  // storage buffer, so it picks up the next instance's matrices and drags the
  // vertex somewhere arbitrary. Checked once here rather than clamped per
  // vertex on the GPU forever.
  await assert.rejects(
    loadGLTF(skinnedGLB({
      joints: Uint16Array.from([0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0]),
    })),
    /past the 2 the skin declares/,
  );
});

await atest('the joint range check pairs a mesh with the skin that drives it', async () => {
  // The check cannot happen while a primitive is built, because which skin
  // applies comes from the NODE. This asset's indices are fine against a
  // two-joint skin and would be out of range against a one-joint one.
  const model = await loadGLTF(skinnedGLB());
  assert.equal(model.skins[0].joints.length, 2);
  assert.equal(Math.max(...model.meshes[0].primitives[0].jointIndices), 1);
});

// --------------------------------------------------- malformed but plausible

console.log('\nfiles that used to load wrong rather than fail');

await atest('a float index accessor is refused, not truncated', async () => {
  // FLOAT is signed, and the signed guard excluded it so the float reader
  // could share it. The fast path then views the buffer as Float32Array and
  // copies into a Uint32Array, which truncates toward zero: an index of 2.0
  // reads as 2 and nothing looks wrong until one is 65535.9.
  const glb = quadGLB();
  const { json, binary } = parseContainer(glb);
  json.accessors[1].componentType = 5126;              // FLOAT indices
  await assert.rejects(
    () => loadGLTF(makeGLB(json, binary)),
    /stores FLOAT where an unsigned integer is required/,
  );
});

await atest('a float JOINTS_0 accessor is refused too', async () => {
  const glb = skinnedGLB();
  const { json, binary } = parseContainer(glb);
  json.accessors[4].componentType = 5126;
  await assert.rejects(
    () => loadGLTF(makeGLB(json, binary)),
    /stores FLOAT where an unsigned integer is required/,
  );
});

await atest('a MAT3 of bytes is refused rather than read at the wrong stride', async () => {
  // glTF pads each COLUMN of a MAT2/MAT3 to four bytes when the component is
  // smaller, so the element is not bytes * count long and every matrix after
  // the first would be read from the wrong place. Nothing here can reach it --
  // the only matrices read are MAT4 inverse binds, where the rule does not
  // apply -- so it says so rather than implementing a layout nothing uses.
  const glb = skinnedGLB();
  const { json, binary } = parseContainer(glb);
  json.accessors[6].type = 'MAT3';
  json.accessors[6].componentType = 5121;
  await assert.rejects(
    () => loadGLTF(makeGLB(json, binary)),
    /MAT3 of UNSIGNED_BYTE needs column padding/,
  );
});

await atest('a MAT4 of floats needs no padding and still loads', async () => {
  // The other half of the rule: four-byte components are already aligned, so
  // the guard above must not refuse the one matrix layout this engine reads.
  const model = await loadGLTF(skinnedGLB());
  assert.equal(model.skins[0].inverseBind.length, 32);
});

await atest('a declared but empty default scene instantiates nothing', async () => {
  // `scenes: [{}]` is legal and means a document whose contents are all
  // referenced rather than instantiated -- a library of meshes, which is a
  // real way to ship one. Falling through to orphan detection loaded every
  // node in the file, which is the opposite of what the document says.
  const model = await loadGLTF(quadGLB({ scenes: [{}] }));
  assert.equal(model.nodes.length, 1, 'the node is still in the document');
  assert.deepEqual(model.roots, [], 'and none of it is a root');

  const entities = new HandleAllocator(16);
  const transforms = new TransformStore(16);
  const { renderables } = instantiate(model, entities, transforms);
  assert.equal(renderables.length, 0, 'so nothing is instantiated');
});

await atest('a default scene index that names nothing is refused', async () => {
  const glb = quadGLB();
  const { json, binary } = parseContainer(glb);
  json.scene = 7;
  await assert.rejects(
    () => loadGLTF(makeGLB(json, binary)),
    /scene 7 is the default scene but does not exist/,
  );
});

await atest('a document with no scenes at all still loads its orphans', async () => {
  // The documented fallback, which the change above must not have eaten: with
  // no scene declared the spec leaves the choice to the runtime.
  const glb = quadGLB();
  const { json, binary } = parseContainer(glb);
  delete json.scenes;
  delete json.scene;
  const model = await loadGLTF(makeGLB(json, binary));
  assert.deepEqual(model.roots, [0]);
});

console.log(`\n${passed} checks passed\n`);
