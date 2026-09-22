// glTF import self-check. Run: node test/gltf.test.js
//
// Every asset here is built in memory. No binary fixtures to go stale, no
// network, and each test states exactly which part of the spec it exercises.

import assert from 'node:assert/strict';

import { parseContainer } from '../src/scene/gltf/glb.js';
import { readAccessorAsFloat32, readAccessorAsUint32 } from '../src/scene/gltf/accessor.js';
import {
  loadGLTF, instantiate, VERTEX_STRIDE_FLOATS, VERTEX_STRIDE_BYTES, VERTEX_BUFFER_LAYOUT,
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

function quadGLB({ includeNormals = true, includeUVs = true, nodes, scenes } = {}) {
  const arrays = [QUAD.positions, QUAD.indices];
  if (includeNormals) arrays.push(QUAD.normals);
  if (includeUVs) arrays.push(QUAD.uvs);

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
  assert.equal(VERTEX_STRIDE_BYTES, 48);
  assert.equal(VERTEX_BUFFER_LAYOUT.arrayStride, VERTEX_STRIDE_BYTES,
    'the layout the pipeline declares must be the layout the importer writes');

  // Vertex 1: position (1,0,0), normal (0,0,1), uv (1,0)
  const v1 = primitive.vertices.subarray(VERTEX_STRIDE_FLOATS, VERTEX_STRIDE_FLOATS * 2);
  vecClose(v1.subarray(0, 3), [1, 0, 0], EPS, 'position');
  vecClose(v1.subarray(3, 6), [0, 0, 1], EPS, 'normal');
  vecClose(v1.subarray(6, 8), [1, 0], EPS, 'uv');
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

console.log(`\n${passed} checks passed\n`);
