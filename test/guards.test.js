// Guards and bookkeeping self-check. Run: node test/guards.test.js
//
// Written from a mutation pass: each guard, boundary and piece of reuse logic
// below was broken on purpose, one at a time, and the other suites went on
// passing. A check that cannot fail is not a check, so each test here is the
// smallest thing that fails when its line breaks.
//
// No GPU. The WebGPU objects are stand-ins that record what was asked of them,
// which is all a CPU-side guard can be tested against.

import assert from 'node:assert/strict';

import { HandleAllocator } from '../src/core/handle.js';
import { grownCapacity } from '../src/core/grow.js';
import { frustumCreate, frustumFromViewProjection, frustumTestAABB } from '../src/core/math/frustum.js';
import { mat4Create, mat4Decompose, mat4FromQuatPosScale, mat4PerspectiveReverseZInfinite } from '../src/core/math/mat4.js';
import { quatCreate, quatNormalize, quatSetAxisAngle } from '../src/core/math/quat.js';
import { createBuffer, storageCapacity } from '../src/rhi/buffer.js';
import { mipLevelCountFor, createTexture } from '../src/rhi/texture.js';
import { PipelineCache } from '../src/rhi/pipeline.js';
import { MorphStore, packMorphCountStride } from '../src/render/morph.js';
import {
  MaterialRegistry, MATERIAL_BYTES, extensionSlotCount, EXTENSION_BINDING, VARIANT_EXTENDED, variantKey, variantPipelineState,
} from '../src/render/material.js';
import { EXTENSION_TEXTURES, CORE_TEXTURE_COUNT, MATERIAL_TEXTURES } from '../src/scene/gltf/images.js';
import { DebugLines } from '../src/render/debug.js';

/** Where the material uniform's texture rows start, in floats: they end it, eight floats a texture. */
const UV_ROWS = MATERIAL_BYTES / 4 - MATERIAL_TEXTURES.length * 8;
import { SkinPalette } from '../src/render/skin.js';
import { ShadowMaps, MAX_CASCADES } from '../src/render/shadows.js';
import { Scene } from '../src/scene/scene.js';
import { TransformStore } from '../src/scene/transform.js';
import { checkJointIndices } from '../src/scene/gltf/skin.js';
import { ClusteredLights } from '../src/render/clustered.js';
import { GpuDriven, addToRuns } from '../src/render/gpudriven.js';
import { Environment } from '../src/render/ibl.js';
import { createTexture2D } from '../src/rhi/texture.js';
import { createPipelineLayout } from '../src/rhi/bindgroups.js';
import { Clock } from '../src/core/time.js';

globalThis.GPUBufferUsage ??= {
  MAP_READ: 0x0001, COPY_SRC: 0x0004, COPY_DST: 0x0008, INDEX: 0x0010, VERTEX: 0x0020,
  UNIFORM: 0x0040, STORAGE: 0x0080, INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
};
globalThis.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
globalThis.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

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

/** A device that records what it was asked, with WebGPU's default limits. */
function fakeRhi(limits = {}) {
  const log = { writes: [], copies: [], bindGroups: [], pipelines: 0 };
  const device = {
    createBuffer: (d) => ({ ...d, destroyed: false, destroy() { this.destroyed = true; } }),
    createTexture: (d) => ({ ...d, createView: () => ({}), destroy() {} }),
    createSampler: () => ({}),
    createBindGroupLayout: () => ({}),
    createBindGroup: (d) => { log.bindGroups.push(d); return d; },
    createRenderPipeline: () => { log.pipelines++; return {}; },
    createCommandEncoder: () => ({
      copyBufferToBuffer: (src, srcOffset, dst, dstOffset, size) => log.copies.push(size),
      finish: () => ({}),
    }),
  };
  const queue = {
    writeBuffer: (buffer, offset, data, dataOffset = 0, size) => log.writes.push({ buffer, offset, data, dataOffset, size }),
    writeTexture() {},
    submit() {},
  };
  return {
    device, queue, log,
    limits: {
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 256 << 20,
      maxStorageBufferBindingSize: 128 << 20,
      maxTextureDimension2D: 8192,
      maxSampledTexturesPerShaderStage: 16,
      ...limits,
    },
  };
}

console.log('\nhandles');

test('a slot retired at generation 0 never answers for NULL_HANDLE', () => {
  // Slot 0's generation 0 IS the null handle. Once the slot retires there,
  // only the generation-0 check keeps alive(0) false.
  const h = new HandleAllocator(1);
  for (let i = 0; i < 255; i++) h.free(h.alloc());
  assert.equal(h.generations[0], 0, 'slot 0 retired');
  assert.equal(h.alive(0), false);
  assert.throws(() => h.free(0));
  assert.notEqual(h.alloc() >>> 8, 0, 'and it is not handed out again');
});

test('the 24-bit index space is enforced at construction', () => {
  assert.throws(() => new HandleAllocator(2 ** 24 + 1), /24-bit/);
});

console.log('\nscene bookkeeping');

const PRIMITIVE = { indexCount: 3, materialId: 0, bounds: { min: [-1, -1, -1], max: [1, 1, 1] } };

test('removing a mesh changes the revision, so batches are rebuilt', () => {
  // Without it the renderer kept the removed mesh's batch and drew it from
  // buffers an unload may already have destroyed.
  const scene = new Scene({ capacity: 8 });
  const node = scene.createNode();
  scene._addRenderable(node.entity, { ...PRIMITIVE });
  const before = scene.revision;
  scene.remove(node);
  assert.notEqual(scene.revision, before);
});

test('scenes never share a revision, from creation on', () => {
  const a = new Scene({ capacity: 8 });
  const b = new Scene({ capacity: 8 });
  assert.notEqual(a.revision, b.revision, 'fresh scenes differ');
  const node = a.createNode();
  a._addRenderable(node.entity, { ...PRIMITIVE });
  const c = new Scene({ capacity: 8 });
  assert.notEqual(a.revision, c.revision);
});

test('the survivor of a swap-remove keeps its world box', () => {
  const scene = new Scene({ capacity: 8 });
  const first = scene.createNode();
  const second = scene.createNode();
  second.setPosition(10, 0, 0);
  scene._addRenderable(first.entity, { ...PRIMITIVE });
  scene._addRenderable(second.entity, { ...PRIMITIVE });
  scene.renderableMorphPad[1] = 5;
  scene.update();
  scene._refreshBounds();
  const box = Array.from(scene.worldMin.subarray(3, 6));
  assert.notDeepEqual(box, Array.from(scene.worldMin.subarray(0, 3)), 'the two boxes differ to begin with');
  scene.remove(first);
  assert.deepEqual(Array.from(scene.worldMin.subarray(0, 3)), box, 'moved to index 0 with its box');
  assert.equal(scene.renderableMorphPad[0], 5, 'and its morph padding');
});

console.log('\na frame where nothing moved');

test('a clean update composes nothing, and a dirty one composes what it must', () => {
  const ids = new HandleAllocator(8);
  const store = new TransformStore(8);
  const parent = ids.alloc();
  const child = ids.alloc();
  store.add(parent, { position: [1, 0, 0] });
  store.add(child, { position: [0, 2, 0], parent });
  assert.equal(store.update(), 2);
  assert.equal(store.movedPending, true, 'a compose that moved something says so');
  store.movedPending = false;

  assert.equal(store.update(), 0, 'nothing dirty: nothing composed');
  assert.equal(store.movedPending, false);

  // Across that skipped frame, the parent's recomputed flag is stale. The
  // child alone moves, and must still compose against the right parent.
  store.setPosition(child, 0, 3, 0);
  assert.equal(store.update(), 1);
  assert.deepEqual(Array.from(store.world.subarray(16 + 12, 16 + 15)), [1, 3, 0]);

  store.setPosition(parent, 5, 0, 0);
  assert.equal(store.update(), 2, 'a parent carries its children');
  assert.deepEqual(Array.from(store.world.subarray(16 + 12, 16 + 15)), [5, 3, 0]);
});

test('changed renderables upload as runs, with short gaps merged', () => {
  // Two movers at opposite ends used to send everything between them.
  const runs = [];
  for (const i of [0, 9999]) addToRuns(runs, i, 3);
  assert.deepEqual(runs, [0, 0, 9999, 9999], 'two runs of one, not one of ten thousand');

  const close = [];
  for (const i of [10, 12, 16, 20, 21, 40]) addToRuns(close, i, 3);
  assert.deepEqual(close, [10, 21, 40, 40], 'gaps of up to 3 are sent rather than paid a call for');
});

console.log('\nmorph arena');

test('two allocations from one hole do not overlap, and each is written', () => {
  const rhi = fakeRhi();
  const store = new MorphStore(rhi, { deltaCapacity: 64 });
  const a = store.allocate(new Float32Array(10));
  store.allocate(new Float32Array(10));
  store.allocate(new Float32Array(10));   // keeps the hole off the end
  store.free(a, 20);
  rhi.log.writes.length = 0;
  const x = store.allocate(new Float32Array(10));
  const y = store.allocate(new Float32Array(10));
  assert.deepEqual([x, y], [0, 10], 'side by side, not on top of each other');
  assert.deepEqual(rhi.log.writes.map((w) => w.offset), [0, 40], 'both uploaded, in bytes');
  assert.equal(store._holes.length, 0, 'an exact fit leaves no empty hole behind');
});

test('growing the arena copies every byte and says it moved', () => {
  const rhi = fakeRhi();
  const store = new MorphStore(rhi, { deltaCapacity: 16 });
  store.allocate(new Float32Array(10));
  const revision = store.revision;
  store.allocate(new Float32Array(10));
  assert.deepEqual(rhi.log.copies, [40], '10 floats of existing deltas, in bytes');
  assert.ok(store.revision > revision, 'the bind groups naming the old buffer rebuild');
});

test('more weights than fit grow the weight buffer and say so', () => {
  const store = new MorphStore(fakeRhi(), { weightCapacity: 4 });
  const revision = store.revision;
  store.update({ morphs: [{ weights: new Float32Array(5) }] });
  assert.ok(store.weightCapacity >= 5, 'one past capacity is enough to grow');
  assert.ok(store.revision > revision);
});

test('freed holes merge whichever order they come back in', () => {
  const store = new MorphStore(fakeRhi(), { deltaCapacity: 64 });
  const [a, b] = [store.allocate(new Float32Array(10)), store.allocate(new Float32Array(10))];
  store.allocate(new Float32Array(10));
  store.free(b, 10);
  store.free(a, 10);
  assert.deepEqual(store._holes, [{ base: 0, length: 20 }]);
});

test('target count and stride pack into one word, and 65536 targets do not', () => {
  assert.equal(packMorphCountStride(2, 6), 2 | (6 << 16));
  assert.throws(() => packMorphCountStride(0x10000, 3), /65535/);
});

console.log('\nmaterials');

const MATERIAL = { baseColorFactor: [0.25, 0.5, 0.75, 1] };

test('a texture past the device is refused by name, not made invalid', () => {
  const made = [];
  const rhi = { device: { createTexture: (d) => { made.push(d); return d; } }, limits: { maxTextureDimension2D: 4096, maxTextureArrayLayers: 256 } };
  assert.throws(() => createTexture(rhi, { label: 'big', size: [8192, 64, 1] }), /big is 8192x64, past this device's 4096/);
  assert.throws(() => createTexture(rhi, { label: 'deep', size: { width: 64, height: 64, depthOrArrayLayers: 300 } }), /deep has 300 layers, past this device's 256/);
  createTexture(rhi, { label: 'fits', size: [4096, 4096, 256] });
  assert.deepEqual(made.map((d) => d.label), ['fits'], 'only the one that fits reached the device');
});

test('compute pipelines are cached by what makes them different', () => {
  let created = 0;
  const cache = new PipelineCache({ createComputePipeline: () => ({ n: ++created }) });
  const layout = { id: 1, gpu: {} };
  const shader = { id: 7, module: {} };
  const a = cache.compute({ label: 'a', layout, shader, entry: 'main' });
  assert.equal(cache.compute({ label: 'again', layout, shader, entry: 'main' }), a, 'same layout, shader and entry');
  assert.notEqual(cache.compute({ layout, shader, entry: 'other' }), a, 'another entry');
  assert.notEqual(cache.compute({ layout, shader, entry: 'main', constants: { N: 2 } }), a, 'other constants');
  assert.equal(created, 3);
});


test('growth keeps every material: its values, its pipeline and its bind group', () => {
  const rhi = fakeRhi();
  const registry = new MaterialRegistry(rhi, { capacity: 2 });
  registry.register({ ...MATERIAL, uvSets: { baseColor: 1 } });
  registry.register({ ...MATERIAL, alphaMode: 'BLEND' });
  const pipeline = registry.pipelineIdOf[1];
  assert.notEqual(pipeline, 0, 'a second variant, so a zeroed table would show');
  rhi.log.writes.length = 0;
  registry.register(MATERIAL);   // the third: grows

  const upload = rhi.log.writes.find((w) => w.buffer === registry.buffer && w.offset === 0);
  assert.ok(upload, 'the new buffer was filled from the old contents');
  assert.equal(new Float32Array(upload.data, 0, 4)[1], 0.5, "material 0's colour came across");
  assert.equal(registry.bindGroups[0].entries[0].resource.buffer, registry.buffer, 'its bind group names the new buffer');
  assert.equal(registry.pipelineIdOf[1], pipeline, 'and its pipeline is unchanged');
  assert.equal(new Float32Array(upload.data, 0, 16)[12], 1, 'its UV set mask too');
});

test('a released id is reused, and its textures are let go', () => {
  const registry = new MaterialRegistry(fakeRhi(), { capacity: 4 });
  const id = registry.register(MATERIAL, { baseColor: { createView: () => ({}) } });
  registry.release(id);
  assert.equal(registry.bindGroups[id], undefined);
  assert.equal(registry._textures[id], undefined);
  assert.equal(registry.register(MATERIAL), id);
});

test('an animated material uploads again, unless its id went to another asset', () => {
  const rhi = fakeRhi();
  const registry = new MaterialRegistry(rhi, { capacity: 4 });
  const record = { ...MATERIAL, baseColorFactor: Float32Array.of(0.25, 0.5, 0.75, 1) };
  const id = registry.register(record);
  record.baseColorFactor[1] = 0.125;
  rhi.log.writes.length = 0;
  registry.update(id, record);
  assert.equal(rhi.log.writes.length, 1);
  const { data, offset } = rhi.log.writes[0];
  assert.equal(new Float32Array(data, offset, 4)[1], 0.125, 'the new colour went up');

  // Unloaded between the clip's change and the frame's upload: nothing to
  // upload. And once the id is handed to someone else, their material must
  // not be overwritten.
  registry.release(id);
  rhi.log.writes.length = 0;
  registry.update(id, record);
  assert.equal(rhi.log.writes.length, 0, 'a released id');
  const other = registry.register(MATERIAL);
  assert.equal(other, id, 'the id really was reused');
  rhi.log.writes.length = 0;
  registry.update(id, record);
  assert.equal(rhi.log.writes.length, 0);
});


test('extension textures share a binding per image, each kind keeping its own UVs', () => {
  const rhi = fakeRhi();
  const registry = new MaterialRegistry(rhi, { capacity: 4, frameTextures: 4 });
  assert.equal(registry.extensionSlots, Math.min(EXTENSION_TEXTURES.length, 16 - 4 - CORE_TEXTURE_COUNT));
  const image = { createView: () => ({ image: 'shared' }) };
  const id = registry.register(
    { ...MATERIAL, name: 'shared', uvSets: { specularColor: 1 } },
    { specular: image, specularColor: image },
  );
  const f32 = new Float32Array(rhi.log.writes.at(-1).data, 0, MATERIAL_BYTES / 4);
  const rows = (kind) => f32.subarray(UV_ROWS + (CORE_TEXTURE_COUNT + kind) * 8, UV_ROWS + (CORE_TEXTURE_COUNT + kind + 1) * 8);
  assert.deepEqual([rows(0)[3], rows(0)[7]], [0, 0], 'specular: UV set 0, binding 0');
  assert.deepEqual([rows(1)[3], rows(1)[7]], [1, 0], 'specular colour: UV set 1, the same binding');
  assert.deepEqual([f32[13], f32[14], f32[15], ...f32.subarray(16, 19)], [1.5, 1, 0, 1, 1, 1], 'the extension defaults');
  const entries = registry.shadingGroup(id).entries;
  assert.equal(entries.find((e) => e.binding === EXTENSION_BINDING).resource.image, 'shared');
  assert.equal(entries.length, EXTENSION_BINDING + registry.extensionSlots, 'every slot bound, spares to white');

  const plain = registry.register(MATERIAL);
  const g32 = new Float32Array(rhi.log.writes.at(-1).data, plain * registry.stride, MATERIAL_BYTES / 4);
  assert.equal(g32[UV_ROWS + CORE_TEXTURE_COUNT * 8 + 7], -1, 'no texture, no binding: the shader reads 1');
});

test('a material with more extension images than the device binds is refused, and takes no id', () => {
  assert.equal(extensionSlotCount(16, 4), Math.min(EXTENSION_TEXTURES.length, 7));
  assert.equal(extensionSlotCount(10, 5), 0, 'never negative');
  const registry = new MaterialRegistry(fakeRhi({ maxSampledTexturesPerShaderStage: 10 }), { capacity: 4, frameTextures: 4 });
  assert.equal(registry.extensionSlots, 1);
  const a = { createView: () => ({}) };
  const b = { createView: () => ({}) };
  assert.throws(() => registry.register({ name: 'rich' }, { specular: a, specularColor: b }),
    /material "rich" samples 2 distinct extension textures; this device binds 1 \(maxSampledTexturesPerShaderStage 10/);
  assert.equal(registry.count, 0);
  registry.register({ name: 'fits' }, { specular: a, specularColor: a });
});

test('a transmissive material leaves the opaque path, with a variant of its own', () => {
  // It must draw after the opaque scene is copied, with the one colour target
  // that pass has, so it cannot share a pipeline with an opaque material.
  const rhi = fakeRhi();
  const registry = new MaterialRegistry(rhi, { capacity: 4 });
  const glass = registry.register({ ...MATERIAL, transmission: 1, attenuationDistance: Infinity });
  const solid = registry.register(MATERIAL);
  assert.equal(registry.isTransparent(glass), true);
  assert.equal(registry.isTransmissive(glass), true);
  assert.equal(registry.isTransparent(solid), false);
  assert.notEqual(registry.variants[glass], registry.variants[solid]);
  assert.notEqual(registry.pipelineIdOf[glass], registry.pipelineIdOf[solid]);
  const f32 = new Float32Array(rhi.log.writes.find((w) => w.offset === glass * registry.stride).data, glass * registry.stride, MATERIAL_BYTES / 4);
  assert.deepEqual([f32[36], f32[38]], [1, 0], 'an infinite attenuation distance is written as 0, which the shader reads as none');
});

test('only a material that uses an extension layer pays for the extended shader', () => {
  const registry = new MaterialRegistry(fakeRhi(), { capacity: 8 });
  const texture = { createView: () => ({}) };
  const extended = (material, textures) => (registry.variants[registry.register(material, textures)] & VARIANT_EXTENDED) !== 0;
  assert.equal(extended(MATERIAL), false, 'plain');
  assert.equal(extended({ ...MATERIAL, ior: 1.33, specular: 0.5, specularColor: [1, 0, 0] }), false,
    'ior and specular factors need no layer');
  assert.equal(extended({ ...MATERIAL, clearcoat: 0.5 }), true);
  assert.equal(extended({ ...MATERIAL, extendedShading: true }), true, 'declared at zero, so a clip can raise it');
  assert.equal(extended(MATERIAL, { specular: texture }), true, 'an extension texture');
  assert.equal(variantPipelineState(variantKey(0, false, false, false, false, true)).constants.EXTENSIONS, 1);
  assert.equal(variantPipelineState(variantKey(0, false)).constants.EXTENSIONS, 0);
});

test('debug lines: what each shape writes, growth, and a frame with none adds no pass', () => {
  const rhi = fakeRhi();
  const debug = new DebugLines(rhi, null, {}, {});
  debug.box([0, 0, 0], [1, 2, 3], [1, 0.5, 0]);
  assert.equal(debug.count, 24, 'twelve edges');
  const f32 = new Float32Array(debug._floats.buffer, 0, debug.count * 4);
  const corners = new Set();
  for (let v = 0; v < debug.count; v++) corners.add(`${f32[v * 4]},${f32[v * 4 + 1]},${f32[v * 4 + 2]}`);
  assert.equal(corners.size, 8, 'every edge ends on a corner, and all eight are used');
  assert.deepEqual([...debug._bytes.subarray(12, 16)], [255, 128, 0, 255], 'colour as unorm bytes, opaque');

  debug.clear();
  debug.sphere([1, 1, 1], 2);
  assert.equal(debug.count, 3 * 32 * 2);
  const g32 = new Float32Array(debug._floats.buffer, 0, debug.count * 4);
  for (let v = 0; v < debug.count; v++) {
    const r = Math.hypot(g32[v * 4] - 1, g32[v * 4 + 1] - 1, g32[v * 4 + 2] - 1);
    assert.ok(Math.abs(r - 2) < 1e-5, 'every point on the sphere');
  }

  // Past the first allocation, and the earlier lines survive the move.
  debug.clear();
  debug.line([7, 8, 9], [0, 0, 0], [0, 1, 0]);
  for (let i = 0; i < 300; i++) debug.axes([i, 0, 0]);
  assert.equal(debug.count, 2 + 300 * 6);
  assert.deepEqual([...debug._floats.subarray(0, 3)], [7, 8, 9]);

  const passes = [];
  const graph = { addPass: (p) => passes.push(p) };
  debug.addPass(graph, { surface: 1, depth: 2, viewProjection: new Float32Array(16) });
  assert.equal(passes.length, 1);
  debug.clear();
  debug.addPass(graph, { surface: 1, depth: 2, viewProjection: new Float32Array(16) });
  assert.equal(passes.length, 1, 'nothing to draw, no pass');
});

test('the 4096th material fits the sort key and the 4097th does not', () => {
  const registry = new MaterialRegistry(fakeRhi(), { capacity: 1024 });
  for (let i = 0; i < 4096; i++) registry.register(MATERIAL);
  assert.throws(() => registry.register(MATERIAL), /4096/);
});

console.log('\nculling');

test('a box across the left edge of the view is kept', () => {
  // The plane test picks, per axis, the corner furthest along the normal. With
  // x picked the wrong way round, anything crossing a side edge vanished.
  const projection = mat4PerspectiveReverseZInfinite(mat4Create(), Math.PI / 2, 1, 0.1);
  const frustum = frustumFromViewProjection(frustumCreate(), projection);
  assert.equal(frustumTestAABB(frustum, [-100, -1, -11], [-5, 1, -9]), true, 'straddling the left plane');
  assert.equal(frustumTestAABB(frustum, [5, -1, -11], [100, 1, -9]), true, 'straddling the right plane');
  assert.equal(frustumTestAABB(frustum, [-100, -1, -11], [-50, 1, -9]), false, 'wholly outside');
});

console.log('\ndevice ceilings, where each store grows');

const TINY = { maxStorageBufferBindingSize: 1024, maxBufferSize: 1024 };

test('the morph arena and its weights stop at the binding limit', () => {
  const store = new MorphStore(fakeRhi(TINY), { deltaCapacity: 16, weightCapacity: 4 });
  assert.throws(() => store.allocate(new Float32Array(300)), RangeError, '1200 bytes of deltas');
  assert.throws(() => store.update({ morphs: [{ weights: new Float32Array(300) }] }), RangeError);
});

test('the joint palette stops at the binding limit', () => {
  const palette = new SkinPalette(fakeRhi(TINY), 4);
  assert.throws(() => palette._grow(17), RangeError, '17 joints of 64 bytes');
  palette._grow(16);
});

test('a buffer exactly at the device limit is allowed, one byte past is not', () => {
  const rhi = fakeRhi({ maxBufferSize: 16 });
  createBuffer(rhi, { size: 16, usage: 0 });
  assert.throws(() => createBuffer(rhi, { size: 20, usage: 0 }), RangeError);
});

test('a six-byte triangle of indices becomes an eight-byte buffer', () => {
  let size;
  const rhi = fakeRhi();
  rhi.device.createBuffer = (d) => { size = d.size; return { getMappedRange: () => new ArrayBuffer(d.size), unmap() {} }; };
  createBuffer(rhi, { data: new Uint16Array([0, 1, 2]), usage: 0 });
  assert.equal(size, 8);
});

test('storage capacity rounds down, never up', () => {
  assert.equal(storageCapacity(fakeRhi({ maxStorageBufferBindingSize: 100, maxBufferSize: 100 }), 16), 6);
});

test('growth reaches the ceiling exactly and doubles from nothing', () => {
  assert.equal(grownCapacity(1024, 2048), 2048);
  assert.equal(grownCapacity(0, 5), 8, 'no infinite loop from a zero capacity');
  assert.equal(grownCapacity(1024, 1800, 1800), 1800, 'needing exactly the ceiling is not past it');
});

test('shadow maps refuse a cascade count or a size the device cannot hold', () => {
  const rhi = fakeRhi();
  assert.throws(() => new ShadowMaps(rhi, { cascades: MAX_CASCADES + 1 }), /cascades/);
  assert.throws(() => new ShadowMaps(rhi, { size: 16384 }), /past this device's 8192/);
});

console.log('\npipelines');

function pipelineDesc(extra = {}) {
  return {
    label: 'p', layout: { id: 1, gpu: {} }, shader: { id: 1, module: {} },
    targets: [{ format: 'rgba8unorm' }], ...extra,
  };
}

test('every field that makes two pipelines different is in the key', () => {
  const rhi = fakeRhi();
  const cache = new PipelineCache(rhi.device);
  cache.get(pipelineDesc());
  cache.get(pipelineDesc({ depth: { depthBias: 1 } }));
  cache.get(pipelineDesc({ multisample: { count: 4 } }));
  cache.get(pipelineDesc({ targets: [{ format: 'rgba8unorm', writeMask: 0 }] }));
  assert.equal(rhi.log.pipelines, 4, 'depth bias, sample count and write mask each make a new pipeline');

  cache.get(pipelineDesc({ constants: { A: 1, B: 2 } }));
  cache.get(pipelineDesc({ constants: { B: 2, A: 1 } }));
  assert.equal(rhi.log.pipelines, 5, 'but constants in another order are the same pipeline');
});

await atest('warm() compiles once, counts it, and get() then finds it', async () => {
  let compiled = 0;
  const device = {
    createRenderPipelineAsync: async () => { compiled++; return {}; },
    createRenderPipeline: () => { compiled++; return {}; },
  };
  const cache = new PipelineCache(device);
  await cache.warm([pipelineDesc()]);
  await cache.warm([pipelineDesc()]);
  cache.get(pipelineDesc());
  assert.equal(compiled, 1);
  assert.equal(cache.created, 1);
});

test('a pipeline with no colour targets has no fragment stage at all', () => {
  let descriptor;
  const cache = new PipelineCache({ createRenderPipeline: (d) => { descriptor = d; return {}; } });
  cache.get(pipelineDesc({ targets: [] }));
  assert.equal(descriptor.fragment, undefined, 'a depth-only pass; an empty stage is a validation error');
});

test('a pipeline layout refuses a fifth bind group', () => {
  assert.throws(() => createPipelineLayout(fakeRhi().device, { 4: {} }, 'x'), /bind group 4 exceeds the 4/);
});

console.log('\nmore ceilings');

test('lights and renderables stop at the binding limit where they grow', () => {
  const lights = Object.assign(Object.create(ClusteredLights.prototype), {
    rhi: fakeRhi(TINY), lightCapacity: 4, lightBuffer: { destroy() {} }, _makeBindGroup: () => ({}), buffersRevision: 0,
  });
  assert.throws(() => lights._grow(17), RangeError, '17 lights of 64 bytes');
  const gpu = Object.assign(Object.create(GpuDriven.prototype), { rhi: fakeRhi(TINY), capacity: 4 });
  assert.throws(() => gpu._grow(9), RangeError, '9 renderables of 128 bytes');
});

test('an environment or texture past the device limit in EITHER direction is refused', () => {
  assert.throws(() => new Environment(fakeRhi(), { irradianceSize: 16384 }), /past this device's 8192/);
  assert.throws(() => createTexture2D(fakeRhi(), { width: 100, height: 9000 }), /past this device's 8192/);
});

test('a frame that is exactly one step long takes one step', () => {
  const clock = new Clock(0.25);   // exact in binary, so the edge is exact too
  clock.begin(0);
  clock.begin(0.25);
  assert.equal(clock.step(), true);
  assert.equal(clock.step(), false);
});

console.log('\nmath');

test('a rotated mirror decomposes into something that rebuilds it', () => {
  // An axis off every plane, so every term of the determinant is nonzero.
  const rotation = quatSetAxisAngle(quatCreate(), [1 / Math.sqrt(14), 2 / Math.sqrt(14), 3 / Math.sqrt(14)], 0.9);
  const m = mat4FromQuatPosScale(mat4Create(), rotation, [1, 2, 3], [-1, 2, 3]);
  const [p, r, s] = [new Float32Array(3), new Float32Array(4), new Float32Array(3)];
  assert.ok(mat4Decompose(p, r, s, m));
  assert.ok(s[0] * s[1] * s[2] < 0, 'the flip survives');
  const rebuilt = mat4FromQuatPosScale(mat4Create(), r, p, s);
  for (let i = 0; i < 16; i++) assert.ok(Math.abs(rebuilt[i] - m[i]) < 1e-5, `element ${i}`);
});

test('a mirror whose determinant lives in one cofactor is still found', () => {
  // A rotation that cycles the axes: every term of the 3x3 determinant is
  // zero but one, so dropping that term reads a mirror as a degenerate matrix.
  const cycle = quatSetAxisAngle(quatCreate(), [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)], (2 * Math.PI) / 3);
  const m = mat4FromQuatPosScale(mat4Create(), cycle, [0, 0, 0], [-1, 1, 1]);
  const [p, r, s] = [new Float32Array(3), new Float32Array(4), new Float32Array(3)];
  assert.ok(mat4Decompose(p, r, s, m), 'decomposes');
  assert.ok(s[0] * s[1] * s[2] < 0, 'and keeps the flip');
});

test('normalizing a zero quaternion is caught, not turned into NaN', () => {
  // Debug builds say so; release builds fall back to the identity.
  assert.throws(() => quatNormalize(quatCreate(), [0, 0, 0, 0]), /zero quaternion/);
  assert.deepEqual(Array.from(quatNormalize(quatCreate(), [0, 0, 0, 2])), [0, 0, 0, 1]);
});

test('a 256 texture has nine mip levels', () => {
  assert.equal(mipLevelCountFor(256, 256), 9);
  assert.equal(mipLevelCountFor(256, 1), 9, 'the longer side decides');
});

console.log('\nglTF boundaries');

test('a joint index equal to the joint count is refused', () => {
  checkJointIndices(Uint32Array.from([0, 1, 2, 3]), 4, 'mesh');
  assert.throws(() => checkJointIndices(Uint32Array.from([0, 1, 2, 4]), 4, 'mesh'));
});

console.log(`\n${passed} checks passed\n`);
