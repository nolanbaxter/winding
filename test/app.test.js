// Scene and Node self-check. Run: node test/app.test.js
//
// No GPU anywhere here. Scene holds references to primitives that live on the
// GPU but never calls a WebGPU function, which is exactly what makes this
// testable -- and is why the layering was worth keeping strict.

import assert from 'node:assert/strict';

import { Scene, DIRECTIONAL_FLOATS } from '../src/scene/scene.js';
import { packSprites, SPRITE_FLOATS, SpritePass } from '../src/render/sprites.js';
import { order2D, write2D, View2D, SPRITE2D_FLOATS } from '../src/render/view2d.js';
import { Camera2D } from '../src/scene/camera2d.js';
import { spriteSheet, shapeRadius } from '../src/scene/scene.js';
import { ringCapacity, packEmitter } from '../src/render/particles.js';
import { packDecals, DECAL_FLOATS } from '../src/render/decals.js';
import { Node } from '../src/scene/node.js';
import { Camera } from '../src/scene/camera.js';
import { OrbitController } from '../src/app/controllers.js';
import { Clock } from '../src/core/time.js';
import { NO_PARENT } from '../src/scene/transform.js';
import { handleIndex } from '../src/core/handle.js';
import { quatCreate, quatFromEuler } from '../src/core/math/quat.js';
import { vec3Create, vec3TransformQuat } from '../src/core/math/vec3.js';
import { createModuleWorker, workerShimSource, Winding } from '../src/app/engine.js';
import { Environment } from '../src/render/ibl.js';

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

/** A stand-in for what engine.load() returns. The buffers are never touched. */
function fakeAsset({ nodes, roots, meshCount = 1, primitivesPerMesh = 1 } = {}) {
  const meshes = [];
  for (let m = 0; m < meshCount; m++) {
    meshes.push({
      name: `mesh_${m}`,
      primitives: Array.from({ length: primitivesPerMesh }, (_, p) => ({
        vertexBuffer: `vb_${m}_${p}`,
        indexBuffer: `ib_${m}_${p}`,
        indexCount: 36,
        materialId: m,
        bounds: { min: Float32Array.from([-1, -1, -1]), max: Float32Array.from([1, 1, 1]) },
      })),
    });
  }
  return {
    meshes,
    nodes: nodes ?? [{ name: 'root', position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: 0 }],
    roots: roots ?? [0],
  };
}

/**
 * Entities a scene holds before anything is added.
 * Measured rather than written down, so a count that means "what the asset
 * made" stays right whatever a fresh scene starts with.
 */
const EMPTY_SCENE_ENTITIES = new Scene({ capacity: 8 }).entities.liveCount;

function node(name, extra = {}) {
  return {
    name,
    position: [0, 0, 0],
    rotation: [0, 0, 0, 1],
    scale: [1, 1, 1],
    children: [],
    mesh: -1,
    ...extra,
  };
}

// ------------------------------------------------------------------ scene

console.log('\nscene');

test('an LOD group: each level draws its own coverage, all measured on one sphere', () => {
  // Three levels on three nodes, the lower two listed only by MSFT_lod.
  // Node 2 is exported where node 0 is; node 1 a unit to the side.
  const asset = fakeAsset({
    meshCount: 3,
    nodes: [
      node('high', { mesh: 0, position: [5, 0, 0], lod: { ids: [1, 2], coverage: [0.5, 0.2, 0.01] } }),
      node('medium', { mesh: 1, position: [6, 0, 0] }),
      node('low', { mesh: 2, position: [5, 0, 0] }),
    ],
    roots: [0, 1],   // a file that ALSO lists a level as a root must not draw it twice
  });
  const scene = new Scene({ capacity: 8 });
  const root = scene.add(asset);
  scene.update();
  assert.equal(scene.renderableCount, 3, 'one renderable a level, none twice');
  const byMaterial = (m) => [...Array(scene.renderableCount).keys()].find((i) => scene.renderableMaterial[i] === m);
  const range = (i) => [...scene.renderableCoverage.subarray(i * 2, i * 2 + 2)];
  assert.deepEqual(range(byMaterial(0)).slice(0, 1), [0.5]);
  assert.ok(range(byMaterial(0))[1] > 1e38, 'the finest has no upper bound');
  vecClose(range(byMaterial(1)), [0.2, 0.5]);
  vecClose(range(byMaterial(2)), [0.01, 0.2]);
  const slot = scene.renderableLodSlot[byMaterial(0)];
  for (let m = 0; m < 3; m++) {
    assert.equal(scene.renderableLodSlot[byMaterial(m)], slot, 'all measured on the group node');
    vecClose(scene.renderableLodSphere.subarray(byMaterial(m) * 4, byMaterial(m) * 4 + 4), [0, 0, 0, Math.sqrt(3)]);
  }
  // Each level where the file put it, and moving the group moves them all.
  const worldX = (m) => scene.transforms.world[scene.renderableMatrixSlot[byMaterial(m)] * 16 + 12];
  vecClose([worldX(0), worldX(1), worldX(2)], [5, 6, 5]);
  root.setPosition(0, 0, 0);
  scene.update();
  vecClose([worldX(0), worldX(1), worldX(2)], [0, 1, 0]);

  // Removing a renderable keeps every survivor's LOD with it.
  scene.remove(root);
  assert.equal(scene.renderableCount, 0);
});

test('without coverage hints only the finest level draws, as a client without MSFT_lod would', () => {
  const scene = new Scene({ capacity: 8 });
  scene.add(fakeAsset({
    meshCount: 2,
    nodes: [node('high', { mesh: 0, lod: { ids: [1], coverage: null } }), node('low', { mesh: 1 })],
    roots: [0],
  }));
  assert.equal(scene.renderableCount, 1);
  assert.equal(scene.renderableLodSlot[0], -1, 'and it is in no group');
});

test('a sprite takes the aspect of its texture, or its pixels, and names a bad option', () => {
  const texture = { view: {}, width: 64, height: 32 };
  const scene = new Scene({ capacity: 8 });
  const node = scene.addSprite({ texture, position: [1, 2, 3] });
  const sprite = scene.spriteOf(node);
  assert.deepEqual([...sprite.size], [1, 0.5], 'one unit wide, at the aspect of the texture');
  assert.deepEqual([sprite.facing, sprite.blend, sprite.cutoff, sprite.pixels], ['camera', 'alpha', 0.5, false]);
  scene.setSprite(node, { pixels: true });
  assert.deepEqual([...scene.spriteOf(node).size], [64, 32], 'in pixels, the size of the texture');
  scene.setSprite(node, { size: [10, 10] });
  scene.setSprite(node, { texture: { view: {}, width: 8, height: 8 } });
  assert.deepEqual([...scene.spriteOf(node).size], [10, 10], 'a size asked for stays');
  for (const [options, why] of [
    [{}, /texture must be/], [{ texture, size: [1, 0] }, /size must be positive/],
    [{ texture, facing: 'down' }, /facing/], [{ texture, blend: 'overlay' }, /blend/],
    [{ texture, cutoff: 2 }, /cutoff/], [{ texture, color: [1, 1, 1] }, /color must be 4/],
  ]) assert.throws(() => scene.addSprite(options), why);
  scene.remove(node);
  assert.equal(scene.sprites.size, 0, 'removed with its node');
});

test('sprites draw cutouts, then additive, then alpha from far to near, in runs of one texture', () => {
  const a = { view: {}, width: 4, height: 4 };
  const b = { view: {}, width: 4, height: 4 };
  const scene = new Scene({ capacity: 8 });
  const near = scene.addSprite({ texture: a, position: [0, 0, -2] });
  scene.addSprite({ texture: a, position: [0, 0, -9] });                         // alpha, far
  scene.addSprite({ texture: b, position: [0, 0, -5], blend: 'additive' });
  scene.addSprite({ texture: a, position: [0, 0, -4], blend: 'cutout' });
  scene.addSprite({ texture: a, position: [0, 0, -6], blend: 'additive' });
  near.setScale(2, 3, 1);
  scene.update();
  const camera = new Camera({ fovY: 1, near: 0.1 });
  camera.position.set([0, 0, 0]);
  camera.target.set([0, 0, -1]);
  camera.update(1);
  const out = new Float32Array(8 * SPRITE_FLOATS);
  const { count, runs } = packSprites(scene, camera, out);
  assert.equal(count, 5);
  const z = [...Array(count).keys()].map((k) => out[k * SPRITE_FLOATS + 2]);
  assert.equal(z[0], -4, 'the cutout first');
  assert.deepEqual(z.slice(3), [-9, -2], 'alpha last, far to near');
  assert.deepEqual(runs.map((r) => [r.blend, r.count]), [['cutout', 1], ['additive', 1], ['additive', 1], ['alpha', 2]],
    'additive grouped by texture, one run each');
  assert.deepEqual([out[4 * SPRITE_FLOATS + 4], out[4 * SPRITE_FLOATS + 5]], [2, 3], 'scaled by its node');
});

test('one convention everywhere: pivots from the top-left, two numbers mean 2D, a getter and a node for every kind', () => {
  const texture = { view: {}, width: 4, height: 4 };
  const scene = new Scene({ capacity: 16 });

  // A pivot's [0, 0] is the image's top-left in 3D as in 2D: [0.5, 1] stands it on the ground.
  const tree = scene.addSprite({ texture, pivot: [0.5, 1], position: [0, 0, -3] });
  scene.update();
  const camera = new Camera({ fovY: 1, near: 0.1 });
  camera.target.set([0, 0, -1]);
  camera.update(1);
  const out = new Float32Array(2 * SPRITE_FLOATS);
  packSprites(scene, camera, out);
  vecClose(out.subarray(6, 8), [0.5, 0], 1e-6, 'bottom middle, where the 3D quad measures y up');

  // With no size, a sprite is the shape of the frame it shows, not of the whole sheet.
  const sheet = { view: {}, width: 64, height: 16 };
  const walker = scene.addSprite({ texture: sheet, animation: { frames: spriteSheet({ columns: 4 }) } });
  assert.deepEqual([...scene.spriteOf(walker).size], [1, 1], 'a 16 x 16 frame of a 64 x 16 sheet: square');
  scene.remove(walker);

  // Two numbers leave z alone, as setPosition(x, y) does; one scales every axis.
  tree.setScale(-1, 2);
  scene.update();
  const m = handleIndex(tree.entity) * 16;
  vecClose([scene.transforms.world[m], scene.transforms.world[m + 5], scene.transforms.world[m + 10]], [-1, 2, 1], 1e-6, 'z stays 1');
  tree.setScale(3);
  scene.update();
  close(scene.transforms.world[m + 10], 3);

  // Every addX has an xOf, a copy.
  const ring = scene.addShape({ shape: 'ellipse', size: [8, 8] });
  assert.equal(scene.shapeOf(ring).shape, 'ellipse');
  assert.equal(scene.shapeOf(tree), null, 'a node without one');
  scene.shapeOf(ring).size[0] = 99;
  assert.equal(scene.shapes.get(ring.entity).size[0], 8, 'a copy: changing it changes nothing');
  assert.throws(() => scene.setShape(ring, { size: [0, 1] }), /^Error: setShape:/, 'a set call names itself');
  const route = scene.addPath({ points: [[0, 0], [5, 5]], closed: false, strokeWidth: 2 });
  scene.setPath(route, scene.pathOf(route));
  assert.deepEqual(scene.pathOf(route).points, [[0, 0], [5, 5]], 'read back as addPath took it, and taken back');
  const lamp = scene.addLight({ direction: [0, -1, 0], radius: 7, outerAngle: 0.4, castShadow: true });
  const spot = scene.lightOf(lamp);
  assert.deepEqual([spot.type, [...spot.color], spot.intensity, spot.radius, spot.castShadow], ['spot', [1, 1, 1], 1, 7, true]);
  vecClose([spot.innerAngle, spot.outerAngle], [0.2, 0.4], 1e-6, 'its cone, as stored: 32-bit');
  scene.setLight(lamp, { intensity: 3 });
  assert.equal(scene.lightOf(lamp).intensity, 3);
  assert.throws(() => scene.setLight(ring, { intensity: 1 }), /not a light/);

  // A reflection probe is a node: boxed around where it is, moved with it, removed with it.
  const hall = scene.addProbe({ position: [0, 2, 0], size: [10, 4, 16] });
  scene.update();
  const probe = scene.reflectionProbes[0];
  assert.deepEqual([...probe.min, ...probe.max], [-5, 0, -8, 5, 4, 8]);
  probe.captured = true;
  hall.setPosition(1, 2, 0);
  scene.update();
  assert.deepEqual([probe.min[0], probe.captured], [-4, false], 'moved: its capture was of somewhere else');
  assert.deepEqual([...scene.probeOf(hall).size], [10, 4, 16]);
  assert.throws(() => scene.addProbe({ min: [0, 0, 0], max: [1, 1, 1] }), /a node now/);
  scene.remove(hall);
  assert.equal(scene.reflectionProbes.length, 0);

  // One way to free anything a load call returned.
  const engine = Object.create(Winding.prototype);
  let destroyed = 0;
  const loaded = { texture: { destroy: () => destroyed++ }, view: {}, width: 4, height: 4 };
  engine.unload(loaded);
  engine.unload(loaded);
  assert.equal(destroyed, 1, 'freed once');
  assert.throws(() => engine.unload({}), /not something load/);
});

test('one name per idea: a renamed option says what it is called now', () => {
  const texture = { view: {}, width: 4, height: 4 };
  const scene = new Scene({ capacity: 16 });
  assert.throws(() => scene.addSprite({ texture, rotation: 1 }), /^Error: addSprite: rotation is now angle/);
  assert.equal(scene.spriteOf(scene.addSprite({ texture, angle: 1 })).angle, 1);
  assert.throws(() => new Camera2D({ rotation: 1 }), /Camera2D: rotation is now angle/);
  assert.equal(new Camera2D({ angle: 1 }).angle, 1);
  assert.throws(() => scene.addProbe({ size: [1, 1, 1], blend: 1 }), /addProbe: blend is now fade/);
  assert.equal(scene.probeOf(scene.addProbe({ size: [1, 1, 1], fade: 0.5 })).fade, 0.5);

  // A node sets where it is; what a kind is goes through the scene, for lights as for the rest.
  const node = scene.createNode().setAxisAngle([0, 1, 0], 1).setEuler(1, 0);
  assert.equal(node.setLight, undefined);
  assert.deepEqual([...node.setPosition(1, 2, 3).worldPosition([0, 0, 0])], [0, 0, 0], 'as of the last update');
  scene.update();
  assert.deepEqual(node.worldPosition([0, 0, 0]), [1, 2, 3]);

  // One call moves a scene on: its clips, its sprite frames and its particles.
  const walker = scene.addSprite({ texture, animation: { frames: spriteSheet({ columns: 2 }), fps: 10 } });
  const sparks = scene.addEmitter({ rate: 10, lifetime: 1, size: 1 });
  scene.advance(0.15);
  assert.equal(scene.sprites.get(walker.entity).frame, 1);
  close(scene.emitters.get(sparks.entity).owed, 1.5, 1e-9);

  // run takes the scene and camera first, as renderFrame does; the old one object says so.
  const engine = Object.create(Winding.prototype);
  assert.throws(() => engine.run({ scene, camera: new Camera2D() }), /^Error: run: takes \(scene, camera, \{ update, frame, hud \}\)/);
});

test('run() skips a frame only when nothing it is drawn from has changed', () => {
  const engine = Object.create(Winding.prototype);
  engine.gpu = { width: 320, height: 240 };
  const sets = new Map([[0, { ready: true }]]);
  engine.renderer = {
    exposure: 1, fog: null, dof: null, skybox: true, shadowDistance: null, lightDistance: null, ao: null,
    debug: { count: 0, depthTest: true }, _variantSets: sets,
    post: { threshold: 1, knee: 0.5, filterRadius: 1, strength: 0.06, levels: 5, antialias: true, grading: null, fxaaPipeline: {} },
    upscaler: { pending: false },
    shadows: { lambda: 0.7, casterExtent: 4, normalBias: 1.5 },
  };
  engine._drawn = null;
  const scene = new Scene({ capacity: 16 });
  const node = scene.createNode();
  const camera = new Camera({ fovY: 1, near: 0.1 });
  camera.position.set([0, 0, 5]);
  scene.update();
  // What renderFrame leaves behind: composed transforms, spent moves, an updated camera.
  const drawn = () => { scene.update(); scene.transforms.movedPending = false; camera.update(engine.gpu.width / engine.gpu.height); engine._remember(scene, camera); };
  const check = (why, change, restore = () => {}) => {
    drawn();
    assert.equal(engine._idle(scene, camera), true, `settled, before: ${why}`);
    change();
    assert.equal(engine._idle(scene, camera), false, why);
    restore();
  };

  assert.equal(engine._idle(scene, camera), false, 'nothing drawn yet');
  check('a node moved', () => node.setPosition(1, 0, 0));
  check('the camera moved', () => camera.position.set([0, 1, 5]));
  check('a setting changed', () => { engine.renderer.exposure = 2; });
  check('a LUT was swapped for another', () => { engine.renderer.post.grading = { lut: {} }; }, () => {});
  check('a light changed', () => scene.addLight({ position: [0, 1, 0], radius: 2 }));
  check('the canvas resized', () => { engine.gpu.width = 640; });
  check('debug lines are queued', () => { engine.renderer.debug.count = 2; }, () => { engine.renderer.debug.count = 0; });
  check('a pipeline set is still building', () => sets.set(2, { ready: false }), () => sets.delete(2));
  check('a material changed', () => scene.changedMaterials.set(0, {}), () => scene.changedMaterials.clear());
  check('the environment was swapped', () => { scene.environment = {}; });
  check('a shadow setting changed', () => { engine.renderer.shadows.lambda = 0.5; });
  check('order-independent transparency was switched on', () => { engine.renderer.oit = true; });
  check('antialiasing was asked for and is still building', () => { engine.renderer.post.fxaaPipeline = undefined; },
    () => { engine.renderer.post.fxaaPipeline = {}; });
  const emitter = scene.addEmitter({ rate: 0, lifetime: 2, size: 0.1 });
  check('particles were born', () => { scene.burst(emitter, 5); scene.advance(0.016); });
  scene.emitters.get(emitter.entity).owed = 0;   // the frame that births them settles what is owed
  scene.advance(2.1);
  drawn();
  assert.equal(engine._idle(scene, camera), true, 'the burst has lived out its lifetime');
  check('invalidate() was called', () => engine.invalidate());
});

test('a HUD is part of what run() compares before skipping a frame', () => {
  const engine = Object.create(Winding.prototype);
  engine.gpu = { width: 320, height: 240 };
  engine.renderer = {
    exposure: 1, fog: null, dof: null, skybox: true, shadowDistance: null, lightDistance: null, ao: null,
    debug: { count: 0, depthTest: true }, _variantSets: new Map(),
    post: { threshold: 1, knee: 0.5, filterRadius: 1, strength: 0.06, levels: 5, antialias: true, grading: null, fxaaPipeline: {} },
    upscaler: { pending: false },
    shadows: { lambda: 0.7, casterExtent: 4, normalBias: 1.5 },
  };
  engine._drawn = null;
  const scene = new Scene({ capacity: 16 });
  const camera = new Camera({ fovY: 1, near: 0.1 });
  camera.position.set([0, 0, 5]);
  const hudScene = new Scene({ capacity: 16 });
  const bar = hudScene.addShape({ size: [100, 10] });
  const hud = engine._hud('run', { hud: { scene: hudScene } });
  assert.equal(hud.camera.is2D, true, 'a plain Camera2D when none is given');
  assert.equal(engine._hud('run', { hud: { scene: hudScene } }), hud, 'and the same one each time');
  assert.throws(() => engine._hud('run', { hud: { scene: hudScene, camera } }), /^Error: run: the hud's camera must be a Camera2D/);
  assert.throws(() => engine._hud('renderFrame', { overlay: { scene: hudScene } }), /renderFrame: overlay is now hud/);

  const drawn = () => {
    for (const s of [scene, hudScene]) { s.update(); s.transforms.movedPending = false; }
    camera.update(engine.gpu.width / engine.gpu.height);
    hud.camera.update(1, engine.gpu.width, engine.gpu.height);
    engine._remember(scene, camera, hud);
  };
  drawn();
  assert.equal(engine._idle(scene, camera, hud), true, 'settled');
  assert.equal(engine._idle(scene, camera), false, 'the HUD went away');
  hudScene.setShape(bar, { color: [1, 0, 0, 1] });
  assert.equal(engine._idle(scene, camera, hud), false, 'a HUD shape changed');
  drawn();
  bar.setPosition(5, 5);
  assert.equal(engine._idle(scene, camera, hud), false, 'a HUD node moved');
  drawn();
  hud.camera.zoom = 2;
  assert.equal(engine._idle(scene, camera, hud), false, 'the HUD camera zoomed');
});

test('sprites are packed again only when something they are drawn from changed', () => {
  let packs = 0;
  const pass = Object.create(SpritePass.prototype);
  pass._data = null;
  pass._capacity = 1 << 20;
  pass._buffer = {};
  pass._params = new Float32Array(12);
  pass._paramsBuffer = {};
  pass.rhi = { queue: { writeBuffer(buffer) { if (buffer === pass._buffer) packs++; } } };
  const texture = { view: {}, width: 4, height: 4 };
  const scene = new Scene({ capacity: 8 });
  const sprite = scene.addSprite({ texture, position: [0, 0, -5] });
  scene.update();
  const camera = new Camera({ fovY: 1, near: 0.1 });
  camera.position.set([0, 0, 5]);
  camera.target.set([0, 0, 0]);
  camera.update(1);
  const frame = (moved = false) => pass.prepare(scene, camera, null, 320, 240, moved);

  frame(true);
  assert.equal(packs, 1);
  frame();
  assert.equal(packs, 1, 'nothing changed: the last pack stands');
  frame(true);
  assert.equal(packs, 2, 'a transform moved');
  camera.position.set([0, 1, 5]);
  camera.update(1);
  frame();
  assert.equal(packs, 3, 'the camera moved');
  scene.setSprite(sprite, { color: [1, 0, 0, 1] });
  frame();
  assert.equal(packs, 4, 'a sprite changed');
  pass.prepare(scene, camera, null, 640, 240, false);
  assert.equal(packs, 5, 'the viewport changed, which pixel-sized sprites read');
});

test('an emitter owes particles by its rate and carries time until a frame settles them', () => {
  const scene = new Scene({ capacity: 8 });
  const node = scene.addEmitter({ rate: 30, lifetime: [0.5, 2], size: 0.2, sizeEnd: 0 });
  const record = scene.emitters.get(node.entity);
  assert.deepEqual([...record.lifetime, ...record.speed, record.blend], [0.5, 2, 0, 0, 'additive']);
  assert.deepEqual([...record.colorEnd], [...record.color], 'the end colour is the start unless given');
  scene.advance(0.25);
  scene.advance(0.25);
  assert.deepEqual([record.owed, record.time], [15, 0.5]);
  scene.burst(node, 7);
  assert.equal(record.owed, 22);
  scene.setEmitter(node, { rate: 0 });
  const after = scene.emitters.get(node.entity);
  assert.deepEqual([after.owed, after.time, after.seed, after.rate], [22, 0.5, record.seed, 0], 'what it owes survives a change');
  assert.equal(ringCapacity(record, 22), 30 * 2 + 22, 'rate times the longest life, plus what is being born');
  assert.equal(ringCapacity(after, 0), 1);
  for (const [options, why] of [
    [{ size: 1 }, /lifetime is required/], [{ lifetime: 1 }, /size is required/],
    [{ lifetime: [2, 1], size: 1 }, /lifetime/], [{ lifetime: 0, size: 1 }, /lifetime must be a positive/],
    [{ lifetime: 1, size: 1, spread: 4 }, /spread/], [{ lifetime: 1, size: 1, direction: [0, 0, 0] }, /direction/],
    [{ lifetime: 1, size: 1, blend: 'cutout' }, /blend/], [{ lifetime: 1, size: 1, rate: -1 }, /rate/],
    [{ lifetime: 1, size: [1, 0] }, /sizeEnd is the size at death/],
  ]) assert.throws(() => scene.addEmitter(options), why);
  assert.throws(() => scene.burst(node, 1.5), /whole number/);
  scene.remove(node);
  assert.equal(scene.emitters.size, 0, 'removed with its node');
});

test('the emitter uniform lands where the WGSL struct reads it', () => {
  const scene = new Scene({ capacity: 4 });
  const node = scene.addEmitter({
    lifetime: [1, 3], size: 0.5, sizeEnd: 0.25, speed: [2, 4], spread: 0.5, radius: 0.75, drag: 0.1,
    direction: [0, 0, 1], acceleration: [0, -9, 0], color: [1, 2, 3, 4], colorEnd: [5, 6, 7, 8],
  });
  const record = scene.emitters.get(node.entity);
  const out = new Uint8Array(256 + 192);
  const world = new Float32Array(16).map((_, i) => i);
  packEmitter(out, 256, record, world, 0, { offset: 10, capacity: 20, head: 3 }, 4, 0.016, 99);
  const f = new Float32Array(out.buffer, 256, 48);
  const u = new Uint32Array(out.buffer, 256, 48);
  // Offsets from the struct: 64 offset.., 80 dt.., 96 direction, 112 speed,
  // 120 lifetime, 128 acceleration (w drag), 144 size, 160 and 176 colours.
  assert.deepEqual([...f.subarray(0, 16)], [...world]);
  assert.deepEqual([...u.subarray(16, 20)], [10, 20, 3, 4]);
  assert.deepEqual([Math.fround(0.016), 99, 0.75, 0.5], [f[20], u[21], f[22], f[23]]);
  assert.deepEqual([...f.subarray(24, 28)], [0, 0, 1, 0], 'no texture');
  assert.deepEqual([...f.subarray(28, 32)], [2, 4, 1, 3]);
  assert.deepEqual([...f.subarray(32, 36)], [0, -9, 0, Math.fround(0.1)]);
  assert.deepEqual([...f.subarray(36, 38)], [0.5, 0.25]);
  assert.deepEqual([...f.subarray(40, 48)], [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('a decal maps its box onto [-1, 1] and faces along its +Z', () => {
  const texture = { view: {}, width: 8, height: 8 };
  const scene = new Scene({ capacity: 8 });
  // A 4 x 2 x 1 box at (5, 0, 0), turned to project straight down.
  const node = scene.addDecal({ texture, size: [4, 2, 1], color: [1, 0.5, 0.25, 0.75], position: [5, 0, 0] });
  node.setAxisAngle([1, 0, 0], -Math.PI / 2);
  scene.addDecal({ texture, size: [1, 1, 1] });
  scene.update();
  const out = new Float32Array(2 * DECAL_FLOATS);
  const spheres = new Float32Array(8);
  assert.equal(packDecals(scene, out, new Map([[texture, 3]]), spheres), 2);
  const apply = (p) => [0, 1, 2].map((r) => out[r] * p[0] + out[4 + r] * p[1] + out[8 + r] * p[2] + out[12 + r]);
  // Its x runs along world x; its y, after the turn, along world -z.
  vecClose(apply([7, 0, 0]), [1, 0, 0], 1e-6, 'the far x face');
  vecClose(apply([5, 0, -1]), [0, 1, 0], 1e-6, 'the far y face');
  vecClose(apply([5, -0.5, 0]), [0, 0, -1], 1e-6, 'down, along its -Z, half its depth: the box is centred on the node');
  vecClose(out.subarray(20, 23), [0, 1, 0], 1e-6, 'it paints what faces up');
  assert.equal(out[23], 3, 'its texture layer');
  vecClose(out.subarray(16, 20), [1, 0.5, 0.25, 0.75]);
  // Clustered by the sphere through its corners: half of the 4 x 2 x 1 diagonal.
  vecClose(spheres.subarray(0, 4), [5, 0, 0, Math.hypot(2, 1, 0.5)], 1e-6, 'its bounding sphere');
  // Sheared by a non-uniform scale under a turned parent, the farthest corner
  // is not the axis-aligned half-diagonal; all eight corners bound it.
  const parent = scene.createNode();
  parent.setScale(3, 1, 1);
  const sheared = scene.addDecal({ texture, size: [1, 2, 0.5], parent });
  sheared.setAxisAngle([0, 0, 1], Math.PI / 5);
  scene.update();
  const three = new Float32Array(12);
  packDecals(scene, new Float32Array(3 * DECAL_FLOATS), new Map([[texture, 0]]), three);
  const w = scene.transforms.world.subarray(handleIndex(sheared.entity) * 16);
  let farthest = 0;
  for (let corner = 0; corner < 8; corner++) {
    const h = [0.5, 1, 0.25].map((v, a) => (corner >> a) & 1 ? v : -v);
    farthest = Math.max(farthest, Math.hypot(...[0, 1, 2].map((c) => w[c] * h[0] + w[4 + c] * h[1] + w[8 + c] * h[2])));
  }
  assert.ok(Math.abs(three[11] - farthest) < 1e-5, `sheared: ${three[11]} against ${farthest}`);
  scene.remove(sheared);
  for (const [options, why] of [
    [{ size: [1, 1, 1] }, /texture must be/], [{ texture, size: [1, 1] }, /size must be/],
    [{ texture, size: [1, 0, 1] }, /size must be/], [{ texture, size: [1, 1, 1], color: [1, 1, 1] }, /color/],
  ]) assert.throws(() => scene.addDecal(options), why);
  scene.remove(node);
  assert.equal(scene.decals.size, 1, 'removed with its node');
});

test('adding an asset creates entities, transforms and renderables', () => {
  const scene = new Scene({ capacity: 64 });
  const root = scene.add(fakeAsset());

  assert.ok(root instanceof Node);
  assert.equal(scene.entities.liveCount - EMPTY_SCENE_ENTITIES, 1);
  assert.equal(scene.renderableCount, 1);
  assert.equal(scene.renderablePrimitive[0].indexCount, 36);
  assert.equal(scene.renderableMaterial[0], 0);
});

test('a mesh with several primitives becomes several renderables on one entity', () => {
  const scene = new Scene({ capacity: 64 });
  const root = scene.add(fakeAsset({ primitivesPerMesh: 3 }));

  assert.equal(scene.entities.liveCount - EMPTY_SCENE_ENTITIES, 1, 'still one node');
  assert.equal(scene.renderableCount, 3, 'but three things to draw');
  for (let i = 0; i < 3; i++) {
    assert.equal(scene.renderableEntity[i], root.entity);
  }
});

test('local bounds are copied in so culling never touches the asset again', () => {
  const scene = new Scene({ capacity: 64 });
  scene.add(fakeAsset());
  vecClose(scene.localMin.subarray(0, 3), [-1, -1, -1]);
  vecClose(scene.localMax.subarray(0, 3), [1, 1, 1]);
});

test('a multi-root asset gets one wrapper node', () => {
  // So the caller always gets a single handle back and can move the whole
  // thing with one setPosition.
  const scene = new Scene({ capacity: 64 });
  const root = scene.add(fakeAsset({
    nodes: [node('a'), node('b')],
    roots: [0, 1],
  }));

  assert.equal(scene.entities.liveCount - EMPTY_SCENE_ENTITIES, 3, 'two nodes plus the wrapper');
  assert.equal(root.children().length, 2);
});

test('a single-root asset is returned directly, with no wrapper', () => {
  const scene = new Scene({ capacity: 64 });
  const root = scene.add(fakeAsset());
  assert.equal(scene.entities.liveCount - EMPTY_SCENE_ENTITIES, 1, 'no extra node invented');
  assert.equal(scene.transforms.parent[handleIndex(root.entity)], NO_PARENT);
});

test('nested nodes become a real transform hierarchy', () => {
  const scene = new Scene({ capacity: 64 });
  const root = scene.add(fakeAsset({
    nodes: [
      node('parent', { position: [10, 0, 0], children: [1] }),
      node('child', { position: [0, 5, 0], mesh: 0 }),
    ],
    roots: [0],
  }));

  scene.update();
  const child = root.children()[0];
  vecClose(child.worldPosition(vec3Create()), [10, 5, 0]);
});

test('adding under a parent node inherits its transform', () => {
  const scene = new Scene({ capacity: 64 });
  const group = scene.createNode();
  group.setPosition(100, 0, 0);

  const child = scene.add(fakeAsset({ nodes: [node('x', { position: [1, 2, 3], mesh: 0 })] }), { parent: group });
  scene.update();
  vecClose(child.worldPosition(vec3Create()), [101, 2, 3]);
});

// ------------------------------------------------------------------- node

console.log('\nnode');

test('setters mark the transform dirty, so the world matrix follows', () => {
  // The reason there is no node.position.x -- a live view into the column
  // would not do this, and the node would silently not move.
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode();
  scene.update();

  n.setPosition(4, 5, 6);
  assert.equal(scene.update(), 1, 'exactly one transform recomposed');
  vecClose(n.worldPosition(vec3Create()), [4, 5, 6]);
});

test('setters chain', () => {
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode().setPosition(1, 0, 0).setScale(2);
  scene.update();
  vecClose(n.worldPosition(vec3Create()), [1, 0, 0]);
  assert.equal(scene.transforms.scale[handleIndex(n.entity) * 3], 2);
});

test('setScale with one argument scales uniformly', () => {
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode().setScale(3);
  const o = handleIndex(n.entity) * 3;
  assert.deepEqual([...scene.transforms.scale.subarray(o, o + 3)], [3, 3, 3]);
});

test('setEuler converts at the edge and stores a quaternion', () => {
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode().setEuler(Math.PI / 2, 0, 0);

  const o = handleIndex(n.entity) * 4;
  const stored = scene.transforms.rotation.subarray(o, o + 4);
  vecClose(stored, [0, Math.SQRT1_2, 0, Math.SQRT1_2], 1e-5, '90 degrees of yaw');

  // And it means what it says: yaw maps +X to -Z.
  vecClose(vec3TransformQuat(vec3Create(), vec3Create(1, 0, 0), stored), [0, 0, -1], 1e-5);
});

test('setParent reparents, and null detaches to the root', () => {
  const scene = new Scene({ capacity: 16 });
  const a = scene.createNode().setPosition(10, 0, 0);
  const b = scene.createNode().setPosition(1, 0, 0);

  b.setParent(a);
  scene.update();
  vecClose(b.worldPosition(vec3Create()), [11, 0, 0]);

  b.setParent(null);
  scene.update();
  vecClose(b.worldPosition(vec3Create()), [1, 0, 0]);
});

test('identity is by entity, not by object', () => {
  // Two cursors onto the same entity are different JS objects on purpose --
  // the object is a handle with methods, not the storage.
  const scene = new Scene({ capacity: 16 });
  const a = scene.createNode();
  const b = scene.node(a.entity);

  assert.notEqual(a, b, 'different objects');
  assert.equal(a.entity, b.entity, 'same entity');

  b.setPosition(7, 0, 0);
  scene.update();
  vecClose(a.worldPosition(vec3Create()), [7, 0, 0], EPS, 'both see the same data');
});

test('worldPosition reads the LAST update, not pending edits', () => {
  // Stated plainly because it is the one surprising thing about a cursor API.
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode();
  scene.update();

  n.setPosition(9, 9, 9);
  vecClose(n.worldPosition(vec3Create()), [0, 0, 0], EPS, 'stale until update()');
  scene.update();
  vecClose(n.worldPosition(vec3Create()), [9, 9, 9]);
});

// ---------------------------------------------------------------- removal

console.log('\nremoval');

test('removing a node takes its subtree and its renderables', () => {
  const scene = new Scene({ capacity: 64 });
  const keep = scene.add(fakeAsset());
  const doomed = scene.add(fakeAsset({
    nodes: [node('p', { children: [1], mesh: 0 }), node('c', { mesh: 0 })],
    roots: [0],
  }));

  assert.equal(scene.renderableCount, 3);
  assert.equal(scene.entities.liveCount - EMPTY_SCENE_ENTITIES, 3);

  doomed.destroy();

  assert.equal(scene.renderableCount, 1, 'both of its renderables went');
  assert.equal(scene.entities.liveCount - EMPTY_SCENE_ENTITIES, 1);
  assert.equal(scene.renderableEntity[0], keep.entity, 'the survivor is intact');
  assert.equal(doomed.alive, false);
});

test('every scene counts what draws a primitive, so unload knows when it is free', () => {
  const asset = fakeAsset({
    nodes: [node('p', { children: [1], mesh: 0 }), node('c', { mesh: 0 })],
    roots: [0],
  });
  const primitive = asset.meshes[0].primitives[0];
  const a = new Scene({ capacity: 16 });
  const b = new Scene({ capacity: 16 });

  const first = a.add(asset);
  const second = a.add(asset);
  b.add(asset);
  assert.equal(primitive.instances, 6, 'two nodes each, three adds, two scenes');

  first.destroy();
  assert.equal(primitive.instances, 4);
  second.destroy();
  assert.equal(primitive.instances, 2, 'the other scene still draws it');
});

await atest('create refuses an Environment, which belongs to another device', async () => {
  // It was accepted "to share", and could never work: create() makes a new
  // device every time, and the Environment's cubemaps live on the old one.
  // Refused before any device is asked for, which is why Node can check it.
  const foreign = Object.create(Environment.prototype);
  await assert.rejects(() => Winding.create(null, { environment: foreign }), /belongs to the engine/);
});

test('an unloaded asset is refused, not drawn from freed buffers', () => {
  const asset = fakeAsset();
  asset.unloaded = true;
  const scene = new Scene({ capacity: 8 });
  assert.throws(() => scene.add(asset), /unloaded/);
  assert.equal(scene.renderableCount, 0);
});

test('removal swap-removes, so renderable indices are not stable', () => {
  const scene = new Scene({ capacity: 64 });
  const first = scene.add(fakeAsset());
  const second = scene.add(fakeAsset());

  assert.equal(scene.renderableEntity[1], second.entity);
  first.destroy();
  assert.equal(scene.renderableCount, 1);
  assert.equal(scene.renderableEntity[0], second.entity, 'the last one moved down');
});

test('a stale node reports itself dead rather than acting on a reused slot', () => {
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode();
  const stale = scene.node(n.entity);

  n.destroy();
  assert.equal(stale.alive, false);

  // The slot gets reused, and the old handle still does not validate --
  // that is the generation counter doing its job.
  const fresh = scene.createNode();
  assert.equal(handleIndex(fresh.entity), handleIndex(stale.entity));
  assert.equal(stale.alive, false);
  assert.equal(fresh.alive, true);
});

test('a scene grows past its initial capacity instead of refusing the add', () => {
  // The constructor argument is a starting size, not a budget. Nothing a caller
  // could pass here is a number they had any way to know in advance.
  const scene = new Scene({ capacity: 2, renderableCapacity: 2 });
  const nodes = [];
  for (let i = 0; i < 40; i++) nodes.push(scene.add(fakeAsset()));

  assert.equal(scene.renderableCount, 40);
  assert.ok(scene.renderableCapacity >= 40);

  // Everything added before the growth has to still be intact and addressable.
  scene.update();
  for (let i = 0; i < 40; i++) {
    assert.equal(nodes[i].alive, true, `node ${i} did not survive growth`);
  }
  assert.equal(scene.renderablePrimitive[0].indexCount, 36, 'column 0 survived');
  assert.equal(scene.renderablePrimitive[39].indexCount, 36, 'column 39 was written');
});

test('growth preserves transform data already composed', () => {
  const scene = new Scene({ capacity: 2 });
  const first = scene.createNode().setPosition(7, 8, 9);
  scene.update();

  for (let i = 0; i < 50; i++) scene.createNode();   // forces several growths
  scene.update();

  const o = scene.transforms.worldOffset(first.entity);
  assert.deepEqual(
    [scene.transforms.world[o + 12], scene.transforms.world[o + 13], scene.transforms.world[o + 14]],
    [7, 8, 9],
    'the first node lost its transform when the columns were reallocated',
  );
});

test('a node added after growth defaults to no parent, not to entity 0', () => {
  // parent fills with NO_PARENT (-1), so a zeroed tail would silently make
  // every new node a child of whatever lives in slot 0.
  const scene = new Scene({ capacity: 2 });
  const root = scene.createNode().setPosition(100, 0, 0);
  for (let i = 0; i < 20; i++) scene.createNode();

  const late = scene.createNode().setPosition(1, 0, 0);
  scene.update();

  const o = scene.transforms.worldOffset(late.entity);
  assert.equal(scene.transforms.world[o + 12], 1, 'inherited a parent it never had');
  assert.equal(root.alive, true);
});

test('lights grow too', () => {
  const scene = new Scene({ capacity: 8, lightCapacity: 2 });
  for (let i = 0; i < 10; i++) {
    scene.addLight({ position: [i, 0, 0], color: [1, 1, 1], intensity: 1, radius: 1 });
  }
  // Lights are entities now, so this also grows the handle allocator and the
  // transform store past a capacity of 8 -- and positions arrive from those
  // transforms when a frame refreshes them, not at addLight.
  scene.update();
  scene.refreshLights();

  assert.equal(scene.lightCount, 10);
  assert.equal(scene.lights[0], 0, 'first light survived');
  assert.equal(scene.lights[9 * 16], 9, 'tenth light was written');
});

// ------------------------------------------------------------ euler angles

console.log('\neuler conversion');

test('yaw, pitch and roll map to the axes they should', () => {
  const q = quatCreate();
  const out = vec3Create();

  quatFromEuler(q, Math.PI / 2, 0, 0);
  vecClose(vec3TransformQuat(out, vec3Create(1, 0, 0), q), [0, 0, -1], 1e-5, 'yaw turns +X to -Z');

  quatFromEuler(q, 0, Math.PI / 2, 0);
  vecClose(vec3TransformQuat(out, vec3Create(0, 0, -1), q), [0, 1, 0], 1e-5, 'pitch looks up');

  quatFromEuler(q, 0, 0, Math.PI / 2);
  vecClose(vec3TransformQuat(out, vec3Create(1, 0, 0), q), [0, 1, 0], 1e-5, 'roll tilts');
});

test('the conversion produces unit quaternions', () => {
  const q = quatCreate();
  for (const angles of [[0.3, -1.2, 2.0], [Math.PI, Math.PI / 3, -Math.PI / 4], [0, 0, 0]]) {
    quatFromEuler(q, ...angles);
    close(Math.hypot(q[0], q[1], q[2], q[3]), 1, 1e-6, `angles ${angles}`);
  }
});

test('YXZ order keeps yaw and pitch independent', () => {
  // The property a look-around control needs: yawing then pitching must not
  // introduce roll, which a different order would.
  const q = quatCreate();
  quatFromEuler(q, 0.9, 0.4, 0);

  // The camera's right vector must stay level -- no roll means no Y component.
  const right = vec3TransformQuat(vec3Create(), vec3Create(1, 0, 0), q);
  close(right[1], 0, 1e-6, 'horizon stays level');
});

// ------------------------------------------------- workers from another origin

console.log('\nworkers from another origin');

/** Records what it was constructed with, so the decision is observable. */
function fakeWorker() {
  const built = [];
  class Fake {
    constructor(url, options) {
      built.push({ url: String(url), options });
    }
  }
  return { Fake, built };
}

const PAGE = 'https://app.example.com';
const CDN = 'https://cdn.jsdelivr.net/npm/winding@0.6.2/src/core/jobWorker.js';

await atest('a same-origin worker is constructed directly', async () => {
  // The path every existing user is on. A shim here would cost a fetch hop to
  // get around a restriction that is not there.
  const { Fake, built } = fakeWorker();
  const url = new URL(`${PAGE}/src/core/jobWorker.js`);

  createModuleWorker(url, { WorkerClass: Fake, origin: PAGE });

  assert.equal(built.length, 1);
  assert.equal(built[0].url, url.href, 'the real script, not a shim');
  assert.deepEqual(built[0].options, { type: 'module' });
});

await atest('a cross-origin worker is never handed to the constructor', async () => {
  // THE fix. `new Worker(crossOriginUrl)` throws -- it does not degrade -- and
  // an engine served from a CDN is cross-origin by definition. Worse, this only
  // happens on a page that set COOP and COEP, because that is the only case
  // where workers are spawned at all: better configuration, harder failure.
  const { Fake, built } = fakeWorker();

  createModuleWorker(new URL(CDN), { WorkerClass: Fake, origin: PAGE });

  assert.equal(built.length, 1);
  assert.notEqual(built[0].url, CDN, 'the cross-origin URL must not reach Worker');
  assert.ok(built[0].url.startsWith('blob:'), `expected a blob URL, got ${built[0].url}`);
  assert.deepEqual(built[0].options, { type: 'module' });
});

await atest('the shim imports the real script by absolute URL', async () => {
  // What the blob actually contains, read back rather than assumed. A module's
  // own imports go through CORS, which is what a CDN serves and the Worker
  // constructor does not.
  const { Fake, built } = fakeWorker();
  createModuleWorker(new URL(CDN), { WorkerClass: Fake, origin: PAGE });

  const source = await (await fetch(built[0].url)).text();
  assert.equal(source, `import ${JSON.stringify(CDN)};`);

  // And it is a real import statement, not a string that looks like one.
  assert.doesNotThrow(() => new Function(`return () => { ${''} }`));
  assert.ok(source.startsWith('import "') || source.startsWith("import '"),
    `the specifier must be quoted: ${source}`);
});

await atest('a URL containing a quote cannot break out of the import', async () => {
  // JSON.stringify rather than quotes by hand. A path is not a safe thing to
  // paste into source, and this one is pasted into a module that gets executed.
  const nasty = 'https://cdn.example.com/a"; globalThis.pwned = 1; import "b.js';
  const source = workerShimSource(nasty);

  assert.ok(source.includes('\\"'), 'the quote must be escaped');
  assert.equal(JSON.parse(source.slice('import '.length, -1)), nasty,
    'and the specifier must still round-trip to the original URL');
});

await atest('two workers for one script share a shim', async () => {
  const { Fake, built } = fakeWorker();
  createModuleWorker(new URL(CDN), { WorkerClass: Fake, origin: PAGE });
  createModuleWorker(new URL(CDN), { WorkerClass: Fake, origin: PAGE });

  assert.equal(built[0].url, built[1].url, 'one blob, however many workers');
});


// ------------------------------------------------------------ orbit control

console.log('\norbit control');

/** Enough of an element for the controller to attach to and be driven. */
function stubElement() {
  const handlers = new Map();
  return {
    handlers,
    addEventListener: (type, fn) => handlers.set(type, fn),
    removeEventListener: (type) => handlers.delete(type),
    setPointerCapture() {},
    hasPointerCapture: () => false,
    releasePointerCapture() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
    send(type, event) { handlers.get(type)?.({ preventDefault() {}, button: 0, pointerId: 1, ...event }); },
  };
}

/** Drag from one point to another with the left button held. */
function drag(element, fromX, fromY, toX, toY) {
  element.send('pointerdown', { clientX: fromX, clientY: fromY });
  element.send('pointermove', { clientX: toX, clientY: toY });
  element.send('pointerup', { clientX: toX, clientY: toY });
}

test('dragging up shows the underside, like grabbing the object', () => {
  // The bug: this was inverted relative to the horizontal drag, so turning the
  // object left and right felt like turning the OBJECT and tilting felt like
  // moving the CAMERA. Two metaphors in one gesture, which reads as the model
  // being hinged behind itself.
  //
  // Grab the front of a ball and pull up: the front goes over the top and the
  // UNDERSIDE rotates toward you. So the camera has to go down.
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const element = stubElement();
  const controller = new OrbitController(camera, element, { distance: 10, pitch: 0, yaw: 0 });

  const before = camera.position[1];
  drag(element, 400, 400, 400, 300);     // 100px UP
  controller.update(0);                  // 0 snaps past the damping

  assert.ok(camera.position[1] < before,
    `drag up must lower the camera: ${before} -> ${camera.position[1]}`);
  controller.detach();
});

test('dragging down shows the top', () => {
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const element = stubElement();
  const controller = new OrbitController(camera, element, { distance: 10, pitch: 0, yaw: 0 });

  const before = camera.position[1];
  drag(element, 400, 300, 400, 400);     // 100px DOWN
  controller.update(0);

  assert.ok(camera.position[1] > before,
    `drag down must raise the camera: ${before} -> ${camera.position[1]}`);
  controller.detach();
});

test('both axes turn the object the way the hand moves', () => {
  // The property the vertical drag was breaking: one metaphor, not two. A
  // drag right and a drag up must both move the camera the OPPOSITE way, so
  // the surface under the cursor follows it.
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const element = stubElement();
  const controller = new OrbitController(camera, element, { distance: 10, pitch: 0, yaw: 0 });

  drag(element, 400, 400, 500, 400);     // 100px RIGHT
  controller.update(0);
  assert.ok(camera.position[0] < 0, `drag right must send the camera left: ${camera.position[0]}`);

  controller.detach();
});

test('pitch cannot reach the pole', () => {
  // At exactly straight up the up-vector is ambiguous and the view flips.
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const element = stubElement();
  const controller = new OrbitController(camera, element, { distance: 10, pitch: 0 });

  drag(element, 400, 400, 400, 9999);    // absurdly far down
  controller.update(0);
  assert.ok(Math.abs(controller.pitch) < Math.PI / 2,
    `pitch must stay off the pole: ${controller.pitch}`);
  controller.detach();
});

test('the controller and a bare camera agree on how far to back off', () => {
  // Two things frame, and they used to disagree: the controller worked the
  // distance out inline and left out the aspect term. One definition now.
  const a = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const b = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  a.update(0.5);            // portrait, where the missing aspect term showed
  b.update(0.5);

  const element = stubElement();
  const controller = new OrbitController(b, element, { distance: 99 });

  a.frameBounds([-1, -1, -1], [1, 1, 1]);
  controller.frameBounds([-1, -1, -1], [1, 1, 1]);

  const distanceOf = (c) => Math.hypot(
    c.position[0] - c.target[0], c.position[1] - c.target[1], c.position[2] - c.target[2],
  );
  close(distanceOf(b), distanceOf(a), 1e-4, 'same fit');
  controller.detach();
});


test('syncFromCamera reproduces the pose it adopted', () => {
  // The inverse of what update() does, so the two have to agree exactly.
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const controller = new OrbitController(camera, stubElement(), { distance: 6 });

  camera.position.set([3, 4, 5]);
  camera.target.set([1, 1, 1]);
  controller.syncFromCamera();

  vecClose(camera.position, [3, 4, 5], 1e-5, 'position survives the round trip');
  vecClose(camera.target, [1, 1, 1], 1e-5, 'target too');
  controller.detach();
});

test('a camera moved directly is no longer snapped back', () => {
  // THE gap. The controller rebuilds position from yaw/pitch/distance every
  // frame, so anything that moved the camera itself lasted exactly one frame
  // and was then silently undone. There was no way to hand control back.
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const controller = new OrbitController(camera, stubElement(), { distance: 6 });

  camera.position.set([9, 2, 4]);
  camera.target.set([0, 0, 0]);
  controller.syncFromCamera();

  // Several frames of the normal loop, which is what used to undo it.
  for (let i = 0; i < 10; i++) controller.update(1 / 60);

  vecClose(camera.position, [9, 2, 4], 1e-4, 'still where it was put');
  controller.detach();
});

test('framing a camera then handing it back holds', () => {
  // The combination this exists for: Camera.frameBounds writes position and
  // target, which the controller owns. Without the sync the frame is gone by
  // the next frame.
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const controller = new OrbitController(camera, stubElement(), { distance: 50 });
  camera.update(1);

  camera.frameBounds([-1, -1, -1], [1, 1, 1]);
  const framed = [...camera.position];
  controller.syncFromCamera();
  for (let i = 0; i < 10; i++) controller.update(1 / 60);

  vecClose(camera.position, framed, 1e-4, 'the framing survived the controller');
  controller.detach();
});

test('a pose the controller cannot hold is adopted at the nearest it can', () => {
  // Straight down is refused on purpose: at the pole the up vector is
  // ambiguous and the view flips. Adopting it has to clamp, and saying so is
  // better than a silent flip later.
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const controller = new OrbitController(camera, stubElement(), { distance: 6 });

  camera.position.set([0, 10, 0]);       // directly above
  camera.target.set([0, 0, 0]);
  controller.syncFromCamera();

  assert.ok(Number.isFinite(controller.pitch), 'pitch must not be NaN');
  assert.ok(Math.abs(controller.pitch) < Math.PI / 2, `pitch stays off the pole: ${controller.pitch}`);
  assert.ok(Number.isFinite(camera.position[0]) && Number.isFinite(camera.position[2]));
  controller.detach();
});

test('a camera sitting on its own target keeps the angles it had', () => {
  // No offset means no direction to derive. Inventing one would spin the view
  // for no reason the user could see.
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  const controller = new OrbitController(camera, stubElement(), { distance: 6, yaw: 1.2, pitch: 0.3 });

  camera.position.set([5, 5, 5]);
  camera.target.set([5, 5, 5]);
  controller.syncFromCamera();

  close(controller.yaw, 1.2, EPS, 'yaw kept');
  close(controller.pitch, 0.3, EPS, 'pitch kept');
  vecClose(controller.target, [5, 5, 5], EPS, 'but the target moved');
  controller.detach();
});

test('a survivor keeps its own skin and morph when another renderable is removed', () => {
  // Removal swap-compacts the renderable columns. It moved entity, material
  // and bounds down but left skin and morph behind, so the survivor took the
  // deleted object's palette and weights: remove one character and another
  // starts wearing its pose.
  const scene = new Scene({ capacity: 16 });
  const primitive = fakeAsset().meshes[0].primitives[0];
  const doomed = scene.createNode();
  const survivor = scene.createNode();
  scene._addRenderable(doomed.entity, primitive, 0, 0);
  scene._addRenderable(survivor.entity, primitive, 1, 1);

  doomed.destroy();

  assert.equal(scene.renderableEntity[0], survivor.entity, 'it moved into slot 0');
  assert.equal(scene.renderableSkin[0], 1, 'with its own skin');
  assert.equal(scene.renderableMorph[0], 1, 'and its own weights');
});

// ------------------------------------------------ imported lights and cameras

console.log('\nimported lights and cameras');

/** A child node under a root, carrying whatever `extra` says. */
function carrierAsset(extra, assetExtra) {
  return {
    ...fakeAsset({
      nodes: [node('root', { children: [1] }), node('carrier', { position: [0, 1, 0], ...extra })],
      roots: [0],
    }),
    ...assetExtra,
  };
}

test('an imported light is its node, and follows the asset', () => {
  const scene = new Scene({ capacity: 16 });
  const root = scene.add(carrierAsset({ light: 0 }, {
    lights: [{ color: [1, 0.5, 0.25], intensity: 9, radius: 4, type: 'point', innerAngle: 0, outerAngle: 0.7 }],
  }));
  root.setPosition(5, 0, 0);
  scene.update();
  scene.refreshLights();

  assert.equal(scene.lightCount, 1);
  vecClose([...scene.lights.subarray(0, 4)], [5, 1, 0, 4], EPS, 'asset offset + node offset, radius');
  vecClose([...scene.lights.subarray(4, 8)], [1, 0.5, 0.25, 9], EPS, 'colour + intensity');
});

test('an imported spot turns with the asset', () => {
  const scene = new Scene({ capacity: 16 });
  const root = scene.add(carrierAsset({ light: 0 }, {
    lights: [{ color: [1, 1, 1], intensity: 1, radius: 4, type: 'spot', innerAngle: 0.1, outerAngle: 0.5 }],
  }));
  root.setAxisAngle([0, 1, 0], Math.PI / 2);   // quarter turn about +Y
  scene.update();
  scene.refreshLights();
  vecClose([...scene.lights.subarray(8, 11)], [-1, 0, 0], 1e-5, '-Z swung onto -X');
});

test('an unknown light type in an asset is skipped, not misread', () => {
  const scene = new Scene({ capacity: 16 });
  scene.add(carrierAsset({ light: 0 }, { lights: [null] }));
  assert.equal(scene.lightCount, 0);
});

// ------------------------------------------------------------ directional lights

console.log('\ndirectional lights');

/** Compose and refresh, as a frame would before reading the lights. */
function settleLights(scene) {
  scene.update();
  scene.refreshLights();
}

test('a new scene has no lights: its environment lights it until one is added', () => {
  const scene = new Scene({ capacity: 16 });
  settleLights(scene);
  assert.equal(scene.directionalCount, 0);
  assert.equal(scene.lightCount, 0);
  assert.equal(scene.entities.liveCount, 0, 'no node made on its behalf');
  assert.equal('sun' in scene, false, 'and nothing called the sun');
});

test('aiming and recolouring a directional light is aiming and recolouring its node', () => {
  const scene = new Scene({ capacity: 16 });
  const light = scene.addLight({ type: 'directional', direction: [-0.35, -0.55, -0.45], intensity: 3 });
  light.setDirection(0, -1, 0);
  scene.setLight(light, { color: [1, 0.5, 0.25], intensity: 2 });
  settleLights(scene);
  vecClose(scene.directionals.subarray(0, 3), [0, -1, 0], 1e-6, 'straight down: the degenerate look-along case');
  vecClose(scene.directionals.subarray(4, 7), [2, 1, 0.5], 1e-6, 'colour at intensity');
  assert.equal(scene.lightCount, 0, 'not a clustered light');

  scene.setLight(light, { intensity: 4 });
  settleLights(scene);
  vecClose(scene.directionals.subarray(4, 7), [4, 2, 1], 1e-6, 'partial: the colour stayed');
});

test('a directional light turns with its parent, so a day cycle is one rotating node', () => {
  const scene = new Scene({ capacity: 16 });
  const sky = scene.createNode();
  const light = scene.addLight({ type: 'directional', direction: [0, 0, -1] });
  light.setParent(sky);
  sky.setAxisAngle([1, 0, 0], -Math.PI / 2);   // tip -Z down to -Y
  settleLights(scene);
  vecClose(scene.directionals.subarray(0, 3), [0, -1, 0], 1e-5);
});

test('every light casts or not by the same switch, defaulting by what it costs', () => {
  // No light is the sun, by kind, order or brightness. A directional light
  // casts unless told not to; a point or spot light does not unless told to.
  // A file's lights take the same defaults, since glTF has no say in it.
  const scene = new Scene({ capacity: 16 });
  const key = scene.addLight({ type: 'directional', direction: [0, -1, 0], intensity: 10 });
  const fill = scene.addLight({ type: 'directional', direction: [1, -1, 0], intensity: 1, castShadow: false });
  const lamp = scene.addLight({ position: [0, 2, 0] });
  const torch = scene.addLight({ position: [1, 2, 0], castShadow: true });
  const imported = scene.add(carrierAsset({ light: 0 }, {
    lights: [{ type: 'directional', color: [1, 0, 0], intensity: 5 }],
  }));
  const casters = scene.shadowCasters;
  assert.ok(casters.has(key.entity), 'a directional light casts by default');
  assert.ok(!casters.has(fill.entity), 'unless told not to');
  assert.ok(!casters.has(lamp.entity), 'a point light does not by default');
  assert.ok(casters.has(torch.entity), 'unless told to');
  // The light is on the asset's child, the carrier node.
  const carrier = scene.childrenOf(imported)[0];
  assert.ok(casters.has(carrier.entity), 'a file\'s directional light casts, as one added by hand does');

  scene.setLight(fill, { castShadow: true });
  scene.setLight(key, { castShadow: false });
  assert.ok(casters.has(fill.entity) && !casters.has(key.entity), 'and the switch works both ways, for every kind');

  settleLights(scene);
  assert.equal(scene.directionalCount, 3, 'every directional light lights, casting or not');
  key.destroy();
  fill.destroy();
  imported.destroy();
  settleLights(scene);
  assert.equal(scene.directionalCount, 0);
  assert.equal(casters.size, 1, 'only the torch is left casting');
});

test('the packed directionals grow past their starting size, with their entities', () => {
  const scene = new Scene({ capacity: 16 });
  const lights = [];
  for (let i = 0; i < 9; i++) {
    lights.push(scene.addLight({ type: 'directional', direction: [0, -1, 0], intensity: 0.1 * (i + 1) }));
  }
  settleLights(scene);
  assert.equal(scene.directionalCount, 9);
  assert.ok(scene.directionals.length >= 9 * DIRECTIONAL_FLOATS);
  assert.deepEqual(scene.directionalEntity, lights.map((l) => l.entity), 'in the order they were added');
});

test('addLight refuses a type it does not have', () => {
  const scene = new Scene({ capacity: 16 });
  assert.throws(() => scene.addLight({ type: 'area' }), /^Error: addLight: .*point, spot or directional/);
});

test('addLight and setLight check the values they are given, and name the call', () => {
  const scene = new Scene({ capacity: 16 });
  assert.throws(() => scene.addLight({ color: [1, 1] }), /^Error: addLight: color/);
  assert.throws(() => scene.addLight({ color: [1, -1, 1] }), /addLight: color/);
  assert.throws(() => scene.addLight({ intensity: NaN }), /addLight: intensity/);
  assert.throws(() => scene.addLight({ radius: 0 }), /addLight: radius must be positive/);
  assert.throws(() => scene.addLight({ direction: [0, -1, 0], innerAngle: 0.6, outerAngle: 0.5 }), /addLight: angles/);
  assert.throws(() => scene.addLight({ direction: [0, -1, 0], outerAngle: 2 }), /addLight: angles/);
  assert.equal(scene.lightCount, 0, 'nothing half-added');

  const spot = scene.addLight({ direction: [0, -1, 0] });
  const sun = scene.addLight({ type: 'directional', direction: [0, -1, 0] });
  const bulb = scene.addLight({});
  assert.throws(() => scene.setLight(spot, { intensity: -1 }), /^Error: setLight: intensity/);
  assert.throws(() => scene.setLight(spot, { innerAngle: 0.6 }), /setLight: angles/, 'checked against the outer angle it keeps');
  assert.throws(() => scene.setLight(spot, { type: 'point' }), /setLight: type can't change/);
  assert.throws(() => scene.setLight(spot, { position: [1, 1, 1] }), /setLight: position can't change here; use the node/);
  assert.throws(() => scene.setLight(sun, { radius: 4 }), /a directional light has no radius/);
  assert.throws(() => scene.setLight(bulb, { outerAngle: 0.4 }), /a point light has no outerAngle/);
  scene.setLight(spot, { innerAngle: 0.1, outerAngle: 0.3 });
  assert.equal(scene.lightOf(spot).outerAngle, Math.fround(0.3));
});

test('a transform setter that throws leaves the node as it was', () => {
  const scene = new Scene({ capacity: 16 });
  const node = scene.createNode().setPosition(1, 2, 3).setScale(2);
  assert.throws(() => node.setPosition(4, NaN), /setPosition: non-finite/);
  assert.throws(() => node.setScale(Infinity), /setScale: non-finite/);
  assert.throws(() => node.setRotation([0, 0, NaN, 1]), /setRotation: non-finite/);
  const o = handleIndex(node.entity);
  assert.deepEqual([...scene.transforms.position.subarray(o * 3, o * 3 + 3)], [1, 2, 3]);
  assert.deepEqual([...scene.transforms.scale.subarray(o * 3, o * 3 + 3)], [2, 2, 2]);
  assert.deepEqual([...scene.transforms.rotation.subarray(o * 4, o * 4 + 4)], [0, 0, 0, 1]);
});

test('setDirection keeps the node upright, where the shortest turn would roll it', () => {
  // Up and to the side at once is the case the shortest turn gets wrong: it
  // tips the node about its own axis. An upright node's +X stays level.
  const scene = new Scene({ capacity: 16 });
  const node = scene.createNode();
  node.setDirection(1, 1, -1);
  scene.update();
  const m = scene.transforms.world;
  const o = handleIndex(node.entity) * 16;
  close(m[o + 1], 0, 1e-6, 'right axis has no vertical component');
  const f = [-m[o + 8], -m[o + 9], -m[o + 10]];
  vecClose(f, [1, 1, -1].map((v) => v / Math.sqrt(3)), 1e-6, 'and -Z is where it was sent');
});

test('removing an asset removes its lights and its cameras', () => {
  const scene = new Scene({ capacity: 16 });
  const kept = scene.add(carrierAsset({ camera: 0 }, { cameras: [{ orthographic: false, fovY: 1, near: 0.1 }] }));
  const doomed = scene.add(carrierAsset({ light: 0, camera: 0 }, {
    lights: [{ color: [1, 1, 1], intensity: 1, radius: 4, type: 'point' }],
    cameras: [{ orthographic: false, fovY: 1, near: 0.1 }],
  }));
  assert.equal(scene.cameras.length, 2);

  doomed.destroy();
  assert.equal(scene.lightCount, 0);
  assert.equal(scene.cameras.length, 1, 'only the doomed asset\'s camera went');
  assert.ok(scene.cameras[0].following.alive, 'and the one left still has a live node');
  assert.ok(kept.alive);
});

test('an imported camera sits on its node and looks down its -Z', () => {
  const scene = new Scene({ capacity: 16 });
  const root = scene.add(carrierAsset({ camera: 0 }, {
    cameras: [{ orthographic: false, fovY: 0.8, near: 0.05 }],
  }));
  root.setPosition(0, 0, 10);
  scene.update();

  const [camera] = scene.cameras;
  assert.equal(camera.fovY, 0.8);
  assert.equal(camera.near, 0.05);
  camera.update(1);
  vecClose(camera.position, [0, 1, 10], EPS, 'on the node');
  const dz = camera.target[2] - camera.position[2];
  assert.ok(dz < 0 && Math.abs(camera.target[0]) < EPS, 'looking down -Z');
});

test('an imported orthographic camera shows the height the file asked for', () => {
  // Orthographic height comes from distance to target, and following keeps
  // that distance. The scene places it so the view is exactly 2 * ymag tall.
  const scene = new Scene({ capacity: 16 });
  scene.add(carrierAsset({ camera: 0 }, {
    cameras: [{ orthographic: true, near: 0.01, far: 40, halfHeight: 3 }],
  }));
  scene.update();
  const [camera] = scene.cameras;
  camera.update(1);
  close(camera.orthographicHalfHeight(), 3, 1e-5, 'ymag');
  assert.equal(camera.far, 40);
});

// --------------------------------------------------------------- follow

console.log('\ncamera follow');

test('a following camera goes where its node goes, and points where it points', () => {
  const scene = new Scene({ capacity: 16 });
  const car = scene.createNode();
  const mount = scene.createNode({ parent: car });
  mount.setPosition(0, 2, 6);
  const camera = new Camera().follow(mount);

  car.setPosition(10, 0, 0);
  car.setAxisAngle([0, 1, 0], Math.PI / 2);   // car turns to face -X
  scene.update();
  camera.update(1);

  // The mount's offset (0, 2, 6) turned a quarter about +Y is (6, 2, 0), and
  // its -Z now points down -X: behind the car, looking the way it faces.
  vecClose(camera.position, [16, 2, 0], 1e-5, 'behind the car');
  const forward = [0, 1, 2].map((i) => camera.target[i] - camera.position[i]);
  const length = Math.hypot(...forward);
  vecClose(forward.map((v) => v / length), [-1, 0, 0], 1e-5, 'facing where the car faces');
  vecClose(camera.up, [0, 1, 0], 1e-5, 'up is the node\'s +Y');
});

test('scale on the node does not leak into the view', () => {
  const scene = new Scene({ capacity: 16 });
  const mount = scene.createNode();
  mount.setScale(3);
  const camera = new Camera().follow(mount);
  const before = Math.hypot(...[0, 1, 2].map((i) => camera.position[i] - camera.target[i]));
  scene.update();
  camera.update(1);
  const after = Math.hypot(...[0, 1, 2].map((i) => camera.position[i] - camera.target[i]));
  close(after, before, 1e-5, 'distance to target is kept, not scaled');
  close(Math.hypot(...camera.up), 1, 1e-6, 'up stays unit');
});

test('a destroyed node ends the follow and leaves the camera where it was', () => {
  const scene = new Scene({ capacity: 16 });
  const mount = scene.createNode();
  mount.setPosition(1, 2, 3);
  const camera = new Camera().follow(mount);
  scene.update();
  camera.update(1);

  mount.destroy();
  camera.update(1);
  assert.equal(camera.following, null);
  vecClose(camera.position, [1, 2, 3], EPS, 'stayed put');
});

test('an orbit controller stands aside while its camera follows, and takes back over', () => {
  const scene = new Scene({ capacity: 16 });
  const mount = scene.createNode();
  mount.setPosition(0, 0, 4);
  const camera = new Camera();
  const element = stubElement();
  const controller = new OrbitController(camera, element, { distance: 10, pitch: 0, yaw: 0 });

  camera.follow(mount);
  scene.update();
  camera.update(1);
  drag(element, 400, 300, 500, 300);
  controller.update(0);
  // Checked BEFORE the camera's own update, which would paper over a write:
  // picking reads camera.position directly, so a controller that moved it
  // would aim the pick ray from the drag's pose rather than the node's.
  vecClose(camera.position, [0, 0, 4], EPS, 'the node is in charge, not the drag');

  // Handing back is the documented two steps, and nothing jumps.
  camera.follow(null);
  controller.syncFromCamera();
  controller.update(0);
  vecClose(camera.position, [0, 0, 4], 1e-5, 'the controller adopted the pose');
});

// ------------------------------------------------ hierarchy and lifecycle

console.log('\nhierarchy and lifecycle');

/** A mesh node rigged to one joint, with one morph target. */
function riggedAsset() {
  const primitive = {
    materialId: 0,
    skinned: true,
    morphExtent: Float32Array.from([0.5]),
    bounds: { min: Float32Array.from([-1, -1, -1]), max: Float32Array.from([1, 1, 1]) },
  };
  return {
    meshes: [{ targetCount: 1, primitives: [primitive] }],
    nodes: [node('mesh', { mesh: 0, skin: 0, weights: [0.5], children: [1] }), node('joint')],
    roots: [0],
    skins: [{ name: 's', joints: [1], inverseBind: new Float32Array(16), jointRadii: Float32Array.from([1]) }],
  };
}

test('destroying a child and then its parent frees everything, and nothing else', () => {
  // The scene kept a second list of children that remove() never unlinked
  // from, so the parent's removal walked a dead handle: it threw halfway, or
  // -- once the slot was reused -- wiped an unrelated node's transform.
  const scene = new Scene({ capacity: 16 });
  const root = scene.add(fakeAsset({ nodes: [node('p', { children: [1] }), node('c', { mesh: 0 })], roots: [0] }));
  root.children()[0].destroy();
  const other = scene.createNode();            // takes the freed slot
  other.setPosition(5, 0, 0);
  root.destroy();

  assert.ok(other.alive, 'the unrelated node survived');
  scene.update();
  vecClose(other.worldPosition(vec3Create()), [5, 0, 0], EPS, 'with its transform intact');
});

test('destroying a node twice does nothing the second time', () => {
  const scene = new Scene({ capacity: 16 });
  const a = scene.createNode();
  a.destroy();
  const b = scene.createNode();                // same slot, new generation
  b.setPosition(1, 2, 3);
  a.destroy();

  assert.ok(b.alive);
  scene.update();
  vecClose(b.worldPosition(vec3Create()), [1, 2, 3], EPS, 'b was not touched');
});

test('every way of making a child makes one its parent takes with it', () => {
  const scene = new Scene({ capacity: 16 });
  const parent = scene.createNode();
  const made = scene.createNode({ parent });
  const added = scene.add(fakeAsset(), { parent });
  const moved = scene.createNode();
  moved.setParent(parent);

  parent.destroy();
  assert.equal(made.alive, false, 'createNode({ parent })');
  assert.equal(added.alive, false, 'add(asset, { parent })');
  assert.equal(moved.alive, false, 'setParent');
  assert.equal(scene.renderableCount, 0, 'the added asset took its renderables');
});

test('a node reparented away is not destroyed with its old parent', () => {
  const scene = new Scene({ capacity: 16 });
  const first = scene.createNode();
  const second = scene.createNode();
  const lamp = scene.addLight({ parent: first });
  lamp.setParent(second);
  first.destroy();
  assert.ok(lamp.alive);
  assert.equal(scene.lightCount, 1);
});

test('children come back in the order they were added, even in reused slots', () => {
  const scene = new Scene({ capacity: 16 });
  const spare = scene.createNode();            // a low slot, freed below
  const parent = scene.createNode();
  const first = scene.createNode({ parent });
  spare.destroy();
  const second = scene.createNode({ parent }); // reuses the lower slot
  assert.ok(handleIndex(second.entity) < handleIndex(first.entity), 'the setup puts second in a lower slot');
  assert.deepEqual(parent.children().map((c) => c.entity), [first.entity, second.entity]);
});

test('a failed add leaves the scene exactly as it was', () => {
  const scene = new Scene({ capacity: 16 });
  const before = scene.entities.liveCount;

  const badSkin = riggedAsset();
  badSkin.skins[0].joints = [5];               // a node the default scene never reaches
  badSkin.nodes.push(node('x'), node('x'), node('x'), node('x'));
  assert.throws(() => scene.add(badSkin), /not in the asset's default scene/);

  const notATree = fakeAsset({
    nodes: [node('a', { mesh: 0, children: [2] }), node('b', { children: [2] }), node('c', { mesh: 0 })],
    roots: [0, 1],
  });
  assert.throws(() => scene.add(notATree), /not a tree/);

  assert.equal(scene.entities.liveCount, before, 'no entities left behind');
  assert.equal(scene.renderableCount, 0);
  assert.equal(scene.skins.length, 0);
  assert.equal(scene.morphs.length, 0);
  // And the next add is not poisoned by what the failed one left pending.
  scene.add(fakeAsset());
  assert.equal(scene.renderableCount, 1);
});

test('removing an asset frees its skins and morph weights', () => {
  const scene = new Scene({ capacity: 16 });
  for (let i = 0; i < 20; i++) scene.add(riggedAsset()).destroy();
  assert.equal(scene.skins.length, 0);
  assert.equal(scene.morphs.length, 0);
});

test('a survivor still reads its own skin and weights after another is removed', () => {
  // Removal swap-compacts the skin and morph arrays. Whatever moves into a
  // gap must take every index that pointed at it along.
  const scene = new Scene({ capacity: 32 });
  const doomed = scene.add(riggedAsset());
  const survivor = scene.add(riggedAsset());
  survivor.weights[0] = 0.9;
  doomed.destroy();

  assert.equal(scene.skins.length, 1);
  assert.equal(scene.morphs.length, 1);
  const r = [...Array(scene.renderableCount).keys()].find((i) => scene.renderableEntity[i] === survivor.entity);
  assert.equal(scene.skins[scene.renderableSkin[r]].owner, survivor.entity, 'its own palette');
  assert.equal(scene.morphs[scene.renderableMorph[r]].owner, survivor.entity, 'its own weights');
  close(survivor.weights[0], 0.9, 1e-6, 'node.weights still finds them');
});

// ---------------------------------------------------------------------------
// 2D

test('Camera2D maps the world to screen pixels, and back', () => {
  // Default: a unit is a pixel, the origin is the top-left.
  const camera = new Camera2D().update(2, 200, 100);
  vecClose(camera.worldToScreen(10, 20), [10, 20]);

  // Centred on (500, 300), twice as large, turned a quarter.
  camera.position.set([500, 300]);
  camera.pivot.set([0.5, 0.5]);
  camera.zoom = 2;
  camera.angle = Math.PI / 2;
  camera.update(2, 200, 100);
  vecClose(camera.worldToScreen(500, 300), [100, 50], 1e-3, 'position sits at the pivot');
  vecClose(camera.worldToScreen(510, 300), [100, 30], 1e-3, 'turned and scaled');
  vecClose(camera.screenToWorld(100, 30), [510, 300], 1e-3, 'and back');

  // The view projection takes the corners of the screen to clip space's.
  const p = camera.projection;
  close(p[0] * 0 + p[12], -1);
  close(p[5] * 100 + p[13], -1);

  // Snapped, the view's offset is whole pixels.
  const snapped = new Camera2D({ position: [0.3, 0.6], pixelSnap: true }).update(2, 200, 100);
  close(snapped.view[12], 0);
  close(snapped.view[13], -1);
  // And a unit is a whole number of screen pixels: on a 1.5x screen, 2 of them.
  snapped.update(2, 300, 150, 1.5);
  close(snapped.view[0], 2);
  snapped.zoom = 0.1;
  snapped.update(2, 300, 150, 1.5);
  close(snapped.view[0], 1, 1e-9, 'never less than one');
  snapped.pixelSnap = false;
  snapped.update(2, 300, 150, 1.5);
  close(snapped.view[0], 0.15, 1e-6, 'unsnapped, exactly as asked');
});

test('spriteSheet gives frames in reading order', () => {
  const frames = spriteSheet({ columns: 4, rows: 2 });
  assert.equal(frames.length, 8);
  assert.deepEqual(frames[0], [0, 0, 0.25, 0.5]);
  assert.deepEqual(frames[5], [0.25, 0.5, 0.5, 1]);
  assert.deepEqual(spriteSheet({ columns: 4, rows: 2, first: 2, count: 3 })[0], [0.5, 0, 0.75, 0.5]);
  assert.throws(() => spriteSheet({ columns: 4, rows: 2, first: 6, count: 3 }), /not all in/);
  assert.throws(() => spriteSheet({ columns: 0 }), /whole numbers/);
});

test('2D draws by layer, then in the order added, merging runs that share a texture', () => {
  const A = { view: {}, width: 4, height: 4 };
  const B = { view: {}, width: 4, height: 4 };
  const scene = new Scene({ capacity: 8 });
  const a = scene.addSprite({ texture: B, layer: 1 });
  const b = scene.addSprite({ texture: A });
  const c = scene.addSprite({ texture: B });
  const d = scene.addSprite({ texture: A, layer: 2 });
  const { entries, runs } = order2D(scene);
  assert.deepEqual(entries.map((e) => e[2]), [b, c, a, d].map((n) => n.entity));
  assert.deepEqual(runs.map((r) => [r.first, r.count]), [[0, 1], [1, 2], [3, 1]]);
  assert.deepEqual(runs.map((r) => r.texture), [A, B, A]);
});

test('write2D: frame-sized by default, mirrored by a negative scale, animated by frame', () => {
  const sheet = { view: {}, width: 64, height: 32 };
  const scene = new Scene({ capacity: 8 });
  const node = scene.addSprite({ texture: sheet, position: [40, 30], animation: { frames: spriteSheet({ columns: 4, rows: 2 }), fps: 10 } });
  node.setAngle(0.5).setScale(-1, 1, 1);
  scene.update();
  const out = new Float32Array(SPRITE2D_FLOATS);
  const write = () => write2D(out, 0, scene.transforms.world, node.entity, scene.sprites.get(node.entity), null);

  write();
  vecClose(out.subarray(0, 2), [40, 30]);
  close(out[2], 0.5, 1e-5, 'the angle survives the mirror');
  close(out[4], -16, 1e-5, 'a frame is 16 texels wide, and mirrored');
  close(out[5], 16, 1e-5, 'and 16 tall');
  vecClose(out.subarray(8, 12), [0, 0, 0.25, 0.5]);

  scene.advance(0.25);
  assert.ok(scene.spritesChanged.has(node.entity), 'a new frame marks the sprite changed');
  write();
  vecClose(out.subarray(8, 12), [0.5, 0, 0.75, 0.5], 1e-6, 'frame 2 at 0.25 s and 10 fps');

  scene.setSprite(node, { size: [8, 4] });
  write();
  close(out[4], -8);
  close(out[5], 4);
});

test('setSprite reorders only for what changes the draw order', () => {
  const texture = { view: {}, width: 4, height: 4 };
  const scene = new Scene({ capacity: 8 });
  const node = scene.addSprite({ texture });
  const order = scene.spriteOrder;
  scene.setSprite(node, { color: [1, 0, 0, 1] });
  assert.equal(scene.spriteOrder, order, 'a colour is written in place');
  assert.ok(scene.spritesChanged.has(node.entity));
  scene.setSprite(node, { layer: 3 });
  assert.equal(scene.spriteOrder, order + 1, 'a layer moves it in the list');
  assert.equal(scene.sprites.get(node.entity).added, 0, 'and it keeps its place within the layer');
});

test('the 2D view rewrites only the slots of what moved or changed', () => {
  const view = Object.create(View2D.prototype);
  view.rhi = { queue: { writeBuffer() {} } };
  view._uniform = {};
  view._uniformData = new Float32Array(24);
  view._upload = () => {};
  view._data = new Float32Array(0);
  view._grow = function (count) {
    if (count * SPRITE2D_FLOATS > this._data.length) {
      const data = new Float32Array(count * SPRITE2D_FLOATS);
      data.set(this._data);
      this._data = data;
    }
  };
  view._buffer = {};
  view._tilemaps = new Map();
  view._slots = new Map();
  view._entries = [];
  view._runs = [];
  view._scene = null;
  view._order = -1;
  view.count = 0;
  view.written = 0;

  const texture = { view: {}, width: 4, height: 4 };
  const scene = new Scene({ capacity: 32 });
  const nodes = [];
  for (let i = 0; i < 10; i++) nodes.push(scene.addSprite({ texture, position: [i * 10, 0] }));
  const camera = new Camera2D();
  const frame = () => {
    scene.update();
    camera.update(1, 320, 240);
    const t = scene.transforms;
    view.prepare(scene, camera, 320, 240, t.movedPending ? t.moved : null);
    t.moved.fill(0, 0, t.capacity);
    t.movedPending = false;
    return view.written;
  };

  assert.equal(frame(), 10, 'the first frame writes every slot');
  assert.equal(frame(), 0, 'a still frame writes none');
  camera.position.set([50, 50]);
  assert.equal(frame(), 0, 'the camera is a uniform: no slot');
  close(view._uniformData[12], -50);
  nodes[4].setPosition(45, 5);
  assert.equal(frame(), 1, 'one node moved: one slot');
  close(view._data[4 * SPRITE2D_FLOATS], 45);
  nodes[9].setPosition(99, 1);
  nodes[0].setPosition(1, 1);
  assert.equal(frame(), 2, 'two movers at either end: two slots, not the ten from one to the other');
  nodes[2].setPosition(21, 1);
  nodes[4].setPosition(41, 1);
  assert.equal(frame(), 3, 'two a slot apart: one upload, the slot between with them');
  scene.setSprite(nodes[7], { color: [0, 1, 0, 1] });
  assert.equal(frame(), 1, 'one sprite changed: one slot');
  assert.deepEqual([...view._data.subarray(7 * SPRITE2D_FLOATS + 12, 7 * SPRITE2D_FLOATS + 16)], [0, 1, 0, 1], 'with what it changed to');
  scene.setSprite(nodes[7], { size: [9, 3] });
  frame();
  assert.deepEqual([...view._data.subarray(7 * SPRITE2D_FLOATS + 4, 7 * SPRITE2D_FLOATS + 6)], [9, 3], 'and again: a set call replaces the record');
  const bar = scene.addShape({ size: [10, 10] });
  frame();
  scene.setShape(bar, { size: [50, 10] });
  frame();
  assert.deepEqual([...view._data.subarray(10 * SPRITE2D_FLOATS + 4, 10 * SPRITE2D_FLOATS + 6)], [50, 10], 'a health bar shrinks and grows');
  const order = scene.spriteOrder;
  const late = scene.addSprite({ texture, color: [1, 0, 0, 1] });
  assert.equal(frame(), 1, 'a new sprite goes on the end: one slot');
  assert.equal(scene.spriteOrder, order, 'no rebuild');
  assert.deepEqual([...view._data.subarray(11 * SPRITE2D_FLOATS + 12, 11 * SPRITE2D_FLOATS + 16)], [1, 0, 0, 1]);
  assert.deepEqual(view._data.subarray(7 * SPRITE2D_FLOATS + 12, 7 * SPRITE2D_FLOATS + 16), Float32Array.of(0, 1, 0, 1), 'and the list is kept');
  assert.equal(scene.spritesChanged.size + scene.added2D.size, 0, 'and the changes are consumed');
  scene.remove(nodes[3]);
  assert.equal(frame(), 1, 'a removed sprite leaves a hole: one slot');
  assert.deepEqual([view._data[3 * SPRITE2D_FLOATS + 4], view._data[3 * SPRITE2D_FLOATS + 5]], [0, 0], 'drawn as nothing');
  assert.equal(view.count, 12, 'the list keeps its length until a rebuild');
  scene.setSprite(late, { color: [0, 0, 1, 1] });
  assert.equal(frame(), 1, 'what came after the hole is where it was');
  scene.addSprite({ texture, layer: -1 });
  assert.equal(frame(), 12, 'one added below the last layer goes in the middle: a rebuild, and the hole is gone');

  // A font's atlas grows when any scene asks for new glyphs, moving every
  // glyph: a view drawing its text rebuilds, though its own order is the same.
  const font = { metrics: { ascent: 0.8, descent: 0.2, glyphs: new Map([['a', { advance: 0.5, left: 0, width: 0.5, height: 1, descent: 0, rect: [0, 0, 0.5, 0.5] }], [' ', { advance: 0.5, left: 0, width: 0, height: 0, descent: 0, rect: null }]]) }, ensure() {}, texture: {} };
  const label = scene.addText({ font, text: 'aa', size: 10 });
  frame();
  assert.equal(frame(), 0, 'settled');
  const glyphs = view.count;
  scene.setText(label, { text: 'a a' });
  assert.equal(frame(), 2, 'as many glyphs: rewritten where they are');
  assert.equal(view.count, glyphs);
  close(view._data[(glyphs - 1) * SPRITE2D_FLOATS + 6], -0.5, 1e-6, 'from the new layout: its second glyph a space further on');
  font.texture = {};
  font.metrics.glyphs.get('a').rect = [0, 0, 0.25, 0.25];
  assert.equal(frame(), 14, 'the atlas grew: every slot again');
  assert.equal(view._runs.at(-1).texture, font.texture, 'drawn from the new texture');
});

test('2D: pick where it is drawn, position at the pivot, one empty tile, and idle between sprite frames', () => {
  // A camera's position is at its pivot however the view is turned.
  const camera = new Camera2D({ position: [30, 40], pivot: [0, 0], angle: Math.PI / 2 }).update(2, 200, 100);
  vecClose(camera.worldToScreen(30, 40), [0, 0], 1e-4, 'the top-left pivot, turned a quarter');
  camera.pivot.set([0.5, 0.9]);
  camera.angle = 0.5;
  camera.update(2, 200, 100);
  vecClose(camera.worldToScreen(30, 40), [100, 90], 1e-4);

  // A sprite's own angle on a node scaled unevenly: picked where it is drawn.
  // Drawn 20 x 3 = 60 along the node's x, then turned a quarter: x 99..101, y 70..130.
  const scene = new Scene({ capacity: 16 });
  const texture = { view: {}, width: 16, height: 16 };
  const bar = scene.addSprite({ texture, size: [20, 2], angle: Math.PI / 2, position: [100, 100] });
  bar.setScale(3, 1);
  const view = new Camera2D().update(1, 320, 240);
  assert.equal(scene.pick(view, 100, 125, 320, 240)?.node.entity, bar.entity, 'drawn there, so hit there');
  assert.equal(scene.pick(view, 102.5, 105, 320, 240), null, 'nothing drawn there');

  // An empty tile is 0, and only 0.
  const map = scene.addTilemap({ tileset: { view: {}, width: 32, height: 32 }, tileSize: [16, 16], columns: 2, rows: 2 });
  assert.throws(() => scene.setTile(map, 0, 0, 0x80000000), /^Error: setTile: tile 2147483648 flips no tile; an empty tile is 0/);

  // A sprite animation is a change when its frame turns, and nothing between.
  const sheet = { view: {}, width: 64, height: 16 };
  scene.addSprite({ texture: sheet, animation: { frames: spriteSheet({ columns: 4 }), fps: 10 } });
  let changes = scene.changes;
  scene.advance(0.01);
  assert.equal(scene.animating, false, 'a sprite animation does not keep the loop drawing');
  assert.equal(scene.changes, changes, 'no new frame yet');
  scene.advance(0.1);
  assert.ok(scene.changes > changes, 'a new frame is a change, so it is drawn');

  // An emitter seen in 2D is told so, to stay in the screen's plane.
  const puff = scene.addEmitter({ rate: 1, lifetime: 1, size: 1, direction: [0, -1], spread: Math.PI });
  const out = new Uint8Array(512);
  const f32 = new Float32Array(out.buffer);
  const ring = { offset: 0, capacity: 4, head: 0 };
  packEmitter(out, 0, scene.emitters.get(puff.entity), scene.transforms.world, 0, ring, 0, 0, 1, true);
  assert.equal(f32[27], 2, 'flat, no texture');
  packEmitter(out, 0, scene.emitters.get(puff.entity), scene.transforms.world, 0, ring, 0, 0, 1);
  assert.equal(f32[27], 0, 'in 3D, the cone');
});

test('a 2D view redraws when only the page zoom changes', () => {
  const engine = Object.create(Winding.prototype);
  engine.gpu = { width: 320, height: 240, pixelRatio: 1 };
  engine.renderer = {
    exposure: 1, fog: null, dof: null, skybox: true, shadowDistance: null, lightDistance: null, ao: null,
    debug: { count: 0, depthTest: true }, _variantSets: new Map(),
    post: { threshold: 1, knee: 0.5, filterRadius: 1, strength: 0.06, levels: 5, antialias: true, grading: null, fxaaPipeline: {} },
    upscaler: { pending: false },
    shadows: { lambda: 0.7, casterExtent: 4, normalBias: 1.5 },
  };
  engine._drawn = null;
  const scene = new Scene({ capacity: 16 });
  scene.addShape({ size: [50, 50] });
  const camera = new Camera2D();
  scene.update();
  scene.transforms.movedPending = false;
  camera.update(1, 320, 240, 1);
  engine._remember(scene, camera);
  assert.equal(engine._idle(scene, camera), true, 'settled');
  engine.gpu.pixelRatio = 2;   // the same canvas pixels, twice as many to a CSS pixel
  assert.equal(engine._idle(scene, camera), false, 'everything is drawn twice the size');
});

test('a tilemap keeps its ids, and marks the block that changed', () => {
  const tileset = { view: {}, width: 64, height: 32 };
  const scene = new Scene({ capacity: 8 });
  assert.throws(() => scene.addTilemap({ tileset, tileSize: [16, 16], columns: 4, rows: 2, tiles: [1, 2] }), /needs 8/);
  assert.throws(() => scene.addTilemap({ tileset, tileSize: [16.5, 16], columns: 4, rows: 2 }), /whole texels/);
  assert.throws(() => scene.addTilemap({ tileset, tileSize: [16, 16], columns: 0, rows: 2 }), /above zero/);

  const map = scene.addTilemap({ tileset, tileSize: [16, 16], columns: 4, rows: 3 });
  assert.equal(scene.tileAt(map, 1, 1), 0, 'empty to start');
  scene.setTile(map, 1, 1, 5);
  scene.setTiles(map, 2, 2, 2, [7, 8]);
  assert.equal(scene.tileAt(map, 1, 1), 5);
  assert.equal(scene.tileAt(map, 3, 2), 8);
  assert.equal(scene.tileAt(map, 3.7, 2.2), 8, 'a point inside a tile reads it');
  assert.equal(scene.tileAt(map, -1, 0), 0, 'off the map is empty');
  const record = scene.tilemaps.get(map.entity);
  assert.deepEqual(record.dirty, [1, 1, 4, 3], 'both blocks, as one');
  assert.throws(() => scene.setTiles(map, 3, 0, 2, [1, 1]), /not inside/);
  const flipped = (0x80000000 | 3) >>> 0;
  scene.setTile(map, 0, 0, flipped);
  assert.equal(scene.tileAt(map, 0, 0), flipped, 'flip bits kept');
  // 64 x 32 in 16 x 16 tiles is eight: 8 is the last, 9 is past it, flipped or not.
  assert.throws(() => scene.setTile(map, 0, 1, 9), /^Error: setTile: tile 9 is not in the tileset, which holds 8/);
  assert.throws(() => scene.setTile(map, 0, 1, (0x40000000 | 9) >>> 0), /setTile: tile/);
  assert.throws(() => scene.setTiles(map, 0, 1, 2, [1, 2.5]), /setTiles: tile 2.5/);
  assert.throws(() => scene.setTile(map, 0, 1, -1), /setTile: tile -1/);
  assert.equal(scene.tileAt(map, 0, 1), 0, 'a refused id writes nothing');
  assert.throws(() => scene.addTilemap({ tileset, tileSize: [16, 16], columns: 1, rows: 1, tiles: [9] }), /addTilemap: tile 9/);
  assert.throws(() => scene.setTilemap(map, { tileSize: [32, 32] }), /setTilemap: tile \d+ is not in the tileset, which holds 2/,
    'bigger tiles mean fewer of them, and the ids are checked against that');

  const order = scene.spriteOrder;
  scene.setTilemap(map, { color: [1, 0, 0, 1] });
  assert.equal(scene.tileAt(map, 1, 1), 5, 'setTilemap keeps the tiles');
  assert.ok(scene.spriteOrder > order);
  assert.throws(() => scene.setTilemap(map, { columns: 8 }), /needs 24/);
  scene.remove(map);
  assert.equal(scene.tilemaps.size, 0);
});

test('a tilemap draws in its layer as a run of its own, one quad over the map', () => {
  const tileset = { view: {}, width: 64, height: 32 };
  const scene = new Scene({ capacity: 8 });
  scene.addSprite({ texture: tileset, layer: 1 });
  const a = scene.addTilemap({ tileset, tileSize: [16, 8], columns: 10, rows: 5, pivot: [0, 0] });
  scene.addTilemap({ tileset, tileSize: [16, 8], columns: 10, rows: 5 });
  const { entries, runs } = order2D(scene);
  assert.equal(entries[0][2], a.entity, 'layer 0 first');
  assert.deepEqual(runs.map((r) => [r.blend, r.count]), [['tilemap', 1], ['tilemap', 1], ['alpha', 1]], 'never merged');

  scene.update();
  const out = new Float32Array(SPRITE2D_FLOATS);
  write2D(out, 0, scene.transforms.world, a.entity, scene.tilemaps.get(a.entity), null);
  vecClose(out.subarray(4, 8), [160, 40, 0, 0], 1e-6, 'size in pixels, pivot top-left');
  vecClose(out.subarray(8, 12), [0, 0, 10, 5], 1e-6, 'uv counts tiles');
  vecClose(out.subarray(16, 19), [16, 8, 4], 1e-6, 'tile size, and the tileset is 4 tiles wide');
  assert.equal(out[3], 4, 'flagged a tilemap');
  // A 1-texel margin and 2-texel gaps: 1 + 16 + 2 + 16 + 2 + 16 = 53 of 64, and a fourth won't fit.
  const spaced = scene.addTilemap({ tileset, tileSize: [16, 8], columns: 2, rows: 2, margin: 1, spacing: 2 });
  scene.update();
  write2D(out, 0, scene.transforms.world, spaced.entity, scene.tilemaps.get(spaced.entity), null);
  assert.equal(out[18], 3, 'three tiles fit between the margins');
  vecClose(out.subarray(20, 22), [1, 2], 1e-6, 'margin and spacing ride along');
  assert.throws(() => scene.addTilemap({ tileset, tileSize: [16, 8], columns: 1, rows: 1, margin: 0.5 }), /whole texels/);
});

test('shapes: checked, drawn in their layer, batched together, and scaled by their node', () => {
  const scene = new Scene({ capacity: 16 });
  assert.throws(() => scene.addShape({ shape: 'star', size: [1, 1] }), /'rect' or 'ellipse'/);
  assert.throws(() => scene.addShape({ size: [0, 1] }), /positive/);
  assert.throws(() => scene.addShape({ size: [1, 1], strokeWidth: -1 }), /0 or more/);
  const pill = scene.addShape({ size: [40, 10], radius: 99, color: [1, 0, 0, 1], stroke: [0, 0, 1, 1], strokeWidth: 2, position: [100, 50] });
  assert.equal(shapeRadius(scene.shapes.get(pill.entity)), 5, 'drawn no bigger than half the shorter side');
  assert.equal(scene.shapeOf(pill).radius, 99, 'and kept as asked');
  scene.addShape({ shape: 'ellipse', size: [8, 8] });
  scene.addShape({ size: [8, 8], blend: 'additive' });
  const texture = { view: {}, width: 4, height: 4 };
  scene.addSprite({ texture, layer: -1 });

  const { runs } = order2D(scene);
  assert.deepEqual(runs.map((r) => [r.blend, r.count, r.texture]), [['alpha', 1, texture], ['shape', 2, null], ['shapeAdditive', 1, null]],
    'shapes share one draw, whatever their kind');

  pill.setScale(2, 3, 1);
  scene.update();
  const out = new Float32Array(SPRITE2D_FLOATS);
  write2D(out, 0, scene.transforms.world, pill.entity, scene.shapes.get(pill.entity), null);
  vecClose(out.subarray(4, 6), [80, 30], 1e-5, 'size scaled by the node');
  vecClose(out.subarray(20, 24), [0, 0, 1, 1], 1e-6, 'the outline colour');
  vecClose(out.subarray(12, 16), [1, 0, 0, 1], 1e-6, 'the fill');
  vecClose(out.subarray(16, 20), [40, 15, 10, 4], 1e-5, 'half size; corner and outline by the lesser scale');
  assert.equal(out[3], 8, 'a shape, not an ellipse');

  const order = scene.spriteOrder;
  scene.setShape(pill, { strokeWidth: 3 });
  assert.equal(scene.spriteOrder, order, 'an outline is rewritten in place');
  scene.setShape(pill, { blend: 'additive' });
  assert.equal(scene.spriteOrder, order + 1, 'a blend moves it to another run');
});

test('paths: checked, batched with shapes, their points packed, their quad around the line', () => {
  const scene = new Scene({ capacity: 16 });
  assert.throws(() => scene.addPath({ points: [[0, 0]] }), /at least 2/);
  assert.throws(() => scene.addPath({ points: [[0, 0], [1, NaN]] }), /point 1/);
  scene.addShape({ size: [4, 4] });
  const tri = scene.addPath({ points: [[0, 0], [60, 20], [0, 40]], color: [1, 0.8, 0, 1], stroke: [0, 0, 1, 1], strokeWidth: 4, lit: true, position: [100, 50] });
  const line = scene.addPath({ points: [[-10, 5], [10, 5]], closed: false, strokeWidth: 2 });
  const { entries, runs, points } = order2D(scene);
  assert.deepEqual(runs.map((r) => [r.blend, r.count]), [['shape', 3]], 'shapes and paths, one draw');
  assert.deepEqual([entries[1][3], entries[2][3]], [0, 3], 'where each path starts in the points');
  assert.deepEqual([...points], [0, 0, 60, 20, 0, 40, -10, 5, 10, 5]);

  tri.setScale(2, 2, 1);
  scene.update();
  const out = new Float32Array(SPRITE2D_FLOATS);
  write2D(out, 0, scene.transforms.world, tri.entity, scene.paths.get(tri.entity), null, 0);
  vecClose(out.subarray(8, 12), [-2, -2, 62, 42], 1e-6, 'the points, and half the line, in its own units');
  vecClose(out.subarray(4, 8), [128, 88, 2 / 64, 2 / 44], 1e-5, 'scaled by the node; the pivot at its origin');
  vecClose(out.subarray(16, 20), [0, 3, 4, 2], 1e-6, 'first point, count, line width, node scale');
  vecClose(out.subarray(20, 24), [0, 0, 1, 1], 1e-6, 'the line colour');
  assert.equal(out[3], 32 | 64 | 128, 'a path, closed, lit');
  write2D(out, 0, scene.transforms.world, line.entity, scene.paths.get(line.entity), null, 3);
  assert.equal(out[3], 32, 'open, unlit');

  const order = scene.spriteOrder;
  scene.setPath(tri, { color: [1, 0, 0, 1] });
  assert.equal(scene.spriteOrder, order, 'a colour is rewritten in place');
  assert.equal(scene.paths.get(tri.entity).points.length, 6, 'and keeps its points');
  scene.setPath(tri, { points: [[0, 0], [1, 1], [0, 1]] });
  assert.equal(scene.spriteOrder, order + 1, 'new points are new GPU data: rebuilt');
});

test('picking a path: inside by winding, as it fills; near its line', () => {
  const scene = new Scene({ capacity: 16 });
  // An L: the notch at the top right is outside, though inside its bounds.
  const ell = scene.addPath({ points: [[0, 0], [10, 0], [10, 30], [30, 30], [30, 40], [0, 40]], color: [0, 0, 0, 0], position: [100, 100] });
  const road = scene.addPath({ points: [[0, 0], [100, 0]], closed: false, strokeWidth: 6, position: [0, 200] });
  const camera = new Camera2D().update(1, 400, 300, 1);
  const hit = (x, y) => scene.pick(camera, x, y, 400, 300)?.node.entity ?? null;
  assert.equal(hit(105, 105), ell.entity, 'in the upright, even with a clear fill');
  assert.equal(hit(125, 135), ell.entity, 'in the foot');
  assert.equal(hit(125, 110), null, 'in the notch: outside');
  assert.equal(hit(50, 202.5), road.entity, 'on the line');
  assert.equal(hit(50, 204), null, 'past half its width');
});

test('an emitter holds its layer among sprites in 2D, and takes [x, y]', () => {
  const scene = new Scene({ capacity: 16 });
  const texture = { view: {}, width: 4, height: 4 };
  scene.addSprite({ texture, layer: 2 });
  const sparks = scene.addEmitter({ lifetime: 1, size: 4, layer: 1, direction: [0, -1], acceleration: [0, 300] });
  scene.addSprite({ texture });
  const { entries, runs } = order2D(scene);
  assert.deepEqual(runs.map((r) => [r.blend, r.count]), [['alpha', 1], ['emitter', 0], ['alpha', 1]], 'between the layers, taking no slot');
  assert.equal(runs[1].emitter, sparks.entity);
  assert.equal(entries.length, 2);
  const record = scene.emitters.get(sparks.entity);
  assert.deepEqual([...record.direction], [0, -1, 0]);
  assert.deepEqual([...record.acceleration], [0, 300, 0]);
  const order = scene.spriteOrder;
  scene.setEmitter(sparks, { layer: 3 });
  assert.equal(scene.spriteOrder, order + 1, 'a new layer reorders');
});

test('a light takes [x, y] in 2D, and a spot aims across the view', () => {
  const scene = new Scene({ capacity: 8 });
  const lamp = scene.addLight({ position: [40, 60], radius: 100 });
  const torch = scene.addLight({ position: [0, 0], direction: [0, 1] });
  scene.update();
  scene.refreshLights();
  vecClose(scene.lights.subarray(0, 4), [40, 60, 0, 100]);
  vecClose(scene.lights.subarray(16 + 8, 16 + 11), [0, 1, 0], 1e-6, 'down the screen');
  torch.setDirection(-1, 0);
  scene.update();
  scene.refreshLights();
  vecClose(scene.lights.subarray(16 + 8, 16 + 11), [-1, 0, 0], 1e-6, 'turned to the left');
  assert.ok(lamp.alive);
});

test('a 2D unit is a CSS pixel on any screen', () => {
  // A 100 x 50 CSS canvas on a screen with two pixels to a CSS pixel.
  const camera = new Camera2D({ pivot: [0.5, 0.5], position: [50, 25] }).update(2, 200, 100, 2);
  vecClose(camera.worldToScreen(10, 20), [20, 40], 1e-4, 'twice as many canvas pixels');
  vecClose(camera.worldToScreen(50, 25), [100, 50], 1e-4, 'and still centred');
  vecClose(camera.screenToWorld(20, 40), [10, 20], 1e-4);
  camera.update(2);
  assert.equal(camera.pixelRatio, 2, 'kept between frames');
});

test('picking a 2D view finds what is drawn on top, by its real shape', () => {
  const scene = new Scene({ capacity: 32 });
  const texture = { view: {}, width: 16, height: 16 };
  const font = { metrics: { ascent: 0.8, descent: 0.2, glyphs: new Map([['a', { advance: 0.5, left: 0, width: 0.5, height: 1, descent: 0 }], [' ', { advance: 0.5, left: 0, width: 0, height: 0, descent: 0 }]]) }, ensure() {}, texture: {} };
  const ball = scene.addShape({ shape: 'ellipse', size: [20, 20], position: [60, 20] });
  const coin = scene.addSprite({ texture, position: [60, 20], layer: 1 });   // 16 x 16, over the ball's middle
  const bar = scene.addSprite({ texture, size: [20, 2], position: [20, 60] });
  bar.setAngle(Math.PI / 4);
  const label = scene.addText({ font, text: 'a a', size: 10, pivot: [0, 1], position: [100, 100] });
  const floor = scene.addSprite({ texture, size: [400, 400], pivot: [0, 0], layer: -5 });
  const map = scene.addTilemap({ tileset: texture, tileSize: [8, 8], columns: 4, rows: 2, tiles: [1, 0, 0, 0, 0, 0, 0, 1], position: [200, 200] });

  // A 400 x 300 CSS canvas at twice the pixels, never moved.
  const camera = new Camera2D().update(1, 800, 600, 2);
  const at = (x, y) => scene.pick(camera, x, y, 400, 300);
  const hit = (x, y) => at(x, y)?.node.entity;

  assert.equal(hit(60, 20), coin.entity, 'the higher layer wins');
  assert.equal(hit(60, 29), ball.entity, 'below the coin, inside the ball');
  assert.equal(hit(68.5, 28.5), floor.entity, "inside the ball's box but outside the ball");
  assert.equal(hit(25, 65), bar.entity, 'along the turned bar');
  assert.equal(hit(29, 60), floor.entity, 'where the bar would be unturned');
  // pivot [0, 1] is the block's bottom-left, so the text sits above the node.
  assert.equal(hit(107, 96), label.entity, 'between two letters: text is hit by its block');
  assert.deepEqual(at(200 + 28, 200 + 12).tile, [3, 1], 'a tilemap says which tile');
  assert.equal(hit(200 + 12, 200 + 4), floor.entity, 'an empty tile shows what is under it');
  vecClose(at(123, 45).point, [123, 45], 1e-4, 'the world point');
  // A sprite's own angle on a mirrored node: drawn down the diagonal, so hit down it.
  const mirrored = scene.addSprite({ texture, size: [20, 2], angle: Math.PI / 4, position: [300, 120], layer: 3 });
  mirrored.setScale(-1, 1);
  assert.equal(hit(307, 127), mirrored.entity, 'along the diagonal it is drawn on');
  assert.equal(hit(307, 113), floor.entity, 'not along the other one');
  scene.remove(floor);
  assert.equal(at(399, 299), null, 'nothing there');
});

test('the 2D view uploads all of a new tilemap, then only the tiles changed', () => {
  globalThis.GPUTextureUsage ??= { TEXTURE_BINDING: 4, COPY_DST: 2 };
  const writes = [];
  const destroyed = [];
  const view = Object.create(View2D.prototype);
  view._tilemaps = new Map();
  view.rhi = {
    device: {
      limits: { maxTextureDimension2D: 8192 },
      createTexture: () => { const t = { destroy: () => destroyed.push(t) }; return t; },
    },
    queue: { writeTexture: (dst, data, layout, size) => writes.push([dst.origin.x, dst.origin.y, size.width, size.height, layout.offset]) },
  };
  const tileset = { view: {}, width: 64, height: 32 };
  const scene = new Scene({ capacity: 8 });
  const map = scene.addTilemap({ tileset, tileSize: [16, 16], columns: 50, rows: 20 });

  view._uploadTiles(scene, true);
  assert.equal(view.tilesWritten, 1000, 'all of it');
  view._uploadTiles(scene, false);
  assert.equal(view.tilesWritten, 0, 'nothing changed');
  scene.setTile(map, 7, 3, 2);
  view._uploadTiles(scene, false);
  assert.equal(view.tilesWritten, 1, 'one tile');
  assert.deepEqual(writes.at(-1), [7, 3, 1, 1, (3 * 50 + 7) * 4], 'read from its place in the map');
  scene.remove(map);
  view._uploadTiles(scene, true);
  assert.equal(destroyed.length, 1, 'a removed map gives its texture back');
  assert.equal(view._tilemaps.size, 0);
});


test('1.0.1: run sees a frame drawn by hand, and a restart is no jump', () => {
  const engine = Object.create(Winding.prototype);
  engine.gpu = { width: 320, height: 240, destroyed: false, canvas: { isConnected: true } };
  const drawn = [];
  engine.renderer = {
    exposure: 1, fog: null, dof: null, skybox: true, shadowDistance: null, lightDistance: null, ao: null, oit: false,
    debug: { count: 0, depthTest: true }, _variantSets: new Map(),
    post: { threshold: 1, knee: 0.5, filterRadius: 1, strength: 0.06, levels: 5, antialias: true, grading: null, fxaaPipeline: {} },
    upscaler: { pending: false },
    shadows: { lambda: 0.7, casterExtent: 4, normalBias: 1.5 },
    render(scene, camera) { scene.update(); scene.transforms.moved.fill(0); scene.transforms.movedPending = false; camera.update(320 / 240); drawn.push(scene.name); },
  };
  engine._drawn = null; engine.onDemand = true; engine.skippedFrames = 0; engine._running = false;
  engine.clock = new Clock();
  engine._fpsAccum = 0; engine._fpsFrames = 0;
  let raf = null;
  const [request, cancel] = [globalThis.requestAnimationFrame, globalThis.cancelAnimationFrame];
  globalThis.requestAnimationFrame = (f) => { raf = f; return 1; };
  globalThis.cancelAnimationFrame = () => {};
  try {
    const a = new Scene({ capacity: 8 }); a.name = 'A';
    const b = new Scene({ capacity: 8 }); b.name = 'B';
    const camera = new Camera();
    engine.run(a, camera); raf(0); raf(16); engine.stop();
    engine.renderFrame(b, camera);
    engine.run(a, camera); raf(10000); raf(10016);
    assert.deepEqual(drawn, ['A', 'B', 'A'], 'A drawn again over B, then idle');
    close(engine.clock.realDelta, 0.016, 1e-9, 'ten seconds stopped is not a frame of time');
    // Destroyed from inside frame(): nothing more is drawn that frame.
    let destroying = false;
    engine.stop();
    engine.run(a, camera, { frame: () => { if (destroying) engine.stop(); } });
    destroying = true;
    const before = drawn.length;
    engine.invalidate();
    raf(10032);
    assert.equal(drawn.length, before, 'stopped in frame(): not drawn');
  } finally {
    globalThis.requestAnimationFrame = request;
    globalThis.cancelAnimationFrame = cancel;
  }
});

test('1.0.1: a removed parent, a morph back at rest, levels of detail, renamed names, and a clamped orbit', () => {
  // A removed node's slot is the next one's: hanging something off it is an error.
  const scene = new Scene({ capacity: 16 });
  const a = scene.createNode();
  scene.remove(a);
  const b = scene.createNode().setPosition(100, 0, 0);
  assert.throws(() => scene.addLight({ parent: a }), /^Error: addLight: parent was removed/);
  assert.throws(() => scene.createNode({ parent: a }), /^Error: createNode: parent was removed/);
  assert.throws(() => scene.createNode().setParent(a), /^Error: setParent: parent was removed/);
  assert.throws(() => scene.createNode({ parent: new Scene({ capacity: 4 }).createNode() }), /another scene/);
  assert.equal(b.children().length, 0, 'nothing was hung off the stranger');

  // A morph mesh whose weights return to 0 gets its own box back.
  const morphed = new Scene({ capacity: 16 });
  const node = morphed.add({
    nodes: [{ name: 'head', position: [0, 0, -5], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [], mesh: 0, skin: -1, weights: Float32Array.from([0]) }],
    meshes: [{ name: 'face', targetCount: 1, primitives: [{ indexCount: 6, materialId: 0, bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] }, morphExtent: Float32Array.from([2]) }] }],
    roots: [0],
  });
  const min = [0, 0, 0], max = [0, 0, 0];
  const drawn = () => { morphed.transforms.moved.fill(0); morphed.transforms.movedPending = false; };
  morphed.bounds(min, max); drawn();
  node.weights[0] = 1;
  morphed.bounds(min, max); drawn();
  node.weights[0] = 0;
  morphed.bounds(min, max); drawn();
  vecClose([...min, ...max], [-0.5, -0.5, -5.5, 0.5, 0.5, -4.5], 1e-6, 'the box at rest');
  assert.equal(morphed.raycast([2, 0, 0], [0, 0, -1]), null, 'a ray beside it misses');

  // A ray hits a mesh in levels of detail by its finest.
  const lods = new Scene({ capacity: 8 });
  const prim = (h) => ({ indexCount: 3, materialId: 0, bounds: { min: [-h, -h, -h], max: [h, h, h] } });
  const trs = { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] };
  lods.add({
    nodes: [
      { name: 'fine', ...trs, children: [], mesh: 0, skin: -1, lod: { ids: [1], coverage: [0.5, 0] } },
      { name: 'coarse', ...trs, children: [], mesh: 1, skin: -1 },
    ],
    meshes: [{ name: 'fine', targetCount: 0, primitives: [prim(0.5)] }, { name: 'coarse', targetCount: 0, primitives: [prim(2)] }],
    roots: [0],
  });
  assert.equal(lods.raycast([1.5, 0, 10], [0, 0, -1]), null, 'beside the fine level, inside the coarse');
  assert.equal(lods.raycast([0, 0, 10], [0, 0, -1])?.distance, 9.5);

  // Names 1.0 changed say what they're called now, read or written.
  const engine = Object.create(Winding.prototype);
  assert.throws(() => engine.rhi, /engine.rhi is now engine.gpu/);
  const view = new Camera2D();
  assert.throws(() => { view.rotation = 1; }, /camera.rotation is now camera.angle/);

  // An orbit's pitch and distance are held in range however they're set.
  const listeners = { addEventListener() {}, removeEventListener() {}, style: { touchAction: 'auto' } };
  const camera = new Camera({ fovY: 1, near: 0.1 });
  const orbit = new OrbitController(camera, listeners, { pitch: Math.PI / 2 });
  camera.update(1);
  assert.ok(camera.view.every(Number.isFinite), 'straight up is just short of it');
  assert.equal(listeners.style.touchAction, 'none', 'a finger turns it, not the page');
  orbit.desired.distance = 0;
  orbit.update(0);
  close(orbit.distance, orbit.minDistance, 1e-9, "held at its nearest");
  orbit.detach();
  assert.equal(listeners.style.touchAction, 'auto', 'put back');
});

console.log(`\n${passed} checks passed\n`);
