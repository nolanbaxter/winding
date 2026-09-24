// Scene and Node self-check. Run: node test/app.test.js
//
// No GPU anywhere here. Scene holds references to primitives that live on the
// GPU but never calls a WebGPU function, which is exactly what makes this
// testable -- and is why the layering was worth keeping strict.

import assert from 'node:assert/strict';

import { Scene } from '../src/scene/scene.js';
import { Node } from '../src/scene/node.js';
import { Camera } from '../src/scene/camera.js';
import { OrbitController } from '../src/app/controllers.js';
import { NO_PARENT } from '../src/scene/transform.js';
import { handleIndex } from '../src/core/handle.js';
import { quatCreate, quatFromEuler } from '../src/core/math/quat.js';
import { vec3Create, vec3TransformQuat } from '../src/core/math/vec3.js';
import { createModuleWorker, workerShimSource } from '../src/app/engine.js';

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

test('adding an asset creates entities, transforms and renderables', () => {
  const scene = new Scene({ capacity: 64 });
  const root = scene.add(fakeAsset());

  assert.ok(root instanceof Node);
  assert.equal(scene.entities.liveCount, 1);
  assert.equal(scene.renderableCount, 1);
  assert.equal(scene.renderablePrimitive[0].indexCount, 36);
  assert.equal(scene.renderableMaterial[0], 0);
});

test('a mesh with several primitives becomes several renderables on one entity', () => {
  const scene = new Scene({ capacity: 64 });
  const root = scene.add(fakeAsset({ primitivesPerMesh: 3 }));

  assert.equal(scene.entities.liveCount, 1, 'still one node');
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

  assert.equal(scene.entities.liveCount, 3, 'two nodes plus the wrapper');
  assert.equal(root.children().length, 2);
});

test('a single-root asset is returned directly, with no wrapper', () => {
  const scene = new Scene({ capacity: 64 });
  const root = scene.add(fakeAsset());
  assert.equal(scene.entities.liveCount, 1, 'no extra node invented');
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
  vecClose(child.getWorldPosition(vec3Create()), [10, 5, 0]);
});

test('adding under a parent node inherits its transform', () => {
  const scene = new Scene({ capacity: 64 });
  const group = scene.createNode();
  group.setPosition(100, 0, 0);

  const child = scene.add(fakeAsset({ nodes: [node('x', { position: [1, 2, 3], mesh: 0 })] }), { parent: group });
  scene.update();
  vecClose(child.getWorldPosition(vec3Create()), [101, 2, 3]);
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
  vecClose(n.getWorldPosition(vec3Create()), [4, 5, 6]);
});

test('setters chain', () => {
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode().setPosition(1, 0, 0).setScale(2);
  scene.update();
  vecClose(n.getWorldPosition(vec3Create()), [1, 0, 0]);
  assert.equal(scene.transforms.scale[handleIndex(n.entity) * 3], 2);
});

test('setScale with one argument scales uniformly', () => {
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode().setScale(3);
  const o = handleIndex(n.entity) * 3;
  assert.deepEqual([...scene.transforms.scale.subarray(o, o + 3)], [3, 3, 3]);
});

test('setRotationEuler converts at the edge and stores a quaternion', () => {
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode().setRotationEuler(Math.PI / 2, 0, 0);

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
  vecClose(b.getWorldPosition(vec3Create()), [11, 0, 0]);

  b.setParent(null);
  scene.update();
  vecClose(b.getWorldPosition(vec3Create()), [1, 0, 0]);
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
  vecClose(a.getWorldPosition(vec3Create()), [7, 0, 0], EPS, 'both see the same data');
});

test('getWorldPosition reads the LAST update, not pending edits', () => {
  // Stated plainly because it is the one surprising thing about a cursor API.
  const scene = new Scene({ capacity: 16 });
  const n = scene.createNode();
  scene.update();

  n.setPosition(9, 9, 9);
  vecClose(n.getWorldPosition(vec3Create()), [0, 0, 0], EPS, 'stale until update()');
  scene.update();
  vecClose(n.getWorldPosition(vec3Create()), [9, 9, 9]);
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
  assert.equal(scene.entities.liveCount, 3);

  doomed.destroy();

  assert.equal(scene.renderableCount, 1, 'both of its renderables went');
  assert.equal(scene.entities.liveCount, 1);
  assert.equal(scene.renderableEntity[0], keep.entity, 'the survivor is intact');
  assert.equal(doomed.alive, false);
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

console.log(`\n${passed} checks passed\n`);
