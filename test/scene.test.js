// Transforms, camera and picking. Run: node test/scene.test.js

import assert from 'node:assert/strict';

import { HandleAllocator } from '../src/core/handle.js';
import { NULL_HANDLE, handleIndex } from '../src/core/handle.js';
import { TransformStore, NO_PARENT } from '../src/scene/transform.js';
import { Camera } from '../src/scene/camera.js';
import { quatCreate, quatSetAxisAngle } from '../src/core/math/quat.js';
import { vec3Create } from '../src/core/math/vec3.js';
import { Scene } from '../src/scene/scene.js';

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

/** World-space translation of an entity, straight out of the SoA column. */
function worldPos(store, entity) {
  const o = store.worldOffset(entity);
  return [store.world[o + 12], store.world[o + 13], store.world[o + 14]];
}
function posClose(store, entity, expected, what = '') {
  const p = worldPos(store, entity);
  for (let i = 0; i < 3; i++) close(p[i], expected[i], EPS, `${what} axis ${i}`);
}

// -------------------------------------------------------------- transform

console.log('\ntransform hierarchy');

function makeScene(capacity = 64) {
  return { ids: new HandleAllocator(capacity), store: new TransformStore(capacity) };
}

test('a root transform composes to its own TRS', () => {
  const { ids, store } = makeScene();
  const e = ids.alloc();
  store.add(e, { position: [3, -2, 7] });
  store.update();
  posClose(store, e, [3, -2, 7]);
});

test('a child composes against its parent', () => {
  const { ids, store } = makeScene();
  const parent = ids.alloc();
  const child = ids.alloc();
  store.add(parent, { position: [10, 0, 0] });
  store.add(child, { position: [0, 5, 0], parent });

  store.update();
  posClose(store, child, [10, 5, 0]);
});

test('parent scale and rotation apply to children', () => {
  const { ids, store } = makeScene();
  const parent = ids.alloc();
  const child = ids.alloc();

  // Rotate the parent 90 degrees about +Y, which maps its local +X to -Z.
  const q = quatSetAxisAngle(quatCreate(), vec3Create(0, 1, 0), Math.PI / 2);
  store.add(parent, { position: [10, 0, 0], rotation: q, scale: [2, 2, 2] });
  store.add(child, { position: [1, 0, 0], parent });

  store.update();
  // child offset (1,0,0) -> scaled to (2,0,0) -> rotated to (0,0,-2) -> +parent
  posClose(store, child, [10, 0, -2]);
});

test('moving a parent moves its whole subtree in one pass', () => {
  const { ids, store } = makeScene();
  const a = ids.alloc(), b = ids.alloc(), c = ids.alloc();
  store.add(a, { position: [0, 0, 0] });
  store.add(b, { position: [1, 0, 0], parent: a });
  store.add(c, { position: [0, 1, 0], parent: b });
  store.update();

  store.setPosition(a, 100, 0, 0);
  store.update();                            // ONE pass, no recursion

  posClose(store, a, [100, 0, 0]);
  posClose(store, b, [101, 0, 0]);
  posClose(store, c, [101, 1, 0], 'grandchild');
});

test('a four-deep chain accumulates correctly', () => {
  const { ids, store } = makeScene();
  let parent = NULL_HANDLE;
  const chain = [];
  for (let i = 0; i < 4; i++) {
    const e = ids.alloc();
    store.add(e, { position: [1, 0, 0], parent });
    chain.push(e);
    parent = e;
  }
  store.update();
  posClose(store, chain[3], [4, 0, 0]);
});

test('depth order is correct even when children are created before parents', () => {
  // Index order and depth order disagree here, which is the case a naive
  // "just iterate the array" update gets silently wrong.
  const { ids, store } = makeScene();
  const child = ids.alloc();                 // index 0
  const parent = ids.alloc();                // index 1

  store.add(child, { position: [0, 5, 0] });
  store.add(parent, { position: [10, 0, 0] });
  store.setParent(child, parent);

  store.update();
  posClose(store, child, [10, 5, 0]);

  const order = [...store.order.subarray(0, store.orderCount)];
  assert.ok(
    order.indexOf(handleIndex(parent)) < order.indexOf(handleIndex(child)),
    'parent must be visited before child',
  );
});

// --------------------------------------------------- dirty propagation

console.log('\ndirty propagation');

test('a settled scene recomputes nothing', () => {
  const { ids, store } = makeScene();
  const a = ids.alloc(), b = ids.alloc();
  store.add(a);
  store.add(b, { parent: a });

  assert.equal(store.update(), 2, 'first pass composes everything');
  assert.equal(store.update(), 0, 'second pass does no work at all');
});

test('one write recomputes exactly the subtree, and nothing else', () => {
  //   a --> b --> d
  //     \-> c
  //   e  (separate root)
  const { ids, store } = makeScene();
  const a = ids.alloc(), b = ids.alloc(), c = ids.alloc(), d = ids.alloc(), e = ids.alloc();
  store.add(a);
  store.add(b, { parent: a });
  store.add(c, { parent: a });
  store.add(d, { parent: b });
  store.add(e);
  store.update();

  store.setPosition(a, 5, 0, 0);
  assert.equal(store.update(), 4, 'a, b, c, d -- but not e');

  store.setPosition(d, 0, 1, 0);
  assert.equal(store.update(), 1, 'a leaf write recomputes only the leaf');
});

test('moving a parent and a child in the same frame composes each once', () => {
  const { ids, store } = makeScene();
  const a = ids.alloc(), b = ids.alloc();
  store.add(a);
  store.add(b, { parent: a });
  store.update();

  store.setPosition(a, 10, 0, 0);
  store.setPosition(b, 0, 3, 0);
  assert.equal(store.update(), 2, 'no node is composed twice');
  posClose(store, b, [10, 3, 0]);
});

// ------------------------------------------------------------ restructure

console.log('\nrestructuring');

test('reparenting rebuilds the order and recomposes', () => {
  const { ids, store } = makeScene();
  const x = ids.alloc(), y = ids.alloc(), child = ids.alloc();
  store.add(x, { position: [10, 0, 0] });
  store.add(y, { position: [0, 20, 0] });
  store.add(child, { position: [1, 1, 1], parent: x });
  store.update();
  posClose(store, child, [11, 1, 1]);

  store.setParent(child, y);
  store.update();
  posClose(store, child, [1, 21, 1]);
});

test('detaching to the root drops the parent transform', () => {
  const { ids, store } = makeScene();
  const p = ids.alloc(), child = ids.alloc();
  store.add(p, { position: [10, 0, 0] });
  store.add(child, { position: [1, 0, 0], parent: p });
  store.update();

  store.setParent(child, NULL_HANDLE);
  store.update();
  posClose(store, child, [1, 0, 0]);
  assert.equal(store.parent[handleIndex(child)], NO_PARENT);
});

test('a cycle is rejected, not allowed to hang the update loop', () => {
  const { ids, store } = makeScene();
  const a = ids.alloc(), b = ids.alloc();
  store.add(a);
  store.add(b, { parent: a });
  assert.throws(() => store.setParent(a, b), /cycle/);
});

test('self-parenting is rejected', () => {
  const { ids, store } = makeScene();
  const a = ids.alloc();
  store.add(a);
  assert.throws(() => store.setParent(a, a), /cycle/);
});

test('removing a parent re-roots its children instead of orphaning them', () => {
  const { ids, store } = makeScene();
  const p = ids.alloc(), child = ids.alloc();
  store.add(p, { position: [10, 0, 0] });
  store.add(child, { position: [1, 0, 0], parent: p });
  store.update();

  store.remove(p);
  store.update();

  assert.equal(store.parent[handleIndex(child)], NO_PARENT);
  posClose(store, child, [1, 0, 0], 'child keeps its local transform');
});

test('a removed transform stops being visited', () => {
  const { ids, store } = makeScene();
  const a = ids.alloc(), b = ids.alloc();
  store.add(a);
  store.add(b);
  store.update();

  store.remove(a);
  store.setPosition(b, 1, 0, 0);
  store.update();
  assert.equal(store.orderCount, 1);
});

// ----------------------------------------------------------------- camera

console.log('\ncamera');

test('view and projection combine, and there is no far plane to set', () => {
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  camera.position.set([0, 0, 5]);
  camera.update(16 / 9);

  assert.equal('far' in camera, false, 'the API must not have a far distance');

  // A point at the origin is 5 in front of the camera. Reverse-Z puts it at
  // near/distance = 0.1/5 = 0.02, well inside [0,1].
  const m = camera.viewProjection;
  const clipZ = m[14];                        // origin: x=y=z=0, w=1
  const clipW = m[15];
  close(clipZ / clipW, 0.1 / 5, 1e-6, 'ndc z');
});

test('camera depth stays in range across a huge distance span', () => {
  const camera = new Camera({ near: 0.05 });
  camera.position.set([0, 0, 0]);
  camera.target.set([0, 0, -1]);
  camera.update(1);

  const m = camera.viewProjection;
  let prev = Infinity;
  for (const d of [0.05, 1, 100, 1e6]) {
    // View-space point (0,0,-d) through the combined matrix.
    const z = (m[10] * -d + m[14]) / (m[11] * -d + m[15]);
    assert.ok(z <= 1 + 1e-6 && z >= -1e-6, `z out of range at ${d}: ${z}`);
    assert.ok(z < prev, `not monotonic at ${d}`);
    prev = z;
  }
});

// ---------------------------------------------------------------- picking

console.log('\npicking');

/** A scene holding unit cubes at the given positions, one renderable each. */
function pickScene(positions) {
  const scene = new Scene({ capacity: 64 });
  const primitive = { indexCount: 36, bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] } };
  const nodes = positions.map((p) => {
    const entity = scene.entities.alloc();
    scene.transforms.add(entity, { position: p });
    scene._addRenderable(entity, { ...primitive, materialId: 0 });
    return entity;
  });
  return { scene, nodes };
}

test('a ray hits the nearest box, not merely the first one in the list', () => {
  // Far one added FIRST, so a loop that returns on first hit gets this wrong.
  const { scene, nodes } = pickScene([[0, 0, -20], [0, 0, -5]]);

  const hit = scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 0, -1));
  assert.ok(hit, 'expected a hit');
  assert.equal(hit.node.entity, nodes[1], 'nearest box must win');
  close(hit.distance, 4.5, EPS, 'distance to the near face');
});

test('a ray that misses everything returns null', () => {
  const { scene } = pickScene([[0, 0, -5]]);
  assert.equal(scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 1, 0)), null);
});

test('a ray pointing away from a box does not hit it behind the origin', () => {
  const { scene } = pickScene([[0, 0, -5]]);
  assert.equal(scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 0, 1)), null);
});

test('an origin inside a box hits it at distance 0', () => {
  const { scene } = pickScene([[0, 0, 0]]);
  const hit = scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 0, -1));
  assert.ok(hit, 'expected a hit');
  close(hit.distance, 0, EPS);
});

test('maxDistance excludes boxes beyond it', () => {
  const { scene } = pickScene([[0, 0, -20]]);
  const direction = vec3Create(0, 0, -1);
  assert.ok(scene.raycast(vec3Create(0, 0, 0), direction), 'hit without a limit');
  assert.equal(scene.raycast(vec3Create(0, 0, 0), direction, { maxDistance: 10 }), null);
});

test('raycast sees a move that has not been rendered yet', () => {
  // The whole point of composing inside raycast: nothing here calls update(),
  // and picking must still answer about where things ARE.
  const { scene, nodes } = pickScene([[0, 0, -5]]);
  scene.node(nodes[0]).setPosition(50, 0, 0);

  assert.equal(scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 0, -1)), null,
    'the box is no longer on the -Z axis');
  assert.ok(scene.raycast(vec3Create(0, 0, 0), vec3Create(1, 0, 0)), 'it is on +X now');
});

test('a ray parallel to a slab, exactly on its plane, is not a false hit', () => {
  // The 0 * Infinity -> NaN case the slab test deliberately does not branch on.
  const { scene } = pickScene([[0, 0, -5]]);
  // y = 0.5 is exactly the top face; travelling along -Z stays on it.
  const hit = scene.raycast(vec3Create(0, 0.5, 0), vec3Create(0, 0, -1));
  assert.ok(hit === null || Number.isFinite(hit.distance), 'must not return NaN');
});

test('a degenerate ray is refused, not answered with a false hit', () => {
  // A NaN direction contributes no constraint on any axis, so an unchecked
  // slab test reports a hit at distance 0 on whatever it looks at first. A
  // canvas that is not laid out yet (0x0) is enough to produce one.
  const { scene } = pickScene([[0, 0, -5]]);
  assert.throws(
    () => scene.raycast(vec3Create(0, 0, 0), vec3Create(NaN, NaN, NaN)),
    /non-finite/,
  );
});

test('camera rays agree with the projection they have to match', () => {
  const camera = new Camera({ fovY: Math.PI / 2, near: 0.1 });
  camera.position.set([0, 0, 0]);
  camera.target.set([0, 0, -1]);
  camera.update(1);

  const origin = vec3Create();
  const direction = vec3Create();

  // Screen centre looks straight down the view direction.
  camera.rayFromScreen(50, 50, 100, 100, origin, direction);
  close(direction[0], 0, EPS, 'centre x');
  close(direction[1], 0, EPS, 'centre y');
  close(direction[2], -1, EPS, 'centre z');

  // At a 90 degree vertical FOV the top edge is 45 degrees up, so the ray's
  // y and -z components are equal. That ties the ray to fovY, which is the
  // thing that would silently drift if the two were derived separately.
  camera.rayFromScreen(50, 0, 100, 100, origin, direction);
  close(direction[1], -direction[2], EPS, 'top edge sits at 45 degrees');
  assert.ok(direction[1] > 0, 'screen y=0 is UP in world space');
});

test('picking through the camera finds the box under the cursor', () => {
  const { scene, nodes } = pickScene([[0, 0, -5]]);
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  camera.position.set([0, 0, 0]);
  camera.target.set([0, 0, -1]);
  camera.update(1);

  const hit = scene.pick(camera, 50, 50, 100, 100);
  assert.ok(hit, 'centre of the screen should hit');
  assert.equal(hit.node.entity, nodes[0]);

  assert.equal(scene.pick(camera, 0, 0, 100, 100), null, 'the far corner should miss');
});

console.log(`\n${passed} checks passed\n`);
