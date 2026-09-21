// Scene and Node self-check. Run: node test/app.test.js
//
// No GPU anywhere here. Scene holds references to primitives that live on the
// GPU but never calls a WebGPU function, which is exactly what makes this
// testable -- and is why the layering was worth keeping strict.

import assert from 'node:assert/strict';

import { Scene } from '../src/scene/scene.js';
import { Node } from '../src/scene/node.js';
import { NO_PARENT } from '../src/scene/transform.js';
import { handleIndex } from '../src/core/handle.js';
import { quatCreate, quatFromEuler } from '../src/core/math/quat.js';
import { vec3Create, vec3TransformQuat } from '../src/core/math/vec3.js';

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

console.log(`\n${passed} checks passed\n`);
