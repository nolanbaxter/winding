// Transforms, camera and picking. Run: node test/scene.test.js

import assert from 'node:assert/strict';

import { HandleAllocator } from '../src/core/handle.js';
import { NULL_HANDLE, handleIndex } from '../src/core/handle.js';
import { TransformStore, NO_PARENT } from '../src/scene/transform.js';
import { Camera } from '../src/scene/camera.js';
import { quatCreate, quatSetAxisAngle } from '../src/core/math/quat.js';
import { vec3Create } from '../src/core/math/vec3.js';
import { Scene } from '../src/scene/scene.js';
import { updateWorldBounds, applyMorphBounds } from '../src/scene/bounds.js';

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

// ------------------------------------------------- triangle-exact picking

console.log('\npicking against triangles');

/**
 * One renderable whose bounding box is the unit cube but whose geometry is a
 * single small quad at its centre. Every ray that enters the box away from the
 * middle is a box hit and a triangle miss, which is the whole difference.
 */
function retainedScene({ position = [0, 0, -5], scale } = {}) {
  const scene = new Scene({ capacity: 8 });
  const entity = scene.entities.alloc();
  scene.transforms.add(entity, { position, scale });
  scene._addRenderable(entity, {
    indexCount: 6,
    materialId: 0,
    bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
    positions: new Float32Array([
      -0.2, -0.2, 0,
      0.2, -0.2, 0,
      0.2, 0.2, 0,
      -0.2, 0.2, 0,
    ]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  });
  return { scene, entity };
}

test('a ray through the geometry hits, and reports the distance to it', () => {
  const { scene, entity } = retainedScene();
  const hit = scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 0, -1));
  assert.ok(hit, 'straight down the middle');
  assert.equal(hit.node.entity, entity);
  // The box front face is at z = -4.5; the quad is at z = -5. Reporting 4.5
  // would mean the broad phase answered.
  close(hit.distance, 5, EPS, 'distance is to the triangle, not to the box');
});

test('the empty corner of the box is a miss once geometry is retained', () => {
  const { scene } = retainedScene();
  // Inside the box, outside the 0.4 x 0.4 quad.
  assert.equal(scene.raycast(vec3Create(0.4, 0.4, 0), vec3Create(0, 0, -1)), null);
});

test('the same corner still hits when geometry was not retained', () => {
  // The contrast that makes the option mean something. Same box, same ray.
  const { scene } = pickScene([[0, 0, -5]]);
  assert.ok(scene.raycast(vec3Create(0.4, 0.4, 0), vec3Create(0, 0, -1)));
});

test('a scaled instance reports its distance in world units', () => {
  // The narrow phase inverts the model matrix and does NOT renormalize the
  // transformed direction. If it did, this would come back as 2.5 -- the
  // distance measured in the object's own halved local units.
  const { scene } = retainedScene({ position: [0, 0, -5], scale: [2, 2, 2] });
  const hit = scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 0, -1));
  assert.ok(hit, 'a doubled quad is still under the ray');
  close(hit.distance, 5, EPS, 'world units, not local');
});

test('a scaled instance widens what it can be hit by', () => {
  // The quad spans +/-0.2 locally, so at scale 2 it reaches 0.4 and a ray that
  // misses the unscaled one now connects. Proves the ray really is being
  // pushed through the transform rather than tested in local space as-is.
  assert.equal(
    retainedScene().scene.raycast(vec3Create(0.3, 0, 0), vec3Create(0, 0, -1)), null,
    'misses at scale 1',
  );
  assert.ok(
    retainedScene({ scale: [2, 2, 2] }).scene.raycast(vec3Create(0.3, 0, 0), vec3Create(0, 0, -1)),
    'hits at scale 2',
  );
});

test('a zero scale collapses the mesh and cannot be hit', () => {
  // The model matrix is singular, so there is no inverse and nothing to test
  // against. Returning a box hit here would be reporting geometry that has no
  // extent at all.
  const { scene } = retainedScene({ scale: [0, 0, 0] });
  assert.equal(scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 0, -1)), null);
});

test('a nearer plain box wins over a further exact hit', () => {
  // Mixed retention in one scene. The un-retained box at -2 is in front of the
  // quad at -5, and the near-to-far walk must stop at it.
  const { scene } = retainedScene();
  const blocker = scene.entities.alloc();
  scene.transforms.add(blocker, { position: [0, 0, -2] });
  scene._addRenderable(blocker, {
    indexCount: 36, materialId: 0,
    bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
  });

  const hit = scene.raycast(vec3Create(0, 0, 0), vec3Create(0, 0, -1));
  assert.ok(hit);
  assert.equal(hit.node.entity, blocker);
  close(hit.distance, 1.5, EPS, 'the blocker front face');
});

test('a further exact hit wins when the nearer box is a triangle miss', () => {
  // The reverse: the retained object is FIRST in the list and nearer by box,
  // but the ray slips past its geometry, so the plain box behind it answers.
  const { scene } = retainedScene({ position: [0, 0, -2] });
  const behind = scene.entities.alloc();
  scene.transforms.add(behind, { position: [0, 0, -5] });
  scene._addRenderable(behind, {
    indexCount: 36, materialId: 0,
    bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
  });

  const hit = scene.raycast(vec3Create(0.4, 0.4, 0), vec3Create(0, 0, -1));
  assert.ok(hit, 'the box behind is still there');
  assert.equal(hit.node.entity, behind);
});

// --------------------------------------------------------------------- morphs

console.log('\nmorph instances and bounds');

/**
 * An asset stand-in: one node, one morphed mesh, one primitive whose box is
 * the unit cube. Built the way engine.load() hands one over, so scene.add()
 * does the wiring rather than the test reaching past it.
 */
function morphAsset({
  extent = [1, 2], weights = [0, 0], position = [0, 0, -5], retain = false, animations,
} = {}) {
  const primitive = {
    indexCount: 6,
    materialId: 0,
    bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
    morphExtent: Float32Array.from(extent),
  };
  if (retain) {
    primitive.positions = new Float32Array([
      -0.2, -0.2, 0, 0.2, -0.2, 0, 0.2, 0.2, 0, -0.2, 0.2, 0,
    ]);
    primitive.indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  }
  return {
    nodes: [{
      name: 'head',
      position: Float32Array.from(position),
      rotation: Float32Array.from([0, 0, 0, 1]),
      scale: Float32Array.from([1, 1, 1]),
      children: [],
      mesh: 0,
      skin: -1,
      weights: Float32Array.from(weights),
    }],
    meshes: [{ name: 'face', targetCount: extent.length, primitives: [primitive] }],
    roots: [0],
    animations,
  };
}

/** The renderable's world box after a bounds refresh, as [min, max]. */
function worldBox(scene, i = 0) {
  scene.update();
  updateWorldBounds(
    scene.renderableCount, scene.localMin, scene.localMax, scene.worldMin, scene.worldMax,
    scene.transforms.world, scene.renderableMatrixSlot, null,
  );
  scene.applyMorphBounds();
  const o = i * 3;
  return [
    [scene.worldMin[o], scene.worldMin[o + 1], scene.worldMin[o + 2]],
    [scene.worldMax[o], scene.worldMax[o + 1], scene.worldMax[o + 2]],
  ];
}

test('adding a morphed asset creates one weight array per instance', () => {
  const scene = new Scene({ capacity: 16 });
  const a = scene.add(morphAsset({ weights: [0.25, 0.5] }));
  const b = scene.add(morphAsset({ weights: [0.25, 0.5] }));

  assert.equal(scene.morphs.length, 2);
  assert.equal(scene.renderableMorph[0], 0);
  assert.equal(scene.renderableMorph[1], 1);

  // Two instances of one asset deform independently -- the same reason the
  // animation player is per instance.
  a.weights[0] = 1;
  close(a.weights[0], 1, EPS, 'a');
  close(b.weights[0], 0.25, EPS, 'b is untouched');
});

test('node.weights is a live view, not a copy', () => {
  const scene = new Scene({ capacity: 16 });
  const node = scene.add(morphAsset());
  node.weights[1] = 0.75;
  close(scene.morphs[0].weights[1], 0.75, EPS, 'the write reached the instance');
});

test('a node whose mesh has no targets has no weights', () => {
  const scene = new Scene({ capacity: 16 });
  const asset = morphAsset();
  asset.meshes[0].targetCount = 0;
  asset.meshes[0].primitives[0].morphExtent = null;
  asset.nodes[0].weights = null;

  const node = scene.add(asset);
  assert.equal(node.weights, null);
  assert.equal(scene.morphs.length, 0);
  assert.equal(scene.renderableMorph[0], -1);
});

test('weights at zero leave the authored box alone', () => {
  const scene = new Scene({ capacity: 16 });
  scene.add(morphAsset());
  const [min, max] = worldBox(scene);
  for (let i = 0; i < 3; i++) close(min[i], [-0.5, -0.5, -5.5][i], EPS, `min ${i}`);
  for (let i = 0; i < 3; i++) close(max[i], [0.5, 0.5, -4.5][i], EPS, `max ${i}`);
});

test('a weight grows the box by how far that target reaches', () => {
  const scene = new Scene({ capacity: 16 });
  const node = scene.add(morphAsset({ extent: [1, 2] }));

  node.weights[0] = 1;                      // reach 1
  let [min, max] = worldBox(scene);
  close(min[0], -1.5, EPS, 'min x');
  close(max[0], 1.5, EPS, 'max x');
  close(min[2], -6.5, EPS, 'min z');

  node.weights[1] = 0.5;                    // + reach 1 => 2 in total
  [min, max] = worldBox(scene);
  close(min[0], -2.5, EPS, 'both targets');
  close(max[0], 2.5, EPS, 'both targets');
});

test('a negative weight grows the box too', () => {
  // glTF does not bound weights to [0,1]: a negative one is how "the opposite
  // of this expression" is authored. Summing without the absolute value would
  // SHRINK the box and cull geometry that is on screen.
  const scene = new Scene({ capacity: 16 });
  const node = scene.add(morphAsset({ extent: [1, 2] }));
  node.weights[0] = -1;

  const [min, max] = worldBox(scene);
  close(min[0], -1.5, EPS, 'min x');
  close(max[0], 1.5, EPS, 'max x');
});

test('the pass is idempotent: it rebuilds rather than grows', () => {
  const scene = new Scene({ capacity: 16 });
  const node = scene.add(morphAsset({ extent: [1, 0] }));
  node.weights[0] = 1;

  worldBox(scene);
  scene.applyMorphBounds();
  scene.applyMorphBounds();
  const [min, max] = worldBox(scene);
  close(min[0], -1.5, EPS, 'min x after four passes');
  close(max[0], 1.5, EPS, 'max x after four passes');
});

test('the pass reports only the instances whose padding actually changed', () => {
  // The renderer derives "does the scene extent need recomputing" from this.
  // A transform that does not move and a weight that does not change is a
  // frame with nothing to redo, and a morphed scene must not lose that.
  const scene = new Scene({ capacity: 16 });
  const node = scene.add(morphAsset({ extent: [1, 0] }));

  node.weights[0] = 1;
  assert.equal(scene.applyMorphBounds(), 1, 'the weight changed');
  assert.equal(scene.applyMorphBounds(), 0, 'nothing changed');
  node.weights[0] = 0.5;
  assert.equal(scene.applyMorphBounds(), 1, 'changed again');
});

test('a morphed renderable is picked at its box, not at its triangles', () => {
  // The retained triangles are the UNDEFORMED mesh. Answering with them would
  // report where a vertex was authored rather than where the weights put it.
  const scene = new Scene({ capacity: 16 });
  scene.add(morphAsset({ retain: true }));

  // A ray through the empty corner of the box: a box hit, a triangle miss.
  const hit = scene.raycast(vec3Create(0.4, 0.4, 0), vec3Create(0, 0, -1));
  assert.ok(hit, 'the box answers');
  close(hit.distance, 4.5, EPS, 'distance is to the box face');
});

test('a skinned and morphed renderable gets both corrections', () => {
  // applySkinBounds builds a world box from where the joints are, so there is
  // no local box left to pad -- the padding goes straight onto the world box,
  // which holds because joint matrices are rigid.
  const min = new Float32Array([-1, -1, -1]);
  const max = new Float32Array([1, 1, 1]);
  const changed = applyMorphBounds(
    1, Int32Array.from([0]), Int32Array.from([0]),
    [{ weights: Float32Array.from([0.5]) }], [Float32Array.from([4])],
    new Float32Array(3), new Float32Array(3), min, max,
    new Float32Array(16), Uint32Array.from([0]), new Float32Array(1),
  );
  assert.equal(changed, 1);
  close(min[0], -3, EPS, 'grown by 0.5 * 4');
  close(max[2], 3, EPS, 'grown by 0.5 * 4');
});

console.log(`\n${passed} checks passed\n`);
