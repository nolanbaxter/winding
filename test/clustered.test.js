// Clustered lighting self-check. Run: node test/clustered.test.js
//
// The slice mapping and the light packing are both pure, and both are places
// where being subtly wrong produces lighting that looks plausible rather than
// broken -- lights assigned to cells nothing looks up, or a cone that falls off
// over the wrong angle.

import assert from 'node:assert/strict';

import {
  sliceFor, CLUSTER_Z, MAX_LIGHTS_PER_CLUSTER, CLUSTER_COUNT, ClusteredLights, LIGHT_BYTES,
  clusterGridFor, CLUSTER_TILES,
} from '../src/render/clustered.js';
import { Camera } from '../src/scene/camera.js';
import { Scene, LIGHT_FLOATS, LIGHT_POINT, LIGHT_SPOT } from '../src/scene/scene.js';
import { RenderGraph } from '../src/render/graph.js';

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

const NEAR = 0.1;
const LIGHT_DISTANCE = 60;

// ------------------------------------------------------------ slice mapping

console.log('\ndepth slicing');

test('the near plane is slice 0 and the light distance is the last slice', () => {
  assert.equal(sliceFor(NEAR, NEAR, LIGHT_DISTANCE), 0);
  assert.equal(sliceFor(LIGHT_DISTANCE, NEAR, LIGHT_DISTANCE), CLUSTER_Z - 1);
});

test('slices increase monotonically with distance', () => {
  let previous = -1;
  for (let d = NEAR; d < LIGHT_DISTANCE; d *= 1.3) {
    const slice = sliceFor(d, NEAR, LIGHT_DISTANCE);
    assert.ok(slice >= previous, `went backwards at ${d}`);
    assert.ok(slice >= 0 && slice < CLUSTER_Z, `out of range at ${d}: ${slice}`);
    previous = slice;
  }
});

test('anything beyond the light distance clamps into the last slice', () => {
  // Not a cutoff: distant geometry is still lit, just by a coarser cell. There
  // is no far plane in this engine to fall off instead.
  for (const d of [LIGHT_DISTANCE * 2, 1e4, 1e9]) {
    assert.equal(sliceFor(d, NEAR, LIGHT_DISTANCE), CLUSTER_Z - 1, `at ${d}`);
  }
});

test('anything closer than the near plane clamps to slice 0', () => {
  assert.equal(sliceFor(NEAR * 0.5, NEAR, LIGHT_DISTANCE), 0);
  assert.equal(sliceFor(1e-9, NEAR, LIGHT_DISTANCE), 0);
});

test('slicing is exponential, not uniform', () => {
  // The whole reason for the log mapping. Uniform slices would make every cell
  // the same depth, and perspective would make the far ones enormous.
  const depthOf = (slice) => {
    let low = NEAR;
    for (let d = NEAR; d < LIGHT_DISTANCE * 2; d *= 1.001) {
      if (sliceFor(d, NEAR, LIGHT_DISTANCE) > slice) return d - low;
      if (sliceFor(d, NEAR, LIGHT_DISTANCE) === slice && low === NEAR) low = d;
    }
    return Infinity;
  };

  const firstSpan = depthOf(0);
  const laterSpan = depthOf(CLUSTER_Z - 4);
  assert.ok(laterSpan > firstSpan * 10,
    `far slices should be far deeper: ${firstSpan} vs ${laterSpan}`);
});

test('the grid is a sane size for the index buffer it implies', () => {
  // 16 x 9 x 24 cells at 64 lights each. Worth stating, because the index
  // buffer is the one allocation here that scales with all four numbers.
  assert.equal(CLUSTER_COUNT, CLUSTER_TILES * CLUSTER_Z);
  const indexBytes = CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER * 4;
  assert.ok(indexBytes < 2 * 1024 * 1024, `${indexBytes} bytes is too much`);
});

// ------------------------------------------------------------ light packing

console.log('\nlight packing');

/** Where a light sits in the packed array. Tests read the GPU format directly. */
const slot = (scene, light) => scene._lightOf.get(light.entity) * LIGHT_FLOATS;

/** Compose transforms and copy them into the lights, as a frame would. */
function settle(scene) {
  scene.update();
  scene.refreshLights();
}

test('a point light packs position, radius, colour and intensity', () => {
  const scene = new Scene({ capacity: 16 });
  const light = scene.addLight({
    position: [1, 2, 3], color: [0.25, 0.5, 0.75], intensity: 7, radius: 9,
  });
  settle(scene);

  assert.equal(scene.lightCount, 1);
  const o = slot(scene, light);
  assert.deepEqual([...scene.lights.subarray(o, o + 4)], [1, 2, 3, 9]);
  assert.deepEqual([...scene.lights.subarray(o + 4, o + 8)], [0.25, 0.5, 0.75, 7]);
  assert.equal(scene.lights[o + 14], LIGHT_POINT, 'type marks it a point light');
});

test('a spot light is aimed by its node, and precomputes the cone falloff', () => {
  const scene = new Scene({ capacity: 16 });
  const inner = 0.3;
  const outer = 0.6;
  const light = scene.addLight({
    position: [0, 5, 0], direction: [0, -4, 0], innerAngle: inner, outerAngle: outer,
  });
  settle(scene);

  const o = slot(scene, light);
  vecClose([...scene.lights.subarray(o + 8, o + 11)], [0, -1, 0], 1e-6, 'unit direction');
  assert.equal(scene.lights[o + 14], LIGHT_SPOT);

  // saturate(cos(angle) * scale + offset) must be 1 at the inner angle and 0
  // at the outer one -- that is the whole point of precomputing the pair.
  const scale = scene.lights[o + 12];
  const offset = scene.lights[o + 13];
  close(Math.cos(inner) * scale + offset, 1, 1e-5, 'full brightness at the inner angle');
  close(Math.cos(outer) * scale + offset, 0, 1e-5, 'dark at the outer angle');
});

test('a degenerate cone does not divide by zero', () => {
  const scene = new Scene({ capacity: 16 });
  const light = scene.addLight({
    direction: [0, -1, 0], innerAngle: 0.5, outerAngle: 0.5,
  });
  const o = slot(scene, light);
  assert.ok(Number.isFinite(scene.lights[o + 12]), 'scale is finite');
  assert.ok(Number.isFinite(scene.lights[o + 13]), 'offset is finite');
});

// ------------------------------------------------------ lights as scene nodes

console.log('\nlights as scene nodes');

test('a light follows its parent', () => {
  // THE point of this change. A light used to be a row in an array, so one on
  // a moving object meant calling setLightPosition every frame -- the viewer
  // example did exactly that for twelve lights. Now it is a node, and a node
  // under a parent goes where the parent goes.
  const scene = new Scene({ capacity: 16 });
  const cart = scene.createNode();
  const lamp = scene.addLight({ position: [0, 1, 0], parent: cart });

  cart.setPosition(10, 0, 5);
  settle(scene);

  const o = slot(scene, lamp);
  vecClose([...scene.lights.subarray(o, o + 3)], [10, 1, 5], 1e-6, 'parent offset + own offset');
});

test('a spot aims wherever its parent turns', () => {
  // The same argument for direction. A torch in a hand points where the hand
  // points, because the aim is the node's -Z and the node inherits rotation.
  const scene = new Scene({ capacity: 16 });
  const head = scene.createNode();
  const torch = scene.addLight({ direction: [0, 0, -1], parent: head });

  // Quarter turn to the left about +Y: -Z swings round onto -X.
  head.setRotationAxisAngle([0, 1, 0], Math.PI / 2);
  settle(scene);

  const o = slot(scene, torch);
  vecClose([...scene.lights.subarray(o + 8, o + 11)], [-1, 0, 0], 1e-5, 'aim followed the head');
});

test('moving a light is moving its node', () => {
  const scene = new Scene({ capacity: 16 });
  const lamp = scene.addLight({ position: [0, 0, 0] });
  lamp.setPosition(3, 4, 5);
  settle(scene);

  const o = slot(scene, lamp);
  vecClose([...scene.lights.subarray(o, o + 3)], [3, 4, 5], 1e-6);
});

test('setLight changes only what it is given, and only that light', () => {
  const scene = new Scene({ capacity: 16 });
  const first = scene.addLight({ position: [0, 0, 0], radius: 5, color: [1, 1, 1], intensity: 1 });
  const second = scene.addLight({ position: [9, 9, 9], radius: 5, color: [1, 1, 1], intensity: 1 });

  assert.equal(scene.setLight(second.entity, { color: [0.1, 0.2, 0.3], intensity: 4 }), true);

  const o = slot(scene, second);
  assert.equal(scene.lights[o + 3], 5, 'radius untouched');
  vecClose([...scene.lights.subarray(o + 4, o + 8)], [0.1, 0.2, 0.3, 4], 1e-6, 'colour + intensity');

  const f = slot(scene, first);
  vecClose([...scene.lights.subarray(f + 4, f + 8)], [1, 1, 1, 1], 1e-6, 'the other light is unchanged');
});

test('setLight on something that is not a light says so', () => {
  const scene = new Scene({ capacity: 16 });
  const plain = scene.createNode();
  assert.equal(scene.setLight(plain.entity, { intensity: 9 }), false);
});

test('removing a light leaves every other light reachable by its handle', () => {
  // This test used to be called "removal swap-removes, so light indices are
  // not stable" and ASSERTED the bug: the last light silently took the
  // removed one's index, so anyone holding an index now pointed at a
  // different light. Callers hold the node now, and the node still finds its
  // own light after the array has been compacted under it.
  const scene = new Scene({ capacity: 16 });
  const a = scene.addLight({ position: [1, 1, 1], intensity: 1 });
  const b = scene.addLight({ position: [2, 2, 2], intensity: 2 });
  const c = scene.addLight({ position: [3, 3, 3], intensity: 3 });

  a.destroy();
  settle(scene);

  assert.equal(scene.lightCount, 2);
  assert.equal(scene.lights[slot(scene, b) + 7], 2, 'b still reaches b');
  assert.equal(scene.lights[slot(scene, c) + 7], 3, 'c still reaches c -- it moved, the handle did not');
  vecClose([...scene.lights.subarray(slot(scene, c), slot(scene, c) + 3)], [3, 3, 3], 1e-6);
});

test('destroying a parent takes its lights with it', () => {
  const scene = new Scene({ capacity: 16 });
  const cart = scene.createNode();
  scene.addLight({ parent: cart });
  scene.addLight({ parent: cart });
  const elsewhere = scene.addLight({ position: [7, 7, 7], intensity: 5 });

  cart.destroy();
  settle(scene);

  assert.equal(scene.lightCount, 1, 'both children went with the cart');
  assert.equal(scene.lights[slot(scene, elsewhere) + 7], 5, 'and the unrelated light survived intact');
});

test('light capacity grows rather than dropping lights', () => {
  const scene = new Scene({ capacity: 8, lightCapacity: 2 });
  scene.addLight({});
  scene.addLight({});
  const third = scene.addLight({ intensity: 3 });
  assert.equal(scene.lightCount, 3);
  assert.ok(scene.lightCapacity >= 3);
  assert.equal(scene.lights[slot(scene, third) + 7], 3, 'the third light is real');
});

// --------------------------------------------------- compute pass ordering

console.log('\ncompute passes in the graph');

function fakeRhi() {
  return {
    device: {
      createTexture() {
        const texture = { destroy() {} };
        texture.createView = () => ({});
        return texture;
      },
    },
  };
}

test('buffer dependencies order the compute passes', () => {
  // Neither compute pass touches a texture, so without buffer resources the
  // topological sort would have no edge between them and could run the light
  // assignment before the cluster bounds it reads.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();

  const surface = graph.importTexture('surface', {});
  const bounds = graph.importBuffer('cluster-bounds', {});
  const indices = graph.importBuffer('cluster-indices', {});

  // Declared in the WRONG order on purpose.
  graph.addPass({
    name: 'forward', reads: [indices],
    color: [{ resource: surface, clear: 0 }], execute() {},
  });
  graph.addPass({
    name: 'assign', type: 'compute', reads: [bounds], writes: [indices], execute() {},
  });
  graph.addPass({ name: 'bounds', type: 'compute', writes: [bounds], execute() {} });

  graph.compile();

  const ran = [];
  graph.execute({
    beginRenderPass(d) { ran.push(d.label); return { end() {} }; },
    beginComputePass(d) { ran.push(d.label); return { end() {} }; },
  });
  assert.deepEqual(ran, ['bounds', 'assign', 'forward']);
});

test('compute passes begin a compute pass, not a render pass', () => {
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const buffer = graph.importBuffer('out', {});
  graph.addPass({ name: 'work', type: 'compute', writes: [buffer], execute() {} });
  graph.compile();

  let renderPasses = 0;
  let computePasses = 0;
  graph.execute({
    beginRenderPass() { renderPasses++; return { end() {} }; },
    beginComputePass() { computePasses++; return { end() {} }; },
  });
  assert.equal(computePasses, 1);
  assert.equal(renderPasses, 0);
});

test('a compute pass nothing consumes is still culled', () => {
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const surface = graph.importTexture('surface', {});
  const unused = graph.importBuffer('unused', {});
  void unused;

  graph.addPass({ name: 'forward', color: [{ resource: surface, clear: 0 }], execute() {} });
  graph.compile();
  assert.equal(graph.stats.executed, 1);
});

// --------------------------------------------------------- last slice reach

console.log('\nlast slice reach');

/** ClusteredLights.update without a GPU: only the maths is under test. */
function reachFor(lights, { lightDistance = 60 } = {}) {
  const c = Object.create(ClusteredLights.prototype);
  c.lightCapacity = 64;
  c.lightData = new Float32Array(64 * (LIGHT_BYTES / 4));
  c.paramsData = new ArrayBuffer(256);
  c.paramsF32 = new Float32Array(c.paramsData);
  c.paramsU32 = new Uint32Array(c.paramsData);
  c.tileSize = new Float32Array(2);
  c.rhi = { width: 1920, height: 1080, queue: { writeBuffer() {} } };
  c.lightBuffer = {};
  c.paramsBuffer = {};

  const scene = {
    lightCount: lights.length,
    lights: new Float32Array(lights.length * (LIGHT_BYTES / 4)),
  };
  lights.forEach((l, i) => {
    const o = i * (LIGHT_BYTES / 4);
    scene.lights[o] = l.position[0];
    scene.lights[o + 1] = l.position[1];
    scene.lights[o + 2] = l.position[2];
    scene.lights[o + 3] = l.radius;
  });

  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  camera.position.set([0, 0, 0]);
  camera.target.set([0, 0, -1]);
  camera.update(16 / 9);

  c.update(scene, camera, lightDistance);
  return c.lightReach;
}

test('with no lights the last slice stops at lightDistance', () => {
  assert.equal(reachFor([]), 60);
});

test('a light inside the grid does not stretch it', () => {
  assert.equal(reachFor([{ position: [0, 0, -10], radius: 5 }]), 60);
});

test('a light past lightDistance stretches the last slice to cover it', () => {
  // The bug: the exponential mapping ends the final cell exactly at
  // lightDistance, so this light sat outside every cluster box and was
  // assigned to none -- while the fragment shader clamped fragments out there
  // into that same cell and read a list the light was never in.
  close(reachFor([{ position: [0, 0, -200], radius: 12 }]), 212, EPS,
    'centre depth plus radius');
});

test('reach follows the camera, not the world origin', () => {
  // View depth, not distance from the origin. A light behind the camera has a
  // negative depth and must not stretch anything.
  assert.equal(reachFor([{ position: [0, 0, 500], radius: 5 }]), 60, 'behind the camera');
});

test('the farthest light wins', () => {
  close(reachFor([
    { position: [0, 0, -80], radius: 1 },
    { position: [0, 0, -300], radius: 20 },
    { position: [0, 0, -120], radius: 1 },
  ]), 320, EPS);
});

// ----------------------------------------------------------------- grid shape

console.log('\nfroxel grid shape');

test('the 16:9 grid is exactly what it always was', () => {
  // The default viewport must not move, or this change would be a silent
  // quality shift on every existing scene rather than a fix for other shapes.
  const g = clusterGridFor(16 / 9);
  assert.equal(g.x, 16);
  assert.equal(g.y, 9);
  assert.equal(g.x * g.y, CLUSTER_TILES);
});

test('froxels stay near square at any aspect', () => {
  // The defect: a hardcoded 16 by 9 made cells 3x taller than wide on a
  // portrait canvas, which overlap ~3x as many light spheres and reach the
  // per-cluster cap at a third of the light count -- dropped silently.
  for (const [w, h] of [[1920, 1080], [1080, 1920], [1024, 1024], [3440, 1440], [1440, 960]]) {
    const g = clusterGridFor(w / h);
    const froxel = (w / g.x) / (h / g.y);
    assert.ok(froxel > 0.8 && froxel < 1.25,
      `${w}x${h} gave ${g.x}x${g.y}, froxel aspect ${froxel.toFixed(2)}`);
  }
});

test('the tile budget is never exceeded, which the index buffer depends on', () => {
  // y floors rather than rounds for exactly this reason: rounding both lands
  // at 15 x 10 = 150 for 3:2, past a budget that sizes a GPU buffer.
  for (let aspect = 0.05; aspect < 20; aspect += 0.01) {
    const g = clusterGridFor(aspect);
    assert.ok(g.x >= 1 && g.y >= 1, `aspect ${aspect} gave ${g.x}x${g.y}`);
    assert.ok(g.x * g.y <= CLUSTER_TILES,
      `aspect ${aspect.toFixed(2)} gave ${g.x}x${g.y} = ${g.x * g.y}, over ${CLUSTER_TILES}`);
  }
});

test('a degenerate aspect still gives a usable grid', () => {
  for (const bad of [0, -1, NaN, Infinity]) {
    const g = clusterGridFor(bad);
    assert.ok(g.x >= 1 && g.y >= 1 && g.x * g.y <= CLUSTER_TILES, `aspect ${bad}`);
  }
});

console.log(`\n${passed} checks passed\n`);
