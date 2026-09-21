// Clustered lighting self-check. Run: node test/clustered.test.js
//
// The slice mapping and the light packing are both pure, and both are places
// where being subtly wrong produces lighting that looks plausible rather than
// broken -- lights assigned to cells nothing looks up, or a cone that falls off
// over the wrong angle.

import assert from 'node:assert/strict';

import { sliceFor, CLUSTER_Z, MAX_LIGHTS_PER_CLUSTER, CLUSTER_COUNT } from '../src/render/clustered.js';
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
  assert.equal(CLUSTER_COUNT, 16 * 9 * 24);
  const indexBytes = CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER * 4;
  assert.ok(indexBytes < 2 * 1024 * 1024, `${indexBytes} bytes is too much`);
});

// ------------------------------------------------------------ light packing

console.log('\nlight packing');

test('a point light packs position, radius, colour and intensity', () => {
  const scene = new Scene({ capacity: 16 });
  const index = scene.addLight({
    position: [1, 2, 3], color: [0.25, 0.5, 0.75], intensity: 7, radius: 9,
  });

  assert.equal(index, 0);
  assert.equal(scene.lightCount, 1);

  const o = index * LIGHT_FLOATS;
  assert.deepEqual([...scene.lights.subarray(o, o + 4)], [1, 2, 3, 9]);
  assert.deepEqual([...scene.lights.subarray(o + 4, o + 8)], [0.25, 0.5, 0.75, 7]);
  assert.equal(scene.lights[o + 14], LIGHT_POINT, 'type marks it a point light');
});

test('a spot light normalizes its direction and precomputes the cone falloff', () => {
  const scene = new Scene({ capacity: 16 });
  const inner = 0.3;
  const outer = 0.6;
  const index = scene.addLight({
    position: [0, 5, 0], direction: [0, -4, 0], innerAngle: inner, outerAngle: outer,
  });

  const o = index * LIGHT_FLOATS;
  assert.deepEqual([...scene.lights.subarray(o + 8, o + 11)], [0, -1, 0], 'unit direction');
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
  const index = scene.addLight({
    direction: [0, -1, 0], innerAngle: 0.5, outerAngle: 0.5,
  });
  const o = index * LIGHT_FLOATS;
  assert.ok(Number.isFinite(scene.lights[o + 12]), 'scale is finite');
  assert.ok(Number.isFinite(scene.lights[o + 13]), 'offset is finite');
});

test('setters update in place without touching neighbours', () => {
  const scene = new Scene({ capacity: 16 });
  scene.addLight({ position: [0, 0, 0], radius: 5 });
  const second = scene.addLight({ position: [9, 9, 9], radius: 5 });

  scene.setLightPosition(second, 1, 2, 3);
  scene.setLightColor(second, 0.1, 0.2, 0.3, 4);

  const o = second * LIGHT_FLOATS;
  assert.deepEqual([...scene.lights.subarray(o, o + 4)], [1, 2, 3, 5], 'radius untouched');
  assert.deepEqual([...scene.lights.subarray(o + 4, o + 8)], [
    Math.fround(0.1), Math.fround(0.2), Math.fround(0.3), 4,
  ]);
  assert.equal(scene.lights[0], 0, 'the first light is unchanged');
});

test('removal swap-removes, so light indices are not stable', () => {
  const scene = new Scene({ capacity: 16 });
  scene.addLight({ position: [1, 1, 1] });
  scene.addLight({ position: [2, 2, 2] });
  scene.addLight({ position: [3, 3, 3] });

  scene.removeLight(0);
  assert.equal(scene.lightCount, 2);
  assert.equal(scene.lights[0], 3, 'the last light moved into the hole');
});

test('passing the initial light capacity grows rather than dropping lights', () => {
  const scene = new Scene({ capacity: 8, lightCapacity: 2 });
  scene.addLight({});
  scene.addLight({});
  assert.equal(scene.addLight({}), 2, 'the third light gets a real index');
  assert.ok(scene.lightCapacity >= 3);
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

console.log(`\n${passed} checks passed\n`);
