// Render graph self-check. Run: node test/graph.test.js
//
// The graph is pure scheduling logic, so all of it tests without a GPU. The
// only thing faked is texture creation, which the aliasing pass calls.

import assert from 'node:assert/strict';
import { RenderGraph } from '../src/render/graph.js';

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`  ok  ${name}`);
}

/** Enough of an rhi for the aliasing pass to allocate against. */
function fakeRhi() {
  let created = 0;
  return {
    get created() { return created; },
    device: {
      createTexture(desc) {
        created++;
        const texture = { desc, id: created, destroy() {} };
        texture.createView = () => ({ texture });
        return texture;
      },
    },
  };
}

/** Record which passes ran, in order. */
function recorder() {
  const ran = [];
  const encoder = {
    beginRenderPass(desc) {
      ran.push(desc);
      return { end() {} };
    },
  };
  return { ran, encoder };
}

const COLOR = { width: 8, height: 8, format: 'rgba8unorm', usage: 0x10 };

// ------------------------------------------------------------------- order

console.log('\nordering');

test('passes run in dependency order, not declaration order', () => {
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const target = graph.importTexture('target', {});
  const intermediate = graph.createTexture('intermediate', COLOR);

  // Declared consumer-first on purpose.
  graph.addPass({
    name: 'consumer',
    reads: [intermediate],
    color: [{ resource: target, clear: 0 }],
    execute() {},
  });
  graph.addPass({
    name: 'producer',
    color: [{ resource: intermediate, clear: 0 }],
    execute() {},
  });
  graph.compile();

  const { ran, encoder } = recorder();
  graph.execute(encoder);
  assert.deepEqual(ran.map((p) => p.label), ['producer', 'consumer']);
});

test('write-after-write keeps repeated writers in declaration order', () => {
  // The four shadow cascades all write one texture array; nothing reads
  // between them, so only this edge keeps them from being reordered.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const shadows = graph.importTexture('shadows', {});

  for (let i = 0; i < 4; i++) {
    graph.addPass({ name: `shadow:${i}`, depth: { resource: shadows, clear: 0 }, execute() {} });
  }
  graph.compile();

  const { ran, encoder } = recorder();
  graph.execute(encoder);
  assert.deepEqual(ran.map((p) => p.label), ['shadow:0', 'shadow:1', 'shadow:2', 'shadow:3']);
});

test('a cycle is detected rather than silently dropping passes', () => {
  // Each pass reads what the other writes. There is no order that satisfies
  // both, and the failure mode without a check is a pass quietly vanishing
  // from the frame.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const a = graph.importTexture('a', {});
  const b = graph.importTexture('b', {});

  graph.addPass({
    name: 'first', reads: [b], color: [{ resource: a, clear: 0 }], execute() {},
  });
  graph.addPass({
    name: 'second', reads: [a], color: [{ resource: b, clear: 0 }], execute() {},
  });

  assert.throws(() => graph.compile(), /cycle/);
});

// ------------------------------------------------------------- store / load

console.log('\nload and store derivation');

test('depth nothing reads afterwards is discarded', () => {
  // The payoff: on a tile-based GPU a discarded depth buffer is never written
  // back to main memory at all.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const color = graph.importTexture('color', {});
  const depth = graph.createTexture('depth', { ...COLOR, format: 'depth32float' });

  graph.addPass({
    name: 'forward',
    color: [{ resource: color, clear: 0 }],
    depth: { resource: depth, clear: 0 },
    execute() {},
  });
  graph.compile();

  const { ran, encoder } = recorder();
  graph.execute(encoder);
  assert.equal(ran[0].depthStencilAttachment.depthStoreOp, 'discard');
  assert.equal(ran[0].colorAttachments[0].storeOp, 'store', 'imported colour is always kept');
});

test('a resource a later pass reads is stored', () => {
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const target = graph.importTexture('target', {});
  const shadows = graph.createTexture('shadows', { ...COLOR, format: 'depth32float' });

  graph.addPass({ name: 'shadow', depth: { resource: shadows, clear: 0 }, execute() {} });
  graph.addPass({
    name: 'forward',
    reads: [shadows],
    color: [{ resource: target, clear: 0 }],
    execute() {},
  });
  graph.compile();

  const { ran, encoder } = recorder();
  graph.execute(encoder);
  assert.equal(ran[0].depthStencilAttachment.depthStoreOp, 'store', 'forward reads it');
});

test('adding a reader flips discard to store with no declaration change', () => {
  // The reason this is derived rather than written down: the correct answer
  // changes when a DIFFERENT pass is added, and nobody would remember to go
  // back and edit the first one.
  const build = (withReader) => {
    const graph = new RenderGraph(fakeRhi());
    graph.begin();
    const target = graph.importTexture('target', {});
    const depth = graph.createTexture('depth', { ...COLOR, format: 'depth32float' });
    graph.addPass({
      name: 'forward',
      color: [{ resource: target, clear: 0 }],
      depth: { resource: depth, clear: 0 },
      execute() {},
    });
    if (withReader) {
      graph.addPass({ name: 'ssao', reads: [depth], color: [{ resource: target }], execute() {} });
    }
    graph.compile();
    const { ran, encoder } = recorder();
    graph.execute(encoder);
    return ran[0].depthStencilAttachment.depthStoreOp;
  };

  assert.equal(build(false), 'discard');
  assert.equal(build(true), 'store');
});

test('a second writer loads instead of clearing', () => {
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const color = graph.importTexture('color', {});

  graph.addPass({ name: 'sky', color: [{ resource: color, clear: 0 }], execute() {} });
  // No clear value: it must preserve what the sky pass drew.
  graph.addPass({ name: 'geometry', color: [{ resource: color }], execute() {} });
  graph.compile();

  const { ran, encoder } = recorder();
  graph.execute(encoder);
  assert.equal(ran[0].colorAttachments[0].loadOp, 'clear');
  assert.equal(ran[1].colorAttachments[0].loadOp, 'load');
});

test('loading a resource nothing wrote is rejected', () => {
  // Would sample undefined memory. A declaration bug, not something to paper
  // over with a silent clear.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const color = graph.importTexture('color', {});
  graph.addPass({ name: 'bad', color: [{ resource: color }], execute() {} });
  assert.throws(() => graph.compile(), /no clear value/);
});

// -------------------------------------------------------------------- cull

console.log('\ndead pass elimination');

test('a pass whose output nothing consumes does not execute', () => {
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const target = graph.importTexture('target', {});
  const orphan = graph.createTexture('orphan', COLOR);

  graph.addPass({ name: 'wasted', color: [{ resource: orphan, clear: 0 }], execute() {} });
  graph.addPass({ name: 'forward', color: [{ resource: target, clear: 0 }], execute() {} });
  graph.compile();

  assert.equal(graph.stats.culled, 1);
  const { ran, encoder } = recorder();
  graph.execute(encoder);
  assert.deepEqual(ran.map((p) => p.label), ['forward']);
});

test('a producer whose consumer survives is kept', () => {
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const target = graph.importTexture('target', {});
  const used = graph.createTexture('used', COLOR);

  graph.addPass({ name: 'producer', color: [{ resource: used, clear: 0 }], execute() {} });
  graph.addPass({
    name: 'consumer', reads: [used], color: [{ resource: target, clear: 0 }], execute() {},
  });
  graph.compile();

  assert.equal(graph.stats.culled, 0);
  assert.equal(graph.stats.executed, 2);
});

// ----------------------------------------------------------------- aliasing

console.log('\nresource aliasing');

test('transients with disjoint lifetimes share one texture', () => {
  // A bloom ladder is a chain of these. Without aliasing it is one allocation
  // per step; with it, two.
  const rhi = fakeRhi();
  const graph = new RenderGraph(rhi);
  graph.begin();
  const target = graph.importTexture('target', {});
  const a = graph.createTexture('a', COLOR);
  const b = graph.createTexture('b', COLOR);

  graph.addPass({ name: 'p0', color: [{ resource: a, clear: 0 }], execute() {} });
  graph.addPass({ name: 'p1', reads: [a], color: [{ resource: b, clear: 0 }], execute() {} });
  // 'a' is dead after p1, so 'c' can take its memory.
  const c = graph.createTexture('c', COLOR);
  graph.addPass({ name: 'p2', reads: [b], color: [{ resource: c, clear: 0 }], execute() {} });
  graph.addPass({
    name: 'present', reads: [c], color: [{ resource: target, clear: 0 }], execute() {},
  });
  graph.compile();

  assert.equal(graph.stats.transient, 3);
  assert.equal(graph.stats.aliased, 1, 'c reused a');
  assert.equal(rhi.created, 2, 'three resources, two allocations');
});

test('overlapping lifetimes never share', () => {
  const rhi = fakeRhi();
  const graph = new RenderGraph(rhi);
  graph.begin();
  const target = graph.importTexture('target', {});
  const a = graph.createTexture('a', COLOR);
  const b = graph.createTexture('b', COLOR);

  graph.addPass({ name: 'p0', color: [{ resource: a, clear: 0 }], execute() {} });
  graph.addPass({ name: 'p1', color: [{ resource: b, clear: 0 }], execute() {} });
  // Both are still live here, so they cannot be the same memory.
  graph.addPass({
    name: 'combine', reads: [a, b], color: [{ resource: target, clear: 0 }], execute() {},
  });
  graph.compile();

  assert.equal(graph.stats.aliased, 0);
  assert.equal(rhi.created, 2);
});

test('different descriptors never share, even when lifetimes allow it', () => {
  const rhi = fakeRhi();
  const graph = new RenderGraph(rhi);
  graph.begin();
  const target = graph.importTexture('target', {});
  const small = graph.createTexture('small', COLOR);
  const large = graph.createTexture('large', { ...COLOR, width: 64, height: 64 });

  graph.addPass({ name: 'p0', color: [{ resource: small, clear: 0 }], execute() {} });
  graph.addPass({ name: 'p1', reads: [small], color: [{ resource: large, clear: 0 }], execute() {} });
  graph.addPass({
    name: 'present', reads: [large], color: [{ resource: target, clear: 0 }], execute() {},
  });
  graph.compile();

  assert.equal(graph.stats.aliased, 0, 'different sizes are not interchangeable');
  assert.equal(rhi.created, 2);
});

test('textures are pooled across frames, not recreated', () => {
  const rhi = fakeRhi();
  const graph = new RenderGraph(rhi);

  const buildFrame = () => {
    graph.begin();
    const target = graph.importTexture('target', {});
    const temp = graph.createTexture('temp', COLOR);
    graph.addPass({ name: 'p0', color: [{ resource: temp, clear: 0 }], execute() {} });
    graph.addPass({
      name: 'p1', reads: [temp], color: [{ resource: target, clear: 0 }], execute() {},
    });
    graph.compile();
  };

  buildFrame();
  assert.equal(rhi.created, 1);
  buildFrame();
  buildFrame();
  assert.equal(rhi.created, 1, 'the second and third frames reuse the first frame texture');
});

test('rebuilding a frame allocates no new pass or resource records', () => {
  // begin() resets counters over pooled records rather than freeing them, so a
  // steady-state frame adds nothing to the heap.
  const graph = new RenderGraph(fakeRhi());
  const firstPass = graph._passes[0];
  const firstResource = graph._resources[0];

  for (let frame = 0; frame < 3; frame++) {
    graph.begin();
    const target = graph.importTexture('target', {});
    graph.addPass({ name: `f${frame}`, color: [{ resource: target, clear: 0 }], execute() {} });
    graph.compile();
  }

  assert.equal(graph._passes[0], firstPass, 'pass record reused');
  assert.equal(graph._resources[0], firstResource, 'resource record reused');
  assert.equal(graph.passCount, 1, 'counters reset each frame');
});

// ------------------------------------------------------------------ describe

console.log('\nintrospection');

test('describe() reports the plan, including what was culled', () => {
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const target = graph.importTexture('target', {});
  const depth = graph.createTexture('depth', { ...COLOR, format: 'depth32float' });
  const orphan = graph.createTexture('orphan', COLOR);

  graph.addPass({ name: 'dead', color: [{ resource: orphan, clear: 0 }], execute() {} });
  graph.addPass({
    name: 'forward',
    color: [{ resource: target, clear: 0 }],
    depth: { resource: depth, clear: 0 },
    execute() {},
  });
  graph.compile();

  const description = graph.describe();
  assert.match(description, /forward/);
  assert.match(description, /depth:clear\/discard/);
  assert.match(description, /dead.*culled/);
});

// ------------------------------------------------- write-after-read limits

console.log('\nwrite-after-read');

test('writing a texture a later pass reads is reported as a cycle', () => {
  // A KNOWN LIMITATION, pinned here on purpose. Real frame graphs version
  // resources -- every write produces a new version and each read binds to a
  // specific one -- which lets a pass overwrite something an earlier-ordered
  // pass read. This graph has no versions, so it cannot order that and says so
  // rather than picking an order that is silently wrong.
  //
  // It is what stopped bloom from accumulating in place: the upsample wanted to
  // blend into a texture the next downsample had already read.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const surface = graph.importTexture('surface', {});
  const shared = graph.createTexture('shared', COLOR);
  const other = graph.createTexture('other', COLOR);

  graph.addPass({ name: 'produce', color: [{ resource: shared, clear: 0 }], execute() {} });
  graph.addPass({
    name: 'consume', reads: [shared], color: [{ resource: other, clear: 0 }], execute() {},
  });
  // Wants to run after 'consume', but writes what 'consume' reads.
  graph.addPass({ name: 'overwrite', reads: [other], color: [{ resource: shared }], execute() {} });
  graph.addPass({
    name: 'present', reads: [shared], color: [{ resource: surface, clear: 0 }], execute() {},
  });

  assert.throws(() => graph.compile(), /cycle/);
});

test('the cycle message names the passes that could not be ordered', () => {
  // "Contains a cycle" alone sends you reading the entire frame declaration.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const a = graph.importTexture('a', {});
  const b = graph.importTexture('b', {});
  graph.addPass({ name: 'alpha', reads: [b], color: [{ resource: a, clear: 0 }], execute() {} });
  graph.addPass({ name: 'beta', reads: [a], color: [{ resource: b, clear: 0 }], execute() {} });

  assert.throws(() => graph.compile(), /alpha|beta/);
});

test('the bloom chain shape compiles and orders correctly', () => {
  // Reading both inputs and writing a third texture has no hazard, which is
  // why the upsample adds in the shader instead of blending in place.
  const graph = new RenderGraph(fakeRhi(), { maxResources: 64 });
  graph.begin();

  const surface = graph.importTexture('surface', {});
  const scene = graph.createTexture('scene', COLOR);
  graph.addPass({ name: 'forward', color: [{ resource: scene, clear: 0 }], execute() {} });

  const levels = 4;
  const down = [];
  for (let i = 0; i < levels; i++) {
    down.push(graph.createTexture(`d${i}`, { ...COLOR, width: 8 >> i, height: 8 >> i }));
    graph.addPass({
      name: `down:${i}`,
      reads: [i === 0 ? scene : down[i - 1]],
      color: [{ resource: down[i], clear: 0 }],
      execute() {},
    });
  }

  let smaller = down[levels - 1];
  for (let i = levels - 2; i >= 0; i--) {
    const target = graph.createTexture(`u${i}`, { ...COLOR, width: 8 >> i, height: 8 >> i });
    graph.addPass({
      name: `up:${i}`,
      reads: [smaller, down[i]],
      color: [{ resource: target, clear: 0 }],
      execute() {},
    });
    smaller = target;
  }

  graph.addPass({
    name: 'tonemap', reads: [scene, smaller],
    color: [{ resource: surface, clear: 0 }], execute() {},
  });

  graph.compile();
  const { ran, encoder } = recorder();
  graph.execute(encoder);

  const order = ran.map((p) => p.label);
  assert.equal(order[0], 'forward');
  assert.equal(order[order.length - 1], 'tonemap');
  assert.ok(order.indexOf('down:3') < order.indexOf('up:2'), 'downsample before upsample');
  assert.ok(order.indexOf('up:2') < order.indexOf('up:0'), 'upsample walks back up');
});

console.log(`\n${passed} checks passed\n`);
