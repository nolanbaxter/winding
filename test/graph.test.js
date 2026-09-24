// Render graph self-check. Run: node test/graph.test.js
//
// The graph is pure scheduling logic, so all of it tests without a GPU. The
// only thing faked is texture creation, which the aliasing pass calls.

import assert from 'node:assert/strict';
import { RenderGraph } from '../src/render/graph.js';
import { GpuProfiler } from '../src/render/timing.js';

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
  // steady-state frame adds nothing to the heap. The pools start empty and
  // extend to the high-water mark, so the record to compare against is the one
  // the FIRST frame created -- which also makes this a test of reuse across
  // frames rather than of a preallocated slot's identity.
  const graph = new RenderGraph(fakeRhi());

  const buildFrame = (frame) => {
    graph.begin();
    const target = graph.importTexture('target', {});
    graph.addPass({ name: `f${frame}`, color: [{ resource: target, clear: 0 }], execute() {} });
    graph.compile();
  };

  buildFrame(0);
  const firstPass = graph._passes[0];
  const firstResource = graph._resources[0];
  assert.ok(firstPass && firstResource, 'the first frame created the records');

  buildFrame(1);
  buildFrame(2);

  assert.equal(graph._passes[0], firstPass, 'pass record reused');
  assert.equal(graph._resources[0], firstResource, 'resource record reused');
  assert.equal(graph._passes.length, 1, 'the pool did not grow past the high-water mark');
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

// ------------------------------------------------------ repeated writes

console.log('\nrepeated writes');

test('a pass may overwrite something an earlier pass read', () => {
  // Resource versioning, positional: a resource written more than once is a
  // SEQUENCE of values, and a read means the one current where the read was
  // declared. Without that, 'present' would edge from 'overwrite' as well as
  // 'produce' and there would be no order at all -- which is what this graph
  // used to report as a cycle.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const surface = graph.importTexture('surface', {});
  const shared = graph.createTexture('shared', COLOR);
  const other = graph.createTexture('other', COLOR);

  graph.addPass({ name: 'produce', color: [{ resource: shared, clear: 0 }], execute() {} });
  graph.addPass({
    name: 'consume', reads: [shared], color: [{ resource: other, clear: 0 }], execute() {},
  });
  // Runs after 'consume', and writes what 'consume' read.
  graph.addPass({ name: 'overwrite', reads: [other], color: [{ resource: shared }], execute() {} });
  graph.addPass({
    name: 'present', reads: [shared], color: [{ resource: surface, clear: 0 }], execute() {},
  });
  graph.compile();

  const { ran, encoder } = recorder();
  graph.execute(encoder);
  assert.deepEqual(ran.map((d) => d.label), ['produce', 'consume', 'overwrite', 'present']);
  assert.equal(ran[2].colorAttachments[0].loadOp, 'load',
    'overwrite adds to what produce wrote, so it must not clear');
});

test('a read still finds its producer when the producer is declared later', () => {
  // The single-writer rule is unchanged, and it is the one that matters most:
  // declaration order must not be load-bearing where there is nothing to
  // sequence.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const surface = graph.importTexture('surface', {});
  const data = graph.createTexture('data', COLOR);

  graph.addPass({
    name: 'consumer', reads: [data], color: [{ resource: surface, clear: 0 }], execute() {},
  });
  graph.addPass({ name: 'producer', color: [{ resource: data, clear: 0 }], execute() {} });
  graph.compile();

  const { ran, encoder } = recorder();
  graph.execute(encoder);
  assert.deepEqual(ran.map((d) => d.label), ['producer', 'consumer']);
});

test('two passes each waiting on the other is still a cycle', () => {
  // What versioning does not and cannot fix: a mutual dependency through two
  // single-writer resources has no valid order in either direction.
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const surface = graph.importTexture('surface', {});
  const x = graph.createTexture('x', COLOR);
  const y = graph.createTexture('y', COLOR);

  graph.addPass({ name: 'a', reads: [y], color: [{ resource: x, clear: 0 }], execute() {} });
  graph.addPass({ name: 'b', reads: [x], color: [{ resource: y, clear: 0 }], execute() {} });
  graph.addPass({
    name: 'present', reads: [x], color: [{ resource: surface, clear: 0 }], execute() {},
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
  const graph = new RenderGraph(fakeRhi());
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

// ------------------------------------------------------------------ timing

console.log('\nGPU pass timing');

// The profiler names WebGPU's usage flags the way every other module does, so
// Node needs them to exist. Real values, in case anything ever ORs them.
globalThis.GPUBufferUsage ??= {
  MAP_READ: 0x0001, COPY_SRC: 0x0004, COPY_DST: 0x0008, QUERY_RESOLVE: 0x0200,
};
globalThis.GPUMapMode ??= { READ: 0x0001 };

/**
 * Run one frame of `passes` passes so the profiler sizes itself, then open the
 * next frame. The first frame is always untimed by design: the pass count is
 * whatever the graph declares, so the profiler learns it by being asked.
 */
function warmed(profiler, passes) {
  profiler.begin();
  for (let i = 0; i < passes; i++) profiler.writesFor(i, `warm${i}`);
  profiler.begin();
  return profiler;
}



/**
 * A device that records what the profiler asks of it. Timestamps are handed
 * back as a fixed ramp so durations are predictable: pass i takes (i+1) ms.
 */
function timingRhi({ feature = true } = {}) {
  const copies = [];
  const resolves = [];
  return {
    copies,
    resolves,
    features: { has: (name) => feature && name === 'timestamp-query' },
    device: {
      createQuerySet: (desc) => ({ ...desc, destroy() {} }),
      createBuffer(desc) {
        const bytes = new ArrayBuffer(desc.size);
        return {
          size: desc.size,
          destroy() {},
          mapAsync: () => Promise.resolve(),
          getMappedRange() {
            const stamps = new BigInt64Array(bytes);
            for (let i = 0; i < stamps.length / 2; i++) {
              stamps[i * 2] = BigInt(i) * 1000000n;
              stamps[i * 2 + 1] = BigInt(i) * 1000000n + BigInt(i + 1) * 1000000n;
            }
            return bytes;
          },
          unmap() {},
        };
      },
    },
  };
}

const encoderStub = () => ({
  resolves: [],
  resolveQuerySet(set, first, count) { this.resolves.push({ first, count }); },
  copyBufferToBuffer() {},
});

test('a device without the feature makes every call inert', () => {
  const profiler = new GpuProfiler(timingRhi({ feature: false }));
  assert.equal(profiler.supported, false);
  profiler.begin();
  assert.equal(profiler.writesFor(0, 'forward'), undefined);
  profiler.resolve(encoderStub());
  assert.deepEqual(profiler.results, []);
});

test('enabled:false turns it off on a device that does support it', () => {
  const profiler = new GpuProfiler(timingRhi(), { enabled: false });
  assert.equal(profiler.supported, false);
  profiler.begin();
  assert.equal(profiler.writesFor(0, 'forward'), undefined);
});

test('each pass gets its own pair of query indices', () => {
  const profiler = warmed(new GpuProfiler(timingRhi()), 2);
  const first = profiler.writesFor(0, 'cull');
  const second = profiler.writesFor(1, 'forward');
  assert.equal(first.beginningOfPassWriteIndex, 0);
  assert.equal(first.endOfPassWriteIndex, 1);
  assert.equal(second.beginningOfPassWriteIndex, 2);
  assert.equal(second.endOfPassWriteIndex, 3);
  assert.equal(first.querySet, second.querySet, 'one set for the whole frame');
});

test('a pass past what is allocated is untimed for one frame, then fits', () => {
  // Nothing is dropped for good: the demand is recorded and the next begin()
  // widens to it, which is what lets the profiler follow a graph that has no
  // pass ceiling of its own.
  const profiler = warmed(new GpuProfiler(timingRhi()), 1);
  assert.ok(profiler.writesFor(0, 'in'));
  assert.equal(profiler.writesFor(1, 'out'), undefined, 'untimed this frame');

  profiler.begin();
  assert.ok(profiler.writesFor(0, 'in'));
  assert.ok(profiler.writesFor(1, 'out'), 'and timed the next');
});

test('begin() forgets the previous frame, so names cannot accumulate', () => {
  const profiler = warmed(new GpuProfiler(timingRhi()), 8);
  profiler.writesFor(0, 'a');
  profiler.begin();
  const again = profiler.writesFor(0, 'a');
  assert.equal(again.beginningOfPassWriteIndex, 0, 'indices restart with the frame');
});

test('resolve only covers the passes that actually ran', () => {
  const profiler = warmed(new GpuProfiler(timingRhi()), 64);
  profiler.writesFor(0, 'a');
  profiler.writesFor(1, 'b');
  const encoder = encoderStub();
  profiler.resolve(encoder);
  assert.deepEqual(encoder.resolves, [{ first: 0, count: 4 }],
    'resolving all 64 slots would read queries nothing wrote');
});

test('a frame with no passes resolves nothing', () => {
  const profiler = new GpuProfiler(timingRhi());
  profiler.begin();
  const encoder = encoderStub();
  profiler.resolve(encoder);
  assert.deepEqual(encoder.resolves, []);
});

test('the ring skips instead of stalling when every buffer is in flight', () => {
  const profiler = warmed(new GpuProfiler(timingRhi(), { depth: 2 }), 1);
  const encoder = encoderStub();
  // Three frames back to back, nothing given a chance to unmap in between.
  for (let i = 0; i < 3; i++) {
    profiler.begin();
    profiler.writesFor(0, 'a');
    profiler.resolve(encoder);
    profiler.readback();
  }
  assert.equal(encoder.resolves.length, 2, 'the third frame is dropped, not queued');
});

test('readback without a resolve does nothing, and does not repeat itself', () => {
  const profiler = warmed(new GpuProfiler(timingRhi(), { depth: 1 }), 1);
  profiler.readback();

  profiler.begin();
  profiler.writesFor(0, 'a');
  profiler.resolve(encoderStub());
  profiler.readback();
  // A second readback must not map the same buffer twice: WebGPU rejects that,
  // and the ring would lose the slot.
  profiler.readback();
});

test('the graph hands every pass its timestamp writes, render and compute', () => {
  const profiler = warmed(new GpuProfiler(timingRhi()), 2);
  const graph = new RenderGraph(fakeRhi(), { profiler });
  graph.begin();
  // An imported target: a transient nothing reads is a dead pass, and a dead
  // pass is never recorded, so there would be nothing to check.
  const target = graph.importTexture('surface', {});
  const buffer = graph.importBuffer('counts', {});
  graph.addPass({ name: 'cull', type: 'compute', writes: [buffer], execute: () => {} });
  graph.addPass({
    name: 'forward',
    color: [{ resource: target, clear: [0, 0, 0, 1] }],
    reads: [buffer],
    execute: () => {},
  });
  graph.compile();

  const seen = [];
  graph.execute({
    beginComputePass(desc) { seen.push(desc); return { end() {} }; },
    beginRenderPass(desc) { seen.push(desc); return { end() {} }; },
  });

  assert.equal(seen.length, 2);
  for (const desc of seen) {
    assert.ok(desc.timestampWrites, `${desc.label} is timed`);
  }
  assert.equal(seen[0].timestampWrites.endOfPassWriteIndex + 1,
    seen[1].timestampWrites.beginningOfPassWriteIndex,
    'indices follow execution order, not declaration order');
});

test('no profiler means no timestampWrites key to confuse a driver', () => {
  const graph = new RenderGraph(fakeRhi());
  graph.begin();
  const target = graph.importTexture('surface', {});
  graph.addPass({ name: 'only', color: [{ resource: target, clear: [0, 0, 0, 1] }], execute: () => {} });
  graph.compile();

  let seen = null;
  graph.execute({ beginRenderPass(desc) { seen = desc; return { end() {} }; } });
  assert.equal(seen.timestampWrites, undefined);
});

// Readback is a promise, so this one sits outside the sync helper.
{
  const profiler = warmed(new GpuProfiler(timingRhi()), 2);
  profiler.writesFor(0, 'cull');
  profiler.writesFor(1, 'forward');
  profiler.resolve(encoderStub());
  profiler.readback();
  await new Promise((resolve) => setTimeout(resolve, 0));

  // The fake ramps pass i to (i+1) ms.
  assert.deepEqual(profiler.results, [{ name: 'cull', ms: 1 }, { name: 'forward', ms: 2 }]);
  assert.equal(profiler.totalMs, 3);
  assert.deepEqual(profiler.slowest(1), [{ name: 'forward', ms: 2 }]);
  passed++;
  console.log('  ok  a resolved readback reports a duration per named pass');
}

// ---------------------------------------------------------- compile reuse

console.log('\ncompile reuse');

/** A small frame: a transient bloom-like chain into an imported target. */
function declareFrame(graph, { size = 8, reader = true, clearTarget = true, extraPass = false } = {}) {
  graph.begin();
  const target = graph.importTexture('target', { id: 'target-view' });
  const scratch = graph.createTexture('scratch', { ...COLOR, width: size, height: size });
  const unused = graph.createTexture('unused', COLOR);
  graph.addPass({ name: 'produce', color: [{ resource: scratch, clear: 0 }], execute() {} });
  // Writes a transient nothing reads unless `reader`: culled or live by that.
  graph.addPass({ name: 'maybe-dead', color: [{ resource: unused, clear: 0 }], execute() {} });
  graph.addPass({
    name: 'consume',
    reads: reader ? [scratch, unused] : [scratch],
    color: [{ resource: target, clear: clearTarget ? 0 : undefined }],
    execute() {},
  });
  if (extraPass) {
    graph.addPass({ name: 'overlay', color: [{ resource: target }], execute() {} });
  }
  graph.compile();
  const { ran, encoder } = recorder();
  graph.execute(encoder);
  return ran.map((p) => ({
    label: p.label,
    ops: p.colorAttachments.map((a) => `${a.loadOp}/${a.storeOp}`).join(),
    view: p.colorAttachments[0].view,
  }));
}

/** An rhi that counts what it creates and destroys. */
function countingRhi() {
  const counts = { created: 0, destroyed: 0 };
  return {
    counts,
    device: {
      createTexture(desc) {
        counts.created++;
        const texture = { desc, destroy() { counts.destroyed++; } };
        texture.createView = () => ({ texture });
        return texture;
      },
    },
  };
}

test('an identical frame reuses the last compile, and gets the same answer', () => {
  const rhi = countingRhi();
  const graph = new RenderGraph(rhi);
  const first = declareFrame(graph);
  assert.equal(graph.stats.reused, false, 'the first frame compiles');

  const second = declareFrame(graph);
  assert.equal(graph.stats.reused, true, 'the second is the same declaration');
  assert.deepEqual(second.map((p) => p.label), first.map((p) => p.label), 'same order');
  assert.deepEqual(second.map((p) => p.ops), first.map((p) => p.ops), 'same load/store ops');
  assert.deepEqual(second.map((p) => p.view), first.map((p) => p.view), 'same physical textures');
  assert.equal(rhi.counts.created, 2, 'nothing new allocated');
});

test('reused textures are kept alive, not evicted as unasked-for', () => {
  // The pool evicts what a frame did not acquire. A reused compile acquires
  // nothing, so without re-marking its textures they died two frames later
  // while the graph was still handing out their views.
  const rhi = countingRhi();
  const graph = new RenderGraph(rhi);
  for (let f = 0; f < 10; f++) declareFrame(graph);
  assert.equal(rhi.counts.destroyed, 0);
  assert.equal(rhi.counts.created, 2);
});

test('a pass that comes alive recompiles, and runs', () => {
  const graph = new RenderGraph(countingRhi());
  const without = declareFrame(graph, { reader: false });
  assert.ok(!without.some((p) => p.label === 'maybe-dead'), 'culled while nothing reads it');

  const withReader = declareFrame(graph, { reader: true });
  assert.equal(graph.stats.reused, false);
  assert.ok(withReader.some((p) => p.label === 'maybe-dead'), 'live once something reads it');
});

test('dropping a clear recompiles the load op', () => {
  const graph = new RenderGraph(countingRhi());
  declareFrame(graph, { extraPass: true });
  const overlay = declareFrame(graph, { extraPass: true }).find((p) => p.label === 'overlay');
  assert.equal(overlay.ops, 'load/store', 'loads what consume wrote');

  // Now consume does not clear either, and nothing wrote target before it.
  // DEBUG builds refuse that declaration outright; either way it must not be
  // mistaken for the frame before.
  let reused = true;
  try {
    declareFrame(graph, { extraPass: true, clearTarget: false });
    reused = graph.stats.reused;
  } catch { reused = false; }
  assert.equal(reused, false);
});

test('a resize recompiles and takes a texture of the new size', () => {
  const rhi = countingRhi();
  const graph = new RenderGraph(rhi);
  const small = declareFrame(graph, { size: 8 });
  const large = declareFrame(graph, { size: 16 });
  assert.equal(graph.stats.reused, false);
  const produced = (frame) => frame.find((p) => p.label === 'produce').view.texture.desc.size[0];
  assert.equal(produced(small), 8);
  assert.equal(produced(large), 16);
});

test('a pass added recompiles', () => {
  const graph = new RenderGraph(countingRhi());
  declareFrame(graph);
  const frame = declareFrame(graph, { extraPass: true });
  assert.equal(graph.stats.reused, false);
  assert.ok(frame.some((p) => p.label === 'overlay'));
});

test('destroy forgets the kept compile, whose textures it just destroyed', () => {
  const rhi = countingRhi();
  const graph = new RenderGraph(rhi);
  declareFrame(graph);
  graph.destroy();
  declareFrame(graph);
  assert.equal(graph.stats.reused, false);
  assert.equal(rhi.counts.created, 4, 'fresh textures, not the destroyed ones');
});

test('a clear with no clear value still gets one when executed', () => {
  // The release-build fallback for an attachment nothing wrote used to write
  // its zero into the declaration's clear field, which the next addPass
  // resets -- so a reused compile kept loadOp 'clear' with no value, which
  // WebGPU rejects. execute() now supplies the value.
  const graph = new RenderGraph(countingRhi());
  graph.begin();
  const target = graph.importTexture('target', {});
  const depth = graph.createTexture('depth', { ...COLOR, format: 'depth32float' });
  graph.addPass({ name: 'draw', color: [{ resource: target, clear: 0 }], depth: { resource: depth, clear: 0 }, execute() {} });
  graph.compile();
  // What the fallback leaves behind on a reused frame: 'clear', and no value.
  const pass = graph._passes[0];
  pass.color[0].clear = undefined;
  pass.depth.clear = undefined;
  const { ran, encoder } = recorder();
  graph.execute(encoder);
  assert.deepEqual(ran[0].colorAttachments[0].clearValue, { r: 0, g: 0, b: 0, a: 0 });
  assert.equal(ran[0].depthStencilAttachment.depthClearValue, 0);
});

console.log(`\n${passed} checks passed\n`);
