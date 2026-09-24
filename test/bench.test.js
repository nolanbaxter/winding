// Benchmark self-check. Run: node test/bench.test.js
//
// The renderer's side -- that every phase is marked, in order, and the marks
// cost nothing when no benchmark is attached -- needs a GPU and is checked in
// gpu.test.js. This is the benchmark's own bookkeeping, against a stand-in.

import assert from 'node:assert/strict';

import { Benchmark, summarize, boundBy } from '../src/bench.js';

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

/** Just enough engine: the two renderer fields a Benchmark touches. */
function fakeEngine({ available = true } = {}) {
  return {
    renderer: {
      profiler: null,
      gpuTiming: {
        available, enabled: false, average: [], averageTotalMs: 0, samples: 0,
        resets: 0,
        resetAverages() { this.resets++; },
      },
    },
  };
}

/** One frame of three phases, driven the way Renderer.render drives it. */
function frame(bench, phases) {
  bench.frameStart();
  for (const name of phases) bench.mark(name);
  bench.frameEnd();
}

console.log('\nbenchmark');

test('attaching is the only thing that turns anything on, and stop puts it back', () => {
  const engine = fakeEngine();
  const bench = new Benchmark(engine);
  assert.equal(engine.renderer.profiler, null, 'constructing attaches nothing');
  assert.equal(engine.renderer.gpuTiming.enabled, false);

  bench.start();
  assert.equal(engine.renderer.profiler, bench);
  assert.equal(engine.renderer.gpuTiming.enabled, true, 'GPU timing only while running');
  assert.equal(engine.renderer.gpuTiming.resets, 1, 'and from a clean slate');

  bench.stop();
  assert.equal(engine.renderer.profiler, null);
  assert.equal(engine.renderer.gpuTiming.enabled, false, 'restored, not merely switched off');
});

test('a renderer that already had GPU timing on keeps it on', () => {
  const engine = fakeEngine();
  engine.renderer.gpuTiming.enabled = true;
  new Benchmark(engine).start().stop();
  assert.equal(engine.renderer.gpuTiming.enabled, true);
});

test('two benchmarks cannot share a renderer', () => {
  const engine = fakeEngine();
  new Benchmark(engine).start();
  assert.throws(() => new Benchmark(engine).start(), /already attached/);
});

test('the phases partition the frame: they sum to its total', () => {
  // Each mark closes the span since the previous one, so there is no gap for
  // time to hide in. Checked with real clock time, not fabricated numbers.
  const bench = new Benchmark(fakeEngine()).start();
  for (let f = 0; f < 20; f++) {
    frame(bench, ['a', 'b', 'c']);
  }
  const report = bench.stop();
  assert.equal(report.frames, 20);
  assert.deepEqual(report.cpu.map((row) => row.name), ['a', 'b', 'c', 'frame (cpu)'], 'in frame order');

  const phases = report.cpu.filter((row) => row.name !== 'frame (cpu)');
  const total = report.cpu.find((row) => row.name === 'frame (cpu)');
  const sum = phases.reduce((s, row) => s + row.mean, 0);
  assert.ok(Math.abs(sum - total.mean) < 1e-9, `phases ${sum} vs frame ${total.mean}`);
  const shares = phases.reduce((s, row) => s + row.share, 0);
  assert.ok(total.mean === 0 || Math.abs(shares - 1) < 1e-9, `shares sum to ${shares}`);
});

test('start clears the previous run', () => {
  const bench = new Benchmark(fakeEngine()).start();
  frame(bench, ['a']);
  bench.stop();
  bench.start();
  frame(bench, ['z']);
  const report = bench.stop();
  assert.equal(report.frames, 1);
  assert.deepEqual(report.cpu.map((row) => row.name), ['z', 'frame (cpu)']);
});

test('summarize takes the median and p95 from the sorted samples', () => {
  const samples = Array.from({ length: 100 }, (_, i) => 100 - i);   // 100..1, unsorted
  const s = summarize(samples);
  assert.equal(s.mean, 50.5);
  assert.equal(s.median, 51);
  assert.equal(s.p95, 96);
  assert.equal(s.max, 100);
  assert.deepEqual(summarize([]), { mean: 0, median: 0, p95: 0, max: 0 });
});

test('a device without timestamps reports gpu as null, and format says so', () => {
  const bench = new Benchmark(fakeEngine({ available: false })).start();
  frame(bench, ['a']);
  const report = bench.stop();
  assert.equal(report.gpu, null);
  assert.equal(report.wall, null, 'wall exists only when run() waited');
  assert.match(Benchmark.format(report), /no timestamp-query/);
});

test('a frame is labelled by the largest of cpu, gpu and display wait', () => {
  const cpuRow = (median) => [{ name: 'frame (cpu)', mean: median, median, p95: median, max: median, share: 1 }];
  const gpuRows = (mean) => [{ name: 'forward', mean, share: 1 }];
  const wall = (median) => ({ mean: median, median, p95: median, max: median });
  const label = (cpu, gpu, w) => boundBy({ cpu: cpuRow(cpu), gpu: gpu === null ? null : gpuRows(gpu), wall: wall(w) });

  assert.match(label(3, 0.5, 4), /^CPU-bound/);
  assert.match(label(0.5, 3, 4), /^GPU-bound/);
  // The case the first real run hit: a 60Hz display pacing a light frame.
  assert.match(label(0.9, 1.2, 15.8), /^display-paced/);
  // No timestamps: the GPU and the display cannot be told apart, and it says so.
  assert.match(label(1, null, 4), /^waiting on the GPU or the display/);
  assert.match(label(3, null, 4), /^CPU-bound/);
  assert.match(Benchmark.format({ frames: 1, cpu: cpuRow(1), gpu: null, gpuSamples: 0, wall: wall(4) }), /GPU or the display/);
});

await atest('starting twice, or starting and then running, still restores GPU timing', async () => {
  // A second start used to save the "on" the first had set, so stop() put
  // GPU timing back on for a renderer that had it off.
  const engine = fakeEngine();
  const bench = new Benchmark(engine);
  bench.start();
  bench.start();
  bench.stop();
  assert.equal(engine.renderer.gpuTiming.enabled, false, 'start twice');

  const runnable = runnableEngine();
  const other = new Benchmark(runnable);
  other.start();
  await other.run({}, {}, { frames: 3, warmup: 2 });
  assert.equal(runnable.renderer.gpuTiming.enabled, false, 'start then run');
});

await atest('the warmup is measured attached, then forgotten', async () => {
  // GPU timing's buffers are created on its first frames on, so it must be on
  // for the warmup -- and those frames must not be in the report.
  const engine = runnableEngine();
  let enabledDuringWarmup = null;
  engine.onFrame = (n) => { if (n === 0) enabledDuringWarmup = engine.renderer.gpuTiming.enabled; };
  const report = await new Benchmark(engine).run({}, {}, { frames: 4, warmup: 3 });
  assert.equal(enabledDuringWarmup, true);
  assert.equal(report.frames, 4, 'only the recorded frames');
});

/** A stand-in engine whose renderFrame drives the profiler the way Renderer.render does. */
function runnableEngine() {
  const engine = fakeEngine();
  let n = 0;
  engine.rhi = { device: { queue: { onSubmittedWorkDone: async () => {} } } };
  engine.renderFrame = () => {
    engine.onFrame?.(n++);
    const p = engine.renderer.profiler;
    p?.frameStart();
    p?.mark('work');
    p?.frameEnd();
  };
  return engine;
}

console.log(`\n${passed} checks passed\n`);
