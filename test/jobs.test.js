// Job system self-check. Run: node test/jobs.test.js
//
// Real threads, not a simulation. node:worker_threads gives genuine parallelism
// over a SharedArrayBuffer, which is the only way to catch the failure this
// design can actually have: an item processed twice, or none.

import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { JobSystem, runChunks, CONTROL } from '../src/core/jobs.js';
import { sharedInt32Array, sharedMemoryAvailable } from '../src/core/shared.js';

let passed = 0;
async function test(name, fn) {
  await fn();
  passed++;
  console.log(`  ok  ${name}`);
}

const JOB_DOUBLE = 99;
const JOB_TOUCH = 98;
const WORKER_URL = new URL('./fixtures/jobWorker.js', import.meta.url);

function makeSystem(workerCount, count) {
  const input = sharedInt32Array(count);
  const output = sharedInt32Array(count);
  const touches = sharedInt32Array(count);
  for (let i = 0; i < count; i++) input[i] = i;

  const jobs = new JobSystem({
    parallelThreshold: 1,
    workerCount,
    createWorker: workerCount > 0 ? () => new Worker(WORKER_URL) : null,
  });

  // The dispatching thread runs the same handlers as the workers.
  jobs.register(JOB_DOUBLE, (start, end) => {
    for (let i = start; i < end; i++) output[i] = input[i] * 2;
  });
  jobs.register(JOB_TOUCH, (start, end) => {
    for (let i = start; i < end; i++) Atomics.add(touches, i, 1);
  });

  jobs.setSharedData({
    input: input.buffer, output: output.buffer, touches: touches.buffer,
  });
  return { jobs, input, output, touches };
}

console.log('\nenvironment');

await test('shared memory is available under node', () => {
  assert.equal(sharedMemoryAvailable, true);
});

console.log('\nparallel execution');

await test('every item is processed exactly once', async () => {
  // THE property. Too few and there is a hole; too many and two threads wrote
  // the same slot, which for a transform buffer means a torn matrix.
  const count = 10_000;
  const { jobs, touches } = makeSystem(3, count);

  jobs.dispatch(JOB_TOUCH, count);
  for (let i = 0; i < count; i++) {
    assert.equal(touches[i], 1, `item ${i} was touched ${touches[i]} times`);
  }
  jobs.destroy();
});

await test('results match the serial computation', async () => {
  const count = 5000;
  const { jobs, input, output } = makeSystem(3, count);

  jobs.dispatch(JOB_DOUBLE, count);
  for (let i = 0; i < count; i++) assert.equal(output[i], input[i] * 2, `at ${i}`);
  jobs.destroy();
});

await test('repeated dispatches stay correct', async () => {
  // The counters are reset per dispatch. Forget one and the second dispatch
  // returns immediately with most of the work undone.
  const count = 2000;
  const { jobs, touches } = makeSystem(3, count);

  for (let round = 0; round < 5; round++) jobs.dispatch(JOB_TOUCH, count);
  for (let i = 0; i < count; i++) assert.equal(touches[i], 5, `at ${i}`);
  jobs.destroy();
});

await test('a tiny job is still correct', async () => {
  // Fewer items than threads: most threads claim nothing at all.
  const count = 3;
  const { jobs, touches } = makeSystem(3, count);
  jobs.dispatch(JOB_TOUCH, count);
  assert.deepEqual([...touches], [1, 1, 1]);
  jobs.destroy();
});

await test('an empty job does nothing and returns', async () => {
  const { jobs, touches } = makeSystem(2, 8);
  assert.equal(jobs.dispatch(JOB_TOUCH, 0), 0);
  assert.equal(touches.reduce((a, b) => a + b, 0), 0);
  jobs.destroy();
});

console.log('\nserial fallback');

await test('no workers means the calling thread does everything', async () => {
  // The fallback is not a different algorithm -- identical results, one thread.
  const count = 1000;
  const { jobs, input, output } = makeSystem(0, count);
  assert.equal(jobs.parallel, false);

  jobs.dispatch(JOB_DOUBLE, count);
  for (let i = 0; i < count; i++) assert.equal(output[i], input[i] * 2, `at ${i}`);
  jobs.destroy();
});

console.log('\nchunk claiming');

await test('claiming stops at the item count, never past it', async () => {
  // runChunks in isolation: the cursor overshoots by design, and the guard is
  // what keeps a handler from being called with a range past the end.
  const control = sharedInt32Array(CONTROL.CONTROL_SLOTS);
  Atomics.store(control, CONTROL.ITEM_COUNT, 10);
  Atomics.store(control, CONTROL.CHUNK, 4);

  const ranges = [];
  runChunks(control, (start, end) => ranges.push([start, end]), 0, 0);

  assert.deepEqual(ranges, [[0, 4], [4, 8], [8, 10]], 'last chunk is clamped');
  assert.equal(Atomics.load(control, CONTROL.COMPLETED), 10);
});

await test('completion counts items, not chunks', async () => {
  // Waiting on the cursor instead would return while the final chunk was still
  // being written.
  const control = sharedInt32Array(CONTROL.CONTROL_SLOTS);
  Atomics.store(control, CONTROL.ITEM_COUNT, 7);
  Atomics.store(control, CONTROL.CHUNK, 3);
  runChunks(control, () => {}, 0, 0);

  assert.equal(Atomics.load(control, CONTROL.COMPLETED), 7);
  assert.ok(Atomics.load(control, CONTROL.CURSOR) >= 7, 'cursor may overshoot');
});


// ------------------------------------------- parallel transform composition

console.log('\nparallel transforms');

const { TransformStore } = await import('../src/scene/transform.js');
const { HandleAllocator } = await import('../src/core/handle.js');
const { composeRange } = await import('../src/scene/transformJob.js');
const { JOB_COMPOSE_TRANSFORMS } = await import('../src/core/jobs.js');

/** A wide, deep hierarchy: 4 levels, branching 6. */
function buildHierarchy(store, ids) {
  const roots = [];
  for (let r = 0; r < 6; r++) {
    const root = ids.alloc();
    store.add(root, { position: [r * 3, 0, 0] });
    roots.push(root);

    for (let a = 0; a < 6; a++) {
      const child = ids.alloc();
      store.add(child, { position: [0, 1, 0], parent: root });
      for (let b = 0; b < 6; b++) {
        const grand = ids.alloc();
        store.add(grand, { position: [0, 0, 1], parent: child });
        for (let c = 0; c < 3; c++) {
          const leaf = ids.alloc();
          store.add(leaf, { position: [0.5, 0, 0], parent: grand });
        }
      }
    }
  }
  return roots;
}

await test('depth levels are recorded and cover every node', () => {
  const ids = new HandleAllocator(2048);
  const store = new TransformStore(2048);
  buildHierarchy(store, ids);
  store.update();

  assert.equal(store.levelCount, 4, 'root, child, grandchild, leaf');
  assert.equal(store.levelStart[0], 0);
  assert.equal(store.levelStart[store.levelCount], store.orderCount, 'levels tile the order');
});

await test('parallel composition matches serial, matrix for matrix', () => {
  // THE test. Depth ordering is what makes a level safe to split; if that
  // reasoning is wrong, a child composes against a stale parent and the two
  // paths diverge.
  const ids = new HandleAllocator(2048);
  const serialIds = new HandleAllocator(2048);

  const parallelStore = new TransformStore(2048);
  const serialStore = new TransformStore(2048);
  buildHierarchy(parallelStore, ids);
  buildHierarchy(serialStore, serialIds);

  serialStore.update();

  const jobs = new JobSystem({ parallelThreshold: 1,
    workerCount: 3,
    createWorker: () => new Worker(new URL('./fixtures/transformWorker.js', import.meta.url)),
  });
  jobs.register(JOB_COMPOSE_TRANSFORMS,
    (start, end, base) => composeRange(parallelStore, base, start, end));
  jobs.setSharedData(parallelStore.sharedBuffers());

  parallelStore.updateParallel(jobs);

  assert.equal(parallelStore.orderCount, serialStore.orderCount);
  for (let i = 0; i < parallelStore.world.length; i++) {
    assert.equal(parallelStore.world[i], serialStore.world[i], `world[${i}] diverged`);
  }
  jobs.destroy();
});

console.log(`\n${passed} checks passed\n`);
