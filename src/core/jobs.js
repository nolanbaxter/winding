// Job system: a parallel-for over shared memory.
//
// NOT a work-stealing scheduler. Chase-Lev deques and per-worker queues earn
// their complexity when tasks spawn subtasks of wildly varying cost and a
// worker can genuinely run dry while another is buried. A flat loop over N
// array elements is not that shape, and the simpler structure below
// load-balances it just as well:
//
//   a single atomic cursor, and every thread repeatedly claims the next chunk
//
// A thread that gets unlucky work simply claims fewer chunks. There is no
// queue, no stealing, no per-worker state, and the only synchronisation is one
// atomicAdd per chunk. Chunks are sized so that cost dominates the atomic --
// too small and the cursor becomes the bottleneck, too large and the last
// chunk decides when everyone finishes.
//
// THE MAIN THREAD PARTICIPATES. It has to: a browser's main thread is not
// allowed to block in Atomics.wait, so it cannot simply hand work off and
// sleep. It claims chunks alongside the workers and then spins on the
// completion counter, which is cheap precisely because it has been doing a
// share of the work rather than waiting for it.
//
// Without shared memory the whole thing degrades to a plain loop on the calling
// thread, producing identical results.

import { DEBUG, assert } from './assert.js';
import { sharedInt32Array, sharedMemoryAvailable } from './shared.js';

// Control block layout. Int32Array so Atomics can operate on it.
const EPOCH = 0;       // bumped per dispatch; what workers wait on
const CURSOR = 1;      // next unclaimed item
const COMPLETED = 2;   // items finished
const ITEM_COUNT = 3;
const CHUNK = 4;
const KIND = 5;        // which job to run
const ARG0 = 6;        // job-specific scalars
const ARG1 = 7;
const SHUTDOWN = 8;
const CONTROL_SLOTS = 16;

/**
 * How long the dispatching thread spins before declaring the workers lost.
 *
 * This is a HANG bound, not a performance bound, and the distinction matters:
 * a legitimate chunk is sub-millisecond, but a backgrounded tab, a breakpoint
 * or a busy machine can deschedule this thread for far longer than the work
 * takes. Set it near the real cost and a scheduling hiccup looks like a broken
 * worker; set it here and only a genuine failure trips it.
 */
const SPIN_DEADLINE_MS = 2000;

const nowMs = () => (globalThis.performance?.now?.() ?? Date.now());

/**
 * Below this many items, dispatch runs inline instead of waking the workers.
 *
 * MEASURED, not guessed. A scene of 810 transforms over depth levels of 6, 36,
 * 216 and 648 is around 50 microseconds of actual work. Waking
 * seven threads four times a frame, and busy-spinning until the slowest returns,
 * cost enough to drop the frame rate from 60 to 4.
 *
 * Thread wake latency is tens of microseconds and does not shrink with the
 * work, so there is a floor below which parallelism can only lose. This is that
 * floor. Raise it if the per-item work is trivial, lower it if each item is
 * expensive -- it is a property of the JOB, not of the machine.
 */
const DEFAULT_PARALLEL_THRESHOLD = 4096;

/** Jobs are identified by a small integer so the control block stays an Int32Array. */
export const JOB_NONE = 0;
export const JOB_COMPOSE_TRANSFORMS = 1;

export class JobSystem {
  /**
   * @param options.workerCount   how many workers to spawn; 0 runs inline
   * @param options.createWorker  () => Worker. Injected so the same system runs
   *                              under node:worker_threads in a test and under
   *                              a browser Worker in the engine.
   */
  constructor({
    workerCount = defaultWorkerCount(),
    createWorker = null,
    parallelThreshold = DEFAULT_PARALLEL_THRESHOLD,
  } = {}) {
    this.parallelThreshold = parallelThreshold;
    this.control = sharedInt32Array(CONTROL_SLOTS);
    this.workers = [];
    this.parallel = sharedMemoryAvailable && workerCount > 0 && createWorker !== null;
    this.workerCount = this.parallel ? workerCount : 0;

    /** Registered job bodies, by kind. Run on every thread including this one. */
    this._handlers = new Map();
    this._buffers = null;

    // Workers report in once their module has loaded and their handlers are
    // installed. Until every one has, dispatch runs inline -- a worker that is
    // not listening would otherwise claim nothing, and the dispatcher would
    // wait forever for work nobody is doing.
    this.readyCount = 0;
    if (this.parallel) {
      for (let i = 0; i < workerCount; i++) {
        const worker = createWorker();
        const onMessage = (event) => {
          const data = event?.data ?? event;
          if (data?.type === 'ready') this.readyCount++;
        };
        if (worker.addEventListener) worker.addEventListener('message', onMessage);
        else worker.on?.('message', onMessage);
        this.workers.push(worker);
      }
    }
  }

  /**
   * Hand every worker the shared arrays they operate on. Sent once; nothing is
   * copied, because these are views onto SharedArrayBuffers.
   */
  setSharedData(buffers) {
    this._buffers = buffers;
    for (const worker of this.workers) {
      worker.postMessage({ type: 'init', control: this.control.buffer, buffers });
    }
  }

  /** Register a job body. Must be registered identically on every thread. */
  register(kind, handler) {
    this._handlers.set(kind, handler);
  }

  /**
   * Run `kind` over [0, itemCount) and return when every item is done.
   *
   * Synchronous by design. The transform hierarchy has to be finished before
   * the frame can be encoded, so an async API would only move the wait
   * somewhere less obvious.
   */
  dispatch(kind, itemCount, { chunkSize = 0, arg0 = 0, arg1 = 0 } = {}) {
    if (itemCount <= 0) return 0;

    const handler = this._handlers.get(kind);
    if (DEBUG) assert(handler, `job kind ${kind} has no handler on this thread`);

    // Inline whenever the workers cannot help, or would not be worth waking:
    // no shared memory, not all of them checked in, or too few items for the
    // wake cost to pay for itself. The control block is left untouched, so a
    // worker that wakes later sees no new epoch and stays asleep.
    if (!this.parallel
      || this.readyCount < this.workerCount
      || itemCount < this.parallelThreshold) {
      handler(0, itemCount, arg0, arg1);
      return itemCount;
    }

    const control = this.control;
    const chunk = chunkSize > 0
      ? chunkSize
      : Math.max(32, Math.ceil(itemCount / (this.workerCount + 1) / 4));

    Atomics.store(control, CURSOR, 0);
    Atomics.store(control, COMPLETED, 0);
    Atomics.store(control, ITEM_COUNT, itemCount);
    Atomics.store(control, CHUNK, chunk);
    Atomics.store(control, ARG0, arg0);
    Atomics.store(control, ARG1, arg1);
    Atomics.store(control, KIND, kind);

    // Publishing the epoch last is what makes the rest of the block visible:
    // a worker only reads the parameters after observing a new epoch, and
    // Atomics give that pairing the ordering guarantee it needs.
    Atomics.add(control, EPOCH, 1);
    Atomics.notify(control, EPOCH);

    // This thread is a worker too.
    runChunks(control, handler, arg0, arg1);

    // Spin rather than wait: blocking is forbidden on a browser's main thread,
    // and by this point the remaining work is at most one chunk per worker.
    //
    // BOUNDED, though. An unbounded spin here turns any worker-side failure --
    // a module that failed to load, a thread that died mid-chunk -- into a
    // frozen page with no diagnostic. A deadline turns the same failure into a
    // sentence someone can act on.
    const deadline = nowMs() + SPIN_DEADLINE_MS;
    while (Atomics.load(control, COMPLETED) < itemCount) {
      if (nowMs() > deadline) {
        this.parallel = false;
        throw new Error(
          `JobSystem: workers stopped responding after ${SPIN_DEADLINE_MS}ms ` +
          `(${Atomics.load(control, COMPLETED)}/${itemCount} items done). ` +
          'Falling back to single-threaded for the rest of the session.',
        );
      }
    }

    Atomics.store(control, KIND, JOB_NONE);
    return itemCount;
  }

  destroy() {
    Atomics.store(this.control, SHUTDOWN, 1);
    Atomics.add(this.control, EPOCH, 1);
    Atomics.notify(this.control, EPOCH);
    for (const worker of this.workers) worker.terminate?.();
    this.workers.length = 0;
  }
}

/**
 * The loop every thread runs, including the dispatching one.
 *
 * Claiming a chunk and finishing it are two separate counters on purpose. The
 * cursor says what has been handed out; COMPLETED says what is actually done.
 * Waiting on the cursor instead would return while the last chunk was still
 * being written.
 */
export function runChunks(control, handler, arg0, arg1) {
  const itemCount = Atomics.load(control, ITEM_COUNT);
  const chunk = Atomics.load(control, CHUNK);

  for (;;) {
    const start = Atomics.add(control, CURSOR, chunk);
    if (start >= itemCount) return;

    const end = Math.min(start + chunk, itemCount);
    handler(start, end, arg0, arg1);
    Atomics.add(control, COMPLETED, end - start);
  }
}

/**
 * The loop a worker thread sits in: wait for a new epoch, run, wait again.
 * Exported so the worker entry module is a handful of lines.
 */
export function workerLoop(control, handlers) {
  let seen = Atomics.load(control, EPOCH);
  for (;;) {
    Atomics.wait(control, EPOCH, seen);
    seen = Atomics.load(control, EPOCH);

    if (Atomics.load(control, SHUTDOWN) === 1) return;

    const kind = Atomics.load(control, KIND);
    const handler = handlers.get(kind);
    if (!handler) continue;

    runChunks(control, handler, Atomics.load(control, ARG0), Atomics.load(control, ARG1));
  }
}

export const CONTROL = { EPOCH, CURSOR, COMPLETED, ITEM_COUNT, CHUNK, KIND, ARG0, ARG1, SHUTDOWN, CONTROL_SLOTS };

/**
 * One worker per core, minus the one this thread is already using.
 *
 * Oversubscribing is worse than undersubscribing here: every thread spins at
 * the end of a dispatch, so more threads than cores means threads burning a
 * core doing nothing while the thread they are waiting for cannot run.
 */
export function defaultWorkerCount() {
  const cores = globalThis.navigator?.hardwareConcurrency ?? 4;
  return Math.max(0, Math.min(cores - 1, 7));
}
