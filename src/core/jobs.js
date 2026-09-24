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
// queue, no stealing, and no per-worker state. Chunks are sized so that their
// cost dominates the atomic -- too small and the cursor becomes the
// bottleneck, too large and the last chunk decides when everyone finishes.
//
// THE CURSOR CARRIES THE EPOCH, and that is not decoration. It was a plain
// counter reset per dispatch, claimed with one atomicAdd, and that made a
// claim anonymous: a worker descheduled between reading its dispatch's
// parameters and claiming could wake into the NEXT dispatch, take its chunks,
// run them with the PREVIOUS dispatch's arguments, and then credit the new
// dispatch's completion counter -- which returned believing work was done that
// its own handler never touched. Tagging the cursor makes a claim belong to a
// dispatch, so a stale thread's compare-exchange simply fails and it claims
// nothing. The cost is a CAS where there was an add.
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
// (epoch << 16) | next unclaimed CHUNK. Tagged so a claim cannot cross a
// dispatch boundary; see the header. Chunks rather than items because the
// index has to fit beside the tag, and because it can then never overshoot.
const CURSOR = 1;
const CURSOR_TAG = 0xffff0000 | 0;
const CURSOR_INDEX = 0xffff;
const COMPLETED = 2;   // items finished
const ITEM_COUNT = 3;
const CHUNK = 4;
const KIND = 5;        // which job to run
const ARG0 = 6;        // job-specific scalars
const ARG1 = 7;
const SHUTDOWN = 8;
const DATA = 9;        // bumped per setSharedData; which buffers are current
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

    /**
     * Which path each dispatch took.
     *
     * Not instrumentation for its own sake: whether a dispatch went parallel
     * is otherwise UNOBSERVABLE. The serial fallback is bit-identical by
     * construction -- that is the whole point of it -- so no assertion about
     * results can tell the two apart, and a suite that dispatched only
     * inline would pass every check it had while testing nothing it claimed
     * to. That is exactly what this one did.
     */
    this.stats = { parallel: 0, inline: 0 };

    // Resolved once every worker has checked in. Created here rather than
    // lazily because the messages can arrive before anyone asks.
    this._onReady = null;
    this._ready = this.parallel
      ? new Promise((resolve) => { this._onReady = resolve; })
      : Promise.resolve();

    if (this.parallel) {
      for (let i = 0; i < workerCount; i++) {
        const worker = createWorker();
        const onMessage = (event) => {
          const data = event?.data ?? event;
          if (data?.type !== 'ready') return;
          this.readyCount++;
          if (this.readyCount >= this.workerCount) this._onReady?.();
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
   *
   * `owner` is whatever those arrays belong to. It is published alongside them
   * so a job handler can find the CURRENT owner instead of closing over one,
   * which is what lets two scenes exist: the workers can only hold one set of
   * buffers at a time, and the handler has to agree with them about whose.
   */
  setSharedData(buffers, owner = null) {
    this._buffers = buffers;
    /** Whatever last published its buffers here. See setSharedData. */
    this.sharedOwner = owner;
    // A worker sits inside workerLoop and never returns to its event loop on
    // its own, so a second 'init' would queue unread while the worker went on
    // composing into the FIRST buffers it was given -- memory nobody reads
    // once a store has grown. The revision is how it learns to go and read it.
    const revision = Atomics.add(this.control, DATA, 1) + 1;
    for (const worker of this.workers) {
      worker.postMessage({ type: 'init', control: this.control.buffer, buffers, revision });
    }
  }

  /**
   * Resolve once every worker has loaded its module and installed its
   * handlers -- true if they did, false if dispatch will run inline anyway.
   *
   * NOT for the frame loop. The engine deliberately does not await this: a
   * worker that has not checked in yet simply means the next few dispatches
   * run on the calling thread, which is correct and invisible. What needs it
   * is anything that must know the parallel path was actually exercised,
   * because readiness arrives as a MESSAGE and a message cannot be delivered
   * to a thread that has not returned to its event loop. A caller that
   * constructs a JobSystem and dispatches in the same synchronous run will
   * never see a single worker, however many it spawned.
   *
   * Bounded by the same deadline a lost worker gets mid-dispatch, because it
   * is the same question -- a worker that has not answered in that long is
   * not coming -- and an unbounded wait here would hang a test runner rather
   * than fail it.
   */
  ready() {
    if (!this.parallel) return Promise.resolve(false);
    if (this.readyCount >= this.workerCount) return Promise.resolve(true);

    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve(false), SPIN_DEADLINE_MS);
      // Node keeps the process alive for a pending timer; a browser does not
      // have this method at all.
      timer?.unref?.();
      this._ready.then(() => {
        clearTimeout(timer);
        resolve(true);
      });
    });
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
      this.stats.inline++;
      handler(0, itemCount, arg0, arg1);
      return itemCount;
    }

    this.stats.parallel++;
    const control = this.control;
    let chunk = chunkSize > 0
      ? chunkSize
      : Math.max(32, Math.ceil(itemCount / (this.workerCount + 1) / 4));

    // The chunk INDEX shares a word with the epoch tag, so there is a ceiling
    // on how many chunks a dispatch can have. Raised rather than refused: a
    // chunk size is a performance hint, and doing more work per claim cannot
    // make a result wrong, where rejecting the caller's number would.
    if (Math.ceil(itemCount / chunk) > CURSOR_INDEX) {
      chunk = Math.ceil(itemCount / CURSOR_INDEX);
    }

    const epoch = Atomics.load(control, EPOCH) + 1;
    Atomics.store(control, CURSOR, epochTag(epoch));
    Atomics.store(control, COMPLETED, 0);
    Atomics.store(control, ITEM_COUNT, itemCount);
    Atomics.store(control, CHUNK, chunk);
    Atomics.store(control, ARG0, arg0);
    Atomics.store(control, ARG1, arg1);
    Atomics.store(control, KIND, kind);

    // Publishing the epoch last is what makes the rest of the block visible:
    // a worker only reads the parameters after observing a new epoch, and
    // Atomics give that pairing the ordering guarantee it needs.
    Atomics.store(control, EPOCH, epoch);
    Atomics.notify(control, EPOCH);

    // This thread is a worker too.
    runChunks(control, handler, arg0, arg1, epoch);

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

/** The high half of the cursor word, identifying which dispatch owns it. */
export function epochTag(epoch) {
  return (epoch & CURSOR_INDEX) << 16;
}

/**
 * The loop every thread runs, including the dispatching one.
 *
 * Claiming a chunk and finishing it are two separate counters on purpose. The
 * cursor says what has been handed out; COMPLETED says what is actually done.
 * Waiting on the cursor instead would return while the last chunk was still
 * being written.
 *
 * `epoch` is which dispatch the caller believes it is serving. A claim is a
 * compare-exchange against the cursor's tag, so a thread carrying a stale
 * epoch takes nothing rather than stealing the current dispatch's work and
 * running it with the wrong arguments. Defaults to 0 for direct callers that
 * drive the control block themselves.
 *
 * The tag is the epoch's low 16 bits, so a thread would have to be stranded
 * across exactly 65536 dispatches to alias a live one. At a handful of
 * dispatches per frame that is minutes of being descheduled, by which point
 * the spin deadline has already declared the workers lost.
 */
export function runChunks(control, handler, arg0, arg1, epoch = 0) {
  const itemCount = Atomics.load(control, ITEM_COUNT);
  const chunk = Atomics.load(control, CHUNK);
  const tag = epochTag(epoch);

  for (;;) {
    const cursor = Atomics.load(control, CURSOR);
    // A different dispatch owns the cursor now. Nothing here belongs to this
    // thread, and the loop it returns to will observe the new epoch.
    if ((cursor & CURSOR_TAG) !== tag) return;

    const index = cursor & CURSOR_INDEX;
    const start = index * chunk;
    if (start >= itemCount) return;

    // Lost the race to another thread, or to a new dispatch. Re-read and
    // decide again rather than assuming either.
    if (Atomics.compareExchange(control, CURSOR, cursor, tag | (index + 1)) !== cursor) continue;

    const end = Math.min(start + chunk, itemCount);
    handler(start, end, arg0, arg1);
    Atomics.add(control, COMPLETED, end - start);
  }
}

/**
 * The loop a worker thread sits in: wait for a new epoch, run, wait again.
 * Exported so the worker entry module is a handful of lines.
 *
 * `revision` is the one its 'init' message carried. The loop returns as soon
 * as the buffers it holds are no longer current, so the worker's event loop
 * can deliver the newer 'init' that is already queued. Claiming nothing in the
 * meantime is safe: the dispatching thread runs every chunk nobody takes.
 */
export function workerLoop(control, handlers, revision = Atomics.load(control, DATA)) {
  let seen = Atomics.load(control, EPOCH);
  for (;;) {
    if (Atomics.load(control, DATA) !== revision) return;
    Atomics.wait(control, EPOCH, seen);
    seen = Atomics.load(control, EPOCH);

    if (Atomics.load(control, SHUTDOWN) === 1) return;
    if (Atomics.load(control, DATA) !== revision) return;

    const kind = Atomics.load(control, KIND);
    const handler = handlers.get(kind);
    if (!handler) continue;

    // `seen` is the dispatch these arguments came from. If a new one starts
    // before this thread claims anything, the tag check inside runChunks is
    // what keeps it from taking work it would run with the wrong arguments.
    runChunks(
      control, handler,
      Atomics.load(control, ARG0), Atomics.load(control, ARG1),
      seen,
    );
  }
}

export const CONTROL = { EPOCH, CURSOR, COMPLETED, ITEM_COUNT, CHUNK, KIND, ARG0, ARG1, SHUTDOWN, DATA, CONTROL_SLOTS };

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
