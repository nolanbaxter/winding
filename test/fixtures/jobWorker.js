// Worker side of the job system test. Runs under node:worker_threads.
//
// Deliberately tiny: the whole worker is "receive the shared arrays, register
// the same handlers the main thread has, sit in workerLoop". If this file had
// logic of its own, the parallel path could differ from the serial one.

import { parentPort, threadId } from 'node:worker_threads';
import { workerLoop } from '../../src/core/jobs.js';

const JOB_DOUBLE = 99;
const JOB_TOUCH = 98;
const JOB_SLOW = 97;

/**
 * Who this thread is, in a form that survives an Int32Array whose zero means
 * "nothing touched this item". The main thread's `threadId` is 0, so every
 * thread is one more than its id and none of them collides with untouched.
 */
const WHO = threadId + 1;

const postReady = () => parentPort.postMessage({ type: 'ready' });
let reported = false;

parentPort.on('message', (message) => {
  if (message.type !== 'init') return;

  const control = new Int32Array(message.control);
  const input = new Int32Array(message.buffers.input);
  const output = new Int32Array(message.buffers.output);
  const touches = new Int32Array(message.buffers.touches);
  const who = new Int32Array(message.buffers.who);

  const handlers = new Map([
    [JOB_DOUBLE, (start, end) => {
      for (let i = start; i < end; i++) output[i] = input[i] * 2;
    }],
    // Atomic increment per item: if any item is processed twice, or by two
    // threads at once, the count says so.
    [JOB_TOUCH, (start, end) => {
      for (let i = start; i < end; i++) Atomics.add(touches, i, 1);
    }],
    // The same, plus a signature and real time per chunk. Both halves are
    // needed to prove parallelism rather than assume it: the signature says
    // WHICH thread ran an item, and the delay is what makes it possible for
    // more than one to -- a trivial handler lets the dispatching thread claim
    // every chunk before a worker has finished waking, which is legal and
    // would make the assertion flaky rather than wrong.
    [JOB_SLOW, (start, end) => {
      for (let i = start; i < end; i++) {
        Atomics.add(touches, i, 1);
        who[i] = WHO;
      }
      const until = Date.now() + 2;
      while (Date.now() < until) { /* burn a chunk's worth of time */ }
    }],
  ]);

  if (!reported) { reported = true; postReady(); }
  workerLoop(control, handlers, message.revision);
});
