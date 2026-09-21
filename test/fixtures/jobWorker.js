// Worker side of the job system test. Runs under node:worker_threads.
//
// Deliberately tiny: the whole worker is "receive the shared arrays, register
// the same handlers the main thread has, sit in workerLoop". If this file had
// logic of its own, the parallel path could differ from the serial one.

import { parentPort } from 'node:worker_threads';
import { workerLoop } from '../../src/core/jobs.js';

const JOB_DOUBLE = 99;
const JOB_TOUCH = 98;

const postReady = () => parentPort.postMessage({ type: 'ready' });

parentPort.on('message', (message) => {
  if (message.type !== 'init') return;

  const control = new Int32Array(message.control);
  const input = new Int32Array(message.buffers.input);
  const output = new Int32Array(message.buffers.output);
  const touches = new Int32Array(message.buffers.touches);

  const handlers = new Map([
    [JOB_DOUBLE, (start, end) => {
      for (let i = start; i < end; i++) output[i] = input[i] * 2;
    }],
    // Atomic increment per item: if any item is processed twice, or by two
    // threads at once, the count says so.
    [JOB_TOUCH, (start, end) => {
      for (let i = start; i < end; i++) Atomics.add(touches, i, 1);
    }],
  ]);

  postReady();
  workerLoop(control, handlers);
});
