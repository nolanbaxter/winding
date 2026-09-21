// Browser worker entry for the job system.
//
// Deliberately identical in shape to the Node test fixture: receive the shared
// buffers, register the same handlers the main thread registers, sit in
// workerLoop. Any logic that lived here would be logic the serial path does not
// run, and the two would eventually disagree.

import { workerLoop, JOB_COMPOSE_TRANSFORMS } from './jobs.js';
import { composeRange, columnsFromBuffers } from '../scene/transformJob.js';

const postReady = () => self.postMessage({ type: 'ready' });

self.addEventListener('message', (event) => {
  const message = event.data;
  if (message.type !== 'init') return;

  const control = new Int32Array(message.control);
  const columns = columnsFromBuffers(message.buffers);

  postReady();
  workerLoop(control, new Map([
    [JOB_COMPOSE_TRANSFORMS, (start, end, base) => composeRange(columns, base, start, end)],
  ]));
});
