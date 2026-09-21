// Worker side of the parallel transform test.
import { parentPort } from 'node:worker_threads';
import { workerLoop, JOB_COMPOSE_TRANSFORMS } from '../../src/core/jobs.js';
import { composeRange, columnsFromBuffers } from '../../src/scene/transformJob.js';

const postReady = () => parentPort.postMessage({ type: 'ready' });

parentPort.on('message', (message) => {
  if (message.type !== 'init') return;
  const control = new Int32Array(message.control);
  const columns = columnsFromBuffers(message.buffers);

  postReady();
  workerLoop(control, new Map([
    [JOB_COMPOSE_TRANSFORMS, (start, end, base) => composeRange(columns, base, start, end)],
  ]));
});
