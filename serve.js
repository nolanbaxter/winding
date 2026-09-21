// Minimal static server. `node serve.js` then open http://localhost:8080/test/gpu.html
//
// Sends COOP/COEP so the page is cross-origin isolated and SharedArrayBuffer
// exists. Everything is same-origin here, so nothing else has to change.
//
// Needed because ES modules and fetch() are blocked over file://. Node's own
// http + fs cover it, so there is no dependency here and never needs to be one.

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';

const ROOT = import.meta.dirname;
const PORT = Number(process.argv[2] ?? 8080);

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
};

createServer(async (req, res) => {
  try {
    let path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (path.endsWith('/')) path += 'index.html';

    // Reject traversal before touching the filesystem: normalize resolves the
    // '..' segments, then we confirm the result is still inside ROOT.
    const full = normalize(join(ROOT, path));
    if (!full.startsWith(ROOT)) {
      res.writeHead(403).end('forbidden');
      return;
    }

    const body = await readFile(full);
    res.writeHead(200, {
      'content-type': TYPES[extname(full)] ?? 'application/octet-stream',
      'cache-control': 'no-store',
      // SharedArrayBuffer is only handed out to a cross-origin isolated page,
      // which needs BOTH of these. Without them crossOriginIsolated is false
      // and the constructor is simply not there -- no error, no hint.
      //
      // Static hosting that cannot set headers therefore cannot run the worker
      // path. The engine still works there: jobs fall back to running inline.
      'cross-origin-opener-policy': 'same-origin',
      'cross-origin-embedder-policy': 'require-corp',
    }).end(body);
  } catch {
    res.writeHead(404).end('not found');
  }
}).listen(PORT, () => {
  console.log(`http://localhost:${PORT}/test/gpu.html   (GPU smoke test)`);
});
