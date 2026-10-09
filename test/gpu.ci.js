// The GPU suite, headless: node test/gpu.ci.js
//
// Serves the repo, opens test/gpu.html in Chrome with no window, waits for the
// page's own verdict (globalThis.__gpuTest, set by finish() in gpu.test.js) and
// exits 0 only if every check passed. CI runs it on a runner with no GPU:
// Chrome needs Mesa's Vulkan driver installed to start under these flags, and
// WebGPU then comes up on SwiftShader, Chrome's own software device. Slow, but
// it compiles and runs every shader, which is the point of the suite.
//
// No Puppeteer: Chrome is driven over its DevTools protocol with Node's own
// WebSocket (Node 22+), so nothing is installed and nothing is downloaded.
// Chrome comes from $CHROME, or the usual install paths.

import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const PORT = 8137;
const TIMEOUT = 15 * 60_000;

const CHROMES = [
  process.env.CHROME,
  '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
].filter(Boolean);

const chromePath = CHROMES.find((path) => existsSync(path));
if (!chromePath) throw new Error(`gpu.ci: no Chrome found; set CHROME. Looked in:\n  ${CHROMES.join('\n  ')}`);
if (typeof WebSocket === 'undefined') throw new Error('gpu.ci: needs Node 22 or later, for its WebSocket');

const server = spawn(process.execPath, ['serve.js', String(PORT)], { cwd: new URL('..', import.meta.url), stdio: 'ignore' });
const profile = mkdtempSync(join(tmpdir(), 'winding-gpu-'));
const chrome = spawn(chromePath, [
  '--headless=new',
  '--remote-debugging-port=0',
  `--user-data-dir=${profile}`,
  '--no-first-run',
  '--no-default-browser-check',
  // WebGPU on a machine Chrome would otherwise refuse it on: a software
  // adapter, no display. The same flags three.js runs its WebGPU tests under.
  // Vulkan is how Linux gets there; elsewhere it would turn off the real path.
  '--enable-unsafe-webgpu',
  '--ignore-gpu-blocklist',
  ...(process.platform === 'linux'
    ? ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-vulkan-surface', '--enable-unsafe-swiftshader'] : []),
  // A background tab is throttled, and this one is never in front.
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  // So the leak soak can force a collection and see what was let go of.
  '--js-flags=--expose-gc',
  '--window-size=800,600',
  'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
// Kept, and shown only if Chrome never gets going: otherwise it is noise.
let chromeLog = '';
chrome.stderr.on('data', (chunk) => { chromeLog = (chromeLog + chunk).slice(-4000); });

let exitCode = 1;
try {
  exitCode = await runSuite();
} catch (error) {
  console.error(`gpu.ci: ${error.message}`);
} finally {
  chrome.kill();
  server.kill();
  // Chrome holds the profile for a moment after it is told to go.
  setTimeout(() => {
    try { rmSync(profile, { recursive: true, force: true }); } catch {}
    process.exit(exitCode);
  }, 500);
}

async function runSuite() {
  const port = await until(() => {
    const file = join(profile, 'DevToolsActivePort');
    return existsSync(file) && Number(readFileSync(file, 'utf8').split('\n')[0]);
  }, 30_000, () => `Chrome never opened its DevTools port. It said:
${chromeLog}`);

  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const page = targets.find((t) => t.type === 'page');
  const cdp = await connect(page.webSocketDebuggerUrl);

  cdp.on('Runtime.consoleAPICalled', ({ type, args }) => {
    if (type === 'error') console.error(args.map((a) => a.value ?? a.description).join(' '));
  });
  cdp.on('Runtime.exceptionThrown', ({ exceptionDetails: e }) => {
    console.error(`page error: ${e.exception?.description ?? e.text}`);
  });
  await cdp.send('Runtime.enable');
  // On a runner the software adapter can take a moment to appear after Chrome
  // starts, and a page that asks too early gets none. Wait for it on a page of
  // the same origin, so the suite starts on a GPU that is there.
  await cdp.send('Page.navigate', { url: `http://localhost:${PORT}/serve.js` });
  const adapter = await until(() => cdp.eval(`navigator.gpu?.requestAdapter().then((a) => a
    && ([a.info.vendor, a.info.architecture, a.info.device, a.info.description].filter(Boolean).join(' ') || 'an unnamed adapter'))`),
  60_000, () => `no WebGPU adapter after a minute. Chrome said:
${chromeLog}`);
  console.log(`adapter: ${adapter}`);

  // GPU_STEPS and GPU_SHARD: the checks this runner makes (see gpu.test.js).
  // GPU_MAP=1: record which of src/ each check runs, into test/gpu-map.json.
  const query = new URLSearchParams();
  if (process.env.GPU_STEPS) query.set('steps', process.env.GPU_STEPS);
  if (process.env.GPU_SHARD) query.set('shard', process.env.GPU_SHARD);
  const map = process.env.GPU_MAP === '1' || process.argv.includes('--map') ? await recordCoverage(cdp) : null;

  const started = Date.now();
  await cdp.send('Page.navigate', { url: `http://localhost:${PORT}/test/gpu.html${query.size ? `?${query}` : ''}` });

  const summary = await until(async () => {
    const title = await cdp.eval('document.title');
    if (title.startsWith('FAIL harness')) throw new Error('the page could not run the suite; see the errors above');
    return /^(PASS|FAIL) /.test(title) && cdp.eval('globalThis.__gpuTest');
  }, TIMEOUT, `the suite did not finish in ${TIMEOUT / 60_000} minutes`);

  for (const r of summary.results) {
    // The time goes back into test/gpu-map.json (.github/ci-times.js), so the
    // runners are balanced by what each check takes here, not on a real GPU.
    console.log(`${r.ok ? 'ok  ' : 'FAIL'}  ${r.name}${r.ms !== undefined ? `  [${r.id} ${r.ms} ms]` : ''}`);
    if (r.detail) console.log(`      ${r.detail.replaceAll('\n', '\n      ')}`);
  }
  const seconds = ((Date.now() - started) / 1000).toFixed(0);
  console.log(`\n${summary.passed} passed, ${summary.failed} failed, in ${seconds} s`);
  if (map !== null) {
    if (summary.failed > 0) throw new Error('not writing the map from a run that failed');
    for (const r of summary.results) if (r.id && map.steps[r.id]) map.steps[r.id].ms = r.ms;
    writeFileSync(new URL('./gpu-map.json', import.meta.url), `${JSON.stringify(map)}\n`);
    const fns = Object.values(map.files).flatMap((f) => Object.values(f));
    console.log(`wrote test/gpu-map.json: ${Object.keys(map.steps).length} checks, ${fns.length} functions run, `
      + `${fns.filter((ids) => ids.includes('*')).length} of them by the setup every check needs`);
  }
  return summary.failed === 0 && summary.passed > 0 ? 0 : 1;
}

/**
 * Coverage, check by check, function by function: for every function in src/
 * that ran, its lines and the checks that ran it. What runs before the first
 * BASE checks end, and between checks, is the setup every check draws on, and
 * is recorded as '*'. The commit it was taken at goes with it, so the planner
 * can carry its line numbers to later code. Returns the map, filled in as the
 * suite runs.
 */
async function recordCoverage(cdp) {
  const commit = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const dirty = execFileSync('git', ['status', '--porcelain', '--', 'src'], { encoding: 'utf8' }).trim();
  if (dirty) throw new Error(`commit src/ first: the map's line numbers are the commit's\n${dirty}`);
  const map = { note: 'Made by GPU_MAP=1 node test/gpu.ci.js; read by .github/affected.js.', commit, steps: {}, files: {} };
  const lines = new Map();   // path -> offsets where each line starts
  const lineOf = (path, offset) => {
    let starts = lines.get(path);
    if (!starts) {
      starts = [0];
      const text = readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
      for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) starts.push(i + 1);
      lines.set(path, starts);
    }
    let lo = 0, hi = starts.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (starts[mid] <= offset) lo = mid; else hi = mid - 1; }
    return lo + 1;
  };
  const record = async (who) => {
    const { result } = await cdp.send('Profiler.takePreciseCoverage');
    for (const script of result) {
      const path = new URL(script.url || 'about:blank').pathname.replace(/^\//, '');
      if (!path.startsWith('src/')) continue;
      for (const f of script.functions) {
        const range = f.ranges[0];
        if (!(range?.count > 0)) continue;
        const key = `${lineOf(path, range.startOffset)}-${lineOf(path, range.endOffset)}`;
        const byFunction = (map.files[path] ??= {});
        const ids = (byFunction[key] ??= []);
        if (!ids.includes(who)) ids.push(who);
      }
    }
  };
  const BASE_STEPS = 5;
  let started = 0;
  await cdp.send('Profiler.enable');
  await cdp.send('Profiler.startPreciseCoverage', { callCount: true, detailed: false });
  await cdp.send('Runtime.addBinding', { name: '__coverageMark' });
  cdp.on('Runtime.bindingCalled', async ({ name, payload }) => {
    if (name !== '__coverageMark') return;
    const { phase, id, name: check } = JSON.parse(payload);
    if (phase === 'start') {
      started++;
      await record('*');   // between checks: setup
    } else if (started <= BASE_STEPS) {
      await record('*');
    } else {
      map.steps[id] = { name: check };
      await record(id);
    }
    await cdp.eval('globalThis.__markDone()');
  });
  return map;
}

/** Polls fn until it returns something truthy. */
async function until(fn, ms, message) {
  const end = Date.now() + ms;
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() > end) throw new Error(typeof message === 'function' ? message() : message);
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

/** The smallest DevTools protocol client that works: send, eval, on. */
function connect(url) {
  const ws = new WebSocket(url);
  const pending = new Map();
  const listeners = new Map();
  let next = 0;
  ws.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.id !== undefined) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message));
      else resolve(message.result);
    } else listeners.get(message.method)?.(message.params);
  };
  const client = {
    send: (method, params = {}) => new Promise((resolve, reject) => {
      const id = next++;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    }),
    on: (method, fn) => listeners.set(method, fn),
    async eval(expression) {
      const { result, exceptionDetails } = await client.send('Runtime.evaluate', {
        expression, returnByValue: true, awaitPromise: true,
      });
      if (exceptionDetails) throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
      return result.value;
    },
  };
  return new Promise((resolve, reject) => {
    ws.onopen = () => resolve(client);
    ws.onerror = () => reject(new Error(`could not connect to ${url}`));
  });
}
