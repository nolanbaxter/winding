// The GPU suite, headless: node test/gpu.ci.js
//
// Serves the repo, opens test/gpu.html in Chrome with no window, waits for the
// page's own verdict (globalThis.__gpuTest, set by finish() in gpu.test.js) and
// exits 0 only if every check passed. CI runs it on a runner with no GPU, where
// Chrome falls back to SwiftShader, its own software device: slow, but it
// compiles and runs every shader, which is the point of the suite.
//
// No Puppeteer: Chrome is driven over its DevTools protocol with Node's own
// WebSocket (Node 22+), so nothing is installed and nothing is downloaded.
// Chrome comes from $CHROME, or the usual install paths.

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
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
    ? ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-vulkan-surface'] : []),
  // A background tab is throttled, and this one is never in front.
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--window-size=800,600',
  'about:blank',
], { stdio: 'ignore' });

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
  }, 30_000, 'Chrome never opened its DevTools port');

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
  await cdp.send('Page.navigate', { url: `http://localhost:${PORT}/test/gpu.html` });

  const started = Date.now();
  const adapter = await cdp.eval(`navigator.gpu?.requestAdapter().then((a) => a
    ? [a.info.vendor, a.info.architecture, a.info.device, a.info.description].filter(Boolean).join(' ') || 'unnamed adapter'
    : 'no adapter')`);
  console.log(`adapter: ${adapter ?? 'navigator.gpu is undefined'}`);

  const summary = await until(async () => {
    const title = await cdp.eval('document.title');
    if (title.startsWith('FAIL harness')) throw new Error('the page could not run the suite; see the errors above');
    return /^(PASS|FAIL) /.test(title) && cdp.eval('globalThis.__gpuTest');
  }, TIMEOUT, `the suite did not finish in ${TIMEOUT / 60_000} minutes`);

  for (const r of summary.results) {
    console.log(`${r.ok ? 'ok  ' : 'FAIL'}  ${r.name}`);
    if (r.detail) console.log(`      ${r.detail.replaceAll('\n', '\n      ')}`);
  }
  const seconds = ((Date.now() - started) / 1000).toFixed(0);
  console.log(`\n${summary.passed} passed, ${summary.failed} failed, in ${seconds} s`);
  return summary.failed === 0 && summary.passed > 0 ? 0 : 1;
}

/** Polls fn until it returns something truthy. */
async function until(fn, ms, message) {
  const end = Date.now() + ms;
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() > end) throw new Error(message);
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
