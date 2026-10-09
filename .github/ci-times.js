// What each GPU check takes on CI, into test/gpu-map.json, so .github/affected.js
// balances the runners by it. CI's GPU is software, and it does not slow every
// check alike: balanced by a real GPU's times, four runners took 113 to 227 s.
//
//   gh run view <run> --log | node .github/ci-times.js
//
// Reads the "[id ms]" gpu.ci.js prints after each check; a check run on more
// than one runner keeps its longest.

import { readFileSync, writeFileSync } from 'node:fs';

const path = new URL('../test/gpu-map.json', import.meta.url);
const map = JSON.parse(readFileSync(path, 'utf8'));
const log = readFileSync(0, 'utf8');
const times = new Map();
for (const [, id, ms] of log.matchAll(/\[([0-9a-z]+) (\d+) ms\]/g)) times.set(id, Math.max(times.get(id) ?? 0, Number(ms)));
let n = 0;
for (const [id, ms] of times) if (map.steps[id]) { map.steps[id].ciMs = ms; n++; }
writeFileSync(path, `${JSON.stringify(map)}\n`);
console.log(`${n} checks timed, of ${Object.keys(map.steps).length}`);
