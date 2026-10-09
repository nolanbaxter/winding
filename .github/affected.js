// What a change has to be checked by, and nothing else.
//
//   node .github/affected.js <base> <head>     -> GITHUB_OUTPUT lines: node, types, gpu
//
// NODE SUITES: a suite runs when a changed file is in what it imports, or reads
// through new URL(..., import.meta.url), followed all the way down. They take
// seconds; the point is the GPU suite.
//
// THE GPU SUITE takes minutes on a runner with no GPU, and nearly every change
// reaches only some of it. test/gpu-map.json says, for every function in src/
// the suite runs, which checks run it -- '*' for the setup every check draws on
// (GPU_MAP=1 node test/gpu.ci.js makes it, at a commit it records). A changed
// line is carried from this change's code back to the map's commit through git
// diff, and its innermost function names the checks. A line in no function --
// a shader's source, a constant -- reaches every check of its file. The checks
// chosen are split across up to RUNNERS runners, balanced by the time each took.
//
// WHEN IN DOUBT, EVERYTHING: a change to the suite, its fixtures or runner, to
// CI or to this file, a new file in src/, a map from a commit git cannot find,
// or a base it cannot diff against.

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, appendFileSync } from 'node:fs';
import { dirname, join, normalize, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
export const RUNNERS = 4;
/**
 * A runner's worth of checks, in CI's time (test/gpu-map.json's ciMs, from
 * .github/ci-times.js; where a check has none, its time on the real GPU the map
 * was made on, about seven times quicker). Each runner also starts Chrome and
 * builds the setup every check needs, a minute or so, so a handful of quick
 * checks share one rather than taking four.
 */
const PER_RUNNER_MS = 60000;

/** Changes that are checked by everything there is. */
const EVERYTHING = [/^package\.json$/, /^\.github\/workflows\//, /^\.github\/affected\.js$/, /^serve\.js$/];
/** Changes that the GPU suite as a whole checks. */
const WHOLE_GPU = [/^test\/gpu\.(test|ci)\.js$/, /^test\/gpu\.html$/, /^test\/fixtures\//];

const git = (...args) => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', maxBuffer: 64 << 20 });

/** Every file `file` reaches: its imports and new URL() references, followed down. Paths relative to the root. */
export function reaches(file, read = (f) => readFileSync(join(ROOT, f), 'utf8'), seen = new Set()) {
  if (seen.has(file)) return seen;
  seen.add(file);
  if (!/\.m?js$/.test(file)) return seen;
  let text;
  try { text = read(file); } catch { return seen; }
  const refs = [
    ...text.matchAll(/\bfrom\s*['"](\.{1,2}\/[^'"]+)['"]/g),
    ...text.matchAll(/\bimport\s*\(?\s*['"](\.{1,2}\/[^'"]+)['"]/g),
    ...text.matchAll(/new URL\(\s*['"](\.{1,2}\/[^'"]+)['"]\s*,\s*import\.meta\.url\s*\)/g),
  ];
  for (const [, ref] of refs) reaches(normalize(join(dirname(file), ref)).replaceAll('\\', '/'), read, seen);
  return seen;
}

/** The lines of `path` a diff touches, in its new version: [first, last] pairs. A deletion touches the line either side. */
export function touched(diff) {
  const out = [];
  for (const [, , , start, count = '1'] of diff.matchAll(/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/gm)) {
    const s = Number(start), n = Number(count);
    out.push(n === 0 ? [s, s + 1] : [s, s + n - 1]);
  }
  return out;
}

/**
 * Carry a range of lines in a file's new version back to its old one, through
 * the hunks of the diff between them. A line inside a changed hunk is the whole
 * of that hunk's old lines; one outside moves by what the hunks before it added.
 */
export function carry([first, last], diff) {
  // A count of 0 places the start BEFORE the lines: an insertion sits after old
  // line os, a deletion after new line ns. The last line each side is then os
  // or ns itself, and the line after a hunk moves by the difference of the two.
  const hunks = [...diff.matchAll(/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/gm)].map(([, os, oc = '1', ns, nc = '1']) => {
    const o = { start: +os, count: +oc }, n = { start: +ns, count: +nc };
    return {
      o, n,
      oldFirst: o.count === 0 ? o.start : o.start, oldLast: o.count === 0 ? o.start + 1 : o.start + o.count - 1,
      newLast: n.count === 0 ? n.start : n.start + n.count - 1,
    };
  });
  const one = (line, end) => {
    let shift = 0;
    for (const h of hunks) {
      if (h.n.count > 0 && line >= h.n.start && line <= h.newLast) return end ? h.oldLast : h.oldFirst;
      if (line <= h.newLast) break;
      shift = (h.o.count === 0 ? h.o.start : h.o.start + h.o.count - 1) - h.newLast;
    }
    return line + shift;
  };
  return [one(first, false), one(last, true)];
}

/**
 * The checks a touched range of `path` (map lines) reaches, by the map's
 * functions: '*' when it reaches the setup, so every check.
 */
export function checksFor(functions, [first, last]) {
  const out = new Set();
  const ranges = Object.entries(functions).map(([key, ids]) => [...key.split('-').map(Number), ids]);
  for (let line = first; line <= last; line++) {
    let best = null;
    for (const r of ranges) if (r[0] <= line && line <= r[1] && (best === null || r[1] - r[0] < best[1] - best[0])) best = r;
    // In no function: a shader's source, a constant -- whatever reads it.
    const ids = best !== null ? best[2] : ranges.flatMap((r) => r[2]);
    for (const id of ids) out.add(id);
  }
  return out;
}

/** Split checks across up to RUNNERS runners, the longest first onto the least loaded. */
export function shard(ids, ms, runners = RUNNERS) {
  const total = ids.reduce((t, id) => t + (ms[id] ?? 7000), 0);
  const n = Math.max(1, Math.min(runners, ids.length, Math.ceil(total / PER_RUNNER_MS)));
  const bins = Array.from({ length: n }, () => ({ total: 0, ids: [] }));
  for (const id of [...ids].sort((a, b) => (ms[b] ?? 1000) - (ms[a] ?? 1000))) {
    const bin = bins.reduce((x, y) => (y.total < x.total ? y : x));
    bin.ids.push(id);
    bin.total += ms[id] ?? 7000;
  }
  return bins.map((bin, i) => ({ shard: `${i}/${n}`, steps: bin.ids.join(',') }));
}

/** The plan for the change base..head. */
export function plan(base, head) {
  const suites = readdirSync(join(ROOT, 'test')).filter((f) => f.endsWith('.test.js') && f !== 'gpu.test.js').map((f) => `test/${f}`);
  const all = { node: suites, types: true, gpu: Array.from({ length: RUNNERS }, (_, i) => ({ shard: `${i}/${RUNNERS}` })), why: [] };
  let changed;
  try {
    if (/^0+$/.test(base)) throw new Error('no base');
    changed = git('diff', '--name-only', base, head).split('\n').filter(Boolean);
  } catch {
    return { ...all, why: ['no base to compare with: everything'] };
  }
  // A release bumps package.json's version and nothing else of it: that runs nothing.
  const versionOnly = (f) => f === 'package.json'
    && git('diff', '-U0', base, head, '--', f).split('\n').filter((l) => /^[+-](?![+-])/.test(l)).every((l) => /"version":/.test(l));
  if (changed.some((f) => EVERYTHING.some((re) => re.test(f)) && !versionOnly(f))) return { ...all, why: ['CI itself changed: everything'] };

  const node = suites.filter((suite) => { const set = reaches(suite); return changed.some((f) => set.has(f)); });
  const types = changed.some((f) => /\.d\.ts$/.test(f) || f === 'test/types.ts');

  const why = [];
  let map = null;
  try {
    map = JSON.parse(readFileSync(join(ROOT, 'test/gpu-map.json'), 'utf8'));
    git('cat-file', '-e', `${map.commit}^{commit}`);
  } catch { map = null; }
  const ms = map === null ? {} : Object.fromEntries(Object.entries(map.steps).map(([id, s]) => [id, s.ciMs ?? (s.ms ?? 1000) * 7]));
  // Every check, balanced by the map's times -- one it does not know yet runs
  // in shard 0 -- or dealt out in turn without one.
  const whole = map === null ? all.gpu : shard(Object.keys(ms), ms);

  let gpu = [];
  const source = changed.filter((f) => /^src\/.*\.js$/.test(f));
  if (changed.some((f) => WHOLE_GPU.some((re) => re.test(f)))) {
    gpu = whole;
    why.push('the GPU suite itself changed: all of it');
  } else if (source.length > 0 && map === null) {
    gpu = whole;
    why.push('no usable test/gpu-map.json: all of the GPU suite');
  } else if (source.length > 0) {
    const chosen = new Set();
    for (const file of source) {
      if (!existsSync(join(ROOT, file))) continue;   // deleted: what used it changed too
      try { git('cat-file', '-e', `${map.commit}:${file}`); } catch {
        chosen.add('*');
        why.push(`${file} is new since the map: all of the GPU suite`);
        break;
      }
      const functions = map.files[file];
      if (!functions) continue;   // nothing the suite runs is in it
      const sinceMap = git('diff', '-U0', map.commit, head, '--', file);
      for (const range of touched(git('diff', '-U0', base, head, '--', file))) {
        for (const id of checksFor(functions, carry(range, sinceMap))) chosen.add(id);
      }
      if (chosen.has('*')) { why.push(`${file} changes what every check sets up: all of the GPU suite`); break; }
    }
    if (chosen.has('*')) gpu = whole;
    else if (chosen.size > 0) {
      gpu = shard([...chosen], ms);
      why.push(`${chosen.size} of ${Object.keys(map.steps).length} GPU checks reach the change`);
    }
  }
  return { node, types, gpu, why: [...why, `${node.length} of ${suites.length} Node suites`] };
}

if (process.argv[1] && relative(process.argv[1], fileURLToPath(import.meta.url)) === '') {
  const [base = 'HEAD~1', head = 'HEAD'] = process.argv.slice(2);
  const p = plan(base, head);
  const lines = [`node=${p.node.join(' ')}`, `types=${p.types}`, `gpu=${JSON.stringify(p.gpu)}`];
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `${lines.join('\n')}\n`);
  console.log(lines.join('\n'));
  for (const reason of p.why) console.log(`# ${reason}`);
  if (process.env.GITHUB_STEP_SUMMARY) {
    appendFileSync(process.env.GITHUB_STEP_SUMMARY, `### What this change is checked by\n\n${p.why.map((w) => `- ${w}`).join('\n')}\n- GPU runners: ${p.gpu.length}\n`);
  }
}
