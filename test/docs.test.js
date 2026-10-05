// The API reference is complete, and every link in it goes somewhere.
// Run: node test/docs.test.js
//
// Written by hand, docs/API.md drifts the moment a method is added without an
// entry or an entry is renamed out from under a link. So what it must cover is
// read from the code itself -- every public method and getter of every
// exported class -- and a missing entry fails here, not in a reader's search.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { Winding } from '../src/app/engine.js';
import { Scene } from '../src/scene/scene.js';
import { Node } from '../src/scene/node.js';
import { Camera } from '../src/scene/camera.js';
import { Camera2D } from '../src/scene/camera2d.js';
import { OrbitController } from '../src/app/controllers.js';
import { StatsOverlay } from '../src/app/overlay.js';
import * as winding from '../src/winding.js';
import * as bench from '../src/bench.js';

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`  ok  ${name}`);
}

const API = readFileSync(new URL('../docs/API.md', import.meta.url), 'utf8');
const README = readFileSync(new URL('../README.md', import.meta.url), 'utf8');
const anchors = new Set([...API.matchAll(/<a id="([^"]+)"><\/a>/g)].map((m) => m[1]));

/**
 * Public, but not for users: the renderer's and the engine's own calls on a
 * scene, each named with why. A method is either documented or listed here.
 */
const INTERNAL = new Map([
  ['scene-morphweights', 'the renderer reads a morph target set through it'],
  ['scene-applymorphbounds', 'the renderer pads bounds for morphs with it'],
  ['scene-refreshlights', 'the renderer copies light positions out of transforms with it'],
]);

/** Every public method and getter a class's instances have, as `prefix-name` anchors. */
function surface(prefix, cls) {
  return Object.getOwnPropertyNames(cls.prototype)
    .filter((name) => name !== 'constructor' && !name.startsWith('_'))
    .map((name) => `${prefix}-${name.toLowerCase()}`);
}

console.log('\ndocs/API.md');

test('every public method and getter has an entry', () => {
  const expected = [
    ...surface('engine', Winding), 'winding-create',
    ...surface('scene', Scene), ...surface('node', Node),
    ...surface('camera', Camera), ...surface('camera2d', Camera2D),
    ...surface('orbitcontroller', OrbitController), ...surface('statsoverlay', StatsOverlay),
  ];
  const missing = expected.filter((id) => !anchors.has(id) && !INTERNAL.has(id));
  assert.deepEqual(missing, [], `no entry for: ${missing.join(', ')}`);
});

test('every link to an entry lands on one', () => {
  const inPage = [...API.matchAll(/\]\(#([^)]+)\)/g)].map((m) => m[1]);
  const fromReadme = [...README.matchAll(/\]\(docs\/API\.md#([^)]+)\)/g)].map((m) => m[1]);
  const broken = [...new Set([...inPage, ...fromReadme])].filter((id) => !anchors.has(id));
  assert.deepEqual(broken, [], `links to nothing: ${broken.join(', ')}`);
});

test('the index lists every entry', () => {
  const start = API.indexOf('\n## Index');
  assert.ok(start >= 0, 'an index');
  // To the next section's heading, after the index's own.
  const end = API.indexOf('\n## ', start + 1);
  const index = API.slice(start, end < 0 ? undefined : end);
  const listed = new Set([...index.matchAll(/\]\(#([^)]+)\)/g)].map((m) => m[1]));
  const entries = [...API.matchAll(/<a id="([^"]+)"><\/a>\s*\n###/g)].map((m) => m[1]);
  const missing = entries.filter((id) => !listed.has(id));
  assert.deepEqual(missing, [], `not in the index: ${missing.join(', ')}`);
});

console.log('\nsrc/winding.d.ts');

const TYPES = readFileSync(new URL('../src/winding.d.ts', import.meta.url), 'utf8')
  + readFileSync(new URL('../src/bench.d.ts', import.meta.url), 'utf8');

/** Public, but not for users, so not typed: each named with why. */
const UNTYPED = new Map([
  ['Scene.morphWeights', 'the renderer reads a morph target set through it'],
  ['Scene.applyMorphBounds', 'the renderer pads bounds for morphs with it'],
  ['Scene.refreshLights', 'the renderer copies light positions out of transforms with it'],
  ['Environment.convolve', 'a probe is prefiltered through it once captured'],
  ['Benchmark.frameStart', "the renderer's profiling hook"],
  ['Benchmark.mark', "the renderer's profiling hook"],
  ['Benchmark.frameEnd', "the renderer's profiling hook"],
  ['boundBy', "Benchmark.format's helper"],
  ['summarize', "Benchmark.report's helper"],
]);

/** The body of `export class name { ... }`, to its closing brace. */
function classBody(name) {
  const start = TYPES.indexOf(`export class ${name} `);
  assert.ok(start >= 0, `no class ${name}`);
  return TYPES.slice(start, TYPES.indexOf('\n}', start));
}

test('every export is typed', () => {
  const missing = [...Object.keys(winding), ...Object.keys(bench)].filter((name) => !UNTYPED.has(name)
    && !new RegExp(`export (class|function) ${name}\\b`).test(TYPES));
  assert.deepEqual(missing, [], `not typed: ${missing.join(', ')}`);
});

test('every public method and getter is typed', () => {
  const classes = {
    Winding, Scene, Node, Camera, Camera2D, OrbitController, StatsOverlay,
    Environment: winding.Environment, Benchmark: bench.Benchmark,
  };
  const missing = [];
  for (const [name, cls] of Object.entries(classes)) {
    const body = classBody(name);
    const statics = Object.getOwnPropertyNames(cls).filter((m) => !['length', 'name', 'prototype'].includes(m));
    for (const m of [...Object.getOwnPropertyNames(cls.prototype), ...statics]) {
      if (m === 'constructor' || m.startsWith('_') || UNTYPED.has(`${name}.${m}`)) continue;
      if (!new RegExp(`\\n  (static |readonly |get |set )*${m}\\??[(:<]`).test(body)) missing.push(`${name}.${m}`);
    }
  }
  assert.deepEqual(missing, [], `not typed: ${missing.join(', ')}`);
});

console.log(`\n${passed} checks passed\n`);
