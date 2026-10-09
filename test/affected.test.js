// What a change is checked by (.github/affected.js). Run: node test/affected.test.js
//
// The plan as CI makes it needs git and a recorded map; this is its arithmetic:
// carrying lines through a diff, the function a line is in, the import graph,
// and splitting checks across runners.

import assert from 'node:assert/strict';

import { reaches, touched, carry, checksFor, shard, plan } from '../.github/affected.js';

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`  ok  ${name}`);
}

test('a diff touches the lines it changes, and either side of a deletion', () => {
  const diff = '@@ -10,2 +10,3 @@\n@@ -40 +41 @@\n@@ -60,3 +61,0 @@\n';
  assert.deepEqual(touched(diff), [[10, 12], [41, 41], [61, 62]]);
});

test('lines are carried back through the hunks before them', () => {
  // Old 5..6 became new 5..8 (two lines more); old 20 was deleted, after new 21;
  // a line went in after old 30, which by then is new 31: the new line is 32.
  const diff = '@@ -5,2 +5,4 @@\n@@ -20 +21,0 @@\n@@ -30,0 +32 @@\n';
  assert.deepEqual(carry([2, 3], diff), [2, 3], 'before every hunk: unmoved');
  assert.deepEqual(carry([6, 7], diff), [5, 6], 'inside a changed hunk: all of its old lines');
  assert.deepEqual(carry([10, 10], diff), [8, 8], 'after a hunk two longer: two up');
  assert.deepEqual(carry([25, 25], diff), [24, 24], 'after a deletion: one further down again');
  assert.deepEqual(carry([31, 31], diff), [30, 30], 'the line before it');
  assert.deepEqual(carry([32, 32], diff), [30, 31], 'an inserted line: where it went in');
  assert.deepEqual(carry([40, 40], diff), [38, 38], 'after an insertion: one up');
});

test('a line reaches the checks of its innermost function, or of the whole file outside any', () => {
  const functions = { '1-100': ['a'], '10-20': ['b'], '30-40': ['*'] };
  assert.deepEqual([...checksFor(functions, [12, 12])], ['b']);
  assert.deepEqual([...checksFor(functions, [50, 50])], ['a']);
  assert.ok(checksFor(functions, [35, 35]).has('*'), 'the setup: every check');
  assert.deepEqual([...checksFor(functions, [150, 150])].sort(), ['*', 'a', 'b'], 'outside any function: the file');
});

test('checks are split across runners, the longest first onto the least loaded', () => {
  const ms = { a: 9000, b: 5000, c: 4000, d: 1000, e: 500 };
  const shards = shard(['e', 'd', 'c', 'b', 'a'], ms, 2);
  assert.deepEqual(shards.map((s) => s.shard), ['0/2', '1/2']);
  assert.deepEqual(shards.map((s) => s.steps), ['a,d', 'b,c,e'], '10 s and 9.5 s');
  assert.equal(shard(['a'], ms).length, 1, 'never more runners than checks');
});

test('a file reaches what it imports and loads by URL, all the way down', () => {
  const files = {
    'test/x.test.js': "import { a } from '../src/a.js';\nconst w = new URL('../src/w.js', import.meta.url);",
    'src/a.js': "export { b } from './b.js';\nimport('./lazy.js');",
    'src/b.js': '',
    'src/lazy.js': "import { a } from './a.js';",
    'src/w.js': '',
  };
  const set = reaches('test/x.test.js', (f) => files[f]);
  assert.deepEqual([...set].sort(), ['src/a.js', 'src/b.js', 'src/lazy.js', 'src/w.js', 'test/x.test.js']);
});

test('with no base to compare with, everything runs', () => {
  const p = plan('0000000000000000000000000000000000000000', 'HEAD');
  assert.ok(p.node.length > 10 && p.types === true && p.gpu.length === 4);
});

console.log(`\n${passed} checks passed\n`);
