// HZB occlusion math. Run: node test/hzb.test.js
//
// The pyramid itself is a GPU pass, but the two decisions that make or break it
// -- how big the pyramid is, and which level answers a given screen rectangle --
// are pure and sit here.

import assert from 'node:assert/strict';
import { previousPowerOfTwo, mipLevelForExtent } from '../src/render/hzb.js';

let passed = 0;
function test(name, fn) { fn(); passed++; console.log(`  ok  ${name}`); }

console.log('\npyramid sizing');

test('the pyramid is the largest power of two that fits', () => {
  // Power-of-two dimensions make every reduction an exact 2x2, which is what
  // lets the reduce shader skip footprint arithmetic entirely.
  assert.equal(previousPowerOfTwo(1268), 1024);
  assert.equal(previousPowerOfTwo(910), 512);
  assert.equal(previousPowerOfTwo(1024), 1024, 'an exact power stays put');
  assert.equal(previousPowerOfTwo(1), 1);
  assert.equal(previousPowerOfTwo(0), 1, 'degenerate size still yields a valid texture');
});

test('one output texel covers between one and two input texels', () => {
  // The first level gathers 2x2 from the floored source position, which only
  // covers the full footprint while this holds. A missed texel would make the
  // stored value too large, over-occlude, and delete visible geometry.
  for (const size of [1268, 910, 1920, 1080, 3, 5, 100]) {
    const ratio = size / previousPowerOfTwo(size);
    assert.ok(ratio >= 1 && ratio < 2, `ratio ${ratio} at ${size} breaks the 2x2 gather`);
  }
});

console.log('\nlevel selection');

test('a rectangle spanning n texels picks the level where it spans about two', () => {
  const levels = 11;
  assert.equal(mipLevelForExtent(1, 1, levels), 0, 'one texel needs no reduction');
  assert.equal(mipLevelForExtent(2, 2, levels), 1);
  assert.equal(mipLevelForExtent(4, 1, levels), 2);
  assert.equal(mipLevelForExtent(5, 3, levels), 3, 'rounds up, never down');
  assert.equal(mipLevelForExtent(256, 200, levels), 8);
});

test('the level is clamped to the pyramid', () => {
  // A huge object must not index past the 1x1 top, and a degenerate rectangle
  // must not index below zero.
  assert.equal(mipLevelForExtent(1e9, 1e9, 11), 10);
  assert.equal(mipLevelForExtent(0, 0, 11), 0);
  assert.equal(mipLevelForExtent(-5, -5, 11), 0);
});

test('level selection is monotonic in extent', () => {
  let previous = -1;
  for (let extent = 1; extent <= 1024; extent *= 1.5) {
    const level = mipLevelForExtent(extent, extent, 11);
    assert.ok(level >= previous, `went backwards at ${extent}`);
    previous = level;
  }
});

console.log(`\n${passed} checks passed\n`);
