// Bind group frequency model.
//
// WebGPU gives you 4 bind groups by default (maxBindGroups). That is not a
// limit to work around -- it maps exactly onto the four rates at which data
// actually changes, so we spend one group on each:
//
//   0  FRAME     camera matrices, time, light clusters   -- rebound once a frame
//   1  PASS      shadow cascade matrices, target info    -- rebound per pass
//   2  MATERIAL  textures, samplers, material params     -- rebound per material
//   3  DRAW      model matrix, via dynamic offset        -- rebound per object
//
// Cost is rebinding, so the cheap groups sit at the top and the expensive one
// at the bottom. This ordering is why the draw list sorts by material: crossing
// a material boundary is the rebind we are trying to avoid.
//
// A pipeline that skips a group still has to declare it, because a pipeline
// layout is a dense array. Unused slots get an empty layout and an empty bind
// group, both shared process-wide -- see emptyLayoutFor().

import { DEBUG, assert } from '../core/assert.js';

export const GROUP_FRAME = 0;
export const GROUP_PASS = 1;
export const GROUP_MATERIAL = 2;
export const GROUP_DRAW = 3;
export const GROUP_COUNT = 4;

// One empty layout + bind group per GPUDevice. Creating these per pipeline
// would be pure waste; they carry no data and are interchangeable.
const emptyCache = new WeakMap();

function emptyLayoutFor(device) {
  let cached = emptyCache.get(device);
  if (!cached) {
    const layout = device.createBindGroupLayout({ label: 'empty', entries: [] });
    const group = device.createBindGroup({ label: 'empty', layout, entries: [] });
    cached = { layout, group };
    emptyCache.set(device, cached);
  }
  return cached;
}

let nextLayoutId = 1;

/**
 * A pipeline layout plus the empty bind groups it needs at draw time.
 *
 * `groups` is sparse -- pass only the slots you use, keyed by the GROUP_*
 * constants. Gaps are filled with the shared empty layout.
 *
 *   const layout = createPipelineLayout(device, {
 *     [GROUP_FRAME]: frameLayout,
 *     [GROUP_DRAW]:  drawLayout,
 *   });
 */
export function createPipelineLayout(device, groups, label) {
  const { layout: emptyLayout, group: emptyGroup } = emptyLayoutFor(device);

  const layouts = [];
  const emptySlots = [];
  for (let i = 0; i < GROUP_COUNT; i++) {
    if (groups[i]) {
      layouts.push(groups[i]);
    } else {
      layouts.push(emptyLayout);
      emptySlots.push(i);
    }
  }

  if (DEBUG) {
    for (const key of Object.keys(groups)) {
      assert(Number(key) < GROUP_COUNT, `bind group ${key} exceeds GROUP_COUNT`);
    }
  }

  return new PipelineLayout(
    device.createPipelineLayout({ label, bindGroupLayouts: layouts }),
    emptySlots,
    emptyGroup,
    label,
  );
}

export class PipelineLayout {
  constructor(gpu, emptySlots, emptyGroup, label) {
    this.gpu = gpu;
    this.label = label;
    /** Stable identity for the pipeline cache key -- GPU objects are not comparable. */
    this.id = nextLayoutId++;
    this._emptySlots = emptySlots;
    this._emptyGroup = emptyGroup;
  }

  /**
   * Bind the placeholder groups. Required: WebGPU validates that every slot in
   * the pipeline layout has a bind group set, even one holding nothing.
   * Call once per pass, not per draw -- they never change.
   */
  bindEmptyGroups(pass) {
    for (let i = 0; i < this._emptySlots.length; i++) {
      pass.setBindGroup(this._emptySlots[i], this._emptyGroup);
    }
  }
}
