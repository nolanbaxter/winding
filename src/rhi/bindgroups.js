// Bind group frequency model.
//
// WebGPU gives you 4 bind groups by default (maxBindGroups). That is not a
// limit to work around -- it maps onto the rates at which data actually
// changes, so we spend one group on each:
//
//   0  FRAME     camera matrices, time, light clusters,
//                shadow cascade matrices                 -- rebound once a frame
//   1  --        reserved, currently unused
//   2  MATERIAL  textures, samplers, material params     -- rebound per material
//   3  DRAW      model matrix, via dynamic offset        -- rebound per object
//
// Slot 1 was a PASS rate, for anything that changes between passes but not
// within one. Nothing turned out to need it: the shadow pass rebinds nothing
// of its own, because the cascade matrices are uniform across the frame and
// live in FRAME with everything else at that rate. The slot is left empty
// rather than renumbered, since the numbering is what the shaders name.
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
/** Reserved. Nothing binds at this rate; see the header. */
export const GROUP_RESERVED = 1;
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

  // Unconditional, and before the layouts are built. The loop below only ever
  // reads slots 0..GROUP_COUNT-1, so a group handed in at index 4 was silently
  // dropped in a release build and the pipeline was created with an empty
  // layout in its place -- surfacing much later as a WebGPU validation error
  // naming the wrong thing. A tier-2 caller supplying its own groups is exactly
  // who would hit it.
  for (const key of Object.keys(groups)) {
    if (!(Number(key) < GROUP_COUNT)) {
      throw new Error(`createPipelineLayout: bind group ${key} exceeds the ${GROUP_COUNT} WebGPU guarantees`);
    }
  }

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
