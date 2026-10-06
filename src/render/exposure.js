// Auto exposure: the camera opens up in the dark and stops down in the light.
//
// Two compute passes, both tiny, and nothing read back on the way:
//
//   exposure-histogram  every fourth pixel each way of the scene, before
//                       exposure, sorted by log brightness into 64 bins
//   exposure-adapt      one workgroup: the bins' mean, its darkest and
//                       brightest tenths left out, gives the exposure that
//                       puts it at middle grey; the exposure moves toward it
//                       at so many stops a second, and the tonemap reads it
//
// The histogram is taken before exposure is applied, so what it measures does
// not depend on what it decides: no feedback, and a still scene settles on
// one value. Each step is clamped to land exactly on the target, which is how
// the engine knows, from a four-float readback a frame or two late, that the
// image has stopped changing and an idle frame can be skipped again.
//
// Trimming the tails is Bevy's choice too: a lamp in shot, or a black corner,
// is not what the eye adapts to.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';

const BINS = 64;
/** log2 luminance the bins span; bin 0 holds anything darker, and black. */
const LOG_MIN = -12;
const LOG_MAX = 12;
/** Pixels sampled: one in STRIDE each way. */
const STRIDE = 4;
const GROUP = 16;
/** Middle grey, where the trimmed mean is put. */
const KEY = 0.18;
/** The share of samples left out at each end. */
const TRIM = 0.1;

/** min, max, brighten, darken, dt, pad x3 */
const PARAMS_BYTES = 32;
/** ev, valid, target, pad */
const STATE_BYTES = 16;

/** The settings, and their defaults: stops of range, and stops a second. */
export const AUTO_EXPOSURE_DEFAULTS = Object.freeze({ min: -8, max: 8, brighten: 3, darken: 1 });

const SHADER = /* wgsl */ `
struct Params {
  minEv    : f32,
  maxEv    : f32,
  brighten : f32,
  darken   : f32,
  dt       : f32,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var scene : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> bins : array<atomic<u32>, ${BINS}>;
@group(0) @binding(3) var<storage, read_write> state : array<f32, 4>;

var<workgroup> local : array<atomic<u32>, ${BINS}>;
var<workgroup> counts : array<u32, ${BINS}>;

fn binCentre(k : u32) -> f32 {
  return ${LOG_MIN}.0 + (f32(k) - 0.5) / ${BINS - 1}.0 * ${LOG_MAX - LOG_MIN}.0;
}

@compute @workgroup_size(${GROUP}, ${GROUP})
fn histogram(@builtin(global_invocation_id) id : vec3<u32>, @builtin(local_invocation_index) i : u32) {
  if (i < ${BINS}u) { atomicStore(&local[i], 0u); }
  workgroupBarrier();
  let p = id.xy * ${STRIDE}u;
  if (all(p < textureDimensions(scene))) {
    let c = textureLoad(scene, p, 0).rgb;
    let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    var bin = 0u;
    if (l > exp2(${LOG_MIN}.0)) {
      let t = (log2(l) - ${LOG_MIN}.0) / ${LOG_MAX - LOG_MIN}.0;
      bin = 1u + u32(clamp(t * ${BINS - 1}.0, 0.0, ${BINS - 2}.0));
    }
    atomicAdd(&local[bin], 1u);
  }
  workgroupBarrier();
  if (i < ${BINS}u) { atomicAdd(&bins[i], atomicLoad(&local[i])); }
}

@compute @workgroup_size(${BINS})
fn adapt(@builtin(local_invocation_index) i : u32) {
  counts[i] = atomicLoad(&bins[i]);
  atomicStore(&bins[i], 0u);
  workgroupBarrier();
  if (i != 0u) { return; }

  // Black is left out: it says nothing about how bright the light is.
  var total = 0.0;
  for (var k = 1u; k < ${BINS}u; k++) { total += f32(counts[k]); }
  let low = total * ${TRIM};
  let high = total * ${1 - TRIM};
  var below = 0.0;
  var sum = 0.0;
  var weight = 0.0;
  for (var k = 1u; k < ${BINS}u; k++) {
    let c = f32(counts[k]);
    let kept = max(min(below + c, high) - max(below, low), 0.0);
    sum += kept * binCentre(k);
    weight += kept;
    below += c;
  }

  let valid = state[1] > 0.5;
  var ev = state[0];
  // Nothing lit to measure: hold where it is.
  if (weight <= 0.0) {
    state[1] = select(0.0, 1.0, valid);
    state[2] = ev;
    return;
  }
  let goal = clamp(log2(${KEY}) - sum / weight, params.minEv, params.maxEv);
  if (!valid) {
    ev = goal;
  } else {
    let delta = goal - ev;
    let step = select(params.darken, params.brighten, delta > 0.0) * params.dt;
    ev += clamp(delta, -step, step);
  }
  state[0] = ev;
  state[1] = 1.0;
  state[2] = goal;
}
`;

/** Checked once a frame, as fog and depth of field are. */
export function autoExposureSettings(value) {
  if (value === null || value === undefined) return null;
  if (value === true) return { ...AUTO_EXPOSURE_DEFAULTS };
  if (typeof value !== 'object') {
    throw new Error(`autoExposure: true, { min, max, brighten, darken }, or null to turn it off, got ${value}`);
  }
  const s = { ...AUTO_EXPOSURE_DEFAULTS, ...value };
  for (const key of ['min', 'max']) {
    if (!Number.isFinite(s[key])) throw new Error(`autoExposure: ${key} must be a finite number of stops, got ${s[key]}`);
  }
  if (!(s.min <= s.max)) throw new Error(`autoExposure: min must be at most max, got ${s.min} and ${s.max}`);
  for (const key of ['brighten', 'darken']) {
    if (!(s[key] > 0)) throw new Error(`autoExposure: ${key} must be a positive number of stops a second, got ${s[key]}`);
  }
  return s;
}

export class AutoExposure {
  static async create(rhi, pipelines) {
    const device = rhi.device;
    const shader = await compileShader(device, SHADER, 'exposure.wgsl');
    const compute = GPUShaderStage.COMPUTE;
    const layout = device.createBindGroupLayout({
      label: 'exposure',
      entries: [
        { binding: 0, visibility: compute, buffer: { type: 'uniform' } },
        { binding: 1, visibility: compute, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: compute, buffer: { type: 'storage' } },
        { binding: 3, visibility: compute, buffer: { type: 'storage' } },
      ],
    });
    const pipelineLayout = createPipelineLayout(device, { 0: layout }, 'exposure');
    const pipeline = (entry) => pipelines.compute({ label: `exposure-${entry}`, layout: pipelineLayout, shader, entry });
    return new AutoExposure(rhi, layout, pipeline('histogram'), pipeline('adapt'));
  }

  constructor(rhi, layout, histogram, adapt) {
    this.rhi = rhi;
    this.layout = layout;
    this.histogramPipeline = histogram;
    this.adaptPipeline = adapt;
    this.params = createBuffer(rhi, { label: 'exposure-params', size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bins = createBuffer(rhi, { label: 'exposure-bins', size: BINS * 4, usage: GPUBufferUsage.STORAGE });
    /** What the tonemap reads: x is the exposure in stops, 0 while auto exposure is off. */
    this.state = createBuffer(rhi, {
      label: 'exposure-state', size: STATE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._data = new Float32Array(PARAMS_BYTES / 4);
    this._ring = [0, 1].map(() => ({
      buffer: createBuffer(rhi, { label: 'exposure-readback', size: STATE_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      inFlight: false,
    }));
    this._pending = null;
    this._on = false;
    this._last = 0;
    /** Whether the last exposure read back was still moving toward its target. */
    this.adapting = false;
    /** The exposure auto exposure chose, in stops, as last read back. */
    this.stops = 0;
  }

  /**
   * This frame's passes, or none. `adapt` is false for a frame drawn into a
   * target: it is shown at the exposure the canvas has, and does not move it.
   * Returns the state buffer's graph handle for the tonemap to read, or null.
   */
  addPasses(graph, { settings, sceneColor, width, height, adapt }) {
    if ((settings !== null) !== this._on) {
      // On: snap to the first measurement. Off: back to no change at all.
      this.rhi.queue.writeBuffer(this.state, 0, new Float32Array(4));
      this._on = settings !== null;
      this.adapting = this._on;
      this.stops = 0;
      this._last = 0;
    }
    if (settings === null || !adapt) return null;

    const now = performance.now();
    // Real seconds, as the clock counts them, capped as it caps them.
    const dt = this._last === 0 ? 0 : Math.min((now - this._last) / 1000, 0.25);
    this._last = now;
    const d = this._data;
    d[0] = settings.min;
    d[1] = settings.max;
    d[2] = settings.brighten;
    d[3] = settings.darken;
    d[4] = dt;
    this.rhi.queue.writeBuffer(this.params, 0, d);

    const bins = graph.importBuffer('exposure-bins', this.bins);
    const state = graph.importBuffer('exposure-state', this.state);
    const groups = [Math.ceil(width / STRIDE / GROUP), Math.ceil(height / STRIDE / GROUP)];
    graph.addPass({
      name: 'exposure-histogram', type: 'compute', reads: [sceneColor], writes: [bins],
      execute: (pass) => this._dispatch(pass, graph, sceneColor, this.histogramPipeline, groups),
    });
    graph.addPass({
      name: 'exposure-adapt', type: 'compute', reads: [bins], writes: [state],
      execute: (pass) => this._dispatch(pass, graph, sceneColor, this.adaptPipeline, [1, 1]),
    });
    this._readback = true;
    return state;
  }

  _dispatch(pass, graph, sceneColor, pipeline, [x, y]) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.rhi.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: graph.viewOf(sceneColor) },
        { binding: 2, resource: { buffer: this.bins } },
        { binding: 3, resource: { buffer: this.state } },
      ],
    }));
    pass.dispatchWorkgroups(x, y);
  }

  /** After the last pass, before the encoder closes: copy the state out, if a buffer is free. */
  resolve(encoder) {
    if (!this._readback) return;
    this._readback = false;
    const slot = this._ring.find((s) => !s.inFlight);
    // Both still mapping: the GPU is behind, and the next frame will ask again.
    if (!slot) return;
    encoder.copyBufferToBuffer(this.state, 0, slot.buffer, 0, STATE_BYTES);
    slot.inFlight = true;
    this._pending = slot;
  }

  /** After the submit, as the profiler's readback is. */
  readback() {
    const slot = this._pending;
    if (!slot) return;
    this._pending = null;
    slot.buffer.mapAsync(GPUMapMode.READ)
      .then(() => {
        const [ev, valid, target] = new Float32Array(slot.buffer.getMappedRange());
        slot.buffer.unmap();
        if (!this._on) return;
        this.stops = ev;
        this.adapting = valid < 0.5 || Math.abs(target - ev) > 1e-4;
      })
      .catch(() => { /* device lost, or destroyed mid-flight */ })
      .finally(() => { slot.inFlight = false; });
  }

  destroy() {
    for (const buffer of [this.params, this.bins, this.state, ...this._ring.map((s) => s.buffer)]) buffer.destroy();
  }
}
