// Reflection probes: the scene seen from a point, for the surfaces in a box
// around it to reflect instead of the sky.
//
// A room lit by the sky reflects the sky, on its floor, its walls and every
// metal thing in it. A probe captures the room itself -- six renders of the
// scene from its position -- and prefilters them exactly as the environment
// is prefiltered, at the environment's resolution and mips, so one roughness
// indexes both. Surfaces in its box reflect that, projected onto the box
// rather than read from infinity, so a reflection of the far wall sits on the
// far wall.
//
// Captured on request, not every frame: six scene renders and a prefilter is
// a load-time cost, like the environment's bake. Recapture after the room
// changes.
//
// Specular only. What a probe's room contributes to diffuse light is a light
// probe's job, which this is not.

import { createTexture } from '../rhi/texture.js';
import { createBuffer } from '../rhi/buffer.js';
import { compileShaderSync } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { sharedPipelines } from '../rhi/pipeline.js';
import { grownCapacity } from '../core/grow.js';

/** One probe for the shader: box min (w = blend), box max (w = layer), position. */
export const PROBE_FLOATS = 12;

/**
 * The probes packed for the shader, smallest box first: where boxes nest,
 * the tighter one is the better answer, and the shader lets each one take
 * what the ones before it left. Only captured probes; a probe that has never
 * been captured has nothing to show.
 */
export function packProbes(probes, layers) {
  const captured = probes.filter((p) => p.captured);
  const volume = (p) => (p.max[0] - p.min[0]) * (p.max[1] - p.min[1]) * (p.max[2] - p.min[2]);
  captured.sort((a, b) => volume(a) - volume(b));
  const out = new Float32Array(Math.max(captured.length, 1) * PROBE_FLOATS);
  captured.forEach((p, k) => {
    const o = k * PROBE_FLOATS;
    out.set(p.min, o);
    out[o + 3] = p.blend;
    out.set(p.max, o + 4);
    out[o + 7] = layers.get(p);
    out.set(p.position, o + 8);
  });
  return { data: out, count: captured.length };
}

/**
 * Each cube face's camera, as look-at direction and up. Rendered with an
 * ordinary right-handed camera and copied in mirrored across u (see FLIP),
 * texel (u, v) of face f then shows exactly cubeDirection(f, u, v) -- the
 * direction the shader samples it by.
 */
export const FACE_CAMERAS = [
  { forward: [1, 0, 0], up: [0, 1, 0] },
  { forward: [-1, 0, 0], up: [0, 1, 0] },
  { forward: [0, 1, 0], up: [0, 0, -1] },
  { forward: [0, -1, 0], up: [0, 0, 1] },
  { forward: [0, 0, 1], up: [0, 1, 0] },
  { forward: [0, 0, -1], up: [0, 1, 0] },
];

const FLIP_SHADER = /* wgsl */ `
@group(0) @binding(0) var source : texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) index : u32) -> @builtin(position) vec4<f32> {
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

/** A cube face is the camera's image mirrored across u. */
@fragment
fn fs(@builtin(position) at : vec4<f32>) -> @location(0) vec4<f32> {
  let size = i32(textureDimensions(source).x);
  return textureLoad(source, vec2<i32>(size - 1 - i32(at.x), i32(at.y)), 0);
}
`;

/**
 * A scene's probes on the GPU: one cube array, six layers a probe, every mip
 * of the prefilter; and their boxes, packed. Held by the renderer per scene,
 * since a scene holds no GPU objects.
 */
export class ProbeSet {
  constructor(rhi, size, mips) {
    this.rhi = rhi;
    this.size = size;
    this.mips = mips;
    this.capacity = 0;
    this.texture = null;
    this.view = null;
    this.count = 0;
    /** Which probe sits in which array slot: slots outlive reordering. */
    this.layers = new Map();
    this.buffer = createBuffer(rhi, {
      label: 'reflection-probes', size: PROBE_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.revision = 0;
    this._grow(1);
  }

  /** The slot a probe's cube lives in, allocated on first capture. */
  slotFor(probe) {
    let slot = this.layers.get(probe);
    if (slot === undefined) {
      slot = this.layers.size;
      if (slot >= this.capacity) this._grow(slot + 1);
      this.layers.set(probe, slot);
    }
    return slot;
  }

  _grow(needed) {
    const capacity = grownCapacity(this.capacity, needed);
    const texture = createTexture(this.rhi, {
      label: 'reflection-probes',
      size: [this.size, this.size, capacity * 6],
      format: 'rgba16float',
      mipLevelCount: this.mips,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });
    if (this.texture !== null) {
      const encoder = this.rhi.device.createCommandEncoder({ label: 'reflection-probes-grow' });
      for (let mip = 0; mip < this.mips; mip++) {
        const s = Math.max(this.size >> mip, 1);
        encoder.copyTextureToTexture({ texture: this.texture, mipLevel: mip }, { texture, mipLevel: mip }, [s, s, this.capacity * 6]);
      }
      this.rhi.queue.submit([encoder.finish()]);
      this.texture.destroy();
    }
    this.texture = texture;
    this.view = texture.createView({ dimension: 'cube-array', label: 'reflection-probes' });
    this.capacity = capacity;
    this.revision++;
  }

  /** Upload the boxes of every captured probe. */
  upload(probes) {
    const { data, count } = packProbes(probes, this.layers);
    if (data.byteLength > this.buffer.size) {
      this.buffer.destroy();
      this.buffer = createBuffer(this.rhi, {
        label: 'reflection-probes', size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.revision++;
    }
    this.rhi.queue.writeBuffer(this.buffer, 0, data);
    this.count = count;
  }

  /** Copy a prefiltered cube, every mip, into a probe's slot. */
  store(slot, prefiltered) {
    const encoder = this.rhi.device.createCommandEncoder({ label: 'reflection-probe-store' });
    for (let mip = 0; mip < this.mips; mip++) {
      const s = Math.max(this.size >> mip, 1);
      encoder.copyTextureToTexture(
        { texture: prefiltered, mipLevel: mip },
        { texture: this.texture, mipLevel: mip, origin: { x: 0, y: 0, z: slot * 6 } },
        [s, s, 6],
      );
    }
    this.rhi.queue.submit([encoder.finish()]);
  }

  destroy() {
    this.texture?.destroy();
    this.buffer.destroy();
  }
}

/** Copy a rendered face into a cube face, mirrored across u; see FACE_CAMERAS. */
export function flipInto(rhi, sourceView, targetView) {
  const device = rhi.device;
  const { layout, pipeline } = flipPipeline(device);
  const encoder = device.createCommandEncoder({ label: 'probe-flip' });
  const pass = encoder.beginRenderPass({
    label: 'probe-flip',
    colorAttachments: [{ view: targetView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout, entries: [{ binding: 0, resource: sourceView }] }));
  pass.draw(3);
  pass.end();
  rhi.queue.submit([encoder.finish()]);
}

/** One per device: the layout is part of the pipeline's identity. */
const flipPipelines = new WeakMap();
function flipPipeline(device) {
  let built = flipPipelines.get(device);
  if (!built) {
    const layout = device.createBindGroupLayout({
      label: 'probe-flip',
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
    });
    const pipeline = sharedPipelines(device).get({
      label: 'probe-flip',
      layout: createPipelineLayout(device, { 0: layout }, 'probe-flip'),
      shader: compileShaderSync(device, FLIP_SHADER, 'probe-flip.wgsl'),
      targets: [{ format: 'rgba16float' }],
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depth: null,
    });
    built = { layout, pipeline };
    flipPipelines.set(device, built);
  }
  return built;
}
