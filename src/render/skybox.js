// Skybox pass: draws the environment cubemap behind everything.
//
// A renderer with image-based lighting and no visible environment is a renderer
// you cannot debug -- there is no way to tell
// a correct reflection from a plausible one when the thing being reflected is
// invisible.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { DEPTH_FORMAT } from '../rhi/device.js';
import { HDR_FORMAT } from './post.js';
import { createBuffer } from '../rhi/buffer.js';
import { FOG_WGSL } from './fog.js';

/** right, up, forward, lens, then the eye and the fog's three: eight vec4s. */
const PARAMS_BYTES = 128;

const SKYBOX_SHADER = /* wgsl */ `
struct Params {
  right   : vec4<f32>,
  up      : vec4<f32>,
  forward : vec4<f32>,
  lens    : vec4<f32>,      // x = tan(fovY/2), y = aspect, z = exposure
  eye     : vec4<f32>,
  fog     : vec4<f32>,      // as the frame's; see render/fog.js
  fogAlbedo : vec4<f32>,
  fogLight  : vec4<f32>,
};

@group(0) @binding(0) var<uniform> params      : Params;
@group(0) @binding(1) var          environment : texture_cube<f32>;
@group(0) @binding(2) var          envSampler  : sampler;
@group(0) @binding(3) var          irradiance  : texture_cube<f32>;
${FOG_WGSL}

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0)       ndc      : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32) -> VertexOut {
  var out : VertexOut;
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  let clip = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
  out.ndc = clip;
  out.position = vec4<f32>(clip, 0.0, 1.0);
  return out;
}

@fragment
fn fs(v : VertexOut) -> @location(0) vec4<f32> {
  // The view ray straight from the camera basis. No inverse projection, and it
  // works unchanged with an infinite far plane.
  let direction = normalize(
      params.forward.xyz
    + params.right.xyz * (v.ndc.x * params.lens.x * params.lens.y)
    + params.up.xyz    * (v.ndc.y * params.lens.x)
  );
  // Linear HDR: the sun disc really is 60x white, and the post stack is what
  // brings it into range.
  let sky = textureSampleLevel(environment, envSampler, direction, 0.0).rgb;
  if (params.fog.x <= 0.0) { return vec4<f32>(sky, 1.0); }
  // A ray to infinity: all fog, unless it rises out of a thinning layer.
  let through = exp(-fogDepth(params.fog, params.eye.xyz, direction, -1.0));
  let inscatter = params.fogAlbedo.rgb * fogMeanRadiance(irradiance, envSampler) + params.fogLight.rgb;
  return vec4<f32>(sky * through + inscatter * (1.0 - through), 1.0);
}
`;

export class SkyboxPass {
  /** `ambientFormat`: the forward passes' second target, when ambient occlusion is on. */
  static async create(rhi, pipelines, ambientFormat = null) {
    const shader = await compileShader(rhi.device, SKYBOX_SHADER, 'skybox.wgsl');

    const layout = rhi.device.createBindGroupLayout({
      label: 'skybox',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
      ],
    });

    const buffer = createBuffer(rhi, {
      label: 'skybox', size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const descriptor = {
      label: 'skybox',
      layout: createPipelineLayout(rhi.device, { 0: layout }, 'skybox'),
      shader,
      // The sky has no ambient term to occlude: its second target, when there
      // is one, keeps the zero it was cleared to.
      targets: ambientFormat === null
        ? [{ format: HDR_FORMAT }]
        : [{ format: HDR_FORMAT }, { format: ambientFormat, writeMask: 0 }],
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      // Drawn first, writing no depth and testing nothing, so everything else
      // draws over it normally. Simpler to reason about than fighting the
      // reverse-Z clear value for a far-plane trick.
      depth: { format: DEPTH_FORMAT, depthCompare: 'always', depthWriteEnabled: false },
    };
    await pipelines.warm([descriptor]);

    return new SkyboxPass(rhi, pipelines, descriptor, layout, buffer);
  }

  constructor(rhi, pipelines, descriptor, layout, buffer) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    this.descriptor = descriptor;
    this.layout = layout;
    this.buffer = buffer;
    this.data = new Float32Array(PARAMS_BYTES / 4);
    // Keyed by environment: the real one arrives per frame in draw().
    this._bindGroups = new WeakMap();
  }

  _bindGroupFor(environment) {
    let bindGroup = this._bindGroups.get(environment);
    if (!bindGroup) {
      bindGroup = this.rhi.device.createBindGroup({
        label: 'skybox',
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.buffer } },
          { binding: 1, resource: environment.environmentView },
          { binding: 2, resource: environment.sampler },
          { binding: 3, resource: environment.irradianceView },
        ],
      });
      this._bindGroups.set(environment, bindGroup);
    }
    return bindGroup;
  }

  /** `fog`: the frame's 12 fog floats (see packFog), or null. */
  update(camera, exposure, fog = null) {
    // The view matrix's rows are the camera basis expressed in world space.
    const v = camera.view;
    this.data[0] = v[0]; this.data[1] = v[4]; this.data[2] = v[8];
    this.data[4] = v[1]; this.data[5] = v[5]; this.data[6] = v[9];
    this.data[8] = -v[2]; this.data[9] = -v[6]; this.data[10] = -v[10];
    this.data[12] = Math.tan(camera.fovY * 0.5);
    this.data[13] = camera.aspect;
    this.data[14] = exposure;
    this.data.set(camera.position, 16);
    if (fog === null) this.data.fill(0, 20, 32);
    else this.data.set(fog, 20);
    this.rhi.queue.writeBuffer(this.buffer, 0, this.data);
  }

  /** The parameter buffer is the only GPU object this owns. */
  destroy() {
    this.buffer.destroy();
  }

  draw(pass, environment) {
    pass.setPipeline(this.pipelines.get(this.descriptor));
    pass.setBindGroup(0, this._bindGroupFor(environment));
    pass.draw(3);
  }
}
