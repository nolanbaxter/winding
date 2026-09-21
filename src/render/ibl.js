// Image-based lighting.
//
// Ambient light is an integral over every incoming direction, which is far too
// expensive per pixel. The standard trick (Karis, "split sum") is to precompute
// it into two cubemaps at load time and reduce the shader to two texture reads:
//
//   irradiance   cosine-convolved environment -> diffuse ambient, one sample
//   prefiltered  GGX-convolved, one mip per roughness -> specular ambient
//
// The third piece, the BRDF integration term, is normally a lookup texture.
// Here it is an analytic polynomial (see envBRDFApprox in brdf.js), which
// removes a texture and a whole generation pass for an error nobody can see.
//
// Everything happens once, at startup. Nothing here runs per frame.

import { BRDF_WGSL } from './shaders/brdf.js';
import { createCubemap, cubeView, cubeFaceView, clampSampler } from '../rhi/texture.js';
import { compileShaderSync } from '../rhi/shader.js';

const FACE_COUNT = 6;
const PARAMS_BYTES = 16;   // face:u32, roughness:f32, sampleCount:u32, pad

// 64 samples is enough for a smooth result at these resolutions; the
// low-discrepancy Hammersley sequence is what makes that true, since uniform
// random sampling would still be visibly noisy at 10x the count.
const PREFILTER_SAMPLES = 64;
const IRRADIANCE_SAMPLES = 64;

const SKY_SHADER = /* wgsl */ `
${BRDF_WGSL}

struct Params {
  face        : u32,
  roughness   : f32,
  sampleCount : u32,
  padding     : u32,
};
@group(0) @binding(0) var<uniform> params : Params;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0)       uv       : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32) -> VertexOut {
  var out : VertexOut;
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  out.uv = vec2<f32>(x, y);
  out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs(v : VertexOut) -> @location(0) vec4<f32> {
  return vec4<f32>(skyRadiance(cubeDirection(params.face, v.uv)), 1.0);
}
`;

const CONVOLVE_SHADER = /* wgsl */ `
${BRDF_WGSL}

struct Params {
  face        : u32,
  roughness   : f32,
  sampleCount : u32,
  padding     : u32,
};
@group(0) @binding(0) var<uniform> params      : Params;
@group(0) @binding(1) var          environment : texture_cube<f32>;
@group(0) @binding(2) var          envSampler  : sampler;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0)       uv       : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32) -> VertexOut {
  var out : VertexOut;
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  out.uv = vec2<f32>(x, y);
  out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

// Cosine-weighted convolution: the diffuse response of a Lambertian surface
// facing this direction. Sampled rather than marched over the full hemisphere,
// using the same Hammersley set as the specular pass.
@fragment
fn fsIrradiance(v : VertexOut) -> @location(0) vec4<f32> {
  let n = cubeDirection(params.face, v.uv);

  let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.999);
  let tangentX = normalize(cross(up, n));
  let tangentY = cross(n, tangentX);

  var total = vec3<f32>(0.0);
  for (var i = 0u; i < params.sampleCount; i = i + 1u) {
    let xi = hammersley(i, params.sampleCount);

    // Cosine-weighted hemisphere sample. Drawing proportional to cos means the
    // cosine term and the pdf cancel, so the estimator is a plain average.
    let phi = 2.0 * PI * xi.x;
    let cosTheta = sqrt(1.0 - xi.y);
    let sinTheta = sqrt(xi.y);

    let direction = tangentX * (cos(phi) * sinTheta)
                  + tangentY * (sin(phi) * sinTheta)
                  + n * cosTheta;

    total = total + textureSampleLevel(environment, envSampler, direction, 0.0).rgb;
  }

  return vec4<f32>(total / f32(params.sampleCount), 1.0);
}

// GGX convolution for one roughness. The usual N = V = R simplification: it
// makes the result independent of view direction, which is what lets the answer
// live in a cubemap at all. The cost is that stretched grazing-angle
// reflections are lost -- every real-time engine accepts this.
@fragment
fn fsPrefilter(v : VertexOut) -> @location(0) vec4<f32> {
  let n = cubeDirection(params.face, v.uv);
  let r = n;
  let view = n;

  var total = vec3<f32>(0.0);
  var totalWeight = 0.0;

  for (var i = 0u; i < params.sampleCount; i = i + 1u) {
    let xi = hammersley(i, params.sampleCount);
    let h = importanceSampleGGX(xi, n, params.roughness);
    let l = normalize(2.0 * dot(view, h) * h - view);

    let NoL = dot(n, l);
    if (NoL > 0.0) {
      // Weighting by NoL rather than averaging flat is a small cheat that
      // visibly reduces the bright fringe at the edge of rough reflections.
      total = total + textureSampleLevel(environment, envSampler, l, 0.0).rgb * NoL;
      totalWeight = totalWeight + NoL;
    }
  }

  return vec4<f32>(total / max(totalWeight, 1e-4), 1.0);
}
`;

/**
 * A prebaked lighting environment.
 *
 * Generated from the procedural sky in brdf.js. Swapping in a loaded HDR map
 * means replacing the sky pass with an equirectangular blit -- the irradiance
 * and prefilter passes below do not change.
 */
export class Environment {
  constructor(rhi, { size = 128, irradianceSize = 32, prefilterMips = 6, label = 'env' } = {}) {
    this.rhi = rhi;
    this.prefilterMips = prefilterMips;

    // rgba16float, not rgba8unorm: the sun is 60x brighter than white and
    // clamping it to 1.0 would flatten every reflection in the scene.
    this.environment = createCubemap(rhi, { size, format: 'rgba16float', label: `${label}-sky` });
    this.irradiance = createCubemap(rhi, {
      size: irradianceSize, format: 'rgba16float', label: `${label}-irradiance`,
    });
    this.prefiltered = createCubemap(rhi, {
      size, mipLevelCount: prefilterMips, format: 'rgba16float', label: `${label}-prefiltered`,
    });

    this.environmentView = cubeView(this.environment, `${label}-sky`);
    this.irradianceView = cubeView(this.irradiance, `${label}-irradiance`);
    this.prefilteredView = cubeView(this.prefiltered, `${label}-prefiltered`);
    this.sampler = clampSampler(rhi);

    this._bake(size, irradianceSize, prefilterMips);
  }

  _bake(size, irradianceSize, prefilterMips) {
    const rhi = this.rhi;
    const device = rhi.device;

    // Every pass needs different params, and queue.writeBuffer cannot be
    // interleaved with render passes inside one encoder. So: one buffer, all
    // the params written up front, a dynamic offset per pass.
    const alignment = rhi.limits.minUniformBufferOffsetAlignment;
    const passCount = FACE_COUNT * (2 + prefilterMips);
    const params = new ArrayBuffer(alignment * passCount);
    const paramsBuffer = device.createBuffer({
      label: 'ibl-params',
      size: params.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let slot = 0;
    const writeParams = (face, roughness, sampleCount) => {
      const offset = slot++ * alignment;
      const u32 = new Uint32Array(params, offset, 4);
      const f32 = new Float32Array(params, offset, 4);
      u32[0] = face;
      f32[1] = roughness;
      u32[2] = sampleCount;
      u32[3] = 0;
      return offset;
    };

    const skyOffsets = [];
    for (let face = 0; face < FACE_COUNT; face++) skyOffsets.push(writeParams(face, 0, 0));

    const irradianceOffsets = [];
    for (let face = 0; face < FACE_COUNT; face++) {
      irradianceOffsets.push(writeParams(face, 0, IRRADIANCE_SAMPLES));
    }

    const prefilterOffsets = [];
    for (let mip = 0; mip < prefilterMips; mip++) {
      // Mip 0 is a mirror; the last mip is fully rough. Linear in perceptual
      // roughness, which is what the surface shader will index it with.
      const roughness = prefilterMips > 1 ? mip / (prefilterMips - 1) : 0;
      const row = [];
      for (let face = 0; face < FACE_COUNT; face++) {
        row.push(writeParams(face, roughness, PREFILTER_SAMPLES));
      }
      prefilterOffsets.push(row);
    }

    rhi.queue.writeBuffer(paramsBuffer, 0, params);

    // ---- pipelines -------------------------------------------------------

    const paramsLayout = device.createBindGroupLayout({
      label: 'ibl-params',
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_BYTES },
      }],
    });
    const convolveLayout = device.createBindGroupLayout({
      label: 'ibl-convolve',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_BYTES },
        },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    const skyModule = compileShaderSync(device, SKY_SHADER, 'ibl-sky.wgsl').module;
    const convolveModule = compileShaderSync(device, CONVOLVE_SHADER, 'ibl-convolve.wgsl').module;
    const target = [{ format: 'rgba16float' }];

    const skyPipeline = device.createRenderPipeline({
      label: 'ibl-sky',
      layout: device.createPipelineLayout({ bindGroupLayouts: [paramsLayout] }),
      vertex: { module: skyModule, entryPoint: 'vs' },
      fragment: { module: skyModule, entryPoint: 'fs', targets: target },
      primitive: { topology: 'triangle-list' },
    });

    const convolvePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [convolveLayout] });
    const irradiancePipeline = device.createRenderPipeline({
      label: 'ibl-irradiance',
      layout: convolvePipelineLayout,
      vertex: { module: convolveModule, entryPoint: 'vs' },
      fragment: { module: convolveModule, entryPoint: 'fsIrradiance', targets: target },
      primitive: { topology: 'triangle-list' },
    });
    const prefilterPipeline = device.createRenderPipeline({
      label: 'ibl-prefilter',
      layout: convolvePipelineLayout,
      vertex: { module: convolveModule, entryPoint: 'vs' },
      fragment: { module: convolveModule, entryPoint: 'fsPrefilter', targets: target },
      primitive: { topology: 'triangle-list' },
    });

    const paramsBindGroup = device.createBindGroup({
      layout: paramsLayout,
      entries: [{ binding: 0, resource: { buffer: paramsBuffer, size: PARAMS_BYTES } }],
    });
    const convolveBindGroup = device.createBindGroup({
      layout: convolveLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer, size: PARAMS_BYTES } },
        { binding: 1, resource: this.environmentView },
        { binding: 2, resource: this.sampler },
      ],
    });

    // ---- passes ----------------------------------------------------------

    const encoder = device.createCommandEncoder({ label: 'ibl-bake' });

    const facePass = (pipeline, bindGroup, offset, view, label) => {
      const pass = encoder.beginRenderPass({
        label,
        colorAttachments: [{
          view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup, [offset]);
      pass.draw(3);
      pass.end();
    };

    for (let face = 0; face < FACE_COUNT; face++) {
      facePass(skyPipeline, paramsBindGroup, skyOffsets[face],
        cubeFaceView(this.environment, face), `sky:${face}`);
    }

    // The convolution passes read `environment`, which the passes above wrote.
    // Same encoder, same queue, so ordering is guaranteed -- but only because
    // they are separate passes; a texture cannot be read and written in one.
    for (let face = 0; face < FACE_COUNT; face++) {
      facePass(irradiancePipeline, convolveBindGroup, irradianceOffsets[face],
        cubeFaceView(this.irradiance, face), `irradiance:${face}`);
    }

    for (let mip = 0; mip < prefilterMips; mip++) {
      for (let face = 0; face < FACE_COUNT; face++) {
        facePass(prefilterPipeline, convolveBindGroup, prefilterOffsets[mip][face],
          cubeFaceView(this.prefiltered, face, mip), `prefilter:${mip}:${face}`);
      }
    }

    rhi.queue.submit([encoder.finish()]);
    paramsBuffer.destroy();

    this.passCount = passCount;
  }

  destroy() {
    this.environment.destroy();
    this.irradiance.destroy();
    this.prefiltered.destroy();
  }
}
