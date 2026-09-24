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
import {
  createCubemap, cubeView, cubeFaceView, clampSampler, generateMipmaps, mipLevelCountFor,
} from '../rhi/texture.js';
import { compileShaderSync } from '../rhi/shader.js';

const FACE_COUNT = 6;
const PARAMS_BYTES = 16;   // face:u32, roughness:f32, sampleCount:u32, envSize:f32

// 64 samples is enough for a smooth result at these resolutions; the
// low-discrepancy Hammersley sequence is what makes that true, since uniform
// random sampling would still be visibly noisy at 10x the count.
const PREFILTER_SAMPLES = 64;
const IRRADIANCE_SAMPLES = 64;

/**
 * The sky the environment is baked from, and therefore also the background.
 *
 * Linear HDR. The sun is far brighter than 1.0, which is the point of keeping
 * the cubemap in a float format -- clamp it to 1 and every reflection in the
 * scene flattens.
 *
 * `sun` points TOWARD the sun, which is the opposite of `scene.sun.direction`
 * (the direction light travels). They are separate on purpose: this one is
 * baked once into a cubemap and the other is a per-frame analytic light. If
 * you move one and want the disc to stay under the highlight, move both.
 */
export const DEFAULT_SKY = Object.freeze({
  ground: Object.freeze([0.10, 0.09, 0.08]),
  horizon: Object.freeze([0.62, 0.66, 0.74]),
  zenith: Object.freeze([0.16, 0.30, 0.60]),
  sun: Object.freeze([0.35, 0.55, 0.45]),
  sunColor: Object.freeze([1.0, 0.93, 0.80]),
  /** Radiance of the disc itself. Zero removes the sun from the sky. */
  sunIntensity: 60,
  /** How far the halo around it reaches. Zero removes it. */
  glow: 0.5,
});

const vec3 = (v) => `vec3<f32>(${v[0]}, ${v[1]}, ${v[2]})`;

/**
 * Substituted into the source rather than uploaded as a uniform, because these
 * are BAKE-TIME constants: the cubemap is generated once when an Environment
 * is constructed and read every frame thereafter. A uniform would add a buffer
 * and a binding to describe values that never change after the bake.
 */
function skyShader(sky) {
  return /* wgsl */ `
${BRDF_WGSL}

fn skyRadiance(dir : vec3<f32>) -> vec3<f32> {
  let sunDirection = normalize(${vec3(sky.sun)});
  let height = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

  let ground  = ${vec3(sky.ground)};
  let horizon = ${vec3(sky.horizon)};
  let zenith  = ${vec3(sky.zenith)};

  var color = mix(horizon, zenith, smoothstep(0.5, 1.0, height));
  color = mix(ground, color, smoothstep(0.47, 0.53, height));

  let toSun = max(dot(dir, sunDirection), 0.0);
  let disc  = pow(toSun, 900.0) * ${sky.sunIntensity};
  let glow  = pow(toSun, 8.0) * ${sky.glow};
  return color + ${vec3(sky.sunColor)} * (disc + glow);
}

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
}

const CONVOLVE_SHADER = /* wgsl */ `
${BRDF_WGSL}

struct Params {
  face        : u32,
  roughness   : f32,
  sampleCount : u32,
  // Edge length of ONE face of the source cubemap, in texels. Needed for the
  // solid angle a single texel covers, which is what sets the mip below.
  envSize     : f32,
};

/**
 * Which source mip a sample should read, from the solid angle it represents.
 *
 * Karis, "Real Shading in Unreal Engine 4", the section on solving the bright
 * dots. A fixed sample count over a full-resolution environment undersamples
 * it: each sample stands for a cone of directions but reads a single texel, so
 * a source with its energy concentrated in a few texels -- a sun, and every
 * captured HDR -- either gets hit and blows the estimate up, or gets missed
 * and vanishes. Neighbouring output texels make different choices frame to
 * frame, which is the speckle that swims across rough metal.
 *
 * Reading a mip whose texels cover the sample's own solid angle turns the
 * point sample into an average of what the cone actually contains.
 *
 *   saTexel  = 4pi / (6 * size * size)      one texel of the source cube
 *   saSample = 1 / (sampleCount * pdf)      what one sample stands for
 *   level    = 0.5 * log2(saSample / saTexel)
 */
fn sampleMip(pdf : f32, sampleCount : u32, envSize : f32) -> f32 {
  let saTexel = 4.0 * PI / (6.0 * envSize * envSize);
  let saSample = 1.0 / (f32(sampleCount) * max(pdf, 1e-6));
  return max(0.5 * log2(saSample / saTexel), 0.0);
}
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

    // pdf of a cosine-weighted hemisphere sample is cos(theta)/PI, and here
    // cos(theta) IS cosTheta -- the sample was drawn about n.
    let mip = sampleMip(cosTheta / PI, params.sampleCount, params.envSize);
    total = total + textureSampleLevel(environment, envSampler, direction, mip).rgb;
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
      // The GGX sample pdf, in the same half-vector form importanceSampleGGX
      // drew from: D(h) * NoH / (4 * VoH). With N = V = R the two dots are the
      // same, which is why only one appears.
      let NoH = max(dot(n, h), 0.0);
      let VoH = max(dot(view, h), 0.0);
      let pdf = distributionGGX(NoH, params.roughness) * NoH / (4.0 * max(VoH, 1e-4));
      // Roughness 0 is a mirror: one direction, no cone, so no blur.
      let mip = select(sampleMip(pdf, params.sampleCount, params.envSize), 0.0, params.roughness == 0.0);

      // Weighting by NoL rather than averaging flat is a small cheat that
      // visibly reduces the bright fringe at the edge of rough reflections.
      total = total + textureSampleLevel(environment, envSampler, l, mip).rgb * NoL;
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
  constructor(rhi, {
    size = 128, irradianceSize = 32, prefilterMips = 6, label = 'env', sky = null,
  } = {}) {
    this.rhi = rhi;
    const max = rhi.limits.maxTextureDimension2D;
    if (size > max || irradianceSize > max) {
      throw new RangeError(`Environment: size ${Math.max(size, irradianceSize)} is past this device's ${max}`);
    }
    // No more levels than the cube has. Asking for more made an invalid
    // texture -- size 16 has five -- and every frame after it was black.
    prefilterMips = Math.min(prefilterMips, mipLevelCountFor(size, size));
    this.prefilterMips = prefilterMips;
    /** What this environment was baked from. See DEFAULT_SKY. */
    this.sky = { ...DEFAULT_SKY, ...(sky ?? {}) };

    // rgba16float, not rgba8unorm: the sun is 60x brighter than white and
    // clamping it to 1.0 would flatten every reflection in the scene.
    // A mip chain, because the convolutions below read DOWN it: a sample that
    // stands for a wide cone reads a level whose texels cover that cone. The
    // skybox samples level 0 and is unaffected.
    this.environment = createCubemap(rhi, {
      size, mipLevelCount: mipLevelCountFor(size, size),
      format: 'rgba16float', label: `${label}-sky`,
    });
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
      f32[3] = size;
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

    const skyModule = compileShaderSync(device, skyShader(this.sky), 'ibl-sky.wgsl').module;
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

    // Submitted before the convolutions, because generateMipmaps runs on its
    // own encoder and the convolutions read the levels it produces. Queue
    // order is what sequences the three.
    rhi.queue.submit([encoder.finish()]);
    generateMipmaps(rhi, this.environment);

    const convolve = device.createCommandEncoder({ label: 'ibl-convolve-passes' });
    const convolveFace = (pipeline, bindGroup, offset, view, passLabel) => {
      const pass = convolve.beginRenderPass({
        label: passLabel,
        colorAttachments: [{
          view, loadOp: 'clear', storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup, [offset]);
      pass.draw(3);
      pass.end();
    };

    for (let face = 0; face < FACE_COUNT; face++) {
      convolveFace(irradiancePipeline, convolveBindGroup, irradianceOffsets[face],
        cubeFaceView(this.irradiance, face), `irradiance:${face}`);
    }

    for (let mip = 0; mip < prefilterMips; mip++) {
      for (let face = 0; face < FACE_COUNT; face++) {
        convolveFace(prefilterPipeline, convolveBindGroup, prefilterOffsets[mip][face],
          cubeFaceView(this.prefiltered, face, mip), `prefilter:${mip}:${face}`);
      }
    }

    rhi.queue.submit([convolve.finish()]);
    paramsBuffer.destroy();

    this.passCount = passCount;
  }

  destroy() {
    this.environment.destroy();
    this.irradiance.destroy();
    this.prefiltered.destroy();
  }
}
