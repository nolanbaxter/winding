// Textures, mip chains and cubemaps.
//
// Two things here are easy to get wrong and invisible when you do:
//
// COLOR SPACE. Base colour and emissive maps hold sRGB-encoded bytes; normal,
// occlusion, roughness and metallic maps hold raw linear numbers. Loading both
// with the same format means either doing lighting on gamma-encoded colour or
// gamma-decoding a normal vector. Neither errors, both look "slightly off" in a
// way that is very hard to trace, so the format is chosen per usage and the
// caller has to say which it is.
//
// MIP FILTERING. Mips must be averaged in LINEAR light. Rendering into an -srgb
// view gets that for free: the hardware decodes on sample, blends linearly, and
// re-encodes on write.

import { DEBUG, assert } from '../core/assert.js';
import { compileShaderSync } from './shader.js';
import { sharedPipelines } from './pipeline.js';
import { createPipelineLayout } from './bindgroups.js';

/** Fullscreen-triangle blit, used to build each mip from the level above it. */
const MIP_SHADER = /* wgsl */ `
@group(0) @binding(0) var source : texture_2d<f32>;
@group(0) @binding(1) var samp   : sampler;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0)       uv       : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32) -> VertexOut {
  // One oversized triangle rather than two triangles: no vertex buffer, no
  // index buffer, and no seam along the quad's diagonal.
  var out : VertexOut;
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  out.uv = vec2<f32>(x, y);
  out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs(v : VertexOut) -> @location(0) vec4<f32> {
  return textureSample(source, samp, v.uv);
}
`;

export function mipLevelCountFor(width, height) {
  return Math.floor(Math.log2(Math.max(width, height))) + 1;
}

/**
 * Any texture, checked against the device first. WebGPU does not throw for
 * an oversized descriptor: it hands back an invalid texture, and every pass
 * that touches it becomes a silent no-op. Every texture made outside this
 * folder comes through here, so the check is in one place.
 */
export function createTexture(rhi, descriptor) {
  const size = descriptor.size;
  const [width, height = 1, layers = 1] = Array.isArray(size)
    ? size
    : [size.width, size.height, size.depthOrArrayLayers];
  const limits = rhi.limits ?? {};
  const max = limits.maxTextureDimension2D ?? Infinity;
  if (width > max || height > max) {
    throw new RangeError(`createTexture: ${descriptor.label ?? 'texture'} is ${width}x${height}, past this device's ${max}`);
  }
  if (layers > (limits.maxTextureArrayLayers ?? Infinity)) {
    throw new RangeError(`createTexture: ${descriptor.label ?? 'texture'} has ${layers} layers, past this device's ${limits.maxTextureArrayLayers}`);
  }
  return rhi.device.createTexture(descriptor);
}

/**
 * @param srgb true for colour data authored in sRGB (base colour, emissive),
 *             false for data that is already linear (normal, ORM, depth-like).
 */
export function createTexture2D(rhi, {
  width, height, srgb = false, mipmapped = false, label,
  usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
}) {
  // Unconditional, because the failure it prevents is silent. createTexture
  // does not throw for an oversized descriptor -- it raises a validation error
  // on the device's error scope and hands back an INVALID texture, after which
  // the upload, the mip generation and every draw that binds it are no-ops.
  // The result is a material that renders with no texture and no message. The
  // limit is 8192 on a lot of hardware and scanned or film assets do exceed it.
  const max = rhi.limits.maxTextureDimension2D;
  if (width > max || height > max) {
    throw new Error(
      `createTexture2D: ${label ?? 'texture'} is ${width}x${height}, past this device's ${max} limit`,
    );
  }

  const mipLevelCount = mipmapped ? mipLevelCountFor(width, height) : 1;
  return rhi.device.createTexture({
    label,
    size: [width, height, 1],
    format: srgb ? 'rgba8unorm-srgb' : 'rgba8unorm',
    mipLevelCount,
    // RENDER_ATTACHMENT is required to generate mips by rendering into them.
    usage: mipmapped ? usage | GPUTextureUsage.RENDER_ATTACHMENT : usage,
  });
}

/** Upload an ImageBitmap (or canvas) into mip 0. */
export function uploadImage(rhi, texture, source) {
  rhi.queue.copyExternalImageToTexture(
    { source, flipY: false },
    { texture, premultipliedAlpha: false },
    [source.width, source.height],
  );
}

/**
 * Fill mips 1..n by successively halving. Each level is a render pass sampling
 * the level above with a linear filter, which is a box filter -- good enough
 * for everything except normal maps, where it slowly flattens the surface.
 *
 * ponytail: box filter. A Kaiser or tent filter is better for detail
 * preservation, and normal maps really want renormalization per level. Both are
 * a different fragment shader in this same loop.
 */
export function generateMipmaps(rhi, texture) {
  if (texture.mipLevelCount <= 1) return;

  const { pipeline, layout } = mipPipelineFor(rhi, texture.format);
  const sampler = linearSampler(rhi);
  const encoder = rhi.device.createCommandEncoder({ label: 'mipmaps' });

  // Every array layer, which is what makes this work for a cubemap: its six
  // faces are six layers, and reducing only layer 0 would leave five of them
  // undefined at every level but the base. Each face is filtered on its own --
  // a box filter across a cube seam is not defined without neighbour lookups,
  // and the error is confined to one texel at the edge.
  const layers = texture.depthOrArrayLayers;

  for (let layer = 0; layer < layers; layer++) {
    for (let level = 1; level < texture.mipLevelCount; level++) {
      const sourceView = texture.createView({
        baseMipLevel: level - 1, mipLevelCount: 1,
        baseArrayLayer: layer, arrayLayerCount: 1, dimension: '2d',
      });
      const targetView = texture.createView({
        baseMipLevel: level, mipLevelCount: 1,
        baseArrayLayer: layer, arrayLayerCount: 1, dimension: '2d',
      });

      const bindGroup = rhi.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: sourceView },
          { binding: 1, resource: sampler },
        ],
      });

      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: targetView, loadOp: 'clear', storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    }
  }

  rhi.queue.submit([encoder.finish()]);
}

/**
 * The mip pipeline for a format, and the layout its bind groups use. One per
 * (device, format), from the shared cache: building it per texture would be
 * the pipeline-creation stall this engine spends a whole cache avoiding. An
 * explicit layout, not 'auto', so it is a plain descriptor like the rest.
 */
export function mipPipelineFor(rhi, format) {
  const device = rhi.device;
  const { shader, layout, pipelineLayout } = cached(rhi, 'mip', () => {
    const layout = device.createBindGroupLayout({
      label: 'mipmap',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    return {
      shader: compileShaderSync(device, MIP_SHADER, 'mipmap.wgsl'),
      layout,
      pipelineLayout: createPipelineLayout(device, { 0: layout }, 'mipmap'),
    };
  });
  const pipeline = sharedPipelines(device).get({
    label: `mipmap:${format}`,
    layout: pipelineLayout,
    shader,
    targets: [{ format }],
    // The full-screen triangle winds clockwise, and the cache culls back faces
    // unless told otherwise.
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depth: null,
  });
  return { pipeline, layout };
}

// Per-device cache for anything created once and shared: samplers, and the 1x1
// default textures. Named for what it mostly holds; defaultTextures() puts
// GPUTextures in it too, so texture lifetime IS tied to this map.
const perDevice = new WeakMap();

function cached(rhi, key, make) {
  let byKey = perDevice.get(rhi.device);
  if (!byKey) {
    byKey = new Map();
    perDevice.set(rhi.device, byKey);
  }
  let value = byKey.get(key);
  if (value === undefined) {
    value = make();
    byKey.set(key, value);
  }
  return value;
}

export function linearSampler(rhi) {
  return cached(rhi, 'linear', () => rhi.device.createSampler({
    label: 'linear',
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    addressModeU: 'repeat', addressModeV: 'repeat',
    // 16x is the common hardware maximum. Queried limits do not expose it, so
    // the driver silently clamps -- which is the one case where asking for more
    // than you can have is safe.
    maxAnisotropy: 16,
  }));
}

/** Clamped, non-anisotropic. For cubemaps and full-screen work. */
export function clampSampler(rhi) {
  return cached(rhi, 'clamp', () => rhi.device.createSampler({
    label: 'clamp',
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge',
  }));
}

/**
 * 1x1 stand-ins so a material with no texture still has something to bind.
 *
 * This is why there are no shader variants for texture presence: every
 * material binds four textures, absent ones point at these, and the factors in
 * the material uniform do the rest. Sampling a 1x1 texture is free, and the
 * alternative is a pipeline variant per combination of present maps.
 */
export function defaultTextures(rhi) {
  return cached(rhi, 'defaults', () => ({
    // Every colour slot that can be absent defaults to white, because glTF
    // says an absent texture reads as 1.0 and the material factors do the
    // scaling. There is deliberately no black default: the one binding that
    // used it -- emissive -- was multiplying its own factor away.
    white: solidTexture(rhi, [255, 255, 255, 255], true, 'default-white'),
    // (0.5, 0.5, 1.0) decodes to a normal of (0, 0, 1): no perturbation.
    flatNormal: solidTexture(rhi, [128, 128, 255, 255], false, 'default-normal'),
    // Occlusion 1, roughness 1, metallic 1 -- glTF's channel packing is
    // occlusion/roughness/metallic in R/G/B, and the factors scale it.
    orm: solidTexture(rhi, [255, 255, 255, 255], false, 'default-orm'),
  }));
}

function solidTexture(rhi, rgba, srgb, label) {
  const texture = rhi.device.createTexture({
    label,
    size: [1, 1, 1],
    format: srgb ? 'rgba8unorm-srgb' : 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  rhi.queue.writeTexture({ texture }, new Uint8Array(rgba), { bytesPerRow: 4 }, [1, 1, 1]);
  return texture;
}

/**
 * Cubemap. `mipLevelCount` above 1 is what the prefiltered specular chain
 * needs: each level holds the environment convolved for a rougher surface.
 */
export function createCubemap(rhi, { size, mipLevelCount = 1, format = 'rgba16float', label }) {
  if (DEBUG) assert(size > 0 && (size & (size - 1)) === 0, 'cubemap size should be a power of two');
  return rhi.device.createTexture({
    label,
    size: [size, size, 6],
    format,
    mipLevelCount,
    usage: GPUTextureUsage.TEXTURE_BINDING
      | GPUTextureUsage.RENDER_ATTACHMENT
      | GPUTextureUsage.COPY_DST
      // Copied from: a reflection probe's prefilter goes into its scene's array.
      | GPUTextureUsage.COPY_SRC,
  });
}

export function cubeView(texture, label) {
  return texture.createView({ dimension: 'cube', label });
}

/** A single face at a single mip, as a render target. */
export function cubeFaceView(texture, face, mipLevel = 0) {
  return texture.createView({
    dimension: '2d',
    baseArrayLayer: face,
    arrayLayerCount: 1,
    baseMipLevel: mipLevel,
    mipLevelCount: 1,
  });
}
