// Decals (Scene.addDecal) for the surface shader: each one's box, and every
// decal texture in one array, since a shader cannot index separate textures.
//
// The surface shader paints them into the base colour before lighting, so a
// decal is lit, shadowed and fogged exactly as the surface under it. A
// fragment tests only the decals whose bounding spheres reach its light
// cluster (render/clustered.js), and the test is compiled out of every
// pipeline while no scene has decals (DECALS in pbr.js).

import { createBuffer } from '../rhi/buffer.js';
import { createTexture, mipLevelCountFor, mipPipelineFor, generateMipmaps, clampSampler } from '../rhi/texture.js';
import { mat4Create, mat4Invert } from '../core/math/mat4.js';
import { handleIndex } from '../core/handle.js';

/** worldToDecal (16), colour (4), facing and layer (4). */
export const DECAL_FLOATS = 24;
const FORMAT = 'rgba8unorm-srgb';

/**
 * Pack a scene's decals for the shader, in the order they were added, and
 * say which texture each reads. worldToDecal maps the decal's box to
 * [-1, 1] on every axis; facing is its +Z in world space, the way a surface
 * it paints must face. `spheres` gets each box's centre and the radius of
 * its farthest corner, for clustering.
 */
export function packDecals(scene, out, layers, spheres) {
  const world = scene.transforms.world;
  const inverse = mat4Create();
  let k = 0;
  for (const [entity, decal] of scene.decals) {
    const m = handleIndex(entity) * 16;
    mat4Invert(inverse, world.subarray(m, m + 16));
    const o = k * DECAL_FLOATS;
    for (let column = 0; column < 4; column++) {
      for (let row = 0; row < 3; row++) {
        out[o + column * 4 + row] = inverse[column * 4 + row] * (2 / decal.size[row]);
      }
      out[o + column * 4 + 3] = inverse[column * 4 + 3];
    }
    out.set(decal.color, o + 16);
    const zx = world[m + 8];
    const zy = world[m + 9];
    const zz = world[m + 10];
    const length = Math.hypot(zx, zy, zz) || 1;
    out[o + 20] = zx / length;
    out[o + 21] = zy / length;
    out[o + 22] = zz / length;
    out[o + 23] = layers.get(decal.texture);
    // The box's half-axes are the node's axes times half its size. Corners
    // come in opposite pairs, so four of them give the farthest -- exactly,
    // however the node is scaled or sheared.
    let radiusSq = 0;
    for (let corner = 0; corner < 4; corner++) {
      let lengthSq = 0;
      for (let c = 0; c < 3; c++) {
        const x = world[m + c] * decal.size[0] / 2;
        const y = world[m + 4 + c] * decal.size[1] / 2 * (corner & 1 ? -1 : 1);
        const z = world[m + 8 + c] * decal.size[2] / 2 * (corner & 2 ? -1 : 1);
        lengthSq += (x + y + z) ** 2;
      }
      radiusSq = Math.max(radiusSq, lengthSq);
    }
    spheres[k * 4] = world[m + 12];
    spheres[k * 4 + 1] = world[m + 13];
    spheres[k * 4 + 2] = world[m + 14];
    spheres[k * 4 + 3] = Math.sqrt(radiusSq);
    k++;
  }
  return k;
}

export class DecalSet {
  constructor(rhi) {
    this.rhi = rhi;
    this.count = 0;
    this.revision = 0;
    this._data = new Float32Array(DECAL_FLOATS);
    /** Centre and radius per decal, for the cluster pass. */
    this.spheres = new Float32Array(4);
    this.buffer = createBuffer(rhi, {
      label: 'decals', size: DECAL_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._layers = new Map();
    this._textures = [];
    this._array = createTexture(rhi, {
      label: 'no-decals', size: [1, 1, 1], format: FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    this.view = this._array.createView({ dimension: '2d-array' });
  }

  /** Pack this frame's decals, rebuilding the texture array if the textures changed. */
  prepare(scene) {
    this.count = scene.decals.size;
    if (this.count === 0) return 0;
    const textures = [];
    for (const decal of scene.decals.values()) if (!textures.includes(decal.texture)) textures.push(decal.texture);
    if (textures.length !== this._textures.length || textures.some((t, k) => t !== this._textures[k])) {
      this._build(textures);
    }
    if (this._data.length < this.count * DECAL_FLOATS) {
      this._data = new Float32Array(this.count * DECAL_FLOATS * 2);
      this.spheres = new Float32Array(this.count * 4 * 2);
    }
    packDecals(scene, this._data, this._layers, this.spheres);
    const bytes = this.count * DECAL_FLOATS * 4;
    if (bytes > this.buffer.size) {
      this.buffer.destroy();
      this.buffer = createBuffer(this.rhi, {
        label: 'decals', size: this._data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.revision++;
    }
    this.rhi.queue.writeBuffer(this.buffer, 0, this._data, 0, this.count * DECAL_FLOATS);
    return this.count;
  }

  /**
   * Every decal texture as a layer of one array, at the largest one's size
   * rounded up to a power of two, mipmapped: resampled into it by the mip
   * pipeline's filter, as every other mip chain in the engine.
   */
  _build(textures) {
    const rhi = this.rhi;
    const largest = Math.max(...textures.map((t) => Math.max(t.width, t.height)));
    const size = 2 ** Math.ceil(Math.log2(largest));
    const array = createTexture(rhi, {
      label: 'decals', size: [size, size, textures.length], format: FORMAT,
      mipLevelCount: mipLevelCountFor(size, size),
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });
    const { pipeline, layout } = mipPipelineFor(rhi, FORMAT);
    const encoder = rhi.device.createCommandEncoder({ label: 'decal-array' });
    textures.forEach((texture, layer) => {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: array.createView({ dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 }),
          loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 },
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, rhi.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: texture.texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
          { binding: 1, resource: clampSampler(rhi) },
        ],
      }));
      pass.draw(3);
      pass.end();
    });
    rhi.queue.submit([encoder.finish()]);
    generateMipmaps(rhi, array);
    this._array.destroy();
    this._array = array;
    this.view = array.createView({ dimension: '2d-array' });
    this._textures = textures;
    this._layers = new Map(textures.map((t, k) => [t, k]));
    this.revision++;
  }

  destroy() {
    this._array.destroy();
    this.buffer.destroy();
  }
}
