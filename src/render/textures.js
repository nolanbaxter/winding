// Uploading decoded glTF images to the GPU.
//
// The layer boundary: scene/gltf/images.js produces ImageBitmaps and knows
// nothing about a GPU; this file turns them into textures and knows nothing
// about glTF's container format. Either half can be replaced without the other
// noticing.

import {
  createTexture2D, uploadImage, generateMipmaps, linearSampler,
} from '../rhi/texture.js';
import {
  materialTextureSlots, samplerDescriptor, textureImageIndex, textureSamplerIndex,
} from '../scene/gltf/images.js';

/**
 * Uploads on demand and caches, so an image shared by twenty materials is
 * uploaded once.
 *
 * The cache key is `imageIndex:srgb`, not just the image index. The same file
 * used as both a base colour map and a roughness map genuinely needs two GPU
 * textures with different formats -- one that decodes on sample and one that
 * does not.
 */
export class GLTFTextures {
  constructor(rhi, json, bitmaps) {
    this.rhi = rhi;
    this.json = json;
    this.bitmaps = bitmaps;
    this._textures = new Map();
    this._samplers = new Map();
    this.uploaded = 0;
  }

  /** {baseColor, normal, metallicRoughness, occlusion, emissive, sampler} */
  texturesFor(material) {
    const result = {};
    for (const { slot, texture, srgb } of materialTextureSlots(material)) {
      const gpu = this._texture(texture, srgb);
      if (gpu) result[slot] = gpu;
    }
    result.sampler = this._samplerFor(material);
    return result;
  }

  _texture(textureIndex, srgb) {
    const imageIndex = textureImageIndex(this.json, textureIndex);
    if (imageIndex < 0) return null;

    const bitmap = this.bitmaps[imageIndex];
    if (!bitmap) return null;            // decode failed; the factor stands in

    const key = `${imageIndex}:${srgb ? 1 : 0}`;
    let texture = this._textures.get(key);
    if (texture) return texture;

    texture = createTexture2D(this.rhi, {
      label: `gltf-image-${imageIndex}${srgb ? '-srgb' : ''}`,
      width: bitmap.width,
      height: bitmap.height,
      srgb,
      mipmapped: true,
    });
    uploadImage(this.rhi, texture, bitmap);
    // Mips are averaged through an -srgb view when the format is sRGB, so the
    // filtering happens in linear light without any manual conversion.
    generateMipmaps(this.rhi, texture);

    this._textures.set(key, texture);
    this.uploaded++;
    return texture;
  }

  /**
   * One sampler per material, taken from its base colour texture.
   *
   * ponytail: glTF allows a different sampler per texture. Honouring that means
   * five sampler bindings in group 2 instead of one. Assets that mix wrap modes
   * across a single material's maps are rare enough that this has not been
   * worth the extra slots -- widen the bind group layout if one shows up.
   */
  _samplerFor(material) {
    const baseColorTexture = material.textures?.baseColor ?? -1;
    const samplerIndex = textureSamplerIndex(this.json, baseColorTexture);
    if (samplerIndex < 0) return linearSampler(this.rhi);

    let sampler = this._samplers.get(samplerIndex);
    if (!sampler) {
      const descriptor = samplerDescriptor(this.json.samplers?.[samplerIndex]);
      sampler = this.rhi.device.createSampler({ label: `gltf-sampler-${samplerIndex}`, ...descriptor });
      this._samplers.set(samplerIndex, sampler);
    }
    return sampler;
  }

  destroy() {
    for (const texture of this._textures.values()) texture.destroy();
    this._textures.clear();
  }
}
