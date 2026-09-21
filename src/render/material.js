// Material registry. Bind group 2 in the frequency model.
//
// Two decisions shape this file.
//
// NO VARIANTS FOR TEXTURE PRESENCE. A material with no normal map still binds
// one -- a 1x1 flat-normal default. The alternative is a shader permutation per
// combination of present maps, which is 16 pipelines for four slots and is the
// classic way shader compile times get out of hand. Sampling a 1x1 texture
// costs nothing measurable.
//
// VARIANTS ONLY WHERE THE PIPELINE GENUINELY DIFFERS. Three things cannot be
// expressed by a uniform: cull mode, blend state, and whether the shader
// discards. Those are the whole variant set, and there are at most eight.

import { DEPTH_COMPARE, DEPTH_FORMAT } from '../rhi/device.js';
import { defaultTextures, linearSampler } from '../rhi/texture.js';
import { grownCapacity, growArray } from '../core/grow.js';
import { OPAQUE_MATERIAL_BITS } from './drawlist.js';

/**
 * The most materials a sort key can address. Derived from the key layout in
 * drawlist.js, not chosen here.
 */
export const MATERIAL_ID_LIMIT = 1 << OPAQUE_MATERIAL_BITS;

export const ALPHA_OPAQUE = 0;
export const ALPHA_MASK = 1;
export const ALPHA_BLEND = 2;

const ALPHA_MODES = { OPAQUE: ALPHA_OPAQUE, MASK: ALPHA_MASK, BLEND: ALPHA_BLEND };

/** baseColor(16) + emissive/metallic(16) + roughness, normalScale, cutoff, pad(16) */
export const MATERIAL_BYTES = 48;

/**
 * A pipeline variant. Packed small on purpose: it is also the pipeline id that
 * goes into the sort key, and that field is 10 bits wide.
 */
export function variantKey(alphaMode, doubleSided) {
  return alphaMode | (doubleSided ? 4 : 0);
}

export class MaterialRegistry {
  constructor(rhi, { capacity = 256, label = 'materials' } = {}) {
    this.rhi = rhi;
    this.capacity = capacity;
    this.count = 0;
    this._label = label;

    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    this.buffer = rhi.device.createBuffer({
      label,
      size: this.alignment * capacity,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.staging = new ArrayBuffer(this.alignment * capacity);

    this.layout = rhi.device.createBindGroupLayout({
      label: 'material',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: MATERIAL_BYTES } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },   // base colour
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },   // normal
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },   // metallic/roughness
        // Occlusion gets its own slot rather than riding in the R channel of
        // the metallic-roughness map. Exporters usually pack them together, but
        // glTF permits separate images and reading R from the wrong texture
        // would darken the model with no error anywhere.
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },   // occlusion
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} },   // emissive
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    this.bindGroups = [];
    /**
     * The textures each bind group was built from.
     *
     * Kept because growth recreates the uniform buffer, and a bind group holds
     * a direct reference to it -- every one has to be rebuilt, which means
     * knowing what went into it. These are references to textures the asset
     * already owns, not copies.
     */
    this._textures = [];
    this._names = [];
    this.variants = new Uint8Array(capacity);
    this.alphaModes = new Uint8Array(capacity);

    // Variant -> dense pipeline id, assigned in first-seen order so the id
    // stays small however sparse the variant space is.
    this._pipelineIds = new Map();
    this.pipelineIdOf = new Uint8Array(capacity);

    this._defaults = defaultTextures(rhi);
    this._sampler = linearSampler(rhi);
  }

  /**
   * @param material a material record from the glTF importer
   * @param textures optional {baseColor, normal, orm, emissive} GPUTextures;
   *                 anything missing falls back to a 1x1 default
   * @returns the material id, which is also what goes into the sort key
   */
  register(material, textures = {}) {
    // Unconditional, and this is the right place for it: the sort key's
    // material field is 12 bits, and an id that does not fit would silently
    // alias two materials into one bucket. drawlist.js only DEBUG-checks
    // because by then the id has already been minted here.
    if (this.count >= this.capacity) this._grow(this.count + 1);

    const id = this.count++;
    const offset = id * this.alignment;

    const f32 = new Float32Array(this.staging, offset, MATERIAL_BYTES / 4);
    f32.set(material.baseColorFactor ?? [1, 1, 1, 1], 0);
    f32[4] = material.emissive?.[0] ?? 0;
    f32[5] = material.emissive?.[1] ?? 0;
    f32[6] = material.emissive?.[2] ?? 0;
    f32[7] = material.metallic ?? 1;          // packed into emissive.w
    f32[8] = material.roughness ?? 1;
    f32[9] = material.normalScale ?? 1;
    f32[10] = material.alphaCutoff ?? 0.5;
    f32[11] = material.occlusionStrength ?? 1;

    this.rhi.queue.writeBuffer(this.buffer, offset, this.staging, offset, MATERIAL_BYTES);

    const alphaMode = ALPHA_MODES[material.alphaMode] ?? ALPHA_OPAQUE;
    this.alphaModes[id] = alphaMode;

    const variant = variantKey(alphaMode, material.doubleSided === true);
    this.variants[id] = variant;

    let pipelineId = this._pipelineIds.get(variant);
    if (pipelineId === undefined) {
      pipelineId = this._pipelineIds.size;
      this._pipelineIds.set(variant, pipelineId);
    }
    this.pipelineIdOf[id] = pipelineId;

    this._textures[id] = textures;
    this._names[id] = material.name ?? String(id);
    this.bindGroups[id] = this._makeBindGroup(id);

    return id;
  }

  _makeBindGroup(id) {
    const textures = this._textures[id] ?? {};
    return this.rhi.device.createBindGroup({
      label: `material:${this._names[id]}`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.buffer, offset: id * this.alignment, size: MATERIAL_BYTES } },
        { binding: 1, resource: (textures.baseColor ?? this._defaults.white).createView() },
        { binding: 2, resource: (textures.normal ?? this._defaults.flatNormal).createView() },
        { binding: 3, resource: (textures.metallicRoughness ?? this._defaults.orm).createView() },
        { binding: 4, resource: (textures.occlusion ?? this._defaults.white).createView() },
        { binding: 5, resource: (textures.emissive ?? this._defaults.black).createView() },
        { binding: 6, resource: textures.sampler ?? this._sampler },
      ],
    });
  }

  /**
   * Widen for more materials.
   *
   * The 12-bit ceiling is real and stays a throw: the sort key packs the
   * material id into 12 bits, so a 4097th material would alias onto an existing
   * one and silently draw with the wrong surface. That is a property of the key
   * layout, not a capacity, which is why more memory does not fix it.
   */
  _grow(needed) {
    const capacity = grownCapacity(this.capacity, needed);
    if (capacity > MATERIAL_ID_LIMIT) {
      throw new Error(
        `MaterialRegistry: ${needed} materials exceeds the ${MATERIAL_ID_LIMIT} the sort key can address`,
      );
    }

    const staging = new ArrayBuffer(this.alignment * capacity);
    new Uint8Array(staging).set(new Uint8Array(this.staging));
    this.staging = staging;

    this.buffer.destroy();
    this.buffer = this.rhi.device.createBuffer({
      label: this._label,
      size: this.alignment * capacity,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Re-upload everything registered so far: the new buffer starts empty.
    this.rhi.queue.writeBuffer(this.buffer, 0, this.staging, 0, this.alignment * this.count);

    this.variants = growArray(this.variants, capacity);
    this.alphaModes = growArray(this.alphaModes, capacity);
    this.pipelineIdOf = growArray(this.pipelineIdOf, capacity);
    this.capacity = capacity;

    // Every existing bind group points at the destroyed buffer.
    for (let id = 0; id < this.count; id++) this.bindGroups[id] = this._makeBindGroup(id);
  }

  bindGroup(materialId) {
    return this.bindGroups[materialId];
  }

  isTransparent(materialId) {
    return this.alphaModes[materialId] === ALPHA_BLEND;
  }

  /** Every distinct variant registered so far, for warming the pipeline cache. */
  variantList() {
    return [...this._pipelineIds.keys()];
  }
}

/**
 * Pipeline state for one variant. Everything here is something a uniform
 * cannot express, which is exactly the test for whether a variant is warranted.
 */
export function variantPipelineState(variant) {
  const alphaMode = variant & 3;
  const doubleSided = (variant & 4) !== 0;

  const blend = alphaMode === ALPHA_BLEND
    ? {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    }
    : undefined;

  return {
    primitive: {
      topology: 'triangle-list',
      cullMode: doubleSided ? 'none' : 'back',
      frontFace: 'ccw',
    },
    depth: {
      format: DEPTH_FORMAT,
      depthCompare: DEPTH_COMPARE,
      // Blended surfaces must not write depth: they do not occlude what is
      // behind them, and writing would make the result depend on draw order
      // in a way even a correct back-to-front sort cannot fix for coplanar
      // geometry.
      depthWriteEnabled: alphaMode !== ALPHA_BLEND,
    },
    blend,
    // Only the MASK variant discards. A discard anywhere in a shader disables
    // early-Z for every draw using it, so opaque geometry must not pay for it.
    constants: { USE_ALPHA_MASK: alphaMode === ALPHA_MASK ? 1 : 0 },
  };
}
