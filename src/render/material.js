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
import { createBuffer } from '../rhi/buffer.js';
import { MATERIAL_TEXTURES, CORE_TEXTURE_COUNT, EXTENSION_TEXTURES } from '../scene/gltf/images.js';

/**
 * The most materials a sort key can address. Derived from the key layout in
 * drawlist.js, not chosen here.
 */
/** No texture transform, for every slot: the rows (1 0 0) and (0 1 0). */
const IDENTITY_UV_TRANSFORMS = Float32Array.from(
  { length: MATERIAL_TEXTURES.length * 6 }, (_, i) => (i % 6 === 0 || i % 6 === 4 ? 1 : 0),
);

const WHITE = [1, 1, 1];
const BLACK = [0, 0, 0];

export const MATERIAL_ID_LIMIT = 1 << OPAQUE_MATERIAL_BITS;

export const ALPHA_OPAQUE = 0;
export const ALPHA_MASK = 1;
export const ALPHA_BLEND = 2;

const ALPHA_MODES = { OPAQUE: ALPHA_OPAQUE, MASK: ALPHA_MASK, BLEND: ALPHA_BLEND };

/** Where the texture rows start, in floats. */
const UV_ROWS_FLOAT = 44;

/**
 * The material uniform, as WGSL. One definition: the forward pass and the
 * alpha-tested shadow casters both read it, and _writeFactors below writes it.
 */
export const MATERIAL_WGSL = /* wgsl */ `
struct Material {
  baseColor         : vec4<f32>,  //  0
  emissive          : vec4<f32>,  // 16   w = metallic
  roughness         : f32,        // 32
  normalScale       : f32,        // 36
  alphaCutoff       : f32,        // 40
  occlusionStrength : f32,        // 44
  // Bit per core texture: set means that map samples UV set 1. Carried as f32
  // because the rest of the struct is.
  uvSets            : f32,        // 48
  ior               : f32,        // 52   KHR_materials_ior; 1.5 without
  specular          : f32,        // 56   KHR_materials_specular's strength; 1 without
  unlit             : f32,        // 60   KHR_materials_unlit: 1 for unlit
  specularColor     : vec4<f32>,  // 64   KHR_materials_specular's tint; white without
  // KHR_materials_clearcoat, all 0 but the scale without it, and
  // KHR_materials_sheen's roughness.
  clearcoat          : f32,       // 80
  clearcoatRoughness : f32,       // 84
  clearcoatNormalScale : f32,     // 88
  sheenRoughness     : f32,       // 92
  sheenColor        : vec4<f32>,  // 96   KHR_materials_sheen; black, so none, without
  // KHR_materials_anisotropy and KHR_materials_iridescence; 0 strength for none.
  anisotropyStrength : f32,       // 112
  anisotropyRotation : f32,       // 116  radians from the tangent
  iridescence        : f32,       // 120
  iridescenceIor     : f32,       // 124
  iridescenceThickness : vec4<f32>, // 128  x minimum, y maximum, in nanometres
  // KHR_materials_transmission and KHR_materials_volume. An attenuation
  // distance of 0 stands for the infinite one: no attenuation.
  transmission       : f32,       // 144
  thickness          : f32,       // 148
  attenuationDistance : f32,      // 152
  attenuationColor  : vec4<f32>,  // 160
  // Two rows a texture, (u, v, 1) dotted with each -- KHR_texture_transform,
  // the identity when a file gives none -- in MATERIAL_TEXTURES order. An
  // extension texture's first row carries its UV set in w, and its second the
  // binding it samples, -1 for none.
  uvTransforms      : array<vec4<f32>, ${MATERIAL_TEXTURES.length * 2}>,   // 176
};`;

export const MATERIAL_BYTES = (UV_ROWS_FLOAT + MATERIAL_TEXTURES.length * 8) * 4;

/** Where the extension textures bind: after the core five and the sampler. */
export const EXTENSION_BINDING = 7;

/**
 * How many distinct extension textures one material can bind: the per-stage
 * limit, less what the frame and the core material maps already take -- or
 * every extension texture there is, if the device has room for them all.
 */
export function extensionSlotCount(limit, frameTextures) {
  return Math.max(0, Math.min(EXTENSION_TEXTURES.length, limit - frameTextures - CORE_TEXTURE_COUNT));
}

/**
 * Bit per texture slot: set means that map samples UV set 1.
 *
 * glTF puts `texCoord` on the texture REFERENCE rather than the material, so
 * one material's maps can disagree about which set they use -- baked occlusion
 * on set 1 beside a base colour on set 0 is the ordinary case out of Blender
 * and Max. Five bits in one float beats five floats, and the shader picks per
 * sample with a select rather than branching.
 */
export const UV_SET_BASE_COLOR = 1;
export const UV_SET_METALLIC_ROUGHNESS = 2;
export const UV_SET_NORMAL = 4;
export const UV_SET_OCCLUSION = 8;
export const UV_SET_EMISSIVE = 16;

/** Pack a material's five `texCoord` values into that bitfield. */
export function uvSetMask(uvSets = {}) {
  return (uvSets.baseColor ? UV_SET_BASE_COLOR : 0)
    | (uvSets.metallicRoughness ? UV_SET_METALLIC_ROUGHNESS : 0)
    | (uvSets.normal ? UV_SET_NORMAL : 0)
    | (uvSets.occlusion ? UV_SET_OCCLUSION : 0)
    | (uvSets.emissive ? UV_SET_EMISSIVE : 0);
}

/**
 * Bits of a pipeline variant.
 *
 * MIRRORED is not a property of the material, unlike the other two. It comes
 * from the INSTANCE -- whether its world transform has a negative determinant
 * -- so the registry stores a variant without it and the renderer ORs it in
 * per batch. Two instances of one material, one mirrored, need two pipelines.
 */
export const VARIANT_DOUBLE_SIDED = 4;
export const VARIANT_MIRRORED = 8;
/**
 * Skinned. A vertex-stage difference only -- the fragment side is identical --
 * but a pipeline is one object, so it is a variant like the rest. It also
 * carries a second vertex buffer, which is the other half of why it cannot be
 * anything smaller.
 */
export const VARIANT_SKINNED = 16;
/**
 * Transmissive (KHR_materials_transmission). It draws after the opaque scene
 * has been copied, since that copy is what it shows through it, so it has
 * the one colour target that pass has -- never the ambient one ambient
 * occlusion adds.
 */
export const VARIANT_TRANSMISSIVE = 32;
/**
 * Draws the extension layers: clearcoat, sheen, anisotropy, iridescence,
 * transmission, and any extension texture. The rest compile that code out,
 * through the shader's EXTENSIONS constant. Measured on Sponza at 720p, which
 * uses none of it, carrying it cost the forward passes 7.65 ms against 5.62 --
 * a third of their time, for branches that never run.
 */
export const VARIANT_EXTENDED = 64;

/** Whether a material needs the extended shader; see VARIANT_EXTENDED. */
export function usesExtendedShading(material, extensionTextures) {
  return extensionTextures > 0 || material.extendedShading === true
    || (material.transmission ?? 0) > 0 || (material.clearcoat ?? 0) > 0
    || (material.anisotropyStrength ?? 0) > 0 || (material.iridescence ?? 0) > 0
    || (material.sheenColor ?? BLACK).some((c) => c > 0);
}

/**
 * A pipeline variant. Packed small on purpose: it is also the pipeline id that
 * goes into the sort key, and the narrower of those fields is 4 bits wide.
 * Three alpha modes times two sidedness times two windings is 12 of 16.
 */
export function variantKey(alphaMode, doubleSided, mirrored = false, skinned = false, transmissive = false, extended = false) {
  return alphaMode
    | (doubleSided ? VARIANT_DOUBLE_SIDED : 0)
    | (mirrored ? VARIANT_MIRRORED : 0)
    | (skinned ? VARIANT_SKINNED : 0)
    | (transmissive ? VARIANT_TRANSMISSIVE : 0)
    | (extended || transmissive ? VARIANT_EXTENDED : 0);
}

export class MaterialRegistry {
  /**
   * @param frameTextures how many textures the frame's bind group gives the
   *                      fragment stage, which the extension slots share a
   *                      limit with
   */
  constructor(rhi, { capacity = 256, label = 'materials', frameTextures = 0 } = {}) {
    this.rhi = rhi;
    this.capacity = capacity;
    this.count = 0;
    this._label = label;

    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    // Where each material starts: its size, rounded up to what a uniform
    // offset must be a multiple of. The alignment alone was the stride while a
    // material fitted in one, which a device with a small alignment breaks.
    this.stride = Math.ceil(MATERIAL_BYTES / this.alignment) * this.alignment;
    this.buffer = createBuffer(rhi, {
      label,
      size: this.stride * capacity,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.staging = new ArrayBuffer(this.stride * capacity);

    this._limit = rhi.limits.maxSampledTexturesPerShaderStage;
    this._frameTextures = frameTextures;
    this.extensionSlots = extensionSlotCount(this._limit, frameTextures);

    // Two layouts: the core textures, which every material binds and the
    // shadow pass reads, and those plus the extension slots, for materials
    // the extended shader draws. A plain material binding a dozen unused
    // slots cost the CPU about 0.14 ms a frame on Sponza, in resource
    // tracking on every material change.
    const core = [
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
    ];
    this.layout = rhi.device.createBindGroupLayout({ label: 'material', entries: core });
    this.extendedLayout = rhi.device.createBindGroupLayout({
      label: 'material-extended',
      entries: [...core, ...Array.from({ length: this.extensionSlots }, (_, i) => (
        { binding: EXTENSION_BINDING + i, visibility: GPUShaderStage.FRAGMENT, texture: {} }))],
    });

    /** Per id: the core group, and the group its own pipelines bind (the extended one, if it has one). */
    this.bindGroups = [];
    this.shadingGroups = [];
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
    /** The importer record each id was registered from, for update(). */
    this._records = [];
    /** Each id's extension textures: the distinct ones, and which each kind samples. */
    this._extensions = [];
    /** Ids given back by release(), handed out again before new ones. */
    this._free = [];
    this.variants = new Uint8Array(capacity);
    this.alphaModes = new Uint8Array(capacity);
    this.transmissive = new Uint8Array(capacity);

    // Variant -> dense pipeline id, assigned in first-seen order so the id
    // stays small however sparse the variant space is.
    this._pipelineIds = new Map();
    this.pipelineIdOf = new Uint8Array(capacity);

    this._defaults = defaultTextures(rhi);
    this._sampler = linearSampler(rhi);
  }

  /**
   * @param material a material record from the glTF importer
   * @param textures optional GPUTextures keyed by slot: baseColor, normal,
   *                 metallicRoughness, occlusion, emissive, plus an optional
   *                 sampler. Anything missing falls back to a 1x1 default.
   *                 The key names matter -- an unrecognised one is ignored in
   *                 silence, and the material renders flat with no error.
   * @returns the material id, which is also what goes into the sort key
   */
  register(material, textures = {}) {
    // Before an id is minted, so a material that cannot bind leaks nothing.
    const extensions = this._extensionSlotsFor(material, textures);
    // Unconditional, and this is the right place for it: the sort key's
    // material field is 12 bits, and an id that does not fit would silently
    // alias two materials into one bucket. drawlist.js only DEBUG-checks
    // because by then the id has already been minted here.
    // A released id first. Without reuse, loading and unloading one asset over
    // and over walked the id up to the 4096 the sort key allows, and threw.
    let id = this._free.pop();
    if (id === undefined) {
      if (this.count >= this.capacity) this._grow(this.count + 1);
      id = this.count++;
    }
    this._records[id] = material;
    this._extensions[id] = extensions;
    this._writeFactors(id, material);

    const alphaMode = ALPHA_MODES[material.alphaMode] ?? ALPHA_OPAQUE;
    this.alphaModes[id] = alphaMode;

    const transmissive = (material.transmission ?? 0) > 0;
    this.transmissive[id] = transmissive ? 1 : 0;
    const extended = usesExtendedShading(material, extensions.bound.length);
    const variant = variantKey(alphaMode, material.doubleSided === true, false, false, transmissive, extended);
    this.variants[id] = variant;

    let pipelineId = this._pipelineIds.get(variant);
    if (pipelineId === undefined) {
      pipelineId = this._pipelineIds.size;
      this._pipelineIds.set(variant, pipelineId);
    }
    this.pipelineIdOf[id] = pipelineId;

    this._textures[id] = textures;
    this._names[id] = material.name ?? String(id);
    this._makeBindGroups(id);

    return id;
  }

  /**
   * Upload a material's factors again, after a clip changed them. Skipped
   * when the id no longer holds that record: the asset was unloaded between
   * the change and this upload, and the id may be someone else's by now.
   */
  /**
   * The extension textures a material binds. Kinds reading the same GPU
   * texture share one binding -- clearcoat and its roughness are usually one
   * image -- each keeping its own UV set and transform.
   */
  _extensionSlotsFor(material, textures) {
    const bound = [];
    const slots = EXTENSION_TEXTURES.map(({ slot }) => {
      const texture = textures[slot];
      if (!texture) return -1;
      const at = bound.indexOf(texture);
      return at >= 0 ? at : bound.push(texture) - 1;
    });
    if (bound.length > this.extensionSlots) {
      throw new Error(
        `MaterialRegistry: material "${material.name}" samples ${bound.length} distinct extension textures; `
        + `this device binds ${this.extensionSlots} (maxSampledTexturesPerShaderStage ${this._limit}, less `
        + `${this._frameTextures} for the frame and ${CORE_TEXTURE_COUNT} for the core maps)`,
      );
    }
    return { bound, slots };
  }

  update(id, material) {
    if (this._records[id] === material) this._writeFactors(id, material);
  }

  _writeFactors(id, material) {
    const offset = id * this.stride;
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
    f32[12] = uvSetMask(material.uvSets);
    f32[13] = material.ior ?? 1.5;
    f32[14] = material.specular ?? 1;
    f32[15] = material.unlit ? 1 : 0;
    f32.set(material.specularColor ?? WHITE, 16);
    f32[20] = material.clearcoat ?? 0;
    f32[21] = material.clearcoatRoughness ?? 0;
    f32[22] = material.clearcoatNormalScale ?? 1;
    f32[23] = material.sheenRoughness ?? 0;
    f32.set(material.sheenColor ?? BLACK, 24);
    f32[28] = material.anisotropyStrength ?? 0;
    f32[29] = material.anisotropyRotation ?? 0;
    f32[30] = material.iridescence ?? 0;
    f32[31] = material.iridescenceIor ?? 1.3;
    f32[32] = material.iridescenceThicknessMinimum ?? 100;
    f32[33] = material.iridescenceThicknessMaximum ?? 400;
    f32[36] = material.transmission ?? 0;
    f32[37] = material.thickness ?? 0;
    const distance = material.attenuationDistance ?? Infinity;
    f32[38] = Number.isFinite(distance) ? distance : 0;
    f32.set(material.attenuationColor ?? WHITE, 40);
    // Two rows a texture, each a vec4 for alignment, in MATERIAL_TEXTURES order.
    const slots = this._extensions[id].slots;
    for (let k = 0; k < MATERIAL_TEXTURES.length; k++) {
      const t = material.uvTransforms?.length >= k * 6 + 6 ? material.uvTransforms : IDENTITY_UV_TRANSFORMS;
      const at = UV_ROWS_FLOAT + k * 8;
      const extension = k - CORE_TEXTURE_COUNT;
      f32.set(t.subarray(k * 6, k * 6 + 3), at);
      f32[at + 3] = extension < 0 ? 0 : (material.uvSets?.[MATERIAL_TEXTURES[k].slot] ? 1 : 0);
      f32.set(t.subarray(k * 6 + 3, k * 6 + 6), at + 4);
      f32[at + 7] = extension < 0 ? 0 : slots[extension];
    }

    this.rhi.queue.writeBuffer(this.buffer, offset, this.staging, offset, MATERIAL_BYTES);
  }

  _makeBindGroups(id) {
    const textures = this._textures[id] ?? {};
    const core = [
      { binding: 0, resource: { buffer: this.buffer, offset: id * this.stride, size: MATERIAL_BYTES } },
      { binding: 1, resource: (textures.baseColor ?? this._defaults.white).createView() },
      { binding: 2, resource: (textures.normal ?? this._defaults.flatNormal).createView() },
      { binding: 3, resource: (textures.metallicRoughness ?? this._defaults.orm).createView() },
      { binding: 4, resource: (textures.occlusion ?? this._defaults.white).createView() },
      // White, not black. glTF says an absent texture means 1.0 on every
      // channel, and the shader multiplies this by emissiveFactor -- so a
      // black default silently throws the factor away, and a material that
      // asked to glow renders dark with nothing reported. Emissive WITHOUT a
      // texture is how a simple glowing object is authored, so this was the
      // common case rather than the corner one.
      { binding: 5, resource: (textures.emissive ?? this._defaults.white).createView() },
      { binding: 6, resource: textures.sampler ?? this._sampler },
    ];
    const label = `material:${this._names[id]}`;
    this.bindGroups[id] = this.rhi.device.createBindGroup({ label, layout: this.layout, entries: core });
    this.shadingGroups[id] = (this.variants[id] & VARIANT_EXTENDED) === 0 ? this.bindGroups[id] : this.rhi.device.createBindGroup({
      label: `${label}:extended`,
      layout: this.extendedLayout,
      entries: [...core, ...Array.from({ length: this.extensionSlots }, (_, i) => ({
        binding: EXTENSION_BINDING + i,
        resource: (this._extensions[id].bound[i] ?? this._defaults.white).createView(),
      }))],
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

    const staging = new ArrayBuffer(this.stride * capacity);
    new Uint8Array(staging).set(new Uint8Array(this.staging));
    this.staging = staging;

    this.buffer.destroy();
    this.buffer = createBuffer(this.rhi, {
      label: this._label,
      size: this.stride * capacity,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Re-upload everything registered so far: the new buffer starts empty.
    this.rhi.queue.writeBuffer(this.buffer, 0, this.staging, 0, this.stride * this.count);

    this.variants = growArray(this.variants, capacity);
    this.alphaModes = growArray(this.alphaModes, capacity);
    this.transmissive = growArray(this.transmissive, capacity);
    this.pipelineIdOf = growArray(this.pipelineIdOf, capacity);
    this.capacity = capacity;

    // Every existing bind group points at the destroyed buffer.
    for (let id = 0; id < this.count; id++) if (this._records[id] !== undefined) this._makeBindGroups(id);
  }

  /**
   * Give an id back. Nothing may still draw with it -- engine.unload checks
   * that -- because the next register() will reuse it for another surface.
   * The references go too, so the textures behind it are not kept alive.
   */
  release(id) {
    this._textures[id] = undefined;
    this._records[id] = undefined;
    this._extensions[id] = undefined;
    this.bindGroups[id] = undefined;
    this.shadingGroups[id] = undefined;
    this._free.push(id);
  }

  /** The core textures: what the shadow pass binds. */
  bindGroup(materialId) {
    return this.bindGroups[materialId];
  }

  /** What the material's own pipelines bind: extended if its variant is. */
  shadingGroup(materialId) {
    return this.shadingGroups[materialId];
  }

  /**
   * Whether it leaves the batched opaque path for the sorted one: blended, or
   * transmissive, which has to see the opaque scene finished before it draws.
   */
  isTransparent(materialId) {
    return this.alphaModes[materialId] === ALPHA_BLEND || this.transmissive[materialId] === 1;
  }

  isTransmissive(materialId) {
    return this.transmissive[materialId] === 1;
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
  const doubleSided = (variant & VARIANT_DOUBLE_SIDED) !== 0;
  const mirrored = (variant & VARIANT_MIRRORED) !== 0;

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
      // glTF 3.7.4: a node whose global transform has a negative determinant
      // has its winding reversed. Mirroring one chair to make its pair is the
      // everyday case, and with a fixed 'ccw' both of them culled the faces
      // that should be visible and kept the ones that should not.
      frontFace: mirrored ? 'cw' : 'ccw',
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
    constants: {
      USE_ALPHA_MASK: alphaMode === ALPHA_MASK ? 1 : 0,
      EXTENSIONS: (variant & VARIANT_EXTENDED) !== 0 ? 1 : 0,
    },
  };
}
