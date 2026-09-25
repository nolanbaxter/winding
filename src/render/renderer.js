// The forward renderer. Owns everything between "here is a scene" and "pixels".
//
// The layouts below are the important part: they are the engine's contract with
// its own shader, and a Tier 2 custom material is handed them to build against.

import { compileShader } from '../rhi/shader.js';
import { PipelineCache } from '../rhi/pipeline.js';
import { DrawList, opaqueSortKey, transparentSortKey, transparentDepthBucket } from './drawlist.js';
import {
  createPipelineLayout, GROUP_FRAME, GROUP_MATERIAL, GROUP_DRAW,
} from '../rhi/bindgroups.js';
import { DEPTH_CLEAR_VALUE, DEPTH_FORMAT, DEPTH_COMPARE } from '../rhi/device.js';

/** OIT targets. accum sums weighted colour; reveal is the surviving background. */
export const OIT_ACCUM_FORMAT = 'rgba16float';
export const OIT_REVEAL_FORMAT = 'r16float';

import { mat4NormalMatrix } from '../core/math/mat4.js';
import { vec3Create, vec3Sub, vec3Normalize } from '../core/math/vec3.js';
import { frustumCreate, frustumFromViewProjection, frustumTestAABB } from '../core/math/frustum.js';
import { grownCapacity } from '../core/grow.js';
import { DIRECTIONAL_FLOATS } from '../scene/scene.js';

import {
  MaterialRegistry, variantPipelineState, VARIANT_MIRRORED, VARIANT_SKINNED, VARIANT_TRANSMISSIVE, VARIANT_EXTENDED, ALPHA_BLEND,
} from './material.js';
import { pbrShader, FRAME_BYTES } from './shaders/pbr.js';
import { SHEEN_ALBEDO } from './sheen.js';
import { OpaqueCopy } from './transmission.js';
import { packFog } from './fog.js';
import { DebugLines } from './debug.js';
import { SpritePass } from './sprites.js';
import { ParticleSystem } from './particles.js';
import { DecalSet } from './decals.js';
import { DepthOfField } from './dof.js';
import { ProbeSet, FACE_CAMERAS, flipInto, PROBE_FLOATS } from './probes.js';
import { Environment } from './ibl.js';
import { Camera } from '../scene/camera.js';
import { createTexture, cubeFaceView } from '../rhi/texture.js';
import { OIT_RESOLVE_SHADER } from './shaders/oit.js';
import { SkyboxPass } from './skybox.js';
import { ShadowMaps, stableShadowDistance } from './shadows.js';
import { RenderGraph } from './graph.js';
import { GpuProfiler } from './timing.js';
import { SkinPalette } from './skin.js';
import { MorphStore } from './morph.js';
import { ClusteredLights, CLUSTER_Z } from './clustered.js';
import { PostStack, HDR_FORMAT } from './post.js';
import { GpuDriven, BATCH_BYTES, INDIRECT_BYTES, CULL_SHADOW } from './gpudriven.js';
import {
  updateWorldBounds, unionWorldBounds, farthestViewDepth, farthestDistance,
  updateSkinBounds, applySkinBounds,
} from '../scene/bounds.js';
import { HierarchicalDepth } from './hzb.js';
import { AmbientOcclusion, AMBIENT_FORMAT } from './ao.js';
import { VERTEX_BUFFER_LAYOUT as VERTEX_LAYOUT, SKIN_BUFFER_LAYOUT } from './vertex.js';
import { createBuffer } from '../rhi/buffer.js';

const DEFAULT_MAX_DRAWS = 4096;

/**
 * Refuse, by name, a device with fewer storage buffers per stage than the
 * frame layout reads. Counted from the layout itself, so it cannot drift from
 * it. Every core adapter allows 8 and this needs 5 in the vertex stage, but
 * compatibility-mode devices may allow none there -- and a layout the device
 * rejects is a validation error and a black frame, nowhere near the cause.
 */
function checkStorageStages(rhi, entries) {
  const limits = rhi.limits;
  if (!limits) return;
  for (const [stage, name, limit] of [
    [GPUShaderStage.VERTEX, 'vertex', limits.maxStorageBuffersInVertexStage ?? limits.maxStorageBuffersPerShaderStage],
    [GPUShaderStage.FRAGMENT, 'fragment', limits.maxStorageBuffersInFragmentStage ?? limits.maxStorageBuffersPerShaderStage],
  ]) {
    const needed = entries.filter((e) => (e.visibility & stage) && e.buffer?.type?.endsWith('storage')).length;
    if (limit !== undefined && needed > limit) {
      throw new Error(`This GPU allows ${limit} storage buffers in the ${name} stage; the renderer reads ${needed}.`);
    }
  }
}

/** Scratch for the camera forward axis used by transparent depth sorting. */
const FORWARD = vec3Create();

const now = () => (globalThis.performance?.now?.() ?? Date.now());

/** Shader features a pipeline set compiles in; see PROBES and DECALS in pbr.js. */
export const FEATURE_PROBES = 1;
export const FEATURE_DECALS = 2;

function featureConstants(features) {
  return { PROBES: features & FEATURE_PROBES ? 1 : 0, DECALS: features & FEATURE_DECALS ? 1 : 0 };
}

export class Renderer {
  static async create(rhi, {
    maxDraws = DEFAULT_MAX_DRAWS, exposure = 1.0, shadows, lightDistance = null,
    shadowDistance = null, post, gpuTiming = false, oit = false, ao = false,
  } = {}) {
    const renderer = new Renderer(rhi, {
      maxDraws, exposure, shadows, lightDistance, shadowDistance, post, gpuTiming, oit, ao,
    });
    await renderer._init();
    return renderer;
  }

  constructor(rhi, {
    maxDraws, exposure, shadows, lightDistance, shadowDistance, post, gpuTiming = false,
    oit = false, ao = false, fog = null, dof = null,
  }) {
    this.shadowOptions = shadows ?? {};
    this._fogOption = fog;
    this._dofOption = dof;
    this.postOptions = post ?? {};
    this.rhi = rhi;
    this.maxDraws = maxDraws;
    this.exposure = exposure;

    this.pipelines = new PipelineCache(rhi.device);
    const frameEntries = [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      {
        binding: 4,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'depth', viewDimension: '2d-array' },
      },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      // Storage rather than uniform: the light list can exceed the 64KB
      // uniform binding limit, and the shader indexes it dynamically.
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      // Per-object transforms and the GPU-written visible list. Read in the
      // VERTEX stage: this is what replaces a bind group per draw.
      { binding: 9, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 10, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      // Every skinned instance's joint matrices. Bound for every pipeline,
      // skinned or not, because a bind group layout is one object -- an
      // unskinned vertex shader simply never reads it.
      { binding: 11, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      // Morph deltas, then morph weights. Same call as the palette: bound
      // for every pipeline, read only by a draw whose target count is not
      // zero, which is a number the shader already has in hand.
      { binding: 12, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 13, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      // Every directional light but the shadowed one.
      { binding: 14, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      // Point and spot shadows: their maps, a layer a view, and the views.
      {
        binding: 15,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'depth', viewDimension: '2d-array' },
      },
      { binding: 16, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      // Every casting directional light's cascade matrices, light by light.
      { binding: 17, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      // The opaque scene, mipped, for transmissive surfaces to see through.
      { binding: 18, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      // Reflection probes: their prefiltered cubes, and their boxes.
      { binding: 19, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube-array' } },
      { binding: 20, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      // Decals: their textures, a layer each, and their boxes. The boxes are the
      // fragment stage's eighth storage buffer, which is as many as WebGPU
      // guarantees: the next thing that needs one has to share.
      { binding: 21, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '2d-array' } },
      { binding: 22, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ];
    checkStorageStages(rhi, frameEntries);
    this.materials = new MaterialRegistry(rhi, {
      capacity: 1024,
      frameTextures: frameEntries.filter((e) => e.texture && (e.visibility & GPUShaderStage.FRAGMENT)).length,
    });
    this.frameLayout = rhi.device.createBindGroupLayout({ label: 'frame', entries: frameEntries });
    this.drawLayout = rhi.device.createBindGroupLayout({
      label: 'draw',
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: BATCH_BYTES },
      }],
    });
    this.pipelineLayout = createPipelineLayout(rhi.device, {
      [GROUP_FRAME]: this.frameLayout,
      [GROUP_MATERIAL]: this.materials.layout,
      [GROUP_DRAW]: this.drawLayout,
    }, 'pbr');
    // The extended variants' own: the material group with its extension slots.
    this.extendedPipelineLayout = createPipelineLayout(rhi.device, {
      [GROUP_FRAME]: this.frameLayout,
      [GROUP_MATERIAL]: this.materials.extendedLayout,
      [GROUP_DRAW]: this.drawLayout,
    }, 'pbr-extended');

    this.frameBuffer = createBuffer(rhi, {
      label: 'frame', size: FRAME_BYTES + SHEEN_ALBEDO.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // The sheen albedo table rides at the end of the frame's uniform: it never
    // changes, so it is written once here and the per-frame write stops short.
    rhi.queue.writeBuffer(this.frameBuffer, FRAME_BYTES, SHEEN_ALBEDO);
    this.frameData = new Float32Array(FRAME_BYTES / 4);
    // The directional lights that do not cast the shadow. Grown to the scene's
    // count on demand; one entry to start, since a binding cannot be empty.
    this.directionalCapacity = 1;
    this.directionalBuffer = this._createDirectionalBuffer(1);
    // Same memory, integer view: the cluster grid dimensions are u32 in WGSL.
    this.frameU32 = new Uint32Array(this.frameData.buffer);

    this.frustum = frustumCreate();

    /**
     * Pipelines by variant: the forward ones and the OIT ones, first without
     * reflection probes compiled in, then -- once a scene has captured any --
     * with. Each frame draws from the set its scene needs; see PROBES in
     * pbr.js for what carrying the lookup costs a scene without probes.
     */
    this._variantSets = new Map([[0, { forward: new Map(), oit: new Map(), ready: true }]]);
    this._pipelineByVariant = this._variantSets.get(0).forward;
    this._oitPipelineByVariant = this._variantSets.get(0).oit;
    /**
     * Weighted-blended order-independent transparency, off by default.
     *
     * The sorted path is EXACT for separated convex objects and has no answer
     * for interpenetrating ones; this is approximate everywhere and needs no
     * order. Different tools, so this is a choice rather than a replacement --
     * architectural glass wants the sorted path, smoke and foliage want this.
     */
    this.oit = oit;
    /**
     * Screen-space ambient occlusion (ao.js), off by default: `true`, or
     * { radius } in world units. Without a radius it is a thirty-second of
     * the scene's bounding radius, found each frame -- the reach of a contact
     * shadow has to be picked by someone, and this picks it relative to the
     * scene so that a helmet and a cathedral both get one in proportion.
     */
    this.ao = ao ? { radius: ao.radius ?? null } : null;
    this._frameBindGroups = new WeakMap();
    this._clusterRevision = 0;

    /**
     * How far shadows are fitted and clustered lights resolved, in world units.
     *
     * Null means derive from the scene, which is the default and the only
     * answer that is not a guess about scale. Both used to be a flat 60 -- true
     * of Sponza and of nothing an order of magnitude either side of it, while
     * the culler maintained the world bounds that answer the question every
     * frame and nothing read them. A number you have to know to change is the
     * failure this engine's design rule names.
     *
     * Pass a number to pin either one.
     */
    this.shadowDistance = shadowDistance ?? null;
    this.lightDistance = lightDistance ?? null;

    this._sceneMin = new Float32Array(3);
    this._sceneMax = new Float32Array(3);
    this._hasSceneBounds = false;
    this._boundsRevision = -1;
    /**
     * Batches in draw order, sorted by pipeline then material.
     *
     * Sorting BATCHES rather than objects is what the sort key does, because
     * the GPU decides which objects survive. The list is far shorter -- one
     * entry per (primitive, material) pair rather than per instance -- and it
     * is rebuilt only when the scene's contents change.
     *
     * The depth field is zero for every batch on purpose: a batch has no single
     * depth, so state grouping is all this ordering can give. Front-to-back for
     * early-Z would need a depth prepass, not a different sort.
     */
    this.batchList = new DrawList(maxDraws);
    this._batchRevision = -1;

    // Blended draws, rebuilt every frame rather than on scene change: their
    // order is a function of where the camera is, so nothing about it survives
    // a frame. One entry per blended object, not per batch.
    this.transparentList = new DrawList(maxDraws);
    this._transparentOrder = new Uint32Array(maxDraws);
    /** Joint matrices for every skinned instance, rebuilt each frame. */
    this.skinPalette = new SkinPalette(rhi);
    /** Morph deltas (static, per primitive) and weights (per frame, per instance). */
    this.morph = new MorphStore(rhi);
    this._paletteRevision = 0;

    /**
     * GPU milliseconds per pass, to sit beside the CPU phases below.
     *
     * Off unless asked for (`gpuTiming: true`, or a Benchmark for the length of
     * its run): every pass would stamp two timestamps and copy them back, for
     * a number nothing in the engine reads. Switchable at any time through
     * `gpuTiming.enabled`. Inert on a device without `timestamp-query`.
     */
    this.gpuTiming = new GpuProfiler(rhi, { enabled: gpuTiming });
    this.graph = new RenderGraph(rhi, { profiler: this.gpuTiming });
    // Bound once: the graph holds a function per pass, and rebuilding these
    // every frame would allocate a closure per pass per frame.
    this._forwardEarly = (pass) => this._encodeForward(pass, 0);
    this._oitExecute = (pass) => this._encodeOIT(pass);
    this._forwardLate = (pass) => this._encodeForward(pass, 1);
    this._forwardBlend = (pass) => {
      pass.setBindGroup(GROUP_FRAME, this._frameBindGroup(this._frameEnvironment));
      this.pipelineLayout.bindEmptyGroups(pass);
      this._encodeTransparent(pass, this._pipelineByVariant, null, -1, false);
    };
    this._forwardTransmission = (pass) => {
      pass.setBindGroup(GROUP_FRAME, this._frameBindGroup(this._frameEnvironment));
      this.pipelineLayout.bindEmptyGroups(pass);
      this._encodeTransparent(pass, this._pipelineByVariant, null, -1, true);
    };
    /** The opaque scene, for transmissive surfaces; see transmission.js. */
    this.opaqueCopy = new OpaqueCopy(rhi);
    /** Each scene's reflection probes on the GPU; see probes.js. */
    this._probeSets = new WeakMap();
    /** What the frame binds for probes when a scene has none: an empty set. */
    this._noProbes = {
      view: createTexture(rhi, {
        label: 'no-probes', size: [1, 1, 6], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING,
      }).createView({ dimension: 'cube-array' }),
      buffer: createBuffer(rhi, { label: 'no-probes', size: PROBE_FLOATS * 4, usage: GPUBufferUsage.STORAGE }),
    };
    this._frameProbes = null;
    /** Every scene's decals go through this one set, packed each frame. */
    this.decals = new DecalSet(rhi);
    this._decalRevision = this.decals.revision;
    /** How many transmissive items this frame draws, from _orderTransparent. */
    this._transmissiveCount = 0;
    /** Whether blended items draw at the end of forward:late, or in a pass of their own. */
    this._blendInLate = true;
    this._frameScene = null;
    this._frameEnvironment = null;

    this.stats = { renderables: 0, draws: 0, recomposed: 0, transparent: 0, transparentDraws: 0 };
    /**
     * CPU milliseconds per phase, for the optimization pass.
     *
     * Cheap enough to leave on: five performance.now() calls a frame against a
     * frame budget of 16.7ms. Guessing which phase is expensive is how people
     * optimize the wrong one.
     */
    this.timing = { transforms: 0, upload: 0, graph: 0, encode: 0, total: 0 };

    /**
     * A fine-grained CPU profiler, or null. Nothing sets it but a Benchmark
     * (src/bench.js). While it is null each phase boundary below costs one
     * null check: the marks are `p?.mark(...)`, which does not even evaluate
     * its argument. See bench.js for what it measures and how to read it.
     */
    this.profiler = null;
  }

  async _init() {
    this.gpu = await GpuDriven.create(this.rhi, this.maxDraws, this.materials);
    this._makeDrawBindGroup();
    this.clusters = await ClusteredLights.create(this.rhi);
    // Two modules: the plain one declares no extension textures, so its
    // pipelines take the core material layout.
    [this.shader, this.extendedShader] = await Promise.all([
      compileShader(this.rhi.device, pbrShader(0), 'pbr.wgsl'),
      compileShader(this.rhi.device, pbrShader(this.materials.extensionSlots), 'pbr-extended.wgsl'),
    ]);
    if (this.oit) await this._initOit();
    this.shadows = await ShadowMaps.create(
      this.rhi, this.pipelines, this.drawLayout, this.shadowOptions, this.materials.layout,
    );
    this.skybox = await SkyboxPass.create(this.rhi, this.pipelines, this.ao ? AMBIENT_FORMAT : null);
    if (this.ao) this.aoPass = await AmbientOcclusion.create(this.rhi, this.pipelines);
    /**
     * Whether the environment is drawn as the background.
     *
     * A plain mutable field: per-renderer state a caller reads
     * and writes, not hidden configuration. Turning it off leaves the clear
     * colour showing and changes NOTHING about lighting -- the same cubemap is
     * still the ambient term, because the sky IS the light. Setting the sky
     * colours to black would turn the background off too, and take the
     * lighting with it.
     */
    this.drawSkybox = true;
    /**
     * Fog, or null for none: { visibility, height, scaleHeight, albedo } --
     * see fogCoefficients in fog.js. A plain field like drawSkybox, checked
     * each frame it is used.
     */
    this.fog = this._fogOption;
    this._fogData = new Float32Array(12);
    this.post = await PostStack.create(this.rhi, this.pipelines, this.postOptions);
    /** Lines for one frame, drawn over the finished picture; see debug.js. */
    this.debug = await DebugLines.create(this.rhi, this.pipelines);
    this.sprites = await SpritePass.create(this.rhi, this.pipelines, this.frameBuffer, HDR_FORMAT);
    this.particles = await ParticleSystem.create(this.rhi, this.pipelines, this.frameBuffer, HDR_FORMAT);
    this.dofPass = await DepthOfField.create(this.rhi, this.pipelines, HDR_FORMAT);
    /**
     * Depth of field, or null for none: { focusDistance, fStop, sensorHeight }
     * -- see render/dof.js. A plain field, like fog.
     */
    this.dof = this._dofOption;
    this.hzb = await HierarchicalDepth.create(this.rhi, this.pipelines);
    // The opaque, single-sided variant is what almost every asset uses; having
    // it ready means the first model added never waits on a compile.
    await this.ensureVariants([0]);
  }

  /**
   * Rebuild everything that names a GpuDriven buffer.
   *
   * Growing destroys and replaces those buffers, and TWO of this renderer's
   * bind groups name them: the draw group (batchBuffer) and the frame group
   * (drawDataBuffer and the visible list, bindings 9 and 10). Missing either
   * one shows up as "used in submit while destroyed" several frames later,
   * which is why both are invalidated from one place off one revision.
   */
  _makeDrawBindGroup() {
    this.drawBindGroup = this.rhi.device.createBindGroup({
      label: 'batch', layout: this.drawLayout,
      entries: [{ binding: 0, resource: { buffer: this.gpu.batchBuffer, size: BATCH_BYTES } }],
    });
    this._frameBindGroups = new WeakMap();
    this._drawBindGroupRevision = this.gpu.buffersRevision;
  }

  /**
   * Compile the pipelines for a set of material variants.
   *
   * Called by engine.load() before an asset is handed back, so that by the time
   * anything is in a scene every pipeline it needs already exists. Nothing in
   * render() is allowed to create one.
   */
  async ensureVariants(variants) {
    // Both windings of each, because whether an instance mirrors is a property
    // of the scene and is not known here -- and the contract of this method is
    // that nothing in render() ever has to create a pipeline.
    const wanted = [];
    for (const variant of variants) {
      const base = variant & ~(VARIANT_MIRRORED | VARIANT_SKINNED);
      wanted.push(base, base | VARIANT_MIRRORED,
        base | VARIANT_SKINNED, base | VARIANT_MIRRORED | VARIANT_SKINNED);
    }
    // Every set in use: a material loaded after probes were captured needs
    // its pipelines every way the scenes may draw it.
    for (const features of this._variantSets.keys()) {
      await this._ensureVariantSet(wanted, features);
    }
  }

  /**
   * The pipelines with some features compiled in -- FEATURE_PROBES,
   * FEATURE_DECALS -- built the first time a frame needs them, for every
   * variant then known. Until they are ready a frame draws with the largest
   * set that is, so render() itself never compiles.
   */
  async _enableFeatures(features) {
    const existing = this._variantSets.get(features);
    if (existing) return existing.building;
    const set = { forward: new Map(), oit: new Map(), ready: false };
    this._variantSets.set(features, set);
    set.building = this._ensureVariantSet([...this._variantSets.get(0).forward.keys()], features)
      .then(() => { set.ready = true; });
    return set.building;
  }

  /** The largest ready set within these features: every one, or a subset, or none. */
  _readySet(features) {
    for (let subset = features; ; subset = (subset - 1) & features) {
      const set = this._variantSets.get(subset);
      if (set?.ready) return set;
      if (subset === 0) return this._variantSets.get(0);
    }
  }

  /** The module and layout a variant's pipelines use; see extendedPipelineLayout. */
  _shaderFor(variant) {
    return (variant & VARIANT_EXTENDED) === 0
      ? { layout: this.pipelineLayout, shader: this.shader }
      : { layout: this.extendedPipelineLayout, shader: this.extendedShader };
  }

  async _ensureVariantSet(wanted, features) {
    const set = this._variantSets.get(features);
    const pending = [];
    for (const variant of wanted) {
      // Every wanted variant goes to warm(), built or not: one another load
      // is still compiling has to be waited on, and warm() knows which.
      const known = set.forward.get(variant);
      if (known) { pending.push(known); continue; }

      const state = variantPipelineState(variant);
      const skinned = (variant & VARIANT_SKINNED) !== 0;
      // With ambient occlusion on, opaque and masked surfaces also write their
      // ambient term, for the occlusion pass to take its share back out.
      // Blended ones draw after it, in a pass with the one target.
      const ambient = this.ao !== null && (variant & 3) !== ALPHA_BLEND && (variant & VARIANT_TRANSMISSIVE) === 0;
      const descriptor = {
        label: `pbr:v${variant}:f${features}`,
        ...this._shaderFor(variant),
        vertexEntry: skinned ? 'vsSkinned' : 'vs',
        fragmentEntry: ambient ? 'fsAO' : undefined,
        buffers: skinned ? [VERTEX_LAYOUT, SKIN_BUFFER_LAYOUT] : [VERTEX_LAYOUT],
        targets: ambient
          ? [{ format: HDR_FORMAT, blend: state.blend }, { format: AMBIENT_FORMAT }]
          : [{ format: HDR_FORMAT, blend: state.blend }],
        primitive: state.primitive,
        depth: state.depth,
        constants: { ...state.constants, ...featureConstants(features) },
      };
      set.forward.set(variant, descriptor);
      pending.push(descriptor);
    }
    await this.pipelines.warm(pending);
    if (this.oit) await this._ensureOitVariants(wanted, features);
  }

  /**
   * The OIT copies of every BLEND variant.
   *
   * Only those: opaque and masked geometry never reaches the transparent path,
   * so building them would compile pipelines nothing can bind. Two targets and
   * two blend states rather than one, which is why these cannot just be the
   * same descriptors with a different entry point.
   */
  async _ensureOitVariants(variants, features = 0) {
    const set = this._variantSets.get(features);
    const pending = [];
    for (const variant of variants) {
      // Transmissive ones never go to OIT: they draw in their own pass.
      if ((variant & 3) !== ALPHA_BLEND || (variant & VARIANT_TRANSMISSIVE) !== 0) continue;
      const known = set.oit.get(variant);
      if (known) { pending.push(known); continue; }

      const state = variantPipelineState(variant);
      // Skinned variants skin, exactly as the forward ones do. These used to
      // be built without the entry point or the joint buffer, so a skinned
      // blended mesh under OIT drew in its rest pose.
      const skinned = (variant & VARIANT_SKINNED) !== 0;
      const descriptor = {
        label: `pbr-oit:v${variant}:f${features}`,
        ...this._shaderFor(variant),
        vertexEntry: skinned ? 'vsSkinned' : 'vs',
        buffers: skinned ? [VERTEX_LAYOUT, SKIN_BUFFER_LAYOUT] : [VERTEX_LAYOUT],
        fragmentEntry: 'fsOIT',
        targets: [
          // accum sums, so it adds; reveal multiplies what is left of the
          // background, so it is a product. Both are the standard weighted
          // blended pair and neither is a choice.
          { format: OIT_ACCUM_FORMAT, blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          } },
          { format: OIT_REVEAL_FORMAT, blend: {
            color: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
          } },
        ],
        primitive: state.primitive,
        // Tested against the opaque depth so nothing behind a wall accumulates,
        // and never written: a blended surface does not occlude what is behind
        // it, and the whole point of this path is that order does not matter.
        depth: { format: DEPTH_FORMAT, depthCompare: DEPTH_COMPARE, depthWriteEnabled: false },
        constants: { ...state.constants, ...featureConstants(features) },
      };
      set.oit.set(variant, descriptor);
      pending.push(descriptor);
    }
    await this.pipelines.warm(pending);
  }

  /**
   * A scene's probe set, made at its environment's resolution and mips so one
   * roughness indexes probes and sky alike. Made again -- every probe then
   * waiting for a capture -- if the environment changes size.
   */
  _probesFor(scene, environment) {
    if (scene.reflectionProbes.length === 0) return null;
    let set = this._probeSets.get(scene);
    if (set && (set.size !== environment.size || set.mips !== environment.prefilterMips)) {
      set.destroy();
      for (const probe of scene.reflectionProbes) probe.captured = false;
      set = undefined;
    }
    if (!set) {
      set = new ProbeSet(this.rhi, environment.size, environment.prefilterMips);
      this._probeSets.set(scene, set);
    }
    if (set.sceneRevision !== scene.probeRevision) {
      set.upload(scene.reflectionProbes);
      set.sceneRevision = scene.probeRevision;
    }
    return set;
  }

  /**
   * Render each probe's six faces from where it stands, prefilter them as the
   * environment is prefiltered, and store them in the scene's probe set.
   * Every probe the scene has unless given a list. A load-time cost: six
   * scene renders and a prefilter each.
   */
  async captureReflectionProbes(scene, probes = scene.reflectionProbes) {
    const environment = scene.environment;
    if (!environment) throw new Error('captureReflectionProbes: the scene has no environment');
    if (probes.length === 0) return;
    // Built before any frame reads probes, so render() never has to compile.
    await this._enableFeatures(FEATURE_PROBES);
    const set = this._probesFor(scene, environment);
    const size = environment.size;
    if (this._captureColor?.width !== size) {
      this._captureColor?.destroy();
      this._captureDepth?.destroy();
      this._captureColor = createTexture(this.rhi, {
        label: 'probe-capture', size: [size, size], format: HDR_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this._captureDepth = createTexture(this.rhi, {
        label: 'probe-capture-depth', size: [size, size], format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    }
    const target = {
      width: size, height: size,
      colorView: this._captureColor.createView(), depthView: this._captureDepth.createView(),
    };
    for (const probe of probes) {
      const cube = new Environment(this.rhi, {
        capture: true, size, prefilterMips: environment.prefilterMips, label: 'probe',
      });
      // Near enough for anything in the box: a thousandth of its diagonal.
      const near = 1e-3 * Math.hypot(probe.max[0] - probe.min[0], probe.max[1] - probe.min[1], probe.max[2] - probe.min[2]);
      FACE_CAMERAS.forEach(({ forward, up }, face) => {
        const camera = new Camera({ fovY: Math.PI / 2, near });
        camera.position.set(probe.position);
        camera.target.set([0, 1, 2].map((a) => probe.position[a] + forward[a]));
        camera.up.set(up);
        this.render(scene, camera, null, target);
        flipInto(this.rhi, target.colorView, cubeFaceView(cube.environment, face));
      });
      cube.convolve();
      set.store(set.slotFor(probe), cube.prefiltered);
      cube.destroy();
      probe.captured = true;
    }
    scene.probeRevision++;
  }

  _frameBindGroup(environment) {
    let bindGroup = this._frameBindGroups.get(environment);
    if (!bindGroup) {
      bindGroup = this.rhi.device.createBindGroup({
        label: 'frame',
        layout: this.frameLayout,
        entries: [
          { binding: 0, resource: { buffer: this.frameBuffer } },
          { binding: 1, resource: environment.irradianceView },
          { binding: 2, resource: environment.prefilteredView },
          { binding: 3, resource: environment.sampler },
          { binding: 4, resource: this.shadows.view },
          { binding: 5, resource: this.shadows.sampler },
          { binding: 6, resource: { buffer: this.clusters.lightBuffer } },
          { binding: 7, resource: { buffer: this.clusters.indexBuffer } },
          { binding: 8, resource: { buffer: this.clusters.countBuffer } },
          { binding: 9, resource: { buffer: this.gpu.drawDataBuffer } },
          { binding: 10, resource: { buffer: this.gpu.visibleBuffer } },
          { binding: 11, resource: { buffer: this.skinPalette.buffer } },
          { binding: 12, resource: { buffer: this.morph.deltaBuffer } },
          { binding: 13, resource: { buffer: this.morph.weightBuffer } },
          { binding: 14, resource: { buffer: this.directionalBuffer } },
          { binding: 15, resource: this.shadows.localView },
          { binding: 16, resource: { buffer: this.shadows.localBuffer } },
          { binding: 17, resource: { buffer: this.shadows.cascadeList } },
          { binding: 18, resource: this.opaqueCopy.view },
          { binding: 19, resource: this._frameProbes?.view ?? this._noProbes.view },
          { binding: 20, resource: { buffer: this._frameProbes?.buffer ?? this._noProbes.buffer } },
          { binding: 21, resource: this.decals.view },
          { binding: 22, resource: { buffer: this.decals.buffer } },
        ],
      });
      this._frameBindGroups.set(environment, bindGroup);
    }
    return bindGroup;
  }

  _createDirectionalBuffer(count) {
    return createBuffer(this.rhi, {
      label: 'directionals',
      size: count * DIRECTIONAL_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * `target`: render into { width, height, colorView, depthView } instead of
   * the canvas -- linear HDR, before post, and with no reflection probes read,
   * since a probe capture is what renders into one. See captureReflectionProbes.
   */
  render(scene, camera, jobs = null, target = null) {
    const rhi = this.rhi;
    const environment = scene.environment;
    if (!environment) throw new Error('Renderer: the scene has no environment');
    const width = target?.width ?? rhi.width;
    const height = target?.height ?? rhi.height;
    const depthView = target?.depthView ?? rhi.depthView();

    // This scene's probes, uploaded again when its set of them changed. None
    // while capturing: the probe being captured would be read half-written.
    const probes = target === null ? this._probesFor(scene, environment) : null;
    // The pipelines that read probes and decals only when there are any to
    // read. A set not built yet starts building, and this frame draws
    // without what it adds.
    const decalCount = this.decals.prepare(scene);
    const features = (probes !== null && probes.count > 0 ? FEATURE_PROBES : 0) | (decalCount > 0 ? FEATURE_DECALS : 0);
    if (!this._variantSets.has(features)) this._enableFeatures(features).catch((error) => console.error(error));
    const variantSet = this._readySet(features);
    this._pipelineByVariant = variantSet.forward;
    this._oitPipelineByVariant = variantSet.oit;
    if (this.decals.revision !== this._decalRevision) {
      this._decalRevision = this.decals.revision;
      this._frameBindGroups = new WeakMap();
    }
    if (probes !== this._frameProbes || probes?.revision !== this._frameProbesRevision) {
      this._frameProbes = probes;
      this._frameProbesRevision = probes?.revision;
      this._frameBindGroups = new WeakMap();
    }

    const p = this.profiler;
    p?.frameStart();
    const tFrame = now();
    this.stats.recomposed = scene.update(jobs);
    const tAfterTransforms = now();
    p?.mark('transforms');
    camera.update(width / height);
    frustumFromViewProjection(this.frustum, camera.viewProjection);
    p?.mark('camera');

    const count = scene.renderableCount;
    this.stats.renderables = count;

    // Only a compose that moved something can have changed a world box.
    const moved = scene.transforms.movedPending ? updateWorldBounds(
      count, scene.localMin, scene.localMax, scene.worldMin, scene.worldMax,
      scene.transforms.world, scene.renderableMatrixSlot, scene.transforms.moved,
    ) : 0;

    // Skinned renderables get their bounds replaced: the pass above gave them
    // a bind-pose box transformed by a model matrix the vertices do not follow.
    // Counted like `moved`: a skeleton moves its mesh's box with no transform
    // of the mesh's own changing, so the union below has to hear about it.
    let skinned = 0;
    if (scene.skins.length > 0) {
      updateSkinBounds(scene.skins, scene.transforms.world);
      skinned = applySkinBounds(count, scene.renderableSkin, scene.skins, scene.worldMin, scene.worldMax);
    }

    // And morphed ones get theirs grown, by how far their weights can carry a
    // vertex. Counted into `moved`, because a weight changes without any
    // transform changing and the union below is derived from both.
    const morphed = scene.morphs.length > 0 ? scene.applyMorphBounds() : 0;

    // The scene's own extent, which is what the shadow and cluster ranges are
    // derived from. Recomputed only when something moved or the contents
    // changed -- a union cannot be updated in place, because a renderable that
    // moves can shrink it as easily as grow it, but on a settled scene that
    // means never paying for it at all.
    if (moved > 0 || morphed > 0 || skinned > 0 || this._boundsRevision !== scene.revision) {
      this._hasSceneBounds = unionWorldBounds(
        count, scene.worldMin, scene.worldMax, this._sceneMin, this._sceneMax,
      );
      this._boundsRevision = scene.revision;
    }
    p?.mark('bounds');

    // Culling happens on the GPU. Nothing here reads back which objects survived
    // -- that answer only ever exists in the indirect argument buffer, which
    // is exactly the point: a readback would cost a pipeline stall.
    // The pyramid is sized to the surface, so it is rebuilt on resize. Doing it
    // before the cull means the bind group always points at a live texture.
    this.hzb.resize(width, height, depthView);
    this.gpu.bindHzb(this.hzb.view);
    // Weights first: draw data records where each instance's slice begins,
    // and the gather below is what decides those offsets.
    this.morph.update(scene);
    this.gpu.update(scene, this.frustum, this.hzb, camera.viewProjection, writeDrawData,
      this.skinPalette.offsets, this.morph, camera.projection[5]);
    if (this._drawBindGroupRevision !== this.gpu.buffersRevision) this._makeDrawBindGroup();
    p?.mark('draw data');

    // Palettes are rebuilt from this frame's pose, before anything reads them.
    // A grow replaces the buffer the frame group names, so that invalidates
    // the cache the same way a gpu grow does.
    this.skinPalette.update(scene);
    // A grown morph buffer invalidates the frame group for the same reason a
    // grown palette does: the group names a buffer that no longer exists.
    if (this._morphRevision !== this.morph.revision) {
      this._frameBindGroups = new WeakMap();
      this._morphRevision = this.morph.revision;
    }
    if (this._paletteRevision !== this.skinPalette.revision) {
      this._frameBindGroups = new WeakMap();
      this._paletteRevision = this.skinPalette.revision;
    }

    // Both consumers of `moved` have now read it, so the record is spent.
    // Clearing here rather than in update() is what makes scene.update() safe
    // to call any number of times before a frame.
    if (scene.transforms.movedPending) {
      scene.transforms.moved.fill(0, 0, scene.transforms.capacity);
      scene.transforms.movedPending = false;
    }

    const tAfterUpload = now();
    p?.mark('skin + morph');

    if (this._batchRevision !== this.gpu.sceneRevision) {
      this.batchList.clear();
      for (let b = 0; b < this.gpu.batchCount; b++) {
        const materialId = this.gpu.batchMaterial[b];
        // Winding is pipeline state, so it belongs in the pipeline field. Two
        // ids per material variant, which keeps the worst case at 12 of the 16
        // the narrower key can address.
        // Winding and skinning are both pipeline state, so both belong in the
        // pipeline field or the key claims two pipelines are one bucket.
        const pipelineId = (this.materials.pipelineIdOf[materialId] * 2
          + this.gpu.batchMirrored[b]) * 2 + this.gpu.batchSkinned[b];
        this.batchList.push(opaqueSortKey(pipelineId, materialId, 0), b);
      }
      this.batchList.sort();
      this._batchRevision = this.gpu.sceneRevision;
    }
    this.stats.draws = this.gpu.batchCount;
    p?.mark('batch sort');

    this._orderTransparent(scene, camera);
    p?.mark('transparent sort');

    // These three recompute what the frame uniform is about to copy, so they
    // have to run FIRST. Filling the uniform before them uploaded the previous
    // frame's cascade matrices, splits and cluster parameters while the shadow
    // pass rasterised with this frame's -- so a moving camera looked up its
    // shadows in the wrong patch of the map, and the first frame of all read
    // matrices that were still zero.
    // Derived per frame unless pinned. The scene's far corner in view depth is
    // exactly how far there is anything to shadow or light; beyond it both
    // would be resolving empty space.
    //
    // The floor is the smallest legal value rather than a chosen one: both
    // consumers require strictly more than the near plane, so with nothing in
    // the scene -- where the number cannot matter -- twice near is the minimum
    // that satisfies them.
    //
    // The two ranges are derived differently on purpose. Clusters want the
    // tight view-space depth, and re-slicing them as the camera turns costs
    // nothing visible. Shadows cannot follow the view direction at all: every
    // change resizes the cascades and makes their edges crawl, so theirs is
    // radial and held to powers of two (see stableShadowDistance).
    const floor = camera.near * 2;
    const sceneDepth = this._hasSceneBounds
      ? farthestViewDepth(camera.view, this._sceneMin, this._sceneMax)
      : 0;
    const sceneReach = this._hasSceneBounds
      ? farthestDistance(camera.position, this._sceneMin, this._sceneMax)
      : 0;
    const shadowRange = this.shadowDistance ?? stableShadowDistance(sceneReach, floor);
    const lightRange = this.lightDistance ?? Math.max(sceneDepth, floor);

    this.shadows.shadowDistance = shadowRange;
    // Lights are scene objects: their position and aim live in their
    // transforms, which have composed by now. Copied into the packed array
    // here, immediately before upload, so a light parented to something that
    // moved this frame is lit from where it is rather than where it was.
    // Before the shadow fit, which reads the directional lights' directions.
    scene.refreshLights();
    // After the lights, whose colours the fog scatters.
    packFog(this._fogData, 0, this.fog, scene.directionals, scene.directionalCount, DIRECTIONAL_FLOATS);
    this.skybox.update(camera, 1.0, this.fog === null ? null : this._fogData);
    // Materials a clip changed since the last frame. Uploaded here rather than
    // by the player, because the scene does not own the GPU.
    if (scene.changedMaterials.size > 0) {
      for (const [id, record] of scene.changedMaterials) this.materials.update(id, record);
      scene.changedMaterials.clear();
    }
    p?.mark('lights');
    // Every casting light's shadow views: the directional lights' cascades,
    // then point and spot views. Each writes its slot into the light's record,
    // so both run before the records are uploaded. Growing either array
    // replaces what the frame group names.
    this.shadows.update(camera, scene);
    this.shadows.updateLocal(scene, camera);
    if (this._shadowRevision !== this.shadows.revision) {
      this._frameBindGroups = new WeakMap();
      this._shadowRevision = this.shadows.revision;
    }
    // The directional records go up after the fit, which wrote their slots.
    if (scene.directionalCount > this.directionalCapacity) {
      this.directionalBuffer.destroy();
      this.directionalCapacity = grownCapacity(this.directionalCapacity, scene.directionalCount);
      this.directionalBuffer = this._createDirectionalBuffer(this.directionalCapacity);
      this._frameBindGroups = new WeakMap();   // they name the buffer just replaced
    }
    if (scene.directionalCount > 0) {
      rhi.queue.writeBuffer(this.directionalBuffer, 0, scene.directionals, 0,
        scene.directionalCount * DIRECTIONAL_FLOATS);
    }
    p?.mark('shadow fit');
    this.clusters.update(scene, camera, lightRange, width, height, this.decals.spheres, decalCount);
    // Growing the light list replaced lightBuffer, which every cached frame
    // bind group names. Dropping the cache rebuilds them on next use.
    if (this._clusterRevision !== this.clusters.buffersRevision) {
      this._frameBindGroups = new WeakMap();
      this._clusterRevision = this.clusters.buffersRevision;
    }

    // --- frame uniform ------------------------------------------------------
    this.frameData.set(camera.viewProjection, 0);
    this.frameData.set(camera.position, 16);
    // Exposure moved to the tonemap pass, where it applies BEFORE the curve.
    // Scaling an already-compressed image would only wash it out.
    this.frameData[19] = 1.0;
    this.frameData[20] = environment.prefilterMips;
    // Cascades a casting directional light has, zero when none casts. Their
    // matrices are per light, in the cascade list; the slices are shared.
    this.frameData[24] = this.shadows.activeCascades;
    this.frameData.set(this.shadows.splits, 28);
    this.frameData.set(this.shadows.texelSizes, 32);
    this.frameData[36] = this.shadows.normalBias;
    this.frameData[37] = this.shadows.size;
    // Where the first cascade's slice begins, for the tap spacing that walks
    // each cascade's blur up to the next one's (directionalVisibility in pbr.js).
    this.frameData[38] = camera.near;
    this.frameData[39] = this.shadows.localSize;
    if (this.ao !== null) {
      const radius = this.ao.radius ?? (this._hasSceneBounds
        ? 0.5 * Math.hypot(this._sceneMax[0] - this._sceneMin[0], this._sceneMax[1] - this._sceneMin[1],
          this._sceneMax[2] - this._sceneMin[2]) / 32
        : 1);
      this.aoPass.update(camera, radius, width, height);
    }

    // Cluster grid dims are u32 in the shader, so they are written through a
    // Uint32 view of the same buffer rather than as floats.
    // The grid follows the viewport aspect, so the shader is told the shape
    // this frame has rather than a constant it would disagree with.
    this.frameU32[40] = this.clusters.gridX;
    this.frameU32[41] = this.clusters.gridY;
    this.frameU32[42] = CLUSTER_Z;
    this.frameU32[43] = this.clusters.lightCount;
    this.frameData[44] = this.clusters.sliceScale;
    this.frameData[45] = this.clusters.sliceBias;
    this.frameData[46] = this.clusters.tileSize[0];
    this.frameData[47] = this.clusters.tileSize[1];
    // The view axis, for view depth in the shader. The view matrix's third row
    // is the camera's +Z in world space; the camera looks down -Z.
    this.frameData[48] = -camera.view[2];
    this.frameData[49] = -camera.view[6];
    this.frameData[50] = -camera.view[10];
    this.frameData[51] = scene.directionalCount;   // a value; see the shader
    this.frameData.set(this._fogData, 52);
    this.frameData[64] = probes?.count ?? 0;
    rhi.queue.writeBuffer(this.frameBuffer, 0, this.frameData);
    p?.mark('clusters + frame uniform');



    // --- declare the frame ---------------------------------------------------
    // Nothing below says what order to run in, when to clear, or when to store.
    // The graph works all three out from reads and writes.
    this._frameScene = scene;
    this._frameEnvironment = environment;

    const graph = this.graph;
    graph.begin();

    const surface = target === null ? graph.importTexture('surface', rhi.currentColorView()) : null;
    // Imported but NOT external: the device owns the memory, and nothing reads
    // it after the frame, so the graph is free to derive `discard` for it.
    const depth = graph.importTexture('depth', depthView, { external: false });
    const shadowMap = graph.importTexture('shadows', this.shadows.view);
    const localShadowMap = graph.importTexture('local-shadows', this.shadows.localView);

    const clusterBounds = graph.importBuffer('cluster-bounds', this.clusters.boundsBuffer);
    const lightBuffer = graph.importBuffer('lights', this.clusters.lightBuffer);
    const clusterIndices = graph.importBuffer('cluster-indices', this.clusters.indexBuffer);
    const clusterCounts = graph.importBuffer('cluster-counts', this.clusters.countBuffer);

    const drawDataBuffer = graph.importBuffer('draw-data', this.gpu.drawDataBuffer);
    // The same two GPU buffers, imported twice each. The graph tracks buffers
    // only to derive ordering, and the two cull phases write disjoint halves,
    // so describing them as one resource would be a lie that costs a cycle:
    // forward:early reads the indirect args, and a read edges from EVERY
    // writer, so it would be forced after cull:late -- which must itself come
    // after forward:early.
    const indirectEarly = graph.importBuffer('indirect:early', this.gpu.indirectBuffer);
    const visibleEarly = graph.importBuffer('visible:early', this.gpu.visibleBuffer);
    const indirectLate = graph.importBuffer('indirect:late', this.gpu.indirectBuffer);
    const visibleLate = graph.importBuffer('visible:late', this.gpu.visibleBuffer);

    this.gpu.addCullPass(graph, {
      phase: 0,
      boundsResource: drawDataBuffer,
      indirectResource: indirectEarly,
      visibleResource: visibleEarly,
    });

    // Shadows draw the level of detail the camera chose. Only when there is
    // any: otherwise they walk the static order, and this pass is not run.
    const shadowReads = [];
    if (this.gpu.hasLod) {
      const indirectShadow = graph.importBuffer('indirect:shadow', this.gpu.indirectBuffer);
      const visibleShadow = graph.importBuffer('visible:shadow', this.gpu.visibleBuffer);
      this.gpu.addCullPass(graph, {
        phase: CULL_SHADOW,
        boundsResource: drawDataBuffer,
        indirectResource: indirectShadow,
        visibleResource: visibleShadow,
      });
      shadowReads.push(indirectShadow, visibleShadow);
    }
    this.shadows.addPasses(
      graph, shadowMap, this.gpu, this.drawBindGroup, this.skinPalette, this.morph, localShadowMap, shadowReads,
    );
    this.clusters.addPasses(graph, {
      boundsResource: clusterBounds,
      lightsResource: lightBuffer,
      indicesResource: clusterIndices,
      countsResource: clusterCounts,
    });

    // Geometry renders into a linear HDR target, not the swap chain. Bloom has
    // to see that a highlight was at 60x white rather than clipped to 1, and
    // one pass at the end tonemaps the result into the sRGB surface.
    // A target takes the HDR picture itself: nothing after this is for it.
    const sceneColor = target !== null ? graph.importTexture('capture', target.colorView) : graph.createTexture('scene-hdr', {
      width,
      height,
      format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    // The opaque passes' ambient term, when ambient occlusion is on.
    const ambient = this.ao === null ? null : graph.createTexture('ambient', {
      width,
      height,
      format: AMBIENT_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const black = { r: 0, g: 0, b: 0, a: 1 };

    // EARLY. Everything that was on screen last frame, which is both the
    // picture so far and the set of occluders the pyramid is built from.
    graph.addPass({
      name: 'forward:early',
      reads: [shadowMap, localShadowMap, lightBuffer, clusterIndices, clusterCounts, indirectEarly, visibleEarly],
      color: ambient === null
        ? [{ resource: sceneColor, clear: black }]
        : [{ resource: sceneColor, clear: black }, { resource: ambient, clear: { r: 0, g: 0, b: 0, a: 0 } }],
      depth: { resource: depth, clear: DEPTH_CLEAR_VALUE },
      execute: this._forwardEarly,
    });

    // Built from the early depth and consumed later in the SAME frame, which
    // is the whole difference: the occlusion test is no longer a frame behind.
    const hzbLevels = [];
    for (let level = 0; level < this.hzb.levelCount; level++) {
      hzbLevels.push(graph.importTexture(`hzb${level}`, this.hzb.levelViews[level]));
    }
    this.hzb.addPasses(graph, depth, hzbLevels);

    this.gpu.addCullPass(graph, {
      phase: 1,
      boundsResource: drawDataBuffer,
      indirectResource: indirectLate,
      visibleResource: visibleLate,
      hzbResources: hzbLevels,
    });

    // Transmissive surfaces need the opaque scene finished and copied before
    // they draw, so blended ones -- which must come after them -- leave the
    // late pass for one of their own, as they already do for ambient occlusion.
    const transmissive = this._transmissiveCount > 0;
    // Sprites draw after the opaque scene and before anything blended, so
    // they push blended geometry out of the late pass too.
    const sprites = this.sprites.prepare(scene, camera, environment, width, height);
    // Particles likewise; and during a probe capture they are seen, not moved.
    const emitters = this.particles.prepare(scene, camera, environment, target !== null);
    this._blendInLate = !this.oit && ambient === null && !transmissive && sprites === 0 && emitters === 0;

    // LATE. Whatever the fresh pyramid says is visible and was not drawn above,
    // then the blended geometry, which has to follow every opaque draw. No
    // clear on either attachment: the graph derives `load` from the early pass
    // having written them.
    graph.addPass({
      name: 'forward:late',
      reads: [shadowMap, localShadowMap, lightBuffer, clusterIndices, clusterCounts, indirectLate, visibleLate],
      color: ambient === null ? [{ resource: sceneColor }] : [{ resource: sceneColor }, { resource: ambient }],
      depth: { resource: depth },
      execute: this._forwardLate,
    });

    // Ambient occlusion on the finished opaque depth, then the blended
    // geometry the late pass left for after it.
    if (ambient !== null) {
      this.aoPass.addPasses(graph, { depth, ambient, sceneColor, width, height });
    }

    // Sprites, on the finished opaque scene: before the transmission copy, so
    // glass shows what is behind it, and before blended geometry.
    this.sprites.addPass(graph, { sceneColor, depth });
    this.particles.addPasses(graph, { sceneColor, depth });

    // Transmission: the opaque scene copied down a mip chain, then the
    // transmissive surfaces, which read it.
    if (transmissive) {
      if (this.opaqueCopy.ensure(width, height)) this._frameBindGroups = new WeakMap();
      const behind = this.opaqueCopy.addPasses(graph, sceneColor);
      graph.addPass({
        name: 'forward:transmission',
        reads: [shadowMap, localShadowMap, lightBuffer, clusterIndices, clusterCounts, ...behind],
        color: [{ resource: sceneColor }],
        depth: { resource: depth },
        execute: this._forwardTransmission,
      });
    }

    if (!this.oit && !this._blendInLate) {
      graph.addPass({
        name: 'forward:blend',
        reads: [shadowMap, localShadowMap, lightBuffer, clusterIndices, clusterCounts],
        color: [{ resource: sceneColor }],
        depth: { resource: depth },
        execute: this._forwardBlend,
      });
    }

    // OIT, when it is on. Blended geometry skipped the pass above, so it is
    // drawn here into its own two targets in whatever order it comes -- that
    // is the point -- and composited over the scene by the resolve.
    if (this.oit) {
      const accum = graph.createTexture('oit-accum', {
        width,
        height,
        format: OIT_ACCUM_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      const reveal = graph.createTexture('oit-reveal', {
        width,
        height,
        format: OIT_REVEAL_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });

      graph.addPass({
        name: 'oit',
        reads: [shadowMap, localShadowMap, lightBuffer, clusterIndices, clusterCounts],
        color: [
          // accum starts empty; reveal starts at 1, meaning all background.
          { resource: accum, clear: { r: 0, g: 0, b: 0, a: 0 } },
          { resource: reveal, clear: { r: 1, g: 1, b: 1, a: 1 } },
        ],
        // Read for the depth test, never written. Declaring the depth here is
        // also what keeps the graph from discarding it after forward:late.
        depth: { resource: depth },
        execute: this._oitExecute,
      });

      graph.addPass({
        name: 'oit-resolve',
        reads: [accum, reveal],
        color: [{ resource: sceneColor }],
        execute: (pass) => this._encodeOitResolve(pass, graph.viewOf(accum), graph.viewOf(reveal)),
      });
    }

    if (target === null) {
      // Depth of field last in HDR, on everything drawn, before bloom and the tonemap.
      const lensed = this.dof ? this.dofPass.addPasses(graph, { sceneColor, depth, camera, width, height, dof: this.dof }) : sceneColor;
      this.post.addPasses(graph, { sceneColor: lensed, surface, width, height, exposure: this.exposure });
      this.debug.addPass(graph, { surface, depth, viewProjection: camera.viewProjection });
    }

    graph.compile();
    const tAfterGraph = now();
    p?.mark('graph build');

    const encoder = rhi.device.createCommandEncoder({ label: 'frame' });
    graph.execute(encoder);

    // After the last pass is recorded and before the encoder is closed: this
    // only copies queries the GPU will have written by the time it runs.
    this.gpuTiming.resolve(encoder);

    const tEnd = now();
    this.timing.transforms = tAfterTransforms - tFrame;
    this.timing.upload = tAfterUpload - tAfterTransforms;
    this.timing.graph = tAfterGraph - tAfterUpload;
    this.timing.encode = tEnd - tAfterGraph;
    this.timing.total = tEnd - tFrame;
    p?.mark('encode');
    rhi.queue.submit([encoder.finish()]);
    if (target === null) this.debug.clear();
    // After the submit, never before: the command buffer above writes the
    // buffer this maps, and a buffer with a map pending cannot be written.
    this.gpuTiming.readback();
    p?.mark('submit');
    p?.frameEnd();
  }

  /**
   * Release every GPU object this renderer owns.
   *
   * Eight subsystems here each defined a destroy() and not one was ever called,
   * so the whole teardown path existed on paper only. device.destroy() does
   * reclaim the memory, which is why nothing showed -- but it does not help the
   * cases that are not a full teardown: an Engine destroyed while another still
   * uses the device, or a profiler left holding buffers mid-mapAsync.
   *
   * Not the materials or the environment: the registry's textures come from
   * assets the caller loaded, and the environment may be shared.
   */
  destroy() {
    this.gpuTiming.destroy();
    this.graph.destroy();
    this.gpu.destroy();
    this.shadows.destroy();
    this.clusters.destroy();
    this.hzb.destroy();
    this.post.destroy();
    this.frameBuffer.destroy();
    this.directionalBuffer.destroy();
    this.skinPalette.destroy();
    this.morph.destroy();
    this.skybox.destroy();
    this.opaqueCopy.destroy();
    this.debug.destroy();
    this.sprites.destroy();
    this.particles.destroy();
    this.decals.destroy();
    this.dofPass.destroy();
    this._captureColor?.destroy();
    this._captureDepth?.destroy();
    this.aoPass?.destroy();
  }

  /**
   * Cull and order this frame's blended geometry, back to front.
   *
   * This is the one place the engine sorts on the CPU, and it is not an
   * oversight. Batched geometry is compacted by an atomic in the cull shader,
   * which hands out slots in whatever order threads happen to finish -- fine
   * when the result is order-independent, useless when it is not. Blending is
   * not order-independent, so these objects leave the batched path entirely.
   *
   * Sorting by centroid is exact for separated convex objects and wrong for
   * interpenetrating ones. No per-object ordering can fix that case; it needs
   * per-fragment work (depth peeling, or the weighted-blended approximation).
   */
  _orderTransparent(scene, camera) {
    const gpu = this.gpu;
    this.transparentList.clear();
    this.stats.transparent = 0;
    this.stats.transparentDraws = 0;
    this._transmissiveCount = 0;
    if (gpu.transparentCount === 0) return;

    const near = camera.near;
    const eye = camera.position;
    vec3Sub(FORWARD, camera.target, camera.position);
    vec3Normalize(FORWARD, FORWARD);

    const projectionScale = camera.projection[5];
    for (let t = 0; t < gpu.transparentCount; t++) {
      const i = gpu.transparentItems[t];
      const o = i * 3;
      // The level the camera shows is the level that casts, on screen or not.
      const selected = !gpu.hasLod || gpu.lodSelected(i, camera.viewProjection, projectionScale);
      gpu.casterSelected[t] = selected ? 1 : 0;
      if (!selected || !frustumTestAABB(this.frustum, scene.worldMin, scene.worldMax, o)) continue;

      // VIEW DEPTH to the bounds centre, not radial distance: the bucket runs
      // it through the projection's own near/depth curve, which is defined
      // against z along the view axis. Clamped at the near plane, where that
      // curve starts.
      const dx = (scene.worldMin[o] + scene.worldMax[o]) * 0.5 - eye[0];
      const dy = (scene.worldMin[o + 1] + scene.worldMax[o + 1]) * 0.5 - eye[1];
      const dz = (scene.worldMin[o + 2] + scene.worldMax[o + 2]) * 0.5 - eye[2];
      const depth = Math.max(dx * FORWARD[0] + dy * FORWARD[1] + dz * FORWARD[2], near);

      const materialId = scene.renderableMaterial[i];
      if (this.materials.isTransmissive(materialId)) this._transmissiveCount++;
      this.transparentList.push(
        transparentSortKey(
          this.materials.pipelineIdOf[materialId],
          materialId,
          transparentDepthBucket(near, depth),
        ),
        i,
      );
    }

    this.transparentList.sort();
    const count = this.transparentList.count;
    // The list grows on push; this staging array has to follow it, or the
    // upload below reads past its own end.
    if (this._transparentOrder.length < count) {
      this._transparentOrder = new Uint32Array(grownCapacity(this._transparentOrder.length, count));
    }
    for (let k = 0; k < count; k++) this._transparentOrder[k] = this.transparentList.payloads[k];
    gpu.writeTransparentOrder(this._transparentOrder, count);
    this.stats.transparent = count;
  }

  /**
   * Draw one phase of the opaque geometry, plus the things that bracket it.
   *
   * Phase 0 opens the frame: the skybox, then everything that was on screen
   * last frame. Phase 1 closes it: whatever the fresh depth pyramid newly
   * admitted, then the blended geometry, which has to follow every opaque draw
   * in either phase.
   *
   * The batch loop is identical in both; only which half of the indirect and
   * visible lists it reads differs, and that is one offset.
   */
  _encodeForward(pass, phase) {
    const scene = this._frameScene;
    const environment = this._frameEnvironment;

    if (phase === 0 && this.drawSkybox) {
      // The skybox binds its own layout at group 0, so the frame group has to
      // be set AFTER it -- a bind group set at an index is overwritten
      // regardless of which pipeline layout put it there.
      this.skybox.draw(pass, environment);
    }
    pass.setBindGroup(GROUP_FRAME, this._frameBindGroup(environment));
    this.pipelineLayout.bindEmptyGroups(pass);

    const gpu = this.gpu;
    let boundPipeline = null;
    let boundMaterial = -1;

    // One call per batch, not per object, and the instance count inside each
    // one was written by the compute shader. A fully culled batch still costs a
    // call, but it draws nothing and the GPU discards it immediately. The late
    // phase is mostly such batches, which is what makes a second pass affordable.
    for (let d = 0; d < this.batchList.count; d++) {
      const b = this.batchList.payloads[d];
      const materialId = gpu.batchMaterial[b];
      const primitive = gpu.batchPrimitive[b];

      const skinned = gpu.batchSkinned[b] === 1;
      const variant = this.materials.variants[materialId]
        | (gpu.batchMirrored[b] ? VARIANT_MIRRORED : 0)
        | (skinned ? VARIANT_SKINNED : 0);
      const pipeline = this.pipelines.get(this._pipelineByVariant.get(variant));
      if (pipeline !== boundPipeline) {
        pass.setPipeline(pipeline);
        boundPipeline = pipeline;
      }
      if (materialId !== boundMaterial) {
        pass.setBindGroup(GROUP_MATERIAL, this.materials.shadingGroup(materialId));
        boundMaterial = materialId;
      }

      pass.setBindGroup(GROUP_DRAW, this.drawBindGroup, [gpu.batchOffset(b, phase)]);
      pass.setVertexBuffer(0, primitive.vertexBuffer);
      // Slot 1 only for skinned pipelines. A pipeline declares how many vertex
      // buffers it reads, so binding this on an unskinned one is a validation
      // error rather than something harmlessly ignored.
      if (skinned) pass.setVertexBuffer(1, primitive.skinBuffer);
      pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
      pass.drawIndexedIndirect(gpu.indirectBuffer, gpu.indirectOffset(b, phase));
    }

    if (phase === 0) return;

    // Blended geometry, strictly after every opaque draw. With OIT on it is
    // drawn into its own targets by a later pass instead, and needs no order
    // at all -- which is the whole reason that path exists.
    if (this._blendInLate) this._encodeTransparent(pass, this._pipelineByVariant, boundPipeline, boundMaterial, false);
  }

  /**
   * Draw every blended object, in sorted order, as few calls as that allows.
   *
   * NEIGHBOURS IN THE SORTED ORDER THAT DRAW ALIKE ARE ONE INSTANCED CALL.
   * This used to be one call per object, on the reasoning that two blended
   * objects at different depths cannot share an instanced draw without losing
   * their order. That holds for objects apart in the order, and not for
   * adjacent ones: a GPU rasterizes and blends a draw's primitives in API
   * order, instance by instance, so a run of neighbours drawn as consecutive
   * instances blends exactly as the separate calls did. The draw data is
   * already laid out in sorted order (slot opaqueCount + k is the k-th), so a
   * run of r is one call of r instances starting at its first slot.
   *
   * "Alike" is everything a call binds: primitive, material, and the pipeline
   * variant that skinning and mirroring pick. A scene of three hundred panes of
   * one kind of glass was three hundred calls and is now however many runs the
   * depth order makes of them -- one, when nothing else is between them.
   *
   * firstInstance carries the slot, which a DIRECT draw may set freely -- the
   * optional-feature restriction the vertex shader mentions applies to
   * indirect draws only. So the shader is the same one the batches use, with a
   * batch base of zero.
   */
  _encodeTransparent(pass, pipelineByVariant, boundPipeline = null, boundMaterial = -1, transmissive = false) {
    const scene = this._frameScene;
    const gpu = this.gpu;
    const payloads = this.transparentList.payloads;
    const transparentCount = this.transparentList.count;
    if (transparentCount === 0) return;

    pass.setBindGroup(GROUP_DRAW, this.drawBindGroup, [gpu.transparentBatchOffset()]);

    let boundPrimitive = null;
    let boundSkinned = false;
    let k = 0;
    while (k < transparentCount) {
      const i = payloads[k];
      const materialId = scene.renderableMaterial[i];
      // Transmissive items draw in their own pass, blended ones in theirs;
      // a run never mixes the two, since it is one material.
      if (this.materials.isTransmissive(materialId) !== transmissive) { k++; continue; }
      const primitive = scene.renderablePrimitive[i];
      const skinned = scene.renderableSkin[i] >= 0;
      const mirrored = gpu.itemMirrored[i];

      // How far the run of neighbours that draw exactly like this one goes.
      let run = 1;
      while (k + run < transparentCount) {
        const j = payloads[k + run];
        if (scene.renderablePrimitive[j] !== primitive
          || scene.renderableMaterial[j] !== materialId
          || (scene.renderableSkin[j] >= 0) !== skinned
          || gpu.itemMirrored[j] !== mirrored) break;
        run++;
      }

      const variant = this.materials.variants[materialId]
        | (mirrored ? VARIANT_MIRRORED : 0)
        | (skinned ? VARIANT_SKINNED : 0);
      const pipeline = this.pipelines.get(pipelineByVariant.get(variant));
      if (pipeline !== boundPipeline) {
        pass.setPipeline(pipeline);
        boundPipeline = pipeline;
      }
      if (materialId !== boundMaterial) {
        pass.setBindGroup(GROUP_MATERIAL, this.materials.shadingGroup(materialId));
        boundMaterial = materialId;
      }
      // Buffers too only when they change: two runs of one mesh split by
      // something else in between still share them.
      if (primitive !== boundPrimitive || skinned !== boundSkinned) {
        pass.setVertexBuffer(0, primitive.vertexBuffer);
        if (skinned) pass.setVertexBuffer(1, primitive.skinBuffer);
        pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
        boundPrimitive = primitive;
        boundSkinned = skinned;
      }
      pass.drawIndexed(primitive.indexCount, run, 0, 0, gpu.opaqueCount + k);
      this.stats.transparentDraws++;
      k += run;
    }
  }

  /**
   * Build the full-screen pass that composites the OIT targets.
   *
   * Its own layout and pipeline, because it binds two textures and nothing
   * else -- no material, no lights, no geometry. The blend is a plain
   * source-over: the resolve returns coverage in alpha, so `1 - reveal` of the
   * blended colour lands on top of whatever the forward passes left.
   */
  async _initOit() {
    const device = this.rhi.device;
    const shader = await compileShader(device, OIT_RESOLVE_SHADER, 'oit.wgsl');

    this.oitLayout = device.createBindGroupLayout({
      label: 'oit-resolve',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    this.oitResolveDescriptor = {
      label: 'oit-resolve',
      layout: createPipelineLayout(device, { 0: this.oitLayout }, 'oit-resolve'),
      shader,
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depth: null,
      targets: [{
        format: HDR_FORMAT,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    };
    await this.pipelines.warm([this.oitResolveDescriptor]);
    this._oitKey = '';
    this._oitBindGroup = null;
  }

  /**
   * The resolve's bind group for this (accum, reveal) view pair, which the
   * graph pool keeps stable from frame to frame.
   *
   * ONE, not a cache of them. A frame resolves one pair, so the only question
   * is whether it is still the last one. This was a map that nothing evicted
   * -- every resize with OIT on added an entry naming views the pool had
   * since destroyed.
   */
  _oitBindGroupFor(accumView, revealView) {
    const key = `${viewKey(accumView)}:${viewKey(revealView)}`;
    if (key !== this._oitKey) {
      this._oitBindGroup = this.rhi.device.createBindGroup({
        label: 'oit-resolve',
        layout: this.oitLayout,
        entries: [
          { binding: 0, resource: accumView },
          { binding: 1, resource: revealView },
        ],
      });
      this._oitKey = key;
    }
    return this._oitBindGroup;
  }

  _encodeOitResolve(pass, accumView, revealView) {
    pass.setPipeline(this.pipelines.get(this.oitResolveDescriptor));
    pass.setBindGroup(0, this._oitBindGroupFor(accumView, revealView));
    pass.draw(3);
  }

  /** The OIT geometry pass: the same objects, unsorted, into accum and reveal. */
  _encodeOIT(pass) {
    pass.setBindGroup(GROUP_FRAME, this._frameBindGroup(this._frameEnvironment));
    this.pipelineLayout.bindEmptyGroups(pass);
    this._encodeTransparent(pass, this._oitPipelineByVariant, null, -1, false);
  }
}

// Views have no identity of their own, so one is stamped on first use. Stable
// because the graph's texture pool hands back the same view objects for the
// same descriptor, and its eviction drops the entries that stop coming back.
let nextViewKey = 1;
function viewKey(view) {
  if (!view.__oitKey) view.__oitKey = nextViewKey++;
  return view.__oitKey;
}

/** Model matrix plus its normal matrix, packed for the draw-data buffer. */
function writeDrawData(out, offset, scene, renderable) {
  const worldOffset = scene.renderableMatrixSlot[renderable] * 16;
  for (let k = 0; k < 16; k++) out[offset + k] = scene.transforms.world[worldOffset + k];
  mat4NormalMatrix(out, scene.transforms.world, offset + 16, worldOffset);
}

function alignUp(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}
