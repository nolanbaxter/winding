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
  MaterialRegistry, variantPipelineState, VARIANT_MIRRORED, VARIANT_SKINNED, ALPHA_BLEND,
} from './material.js';
import { PBR_SHADER, FRAME_BYTES } from './shaders/pbr.js';
import { OIT_RESOLVE_SHADER } from './shaders/oit.js';
import { SkyboxPass } from './skybox.js';
import { ShadowMaps, stableShadowDistance } from './shadows.js';
import { RenderGraph } from './graph.js';
import { GpuProfiler } from './timing.js';
import { SkinPalette } from './skin.js';
import { MorphStore } from './morph.js';
import { ClusteredLights, CLUSTER_Z } from './clustered.js';
import { PostStack, HDR_FORMAT } from './post.js';
import { GpuDriven, BATCH_BYTES, INDIRECT_BYTES } from './gpudriven.js';
import {
  updateWorldBounds, unionWorldBounds, farthestViewDepth, farthestDistance,
  updateSkinBounds, applySkinBounds,
} from '../scene/bounds.js';
import { HierarchicalDepth } from './hzb.js';
import { VERTEX_BUFFER_LAYOUT as VERTEX_LAYOUT, SKIN_BUFFER_LAYOUT } from './vertex.js';

const DEFAULT_MAX_DRAWS = 4096;

/** Scratch for the camera forward axis used by transparent depth sorting. */
const FORWARD = vec3Create();

const now = () => (globalThis.performance?.now?.() ?? Date.now());

export class Renderer {
  static async create(rhi, {
    maxDraws = DEFAULT_MAX_DRAWS, exposure = 1.0, shadows, lightDistance = null,
    shadowDistance = null, post, gpuTiming = false, oit = false,
  } = {}) {
    const renderer = new Renderer(rhi, {
      maxDraws, exposure, shadows, lightDistance, shadowDistance, post, gpuTiming, oit,
    });
    await renderer._init();
    return renderer;
  }

  constructor(rhi, {
    maxDraws, exposure, shadows, lightDistance, shadowDistance, post, gpuTiming = false,
    oit = false,
  }) {
    this.shadowOptions = shadows ?? {};
    this.postOptions = post ?? {};
    this.rhi = rhi;
    this.maxDraws = maxDraws;
    this.exposure = exposure;

    this.pipelines = new PipelineCache(rhi.device);
    this.materials = new MaterialRegistry(rhi, { capacity: 1024 });

    this.frameLayout = rhi.device.createBindGroupLayout({
      label: 'frame',
      entries: [
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
      ],
    });
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

    this.frameBuffer = rhi.device.createBuffer({
      label: 'frame', size: FRAME_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.frameData = new Float32Array(FRAME_BYTES / 4);
    // The directional lights that do not cast the shadow. Grown to the scene's
    // count on demand; one entry to start, since a binding cannot be empty.
    this.directionalCapacity = 1;
    this.directionalBuffer = this._createDirectionalBuffer(1);
    // Same memory, integer view: the cluster grid dimensions are u32 in WGSL.
    this.frameU32 = new Uint32Array(this.frameData.buffer);

    this.frustum = frustumCreate();

    this._pipelineByVariant = new Map();
    this._oitPipelineByVariant = new Map();
    /**
     * Weighted-blended order-independent transparency, off by default.
     *
     * The sorted path is EXACT for separated convex objects and has no answer
     * for interpenetrating ones; this is approximate everywhere and needs no
     * order. Different tools, so this is a choice rather than a replacement --
     * architectural glass wants the sorted path, smoke and foliage want this.
     */
    this.oit = oit;
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
    this.shader = await compileShader(this.rhi.device, PBR_SHADER, 'pbr.wgsl');
    if (this.oit) await this._initOit();
    this.shadows = await ShadowMaps.create(
      this.rhi, this.pipelines, this.drawLayout, this.shadowOptions,
    );
    this.skybox = await SkyboxPass.create(this.rhi, this.pipelines);
    /**
     * Whether the environment is drawn as the background.
     *
     * A plain mutable field, like scene.sun: per-renderer state a caller reads
     * and writes, not hidden configuration. Turning it off leaves the clear
     * colour showing and changes NOTHING about lighting -- the same cubemap is
     * still the ambient term, because the sky IS the light. Setting the sky
     * colours to black would turn the background off too, and take the
     * lighting with it.
     */
    this.drawSkybox = true;
    this.post = await PostStack.create(this.rhi, this.pipelines, this.postOptions);
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
    const pending = [];
    // Both windings of each, because whether an instance mirrors is a property
    // of the scene and is not known here -- and the contract of this method is
    // that nothing in render() ever has to create a pipeline.
    const wanted = [];
    for (const variant of variants) {
      const base = variant & ~(VARIANT_MIRRORED | VARIANT_SKINNED);
      wanted.push(base, base | VARIANT_MIRRORED,
        base | VARIANT_SKINNED, base | VARIANT_MIRRORED | VARIANT_SKINNED);
    }
    for (const variant of wanted) {
      if (this._pipelineByVariant.has(variant)) continue;

      const state = variantPipelineState(variant);
      const skinned = (variant & VARIANT_SKINNED) !== 0;
      const descriptor = {
        label: `pbr:v${variant}`,
        layout: this.pipelineLayout,
        shader: this.shader,
        vertexEntry: skinned ? 'vsSkinned' : 'vs',
        buffers: skinned ? [VERTEX_LAYOUT, SKIN_BUFFER_LAYOUT] : [VERTEX_LAYOUT],
        targets: [{ format: HDR_FORMAT, blend: state.blend }],
        primitive: state.primitive,
        depth: state.depth,
        constants: state.constants,
      };
      this._pipelineByVariant.set(variant, descriptor);
      pending.push(descriptor);
    }
    if (pending.length > 0) await this.pipelines.warm(pending);
    if (this.oit) await this._ensureOitVariants(wanted);
  }

  /**
   * The OIT copies of every BLEND variant.
   *
   * Only those: opaque and masked geometry never reaches the transparent path,
   * so building them would compile pipelines nothing can bind. Two targets and
   * two blend states rather than one, which is why these cannot just be the
   * same descriptors with a different entry point.
   */
  async _ensureOitVariants(variants) {
    const pending = [];
    for (const variant of variants) {
      if ((variant & 3) !== ALPHA_BLEND) continue;
      if (this._oitPipelineByVariant.has(variant)) continue;

      const state = variantPipelineState(variant);
      const descriptor = {
        label: `pbr-oit:v${variant}`,
        layout: this.pipelineLayout,
        shader: this.shader,
        buffers: [VERTEX_LAYOUT],
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
        constants: state.constants,
      };
      this._oitPipelineByVariant.set(variant, descriptor);
      pending.push(descriptor);
    }
    if (pending.length > 0) await this.pipelines.warm(pending);
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
        ],
      });
      this._frameBindGroups.set(environment, bindGroup);
    }
    return bindGroup;
  }

  _createDirectionalBuffer(count) {
    return this.rhi.device.createBuffer({
      label: 'directionals',
      size: count * DIRECTIONAL_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  render(scene, camera, jobs = null) {
    const rhi = this.rhi;
    const environment = scene.environment;
    if (!environment) throw new Error('Renderer: the scene has no environment');

    const p = this.profiler;
    p?.frameStart();
    const tFrame = now();
    this.stats.recomposed = scene.update(jobs);
    const tAfterTransforms = now();
    p?.mark('transforms');
    camera.update(rhi.aspect);
    frustumFromViewProjection(this.frustum, camera.viewProjection);
    p?.mark('camera');

    const count = scene.renderableCount;
    this.stats.renderables = count;

    const moved = updateWorldBounds(
      count, scene.localMin, scene.localMax, scene.worldMin, scene.worldMax,
      scene.transforms.world, scene.renderableMatrixSlot, scene.transforms.moved,
    );

    // Skinned renderables get their bounds replaced: the pass above gave them
    // a bind-pose box transformed by a model matrix the vertices do not follow.
    if (scene.skins.length > 0) {
      updateSkinBounds(scene.skins, scene.transforms.world);
      applySkinBounds(count, scene.renderableSkin, scene.skins, scene.worldMin, scene.worldMax);
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
    if (moved > 0 || morphed > 0 || this._boundsRevision !== scene.revision) {
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
    this.hzb.resize(rhi.width, rhi.height, rhi.depthView());
    this.gpu.bindHzb(this.hzb.view);
    // Weights first: draw data records where each instance's slice begins,
    // and the gather below is what decides those offsets.
    this.morph.update(scene);
    this.gpu.update(scene, this.frustum, this.hzb, camera.viewProjection, writeDrawData,
      this.skinPalette.offsets, this.morph);
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
    scene.transforms.moved.fill(0, 0, scene.transforms.capacity);

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
    this.skybox.update(camera, 1.0);
    // Lights are scene objects: their position and aim live in their
    // transforms, which have composed by now. Copied into the packed array
    // here, immediately before upload, so a light parented to something that
    // moved this frame is lit from where it is rather than where it was.
    // Before the shadow fit, because the sun is one of them.
    scene.refreshLights();
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
    p?.mark('lights');
    this.shadows.update(camera, scene.sunDirection);
    p?.mark('shadow fit');
    this.clusters.update(scene, camera, lightRange);
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
    // The shader wants the direction TOWARD the light; the scene stores the
    // direction light travels, which is what a user means by "sun direction".
    this.frameData[20] = -scene.sunDirection[0];
    this.frameData[21] = -scene.sunDirection[1];
    this.frameData[22] = -scene.sunDirection[2];
    this.frameData[23] = environment.prefilterMips;
    this.frameData.set(scene.sunColor, 24);
    this.frameData[27] = this.shadows.activeCascades;

    // Cascade matrices (4 x mat4), splits, texel sizes, then the bias params.
    this.frameData.set(this.shadows.matrices, 28);
    this.frameData.set(this.shadows.splits, 92);
    this.frameData.set(this.shadows.texelSizes, 96);
    this.frameData[100] = this.shadows.normalBias;
    this.frameData[101] = this.shadows.size;

    // Cluster grid dims are u32 in the shader, so they are written through a
    // Uint32 view of the same buffer rather than as floats.
    // The grid follows the viewport aspect, so the shader is told the shape
    // this frame has rather than a constant it would disagree with.
    this.frameU32[104] = this.clusters.gridX;
    this.frameU32[105] = this.clusters.gridY;
    this.frameU32[106] = CLUSTER_Z;
    this.frameU32[107] = this.clusters.lightCount;
    this.frameData[108] = this.clusters.sliceScale;
    this.frameData[109] = this.clusters.sliceBias;
    this.frameData[110] = this.clusters.tileSize[0];
    this.frameData[111] = this.clusters.tileSize[1];
    // The view axis, for view depth in the shader. The view matrix's third row
    // is the camera's +Z in world space; the camera looks down -Z.
    this.frameData[112] = -camera.view[2];
    this.frameData[113] = -camera.view[6];
    this.frameData[114] = -camera.view[10];
    this.frameU32[115] = scene.directionalCount;
    rhi.queue.writeBuffer(this.frameBuffer, 0, this.frameData);
    p?.mark('clusters + frame uniform');



    // --- declare the frame ---------------------------------------------------
    // Nothing below says what order to run in, when to clear, or when to store.
    // The graph works all three out from reads and writes.
    this._frameScene = scene;
    this._frameEnvironment = environment;

    const graph = this.graph;
    graph.begin();

    const surface = graph.importTexture('surface', rhi.currentColorView());
    // Imported but NOT external: the device owns the memory, and nothing reads
    // it after the frame, so the graph is free to derive `discard` for it.
    const depth = graph.importTexture('depth', rhi.depthView(), { external: false });
    const shadowMap = graph.importTexture('shadows', this.shadows.view);

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

    this.shadows.addPasses(
      graph, shadowMap, this.gpu, this.drawBindGroup, this.skinPalette, this.morph,
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
    const sceneColor = graph.createTexture('scene-hdr', {
      width: rhi.width,
      height: rhi.height,
      format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    // EARLY. Everything that was on screen last frame, which is both the
    // picture so far and the set of occluders the pyramid is built from.
    graph.addPass({
      name: 'forward:early',
      reads: [shadowMap, lightBuffer, clusterIndices, clusterCounts, indirectEarly, visibleEarly],
      color: [{ resource: sceneColor, clear: { r: 0, g: 0, b: 0, a: 1 } }],
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

    // LATE. Whatever the fresh pyramid says is visible and was not drawn above,
    // then the blended geometry, which has to follow every opaque draw. No
    // clear on either attachment: the graph derives `load` from the early pass
    // having written them.
    graph.addPass({
      name: 'forward:late',
      reads: [shadowMap, lightBuffer, clusterIndices, clusterCounts, indirectLate, visibleLate],
      color: [{ resource: sceneColor }],
      depth: { resource: depth },
      execute: this._forwardLate,
    });

    // OIT, when it is on. Blended geometry skipped the pass above, so it is
    // drawn here into its own two targets in whatever order it comes -- that
    // is the point -- and composited over the scene by the resolve.
    if (this.oit) {
      const accum = graph.createTexture('oit-accum', {
        width: rhi.width,
        height: rhi.height,
        format: OIT_ACCUM_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      const reveal = graph.createTexture('oit-reveal', {
        width: rhi.width,
        height: rhi.height,
        format: OIT_REVEAL_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });

      graph.addPass({
        name: 'oit',
        reads: [shadowMap, lightBuffer, clusterIndices, clusterCounts],
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

    this.post.addPasses(graph, {
      sceneColor,
      surface,
      width: rhi.width,
      height: rhi.height,
      exposure: this.exposure,
    });

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
    if (gpu.transparentCount === 0) return;

    const near = camera.near;
    const eye = camera.position;
    vec3Sub(FORWARD, camera.target, camera.position);
    vec3Normalize(FORWARD, FORWARD);

    for (let t = 0; t < gpu.transparentCount; t++) {
      const i = gpu.transparentItems[t];
      const o = i * 3;
      if (!frustumTestAABB(this.frustum, scene.worldMin, scene.worldMax, o)) continue;

      // VIEW DEPTH to the bounds centre, not radial distance: the bucket runs
      // it through the projection's own near/depth curve, which is defined
      // against z along the view axis. Clamped at the near plane, where that
      // curve starts.
      const dx = (scene.worldMin[o] + scene.worldMax[o]) * 0.5 - eye[0];
      const dy = (scene.worldMin[o + 1] + scene.worldMax[o + 1]) * 0.5 - eye[1];
      const dz = (scene.worldMin[o + 2] + scene.worldMax[o + 2]) * 0.5 - eye[2];
      const depth = Math.max(dx * FORWARD[0] + dy * FORWARD[1] + dz * FORWARD[2], near);

      const materialId = scene.renderableMaterial[i];
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
        pass.setBindGroup(GROUP_MATERIAL, this.materials.bindGroup(materialId));
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
    if (!this.oit) this._encodeTransparent(pass, this._pipelineByVariant, boundPipeline, boundMaterial);
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
  _encodeTransparent(pass, pipelineByVariant, boundPipeline = null, boundMaterial = -1) {
    const scene = this._frameScene;
    const gpu = this.gpu;
    const payloads = this.transparentList.payloads;
    const transparentCount = this.transparentList.count;
    this.stats.transparentDraws = 0;
    if (transparentCount === 0) return;

    pass.setBindGroup(GROUP_DRAW, this.drawBindGroup, [gpu.transparentBatchOffset()]);

    let boundPrimitive = null;
    let boundSkinned = false;
    let k = 0;
    while (k < transparentCount) {
      const i = payloads[k];
      const materialId = scene.renderableMaterial[i];
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
        pass.setBindGroup(GROUP_MATERIAL, this.materials.bindGroup(materialId));
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
    this._oitBindGroups = new Map();
  }

  /** Bind group per (accum, reveal) view pair, which the graph pool keeps stable. */
  _oitBindGroupFor(accumView, revealView) {
    const key = `${viewKey(accumView)}:${viewKey(revealView)}`;
    let bindGroup = this._oitBindGroups.get(key);
    if (!bindGroup) {
      bindGroup = this.rhi.device.createBindGroup({
        label: 'oit-resolve',
        layout: this.oitLayout,
        entries: [
          { binding: 0, resource: accumView },
          { binding: 1, resource: revealView },
        ],
      });
      this._oitBindGroups.set(key, bindGroup);
    }
    return bindGroup;
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
    this._encodeTransparent(pass, this._oitPipelineByVariant);
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
