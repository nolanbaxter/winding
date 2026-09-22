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
import { DEPTH_CLEAR_VALUE } from '../rhi/device.js';

import { mat4NormalMatrix } from '../core/math/mat4.js';
import { vec3Create, vec3Sub, vec3Normalize } from '../core/math/vec3.js';
import { frustumCreate, frustumFromViewProjection, frustumTestAABB } from '../core/math/frustum.js';
import { grownCapacity } from '../core/grow.js';

import { MaterialRegistry, variantPipelineState, VARIANT_MIRRORED } from './material.js';
import { PBR_SHADER, FRAME_BYTES } from './shaders/pbr.js';
import { SkyboxPass } from './skybox.js';
import { ShadowMaps } from './shadows.js';
import { RenderGraph } from './graph.js';
import { GpuProfiler } from './timing.js';
import { ClusteredLights, CLUSTER_X, CLUSTER_Y, CLUSTER_Z } from './clustered.js';
import { PostStack, HDR_FORMAT } from './post.js';
import { GpuDriven, BATCH_BYTES, INDIRECT_BYTES } from './gpudriven.js';
import { updateWorldBounds } from '../scene/bounds.js';
import { HierarchicalDepth } from './hzb.js';
import { VERTEX_BUFFER_LAYOUT as VERTEX_LAYOUT } from './vertex.js';

const DEFAULT_MAX_DRAWS = 4096;

/** Scratch for the camera forward axis used by transparent depth sorting. */
const FORWARD = vec3Create();

const now = () => (globalThis.performance?.now?.() ?? Date.now());

export class Renderer {
  static async create(rhi, {
    maxDraws = DEFAULT_MAX_DRAWS, exposure = 1.0, shadows, lightDistance = 60, post, gpuTiming = true,
  } = {}) {
    const renderer = new Renderer(rhi, { maxDraws, exposure, shadows, lightDistance, post, gpuTiming });
    await renderer._init();
    return renderer;
  }

  constructor(rhi, { maxDraws, exposure, shadows, lightDistance, post, gpuTiming = true }) {
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
    // Same memory, integer view: the cluster grid dimensions are u32 in WGSL.
    this.frameU32 = new Uint32Array(this.frameData.buffer);

    this.frustum = frustumCreate();

    this._pipelineByVariant = new Map();
    this._frameBindGroups = new WeakMap();
    this._clusterRevision = 0;

    this.lightDistance = lightDistance;
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
    /**
     * GPU milliseconds per pass, to sit beside the CPU phases below.
     *
     * On by default for the same reason those are: the cost is two timestamps
     * per pass and an async copy, and a renderer that makes you remember to
     * turn on the thing that says what is slow gets optimized by guesswork.
     * Inert on a device without `timestamp-query`.
     */
    this.gpuTiming = new GpuProfiler(rhi, { enabled: gpuTiming });
    this.graph = new RenderGraph(rhi, { profiler: this.gpuTiming });
    // Bound once: the graph holds a function per pass, and rebuilding these
    // every frame would allocate a closure per pass per frame.
    this._forwardEarly = (pass) => this._encodeForward(pass, 0);
    this._forwardLate = (pass) => this._encodeForward(pass, 1);
    this._frameScene = null;
    this._frameEnvironment = null;

    this.stats = { renderables: 0, draws: 0, recomposed: 0, transparent: 0 };
    /**
     * CPU milliseconds per phase, for the optimization pass.
     *
     * Cheap enough to leave on: five performance.now() calls a frame against a
     * frame budget of 16.7ms. Guessing which phase is expensive is how people
     * optimize the wrong one.
     */
    this.timing = { transforms: 0, upload: 0, graph: 0, encode: 0, total: 0 };
  }

  async _init() {
    this.gpu = await GpuDriven.create(this.rhi, this.maxDraws, this.materials);
    this._makeDrawBindGroup();
    this.clusters = await ClusteredLights.create(this.rhi);
    this.shader = await compileShader(this.rhi.device, PBR_SHADER, 'pbr.wgsl');
    this.shadows = await ShadowMaps.create(
      this.rhi, this.pipelines, this.drawLayout, this.shadowOptions,
    );
    this.skybox = await SkyboxPass.create(this.rhi, this.pipelines);
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
      wanted.push(variant & ~VARIANT_MIRRORED, variant | VARIANT_MIRRORED);
    }
    for (const variant of wanted) {
      if (this._pipelineByVariant.has(variant)) continue;

      const state = variantPipelineState(variant);
      const descriptor = {
        label: `pbr:v${variant}`,
        layout: this.pipelineLayout,
        shader: this.shader,
        buffers: [VERTEX_LAYOUT],
        targets: [{ format: HDR_FORMAT, blend: state.blend }],
        primitive: state.primitive,
        depth: state.depth,
        constants: state.constants,
      };
      this._pipelineByVariant.set(variant, descriptor);
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
        ],
      });
      this._frameBindGroups.set(environment, bindGroup);
    }
    return bindGroup;
  }

  render(scene, camera, jobs = null) {
    const rhi = this.rhi;
    const environment = scene.environment;
    if (!environment) throw new Error('Renderer: the scene has no environment');

    const tFrame = now();
    this.stats.recomposed = scene.update(jobs);
    const tAfterTransforms = now();
    camera.update(rhi.aspect);
    frustumFromViewProjection(this.frustum, camera.viewProjection);

    const count = scene.renderableCount;
    this.stats.renderables = count;

    updateWorldBounds(
      count, scene.localMin, scene.localMax, scene.worldMin, scene.worldMax,
      scene.transforms.world, scene.renderableMatrixSlot, scene.transforms.moved,
    );

    // Culling happens on the GPU. Nothing here reads back which objects survived
    // -- that answer only ever exists in the indirect argument buffer, which
    // is exactly the point: a readback would cost a pipeline stall.
    // The pyramid is sized to the surface, so it is rebuilt on resize. Doing it
    // before the cull means the bind group always points at a live texture.
    this.hzb.resize(rhi.width, rhi.height, rhi.depthView());
    this.gpu.bindHzb(this.hzb.view);
    this.gpu.update(scene, this.frustum, this.hzb, camera.viewProjection, writeDrawData);
    if (this._drawBindGroupRevision !== this.gpu.buffersRevision) this._makeDrawBindGroup();

    // Both consumers of `moved` have now read it, so the record is spent.
    // Clearing here rather than in update() is what makes scene.update() safe
    // to call any number of times before a frame.
    scene.transforms.moved.fill(0, 0, scene.transforms.capacity);

    const tAfterUpload = now();

    if (this._batchRevision !== this.gpu.sceneRevision) {
      this.batchList.clear();
      for (let b = 0; b < this.gpu.batchCount; b++) {
        const materialId = this.gpu.batchMaterial[b];
        // Winding is pipeline state, so it belongs in the pipeline field. Two
        // ids per material variant, which keeps the worst case at 12 of the 16
        // the narrower key can address.
        const pipelineId = this.materials.pipelineIdOf[materialId] * 2 + this.gpu.batchMirrored[b];
        this.batchList.push(opaqueSortKey(pipelineId, materialId, 0), b);
      }
      this.batchList.sort();
      this._batchRevision = this.gpu.sceneRevision;
    }
    this.stats.draws = this.gpu.batchCount;

    this._orderTransparent(scene, camera);

    // These three recompute what the frame uniform is about to copy, so they
    // have to run FIRST. Filling the uniform before them uploaded the previous
    // frame's cascade matrices, splits and cluster parameters while the shadow
    // pass rasterised with this frame's -- so a moving camera looked up its
    // shadows in the wrong patch of the map, and the first frame of all read
    // matrices that were still zero.
    this.skybox.update(camera, 1.0);
    this.shadows.update(camera, scene.sun.direction);
    this.clusters.update(scene, camera, this.lightDistance);
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
    this.frameData[20] = -scene.sun.direction[0];
    this.frameData[21] = -scene.sun.direction[1];
    this.frameData[22] = -scene.sun.direction[2];
    this.frameData[23] = environment.prefilterMips;
    this.frameData.set(scene.sun.color, 24);
    this.frameData[27] = this.shadows.activeCascades;

    // Cascade matrices (4 x mat4), splits, texel sizes, then the bias params.
    this.frameData.set(this.shadows.matrices, 28);
    this.frameData.set(this.shadows.splits, 92);
    this.frameData.set(this.shadows.texelSizes, 96);
    this.frameData[100] = this.shadows.normalBias;
    this.frameData[101] = this.shadows.size;

    // Cluster grid dims are u32 in the shader, so they are written through a
    // Uint32 view of the same buffer rather than as floats.
    this.frameU32[104] = CLUSTER_X;
    this.frameU32[105] = CLUSTER_Y;
    this.frameU32[106] = CLUSTER_Z;
    this.frameU32[107] = this.clusters.lightCount;
    this.frameData[108] = this.clusters.sliceScale;
    this.frameData[109] = this.clusters.sliceBias;
    this.frameData[110] = this.clusters.tileSize[0];
    this.frameData[111] = this.clusters.tileSize[1];
    rhi.queue.writeBuffer(this.frameBuffer, 0, this.frameData);



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

    this.shadows.addPasses(graph, shadowMap, this.gpu, this.drawBindGroup);
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

    this.post.addPasses(graph, {
      sceneColor,
      surface,
      width: rhi.width,
      height: rhi.height,
      exposure: this.exposure,
    });

    graph.compile();
    const tAfterGraph = now();

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
    rhi.queue.submit([encoder.finish()]);
    // After the submit, never before: the command buffer above writes the
    // buffer this maps, and a buffer with a map pending cannot be written.
    this.gpuTiming.readback();
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

    if (phase === 0) {
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

      const variant = this.materials.variants[materialId]
        | (gpu.batchMirrored[b] ? VARIANT_MIRRORED : 0);
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
      pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
      pass.drawIndexedIndirect(gpu.indirectBuffer, gpu.indirectOffset(b, phase));
    }

    if (phase === 0) return;

    // Blended geometry, strictly after every opaque draw and in the order
    // _orderTransparent worked out. One call each: two blended objects at
    // different depths cannot share an instanced draw without losing the
    // ordering that is the entire point of this pass.
    //
    // firstInstance carries the slot, which a DIRECT draw may set freely --
    // the optional-feature restriction the vertex shader mentions applies to
    // indirect draws only. So the shader is the same one the batches use, with
    // a batch base of zero.
    const transparentCount = this.transparentList.count;
    if (transparentCount > 0) {
      pass.setBindGroup(GROUP_DRAW, this.drawBindGroup, [gpu.transparentBatchOffset()]);

      for (let k = 0; k < transparentCount; k++) {
        const i = this.transparentList.payloads[k];
        const materialId = scene.renderableMaterial[i];
        const primitive = scene.renderablePrimitive[i];

        const variant = this.materials.variants[materialId]
          | (gpu.itemMirrored[i] ? VARIANT_MIRRORED : 0);
        const pipeline = this.pipelines.get(this._pipelineByVariant.get(variant));
        if (pipeline !== boundPipeline) {
          pass.setPipeline(pipeline);
          boundPipeline = pipeline;
        }
        if (materialId !== boundMaterial) {
          pass.setBindGroup(GROUP_MATERIAL, this.materials.bindGroup(materialId));
          boundMaterial = materialId;
        }

        pass.setVertexBuffer(0, primitive.vertexBuffer);
        pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
        pass.drawIndexed(primitive.indexCount, 1, 0, 0, gpu.opaqueCount + k);
      }
    }
  }
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
