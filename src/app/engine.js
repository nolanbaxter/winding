// Tier 3. The facade that makes the other three tiers optional.
//
//   const engine = await Winding.create(canvas);
//   const scene  = engine.createScene();
//   const camera = new Camera();
//
//   scene.add(await engine.load('helmet.glb'));
//   engine.run({ scene, camera });
//
// Two rules shape everything below.
//
// ALL ASYNC WORK HAPPENS IN load(). Parsing, image decode, buffer upload,
// material registration and pipeline compilation all finish before an asset is
// handed back, so scene.add() is synchronous and no frame can ever stall on a
// shader compile: enforced here rather than left to the caller.
//
// THERE IS NO engine.camera. A property like that is global mutable state that
// render() reads invisibly. The camera is an argument, which costs
// one parameter and buys split-screen and shadow passes for free, since those
// are just "render this scene from that camera".

import { createDevice } from '../rhi/device.js';
import { createBuffer } from '../rhi/buffer.js';
import { Environment } from '../render/ibl.js';
import { Renderer } from '../render/renderer.js';
import { GLTFTextures } from '../render/textures.js';
import { Scene } from '../scene/scene.js';
import { loadGLTF, DEFAULT_MATERIAL } from '../scene/gltf/parse.js';
import { decodeImages } from '../scene/gltf/images.js';
import { Clock } from '../core/time.js';
import { JobSystem, JOB_COMPOSE_TRANSFORMS } from '../core/jobs.js';
import { composeRange } from '../scene/transformJob.js';
import { sharedMemoryAvailable } from '../core/shared.js';

export class Winding {
  /**
   * @param canvas a <canvas>; it is sized, configured and observed for you
   * @param options.environment  Environment settings, or an Environment to share
   * @param options.onDeviceLost called when the GPU goes away
   */
  static async create(canvas, options = {}) {
    const rhi = await createDevice(canvas, {
      label: options.label ?? 'winding',
      onDeviceLost: options.onDeviceLost,
      onError: options.onError,
    });

    const renderer = await Renderer.create(rhi, {
      maxDraws: options.maxDraws,
      exposure: options.exposure,
    });

    const sharedEnvironment = options.environment instanceof Environment;
    const environment = sharedEnvironment
      ? options.environment
      : new Environment(rhi, options.environment ?? {});

    // Workers need cross-origin isolation (COOP + COEP). Without it the job
    // system falls back to running inline, which is slower and identical.
    const jobs = new JobSystem({
      workerCount: options.workerCount,
      createWorker: sharedMemoryAvailable
        ? () => new Worker(new URL('../core/jobWorker.js', import.meta.url), { type: 'module' })
        : null,
    });

    return new Winding(rhi, renderer, environment, jobs, !sharedEnvironment);
  }

  constructor(rhi, renderer, environment, jobs, ownsEnvironment = true) {
    /** The RHI. Public: dropping a tier must never require a fork. */
    this.rhi = rhi;
    this.renderer = renderer;
    this.environment = environment;
    this.jobs = jobs;
    /** False when create() was handed an Environment to share. */
    this._ownsEnvironment = ownsEnvironment;

    // Registered once, for the engine, not once per scene. The handler reads
    // the CURRENT owner of the shared buffers rather than closing over a
    // scene: a closure meant every createScene() overwrote the last one, so
    // with two scenes the workers composed one scene's columns while the
    // frame being drawn belonged to the other. Silently, and only on the
    // parallel path.
    this.jobs.register(JOB_COMPOSE_TRANSFORMS,
      (start, end, base) => composeRange(this.jobs.sharedOwner, base, start, end));

    this.clock = new Clock(1 / 60);
    this.fps = 0;
    this._raf = 0;
    this._running = false;
    this._fpsAccum = 0;
    this._fpsFrames = 0;
    this._defaultMaterialId = -1;
  }

  createScene(options = {}) {
    const scene = new Scene(options);
    scene.environment = options.environment ?? this.environment;
    return scene;
  }

  /**
   * Load a .glb/.gltf and get back something scene.add() can take.
   *
   * `source` is a URL string, an ArrayBuffer, or a Uint8Array. This is the only
   * slow call in the API, which is exactly where a progress indicator belongs.
   *
   * `retainGeometry` keeps each primitive's positions and indices on the CPU
   * after upload, which is what Scene.raycast needs to answer with triangles
   * instead of bounding boxes. Off by default: it costs 12 bytes per vertex
   * plus 4 per index on the JS heap for as long as the asset lives, and most
   * scenes never pick.
   */
  async load(source, options = {}) {
    const { retainGeometry = false } = options;
    let bytes = source;
    let baseURL = options.baseURL;

    if (typeof source === 'string') {
      baseURL = baseURL ?? new URL(source, globalThis.location?.href ?? 'http://localhost/');
      const response = await fetch(source);
      if (!response.ok) throw new Error(`load: ${source} returned ${response.status}`);
      bytes = new Uint8Array(await response.arrayBuffer());
    } else if (source instanceof ArrayBuffer) {
      bytes = new Uint8Array(source);
    }

    const model = await loadGLTF(bytes, { baseURL });
    const bitmaps = await decodeImages(model.source.json, model.source.buffers, { baseURL });
    const textures = new GLTFTextures(this.rhi, model.source.json, bitmaps);

    // Materials first: a primitive needs its material id before it can be
    // turned into something the draw list can sort.
    const materialIds = model.materials.map(
      (material) => this.renderer.materials.register(material, textures.texturesFor(material)),
    );

    // Every image that was going to be uploaded now has been, so the decoded
    // copies are dead weight. An ImageBitmap holds native memory the collector
    // frees only when it gets round to it, and a texture-heavy asset is
    // hundreds of megabytes of them -- Sponza decodes to over a gigabyte.
    // close() gives it back at a known moment instead.
    textures.releaseBitmaps();

    const meshes = model.meshes.map((mesh, m) => ({
      name: mesh.name,
      primitives: mesh.primitives.map((primitive, p) => ({
        vertexBuffer: createBuffer(this.rhi, {
          label: `${mesh.name}[${p}].vertices`,
          data: primitive.vertices,
          usage: GPUBufferUsage.VERTEX,
        }),
        indexBuffer: createBuffer(this.rhi, {
          label: `${mesh.name}[${p}].indices`,
          data: primitive.indices,
          usage: GPUBufferUsage.INDEX,
        }),
        indexCount: primitive.indexCount,
        bounds: primitive.bounds,
        // Only when asked. Scene.raycast tests triangles for primitives that have
        // these and falls back to the bounding box for those that do not, so the
        // flag buys precision with memory and nothing else changes.
        positions: retainGeometry ? primitive.positions : undefined,
        indices: retainGeometry ? primitive.indices : undefined,
        materialId: primitive.material >= 0
          ? materialIds[primitive.material]
          : this._defaultMaterial(),
        meshIndex: m,
      })),
    }));

    // Every pipeline the asset can possibly need, compiled off-thread, before
    // the caller is even told the asset exists.
    const variants = new Set();
    for (const mesh of meshes) {
      for (const primitive of mesh.primitives) {
        variants.add(this.renderer.materials.variants[primitive.materialId]);
      }
    }
    await this.renderer.ensureVariants([...variants]);

    return {
      nodes: model.nodes, meshes, roots: model.roots, materialIds,
      animations: model.animations, source: model.source,
    };
  }

  /** glTF lets a primitive have no material; the spec's default is a white dielectric. */
  _defaultMaterial() {
    if (this._defaultMaterialId < 0) {
      this._defaultMaterialId = this.renderer.materials.register(DEFAULT_MATERIAL);
    }
    return this._defaultMaterialId;
  }

  /**
   * Own the frame loop.
   *
   * `update(dt)` runs at a FIXED 60 Hz, however fast the display is, and may
   * run zero or several times per rendered frame. `frame(alpha)` runs once per
   * rendered frame, with alpha in [0,1) being how far between the last two
   * simulation states this frame sits -- that is where interpolation goes.
   *
   * Getting this accumulator right is the thing hobby engines most reliably
   * miss, which is why the engine owns it by default. renderFrame() stays
   * public for anyone who needs to drive it themselves.
   */
  run({ scene, camera, update, frame }) {
    if (this._running) throw new Error('run: already running; call stop() first');
    this._running = true;

    const loop = (nowMs) => {
      // The device going away ends the loop for good, so tear the running
      // flag down with it. Returning without stop() left _running true, and
      // every later run() threw 'already running' with no way back short of
      // destroy() -- a lost device wedged the engine rather than stopping it.
      if (!this._running) return;
      if (this.rhi.destroyed) { this.stop(); return; }
      this._raf = requestAnimationFrame(loop);

      this.clock.begin(nowMs / 1000);
      // Before update(), so a callback that reads a bone's world position sees
      // this frame's pose rather than last frame's.
      scene.advanceAnimations(this.clock.realDelta);
      if (update) while (this.clock.step()) update(this.clock.fixedDt, this.clock.elapsed);
      else this.clock.accumulator = 0;   // nothing to simulate; do not let it grow

      if (frame) frame(this.clock.alpha, this.clock);
      this.renderFrame(scene, camera);

      this._fpsAccum += this.clock.realDelta;
      this._fpsFrames++;
      if (this._fpsAccum >= 0.5) {
        this.fps = this._fpsFrames / this._fpsAccum;
        this._fpsAccum = 0;
        this._fpsFrames = 0;
      }
    };
    this._raf = requestAnimationFrame(loop);
  }

  stop() {
    this._running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;
  }

  /** One frame, synchronously. Use this when you own the loop. */
  renderFrame(scene, camera) {
    this.renderer.render(scene, camera, this.jobs);
  }

  get stats() {
    return this.renderer.stats;
  }

  /**
   * Release everything this engine owns.
   *
   * The environment is destroyed only if this engine made it. `create` accepts
   * one to SHARE, and tearing down a borrowed environment's cubemaps breaks
   * whichever engine is still using them.
   */
  destroy() {
    this.stop();
    this.jobs.destroy();
    this.renderer.destroy();
    if (this._ownsEnvironment) this.environment.destroy();
    this.rhi.destroy();
  }
}
