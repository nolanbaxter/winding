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
import { packSkinVertices } from '../render/vertex.js';
import { packMorphCountStride } from '../render/morph.js';
import { Scene } from '../scene/scene.js';
import { loadGLTF, DEFAULT_MATERIAL } from '../scene/gltf/parse.js';
import { decodeImages } from '../scene/gltf/images.js';
import { Clock } from '../core/time.js';
import { JobSystem, JOB_COMPOSE_TRANSFORMS } from '../core/jobs.js';
import { composeRange } from '../scene/transformJob.js';
import { sharedMemoryAvailable } from '../core/shared.js';


/**
 * The shim module a cross-origin worker is loaded through.
 *
 * A module's own imports are fetched with CORS, which a CDN serves happily.
 * The Worker constructor is the thing that refuses, so the way past it is a
 * worker script that is same-origin and does nothing but import the real one.
 *
 * JSON.stringify rather than quotes by hand: it is the specifier of a real
 * import statement, and a URL may legally contain a quote.
 */
export function workerShimSource(href) {
  return `import ${JSON.stringify(href)};`;
}

/** Blob URL of the shim for each worker href. One per URL, for the page's life. */
const shimUrls = new Map();

/**
 * A module Worker, whether or not the engine was served from the page's origin.
 *
 * `new Worker(url)` REFUSES a cross-origin script -- it THROWS rather than
 * degrading -- and an engine loaded from a CDN is cross-origin by definition.
 * So the case this exists for is the good one: a page that set COOP and COEP,
 * which is the only case where the job system spawns workers at all, loading
 * Winding from jsDelivr. Without this, better configuration produces a harder
 * failure, which is exactly backwards.
 *
 * Same-origin stays a direct construction. The shim costs a fetch hop and
 * exists to get around a restriction that is not there.
 *
 * The blob URL is deliberately not revoked. It is a few dozen bytes, shared by
 * every worker that loads the same script, and revoking it while a worker is
 * still fetching it is a race whose best outcome is nothing.
 */
export function createModuleWorker(
  url,
  { WorkerClass = globalThis.Worker, origin = globalThis.location?.origin } = {},
) {
  if (url.origin === origin) return new WorkerClass(url, { type: 'module' });

  let shim = shimUrls.get(url.href);
  if (shim === undefined) {
    shim = URL.createObjectURL(
      new Blob([workerShimSource(url.href)], { type: 'text/javascript' }),
    );
    shimUrls.set(url.href, shim);
  }
  return new WorkerClass(shim, { type: 'module' });
}

export class Winding {
  /**
   * @param canvas a <canvas>; it is sized, configured and observed for you
   * @param options.environment  Environment settings, or an Environment to share
   * @param options.onDeviceLost called when the GPU goes away
   *
   * `shadows`, `post`, `shadowDistance` and `lightDistance` reach the renderer
   * from here. They used to stop at this function -- accepted by
   * Renderer.create and never passed on -- so the shadow map's size and cascade
   * count, the bloom settings, and the two world-scale ranges were unreachable
   * without constructing a Renderer yourself. The scale ones now derive from
   * the scene by default, which is the better fix, but the others were simply
   * lost.
   */
  static async create(canvas, options = {}) {
    const rhi = await createDevice(canvas, {
      label: options.label ?? 'winding',
      powerPreference: options.powerPreference,
      onDeviceLost: options.onDeviceLost,
      onError: options.onError,
    });

    // A failure past this point would otherwise strand the device and its
    // resize observer, with no engine for anyone to call destroy() on.
    try {
      const renderer = await Renderer.create(rhi, {
        maxDraws: options.maxDraws,
        exposure: options.exposure,
        shadows: options.shadows,
        post: options.post,
        shadowDistance: options.shadowDistance,
        lightDistance: options.lightDistance,
        gpuTiming: options.gpuTiming,
        oit: options.oit,
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
          ? () => createModuleWorker(new URL('../core/jobWorker.js', import.meta.url))
          : null,
      });

      return new Winding(rhi, renderer, environment, jobs, !sharedEnvironment);
    } catch (error) {
      rhi.destroy();
      throw error;
    }
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

    // The device was lost, or the canvas left the document. Either way nothing
    // this engine draws can be seen again, so it lets go of everything --
    // including its workers, which nothing else would ever end. Whether or not
    // run() is driving it: an engine that failed before run(), or one driven
    // through renderFrame(), has no loop to notice.
    rhi.onUnusable = () => this.destroy();
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
      // How many morph targets every primitive here carries. The scene reads
      // it to decide whether a node instancing this mesh needs weights.
      targetCount: mesh.targetCount,
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
        /** Whether this primitive carries influences. The scene reads this. */
        skinned: primitive.jointIndices != null,
        // Null unless the mesh is rigged. A second vertex buffer, bound at
        // slot 1 by skinned pipelines only, so static meshes carry nothing.
        skinBuffer: primitive.jointIndices
          ? createBuffer(this.rhi, {
            label: `${mesh.name}[${p}].skin`,
            data: new Uint8Array(packSkinVertices(
              primitive.jointIndices, primitive.jointWeights, primitive.vertexCount,
            )),
            usage: GPUBufferUsage.VERTEX,
          })
          : null,
        indexCount: primitive.indexCount,
        bounds: primitive.bounds,
        // How far each target reaches, which is all a bound needs. The deltas
        // themselves are a GPU buffer; this is the one number the CPU keeps.
        morphExtent: primitive.morph?.extent ?? null,
        // Where this primitive's deltas landed in the engine-wide arena, and
        // its target count and stride packed as draw data carries them. Zero
        // for an unmorphed primitive, which is what makes the vertex shader
        // skip the loop entirely.
        morphBase: primitive.morph ? this.renderer.morph.allocate(primitive.morph.deltas) : 0,
        /** How many floats of the arena that is, for unload() to give back. */
        morphFloats: primitive.morph ? primitive.morph.deltas.length : 0,
        morphCountStride: primitive.morph
          ? packMorphCountStride(primitive.morph.targetCount, primitive.morph.stride)
          : 0,
        // Only when asked. Scene.raycast tests triangles for primitives that have
        // these and falls back to the bounding box for those that do not, so the
        // flag buys precision with memory and nothing else changes.
        positions: retainGeometry ? primitive.positions : undefined,
        indices: retainGeometry ? primitive.indices : undefined,
        materialId: primitive.material >= 0
          ? materialIds[primitive.material]
          : this._defaultMaterial(),
        meshIndex: m,
        /** Renderables drawing this, across every scene. See Scene.add. */
        instances: 0,
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
      animations: model.animations, skins: model.skins, source: model.source,
      lights: model.lights, cameras: model.cameras, textures,
    };
  }

  /**
   * Free what load() made for an asset: its buffers, textures, material ids
   * and morph deltas. Remove it from every scene first; this throws if any
   * scene still draws it.
   *
   * load() allocates on every call, the same file included, and until this
   * existed nothing ever gave it back. An editor that re-imports on each save
   * kept every version it had ever loaded, and the 4097th material threw.
   */
  unload(asset) {
    if (asset.unloaded) return;
    for (const mesh of asset.meshes) {
      for (const primitive of mesh.primitives) {
        if (primitive.instances > 0) {
          throw new Error(`unload: mesh "${mesh.name}" is still in a scene; remove it first`);
        }
      }
    }
    asset.unloaded = true;

    for (const mesh of asset.meshes) {
      for (const primitive of mesh.primitives) {
        primitive.vertexBuffer.destroy();
        primitive.indexBuffer.destroy();
        primitive.skinBuffer?.destroy();
        if (primitive.morphFloats > 0) {
          this.renderer.morph.free(primitive.morphBase, primitive.morphFloats);
        }
      }
    }
    for (const id of asset.materialIds) this.renderer.materials.release(id);
    asset.textures.destroy();
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
   *
   * If the canvas is removed from the document, the next frame destroys the
   * engine: nothing drawn into a detached canvas can be seen. (An engine that
   * is not running learns the same thing from its resize observer.)
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
      // A canvas that has left the document can never be seen again, so every
      // frame drawn into it is waste. This is what a live editor's reload
      // leaves behind: it rewrites the page in place (document.open/write), no
      // pagehide fires, and each old engine kept rendering and kept its GPU
      // device -- one more full loop per edit, until the adapter ran out.
      if (this.rhi.canvas.isConnected === false) { this.destroy(); return; }
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
    if (this._destroyed) return;   // the loop may already have done it
    this._destroyed = true;
    this.stop();
    this.jobs.destroy();
    this.renderer.destroy();
    if (this._ownsEnvironment) this.environment.destroy();
    this.rhi.destroy();
  }
}
