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

import { renamed } from '../core/assert.js';
import { createDevice } from '../rhi/device.js';
import { createBuffer } from '../rhi/buffer.js';
import { Environment } from '../render/ibl.js';
import { parseHDR } from '../render/hdr.js';
import { Renderer, RenderTarget } from '../render/renderer.js';
import { Splats } from '../render/splats.js';
import { openSurface } from '../render/shadows.js';
import { readSplats } from '../scene/splats.js';
import { GLTFTextures } from '../render/textures.js';
import { createTexture2D, uploadImage, generateMipmaps, decodeImageBytes } from '../rhi/texture.js';
import { Font } from '../render/text.js';
import { parseCube, uploadLUT } from '../render/grading.js';
import { packSkinVertices } from '../render/vertex.js';
import { packMorphCountStride } from '../render/morph.js';
import { Scene } from '../scene/scene.js';
import { Camera2D } from '../scene/camera2d.js';
import { loadGLTF, DEFAULT_MATERIAL } from '../scene/gltf/parse.js';
import { decodeImages, imageColorSpaces } from '../scene/gltf/images.js';
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
   * @param options.environment  Environment settings (share one between scenes
   *                             with createScene({ environment }))
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
    // This used to be accepted as "an Environment to share". It could never
    // work: its cubemaps belong to the device that baked them, and every
    // create() makes a new device, so every frame was a validation error.
    if (options.environment instanceof Environment) {
      throw new Error(
        'create: an Environment belongs to the engine that made it. Pass its settings, '
        + 'or share it between scenes with createScene({ environment }).',
      );
    }
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
        // `antialias` sits at the top, where a canvas's own option would.
        post: { antialias: options.antialias, grading: options.grading ?? null, ...(options.post ?? {}) },
        shadowDistance: options.shadowDistance,
        lightDistance: options.lightDistance,
        gpuTiming: options.gpuTiming,
        oit: options.oit,
        ao: options.ao,
        fog: options.fog,
        dof: options.dof,
        autoExposure: options.autoExposure,
      });

      const environment = new Environment(rhi, options.environment ?? {});

      // Workers need cross-origin isolation (COOP + COEP). Without it the job
      // system falls back to running inline, which is slower and identical.
      const jobs = new JobSystem({
        workerCount: options.workerCount,
        createWorker: sharedMemoryAvailable
          ? () => createModuleWorker(new URL('../core/jobWorker.js', import.meta.url))
          : null,
      });

      const engine = new Winding(rhi, renderer, environment, jobs);
      engine.onDemand = options.onDemand !== false;
      return engine;
    } catch (error) {
      rhi.destroy();
      throw error;
    }
  }

  constructor(rhi, renderer, environment, jobs, ownsEnvironment = true) {
    /** The RHI. Public: dropping a tier must never require a fork. */
    this.gpu = rhi;
    this.renderer = renderer;
    this.environment = environment;
    this.jobs = jobs;
    /** False when whoever constructs this directly lends an Environment it keeps. */
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
    /**
     * Whether run() skips a frame that would draw exactly what the last one
     * did: nothing moved, animated, emitted or changed, and neither the camera
     * nor a setting did. A still scene then costs the GPU nothing. On unless
     * `onDemand: false`; engine.invalidate() forces the next frame.
     */
    this.onDemand = true;
    /** Frames run() skipped because nothing had changed. */
    this.skippedFrames = 0;
    /** What the last frame run() drew was drawn from; see _idle. */
    this._drawn = null;

    // The device was lost, or the canvas left the document. Either way nothing
    // this engine draws can be seen again, so it lets go of everything --
    // including its workers, which nothing else would ever end. Whether or not
    // run() is driving it: an engine that failed before run(), or one driven
    // through renderFrame(), has no loop to notice.
    rhi.onUnusable = () => this.destroy();
    // It may already have happened: create() compiles pipelines for hundreds
    // of milliseconds before this runs, and nobody was listening then.
    if (rhi.destroyed || rhi.canvas.isConnected === false) queueMicrotask(() => this.destroy());
  }

  /** Every entry point that would use the GPU fails here, by name, once destroyed. */
  _assertAlive(what) {
    if (this._destroyed) throw new Error(`${what}: this engine was destroyed`);
  }

  createScene(options = {}) {
    this._assertAlive('createScene');
    const environment = options.environment ?? this.environment;
    if (environment.rhi !== this.gpu) {
      throw new Error('createScene: that Environment belongs to another engine');
    }
    const scene = new Scene(options);
    scene.environment = environment;
    return scene;
  }

  /**
   * An image as a GPU texture, mipmapped: for sprites (scene.addSprite).
   *
   *   const pin = await engine.loadTexture('pin.png');
   *
   * `source` is a URL, a Blob, an ImageBitmap, or anything createImageBitmap
   * takes. `fetch` downloads a URL, as every loader's does: pass one to decide
   * which URLs may be reached. `srgb` for colour, which is almost every image;
   * false for data.
   * `pixelated` for pixel art: enlarged, each texel stays a hard square, as
   * CSS's image-rendering: pixelated draws it.
   * Returns { texture, view, width, height, pixelated }; engine.unload() it when done.
   */
  async loadTexture(source, { srgb = true, pixelated = false, label = 'texture', fetch = globalThis.fetch } = {}) {
    this._assertAlive('loadTexture');
    let image = source;
    if (typeof source === 'string') {
      const response = await fetch(source);
      if (!response.ok) throw new Error(`loadTexture: ${source} returned ${response.status}`);
      image = await response.blob();
    }
    // The bytes as authored: no colour conversion and no premultiplying, as
    // for glTF images (scene/gltf/images.js). Whether they are sRGB is the
    // texture's format's business.
    const bitmap = image instanceof ImageBitmap
      ? image
      : await createImageBitmap(image, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    this._assertAlive('loadTexture');
    const texture = createTexture2D(this.gpu, {
      label, width: bitmap.width, height: bitmap.height, srgb, mipmapped: true,
    });
    uploadImage(this.gpu, texture, bitmap);
    // Bled: a sprite is often enlarged, and its clear pixels must not fringe it.
    generateMipmaps(this.gpu, texture, { bleed: true });
    if (bitmap !== source) bitmap.close();
    return { texture, view: texture.createView(), width: texture.width, height: texture.height, pixelated: pixelated === true };
  }

  /**
   * A font for scene.addText: any CSS font the page can use, rasterised as a
   * distance field at the size it names. Waits for a web font to load.
   *
   *   const font = await engine.loadFont('64px Inter');
   *
   * Rasterise near the size text is mostly seen at: sharp above it, and
   * sharp down to an eighth of it (see SPREAD in render/text.js).
   */
  async loadFont(css) {
    this._assertAlive('loadFont');
    const px = /(\d+(?:\.\d+)?)px/.exec(css);
    if (px === null) throw new Error(`loadFont: the font needs a size in pixels, like '64px sans-serif'; got '${css}'`);
    await document.fonts?.load(css);
    this._assertAlive('loadFont');
    return new Font(this.gpu, css, Number(px[1]));
  }

  /**
   * A 3D colour LUT from an Adobe .cube file, for grading:
   *
   *   engine.grading = { lut: await engine.loadLUT('film.cube') };
   *
   * `source` is a URL or the file's text. `fetch` downloads a URL, as every
   * loader's does.
   */
  async loadLUT(source, { fetch = globalThis.fetch } = {}) {
    this._assertAlive('loadLUT');
    let text = source;
    if (!/LUT_3D_SIZE/i.test(source)) {
      const response = await fetch(source);
      if (!response.ok) throw new Error(`loadLUT: ${source} returned ${response.status}`);
      text = await response.text();
    }
    this._assertAlive('loadLUT');
    return uploadLUT(this.gpu, parseCube(text));
  }

  /** Colour grading, read and set any time: { whiteBalance, contrast, saturation, lut }. See render/grading.js. */
  get grading() {
    return this.renderer.post.grading;
  }

  set grading(value) {
    this.renderer.post.grading = value ?? null;
  }

  /**
   * Load a Radiance .hdr panorama and bake it into an Environment: ambient
   * light, reflections and background. Hand it to createScene({ environment }).
   *
   *   const studio = await engine.loadEnvironment('studio.hdr');
   *   const scene = engine.createScene({ environment: studio });
   *
   * `source` is a URL, an ArrayBuffer or a Uint8Array. Other options are the
   * Environment's -- `size` to bake the cube smaller than the map, say. The
   * environment is yours: engine.unload() it when no scene uses it any more.
   */
  async loadEnvironment(source, options = {}) {
    this._assertAlive('loadEnvironment');
    const { fetch: fetchImpl = globalThis.fetch, ...environmentOptions } = options;
    let bytes = source;
    if (typeof source === 'string') {
      const response = await fetchImpl(source);
      if (!response.ok) throw new Error(`loadEnvironment: ${source} returned ${response.status}`);
      bytes = new Uint8Array(await response.arrayBuffer());
      this._assertAlive('loadEnvironment');
    } else if (source instanceof ArrayBuffer) {
      bytes = new Uint8Array(source);
    }
    const map = parseHDR(bytes, { maxDimension: this.gpu.limits.maxTextureDimension2D });
    return new Environment(this.gpu, { ...environmentOptions, map });
  }

  /**
   * A Gaussian splat capture, for scene.addSplats: a .ply as 3D Gaussian
   * Splatting training writes it, or a .splat.
   *
   *   const room = await engine.loadSplats('room.ply');
   *   scene.addSplats({ splats: room });
   *
   * `source` is a URL, an ArrayBuffer or a Uint8Array. `fetch` downloads a
   * URL, as every loader's does. Returns { count, min, max }: how many
   * splats, and the box around their centres. engine.unload() it when no
   * scene draws it any more.
   */
  async loadSplats(source, { fetch = globalThis.fetch } = {}) {
    this._assertAlive('loadSplats');
    let bytes = source;
    if (typeof source === 'string') {
      const response = await fetch(source);
      if (!response.ok) throw new Error(`loadSplats: ${source} returned ${response.status}`);
      bytes = new Uint8Array(await response.arrayBuffer());
      this._assertAlive('loadSplats');
    }
    const data = await readSplats(bytes, { decodeImage: (image) => decodeImageBytes(this.gpu, image) });
    // The pipeline for its harmonics, built before it can be drawn.
    await this.renderer.splats.ready(data.degree);
    this._assertAlive('loadSplats');
    return new Splats(this.gpu, data);
  }

  /**
   * Load a .glb/.gltf and get back something scene.add() can take.
   *
   * `source` is a URL string, an ArrayBuffer, or a Uint8Array. This is the only
   * slow call in the API, which is exactly where a progress indicator belongs.
   *
   * `retainGeometry` keeps each primitive's positions and indices on the CPU
   * after upload, which is what Scene.raycast needs to answer with triangles
   * instead of bounding boxes -- and, for a skinned or morphed mesh, its joint
   * influences and morph deltas, so it is picked as posed. Off by default: it
   * costs 12 bytes per vertex plus 4 per index on the JS heap (32 more per
   * skinned vertex, and the deltas) for as long as the asset lives, and most
   * scenes never pick.
   *
   * `fetch` replaces the fetch used for the buffers and images a .gltf names.
   * Those URLs come from the file, so an app loading files it did not write
   * can use this to refuse, rewrite or restrict them -- same-origin only, say,
   * or `credentials: 'omit'`. Bytes with no `baseURL` can reach no URL at all.
   */
  async load(source, options = {}) {
    const { retainGeometry = false } = options;
    this._assertAlive('load');
    let bytes = source;
    // Against the page, as a URL in it would be: a relative one, 'models/',
    // reached no file at all -- new URL() refuses to resolve against it.
    const page = globalThis.location?.href ?? 'http://localhost/';
    let baseURL = options.baseURL === undefined ? undefined : new URL(options.baseURL, page);

    // Every await is a point where the engine may have been destroyed -- the
    // canvas left the page or the device was lost, and both now do that on
    // their own. Carrying on built an asset on a dead device, which a
    // recovering app then handed to its new engine.
    if (typeof source === 'string') {
      baseURL = baseURL ?? new URL(source, page);
      // Through options.fetch, as everything the load downloads is: the one rule every loader keeps.
      const response = await (options.fetch ?? globalThis.fetch)(source);
      if (!response.ok) throw new Error(`load: ${source} returned ${response.status}`);
      bytes = new Uint8Array(await response.arrayBuffer());
      this._assertAlive('load');
    } else if (source instanceof ArrayBuffer) {
      bytes = new Uint8Array(source);
    }

    // What the file may ask for is bounded by what this device can hold, and
    // what it may fetch by `options.fetch`, when the app supplies one: a
    // .gltf names its own buffers and images, so loading one from a stranger
    // requests URLs the stranger chose.
    const { limits } = this.gpu;
    const fetchImpl = options.fetch ?? globalThis.fetch;
    // Images start as soon as the document is read: fetched and decoded
    // while its buffers download and its geometry builds, and each uploaded
    // as it lands, while the rest are still arriving. Measured on Sponza,
    // that overlap took its load from 2.2 s to 1.3 s.
    let decoding = null;
    let textures = null;
    let model;
    try {
      model = await loadGLTF(bytes, {
        baseURL, fetchImpl, maxBytes: limits.maxBufferSize,
        onDocument: (json, ready) => {
          textures = new GLTFTextures(this.gpu, json, []);
          const spaces = imageColorSpaces(json);
          decoding = decodeImages(json, ready, {
            baseURL, fetchImpl, maxDimension: limits.maxTextureDimension2D,
            onImage: (index, bitmap) => { if (!this._destroyed) textures.upload(index, bitmap, spaces.get(index)); },
          });
        },
      });
    } catch (error) {
      // A geometry that would not build leaves images decoding and uploading:
      // free both when they land.
      decoding?.then((bitmaps) => {
        for (const bitmap of bitmaps) bitmap?.close?.();
        textures.destroy();
      });
      throw error;
    }
    const bitmaps = await decoding;
    if (this._destroyed) {
      for (const bitmap of bitmaps) bitmap?.close?.();
      textures.destroy();
      this._assertAlive('load');
    }
    // Built as the images landed; given them too, for anything not made yet.
    textures.bitmaps = bitmaps;

    // Filled in as it is built, so a throw part way through -- a texture past
    // the device limit, a mesh with too many targets, a pipeline that will
    // not compile -- frees exactly what exists. It used to leak all of it,
    // material ids included, which then counted against the 4096 for good.
    const asset = {
      nodes: model.nodes, meshes: [], roots: model.roots, materials: model.materials, materialIds: [],
      animations: model.animations, skins: model.skins, source: model.source,
      lights: model.lights, cameras: model.cameras, textures,
      /** Which engine made it. unload() refuses an asset from another. */
      engine: this,
    };

    try {
      // Materials first: a primitive needs its material id before it can be
      // turned into something the draw list can sort.
      for (const material of model.materials) {
        asset.materialIds.push(
          this.renderer.materials.register(material, textures.texturesFor(material)),
        );
      }

      // Every image that was going to be uploaded now has been, so the decoded
      // copies are dead weight. An ImageBitmap holds native memory the collector
      // frees only when it gets round to it, and a texture-heavy asset is
      // hundreds of megabytes of them -- Sponza decodes to over a gigabyte.
      // close() gives it back at a known moment instead.
      textures.releaseBitmaps();

      model.meshes.forEach((mesh, m) => {
        const primitives = [];
        asset.meshes.push({
          name: mesh.name,
          // How many morph targets every primitive here carries. The scene reads
          // it to decide whether a node instancing this mesh needs weights.
          targetCount: mesh.targetCount,
          primitives,
        });
        mesh.primitives.forEach((primitive, p) => {
          const built = {
            vertexBuffer: null,
            indexBuffer: null,
            /** Whether this primitive carries influences. The scene reads this. */
            skinned: primitive.jointIndices != null,
            // Null unless the mesh is rigged. A second vertex buffer, bound at
            // slot 1 by skinned pipelines only, so static meshes carry nothing.
            skinBuffer: null,
            indexCount: primitive.indexCount,
            bounds: primitive.bounds,
            /** Whether an edge has one triangle: the shadow pass draws it whole. See openSurface. */
            open: openSurface(primitive.positions, primitive.indices),
            // How far each target reaches, which is all a bound needs. The deltas
            // themselves are a GPU buffer; this is the one number the CPU keeps.
            morphExtent: primitive.morph?.extent ?? null,
            // Its target count and stride packed as draw data carries them, and
            // where its deltas landed in the engine-wide arena. Zero for an
            // unmorphed primitive, which is what makes the vertex shader skip
            // the loop entirely. Packed first: it is the check that can throw.
            morphCountStride: primitive.morph
              ? packMorphCountStride(primitive.morph.targetCount, primitive.morph.stride)
              : 0,
            morphBase: 0,
            /** How many floats of the arena that is, for unload() to give back. */
            morphFloats: 0,
            // Only when asked. Scene.raycast tests triangles for primitives that have
            // these and falls back to the bounding box for those that do not, so the
            // flag buys precision with memory and nothing else changes.
            positions: retainGeometry ? primitive.positions : undefined,
            indices: retainGeometry ? primitive.indices : undefined,
            // And what deforms them, so a skinned or morphed mesh is picked at
            // its posed triangles rather than its box.
            jointIndices: retainGeometry && primitive.jointIndices ? primitive.jointIndices : undefined,
            jointWeights: retainGeometry && primitive.jointIndices ? primitive.jointWeights : undefined,
            morphDeltas: retainGeometry && primitive.morph ? primitive.morph.deltas : undefined,
            materialId: primitive.material >= 0
              ? asset.materialIds[primitive.material]
              : this._defaultMaterial(),
            meshIndex: m,
            /** Renderables drawing this, across every scene. See Scene.add. */
            instances: 0,
          };
          primitives.push(built);

          built.vertexBuffer = createBuffer(this.gpu, {
            label: `${mesh.name}[${p}].vertices`,
            data: primitive.vertices,
            usage: GPUBufferUsage.VERTEX,
          });
          built.indexBuffer = createBuffer(this.gpu, {
            label: `${mesh.name}[${p}].indices`,
            data: primitive.indices,
            usage: GPUBufferUsage.INDEX,
          });
          if (primitive.jointIndices) {
            built.skinBuffer = createBuffer(this.gpu, {
              label: `${mesh.name}[${p}].skin`,
              data: new Uint8Array(packSkinVertices(
                primitive.jointIndices, primitive.jointWeights, primitive.vertexCount,
              )),
              usage: GPUBufferUsage.VERTEX,
            });
          }
          if (primitive.morph) {
            built.morphBase = this.renderer.morph.allocate(primitive.morph.deltas);
            built.morphFloats = primitive.morph.deltas.length;
          }
        });
      });

      // Every pipeline the asset can possibly need, compiled off-thread, before
      // the caller is even told the asset exists.
      const variants = new Set();
      for (const mesh of asset.meshes) {
        for (const primitive of mesh.primitives) {
          variants.add(this.renderer.materials.variants[primitive.materialId]);
        }
      }
      await this.renderer.ensureVariants([...variants]);
      this._assertAlive('load');
    } catch (error) {
      textures.releaseBitmaps();
      this._free(asset);
      throw error;
    }

    return asset;
  }

  /** glTF lets a primitive have no material; the spec's default is a white dielectric. */
  _defaultMaterial() {
    if (this._defaultMaterialId < 0) {
      this._defaultMaterialId = this.renderer.materials.register(DEFAULT_MATERIAL);
    }
    return this._defaultMaterialId;
  }

  /**
   * Free what any load call made: a model from load(), a texture from
   * loadTexture(), a font from loadFont(), a LUT from loadLUT(), an
   * environment from loadEnvironment(), splats from loadSplats(). Freeing
   * twice does nothing.
   *
   * A model's buffers, textures, material ids and morph deltas: remove it
   * from every scene first, and this throws if any scene still draws it. A
   * texture, font, LUT or environment is freed as it stands -- stop using it
   * first, as nothing here can see what still reads it.
   *
   * load() allocates on every call, the same file included, and until this
   * existed nothing ever gave it back. An editor that re-imports on each save
   * kept every version it had ever loaded, and the 4097th material threw.
   */
  unload(asset) {
    if (asset.unloaded) return;
    if (!Array.isArray(asset.meshes)) {
      // Everything but a model: a font or an environment frees itself; a
      // texture or a LUT is its GPU texture.
      // A target's pooled frame textures go with it.
      if (asset instanceof RenderTarget) this.renderer.graph.forget(asset);
      if (asset instanceof Font || asset instanceof Environment || asset instanceof RenderTarget || asset instanceof Splats) asset.destroy();
      else if (typeof asset.texture?.destroy === 'function') asset.texture.destroy();
      else throw new Error('unload: this is not something load, loadTexture, loadFont, loadLUT, loadEnvironment, loadSplats or createTarget returned');
      asset.unloaded = true;
      return;
    }
    // Its ids and arena ranges index THIS engine's registry and morph store.
    // Unloading another engine's asset -- the one from before a device loss,
    // say -- freed live entries here, and the next load overwrote them.
    if (asset.engine !== this) throw new Error('unload: this asset was loaded by another engine');
    for (const mesh of asset.meshes) {
      for (const primitive of mesh.primitives) {
        if (primitive.instances > 0) {
          throw new Error(`unload: mesh "${mesh.name}" is still in a scene; remove it first`);
        }
      }
    }
    this._free(asset);
  }

  /** Everything an asset holds, freed. Also what a failed load() undoes. */
  _free(asset) {
    asset.unloaded = true;
    for (const mesh of asset.meshes) {
      for (const primitive of mesh.primitives) {
        primitive.vertexBuffer?.destroy();
        primitive.indexBuffer?.destroy();
        primitive.skinBuffer?.destroy();
        if (primitive.morphFloats > 0) {
          this.renderer.morph.free(primitive.morphBase, primitive.morphFloats);
        }
      }
    }
    for (const id of asset.materialIds) this.renderer.materials.release(id);
    asset.textures.destroy();
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
   *
   * `hud: { scene, camera }` draws a second, 2D scene over every frame
   * through its Camera2D, or a plain one if none is given.
   */
  run(scene, camera, options = {}) {
    this._assertAlive('run');
    if (!(scene instanceof Scene)) {
      throw new Error('run: takes (scene, camera, { update, frame, hud }), the scene first, from createScene()');
    }
    const { update, frame } = options;
    const hud = this._hud('run', options);
    if (this._running) throw new Error('run: already running; call stop() first');
    this._running = true;
    // The time spent stopped is not a frame's worth of time: without this,
    // the first frame after a restart jumped clips, particles and update() a
    // quarter of a second.
    this.clock._last = -1;

    const loop = (nowMs) => {
      // The device going away ends the loop for good, so tear the running
      // flag down with it. Returning without stop() left _running true, and
      // every later run() threw 'already running' with no way back short of
      // destroy() -- a lost device wedged the engine rather than stopping it.
      if (!this._running) return;
      if (this.gpu.destroyed) { this.destroy(); return; }
      // A canvas that has left the document can never be seen again, so every
      // frame drawn into it is waste. This is what a live editor's reload
      // leaves behind: it rewrites the page in place (document.open/write), no
      // pagehide fires, and each old engine kept rendering and kept its GPU
      // device -- one more full loop per edit, until the adapter ran out.
      if (this.gpu.canvas.isConnected === false) { this.destroy(); return; }
      this._raf = requestAnimationFrame(loop);

      this.clock.begin(nowMs / 1000);
      // Before update(), so a callback that reads a bone's world position sees
      // this frame's pose rather than last frame's.
      scene.advance(this.clock.realDelta);
      hud?.scene.advance(this.clock.realDelta);
      if (update) while (this.clock.step()) update(this.clock.fixedDt, this.clock.elapsed);
      else this.clock.accumulator = 0;   // nothing to simulate; do not let it grow

      if (frame) frame(this.clock.alpha, this.clock);
      // update() or frame() may have stopped the loop, or destroyed the engine.
      if (!this._running) return;
      if (this.onDemand && this._idle(scene, camera, hud)) {
        this.skippedFrames++;
      } else {
        this.renderer.render(scene, camera, this.jobs, null, hud);
        this._remember(scene, camera, hud);
      }

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

  /** Draw the next frame run() reaches, even if nothing seems to have changed. */
  invalidate() {
    this._drawn = null;
  }

  /**
   * Would this frame draw exactly what the last one run() drew? Everything a
   * frame is drawn from, checked: the scene's structure, its transforms,
   * materials, properties, animations, particles and morph weights; the
   * camera; the canvas size; every live setting; pipelines still building,
   * which a frame draws without; and debug lines, which are drawn per frame.
   * No shader here depends on time, so the same inputs give the same image.
   */
  _idle(scene, camera, hud = null) {
    const last = this._drawn;
    if (last === null || last.scene !== scene || last.camera !== camera) return false;
    if (last.hud !== hud || (hud !== null && !this._hudIdle(hud, last))) return false;
    if (scene.revision !== last.revision || scene.changes !== last.changes) return false;
    if (scene.transforms._anyDirty || scene.transforms.movedPending) return false;
    if (scene.changedMaterials.size > 0 || scene.animating || scene.particlesActive) return false;
    if (this.gpu.width !== last.width || this.gpu.height !== last.height) return false;
    const renderer = this.renderer;
    if (renderer.debug.count > 0) return false;
    // A frame drawn without something still building is not the last word,
    // and neither is the next while it builds. Asking only now missed a build
    // that finished between two frames: the one drawn without it stayed up.
    if (last.building || this._building()) return false;
    // Still easing toward its exposure: the image changes with nothing moved.
    if (renderer.autoExposure && renderer.post.autoExposure.adapting) return false;
    // With the size and pixel ratio, as a frame updates it: a Camera2D's view
    // changes with the ratio alone, when the page is zoomed.
    camera.update(this.gpu.width / this.gpu.height, this.gpu.width, this.gpu.height, this.gpu.pixelRatio);
    if (!sameFloats(camera.view, last.view) || !sameFloats(camera.projection, last.projection)) return false;
    if (!sameMorphs(scene.morphs, last.morphs)) return false;
    // A 2D view's background, ambient and snapping are not in its matrices.
    if (camera.is2D && (!sameFloats(camera.background, last.background) || !sameFloats(camera.ambient, last.ambient)
      || camera.pixelSnap !== last.pixelSnap)) return false;
    return this._settings() === last.settings;
  }

  /** Would the hud draw what it last did? Its scene and camera, as _idle checks the main one's. */
  _hudIdle({ scene, camera }, last) {
    if (scene.revision !== last.hudRevision || scene.changes !== last.hudChanges) return false;
    if (scene.transforms._anyDirty || scene.transforms.movedPending || scene.animating) return false;
    camera.update(this.gpu.width / this.gpu.height, this.gpu.width, this.gpu.height, this.gpu.pixelRatio);
    return sameFloats(camera.view, last.hudView) && camera.pixelSnap === last.hudSnap
      && sameFloats(camera.ambient, last.hudAmbient);
  }

  /**
   * Whether a frame drawn now is drawn without something it will have once
   * it is built: pipelines for a feature switched on, FXAA, or the scaler
   * for a lower resolution.
   */
  _building() {
    const renderer = this.renderer;
    for (const set of renderer._variantSets.values()) if (!set.ready) return true;
    if (renderer.post.antialias && renderer.post.fxaaPipeline === undefined) return true;
    return renderer.upscaler.pending;
  }

  _remember(scene, camera, hud = null) {
    const last = this._drawn ?? { view: new Float32Array(16), projection: new Float32Array(16), morphs: [], hudView: new Float32Array(16) };
    last.scene = scene;
    last.building = this._building();
    last.camera = camera;
    last.hud = hud;
    if (hud !== null) {
      last.hudRevision = hud.scene.revision;
      last.hudChanges = hud.scene.changes;
      last.hudView.set(hud.camera.view);
      last.hudSnap = hud.camera.pixelSnap;
      last.hudAmbient = Float32Array.from(hud.camera.ambient);
    }
    last.revision = scene.revision;
    last.changes = scene.changes;
    last.width = this.gpu.width;
    last.height = this.gpu.height;
    last.view.set(camera.view);
    last.projection.set(camera.projection);
    last.morphs = scene.morphs.map((m) => Float32Array.from(m.weights));
    last.background = camera.is2D ? Float32Array.from(camera.background) : null;
    last.ambient = camera.is2D ? Float32Array.from(camera.ambient) : null;
    last.pixelSnap = camera.pixelSnap;
    last.settings = this._settings();
    this._drawn = last;
  }

  /** Every live setting a frame reads, as one comparable string. */
  _settings() {
    const r = this.renderer, p = r.post;
    return settingsSignature([
      r.exposure, r.autoExposure, r.resolution, r.fog, r.dof, r.skybox, r.shadowDistance, r.lightDistance, r.ao, r.oit, r.debug.depthTest,
      p.threshold, p.knee, p.filterRadius, p.strength, p.levels, p.antialias, p.grading,
      // The shadow settings a frame reads; the rest are fixed at creation.
      r.shadows.lambda, r.shadows.casterExtent, r.shadows.normalBias,
    ]);
  }

  /**
   * One frame, synchronously. Use this when you own the loop. `hud` is
   * as run() takes it; `target`, one from createTarget, takes the frame in
   * place of the canvas.
   */
  renderFrame(scene, camera, options = {}) {
    this._assertAlive('renderFrame');
    const { target = null } = options;
    if (target !== null && !(target instanceof RenderTarget && target.rhi === this.gpu)) {
      throw new Error("renderFrame: target must be one this engine's createTarget made");
    }
    if (target?.unloaded) throw new Error('renderFrame: that target was unloaded');
    this.renderer.render(scene, camera, this.jobs, null, this._hud('renderFrame', options), target);
    // run's idle check compares with what run last drew. A frame drawn here
    // replaced it on the canvas, or changed a target something shows: either
    // way the next frame of run is drawn.
    this._drawn = null;
  }

  /**
   * A texture to draw a scene into, with renderFrame's `target`: a sprite
   * shows it as it would a loaded image. `size` is [width, height] in its own
   * pixels; `pixelated` shows it enlarged with hard edges, as loadTexture's
   * does -- for pixel art drawn small and shown large.
   *
   *   const map = await engine.createTarget({ size: [256, 256] });
   *   hud.addSprite({ texture: map, position: [16, 16], pivot: [0, 0] });
   *   engine.renderFrame(world, overhead, { target: map });   // whenever it changes
   */
  async createTarget({ size, pixelated = false, label = 'target' } = {}) {
    this._assertAlive('createTarget');
    const most = this.gpu.limits.maxTextureDimension2D;
    const [width, height] = size ?? [];
    if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0 && width <= most && height <= most)) {
      throw new Error(`createTarget: size must be [width, height], whole pixels from 1 to ${most}, got ${size}`);
    }
    return this.renderer.createTarget(width, height, { pixelated: pixelated === true, label });
  }

  /** A run or renderFrame's hud with its camera filled in: the one given, or a plain Camera2D, kept. */
  _hud(caller, { hud = null, overlay }) {
    if (overlay !== undefined) throw new Error(`${caller}: overlay is now hud, with the same { scene, camera }`);
    if (hud === null) return null;
    if (!(hud.scene instanceof Scene)) throw new Error(`${caller}: hud needs { scene, camera }, the scene from createScene()`);
    if (hud.camera !== undefined && hud.camera.is2D !== true) throw new Error(`${caller}: the hud's camera must be a Camera2D`);
    if (hud.camera !== undefined) return hud;
    this._hudCamera ??= new Camera2D();
    // The same object each time for the same scene, so run's idle check can compare it.
    if (this._hudFilled?.scene !== hud.scene) this._hudFilled = { scene: hud.scene, camera: this._hudCamera };
    return this._hudFilled;
  }

  get stats() {
    return this.renderer.stats;
  }

  /**
   * Capture a scene's reflection probes (scene.addProbe): six
   * renders of the scene from each, prefiltered. All of them, or a list of
   * their nodes.
   * Recapture after what they see has changed. Async: the first capture
   * builds the pipelines that read probes.
   */
  async captureProbes(scene, probes) {
    this._assertAlive('captureProbes');
    // Where each probe's node is now, before any is captured from there.
    scene.update();
    const records = probes === undefined ? undefined : probes.map((node) => scene._probe('captureProbes', node));
    await this.renderer.captureProbes(scene, records);
  }

  /** Lines for the next frame: line, box, sphere, axes. See render/debug.js. */
  get debug() {
    return this.renderer.debug;
  }

  /**
   * Release everything this engine owns.
   *
   * The environment is destroyed only if this engine made it -- its default
   * sky. One from loadEnvironment is the caller's, freed with unload.
   */
  destroy() {
    if (this._destroyed) return;   // the loop may already have done it
    this._destroyed = true;
    this.stop();
    this.jobs.destroy();
    this.renderer.destroy();
    if (this._ownsEnvironment) this.environment.destroy();
    this.gpu.destroy();
  }
}

function sameFloats(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function sameMorphs(morphs, last) {
  if (morphs.length !== last.length) return false;
  for (let m = 0; m < morphs.length; m++) {
    if (morphs[m].weights.length !== last[m].length || !sameFloats(morphs[m].weights, last[m])) return false;
  }
  return true;
}

// A setting that is a GPU object -- a LUT's texture -- has no fields worth
// comparing but an identity that is, so it stands in as a number.
const settingIds = new WeakMap();
let nextSettingId = 1;
function settingsSignature(values) {
  return JSON.stringify(values, (key, value) => {
    if (value === null || typeof value !== 'object' || Array.isArray(value) || ArrayBuffer.isView(value)) return value;
    if (Object.getPrototypeOf(value) === Object.prototype) return value;
    let id = settingIds.get(value);
    if (id === undefined) { id = nextSettingId++; settingIds.set(value, id); }
    return `#${id}`;
  });
}

// Names 1.0 changed: the old ones say so. See renamed.
renamed(Winding.prototype, 'rhi', 'gpu', 'engine');
