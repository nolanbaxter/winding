// Device + surface.
//
// createDevice() does everything: adapter negotiation, feature selection,
// canvas configuration, depth buffer, resize observation, error plumbing.
// There is no init order to remember and no second call to forget.
//
//   const rhi = await createDevice(canvas);
//   rhi.device      // the raw GPUDevice; public on purpose
//   rhi.currentColorView()  // this frame's swap chain image

import { DEBUG, assert } from '../core/assert.js';

/**
 * Reverse-Z depends on how floats are DISTRIBUTED, not on bit count, so the
 * depth buffer has to be a real float format. depth24plus would work and cost
 * less bandwidth, but it is normalized-integer backed and throws away the
 * entire precision argument.
 */
export const DEPTH_FORMAT = 'depth32float';

/** Reverse-Z: the near plane is 1.0 and infinity is 0.0, so "empty" is 0.0. */
export const DEPTH_CLEAR_VALUE = 0.0;

/** Reverse-Z: nearer fragments have LARGER depth, so the test flips. */
export const DEPTH_COMPARE = 'greater';

/**
 * Features we take if the adapter offers them, and do without otherwise.
 * Requesting a feature the hardware lacks fails device creation outright, so
 * every entry here must have a working fallback path.
 */
const OPTIONAL_FEATURES = ['timestamp-query'];

/**
 * The limits a scene's size runs into: buffers that grow with objects, lights
 * and morph targets, and textures and a canvas that grow with the asset and
 * the display. Requested at whatever the adapter has.
 */
const SCALING_LIMITS = ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxTextureDimension2D'];

export async function createDevice(canvas, options = {}) {
  if (!navigator.gpu) {
    throw new Error(
      'WebGPU is not available. Needs Chrome/Edge 113+, Safari 18+, or Firefox 141+ ' +
      '(and a secure context: https or localhost).',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference ?? 'high-performance',
  });
  if (!adapter) {
    throw new Error('No WebGPU adapter. The GPU may be blocklisted, or the tab is out of resources.');
  }

  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));

  // What the adapter really has, rather than WebGPU's defaults. Without this
  // every size check in the engine compared against 8192-texel textures and
  // 128 MiB bindings on hardware that commonly allows 16384 and gigabytes.
  // Asking for exactly what the adapter reports always succeeds.
  const requiredLimits = {};
  for (const name of SCALING_LIMITS) requiredLimits[name] = adapter.limits[name];

  const device = await adapter.requestDevice({
    label: options.label ?? 'engine',
    requiredFeatures,
    requiredLimits,
  });

  return new Device(adapter, device, canvas, options);
}

export class Device {
  constructor(adapter, device, canvas, options) {
    this.adapter = adapter;
    /** The raw GPUDevice. Public: dropping to tier 0 must never require a fork. */
    this.device = device;
    this.queue = device.queue;
    this.canvas = canvas;
    this.limits = device.limits;
    this.features = device.features;

    this.width = 0;
    this.height = 0;
    this.depthTexture = null;
    this._depthView = null;
    this.destroyed = false;

    this.onDeviceLost = options.onDeviceLost ?? null;
    this.onError = options.onError ?? null;
    /**
     * Called once nothing drawn here can be seen again: the device was lost,
     * or the canvas left the document. The engine destroys itself on it; a
     * tier-0 user who owns this directly can do the same.
     */
    this.onUnusable = null;

    // WebGPU errors do NOT throw. They surface here, and without a listener
    // they vanish into the console at best.
    device.addEventListener('uncapturederror', (event) => {
      const message = `WebGPU: ${event.error.message}`;
      if (this.onError) this.onError(event.error);
      else console.error(message);
    });

    // Every resource created from this device dies with it: a driver reset, a
    // GPU hang, an out-of-memory, or the browser reclaiming a backgrounded
    // tab. This engine deliberately does NOT rebuild them.
    //
    // Rebuilding would mean keeping a CPU-side description of every GPU object
    // alive for the process lifetime, and the hard part is not the buffers or
    // the pipelines -- those are cheap to re-derive. It is the textures, with
    // their contents: either every decoded image stays resident forever, which
    // is the gigabyte load() explicitly releases, or every asset is fetched
    // and decoded again, which is load() itself. A permanent cost on every
    // module that creates a resource, for a path that fires on a driver reset.
    //
    // So the honest answer is to say so clearly and let the app reload, which
    // is what the user expects from all four causes anyway. What that needs is
    // a callback with enough in it to act on, which is what this builds.
    //
    // Not awaited -- it is a notification, not a step.
    device.lost.then((info) => {
      if (this.destroyed && info.reason === 'destroyed') return;   // we did that
      this.destroyed = true;
      // It kept resizing the canvas and allocating depth textures on a dead
      // device, for as long as the canvas lived.
      this._observer?.disconnect();
      this.onUnusable?.();

      // 'destroyed' here means someone else destroyed it, since our own case
      // returned above. Everything else is the GPU going away underneath us,
      // and reloading is the only route back.
      const detail = {
        reason: info.reason,
        message: info.message,
        /** True when creating a fresh device is likely to work. */
        recoverable: info.reason !== 'destroyed',
        /** What an application should do about it. There is one answer. */
        action: 'reload',
      };

      if (this.onDeviceLost) this.onDeviceLost(detail);
      else {
        console.error(
          `WebGPU device lost (${info.reason}): ${info.message || 'no message'}\n` +
          'Winding does not rebuild GPU state. Reload the page, or pass ' +
          'onDeviceLost to Winding.create to handle it yourself.',
        );
      }
    });

    this.context = canvas.getContext('webgpu');
    if (!this.context) throw new Error('Could not get a webgpu context from the canvas.');

    // Work in linear light, convert to sRGB exactly once, at the surface.
    // The swap chain is stored in a plain unorm format; we render
    // through an -srgb VIEW of it, so the hardware does the encode on write and
    // shaders never do gamma math by hand.
    const storageFormat = navigator.gpu.getPreferredCanvasFormat();
    // Windows and macOS both prefer bgra8unorm, so the bytes a copy produces
    // are B,G,R,A. readPixels undoes that rather than handing callers a
    // platform detail dressed up as RGBA.
    this._swapBlueAndRed = storageFormat.startsWith('bgra');
    this.viewFormat = storageFormat.endsWith('-srgb') ? storageFormat : `${storageFormat}-srgb`;

    this.context.configure({
      device,
      format: storageFormat,
      viewFormats: [this.viewFormat],
      alphaMode: 'opaque',
      // COPY_SRC as well as RENDER_ATTACHMENT, so what was drawn can be read
      // back -- see readPixels. It costs a lazy-clear optimisation the driver
      // could otherwise make on the swap chain, and buys the only way to ask
      // what colour a pixel actually is: a WebGPU canvas has no working
      // toDataURL, so without this the answer does not exist at any price.
      //
      // That matters more than a screenshot API. Until this was here, no check
      // anywhere could assert a rendered RESULT -- only that nothing threw --
      // and "nothing threw" is exactly what a NaN normal or a black default
      // texture produces.
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    this._observeResize();
  }

  _observeResize() {
    const apply = (w, h) => this.resize(w, h);

    this._observer = new ResizeObserver((entries) => {
      // Removing an observed element is itself a resize, to nothing. It is
      // the only notice a page rewritten in place (document.open) ever gives.
      if (!this.canvas.isConnected) { this.onUnusable?.(); return; }
      for (const entry of entries) {
        // device-pixel-content-box gives EXACT device pixels. The usual
        // clientWidth * devicePixelRatio is rounded twice and lands half a
        // pixel off on fractional-DPI displays, which is the real cause of
        // "why is my canvas slightly blurry".
        const box = entry.devicePixelContentBoxSize?.[0];
        if (box) {
          apply(box.inlineSize, box.blockSize);
        } else {
          const dpr = globalThis.devicePixelRatio || 1;
          const content = entry.contentBoxSize[0];
          apply(Math.round(content.inlineSize * dpr), Math.round(content.blockSize * dpr));
        }
      }
    });

    const applyFromClient = () => {
      const dpr = globalThis.devicePixelRatio || 1;
      apply(
        Math.max(1, Math.round(this.canvas.clientWidth * dpr)),
        Math.max(1, Math.round(this.canvas.clientHeight * dpr)),
      );
    };

    try {
      this._observer.observe(this.canvas, { box: 'device-pixel-content-box' });
    } catch {
      // No WebKit has ever shipped this box, so every Safari lands here.
      this._observer.observe(this.canvas);
      // And without it a change of pixel ratio alone -- the window dragged to
      // another display -- resizes no box, so the observer never fires and the
      // canvas stays at the old display's resolution. The ratio is watched
      // directly instead: a query for the current one, renewed each time it
      // stops matching.
      const watchRatio = () => {
        if (this.destroyed || !globalThis.matchMedia) return;
        matchMedia(`(resolution: ${globalThis.devicePixelRatio || 1}dppx)`)
          .addEventListener('change', () => {
            if (this.destroyed) return;
            applyFromClient();
            watchRatio();
          }, { once: true });
      };
      watchRatio();
    }

    // Observe fires asynchronously, so size it now: the object
    // is usable the instant it exists, not one animation frame later.
    applyFromClient();
  }

  resize(width, height) {
    const max = this.limits.maxTextureDimension2D;
    const w = Math.max(1, Math.min(width, max));
    const h = Math.max(1, Math.min(height, max));
    if (w === this.width && h === this.height) return;

    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;

    // The depth buffer must match the color target exactly, so it is recreated
    // rather than resized -- GPU textures are immutable in size.
    this.depthTexture?.destroy();
    this._depthView = null;
    this.depthTexture = this.device.createTexture({
      label: 'depth',
      size: [w, h],
      format: DEPTH_FORMAT,
      // TEXTURE_BINDING as well as RENDER_ATTACHMENT: the HZB build samples this
      // buffer to seed its pyramid.
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  get aspect() {
    return this.width / this.height;
  }

  /**
   * The swap chain image for this frame.
   *
   * getCurrentTexture() must be called fresh every frame -- the swap chain hands
   * out a different texture each time, and caching the view renders into an
   * image nobody is going to show.
   */
  currentColorView() {
    if (DEBUG) assert(!this.destroyed, 'currentColorView() on a destroyed device');
    return this.context.getCurrentTexture().createView({ format: this.viewFormat });
  }

  /** The device-owned depth buffer, resized with the surface. */
  depthView() {
    // Cached, not created per call. A fresh view every frame is not just
    // garbage -- every downstream identity guard (the HZB's resize check, a
    // bind group cache) silently stops holding, because the thing it compares
    // is a new object each time. The view is invalidated with the texture.
    this._depthView ??= this.depthTexture.createView({ label: 'depth' });
    return this._depthView;
  }

  /**
   * Read rendered pixels back as RGBA bytes, row by row, from the top left.
   *
   *     engine.renderFrame(scene, camera);
   *     const pixels = await engine.rhi.readPixels();
   *     const at = (x, y) => pixels.subarray((y * engine.rhi.width + x) * 4, ...);
   *
   * ONCE PER FRAME, and before anything else awaits. The swap chain hands out
   * a fresh texture each frame and presents the old one when the task ends, so
   * the `await` inside this method is already enough to lose it: a second call
   * copies from a texture that has just been handed over and cleared, and
   * returns zeros. Which is why this reads a REGION and callers index into it,
   * rather than offering a readPixel(x, y) that invites the broken shape.
   *
   * Two things are undone on the way out, both of which would otherwise leak a
   * platform detail into every caller. Rows come back 256-byte aligned because
   * that is a GPU rule, and the padding is dropped. And the swap chain is
   * bgra8unorm nearly everywhere, so blue and red are swapped back -- a method
   * that says RGBA and returns BGRA is a trap that reads as a rendering bug.
   */
  async readPixels({ x = 0, y = 0, width = this.width, height = this.height } = {}) {
    const BYTES_PER_ROW_ALIGNMENT = 256;
    const tightRow = width * 4;
    const paddedRow = Math.ceil(tightRow / BYTES_PER_ROW_ALIGNMENT) * BYTES_PER_ROW_ALIGNMENT;

    const staging = this.device.createBuffer({
      label: 'readback',
      size: paddedRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder({ label: 'readback' });
    encoder.copyTextureToBuffer(
      { texture: this.context.getCurrentTexture(), origin: { x, y } },
      { buffer: staging, bytesPerRow: paddedRow, rowsPerImage: height },
      { width, height },
    );
    this.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(staging.getMappedRange());

    const out = new Uint8Array(tightRow * height);
    for (let row = 0; row < height; row++) {
      out.set(padded.subarray(row * paddedRow, row * paddedRow + tightRow), row * tightRow);
    }
    staging.unmap();
    staging.destroy();

    if (this._swapBlueAndRed) {
      for (let i = 0; i < out.length; i += 4) {
        const b = out[i];
        out[i] = out[i + 2];
        out[i + 2] = b;
      }
    }
    return out;
  }

  destroy() {
    this.destroyed = true;
    this._observer?.disconnect();
    this.depthTexture?.destroy();
    this._depthView = null;
    this.device.destroy();
  }
}
