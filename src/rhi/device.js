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

  const device = await adapter.requestDevice({
    label: options.label ?? 'engine',
    requiredFeatures,
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

    // WebGPU errors do NOT throw. They surface here, and without a listener
    // they vanish into the console at best.
    device.addEventListener('uncapturederror', (event) => {
      const message = `WebGPU: ${event.error.message}`;
      if (this.onError) this.onError(event.error);
      else console.error(message);
    });

    // Every resource created from this device dies with it: driver reset, GPU
    // hang, OOM, or the browser reclaiming a backgrounded tab. Recovery means
    // rebuilding all of it, which is only possible because the RHI owns it all.
    // Not awaited -- this is a notification, not a step.
    device.lost.then((info) => {
      if (this.destroyed && info.reason === 'destroyed') return;   // we did that
      this.destroyed = true;
      if (this.onDeviceLost) this.onDeviceLost(info);
      else console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
    });

    this.context = canvas.getContext('webgpu');
    if (!this.context) throw new Error('Could not get a webgpu context from the canvas.');

    // Work in linear light, convert to sRGB exactly once, at the surface.
    // The swap chain is stored in a plain unorm format; we render
    // through an -srgb VIEW of it, so the hardware does the encode on write and
    // shaders never do gamma math by hand.
    const storageFormat = navigator.gpu.getPreferredCanvasFormat();
    this.viewFormat = storageFormat.endsWith('-srgb') ? storageFormat : `${storageFormat}-srgb`;

    this.context.configure({
      device,
      format: storageFormat,
      viewFormats: [this.viewFormat],
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this._observeResize();
  }

  _observeResize() {
    const apply = (w, h) => this.resize(w, h);

    this._observer = new ResizeObserver((entries) => {
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

    try {
      this._observer.observe(this.canvas, { box: 'device-pixel-content-box' });
    } catch {
      this._observer.observe(this.canvas);   // Safari < 18 has no such box
    }

    // Observe fires asynchronously, so size it now: the object
    // is usable the instant it exists, not one animation frame later.
    const dpr = globalThis.devicePixelRatio || 1;
    apply(
      Math.max(1, Math.round(this.canvas.clientWidth * dpr)),
      Math.max(1, Math.round(this.canvas.clientHeight * dpr)),
    );
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

  destroy() {
    this.destroyed = true;
    this._observer?.disconnect();
    this.depthTexture?.destroy();
    this._depthView = null;
    this.device.destroy();
  }
}
