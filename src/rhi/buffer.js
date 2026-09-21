// Buffers, and the per-frame uniform ring.

import { DEBUG, assert } from '../core/assert.js';

export function createBuffer(rhi, { label, size, usage, data }) {
  if (data) {
    // mappedAtCreation writes straight into the buffer's own memory with no
    // staging copy and no queue round-trip. Only available at creation, which
    // is why static geometry is uploaded this way and dynamic data is not.
    const buffer = rhi.device.createBuffer({
      label,
      size: size ?? align4(data.byteLength),
      usage,
      mappedAtCreation: true,
    });
    const dst = new Uint8Array(buffer.getMappedRange());
    dst.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
    return buffer;
  }

  return rhi.device.createBuffer({ label, size, usage });
}

// GPU buffer sizes must be a multiple of 4.
function align4(n) {
  return (n + 3) & ~3;
}

/**
 * Per-frame uniform storage with dynamic offsets.
 *
 * The naive approach is a bind group per object per frame, which allocates
 * constantly and rebinds for every draw. Instead: one big buffer, one bind
 * group, and a byte offset chosen at bind time.
 *
 *   ring.beginFrame();
 *   const slot = ring.alloc(64);              // 64 bytes for a mat4
 *   slot.f32.set(modelMatrix);
 *   ...
 *   ring.flush();                             // ONE writeBuffer for the frame
 *   pass.setBindGroup(GROUP_DRAW, bg, [slot.offset]);
 *
 * Writes land in a CPU staging buffer first and upload in a single call.
 * Calling writeBuffer per object instead would be hundreds of tiny transfers
 * with per-call overhead that dwarfs the 64 bytes each one carries.
 *
 * The ring spans `frames` regions. The CPU writes region N+1 while the GPU is
 * still reading region N, so neither waits on the other.
 */
export class UniformRing {
  constructor(rhi, { bytesPerFrame, frames = 2, label = 'uniform-ring' }) {
    // Queried, never hardcoded. Commonly 256, but 64 on some
    // hardware, and assuming either one is how you get a validation error that
    // only reproduces on someone else's laptop.
    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    /** Largest single binding the hardware allows; alloc() refuses more. */
    this.maxBinding = rhi.limits.maxUniformBufferBindingSize;

    this.bytesPerFrame = alignUp(bytesPerFrame, this.alignment);
    this.frames = frames;

    const total = this.bytesPerFrame * frames;
    this.buffer = rhi.device.createBuffer({
      label,
      size: total,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.staging = new ArrayBuffer(total);
    this.stagingU8 = new Uint8Array(this.staging);
    this.queue = rhi.queue;

    this.frameIndex = 0;
    this.base = 0;      // byte offset of the current frame's region
    this.head = 0;      // write cursor within the whole buffer
    this.highWater = 0;
  }

  /** Advance to the next region. Everything written last frame is now unreachable. */
  beginFrame() {
    this.frameIndex = (this.frameIndex + 1) % this.frames;
    this.base = this.frameIndex * this.bytesPerFrame;
    this.head = this.base;
  }

  /**
   * Reserve `bytes`, aligned for use as a dynamic offset.
   *
   * Returns a view into CPU staging plus the offset to pass to setBindGroup.
   * The view is valid until the next beginFrame() for this region.
   */
  alloc(bytes) {
    if (DEBUG) assert(bytes <= this.maxBinding, `uniform block ${bytes}B exceeds device limit`);

    const offset = alignUp(this.head, this.alignment);
    const end = offset + bytes;
    const limit = this.base + this.bytesPerFrame;
    // Not a DEBUG assert: overrunning would hand out a view into the NEXT
    // frame's region, which the GPU may still be reading. Silent corruption.
    if (end > limit) {
      throw new Error(
        `UniformRing exhausted: frame region is ${this.bytesPerFrame}B, ` +
        `needed ${end - this.base}B (peak ${this.highWater}B)`,
      );
    }

    this.head = end;
    const used = end - this.base;
    if (used > this.highWater) this.highWater = used;

    return {
      offset,
      f32: new Float32Array(this.staging, offset, bytes >> 2),
      u32: new Uint32Array(this.staging, offset, bytes >> 2),
    };
  }

  /** Upload this frame's region. One call, whatever the draw count. */
  flush() {
    const used = this.head - this.base;
    if (used === 0) return;
    // writeBuffer copies immediately, so the staging bytes are free to be
    // overwritten as soon as this returns.
    this.queue.writeBuffer(this.buffer, this.base, this.stagingU8, this.base, used);
  }
}

function alignUp(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}
