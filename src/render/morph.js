// Morph target storage: the deltas and the weights that mix them.
//
// Two buffers with opposite lifetimes, in one place because nothing ever wants
// only one of them.
//
// THE DELTAS are per PRIMITIVE and never change. They arrive at load and stay,
// so this is an arena: every morphed primitive in the engine is appended into
// one buffer and told the float it starts at. One buffer rather than one per
// primitive because the vertex shader reads them through the frame bind group,
// and a binding per primitive would mean a bind group per batch -- which the
// draw path does not have and should not grow one for.
//
// THE WEIGHTS are per INSTANCE and change every frame. Same shape as the joint
// palette for the same reason: two faces sharing a mesh are one batch and one
// draw call, and what makes them different expressions is where each one's
// weights begin.
//
// Growing the arena COPIES ON THE GPU rather than through a CPU staging array.
// Keeping a shadow copy would mean holding every delta in system memory for
// the life of the engine to serve a reallocation that may never happen, and a
// face rig is tens of megabytes. copyBufferToBuffer is what COPY_SRC is for.

import { grownCapacity } from '../core/grow.js';
import { storageCapacity } from '../rhi/buffer.js';

/** Bytes per stored float, in both buffers. */
export const MORPH_FLOAT_BYTES = 4;

/**
 * Pack a primitive's target count and stride into the one word draw data has
 * room for.
 *
 * Both are properties of the primitive and both are small: a stride is 3, 6 or
 * 9, and a target count past 65535 is not a mesh anyone authored. Splitting
 * them into two fields would push DrawData from 128 bytes to 144 for twelve
 * bits of information.
 */
export function packMorphCountStride(count, stride) {
  if (count > 0xffff) {
    throw new Error(`morph: ${count} targets is past the 65535 a draw can address`);
  }
  return (count & 0xffff) | (stride << 16);
}

export class MorphStore {
  constructor(rhi, { deltaCapacity = 4096, weightCapacity = 256 } = {}) {
    this.rhi = rhi;

    /** Floats of delta storage handed out so far. */
    this.deltaCount = 0;
    this.deltaCapacity = deltaCapacity;
    this.deltaBuffer = this._createDeltaBuffer(deltaCapacity);
    /** Freed ranges below deltaCount, as { base, length }, sorted by base. */
    this._holes = [];

    this.weightCapacity = weightCapacity;
    this.weightData = new Float32Array(weightCapacity);
    this.weightBuffer = this._createWeightBuffer(weightCapacity);
    /** Start float of each morph instance, by index into scene.morphs. */
    this.offsets = new Uint32Array(64);
    this.weightCount = 0;

    /** Bumped when either buffer is replaced, so bind groups know to rebuild. */
    this.revision = 0;
  }

  _createDeltaBuffer(capacity) {
    return this.rhi.device.createBuffer({
      label: 'morph-deltas',
      // COPY_SRC so growing can move the existing deltas across on the GPU.
      size: capacity * MORPH_FLOAT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  _createWeightBuffer(capacity) {
    return this.rhi.device.createBuffer({
      label: 'morph-weights',
      size: capacity * MORPH_FLOAT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Append one primitive's deltas, returning the float they start at.
   *
   * Called at load, never during a frame. The returned base goes into the
   * primitive's draw data and is what makes one arena addressable.
   */
  allocate(deltas) {
    // A hole an unloaded asset left behind first, if one is big enough.
    // ponytail: first fit, no compaction -- a hole smaller than every later
    // asset stays a hole. Compacting means moving live deltas and rewriting
    // every primitive's morphBase; worth it only if fragmentation shows up.
    for (let h = 0; h < this._holes.length; h++) {
      const hole = this._holes[h];
      if (hole.length < deltas.length) continue;
      const base = hole.base;
      hole.base += deltas.length;
      hole.length -= deltas.length;
      if (hole.length === 0) this._holes.splice(h, 1);
      this.rhi.queue.writeBuffer(this.deltaBuffer, base * MORPH_FLOAT_BYTES, deltas);
      return base;
    }

    const base = this.deltaCount;
    const needed = base + deltas.length;

    if (needed > this.deltaCapacity) {
      // Two high-end face rigs are enough to pass the default binding limit.
      const capacity = grownCapacity(
        this.deltaCapacity, needed, storageCapacity(this.rhi, MORPH_FLOAT_BYTES), 'morph delta floats',
      );
      const grown = this._createDeltaBuffer(capacity);

      // Only the part that has been written. Copying the whole old buffer
      // would be legal and would move uninitialized bytes.
      if (base > 0) {
        const encoder = this.rhi.device.createCommandEncoder({ label: 'morph-grow' });
        encoder.copyBufferToBuffer(
          this.deltaBuffer, 0, grown, 0, base * MORPH_FLOAT_BYTES,
        );
        this.rhi.queue.submit([encoder.finish()]);
      }

      this.deltaBuffer.destroy();
      this.deltaBuffer = grown;
      this.deltaCapacity = capacity;
      this.revision++;
    }

    this.rhi.queue.writeBuffer(this.deltaBuffer, base * MORPH_FLOAT_BYTES, deltas);
    this.deltaCount = needed;
    return base;
  }

  /**
   * Give back `length` floats at `base`, from an unloaded asset.
   *
   * The arena used to be append-only, so every reload of a face rig added its
   * tens of megabytes again until the buffer passed the binding limit. Holes
   * are kept sorted and merged; one that reaches the end shortens the arena.
   */
  free(base, length) {
    let at = 0;
    while (at < this._holes.length && this._holes[at].base < base) at++;
    this._holes.splice(at, 0, { base, length });

    const next = this._holes[at + 1];
    if (next && base + length === next.base) {
      this._holes[at].length += next.length;
      this._holes.splice(at + 1, 1);
    }
    const previous = this._holes[at - 1];
    if (previous && previous.base + previous.length === base) {
      previous.length += this._holes[at].length;
      this._holes.splice(at, 1);
    }

    const last = this._holes[this._holes.length - 1];
    if (last && last.base + last.length === this.deltaCount) {
      this.deltaCount = last.base;
      this._holes.pop();
    }
  }

  /**
   * Gather every instance's weights into one upload.
   *
   * Unconditional, like the joint palette: a weight can be written by an
   * animation, by user code through node.weights, or by nothing at all, and
   * there is no dirty flag to read because a live array write sets none. The
   * total is one float per target per instanced morphed mesh, which is orders
   * of magnitude below what the deltas already cost.
   */
  update(scene) {
    const morphs = scene.morphs;
    if (morphs.length === 0) {
      this.weightCount = 0;
      return;
    }

    let total = 0;
    for (const morph of morphs) total += morph.weights.length;
    if (total > this.weightCapacity) this._growWeights(total);
    if (this.offsets.length < morphs.length) {
      this.offsets = new Uint32Array(grownCapacity(this.offsets.length, morphs.length));
    }

    let at = 0;
    for (let m = 0; m < morphs.length; m++) {
      this.offsets[m] = at;
      this.weightData.set(morphs[m].weights, at);
      at += morphs[m].weights.length;
    }

    this.weightCount = at;
    this.rhi.queue.writeBuffer(this.weightBuffer, 0, this.weightData, 0, at);
  }

  _growWeights(needed) {
    const capacity = grownCapacity(
      this.weightCapacity, needed, storageCapacity(this.rhi, MORPH_FLOAT_BYTES), 'morph weights',
    );
    this.weightData = new Float32Array(capacity);
    this.weightBuffer.destroy();
    this.weightBuffer = this._createWeightBuffer(capacity);
    this.weightCapacity = capacity;
    this.revision++;
  }

  destroy() {
    this.deltaBuffer.destroy();
    this.weightBuffer.destroy();
  }
}
