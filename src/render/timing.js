// Per-pass GPU timing.
//
// The CPU timings the renderer already keeps say how long it took to BUILD a
// frame, which is a different question from how long the GPU took to run it.
// Answering the second one needs the device to stamp a clock at the start and
// end of every pass, so this hangs off the render graph: the graph already
// knows the pass list and the order, and asking each pass to carry two query
// indices is the whole of it.
//
// Readback is deliberately not synchronised. Mapping a buffer the GPU may still
// be writing would stall the pipeline, which would change the number being
// measured. Instead a small ring of staging buffers is filled in turn and read
// whenever a map resolves, so `results` is a recent frame rather than the last
// one. Frames whose turn comes up while every buffer is still in flight are
// skipped rather than waited on.
//
// Three caveats worth knowing before reading anything into a number here.
//
// Browsers quantise these timestamps. Chrome rounds to 65536ns, so a pass
// shorter than 0.066ms reads as either 0 or one whole quantum and never as its
// real duration. That is why `average` exists and why it is the field to read:
// the two endpoints land on the grid independently, so the mean converges on
// the truth that no single frame can express. A handful of frames says nothing.
//
// And a "pass duration" on a GPU that overlaps work is a span, not an exclusive
// cost. Adjacent spans can sum to more than the frame took.
//
// And the one that matters most in practice: these numbers are not comparable
// ACROSS runs. The same build measured twice a minute apart came back at
// 0.588ms and 1.50ms for the same scene, because GPU clocks and whatever else
// the machine is doing dominate. Within a frame the shares are far steadier --
// the forward pass was 53% and 50% of those two totals -- so this is a tool for
// finding which pass dominates, not for proving a change made the frame faster.
// Doing that honestly needs both versions measured in one process, interleaved.

import { createBuffer } from '../rhi/buffer.js';

const QUERIES_PER_PASS = 2;
const NS_PER_MS = 1e6;

export class GpuProfiler {
  /**
   * @param rhi      the Device; its `features` decides whether this does anything
   * @param enabled  false makes every method a no-op, as an unsupported device does
   * @param depth    staging buffers in flight before frames start being skipped
   */
  constructor(rhi, { enabled = true, depth = 3 } = {}) {
    this.rhi = rhi;
    /** Whether the device can time passes at all. Fixed at device creation. */
    this.available = rhi.features?.has('timestamp-query') ?? false;
    /**
     * Whether it does. Settable at any time: the device asks for the feature
     * whenever the adapter has it, so switching on later needs nothing that
     * construction would have had to decide. Off, every method is a no-op and
     * no pass carries timestamp writes.
     */
    this.enabled = enabled;
    this.depth = depth;

    /**
     * Query slots currently allocated, in passes. Not an option and not a
     * budget: a frame's pass count is whatever the graph declared, and the
     * graph has no ceiling either. This follows it. Starts at zero, so the
     * first frame is untimed and sizes the second.
     */
    this.capacity = 0;
    this._wanted = 0;

    /** Most recent completed readback: `[{ name, ms }]`, in pass order. */
    this.results = [];
    /** Sum of the above. Not the frame time: GPU passes can overlap. */
    this.totalMs = 0;

    /**
     * Running mean per pass over every readback so far, same shape as results.
     *
     * This is the number to read, not a single frame. Timestamps arrive
     * quantised, so a pass below the quantum reads as 0 on most frames and as
     * one whole quantum on the rest; the endpoints land on the grid
     * independently, so the mean converges on the real duration even though no
     * individual frame can express it.
     */
    this.average = [];
    this.averageTotalMs = 0;
    /** Readbacks folded into the mean. A small count means a noisy one. */
    this.samples = 0;
    this._accumulator = new Map();

    this._ring = [];
    this._names = [];
    /** The slot resolve() claimed this frame, waiting for readback(). */
    this._pending = null;
  }

  /** True when this is actually timing: available AND enabled. */
  get supported() {
    return this.available && this.enabled;
  }

  /**
   * Allocate, or reallocate, for `passes` passes.
   *
   * Only ever called from begin(), with nothing in flight, because a query set
   * or staging buffer destroyed while a command buffer still references it is
   * a use-after-free the device reports several frames later.
   */
  _resize(passes) {
    const device = this.rhi.device;
    const bytes = passes * QUERIES_PER_PASS * 8;

    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    for (const entry of this._ring) entry.buffer.destroy();

    this.querySet = device.createQuerySet({
      label: 'pass-timing',
      type: 'timestamp',
      count: passes * QUERIES_PER_PASS,
    });

    // Resolve target and the ring that gets read. Separate because a buffer
    // with QUERY_RESOLVE cannot also be MAP_READ.
    this.resolveBuffer = createBuffer(this.rhi, {
      label: 'pass-timing-resolve',
      size: bytes,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    this._ring = [];
    for (let i = 0; i < this.depth; i++) {
      this._ring.push({
        buffer: createBuffer(this.rhi, {
          label: `pass-timing-read${i}`,
          size: bytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        inFlight: false,
        names: [],
      });
    }
    this.capacity = passes;
  }

  /** Start a frame. Called by the graph before it records anything. */
  begin() {
    if (!this.supported) return;
    // Last frame wanted more slots than exist. Now is the only safe moment to
    // widen: no pass has been recorded yet, so nothing references the old set.
    if (this._wanted > this.capacity && !this._ring.some((e) => e.inFlight)) {
      this._resize(this._wanted);
    }
    this._names.length = 0;
  }

  /**
   * The `timestampWrites` for the pass at `step`, or undefined.
   *
   * Undefined is the correct value to hand a pass descriptor when timing is
   * off, so callers never need to branch. A step past what is currently
   * allocated records the demand and goes untimed for this frame only; the
   * next begin() widens to fit. Nothing is silently dropped for good.
   */
  writesFor(step, name) {
    if (!this.supported) return undefined;
    if (step >= this.capacity) {
      this._wanted = Math.max(this._wanted, step + 1);
      return undefined;
    }
    this._names.push(name);
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: step * QUERIES_PER_PASS,
      endOfPassWriteIndex: step * QUERIES_PER_PASS + 1,
    };
  }

  /**
   * Record the copy that makes this frame's queries readable. Called after the
   * last pass and before the encoder is finished.
   *
   * This only CLAIMS a staging buffer; the map itself has to wait for the
   * submit, because a buffer with a map pending cannot be written by a command
   * buffer. Pair every resolve with a readback().
   */
  resolve(encoder) {
    if (!this.supported || this._names.length === 0) return;

    const slot = this._ring.find((entry) => !entry.inFlight);
    // Every buffer still mapping means the GPU is more than `depth` frames
    // behind. Skipping is the right answer: the alternative is to wait, and
    // waiting is the one thing a profiler must not make the frame do.
    if (!slot) return;

    const count = this._names.length * QUERIES_PER_PASS;
    encoder.resolveQuerySet(this.querySet, 0, count, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, slot.buffer, 0, count * 8);

    slot.inFlight = true;
    slot.names = this._names.slice();
    this._pending = slot;
  }

  /**
   * Start reading the buffer resolve() claimed. Must run AFTER the submit.
   *
   * Calling this before the submit is an error WebGPU reports rather than
   * tolerates -- "used in submit while pending map" -- because the command
   * buffer would be writing a buffer the map already owns.
   */
  readback() {
    const slot = this._pending;
    if (!slot) return;
    this._pending = null;

    slot.buffer.mapAsync(GPUMapMode.READ)
      .then(() => {
        this._read(slot);
        slot.buffer.unmap();
      })
      .catch(() => { /* device lost, or the buffer was destroyed mid-flight */ })
      .finally(() => { slot.inFlight = false; });
  }

  _read(slot) {
    const stamps = new BigInt64Array(slot.buffer.getMappedRange());
    const results = [];
    let total = 0;

    for (let i = 0; i < slot.names.length; i++) {
      // Unsigned nanoseconds, but read as signed: a span is small enough that
      // the sign bit is never in play, and BigInt64Array is what subtracts
      // cleanly. An end before its beginning means the query never landed.
      const ns = Number(stamps[i * QUERIES_PER_PASS + 1] - stamps[i * QUERIES_PER_PASS]);
      const ms = ns > 0 ? ns / NS_PER_MS : 0;
      results.push({ name: slot.names[i], ms });
      total += ms;

      const entry = this._accumulator.get(slot.names[i]) ?? { sum: 0, count: 0 };
      entry.sum += ms;
      entry.count++;
      this._accumulator.set(slot.names[i], entry);
    }

    this.results = results;
    this.totalMs = total;
    this.samples++;

    this.average = results.map(({ name }) => {
      const entry = this._accumulator.get(name);
      return { name, ms: entry.sum / entry.count };
    });
    this.averageTotalMs = this.average.reduce((sum, pass) => sum + pass.ms, 0);
  }

  /** Forget the accumulated means. Call after anything that changes the frame. */
  resetAverages() {
    this._accumulator.clear();
    this.average = [];
    this.averageTotalMs = 0;
    this.samples = 0;
  }

  /** Mean pass timings, slowest first. For printing, not for the frame loop. */
  slowest(limit = 10) {
    return this.average.slice().sort((a, b) => b.ms - a.ms).slice(0, limit);
  }

  destroy() {
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    for (const entry of this._ring) entry.buffer.destroy();
    this._ring = [];
    this.capacity = 0;
  }
}
