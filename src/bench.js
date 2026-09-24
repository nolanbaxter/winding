// Benchmarking.
//
// OPT-IN AND SEPARATE. Nothing in the engine imports this file, so a page that
// never asks for it never loads it. The renderer carries one field for it,
// `renderer.profiler`, which is null until a Benchmark attaches; every phase
// boundary in a frame is then `p?.mark(...)`, one null check that does not even
// evaluate its argument. Nothing is measured, stored or read back unless a
// benchmark is running -- including GPU pass timing, which is off by default
// and switched on only for the length of a run.
//
//   import { Benchmark } from 'winding-engine/bench.js';
//   const report = await new Benchmark(engine).run(scene, camera, { frames: 300 });
//   console.log(Benchmark.format(report));
//
// WHAT IT MEASURES, per frame:
//
//   cpu    every phase of Renderer.render, back to back. The phases PARTITION
//          the frame -- each mark closes the span since the previous one -- so
//          they sum to `frame (cpu)` and nothing goes unaccounted for.
//   gpu    every render-graph pass, from device timestamps. Means only: see
//          render/timing.js for why a single frame's GPU numbers mean nothing.
//   wall   submit to the GPU finishing. Compared with `frame (cpu)` it answers
//          the first question of any optimization: is this frame CPU-bound or
//          GPU-bound? Only run() measures it, because only run() waits.
//
// HOW TO READ IT. Median and p95, not the mean alone: a garbage-collection pause
// or a shader compile lands in one frame and drags a mean around, and p95 is
// where a stutter shows. And the same caution timing.js gives for the GPU holds
// for everything here: numbers from two runs a minute apart differ by more
// than most optimizations change them. Compare shares within a run, or measure
// before and after in the same page, alternating.

const now = () => (globalThis.performance?.now?.() ?? Date.now());

export class Benchmark {
  constructor(engine) {
    this.engine = engine;
    this.renderer = engine.renderer;
    /** Frames recorded since start(). */
    this.frames = 0;
    this._spans = new Map();   // phase name -> milliseconds, one per frame
    this._wall = [];
    this._running = false;
  }

  /**
   * Begin recording every frame the engine renders, however it is driven.
   * Starting again while running starts the recording over; it does not
   * forget what GPU timing was set to before the first start, which a second
   * start used to overwrite with the "on" the first one had set -- so stop()
   * restored "on".
   */
  start() {
    const attached = this.renderer.profiler;
    if (attached && attached !== this) {
      throw new Error('Benchmark: another benchmark is already attached to this renderer');
    }
    const gpu = this.renderer.gpuTiming;
    if (!this._running) {
      this._gpuWasEnabled = gpu.enabled;
      gpu.enabled = true;
      this.renderer.profiler = this;
      this._running = true;
    }
    this._clear();
    return this;
  }

  _clear() {
    this._spans.clear();
    this._wall.length = 0;
    this.frames = 0;
    this.renderer.gpuTiming.resetAverages();
  }

  /** Stop recording, put the renderer back exactly as it was, and report. */
  stop() {
    if (this._running) {
      this.renderer.profiler = null;
      this.renderer.gpuTiming.enabled = this._gpuWasEnabled;
      this._running = false;
    }
    return this.report();
  }

  /**
   * Render `frames` frames and report on them.
   *
   * Waits for the GPU after every frame, which is what makes `wall` mean
   * something and lets the GPU timings read back -- and which also means this
   * measures one frame at a time, not the throughput of a pipelined loop.
   * `warmup` frames run first, unrecorded: the first frames of a scene compile
   * pipelines and grow buffers, and that is a different question.
   *
   * `update(i)` runs before each frame, untimed -- animate the camera or the
   * scene there, so the benchmark sees work that changes.
   */
  async run(scene, camera, { frames = 300, warmup = 30, update = null } = {}) {
    const engine = this.engine;
    const queue = engine.rhi.device.queue;

    // Attached for the warmup too, so everything that only happens the first
    // time something is measured -- GPU timing's query set and staging
    // buffers are created on its first frames on -- happens in the warmup
    // rather than inside the recorded spans. The warmup is then forgotten.
    this.start();
    try {
      for (let i = 0; i < warmup; i++) {
        update?.(i - warmup);
        engine.renderFrame(scene, camera);
        await queue.onSubmittedWorkDone();
      }
      this._clear();

      for (let i = 0; i < frames; i++) {
        update?.(i);
        const t = now();
        engine.renderFrame(scene, camera);
        await queue.onSubmittedWorkDone();
        this._wall.push(now() - t);
      }
      // The GPU timings arrive through a buffer map that resolves after the
      // work does. One turn of the event loop lets the last of them land.
      await new Promise((resolve) => setTimeout(resolve, 0));
    } finally {
      this.stop();
    }
    return this.report();
  }

  // ---------------------------------------------- called by Renderer.render

  frameStart() {
    this._frameStart = this._last = now();
  }

  /** Close the span since the previous mark, under `name`. */
  mark(name) {
    const t = now();
    this._record(name, t - this._last);
    this._last = t;
  }

  /**
   * The frame ends at its last mark, not at a fresh clock read: anything
   * between the two would be in the total and in no phase, and the phases are
   * meant to account for all of it. The renderer's last mark is the submit.
   */
  frameEnd() {
    this._record('frame (cpu)', this._last - this._frameStart);
    this.frames++;
  }

  _record(name, ms) {
    let samples = this._spans.get(name);
    if (!samples) {
      samples = [];
      this._spans.set(name, samples);
    }
    samples.push(ms);
  }

  // ------------------------------------------------------------- reporting

  /**
   * Everything recorded so far, summarised. Plain data, safe to JSON.stringify
   * and paste somewhere.
   *
   *   { frames,
   *     cpu:  [{ name, mean, median, p95, max, share }]   in frame order
   *     gpu:  [{ name, mean, share }] | null              null: no timestamp-query
   *     wall: { mean, median, p95, max } | null }        null: not from run()
   *
   * `share` is the fraction of the frame total the phase or pass took.
   */
  report() {
    const frameCpu = summarize(this._spans.get('frame (cpu)') ?? []);
    const cpu = [...this._spans].map(([name, samples]) => {
      const stats = summarize(samples);
      return { name, ...stats, share: frameCpu.mean > 0 ? stats.mean / frameCpu.mean : 0 };
    });

    const timing = this.renderer.gpuTiming;
    const gpuTotal = timing.averageTotalMs;
    const gpu = timing.available
      ? timing.average.map(({ name, ms }) => ({ name, mean: ms, share: gpuTotal > 0 ? ms / gpuTotal : 0 }))
      : null;

    return {
      frames: this.frames,
      cpu,
      gpu,
      gpuSamples: timing.samples,
      wall: this._wall.length > 0 ? summarize(this._wall) : null,
    };
  }

  /** A report as a fixed-width table, slowest first, for a console or an issue. */
  static format(report) {
    const ms = (v) => v.toFixed(3).padStart(8);
    const pct = (v) => `${(v * 100).toFixed(1)}%`.padStart(7);
    const lines = [`${report.frames} frames`];

    if (report.wall) {
      lines.push(`wall     median ${ms(report.wall.median)}  p95 ${ms(report.wall.p95)}  (${boundBy(report)})`);
    }

    lines.push('', 'cpu phase                    median      p95      max   share');
    const phases = report.cpu.filter((row) => row.name !== 'frame (cpu)')
      .sort((a, b) => b.median - a.median);
    for (const row of phases) {
      lines.push(`  ${row.name.padEnd(24)} ${ms(row.median)} ${ms(row.p95)} ${ms(row.max)} ${pct(row.share)}`);
    }
    const total = report.cpu.find((row) => row.name === 'frame (cpu)');
    if (total) lines.push(`  ${'frame (cpu)'.padEnd(24)} ${ms(total.median)} ${ms(total.p95)} ${ms(total.max)}`);

    if (report.gpu) {
      lines.push('', `gpu pass (mean of ${report.gpuSamples})       mean   share`);
      for (const row of [...report.gpu].sort((a, b) => b.mean - a.mean)) {
        lines.push(`  ${row.name.padEnd(24)} ${ms(row.mean)} ${pct(row.share)}`);
      }
    } else {
      lines.push('', 'gpu: this device has no timestamp-query');
    }
    return lines.join('\n');
  }
}

/**
 * What a frame spent its wall time on: the CPU building it, the GPU running
 * it, or neither -- waiting for the display to take it.
 *
 * run() waits for each frame to finish, so wall = CPU + GPU + whatever is
 * left, and the leftover is presentation pacing: a canvas cannot show frames
 * faster than the display refreshes. Whichever of the three is largest is the
 * answer, which needs no threshold. It matters because the first run of this
 * came back at 15.8ms wall for 0.9ms of CPU and 1.2ms of GPU: a 60Hz display
 * pacing a frame that had plenty of room, which a CPU-or-GPU split would have
 * called GPU-bound.
 *
 * GPU is the sum of the pass means, which can overcount where passes overlap;
 * without timestamps it is unknown and the leftover cannot be split.
 */
export function boundBy(report) {
  const wall = report.wall.median;
  const cpu = report.cpu.find((row) => row.name === 'frame (cpu)')?.median ?? 0;
  if (!report.gpu) {
    return cpu >= wall - cpu
      ? `CPU-bound: cpu ${cpu.toFixed(2)}ms`
      : `waiting on the GPU or the display: cpu ${cpu.toFixed(2)}ms`;
  }
  const gpu = report.gpu.reduce((sum, row) => sum + row.mean, 0);
  const paced = wall - cpu - gpu;
  const split = `cpu ${cpu.toFixed(2)}ms, gpu ${gpu.toFixed(2)}ms`;
  if (cpu >= gpu && cpu >= paced) return `CPU-bound: ${split}`;
  if (gpu >= paced) return `GPU-bound: ${split}`;
  return `display-paced: ${split}, the rest waits for the display`;
}

/** Mean, median, 95th percentile and max of a list of milliseconds. */
export function summarize(samples) {
  if (samples.length === 0) return { mean: 0, median: 0, p95: 0, max: 0 };
  const sorted = Float64Array.from(samples).sort();
  const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  let sum = 0;
  for (const v of sorted) sum += v;
  return { mean: sum / sorted.length, median: at(0.5), p95: at(0.95), max: sorted[sorted.length - 1] };
}
