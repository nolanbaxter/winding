// Types for winding-engine/bench.js. See docs/API.md#benchmark.

import type { Winding, Scene, Camera, Camera2D } from './winding.js';

export interface Timing {
  mean: number;
  median: number;
  p95: number;
  max: number;
}

/** Milliseconds; `share` is the fraction of the frame. Plain data, safe to JSON.stringify. */
export interface Report {
  frames: number;
  cpu: (Timing & { name: string; share: number })[];
  gpu: { name: string; mean: number; share: number }[] | null;
  wall: Timing | null;
}

export class Benchmark {
  constructor(engine: Winding);
  static format(report: Report): string;
  run(
    scene: Scene,
    camera: Camera | Camera2D,
    options?: { frames?: number; warmup?: number; update?: ((i: number) => void) | null },
  ): Promise<Report>;
  start(): void;
  stop(): void;
  report(): Report;
}
