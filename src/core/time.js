// Fixed-timestep clock.
//
// Simulation runs at a fixed rate regardless of framerate; rendering
// interpolates between the last two simulated states. This is why a game
// stays smooth on a 144 Hz display with a 60 Hz sim, and why physics behaves
// identically on every machine.
//
//   clock.begin(performance.now() / 1000);
//   while (clock.step()) simulate(clock.fixedDt);
//   render(clock.alpha);        // alpha in [0, 1): how far between states
//
// Writing `position += velocity * realDelta` in the render loop instead makes
// simulation framerate-dependent -- the same input produces different results
// on different hardware. It is the most common structural bug in hobby engines
// and it is exactly the kind of arbitrary behavior this engine refuses.

/**
 * Longest real delta the accumulator will absorb, in seconds.
 *
 * Without this clamp, a stall (tab backgrounded, a long GC, a breakpoint)
 * queues up hundreds of sim steps. Running them takes longer than the frame,
 * which queues up more -- the "death spiral". Clamping drops simulated time on
 * the floor instead, which is the correct tradeoff: a stalled frame is already
 * visibly broken, and slow motion is better than a hang.
 */
const MAX_FRAME_TIME = 0.25;

export class Clock {
  /** @param {number} fixedDt simulation step in seconds; 1/60 by default */
  constructor(fixedDt = 1 / 60) {
    this.fixedDt = fixedDt;
    this.accumulator = 0;
    /** Total SIMULATED seconds. Not wall-clock -- this is the one shaders want. */
    this.elapsed = 0;
    this.frame = 0;
    /** Real seconds since the previous begin(), after clamping. */
    this.realDelta = 0;
    this._last = -1;
  }

  /** Call once per rendered frame, before stepping. `now` is in seconds. */
  begin(now) {
    if (this._last < 0) this._last = now;   // first frame contributes no time
    let delta = now - this._last;
    this._last = now;

    // A clock that runs backwards (manual adjustment, some timer sources)
    // would otherwise drain the accumulator.
    if (delta < 0) delta = 0;
    else if (delta > MAX_FRAME_TIME) delta = MAX_FRAME_TIME;

    this.realDelta = delta;
    this.accumulator += delta;
    this.frame++;
  }

  /** Drive with `while (clock.step())`. Returns false when the sim has caught up. */
  step() {
    if (this.accumulator < this.fixedDt) return false;
    this.accumulator -= this.fixedDt;
    this.elapsed += this.fixedDt;
    return true;
  }

  /** Fraction of a step left over: how far to interpolate toward the newest state. */
  get alpha() {
    return this.accumulator / this.fixedDt;
  }
}
