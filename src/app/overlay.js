// Frame stats overlay.
//
// Part of Tier 3 because the numbers it shows are the ones that tell you
// whether the engine is behaving: if `pipelines` keeps climbing you are
// compiling shaders mid-frame, and if `visible` never drops below the total
// then culling is not actually wired in.

const STYLE = `
  position: fixed; left: 12px; bottom: 12px; z-index: 9999;
  font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: #8ba3c7; background: rgba(7,8,11,.72);
  padding: 6px 9px; border-radius: 6px; pointer-events: none; white-space: pre;
`;

export class StatsOverlay {
  constructor(engine, { interval = 0.5, parent = document.body } = {}) {
    this.engine = engine;
    this.interval = interval;
    this.element = document.createElement('div');
    this.element.style.cssText = STYLE;
    this.element.textContent = 'starting…';
    parent.appendChild(this.element);
    this._accum = 0;
  }

  /** Call once per rendered frame. */
  update(dt) {
    this._accum += dt;
    if (this._accum < this.interval) return;
    this._accum = 0;

    const { rhi, renderer, stats } = this.engine;
    this.element.textContent = [
      `${this.engine.fps.toFixed(0)} fps`,
      `${rhi.width}x${rhi.height}`,
      // No visible count: culling happens on the GPU and the answer lives only
      // in the indirect argument buffer. Reading it back would stall the frame.
      `${stats.renderables} objects`,
      `${stats.draws} indirect draws`,
      `recomposed ${stats.recomposed}`,
      `pipelines ${renderer.pipelines.created}`,
      `graph ${renderer.graph.stats.executed}/${renderer.graph.stats.passes} passes`,
      `cpu ${renderer.timing.total.toFixed(2)}ms ` +
        `(xform ${renderer.timing.transforms.toFixed(2)} ` +
        `upload ${renderer.timing.upload.toFixed(2)} ` +
        `graph ${renderer.timing.graph.toFixed(2)} ` +
        `encode ${renderer.timing.encode.toFixed(2)})`,
    ].join('   ');
  }

  destroy() {
    this.element.remove();
  }
}
