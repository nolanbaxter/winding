// The opaque scene, copied and filtered down a mip chain, for transmissive
// surfaces to see through (KHR_materials_transmission).
//
// A transmissive surface shows what is behind it, blurred by its roughness.
// Rendering the scene again from every such surface is out of reach, so the
// finished opaque picture stands in for "behind": each surface reads it where
// its view ray comes out, at the mip its roughness asks for. What that cannot
// show is one transmissive surface through another -- only opaque things and
// the sky are behind anything here.
//
// Made on the first frame that has a transmissive surface, and never before:
// a scene without one pays nothing, neither the memory nor the passes.

import { createTexture, defaultTextures, mipLevelCountFor, mipPipelineFor, clampSampler } from '../rhi/texture.js';
import { HDR_FORMAT } from './post.js';

export class OpaqueCopy {
  constructor(rhi) {
    this.rhi = rhi;
    this.texture = null;
    this.width = 0;
    this.height = 0;
    /** What the frame binds. A 1x1 stand-in until a frame needs the real one. */
    this.view = defaultTextures(rhi).white.createView();
    this._levelViews = [];
    this._levelExecutes = [];
    this._sourceView = null;
    this._sourceGroups = new WeakMap();
    // Bound once, like the renderer's pass callbacks: the scene's view is
    // only known when the graph runs the pass.
    this._graph = null;
    this._sceneColor = null;
    this._copyScene = (pass) => {
      this._sourceView = this._graph.viewOf(this._sceneColor);
      this._levelExecutes[0](pass);
    };
  }

  /**
   * Size the copy to the surface. True when the texture was replaced, which
   * leaves any bind group naming the old one stale.
   */
  ensure(width, height) {
    if (this.texture !== null && this.width === width && this.height === height) return false;
    this.texture?.destroy();
    const rhi = this.rhi;
    this.width = width;
    this.height = height;
    this.texture = createTexture(rhi, {
      label: 'opaque-copy',
      size: [width, height],
      format: HDR_FORMAT,
      mipLevelCount: mipLevelCountFor(width, height),
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.view = this.texture.createView();
    this._levelViews = [];
    for (let level = 0; level < this.texture.mipLevelCount; level++) {
      this._levelViews.push(this.texture.createView({ baseMipLevel: level, mipLevelCount: 1 }));
    }

    // Each level is the one above it, filtered: the mip pipeline's own box
    // filter, as every other mip chain in the engine. Level 0 samples the
    // scene at its own size, which is a copy.
    const { pipeline, layout } = mipPipelineFor(rhi, HDR_FORMAT);
    const sampler = clampSampler(rhi);
    const groupFor = (view) => rhi.device.createBindGroup({
      label: 'opaque-copy', layout, entries: [{ binding: 0, resource: view }, { binding: 1, resource: sampler }],
    });
    this._sourceGroups = new WeakMap();
    this._groupFor = groupFor;
    this._levelExecutes = this._levelViews.map((_, level) => {
      const group = level === 0 ? null : groupFor(this._levelViews[level - 1]);
      return (pass) => {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group ?? this._sourceGroup());
        pass.draw(3);
      };
    });
    return true;
  }

  /** The scene's colour target is the graph's, and may be a different view each frame. */
  _sourceGroup() {
    let group = this._sourceGroups.get(this._sourceView);
    if (!group) {
      group = this._groupFor(this._sourceView);
      this._sourceGroups.set(this._sourceView, group);
    }
    return group;
  }

  /**
   * Copy the scene as it stands and filter it down. Returns every level, for
   * the pass that reads them to name.
   */
  addPasses(graph, sceneColor) {
    const levels = this._levelViews.map((view, level) => graph.importTexture(`opaque-copy:${level}`, view));
    this._graph = graph;
    this._sceneColor = sceneColor;
    levels.forEach((resource, level) => {
      graph.addPass({
        name: `opaque-copy:${level}`,
        reads: [level === 0 ? sceneColor : levels[level - 1]],
        color: [{ resource, clear: { r: 0, g: 0, b: 0, a: 0 } }],
        execute: level === 0 ? this._copyScene : this._levelExecutes[level],
      });
    });
    return levels;
  }

  destroy() {
    this.texture?.destroy();
    this.texture = null;
  }
}
