// Text in the scene: fonts as signed distance fields, laid out into glyph
// quads that the sprite pass draws.
//
// A glyph is rasterised once, by the browser, in whatever CSS font was asked
// for, and turned into a distance field: each texel holds how far it is from
// the glyph's edge. Drawn at any size, the edge is where that distance
// crosses zero, found per pixel -- so text stays sharp magnified, where a
// plain bitmap would blur, and one atlas serves every size.
//
// The field reaches SPREAD texels either side of the edge. A pixel's
// smoothing needs the field correct over half its footprint, and a pixel of
// text drawn at 1/k of its raster size covers k texels: so SPREAD texels hold
// down to 1/(2 SPREAD) of the raster size. The atlas has no mips, whose
// averaging would bleed one glyph into the next; text drawn smaller than that
// shimmers. Rasterise near the size it is seen at.

import { createTexture } from '../rhi/texture.js';

/** Texels the distance field reaches past a glyph's edge; see the header. */
export const SPREAD = 4;

// ------------------------------------------------------------- pure parts

/** Stands for no site: far past any real squared distance, and finite, so it subtracts. */
const FAR = 1e20;

/**
 * The exact squared Euclidean distance transform of one row or column
 * (Felzenszwalb and Huttenlocher): f holds 0 at a site and FAR elsewhere,
 * and d gets each entry's squared distance to the nearest site -- the lower
 * envelope of the parabolas rooted at each, in linear time.
 */
function edt1d(f, n, v, z, d) {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  const meet = (q, r) => ((f[q] + q * q) - (f[r] + r * r)) / (2 * q - 2 * r);
  for (let q = 1; q < n; q++) {
    let s = meet(q, v[k]);
    while (s <= z[k]) s = meet(q, v[--k]);
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

/** Squared distance from every texel to the nearest texel where `site` is true. */
function edt(site, width, height) {
  const grid = new Float64Array(width * height);
  for (let i = 0; i < grid.length; i++) grid[i] = site(i) ? 0 : FAR;
  const n = Math.max(width, height);
  const f = new Float64Array(n);
  const d = new Float64Array(n);
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) f[y] = grid[y * width + x];
    edt1d(f, height, v, z, d);
    for (let y = 0; y < height; y++) grid[y * width + x] = d[y];
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) f[x] = grid[y * width + x];
    edt1d(f, width, v, z, d);
    for (let x = 0; x < width; x++) grid[y * width + x] = d[x];
  }
  return grid;
}

/**
 * A glyph's coverage (0..1 a texel, as the browser rasterised it) as a
 * distance field, 0.5 at the edge and SPREAD texels to 0 or 1. Exact
 * distances between texel centres, and, at the edge texels the rasteriser
 * shaded part-way, the sub-texel position its coverage implies -- without
 * which the edge snaps to the texel grid and steps when magnified.
 */
export function distanceField(coverage, width, height, spread = SPREAD) {
  const inside = edt((i) => coverage[i] >= 0.5, width, height);
  const outside = edt((i) => coverage[i] < 0.5, width, height);
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) {
    const c = coverage[i];
    // Positive outside the glyph, in texels, to the nearest edge between centres.
    let d = c >= 0.5 ? -(Math.sqrt(outside[i]) - 0.5) : Math.sqrt(inside[i]) - 0.5;
    if (c > 0 && c < 1) d = 0.5 - c;
    out[i] = Math.round(Math.min(Math.max(0.5 - d / (2 * spread), 0), 1) * 255);
  }
  return out;
}

/**
 * Lay a string out in em units: each glyph's box relative to the block, with
 * the block's anchor point at the origin and y up. `metrics` gives each
 * character's advance and ink box, and the font's ascent and descent; see
 * Font.metrics.
 *   align       'left', 'center' or 'right', each line within the block
 *   lineHeight  in ems; the font's own ascent plus descent by default
 *   anchor      [x, y] in the block, 0..1: the point placed at the node
 * Returns [{ char, x, y, width, height }] -- the ink box's lower left, and
 * size -- for characters with ink.
 */
export function layoutText(text, metrics, { align = 'left', lineHeight, anchor = [0.5, 0.5] } = {}) {
  const lines = String(text).split('\n');
  const step = lineHeight ?? metrics.ascent + metrics.descent;
  const widths = lines.map((line) => [...line].reduce((w, ch) => w + (metrics.glyphs.get(ch)?.advance ?? 0), 0));
  const blockWidth = Math.max(0, ...widths);
  const blockHeight = metrics.ascent + metrics.descent + step * (lines.length - 1);
  const shift = { left: 0, center: 0.5, right: 1 }[align];
  if (shift === undefined) throw new Error(`text: align is 'left', 'center' or 'right', got ${align}`);
  const boxes = [];
  lines.forEach((line, row) => {
    let pen = (blockWidth - widths[row]) * shift;
    // The top of the block is y = 0 before the anchor moves it; baselines run down.
    const baseline = -metrics.ascent - row * step;
    for (const ch of line) {
      const g = metrics.glyphs.get(ch);
      if (g === undefined) continue;
      if (g.width > 0 && g.height > 0) {
        boxes.push({ char: ch, x: pen + g.left, y: baseline - g.descent, width: g.width, height: g.height });
      }
      pen += g.advance;
    }
  });
  const ox = anchor[0] * blockWidth;
  const oy = -blockHeight + anchor[1] * blockHeight;
  for (const box of boxes) {
    box.x -= ox;
    box.y -= oy;
  }
  return boxes;
}

// ----------------------------------------------------------------- fonts

/**
 * A font's glyphs, rasterised on demand into one atlas. Made by
 * engine.loadFont. Glyphs are added the first time text uses them; the atlas
 * doubles when it fills, and `texture` is then a new object, which the sprite
 * pass picks up on the next frame.
 */
export class Font {
  constructor(rhi, css, size) {
    this.rhi = rhi;
    this.css = css;
    /** The raster size, in pixels a em: the CSS font's own size. */
    this.size = size;
    this._canvas = new OffscreenCanvas(1, 1);
    this._context = this._canvas.getContext('2d', { willReadFrequently: true });
    this._context.font = css;
    const probe = this._context.measureText('Hg');
    /** In ems. glyphs: char -> { advance, left, width, height, descent } in ems, and its atlas rect. */
    this.metrics = {
      ascent: probe.fontBoundingBoxAscent / size,
      descent: probe.fontBoundingBoxDescent / size,
      glyphs: new Map(),
    };
    this._atlasSize = 512;
    this._shelf = { x: 0, y: 0, height: 0 };
    this._make(this._atlasSize);
  }

  _make(atlasSize, from = null) {
    const texture = createTexture(this.rhi, {
      label: `font:${this.css}`, size: [atlasSize, atlasSize], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });
    if (from) {
      const encoder = this.rhi.device.createCommandEncoder({ label: 'font-grow' });
      encoder.copyTextureToTexture({ texture: from }, { texture }, [from.width, from.height]);
      this.rhi.queue.submit([encoder.finish()]);
      from.destroy();
      // Every rect is a fraction of the atlas, which has just doubled.
      for (const glyph of this.metrics.glyphs.values()) {
        if (glyph.rect) glyph.rect = glyph.rect.map((v) => v / 2);
      }
    }
    // What a sprite needs of a texture.
    this.texture = { texture, view: texture.createView(), width: atlasSize, height: atlasSize, sdf: true };
  }

  /** Rasterise any characters of `text` not yet in the atlas. */
  ensure(text) {
    for (const ch of new Set(text)) {
      if (ch === '\n' || this.metrics.glyphs.has(ch)) continue;
      this._add(ch);
    }
  }

  _add(ch) {
    const size = this.size;
    const context = this._context;
    const m = context.measureText(ch);
    const left = Math.floor(-m.actualBoundingBoxLeft);
    const right = Math.ceil(m.actualBoundingBoxRight);
    const up = Math.ceil(m.actualBoundingBoxAscent);
    const down = Math.ceil(m.actualBoundingBoxDescent);
    const inkW = right - left;
    const inkH = up + down;
    const glyph = { advance: m.width / size, left: 0, width: 0, height: 0, descent: 0, rect: null };
    this.metrics.glyphs.set(ch, glyph);
    if (inkW <= 0 || inkH <= 0) return;   // a space: advance only

    const w = inkW + 2 * SPREAD;
    const h = inkH + 2 * SPREAD;
    this._canvas.width = w;
    this._canvas.height = h;
    context.font = this.css;
    context.fillStyle = '#fff';
    context.textBaseline = 'alphabetic';
    context.clearRect(0, 0, w, h);
    context.fillText(ch, SPREAD - left, SPREAD + up);
    const pixels = context.getImageData(0, 0, w, h).data;
    const coverage = new Float32Array(w * h);
    for (let i = 0; i < coverage.length; i++) coverage[i] = pixels[i * 4 + 3] / 255;
    const field = distanceField(coverage, w, h);

    // Shelf packing: along a row, then a new row under the tallest so far.
    let shelf = this._shelf;
    if (shelf.x + w > this._atlasSize) { shelf.y += shelf.height; shelf.x = 0; shelf.height = 0; }
    while (shelf.y + h > this._atlasSize || w > this._atlasSize) {
      this._atlasSize *= 2;
      this._make(this._atlasSize, this.texture.texture);
      shelf = this._shelf;
    }
    const rgba = new Uint8Array(w * h * 4).fill(255);
    for (let i = 0; i < field.length; i++) rgba[i * 4 + 3] = field[i];
    this.rhi.queue.writeTexture({ texture: this.texture.texture, origin: [shelf.x, shelf.y] }, rgba, { bytesPerRow: w * 4 }, [w, h]);

    // In ems, the ink box plus its SPREAD, which the quad has to include.
    const s = this._atlasSize;
    glyph.left = (left - SPREAD) / size;
    glyph.width = w / size;
    glyph.height = h / size;
    glyph.descent = (down + SPREAD) / size;
    glyph.rect = [shelf.x / s, shelf.y / s, (shelf.x + w) / s, (shelf.y + h) / s];
    shelf.x += w;
    shelf.height = Math.max(shelf.height, h);
  }

  destroy() {
    this.texture.texture.destroy();
  }
}
