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

/**
 * Texels the distance field reaches past a glyph's edge; see the header. An
 * outline (addText's stroke) is drawn in that reach, so it is at most
 * SPREAD - 1 texels of the raster: 7/64 of an em for a 64px font.
 */
export const SPREAD = 8;
/**
 * The largest a font's atlas grows before it starts again, empty. Glyphs
 * were kept for good, and a script laid out a word at a time (see SHAPED)
 * adds one for every new word: changing text doubled the atlas until it
 * passed the device's limit and threw. 4096 is 64 MB, ~16,000 glyphs at 64 px.
 */
const ATLAS_CAP = 4096;
/** Kerning pairs remembered before the memo starts again. */
const KERN_CAP = 65536;
/** A glyph missing from the atlas: drawn as nothing. */
const NO_RECT = [0, 0, 0, 0];

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

// What a reader sees as one character: a letter and its accents, a flag, an
// emoji and its modifiers. Code points where the browser has no segmenter.
const segmenter = typeof Intl?.Segmenter === 'function' ? new Intl.Segmenter(undefined, { granularity: 'grapheme' }) : null;

// Below U+0300 every code point is a character of its own: no combining
// marks, emoji or CJK, so no segmenting to do -- most text, and much faster.
const SIMPLE = /^[\u0000-\u02ff]*$/;

/** A string's characters as a reader counts them: see segmenter above. */
export function graphemes(text) {
  if (segmenter === null || SIMPLE.test(text)) return [...text];
  const out = [];
  for (const { segment } of segmenter.segment(text)) out.push(segment);
  return out;
}

// Chinese and Japanese are written without spaces, and a line may break
// between any two of their characters -- but not before closing punctuation
// or after opening, as CSS's line breaking holds (kinsoku).
const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\u3000-\u303f\uff00-\uffef]/u;
const NO_BREAK_BEFORE = new Set([...'、。，．：；？！）」』】〕〉》〗〙〛ーぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ々ゝゞヽヾ…‥・']);
const NO_BREAK_AFTER = new Set([...'（「『【〔〈《〖〘〚']);

// Scripts a character at a time cannot draw: Arabic's letters join and change
// shape with their neighbours, the Indic scripts' combine and reorder, Hebrew
// and Arabic run right to left. A word in one is drawn whole, as the browser
// shapes it when it draws text -- one glyph in the atlas, so no shaping
// engine is needed. Ligatures in Latin text are not formed.
const SHAPED = new RegExp(`[${['Arabic', 'Hebrew', 'Syriac', 'Thaana', 'Nko', 'Devanagari', 'Bengali', 'Gurmukhi', 'Gujarati',
  'Oriya', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Sinhala', 'Thai', 'Lao', 'Tibetan', 'Myanmar', 'Khmer', 'Mongolian']
  .map((script) => `\\p{Script=${script}}`).join('')}]`, 'u');
const RTL = /[\p{Script=Arabic}\p{Script=Hebrew}\p{Script=Syriac}\p{Script=Thaana}\p{Script=Nko}]/u;
// Thai, Lao, Khmer and Myanmar put no spaces between words: where a word
// ends, and a line may break, is the browser's word segmenter's to say.
const words = typeof Intl?.Segmenter === 'function' ? new Intl.Segmenter(undefined, { granularity: 'word' }) : null;

/**
 * A word cut where a line may break inside it: between CJK characters, and
 * between the words of a shaped script. Each piece a list of glyphs: graphemes,
 * or for a shaped script the word whole.
 */
function pieces(word) {
  if (!SIMPLE.test(word) && SHAPED.test(word)) {
    if (words === null) return [[word]];
    return [...words.segment(word)].map(({ segment }) => [segment]);
  }
  const chars = graphemes(word);
  if (SIMPLE.test(word) || !chars.some((ch) => CJK.test(ch))) return [chars];
  const out = [[chars[0]]];
  for (let i = 1; i < chars.length; i++) {
    const before = chars[i - 1], ch = chars[i];
    const breaks = (CJK.test(before) || CJK.test(ch)) && !NO_BREAK_BEFORE.has(ch) && !NO_BREAK_AFTER.has(before);
    if (breaks) out.push([ch]); else out[out.length - 1].push(ch);
  }
  return out;
}

/**
 * Lay a string out in em units: each glyph's box relative to the block, with
 * the block's anchor point at the origin and y up. `metrics` gives each
 * character's advance and ink box, the font's ascent and descent, and
 * `kern(a, b)`, how much closer b sits after a; see Font.metrics.
 *   align       'left', 'center' or 'right', each line within the block
 *   lineHeight  in ems; the font's own ascent plus descent by default
 *   anchor      [x, y] in the block, 0..1: the point placed at the node
 *   width       in ems: lines wrap between words to fit it, as CSS wraps
 *               text in a box, and between Chinese and Japanese characters,
 *               and the block is that wide, so align and anchor work within
 *               it. A word wider than it overflows.
 * Characters are graphemes -- what a reader counts as one: a letter and its
 * accents, an emoji and its modifiers -- each one glyph.
 * Returns { boxes, block }: boxes as [{ char, x, y, width, height }] -- the
 * ink box's lower left, and size -- for characters with ink, and block as
 * [left, bottom, right, top], the whole block.
 */
export function layoutText(text, metrics, { align = 'left', lineHeight, anchor = [0.5, 0.5], width = Infinity } = {}) {
  const advance = (ch) => metrics.glyphs.get(ch)?.advance ?? 0;
  const kern = (a, b) => (a === null || metrics.kern === undefined ? 0 : metrics.kern(a, b));
  /** Characters' width, kerned: how far the pen moves over them. */
  const measure = (chars) => {
    let w = 0;
    let before = null;
    for (const ch of chars) {
      if (!metrics.glyphs.has(ch)) continue;
      w += kern(before, ch) + advance(ch);
      before = ch;
    }
    return w;
  };
  const space = advance(' ');
  // Lines as lists of characters. Each line's width is kept as it grows, a
  // piece at a time -- measuring the whole line again for every word made a
  // long unwrapped line quadratic -- and measured exactly, kerning across
  // the joins, once it is done.
  const lines = [];
  for (const paragraph of String(text).split('\n')) {
    let line = null;
    let wide = 0;
    for (const word of paragraph.split(' ')) {
      pieces(word).forEach((piece, k) => {
        // A word's first piece follows a space; the rest join it directly.
        const gap = k === 0 ? space : 0;
        const w = measure(piece);
        if (line === null) {
          line = [...piece];
          wide = w;
        } else if (wide + gap + w <= width) {
          if (k === 0) line.push(' ');
          line.push(...piece);
          wide += gap + w;
        } else {
          lines.push(line);
          line = [...piece];
          wide = w;
        }
      });
    }
    lines.push(line ?? []);
  }
  // A line that starts, at its first letter, in a right-to-left script runs
  // right to left: its words in reverse, each word still drawn as it reads --
  // a shaped one by the browser, any other left to right.
  for (let i = 0; i < lines.length; i++) {
    const first = lines[i].find((ch) => /\p{L}/u.test(ch));
    if (first !== undefined && RTL.test(first)) lines[i] = rightToLeft(lines[i]);
  }
  const widths = lines.map(measure);
  const step = lineHeight ?? metrics.ascent + metrics.descent;
  const blockWidth = Number.isFinite(width) ? width : Math.max(0, ...widths);
  const blockHeight = metrics.ascent + metrics.descent + step * (lines.length - 1);
  const shift = { left: 0, center: 0.5, right: 1 }[align];
  if (shift === undefined) throw new Error(`text: align is 'left', 'center' or 'right', got ${align}`);
  const boxes = [];
  lines.forEach((line, row) => {
    let pen = (blockWidth - widths[row]) * shift;
    // The top of the block is y = 0 before the anchor moves it; baselines run down.
    const baseline = -metrics.ascent - row * step;
    let before = null;
    for (const ch of line) {
      const g = metrics.glyphs.get(ch);
      if (g === undefined) continue;
      pen += kern(before, ch);
      before = ch;
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
  return { boxes, block: [-ox, -blockHeight - oy, blockWidth - ox, -oy] };
}

/** A line's glyphs with its words, the runs between spaces, in reverse order. */
function rightToLeft(line) {
  const runs = [[]];
  for (const ch of line) {
    if (ch === ' ') runs.push([]);
    else runs[runs.length - 1].push(ch);
  }
  return runs.reverse().flatMap((run, k) => (k === 0 ? run : [' ', ...run]));
}

/** Every glyph layoutText could ask for in `text`: what a font has to rasterise. */
export function glyphsOf(text) {
  const out = new Set();
  for (const paragraph of String(text).split('\n')) {
    if (paragraph.includes(' ')) out.add(' ');
    for (const word of paragraph.split(' ')) for (const piece of pieces(word)) for (const glyph of piece) out.add(glyph);
  }
  return out;
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
    const pairs = new Map();
    this.metrics = {
      ascent: probe.fontBoundingBoxAscent / size,
      descent: probe.fontBoundingBoxDescent / size,
      glyphs: new Map(),
      /**
       * How much further along b sits after a than its advance alone puts it,
       * in ems: negative where the font pulls a pair together, as "AV". The
       * browser's own kerning, read off a measurement of the pair, once.
       */
      kern: (a, b) => {
        const key = `${a}\u0000${b}`;
        let k = pairs.get(key);
        if (k === undefined) {
          const context = this._context;
          context.font = this.css;
          k = (context.measureText(a + b).width - context.measureText(a).width - context.measureText(b).width) / size;
          // Below what a pixel would show at any sane size: not a kern, rounding.
          if (Math.abs(k) < 1e-3) k = 0;
          if (pairs.size >= KERN_CAP) pairs.clear();
          pairs.set(key, k);
        }
        return k;
      },
    };
    this._atlasSize = 512;
    this._shelf = { x: 0, y: 0, height: 0 };
    /** Bumped when the atlas starts again: text laid out before has its glyphs to add back (ensureBoxes). */
    this.generation = 0;
    /** ATLAS_CAP, or less where a test wants to see the atlas start again. */
    this.atlasCap = ATLAS_CAP;
    this._make(this._atlasSize);
  }

  /**
   * A text's glyphs back in the atlas, if it started again since this text
   * last looked: what the sprite passes call before packing, so the atlas
   * cannot change in the middle of a pack. `record` is a scene's text record.
   */
  ensureBoxes(record) {
    if (record.fontGeneration === this.generation) return;
    for (const box of record.boxes) if (box.char !== undefined && !this.metrics.glyphs.has(box.char)) this._add(box.char);
    record.fontGeneration = this.generation;
  }

  /** Where a glyph is in the atlas, as a fraction of it; nothing, if it is not there. */
  rectOf(ch) {
    return this.metrics.glyphs.get(ch)?.rect ?? NO_RECT;
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

  /** Rasterise any glyphs of `text` not yet in the atlas: graphemes, and shaped words whole (see SHAPED). */
  ensure(text) {
    for (const glyph of glyphsOf(text)) {
      if (!this.metrics.glyphs.has(glyph)) this._add(glyph);
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
      // Full at the cap: start again, empty, at the same size -- a new texture,
      // which every pass notices -- keeping only this glyph. Text laid out
      // before gets its glyphs back as it is next drawn (ensureBoxes).
      if (this._atlasSize * 2 > Math.min(this.atlasCap, this.rhi.limits?.maxTextureDimension2D ?? this.atlasCap)
        && (shelf.x > 0 || shelf.y > 0)) {
        this.texture.texture.destroy();
        this.metrics.glyphs.clear();
        this.metrics.glyphs.set(ch, glyph);
        this._shelf = shelf = { x: 0, y: 0, height: 0 };
        this.generation++;
        this._make(this._atlasSize);
        continue;
      }
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
