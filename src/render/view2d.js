// The 2D view: what a Camera2D sees (scene/camera2d.js).
//
// Sprites, text, tilemaps, shapes and paths in painter's order -- by layer,
// then in the order they were added -- with no depth and none of the 3D
// passes, lit only if asked (see lit() below). Drawn
// straight onto the canvas through its plain view, not its sRGB one, with
// every colour encoded to sRGB in the shader first: so blending happens
// between sRGB values, as the browser composites a page and an image editor
// its layers, and an opaque colour lands on screen exactly as authored.
//
// RETAINED, NOT REBUILT. Every sprite and glyph has a slot in one storage
// buffer, laid out in draw order. The list is rebuilt only when that order
// can change (scene.spriteOrder); otherwise a frame rewrites the slots of
// what moved or changed and uploads that one range. The camera is a uniform,
// so panning and zooming touch no slot at all.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer, storageCapacity } from '../rhi/buffer.js';
import { clampSampler, pixelatedSampler, spriteSampler, generateMipmaps } from '../rhi/texture.js';
import { grownCapacity } from '../core/grow.js';
import { handleIndex } from '../core/handle.js';
import { spriteRect, frame2D, shapeRadius, repeats, BLENDS } from '../scene/scene.js';
import { BLEND_STATE } from './sprites.js';

/** position, angle, flags; size, pivot; rect; colour; per kind; per kind; a glyph's outline colour. */
export const SPRITE2D_FLOATS = 28;
const SPRITE2D_BYTES = SPRITE2D_FLOATS * 4;
/** Slots between two changed spans that cost less to upload than a second call: 512 bytes' worth. */
const MERGE_GAP = Math.floor(512 / SPRITE2D_BYTES);
const FLAG_SDF = 1;
const FLAG_CUTOUT = 2;
const FLAG_TILEMAP = 4;
const FLAG_SHAPE = 8;
const FLAG_ELLIPSE = 16;
const FLAG_PATH = 32;
const FLAG_CLOSED = 64;
const FLAG_LIT = 128;
const FLAG_CHUNK = 256;
/**
 * Segments in each piece of a long open path. Each piece is its own quad, as
 * tight as its segments, and tests only its own and its neighbours': a line
 * across the screen covered the screen with one quad and tested every
 * segment at every pixel of it.
 */
export const PATH_CHUNK = 16;

const SHADER = /* wgsl */ `
struct View {
  view     : mat4x4<f32>,   // world to canvas pixels
  viewport : vec4<f32>,     // x, y = size in pixels; z = 1 to snap to whole pixels; w = lights
  ambient  : vec4<f32>,     // what lights a lit thing where no light reaches, linear
};
struct Sprite {
  a     : vec4<f32>,   // x, y, angle, flags
  b     : vec4<f32>,   // width, height, pivot x, pivot y
  rect  : vec4<f32>,   // the image's rect; a path's bounds, in its own units
  color : vec4<f32>,
  // A sprite's cutoff; a tilemap's tile width, height and tileset columns; a
  // shape's half width, half height, corner radius and outline width; a
  // path's first point, point count, line width and the node's scale.
  e     : vec4<f32>,
  f     : vec4<f32>,   // a shape's or path's outline colour; a tilemap's margin and spacing; an image's rect
  g     : vec4<f32>,   // a glyph's outline colour
};

@group(0) @binding(0) var<uniform> view : View;
@group(0) @binding(1) var<storage, read> sprites : array<Sprite>;
// Every path's points, one after another (scene.addPath).
@group(0) @binding(2) var<storage, read> points : array<vec2<f32>>;
// The scene's point lights, packed as the scene packs them: 4 vec4s each.
@group(0) @binding(3) var<storage, read> lights : array<vec4<f32>>;
@group(1) @binding(0) var image : texture_2d<f32>;
@group(1) @binding(1) var imageSampler : sampler;
// A tilemap's ids, one texel a tile (scene.addTilemap).
@group(1) @binding(2) var tiles : texture_2d<u32>;

// How it meets what is under it (render/sprites.js, BLEND_STATE): 0 over it,
// 1 adding light, 3 multiplying, 4 screening.
override BLEND : u32 = 0u;

// A colour, straight, as its blend state takes it: premultiplied but for plain
// alpha, and adding no coverage when it adds light.
fn blended(c : vec4<f32>) -> vec4<f32> {
  if (BLEND == 1u) { return vec4<f32>(c.rgb * c.a, 0.0); }
  if (BLEND >= 3u) { return vec4<f32>(c.rgb * c.a, c.a); }
  return c;
}

struct Out {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) color : vec4<f32>,
  @location(2) @interpolate(flat) flags : u32,
  @location(3) @interpolate(flat) extra : vec4<f32>,
  @location(4) @interpolate(flat) f : vec4<f32>,   // as the slot's f
  @location(5) world : vec2<f32>,
  // Where in its quad, 0..1, and the quad's size in screen pixels: a sprite's
  // distance to its own edge, for smoothing it.
  @location(6) edge : vec2<f32>,
  @location(7) @interpolate(flat) span : vec2<f32>,
  @location(8) @interpolate(flat) g : vec4<f32>,   // as the slot's g
};

@vertex
fn vs(@builtin(vertex_index) index : u32, @builtin(instance_index) slot : u32) -> Out {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
  );
  let s = sprites[slot];
  let shape = (u32(s.a.w) & ${FLAG_SHAPE}u) != 0u;
  let path = (u32(s.a.w) & ${FLAG_PATH}u) != 0u;
  // A plain sprite: not a glyph (its edge is in its distance field), a
  // tilemap, or a cutout (hard-edged on purpose).
  let sprite = (u32(s.a.w) & ${FLAG_SDF | FLAG_TILEMAP | FLAG_CUTOUT | FLAG_SHAPE | FLAG_PATH}u) == 0u;
  let span = abs(s.b.xy) * length(view.view[0].xy);
  var corner = corners[index];
  // The quad reaches a screen pixel past its edge, for the edge's smoothing.
  if (shape || path || sprite) { corner += (corner * 2.0 - 1.0) / max(span, vec2<f32>(1e-5)); }
  // y points down, so this turns clockwise on screen, as CSS rotate() does.
  let c = cos(s.a.z);
  let n = sin(s.a.z);
  let local = (corner - s.b.zw) * s.b.xy;
  let turned = vec2<f32>(c * local.x - n * local.y, n * local.x + c * local.y);
  // From the quad's first corner, which snapping puts on a whole pixel: so a
  // sprite a whole number of pixels across has every edge on one, and every
  // pixel samples the middle of a texel, whatever its pivot.
  let first = -s.b.zw * s.b.xy;
  let along = corner * s.b.xy;
  var origin = (view.view * vec4<f32>(s.a.xy + vec2<f32>(c * first.x - n * first.y, n * first.x + c * first.y), 0.0, 1.0)).xy;
  if (view.viewport.z > 0.5) { origin = round(origin); }
  let screen = origin + (view.view * vec4<f32>(c * along.x - n * along.y, n * along.x + c * along.y, 0.0, 0.0)).xy;
  var out : Out;
  out.clip = vec4<f32>(screen.x / view.viewport.x * 2.0 - 1.0, 1.0 - screen.y / view.viewport.y * 2.0, 0.0, 1.0);
  // A shape's uv is where it is from its centre, in its own units.
  out.uv = select(mix(s.rect.xy, s.rect.zw, corner), (corner - 0.5) * abs(s.b.xy), shape);
  out.color = s.color;
  out.flags = u32(s.a.w);
  out.extra = s.e;
  out.f = s.f;
  out.world = s.a.xy + turned;
  out.edge = corner;
  out.g = s.g;
  out.span = select(vec2<f32>(0.0), span, sprite);
  return out;
}

fn encode(c : vec3<f32>) -> vec3<f32> {
  let v = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  return select(v * 12.92, 1.055 * pow(v, vec3<f32>(1.0 / 2.4)) - 0.055, v > vec3<f32>(0.0031308));
}

fn decode(c : vec3<f32>) -> vec3<f32> {
  return select(c / 12.92, pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c > vec3<f32>(0.04045));
}

// A lit thing's colour (FLAG_LIT) under the camera's ambient and every light,
// each fading smoothly to nothing at its radius; a spot within its cone too,
// aimed along its direction's x and y. In linear light, so a full white light
// shows the colour exactly as authored.
// ponytail: every light, every lit pixel -- fine for dozens; bin lights into
// screen tiles, as the 3D view's clusters do, if a 2D scene needs hundreds.
fn lit(c : vec3<f32>, flags : u32, world : vec2<f32>) -> vec3<f32> {
  if ((flags & ${FLAG_LIT}u) == 0u) { return c; }
  var light = view.ambient.rgb;
  for (var i = 0u; i < u32(view.viewport.w); i++) {
    let at = lights[i * 4u];            // position, radius
    let colour = lights[i * 4u + 1u];   // colour, intensity
    let spot = lights[i * 4u + 3u];     // the cone's scale and offset, the type
    let d = world - at.xy;
    let k = clamp(1.0 - dot(d, d) / (at.w * at.w), 0.0, 1.0);
    var cone = 1.0;
    if (spot.z != 0.0) {
      let axis = lights[i * 4u + 2u].xy;
      // Aimed straight out of the screen, a spot lights nothing of a 2D view.
      if (dot(axis, axis) < 1e-8) { continue; }
      // The cone as the 3D view shades it: a multiply-add on the cosine.
      let t = clamp(dot(d, normalize(axis)) / max(length(d), 1e-6) * spot.x + spot.y, 0.0, 1.0);
      cone = t * t;
    }
    light += colour.rgb * colour.w * k * k * cone;
  }
  return encode(decode(c) * light);
}

// A shape's coverage, premultiplied: from the distance to its edge, so exact
// at any size, with its outline inside the edge.
fn shapeCover(v : Out) -> vec4<f32> {
  let half = v.extra.xy;
  let p = v.uv;
  var d : f32;
  if ((v.flags & ${FLAG_ELLIPSE}u) != 0u) {
    // Exact for a circle; for an ellipse, close near its edge, where it counts.
    let k0 = length(p / half);
    let k1 = length(p / (half * half));
    d = select(k0 * (k0 - 1.0) / k1, -min(half.x, half.y), k1 == 0.0);
  } else {
    let r = v.extra.z;
    let q = abs(p) - half + r;
    d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
  }
  // A screen pixel, in the shape's units.
  let pixel = 1.0 / length(view.view[0].xy);
  let fill = clamp(0.5 - d / pixel, 0.0, 1.0);
  let inner = clamp(0.5 - (d + v.extra.w) / pixel, 0.0, 1.0);
  let ring = fill - inner;
  return vec4<f32>(v.color.rgb * v.color.a * inner + v.f.rgb * v.f.a * ring, v.color.a * inner + v.f.a * ring);
}

// A path's coverage, premultiplied: the distance to its nearest segment, and
// inside a closed one by nonzero winding, as a canvas fills. Its line is
// centred on it, so round at every join and end.
// ponytail: every segment, every pixel of its box -- fine for hundreds of
// points; triangulate on the CPU for outlines of thousands.
// The squared distance from p to segment a-b.
fn toSegment(p : vec2<f32>, a : vec2<f32>, b : vec2<f32>) -> f32 {
  let e = b - a;
  let w = p - a;
  let d = w - e * clamp(dot(w, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
  return dot(d, d);
}

// The nearest of segments first..last-1 of the packed points (segment i runs
// from point i to i + 1), squared.
fn nearestOf(p : vec2<f32>, first : u32, last : u32) -> f32 {
  var nearest = 1e30;
  for (var i = first; i < last; i++) { nearest = min(nearest, toSegment(p, points[i], points[i + 1u])); }
  return nearest;
}

fn pathCover(v : Out) -> vec4<f32> {
  let p = v.uv;
  let start = u32(v.extra.x);
  let count = u32(v.extra.y);
  let closed = (v.flags & ${FLAG_CLOSED}u) != 0u;
  var nearest = 1e30;
  var winding = 0i;
  for (var i = 0u; i < select(count - 1u, count, closed); i++) {
    let a = points[start + i];
    let b = points[start + (i + 1u) % count];
    nearest = min(nearest, toSegment(p, a, b));
    let e = b - a;
    let w = p - a;
    let side = e.x * w.y - e.y * w.x;
    if (a.y <= p.y) {
      if (b.y > p.y && side > 0.0) { winding += 1; }
    } else if (b.y <= p.y && side < 0.0) {
      winding -= 1;
    }
  }
  if ((v.flags & ${FLAG_CHUNK}u) != 0u) {
    // A piece of a long open path (PATH_CHUNK): a pixel nearer a neighbour's
    // segments is the neighbour's to draw -- the one before, on a tie -- so
    // no pixel of a see-through line is blended twice where two pieces meet.
    let pathFirst = u32(v.color.x);
    let pathLast = pathFirst + u32(v.color.y) - 1u;
    let before = nearestOf(p, max(start, pathFirst + ${PATH_CHUNK}u) - ${PATH_CHUNK}u, start);
    let after = nearestOf(p, start + count - 1u, min(start + count - 1u + ${PATH_CHUNK}u, pathLast));
    if (before <= nearest || after < nearest) { discard; }
  }
  // One square root, not one a segment.
  nearest = sqrt(nearest);
  // A screen pixel, in the path's own units: the view's scale and the node's.
  let pixel = 1.0 / (length(view.view[0].xy) * v.extra.w);
  let fill = select(0.0, clamp(0.5 - select(nearest, -nearest, winding != 0) / pixel, 0.0, 1.0), closed);
  let line = select(0.0, clamp(0.5 - (nearest - v.extra.z * 0.5) / pixel, 0.0, 1.0), v.extra.z > 0.0);
  let fa = select(v.color.a, 0.0, (v.flags & ${FLAG_CHUNK}u) != 0u) * fill;
  let sa = v.f.a * line;
  return vec4<f32>(v.f.rgb * sa + v.color.rgb * fa * (1.0 - sa), sa + fa * (1.0 - sa));
}

// Shapes and paths, which read no image, so draw in one run together.
@fragment
fn fsShape(v : Out) -> @location(0) vec4<f32> {
  var cover : vec4<f32>;
  if ((v.flags & ${FLAG_PATH}u) != 0u) { cover = pathCover(v); } else { cover = shapeCover(v); }
  if (cover.a <= 0.0) { discard; }
  let colour = lit(cover.rgb / cover.a, v.flags, v.world);
  return blended(vec4<f32>(colour, cover.a));
}

@fragment
fn fs(v : Out) -> @location(0) vec4<f32> {
  // Half a texel inside its rect (f: its corners), so filtering never reaches
  // the next frame of a sheet or the next glyph of an atlas.
  let half = 0.5 / vec2<f32>(textureDimensions(image));
  let texel = textureSample(image, imageSampler, clamp(v.uv, v.f.xy + half, max(v.f.zw - half, v.f.xy + half)));
  // A glyph's alpha is a distance field (render/text.js): its edge is at 0.5,
  // one screen pixel wide at any size.
  // How far the field moves across a pixel. Out here, not in a branch: a
  // derivative needs every pixel around it to be running this line.
  let perPixel = max(fwidth(texel.a), 1e-5);
  let edge = clamp((texel.a - 0.5) / perPixel + 0.5, 0.0, 1.0);
  // A colour texture is sampled as linear light; back to the sRGB it was
  // authored in, which is what this view blends.
  var colour = vec4<f32>(encode(texel.rgb), texel.a) * v.color;
  if ((v.flags & ${FLAG_SDF}u) != 0u) {
    // A glyph's outline runs from its edge out to where the field reaches
    // extra.y, 0.5 without one: the fill over it, premultiplied.
    let outer = clamp((texel.a - v.extra.y) / perPixel + 0.5, 0.0, 1.0);
    let ring = max(outer - edge, 0.0);
    let a = v.color.a * edge + v.g.a * ring;
    colour = vec4<f32>((v.color.rgb * v.color.a * edge + v.g.rgb * v.g.a * ring) / max(a, 1e-6), a);
  }
  if ((v.flags & ${FLAG_CUTOUT}u) != 0u) {
    if (colour.a < v.extra.x) { discard; }
    colour.a = 1.0;
  }
  // A plain sprite's edge, smoothed over a screen pixel as a shape's is: a
  // turned sprite is not jagged, and one on whole pixels is unchanged.
  if (v.span.x > 0.0) {
    let inside = min(v.edge, 1.0 - v.edge) * v.span;
    colour.a *= clamp(min(inside.x, inside.y) + 0.5, 0.0, 1.0);
  }
  return blended(vec4<f32>(lit(colour.rgb, v.flags, v.world), colour.a));
}

// A tilemap: uv counts tiles, and each pixel looks up the tile it is in.
//
// From a padded copy of the tileset (padTileset): each tile alone in a cell,
// its edges stretched out to fill it, so its mips never take in a neighbour,
// and a map zoomed far out is filtered, not shimmering. The mip level comes
// from how fast the map's own coordinate changes, which is smooth, not from
// the tile coordinate, which jumps at every tile's edge. A tileset too large
// to pad is read as it is, at full size: extra.w is 0.
@fragment
fn fsTilemap(v : Out) -> @location(0) vec4<f32> {
  let tile = v.extra.xy;
  // In the tileset's texels a screen pixel along: taken here, before anything branches.
  var dx = dpdx(v.uv) * tile;
  var dy = dpdy(v.uv) * tile;
  let map = vec2<i32>(textureDimensions(tiles));
  let cell = vec2<i32>(floor(v.uv));
  let id = textureLoad(tiles, clamp(cell, vec2<i32>(0), map - 1), 0).r;
  let number = id & 0x1fffffffu;
  // Tiled's flips, undone in the reverse of the order it applies them:
  // diagonal first, then horizontal and vertical. The steps turn with them.
  var f = fract(v.uv);
  if ((id & 0x80000000u) != 0u) { f.x = 1.0 - f.x; dx.x = -dx.x; dy.x = -dy.x; }
  if ((id & 0x40000000u) != 0u) { f.y = 1.0 - f.y; dx.y = -dx.y; dy.y = -dy.y; }
  if ((id & 0x20000000u) != 0u) { f = f.yx; dx = dx.yx; dy = dy.yx; }
  let columns = u32(v.extra.z);
  let index = max(number, 1u) - 1u;
  // A cell at a time, from the first's corner: f.xy the cells' size, f.zw where a tile starts in its cell.
  let origin = v.f.zw + vec2<f32>(f32(index % columns), f32(index / columns)) * v.f.xy;
  let size = vec2<f32>(textureDimensions(image));
  var sampled : vec4<f32>;
  if (v.extra.w > 0.0) {
    sampled = textureSampleGrad(image, imageSampler, (origin + f * tile) / size, dx / size, dy / size);
  } else {
    // Unpadded: half a texel inside the tile, so filtering never reaches a neighbour, and no mips.
    let texel = clamp(origin + f * tile, origin + 0.5, origin + tile - 0.5);
    sampled = textureSampleLevel(image, imageSampler, texel / size, 0.0);
  }
  if (number == 0u || any(cell < vec2<i32>(0)) || any(cell >= map)) { discard; }
  let colour = vec4<f32>(encode(sampled.rgb), sampled.a) * v.color;
  return blended(vec4<f32>(lit(colour.rgb, v.flags, v.world), colour.a));
}
`;

/**
 * The pipeline an item draws with, by what it is -- an image (a sprite or a
 * glyph), a shape or path, or a tilemap -- and its blend: 'alpha', 'shape',
 * 'tilemapAdditive' and so on. A cutout is alpha-blended, its edge discarded.
 */
function pipelineFor(kind, blend) {
  const mode = blend === 'cutout' ? 'alpha' : blend;
  if (kind === 'image') return mode;
  return mode === 'alpha' ? kind : kind + mode[0].toUpperCase() + mode.slice(1);
}

/**
 * The order sprites and glyphs draw in, and what each is: by layer, then by
 * when it was added. Returns { entries, runs, points } -- entries as [source,
 * box, entity, first point] with box null but for a glyph and the first point
 * a path's alone; runs as { texture, blend, first, count, tilemap } with
 * tilemap the record for a tilemap's run of one; points every path's, packed.
 */
export function order2D(scene) {
  const items = [];
  for (const [entity, sprite] of scene.sprites) items.push({ entity, source: sprite, layer: sprite.layer, added: sprite.added ?? 0 });
  for (const [entity, text] of scene.texts) items.push({ entity, source: text, layer: text.layer, added: text.added ?? 0, text: true });
  for (const kind of [scene.tilemaps, scene.shapes, scene.paths]) {
    for (const [entity, record] of kind) items.push({ entity, source: record, layer: record.layer, added: record.added ?? 0 });
  }
  for (const [entity, record] of scene.emitters) items.push({ entity, source: record, layer: record.layer, added: record.added ?? 0, emitter: true });
  items.sort((p, q) => p.layer - q.layer || p.added - q.added);
  const entries = [];
  const runs = [];
  let pointCount = 0;
  for (const item of items) pointCount += item.source.kind === 'path' ? item.source.points.length / 2 : 0;
  const points = new Float32Array(pointCount * 2);
  let packed = 0;
  for (const item of items) {
    const first = packed;
    if (item.source.kind === 'path') {
      points.set(item.source.points, packed * 2);
      packed += item.source.points.length / 2;
    }
    place(item, entries, runs, first);
  }
  return { entries, runs, points, lastLayer: items.length > 0 ? items[items.length - 1].layer : -Infinity };
}

/**
 * One item's entries and its place in the runs, after everything before it:
 * a slot a glyph, or one for anything else, and `first` its first point if
 * it is a path. An emitter's particles are drawn from its own pool
 * (render/particles.js): a run of no slots, holding its place in the order.
 */
function place(item, entries, runs, first) {
  if (item.emitter) {
    runs.push({ texture: null, blend: 'emitter', first: entries.length, count: 0, tilemap: null, emitter: item.entity });
    return;
  }
  const kind = item.source.kind;
  const tilemap = kind === 'tilemap' ? item.source : null;
  const boxes = item.text ? item.source.boxes : [null];
  // A long open path goes in pieces: see PATH_CHUNK.
  const segments = kind === 'path' && !item.source.closed ? item.source.points.length / 2 - 1 : 0;
  const pieces = segments > PATH_CHUNK ? Math.ceil(segments / PATH_CHUNK) : 0;
  // Shapes and paths read no image, and draw together.
  const drawn = kind === 'shape' || kind === 'path';
  const texture = item.text ? item.source.font.texture : tilemap ? tilemap.tileset : drawn ? null : item.source.texture;
  const blend = pipelineFor(tilemap ? 'tilemap' : drawn ? 'shape' : 'image', item.source.blend);
  const repeat = !item.text && kind === undefined && repeats(item.source);
  const add = (box, piece) => {
    const last = runs[runs.length - 1];
    // Each tilemap has its own tiles to bind, so is a run of its own.
    if (last && last.texture === texture && last.blend === blend && last.repeat === repeat && tilemap === null) last.count++;
    else runs.push({ texture, blend, first: entries.length, count: 1, tilemap, repeat, font: item.text ? item.source.font : null });
    entries.push([item.source, box, item.entity, first, piece]);
  };
  if (pieces > 0) for (let piece = 0; piece < pieces; piece++) add(null, piece);
  else for (const box of boxes) add(box, -1);
}

/**
 * One sprite's or glyph's slot, at float offset `o`: placed, turned and
 * scaled by its node, as a canvas transform would. A negative scale on the
 * node mirrors it -- the usual way to turn a character round. `firstPoint` is
 * where a path's points start in the packed list order2D makes, and `piece`
 * which piece of a long open one this slot draws (PATH_CHUNK), or -1.
 */
export function write2D(out, o, world, entity, source, box, firstPoint = 0, piece = -1) {
  const m = handleIndex(entity) * 16;
  const glyph = box !== null;
  const tilemap = source.kind === 'tilemap';
  const shape = source.kind === 'shape';
  const path = source.kind === 'path';
  const [lx, ly, flipped, turn] = frame2D(world, entity, FRAME);
  const mirrored = flipped === 1;
  const angle = turn + (glyph || tilemap || shape || path ? 0 : source.angle);
  let width, height, pivotX, pivotY, rect;
  if (shape) {
    width = source.size[0];
    height = source.size[1];
    pivotX = source.pivot[0];
    pivotY = source.pivot[1];
    rect = [0, 0, 1, 1];
  } else if (path) {
    // The quad covers its points and half its line, and its uv is the node's
    // own units, which the points are in.
    const half = source.strokeWidth / 2;
    let [x0, y0, x1, y1] = source.bounds;
    if (piece >= 0) {
      // A piece's own points' bounds.
      const p = source.points;
      const end = Math.min((piece + 1) * PATH_CHUNK, p.length / 2 - 1);
      x0 = y0 = Infinity;
      x1 = y1 = -Infinity;
      for (let i = piece * PATH_CHUNK; i <= end; i++) {
        x0 = Math.min(x0, p[i * 2]); x1 = Math.max(x1, p[i * 2]);
        y0 = Math.min(y0, p[i * 2 + 1]); y1 = Math.max(y1, p[i * 2 + 1]);
      }
    }
    rect = [x0 - half, y0 - half, x1 + half, y1 + half];
    width = Math.max(rect[2] - rect[0], 1e-3);
    height = Math.max(rect[3] - rect[1], 1e-3);
    pivotX = -rect[0] / width;
    pivotY = -rect[1] / height;
  } else if (tilemap) {
    // The quad covers the whole map, and its uv counts tiles.
    width = source.columns * source.tileSize[0];
    height = source.rows * source.tileSize[1];
    pivotX = source.pivot[0];
    pivotY = source.pivot[1];
    rect = [0, 0, source.columns, source.rows];
  } else if (glyph) {
    // Text is laid out y-up in ems (render/text.js); here y points down, so a
    // glyph's pivot is measured from its top.
    width = box.width * source.size;
    height = box.height * source.size;
    pivotX = -box.x / box.width;
    pivotY = (box.y + box.height) / box.height;
    rect = source.font.rectOf(box.char);
  } else {
    rect = spriteRect(source);
    // A unit is a pixel, so a sprite given no size is its frame's own.
    width = source.sizeGiven ? source.size[0] : source.texture.width * Math.abs(rect[2] - rect[0]);
    height = source.sizeGiven ? source.size[1] : source.texture.height * Math.abs(rect[3] - rect[1]);
    pivotX = source.pivot[0];
    pivotY = source.pivot[1];
  }
  out[o] = world[m + 12];
  out[o + 1] = world[m + 13];
  out[o + 2] = angle;
  out[o + 3] = (glyph ? FLAG_SDF : 0) | (tilemap ? FLAG_TILEMAP : 0) | (!glyph && source.blend === 'cutout' ? FLAG_CUTOUT : 0)
    | (shape ? FLAG_SHAPE | (source.shape === 'ellipse' ? FLAG_ELLIPSE : 0) : 0)
    | (path ? FLAG_PATH | (source.closed ? FLAG_CLOSED : 0) : 0) | (source.lit ? FLAG_LIT : 0) | (piece >= 0 ? FLAG_CHUNK : 0);
  out[o + 4] = width * lx * (mirrored ? -1 : 1);
  out[o + 5] = height * ly;
  out[o + 6] = pivotX;
  out[o + 7] = pivotY;
  for (let c = 0; c < 4; c++) {
    out[o + 8 + c] = rect[c];
    out[o + 12 + c] = source.color[c];
  }
  if (shape) {
    // In the node's scaled units, as its size is; the corner and outline
    // scale by the node's lesser axis, so a corner still fits.
    const s = Math.min(lx, ly);
    out[o + 16] = Math.abs(out[o + 4]) / 2;
    out[o + 17] = out[o + 5] / 2;
    out[o + 18] = shapeRadius(source) * s;
    out[o + 19] = source.strokeWidth * s;
  } else if (path) {
    // In the node's own units, as its points are; the pixel size in them
    // takes the node's scale, the lesser axis's.
    const pointCount = source.points.length / 2;
    out[o + 16] = firstPoint + (piece >= 0 ? piece * PATH_CHUNK : 0);
    out[o + 17] = piece >= 0 ? Math.min(PATH_CHUNK + 1, pointCount - piece * PATH_CHUNK) : pointCount;
    out[o + 18] = source.strokeWidth;
    out[o + 19] = Math.min(lx, ly);
    // A piece has no fill: its colour slot holds where its whole path's points are.
    if (piece >= 0) {
      out[o + 12] = firstPoint;
      out[o + 13] = pointCount;
    }
  } else if (tilemap) {
    const layout = tileLayout(source);
    out[o + 16] = source.tileSize[0];
    out[o + 17] = source.tileSize[1];
    out[o + 18] = layout.columns;
    out[o + 19] = layout.padded ? 1 : 0;
  } else {
    out[o + 16] = glyph ? 0 : source.cutoff;
    out[o + 17] = glyph ? source.strokeEdge : 0;
    out[o + 18] = out[o + 19] = 0;
  }
  // The outline's colour; a tilemap's margin and spacing; a sprite's or glyph's rect, low corner first.
  for (let c = 0; c < 4; c++) out[o + 20 + c] = shape || path ? source.stroke[c] : 0;
  for (let c = 0; c < 4; c++) out[o + 24 + c] = glyph ? source.stroke[c] : 0;
  if (!(shape || path || tilemap)) {
    out[o + 20] = Math.min(rect[0], rect[2]);
    out[o + 21] = Math.min(rect[1], rect[3]);
    out[o + 22] = Math.max(rect[0], rect[2]);
    out[o + 23] = Math.max(rect[1], rect[3]);
  }
  if (tilemap) {
    const layout = tileLayout(source);
    out[o + 20] = layout.cell[0];
    out[o + 21] = layout.cell[1];
    out[o + 22] = layout.pad[0];
    out[o + 23] = layout.pad[1];
  }
}

/** The largest texture every WebGPU device makes: a padded tileset past it is read unpadded. */
const PADDED_MOST = 8192;

/**
 * Where a tilemap's tiles are in the texture it draws from. Padded: each
 * tile in a cell of its own, `pad` texels in from its corner, with room
 * around it for mips down to a tile two texels across -- each level halves
 * the room, and a level's filter reaches a texel past the tile -- and cells
 * a multiple of the last level's texel, so no level's texel straddles two.
 * Unpadded, the tileset as it is: a margin, then tiles and gaps.
 */
export function tileLayout({ tileset, tileSize: [tw, th], margin, spacing }) {
  const columns = Math.floor((tileset.width - 2 * margin + spacing) / (tw + spacing));
  const rows = Math.floor((tileset.height - 2 * margin + spacing) / (th + spacing));
  const levels = Math.max(0, Math.floor(Math.log2(Math.min(tw, th) / 2)));
  const step = 2 ** levels;
  const cell = [Math.ceil((tw + 2 * step) / step) * step, Math.ceil((th + 2 * step) / step) * step];
  const width = columns * cell[0], height = rows * cell[1];
  if (levels > 0 && width <= PADDED_MOST && height <= PADDED_MOST) {
    return { padded: true, columns, rows, cell, pad: [step, step], levels: levels + 1, width, height };
  }
  return { padded: false, columns, rows, cell: [tw + spacing, th + spacing], pad: [margin, margin], levels: 1, width: 0, height: 0 };
}

const FRAME = new Float64Array(4);

/** A 2D entry's record as the scene holds it now: a set call replaces the one order2D saw. */
function live(scene, entity, source, box) {
  const kind = box !== null ? scene.texts
    : source.kind === 'shape' ? scene.shapes : source.kind === 'path' ? scene.paths
      : source.kind === 'tilemap' ? scene.tilemaps : scene.sprites;
  return kind.get(entity) ?? source;
}

export class View2D {
  /** `particles` is the renderer's ParticleSystem, to draw emitters with; null draws none. */
  static async create(rhi, pipelines, particles = null) {
    const device = rhi.device;
    const shader = await compileShader(device, SHADER, 'view2d.wgsl');
    const viewLayout = device.createBindGroupLayout({
      label: 'view2d',
      entries: [
        // The fragment stage too: a shape sizes its edge's smoothing by the zoom.
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });
    const imageLayout = device.createBindGroupLayout({
      label: 'view2d-image',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const tilemapLayout = device.createBindGroupLayout({
      label: 'view2d-tilemap',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'uint' } },
      ],
    });
    const layout = createPipelineLayout(device, { 0: viewLayout, 1: imageLayout }, 'view2d');
    const kinds = {
      image: { layout, fragmentEntry: 'fs' },
      shape: { layout: createPipelineLayout(device, { 0: viewLayout }, 'view2d-shape'), fragmentEntry: 'fsShape' },
      tilemap: { layout: createPipelineLayout(device, { 0: viewLayout, 1: tilemapLayout }, 'view2d-tilemap'), fragmentEntry: 'fsTilemap' },
    };
    // Every kind in every blend. Only the plain ones are built up front; the
    // rest, the first frame that draws one.
    const descriptors = {};
    for (const [kind, { layout: kindLayout, fragmentEntry }] of Object.entries(kinds)) {
      for (const blend of BLENDS) {
        descriptors[pipelineFor(kind, blend)] = {
          label: `view2d:${kind}:${blend}`,
          layout: kindLayout,
          shader,
          fragmentEntry,
          targets: [{ format: rhi.surfaceFormat, blend: BLEND_STATE[blend] }],
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depth: null,
          constants: { BLEND: { alpha: 0, additive: 1, multiply: 3, screen: 4 }[blend] },
        };
      }
    }
    await pipelines.warm([descriptors.alpha, descriptors.shape, descriptors.tilemap]);
    const view = new View2D(rhi, pipelines, descriptors, viewLayout, imageLayout, tilemapLayout);
    view.particles = particles;
    return view;
  }

  constructor(rhi, pipelines, descriptors, viewLayout, imageLayout, tilemapLayout) {
    this.particles = null;
    this.rhi = rhi;
    this.pipelines = pipelines;
    this._descriptors = descriptors;
    this._viewLayout = viewLayout;
    this._imageLayout = imageLayout;
    this._tilemapLayout = tilemapLayout;
    /** Tilemap record -> { texture, group }: its ids on the GPU. */
    this._tilemaps = new Map();
    /** Tileset -> its padded copies, by tile layout: see tileLayout. */
    this._padded = new Map();
    this._uniform = createBuffer(rhi, { label: 'view2d', size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._uniformData = new Float32Array(24);
    /** Paths' points and the scene's lights, each grown as needed: { buffer, size } by name. */
    this._stores = { points: null, lights: null };
    this._data = new Float32Array(64 * SPRITE2D_FLOATS);
    this._buffer = null;
    this._capacity = 0;
    this._group = null;
    this._imageGroups = new WeakMap();
    /** Entity -> [first slot, slot count]. */
    this._slots = new Map();
    this._entries = [];
    this._runs = [];
    this._scene = null;
    this._order = -1;
    this.count = 0;
    /** Slots written this frame: all of them when the list was rebuilt. */
    this.written = 0;
    this._execute = (pass) => this._encode(pass);
  }

  /** Another view drawing with the same pipelines: one a scene, so each keeps its own slots. */
  another(particles = null) {
    const view = new View2D(this.rhi, this.pipelines, this._descriptors, this._viewLayout, this._imageLayout, this._tilemapLayout);
    view.particles = particles;
    return view;
  }

  /**
   * Bring the slots up to date with the scene: rebuilt when the order may
   * have changed, otherwise only what moved or changed. \`moved\` is the
   * transform store's per-node flags for this frame, or null if nothing moved.
   */
  prepare(scene, camera, width, height, moved) {
    const world = scene.transforms.world;
    // Glyphs back in an atlas that started again, before it is built against.
    for (const record of scene.texts.values()) record.font.ensureBoxes(record);
    // A font's atlas grows when any scene needs new glyphs, which moves every
    // glyph's rect and replaces its texture: this view's too.
    // Holes from removals are compacted away once they are a quarter of it.
    const rebuild = scene !== this._scene || scene.spriteOrder !== this._order
      || this._runs.some((run) => run.font !== null && run.font !== undefined && run.font.texture !== run.texture)
      || (this._holes > 64 && this._holes * 4 > this.count) || !this._appendable(scene);
    if (rebuild) {
      const { entries, runs, points, lastLayer } = order2D(scene);
      this._holes = 0;
      this._lastLayer = lastLayer;
      this._grow(entries.length);
      this._upload('points', points);
      this._slots.clear();
      entries.forEach(([source, box, entity, firstPoint, piece], slot) => {
        write2D(this._data, slot * SPRITE2D_FLOATS, world, entity, source, box, firstPoint, piece);
        const span = this._slots.get(entity);
        if (span) span[1]++; else this._slots.set(entity, [slot, 1]);
      });
      this._entries = entries;
      this._runs = runs;
      this.count = entries.length;
      this._scene = scene;
      this._order = scene.spriteOrder;
      this.written = entries.length;
      if (entries.length > 0) this.rhi.queue.writeBuffer(this._buffer, 0, this._data, 0, entries.length * SPRITE2D_FLOATS);
    } else {
      const spans = (this._dirty ??= []);
      spans.length = 0;
      // A removed item's slots are left empty: drawn as nothing until the next rebuild.
      for (const entity of scene.removed2D) {
        const span = this._slots.get(entity);
        if (span === undefined) continue;
        for (let slot = span[0]; slot < span[0] + span[1]; slot++) {
          this._data[slot * SPRITE2D_FLOATS + 4] = this._data[slot * SPRITE2D_FLOATS + 5] = 0;
        }
        spans.push(span[0], span[0] + span[1] - 1);
        this._holes += span[1];
        this._slots.delete(entity);
      }
      if (this._pending.length > 0) this._append(world, spans);
      for (const entity of scene.spritesChanged) this._rewrite(scene, world, entity, spans);
      if (moved !== null) {
        for (const entity of this._slots.keys()) if (moved[handleIndex(entity)] === 1) this._rewrite(scene, world, entity, spans);
      }
      this.written = this._uploadSpans(spans);
    }
    scene.spritesChanged.clear();
    scene.added2D.clear();
    scene.removed2D.clear();
    this._uploadTiles(scene, rebuild);
    // The lights, where their nodes are now: a few floats each, so every frame.
    if (scene.lightCount > 0) scene.refreshLights();
    this._upload('lights', scene.lights.subarray(0, scene.lightCount * 16));

    const u = this._uniformData;
    u.set(camera.view, 0);
    u[16] = width;
    u[17] = height;
    u[18] = camera.pixelSnap ? 1 : 0;
    u[19] = scene.lightCount;
    u.set(camera.ambient, 20);
    this.rhi.queue.writeBuffer(this._uniform, 0, u);
    return this.count;
  }

  /**
   * Whether what was added since the last frame goes at the end of the list:
   * all of it on the last item's layer or higher, as a newly added thing is
   * last in its layer. Keeps those items, in order, for _append.
   */
  _appendable(scene) {
    this._pending ??= [];
    this._pending.length = 0;
    if (scene.added2D.size === 0) return true;
    for (const entity of scene.added2D) {
      const sprite = scene.sprites.get(entity);
      const text = scene.texts.get(entity);
      const source = sprite ?? text ?? scene.shapes.get(entity);
      if (source !== undefined) this._pending.push({ entity, source, layer: source.layer, added: source.added, text: text !== undefined });
    }
    this._pending.sort((p, q) => p.added - q.added);
    let floor = this._lastLayer ?? -Infinity;
    for (const item of this._pending) {
      if (item.layer < floor) return false;
      floor = item.layer;
    }
    return true;
  }

  /** The items _appendable kept, onto the end of the list and its runs. */
  _append(world, spans) {
    const start = this._entries.length;
    for (const item of this._pending) place(item, this._entries, this._runs, 0);
    const count = this._entries.length;
    const buffer = this._buffer;
    this._grow(count);
    for (let slot = start; slot < count; slot++) {
      const [source, box, entity] = this._entries[slot];
      write2D(this._data, slot * SPRITE2D_FLOATS, world, entity, source, box, 0);
      const span = this._slots.get(entity);
      if (span) span[1]++; else this._slots.set(entity, [slot, 1]);
    }
    // A new buffer has none of the old slots in it.
    if (this._buffer !== buffer) spans.push(0, count - 1); else spans.push(start, count - 1);
    this.count = count;
    this._lastLayer = this._pending[this._pending.length - 1].layer;
  }

  /** Rewrite `entity`'s slots from its record as it is now, and note them in `spans`: [first, last, ...]. */
  _rewrite(scene, world, entity, spans) {
    const span = this._slots.get(entity);
    if (span === undefined) return;
    for (let slot = span[0]; slot < span[0] + span[1]; slot++) {
      const entry = this._entries[slot];
      entry[0] = live(scene, entity, entry[0], entry[1]);
      // A text rewritten in place (setText) has new glyphs, as many as before.
      if (entry[1] !== null) entry[1] = entry[0].boxes[slot - span[0]];
      write2D(this._data, slot * SPRITE2D_FLOATS, world, entity, entry[0], entry[1], entry[3], entry[4]);
    }
    spans.push(span[0], span[0] + span[1] - 1);
  }

  /**
   * Upload the slots in `spans`, [first, last, ...], in as few calls as pays:
   * spans a few slots apart go as one, the gap with them, which costs less
   * than another call (as in render/gpudriven.js); spans far apart go
   * separately, so two movers at either end of a long list are two slots, not
   * the whole list. Returns how many slots went.
   */
  _uploadSpans(spans) {
    if (spans.length === 0) return 0;
    // Changed nodes come in any order; moved ones in slot order.
    let sorted = true;
    for (let k = 2; k < spans.length; k += 2) if (spans[k] < spans[k - 2]) { sorted = false; break; }
    const order = [];
    for (let k = 0; k < spans.length; k += 2) order.push(k);
    if (!sorted) order.sort((a, b) => spans[a] - spans[b]);
    let written = 0;
    let first = spans[order[0]], last = spans[order[0] + 1];
    const send = () => {
      const count = last - first + 1;
      this.rhi.queue.writeBuffer(this._buffer, first * SPRITE2D_BYTES, this._data, first * SPRITE2D_FLOATS, count * SPRITE2D_FLOATS);
      written += count;
    };
    for (let i = 1; i < order.length; i++) {
      const k = order[i];
      if (spans[k] - last - 1 <= MERGE_GAP) last = Math.max(last, spans[k + 1]);
      else {
        send();
        first = spans[k];
        last = spans[k + 1];
      }
    }
    send();
    return written;
  }

  /** `data` into the storage buffer named `name`, grown to fit; never empty, so always bindable. */
  _upload(name, data) {
    const bytes = Math.max(data.byteLength, 64);
    let store = this._stores[name];
    if (store === null || store.size < bytes) {
      store?.buffer.destroy();
      const size = grownCapacity(store?.size ?? 0, bytes, storageCapacity(this.rhi, 1), `2D ${name}`);
      store = this._stores[name] = {
        size, buffer: createBuffer(this.rhi, { label: `view2d-${name}`, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      };
      this._group = null;
    }
    if (data.byteLength > 0) this.rhi.queue.writeBuffer(store.buffer, 0, data);
  }

  /**
   * Each tilemap's ids onto the GPU: all of a new one, and of the rest the
   * block changed since the last frame. \`tilesWritten\` counts them.
   */
  _uploadTiles(scene, rebuilt) {
    this.tilesWritten = 0;
    if (rebuilt) {
      const live = new Set(scene.tilemaps.values());
      for (const [map, gpu] of this._tilemaps) {
        if (!live.has(map)) {
          gpu.texture.destroy();
          this._tilemaps.delete(map);
        }
      }
      // And the padded copies of tilesets no map here uses any more: kept,
      // each tileset an editor reloaded left one behind, unloaded or not.
      const tilesets = new Set([...live].map((map) => map.tileset));
      for (const [tileset, byLayout] of this._padded ?? []) {
        if (!tilesets.has(tileset)) {
          for (const padded of byLayout.values()) padded.texture.destroy();
          this._padded.delete(tileset);
        }
      }
    }
    for (const map of scene.tilemaps.values()) {
      let region = map.dirty;
      if (!this._tilemaps.has(map)) {
        const most = this.rhi.device.limits.maxTextureDimension2D;
        if (map.columns > most || map.rows > most) {
          throw new Error(`addTilemap: a ${map.columns} x ${map.rows} map is past this device's ${most} tiles a side`);
        }
        const texture = this.rhi.device.createTexture({
          label: 'view2d-tiles', size: [map.columns, map.rows], format: 'r32uint',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this._tilemaps.set(map, { texture, group: null });
        region = [0, 0, map.columns, map.rows];
      }
      if (region === null) continue;
      const [x0, y0, x1, y1] = region;
      this.rhi.queue.writeTexture(
        { texture: this._tilemaps.get(map).texture, origin: { x: x0, y: y0 } },
        map.tiles, { offset: (y0 * map.columns + x0) * 4, bytesPerRow: map.columns * 4 },
        { width: x1 - x0, height: y1 - y0 },
      );
      this.tilesWritten += (x1 - x0) * (y1 - y0);
      map.dirty = null;
    }
  }

  _grow(count) {
    if (count * SPRITE2D_FLOATS > this._data.length) {
      // Kept: an append writes only its own slots.
      const data = new Float32Array(grownCapacity(this._data.length / SPRITE2D_FLOATS, count) * SPRITE2D_FLOATS);
      data.set(this._data);
      this._data = data;
    }
    const bytes = Math.max(count, 1) * SPRITE2D_BYTES;
    if (bytes > this._capacity) {
      this._buffer?.destroy();
      this._capacity = grownCapacity(this._capacity, bytes, storageCapacity(this.rhi, 1), '2D sprites');
      this._buffer = createBuffer(this.rhi, {
        label: 'view2d-sprites', size: this._capacity, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this._group = null;
    }
  }

  /**
   * The pass: onto \`surface\`, cleared to the camera's background. \`particles\`
   * is the particle pool it reads, when the scene has emitters.
   */
  addPass(graph, { surface, background, particles = null }) {
    graph.addPass({
      name: '2d',
      reads: particles === null ? [] : [particles],
      color: [{ resource: surface, clear: { r: background[0], g: background[1], b: background[2], a: background[3] } }],
      execute: this._execute,
    });
  }

  _encode(pass) {
    if (this._runs.length === 0) return;
    this._group ??= this.rhi.device.createBindGroup({
      label: 'view2d', layout: this._viewLayout,
      entries: [
        { binding: 0, resource: { buffer: this._uniform } },
        { binding: 1, resource: { buffer: this._buffer } },
        { binding: 2, resource: { buffer: this._stores.points.buffer } },
        { binding: 3, resource: { buffer: this._stores.lights.buffer } },
      ],
    });
    pass.setBindGroup(0, this._group);
    let bound = null;
    for (const { texture, blend, first, count, tilemap, emitter, repeat } of this._runs) {
      if (emitter !== undefined) {
        // Particles bring their own pipeline and bind groups; ours go back after.
        if (this.particles === null) continue;
        this.particles.draw2D(pass, emitter);
        pass.setBindGroup(0, this._group);
        bound = null;
        continue;
      }
      if (blend !== bound) {
        pass.setPipeline(this.pipelines.get(this._descriptors[blend]));
        bound = blend;
      }
      if (tilemap !== null) {
        const gpu = this._tilemaps.get(tilemap);
        const padded = this._paddedFor(tilemap);
        if (gpu.padded !== padded) {
          gpu.padded = padded;
          gpu.group = null;
        }
        gpu.group ??= this.rhi.device.createBindGroup({
          label: 'view2d-tilemap', layout: this._tilemapLayout,
          entries: [
            { binding: 0, resource: padded?.view ?? texture.view },
            { binding: 1, resource: texture.pixelated ? pixelatedSampler(this.rhi) : clampSampler(this.rhi) },
            { binding: 2, resource: gpu.texture.createView() },
          ],
        });
        pass.setBindGroup(1, gpu.group);
        pass.draw(6, count, 0, first);
        continue;
      }
      if (texture === null) {
        pass.draw(6, count, 0, first);
        continue;
      }
      // One group per texture and sampler: clamped, or repeating.
      let groups = this._imageGroups.get(texture);
      if (!groups) this._imageGroups.set(texture, groups = []);
      groups[repeat ? 1 : 0] ??= this.rhi.device.createBindGroup({
        label: 'view2d-image', layout: this._imageLayout,
        entries: [
          { binding: 0, resource: texture.view },
          { binding: 1, resource: spriteSampler(this.rhi, texture.pixelated, repeat) },
        ],
      });
      pass.setBindGroup(1, groups[repeat ? 1 : 0]);
      pass.draw(6, count, 0, first);
    }
  }

  /**
   * A tilemap's tileset, padded and mipped as tileLayout lays it out; made
   * the first time it is drawn, and shared by every map that lays it out
   * alike. Null when it is read as it is.
   */
  _paddedFor(map) {
    const layout = tileLayout(map);
    if (!layout.padded) return null;
    const tileset = map.tileset;
    let byLayout = this._padded.get(tileset);
    if (!byLayout) this._padded.set(tileset, byLayout = new Map());
    const key = `${map.tileSize[0]}x${map.tileSize[1]}+${map.margin}+${map.spacing}`;
    let padded = byLayout.get(key);
    if (!padded) {
      padded = padTileset(this.rhi, this.pipelines, tileset, map, layout);
      byLayout.set(key, padded);
    }
    return padded;
  }

  destroy() {
    for (const gpu of this._tilemaps.values()) gpu.texture.destroy();
    for (const byLayout of this._padded.values()) for (const padded of byLayout.values()) padded.texture.destroy();
    for (const store of Object.values(this._stores)) store?.buffer.destroy();
    this._buffer?.destroy();
    this._uniform.destroy();
  }
}

const PAD_SHADER = /* wgsl */ `
struct Pad {
  tile    : vec2<f32>,   // a tile's size, in texels
  cell    : vec2<f32>,   // a cell's, in the padded copy
  pad     : vec2<f32>,   // where the tile starts in its cell
  margin  : f32,
  spacing : f32,
  columns : f32,
};
@group(0) @binding(0) var<uniform> pad : Pad;
@group(0) @binding(1) var source : texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) index : u32) -> @builtin(position) vec4<f32> {
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

/** A texel of the padded copy: its cell's tile, the nearest texel of it -- stretched past its edges. */
@fragment
fn fs(@builtin(position) at : vec4<f32>) -> @location(0) vec4<f32> {
  let cell = floor(at.xy / pad.cell);
  let index = cell.y * pad.columns + cell.x;
  let within = clamp(floor(at.xy - cell * pad.cell - pad.pad), vec2<f32>(0.0), pad.tile - 1.0);
  let texel = pad.margin + vec2<f32>(index % pad.columns, floor(index / pad.columns)) * (pad.tile + pad.spacing) + within;
  return textureLoad(source, vec2<i32>(texel), 0);
}
`;

/**
 * A tileset laid out as tileLayout pads it, drawn once by a pass that reads
 * the original texel for texel, then mipped by coverage, as a sprite's
 * image is.
 */
function padTileset(rhi, pipelines, tileset, map, layout) {
  const device = rhi.device;
  const format = tileset.texture.format;
  const texture = device.createTexture({
    label: 'view2d-tileset-padded', size: [layout.width, layout.height], format, mipLevelCount: layout.levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
  });
  let built = PAD_BUILT.get(device);
  if (!built) {
    const module = device.createShaderModule({ label: 'view2d-pad', code: PAD_SHADER });
    built = { module, pipelines: new Map() };
    PAD_BUILT.set(device, built);
  }
  let pipeline = built.pipelines.get(format);
  if (!pipeline) {
    pipeline = device.createRenderPipeline({
      label: 'view2d-pad', layout: 'auto',
      vertex: { module: built.module, entryPoint: 'vs' },
      fragment: { module: built.module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    built.pipelines.set(format, pipeline);
  }
  const tail = device.createBuffer({ label: 'view2d-pad', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  rhi.queue.writeBuffer(tail, 0, new Float32Array([
    map.tileSize[0], map.tileSize[1], layout.cell[0], layout.cell[1], layout.pad[0], layout.pad[1], map.margin, map.spacing, layout.columns, 0, 0, 0,
  ]));
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: tail } },
      { binding: 1, resource: tileset.texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
    ],
  });
  const encoder = device.createCommandEncoder({ label: 'view2d-pad' });
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.draw(3);
  pass.end();
  rhi.queue.submit([encoder.finish()]);
  tail.destroy();
  generateMipmaps(rhi, texture);
  return { texture, view: texture.createView() };
}

const PAD_BUILT = new WeakMap();
