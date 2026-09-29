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
import { clampSampler, pixelatedSampler } from '../rhi/texture.js';
import { grownCapacity } from '../core/grow.js';
import { handleIndex } from '../core/handle.js';
import { spriteRect } from '../scene/scene.js';

/** position, angle, flags; size, pivot; rect; colour; per kind; outline colour. */
export const SPRITE2D_FLOATS = 24;
const SPRITE2D_BYTES = SPRITE2D_FLOATS * 4;
const FLAG_SDF = 1;
const FLAG_CUTOUT = 2;
const FLAG_TILEMAP = 4;
const FLAG_SHAPE = 8;
const FLAG_ELLIPSE = 16;
const FLAG_PATH = 32;
const FLAG_CLOSED = 64;
const FLAG_LIT = 128;

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
  f     : vec4<f32>,   // a shape's or path's outline colour; a tilemap's margin and spacing
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

override ADDITIVE : bool = false;

struct Out {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) color : vec4<f32>,
  @location(2) @interpolate(flat) flags : u32,
  @location(3) @interpolate(flat) extra : vec4<f32>,
  @location(4) @interpolate(flat) f : vec4<f32>,   // as the slot's f
  @location(5) world : vec2<f32>,
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
  var corner = corners[index];
  // A shape's or path's quad reaches a screen pixel past its edge, for the edge's smoothing.
  if (shape || path) { corner += (corner * 2.0 - 1.0) / max(abs(s.b.xy) * length(view.view[0].xy), vec2<f32>(1e-5)); }
  // y points down, so this turns clockwise on screen, as CSS rotate() does.
  let c = cos(s.a.z);
  let n = sin(s.a.z);
  let local = (corner - s.b.zw) * s.b.xy;
  let turned = vec2<f32>(c * local.x - n * local.y, n * local.x + c * local.y);
  var pivot = (view.view * vec4<f32>(s.a.xy, 0.0, 1.0)).xy;
  if (view.viewport.z > 0.5) { pivot = round(pivot); }
  let screen = pivot + (view.view * vec4<f32>(turned, 0.0, 0.0)).xy;
  var out : Out;
  out.clip = vec4<f32>(screen.x / view.viewport.x * 2.0 - 1.0, 1.0 - screen.y / view.viewport.y * 2.0, 0.0, 1.0);
  // A shape's uv is where it is from its centre, in its own units.
  out.uv = select(mix(s.rect.xy, s.rect.zw, corner), (corner - 0.5) * abs(s.b.xy), shape);
  out.color = s.color;
  out.flags = u32(s.a.w);
  out.extra = s.e;
  out.f = s.f;
  out.world = s.a.xy + turned;
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
    let e = b - a;
    let w = p - a;
    let t = clamp(dot(w, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
    nearest = min(nearest, length(w - e * t));
    let side = e.x * w.y - e.y * w.x;
    if (a.y <= p.y) {
      if (b.y > p.y && side > 0.0) { winding += 1; }
    } else if (b.y <= p.y && side < 0.0) {
      winding -= 1;
    }
  }
  // A screen pixel, in the path's own units: the view's scale and the node's.
  let pixel = 1.0 / (length(view.view[0].xy) * v.extra.w);
  let fill = select(0.0, clamp(0.5 - select(nearest, -nearest, winding != 0) / pixel, 0.0, 1.0), closed);
  let line = select(0.0, clamp(0.5 - (nearest - v.extra.z * 0.5) / pixel, 0.0, 1.0), v.extra.z > 0.0);
  let fa = v.color.a * fill;
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
  if (ADDITIVE) { return vec4<f32>(colour * cover.a, 0.0); }
  return vec4<f32>(colour, cover.a);
}

@fragment
fn fs(v : Out) -> @location(0) vec4<f32> {
  let texel = textureSample(image, imageSampler, v.uv);
  // A glyph's alpha is a distance field (render/text.js): its edge is at 0.5,
  // one screen pixel wide at any size.
  let edge = clamp((texel.a - 0.5) / max(fwidth(texel.a), 1e-5) + 0.5, 0.0, 1.0);
  // A colour texture is sampled as linear light; back to the sRGB it was
  // authored in, which is what this view blends.
  var colour = select(vec4<f32>(encode(texel.rgb), texel.a), vec4<f32>(1.0, 1.0, 1.0, edge), (v.flags & ${FLAG_SDF}u) != 0u) * v.color;
  if ((v.flags & ${FLAG_CUTOUT}u) != 0u) {
    if (colour.a < v.extra.x) { discard; }
    colour.a = 1.0;
  }
  colour = vec4<f32>(lit(colour.rgb, v.flags, v.world), colour.a);
  if (ADDITIVE) { return vec4<f32>(colour.rgb * colour.a, 0.0); }
  return colour;
}

// A tilemap: uv counts tiles, and each pixel looks up the tile it is in.
@fragment
fn fsTilemap(v : Out) -> @location(0) vec4<f32> {
  let tile = v.extra.xy;
  let map = vec2<i32>(textureDimensions(tiles));
  let cell = vec2<i32>(floor(v.uv));
  let id = textureLoad(tiles, clamp(cell, vec2<i32>(0), map - 1), 0).r;
  let number = id & 0x1fffffffu;
  // Tiled's flips, undone in the reverse of the order it applies them:
  // diagonal first, then horizontal and vertical.
  var f = fract(v.uv);
  if ((id & 0x80000000u) != 0u) { f.x = 1.0 - f.x; }
  if ((id & 0x40000000u) != 0u) { f.y = 1.0 - f.y; }
  if ((id & 0x20000000u) != 0u) { f = f.yx; }
  let columns = u32(v.extra.z);
  let index = max(number, 1u) - 1u;
  // Past the margin, a tile and a gap at a time.
  let origin = v.f.x + vec2<f32>(f32(index % columns), f32(index / columns)) * (tile + v.f.y);
  // Half a texel inside the tile, so filtering never reaches its neighbour.
  let texel = clamp(origin + f * tile, origin + 0.5, origin + tile - 0.5);
  // ponytail: the full-size image only, no mips -- a mip would blend
  // neighbouring tiles. Zoomed far out a map shimmers; padded tilesets or a
  // per-tile mip chain if that matters.
  let sampled = textureSampleLevel(image, imageSampler, texel / vec2<f32>(textureDimensions(image)), 0.0);
  if (number == 0u || any(cell < vec2<i32>(0)) || any(cell >= map)) { discard; }
  let colour = vec4<f32>(encode(sampled.rgb), sampled.a) * v.color;
  return vec4<f32>(lit(colour.rgb, v.flags, v.world), colour.a);
}
`;

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
    // An emitter's particles are drawn from its own pool (render/particles.js):
    // a run of no slots, holding its place in the order.
    if (item.emitter) {
      runs.push({ texture: null, blend: 'emitter', first: entries.length, count: 0, tilemap: null, emitter: item.entity });
      continue;
    }
    const kind = item.source.kind;
    const tilemap = kind === 'tilemap' ? item.source : null;
    const boxes = item.text ? item.source.boxes : [null];
    // Shapes and paths read no image, and draw together.
    const drawn = kind === 'shape' || kind === 'path';
    const texture = item.text ? item.source.font.texture : tilemap ? tilemap.tileset : drawn ? null : item.source.texture;
    const additive = !item.text && item.source.blend === 'additive';
    const blend = tilemap ? 'tilemap' : drawn ? (additive ? 'shapeAdditive' : 'shape') : additive ? 'additive' : 'alpha';
    const first = packed;
    if (kind === 'path') {
      points.set(item.source.points, packed * 2);
      packed += item.source.points.length / 2;
    }
    for (const box of boxes) {
      const last = runs[runs.length - 1];
      // Each tilemap has its own tiles to bind, so is a run of its own.
      if (last && last.texture === texture && last.blend === blend && tilemap === null) last.count++;
      else runs.push({ texture, blend, first: entries.length, count: 1, tilemap });
      entries.push([item.source, box, item.entity, first]);
    }
  }
  return { entries, runs, points };
}

/**
 * One sprite's or glyph's slot, at float offset `o`: placed, turned and
 * scaled by its node, as a canvas transform would. A negative scale on the
 * node mirrors it -- the usual way to turn a character round. `firstPoint` is
 * where a path's points start in the packed list order2D makes.
 */
export function write2D(out, o, world, entity, source, box, firstPoint = 0) {
  const m = handleIndex(entity) * 16;
  const glyph = box !== null;
  const tilemap = source.kind === 'tilemap';
  const shape = source.kind === 'shape';
  const path = source.kind === 'path';
  const lx = Math.hypot(world[m], world[m + 1]);
  const ly = Math.hypot(world[m + 4], world[m + 5]);
  const mirrored = world[m] * world[m + 5] - world[m + 1] * world[m + 4] < 0;
  // A mirrored node's x axis points the flipped way, so its turn is read off
  // the y axis, which the mirror leaves alone.
  const angle = (mirrored ? Math.atan2(-world[m + 4], world[m + 5]) : Math.atan2(world[m + 1], world[m]))
    + (glyph || tilemap || shape || path ? 0 : source.rotation);
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
    const [x0, y0, x1, y1] = source.bounds;
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
    rect = source.font.metrics.glyphs.get(box.char).rect;
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
    | (path ? FLAG_PATH | (source.closed ? FLAG_CLOSED : 0) : 0) | (source.lit ? FLAG_LIT : 0);
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
    out[o + 18] = source.radius * s;
    out[o + 19] = source.strokeWidth * s;
  } else if (path) {
    // In the node's own units, as its points are; the pixel size in them
    // takes the node's scale, the lesser axis's.
    out[o + 16] = firstPoint;
    out[o + 17] = source.points.length / 2;
    out[o + 18] = source.strokeWidth;
    out[o + 19] = Math.min(lx, ly);
  } else if (tilemap) {
    const [tw, th] = source.tileSize;
    out[o + 16] = tw;
    out[o + 17] = th;
    // The tileset's columns: as many tiles and gaps as fit inside its margin.
    out[o + 18] = Math.floor((source.tileset.width - 2 * source.margin + source.spacing) / (tw + source.spacing));
    out[o + 19] = 0;
  } else {
    out[o + 16] = glyph ? 0 : source.cutoff;
    out[o + 17] = out[o + 18] = out[o + 19] = 0;
  }
  // The outline's colour; a tilemap's margin and spacing.
  for (let c = 0; c < 4; c++) out[o + 20 + c] = shape || path ? source.stroke[c] : 0;
  if (tilemap) {
    out[o + 20] = source.margin;
    out[o + 21] = source.spacing;
  }
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
    const descriptor = (blend) => ({
      label: `view2d:${blend}`,
      layout,
      shader,
      targets: [{
        format: rhi.surfaceFormat,
        blend: blend === 'additive'
          ? { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' } }
          : { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } },
      }],
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depth: null,
      constants: { ADDITIVE: blend === 'additive' ? 1 : 0 },
    });
    const descriptors = {
      alpha: descriptor('alpha'),
      additive: descriptor('additive'),
      tilemap: {
        ...descriptor('alpha'), label: 'view2d:tilemap', fragmentEntry: 'fsTilemap', constants: {},
        layout: createPipelineLayout(device, { 0: viewLayout, 1: tilemapLayout }, 'view2d-tilemap'),
      },
    };
    const shapeLayout = createPipelineLayout(device, { 0: viewLayout }, 'view2d-shape');
    descriptors.shape = { ...descriptor('alpha'), label: 'view2d:shape', fragmentEntry: 'fsShape', layout: shapeLayout };
    descriptors.shapeAdditive = { ...descriptor('additive'), label: 'view2d:shape-additive', fragmentEntry: 'fsShape', layout: shapeLayout };
    await pipelines.warm(Object.values(descriptors));
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

  /**
   * Bring the slots up to date with the scene: rebuilt when the order may
   * have changed, otherwise only what moved or changed. \`moved\` is the
   * transform store's per-node flags for this frame, or null if nothing moved.
   */
  prepare(scene, camera, width, height, moved) {
    const world = scene.transforms.world;
    const rebuild = scene !== this._scene || scene.spriteOrder !== this._order;
    if (rebuild) {
      const { entries, runs, points } = order2D(scene);
      this._grow(entries.length);
      this._upload('points', points);
      this._slots.clear();
      entries.forEach(([source, box, entity, firstPoint], slot) => {
        write2D(this._data, slot * SPRITE2D_FLOATS, world, entity, source, box, firstPoint);
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
      let lo = Infinity, hi = -1;
      const rewrite = (entity) => {
        const span = this._slots.get(entity);
        if (span === undefined) return;
        for (let slot = span[0]; slot < span[0] + span[1]; slot++) {
          const [source, box, , firstPoint] = this._entries[slot];
          write2D(this._data, slot * SPRITE2D_FLOATS, world, entity, source, box, firstPoint);
        }
        lo = Math.min(lo, span[0]);
        hi = Math.max(hi, span[0] + span[1] - 1);
      };
      for (const entity of scene.spritesChanged) rewrite(entity);
      if (moved !== null) {
        for (const entity of this._slots.keys()) if (moved[handleIndex(entity)] === 1) rewrite(entity);
      }
      this.written = hi >= lo ? hi - lo + 1 : 0;
      if (this.written > 0) {
        this.rhi.queue.writeBuffer(this._buffer, lo * SPRITE2D_BYTES, this._data, lo * SPRITE2D_FLOATS, this.written * SPRITE2D_FLOATS);
      }
    }
    scene.spritesChanged.clear();
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
      this._data = new Float32Array(grownCapacity(this._data.length / SPRITE2D_FLOATS, count) * SPRITE2D_FLOATS);
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
    for (const { texture, blend, first, count, tilemap, emitter } of this._runs) {
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
        gpu.group ??= this.rhi.device.createBindGroup({
          label: 'view2d-tilemap', layout: this._tilemapLayout,
          entries: [
            { binding: 0, resource: texture.view },
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
      let imageGroup = this._imageGroups.get(texture);
      if (!imageGroup) {
        imageGroup = this.rhi.device.createBindGroup({
          label: 'view2d-image', layout: this._imageLayout,
          entries: [
            { binding: 0, resource: texture.view },
            { binding: 1, resource: texture.pixelated ? pixelatedSampler(this.rhi) : clampSampler(this.rhi) },
          ],
        });
        this._imageGroups.set(texture, imageGroup);
      }
      pass.setBindGroup(1, imageGroup);
      pass.draw(6, count, 0, first);
    }
  }

  destroy() {
    for (const gpu of this._tilemaps.values()) gpu.texture.destroy();
    for (const store of Object.values(this._stores)) store?.buffer.destroy();
    this._buffer?.destroy();
    this._uniform.destroy();
  }
}
