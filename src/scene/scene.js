// Scene: entities, transforms, and the renderable list.
//
// Deliberately GPU-free. It stores references to primitives that already live
// on the GPU, but it never calls a WebGPU function, which is what keeps it
// testable under Node and what keeps "what is in the world" separate from "how
// it gets drawn".
//
// Renderables are stored as SoA columns because the renderer walks all of them
// every frame to update bounds and cull. The Node objects users hold are
// cursors over this (see node.js), not entries in it.

import { assertFinite } from '../core/assert.js';
import { HandleAllocator, handleIndex, NULL_HANDLE } from '../core/handle.js';
import { TransformStore } from './transform.js';
import { Node } from './node.js';
import { Camera } from './camera.js';
import {
  updateWorldBounds, unionWorldBounds, updateSkinBounds, applySkinBounds, applyMorphBounds, BoxList,
} from './bounds.js';
import { aabbRayDistance, aabbTransform, aabbUnion, rayTriangleDistance } from '../core/math/aabb.js';
import { AnimationPlayer } from './animation.js';
import { unboundedLightRadius, uvTransformRows } from './gltf/parse.js';
import { layoutText, SPREAD } from '../render/text.js';
import { hypot3, vec3Create, vec3TransformMat4, vec3TransformMat4Dir } from '../core/math/vec3.js';
import {
  mat4Create, mat4Copy, mat4Invert, mat4Multiply, mat4FromQuatPosScale, mat4Decompose,
} from '../core/math/mat4.js';
import { quatCreate, quatLookAlong } from '../core/math/quat.js';
import { grownCapacity, growArray } from '../core/grow.js';

const DEFAULT_CAPACITY = 4096;

/**
 * Identity for a scene, so anything caching work derived from one can tell
 * WHICH one it cached. `revision` cannot do that job: it counts changes within
 * a scene and starts at zero in every scene, so two of them are equal almost
 * immediately and mean entirely different things.
 */
let nextSceneId = 1;

/**
 * Where every scene's revisions come from: one counter for the page, so no
 * two scenes ever report the same revision. Per-scene counters were equal
 * almost at once, and each cache that compared revision alone had to learn to
 * compare the id as well -- the renderer's batch order and scene bounds never
 * did, so rendering a second scene kept the first one's draw list.
 */
let nextRevision = 1;

/**
 * An emitter's options, checked, with each range as [min, max]; see
 * Scene.addEmitter. `owed` is particles owed but not yet born, `time` the
 * seconds not yet simulated, `seed` what makes its randomness its own.
 */
export function emitterRecord(options = {}) {
  const {
    rate = 0, lifetime, size, sizeEnd = size, speed = 0, direction = [0, 1, 0], spread = 0, radius = 0,
    acceleration = [0, 0, 0], drag = 0, color = [1, 1, 1, 1], colorEnd, texture = null, blend = 'additive', layer = 0,
  } = options;
  if (!Number.isFinite(layer)) throw new Error(`addEmitter: layer must be a finite number, got ${layer}`);
  // [x, y] for a 2D view, as node.setPosition(x, y) takes.
  const flat = (v) => (v?.length === 2 ? [v[0], v[1], 0] : v);
  const range = (v, what, positive = false) => {
    const r = typeof v === 'number' ? [v, v] : v;
    if (!(r?.length === 2 && r.every(Number.isFinite) && r[0] <= r[1] && r[0] >= 0 && (!positive || r[0] > 0))) {
      throw new Error(`addEmitter: ${what} must be ${positive ? 'a positive' : 'a non-negative'} number or [min, max], got ${v}`);
    }
    return Float32Array.from(r);
  };
  const numbers = (v, n, what) => {
    if (!(v?.length === n && [...v].every(Number.isFinite))) throw new Error(`addEmitter: ${what} must be ${n} finite numbers, got ${v}`);
    return Float32Array.from(v);
  };
  if (!(rate >= 0 && Number.isFinite(rate))) throw new Error(`addEmitter: rate must be 0 or more, got ${rate}`);
  if (lifetime === undefined) throw new Error('addEmitter: lifetime is required: how long a particle lives, in seconds');
  if (size === undefined) throw new Error('addEmitter: size is required: how wide a particle is, in world units');
  // A size and a sizeEnd, as a colour and a colorEnd: [a, b] elsewhere is a width and height.
  if (typeof size !== 'number') throw new Error(`addEmitter: size is one number; sizeEnd is the size at death. Got ${size}`);
  const sizes = [size, sizeEnd];
  if (!sizes.every((x) => Number.isFinite(x) && x >= 0)) {
    throw new Error(`addEmitter: size and sizeEnd must be 0 or more, got ${size} and ${sizeEnd}`);
  }
  const dir = numbers(flat(direction), 3, 'direction');
  if (Math.hypot(...dir) === 0) throw new Error('addEmitter: direction must not be zero');
  if (!(spread >= 0 && spread <= Math.PI)) throw new Error(`addEmitter: spread must be between 0 and pi, got ${spread}`);
  if (!(radius >= 0 && Number.isFinite(radius))) throw new Error(`addEmitter: radius must be 0 or more, got ${radius}`);
  if (!(drag >= 0 && Number.isFinite(drag))) throw new Error(`addEmitter: drag must be 0 or more, got ${drag}`);
  if (texture !== null && !(texture?.view && texture.width > 0)) {
    throw new Error('addEmitter: texture must be one engine.loadTexture returned');
  }
  if (!['additive', 'alpha'].includes(blend)) throw new Error(`addEmitter: blend is 'additive' or 'alpha', got ${blend}`);
  const start = numbers(color, 4, 'color');
  return {
    options: { ...options },
    rate,
    lifetime: range(lifetime, 'lifetime', true),
    size: Float32Array.from(sizes),
    speed: range(speed, 'speed'),
    direction: dir,
    spread,
    radius,
    acceleration: numbers(flat(acceleration), 3, 'acceleration'),
    drag,
    color: start,
    colorEnd: colorEnd === undefined ? start : numbers(colorEnd, 4, 'colorEnd'),
    texture,
    blend,
    /** For a 2D view: drawn in painter's order with sprites, as their layer is. */
    layer,
    owed: 0,
    // Of what is owed, how much was burst rather than accrued by the rate.
    burstOwed: 0,
    time: 0,
    seed: (nextEmitterSeed++ * 0x9e3779b9) >>> 0,
  };
}
let nextEmitterSeed = 1;

/** A decal's options, checked; see Scene.addDecal. */
export function decalRecord({ texture, size, color = [1, 1, 1, 1] } = {}) {
  if (!(texture?.view && texture.width > 0 && texture.height > 0)) {
    throw new Error('addDecal: texture must be one engine.loadTexture returned');
  }
  if (!(size?.length === 3 && [...size].every((v) => Number.isFinite(v) && v > 0))) {
    throw new Error(`addDecal: size must be [width, height, depth], all positive, got ${size}`);
  }
  if (!(color?.length === 4 && [...color].every(Number.isFinite))) {
    throw new Error(`addDecal: color must be 4 finite numbers, got ${color}`);
  }
  return { texture, size: Float32Array.from(size), color: Float32Array.from(color) };
}

/**
 * A text's options, checked, and its glyphs laid out; see Scene.addText.
 * Rasterises any glyphs its font does not have yet.
 */
export function textRecord(options = {}) {
  const {
    font, text = '', size, color = [1, 1, 1, 1], align = 'left', pivot = [0.5, 0.5], lineHeight,
    facing = 'camera', pixels = false, layer = 0, width = Infinity, lit = false, blend = 'alpha',
    stroke = [0, 0, 0, 1], strokeWidth = 0,
  } = options;
  if (options.anchor !== undefined) throw new Error('addText: anchor is now pivot, and [0, 0] is the block\'s top-left, as a sprite\'s is');
  if (!(width > 0)) throw new Error(`addText: width must be positive, got ${width}`);
  if (!(font?.metrics && typeof font.ensure === 'function')) throw new Error('addText: font must be one engine.loadFont returned');
  if (!(size > 0 && Number.isFinite(size))) throw new Error(`addText: size must be positive, got ${size}`);
  if (!(color?.length === 4 && [...color].every(Number.isFinite))) throw new Error(`addText: color must be 4 finite numbers, got ${color}`);
  if (!['camera', 'upright', 'plane'].includes(facing)) throw new Error(`addText: facing is 'camera', 'upright' or 'plane', got ${facing}`);
  if (!(pivot?.length === 2 && [...pivot].every(Number.isFinite))) throw new Error(`addText: pivot must be [x, y], got ${pivot}`);
  if (!['left', 'center', 'right'].includes(align)) throw new Error(`addText: align is 'left', 'center' or 'right', got ${align}`);
  if (!Number.isFinite(layer)) throw new Error(`addText: layer must be a finite number, got ${layer}`);
  if (!BLENDS.includes(blend)) throw new Error(`addText: blend is ${BLEND_NAMES}, got ${blend}`);
  if (!(stroke?.length === 4 && [...stroke].every(Number.isFinite))) throw new Error(`addText: stroke must be 4 finite numbers, got ${stroke}`);
  // The outline is drawn in the distance field's reach past the edge, which
  // is a few texels of the font's raster: so much of an em, and no more.
  const reach = (SPREAD - 1) / (font.size ?? SPREAD - 1) * size;
  if (!(strokeWidth >= 0 && strokeWidth <= reach)) {
    throw new Error(`addText: strokeWidth must be 0 to ${+reach.toFixed(3)} for this font at this size, got ${strokeWidth}; load the font larger for a wider one`);
  }
  font.ensure(String(text));
  // In ems, as the layout works; `width` is given in the text's own units, as
  // `size` is. The layout measures y up, so the pivot's top-left is its [0, 1].
  const { boxes, block } = layoutText(text, font.metrics, { align, lineHeight, anchor: [pivot[0], 1 - pivot[1]], width: width / size });
  return {
    options: { ...options },
    font,
    text: String(text),
    size,
    color: Float32Array.from(color),
    facing,
    pixels: pixels === true,
    layer,
    blend,
    lit: lit === true,
    stroke: Float32Array.from(stroke),
    strokeWidth,
    /** Where the outline ends, in the glyph's distance field: 0.5 at its edge, less further out. */
    strokeEdge: 0.5 - strokeWidth / size * (font.size ?? 0) / (2 * SPREAD),
    boxes,
    /** The whole block, [left, bottom, right, top] in ems, y up: what picking hits. */
    block,
  };
}

const SPRITE_FACINGS = ['camera', 'upright', 'plane'];
/**
 * How anything drawn in a scene meets what is under it: over it ('alpha'),
 * adding light ('additive'), darkening it ('multiply': white changes nothing)
 * or lightening it ('screen': black changes nothing). A sprite can also be a
 * 'cutout': each pixel drawn fully or not at all.
 */
export const BLENDS = ['alpha', 'additive', 'multiply', 'screen'];
const BLEND_NAMES = "'alpha', 'additive', 'multiply' or 'screen'";
const SPRITE_BLENDS = [...BLENDS, 'cutout'];

/**
 * A sprite's options, checked and filled in; see Scene.addSprite. `sizeGiven`
 * remembers whether the size was asked for or derived from the texture, so a
 * new texture re-derives a derived one.
 */
export function spriteRecord({
  texture, size, color = [1, 1, 1, 1], rect = [0, 0, 1, 1], pivot = [0.5, 0.5], angle = 0,
  facing = 'camera', blend = 'alpha', cutoff = 0.5, pixels = false, sizeGiven = size !== undefined,
  layer = 0, animation = null, frame = 0, time = 0, lit = false, rotation,
} = {}) {
  if (rotation !== undefined) throw new Error('addSprite: rotation is now angle, in the same radians, as node.setAngle takes');
  if (!(texture?.view && texture.width > 0 && texture.height > 0)) {
    throw new Error('addSprite: texture must be one engine.loadTexture returned');
  }
  const numbers = (v, n, what) => {
    if (!(v?.length === n && [...v].every(Number.isFinite))) {
      throw new Error(`addSprite: ${what} must be ${n} finite numbers, got ${v}`);
    }
    return Float32Array.from(v);
  };
  let played = null;
  if (animation !== null) {
    const { frames, fps = 12, loop = true } = animation;
    if (!(Array.isArray(frames) && frames.length > 0)) throw new Error('addSprite: animation.frames must be a list of rects');
    if (!(fps > 0 && Number.isFinite(fps))) throw new Error(`addSprite: animation.fps must be positive, got ${fps}`);
    played = { frames: frames.map((f) => numbers(f, 4, 'an animation frame')), fps, loop: loop !== false };
  }
  // With no size, the shape of what it shows -- its first frame or its rect,
  // not the whole sheet: in pixels, or one unit wide at that aspect.
  const shown = played?.frames[0] ?? numbers(rect, 4, 'rect');
  const fw = texture.width * Math.abs(shown[2] - shown[0]), fh = texture.height * Math.abs(shown[3] - shown[1]);
  const derived = pixels ? [fw, fh] : [1, fh / fw];
  const sized = numbers(sizeGiven ? size : derived, 2, 'size');
  if (!(sized[0] > 0 && sized[1] > 0)) throw new Error('addSprite: size must be positive');
  if (!SPRITE_FACINGS.includes(facing)) throw new Error(`addSprite: facing is 'camera', 'upright' or 'plane', got ${facing}`);
  if (!SPRITE_BLENDS.includes(blend)) throw new Error(`addSprite: blend is 'alpha', 'additive', 'multiply', 'screen' or 'cutout', got ${blend}`);
  if (!(cutoff >= 0 && cutoff <= 1)) throw new Error(`addSprite: cutoff must be between 0 and 1, got ${cutoff}`);
  if (!Number.isFinite(angle)) throw new Error(`addSprite: angle must be a finite number, got ${angle}`);
  if (!Number.isFinite(layer)) throw new Error(`addSprite: layer must be a finite number, got ${layer}`);
  return {
    texture,
    size: sized,
    sizeGiven,
    color: numbers(color, 4, 'color'),
    rect: numbers(rect, 4, 'rect'),
    pivot: numbers(pivot, 2, 'pivot'),
    angle,
    facing,
    blend,
    cutoff,
    pixels: pixels === true,
    layer,
    lit: lit === true,
    animation: played,
    /** The frame shown, and how long the animation has run: see Scene._advanceAnimations. */
    frame: played === null ? 0 : Math.min(frame, played.frames.length - 1),
    time,
  };
}

/**
 * Check what a light is given, naming the call: only the fields present, so
 * setLight checks just its changes. `cone` is the inner and outer angle the
 * light will have once the changes land.
 */
function checkLight(caller, { color, intensity, radius, size }, cone) {
  if (color !== undefined && !(color?.length === 3 && [...color].every((v) => Number.isFinite(v) && v >= 0))) {
    throw new Error(`${caller}: color must be 3 finite numbers, 0 or more, got ${color}`);
  }
  if (intensity !== undefined && !(intensity >= 0 && Number.isFinite(intensity))) {
    throw new Error(`${caller}: intensity must be 0 or more, got ${intensity}`);
  }
  if (radius !== undefined && !(radius > 0 && Number.isFinite(radius))) throw new Error(`${caller}: radius must be positive, got ${radius}`);
  if (size !== undefined && !(size >= 0 && Number.isFinite(size))) throw new Error(`${caller}: size must be 0 or more, got ${size}`);
  if (cone && !(cone[0] >= 0 && cone[0] <= cone[1] && cone[1] <= Math.PI / 2)) {
    throw new Error(`${caller}: angles need 0 <= innerAngle <= outerAngle <= PI/2, got ${cone[0]} and ${cone[1]}`);
  }
}

/**
 * A set call's check, its errors named for that call: the options are
 * checked by the add call's code, whose errors say addX.
 */
function checked(caller, make) {
  try {
    return make();
  } catch (error) {
    error.message = error.message.replace(/^add[A-Za-z]+:/, `${caller}:`);
    throw error;
  }
}

/** The engine's own bookkeeping in a record, which no option names. */
const BOOKKEEPING = new Set(['added', 'owed', 'seed', 'dirty', 'options', 'boxes', 'block', 'kind', 'sizeGiven', 'placed', 'strokeEdge']);

/**
 * A kind's options for a node, or null if it has none: a copy all the way
 * down, so changing it changes nothing -- setX does that. A kind that keeps
 * the options it was given (text, emitters) returns those, so each reads back
 * as addX took it.
 */
function copyOf(records, node) {
  const record = records.get(node.entity);
  if (record === undefined) return null;
  const source = record.options ?? record;
  const copy = {};
  for (const [key, value] of Object.entries(source)) {
    if (source === record && BOOKKEEPING.has(key)) continue;
    copy[key] = ArrayBuffer.isView(value) ? value.slice() : Array.isArray(value) ? structuredClone(value) : value;
  }
  return copy;
}

/** A path's options, checked and filled in; see Scene.addPath. */
export function pathRecord({
  points, closed = true, color = [1, 1, 1, 1], stroke = [0, 0, 0, 1], strokeWidth = 0,
  layer = 0, blend = 'alpha', lit = false,
} = {}) {
  if (!(Array.isArray(points) || ArrayBuffer.isView(points))) throw new Error('addPath: points must be a list of [x, y]');
  const flat = new Float32Array(points.length * 2);
  points.forEach((p, i) => {
    if (!(p?.length === 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]))) throw new Error(`addPath: point ${i} must be [x, y], got ${p}`);
    flat[i * 2] = p[0];
    flat[i * 2 + 1] = p[1];
  });
  if (points.length < 2) throw new Error(`addPath: a path needs at least 2 points, got ${points.length}`);
  const numbers = (v, n, what) => {
    if (!(v?.length === n && [...v].every(Number.isFinite))) throw new Error(`addPath: ${what} must be ${n} finite numbers, got ${v}`);
    return Float32Array.from(v);
  };
  if (!(strokeWidth >= 0 && Number.isFinite(strokeWidth))) throw new Error(`addPath: strokeWidth must be 0 or more, got ${strokeWidth}`);
  if (!Number.isFinite(layer)) throw new Error(`addPath: layer must be a finite number, got ${layer}`);
  if (!BLENDS.includes(blend)) throw new Error(`addPath: blend is ${BLEND_NAMES}, got ${blend}`);
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];
  for (let i = 0; i < flat.length; i += 2) {
    bounds[0] = Math.min(bounds[0], flat[i]);
    bounds[1] = Math.min(bounds[1], flat[i + 1]);
    bounds[2] = Math.max(bounds[2], flat[i]);
    bounds[3] = Math.max(bounds[3], flat[i + 1]);
  }
  return {
    kind: 'path',
    options: { points: points.map((p) => [p[0], p[1]]), closed, color, stroke, strokeWidth, layer, blend, lit },
    points: flat,
    closed: closed !== false,
    color: numbers(color, 4, 'color'),
    stroke: numbers(stroke, 4, 'stroke'),
    strokeWidth,
    layer,
    blend,
    lit: lit === true,
    /** [minX, minY, maxX, maxY] of the points, in the node's units. */
    bounds,
  };
}

/** A shape's options, checked and filled in; see Scene.addShape. */
/** A shape's corner radius as drawn: its own, fitted inside half the shorter side. */
export function shapeRadius(record) {
  return Math.min(record.radius, record.size[0] / 2, record.size[1] / 2);
}

export function shapeRecord({
  shape = 'rect', size, radius = 0, color = [1, 1, 1, 1], stroke = [0, 0, 0, 1], strokeWidth = 0,
  pivot = [0.5, 0.5], layer = 0, blend = 'alpha', lit = false,
} = {}) {
  const numbers = (v, n, what) => {
    if (!(v?.length === n && [...v].every(Number.isFinite))) throw new Error(`addShape: ${what} must be ${n} finite numbers, got ${v}`);
    return Float32Array.from(v);
  };
  if (shape !== 'rect' && shape !== 'ellipse') throw new Error(`addShape: shape is 'rect' or 'ellipse', got ${shape}`);
  const sized = numbers(size, 2, 'size');
  if (!(sized[0] > 0 && sized[1] > 0)) throw new Error('addShape: size must be positive');
  if (!(radius >= 0 && Number.isFinite(radius))) throw new Error(`addShape: radius must be 0 or more, got ${radius}`);
  if (!(strokeWidth >= 0 && Number.isFinite(strokeWidth))) throw new Error(`addShape: strokeWidth must be 0 or more, got ${strokeWidth}`);
  if (!Number.isFinite(layer)) throw new Error(`addShape: layer must be a finite number, got ${layer}`);
  if (!BLENDS.includes(blend)) throw new Error(`addShape: blend is ${BLEND_NAMES}, got ${blend}`);
  return {
    kind: 'shape',
    shape,
    size: sized,
    // As asked: past half the shorter side a corner has nowhere to go, so it
    // is fitted where it is drawn and picked (shapeRadius), and a shape that
    // shrinks and grows back gets its corner back.
    radius,
    color: numbers(color, 4, 'color'),
    stroke: numbers(stroke, 4, 'stroke'),
    strokeWidth,
    pivot: numbers(pivot, 2, 'pivot'),
    layer,
    blend,
    lit: lit === true,
  };
}

/** A tilemap's options, checked and filled in; see Scene.addTilemap. */
/** How many tiles a tilemap's tileset holds, by its grid. */
function tileCount({ tileset, tileSize: [tw, th], margin, spacing }) {
  const across = Math.floor((tileset.width - 2 * margin + spacing) / (tw + spacing));
  const down = Math.floor((tileset.height - 2 * margin + spacing) / (th + spacing));
  return across * down;
}

/** Every id a whole 32-bit number whose tile, flip bits aside, is 0 or in the tileset. */
function checkTiles(caller, tiles, count) {
  for (let i = 0; i < tiles.length; i++) {
    const id = tiles[i];
    if (!(Number.isInteger(id) && id >= 0 && id <= 0xffffffff && (id & 0x1fffffff) <= count)) {
      throw new Error(`${caller}: tile ${id} is not in the tileset, which holds ${count} (ids start at 1; 0 is empty)`);
    }
    // Flip bits on no tile would be a second way to write empty.
    if (id !== 0 && (id & 0x1fffffff) === 0) throw new Error(`${caller}: tile ${id} flips no tile; an empty tile is 0`);
  }
}

export function tilemapRecord({
  tileset, tileSize, columns, rows, tiles = null, layer = 0, color = [1, 1, 1, 1], pivot = [0, 0], lit = false,
  margin = 0, spacing = 0, blend = 'alpha',
} = {}) {
  if (!(tileset?.view && tileset.width > 0 && tileset.height > 0)) {
    throw new Error('addTilemap: tileset must be one engine.loadTexture returned');
  }
  if (!(Number.isInteger(margin) && Number.isInteger(spacing) && margin >= 0 && spacing >= 0)) {
    throw new Error(`addTilemap: margin and spacing must be whole texels, 0 or more, got ${margin} and ${spacing}`);
  }
  const [tw, th] = tileSize ?? [];
  if (!(Number.isInteger(tw) && Number.isInteger(th) && tw > 0 && th > 0
    && tw + 2 * margin <= tileset.width && th + 2 * margin <= tileset.height)) {
    throw new Error(`addTilemap: tileSize must be whole texels inside the ${tileset.width} x ${tileset.height} tileset, got ${tileSize}`);
  }
  if (!(Number.isInteger(columns) && Number.isInteger(rows) && columns > 0 && rows > 0)) {
    throw new Error(`addTilemap: columns and rows must be whole numbers above zero, got ${columns} x ${rows}`);
  }
  if (tiles !== null && tiles.length !== columns * rows) {
    throw new Error(`addTilemap: tiles holds ${tiles.length} ids; a ${columns} x ${rows} map needs ${columns * rows}`);
  }
  if (tiles !== null) checkTiles('addTilemap', tiles, tileCount({ tileset, tileSize: [tw, th], margin, spacing }));
  if (!Number.isFinite(layer)) throw new Error(`addTilemap: layer must be a finite number, got ${layer}`);
  if (!BLENDS.includes(blend)) throw new Error(`addTilemap: blend is ${BLEND_NAMES}, got ${blend}`);
  const numbers = (v, n, what) => {
    if (!(v?.length === n && [...v].every(Number.isFinite))) throw new Error(`addTilemap: ${what} must be ${n} finite numbers, got ${v}`);
    return Float32Array.from(v);
  };
  return {
    kind: 'tilemap',
    tileset,
    tileSize: [tw, th],
    margin,
    spacing,
    columns,
    rows,
    tiles: tiles === null ? new Uint32Array(columns * rows) : Uint32Array.from(tiles),
    layer,
    color: numbers(color, 4, 'color'),
    pivot: numbers(pivot, 2, 'pivot'),
    blend,
    lit: lit === true,
    /** [x0, y0, x1, y1] in tiles changed since a 2D view last uploaded them, or null. */
    dirty: null,
  };
}

/**
 * How a 2D view reads node `entity`'s world transform: its scale along each
 * axis, whether it is mirrored, and its turn, into `out` as [lx, ly, mirrored
 * (1 or 0), angle]. A mirrored node's x axis points the flipped way, so its
 * turn is read off the y axis, which the mirror leaves alone.
 */
export function frame2D(world, entity, out) {
  const m = handleIndex(entity) * 16;
  const mirrored = world[m] * world[m + 5] - world[m + 1] * world[m + 4] < 0;
  out[0] = hypot3(world[m], world[m + 1], 0);
  out[1] = hypot3(world[m + 4], world[m + 5], 0);
  out[2] = mirrored ? 1 : 0;
  out[3] = mirrored ? Math.atan2(-world[m + 4], world[m + 5]) : Math.atan2(world[m + 1], world[m]);
  return out;
}

/** Whether a sprite's rect reaches past its image, so the image repeats. */
export function repeats(record) {
  const r = record.rect;
  return record.animation === null
    && (Math.min(r[0], r[2]) < 0 || Math.min(r[1], r[3]) < 0 || Math.max(r[0], r[2]) > 1 || Math.max(r[1], r[3]) > 1);
}

/** The part of its texture a sprite shows now: its animation's frame, or its rect. */
export function spriteRect(record) {
  return record.animation === null ? record.rect : record.animation.frames[record.frame];
}

/**
 * The frames of a sprite sheet laid out in a grid, in reading order: left to
 * right, then top to bottom. For addSprite's `animation.frames`, or to pick
 * one as its `rect`.
 *
 *   const walk = spriteSheet({ columns: 8, rows: 1 });
 *   scene.addSprite({ texture: hero, animation: { frames: walk, fps: 10 } });
 *
 * `count` stops short of a last row that is not full; `first` skips frames.
 */
export function spriteSheet({ columns, rows = 1, count = columns * rows, first = 0 } = {}) {
  if (!(Number.isInteger(columns) && columns > 0 && Number.isInteger(rows) && rows > 0)) {
    throw new Error(`spriteSheet: columns and rows must be whole numbers above zero, got ${columns} x ${rows}`);
  }
  if (!(Number.isInteger(first) && first >= 0 && Number.isInteger(count) && count > 0 && first + count <= columns * rows)) {
    throw new Error(`spriteSheet: frames ${first} to ${first + count - 1} are not all in a ${columns} x ${rows} sheet`);
  }
  const frames = [];
  for (let k = first; k < first + count; k++) {
    const x = k % columns, y = Math.floor(k / columns);
    frames.push([x / columns, y / rows, (x + 1) / columns, (y + 1) / rows]);
  }
  return frames;
}

/**
 * A reflection probe's description, checked; see Scene.addProbe.
 * Its box and eye are placed around `at`, its node's world position, and
 * again whenever the node moves (Scene._placeProbes).
 */
export function probeRecord({ size, fade = 0, blend, min, max } = {}, at = [0, 0, 0]) {
  if (blend !== undefined) throw new Error('addProbe: blend is now fade, with the same value');
  if (min !== undefined || max !== undefined) {
    throw new Error('addProbe: a probe is a node now -- give its box as size, [width, height, depth], centred on its position');
  }
  if (!(size?.length === 3 && [...size].every((v) => Number.isFinite(v) && v > 0))) {
    throw new Error(`addProbe: size must be [width, height, depth], all positive, got ${size}`);
  }
  if (!(fade >= 0 && Number.isFinite(fade))) throw new Error(`addProbe: fade must be 0 or more, got ${fade}`);
  const half = [0, 1, 2].map((a) => size[a] / 2);
  return {
    size: Float32Array.from(size),
    min: Float32Array.from([0, 1, 2].map((a) => at[a] - half[a])),
    max: Float32Array.from([0, 1, 2].map((a) => at[a] + half[a])),
    position: Float32Array.from(at),
    fade,
    captured: false,
    /** Where its node was when the box was last placed; null to place it again. */
    placed: null,
  };
}

/** The largest finite f32: coverage with no upper bound. */
export const F32_MAX = 3.4028234663852886e38;

/** A mesh's bounding sphere in its node's space: the box's centre and half diagonal. */
function meshSphere(mesh) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const { bounds } of mesh.primitives) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], bounds.min[a]);
      max[a] = Math.max(max[a], bounds.max[a]);
    }
  }
  return [
    (min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2,
    Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2,
  ];
}

/**
 * An LOD level's transform under the finest level's node: the finest's
 * transform undone, then the level's own. Both are in the same parent space,
 * so a level exported where the finest one is -- the usual case -- lands on
 * the identity, and exactly, without a round trip through a matrix.
 */
function relativeTRS(finest, level) {
  const same = (a, b) => a.every((v, k) => v === b[k]);
  if (same(finest.position, level.position) && same(finest.rotation, level.rotation) && same(finest.scale, level.scale)) {
    return { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] };
  }
  const trs = (node) => mat4FromQuatPosScale(mat4Create(), node.rotation, node.position, node.scale);
  const relative = mat4Multiply(mat4Create(), mat4Invert(mat4Create(), trs(finest)), trs(level));
  const position = new Float32Array(3);
  const rotation = new Float32Array(4);
  const scale = new Float32Array(3);
  mat4Decompose(position, rotation, scale, relative);
  return { position, rotation, scale };
}

export class Scene {
  constructor({ capacity = DEFAULT_CAPACITY, renderableCapacity = capacity, lightCapacity = 256 } = {}) {
    /** Unique for the life of the page. See nextSceneId. */
    this.id = nextSceneId++;
    this.capacity = capacity;
    this.entities = new HandleAllocator(capacity);
    this.transforms = new TransformStore(capacity);

    // --- renderable columns -------------------------------------------------
    this.renderableCount = 0;
    this.renderableCapacity = renderableCapacity;
    /**
     * Bumped whenever the set of renderables changes. Batching is O(n log n)
     * and must not run on a scene that is merely moving.
     */
    this.revision = nextRevision++;
    /**
     * The world boxes of renderables removed since the renderer last looked:
     * where a cached shadow map may still show them. An added one needs no
     * entry -- it is new, so it moved, and its box is recorded as any mover's.
     */
    this.removedBoxes = new BoxList();
    this.renderableEntity = new Uint32Array(renderableCapacity);
    /** Which transform slot each renderable reads its world matrix from. */
    this.renderableMatrixSlot = new Uint32Array(renderableCapacity);
    this.renderableMaterial = new Uint16Array(renderableCapacity);
    /** Primitive descriptors: GPU buffers + index count. Held, never called. */
    this.renderablePrimitive = new Array(renderableCapacity);

    /** Index into this.skins, or -1: which palette a renderable reads. */
    this.renderableSkin = new Int32Array(renderableCapacity).fill(-1);

    /** Index into this.morphs, or -1: whose weights deform this renderable. */
    this.renderableMorph = new Int32Array(renderableCapacity).fill(-1);
    /**
     * How far each of this renderable's morph targets reaches, or null.
     *
     * A column rather than a read of renderablePrimitive, because a primitive
     * is the RENDERER's object -- GPU buffers and an index count -- and bounds
     * work is defined by touching scene columns only.
     */
    this.renderableMorphExtent = new Array(renderableCapacity).fill(null);
    /** Last frame's morph padding, so a weight change is detectable at all. */
    this.renderableMorphPad = new Float32Array(renderableCapacity);

    /**
     * Level of detail. The transform slot of the LOD group's node, or -1 for
     * a renderable in no group; that group's bounding sphere in the node's
     * space, centre and radius; and the screen coverage this renderable
     * draws between, [lowest, highest).
     */
    this.renderableLodSlot = new Int32Array(renderableCapacity).fill(-1);
    this.renderableLodSphere = new Float32Array(renderableCapacity * 4);
    this.renderableCoverage = new Float32Array(renderableCapacity * 2);

    this.localMin = new Float32Array(renderableCapacity * 3);
    this.localMax = new Float32Array(renderableCapacity * 3);
    this.worldMin = new Float32Array(renderableCapacity * 3);
    this.worldMax = new Float32Array(renderableCapacity * 3);

    // --- lighting -----------------------------------------------------------
    // Plain mutable fields: they are per-scene data the renderer reads when it
    // is handed this scene, not hidden state a function reaches for.
    /**
     * Every directional light, packed for the GPU: the direction its light
     * travels, then colour times intensity, a vec4 each. The first vec4's w is
     * the light's shadow slot plus one, written by the renderer; zero for none.
     * DERIVED: refreshLights copies them out of the lights' nodes every frame,
     * so read them and never write them. To change a light, change its node.
     */
    this.directionals = new Float32Array(DIRECTIONAL_FLOATS);
    this.directionalCount = 0;
    /** The entity each packed directional light belongs to, in the same order. */
    this.directionalEntity = [];
    this._directional = new Map();   // entity -> { color, intensity }, in the order added
    // See the environment accessors: assigning one is a change a frame must see.
    this._environment = null;

    // --- punctual lights ----------------------------------------------------
    // Packed exactly as the GPU wants them, so uploading is one memcpy rather
    // than a per-light gather. Four vec4s each; see LIGHT_FLOATS below.
    this.lightCount = 0;
    this.lightCapacity = lightCapacity;
    this.lights = new Float32Array(lightCapacity * LIGHT_FLOATS);
    /**
     * The entity each light IS. A light is a scene object: it has a transform,
     * so it can be parented, animated and moved like anything else, and its
     * position and spot direction are READ from that transform every frame
     * rather than written by hand.
     */
    this.lightEntity = new Uint32Array(lightCapacity);
    this._lightOf = new Map();   // entity -> index into the packed arrays

    /** Resolved skin instances: joint ENTITIES plus the bind pose. */
    this.skins = [];
    this._pendingSkins = [];
    /**
     * Morph weights, one array per instanced mesh that has targets.
     *
     * Per (node, mesh), not per primitive: glTF puts the weights on the node,
     * so every primitive of one mesh is deformed by the same set. And not per
     * mesh either -- two nodes instancing one head animate independently,
     * which is the same reason the skin palette and the player are per
     * instance.
     */
    this.morphs = [];
    this._morphOf = new Map();   // entity -> index into this.morphs
    /** Root entity -> AnimationPlayer, for asset instances that have clips. */
    this._players = new Map();
    /**
     * The lights that cast shadows. A choice per light, the same switch for
     * every kind. glTF has no say in it, so the default is the engine's, by
     * cost: a directional light casts, because its shadow is in view almost
     * everywhere and it is one set of cascades; a point or spot light does
     * not, because it is local and a point light is six maps.
     */
    this.shadowCasters = new Set();
    /**
     * The size of each light that has one, by entity: a point or spot light's
     * radius, a directional light's angle across. What makes its shadows soft
     * (PCSS); a light without one casts the plain filtered edge.
     */
    this.lightSizes = new Map();
    /**
     * Reflection probes (render/probes.js): { min, max, position, fade,
     * captured }. engine.captureProbes renders them; the revision
     * tells the renderer the set changed.
     */
    this.reflectionProbes = [];
    this.probeRevision = 0;
    /**
     * Counts changes no transform and no `revision` records: a sprite's,
     * text's, decal's, emitter's, probe's or light's properties. With those,
     * `animating` and `particlesActive`, it is how the engine knows a frame
     * would draw exactly what the last one did, and skips it.
     */
    this.changes = 0;
    /**
     * Bumped when the order sprites and text draw in can change -- one added
     * or removed, or a layer, texture or blend changed -- and a 2D view
     * rebuilds its list. Every other change to one is in spritesChanged, and
     * rewrites that sprite alone.
     */
    this.spriteOrder = 0;
    /**
     * Sprites, text and shapes added and removed since a 2D view last looked:
     * a view appends the one and leaves a hole for the other, where a change
     * to spriteOrder has it rebuild its whole list. layout2D counts both, for
     * picking's cached order.
     */
    this.added2D = new Set();
    this.removed2D = new Set();
    this.layout2D = 0;
    // Counts sprites and texts as they are added: order within a layer.
    this._added = 0;
    /** Sprites and texts whose own data changed since a 2D view last drew, by entity. */
    this.spritesChanged = new Set();
    /** Whether the last _advanceAnimations moved anything. */
    this.animating = false;
    // The particle clock, and until when on it a particle may still be alive.
    this._particleClock = 0;
    this._particlesUntil = 0;
    /** Sprites (see addSprite), by the entity each hangs off. */
    this.sprites = new Map();
    /** Particle emitters (see addEmitter), by the entity each hangs off. */
    this.emitters = new Map();
    /** Gaussian splat captures (see addSplats), by the entity each hangs off. */
    this.splats = new Map();
    /** Decals (see addDecal), by the entity each hangs off, in the order added. */
    this.decals = new Map();
    /** Text (see addText), by the entity each hangs off. */
    this.texts = new Map();
    /** Tilemaps (see addTilemap), by the entity each hangs off. */
    this.tilemaps = new Map();
    /** 2D shapes (see addShape), by the entity each hangs off. */
    this.shapes = new Map();
    /** 2D paths (see addPath), by the entity each hangs off. */
    this.paths = new Map();
    /**
     * Material id -> the importer's record, for materials a clip changed since
     * the renderer last uploaded them. A material belongs to the asset, not to
     * one instance of it, exactly as in glTF: animating one animates it on
     * every copy.
     */
    this.changedMaterials = new Map();

    /**
     * Cameras that came in with assets, in the order they were added. Each
     * already follows its node, so an animated camera plays with the clip.
     * Hand one to engine.run as the camera, or ignore them.
     */
    this.cameras = [];

    // No lights. A scene is lit by its environment until one is added: the
    // directional light a scene used to start with was the procedural sky's
    // sun under another name, which is a special case this no longer has.
  }

  /**
   * Instantiate a loaded asset. Returns the root Node.
   *
   * Synchronous on purpose: every slow step -- parsing, image decode, buffer
   * upload, pipeline compilation -- already happened in engine.load(). Adding
   * to a scene must never be the thing that stalls a frame.
   */
  add(asset, { parent = null } = {}) {
    this.changes++;
    // Its buffers and textures are destroyed; drawing it would fail on the GPU.
    if (asset.unloaded) throw new Error('Scene.add: this asset was unloaded; load it again');
    const created = new Array(asset.nodes.length).fill(NULL_HANDLE);
    const roots = [];
    // Node index -> that node's morph weights, for the animation player. Same
    // shape as `created` and built beside it, because a weights channel names
    // a node exactly the way a translation channel does.
    let weightsOf = null;
    // Light and camera index -> what this instance made of it, for clips that
    // animate them by pointer. One light can sit on several nodes.
    const lightsOf = [];
    const camerasOf = [];

    // Nodes that stand in for another at lower detail (MSFT_lod). They are
    // placed by the node they stand in for, never walked into on their own:
    // a file that also lists one as a child or root would draw it twice.
    const alternates = new Set();
    for (const node of asset.nodes) for (const id of node.lod?.ids ?? []) alternates.add(id);

    // `lod` is the group a node's renderables are a level of, or null; `trs`
    // replaces the node's own transform, for a level placed under the finest.
    const visit = (nodeIndex, parentEntity, lod = null, trs = null) => {
      const node = asset.nodes[nodeIndex];
      // Unconditional, not DEBUG-only. glTF node graphs must be forests, and a
      // node reached twice is a diamond or a cycle. A cycle does not return; a
      // diamond quietly builds the subtree TWICE, and every downstream map --
      // `created`, the animation player's node table, the skin's joint
      // resolution -- keeps only the second copy. The clip then drives one of
      // the two and the other sits frozen, which reads as an asset bug rather
      // than a loader one. One comparison per node is not a reason to ship
      // that in a release build.
      if (created[nodeIndex] !== NULL_HANDLE) {
        throw new Error(
          `Scene.add: node ${nodeIndex} ("${node?.name ?? '?'}") has more than one parent, `
          + 'so the asset graph is not a tree',
        );
      }

      const entity = this.entities.alloc();
      created[nodeIndex] = entity;
      this.transforms.add(entity, {
        position: trs?.position ?? node.position,
        rotation: trs?.rotation ?? node.rotation,
        scale: trs?.scale ?? node.scale,
        parent: parentEntity,
      });

      // The finest level of an LOD group: its renderables draw above the
      // first coverage, and every level measures this node's sphere.
      const coverage = node.lod?.coverage ?? null;
      const group = coverage !== null && node.mesh >= 0
        ? { slot: handleIndex(entity), sphere: meshSphere(asset.meshes[node.mesh]), low: coverage[0], high: F32_MAX }
        : lod;

      if (node.mesh >= 0) {
        // A skin instance per (node, skin), because `created` is this
        // instance's node-to-entity map -- two copies of one character need
        // two palettes, which is the same reason the animation player is per
        // instance. Deferred until after the walk, since a joint node may not
        // have been visited yet.
        // Indexed against this.skins, which accumulates across every add() --
        // the pending list is only this call's tail of it.
        const skinIndex = node.skin >= 0 && asset.skins?.[node.skin] !== undefined
          ? this.skins.length + this._pendingSkins.push({ skin: asset.skins[node.skin], created, owner: entity }) - 1
          : -1;

        // Weights come from the ASSET's node, which the importer already
        // resolved against the mesh's defaults. Copied, because this instance
        // is about to animate them and the asset may be added again.
        let morphIndex = -1;
        if (asset.meshes[node.mesh].targetCount > 0 && node.weights) {
          morphIndex = this.morphs.length;
          // Owned by this node: removed with it (see _dropOwned).
          this.morphs.push({ weights: Float32Array.from(node.weights), owner: entity });
          this._morphOf.set(entity, morphIndex);
          if (weightsOf === null) weightsOf = new Array(asset.nodes.length);
          weightsOf[nodeIndex] = this.morphs[morphIndex].weights;
        }

        for (const primitive of asset.meshes[node.mesh].primitives) {
          // Only a mesh that HAS influences is skinned. A rigged mesh
          // instanced under a node with no skin renders static, which is what
          // the pairing living on the node means.
          // primitive.skinned, not primitive.jointIndices: by the time a
          // primitive reaches the scene it is the renderer's object, which
          // carries GPU buffers rather than the arrays they were built from.
          const skinned = skinIndex >= 0 && primitive.skinned ? skinIndex : -1;
          // A primitive with no targets of its own is never morphed, even on a
          // node that carries weights -- the importer refuses a mesh whose
          // primitives disagree, so in practice this is all or none.
          const morphed = primitive.morphExtent ? morphIndex : -1;
          this._addRenderable(entity, primitive, skinned, morphed, group);
        }
      }

      // A light or camera on a node is the node's, exactly as in glTF: where
      // it is and which way it points are the node's transform. A directional
      // light joins the others; whether it casts is its castShadow, on by default for directional lights.
      const light = node.light >= 0 ? asset.lights?.[node.light] : null;
      if (light) {
        this._attachLight(entity, light);
        (lightsOf[node.light] ??= []).push(entity);
      }

      const spec = node.camera >= 0 ? asset.cameras?.[node.camera] : null;
      if (spec) {
        const camera = cameraFor(spec).follow(new Node(this, entity));
        this.cameras.push(camera);
        (camerasOf[node.camera] ??= []).push(camera);
      }

      for (const child of node.children) if (!alternates.has(child)) visit(child, entity, lod);

      // The lower levels, under this node so they go where it goes. Each is
      // placed where the file puts it -- beside this node, in the same parent
      // space -- which is this node's transform undone, then its own.
      if (group !== lod) {
        node.lod.ids.forEach((id, k) => {
          visit(id, entity, { ...group, low: coverage[k + 1], high: coverage[k] }, relativeTRS(node, asset.nodes[id]));
        });
      }
      return entity;
    };

    const parentEntity = this._parentOf('add', parent);

    // ALL OR NOTHING. A walk that throws -- an asset graph that is not a tree,
    // a skin naming a joint outside the default scene -- used to leave behind
    // everything built before the throw, with no handle to remove it by, and
    // a pending skin that every later add() then tried to resolve and threw
    // on. Now the scene is put back exactly as it was before the error goes on.
    try {
      for (const root of asset.roots) if (!alternates.has(root)) roots.push(visit(root, parentEntity));

      // Joints resolve now, not during the walk: a skin may name a node the
      // walk had not reached yet, and `created` is only complete once it is done.
      for (const pending of this._pendingSkins) {
        const { skin, created: map, owner } = pending;
        const joints = new Uint32Array(skin.joints.length);
        for (let j = 0; j < skin.joints.length; j++) {
          const jointEntity = map[skin.joints[j]];
          if (jointEntity === undefined || jointEntity === NULL_HANDLE) {
            throw new Error(
              `Scene.add: skin "${skin.name}" names node ${skin.joints[j]}, which is not in the ` +
              'asset\'s default scene, so it has no entity to drive it',
            );
          }
          joints[j] = jointEntity;
        }
        this.skins.push({
          joints,
          inverseBind: skin.inverseBind,
          jointRadii: skin.jointRadii,
          // Recomputed each frame from the joints' world positions. A skinned
          // mesh's vertices move without its model matrix moving, so its
          // bounds cannot come from transforming a static box.
          boundsMin: new Float32Array(3),
          boundsMax: new Float32Array(3),
          // The mesh node this palette deforms, which it is removed with.
          owner,
          // Every inverse bind matrix affine, checked at load (gltf/skin.js):
          // the palette may then use the cheaper multiply.
          affine: skin.affine === true,
        });
      }
    } catch (error) {
      this._pendingSkins.length = 0;
      for (const entity of created) {
        if (entity !== NULL_HANDLE && this.entities.alive(entity)) this.remove(new Node(this, entity));
      }
      throw error;
    }
    this._pendingSkins.length = 0;
    // Multi-root assets get a wrapper so the caller always gets one handle back
    // and can move the whole thing with a single setPosition.
    let handle;
    if (roots.length === 1) {
      handle = roots[0];
    } else {
      handle = this.entities.alloc();
      this.transforms.add(handle, { parent: parentEntity });
      for (const root of roots) this.transforms.setParent(root, handle);
    }

    // `created` maps the asset's node indices onto THIS instance's entities,
    // which is the whole reason two copies of one asset can play the same clip
    // at different times. It is kept only when there is something to play.
    if (asset.animations?.length > 0) {
      const player = new AnimationPlayer(asset.animations, created, this.entities, weightsOf, asset.nodes,
        this._propertiesFor(asset, lightsOf, camerasOf));
      // What root motion moves: the handle this instance is placed by.
      player.instance = handle;
      this._players.set(handle, player);
    }

    return new Node(this, handle);
  }

  /** `lod`: { slot, sphere, low, high } for a level of an LOD group; see add(). */
  _addRenderable(entity, primitive, skin = -1, morph = -1, lod = null) {
    if (this.renderableCount >= this.renderableCapacity) {
      this._growRenderables(this.renderableCount + 1);
    }
    const i = this.renderableCount++;
    this.revision = nextRevision++;

    this.renderableEntity[i] = entity;
    this.renderableMatrixSlot[i] = handleIndex(entity);
    this.renderableMaterial[i] = primitive.materialId;
    this.renderablePrimitive[i] = primitive;
    // How many renderables in any scene draw it. engine.unload refuses while
    // this is above zero: freeing buffers a scene still draws is a GPU error,
    // and a reused material id would draw with someone else's surface.
    primitive.instances = (primitive.instances ?? 0) + 1;
    this.renderableSkin[i] = skin;
    this.renderableMorph[i] = morph;
    this.renderableMorphExtent[i] = morph >= 0 ? primitive.morphExtent : null;

    this.renderableLodSlot[i] = lod === null ? -1 : lod.slot;
    if (lod !== null) this.renderableLodSphere.set(lod.sphere, i * 4);
    this.renderableCoverage[i * 2] = lod === null ? 0 : lod.low;
    this.renderableCoverage[i * 2 + 1] = lod === null ? F32_MAX : lod.high;

    this.localMin.set(primitive.bounds.min, i * 3);
    this.localMax.set(primitive.bounds.max, i * 3);
    // Nowhere yet, rather than the box of whatever held this slot last: its
    // first bounds update records the box as it was and as it is, for cached
    // shadow maps, and a stale one there redrew maps the new object is far
    // from. NaN reaches no view and no light; the update replaces it the
    // same frame, as a new node has moved.
    this.worldMin.fill(NaN, i * 3, i * 3 + 3);
    this.worldMax.fill(NaN, i * 3, i * 3 + 3);
    return i;
  }

  _growRenderables(needed) {
    const capacity = grownCapacity(this.renderableCapacity, needed);

    this.renderableEntity = growArray(this.renderableEntity, capacity);
    this.renderableMatrixSlot = growArray(this.renderableMatrixSlot, capacity);
    this.renderableMaterial = growArray(this.renderableMaterial, capacity);
    this.renderablePrimitive.length = capacity;

    this.renderableSkin = growArray(this.renderableSkin, capacity);
    this.renderableMorph = growArray(this.renderableMorph, capacity);
    this.renderableMorphExtent.length = capacity;
    this.renderableMorphPad = growArray(this.renderableMorphPad, capacity);
    this.renderableLodSlot = growArray(this.renderableLodSlot, capacity);
    this.renderableLodSphere = growArray(this.renderableLodSphere, capacity, 4);
    this.renderableCoverage = growArray(this.renderableCoverage, capacity, 2);
    this.localMin = growArray(this.localMin, capacity, 3);
    this.localMax = growArray(this.localMax, capacity, 3);
    // World bounds are recomputed from local every time they are read, so these
    // only need the room, not the contents.
    this.worldMin = growArray(this.worldMin, capacity, 3);
    this.worldMax = growArray(this.worldMax, capacity, 3);

    this.renderableCapacity = capacity;
  }

  /**
   * This entity's morph weights, or null.
   *
   * Handed out LIVE, where a position is not. The rule Node.js states -- no
   * live views, because a write that skips the setter never marks anything
   * dirty -- is about the transform hierarchy, where a stale dirty flag means
   * a stale matrix. Weights have no hierarchy and no flag: they are uploaded
   * whole every frame, so writing one is already the only thing writing one
   * could mean.
   */
  morphWeights(entity) {
    const index = this._morphOf.get(entity);
    return index === undefined ? null : this.morphs[index].weights;
  }

  /**
   * Grow every morphed renderable's world bounds to cover where its weights
   * have put its vertices.
   *
   * Runs after updateWorldBounds and after applySkinBounds -- it reads the
   * boxes both of those wrote. A method rather than a raw call so the renderer
   * and the raycaster cannot pass the ten columns in different orders.
   */
  applyMorphBounds() {
    return applyMorphBounds(
      this.renderableCount, this.renderableMorph, this.renderableSkin,
      this.morphs, this.renderableMorphExtent,
      this.localMin, this.localMax, this.worldMin, this.worldMax,
      this.transforms.world, this.renderableMatrixSlot, this.renderableMorphPad, this.skins,
    );
  }

  /** An empty node, for grouping things you position together. */
  createNode({ parent = null } = {}, caller = 'createNode') {
    const parentEntity = this._parentOf(caller, parent);
    const entity = this.entities.alloc();
    this.transforms.add(entity, { parent: parentEntity });
    return new Node(this, entity);
  }

  /**
   * The entity to hang something off, checked: a removed node's slot is
   * reused by the next one made, so hanging something off it attached it to
   * a stranger -- and removing the stranger took it too.
   */
  _parentOf(caller, parent) {
    if (parent === null || parent === undefined) return NULL_HANDLE;
    if (parent.scene !== this) throw new Error(`${caller}: parent is a node of another scene`);
    if (!parent.alive) throw new Error(`${caller}: parent was removed`);
    return parent.entity;
  }

  node(entity) {
    return new Node(this, entity);
  }

  childrenOf(node) {
    if (!node.alive) return [];
    return this.transforms.childrenOf(handleIndex(node.entity))
      .map((slot) => new Node(this, this.entities.handleAt(slot)));
  }

  /**
   * Remove a node and everything under it.
   *
   * Renderables are swap-removed, so their order changes -- nothing may cache a
   * renderable index across a remove.
   */
  remove(node) {
    this.changes++;
    // A stale handle names a slot something else may own by now. Acting on it
    // erased that other node's transform before failing to free the handle.
    if (!node.alive) return;

    // Everything under it, from the transform hierarchy -- the only record of
    // it, so a child added by createNode({ parent }), add(asset, { parent }) or
    // setParent goes with its parent like any other.
    const doomed = this.transforms.subtree(handleIndex(node.entity))
      .map((slot) => this.entities.handleAt(slot));

    const dying = new Set(doomed);
    for (let i = this.renderableCount - 1; i >= 0; i--) {
      if (!dying.has(this.renderableEntity[i])) continue;
      this.renderablePrimitive[i].instances--;
      this.removedBoxes.push(this.worldMin, this.worldMax, i * 3);

      const last = --this.renderableCount;
      if (i !== last) {
        this.renderableEntity[i] = this.renderableEntity[last];
        this.renderableMatrixSlot[i] = this.renderableMatrixSlot[last];
        this.renderableMaterial[i] = this.renderableMaterial[last];
        this.renderablePrimitive[i] = this.renderablePrimitive[last];
        // Skin and morph move with their renderable. They were left behind,
        // so the survivor took the deleted object's skin palette and morph
        // weights: remove one character and another starts wearing its pose.
        this.renderableSkin[i] = this.renderableSkin[last];
        this.renderableMorph[i] = this.renderableMorph[last];
        this.renderableMorphExtent[i] = this.renderableMorphExtent[last];
        this.renderableMorphPad[i] = this.renderableMorphPad[last];
        this.renderableLodSlot[i] = this.renderableLodSlot[last];
        this.renderableLodSphere.copyWithin(i * 4, last * 4, last * 4 + 4);
        this.renderableCoverage.copyWithin(i * 2, last * 2, last * 2 + 2);
        this.localMin.copyWithin(i * 3, last * 3, last * 3 + 3);
        this.localMax.copyWithin(i * 3, last * 3, last * 3 + 3);
        // World bounds move with their renderable too. Without this the
        // survivor inherits the deleted object's box and keeps it until it
        // happens to move: the GPU culls a visible mesh, and raycast returns
        // the wrong thing.
        this.worldMin.copyWithin(i * 3, last * 3, last * 3 + 3);
        this.worldMax.copyWithin(i * 3, last * 3, last * 3 + 3);
      }
      this.renderablePrimitive[last] = undefined;
      this.renderableMorphExtent[last] = null;
      this.revision = nextRevision++;
    }

    // Imported cameras on a doomed node go with it. A camera the caller made
    // and pointed at the node just stops following on its next update.
    if (this.cameras.length > 0) {
      this.cameras = this.cameras.filter((camera) => !dying.has(camera.following?.entity));
    }

    this._dropOwned(dying);

    // Lights on any doomed entity go too. Before the entities are freed, so
    // the handles are still the ones the map was built from.
    for (const entity of doomed) {
      // A 2D view leaves a hole for a sprite, text or shape; the rest rebuild its list.
      if (this.emitters.has(entity) || this.tilemaps.has(entity) || this.paths.has(entity)) this.spriteOrder++;
      if (this.sprites.has(entity) || this.texts.has(entity) || this.shapes.has(entity)) {
        if (!this.added2D.delete(entity)) this.removed2D.add(entity);
        this.layout2D++;
      }
      this.sprites.delete(entity);
      this.emitters.delete(entity);
      this.splats.delete(entity);
      this.decals.delete(entity);
      this.texts.delete(entity);
      this.tilemaps.delete(entity);
      this.shapes.delete(entity);
      const probe = this.reflectionProbes.findIndex((p) => p.entity === entity);
      if (probe >= 0) {
        this.reflectionProbes.splice(probe, 1);
        this.probeRevision++;
      }
      this.paths.delete(entity);
      const index = this._lightOf.get(entity);
      if (index !== undefined) this._removeLightAt(index);
      this._directional.delete(entity);
      this.shadowCasters.delete(entity);
      this.lightSizes.delete(entity);
    }

    // A scene no 2D view draws never has these read: past a point, a rebuild
    // is cheaper to promise than the list is to keep.
    if (this.removed2D.size > 4096 && this.removed2D.size > this.sprites.size) {
      this.removed2D.clear();
      this.added2D.clear();
      this.spriteOrder++;
    }

    for (const entity of doomed) {
      this.transforms.remove(entity);
      this._players.delete(entity);
      this.entities.free(entity);
    }
  }

  /**
   * Remove the skin palettes and morph weights the dying nodes owned.
   *
   * They used to outlive their asset: a hundred add/remove cycles of a skinned
   * character left a hundred palettes that were still multiplied and uploaded
   * every frame. Swap-compacted like everything else here, and every index
   * that pointed at a moved entry is pointed at its new place. The renderables
   * that used the removed ones are already gone -- remove() drops them first.
   */
  _dropOwned(dying) {
    for (let s = this.skins.length - 1; s >= 0; s--) {
      if (!dying.has(this.skins[s].owner)) continue;
      const last = this.skins.length - 1;
      if (s !== last) {
        this.skins[s] = this.skins[last];
        for (let i = 0; i < this.renderableCount; i++) {
          if (this.renderableSkin[i] === last) this.renderableSkin[i] = s;
        }
      }
      this.skins.pop();
    }

    for (let m = this.morphs.length - 1; m >= 0; m--) {
      const owner = this.morphs[m].owner;
      if (!dying.has(owner)) continue;
      this._morphOf.delete(owner);
      const last = this.morphs.length - 1;
      if (m !== last) {
        this.morphs[m] = this.morphs[last];
        this._morphOf.set(this.morphs[m].owner, m);
        for (let i = 0; i < this.renderableCount; i++) {
          if (this.renderableMorph[i] === last) this.renderableMorph[i] = m;
        }
      }
      this.morphs.pop();
    }
  }

  // --------------------------------------------------------------- sprites

  /**
   * A sprite: a textured quad that turns to face the camera, as a node in the
   * scene -- so it moves, parents and is removed like any other. Returns the
   * Node.
   *
   *   const marker = scene.addSprite({ texture: await engine.loadTexture('pin.png'), parent: npc });
   *   marker.setPosition(0, 2.2, 0);
   *
   *   texture      from engine.loadTexture
   *   size         [width, height], in world units -- or in pixels, with
   *                `pixels: true`, for a marker that stays one size on screen.
   *                One unit wide at the texture's aspect by default, or the
   *                texture's own pixel size.
   *   color        multiplies the texture; linear, and may pass 1 to glow
   *   rect         [u0, v0, u1, v1], the part of the texture to show
   *   pivot        the point of the image placed at the node, 0..1 from its
   *                top-left: [0.5, 0.5], its centre, by default; [0.5, 1] its
   *                bottom middle, for something standing on the ground
   *   angle        radians, about the view direction
   *   facing       'camera', turning every way; 'upright', turning about Y
   *                only, for trees and people seen from the side; or 'plane',
   *                not turning at all -- in the node's own x-y plane, a sign
   *   blend        'alpha' (sorted back to front), 'additive' (light: no order
   *                needed), or 'cutout' (drawn or not, by `cutoff`)
   *   layer        for a 2D view (Camera2D): higher draws over lower, and one
   *                layer draws in the order its sprites were added
   *   animation    { frames, fps = 12, loop = true }: frames are rects, as
   *                spriteSheet makes them, shown in turn by scene.advance
   *
   * Through a Camera2D a unit is a pixel, and a sprite with no size is its
   * frame's own size in texels.
   *
   * Unlit: the colour is the colour, lit by nothing. Scaled by the node.
   */
  addSprite({ texture, position = [0, 0, 0], parent = null, ...options } = {}) {
    this.changes++;
    const record = spriteRecord({ texture, ...options });
    record.added = this._added++;
    const node = this.createNode({ parent }, 'addSprite');
    node.setPosition(...position);
    this._added2D(node.entity);
    this.sprites.set(node.entity, record);
    return node;
  }

  /**
   * The environment that lights the scene and draws its background.
   * engine.createScene sets it; assign another at any time, and the next
   * frame draws with it -- engine.run sees the change.
   */
  get environment() {
    return this._environment;
  }

  set environment(environment) {
    if (environment !== this._environment) this.changes++;
    this._environment = environment;
  }

  /** Change a sprite's options; the same names addSprite takes. */
  setSprite(node, changes) {
    this.changes++;
    const current = this.sprites.get(node.entity);
    if (current === undefined) throw new Error('setSprite: this node has no sprite');
    // A new animation starts from its first frame.
    const restart = changes.animation !== undefined ? { frame: 0, time: 0 } : {};
    const next = checked('setSprite', () => spriteRecord({ ...current, ...changes, ...restart, sizeGiven: changes.size !== undefined || current.sizeGiven }));
    next.added = current.added;
    this.sprites.set(node.entity, next);
    // Repeating is a sampler, which a draw run binds.
    if (next.layer !== current.layer || next.texture !== current.texture || next.blend !== current.blend
      || repeats(next) !== repeats(current)) this.spriteOrder++;
    else this.spritesChanged.add(node.entity);
  }

  /** A sprite's options, or null. A copy: change it through setSprite. */
  spriteOf(node) {
    return copyOf(this.sprites, node);
  }

  // Every other kind's, the same way: its options or null, a copy.
  /** A shape's options, or null. A copy: change it through setShape. */
  shapeOf(node) { return copyOf(this.shapes, node); }
  /** A path's options, or null. A copy: change it through setPath. */
  pathOf(node) { return copyOf(this.paths, node); }
  /** A text's options, or null. A copy: change it through setText. */
  textOf(node) { return copyOf(this.texts, node); }
  /** A tilemap's options, or null. A copy: change it through setTilemap, and its tiles through setTile. */
  tilemapOf(node) { return copyOf(this.tilemaps, node); }
  /** An emitter's options, or null. A copy: change it through setEmitter. */
  emitterOf(node) { return copyOf(this.emitters, node); }
  /** A splat node's options -- { splats } -- or null. */
  splatsOf(node) { return copyOf(this.splats, node); }
  /** A decal's options, or null. A copy: change it through setDecal. */
  decalOf(node) { return copyOf(this.decals, node); }

  /** A reflection probe's options -- size, fade, and whether it is captured -- or null. */
  probeOf(node) {
    const probe = this.reflectionProbes.find((p) => p.entity === node.entity);
    return probe === undefined ? null : { size: Float32Array.from(probe.size), fade: probe.fade, captured: probe.captured };
  }

  /** A light's options, as addLight takes them, or null. A copy: change it through setLight. */
  lightOf(node) {
    const entity = node.entity;
    const castShadow = this.shadowCasters.has(entity);
    const directional = this._directional.get(entity);
    const size = this.lightSizes.get(entity) ?? 0;
    if (directional) return { type: 'directional', color: Float32Array.from(directional.color), intensity: directional.intensity, castShadow, size };
    const index = this._lightOf.get(entity);
    if (index === undefined) return null;
    const o = index * LIGHT_FLOATS, light = this.lights;
    const spot = light[o + 14] === LIGHT_SPOT;
    return {
      type: spot ? 'spot' : 'point',
      color: Float32Array.from(light.subarray(o + 4, o + 7)),
      intensity: light[o + 7],
      radius: light[o + 3],
      ...(spot ? { innerAngle: this._lightCone[index * 2], outerAngle: this._lightCone[index * 2 + 1] } : {}),
      castShadow,
      size,
    };
  }

  // ---------------------------------------------------------------- shapes

  /**
   * A shape, as a node, for a 2D view (Camera2D): a rectangle or an ellipse,
   * worked out per pixel from its distance to the edge rather than read from
   * an image, so it is round and smooth-edged at any size. Returns the Node.
   *
   *   scene.addShape({ shape: 'rect', size: [120, 12], radius: 6, color: [0.9, 0.2, 0.2, 1] });
   *   scene.addShape({ shape: 'ellipse', size: [30, 30], color: [0, 0, 0, 0], stroke: [1, 1, 1, 1], strokeWidth: 2 });
   *
   *   shape        'rect' or 'ellipse' (a circle is an ellipse as wide as tall)
   *   size         [width, height], in pixels through an unzoomed Camera2D
   *   radius       a rect's corner radius; half its height makes a capsule
   *   color        the fill, sRGB 0..1 as in CSS; alpha 0 for an outline alone
   *   stroke       the outline's colour, drawn inside the edge, as a CSS border is
   *   strokeWidth  the outline's width; 0, the default, for none
   *   pivot        the point placed at the node: [0.5, 0.5], the default, is its centre
   *   layer        as for sprites: higher draws over lower
   *   blend        'alpha' or 'additive'
   *
   * Placed, turned (node.setAngle) and scaled by its node. A 3D camera doesn't draw it.
   */
  addShape({ position = [0, 0, 0], parent = null, ...options } = {}) {
    this.changes++;
    const record = shapeRecord(options);
    record.added = this._added++;
    const node = this.createNode({ parent }, 'addShape');
    node.setPosition(...position);
    this._added2D(node.entity);
    this.shapes.set(node.entity, record);
    return node;
  }

  /** Change a shape's options; the same names addShape takes. */
  setShape(node, changes) {
    this.changes++;
    const current = this.shapes.get(node.entity);
    if (current === undefined) throw new Error('setShape: this node has no shape');
    const next = checked('setShape', () => shapeRecord({ ...current, ...changes }));
    next.added = current.added;
    this.shapes.set(node.entity, next);
    if (next.layer !== current.layer || next.blend !== current.blend) this.spriteOrder++;
    else this.spritesChanged.add(node.entity);
  }

  /**
   * A path, as a node, for a 2D view (Camera2D): a polygon to fill, a line to
   * stroke, or both. Worked out per pixel from its distance to its segments,
   * as a shape is, so it is smooth-edged at any size. Returns the Node.
   *
   *   scene.addPath({ points: [[0, 0], [60, 20], [0, 40]], color: [1, 0.8, 0, 1] });   // a triangle
   *   scene.addPath({ points: route, closed: false, color: [0, 0, 0, 0], stroke: [1, 1, 1, 1], strokeWidth: 3 });
   *
   *   points       [x, y] in the node's units, in order
   *   closed       whether the last point joins the first; only a closed path fills.
   *                A crossing or concave outline fills as a canvas's does (nonzero)
   *   color        the fill, sRGB 0..1; alpha 0 for a line alone
   *   stroke       the line's colour, centred on the path, with round joins and ends
   *   strokeWidth  the line's width; 0, the default, for none
   *   layer, blend, lit   as for shapes
   *
   * Placed, turned and scaled by its node. A 3D camera doesn't draw it.
   */
  addPath({ position = [0, 0, 0], parent = null, ...options } = {}) {
    this.changes++;
    this.spriteOrder++;
    const record = pathRecord(options);
    record.added = this._added++;
    const node = this.createNode({ parent }, 'addPath');
    node.setPosition(...position);
    this.paths.set(node.entity, record);
    return node;
  }

  /** Change a path's options; the same names addPath takes. */
  setPath(node, changes) {
    this.changes++;
    const current = this.paths.get(node.entity);
    if (current === undefined) throw new Error('setPath: this node has no path');
    const next = checked('setPath', () => pathRecord({ ...current.options, ...changes }));
    next.added = current.added;
    this.paths.set(node.entity, next);
    // New points are new GPU data, so the list is rebuilt; a colour is not.
    // New points, or opening or closing it -- a long open path is drawn in
    // pieces, a closed one whole -- change its share of a 2D view's list.
    if (changes.points !== undefined || next.closed !== current.closed
      || next.layer !== current.layer || next.blend !== current.blend) this.spriteOrder++;
    else this.spritesChanged.add(node.entity);
  }

  // ------------------------------------------------------------- particles

  /**
   * A particle emitter, as a node: particles leave it along its +Y (turned by
   * its rotation), live their lifetime, and are drawn as camera-facing
   * quads. Simulated on the GPU. Returns the Node.
   *
   *   const sparks = scene.addEmitter({
   *     rate: 200, lifetime: [0.4, 0.8], size: 0.05, sizeEnd: 0, speed: [2, 4], spread: 0.4,
   *     acceleration: [0, -9.81, 0], color: [4, 2, 0.5, 1],
   *   });
   *
   *   rate          particles a second; 0 for bursts only (scene.burst)
   *   lifetime      seconds: one value, or [min, max] for each particle to draw from
   *   size          world units across, at birth; sizeEnd is the size at death,
   *                 the same unless given -- as color and colorEnd
   *   speed         at birth: one value, or [min, max]; 0 by default
   *   direction     in the node's space; its +Y by default
   *   spread        radians off the direction a particle may leave at: 0 is a
   *                 line, pi every way
   *   radius        they are born anywhere in this sphere around the node; 0, a point
   *   acceleration  world space, per second per second: gravity, if you want it
   *   drag          the share of speed lost per second, as a rate: v' = -drag v
   *   color         linear, may pass 1 to glow; colorEnd is what it becomes at
   *                 death, the same unless given
   *   texture       from engine.loadTexture; without one, a soft round dot
   *   blend         'additive' (by default: needs no order, which particles
   *                 are not drawn in) or 'alpha'
   *
   * Particles live in world space once born: a moving emitter leaves a trail.
   * Advance them with scene.advance(dt) -- engine.run does.
   */
  addEmitter({ position = [0, 0, 0], parent = null, ...options } = {}) {
    this.changes++;
    this.spriteOrder++;
    const record = emitterRecord(options);
    record.added = this._added++;
    const node = this.createNode({ parent }, 'addEmitter');
    node.setPosition(...position);
    this.emitters.set(node.entity, record);
    return node;
  }

  /** Change an emitter's options; the same names addEmitter takes. Its particles live on. */
  setEmitter(node, changes) {
    this.changes++;
    const current = this.emitters.get(node.entity);
    if (current === undefined) throw new Error('setEmitter: this node has no emitter');
    const next = checked('setEmitter', () => emitterRecord({ ...current.options, ...changes }));
    next.owed = current.owed;
    next.burstOwed = current.burstOwed;
    next.time = current.time;
    next.seed = current.seed;
    next.added = current.added;
    this.emitters.set(node.entity, next);
    if (next.layer !== current.layer) this.spriteOrder++;
  }

  /** Emit `count` particles at once, on the next frame. */
  burst(node, count) {
    this.changes++;
    const record = this.emitters.get(node.entity);
    if (record === undefined) throw new Error('burst: this node has no emitter');
    if (!(Number.isInteger(count) && count >= 0)) throw new Error(`burst: count must be a whole number, got ${count}`);
    record.owed += count;
    // Kept apart from what the rate owes: a burst's particles hold slots of
    // their own for as long as they may live (render/particles.js).
    record.burstOwed += count;
  }

  /**
   * Move every emitter's clock on by `dt` seconds: what it owes in new
   * particles, and the time its particles have to be carried through. The
   * renderer settles both on the next frame.
   */
  _advanceParticles(dt) {
    if (!(dt >= 0)) return;
    this._particleClock += dt;
    for (const record of this.emitters.values()) {
      record.owed += record.rate * dt;
      record.time += dt;
      // Anything born now lives at most its longest lifetime.
      if (record.owed > 0) this._particlesUntil = Math.max(this._particlesUntil, this._particleClock + record.lifetime[1]);
    }
  }

  /** Whether a particle may still be alive, or one is about to be born. */
  get particlesActive() {
    return this._particleClock < this._particlesUntil;
  }

  // ---------------------------------------------------------------- splats

  /**
   * A Gaussian splat capture, as a node: splats from engine.loadSplats,
   * placed, turned and scaled by the node's transform. Returns the Node.
   *
   *   const room = await engine.loadSplats('room.ply');
   *   scene.addSplats({ splats: room });
   *
   * Lit by nothing: drawn as the capture saw it, hidden by geometry in
   * front of it, and casting no shadow. See render/splats.js.
   */
  addSplats({ splats, position = [0, 0, 0], parent = null } = {}) {
    if (!(splats?.centers !== undefined && Number.isInteger(splats.count))) {
      throw new Error('addSplats: splats must be what engine.loadSplats returned');
    }
    if (splats.unloaded) throw new Error('addSplats: these splats were unloaded');
    this.changes++;
    const node = this.createNode({ parent }, 'addSplats');
    node.setPosition(...position);
    this.splats.set(node.entity, { splats });
    return node;
  }

  // ------------------------------------------------------------------ text

  /**
   * Text, as a node: a string in a font from engine.loadFont, drawn as a
   * quad a glyph and sharp at any size. Returns the Node.
   *
   *   const label = scene.addText({ font, text: 'Gate 3', size: 0.4, parent: gate });
   *   scene.addText({ font, text: 'EXIT', size: 32, pixels: true, color: [0, 4, 0, 1] });
   *
   *   size        the font's em, in world units -- or in pixels, with pixels: true
   *   color       linear; may pass 1 to glow
   *   align       'left', 'center' or 'right', for more than one line
   *   pivot       the point of the block placed at the node, 0..1 from its
   *               top-left, as a sprite's is; [0.5, 0.5], its centre, by default
   *   width       lines wrap between words to fit it, in the units of `size`
   *   lineHeight  in ems; the font's own by default
   *   facing      'camera' (by default), 'upright' or 'plane', as for sprites
   */
  addText({ position = [0, 0, 0], parent = null, ...options } = {}) {
    this.changes++;
    const record = textRecord(options);
    record.added = this._added++;
    const node = this.createNode({ parent }, 'addText');
    node.setPosition(...position);
    this._added2D(node.entity);
    this.texts.set(node.entity, record);
    return node;
  }

  /** Change a text's options -- its string, say; the same names addText takes. */
  setText(node, changes) {
    this.changes++;
    const current = this.texts.get(node.entity);
    if (current === undefined) throw new Error('setText: this node has no text');
    const next = checked('setText', () => textRecord({ ...current.options, ...changes }));
    next.added = current.added;
    this.texts.set(node.entity, next);
    // As many glyphs, of the same atlas, in the same place in the order: a 2D
    // view rewrites them where they are, as a score that ticks over does.
    // Otherwise its share of the list changes, and the list is made again.
    if (next.boxes.length === current.boxes.length && next.font === current.font
      && next.layer === current.layer && next.blend === current.blend) this.spritesChanged.add(node.entity);
    else this.spriteOrder++;
  }

  /** A sprite, text or shape just added, for a 2D view to append. */
  _added2D(entity) {
    this.added2D.add(entity);
    this.layout2D++;
  }

  // ---------------------------------------------------------------- decals

  /**
   * A decal: a texture projected onto whatever surfaces lie in a box, along
   * the box's -Z -- a scorch mark, a poster, a puddle. As a node, so it is
   * placed, turned and parented like one. Returns the Node.
   *
   *   const scorch = scene.addDecal({ texture: burn, size: [2, 2, 0.5] });
   *   scorch.setPosition(0, 0.01, 0).setAxisAngle([1, 0, 0], -Math.PI / 2);
   *
   *   texture  from engine.loadTexture; its alpha is how much it covers
   *   size     the box: [width, height] across the image, and depth along
   *            the projection, in the node's units. Scaled by the node.
   *   color    multiplies the texture; its alpha scales the cover
   *
   * It changes the surface's base colour before lighting, so it is lit,
   * shadowed and fogged as the surface is. A surface facing away from it is
   * not painted. Decals added later paint over earlier ones.
   */
  addDecal({ position = [0, 0, 0], parent = null, ...options } = {}) {
    this.changes++;
    const record = decalRecord(options);
    const node = this.createNode({ parent }, 'addDecal');
    node.setPosition(...position);
    this.decals.set(node.entity, record);
    return node;
  }

  /** Change a decal's options; the same names addDecal takes. */
  setDecal(node, changes) {
    this.changes++;
    const current = this.decals.get(node.entity);
    if (current === undefined) throw new Error('setDecal: this node has no decal');
    this.decals.set(node.entity, checked('setDecal', () => decalRecord({ ...current, ...changes })));
  }

  // -------------------------------------------------------------- tilemaps

  /**
   * A grid of tiles from one tileset image, as a node, for a 2D view
   * (Camera2D). Drawn as one quad whatever its size: each pixel looks its tile
   * up, so a map costs the pixels it covers, and changing a tile uploads that
   * tile alone. Returns the Node, placed at the map's top-left.
   *
   *   const tiles = await engine.loadTexture('tiles.png', { pixelated: true });
   *   const map = scene.addTilemap({ tileset: tiles, tileSize: [16, 16], columns: 100, rows: 40, tiles: level });
   *   scene.setTile(map, 12, 3, 0);   // break a block
   *
   *   tileset   from engine.loadTexture: tiles in a grid, read left to right,
   *             then top to bottom
   *   margin, spacing   texels around the grid, and between its tiles, as
   *             Tiled's tilesets name them; 0 by default
   *   tileSize  [width, height] of a tile, in the tileset's texels -- and in
   *             pixels on screen, before the node's scale
   *   columns, rows   the map's size, in tiles
   *   tiles     columns x rows ids, row by row from the top-left: 0 is empty,
   *             1 the tileset's first tile. The top three bits flip a tile as
   *             Tiled's do -- horizontally (2^31), vertically (2^30) and
   *             diagonally (2^29) -- so Tiled's layer data can be used as is
   *   layer     as for sprites: higher draws over lower
   *   color     multiplies every tile, sRGB 0..1
   *
   * Not drawn by a 3D camera.
   */
  addTilemap({ position = [0, 0, 0], parent = null, ...options } = {}) {
    this.changes++;
    this.spriteOrder++;
    const record = tilemapRecord(options);
    record.added = this._added++;
    const node = this.createNode({ parent }, 'addTilemap');
    node.setPosition(...position);
    this.tilemaps.set(node.entity, record);
    return node;
  }

  /**
   * Change a tilemap's options; the same names addTilemap takes. Its tiles are
   * kept unless given -- and must be, if the map changes size.
   */
  setTilemap(node, changes) {
    this.changes++;
    const current = this.tilemaps.get(node.entity);
    if (current === undefined) throw new Error('setTilemap: this node has no tilemap');
    const next = checked('setTilemap', () => tilemapRecord({ ...current, ...changes }));
    next.added = current.added;
    this.tilemaps.set(node.entity, next);
    // A new record is new GPU data, so the list is rebuilt: rare, and simple.
    this.spriteOrder++;
  }

  /** Set the tile at column x, row y: an id as addTilemap's `tiles` takes. */
  setTile(node, x, y, id) {
    this._setTiles('setTile', node, x, y, 1, [id]);
  }

  /**
   * Set a block of tiles: `tiles` holds rows of `width` ids, the block's
   * top-left at column x, row y.
   */
  setTiles(node, x, y, width, tiles) {
    this._setTiles('setTiles', node, x, y, width, tiles);
  }

  /** setTile and setTiles, whose errors carry the name that was called. */
  _setTiles(caller, node, x, y, width, tiles) {
    const map = this.tilemaps.get(node.entity);
    if (map === undefined) throw new Error(`${caller}: this node has no tilemap`);
    const height = tiles.length / width;
    if (!(Number.isInteger(x) && Number.isInteger(y) && Number.isInteger(width) && width > 0 && Number.isInteger(height)
      && x >= 0 && y >= 0 && x + width <= map.columns && y + height <= map.rows)) {
      throw new Error(caller === 'setTile'
        ? `setTile: (${x}, ${y}) is not on the ${map.columns} x ${map.rows} map`
        : `setTiles: a ${width}-wide block of ${tiles.length} at (${x}, ${y}) is not inside the ${map.columns} x ${map.rows} map`);
    }
    checkTiles(caller, tiles, tileCount(map));
    for (let row = 0; row < height; row++) {
      map.tiles.set(tiles.slice(row * width, (row + 1) * width), (y + row) * map.columns + x);
    }
    // What the next frame uploads: everything changed since the last.
    const d = map.dirty;
    map.dirty = d === null ? [x, y, x + width, y + height]
      : [Math.min(d[0], x), Math.min(d[1], y), Math.max(d[2], x + width), Math.max(d[3], y + height)];
    this.changes++;
  }

  /** The id at column x, row y, flip bits and all; 0 for empty or off the map. */
  tileAt(node, x, y) {
    const map = this.tilemaps.get(node.entity);
    if (map === undefined) throw new Error('tileAt: this node has no tilemap');
    if (!(x >= 0 && y >= 0 && x < map.columns && y < map.rows)) return 0;
    return map.tiles[Math.floor(y) * map.columns + Math.floor(x)];
  }

  // ----------------------------------------------------------- reflections

  /**
   * A reflection probe, as a node: the scene seen from the node, reflected by
   * the surfaces inside a box around it in place of the sky. Nothing shows
   * until it is captured -- engine.captureProbes. Returns the Node.
   *
   *   const hall = scene.addProbe({ position: [0, 2, 0], size: [10, 4, 16] });
   *
   *   size   the box, [width, height, depth] in world units, centred on the node
   *   fade   how far in from the box's faces the probe fades in: 0, the default,
   *          is a hard edge; more hides the seam between neighbours
   *
   * The box stays square to the world: turning or scaling the node does not
   * turn or scale it. Moving the node moves it, and it shows nothing again
   * until it is captured where it now is. Removed like any node.
   */
  addProbe({ position = [0, 0, 0], parent = null, ...options } = {}) {
    this.changes++;
    const node = this.createNode({ parent }, 'addProbe');
    node.setPosition(...position);
    const probe = probeRecord(options);
    probe.entity = node.entity;
    this.reflectionProbes.push(probe);
    this.probeRevision++;
    return node;
  }

  /** Change a probe's options; the same names addProbe takes. It must be captured again. */
  setProbe(node, changes) {
    const probe = this._probe('setProbe', node);
    this.changes++;
    const next = checked('setProbe', () => probeRecord({ size: probe.size, fade: probe.fade, ...changes }));
    Object.assign(probe, { size: next.size, fade: next.fade, captured: false, placed: null });
    this.probeRevision++;
  }

  /** The probe record behind a node, for the renderer's captures. */
  _probe(caller, node) {
    const probe = this.reflectionProbes.find((p) => p.entity === node.entity);
    if (probe === undefined) throw new Error(`${caller}: this node is not a reflection probe`);
    return probe;
  }

  /**
   * Each probe's box and eye, from where its node is now. A probe that moved
   * shows nothing until captured again: what it saw was somewhere else.
   */
  _placeProbes() {
    const world = this.transforms.world;
    for (const probe of this.reflectionProbes) {
      const m = handleIndex(probe.entity) * 16;
      const x = world[m + 12], y = world[m + 13], z = world[m + 14];
      if (probe.placed !== null && probe.placed[0] === x && probe.placed[1] === y && probe.placed[2] === z) continue;
      probe.placed = [x, y, z];
      probe.position.set(probe.placed);
      for (let a = 0; a < 3; a++) {
        probe.min[a] = probe.placed[a] - probe.size[a] / 2;
        probe.max[a] = probe.placed[a] + probe.size[a] / 2;
      }
      probe.captured = false;
      this.probeRevision++;
    }
  }

  // ------------------------------------------------------------------ lights

  /**
   * Add a light -- point, spot or directional -- as a node in the scene. Returns the Node.
   *
   *   const lamp = scene.addLight({ position: [0, 3, 0], color: [1, 0.7, 0.4], intensity: 20 });
   *   lamp.setPosition(2, 3, 0);                     // moves the light
   *
   *   const torch = scene.addLight({ direction: [0, 0, -1], parent: hand });
   *   // follows the hand, aims where the hand aims, and nobody updates it
   *
   * A LIGHT IS A SCENE OBJECT. It used to be a row in a packed array addressed
   * by index: moving one meant calling setLightPosition every frame, a light
   * could not follow anything, and removal swap-deleted -- so the last light
   * silently took the removed one's index and anyone holding it now pointed at
   * a different light. Entities already solved that with generation-tagged
   * handles; lights now use them.
   *
   * POSITION AND AIM COME FROM THE TRANSFORM. A spot shines down its node's -Z,
   * which is glTF's KHR_lights_punctual convention. `direction` here is a
   * convenience that sets the node's rotation to point that way -- after which
   * the node carries it, so parenting and animation aim it for free.
   *
   * `radius` is where the light reaches exactly zero. Physical inverse-square
   * falloff never quite does, so without a cutoff every light would have to be
   * tested against every cluster in the scene -- the radius is what makes
   * clustering possible at all, not a shortcut.
   */
  addLight({
    position = [0, 0, 0],
    color = [1, 1, 1],
    intensity = 1,
    radius = 10,
    direction = null,
    type = direction ? 'spot' : 'point',
    innerAngle = 0.2,
    outerAngle = 0.5,
    parent = null,
    castShadow,
    size = 0,
  } = {}) {
    this.changes++;
    if (type !== 'point' && type !== 'spot' && type !== 'directional') {
      throw new Error(`addLight: type must be point, spot or directional, got ${type}`);
    }
    checkLight('addLight', { color, intensity, radius, size }, type === 'spot' ? [innerAngle, outerAngle] : null);
    // Aim by rotating the node: -Z along the requested direction, upright.
    // [x, y] aims one across a 2D view.
    const rotation = direction ? quatLookAlong(quatCreate(), [direction[0], direction[1], direction[2] ?? 0]) : undefined;

    const parentEntity = this._parentOf('addLight', parent);
    const entity = this.entities.alloc();
    this.transforms.add(entity, {
      // [x, y] places one in a 2D view, as node.setPosition(x, y) does.
      position: [position[0], position[1], position[2] ?? 0],
      rotation,
      parent: parentEntity,
    });
    this._attachLight(entity, { type, color, intensity, radius, innerAngle, outerAngle, castShadow });
    if (size > 0) this.lightSizes.set(entity, size);
    return new Node(this, entity);
  }

  /**
   * Change what a light IS -- colour, brightness, reach, cone, shadows --
   * without touching where it is: the same names addLight takes, only those
   * given changing. Move or aim it through its node.
   */
  setLight(node, changes) {
    const current = this.lightOf(node);
    if (current === null) throw new Error('setLight: this node is not a light');
    for (const key of ['type', 'position', 'direction', 'parent']) {
      if (key in changes) throw new Error(`setLight: ${key} can't change here; ${key === 'type' ? 'add a new light' : 'use the node'}`);
    }
    const coned = current.type === 'spot';
    for (const key of current.type === 'directional' ? ['radius', 'innerAngle', 'outerAngle'] : coned ? [] : ['innerAngle', 'outerAngle']) {
      if (key in changes) throw new Error(`setLight: a ${current.type} light has no ${key}`);
    }
    checkLight('setLight', changes, coned ? [changes.innerAngle ?? current.innerAngle, changes.outerAngle ?? current.outerAngle] : null);
    this._setLight(node.entity, changes);
  }

  /** setLight by entity: false if it is not a light. An animated light's channels come through here. */
  _setLight(entity, changes) {
    this.changes++;
    const directional = this._directional.get(entity);
    if (changes.castShadow !== undefined && (directional || this._lightOf.has(entity))) {
      if (changes.castShadow) this.shadowCasters.add(entity); else this.shadowCasters.delete(entity);
    }
    if (changes.size !== undefined && (directional || this._lightOf.has(entity))) {
      if (changes.size > 0) this.lightSizes.set(entity, changes.size); else this.lightSizes.delete(entity);
    }
    if (directional) {
      // A directional light has no radius and no cone: colour and brightness are all of it.
      if (changes.color) directional.color.set(changes.color);
      if (changes.intensity !== undefined) directional.intensity = changes.intensity;
      return true;
    }

    const index = this._lightOf.get(entity);
    if (index === undefined) return false;
    const o = index * LIGHT_FLOATS;
    const light = this.lights;

    const spot = light[o + 14] === LIGHT_SPOT;
    this._writeLightProperties(index, {
      color: changes.color ?? [light[o + 4], light[o + 5], light[o + 6]],
      intensity: changes.intensity ?? light[o + 7],
      radius: changes.radius ?? light[o + 3],
      spot,
      innerAngle: changes.innerAngle ?? this._lightCone[index * 2] ?? 0.2,
      outerAngle: changes.outerAngle ?? this._lightCone[index * 2 + 1] ?? 0.5,
    });
    return true;
  }

  /**
   * Copy every light's position and aim out of its transform.
   *
   * Runs before the lights are uploaded, after transforms have composed. The
   * packed array stays the GPU's format, unchanged -- what changed is that
   * nobody writes positions into it by hand any more.
   */
  refreshLights() {
    const world = this.transforms.world;
    const light = this.lights;
    for (let i = 0; i < this.lightCount; i++) {
      const m = handleIndex(this.lightEntity[i]) * 16;
      const o = i * LIGHT_FLOATS;

      light[o] = world[m + 12];
      light[o + 1] = world[m + 13];
      light[o + 2] = world[m + 14];

      if (light[o + 14] === LIGHT_SPOT) {
        // -Z of the world matrix. Normalized, because a scaled parent scales
        // this column too, and a cone test on a non-unit axis is wrong.
        const x = -world[m + 8];
        const y = -world[m + 9];
        const z = -world[m + 10];
        const inv = 1 / (hypot3(x, y, z) || 1);
        light[o + 8] = x * inv;
        light[o + 9] = y * inv;
        light[o + 10] = z * inv;
      }
    }

    // Directional lights: -Z of the node, like a spot, and colour at
    // intensity, every one of them. Which cast shadows, and into which slot,
    // the renderer writes into each record's w after this.
    const count = this._directional.size;
    if (this.directionals.length < count * DIRECTIONAL_FLOATS) {
      this.directionals = new Float32Array(grownCapacity(this.directionals.length / DIRECTIONAL_FLOATS, count) * DIRECTIONAL_FLOATS);
    }
    this.directionalEntity.length = count;
    let packed = 0;
    for (const [entity, { color, intensity }] of this._directional) {
      const m = handleIndex(entity) * 16;
      const x = -world[m + 8];
      const y = -world[m + 9];
      const z = -world[m + 10];
      const inv = 1 / (hypot3(x, y, z) || 1);
      this.directionalEntity[packed] = entity;
      const o = packed++ * DIRECTIONAL_FLOATS;
      this.directionals[o] = x * inv;
      this.directionals[o + 1] = y * inv;
      this.directionals[o + 2] = z * inv;
      this.directionals[o + 3] = 0;
      this.directionals[o + 4] = color[0] * intensity;
      this.directionals[o + 5] = color[1] * intensity;
      this.directionals[o + 6] = color[2] * intensity;
      // Its size, an angle: how soft its shadows are (render/shaders/pbr.js, softVisibility).
      this.directionals[o + 7] = this.lightSizes.get(entity) ?? 0;
    }
    this.directionalCount = packed;
  }

  /**
   * Make an existing entity a light. addLight makes the entity first; an
   * imported glTF node already is one.
   */
  _attachLight(entity, properties) {
    if (properties.castShadow ?? properties.type === 'directional') this.shadowCasters.add(entity);
    if (properties.type === 'directional') {
      // Not in the packed array: that is the clustered lights, and a
      // directional light reaches everything, so it has no cluster to be in.
      this._directional.set(entity, {
        color: Float32Array.from(properties.color ?? [1, 1, 1]),
        intensity: properties.intensity ?? 1,
      });
      return;
    }
    if (this.lightCount >= this.lightCapacity) {
      const capacity = grownCapacity(this.lightCapacity, this.lightCount + 1);
      this.lights = growArray(this.lights, capacity, LIGHT_FLOATS);
      this.lightEntity = growArray(this.lightEntity, capacity);
      this.lightCapacity = capacity;
    }
    const index = this.lightCount++;
    this.lightEntity[index] = entity;
    this._lightOf.set(entity, index);
    this._writeLightProperties(index, { ...properties, spot: properties.type === 'spot' });
  }

  /**
   * Everything about a light except where it is and which way it points --
   * those come from its transform, in refreshLights.
   */
  _writeLightProperties(index, { color, intensity, radius, spot, innerAngle, outerAngle }) {
    const o = index * LIGHT_FLOATS;
    const light = this.lights;

    light[o + 3] = radius;
    light[o + 4] = color[0]; light[o + 5] = color[1]; light[o + 6] = color[2];
    light[o + 7] = intensity;

    // The cone is stored as the scale/offset the shader wants, which cannot be
    // turned back into angles exactly -- so the angles are kept alongside, for
    // a partial setLight that changes only one of them.
    if (!this._lightCone || this._lightCone.length < this.lightCapacity * 2) {
      const cone = new Float32Array(this.lightCapacity * 2);
      if (this._lightCone) cone.set(this._lightCone);
      this._lightCone = cone;
    }
    this._lightCone[index * 2] = innerAngle;
    this._lightCone[index * 2 + 1] = outerAngle;

    if (spot) {
      // Frostbite's smooth cone: precomputing scale and offset turns the
      // per-pixel test into a multiply-add instead of two cosines.
      const cosOuter = Math.cos(outerAngle);
      const scale = 1 / Math.max(Math.cos(innerAngle) - cosOuter, 1e-4);
      light[o + 12] = scale;
      light[o + 13] = -cosOuter * scale;
      light[o + 14] = LIGHT_SPOT;
    } else {
      light[o + 8] = 0; light[o + 9] = -1; light[o + 10] = 0;
      light[o + 12] = 1; light[o + 13] = 0;
      light[o + 14] = LIGHT_POINT;
    }
    light[o + 11] = 0;
    light[o + 15] = 0;
  }

  /**
   * Drop one light by index, keeping the packed array dense.
   *
   * Swap-remove, same as renderables -- and safe now, because nothing outside
   * this class holds a light INDEX. Callers hold the entity, and the map from
   * entity to index is fixed up here, so the light that moved into the gap is
   * still found by the handle its owner already has.
   */
  _removeLightAt(index) {
    const last = --this.lightCount;
    this._lightOf.delete(this.lightEntity[index]);
    if (index !== last) {
      this.lights.copyWithin(index * LIGHT_FLOATS, last * LIGHT_FLOATS, (last + 1) * LIGHT_FLOATS);
      this._lightCone.copyWithin(index * 2, last * 2, last * 2 + 2);
      const moved = this.lightEntity[last];
      this.lightEntity[index] = moved;
      this._lightOf.set(moved, index);
    }
  }

  /**
   * Recompose world matrices. Returns how many transforms were recomputed.
   *
   * With a job system it runs one depth level at a time across threads; without
   * one it is the same code on this thread. Identical results either way.
   */
  update(jobs = null) {
    const recomposed = jobs?.parallel ? this.transforms.updateParallel(jobs) : this.transforms.update();
    if (this.reflectionProbes.length > 0) this._placeProbes();
    return recomposed;
  }

  /**
   * What each property a clip animates by pointer writes to, for one
   * instance: key -> { slot, components, rest, write }. Null when no clip
   * drives any. Only what the clips name is built.
   */
  _propertiesFor(asset, lightsOf, camerasOf) {
    let properties = null;
    // One state per light, shared by all its fields: setLight takes the whole
    // light, so a colour written from a stale copy would put back an old
    // intensity.
    const lightState = [];
    for (const clip of asset.animations) {
      for (const channel of clip.channels) {
        if (channel.path !== 'property' || properties?.has(channel.key)) continue;
        const target = channel.kind === 'light'
          ? this._lightProperty(asset.lights[channel.index], lightsOf[channel.index], channel.field,
            (lightState[channel.index] ??= lightStateOf(asset.lights[channel.index])))
          : channel.kind === 'camera' ? cameraProperty(camerasOf[channel.index], channel.field)
            : this._materialProperty(asset, channel.index, channel.field);
        // A light or camera on no node of the default scene has nothing here.
        if (target === null) continue;
        properties ??= new Map();
        properties.set(channel.key, { slot: properties.size, components: channel.components, ...target });
      }
    }
    return properties;
  }

  /**
   * A light's colour, intensity, range or cone, written to every node this
   * instance put it on. A light whose file gave no range keeps deriving its
   * reach from its brightness as that changes.
   *
   * A value the importer would refuse -- a CUBICSPLINE curve can overshoot
   * its keys -- is not written, and the light keeps the last one it would
   * have accepted. Clamping would need an edge to clamp to, and `range > 0`
   * and `inner < outer` have none.
   */
  _lightProperty(spec, entities, field, live) {
    if (entities === undefined) return null;
    const rest = field === 'color' ? live.color
      : field === 'range' ? [live.range ?? spec.radius]
        : [live[field]];
    return {
      rest: Float32Array.from(rest),
      write: (values) => {
        const v = values[0];
        if (field === 'color') {
          if (!(values[0] >= 0 && values[1] >= 0 && values[2] >= 0)) return;
          for (let c = 0; c < 3; c++) live.color[c] = values[c];
        } else if (field === 'intensity') {
          if (!(v >= 0)) return;
          live.intensity = v;
        } else if (field === 'range') {
          if (!(v > 0)) return;
          live.range = v;
        } else if (field === 'innerAngle') {
          if (!(v >= 0 && v < live.outerAngle)) return;
          live.innerAngle = v;
        } else {
          if (!(v > live.innerAngle && v <= Math.PI / 2)) return;
          live.outerAngle = v;
        }
        const changes = { color: live.color, intensity: live.intensity };
        if (spec.type !== 'directional') changes.radius = live.range ?? unboundedLightRadius(live.intensity, live.color);
        if (spec.type === 'spot') {
          changes.innerAngle = live.innerAngle;
          changes.outerAngle = live.outerAngle;
        }
        for (const entity of entities) this._setLight(entity, changes);
      },
    };
  }

  /**
   * A factor of one of the asset's materials. Written into the importer's
   * record and queued for the renderer; emissive is the product of its factor
   * and strength, so either one recomputes it. The importer takes any finite
   * factor, and so does this.
   */
  _materialProperty(asset, index, field) {
    const record = asset.materials?.[index];
    const id = asset.materialIds?.[index];
    if (record === undefined || id === undefined) return null;
    // A texture transform's part: the part changes, and its slot's matrix
    // rows are rebuilt from all three.
    const uv = /^uv(\d+)\.(offset|rotation|scale)$/.exec(field);
    if (uv !== null) {
      const slot = Number(uv[1]);
      const part = record.uvTransformParts[slot];
      const name = uv[2];
      return {
        rest: Float32Array.from(name === 'rotation' ? [part.rotation] : part[name]),
        write: (values) => {
          if (name === 'rotation') part.rotation = values[0];
          else { part[name][0] = values[0]; part[name][1] = values[1]; }
          record.uvTransforms.set(uvTransformRows(part), slot * 6);
          this.changedMaterials.set(id, record);
        },
      };
    }
    const value = record[field];
    const width = typeof value === 'number' ? 1 : value.length;
    return {
      rest: Float32Array.from(width === 1 ? [value] : value),
      write: (values) => {
        if (width === 1) record[field] = values[0];
        else for (let c = 0; c < width; c++) record[field][c] = values[c];
        if (field === 'emissiveFactor' || field === 'emissiveStrength') {
          for (let c = 0; c < 3; c++) record.emissive[c] = record.emissiveFactor[c] * record.emissiveStrength;
        }
        this.changedMaterials.set(id, record);
      },
    };
  }

  /** The AnimationPlayer for an asset instance, or null if it has no clips. */
  _playerFor(node) {
    return this._players.get(node.entity) ?? null;
  }

  /**
   * Move the scene on by `dt` seconds: its animations -- clips and sprite
   * frames -- and its particles. engine.run calls it every frame; call it
   * yourself when you drive renderFrame.
   */
  advance(dt) {
    this._advanceAnimations(dt);
    this._advanceParticles(dt);
  }

  /**
   * Advance every playing clip.
   *
   * engine.run() calls this once per rendered frame, before composition, so
   * playing a clip is all you have to do. Driving renderFrame() yourself means
   * calling this yourself -- that is the deal the manual path makes everywhere
   * else too.
   *
   * Stepped by real elapsed time rather than the fixed simulation step, because
   * animation is presentation: it belongs with the camera controller, not with
   * the physics the accumulator exists to keep deterministic.
   */
  _advanceAnimations(dt) {
    let playing = 0;
    for (const player of this._players.values()) {
      if (player.advance(dt, this.transforms)) playing++;
    }
    // Sprite animations: a frame at a time, at their own rate.
    let sprites = 0;
    for (const [entity, sprite] of this.sprites) {
      const animation = sprite.animation;
      if (animation === null || !(dt > 0)) continue;
      const count = animation.frames.length;
      if (!animation.loop && sprite.frame === count - 1) continue;
      sprite.time += dt;
      const step = Math.floor(sprite.time * animation.fps);
      const frame = animation.loop ? step % count : Math.min(step, count - 1);
      sprites++;
      if (frame !== sprite.frame) {
        sprite.frame = frame;
        this.spritesChanged.add(entity);
        this.changes++;
      }
    }
    // Only clips keep the loop drawing: a sprite's new frame counts as a
    // change when it comes, so between frames a still scene can idle.
    this.animating = playing > 0;
    return playing + sprites;
  }

  /**
   * The nearest renderable a ray hits, or null.
   *
   * Two phases. Every world bounding box the ray enters is collected and sorted
   * by entry distance, then walked near to far; a primitive loaded with
   * `retainGeometry` is tested triangle by triangle, and one loaded without it
   * is taken at its box distance because that is the only answer available. The
   * walk stops as soon as the next box starts further away than the best hit so
   * far, which is what keeps an exact test off geometry that cannot win.
   *
   * Mixing the two in one scene is allowed and means what it looks like: an
   * un-retained object can shadow a retained one, because its box is all this
   * knows about it.
   *
   * Composes transforms and refreshes bounds first, because the alternative is
   * an API where the answer silently depends on whether you happened to render
   * since the last move. Both are no-ops on a settled scene.
   *
   * @param origin    vec3, world space
   * @param direction vec3, world space; normalized, or distances come back scaled
   * @returns `{ node, renderable, distance }`, or null
   */
  raycast(origin, direction, { maxDistance = Infinity } = {}) {
    // Unconditional, not DEBUG-only: this is a trust boundary, and a ray with a
    // NaN component is not merely wrong, it is INVISIBLY wrong. The slab test
    // derives no constraint from a NaN axis, so such a ray "hits" the first
    // renderable at distance zero. A zero-sized canvas is enough to produce one.
    assertFinite(origin, 'raycast origin', 0, 3);
    assertFinite(direction, 'raycast direction', 0, 3);

    this._refreshBounds();

    const candidates = [];

    for (let i = 0; i < this.renderableCount; i++) {
      // Of a mesh drawn in levels of detail, only the finest: a ray has no
      // distance to choose a level by, and a coarser level's looser shape
      // would be hit where the mesh shown up close isn't.
      if (this.renderableCoverage[i * 2 + 1] < F32_MAX) continue;
      const distance = aabbRayDistance(this.worldMin, this.worldMax, origin, direction, i * 3);
      // Not `>= 0`: a miss is -1, and a hit at exactly 0 means the origin is
      // already inside the box, which is a hit.
      if (distance < 0 || distance >= maxDistance) continue;
      candidates.push({ renderable: i, distance });
    }
    candidates.sort(byDistance);

    let bestDistance = maxDistance;
    let best = -1;

    for (const candidate of candidates) {
      if (candidate.distance >= bestDistance) break;

      const primitive = this.renderablePrimitive[candidate.renderable];
      // Skinned and morphed renderables are tested against their triangles AS
      // DEFORMED, the way the vertex shader deforms them. That is work per
      // vertex per click -- and only for the few whose boxes the ray already
      // hit. Without the retained data the box is the answer, as for anything
      // loaded without retainGeometry.
      const renderable = candidate.renderable;
      const skinned = this.renderableSkin[renderable] >= 0;
      const morphed = this.renderableMorph[renderable] >= 0;
      const retained = primitive.positions !== undefined && primitive.indices !== undefined
        && (!skinned || primitive.jointIndices !== undefined)
        && (!morphed || primitive.morphDeltas !== undefined);
      if (!retained) {
        best = renderable;
        bestDistance = candidate.distance;
        continue;
      }

      const distance = skinned || morphed
        ? this._deformedDistance(renderable, primitive, origin, direction)
        : this._triangleDistance(renderable, primitive, origin, direction);
      if (distance < 0 || distance >= bestDistance) continue;
      bestDistance = distance;
      best = candidate.renderable;
    }

    if (best < 0) return null;
    return {
      node: new Node(this, this.renderableEntity[best]),
      renderable: best,
      distance: bestDistance,
    };
  }

  /**
   * Compose transforms and bring every world bound up to date.
   *
   * Shared by raycast and bounds because they want the same thing: the scene
   * as it IS, not as it was when something last rendered. Both are no-ops on a
   * settled scene, which is what makes calling them unconditionally fine.
   */
  _refreshBounds() {
    this.update();
    if (this.transforms.movedPending) {
      updateWorldBounds(
        this.renderableCount, this.localMin, this.localMax, this.worldMin, this.worldMax,
        this.transforms.world, this.renderableMatrixSlot, this.transforms.moved, null, this.movedRenderables(),
      );
    }
    // Picking has to see the pose too, or a click lands on where a character
    // was authored rather than where it is standing.
    if (this.skins.length > 0) {
      updateSkinBounds(this.skins, this.transforms.world);
      applySkinBounds(
        this.renderableCount, this.renderableSkin, this.skins, this.worldMin, this.worldMax,
      );
    }
    if (this.morphs.length > 0) this.applyMorphBounds();
  }

  /**
   * The renderables whose node is in the transforms' moved list, ascending,
   * or null when it lists nothing (a whole compose ran) and `moved` has to be
   * scanned instead. The slot-to-renderables map behind it is rebuilt when the
   * renderables change, which is every add and remove and never a move.
   */
  movedRenderables() {
    const t = this.transforms;
    if (t.movedAll) return null;
    if (this._bySlotRevision !== this.revision) {
      const slots = t.capacity;
      const start = (this._bySlotStart = new Uint32Array(slots + 1));
      const list = (this._bySlot = new Uint32Array(this.renderableCount));
      for (let i = 0; i < this.renderableCount; i++) start[this.renderableMatrixSlot[i] + 1]++;
      for (let s = 0; s < slots; s++) start[s + 1] += start[s];
      const cursor = start.slice(0, slots);
      for (let i = 0; i < this.renderableCount; i++) list[cursor[this.renderableMatrixSlot[i]]++] = i;
      this._bySlotRevision = this.revision;
    }
    const start = this._bySlotStart;
    const out = (this._movedItems ??= []);
    out.length = 0;
    // A slot past the map was made after it, and has no renderable.
    for (const slot of t.movedList) {
      if (slot + 1 < start.length) for (let k = start[slot]; k < start[slot + 1]; k++) out.push(this._bySlot[k]);
    }
    return out.sort((a, b) => a - b);
  }

  /**
   * World-space bounds of everything in the scene, into `outMin`/`outMax`.
   *
   * Returns false and leaves the outputs alone for an empty scene: there is no
   * box meaning "nothing" that a caller would not have to special-case anyway.
   */
  bounds(outMin, outMax) {
    this._refreshBounds();
    let found = unionWorldBounds(this.renderableCount, this.worldMin, this.worldMax, outMin, outMax);
    // Splat clouds by the boxes around their centres.
    for (const [entity, { splats }] of this.splats) {
      aabbTransform(SPLAT_MIN, SPLAT_MAX, splats.min, splats.max, this.transforms.world, handleIndex(entity) * 16);
      if (!found) {
        for (let i = 0; i < 3; i++) { outMin[i] = SPLAT_MIN[i]; outMax[i] = SPLAT_MAX[i]; }
        found = true;
      } else aabbUnion(outMin, outMax, SPLAT_MIN, SPLAT_MAX);
    }
    return found;
  }

  /**
   * Point a camera at the whole scene, from wherever it is already looking.
   *
   * The guess this removes: every example picks a camera distance by eye and
   * nudges it until the model fits. It is derivable from the bounds and the
   * field of view, and the scene already computes those bounds every frame for
   * culling -- so the number was always there, just never offered.
   *
   * Returns false for an empty scene, having moved nothing. See
   * Camera.frameBounds for what the fit actually is.
   */
  frame(camera, options) {
    if (!this.bounds(FRAME_MIN, FRAME_MAX)) return false;
    camera.frameBounds(FRAME_MIN, FRAME_MAX, options);
    return true;
  }

  /**
   * Nearest triangle of one renderable along a ray, or -1.
   *
   * The ray is pushed into local space rather than the triangles into world
   * space: one matrix inverse against however many vertices the primitive has.
   *
   * The transformed direction is deliberately left un-normalized. `M` is
   * linear, so `M(o + t*d)` and `o + t*d` share the same `t`, and the distance
   * comes back on the same scale as the box distances it is compared against.
   * Normalizing here would silently rescale it by the object's scale factor.
   */
  _triangleDistance(renderable, primitive, origin, direction) {
    const slot = this.renderableMatrixSlot[renderable];
    mat4Copy(PICK_WORLD, this.transforms.world, 0, slot * 16);
    // A scale of zero on any axis collapses the mesh to a plane or a point.
    // There is nothing to hit, and inverting would divide by zero.
    if (mat4Invert(PICK_INVERSE, PICK_WORLD) === null) return -1;

    vec3TransformMat4(LOCAL_ORIGIN, origin, PICK_INVERSE);
    vec3TransformMat4Dir(LOCAL_DIRECTION, direction, PICK_INVERSE);

    const { positions, indices } = primitive;
    let nearest = -1;

    for (let i = 0; i + 2 < indices.length; i += 3) {
      const distance = rayTriangleDistance(
        LOCAL_ORIGIN, LOCAL_DIRECTION, positions,
        indices[i] * 3, indices[i + 1] * 3, indices[i + 2] * 3,
      );
      if (distance < 0) continue;
      if (nearest < 0 || distance < nearest) nearest = distance;
    }
    return nearest;
  }

  /**
   * The nearest triangle of a skinned or morphed renderable, in world space.
   *
   * The same deformation the vertex shader makes, in the same order: morph
   * targets first, against the bind pose, then either the joint palette --
   * jointWorld * inverseBind, with the mesh node's own matrix ignored, as
   * glTF says -- or, unskinned, the model matrix. Each vertex is deformed once
   * and shared by the triangles that use it.
   */
  _deformedDistance(renderable, primitive, origin, direction) {
    const { positions, indices } = primitive;
    const vertexCount = positions.length / 3;
    if (DEFORMED.length < positions.length) {
      DEFORMED = new Float32Array(grownCapacity(DEFORMED.length, positions.length));
    }
    const out = DEFORMED;

    const m = this.renderableMorph[renderable];
    const weights = m >= 0 ? this.morphs[m].weights : null;
    const targets = primitive.morphCountStride & 0xffff;
    const stride = primitive.morphCountStride >>> 16;
    const deltas = primitive.morphDeltas;

    const s = this.renderableSkin[renderable];
    const world = this.transforms.world;
    let palette = null;
    if (s >= 0) {
      const { joints, inverseBind } = this.skins[s];
      if (PALETTE.length < joints.length * 16) {
        PALETTE = new Float32Array(grownCapacity(PALETTE.length, joints.length * 16));
      }
      palette = PALETTE;
      for (let j = 0; j < joints.length; j++) {
        mat4Multiply(palette, world, inverseBind, j * 16, handleIndex(joints[j]) * 16, j * 16);
      }
    }
    const model = this.renderableMatrixSlot[renderable] * 16;

    for (let v = 0; v < vertexCount; v++) {
      let x = positions[v * 3], y = positions[v * 3 + 1], z = positions[v * 3 + 2];
      if (weights !== null) {
        for (let t = 0; t < targets; t++) {
          const w = weights[t];
          if (w === 0) continue;
          const o = (v * targets + t) * stride;
          x += w * deltas[o];
          y += w * deltas[o + 1];
          z += w * deltas[o + 2];
        }
      }
      // Skinned: the weighted palette. Otherwise: the model matrix.
      const matrices = palette ?? world;
      let wx = 0, wy = 0, wz = 0;
      for (let k = 0; k < (palette === null ? 1 : 4); k++) {
        const w = palette === null ? 1 : primitive.jointWeights[v * 4 + k];
        if (w === 0) continue;
        const p = palette === null ? model : primitive.jointIndices[v * 4 + k] * 16;
        wx += w * (matrices[p] * x + matrices[p + 4] * y + matrices[p + 8] * z + matrices[p + 12]);
        wy += w * (matrices[p + 1] * x + matrices[p + 5] * y + matrices[p + 9] * z + matrices[p + 13]);
        wz += w * (matrices[p + 2] * x + matrices[p + 6] * y + matrices[p + 10] * z + matrices[p + 14]);
      }
      out[v * 3] = wx; out[v * 3 + 1] = wy; out[v * 3 + 2] = wz;
    }

    let nearest = -1;
    for (let i = 0; i + 2 < indices.length; i += 3) {
      const distance = rayTriangleDistance(
        origin, direction, out, indices[i] * 3, indices[i + 1] * 3, indices[i + 2] * 3,
      );
      if (distance < 0) continue;
      if (nearest < 0 || distance < nearest) nearest = distance;
    }
    return nearest;
  }

  /**
   * The nearest renderable under a point on the canvas.
   *
   * `x`/`y` are CSS pixels from the canvas's top-left, and `width`/`height` its
   * CSS size -- exactly what a pointer event plus getBoundingClientRect give you.
   *
   * Through a Camera2D it is what the view shows there instead, topmost first,
   * as of the camera's last frame: `{ node, point }` -- the world point -- with
   * `tile: [column, row]` for a tilemap, or null. A sprite is hit anywhere in
   * its quad, clear pixels too; a shape inside its edge; text anywhere in its
   * block; a tilemap where its tile isn't empty.
   */
  pick(camera, x, y, width, height, options) {
    if (camera.is2D === true) return this._pick2D(camera.screenToWorld(x * camera.width / width, y * camera.height / height));
    camera.rayFromScreen(x, y, width, height, PICK_ORIGIN, PICK_DIRECTION);
    return this.raycast(PICK_ORIGIN, PICK_DIRECTION, options);
  }

  _pick2D(at) {
    this.update();
    // The reverse of the order they draw in, kept until that order changes --
    // picking on every pointer move sorted the whole scene each time. As
    // entities, with the map each is in: a set call replaces its record.
    if (this._picking?.order !== this.spriteOrder || this._picking.layout !== this.layout2D) {
      const items = [];
      for (const kind of [this.sprites, this.texts, this.tilemaps, this.shapes, this.paths]) {
        for (const [entity, record] of kind) items.push([entity, kind, record.layer, record.added]);
      }
      items.sort((p, q) => q[2] - p[2] || q[3] - p[3]);
      this._picking = { order: this.spriteOrder, layout: this.layout2D, items };
    }
    const world = this.transforms.world;
    for (const [entity, kind] of this._picking.items) {
      const record = kind.get(entity);
      // Into the node's own frame: the inverse of its 2D transform.
      const m = handleIndex(entity) * 16;
      const a = world[m], b = world[m + 1], c = world[m + 4], d = world[m + 5];
      const det = a * d - b * c;
      if (det === 0) continue;
      const dx = at[0] - world[m + 12], dy = at[1] - world[m + 13];
      const hit = isSprite(record) ? hitSprite(record, world, entity, dx, dy)
        : hit2D(record, (d * dx - c * dy) / det, (a * dy - b * dx) / det);
      if (hit !== null) return { node: new Node(this, entity), point: [at[0], at[1]], ...hit };
    }
    return null;
  }
}

/**
 * Whether a 2D view's sprite, text, tilemap or shape covers (x, y) in its
 * node's frame, as render/view2d.js draws it: {} or, for a tilemap,
 * { tile }, if so; null if not.
 */
function hit2D(record, x, y) {
  if (record.kind === 'path') {
    // As render/view2d.js fills it: nonzero winding inside a closed path, or
    // within half the line's width of it.
    const p = record.points, count = p.length / 2;
    const segments = record.closed ? count : count - 1;
    let winding = 0, nearest = Infinity;
    for (let i = 0; i < segments; i++) {
      const j = (i + 1) % count;
      const ax = p[i * 2], ay = p[i * 2 + 1], bx = p[j * 2], by = p[j * 2 + 1];
      const ex = bx - ax, ey = by - ay, px = x - ax, py = y - ay;
      const t = Math.min(Math.max((px * ex + py * ey) / (ex * ex + ey * ey || 1), 0), 1);
      nearest = Math.min(nearest, Math.hypot(px - ex * t, py - ey * t));
      const side = ex * py - ey * px;
      if (ay <= y) { if (by > y && side > 0) winding++; } else if (by <= y && side < 0) winding--;
    }
    // Inside a closed path is a hit even with a clear fill, as inside a shape is.
    return (record.closed && winding !== 0) || nearest <= record.strokeWidth / 2 ? {} : null;
  }
  if (record.kind === 'shape') {
    const [w, h] = record.size;
    const px = x + (record.pivot[0] - 0.5) * w, py = y + (record.pivot[1] - 0.5) * h;
    if (record.shape === 'ellipse') return (px / (w / 2)) ** 2 + (py / (h / 2)) ** 2 <= 1 ? {} : null;
    const r = shapeRadius(record);
    const qx = Math.abs(px) - w / 2 + r, qy = Math.abs(py) - h / 2 + r;
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r <= 0 ? {} : null;
  }
  if (record.kind === 'tilemap') {
    const [tw, th] = record.tileSize;
    const column = Math.floor(x / tw + record.pivot[0] * record.columns);
    const row = Math.floor(y / th + record.pivot[1] * record.rows);
    if (column < 0 || row < 0 || column >= record.columns || row >= record.rows) return null;
    return record.tiles[row * record.columns + column] === 0 ? null : { tile: [column, row] };
  }
  if (record.boxes !== undefined) {
    // Laid out y up, in ems; a 2D view draws it y down, `size` to an em.
    const [left, bottom, right, top] = record.block;
    const s = record.size;
    return x >= left * s && x <= right * s && y >= -top * s && y <= -bottom * s ? {} : null;
  }
  return null;
}

/** Whether a 2D record is a sprite: the one kind with neither a kind nor glyphs. */
function isSprite(record) {
  return record.kind === undefined && record.boxes === undefined;
}

const FRAME = new Float64Array(4);

/**
 * A sprite under the point (dx, dy) from its node, in world units: its quad as
 * the 2D view draws it -- sized by the node's scale, then turned by the node's
 * angle and its own together -- undone.
 */
function hitSprite(record, world, entity, dx, dy) {
  const [lx, ly, mirrored, angle] = frame2D(world, entity, FRAME);
  const turn = angle + record.angle;
  const cos = Math.cos(turn), sin = Math.sin(turn);
  const rect = spriteRect(record);
  const w = (record.sizeGiven ? record.size[0] : record.texture.width * Math.abs(rect[2] - rect[0])) * lx * (mirrored ? -1 : 1);
  const h = (record.sizeGiven ? record.size[1] : record.texture.height * Math.abs(rect[3] - rect[1])) * ly;
  const u = (cos * dx + sin * dy) / w + record.pivot[0], v = (cos * dy - sin * dx) / h + record.pivot[1];
  return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? {} : null;
}

/** Scratch for frame(). Not re-entrant, and it never needs to be. */
const FRAME_MIN = new Float32Array(3);
const FRAME_MAX = new Float32Array(3);
const SPLAT_MIN = new Float32Array(3);
const SPLAT_MAX = new Float32Array(3);

// Scratch for pick(). A scene is not raycast re-entrantly, and the result is
// read before the next call.
const PICK_ORIGIN = vec3Create();
const PICK_DIRECTION = vec3Create();

/** Scratch for the narrow phase: the ray, pushed into one renderable's local space. */
const PICK_WORLD = mat4Create();
/** Scratch for the deformed narrow phase: one renderable's world positions, and its palette. */
let DEFORMED = new Float32Array(0);
let PALETTE = new Float32Array(0);
const PICK_INVERSE = mat4Create();
const LOCAL_ORIGIN = vec3Create();
const LOCAL_DIRECTION = vec3Create();

function byDistance(a, b) {
  return a.distance - b.distance;
}

/** positionRadius, colorIntensity, directionCone, coneFalloff -- four vec4s. */
export const LIGHT_FLOATS = 16;
/** Floats per packed extra directional light: direction.xyz_, colour.rgb_. */
export const DIRECTIONAL_FLOATS = 8;
export const LIGHT_POINT = 0;
export const LIGHT_SPOT = 1;

/** What an animated light is now, starting from what the file said. */
function lightStateOf(spec) {
  return {
    color: Float32Array.from(spec.color),
    intensity: spec.intensity,
    range: spec.range ?? null,
    innerAngle: spec.innerAngle,
    outerAngle: spec.outerAngle,
  };
}

/**
 * A glTF camera's field of view, near or far plane, or orthographic half
 * height, on every Camera this instance made of it. The half height is the
 * distance to the target, as it is when the camera is made (see cameraFor).
 * As with lights, a value the importer would refuse is not written.
 */
function cameraProperty(cameras, field) {
  if (cameras === undefined) return null;
  const first = cameras[0];
  const rest = field === 'halfHeight' ? first.orthographicHalfHeight() : first[field];
  return {
    rest: Float32Array.of(rest),
    write: (values) => {
      const v = values[0];
      for (const camera of cameras) {
        if (field === 'fovY') {
          if (v > 0 && v < Math.PI) camera.fovY = v;
        } else if (field === 'near') {
          // An orthographic znear of 0 gets the importer's sliver of the box.
          if (camera.orthographic ? v >= 0 && v < camera.far : v > 0) camera.near = v > 0 ? v : camera.far * 1e-4;
        } else if (field === 'far') {
          if (v > camera.near) camera.far = v;
        } else if (v !== 0 && Number.isFinite(v)) {
          const scale = Math.abs(v) / camera.orthographicHalfHeight();
          for (let c = 0; c < 3; c++) {
            camera.target[c] = camera.position[c] + (camera.target[c] - camera.position[c]) * scale;
          }
        }
      }
    },
  };
}

function cameraFor(spec) {
  if (!spec.orthographic) return new Camera({ fovY: spec.fovY, near: spec.near });

  // An orthographic Camera shows 2 d tan(fovY / 2) at distance d from its
  // target, and following keeps that distance. So the file's half height,
  // ymag, is set by placing the target at d = ymag / tan(fovY / 2).
  const camera = new Camera({ near: spec.near, far: spec.far, orthographic: true });
  camera.target.set([0, 0, 0]);
  camera.position.set([0, 0, spec.halfHeight / Math.tan(camera.fovY * 0.5)]);
  return camera;
}
