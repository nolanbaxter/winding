// Cascaded shadow maps.
//
// A directional light covers the whole world, so one shadow map stretched over
// the visible range gives you blocky shadows near the camera and wasted texels
// far away. Cascades split the view distance into slices and give each its own
// map, so resolution follows the camera.
//
// Three problems dominate the implementation, and all three are geometry rather
// than shading:
//
// FITTING. Each cascade's projection must cover its slice of the view frustum
// and no more. Fitting to the frustum corners directly makes the box change
// shape as the camera rotates, which makes the shadow crawl. Fitting to the
// slice's bounding SPHERE instead is rotation-invariant -- the sphere is the
// same size whichever way you look -- so the box only moves, never resizes.
//
// SHIMMER. Even a sphere-fitted box slides continuously as the camera moves,
// so every texel covers a slightly different patch of world each frame and
// edges crawl. Snapping the box origin to whole texels removes it entirely.
//
// ACNE vs PETER-PANNING. A surface shadowing itself because its own depth
// rounds below the stored value gives acne -- dark stripes on lit surfaces.
// Biasing it away fixes that and detaches the shadow from the object's feet,
// which is peter-panning. Both are fought here, with different tools: slope-
// scaled hardware bias for the map, and a normal offset at lookup time.
//
// EVERY LIGHT that casts -- castShadow, on by default for directional lights
// and off for the rest -- gets views of its own, each a layer of a depth
// array, all drawn by the same caster passes: a view is a view-projection,
// whatever made it. A directional light gets the cascades; a spot one
// perspective view down its cone; a point light six, one per cube face.
// Cascades and local views are two arrays only because they are two sizes.

import { DEBUG, assert } from '../core/assert.js';
import {
  mat4Create, mat4LookAt, mat4Multiply, mat4OrthographicReverseZ, mat4PerspectiveReverseZInfinite,
} from '../core/math/mat4.js';
import { frustumCreate, frustumFromViewProjection, frustumTestSphere } from '../core/math/frustum.js';
import { grownCapacity } from '../core/grow.js';
import { LIGHT_FLOATS, LIGHT_SPOT, DIRECTIONAL_FLOATS } from '../scene/scene.js';
import { vec3Create, vec3Normalize } from '../core/math/vec3.js';
import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout, GROUP_FRAME, GROUP_MATERIAL, GROUP_DRAW } from '../rhi/bindgroups.js';
import { ALPHA_MASK, MATERIAL_WGSL, VARIANT_DOUBLE_SIDED } from './material.js';
import { CULL_SHADOW, NOT_BATCHED } from './gpudriven.js';
import { boxInOrtho } from '../scene/bounds.js';
import { DEPTH_FORMAT, DEPTH_CLEAR_VALUE, DEPTH_COMPARE } from '../rhi/device.js';
import { VERTEX_BUFFER_LAYOUT, SKIN_BUFFER_LAYOUT } from './vertex.js';
import { createBuffer } from '../rhi/buffer.js';
import { createTexture } from '../rhi/texture.js';

export const MAX_CASCADES = 4;

/** One local shadow view as the forward shader reads it: a mat4 and a vec4. */
export const LOCAL_VIEW_FLOATS = 20;
/** The grid a cascade's box is snapped to, in texels: see _fitCascade. */
const CASCADE_SNAP = 16;
/** A cascade's mode this frame: kept as it is, drawn whole, or scrolled. */
const KEEP = 0, FULL = 1, SCROLL = 2;
/**
 * Rectangles a scrolled cascade redraws, at most: the strip of new columns
 * and the strip of new rows, each split in two by the wrap on either axis.
 */
const MAX_PIECES = 8;
/** Floats a cascade gives the forward pass: its matrix, then where its map's origin sits. */
const CASCADE_VIEW_FLOATS = 20;

/**
 * How far past its own edge each local view reaches, as a tangent scale: the
 * 3x3 PCF reaches two texels beyond the one a fragment lands in, so a view of
 * `size` texels whose inner size - 4 cover the cone (or the cube face) keeps
 * every tap of every fragment inside its own map.
 */
export function localMargin(size) {
  return size / (size - 4);
}

/**
 * Whether a spot light takes one perspective view or six. A cone wider than a
 * cube face (outer angle past 45 degrees) is covered better by the six faces a
 * point light uses than by one frustum stretched towards 180 degrees, where
 * the texels at its rim grow without bound.
 */
export function spotViewCount(outerAngle) {
  return Math.tan(outerAngle) <= 1 ? 1 : 6;
}

/** Cube face axes, in the order the forward shader picks them: +x -x +y -y +z -z. */
const CUBE_AXES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];

/**
 * Depth-only for everything opaque: no fragment stage at all, half the work of
 * the forward pass. Casters shaped by alpha get the fragment stage they need.
 */
export const SHADOW_SHADER = /* wgsl */ `
struct Cascade {
  viewProjection : mat4x4<f32>,
};

struct DrawData {
  model         : mat4x4<f32>,
  normalMatrix  : mat3x3<f32>,
  // Unread by the unskinned path below, and NOT optional. WGSL sizes a struct
  // from its members, so leaving this out makes the shadow pass read a
  // 112-byte stride out of a buffer the CPU writes at 128 -- instance 0 lands
  // correctly and every one after it is misaligned, which is shadows in the
  // wrong places and no error anywhere.
  paletteOffset : u32,
  // Likewise unread here and likewise mandatory: the forward pass's DrawData
  // carries them, and a struct that disagreed would read the buffer at the
  // wrong stride.
  morphBase     : u32,
  morphWeights  : u32,
  morphCount    : u32,
};

struct Batch {
  firstVisible : u32,
};

@group(0) @binding(0) var<uniform> cascade : Cascade;
@group(0) @binding(1) var<storage, read> drawData : array<DrawData>;
// The shadow pass culls nothing, so it walks a static batch-ordered list
// rather than the compacted one the cull shader writes.
@group(0) @binding(2) var<storage, read> order : array<u32>;
@group(0) @binding(3) var<storage, read> palette : array<mat4x4<f32>>;
@group(0) @binding(4) var<storage, read> morphDeltas : array<f32>;
@group(0) @binding(5) var<storage, read> morphWeights : array<f32>;
@group(3) @binding(0) var<uniform> batch : Batch;

// The material, for the casters whose shape comes from alpha. Only the fields
// alpha needs are read, but the struct is the forward pass's, byte for byte.
${MATERIAL_WGSL}
@group(2) @binding(0) var<uniform> material     : Material;
@group(2) @binding(1) var          baseColorMap : texture_2d<f32>;
@group(2) @binding(6) var          surfSampler  : sampler;

// Positions only. Nothing here shades, so the normal and tangent deltas the
// forward pass reads would be fetched and discarded.
fn morphPosition(draw : DrawData, vertex : u32, position : vec3<f32>) -> vec3<f32> {
  let count = draw.morphCount & 0xffffu;
  if (count == 0u) { return position; }

  let stride = draw.morphCount >> 16u;
  var o = draw.morphBase + vertex * count * stride;
  var moved = position;

  for (var t = 0u; t < count; t = t + 1u) {
    let w = morphWeights[draw.morphWeights + t];
    if (w != 0.0) {
      moved = moved + w * vec3<f32>(morphDeltas[o], morphDeltas[o + 1u], morphDeltas[o + 2u]);
    }
    o = o + stride;
  }
  return moved;
}

// A rectangle of a scrolled cascade back to empty, ahead of its casters: one
// triangle over the whole map, cut to the rectangle by the scissor, writing
// the clear depth whatever was there.
@vertex
fn vsClear(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  return vec4<f32>(p, ${DEPTH_CLEAR_VALUE.toFixed(1)}, 1.0);
}

@vertex
fn vs(
  @builtin(instance_index) instance : u32,
  @builtin(vertex_index)   vertex   : u32,
  @location(0) position : vec3<f32>,
) -> @builtin(position) vec4<f32> {
  let draw = drawData[order[batch.firstVisible + instance]];
  let moved = morphPosition(draw, vertex, position);
  return cascade.viewProjection * draw.model * vec4<f32>(moved, 1.0);
}

struct AlphaOut {
  @builtin(position) clip  : vec4<f32>,
  @location(0)       uv    : vec2<f32>,
  @location(1)       uv1   : vec2<f32>,
  @location(2)       alpha : f32,
};

@vertex
fn vsAlpha(
  @builtin(instance_index) instance : u32,
  @builtin(vertex_index)   vertex   : u32,
  @location(0) position : vec3<f32>,
  @location(2) uv       : vec2<f32>,
  @location(4) uv1      : vec2<f32>,
  @location(5) color    : vec4<f32>,
) -> AlphaOut {
  let draw = drawData[order[batch.firstVisible + instance]];
  var out : AlphaOut;
  out.clip = cascade.viewProjection * draw.model * vec4<f32>(morphPosition(draw, vertex, position), 1.0);
  out.uv = uv;
  out.uv1 = uv1;
  out.alpha = color.a;
  return out;
}

@vertex
fn vsSkinnedAlpha(
  @builtin(instance_index) instance : u32,
  @builtin(vertex_index)   vertex   : u32,
  @location(0) position : vec3<f32>,
  @location(2) uv       : vec2<f32>,
  @location(4) uv1      : vec2<f32>,
  @location(5) color    : vec4<f32>,
  @location(6) joints   : vec4<u32>,
  @location(7) weights  : vec4<f32>,
) -> AlphaOut {
  let draw = drawData[order[batch.firstVisible + instance]];
  let base = draw.paletteOffset;
  let skin = palette[base + joints.x] * weights.x
           + palette[base + joints.y] * weights.y
           + palette[base + joints.z] * weights.z
           + palette[base + joints.w] * weights.w;
  var out : AlphaOut;
  out.clip = cascade.viewProjection * skin * vec4<f32>(morphPosition(draw, vertex, position), 1.0);
  out.uv = uv;
  out.uv1 = uv1;
  out.alpha = color.a;
  return out;
}

/** The forward pass's alpha, from the same three factors in the same UV set. */
fn surfaceAlpha(v : AlphaOut) -> f32 {
  let p = vec3<f32>(select(v.uv, v.uv1, (u32(material.uvSets) & 1u) != 0u), 1.0);
  let uv = vec2<f32>(dot(material.uvTransforms[0].xyz, p), dot(material.uvTransforms[1].xyz, p));
  return textureSample(baseColorMap, surfSampler, uv).a * material.baseColor.a * v.alpha;
}

// MASK: the forward pass's own cutoff, so the shadow has the texture's shape.
@fragment
fn fsMask(v : AlphaOut) {
  if (surfaceAlpha(v) < material.alphaCutoff) { discard; }
}

// BLEND: a texel is covered with probability alpha, against a fixed dither
// per shadow-map texel (interleaved gradient noise). The filtered lookup then
// averages neighbouring texels into a shadow as dark as the surface is opaque.
// Fixed to the texel, so a still light and a still caster give a still shadow.
@fragment
fn fsHashed(v : AlphaOut) {
  let threshold = fract(52.9829189 * fract(dot(v.clip.xy, vec2<f32>(0.06711056, 0.00583715))));
  if (surfaceAlpha(v) <= threshold) { discard; }
}

// The same skinning and morphing the forward pass does, for the same reason
// it has to: a caster that deforms and a shadow that does not is a character
// walking beside its own silhouette standing still.
@vertex
fn vsSkinned(
  @builtin(instance_index) instance : u32,
  @builtin(vertex_index)   vertex   : u32,
  @location(0) position : vec3<f32>,
  @location(6) joints   : vec4<u32>,
  @location(7) weights  : vec4<f32>,
) -> @builtin(position) vec4<f32> {
  let draw = drawData[order[batch.firstVisible + instance]];
  let base = draw.paletteOffset;

  let skin = palette[base + joints.x] * weights.x
           + palette[base + joints.y] * weights.y
           + palette[base + joints.z] * weights.z
           + palette[base + joints.w] * weights.w;

  // No draw.model, for the same reason the forward path omits it: the joints
  // place a skinned mesh entirely.
  return cascade.viewProjection * skin * vec4<f32>(morphPosition(draw, vertex, position), 1.0);
}
`;

/**
 * Practical split scheme (Zhang et al.): blend a uniform split, which wastes
 * resolution up close, with a logarithmic one, which wastes it far away.
 *
 * `lambda` of 0 is uniform, 1 is logarithmic. Around 0.5-0.8 is where a scene
 * with both a foreground and a horizon actually looks right.
 */
export function cascadeSplits(near, shadowDistance, count, lambda = 0.7) {
  if (DEBUG) {
    assert(count > 0 && count <= MAX_CASCADES, `cascade count ${count} out of range`);
    assert(shadowDistance > near, 'shadow distance must be beyond the near plane');
  }
  // Unconditional: the logarithmic term is near * (far/near)^p, so a near of
  // zero makes it 0 * Infinity -- NaN, even when lambda weights it to nothing.
  // The result would be a cascade matrix full of NaN, every shadow lookup
  // failing, and no error anywhere.
  if (!(near > 0)) throw new Error(`cascadeSplits: near must be positive, got ${near}`);
  const splits = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const p = (i + 1) / count;
    const uniform = near + (shadowDistance - near) * p;
    const logarithmic = near * (shadowDistance / near) ** p;
    splits[i] = lambda * logarithmic + (1 - lambda) * uniform;
  }
  return splits;
}

/**
 * Bounding sphere of one slice of the view frustum, in world space.
 *
 * Computed from the camera basis rather than by inverting a projection: there
 * is no finite projection to invert, and the corners are two
 * lines of trigonometry anyway.
 *
 * Writes {x,y,z,radius} into `out`.
 */
export function frustumSliceSphere(out, camera, nearDistance, farDistance) {
  const view = camera.view;
  // The view matrix's third row is the camera's +Z in world space; it looks
  // down -Z. Only the centre needs a world direction -- see the radius below.
  const fx = -view[2], fy = -view[6], fz = -view[10];

  const ex = camera.position[0], ey = camera.position[1], ez = camera.position[2];

  // Perspective widens with depth; an orthographic box is the same size at
  // both ends of the slice, so fitting it as a pyramid would leave the near
  // corners -- the part of the view closest to the camera -- unshadowed.
  const tanHalf = Math.tan(camera.fovY * 0.5);
  const nearH = camera.orthographic ? camera.orthographicHalfHeight() : tanHalf * nearDistance;
  const farH = camera.orthographic ? nearH : tanHalf * farDistance;
  const nearW = nearH * camera.aspect;
  const farW = farH * camera.aspect;

  // THE SMALLEST SPHERE, not the one about the centroid. By symmetry its
  // centre is on the view axis, at the depth c where a near corner and a far
  // corner are equally far away:
  //
  //   nearDiag^2 + (c - near)^2 = farDiag^2 + (far - c)^2
  //
  // unless that lands past the far plane, when the far corners' own circle
  // already holds the near ones and the centre stops there. The centroid used
  // before wasted 5-17% of each cascade's texels on space outside the slice.
  const nearDiag2 = nearW * nearW + nearH * nearH;
  const farDiag2 = farW * farW + farH * farH;
  const depth = Math.min(
    (farDiag2 - nearDiag2 + farDistance * farDistance - nearDistance * nearDistance)
      / (2 * (farDistance - nearDistance)),
    farDistance,
  );
  const cx = ex + fx * depth;
  const cy = ey + fy * depth;
  const cz = ez + fz * depth;

  // Radius: the distance from that centre to the farthest corner, worked out
  // in the camera's own frame, where a corner of the plane at distance d is
  // (+-W, +-H, d) and the centre is (0, 0, mid). All four corners of a plane
  // are equally far, so one per plane is enough. Computed from those scalars
  // rather than from world-space corners, because the rotation-invariance the
  // cascades depend on then holds EXACTLY: the world-space form rotated every
  // corner through the camera basis and came back differing in the last bit
  // as the camera turned, resizing the cascade by a rounding error.
  const nearDz = nearDistance - depth;
  const farDz = farDistance - depth;
  const radius = Math.sqrt(Math.max(nearDiag2 + nearDz * nearDz, farDiag2 + farDz * farDz));

  out[0] = cx; out[1] = cy; out[2] = cz; out[3] = radius;
  return out;
}

/**
 * The shadow range when nothing pins it: far enough to reach the whole scene,
 * rounded UP to a power of two.
 *
 * The cascades are only stable -- moving without resizing, so texel snapping
 * can hold their edges still -- while this number holds still. It used to be
 * the scene's farthest point along the VIEW direction, recomputed every
 * frame, so turning the camera in place resized every cascade: 15.0, 15.5,
 * 16.0 over three degrees of yaw, and the shadows crawled.
 *
 * Two things fix that. The reach is RADIAL -- from where the camera is, not
 * along where it looks -- so turning changes nothing. And it is rounded up to
 * a power of two, so walking changes it only at the doublings rather than
 * continuously. The price is resolution: up to twice the distance the scene
 * strictly needs. Pinning shadowDistance trades that back.
 *
 * `floor` is the smallest legal range (the renderer passes twice the near
 * plane), for a scene with nothing in it.
 */
export function stableShadowDistance(reach, floor) {
  const needed = Math.max(reach, floor);
  return Math.max(2 ** Math.ceil(Math.log2(needed)), floor);
}

/**
 * Whether a mesh has an edge that only one triangle uses: an open surface,
 * like a plane, a roof of one sheet, a terrain tile. The shadow pass draws a
 * closed one's back faces, which keeps the faces it shades free of acne; an
 * open one may have no back face toward the light, so it is drawn whole.
 * Vertices are welded by position first: a closed mesh split along its UV
 * seams, or for flat normals, is still closed.
 */
export function openSurface(positions, indices) {
  // Typed hash tables, open addressing: the strings and Maps this began with
  // took 400 ms on Sponza's 262k triangles.
  const bits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
  const vertices = positions.length / 3;
  const weld = new Uint32Array(vertices);
  let size = 1 << Math.ceil(Math.log2(vertices * 2 + 2));
  const table = new Int32Array(size).fill(-1);
  // Minus zero is zero: a seam on an axis is not a hole.
  const plain = (b) => (b === 0x80000000 ? 0 : b);
  for (let v = 0; v < vertices; v++) {
    const x = plain(bits[v * 3]), y = plain(bits[v * 3 + 1]), z = plain(bits[v * 3 + 2]);
    let h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & (size - 1);
    for (;;) {
      const at = table[h];
      if (at < 0) {
        table[h] = v;
        weld[v] = v;
        break;
      }
      if (plain(bits[at * 3]) === x && plain(bits[at * 3 + 1]) === y && plain(bits[at * 3 + 2]) === z) {
        weld[v] = weld[at];
        break;
      }
      h = (h + 1) & (size - 1);
    }
  }
  // Each edge by its two welded ends, lower first; counted to two.
  const triangles = Math.floor(indices.length / 3);
  size = 1 << Math.ceil(Math.log2(triangles * 3 * 2 + 2));
  const lo = new Int32Array(size).fill(-1);
  const hi = new Int32Array(size);
  const uses = new Uint8Array(size);
  const count = (a, b) => {
    if (a > b) { const t = a; a = b; b = t; }
    let h = (Math.imul(a, 73856093) ^ Math.imul(b, 19349663)) & (size - 1);
    for (;;) {
      if (lo[h] < 0) { lo[h] = a; hi[h] = b; uses[h] = 1; return; }
      if (lo[h] === a && hi[h] === b) { if (uses[h] < 2) uses[h]++; return; }
      h = (h + 1) & (size - 1);
    }
  };
  for (let t = 0; t < triangles * 3; t += 3) {
    const a = weld[indices[t]], b = weld[indices[t + 1]], c = weld[indices[t + 2]];
    if (a === b || b === c || c === a) continue;   // degenerate: no edges to speak of
    count(a, b);
    count(b, c);
    count(c, a);
  }
  for (let h = 0; h < size; h++) if (uses[h] === 1) return true;
  return false;
}

export class ShadowMaps {
  static async create(rhi, pipelines, drawLayout, options = {}, materialLayout = null) {
    const maps = new ShadowMaps(rhi, options);
    await maps._init(pipelines, drawLayout, materialLayout);
    return maps;
  }

  constructor(rhi, {
    size = 2048,
    cascades = 4,
    shadowDistance = 60,
    lambda = 0.7,
    /**
     * How far behind the slice to pull the light's near plane, as a multiple of
     * the slice radius. Casters outside the visible slice still cast INTO it --
     * a tall building behind you shadows the street ahead. Too small and those
     * shadows vanish; too large and depth precision is spent on empty space.
     */
    casterExtent = 4,
    /** World-space offset along the surface normal, in texels, at lookup time. */
    normalBias = 1.5,
    /** Hardware slope-scaled bias applied while rendering the map. */
    depthBiasSlope = -2.0,
    depthBiasConstant = -1,
    /**
     * Texels on a side of each point or spot shadow view. A budget, not a
     * derivation: a point light is six of these, at four bytes a texel, so
     * 512 is 6 MB a point light. 512 is also three.js's default.
     */
    localSize = 512,
  } = {}) {
    // Unconditional: past MAX_CASCADES the per-cascade arrays and the cascade
    // uniform are overrun, and the shadows are wrong with nothing reported.
    if (!(cascades > 0 && cascades <= MAX_CASCADES)) {
      throw new RangeError(`Shadows: ${cascades} cascades; 1 to ${MAX_CASCADES} are supported`);
    }
    const maxSize = rhi.limits.maxTextureDimension2D;
    if (size > maxSize) {
      throw new RangeError(`Shadows: a ${size} map is past this device's ${maxSize}`);
    }
    if (!(localSize > 4 && localSize <= maxSize)) {
      throw new RangeError(`Shadows: a local map of ${localSize} is outside 5 to this device's ${maxSize}`);
    }
    this.localSize = localSize;
    /**
     * What each run of local layers last drew, by its first layer: the light
     * and its views exactly. A light whose views match, and that nothing
     * changed near, keeps its maps from that draw; see updateLocal.
     */
    this._cache = [];
    /** Per local view this frame: 1 to draw it, 0 to keep what the layer holds. */
    this.localRedraw = new Uint8Array(0);
    /** How many local views were drawn this frame, of localCount. */
    this.localDrawn = 0;
    /** Bumped when the local array is replaced, which loses every map in it. */
    this._localGeneration = 0;
    /** Counts updateLocal calls, so a record knows whether it was checked in the last one. */
    this._frame = 0;
    // The same, for the cascades: what each layer last drew, by layer, and
    // which to draw this frame. A cascade's matrix is fitted to a texel grid
    // fixed in the world, so it comes out the same while the camera holds
    // still -- and so does its map, unless something moved inside it.
    this._cascadeCache = [];
    this.cascadeRedraw = new Uint8Array(0);
    /** How many cascade layers were drawn this frame. */
    this.cascadesDrawn = 0;
    this._cascadeGeneration = 0;
    this._cascadeFrame = 0;

    this.rhi = rhi;
    this.size = size;
    this.cascadeCount = cascades;
    this._batchBoxes = new Float32Array(0);
    /**
     * Cascades fitted this frame. Zero when there is no sun to fit them to,
     * which is what keeps the shadow passes off the graph entirely.
     */
    this.activeCascades = cascades;
    this.shadowDistance = shadowDistance;
    this.lambda = lambda;
    this.casterExtent = casterExtent;
    this.normalBias = normalBias;
    this.depthBiasSlope = depthBiasSlope;
    this.depthBiasConstant = depthBiasConstant;

    // The cascade array is made in _init and grown with the lights that cast:
    // `cascades` layers for each, light by light.
    this.cascadeCapacity = 0;
    /** Directional lights casting this frame. */
    this.shadowedCount = 0;

    // A comparison sampler does the depth test AND the bilinear blend in one
    // fetch, so a 3x3 PCF kernel costs 9 taps of already-filtered results
    // rather than 9 raw reads plus the comparisons by hand.
    this.sampler = rhi.device.createSampler({
      label: 'shadow-compare',
      compare: DEPTH_COMPARE,
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    // The cascades wrap. One that moves keeps its texels where they lie and
    // draws only the strip it moved onto (see _redrawCascade), so the map's
    // origin wanders and a lookup wraps round its edge instead of clamping.
    this.cascadeSampler = rhi.device.createSampler({
      label: 'shadow-compare-cascades',
      compare: DEPTH_COMPARE,
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });
    /** Cascades scrolled this frame rather than drawn whole. */
    this.cascadesScrolled = 0;

    /** Every cascade of every casting light, a mat4 each, light by light. */
    this.matrices = new Float32Array(0);
    /** Shared by every light: the slices depend on the camera alone. */
    this.splits = new Float32Array(MAX_CASCADES);
    /** World size of one shadow texel, per cascade. The normal-offset bias
     *  scales with it, so a coarse far cascade biases more than a fine near one. */
    this.texelSizes = new Float32Array(MAX_CASCADES);

    this._lightView = mat4Create();
    this._depthRange = new Float64Array(2);
    /** Per layer, the rectangles a scroll redraws: see _redrawCascade. */
    this._pieces = [];
    this._projection = mat4Create();
    this._spheres = Array.from({ length: MAX_CASCADES }, () => new Float32Array(4));
    this._lightDirection = vec3Create(0, -1, 0);
    this._eye = vec3Create();
    this._target = vec3Create();
    this._up = vec3Create();
  }

  async _init(pipelines, drawLayout, materialLayout = null) {
    const rhi = this.rhi;
    this.shader = await compileShader(rhi.device, SHADOW_SHADER, 'shadow.wgsl');

    this.cascadeLayout = rhi.device.createBindGroupLayout({
      label: 'shadow-cascade',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 },
        },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    // `order` is the static batch order, or with LOD the shadow cull's slice.
    this._makeCascadeBindGroup = (gpu, palette, morph, buffer = this.cascadeBuffer) => rhi.device.createBindGroup({
      label: 'shadow-cascade',
      layout: this.cascadeLayout,
      entries: [
        { binding: 0, resource: { buffer, size: 64 } },
        { binding: 1, resource: { buffer: gpu.drawDataBuffer } },
        { binding: 2, resource: { buffer: gpu.hasLod ? gpu.visibleBuffer : gpu.batchOrderBuffer } },
        { binding: 3, resource: { buffer: palette.buffer } },
        { binding: 4, resource: { buffer: morph.deltaBuffer } },
        { binding: 5, resource: { buffer: morph.weightBuffer } },
      ],
    });

    this.pipelineLayout = createPipelineLayout(rhi.device, {
      [GROUP_FRAME]: this.cascadeLayout,
      [GROUP_DRAW]: drawLayout,
    }, 'shadow');

    this.descriptor = {
      label: 'shadow',
      layout: this.pipelineLayout,
      shader: this.shader,
      vertexEntry: 'vs',
      // No fragment stage: nothing is written but depth.
      fragmentEntry: undefined,
      buffers: [VERTEX_BUFFER_LAYOUT],
      targets: [],
      primitive: {
        topology: 'triangle-list',
        // FRONT faces are culled, not back. Rendering only backfaces pushes the
        // recorded depth to the far side of the object, which moves the whole
        // acne problem behind the surface that would show it. Costs correct
        // shadows on open geometry, which is why it is a knob, not a law.
        cullMode: 'front',
        frontFace: 'ccw',
      },
      depth: {
        format: DEPTH_FORMAT,
        depthCompare: DEPTH_COMPARE,
        depthWriteEnabled: true,
        // Slope-scaled: a surface almost edge-on to the light spans many depth
        // values in one texel and needs far more bias than one facing it. A
        // constant bias large enough for the worst case would peter-pan
        // everything else.
        depthBias: this.depthBiasConstant,
        depthBiasSlopeScale: this.depthBiasSlope,
      },
    };

    // The same pipeline with the skinning vertex path and the influence buffer
    // at slot 1. Everything else -- the front-face cull, the bias, the absent
    // fragment stage -- is identical, because a skinned caster casts the same
    // kind of shadow.
    this.skinnedDescriptor = {
      ...this.descriptor,
      label: 'shadow-skinned',
      vertexEntry: 'vsSkinned',
      buffers: [VERTEX_BUFFER_LAYOUT, SKIN_BUFFER_LAYOUT],
    };
    this._pipelines = pipelines;
    // And both again with the winding reversed, for mirrored instances --
    // a negative-determinant world matrix, which the forward pass handles
    // with a 'cw' pipeline. Here there was one pipeline for everything, so a
    // mirrored caster's "front" cull removed its real BACK faces and kept the
    // ones facing the light: the acne front-face culling exists to prevent.
    const mirror = (descriptor) => ({
      ...descriptor,
      label: `${descriptor.label}-mirrored`,
      primitive: { ...descriptor.primitive, frontFace: 'cw' },
    });
    // And all four culling nothing, for double-sided materials and open
    // surfaces (openSurface). Front-face culling assumes a closed surface,
    // whose back faces stand in for its front; an open one -- a leaf, a sail,
    // a sheet, a roof of one plane -- may have nothing facing away from the
    // light, so culling the side facing it took the whole shadow with it.
    const twoSided = (descriptor) => ({
      ...descriptor,
      label: `${descriptor.label}-double-sided`,
      primitive: { ...descriptor.primitive, cullMode: 'none' },
    });
    // Indexed by doubleSided * 4 + skinned * 2 + mirrored.
    const oneSided = [
      this.descriptor, mirror(this.descriptor),
      this.skinnedDescriptor, mirror(this.skinnedDescriptor),
    ];
    this.descriptors = [...oneSided, ...oneSided.map(twoSided)];

    // Casters whose shape comes from alpha: MASK batches, tested against the
    // material's cutoff, and blended items, hashed. Both need the material, so
    // they have their own layout, and a fragment stage that writes nothing.
    // Culling nothing: alpha-shaped geometry is leaves, panes and cards -- open
    // surfaces, where front-face culling would drop whichever side faces the
    // light, and with it the whole shadow. Winding is then moot, so there is
    // no mirrored copy either. Indexed by hashed * 2 + skinned.
    this.alphaDescriptors = [];
    if (materialLayout !== null) {
      const alphaLayout = createPipelineLayout(rhi.device, {
        [GROUP_FRAME]: this.cascadeLayout,
        [GROUP_MATERIAL]: materialLayout,
        [GROUP_DRAW]: drawLayout,
      }, 'shadow-alpha');
      for (const hashed of [false, true]) {
        for (const skinned of [false, true]) {
          this.alphaDescriptors.push({
            ...this.descriptor,
            label: `shadow-${hashed ? 'hashed' : 'mask'}${skinned ? '-skinned' : ''}`,
            layout: alphaLayout,
            vertexEntry: skinned ? 'vsSkinnedAlpha' : 'vsAlpha',
            fragmentEntry: hashed ? 'fsHashed' : 'fsMask',
            buffers: skinned ? [VERTEX_BUFFER_LAYOUT, SKIN_BUFFER_LAYOUT] : [VERTEX_BUFFER_LAYOUT],
            primitive: { ...this.descriptor.primitive, cullMode: 'none' },
          });
        }
      }
    }
    this.clearDescriptor = {
      ...this.descriptor,
      label: 'shadow-clear',
      layout: createPipelineLayout(rhi.device, {}, 'shadow-clear'),
      vertexEntry: 'vsClear',
      buffers: [],
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depth: { format: DEPTH_FORMAT, depthCompare: 'always', depthWriteEnabled: true },
    };
    await pipelines.warm([...this.descriptors, ...this.alphaDescriptors, this.clearDescriptor]);

    // One bound executor per layer, built as the arrays grow. The graph stores
    // a function per pass, and building them per frame would allocate a
    // closure per pass per frame for no reason.
    this._executors = [];

    // Local views: none yet, but the forward pass binds the array and the view
    // list whether or not a light uses them, so both exist from the start.
    this.localCount = 0;
    this.localCapacity = 0;
    /** Bumped when either array or its buffers are replaced: bind groups rebuild. */
    this.revision = 0;
    this._localExecutors = [];
    this._frustum = frustumCreate();
    this._growCascades(this.cascadeCount);
    this._growLocal(1);
  }

  /**
   * Room for `needed` cascade layers: the depth array, a slot each in the
   * render uniform, and a matrix each in the list the forward pass reads. The
   * ceiling is the device's array-layer limit, read from the adapter.
   */
  _growCascades(needed) {
    if (needed <= this.cascadeCapacity) return;
    const rhi = this.rhi;
    const capacity = grownCapacity(this.cascadeCapacity, needed, rhi.limits.maxTextureArrayLayers, 'directional shadow cascades');
    this.texture?.destroy();
    this.cascadeBuffer?.destroy();
    this.cascadeList?.destroy();
    this._cascadeCache = [];
    this._cascadeGeneration++;
    this.texture = createTexture(rhi, {
      label: 'shadow-cascades',
      size: [this.size, this.size, capacity],
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.view = this.texture.createView({ dimension: '2d-array', label: 'shadow-cascades' });
    this.layerViews = [];
    for (let i = 0; i < capacity; i++) {
      this.layerViews.push(this.texture.createView({
        dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1, label: `cascade-${i}`,
      }));
    }
    // A slot a layer, then MAX_PIECES a layer for the rectangles a scroll draws.
    this.cascadeBuffer = createBuffer(rhi, {
      label: 'shadow-cascades',
      size: this.alignment * capacity * (1 + MAX_PIECES),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Grown mid-frame, between one light's cascades and the next: what the
    // lights before it already wrote comes across, or their shadows are lost
    // for the frame the array grew in.
    const staging = new ArrayBuffer(this.alignment * capacity * (1 + MAX_PIECES));
    if (this.cascadeStaging) new Uint8Array(staging).set(new Uint8Array(this.cascadeStaging));
    this.cascadeStaging = staging;
    this._cascadeStagingF32 = new Float32Array(this.cascadeStaging);
    this.cascadeList = createBuffer(rhi, {
      label: 'shadow-cascade-list',
      size: capacity * CASCADE_VIEW_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const matrices = new Float32Array(capacity * 16);
    matrices.set(this.matrices);
    this.matrices = matrices;
    const views = new Float32Array(capacity * CASCADE_VIEW_FLOATS);
    views.set(this._views ?? []);
    this._views = views;
    // Each layer's fit, for the next frame to compare against: the grid cell
    // its box sits on (x, y), its texel, and its depth range (near, far).
    const fits = new Float64Array(capacity * 5);
    fits.set(this._fits ?? []);
    this._fits = fits;
    for (let layer = this._executors.length; layer < capacity; layer++) {
      this._pieces.push({ count: 0, rects: new Int32Array(MAX_PIECES * 4), cull: new Float32Array(MAX_PIECES * 16) });
      this._executors.push((pass) => this._encodeCascade(pass, layer));
    }
    this.cascadeCapacity = capacity;
    this.cascadeBindGroup = undefined;
    this.revision++;
  }

  /**
   * Room for `needed` local views: layers of the depth array, a slot each in
   * the render uniform, and a view each in the list the forward pass reads.
   * The ceiling is the device's array-layer limit, read from the adapter.
   */
  _growLocal(needed) {
    if (needed <= this.localCapacity) return;
    const rhi = this.rhi;
    const capacity = grownCapacity(this.localCapacity, needed, rhi.limits.maxTextureArrayLayers, 'point and spot shadow views');
    this.localTexture?.destroy();
    this.localUniform?.destroy();
    this.localBuffer?.destroy();
    this._cache = [];
    this._localGeneration++;
    const size = this.localSize;
    this.localTexture = createTexture(rhi, {
      label: 'shadow-local',
      size: [size, size, capacity],
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.localView = this.localTexture.createView({ dimension: '2d-array', label: 'shadow-local' });
    this.localLayerViews = [];
    for (let i = 0; i < capacity; i++) {
      this.localLayerViews.push(this.localTexture.createView({
        dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1, label: `shadow-local-${i}`,
      }));
    }
    this.localUniform = createBuffer(rhi, {
      label: 'shadow-local-views',
      size: this.alignment * capacity,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // As with the cascades: views written before the growth come across.
    const staging = new ArrayBuffer(this.alignment * capacity);
    if (this.localStaging) new Uint8Array(staging).set(new Uint8Array(this.localStaging));
    this.localStaging = staging;
    this._localStagingF32 = new Float32Array(this.localStaging);
    this.localBuffer = createBuffer(rhi, {
      label: 'shadow-local-list',
      size: capacity * LOCAL_VIEW_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const data = new Float32Array(capacity * LOCAL_VIEW_FLOATS);
    if (this.localData) data.set(this.localData);
    this.localData = data;
    for (let v = this._localExecutors.length; v < capacity; v++) {
      this._localExecutors.push((pass) => this._encodeView(pass, this.localBindGroup, v * this.alignment));
    }
    this.localCapacity = capacity;
    this.localBindGroup = undefined;
    this.revision++;
  }

  /**
   * Views for every point and spot light that casts a shadow and whose reach
   * is on screen, written into the light records the forward pass reads:
   * directionCone.w is how many views (1 or 6) and coneFalloff.w the first
   * layer plus one, zero for none. Runs after scene.refreshLights, which is
   * where a light's position and aim for this frame come from, and before the
   * light list is uploaded.
   *
   * A light whose sphere is off screen gets no views: nothing it lights is
   * drawn, so nothing would read them.
   *
   * CACHED. A light's maps are drawn again only when they would come out
   * different: its views changed, it landed on other layers, or something in
   * its reach changed -- `changes.boxes` holds every box that moved or
   * deformed this frame, before and after, and `changes.all` says the scene
   * changed in a way no box describes. Only a caster inside the light's
   * sphere can shadow anything it lights, since it has to sit between the
   * light and a receiver the light reaches, so a box outside the sphere
   * cannot change what the maps say about anything that matters. Without
   * `changes`, everything is drawn.
   */
  updateLocal(scene, camera, changes = null) {
    frustumFromViewProjection(this._frustum, camera.viewProjection);
    const lights = scene.lights;
    const casters = scene.shadowCasters;
    const margin = localMargin(this.localSize);
    const generation = this._localGeneration;
    let count = 0;
    let drawn = 0;
    for (let i = 0; i < scene.lightCount; i++) {
      const o = i * LIGHT_FLOATS;
      lights[o + 11] = 0;
      lights[o + 15] = 0;
      if (casters.size === 0 || !casters.has(scene.lightEntity[i])) continue;
      const radius = lights[o + 3];
      this._eye[0] = lights[o]; this._eye[1] = lights[o + 1]; this._eye[2] = lights[o + 2];
      if (!(radius > 0) || !frustumTestSphere(this._frustum, this._eye, radius)) continue;

      const views = lights[o + 14] === LIGHT_SPOT ? spotViewCount(scene._lightCone[i * 2 + 1]) : 6;
      this._growLocal(count + views);
      // Near only clips here: reverse-Z float depth is relative, so precision
      // does not depend on it. A thousandth of the reach clips only casters
      // inside the bulb.
      const near = radius / 1024;
      const size = scene.lightSizes.get(scene.lightEntity[i]) ?? 0;
      if (views === 1) {
        const tanHalf = Math.tan(scene._lightCone[i * 2 + 1]) * margin;
        this._localView(count, this._eye, lights[o + 8], lights[o + 9], lights[o + 10], tanHalf, near, size);
      } else {
        for (let face = 0; face < 6; face++) {
          const [x, y, z] = CUBE_AXES[face];
          this._localView(count + face, this._eye, x, y, z, margin, near, size);
        }
      }
      lights[o + 11] = views;
      lights[o + 15] = count + 1;
      if (this._redrawLight(count, views, scene.lightEntity[i], radius, changes)) drawn += views;
      count += views;
    }
    this.localCount = count;
    // A record not checked this frame missed this frame's changes, so it can
    // never be trusted again: a light off screen for a while, or moved to
    // other layers, draws afresh when it comes back.
    for (let l = 0; l < this._cache.length; l++) {
      if (this._cache[l] !== undefined && this._cache[l].frame !== this._frame) this._cache[l] = undefined;
    }
    this._frame++;
    // Growing the array this frame lost every map in it, those decided above included.
    if (this._localGeneration !== generation) {
      this.localRedraw.fill(1, 0, count);
      drawn = count;
    }
    this.localDrawn = drawn;
    if (count > 0) {
      this.rhi.queue.writeBuffer(this.localUniform, 0, this.localStaging, 0, this.alignment * count);
      this.rhi.queue.writeBuffer(this.localBuffer, 0, this.localData, 0, count * LOCAL_VIEW_FLOATS);
    }
  }

  /**
   * Whether the light whose views fill layers first..first+views-1 must be
   * drawn this frame, marking them in localRedraw and remembering the draw.
   */
  _redrawLight(first, views, entity, radius, changes) {
    if (this.localRedraw.length < this.localCapacity) {
      const grown = new Uint8Array(this.localCapacity);
      grown.set(this.localRedraw);
      this.localRedraw = grown;
    }
    const floats = views * LOCAL_VIEW_FLOATS;
    const at = first * LOCAL_VIEW_FLOATS;
    const cached = this._cache[first];
    let redraw = changes === null || changes.all || cached === undefined
      || cached.entity !== entity || cached.views !== views
      || changes.boxes.touchesSphere(this._eye[0], this._eye[1], this._eye[2], radius);
    for (let k = 0; !redraw && k < floats; k++) redraw = cached.data[k] !== this.localData[at + k];
    this.localRedraw.fill(redraw ? 1 : 0, first, first + views);
    if (!redraw) cached.frame = this._frame;
    if (redraw) {
      const data = cached?.data.length === floats ? cached.data : new Float32Array(floats);
      data.set(this.localData.subarray(at, at + floats));
      this._cache[first] = { entity, views, data, frame: this._frame };
      // Layers inside this run held other runs' maps, which are gone now.
      for (let l = first + 1; l < first + views; l++) this._cache[l] = undefined;
    }
    return redraw;
  }

  /**
   * One perspective view from `eye` along (x, y, z), square, at `tanHalf`.
   * Its params are what the shader needs beside the matrix: tanHalf, near (a
   * stored depth is near / distance), and the light's size, for soft shadows.
   */
  _localView(index, eye, x, y, z, tanHalf, near, size = 0) {
    this._lightDirection[0] = x; this._lightDirection[1] = y; this._lightDirection[2] = z;
    vec3Normalize(this._lightDirection, this._lightDirection);
    this._target[0] = eye[0] + this._lightDirection[0];
    this._target[1] = eye[1] + this._lightDirection[1];
    this._target[2] = eye[2] + this._lightDirection[2];
    mat4LookAt(this._lightView, eye, this._target, chooseUp(this._up, this._lightDirection));
    mat4PerspectiveReverseZInfinite(this._projection, 2 * Math.atan(tanHalf), 1, near);
    const at = index * LOCAL_VIEW_FLOATS;
    mat4Multiply(this.localData, this._projection, this._lightView, at);
    this._localStagingF32.set(this.localData.subarray(at, at + 16), index * this.alignment / 4);
    this.localData[index * LOCAL_VIEW_FLOATS + 16] = tanHalf;
    this.localData[index * LOCAL_VIEW_FLOATS + 17] = near;
    this.localData[index * LOCAL_VIEW_FLOATS + 18] = size;
  }

  /**
   * Fit the cascades of every directional light that casts, and upload them.
   * Call once per frame, after scene.refreshLights and before the directional
   * lights are uploaded: each casting light's record gets its slot in
   * direction.w -- slot + 1, zero for none -- and its cascades are layers
   * slot * cascades + i of the array.
   *
   * The slices, their spheres and their texel sizes depend on the camera
   * alone, so every light shares them; only the matrices are per light.
   */
  update(camera, scene, changes = null, sceneMin = null, sceneMax = null) {
    const count = this.cascadeCount;
    const generation = this._cascadeGeneration;
    let drawn = 0;
    let scrolled = 0;
    const splits = cascadeSplits(camera.near, this.shadowDistance, count, this.lambda);
    let sliceNear = camera.near;
    for (let i = 0; i < count; i++) {
      frustumSliceSphere(this._spheres[i], camera, sliceNear, splits[i]);
      this.splits[i] = splits[i];
      sliceNear = splits[i];
    }
    // Unused cascades get a split of 0, not Infinity. selectCascade tests
    // `viewDepth < split`, and every depth is below Infinity, so the obvious
    // sentinel picks the cascade it was meant to skip.
    for (let i = count; i < MAX_CASCADES; i++) this.splits[i] = 0;

    const directionals = scene.directionals;
    const casters = scene.shadowCasters;
    let lights = 0;
    for (let d = 0; d < scene.directionalCount; d++) {
      const o = d * DIRECTIONAL_FLOATS;
      directionals[o + 3] = 0;
      if (casters.size === 0 || !casters.has(scene.directionalEntity[d])) continue;
      // A zero direction would make eye === target in the look-at and fill
      // every matrix with NaN, which poisons the lighting rather than failing.
      // A node scaled to nothing is the way to get one; it casts nothing.
      const x = directionals[o], y = directionals[o + 1], z = directionals[o + 2];
      if (x * x + y * y + z * z === 0) continue;
      this._growCascades((lights + 1) * count);
      this._lightDirection[0] = x; this._lightDirection[1] = y; this._lightDirection[2] = z;
      vec3Normalize(this._lightDirection, this._lightDirection);
      this._lightBasis();
      const depth = sceneMin === null ? null : this._sceneDepth(sceneMin, sceneMax);
      for (let c = 0; c < count; c++) {
        this._fitCascade(lights * count + c, c, depth);
        const mode = this._redrawCascade(lights * count + c, scene.directionalEntity[d], changes);
        if (mode === FULL) drawn++;
        else if (mode === SCROLL) scrolled++;
      }
      directionals[o + 3] = lights + 1;
      lights++;
    }
    this.shadowedCount = lights;
    // Zero keeps every shadow pass off the graph and every lookup lit.
    this.activeCascades = lights > 0 ? count : 0;
    const layers = lights * count;
    // As for local views: a record not checked this frame is never trusted
    // again, and growing the array lost every map, those decided above too.
    for (let l = 0; l < this._cascadeCache.length; l++) {
      if (this._cascadeCache[l] !== undefined && this._cascadeCache[l].frame !== this._cascadeFrame) this._cascadeCache[l] = undefined;
    }
    this._cascadeFrame++;
    if (this._cascadeGeneration !== generation) {
      // Growing replaced the array: every map starts again, its origin with it.
      this.cascadeRedraw.fill(FULL, 0, layers);
      for (let l = 0; l < layers; l++) this._views.fill(0, l * CASCADE_VIEW_FLOATS + 16, (l + 1) * CASCADE_VIEW_FLOATS);
      drawn = layers;
      scrolled = 0;
    }
    this.cascadesDrawn = drawn;
    this.cascadesScrolled = scrolled;
    if (lights === 0) return;
    for (let l = 0; l < layers; l++) this._views.set(this.matrices.subarray(l * 16, l * 16 + 16), l * CASCADE_VIEW_FLOATS);
    // The layers' own slots, then the slots of their pieces, which sit past every layer's.
    this.rhi.queue.writeBuffer(this.cascadeBuffer, 0, this.cascadeStaging, 0,
      this.alignment * (this.cascadeCapacity + layers * MAX_PIECES));
    this.rhi.queue.writeBuffer(this.cascadeList, 0, this._views, 0, layers * CASCADE_VIEW_FLOATS);
  }

  /** The light-space basis for _lightDirection, anchored at the world origin: see _fitCascade. */
  _lightBasis() {
    const d = this._lightDirection;
    chooseUp(this._up, d);
    this._eye.set([0, 0, 0]);
    this._target.set([d[0], d[1], d[2]]);
    mat4LookAt(this._lightView, this._eye, this._target, this._up);
  }

  /**
   * The scene's extent along the light, as light-space z [low, high], rounded
   * out to a step of a power of two near a quarter of it. Every cascade takes
   * its depth range from this rather than from where the camera is: a range
   * that followed the camera changed whenever it moved, and a changed range
   * means every texel already in the map is wrong, so no cascade could ever
   * be scrolled. Rounded, so a mover at the scene's edge changes it rarely.
   * It also means every caster in the scene casts, however far up the light
   * it stands -- `casterExtent` used to cut those off.
   */
  _sceneDepth(min, max) {
    const v = this._lightView;
    let low = Infinity, high = -Infinity;
    for (let corner = 0; corner < 8; corner++) {
      const x = corner & 1 ? max[0] : min[0], y = corner & 2 ? max[1] : min[1], z = corner & 4 ? max[2] : min[2];
      const lz = v[2] * x + v[6] * y + v[10] * z + v[14];
      if (lz < low) low = lz;
      if (lz > high) high = lz;
    }
    const step = 2 ** Math.ceil(Math.log2(Math.max((high - low) / 4, 1e-3)));
    this._depthRange[0] = Math.floor(low / step) * step - step;
    this._depthRange[1] = Math.ceil(high / step) * step + step;
    return this._depthRange;
  }

  /**
   * What cascade layer `layer`, just fitted for the light `entity`, needs:
   * KEEP, it is as it was; SCROLL, its box moved along the grid and nothing
   * else changed; or FULL, a new map -- no record, another light's, a turned
   * light, a new texel or depth range, a jump of half the map, or something
   * changed inside what it draws.
   *
   * A scroll is the camera walking. The map keeps every texel where it lies
   * and only its origin moves: a world point's texel is (u + origin) mod size,
   * so the strip the box moved onto falls exactly where the strip it left
   * was, and that strip is all that is drawn. Sponza at 720p on Iris Xe,
   * walking: each redraw of a near cascade was 1.5 to 1.9 ms.
   */
  _redrawCascade(layer, entity, changes) {
    if (this.cascadeRedraw.length < this.cascadeCapacity) {
      const grown = new Uint8Array(this.cascadeCapacity);
      grown.set(this.cascadeRedraw);
      this.cascadeRedraw = grown;
    }
    const at = layer * 16;
    const f = layer * 5;
    const fits = this._fits;
    const cached = this._cascadeCache[layer];
    const d = this._lightDirection;
    let mode = changes === null || changes.all || cached === undefined || cached.entity !== entity
      || cached.fit[2] !== fits[f + 2] || cached.fit[3] !== fits[f + 3] || cached.fit[4] !== fits[f + 4]
      || cached.direction[0] !== d[0] || cached.direction[1] !== d[1] || cached.direction[2] !== d[2]
      || changes.boxes.touchesOrtho(this.matrices, at) ? FULL : KEEP;
    // Grid steps the box moved, then texels: u runs with light-space x, v against y.
    const du = mode === FULL ? 0 : (fits[f] - cached.fit[0]) * CASCADE_SNAP;
    const dv = mode === FULL ? 0 : -(fits[f + 1] - cached.fit[1]) * CASCADE_SNAP;
    if (mode === KEEP && (du !== 0 || dv !== 0)) {
      mode = Math.abs(du) * 2 >= this.size || Math.abs(dv) * 2 >= this.size ? FULL : SCROLL;
    }
    this.cascadeRedraw[layer] = mode;
    const view = layer * CASCADE_VIEW_FLOATS + 16;
    // The origin, in uv and again in texels: the forward pass samples with
    // one and loads with the other.
    if (mode === FULL) {
      this._views.fill(0, view, view + 4);
    } else if (mode === SCROLL) {
      const size = this.size;
      const ou = (((this._views[view + 2] + du) % size) + size) % size;
      const ov = (((this._views[view + 3] + dv) % size) + size) % size;
      this._views[view] = ou / size;
      this._views[view + 1] = ov / size;
      this._views[view + 2] = ou;
      this._views[view + 3] = ov;
      this._scrollPieces(layer, du, dv, ou, ov);
    }
    if (mode !== KEEP) {
      const record = cached ?? { fit: new Float64Array(5), direction: new Float32Array(3) };
      record.entity = entity;
      record.fit.set(fits.subarray(f, f + 5));
      record.direction.set(d);
      record.frame = this._cascadeFrame;
      this._cascadeCache[layer] = record;
    } else {
      cached.frame = this._cascadeFrame;
    }
    return mode;
  }

  /**
   * The rectangles a scroll of (du, dv) texels redraws, with the map's origin
   * now at (ou, ov): the new columns at full height, the new rows at full
   * width, each cut where it wraps. Each gets a scissor, a matrix that puts
   * its texels where they now live -- the layer's, moved by the origin, less
   * a whole map where it wraps -- into its uniform slot, and a matrix that
   * stretches the rectangle over [-1, 1] for the caster cull.
   */
  _scrollPieces(layer, du, dv, ou, ov) {
    const size = this.size;
    const piece = this._pieces[layer];
    piece.count = 0;
    // Logical rectangles, [u0, u1) x [v0, v1), in the moved box.
    const columns = du > 0 ? [size - du, size] : du < 0 ? [0, -du] : null;
    const rows = dv > 0 ? [size - dv, size] : dv < 0 ? [0, -dv] : null;
    const rects = [];
    if (columns !== null) rects.push([columns[0], columns[1], 0, size]);
    if (rows !== null) rects.push([0, size, rows[0], rows[1]]);
    // Where a logical range lands: [start, end) in texels, moved by `origin`
    // and wrapped, as up to two ranges, each with the wrap it took.
    const spans = (start, end, origin) => {
      const a = start + origin, b = end + origin;
      if (b <= size) return [[a, b, 0]];
      if (a >= size) return [[a - size, b - size, size]];
      return [[a, size, 0], [0, b - size, size]];
    };
    const m = this.matrices, at = layer * 16;
    for (const [u0, u1, v0, v1] of rects) {
      for (const [x0, x1, wrapU] of spans(u0, u1, ou)) {
        for (const [y0, y1, wrapV] of spans(v0, v1, ov)) {
          const p = piece.count++;
          piece.rects.set([x0, y0, x1 - x0, y1 - y0], p * 4);
          // Logical texels of this piece, for the cull.
          const lu0 = x0 - ou + wrapU, lu1 = x1 - ou + wrapU, lv0 = y0 - ov + wrapV, lv1 = y1 - ov + wrapV;
          // Draw matrix: NDC moved by the origin, less the wrap. u = (x + 1) / 2 * size,
          // v = (1 - y) / 2 * size, and an orthographic w is 1, so translation is all it takes.
          const slot = (this.cascadeCapacity + layer * MAX_PIECES + p) * this.alignment / 4;
          this._cascadeStagingF32.set(m.subarray(at, at + 16), slot);
          this._cascadeStagingF32[slot + 12] += (2 * (ou - wrapU)) / size;
          this._cascadeStagingF32[slot + 13] -= (2 * (ov - wrapV)) / size;
          // Cull matrix: the piece's logical NDC box stretched to [-1, 1].
          const nx0 = (2 * lu0) / size - 1, nx1 = (2 * lu1) / size - 1;
          const ny0 = 1 - (2 * lv1) / size, ny1 = 1 - (2 * lv0) / size;
          const cx = (nx0 + nx1) / 2, hx = (nx1 - nx0) / 2, cy = (ny0 + ny1) / 2, hy = (ny1 - ny0) / 2;
          const c = piece.cull, o = p * 16;
          c.set(m.subarray(at, at + 16), o);
          for (const k of [0, 4, 8]) { c[o + k] /= hx; c[o + k + 1] /= hy; }
          c[o + 12] = (c[o + 12] - cx) / hx;
          c[o + 13] = (c[o + 13] - cy) / hy;
        }
      }
    }
  }

  /** Cascade `layer`, as this frame needs it: whole, or its scrolled pieces. */
  _encodeCascade(pass, layer) {
    if (this.cascadeRedraw[layer] !== SCROLL) {
      this._encodeView(pass, this.cascadeBindGroup, layer * this.alignment, this.matrices, layer * 16);
      return;
    }
    const piece = this._pieces[layer];
    const r = piece.rects;
    for (let p = 0; p < piece.count; p++) {
      pass.setScissorRect(r[p * 4], r[p * 4 + 1], r[p * 4 + 2], r[p * 4 + 3]);
      pass.setPipeline(this._pipelines.get(this.clearDescriptor));
      // Its layout is all empty groups, which still have to be bound.
      this.clearDescriptor.layout.bindEmptyGroups(pass);
      pass.draw(3);
      const slot = (this.cascadeCapacity + layer * MAX_PIECES + p) * this.alignment;
      this._encodeView(pass, this.cascadeBindGroup, slot, piece.cull, p * 16);
    }
  }

  /**
   * One cascade of the light along _lightDirection, into array layer `layer`.
   * `depth` is the scene's light-space z range (_sceneDepth), or null for none.
   */
  _fitCascade(layer, cascade, depth = null) {
    const sphere = this._spheres[cascade];
    const radius = sphere[3];

    // The light-space basis (_lightBasis) is anchored at the WORLD origin, not
    // at the camera: texel snapping below is only meaningful against a grid
    // that does not move when the camera does.
    // Sphere centre in light space.
    const v = this._lightView;
    const cx = v[0] * sphere[0] + v[4] * sphere[1] + v[8] * sphere[2] + v[12];
    const cy = v[1] * sphere[0] + v[5] * sphere[1] + v[9] * sphere[2] + v[13];
    const cz = v[2] * sphere[0] + v[6] * sphere[1] + v[10] * sphere[2] + v[14];

    // Snap to a grid of whole texels. Without this the box slides by fractions
    // of a texel every frame, every texel samples a slightly different patch of
    // world, and shadow edges crawl even when nothing in the scene is moving.
    //
    // The grid is CASCADE_SNAP texels, not one: a box that moved with every
    // texel the camera crossed was a new matrix nearly every frame the camera
    // moved, and a new matrix redraws the cascade -- 4.65 ms of Sponza's frame
    // at 720p on Iris Xe. On a coarser grid the box holds still until the
    // camera has moved that far, and the cache keeps it.
    //
    // Snapping moves the box by up to a grid step, so the box is a step wider
    // on every side than the sphere, with the texel sized so that it still
    // spans the map: 2 (r + n t) = size * t, so t = 2r / (size - 2n). At 2048
    // and 16, texels 1.5% larger than snapping to one.
    const texelSize = (2 * radius) / (this.size - 2 * CASCADE_SNAP);
    const step = texelSize * CASCADE_SNAP;
    const half = radius + step;
    const snappedX = Math.floor(cx / step) * step;
    const snappedY = Math.floor(cy / step) * step;
    // The depth range too, on the same step, and a step deeper each way: it
    // moved with every millimetre the camera did, and was a new matrix every
    // frame by itself.
    const snappedZ = Math.round(cz / step) * step;

    // The light looks down -Z, so a point at light-space z has distance -z.
    //
    // NEGATIVE IS FINE, and usually right. The eye sits at the world origin,
    // so for a scene standing on a floor at y = 0 under an overhead light,
    // every caster is on the light's side of the eye: behind it, at a negative
    // distance. This used to be clamped to at least 0.01, a perspective habit
    // an orthographic box has no use for -- and that clamp cut every one of
    // those casters out of the map. Nothing above the floor cast a shadow.
    //
    // From the scene's extent when there is one, so it does not move with the
    // camera at all (see _sceneDepth); casterExtent is the fallback.
    const nearDistance = depth !== null ? -depth[1] : -(snappedZ + radius + step) - radius * this.casterExtent;
    const farDistance = depth !== null ? -depth[0] : -(snappedZ - radius - step);
    const f = layer * 5;
    this._fits[f] = Math.floor(cx / step);
    this._fits[f + 1] = Math.floor(cy / step);
    this._fits[f + 2] = texelSize;
    this._fits[f + 3] = nearDistance;
    this._fits[f + 4] = farDistance;

    mat4OrthographicReverseZ(
      this._projection,
      snappedX - half, snappedX + half,
      snappedY - half, snappedY + half,
      nearDistance, Math.max(farDistance, nearDistance + 0.02),
    );

    const offset = layer * 16;
    mat4Multiply(this.matrices, this._projection, this._lightView, offset, 0, 0);
    // Same matrix into the GPU-side staging, at its dynamic-offset slot.
    this._cascadeStagingF32.set(this.matrices.subarray(offset, offset + 16), layer * this.alignment / 4);
    this.texelSizes[cascade] = texelSize;
  }

  /**
   * Declare one depth-only pass per cascade.
   *
   * Nothing here says when to run, whether to clear, or whether to store: all
   * four cascades write one resource and the forward pass reads it, which is
   * enough for the graph to work out the rest.
   *
   * `slotOffsets[i]` is where renderable i's model matrix already sits in the
   * draw ring. Written ONCE by the caller and reused across every cascade --
   * the matrix does not depend on which cascade is drawing, and writing it per
   * cascade would quadruple the ring for nothing.
   */
  addPasses(graph, resource, gpu, batchBindGroup, palette, morph, localResource = null, reads = []) {
    this._gpu = gpu;
    this._batchBindGroup = batchBindGroup;
    // Built on first use: the buffers it references belong to GpuDriven, which
    // is created after this object.
    // Names gpu.drawDataBuffer and gpu.batchOrderBuffer, both of which are
    // replaced when GpuDriven grows, and the joint palette, which is replaced
    // when it grows. Caching on first use alone would hold a group pointing at
    // destroyed buffers.
    if (this.cascadeBindGroup === undefined
      || this._gpuRevision !== gpu.buffersRevision
      || this._paletteRevision !== palette.revision
      || this._morphRevision !== morph.revision
      || this._hasLod !== gpu.hasLod) {
      this._hasLod = gpu.hasLod;
      this.cascadeBindGroup = this._makeCascadeBindGroup(gpu, palette, morph);
      this.localBindGroup = undefined;
      this._gpuRevision = gpu.buffersRevision;
      this._paletteRevision = palette.revision;
      this._morphRevision = morph.revision;
    }

    for (let layer = 0; layer < this.shadowedCount * this.cascadeCount; layer++) {
      // A kept cascade is neither drawn nor cleared; the array is imported.
      // A scrolled one keeps what it has and clears only what it redraws.
      const mode = this.cascadeRedraw[layer];
      if (mode === KEEP) continue;
      graph.addPass({
        name: `shadow:${Math.floor(layer / this.cascadeCount)}:${layer % this.cascadeCount}`,
        reads,
        depth: mode === FULL
          ? { resource, view: this.layerViews[layer], clear: DEPTH_CLEAR_VALUE }
          : { resource, view: this.layerViews[layer], keep: true },
        // Prebuilt as the array grew, so declaring a frame allocates no closures.
        execute: this._executors[layer],
      });
    }

    if (localResource === null || this.localCount === 0) return;
    this.localBindGroup ??= this._makeCascadeBindGroup(gpu, palette, morph, this.localUniform);
    for (let v = 0; v < this.localCount; v++) {
      // A kept map is not drawn, and the layer is not cleared: the array is
      // imported, so what it held last frame is still there.
      if (this.localRedraw[v] === 0) continue;
      graph.addPass({
        name: `shadow:local:${v}`,
        reads,
        depth: { resource: localResource, view: this.localLayerViews[v], clear: DEPTH_CLEAR_VALUE },
        execute: this._localExecutors[v],
      });
    }
  }

  /**
   * Each batch's casters as one world box, min xyz then max xyz: what a
   * cascade tests before drawing the batch. Rebuilt when the batches are, or
   * when what moved was not listed; otherwise grown by the renderables that
   * moved (`items`, Scene.movedRenderables). Too big costs a draw the GPU
   * clips away; it never loses a shadow. Skins and morphs move boxes without
   * moving nodes, so a scene with either rebuilds every frame.
   */
  boundBatches(scene, gpu, items, anyMoved) {
    const count = gpu.batchCount;
    const rebuild = this._boxedScene !== scene || this._boxedRevision !== gpu.sceneRevision
      || (anyMoved && items === null) || scene.skins.length > 0 || scene.morphs.length > 0;
    const { worldMin, worldMax } = scene;
    const grow = (b, i) => {
      const k = b * 6, o = i * 3, box = this._batchBoxes;
      for (let a = 0; a < 3; a++) {
        if (worldMin[o + a] < box[k + a]) box[k + a] = worldMin[o + a];
        if (worldMax[o + a] > box[k + 3 + a]) box[k + 3 + a] = worldMax[o + a];
      }
    };
    if (rebuild) {
      if (this._batchBoxes.length < count * 6) this._batchBoxes = new Float32Array(count * 6);
      for (let b = 0; b < count; b++) {
        this._batchBoxes.fill(Infinity, b * 6, b * 6 + 3);
        this._batchBoxes.fill(-Infinity, b * 6 + 3, b * 6 + 6);
        const first = gpu.batchFirst[b];
        for (let k = 0; k < gpu.batchSize[b]; k++) grow(b, gpu.batchOrder[first + k]);
      }
      this._boxedScene = scene;
      this._boxedRevision = gpu.sceneRevision;
    } else if (anyMoved) {
      for (const i of items) if (gpu.itemBatch[i] !== NOT_BATCHED) grow(gpu.itemBatch[i], i);
    }
  }

  /**
   * Every caster, into one view: a cascade or a point or spot light's. A
   * cascade passes `cull`, an orthographic matrix at cull[at], and draws only
   * the batches whose box reaches into it: anything else the GPU would
   * transform and clip, all of it, cascade after cascade. Point and spot
   * views are perspective, which that test is not.
   */
  _encodeView(pass, bindGroup, offset, cull = null, at = 0) {
    const gpu = this._gpu;
    const casts = (b) => cull === null || boxInOrtho(this._batchBoxes, b * 6, cull, at);
    pass.setBindGroup(GROUP_FRAME, bindGroup, [offset]);
    this.pipelineLayout.bindEmptyGroups(pass);
    // Batches arrive sorted by pipeline, so these come in runs and this
    // switches once per run rather than per draw.
    let boundVariant = -1;

    // Instanced, one call per batch. The shadow pass culls nothing, so the
    // instance count is simply the batch size and the shader walks the static
    // batch-ordered list rather than the compacted one.
    //
    // ponytail: culled a batch at a time, by the box around all its casters.
    // A batch of many instances spread wide still draws all of them into a
    // cascade that one reaches; per-instance needs the cull compute per cascade.
    // Depth-only casters first. The alpha ones bind a material in group 2,
    // where these pipelines expect the empty group, so they come after.
    const materials = gpu.materials;
    const alpha = this.alphaDescriptors.length > 0;
    for (let b = 0; b < gpu.batchCount; b++) {
      if (alpha && materials.alphaModes[gpu.batchMaterial[b]] === ALPHA_MASK) continue;
      if (!casts(b)) continue;
      const primitive = gpu.batchPrimitive[b];
      const skinned = gpu.batchSkinned[b];
      // Both sides for a double-sided material, and for an open surface:
      // culling the side facing the light took its whole shadow.
      const doubleSided = (materials.variants[gpu.batchMaterial[b]] & VARIANT_DOUBLE_SIDED) !== 0 || primitive.open === true;
      const variant = (doubleSided ? 4 : 0) + skinned * 2 + gpu.batchMirrored[b];
      if (variant !== boundVariant) {
        pass.setPipeline(this._pipelines.get(this.descriptors[variant]));
        boundVariant = variant;
      }
      pass.setVertexBuffer(0, primitive.vertexBuffer);
      if (skinned) pass.setVertexBuffer(1, primitive.skinBuffer);
      pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
      this._drawBatch(pass, gpu, b, primitive);
    }
    if (alpha) this._encodeAlphaCasters(pass, gpu, materials, casts);
  }

  /**
   * One batch's casters: all of them, or with LOD the ones the shadow cull
   * kept -- the level the camera shows, on screen or off.
   */
  _drawBatch(pass, gpu, b, primitive) {
    if (gpu.hasLod) {
      pass.setBindGroup(GROUP_DRAW, this._batchBindGroup, [gpu.batchOffset(b, CULL_SHADOW)]);
      pass.drawIndexedIndirect(gpu.indirectBuffer, gpu.indirectOffset(b, CULL_SHADOW));
    } else {
      pass.setBindGroup(GROUP_DRAW, this._batchBindGroup, [gpu.batchOffset(b)]);
      pass.drawIndexed(primitive.indexCount, gpu.batchSize[b]);
    }
  }

  /** MASK batches, then blended items, each shaped by its material's alpha. */
  _encodeAlphaCasters(pass, gpu, materials, casts) {
    let bound = null;
    let boundMaterial = -1;
    const draw = (hashed, skinned, material, primitive) => {
      const pipeline = this._pipelines.get(this.alphaDescriptors[(hashed ? 2 : 0) + (skinned ? 1 : 0)]);
      if (pipeline !== bound) { pass.setPipeline(pipeline); bound = pipeline; }
      if (material !== boundMaterial) {
        pass.setBindGroup(GROUP_MATERIAL, materials.bindGroup(material));
        boundMaterial = material;
      }
      pass.setVertexBuffer(0, primitive.vertexBuffer);
      if (skinned) pass.setVertexBuffer(1, primitive.skinBuffer);
      pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
    };

    for (let b = 0; b < gpu.batchCount; b++) {
      const material = gpu.batchMaterial[b];
      if (materials.alphaModes[material] !== ALPHA_MASK || !casts(b)) continue;
      const primitive = gpu.batchPrimitive[b];
      draw(false, gpu.batchSkinned[b] === 1, material, primitive);
      this._drawBatch(pass, gpu, b, primitive);
    }

    // Blended casters sit after the batches in the static order, and are drawn
    // through firstInstance against a batch base of zero -- the slot the
    // transparent pass uses. Neighbours that bind alike share one call.
    const casters = gpu.blendedCasters;
    if (casters.length === 0) return;
    pass.setBindGroup(GROUP_DRAW, this._batchBindGroup,
      [gpu.hasLod ? gpu.shadowTransparentBatchOffset() : gpu.transparentBatchOffset()]);
    const selected = gpu.casterSelected;
    for (let k = 0; k < casters.length;) {
      if (gpu.hasLod && selected[k] === 0) { k++; continue; }
      const { primitive, material, skinned } = casters[k];
      let run = 1;
      while (k + run < casters.length && casters[k + run].primitive === primitive
        && casters[k + run].material === material && casters[k + run].skinned === skinned
        && (!gpu.hasLod || selected[k + run] === 1)) run++;
      draw(true, skinned, material, primitive);
      pass.drawIndexed(primitive.indexCount, run, 0, 0, gpu.opaqueCount + k);
      k += run;
    }
  }

  destroy() {
    this.texture.destroy();
    this.cascadeBuffer.destroy();
    this.cascadeList.destroy();
    this.localTexture?.destroy();
    this.localUniform?.destroy();
    this.localBuffer?.destroy();
  }
}

/**
 * An up vector that is not parallel to the light. Straight down is the common
 * case and is exactly the one where the usual +Y up degenerates.
 */
function chooseUp(out, direction) {
  if (Math.abs(direction[1]) > 0.99) out.set([0, 0, 1]);
  else out.set([0, 1, 0]);
  return out;
}
