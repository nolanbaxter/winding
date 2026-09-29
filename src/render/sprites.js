// Sprites: textured quads that face the camera (Scene.addSprite).
//
// Gathered from the scene every frame -- a sprite's place is its node's --
// packed into one instance buffer, and drawn as instanced quads, one call per
// run of sprites sharing a texture and a blend. Cutouts first, since they
// write depth; then additive, which needs no order; then alpha, sorted back
// to front, which is the order blending needs.
//
// Drawn after the opaque scene and before anything transmissive or blended,
// so glass shows a sprite behind it. What that costs: an alpha sprite in
// front of glass is drawn over by it.
//
// Unlit and in linear HDR, so a colour past 1 glows through bloom like any
// emissive surface, and fogged like every other surface.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { DEPTH_FORMAT, DEPTH_COMPARE } from '../rhi/device.js';
import { spriteSampler } from '../rhi/texture.js';
import { grownCapacity } from '../core/grow.js';
import { FRAME_WGSL } from './shaders/pbr.js';
import { FOG_WGSL } from './fog.js';
import { handleIndex } from '../core/handle.js';
import { spriteRect, repeats } from '../scene/scene.js';

/** position, angle, size, pivot, rect, colour, flags, cutoff, the plane's two axes, and a glyph's outline colour. */
export const SPRITE_FLOATS = 28;
const SPRITE_BYTES = SPRITE_FLOATS * 4;
/** Cutouts, then additive, then alpha; see the header. */
// Cutouts first, as they write depth; additive next, as its order is free;
// the rest far to near, as their order is not.
const DRAW_ORDER = { cutout: 0, additive: 1, alpha: 2, multiply: 2, screen: 2 };
/** The shader's BLEND for each: alpha, additive, cutout, multiply, screen. */
const BLEND_VALUE = { alpha: 0, additive: 1, cutout: 2, multiply: 3, screen: 4 };
const FLAG_UPRIGHT = 1;
const FLAG_PIXELS = 2;
/** In the node's own plane, along its x and y: a sign, not a billboard. */
const FLAG_PLANE = 4;
/** The texture's alpha is a distance field (a font's; see text.js), not coverage. */
const FLAG_SDF = 8;

const SHADER = /* wgsl */ `
${FRAME_WGSL}
${FOG_WGSL}

struct SpriteParams {
  right    : vec4<f32>,   // the camera's right, world space
  up       : vec4<f32>,   // and its up
  viewport : vec4<f32>,   // xy = target size in pixels
};

@group(0) @binding(0) var<uniform> frame      : Frame;
@group(0) @binding(1) var<uniform> params     : SpriteParams;
@group(0) @binding(2) var          irradiance : texture_cube<f32>;
@group(0) @binding(3) var          envSampler : sampler;
@group(1) @binding(0) var          image      : texture_2d<f32>;
@group(1) @binding(1) var          imageSampler : sampler;

// 0 alpha, 1 additive, 2 cutout: one pipeline each.
override BLEND : u32 = 0u;

struct Out {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv     : vec2<f32>,
  @location(1) color  : vec4<f32>,
  @location(2) world  : vec3<f32>,
  @location(3) @interpolate(flat) cutoff : f32,   // a glyph's: where its outline ends
  @location(4) @interpolate(flat) sdf    : f32,
  @location(5) @interpolate(flat) stroke : vec4<f32>,
};

@vertex
fn vs(
  @builtin(vertex_index) index : u32,
  @location(0) position : vec3<f32>,
  @location(1) rotation : f32,
  @location(2) size     : vec2<f32>,
  @location(3) pivot    : vec2<f32>,
  @location(4) rect     : vec4<f32>,
  @location(5) color    : vec4<f32>,
  @location(6) extra    : vec2<f32>,   // x = flags, y = cutoff
  @location(7) planeRight : vec3<f32>,
  @location(8) planeUp    : vec3<f32>,
  @location(9) stroke     : vec4<f32>,
) -> Out {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
  );
  let corner = corners[index];
  let flags = u32(extra.x);
  let c = cos(rotation);
  let s = sin(rotation);
  let unturned = (corner - pivot) * size;
  let offset = vec2<f32>(c * unturned.x - s * unturned.y, s * unturned.x + c * unturned.y);

  var out : Out;
  if ((flags & ${FLAG_PIXELS}u) != 0u) {
    // A size in pixels: the offset goes on after the projection, scaled by w
    // so the divide leaves it as many pixels at any distance.
    let clip = frame.viewProjection * vec4<f32>(position, 1.0);
    out.clip = vec4<f32>(clip.xy + offset * 2.0 / params.viewport.xy * clip.w, clip.zw);
    out.world = position;
  } else {
    var right = params.right.xyz;
    var up = params.up.xyz;
    if ((flags & ${FLAG_PLANE}u) != 0u) {
      right = planeRight;
      up = planeUp;
    } else if ((flags & ${FLAG_UPRIGHT}u) != 0u) {
      // Standing: up is the world's, and right turns to face the eye across it.
      up = vec3<f32>(0.0, 1.0, 0.0);
      let across = cross(up, frame.cameraPosition.xyz - position);
      if (dot(across, across) > 1e-12) { right = normalize(across); }
    }
    out.world = position + right * offset.x + up * offset.y;
    out.clip = frame.viewProjection * vec4<f32>(out.world, 1.0);
  }
  out.uv = mix(rect.xy, rect.zw, vec2<f32>(corner.x, 1.0 - corner.y));
  out.color = color;
  out.cutoff = extra.y;
  out.stroke = stroke;
  out.sdf = select(0.0, 1.0, (flags & ${FLAG_SDF}u) != 0u);
  return out;
}

@fragment
fn fs(v : Out) -> @location(0) vec4<f32> {
  let texel = textureSample(image, imageSampler, v.uv);
  // A distance field's edge is at 0.5; one screen pixel either side of it
  // blends, however large or small the glyph is drawn.
  // How far the field moves across a pixel. Out here, not in a branch: a
  // derivative needs every pixel around it to be running this line.
  let perPixel = max(fwidth(texel.a), 1e-5);
  let edge = clamp((texel.a - 0.5) / perPixel + 0.5, 0.0, 1.0);
  var colour = texel * v.color;
  if (v.sdf > 0.5) {
    // A glyph's outline, from its edge out to where the field reaches
    // v.cutoff: the fill over it, premultiplied.
    let outer = clamp((texel.a - v.cutoff) / perPixel + 0.5, 0.0, 1.0);
    let ring = max(outer - edge, 0.0);
    let a = v.color.a * edge + v.stroke.a * ring;
    colour = vec4<f32>((v.color.rgb * v.color.a * edge + v.stroke.rgb * v.stroke.a * ring) / max(a, 1e-6), a);
  }
  if (BLEND == 2u && colour.a < v.cutoff) { discard; }
  if (frame.fog.x > 0.0) {
    let toSprite = v.world - frame.cameraPosition.xyz;
    let distance = length(toSprite);
    let through = exp(-fogDepth(frame.fog, frame.cameraPosition.xyz, toSprite / max(distance, 1e-6), distance));
    let inscatter = frame.fogAlbedo.rgb * fogMeanRadiance(irradiance, envSampler) + frame.fogLight.rgb;
    // Additive and screen light is dimmed by the fog in front of it, and a
    // multiply's darkening fades to white, which changes nothing; the fog's
    // own glow is already there, under them.
    var fogged = colour.rgb * through + inscatter * (1.0 - through);
    if (BLEND == 1u || BLEND == 4u) { fogged = colour.rgb * through; }
    if (BLEND == 3u) { fogged = mix(vec3<f32>(1.0), colour.rgb, through); }
    colour = vec4<f32>(fogged, colour.a);
  }
  if (BLEND == 1u) { return vec4<f32>(colour.rgb * colour.a, 0.0); }
  if (BLEND >= 3u) { return vec4<f32>(min(colour.rgb, vec3<f32>(65504.0)) * colour.a, colour.a); }
  if (BLEND == 2u) { return vec4<f32>(colour.rgb, 1.0); }
  return vec4<f32>(min(colour.rgb, vec3<f32>(65504.0)), colour.a);
}
`;

/**
 * How each blend meets the target, shared by the 2D view. Multiply and
 * screen take a premultiplied colour: multiply is dst x lerp(1, src, a), and
 * screen src + dst - src x dst, each faded by the alpha.
 */
export const BLEND_STATE = {
  alpha: {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
  additive: {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
  },
  multiply: {
    color: { srcFactor: 'dst', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
  screen: {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
  cutout: undefined,
};

/**
 * This frame's sprites and text, packed and ordered, and the runs to draw
 * them in. Text is a quad a glyph, each placed by its pivot: the point of
 * the quad at the node is the glyph's offset from it, turned round. Pure
 * over the scene's data, so the order is testable without a GPU.
 * Returns { count, runs: [{ texture, blend, first, count }] }.
 *
 * Every frame, for every sprite and glyph, so nothing here allocates per
 * entry: an entry is an index into arrays kept from frame to frame, sorted
 * by keys worked out once each.
 */
export function packSprites(scene, camera, out = null) {
  const world = scene.transforms.world;
  const eye = camera.position;
  // The view axis: alpha sprites sort by depth along it, far first.
  const fx = -camera.view[2];
  const fy = -camera.view[6];
  const fz = -camera.view[10];
  const farness = (m) => -((world[m + 12] - eye[0]) * fx + (world[m + 13] - eye[1]) * fy + (world[m + 14] - eye[2]) * fz);

  let count = scene.sprites.size;
  for (const text of scene.texts.values()) count += text.boxes.length;
  if (scratch.order.length < count) {
    const capacity = grownCapacity(scratch.order.length, count);
    scratch.order = new Uint32Array(capacity);
    scratch.drawOrder = new Uint8Array(capacity);
    scratch.key = new Float64Array(capacity);
    scratch.matrix = new Uint32Array(capacity);
  }
  const { order, drawOrder, key, matrix, sources, boxes } = scratch;

  // Draw order first; then far to near for alpha, and by texture for the
  // rest, so each texture is one run.
  let n = 0;
  for (const [entity, sprite] of scene.sprites) {
    const m = handleIndex(entity) * 16;
    sources[n] = sprite;
    boxes[n] = null;
    matrix[n] = m;
    drawOrder[n] = DRAW_ORDER[sprite.blend];
    key[n] = DRAW_ORDER[sprite.blend] === 2 ? farness(m) : textureId(sprite.texture);
    order[n] = n++;
  }
  for (const [entity, text] of scene.texts) {
    const m = handleIndex(entity) * 16;
    const far = farness(m);
    for (const box of text.boxes) {
      sources[n] = text;
      boxes[n] = box;
      matrix[n] = m;
      drawOrder[n] = DRAW_ORDER[text.blend];
      key[n] = far;
      order[n] = n++;
    }
  }
  sources.length = boxes.length = n;   // hold no record past its removal
  // Ties keep the order entries were gathered in, so a text's glyphs, which
  // share its depth, keep theirs.
  const sorted = order.subarray(0, n).sort((a, b) => drawOrder[a] - drawOrder[b] || key[a] - key[b] || a - b);

  if (out === null || out.length < n * SPRITE_FLOATS) {
    out = new Float32Array(grownCapacity(out === null ? 0 : out.length / SPRITE_FLOATS, n) * SPRITE_FLOATS);
  }
  const runs = [];
  for (let k = 0; k < n; k++) {
    const e = sorted[k];
    const source = sources[e];
    const box = boxes[e];
    const m = matrix[e];
    const o = k * SPRITE_FLOATS;
    // A glyph is a sprite of its font's atlas, sized and pivoted by its box.
    const glyph = box !== null;
    const texture = glyph ? source.font.texture : source.texture;
    const blend = source.blend;
    const rect = glyph ? source.font.metrics.glyphs.get(box.char).rect : spriteRect(source);
    const color = source.color;
    out[o] = world[m + 12];
    out[o + 1] = world[m + 13];
    out[o + 2] = world[m + 14];
    out[o + 3] = glyph ? 0 : source.angle;
    // Scaled by the node, as a mesh would be: its x and y axes' lengths. A
    // size in pixels is on the screen, where the node's scale means nothing.
    const lx = Math.sqrt(world[m] * world[m] + world[m + 1] * world[m + 1] + world[m + 2] * world[m + 2]);
    const ly = Math.sqrt(world[m + 4] * world[m + 4] + world[m + 5] * world[m + 5] + world[m + 6] * world[m + 6]);
    const width = glyph ? box.width * source.size : source.size[0];
    const height = glyph ? box.height * source.size : source.size[1];
    out[o + 4] = width * (source.pixels ? 1 : lx);
    out[o + 5] = height * (source.pixels ? 1 : ly);
    out[o + 6] = glyph ? -box.x / box.width : source.pivot[0];
    // A pivot's [0, 0] is the image's top-left, as in 2D; the quad here is measured y up.
    out[o + 7] = glyph ? -box.y / box.height : 1 - source.pivot[1];
    for (let c = 0; c < 4; c++) {
      out[o + 8 + c] = rect[c];
      out[o + 12 + c] = color[c];
    }
    out[o + 16] = (source.facing === 'upright' ? FLAG_UPRIGHT : 0) | (source.facing === 'plane' ? FLAG_PLANE : 0)
      | (source.pixels ? FLAG_PIXELS : 0) | (glyph ? FLAG_SDF : 0);
    out[o + 17] = glyph ? source.strokeEdge : source.cutoff;
    // The node's own x and y, for a sprite in its plane.
    for (let c = 0; c < 3; c++) {
      out[o + 18 + c] = world[m + c] / (lx || 1);
      out[o + 21 + c] = world[m + 4 + c] / (ly || 1);
    }
    for (let c = 0; c < 4; c++) out[o + 24 + c] = glyph ? source.stroke[c] : 0;
    const repeat = !glyph && repeats(source);
    const last = runs[runs.length - 1];
    if (last && last.texture === texture && last.blend === blend && last.repeat === repeat) last.count++;
    else runs.push({ texture, blend, first: k, count: 1, repeat });
  }
  return { count: n, runs, out };
}

const scratch = {
  order: new Uint32Array(0), drawOrder: new Uint8Array(0), key: new Float64Array(0), matrix: new Uint32Array(0),
  sources: [], boxes: [],
};

let nextTextureId = 1;
const textureIds = new WeakMap();
function textureId(texture) {
  let id = textureIds.get(texture);
  if (id === undefined) { id = nextTextureId++; textureIds.set(texture, id); }
  return id;
}

export class SpritePass {
  static async create(rhi, pipelines, frameBuffer, colorFormat) {
    const device = rhi.device;
    const shader = await compileShader(device, SHADER, 'sprites.wgsl');
    const frameLayout = device.createBindGroupLayout({
      label: 'sprites',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const imageLayout = device.createBindGroupLayout({
      label: 'sprite-image',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const layout = createPipelineLayout(device, { 0: frameLayout, 1: imageLayout }, 'sprites');
    const attributes = [
      ['float32x3', 0], ['float32', 12], ['float32x2', 16], ['float32x2', 24],
      ['float32x4', 32], ['float32x4', 48], ['float32x2', 64], ['float32x3', 72], ['float32x3', 84], ['float32x4', 96],
    ].map(([format, offset], shaderLocation) => ({ format, offset, shaderLocation }));
    const descriptors = {};
    for (const [blend, value] of Object.entries(BLEND_VALUE)) {
      descriptors[blend] = {
        label: `sprites:${blend}`,
        layout,
        shader,
        buffers: [{ arrayStride: SPRITE_BYTES, stepMode: 'instance', attributes }],
        targets: [{ format: colorFormat, blend: BLEND_STATE[blend] }],
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        // Only a cutout occludes; blended sprites are tested and write nothing.
        depth: { format: DEPTH_FORMAT, depthCompare: DEPTH_COMPARE, depthWriteEnabled: blend === 'cutout' },
        constants: { BLEND: value },
      };
    }
    await pipelines.warm(Object.values(descriptors));
    return new SpritePass(rhi, pipelines, descriptors, frameLayout, imageLayout, frameBuffer);
  }

  constructor(rhi, pipelines, descriptors, frameLayout, imageLayout, frameBuffer) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    this._descriptors = descriptors;
    this._frameLayout = frameLayout;
    this._imageLayout = imageLayout;
    this._frameBuffer = frameBuffer;
    this._params = new Float32Array(12);
    this._paramsBuffer = createBuffer(rhi, {
      label: 'sprite-params', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._data = new Float32Array(64 * SPRITE_FLOATS);
    this._buffer = null;
    this._capacity = 0;
    this._frameGroups = new WeakMap();
    this._imageGroups = new WeakMap();
    this.count = 0;
    this._runs = [];
    this._environment = null;
    this._execute = (pass) => this._encode(pass);
  }

  /** Gather and upload this frame's sprites. Returns how many there are. */
  /**
   * `moved`: whether any transform moved this frame. Without that, and with
   * the same scene, contents, camera and size as the last pack, the packed
   * list and its upload stand as they are -- packing is per sprite and per
   * glyph, every frame, and 10,000 of them cost milliseconds.
   */
  prepare(scene, camera, environment, width, height, moved = true) {
    if (scene.sprites.size === 0 && scene.texts.size === 0) {
      this.count = 0;
      this._packed = null;
      return 0;
    }
    const last = this._packed;
    if (!moved && last !== null && last !== undefined && last.scene === scene && last.revision === scene.revision
      && last.changes === scene.changes && last.width === width && last.height === height
      && camera.view.every((v, i) => v === last.view[i])
      && camera.position.every((v, i) => v === last.eye[i])) {
      this._environment = environment;
      return this.count;
    }
    this._packed = {
      scene, revision: scene.revision, changes: scene.changes, width, height,
      view: Float32Array.from(camera.view), eye: Float32Array.from(camera.position),
    };
    const { count, runs, out } = packSprites(scene, camera, this._data);
    this._data = out;
    this.count = count;
    this._runs = runs;
    if (count === 0) return 0;
    const bytes = this.count * SPRITE_BYTES;
    if (bytes > this._capacity) {
      this._buffer?.destroy();
      this._capacity = grownCapacity(this._capacity, bytes);
      this._buffer = createBuffer(this.rhi, {
        label: 'sprites', size: this._capacity, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    this.rhi.queue.writeBuffer(this._buffer, 0, this._data, 0, this.count * SPRITE_FLOATS);
    const v = camera.view;
    this._params.set([v[0], v[4], v[8], 0, v[1], v[5], v[9], 0, width, height, 0, 0]);
    this.rhi.queue.writeBuffer(this._paramsBuffer, 0, this._params);
    this._environment = environment;
    return this.count;
  }

  /** The pass, onto the scene's colour and depth. Nothing when there are no sprites. */
  addPass(graph, { sceneColor, depth, reads = [] }) {
    if (this.count === 0) return;
    graph.addPass({
      name: 'sprites',
      reads,
      color: [{ resource: sceneColor }],
      depth: { resource: depth },
      execute: this._execute,
    });
  }

  _encode(pass) {
    const environment = this._environment;
    let frameGroup = this._frameGroups.get(environment);
    if (!frameGroup) {
      frameGroup = this.rhi.device.createBindGroup({
        label: 'sprites',
        layout: this._frameLayout,
        entries: [
          { binding: 0, resource: { buffer: this._frameBuffer } },
          { binding: 1, resource: { buffer: this._paramsBuffer } },
          { binding: 2, resource: environment.irradianceView },
          { binding: 3, resource: environment.sampler },
        ],
      });
      this._frameGroups.set(environment, frameGroup);
    }
    pass.setBindGroup(0, frameGroup);
    pass.setVertexBuffer(0, this._buffer);
    let bound = null;
    for (const { texture, blend, first, count, repeat } of this._runs) {
      if (blend !== bound) {
        pass.setPipeline(this.pipelines.get(this._descriptors[blend]));
        bound = blend;
      }
      // One group per texture and sampler: clamped, or repeating.
      let groups = this._imageGroups.get(texture);
      if (!groups) this._imageGroups.set(texture, groups = []);
      groups[repeat ? 1 : 0] ??= this.rhi.device.createBindGroup({
        label: 'sprite-image',
        layout: this._imageLayout,
        entries: [
          { binding: 0, resource: texture.view },
          { binding: 1, resource: spriteSampler(this.rhi, texture.pixelated, repeat) },
        ],
      });
      pass.setBindGroup(1, groups[repeat ? 1 : 0]);
      pass.draw(6, count, 0, first);
    }
  }

  destroy() {
    this._buffer?.destroy();
    this._paramsBuffer.destroy();
  }
}

