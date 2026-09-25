// Ground-truth ambient occlusion (Jimenez et al. 2016, as XeGTAO lays it out).
//
// Ambient light comes from every direction, and a crease or a contact sees
// less of the sky than an open face does. The environment cannot know that --
// it is baked once for the whole scene -- so this finds it per pixel from the
// depth buffer: in a few directions around each pixel, how far up the nearby
// geometry rises (its HORIZON), and from that the cosine-weighted share of
// the hemisphere that is still open. It is an integral, not an effect: an
// open flat surface comes out at 1, with no strength to tune.
//
// Where it fits a forward renderer: the ambient term is known only inside
// the forward passes, and the depth it needs is finished only after the last
// of them. So the opaque forward passes write their ambient term to a second
// target, this runs on the finished depth, and a composite takes
// ambient * (1 - occlusion) back out of the scene colour -- by blending, with
// reverse-subtract, so the scene colour is never copied. Blended geometry is
// drawn after that, because what is behind glass is not occluded by the glass.
//
// Three passes, two of them at half resolution, because occlusion changes
// slowly across a surface and its taps are scattered wide: at full resolution
// on integrated graphics it cost 4 ms at 720p, most of it cache misses.
//
//   ao        half res: the horizons, for each 2x2 block's top-left pixel
//   ao-blur   half res: the 4x4 tile average that removes the noise below
//   composite full res: a depth-aware 2x2 upsample, subtracted from the scene
//
// Noise and its removal are one design: each pixel of a 4x4 tile turns the
// slice directions by a different sixteenth and offsets its steps by another,
// and the blur averages exactly that tile. Every rotation is used once per
// output, sixteen times the slices for the price of two. Both averages are
// weighted by distance from the pixel's own plane, in units of the radius, so
// neither smears across an edge.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { HDR_FORMAT } from './post.js';
import { createBuffer } from '../rhi/buffer.js';

/** Where the forward passes put their ambient term. RGB, HDR, like the scene. */
export const AMBIENT_FORMAT = 'rgba16float';
const AO_FORMAT = 'r16float';

/** inverseProjection(64) + size, projection y scale and w terms, radius. */
const PARAMS_BYTES = 96;

/**
 * Slices and steps per side. A budget, as the shadow PCF's nine taps are:
 * two slices times the tile's sixteen rotations is thirty-two directions per
 * blurred pixel, and four steps a side reach the radius in quarters.
 */
const SLICES = 2;
const STEPS = 4;

const COMMON = /* wgsl */ `
const PI = 3.14159265359;

struct Params {
  inverseProjection : mat4x4<f32>,
  size              : vec2<f32>,   // FULL resolution, which the depth is
  // The projection's y scale and the two terms of clip w (w = z * wz + w1):
  // enough to turn a world radius into pixels, perspective or orthographic.
  yScale            : f32,
  wz                : f32,
  w1                : f32,
  radius            : f32,
  pad               : vec2<f32>,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var depthMap : texture_depth_2d;

fn clampPixel(p : vec2<i32>) -> vec2<i32> {
  return clamp(p, vec2<i32>(0), vec2<i32>(params.size) - vec2<i32>(1));
}

fn isSky(p : vec2<i32>) -> bool {
  return textureLoad(depthMap, clampPixel(p), 0) <= 0.0;
}

/** View-space position under a full-resolution pixel, from its depth. */
fn viewPosition(p : vec2<i32>) -> vec3<f32> {
  let q = clampPixel(p);
  let depth = textureLoad(depthMap, q, 0);
  let ndc = vec2<f32>(
    (f32(q.x) + 0.5) / params.size.x * 2.0 - 1.0,
    1.0 - (f32(q.y) + 0.5) / params.size.y * 2.0,
  );
  let v = params.inverseProjection * vec4<f32>(ndc, depth, 1.0);
  return v.xyz / v.w;
}

/**
 * The surface normal, from the depth buffer alone. On each axis the nearer of
 * the two neighbours in depth, so an edge takes its normal from its own side.
 */
fn viewNormal(p : vec2<i32>, center : vec3<f32>) -> vec3<f32> {
  let left = viewPosition(p - vec2<i32>(1, 0));
  let right = viewPosition(p + vec2<i32>(1, 0));
  let up = viewPosition(p - vec2<i32>(0, 1));
  let down = viewPosition(p + vec2<i32>(0, 1));
  let dx = select(right - center, center - left, abs(left.z - center.z) < abs(right.z - center.z));
  let dy = select(down - center, center - up, abs(up.z - center.z) < abs(down.z - center.z));
  return normalize(cross(dy, dx));
}

/** How much a neighbour counts: 1 on this pixel's plane, 0 a radius off it. */
fn planeWeight(center : vec3<f32>, normal : vec3<f32>, other : vec3<f32>) -> f32 {
  return max(1.0 - abs(dot(other - center, normal)) / params.radius, 0.0);
}

struct VertexOut {
  @builtin(position) position : vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32) -> VertexOut {
  var out : VertexOut;
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}
`;

const AO_SHADER = /* wgsl */ `
${COMMON}

/**
 * The view position at a screen offset that lies ON the slice's line, not
 * rounded off it. Rounding moved a step up to half a pixel sideways, and a
 * point on the surface but beside the slice sits at another angle to the view
 * than the slice's own tangent: the nearest steps, where half a pixel is a
 * wide angle, read an open floor as occluded (1.7% dark at 39 degrees). So,
 * as a line is drawn: a whole pixel along the major axis, and the position
 * interpolated between the two pixels across the minor one. Two points on a
 * plane interpolate to a point on it.
 *
 * w is 0 when either pixel is off the screen. A step there knows nothing, and
 * clamping it back onto the edge moved it off the slice again: open floor near
 * the bottom of the view came out 1.7% dark.
 */
fn onSlice(p : vec2<i32>, offset : vec2<f32>) -> vec4<f32> {
  var a : vec2<i32>;
  var b : vec2<i32>;
  var along : f32;
  if (abs(offset.x) >= abs(offset.y)) {
    let x = round(offset.x);
    let y = offset.y * x / offset.x;
    let below = floor(y);
    a = p + vec2<i32>(i32(x), i32(below));
    b = a + vec2<i32>(0, 1);
    along = y - below;
  } else {
    let y = round(offset.y);
    let x = offset.x * y / offset.y;
    let below = floor(x);
    a = p + vec2<i32>(i32(below), i32(y));
    b = a + vec2<i32>(1, 0);
    along = x - below;
  }
  let size = vec2<i32>(params.size);
  if (any(a < vec2<i32>(0)) || any(b >= size)) { return vec4<f32>(0.0); }
  return vec4<f32>(mix(viewPosition(a), viewPosition(b), along), 1.0);
}

/** One side's horizon along a slice: the highest cosine to V, faded by distance. */
fn horizon(p : vec2<i32>, center : vec3<f32>, view : vec3<f32>, span : vec2<f32>, low : f32, jitter : f32) -> f32 {
  var best = low;
  for (var i = 0u; i < ${STEPS}u; i = i + 1u) {
    // Squared spacing: more of the steps where occluders matter most.
    let t = (f32(i) + jitter) / f32(${STEPS});
    let offset = span * max(t * t, 1.0 / length(span));
    let found = onSlice(p, offset);
    // Off the screen, and every later step is further off: nothing more to see.
    if (found.w == 0.0) { break; }
    let delta = found.xyz - center;
    let distance = length(delta);
    let cosine = dot(delta / max(distance, 1e-6), view);
    // Fades to nothing at the radius, so an occluder there has no say.
    let reach = distance / params.radius;
    let weight = clamp(1.0 - reach * reach, 0.0, 1.0);
    best = max(best, mix(low, cosine, weight));
  }
  return best;
}

@fragment
fn fsAO(v : VertexOut) -> @location(0) vec4<f32> {
  let q = vec2<i32>(v.position.xy);
  let p = q * 2;
  if (isSky(p)) { return vec4<f32>(1.0); }

  let center = viewPosition(p);
  let normal = viewNormal(p, center);
  let view = normalize(-center);

  // The radius in full-resolution pixels at this depth: clip w is how far
  // perspective shrinks it.
  let w = center.z * params.wz + params.w1;
  let radiusPixels = params.radius * params.yScale * 0.5 * params.size.y / max(w, 1e-6);
  if (radiusPixels < 1.0) { return vec4<f32>(1.0); }

  // The tile: a rotation per half-resolution pixel, and a step offset that is
  // the same index with its four bits reversed, so the two never run in step.
  let cell = u32(q.x & 3) + u32(q.y & 3) * 4u;
  let rotation = (f32(cell) + 0.5) / 16.0;
  let jitter = (f32(reverseBits(cell) >> 28u) + 0.5) / 16.0;

  var visibility = 0.0;
  var open = 0.0;
  for (var s = 0u; s < ${SLICES}u; s = s + 1u) {
    let phi = (f32(s) + rotation) * PI / f32(${SLICES});
    // Screen y points down, view y up.
    let screen = vec2<f32>(cos(phi), sin(phi));
    let direction = vec3<f32>(screen.x, -screen.y, 0.0);

    // The slice is the plane through the view vector and this direction. The
    // normal, projected into it, sits at angle n from the view vector.
    let ortho = direction - dot(direction, view) * view;
    let axis = normalize(cross(direction, view));
    let projected = normal - axis * dot(normal, axis);
    let projectedLength = length(projected);
    let cosN = clamp(dot(projected, view) / max(projectedLength, 1e-6), -1.0, 1.0);
    let n = sign(dot(ortho, projected)) * acos(cosN);

    // The lowest a horizon can be is the surface's own tangent.
    let cos1 = horizon(p, center, view, screen * radiusPixels, cos(n + PI * 0.5), jitter);
    let cos0 = horizon(p, center, view, -screen * radiusPixels, cos(n - PI * 0.5), jitter);
    let h1 = n + min(acos(clamp(cos1, -1.0, 1.0)) - n, PI * 0.5);
    let h0 = n + max(-acos(clamp(cos0, -1.0, 1.0)) - n, -PI * 0.5);

    // The cosine-weighted arc between the two horizons, in closed form.
    let arc1 = cosN + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n);
    let arc0 = cosN + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n);
    visibility = visibility + projectedLength * (arc0 + arc1) * 0.25;
    // What the same slice gives with nothing above the surface: horizons at
    // its tangent, where the arcs sum to 4 (cos n + n sin n).
    open = open + projectedLength * (cosN + n * sin(n));
  }
  // As a share of what these slices see of an open surface, not of 1. A few
  // slices of an open surface integrate to 1 only on average over every
  // rotation -- one pixel's two can give 0.98 or 1.02 -- and the clamps that
  // keep occlusion in range then drop the overshoots and keep the rest: an
  // open floor came out 0.7% dark. Divided, an open surface is exactly 1
  // everywhere, and what is left is occlusion alone.
  return vec4<f32>(clamp(visibility / max(open, 1e-6), 0.0, 1.0));
}
`;

const BLUR_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(2) var aoMap : texture_2d<f32>;

/** The 4x4 tile the rotations were spread over, off-plane samples left out. */
@fragment
fn fsBlur(v : VertexOut) -> @location(0) vec4<f32> {
  let q = vec2<i32>(v.position.xy);
  let p = q * 2;
  if (isSky(p)) { return vec4<f32>(1.0); }
  let center = viewPosition(p);
  let normal = viewNormal(p, center);
  let last = vec2<i32>(textureDimensions(aoMap)) - vec2<i32>(1);

  var sum = 0.0;
  var weights = 0.0;
  for (var y = -2; y < 2; y = y + 1) {
    for (var x = -2; x < 2; x = x + 1) {
      let r = clamp(q + vec2<i32>(x, y), vec2<i32>(0), last);
      let weight = planeWeight(center, normal, viewPosition(r * 2));
      sum = sum + weight * textureLoad(aoMap, r, 0).r;
      weights = weights + weight;
    }
  }
  return vec4<f32>(clamp(sum / max(weights, 1e-6), 0.0, 1.0));
}
`;

const COMPOSITE_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(2) var aoMap : texture_2d<f32>;
@group(0) @binding(3) var ambientMap : texture_2d<f32>;

/**
 * The ambient light the occlusion removes; blending subtracts it from the
 * scene. The half-resolution occlusion is brought up bilinearly, with each of
 * the four samples weighted by how near it lies to this pixel's plane.
 */
@fragment
fn fsComposite(v : VertexOut) -> @location(0) vec4<f32> {
  let p = vec2<i32>(v.position.xy);
  if (isSky(p)) { return vec4<f32>(0.0); }
  let center = viewPosition(p);
  let normal = viewNormal(p, center);

  let at = (vec2<f32>(p) + 0.5) * 0.5 - 0.5;
  let base = vec2<i32>(floor(at));
  let f = at - floor(at);
  let last = vec2<i32>(textureDimensions(aoMap)) - vec2<i32>(1);
  var sum = 0.0;
  var weights = 0.0;
  var plain = 0.0;
  for (var k = 0; k < 4; k = k + 1) {
    let corner = vec2<i32>(k & 1, k >> 1);
    let r = clamp(base + corner, vec2<i32>(0), last);
    let bilinear = select(1.0 - f.x, f.x, corner.x == 1) * select(1.0 - f.y, f.y, corner.y == 1);
    let value = textureLoad(aoMap, r, 0).r;
    let weight = bilinear * planeWeight(center, normal, viewPosition(r * 2));
    sum = sum + weight * value;
    weights = weights + weight;
    plain = plain + bilinear * value;
  }
  // Nothing on this plane nearby -- a sliver between two surfaces: plain bilinear.
  let visibility = select(plain, sum / weights, weights > 1e-4);
  return vec4<f32>(textureLoad(ambientMap, p, 0).rgb * (1.0 - visibility), 0.0);
}
`;

export class AmbientOcclusion {
  static async create(rhi, pipelines) {
    const [aoShader, blurShader, compositeShader] = await Promise.all([
      compileShader(rhi.device, AO_SHADER, 'ao.wgsl'),
      compileShader(rhi.device, BLUR_SHADER, 'ao-blur.wgsl'),
      compileShader(rhi.device, COMPOSITE_SHADER, 'ao-composite.wgsl'),
    ]);
    const params = { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } };
    const depth = { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } };
    const texture = (binding) => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } });
    const layouts = {
      ao: rhi.device.createBindGroupLayout({ label: 'ao', entries: [params, depth] }),
      blur: rhi.device.createBindGroupLayout({ label: 'ao-blur', entries: [params, depth, texture(2)] }),
      composite: rhi.device.createBindGroupLayout({ label: 'ao-composite', entries: [params, depth, texture(2), texture(3)] }),
    };
    // The full-screen triangle winds clockwise; culling would drop it.
    const primitive = { topology: 'triangle-list', cullMode: 'none' };
    const descriptor = (name, layout, shader, entry, targets) => ({
      label: name,
      layout: createPipelineLayout(rhi.device, { 0: layout }, name),
      shader, fragmentEntry: entry, targets, primitive, depth: null,
    });
    const descriptors = {
      ao: descriptor('ao', layouts.ao, aoShader, 'fsAO', [{ format: AO_FORMAT }]),
      blur: descriptor('ao-blur', layouts.blur, blurShader, 'fsBlur', [{ format: AO_FORMAT }]),
      composite: descriptor('ao-composite', layouts.composite, compositeShader, 'fsComposite', [{
        format: HDR_FORMAT,
        // scene - ambient * occlusion; alpha left as it is.
        blend: {
          color: { operation: 'reverse-subtract', srcFactor: 'one', dstFactor: 'one' },
          alpha: { operation: 'add', srcFactor: 'zero', dstFactor: 'one' },
        },
      }]),
    };
    await pipelines.warm(Object.values(descriptors));
    return new AmbientOcclusion(rhi, pipelines, layouts, descriptors);
  }

  constructor(rhi, pipelines, layouts, descriptors) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    this.layouts = layouts;
    this.descriptors = descriptors;
    this.buffer = createBuffer(rhi, {
      label: 'ao-params', size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.data = new Float32Array(PARAMS_BYTES / 4);
  }

  /** This frame's camera and radius. `radius` is in world units. */
  update(camera, radius, width, height) {
    const d = this.data;
    d.set(camera.inverseProjection, 0);
    d[16] = width;
    d[17] = height;
    d[18] = camera.projection[5];
    // Clip w is row 3 of the projection times (x, y, z, 1): only z and 1 matter.
    d[19] = camera.projection[11];
    d[20] = camera.projection[15];
    d[21] = radius;
    this.rhi.queue.writeBuffer(this.buffer, 0, d);
  }

  /** The occlusion, its blur, and the composite into `sceneColor`. */
  addPasses(graph, { depth, ambient, sceneColor, width, height }) {
    const half = {
      width: Math.ceil(width / 2), height: Math.ceil(height / 2), format: AO_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    };
    const raw = graph.createTexture('ao', half);
    const blurred = graph.createTexture('ao-blurred', half);
    const clear = { r: 1, g: 1, b: 1, a: 1 };
    graph.addPass({
      name: 'ao',
      reads: [depth],
      color: [{ resource: raw, clear }],
      execute: (pass) => this._draw(pass, 'ao', [graph.viewOf(depth)]),
    });
    graph.addPass({
      name: 'ao-blur',
      reads: [depth, raw],
      color: [{ resource: blurred, clear }],
      execute: (pass) => this._draw(pass, 'blur', [graph.viewOf(depth), graph.viewOf(raw)]),
    });
    graph.addPass({
      name: 'ao-composite',
      reads: [depth, blurred, ambient],
      color: [{ resource: sceneColor }],
      execute: (pass) => this._draw(pass, 'composite',
        [graph.viewOf(depth), graph.viewOf(blurred), graph.viewOf(ambient)]),
    });
  }

  _draw(pass, name, views) {
    // Per frame: the graph may hand out different textures after a resize.
    const bindGroup = this.rhi.device.createBindGroup({
      layout: this.layouts[name],
      entries: [
        { binding: 0, resource: { buffer: this.buffer } },
        ...views.map((resource, i) => ({ binding: i + 1, resource })),
      ],
    });
    pass.setPipeline(this.pipelines.get(this.descriptors[name]));
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
  }

  destroy() {
    this.buffer.destroy();
  }
}
