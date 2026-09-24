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

import { DEBUG, assert } from '../core/assert.js';
import {
  mat4Create, mat4LookAt, mat4Multiply, mat4OrthographicReverseZ, mat4Identity,
} from '../core/math/mat4.js';
import { vec3Create, vec3Normalize } from '../core/math/vec3.js';
import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout, GROUP_FRAME, GROUP_DRAW } from '../rhi/bindgroups.js';
import { DEPTH_FORMAT, DEPTH_CLEAR_VALUE, DEPTH_COMPARE } from '../rhi/device.js';
import { VERTEX_BUFFER_LAYOUT, SKIN_BUFFER_LAYOUT } from './vertex.js';

export const MAX_CASCADES = 4;

/** Depth-only. No fragment stage at all -- half the work of the forward pass. */
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

// Positions only. The shadow pass has no fragment stage, so the normal and
// tangent deltas the forward pass reads would be fetched and discarded.
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

export class ShadowMaps {
  static async create(rhi, pipelines, drawLayout, options = {}) {
    const maps = new ShadowMaps(rhi, options);
    await maps._init(pipelines, drawLayout);
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

    this.rhi = rhi;
    this.size = size;
    this.cascadeCount = cascades;
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

    // One depth texture, `cascades` array layers. An array rather than an atlas
    // so the shader indexes by cascade with no UV arithmetic and no bleeding
    // between neighbours at the seams.
    this.texture = rhi.device.createTexture({
      label: 'shadow-cascades',
      size: [size, size, cascades],
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.view = this.texture.createView({ dimension: '2d-array', label: 'shadow-cascades' });
    this.layerViews = [];
    for (let i = 0; i < cascades; i++) {
      this.layerViews.push(this.texture.createView({
        dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1, label: `cascade-${i}`,
      }));
    }

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

    this.matrices = new Float32Array(MAX_CASCADES * 16);
    this.splits = new Float32Array(MAX_CASCADES);
    /** World size of one shadow texel, per cascade. The normal-offset bias
     *  scales with it, so a coarse far cascade biases more than a fine near one. */
    this.texelSizes = new Float32Array(MAX_CASCADES);

    this._lightView = mat4Create();
    this._projection = mat4Create();
    this._sphere = new Float32Array(4);
    this._lightDirection = vec3Create(0, -1, 0);
    this._eye = vec3Create();
    this._target = vec3Create();
    this._up = vec3Create();
  }

  async _init(pipelines, drawLayout) {
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
    this.cascadeBuffer = rhi.device.createBuffer({
      label: 'shadow-cascades',
      size: this.alignment * MAX_CASCADES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.cascadeStaging = new ArrayBuffer(this.alignment * MAX_CASCADES);
    this._makeCascadeBindGroup = (gpu, palette, morph) => rhi.device.createBindGroup({
      label: 'shadow-cascade',
      layout: this.cascadeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cascadeBuffer, size: 64 } },
        { binding: 1, resource: { buffer: gpu.drawDataBuffer } },
        { binding: 2, resource: { buffer: gpu.batchOrderBuffer } },
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
    // Indexed by skinned * 2 + mirrored.
    this.descriptors = [
      this.descriptor, mirror(this.descriptor),
      this.skinnedDescriptor, mirror(this.skinnedDescriptor),
    ];
    await pipelines.warm(this.descriptors);

    // One bound executor per cascade, built once. The graph stores a function
    // per pass, and building them per frame would allocate MAX_CASCADES
    // closures every frame for no reason.
    this._executors = [];
    for (let cascade = 0; cascade < MAX_CASCADES; cascade++) {
      this._executors.push((pass) => this._encodeCascade(pass, cascade));
    }
  }

  /**
   * Fit every cascade to the camera and light, and upload the matrices.
   * Call once per frame, before rendering the shadow pass.
   */
  update(camera, lightDirection) {
    // A zero direction is the natural way to say "no sun", because sun.direction
    // is a plain mutable field the API invites you to write into. Normalizing it
    // yields (0,0,0), which makes eye === target in mat4LookAt and fills every
    // cascade matrix with NaN -- and a NaN shadow lookup does not fail loudly,
    // it just poisons the lighting. So the degenerate case is answered here
    // instead: no cascades, which selectCascade already reads as "unshadowed"
    // because the splits are zero.
    const lengthSq = lightDirection[0] * lightDirection[0]
      + lightDirection[1] * lightDirection[1]
      + lightDirection[2] * lightDirection[2];
    this.activeCascades = lengthSq > 0 ? this.cascadeCount : 0;
    if (this.activeCascades === 0) {
      this.splits.fill(0);
      return;
    }

    vec3Normalize(this._lightDirection, lightDirection);
    const splits = cascadeSplits(camera.near, this.shadowDistance, this.cascadeCount, this.lambda);

    let sliceNear = camera.near;
    for (let i = 0; i < this.cascadeCount; i++) {
      const sliceFar = splits[i];
      this._fitCascade(i, camera, sliceNear, sliceFar);
      this.splits[i] = sliceFar;
      sliceNear = sliceFar;
    }
    // Unused cascades get a split of 0, not Infinity. selectCascade tests
    // `viewDepth < split`, and every depth is below Infinity, so the obvious
    // sentinel picks the cascade it was meant to skip -- an identity matrix and
    // a texture layer that was never allocated.
    for (let i = this.cascadeCount; i < MAX_CASCADES; i++) {
      this.splits[i] = 0;
      mat4Identity(this.matrices.subarray(i * 16, i * 16 + 16));
    }

    this.rhi.queue.writeBuffer(this.cascadeBuffer, 0, this.cascadeStaging);
  }

  _fitCascade(index, camera, sliceNear, sliceFar) {
    const sphere = frustumSliceSphere(this._sphere, camera, sliceNear, sliceFar);
    const radius = sphere[3];

    // A light-space basis anchored at the WORLD origin, not at the camera.
    // Texel snapping below is only meaningful against a grid that does not
    // move when the camera does.
    const d = this._lightDirection;
    chooseUp(this._up, d);
    this._eye.set([0, 0, 0]);
    this._target.set([d[0], d[1], d[2]]);
    mat4LookAt(this._lightView, this._eye, this._target, this._up);

    // Sphere centre in light space.
    const v = this._lightView;
    const cx = v[0] * sphere[0] + v[4] * sphere[1] + v[8] * sphere[2] + v[12];
    const cy = v[1] * sphere[0] + v[5] * sphere[1] + v[9] * sphere[2] + v[13];
    const cz = v[2] * sphere[0] + v[6] * sphere[1] + v[10] * sphere[2] + v[14];

    // Snap to whole texels. Without this the box slides by fractions of a texel
    // every frame, every texel samples a slightly different patch of world, and
    // shadow edges crawl even when nothing in the scene is moving.
    //
    // Snapping moves the box by up to a texel, so a box exactly the sphere's
    // size could leave up to one texel of the sphere uncovered -- fragments
    // there fell outside the map and read as lit. The box is one texel wider
    // on every side, with the texel sized so that it still spans the map:
    // 2 (r + t) = size * t, so t = 2r / (size - 2).
    const texelSize = (2 * radius) / (this.size - 2);
    const half = radius + texelSize;
    const snappedX = Math.floor(cx / texelSize) * texelSize;
    const snappedY = Math.floor(cy / texelSize) * texelSize;

    // The light looks down -Z, so a point at light-space z has distance -z.
    //
    // NEGATIVE IS FINE, and usually right. The eye sits at the world origin,
    // so for a scene standing on a floor at y = 0 under an overhead sun, every
    // caster is on the sun's side of the eye: behind it, at a negative
    // distance. This used to be clamped to at least 0.01, a perspective habit
    // an orthographic box has no use for -- and that clamp cut every one of
    // those casters out of the map. Nothing above the floor cast a shadow.
    const nearDistance = -(cz + radius) - radius * this.casterExtent;
    const farDistance = -(cz - radius);

    mat4OrthographicReverseZ(
      this._projection,
      snappedX - half, snappedX + half,
      snappedY - half, snappedY + half,
      nearDistance, Math.max(farDistance, nearDistance + 0.02),
    );

    const offset = index * 16;
    mat4Multiply(this.matrices, this._projection, this._lightView, offset, 0, 0);

    // Same matrix into the GPU-side staging, at its dynamic-offset slot.
    const slot = new Float32Array(this.cascadeStaging, index * this.alignment, 16);
    for (let k = 0; k < 16; k++) slot[k] = this.matrices[offset + k];

    this.texelSizes[index] = texelSize;
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
  addPasses(graph, resource, gpu, batchBindGroup, palette, morph) {
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
      || this._morphRevision !== morph.revision) {
      this.cascadeBindGroup = this._makeCascadeBindGroup(gpu, palette, morph);
      this._gpuRevision = gpu.buffersRevision;
      this._paletteRevision = palette.revision;
      this._morphRevision = morph.revision;
    }

    for (let cascade = 0; cascade < this.activeCascades; cascade++) {
      graph.addPass({
        name: `shadow:${cascade}`,
        depth: {
          resource,
          view: this.layerViews[cascade],
          clear: DEPTH_CLEAR_VALUE,
        },
        // Prebuilt in _init, so declaring a frame allocates no closures.
        execute: this._executors[cascade],
      });
    }
  }

  _encodeCascade(pass, cascade) {
    const gpu = this._gpu;
    pass.setBindGroup(GROUP_FRAME, this.cascadeBindGroup, [cascade * this.alignment]);
    this.pipelineLayout.bindEmptyGroups(pass);
    // Batches arrive sorted by pipeline, so these come in runs and this
    // switches once per run rather than per draw.
    let boundVariant = -1;

    // Instanced, one call per batch. The shadow pass culls nothing, so the
    // instance count is simply the batch size and the shader walks the static
    // batch-ordered list rather than the compacted one.
    //
    // ponytail: every caster is still drawn into every cascade. Proper CSM runs
    // the cull compute once per cascade against its own ortho box, which needs
    // a six-plane frustum -- the extraction in frustum.js assumes a perspective
    // matrix and produces five.
    for (let b = 0; b < gpu.batchCount; b++) {
      const primitive = gpu.batchPrimitive[b];
      const skinned = gpu.batchSkinned[b];
      const variant = skinned * 2 + gpu.batchMirrored[b];
      if (variant !== boundVariant) {
        pass.setPipeline(this._pipelines.get(this.descriptors[variant]));
        boundVariant = variant;
      }
      pass.setBindGroup(GROUP_DRAW, this._batchBindGroup, [gpu.batchOffset(b)]);
      pass.setVertexBuffer(0, primitive.vertexBuffer);
      if (skinned) pass.setVertexBuffer(1, primitive.skinBuffer);
      pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
      pass.drawIndexed(primitive.indexCount, gpu.batchSize[b]);
    }
  }

  destroy() {
    this.texture.destroy();
    this.cascadeBuffer.destroy();
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
