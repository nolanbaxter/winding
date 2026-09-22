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
@group(3) @binding(0) var<uniform> batch : Batch;

@vertex
fn vs(
  @builtin(instance_index) instance : u32,
  @location(0) position : vec3<f32>,
) -> @builtin(position) vec4<f32> {
  let draw = drawData[order[batch.firstVisible + instance]];
  return cascade.viewProjection * draw.model * vec4<f32>(position, 1.0);
}

// The same skinning the forward pass does, for the same reason it has to: a
// caster that deforms and a shadow that does not is a character walking beside
// its own silhouette standing still.
@vertex
fn vsSkinned(
  @builtin(instance_index) instance : u32,
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
  return cascade.viewProjection * skin * vec4<f32>(position, 1.0);
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
  // The view matrix's rows are the camera basis expressed in world space.
  const rx = view[0], ry = view[4], rz = view[8];
  const ux = view[1], uy = view[5], uz = view[9];
  const fx = -view[2], fy = -view[6], fz = -view[10];

  const ex = camera.position[0], ey = camera.position[1], ez = camera.position[2];

  const tanHalf = Math.tan(camera.fovY * 0.5);
  const nearH = tanHalf * nearDistance, nearW = nearH * camera.aspect;
  const farH = tanHalf * farDistance, farW = farH * camera.aspect;

  // Centroid of the eight corners. It lies on the view axis by symmetry, so
  // this reduces to a point between the two slice centres.
  const cx = ex + fx * (nearDistance + farDistance) * 0.5;
  const cy = ey + fy * (nearDistance + farDistance) * 0.5;
  const cz = ez + fz * (nearDistance + farDistance) * 0.5;

  // Radius is the distance to the furthest corner, which is always a far one.
  let radius = 0;
  for (const [signX, signY] of CORNER_SIGNS) {
    const px = ex + fx * farDistance + rx * farW * signX + ux * farH * signY;
    const py = ey + fy * farDistance + ry * farW * signX + uy * farH * signY;
    const pz = ez + fz * farDistance + rz * farW * signX + uz * farH * signY;
    radius = Math.max(radius, Math.hypot(px - cx, py - cy, pz - cz));

    const nx = ex + fx * nearDistance + rx * nearW * signX + ux * nearH * signY;
    const ny = ey + fy * nearDistance + ry * nearW * signX + uy * nearH * signY;
    const nz = ez + fz * nearDistance + rz * nearW * signX + uz * nearH * signY;
    radius = Math.max(radius, Math.hypot(nx - cx, ny - cy, nz - cz));
  }

  out[0] = cx; out[1] = cy; out[2] = cz; out[3] = radius;
  return out;
}

const CORNER_SIGNS = [[-1, -1], [1, -1], [1, 1], [-1, 1]];

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
    if (DEBUG) assert(cascades > 0 && cascades <= MAX_CASCADES, 'bad cascade count');

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
      ],
    });

    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    this.cascadeBuffer = rhi.device.createBuffer({
      label: 'shadow-cascades',
      size: this.alignment * MAX_CASCADES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.cascadeStaging = new ArrayBuffer(this.alignment * MAX_CASCADES);
    this._makeCascadeBindGroup = (gpu, palette) => rhi.device.createBindGroup({
      label: 'shadow-cascade',
      layout: this.cascadeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cascadeBuffer, size: 64 } },
        { binding: 1, resource: { buffer: gpu.drawDataBuffer } },
        { binding: 2, resource: { buffer: gpu.batchOrderBuffer } },
        { binding: 3, resource: { buffer: palette.buffer } },
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
    await pipelines.warm([this.descriptor, this.skinnedDescriptor]);

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
    const texelSize = (2 * radius) / this.size;
    const snappedX = Math.floor(cx / texelSize) * texelSize;
    const snappedY = Math.floor(cy / texelSize) * texelSize;

    // The light looks down -Z, so a point at light-space z has distance -z.
    const nearDistance = Math.max(-(cz + radius) - radius * this.casterExtent, 0.01);
    const farDistance = -(cz - radius);

    mat4OrthographicReverseZ(
      this._projection,
      snappedX - radius, snappedX + radius,
      snappedY - radius, snappedY + radius,
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
  addPasses(graph, resource, gpu, batchBindGroup, palette) {
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
      || this._paletteRevision !== palette.revision) {
      this.cascadeBindGroup = this._makeCascadeBindGroup(gpu, palette);
      this._gpuRevision = gpu.buffersRevision;
      this._paletteRevision = palette.revision;
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
    // Batches arrive sorted, so skinned and unskinned come in runs and this
    // switches once rather than per draw.
    let boundSkinned = -1;

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
      if (skinned !== boundSkinned) {
        pass.setPipeline(this._pipelines.get(skinned ? this.skinnedDescriptor : this.descriptor));
        boundSkinned = skinned;
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
