// Gaussian splats: drawing a capture (scene.addSplats; the file is read by
// scene/splats.js).
//
// Each splat is a 3D Gaussian. Seen from the camera it projects to a 2D one
// -- the EWA splatting of Zwicker et al., as 3D Gaussian Splatting uses it:
// its covariance carried into view space, then through the projection's
// derivative at its centre, J W S W^T J^T -- drawn as a quad over the ellipse
// it covers, whose fragments fade by the Gaussian. Laid over each other back
// to front, blended, they make the picture.
//
// Back to front is the whole difficulty: a capture is hundreds of thousands
// to millions of splats, and their order changes as the camera moves. It is
// sorted on the GPU, every frame, as the rest of the engine culls there:
//
//   keys     a thread a splat: culled against the view, and the survivors
//            appended with a 16-bit key, how far away they are across the
//            cloud's depth, counted into 65536 buckets as they go
//   scan     one workgroup: where each bucket begins, from the counts --
//            which it then zeroes, ready for the next frame
//   scatter  a thread a survivor: placed at its bucket's next slot
//
// A counting sort, in one pass, because a key fits in 16 bits. The order
// within a bucket comes out of atomics, so it varies, but a bucket is a
// 65536th of the cloud's depth, and splats that close are drawn as one. The
// draw is indirect: the survivors' count is what the keys pass appended, and
// never comes back to the CPU. Nothing is sorted again until the camera, the
// cloud or the canvas moves: a scene with something animating beside a still
// capture keeps its order.
//
// The vertex shader projects each splat at each of its four corners, and that
// is measured, not overlooked: projecting once in the keys pass and storing
// the ellipse cost the sort what it saved the draw (Iris Xe, a million
// splats), and 32 more bytes a splat. What bounds the draw where it is not
// fill is the two triangles a splat, however small: 6 ms for a million of
// them at 320x180.
//
// A splat is lit by nothing: its colour is what the capture saw, decoded from
// sRGB, fogged as the scene is, and tonemapped with everything else. Where the
// capture has higher harmonics, the colour changes with the view as it did in
// the photographs -- a sheen, a reflection -- from the direction each splat
// is seen from. They change only when the camera or the cloud moves, which is
// when it is sorted again, so a fourth compute pass after the sort evaluates
// them once a visible splat, into a colour the draw reads: in the vertex
// shader they were worked out at each of a quad's four corners, and doubled
// the draw (205k splats of degree 3, 720p, Iris Xe: 2.19 -> 4.65 ms). Each
// degree is an override constant, so a capture without them pays nothing.
// Splats are hidden by geometry in front of them, and hide nothing: they
// write no depth, and cast no shadows. So they are drawn straight after the
// opaque scene, before sprites, particles and blended surfaces, which a
// capture -- usually the background -- would otherwise paint over. Two
// clouds are each sorted on their own and drawn far one first, so where two
// overlap, they do not interleave.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer, storageCapacity } from '../rhi/buffer.js';
import { DEPTH_FORMAT, DEPTH_COMPARE } from '../rhi/device.js';
import { handleIndex } from '../core/handle.js';
import { mat4Multiply, mat4Invert } from '../core/math/mat4.js';
import { shWords } from '../scene/splats.js';
import { FRAME_WGSL } from './shaders/pbr.js';
import { FOG_WGSL } from './fog.js';

const BUCKETS = 65536;
const WORKGROUP = 256;
/** modelView, projection, viewport, depth range, count: what a sort depends on. Then model, for fog, and the eye. */
const CLOUD_FLOATS = 37;
const CLOUD_BYTES = 240;
/** Frames a node's sort buffers outlive the last frame that drew it. */
const KEEP_FRAMES = 120;
/**
 * The default cull, in opacity x pixels: see makeKeys. Measured at 1280x720
 * (Iris Xe): a million splats seen whole, sort and draw 27.4 -> 16.8 ms, 0.06%
 * of channels moved by more than 8 levels; a capture seen close, where splats
 * are large and few are culled, no faster and 0.12% moved. At 1 the far one
 * halved but fine detail up close thinned visibly.
 */
export const SPLAT_CULL = 0.5;

const CLOUD_WGSL = /* wgsl */ `
struct Cloud {
  modelView  : mat4x4<f32>,   // multiplied once, on the CPU
  projection : mat4x4<f32>,
  viewport   : vec2<f32>,
  depthMin   : f32,   // the nearest view depth the cloud reaches
  depthScale : f32,   // buckets per unit of depth across it
  count      : u32,
  model      : mat4x4<f32>,   // for fog, which is in world space
  eye        : vec3<f32>,     // the camera, in the capture's own space, for its harmonics
  cull       : f32,           // the least a splat may add to the image and be drawn: opacity x pixels
};
`;

/**
 * Splat i's footprint on screen, as a 2D covariance in pixels (xx, xy, yy),
 * before the third of a pixel the draw adds. The EWA projection, J W S W^T J^T:
 * the projection's derivative at the centre, constant for an orthographic
 * camera; for a perspective one, with the centre held inside a little more
 * than the view, as 3DGS does, so a splat beside it does not stretch without
 * bound. Needs the covariances and the cloud bound.
 */
const FOOTPRINT_WGSL = /* wgsl */ `
fn footprint(i : u32, view : vec4<f32>) -> vec3<f32> {
  let o = i * 6u;
  let sigma = mat3x3<f32>(
    covariances[o], covariances[o + 1u], covariances[o + 2u],
    covariances[o + 1u], covariances[o + 3u], covariances[o + 4u],
    covariances[o + 2u], covariances[o + 4u], covariances[o + 5u],
  );
  let p = cloud.projection;
  let fx = p[0][0] * cloud.viewport.x * 0.5;
  let fy = p[1][1] * cloud.viewport.y * 0.5;
  var J = mat3x3<f32>(vec3<f32>(fx, 0.0, 0.0), vec3<f32>(0.0, fy, 0.0), vec3<f32>(0.0));
  if (p[3][3] == 0.0) {
    let d = -view.z;
    let x = clamp(view.x / d, -1.3 / p[0][0], 1.3 / p[0][0]) * d;
    let y = clamp(view.y / d, -1.3 / p[1][1], 1.3 / p[1][1]) * d;
    J = mat3x3<f32>(vec3<f32>(fx / d, 0.0, 0.0), vec3<f32>(0.0, fy / d, 0.0), vec3<f32>(fx * x / (d * d), fy * y / (d * d), 0.0));
  }
  let mv = cloud.modelView;
  let T = J * mat3x3<f32>(mv[0].xyz, mv[1].xyz, mv[2].xyz);
  let cov = T * sigma * transpose(T);
  return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}
`;

const SORT_SHADER = /* wgsl */ `
${CLOUD_WGSL}
struct DrawArgs {
  vertexCount   : u32,
  instanceCount : atomic<u32>,
  firstVertex   : u32,
  firstInstance : u32,
};

const BUCKETS = ${BUCKETS}u;

@group(0) @binding(0) var<uniform> cloud : Cloud;
@group(0) @binding(1) var<storage, read> centers : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> keys : array<u32>;
@group(0) @binding(3) var<storage, read_write> visible : array<u32>;
@group(0) @binding(4) var<storage, read_write> counts : array<atomic<u32>, BUCKETS>;
@group(0) @binding(5) var<storage, read_write> cursors : array<atomic<u32>, BUCKETS>;
@group(0) @binding(6) var<storage, read_write> order : array<u32>;
@group(0) @binding(7) var<storage, read_write> args : DrawArgs;
@group(0) @binding(8) var<storage, read> covariances : array<f32>;

${FOOTPRINT_WGSL}

/** A thread's splat: a grid of workgroups wraps past the 65535 a dimension allows. */
fn splatOf(id : vec3<u32>, groups : vec3<u32>) -> u32 {
  return id.x + id.y * groups.x * ${WORKGROUP}u;
}

@compute @workgroup_size(${WORKGROUP})
fn makeKeys(@builtin(global_invocation_id) id : vec3<u32>, @builtin(num_workgroups) groups : vec3<u32>) {
  let i = splatOf(id, groups);
  if (i >= cloud.count) { return; }
  let center = centers[i];
  let view = cloud.modelView * vec4<f32>(bitcast<vec3<f32>>(center.xyz), 1.0);
  let clip = cloud.projection * view;
  // Behind the camera, or well outside the view: its centre, with a margin
  // for the part of a splat that reaches in from beside it.
  let w = clip.w;
  // Nearer than the near plane, too: every corner shares the centre's depth,
  // so the whole quad would be clipped.
  if (view.z >= 0.0 || clip.z > w || abs(clip.x) > 1.3 * w || abs(clip.y) > 1.3 * w) { return; }
  // Too faint to reach 1/255 anywhere: the draw would discard all of it.
  let alpha = unpack4x8unorm(center.w).a;
  if (alpha * 255.0 <= 1.0) { return; }
  // Too little to see: its opacity over its area -- the Gaussian's integral,
  // 2 pi sqrt(det), as the draw sizes it -- is the most it can add to the
  // picture, and a splat under a pixel's worth costs a quad all the same.
  if (cloud.cull > 0.0) {
    let f = footprint(i, view);
    let area = 6.2831853 * sqrt(max((f.x + 0.3) * (f.z + 0.3) - f.y * f.y, 0.0));
    if (alpha * area < cloud.cull) { return; }
  }
  // Far first: the farthest key is the smallest.
  let bucket = u32(clamp((-view.z - cloud.depthMin) * cloud.depthScale, 0.0, f32(BUCKETS - 1u)));
  let key = BUCKETS - 1u - bucket;
  let slot = atomicAdd(&args.instanceCount, 1u);
  keys[slot] = key;
  visible[slot] = i;
  atomicAdd(&counts[key], 1u);
}

var<workgroup> sums : array<u32, ${WORKGROUP}>;

/** Where each bucket starts: a thread sums 256 buckets, the 256 sums are scanned, and each thread writes its own run. */
@compute @workgroup_size(${WORKGROUP})
fn scanCounts(@builtin(local_invocation_index) t : u32) {
  let first = t * ${BUCKETS / WORKGROUP}u;
  var total = 0u;
  for (var b = 0u; b < ${BUCKETS / WORKGROUP}u; b++) { total += atomicLoad(&counts[first + b]); }
  sums[t] = total;
  workgroupBarrier();
  for (var step = 1u; step < ${WORKGROUP}u; step <<= 1u) {
    var add = 0u;
    if (t >= step) { add = sums[t - step]; }
    workgroupBarrier();
    sums[t] += add;
    workgroupBarrier();
  }
  var at = sums[t] - total;
  for (var b = 0u; b < ${BUCKETS / WORKGROUP}u; b++) {
    let n = atomicLoad(&counts[first + b]);
    atomicStore(&cursors[first + b], at);
    atomicStore(&counts[first + b], 0u);   // ready for the next frame's keys
    at += n;
  }
}

@compute @workgroup_size(${WORKGROUP})
fn place(@builtin(global_invocation_id) id : vec3<u32>, @builtin(num_workgroups) groups : vec3<u32>) {
  let i = splatOf(id, groups);
  if (i >= atomicLoad(&args.instanceCount)) { return; }
  order[atomicAdd(&cursors[keys[i]], 1u)] = visible[i];
}
`;

const DRAW_SHADER = /* wgsl */ `
${CLOUD_WGSL}
${FRAME_WGSL}
${FOG_WGSL}
@group(0) @binding(0) var<uniform> cloud : Cloud;
@group(0) @binding(1) var<storage, read> centers : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> covariances : array<f32>;
@group(0) @binding(3) var<storage, read> order : array<u32>;
// With harmonics, each visible splat's colour as the shade pass left it.
@group(0) @binding(4) var<storage, read> shaded : array<u32>;
@group(1) @binding(0) var<uniform> frame : Frame;

${FOOTPRINT_WGSL}

/** Whether the capture has harmonics, so the colour is the shade pass's. */
override SHADED : bool = false;
@group(1) @binding(1) var irradiance : texture_cube<f32>;
@group(1) @binding(2) var envSampler : sampler;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  // Where in the quad, in units of the ellipse's axes: the Gaussian is exp(-|offset|^2).
  @location(0) offset : vec2<f32>,
  @location(1) colour : vec4<f32>,
};

fn decodeSRGB(c : vec3<f32>) -> vec3<f32> {
  return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@vertex
fn vs(@builtin(vertex_index) corner : u32, @builtin(instance_index) instance : u32) -> VertexOut {
  var out : VertexOut;
  // Off the screen, for a splat that projects to nothing.
  out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  let i = order[instance];
  let center = centers[i];
  let modelView = cloud.modelView;
  let view = modelView * vec4<f32>(bitcast<vec3<f32>>(center.xyz), 1.0);
  let clip = cloud.projection * view;

  let f = footprint(i, view);
  // Plus a third of a pixel each way: a splat smaller than a pixel still
  // covers one, as 3DGS trains them to.
  let a = f.x + 0.3;
  let b = f.y;
  let c = f.z + 0.3;

  // The ellipse's axes: the 2D covariance's eigenvectors, each as long as
  // the square root of twice its eigenvalue.
  let mid = 0.5 * (a + c);
  let radius = length(vec2<f32>(0.5 * (a - c), b));
  let major = mid + radius;
  let minor = mid - radius;
  // Not drawn when it projects to nothing -- or to NaN, from a size no float holds.
  if (!(minor > 0.0)) { return out; }
  var axis = vec2<f32>(1.0, 0.0);
  if (abs(b) > 1e-12) { axis = normalize(vec2<f32>(b, major - a)); } else if (c > a) { axis = vec2<f32>(0.0, 1.0); }
  let along = min(sqrt(2.0 * major), 1024.0) * axis;
  let across = min(sqrt(2.0 * minor), 1024.0) * vec2<f32>(axis.y, -axis.x);

  // Out to where it fades below 1/255, and no further than e^-4 of its peak:
  // a faint splat covers fewer pixels than an opaque one of the same size.
  var colour = unpack4x8unorm(center.w);
  if (SHADED) { colour = unpack4x8unorm(shaded[i]); }
  let reach = log(255.0 * colour.a);
  if (reach <= 0.0) { return out; }
  let q = (vec2<f32>(f32(corner & 1u), f32(corner >> 1u)) * 2.0 - 1.0) * sqrt(min(reach, 4.0));
  let pixels = q.x * along + q.y * across;
  out.position = vec4<f32>(clip.xy + pixels * 2.0 / cloud.viewport * clip.w, clip.z, clip.w);
  out.offset = q;
  var rgb = decodeSRGB(colour.rgb);
  // Fogged as a particle is, by the distance to its centre.
  if (frame.fog.x > 0.0) {
    let toSplat = (cloud.model * vec4<f32>(bitcast<vec3<f32>>(center.xyz), 1.0)).xyz - frame.cameraPosition.xyz;
    let distance = length(toSplat);
    let through = exp(-fogDepth(frame.fog, frame.cameraPosition.xyz, toSplat / max(distance, 1e-6), distance));
    let inscatter = frame.fogAlbedo.rgb * fogMeanRadiance(irradiance, envSampler) + frame.fogLight.rgb;
    rgb = rgb * through + inscatter * (1.0 - through);
  }
  out.colour = vec4<f32>(rgb, colour.a);
  return out;
}

@fragment
fn fs(v : VertexOut) -> @location(0) vec4<f32> {
  // Premultiplied, and never wholly opaque, as 3DGS draws them.
  let alpha = min(0.99, exp(-dot(v.offset, v.offset)) * v.colour.a);
  if (alpha < 1.0 / 255.0) { discard; }
  return vec4<f32>(v.colour.rgb * alpha, alpha);
}
`;

const SHADE_SHADER = /* wgsl */ `
${CLOUD_WGSL}
struct DrawArgs {
  vertexCount   : u32,
  instanceCount : u32,
  firstVertex   : u32,
  firstInstance : u32,
};

@group(0) @binding(0) var<uniform> cloud : Cloud;
@group(0) @binding(1) var<storage, read> centers : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> sh : array<u32>;
@group(0) @binding(3) var<storage, read> visible : array<u32>;
@group(0) @binding(4) var<storage, read> args : DrawArgs;
@group(0) @binding(5) var<storage, read_write> shaded : array<u32>;

/** The harmonics' degree, 1 to 3. */
override SH_DEGREE : u32 = 1u;

/**
 * Splat i's coefficient k past the constant one, as red, green and blue:
 * three half floats, from the two words they lie across.
 */
fn coefficient(i : u32, k : u32) -> vec3<f32> {
  let n = select(15u, select(8u, 3u, SH_DEGREE == 1u), SH_DEGREE < 3u);
  let h = i * ((n * 3u + 1u) / 2u) * 2u + k * 3u;
  let a = unpack2x16float(sh[h >> 1u]);
  let b = unpack2x16float(sh[(h >> 1u) + 1u]);
  return select(vec3<f32>(a.x, a.y, b.x), vec3<f32>(a.y, b.x, b.y), (h & 1u) == 1u);
}

/** What the harmonics past the constant add, seen along d -- the 3DGS reference's evaluation. */
fn harmonics(i : u32, d : vec3<f32>) -> vec3<f32> {
  let x = d.x; let y = d.y; let z = d.z;
  var c = 0.4886025119029199 * (-y * coefficient(i, 0u) + z * coefficient(i, 1u) - x * coefficient(i, 2u));
  if (SH_DEGREE >= 2u) {
    let xx = x * x; let yy = y * y; let zz = z * z;
    c += 1.0925484305920792 * x * y * coefficient(i, 3u)
      - 1.0925484305920792 * y * z * coefficient(i, 4u)
      + 0.31539156525252005 * (2.0 * zz - xx - yy) * coefficient(i, 5u)
      - 1.0925484305920792 * x * z * coefficient(i, 6u)
      + 0.5462742152960396 * (xx - yy) * coefficient(i, 7u);
    if (SH_DEGREE >= 3u) {
      c += -0.5900435899266435 * y * (3.0 * xx - yy) * coefficient(i, 8u)
        + 2.890611442640554 * x * y * z * coefficient(i, 9u)
        - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * coefficient(i, 10u)
        + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coefficient(i, 11u)
        - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * coefficient(i, 12u)
        + 1.445305721320277 * z * (xx - yy) * coefficient(i, 13u)
        - 0.5900435899266435 * x * (xx - 3.0 * yy) * coefficient(i, 14u);
    }
  }
  return c;
}

/** Each visible splat's colour, seen from the eye: the constant term and the harmonics, clamped as 3DGS clamps them. */
@compute @workgroup_size(${WORKGROUP})
fn shade(@builtin(global_invocation_id) id : vec3<u32>, @builtin(num_workgroups) groups : vec3<u32>) {
  let slot = id.x + id.y * groups.x * ${WORKGROUP}u;
  if (slot >= args.instanceCount) { return; }
  let i = visible[slot];
  let center = centers[i];
  let base = unpack4x8unorm(center.w);
  let d = normalize(bitcast<vec3<f32>>(center.xyz) - cloud.eye);
  let seen = clamp(base.rgb + harmonics(i, d), vec3<f32>(0.0), vec3<f32>(1.0));
  shaded[i] = pack4x8unorm(vec4<f32>(seen, base.a));
}
`;

/**
 * A capture on the GPU: engine.loadSplats makes one, scene.addSplats places
 * it, any number of times. `count` splats; `min` and `max`, the corners of
 * the box around their centres, in the capture's own units.
 */
export class Splats {
  constructor(rhi, data, label = 'splats') {
    // Refused here, by name, rather than as a validation error on the first
    // frame: the covariances are the largest buffer a shader binds whole.
    const most = storageCapacity(rhi, 24);
    if (data.count > most) {
      throw new Error(`splats: ${data.count} splats are more than this device holds in one buffer, ${most}`);
    }
    this.degree = data.degree ?? 0;
    if (this.degree > 0 && data.count > storageCapacity(rhi, shWords(this.degree) * 4)) {
      throw new Error(`splats: ${data.count} splats with harmonics of degree ${this.degree} are more than this device holds in one buffer`);
    }
    this.rhi = rhi;
    this.count = data.count;
    this.min = data.min;
    this.max = data.max;
    this.centers = createBuffer(rhi, { label: `${label}-centers`, data: data.centers, usage: GPUBufferUsage.STORAGE });
    this.covariances = createBuffer(rhi, { label: `${label}-covariances`, data: data.covariances, usage: GPUBufferUsage.STORAGE });
    // One word when there are none: the binding is there for every capture.
    this.sh = createBuffer(rhi, { label: `${label}-sh`, data: data.sh ?? new Uint32Array(1), usage: GPUBufferUsage.STORAGE });
  }

  destroy() {
    this.centers.destroy();
    this.covariances.destroy();
    this.sh.destroy();
  }
}

export class SplatPass {
  static async create(rhi, pipelines, colorFormat, frameBuffer) {
    const device = rhi.device;
    const [sortShader, drawShader, shadeShader] = await Promise.all([
      compileShader(device, SORT_SHADER, 'splats-sort.wgsl'),
      compileShader(device, DRAW_SHADER, 'splats.wgsl'),
      compileShader(device, SHADE_SHADER, 'splats-shade.wgsl'),
    ]);
    const storage = (binding, type = 'storage') => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const sortLayout = device.createBindGroupLayout({
      label: 'splats-sort',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        storage(1, 'read-only-storage'), storage(2), storage(3), storage(4), storage(5), storage(6), storage(7),
        storage(8, 'read-only-storage'),
      ],
    });
    const shadeLayout = device.createBindGroupLayout({
      label: 'splats-shade',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        storage(1, 'read-only-storage'), storage(2, 'read-only-storage'), storage(3, 'read-only-storage'),
        storage(4, 'read-only-storage'), storage(5),
      ],
    });
    const drawLayout = device.createBindGroupLayout({
      label: 'splats-draw',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const frameLayout = device.createBindGroupLayout({
      label: 'splats-frame',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { viewDimension: 'cube' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, sampler: {} },
      ],
    });
    const sort = createPipelineLayout(device, { 0: sortLayout }, 'splats-sort');
    const compute = (entry) => pipelines.compute({ label: `splats-${entry}`, layout: sort, shader: sortShader, entry });
    const drawDescriptor = {
      label: 'splats',
      layout: createPipelineLayout(device, { 0: drawLayout, 1: frameLayout }, 'splats'),
      shader: drawShader,
      buffers: [],
      targets: [{
        format: colorFormat,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
      depth: { format: DEPTH_FORMAT, depthCompare: DEPTH_COMPARE, depthWriteEnabled: false },
    };
    // Plain now; the one that reads shaded colours when a capture with harmonics loads.
    const shadedDescriptor = { ...drawDescriptor, label: 'splats-shaded', constants: { SHADED: 1 } };
    await pipelines.warm([drawDescriptor]);
    const shadeLayoutGPU = createPipelineLayout(device, { 0: shadeLayout }, 'splats-shade');
    return new SplatPass(rhi, frameBuffer, {
      sortLayout, drawLayout, frameLayout, shadeLayout, pipelines, shadedDescriptor,
      shadeFor: (degree) => pipelines.compute({
        label: `splats-shade${degree}`, layout: shadeLayoutGPU, shader: shadeShader, entry: 'shade', constants: { SH_DEGREE: degree },
      }),
      keys: compute('makeKeys'), scan: compute('scanCounts'), scatter: compute('place'),
      draw: pipelines.get(drawDescriptor),
      shades: [],
    });
  }

  constructor(rhi, frameBuffer, built) {
    this.rhi = rhi;
    this._frameBuffer = frameBuffer;
    /** The frame's group, by environment: its irradiance is what fog scatters. */
    this._frameGroups = new WeakMap();
    Object.assign(this, built);
    /** Each placed cloud's sort buffers, by its scene record. */
    this._states = new Map();
    this._items = [];
    this._frame = 0;
    this._modelView = new Float32Array(16);
    this._viewModel = new Float32Array(16);
    this._cloud = new ArrayBuffer(CLOUD_BYTES);
    this._cloudF32 = new Float32Array(this._cloud);
    this._cloudU32 = new Uint32Array(this._cloud);
    this._args = new Uint32Array([4, 0, 0, 0]);
    /** Splats in the clouds the last frame drew, before culling. */
    this.count = 0;
    /** Clouds the last frame sorted; the rest kept the order they had. */
    this.sorted = 0;
    /** The least a splat may add to the picture, opacity x pixels, and be drawn; see makeKeys. */
    this.cull = SPLAT_CULL;
    /**
     * The projection without TAA's jitter, when the frame is jittered: what
     * the sort is decided by. The jitter moves a fraction of a pixel every
     * frame, and a still camera under TAA re-sorted every cloud every frame.
     */
    this.unjittered = null;
    this._sortedFor = new Float32Array(CLOUD_FLOATS);
    this._sort = (pass) => this._encodeSort(pass);
    this._draw = (pass) => this._encodeDraw(pass);
  }

  /**
   * Builds the pipeline for harmonics of `degree`, if it is not built: called
   * by engine.loadSplats, so a frame never compiles one.
   */
  async ready(degree) {
    if (degree === 0 || this.shades[degree] !== undefined) return;
    if (this.drawShaded === undefined) {
      await this.pipelines.warm([this.shadedDescriptor]);
      this.drawShaded = this.pipelines.get(this.shadedDescriptor);
    }
    this.shades[degree] = this.shadeFor(degree);
  }

  /**
   * This frame's clouds: each one whose camera, transform or canvas moved
   * has them written and its count reset, to be sorted again. Returns how
   * many clouds there are.
   */
  prepare(scene, camera, environment, width, height) {
    this._frame++;
    this._environment = environment;
    const items = this._items;
    items.length = 0;
    this.count = 0;
    this.sorted = 0;
    for (const [entity, record] of scene.splats) {
      const splats = record.splats;
      if (splats.unloaded) throw new Error('addSplats: these splats were unloaded; remove the node first');
      const state = this._stateFor(record, splats, scene, entity);
      state.lastFrame = this._frame;
      const modelView = this._modelView;
      mat4Multiply(modelView, camera.view, scene.transforms.world, 0, 0, handleIndex(entity) * 16);
      // The depth the cloud spans in view: its box's eight corners.
      let near = Infinity;
      let far = -Infinity;
      for (let k = 0; k < 8; k++) {
        const x = k & 1 ? splats.max[0] : splats.min[0];
        const y = k & 2 ? splats.max[1] : splats.min[1];
        const z = k & 4 ? splats.max[2] : splats.min[2];
        const depth = -(modelView[2] * x + modelView[6] * y + modelView[10] * z + modelView[14]);
        if (depth < near) near = depth;
        if (depth > far) far = depth;
      }
      if (far <= 0) continue;   // wholly behind the camera
      near = Math.max(near, 0);
      const f = this._cloudF32;
      f.set(modelView, 0);
      f.set(camera.projection, 16);
      f[32] = width;
      f[33] = height;
      f[34] = near;
      // Floored: a cloud with no depth -- one splat, or a flat one edge on --
      // spans none, and is all in one bucket.
      f[35] = (BUCKETS - 1) / Math.max(far - near, 1e-6);
      this._cloudU32[36] = splats.count;
      f.set(scene.transforms.world.subarray(handleIndex(entity) * 16, handleIndex(entity) * 16 + 16), 40);
      // The eye, in the capture's space: where (view x model)^-1 takes the view's origin.
      if (splats.degree > 0 && mat4Invert(this._viewModel, modelView) !== null) f.set(this._viewModel.subarray(12, 15), 56);
      f[59] = this.cull;
      this.rhi.queue.writeBuffer(state.cloud, 0, this._cloud);
      // Sorted again only when something it is sorted by changed. What it
      // was sorted for is kept when the sort is recorded, not here: a frame
      // that throws before then must not leave it looking sorted.
      // A new cull threshold changes which splats are in it, as a move does.
      const key = this._sortedFor;
      key.set(f.subarray(0, CLOUD_FLOATS));
      if (this.unjittered !== null) key.set(this.unjittered, 16);
      const sort = !sameFloats(state.last, key, CLOUD_FLOATS) || state.cull !== this.cull;
      state.cull = this.cull;
      if (sort) {
        state.next.set(key);
        this.rhi.queue.writeBuffer(state.args, 0, this._args);
        this.sorted++;
      }
      const cx = (splats.min[0] + splats.max[0]) / 2, cy = (splats.min[1] + splats.max[1]) / 2, cz = (splats.min[2] + splats.max[2]) / 2;
      items.push({ state, sort, count: splats.count, degree: splats.degree, depth: -(modelView[2] * cx + modelView[6] * cy + modelView[10] * cz + modelView[14]) });
      this.count += splats.count;
    }
    items.sort((a, b) => b.depth - a.depth);
    // A cloud gives its buffers back once its node is removed from this
    // scene, or when no frame has drawn it for a while.
    for (const [record, state] of this._states) {
      const removed = state.scene === scene && scene.splats.get(state.entity) !== record;
      if (removed || state.lastFrame < this._frame - KEEP_FRAMES) {
        destroyState(state);
        this._states.delete(record);
      }
    }
    return items.length;
  }

  /** Sorted on the GPU where it needs to be, then drawn over `sceneColor`, behind what `depth` holds. */
  addPasses(graph, { sceneColor, depth }) {
    if (this._items.length === 0) return;
    // One of the sorted orders stands for all of them: it is what puts the draw after the sort.
    const order = graph.importBuffer('splat-order', this._items[0].state.order);
    if (this.sorted > 0) graph.addPass({ name: 'splats:sort', type: 'compute', writes: [order], execute: this._sort });
    graph.addPass({ name: 'splats', reads: [order], color: [{ resource: sceneColor }], depth: { resource: depth }, execute: this._draw });
  }

  _stateFor(record, splats, scene, entity) {
    let state = this._states.get(record);
    if (state !== undefined && state.splats === splats) return state;
    if (state !== undefined) destroyState(state);
    const rhi = this.rhi;
    const n = Math.max(1, splats.count);
    const make = (label, size, usage = GPUBufferUsage.STORAGE) => createBuffer(rhi, { label: `splats-${label}`, size, usage });
    state = {
      splats,
      scene,
      entity,
      lastFrame: 0,
      // What it was last sorted for. NaN matches nothing, so the first frame sorts.
      last: new Float32Array(CLOUD_FLOATS).fill(NaN),
      next: new Float32Array(CLOUD_FLOATS),
      cloud: make('cloud', CLOUD_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      keys: make('keys', n * 4),
      visible: make('visible', n * 4),
      counts: make('counts', BUCKETS * 4),
      cursors: make('cursors', BUCKETS * 4),
      order: make('order', n * 4),
      args: make('args', 16, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST),
      shaded: make('shaded', splats.degree > 0 ? n * 4 : 4),
    };
    const device = rhi.device;
    const entries = (list) => list.map((buffer, binding) => ({ binding, resource: { buffer } }));
    state.sortGroup = device.createBindGroup({
      label: 'splats-sort',
      layout: this.sortLayout,
      entries: entries([state.cloud, splats.centers, state.keys, state.visible, state.counts, state.cursors, state.order, state.args, splats.covariances]),
    });
    state.drawGroup = device.createBindGroup({
      label: 'splats-draw',
      layout: this.drawLayout,
      entries: entries([state.cloud, splats.centers, splats.covariances, state.order, state.shaded]),
    });
    if (splats.degree > 0) {
      state.shadeGroup = device.createBindGroup({
        label: 'splats-shade',
        layout: this.shadeLayout,
        entries: entries([state.cloud, splats.centers, splats.sh, state.visible, state.args, state.shaded]),
      });
    }
    this._states.set(record, state);
    return state;
  }

  _encodeSort(pass) {
    const limit = this.rhi.limits.maxComputeWorkgroupsPerDimension ?? 65535;
    for (const { state, sort, count, degree } of this._items) {
      if (!sort) continue;
      state.last.set(state.next);
      const groups = Math.ceil(count / WORKGROUP);
      const x = Math.min(groups, limit);
      const y = Math.ceil(groups / x);
      pass.setBindGroup(0, state.sortGroup);
      pass.setPipeline(this.keys);
      pass.dispatchWorkgroups(x, y);
      pass.setPipeline(this.scan);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.scatter);
      pass.dispatchWorkgroups(x, y);
      if (degree > 0) {
        pass.setBindGroup(0, state.shadeGroup);
        pass.setPipeline(this.shades[degree]);
        pass.dispatchWorkgroups(x, y);
      }
    }
  }

  _encodeDraw(pass) {
    const environment = this._environment;
    let frameGroup = this._frameGroups.get(environment);
    if (!frameGroup) {
      frameGroup = this.rhi.device.createBindGroup({
        label: 'splats-frame',
        layout: this.frameLayout,
        entries: [
          { binding: 0, resource: { buffer: this._frameBuffer } },
          { binding: 1, resource: environment.irradianceView },
          { binding: 2, resource: environment.sampler },
        ],
      });
      this._frameGroups.set(environment, frameGroup);
    }
    pass.setBindGroup(1, frameGroup);
    let pipeline = null;
    for (const { state, degree } of this._items) {
      const wanted = degree > 0 ? this.drawShaded : this.draw;
      if (wanted !== pipeline) pass.setPipeline(pipeline = wanted);
      pass.setBindGroup(0, state.drawGroup);
      pass.drawIndirect(state.args, 0);
    }
  }

  destroy() {
    for (const state of this._states.values()) destroyState(state);
    this._states.clear();
  }
}

function sameFloats(a, b, n) {
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return false;
  return true;
}

function destroyState(state) {
  for (const name of ['cloud', 'keys', 'visible', 'counts', 'cursors', 'order', 'args', 'shaded']) state[name].destroy();
}
