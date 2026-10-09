// Particles: born by emitters (Scene.addEmitter), carried on the GPU, drawn
// as camera-facing quads.
//
// Every emitter's particles live in one pool buffer, each emitter a ring in
// it: a new particle takes the slot after the last, and a ring is sized to
// hold everything the emitter can have alive at once -- its rate times its
// longest lifetime, plus a burst -- so a slot comes round again only once its
// particle is dead. Nothing about a particle is touched on the CPU: one
// compute dispatch an emitter births and moves them, and one instanced draw
// an emitter reads the pool where they are.
//
// Motion is solved, not stepped. Under constant acceleration a and drag k,
// v(t) = a/k + (v0 - a/k) e^(-k t), and its integral is the position; with no
// drag, the familiar v0 t + a t^2 / 2. So a particle follows the same path at
// 30 frames a second as at 144.
//
// Drawn after sprites, before glass: alpha emitters far to near, then
// additive ones, which need no order. An alpha emitter's own particles are
// sorted far to near on the GPU every frame, as splats are (render/splats.js):
// the live ones keyed by depth, put in order by a counting sort, and drawn
// indirectly, so dead slots are not drawn at all. Twelve bits of log depth
// are enough for smoke -- a key a third of a percent of the distance wide --
// and keep the sort's own pass over its buckets small for every emitter.
// Measured on Iris Xe: 0.18 ms for an emitter of a thousand particles, 0.7
// ms for ten thousand, about half of it their atomics meeting in the few
// buckets one emitter's depth spans; 65536 buckets made it worse, the scan
// over them costing more than the spread saved. Two alpha emitters are each
// sorted on their own, so where they overlap they do not interleave. A 2D view draws particles in birth order: they
// are flat.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { DEPTH_FORMAT, DEPTH_COMPARE } from '../rhi/device.js';
import { clampSampler, defaultTextures } from '../rhi/texture.js';
import { grownCapacity } from '../core/grow.js';
import { handleIndex } from '../core/handle.js';
import { FRAME_WGSL } from './shaders/pbr.js';
import { FOG_WGSL } from './fog.js';

export const PARTICLE_BYTES = 32;
const EMITTER_BYTES = 192;
const WORKGROUP = 64;
/** Depth buckets for an alpha emitter's sort: 12 bits of log depth. */
const BUCKETS = 4096;
/** The log2 of the depths the buckets span, a millimetre to a thousand kilometres. */
const LOG_NEAR = -10;
const LOG_FAR = 20;

const EMITTER_WGSL = /* wgsl */ `
struct Particle {
  position : vec3<f32>,
  age      : f32,
  velocity : vec3<f32>,
  lifetime : f32,        // 0 for a slot never used: dead, since age >= lifetime
};

struct Emitter {
  spawnMatrix  : mat4x4<f32>,   //   0  the emitter's world transform this frame
  offset       : u32,           //  64  its ring in the pool
  capacity     : u32,           //  68
  spawnFirst   : u32,           //  72  where this frame's births start in the ring
  spawnCount   : u32,           //  76
  dt           : f32,           //  80  the seconds this frame carries its particles
  seed         : u32,           //  84
  radius       : f32,           //  88
  spread       : f32,           //  92
  direction    : vec4<f32>,     //  96  in the emitter's space; w: 1 if it has a texture, + 2 if seen in 2D
  speed        : vec2<f32>,     // 112  [min, max]
  lifetime     : vec2<f32>,     // 120  [min, max]
  acceleration : vec4<f32>,     // 128  world space; w = drag
  size         : vec4<f32>,     // 144  x at birth, y at death
  colorStart   : vec4<f32>,     // 160
  colorEnd     : vec4<f32>,     // 176
};                              // 192
`;

const SIMULATE_SHADER = /* wgsl */ `
${EMITTER_WGSL}
@group(0) @binding(0) var<uniform> emitter : Emitter;
@group(0) @binding(1) var<storage, read_write> pool : array<Particle>;

const TAU = 6.28318530718;

/** PCG, one round: a well-mixed u32 from any u32. */
fn hash(x : u32) -> u32 {
  let state = x * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

/** A uniform number in [0, 1), advancing the stream. */
fn random(n : ptr<function, u32>) -> f32 {
  *n = hash(*n);
  return f32(*n >> 8u) / 16777216.0;
}

/** Carry a particle t seconds along its solved path. */
fn carry(p : ptr<function, Particle>, t : f32) {
  let a = emitter.acceleration.xyz;
  let k = emitter.acceleration.w;
  let v0 = (*p).velocity;
  if (k > 0.0) {
    let terminal = a / k;
    let decay = exp(-k * t);
    (*p).position = (*p).position + terminal * t + (v0 - terminal) * (1.0 - decay) / k;
    (*p).velocity = terminal + (v0 - terminal) * decay;
  } else {
    (*p).position = (*p).position + v0 * t + 0.5 * a * t * t;
    (*p).velocity = v0 + a * t;
  }
  (*p).age = (*p).age + t;
}

@compute @workgroup_size(${WORKGROUP})
fn simulate(@builtin(global_invocation_id) id : vec3<u32>) {
  let i = id.x;
  if (i >= emitter.capacity) { return; }
  let slot = emitter.offset + i;

  // Born this frame: somewhere in its ring's newest stretch.
  if ((i + emitter.capacity - emitter.spawnFirst) % emitter.capacity < emitter.spawnCount) {
    var n = hash(emitter.seed ^ hash(i));
    var p : Particle;
    // Seen in 2D, everything stays in the screen's plane: a direction turned
    // up to \`spread\` either way, and a start in a disc.
    let flat = (u32(emitter.direction.w) & 2u) != 0u;
    var local : vec3<f32>;
    if (flat) {
      let turn = (2.0 * random(&n) - 1.0) * emitter.spread;
      let aim = normalize(emitter.direction.xy);
      local = vec3<f32>(aim.x * cos(turn) - aim.y * sin(turn), aim.x * sin(turn) + aim.y * cos(turn), 0.0);
    } else {
      // A direction in the cone, uniform over the cap of the sphere it cuts.
      let cosTheta = mix(1.0, cos(emitter.spread), random(&n));
      let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
      let phi = TAU * random(&n);
      let axis = normalize(emitter.direction.xyz);
      let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(axis.x) > 0.9);
      let tangent = normalize(cross(helper, axis));
      let bitangent = cross(axis, tangent);
      local = axis * cosTheta + (tangent * cos(phi) + bitangent * sin(phi)) * sinTheta;
    }
    let heading = (emitter.spawnMatrix * vec4<f32>(local, 0.0)).xyz;
    // A point in the ball, uniform by volume; in the disc, by area.
    let z = select(2.0 * random(&n) - 1.0, 0.0, flat);
    let around = TAU * random(&n);
    let ring = sqrt(max(1.0 - z * z, 0.0));
    let reach = select(pow(random(&n), 1.0 / 3.0), sqrt(random(&n)), flat);
    let within = vec3<f32>(ring * cos(around), ring * sin(around), z) * emitter.radius * reach;
    p.position = (emitter.spawnMatrix * vec4<f32>(within, 1.0)).xyz;
    p.velocity = normalize(heading) * mix(emitter.speed.x, emitter.speed.y, random(&n));
    p.lifetime = mix(emitter.lifetime.x, emitter.lifetime.y, random(&n));
    p.age = 0.0;
    // Born at a moment within this frame, and carried from then: a stream
    // arrives as a stream rather than in a pulse a frame.
    carry(&p, random(&n) * emitter.dt);
    pool[slot] = p;
    return;
  }

  var p = pool[slot];
  if (p.age >= p.lifetime) { return; }
  carry(&p, emitter.dt);
  pool[slot] = p;
}
`;

const SORT_SHADER = /* wgsl */ `
${EMITTER_WGSL}
struct View {
  eye     : vec4<f32>,
  forward : vec4<f32>,
};

struct DrawArgs {
  vertexCount   : u32,
  instanceCount : atomic<u32>,
  firstVertex   : u32,
  firstInstance : u32,
};

const BUCKETS = ${BUCKETS}u;

@group(0) @binding(0) var<uniform> emitter : Emitter;
@group(0) @binding(1) var<storage, read> pool : array<Particle>;
@group(0) @binding(2) var<storage, read_write> keys : array<u32>;
@group(0) @binding(3) var<storage, read_write> visible : array<u32>;
@group(0) @binding(4) var<storage, read_write> counts : array<atomic<u32>, BUCKETS>;
@group(0) @binding(5) var<storage, read_write> cursors : array<atomic<u32>, BUCKETS>;
@group(0) @binding(6) var<storage, read_write> order : array<u32>;
@group(0) @binding(7) var<storage, read_write> args : DrawArgs;
@group(0) @binding(8) var<uniform> view : View;

/** Each live particle, appended with its key: far first. */
@compute @workgroup_size(${WORKGROUP})
fn makeKeys(@builtin(global_invocation_id) id : vec3<u32>) {
  let i = id.x;
  if (i >= emitter.capacity) { return; }
  let slot = emitter.offset + i;
  let p = pool[slot];
  if (p.age >= p.lifetime) { return; }
  let depth = max(dot(p.position - view.eye.xyz, view.forward.xyz), 1e-6);
  let bucket = u32(clamp((log2(depth) - ${LOG_NEAR}.0) * f32(BUCKETS - 1u) / ${LOG_FAR - LOG_NEAR}.0, 0.0, f32(BUCKETS - 1u)));
  let key = BUCKETS - 1u - bucket;
  let at = atomicAdd(&args.instanceCount, 1u);
  keys[emitter.offset + at] = key;
  visible[emitter.offset + at] = slot;
  atomicAdd(&counts[key], 1u);
}

var<workgroup> sums : array<u32, 256>;

/** Where each bucket starts, and the counts zeroed for the next emitter. */
@compute @workgroup_size(256)
fn scanCounts(@builtin(local_invocation_index) t : u32) {
  let first = t * ${BUCKETS / 256}u;
  var total = 0u;
  for (var b = 0u; b < ${BUCKETS / 256}u; b++) { total += atomicLoad(&counts[first + b]); }
  sums[t] = total;
  workgroupBarrier();
  for (var step = 1u; step < 256u; step <<= 1u) {
    var add = 0u;
    if (t >= step) { add = sums[t - step]; }
    workgroupBarrier();
    sums[t] += add;
    workgroupBarrier();
  }
  var at = sums[t] - total;
  for (var b = 0u; b < ${BUCKETS / 256}u; b++) {
    let n = atomicLoad(&counts[first + b]);
    atomicStore(&cursors[first + b], at);
    atomicStore(&counts[first + b], 0u);
    at += n;
  }
}

@compute @workgroup_size(${WORKGROUP})
fn place(@builtin(global_invocation_id) id : vec3<u32>) {
  let i = id.x;
  if (i >= atomicLoad(&args.instanceCount)) { return; }
  order[emitter.offset + atomicAdd(&cursors[keys[emitter.offset + i]], 1u)] = visible[emitter.offset + i];
}
`;

const DRAW_SHADER = /* wgsl */ `
${FRAME_WGSL}
${FOG_WGSL}
${EMITTER_WGSL}

struct Camera {
  right : vec4<f32>,
  up    : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame      : Frame;
@group(0) @binding(1) var<uniform> camera     : Camera;
@group(0) @binding(2) var          irradiance : texture_cube<f32>;
@group(0) @binding(3) var          envSampler : sampler;
@group(1) @binding(0) var<uniform> emitter    : Emitter;
@group(1) @binding(1) var<storage, read> pool : array<Particle>;
// An alpha emitter's live particles, far to near: slots in the pool.
@group(1) @binding(2) var<storage, read> order : array<u32>;
@group(2) @binding(0) var          image      : texture_2d<f32>;
@group(2) @binding(1) var          imageSampler : sampler;

// 0 alpha, 1 additive.
override ADDITIVE : bool = true;

struct Out {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv    : vec2<f32>,
  @location(1) color : vec4<f32>,
  @location(2) world : vec3<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32, @builtin(instance_index) instance : u32) -> Out {
  var out : Out;
  // Additive: every slot of the ring, in place. Alpha: the sorted live ones.
  var slot = instance;
  if (!ADDITIVE) { slot = order[emitter.offset + instance]; }
  let p = pool[slot];
  // A dead slot: all six corners outside the depth range, so nothing is drawn.
  if (p.age >= p.lifetime) {
    out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    return out;
  }
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
  );
  let corner = corners[index];
  let t = clamp(p.age / p.lifetime, 0.0, 1.0);
  let size = mix(emitter.size.x, emitter.size.y, t);
  let offset = (corner - vec2<f32>(0.5)) * size;
  out.world = p.position + camera.right.xyz * offset.x + camera.up.xyz * offset.y;
  out.clip = frame.viewProjection * vec4<f32>(out.world, 1.0);
  out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
  out.color = mix(emitter.colorStart, emitter.colorEnd, t);
  return out;
}

@fragment
fn fs(v : Out) -> @location(0) vec4<f32> {
  let sampled = textureSample(image, imageSampler, v.uv);
  // Without a texture, a soft round dot: full at the centre, nothing at the rim.
  let r = v.uv * 2.0 - 1.0;
  let disc = vec4<f32>(1.0, 1.0, 1.0, clamp(1.0 - dot(r, r), 0.0, 1.0));
  var colour = select(disc, sampled, (u32(emitter.direction.w) & 1u) != 0u) * v.color;
  if (frame.fog.x > 0.0) {
    let toParticle = v.world - frame.cameraPosition.xyz;
    let distance = length(toParticle);
    let through = exp(-fogDepth(frame.fog, frame.cameraPosition.xyz, toParticle / max(distance, 1e-6), distance));
    let inscatter = frame.fogAlbedo.rgb * fogMeanRadiance(irradiance, envSampler) + frame.fogLight.rgb;
    colour = vec4<f32>(select(colour.rgb * through + inscatter * (1.0 - through), colour.rgb * through, ADDITIVE), colour.a);
  }
  if (ADDITIVE) { return vec4<f32>(min(colour.rgb * colour.a, vec3<f32>(65504.0)), 0.0); }
  return vec4<f32>(min(colour.rgb, vec3<f32>(65504.0)), colour.a);
}
`;

// The same particles through a Camera2D (render/view2d.js): flat in the view's
// plane, with its colours -- sRGB, as every 2D colour is -- and no depth or fog.
const DRAW_2D_SHADER = /* wgsl */ `
${EMITTER_WGSL}

@group(0) @binding(0) var<uniform> viewProjection : mat4x4<f32>;
@group(1) @binding(0) var<uniform> emitter    : Emitter;
@group(1) @binding(1) var<storage, read> pool : array<Particle>;
@group(2) @binding(0) var          image      : texture_2d<f32>;
@group(2) @binding(1) var          imageSampler : sampler;

override ADDITIVE : bool = true;

struct Out {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv    : vec2<f32>,
  @location(1) color : vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32, @builtin(instance_index) slot : u32) -> Out {
  var out : Out;
  let p = pool[slot];
  if (p.age >= p.lifetime) {
    out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    return out;
  }
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
  );
  let corner = corners[index];
  let t = clamp(p.age / p.lifetime, 0.0, 1.0);
  // y points down, so the image's top is at the corner with the smaller y.
  let at = p.position.xy + (corner - vec2<f32>(0.5)) * mix(emitter.size.x, emitter.size.y, t);
  out.clip = viewProjection * vec4<f32>(at, 0.0, 1.0);
  out.uv = corner;
  out.color = mix(emitter.colorStart, emitter.colorEnd, t);
  return out;
}

fn encode(c : vec3<f32>) -> vec3<f32> {
  let v = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  return select(v * 12.92, 1.055 * pow(v, vec3<f32>(1.0 / 2.4)) - 0.055, v > vec3<f32>(0.0031308));
}

@fragment
fn fs(v : Out) -> @location(0) vec4<f32> {
  let sampled = textureSample(image, imageSampler, v.uv);
  let r = v.uv * 2.0 - 1.0;
  let disc = vec4<f32>(1.0, 1.0, 1.0, clamp(1.0 - dot(r, r), 0.0, 1.0));
  // An image is sampled as linear light: back to the sRGB the 2D view blends.
  let colour = select(disc, vec4<f32>(encode(sampled.rgb), sampled.a), (u32(emitter.direction.w) & 1u) != 0u) * v.color;
  let rgb = clamp(colour.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  if (ADDITIVE) { return vec4<f32>(rgb * colour.a, 0.0); }
  return vec4<f32>(rgb, colour.a);
}
`;

/**
 * How many slots an emitter's ring needs: everything it can have alive at
 * once -- its rate times its longest lifetime, rounded up -- plus what it is
 * about to birth.
 */
export function ringCapacity(record, births) {
  return Math.max(1, Math.ceil(record.rate * record.lifetime[1]) + births);
}

/** Pack an emitter's uniform: see Emitter in the shader. */
export function packEmitter(out, o, record, world, m, ring, births, dt, seed, flat = false) {
  const f32 = new Float32Array(out.buffer, out.byteOffset + o, EMITTER_BYTES / 4);
  const u32 = new Uint32Array(out.buffer, out.byteOffset + o, EMITTER_BYTES / 4);
  f32.set(world.subarray(m, m + 16), 0);
  u32[16] = ring.offset;
  u32[17] = ring.capacity;
  u32[18] = ring.head;
  u32[19] = births;
  f32[20] = dt;
  u32[21] = seed;
  f32[22] = record.radius;
  f32[23] = record.spread;
  f32.set(record.direction, 24);
  f32[27] = (record.texture ? 1 : 0) + (flat ? 2 : 0);
  f32.set(record.speed, 28);
  f32.set(record.lifetime, 30);
  f32.set(record.acceleration, 32);
  f32[35] = record.drag;
  f32[36] = record.size[0];
  f32[37] = record.size[1];
  f32.set(record.color, 40);
  f32.set(record.colorEnd, 44);
}

export class ParticleSystem {
  static async create(rhi, pipelines, frameBuffer, colorFormat) {
    const device = rhi.device;
    const [simulateShader, sortShader, drawShader, draw2DShader] = await Promise.all([
      compileShader(device, SIMULATE_SHADER, 'particles-simulate.wgsl'),
      compileShader(device, SORT_SHADER, 'particles-sort.wgsl'),
      compileShader(device, DRAW_SHADER, 'particles-draw.wgsl'),
      compileShader(device, DRAW_2D_SHADER, 'particles-draw-2d.wgsl'),
    ]);
    const simulateLayout = device.createBindGroupLayout({
      label: 'particles-simulate',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: EMITTER_BYTES } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const frameLayout = device.createBindGroupLayout({
      label: 'particles-frame',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const emitterLayout = device.createBindGroupLayout({
      label: 'particles-emitter',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: EMITTER_BYTES } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const storage = (binding, type = 'storage', extra = {}) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, ...extra } });
    const sortLayout = device.createBindGroupLayout({
      label: 'particles-sort',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: EMITTER_BYTES } },
        storage(1, 'read-only-storage'), storage(2), storage(3), storage(4), storage(5), storage(6),
        storage(7, 'storage', { hasDynamicOffset: true, minBindingSize: 16 }),
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const sortPipelineLayout = createPipelineLayout(device, { 0: sortLayout }, 'particles-sort');
    const sorter = (entry) => pipelines.compute({ label: `particles-${entry}`, layout: sortPipelineLayout, shader: sortShader, entry });
    const sort = { keys: sorter('makeKeys'), scan: sorter('scanCounts'), place: sorter('place') };
    const imageLayout = device.createBindGroupLayout({
      label: 'particles-image',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const simulate = pipelines.compute({
      label: 'particles-simulate',
      layout: createPipelineLayout(device, { 0: simulateLayout }, 'particles-simulate'),
      shader: simulateShader,
      entry: 'simulate',
    });
    const drawLayout = createPipelineLayout(device, { 0: frameLayout, 1: emitterLayout, 2: imageLayout }, 'particles');
    const descriptors = {};
    for (const blend of ['alpha', 'additive']) {
      descriptors[blend] = {
        label: `particles:${blend}`,
        layout: drawLayout,
        shader: drawShader,
        buffers: [],
        targets: [{
          format: colorFormat,
          blend: blend === 'additive'
            ? {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
            }
            : {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
        }],
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depth: { format: DEPTH_FORMAT, depthCompare: DEPTH_COMPARE, depthWriteEnabled: false },
        constants: { ADDITIVE: blend === 'additive' ? 1 : 0 },
      };
    }
    // A 2D view's: onto the canvas's plain view, with no depth.
    const flatLayout = device.createBindGroupLayout({
      label: 'particles-2d',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    });
    const flatPipelineLayout = createPipelineLayout(device, { 0: flatLayout, 1: emitterLayout, 2: imageLayout }, 'particles-2d');
    for (const blend of ['alpha', 'additive']) {
      descriptors[`${blend}2D`] = {
        ...descriptors[blend], label: `particles-2d:${blend}`, layout: flatPipelineLayout, shader: draw2DShader,
        targets: [{ ...descriptors[blend].targets[0], format: rhi.surfaceFormat }], depth: null,
      };
    }
    await pipelines.warm(Object.values(descriptors));
    return new ParticleSystem(rhi, pipelines, { simulate, sort, sortLayout, descriptors, simulateLayout, frameLayout, emitterLayout, imageLayout, flatLayout }, frameBuffer);
  }

  constructor(rhi, pipelines, built, frameBuffer) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    Object.assign(this, built);
    this._frameBuffer = frameBuffer;
    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    this.stride = Math.ceil(EMITTER_BYTES / this.alignment) * this.alignment;
    /**
     * Each emitter's ring, by its seed: { offset, capacity, head }. Not by
     * its entity: every scene numbers its own, so an emitter in the next
     * scene drawn could share one with this scene's, and took over its ring
     * and the particles still alive in it. A seed is the emitter's alone,
     * and setEmitter keeps it.
     */
    this.rings = new Map();
    this._seeds = new Set();
    this.pool = null;
    this.poolSize = 0;
    this._uniform = null;
    this._uniformCapacity = 0;
    this._staging = new Uint8Array(0);
    this._camera = new Float32Array(8);
    this._cameraBuffer = createBuffer(rhi, { label: 'particles-camera', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    /** A 2D view's world-to-clip matrix, and its bind group. */
    this._flatBuffer = createBuffer(rhi, { label: 'particles-2d', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._flatGroup = rhi.device.createBindGroup({
      label: 'particles-2d', layout: this.flatLayout, entries: [{ binding: 0, resource: { buffer: this._flatBuffer } }],
    });
    /** This frame's emitters by entity, for a 2D view drawing them one at a time. */
    this._byEntity = new Map();
    this._frameGroups = new WeakMap();
    this._imageGroups = new WeakMap();
    this._list = [];
    this._frame = 0;
    this._environment = null;
    this._simulate = (pass) => this._encodeSimulate(pass);
    this._sortPass = (pass) => this._encodeSort(pass);
    this._draw = (pass) => this._encodeDraw(pass);
    // The sort's buckets, shared: alpha emitters sort one after another, and
    // each scan leaves the counts zeroed for the next.
    this._counts = createBuffer(rhi, { label: 'particles-counts', size: BUCKETS * 4, usage: GPUBufferUsage.STORAGE });
    this._cursors = createBuffer(rhi, { label: 'particles-cursors', size: BUCKETS * 4, usage: GPUBufferUsage.STORAGE });
    this._view = createBuffer(rhi, { label: 'particles-view', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._viewData = new Float32Array(8);
    /** Each alpha emitter's indirect draw, a stride apart: bound by dynamic offset. */
    this.argsStride = Math.max(16, rhi.limits.minStorageBufferOffsetAlignment);
    this._args = null;
    this._argsCapacity = 0;
    this._argsStaging = new Uint32Array(0);
    /** Alpha emitters this frame, each sorted. */
    this.sorted = 0;
    /** Emitters this frame, and whether any moves anything. */
    this.count = 0;
  }

  /**
   * Settle what the scene's emitters owe -- births and time -- into this
   * frame's uniforms, growing rings that need to. Returns the emitter count.
   */
  prepare(scene, camera, environment, frozen = false) {
    this.count = scene.emitters.size;
    if (this.count === 0) {
      if (this.rings.size > 0) this.rings.clear();
      // And what named the last scene's emitters, which held it.
      this._list.length = 0;
      this._byEntity.clear();
      return 0;
    }
    const world = scene.transforms.world;
    const list = this._list;
    list.length = 0;
    let relayout = this.rings.size !== scene.emitters.size;
    for (const [entity, record] of scene.emitters) {
      // Frozen -- a reflection probe's capture -- draws them where they are
      // and leaves what they owe for the next real frame.
      const births = frozen ? 0 : Math.floor(record.owed);
      const dt = frozen ? 0 : record.time;
      record.owed -= births;
      record.time -= dt;
      let ring = this.rings.get(record.seed);
      // Particles from earlier bursts that may still be alive hold their
      // slots too. Sized by the rate alone, a burst-only emitter's ring held
      // one burst, and the next overwrote particles with seconds to live.
      const clock = (ring?.clock ?? 0) + dt;
      const held = ring?.held ?? [];
      let holding = 0;
      for (let k = held.length - 1; k >= 0; k--) {
        if (held[k].until <= clock) held.splice(k, 1);
        else holding += held[k].count;
      }
      const bursting = Math.min(record.burstOwed, births);
      record.burstOwed -= bursting;
      const needed = ringCapacity(record, births) + holding;
      if (ring === undefined) {
        // placed: how many of its slots the current pool holds. None yet.
        ring = { offset: 0, placed: 0, capacity: needed, head: 0, clock: 0, held };
        this.rings.set(record.seed, ring);
        relayout = true;
      } else if (ring.capacity < needed) {
        // Grown at the end: every live particle keeps its slot, and births
        // go on into the new ones. The head stayed where it was, which after
        // a wrap is the oldest particle still alive, and the next birth took it.
        ring.head = ring.capacity;
        ring.capacity = grownCapacity(ring.capacity, needed);
        relayout = true;
      }
      ring.clock = clock;
      if (bursting > 0) held.push({ count: bursting, until: clock + record.lifetime[1] });
      list.push({ entity, record, ring, births: Math.min(births, ring.capacity), dt });
    }
    const seeds = this._seeds;
    seeds.clear();
    for (const item of list) seeds.add(item.record.seed);
    for (const seed of this.rings.keys()) if (!seeds.has(seed)) { this.rings.delete(seed); relayout = true; }
    if (relayout) this._layout(list);

    for (const item of list) item.m = handleIndex(item.entity) * 16;
    if (camera.is2D !== true) {
      // Alpha emitters draw far to near after one another; additive after them
      // all. A 2D view orders them by layer instead, with its sprites.
      const eye = camera.position;
      const f = [-camera.view[2], -camera.view[6], -camera.view[10]];
      for (const item of list) {
        const m = item.m;
        item.depth = (world[m + 12] - eye[0]) * f[0] + (world[m + 13] - eye[1]) * f[1] + (world[m + 14] - eye[2]) * f[2];
      }
      list.sort((a, b) => (a.record.blend === 'additive') - (b.record.blend === 'additive') || b.depth - a.depth);
    }
    this._byEntity.clear();
    for (const item of list) this._byEntity.set(item.entity, item);

    const bytes = list.length * this.stride;
    if (bytes > this._uniformCapacity) {
      this._uniform?.destroy();
      this._uniformCapacity = grownCapacity(this._uniformCapacity, bytes);
      this._uniform = createBuffer(this.rhi, {
        label: 'particles-emitters', size: this._uniformCapacity, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._staging = new Uint8Array(this._uniformCapacity);
      this._groups = null;
    }
    this._frame++;
    list.forEach((item, k) => {
      const seed = (item.record.seed ^ Math.imul(this._frame, 0x85ebca6b)) >>> 0;
      packEmitter(this._staging, k * this.stride, item.record, world, item.m, item.ring, item.births, item.dt, seed, camera.is2D === true);
      item.slot = k;
      item.ring.head = (item.ring.head + item.births) % item.ring.capacity;
    });
    this.rhi.queue.writeBuffer(this._uniform, 0, this._staging, 0, bytes);
    // Each alpha emitter in 3D is sorted: its draw's count starts at zero, for
    // the sort's key pass to count its live particles into.
    this.sorted = 0;
    if (camera.is2D !== true) {
      for (const item of list) item.sortIndex = item.record.blend === 'alpha' ? this.sorted++ : -1;
      if (this.sorted > 0) {
        const words = this.sorted * this.argsStride / 4;
        if (words * 4 > this._argsCapacity) {
          this._args?.destroy();
          this._argsCapacity = grownCapacity(this._argsCapacity, words * 4);
          this._args = createBuffer(this.rhi, {
            label: 'particles-args', size: this._argsCapacity, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
          });
          this._argsStaging = new Uint32Array(this._argsCapacity / 4);
          this._sortGroup = null;
        }
        for (let k = 0; k < this.sorted; k++) this._argsStaging[k * this.argsStride / 4] = 6;
        this.rhi.queue.writeBuffer(this._args, 0, this._argsStaging, 0, words);
        const v = this._viewData;
        v.set(camera.position, 0);
        v[4] = -camera.view[2];
        v[5] = -camera.view[6];
        v[6] = -camera.view[10];
        this.rhi.queue.writeBuffer(this._view, 0, v);
      }
    }
    if (camera.is2D === true) {
      this.rhi.queue.writeBuffer(this._flatBuffer, 0, camera.viewProjection);
    } else {
      const v = camera.view;
      this._camera.set([v[0], v[4], v[8], 0, v[1], v[5], v[9], 0]);
      this.rhi.queue.writeBuffer(this._cameraBuffer, 0, this._camera);
    }
    this._environment = environment;
    return this.count;
  }

  /**
   * Place every ring in a new pool, back to back, and carry each surviving
   * ring's particles across. A new pool starts zeroed, which is every slot
   * dead: lifetime 0.
   */
  _layout(list) {
    let size = 0;
    const offsets = list.map(({ ring }) => {
      const offset = size;
      size += ring.capacity;
      return offset;
    });
    const pool = createBuffer(this.rhi, {
      label: 'particles', size: Math.max(size, 1) * PARTICLE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    if (this.pool !== null) {
      const encoder = this.rhi.device.createCommandEncoder({ label: 'particles-relayout' });
      list.forEach(({ ring }, k) => {
        if (ring.placed === 0) return;
        encoder.copyBufferToBuffer(this.pool, ring.offset * PARTICLE_BYTES, pool, offsets[k] * PARTICLE_BYTES,
          ring.placed * PARTICLE_BYTES);
      });
      this.rhi.queue.submit([encoder.finish()]);
      this.pool.destroy();
    }
    list.forEach(({ ring }, k) => {
      ring.offset = offsets[k];
      ring.placed = ring.capacity;
    });
    this.pool = pool;
    this.poolSize = size;
    // A slot of each for every slot of the pool: an alpha emitter sorts in its ring's stretch.
    for (const name of ['_keys', '_visible', '_order']) {
      this[name]?.destroy();
      this[name] = createBuffer(this.rhi, { label: `particles${name}`, size: Math.max(size, 1) * 4, usage: GPUBufferUsage.STORAGE });
    }
    this._groups = null;
    this._sortGroup = null;
  }

  /** The two passes: births and motion, then the draw that reads them. */
  addPasses(graph, { sceneColor, depth }) {
    const pool = this.addSimulation(graph);
    if (pool === null) return;
    const reads = [pool];
    if (this.sorted > 0) {
      const order = graph.importBuffer('particles-order', this._order);
      graph.addPass({ name: 'particles:sort', type: 'compute', reads: [pool], writes: [order], execute: this._sortPass });
      reads.push(order);
    }
    graph.addPass({
      name: 'particles', reads, color: [{ resource: sceneColor }], depth: { resource: depth }, execute: this._draw,
    });
  }

  /**
   * Births and motion alone, for a 2D view that draws the particles itself
   * (draw2D). Returns the pool, for its pass to read; null with no emitters.
   */
  addSimulation(graph) {
    if (this.count === 0) return null;
    const pool = graph.importBuffer('particles', this.pool);
    graph.addPass({ name: 'particles:simulate', type: 'compute', writes: [pool], execute: this._simulate });
    return pool;
  }

  /**
   * One emitter's particles, inside a 2D view's pass (render/view2d.js), in
   * its place in the painter's order. Sets its own pipeline and every bind
   * group it uses, so the view binds its own again after.
   */
  draw2D(pass, entity) {
    const item = this._byEntity.get(entity);
    if (item === undefined) return;
    pass.setPipeline(this.pipelines.get(this.descriptors[`${item.record.blend}2D`]));
    pass.setBindGroup(0, this._flatGroup);
    pass.setBindGroup(1, this._bindGroups().emitter, [item.slot * this.stride]);
    pass.setBindGroup(2, this._imageGroup(item.record.texture));
    pass.draw(6, item.ring.capacity, 0, item.ring.offset);
  }

  _bindGroups() {
    if (this._groups) return this._groups;
    const device = this.rhi.device;
    this._groups = {
      simulate: device.createBindGroup({
        label: 'particles-simulate',
        layout: this.simulateLayout,
        entries: [
          { binding: 0, resource: { buffer: this._uniform, size: EMITTER_BYTES } },
          { binding: 1, resource: { buffer: this.pool } },
        ],
      }),
      emitter: device.createBindGroup({
        label: 'particles-emitter',
        layout: this.emitterLayout,
        entries: [
          { binding: 0, resource: { buffer: this._uniform, size: EMITTER_BYTES } },
          { binding: 1, resource: { buffer: this.pool } },
          { binding: 2, resource: { buffer: this._order } },
        ],
      }),
    };
    return this._groups;
  }

  _encodeSimulate(pass) {
    const { simulate } = this._bindGroups();
    pass.setPipeline(this.simulate);
    for (const item of this._list) {
      if (item.dt === 0 && item.births === 0) continue;
      pass.setBindGroup(0, simulate, [item.slot * this.stride]);
      pass.dispatchWorkgroups(Math.ceil(item.ring.capacity / WORKGROUP));
    }
  }

  _encodeSort(pass) {
    this._sortGroup ??= this.rhi.device.createBindGroup({
      label: 'particles-sort',
      layout: this.sortLayout,
      entries: [
        { binding: 0, resource: { buffer: this._uniform, size: EMITTER_BYTES } },
        { binding: 1, resource: { buffer: this.pool } },
        { binding: 2, resource: { buffer: this._keys } },
        { binding: 3, resource: { buffer: this._visible } },
        { binding: 4, resource: { buffer: this._counts } },
        { binding: 5, resource: { buffer: this._cursors } },
        { binding: 6, resource: { buffer: this._order } },
        { binding: 7, resource: { buffer: this._args, size: 16 } },
        { binding: 8, resource: { buffer: this._view } },
      ],
    });
    for (const item of this._list) {
      if (item.sortIndex < 0) continue;
      const groups = Math.ceil(item.ring.capacity / WORKGROUP);
      pass.setBindGroup(0, this._sortGroup, [item.slot * this.stride, item.sortIndex * this.argsStride]);
      pass.setPipeline(this.sort.keys);
      pass.dispatchWorkgroups(groups);
      pass.setPipeline(this.sort.scan);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.sort.place);
      pass.dispatchWorkgroups(groups);
    }
  }

  _encodeDraw(pass) {
    const { emitter } = this._bindGroups();
    const environment = this._environment;
    let frameGroup = this._frameGroups.get(environment);
    if (!frameGroup) {
      frameGroup = this.rhi.device.createBindGroup({
        label: 'particles-frame',
        layout: this.frameLayout,
        entries: [
          { binding: 0, resource: { buffer: this._frameBuffer } },
          { binding: 1, resource: { buffer: this._cameraBuffer } },
          { binding: 2, resource: environment.irradianceView },
          { binding: 3, resource: environment.sampler },
        ],
      });
      this._frameGroups.set(environment, frameGroup);
    }
    pass.setBindGroup(0, frameGroup);
    let bound = null;
    for (const item of this._list) {
      if (item.record.blend !== bound) {
        pass.setPipeline(this.pipelines.get(this.descriptors[item.record.blend]));
        bound = item.record.blend;
      }
      pass.setBindGroup(1, emitter, [item.slot * this.stride]);
      pass.setBindGroup(2, this._imageGroup(item.record.texture));
      if (item.sortIndex >= 0) pass.drawIndirect(this._args, item.sortIndex * this.argsStride);
      else pass.draw(6, item.ring.capacity, 0, item.ring.offset);
    }
  }

  /** A texture's bind group; without one, a white texel the shader ignores. */
  _imageGroup(texture) {
    const key = texture ?? this;
    let group = this._imageGroups.get(key);
    if (!group) {
      group = this.rhi.device.createBindGroup({
        label: 'particles-image',
        layout: this.imageLayout,
        entries: [
          { binding: 0, resource: texture?.view ?? defaultTextures(this.rhi).white.createView() },
          { binding: 1, resource: clampSampler(this.rhi) },
        ],
      });
      this._imageGroups.set(key, group);
    }
    return group;
  }

  destroy() {
    this.pool?.destroy();
    this._uniform?.destroy();
    for (const buffer of [this._keys, this._visible, this._order, this._args]) buffer?.destroy();
    this._counts.destroy();
    this._cursors.destroy();
    this._view.destroy();
    this._cameraBuffer.destroy();
    this._flatBuffer.destroy();
  }
}
