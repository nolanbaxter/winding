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
// Drawn after sprites, before glass: alpha emitters far to near, their
// particles in birth order, then additive ones, which need no order.

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
  direction    : vec4<f32>,     //  96  in the emitter's space; w = 1 if it has a texture
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
    // A direction in the cone, uniform over the cap of the sphere it cuts.
    let cosTheta = mix(1.0, cos(emitter.spread), random(&n));
    let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    let phi = TAU * random(&n);
    let axis = normalize(emitter.direction.xyz);
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(axis.x) > 0.9);
    let tangent = normalize(cross(helper, axis));
    let bitangent = cross(axis, tangent);
    let local = axis * cosTheta + (tangent * cos(phi) + bitangent * sin(phi)) * sinTheta;
    let heading = (emitter.spawnMatrix * vec4<f32>(local, 0.0)).xyz;
    // A point in the ball, uniform by volume.
    let z = 2.0 * random(&n) - 1.0;
    let around = TAU * random(&n);
    let ring = sqrt(max(1.0 - z * z, 0.0));
    let within = vec3<f32>(ring * cos(around), ring * sin(around), z) * emitter.radius * pow(random(&n), 1.0 / 3.0);
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
fn vs(@builtin(vertex_index) index : u32, @builtin(instance_index) slot : u32) -> Out {
  var out : Out;
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
  var colour = select(disc, sampled, emitter.direction.w > 0.5) * v.color;
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

/**
 * How many slots an emitter's ring needs: everything it can have alive at
 * once -- its rate times its longest lifetime, rounded up -- plus what it is
 * about to birth.
 */
export function ringCapacity(record, births) {
  return Math.max(1, Math.ceil(record.rate * record.lifetime[1]) + births);
}

/** Pack an emitter's uniform: see Emitter in the shader. */
export function packEmitter(out, o, record, world, m, ring, births, dt, seed) {
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
  f32[27] = record.texture ? 1 : 0;
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
    const [simulateShader, drawShader] = await Promise.all([
      compileShader(device, SIMULATE_SHADER, 'particles-simulate.wgsl'),
      compileShader(device, DRAW_SHADER, 'particles-draw.wgsl'),
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
      ],
    });
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
    await pipelines.warm(Object.values(descriptors));
    return new ParticleSystem(rhi, pipelines, { simulate, descriptors, simulateLayout, frameLayout, emitterLayout, imageLayout }, frameBuffer);
  }

  constructor(rhi, pipelines, built, frameBuffer) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    Object.assign(this, built);
    this._frameBuffer = frameBuffer;
    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    this.stride = Math.ceil(EMITTER_BYTES / this.alignment) * this.alignment;
    /** Each emitter's ring, by its entity: { offset, capacity, head }. */
    this.rings = new Map();
    this.pool = null;
    this.poolSize = 0;
    this._uniform = null;
    this._uniformCapacity = 0;
    this._staging = new Uint8Array(0);
    this._camera = new Float32Array(8);
    this._cameraBuffer = createBuffer(rhi, { label: 'particles-camera', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._frameGroups = new WeakMap();
    this._imageGroups = new WeakMap();
    this._list = [];
    this._frame = 0;
    this._environment = null;
    this._simulate = (pass) => this._encodeSimulate(pass);
    this._draw = (pass) => this._encodeDraw(pass);
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
      let ring = this.rings.get(entity);
      const needed = ringCapacity(record, births);
      if (ring === undefined) {
        // placed: how many of its slots the current pool holds. None yet.
        ring = { offset: 0, placed: 0, capacity: needed, head: 0 };
        this.rings.set(entity, ring);
        relayout = true;
      } else if (ring.capacity < needed) {
        // Grown at the end: the head and every live particle keep their slots.
        ring.capacity = grownCapacity(ring.capacity, needed);
        relayout = true;
      }
      list.push({ entity, record, ring, births: Math.min(births, ring.capacity), dt });
    }
    for (const entity of this.rings.keys()) if (!scene.emitters.has(entity)) { this.rings.delete(entity); relayout = true; }
    if (relayout) this._layout(list);

    // Alpha emitters draw far to near after one another; additive after them all.
    const eye = camera.position;
    const f = [-camera.view[2], -camera.view[6], -camera.view[10]];
    for (const item of list) {
      const m = handleIndex(item.entity) * 16;
      item.m = m;
      item.depth = (world[m + 12] - eye[0]) * f[0] + (world[m + 13] - eye[1]) * f[1] + (world[m + 14] - eye[2]) * f[2];
    }
    list.sort((a, b) => (a.record.blend === 'additive') - (b.record.blend === 'additive') || b.depth - a.depth);

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
      packEmitter(this._staging, k * this.stride, item.record, world, item.m, item.ring, item.births, item.dt, seed);
      item.slot = k;
      item.ring.head = (item.ring.head + item.births) % item.ring.capacity;
    });
    this.rhi.queue.writeBuffer(this._uniform, 0, this._staging, 0, bytes);
    const v = camera.view;
    this._camera.set([v[0], v[4], v[8], 0, v[1], v[5], v[9], 0]);
    this.rhi.queue.writeBuffer(this._cameraBuffer, 0, this._camera);
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
    this._groups = null;
  }

  /** The two passes: births and motion, then the draw that reads them. */
  addPasses(graph, { sceneColor, depth }) {
    if (this.count === 0) return;
    const pool = graph.importBuffer('particles', this.pool);
    graph.addPass({ name: 'particles:simulate', type: 'compute', writes: [pool], execute: this._simulate });
    graph.addPass({
      name: 'particles', reads: [pool], color: [{ resource: sceneColor }], depth: { resource: depth }, execute: this._draw,
    });
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
      pass.draw(6, item.ring.capacity, 0, item.ring.offset);
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
    this._cameraBuffer.destroy();
  }
}
