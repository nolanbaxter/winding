// Pipeline cache.
//
// A GPURenderPipeline bakes shaders, vertex layout, primitive state, depth
// state and color targets into one immutable object. Creating one compiles
// shaders and validates driver state -- single-digit to hundreds of
// milliseconds. Creating pipelines mid-frame is the direct cause of the
// traversal stutter in modern PC games.
//
// So: never create the same pipeline twice, and create them before the frame
// that needs them.
//
//   const pipe = cache.get(desc);          // cached, or built now
//   await cache.warm([descA, descB]);      // built off-thread during loading
//
// The cache key is a string rather than a hash. A 64-bit hash would be
// FNV-1a here, but a Map already hashes string keys internally, and a real hash
// would only add a collision mode where two different pipelines silently share
// one object. Strings are shorter code AND strictly safer.
//
// A DESCRIPTOR IS NOT CHANGED AFTER ITS FIRST get(). The renderer asks once
// per draw, so each descriptor object remembers its pipeline and the key is
// built only the first time: building it every call cost Sponza 0.4 ms of CPU
// a frame once override constants lengthened it. A different pipeline is a
// different descriptor object.

import { DEPTH_COMPARE, DEPTH_FORMAT } from './device.js';

const DEFAULT_PRIMITIVE = {
  topology: 'triangle-list',
  // glTF winds front faces counter-clockwise, so back-face culling with 'ccw'
  // discards exactly the triangles pointing away from the camera.
  // If geometry renders inside-out, this pair is the first thing to check.
  cullMode: 'back',
  frontFace: 'ccw',
};

const DEFAULT_DEPTH = {
  format: DEPTH_FORMAT,
  depthCompare: DEPTH_COMPARE,   // 'greater' -- reverse-Z
  depthWriteEnabled: true,
};

export class PipelineCache {
  constructor(device) {
    this.device = device;
    this.pipelines = new Map();
    /** Descriptor object -> its pipeline, so a repeat get() skips the key. */
    this._byDescriptor = new WeakMap();
    /** Compiles in flight, by key, so a second warm() waits on the first. */
    this._compiling = new Map();
    /** How many distinct pipelines exist. Watch this stop growing after load. */
    this.created = 0;
    /** Cache hits. If this is 0 in steady state, the key is unstable. */
    this.hits = 0;
  }

  get(desc) {
    const known = this._byDescriptor.get(desc);
    if (known) {
      this.hits++;
      return known;
    }
    const key = pipelineKey(desc);
    let pipeline = this.pipelines.get(key);
    if (pipeline) {
      this.hits++;
    } else {
      pipeline = this.device.createRenderPipeline(gpuDescriptor(desc));
      this.pipelines.set(key, pipeline);
      this.created++;
    }
    this._byDescriptor.set(desc, pipeline);
    return pipeline;
  }

  /**
   * A compute pipeline, from the same kind of plain descriptor:
   * { label, layout, shader, entry, constants }.
   */
  compute(desc) {
    const constants = desc.constants
      ? Object.keys(desc.constants).sort().map((k) => `${k}=${desc.constants[k]}`).join(',')
      : '';
    const key = `compute|${desc.layout.id}|${desc.shader.id}|${desc.entry}|${constants}`;
    const existing = this.pipelines.get(key);
    if (existing) {
      this.hits++;
      return existing;
    }
    const pipeline = this.device.createComputePipeline({
      label: desc.label,
      layout: desc.layout.gpu,
      compute: { module: desc.shader.module, entryPoint: desc.entry, constants: desc.constants },
    });
    this.pipelines.set(key, pipeline);
    this.created++;
    return pipeline;
  }

  /**
   * Compile off the main thread. Call during loading with every variant the
   * scene can use; by the first frame, get() is pure cache hits.
   */
  async warm(descs) {
    // A compile already in flight is awaited, not skipped. Two loads at once
    // used to see each other's variants as handled and return before they
    // were built, and the first frame then compiled them synchronously --
    // the hitch warm() exists to prevent. A failed compile is not cached, so
    // the next warm() tries again.
    await Promise.all(descs.map((desc) => {
      const key = pipelineKey(desc);
      if (this.pipelines.has(key)) return undefined;
      let compiling = this._compiling.get(key);
      if (!compiling) {
        compiling = this.device.createRenderPipelineAsync(gpuDescriptor(desc))
          .then((pipeline) => {
            this.pipelines.set(key, pipeline);
            this.created++;
          })
          .finally(() => this._compiling.delete(key));
        this._compiling.set(key, compiling);
      }
      return compiling;
    }));
  }
}

/**
 * Translate our descriptor into WebGPU's, applying the engine's defaults.
 *
 * Deliberately close to the native shape: learning a second vocabulary for the
 * same concepts would be two ways to say one thing. This adds
 * defaults and caching, nothing else.
 */
function gpuDescriptor(desc) {
  const vertexShader = desc.vertexShader ?? desc.shader;
  const fragmentShader = desc.fragmentShader ?? desc.shader;
  const depth = desc.depth === null ? undefined : { ...DEFAULT_DEPTH, ...desc.depth };

  return {
    label: desc.label,
    layout: desc.layout.gpu,
    vertex: {
      module: vertexShader.module,
      entryPoint: desc.vertexEntry ?? 'vs',
      buffers: desc.buffers ?? [],
      constants: desc.constants,
    },
    // No colour targets means no fragment stage at all. A shadow pass writes
    // only depth, and giving it an empty fragment stage is a validation error
    // rather than a no-op -- the stage has to be absent, not empty.
    // A descriptor that NAMES a fragment entry keeps its stage with no
    // targets: that is a depth-only pass that discards, which is how an
    // alpha-tested caster gets a shadow the shape of its texture.
    fragment: (desc.targets?.length ?? 0) === 0 && desc.fragmentEntry === undefined ? undefined : {
      module: fragmentShader.module,
      entryPoint: desc.fragmentEntry ?? 'fs',
      targets: desc.targets,
      // Pipeline-overridable constants. These change the COMPILED code, so two
      // pipelines differing only here are genuinely different pipelines and
      // the key below has to say so.
      constants: desc.constants,
    },
    primitive: { ...DEFAULT_PRIMITIVE, ...desc.primitive },
    depthStencil: depth,
    multisample: desc.multisample,
  };
}

/**
 * Every field that makes two pipelines different must appear here, in a fixed
 * order. A field left out means two distinct pipelines collapse into one and
 * the second one silently renders with the first one's state -- so this is
 * written out explicitly rather than derived by JSON.stringify, whose key
 * order depends on how the caller happened to build the object.
 */
function pipelineKey(desc) {
  const vertexShader = desc.vertexShader ?? desc.shader;
  const fragmentShader = desc.fragmentShader ?? desc.shader;
  const p = { ...DEFAULT_PRIMITIVE, ...desc.primitive };
  const d = desc.depth === null ? null : { ...DEFAULT_DEPTH, ...desc.depth };

  let key = `L${desc.layout.id}`;
  key += `|VS${vertexShader.id}:${desc.vertexEntry ?? 'vs'}`;
  key += `|FS${fragmentShader.id}:${desc.fragmentEntry ?? 'fs'}`;
  key += `|P${p.topology},${p.cullMode},${p.frontFace},${p.stripIndexFormat ?? '-'}`;
  key += d
    ? `|D${d.format},${d.depthCompare},${d.depthWriteEnabled ? 1 : 0},${d.depthBias ?? 0},${d.depthBiasSlopeScale ?? 0}`
    : '|D-';
  key += `|M${desc.multisample?.count ?? 1}`;

  // Sorted, so two callers that supply the same constants in different object
  // order still land on one cache entry.
  key += '|C';
  if (desc.constants) {
    for (const name of Object.keys(desc.constants).sort()) {
      key += `[${name}=${desc.constants[name]}]`;
    }
  }

  key += '|B';
  for (const buffer of desc.buffers ?? []) {
    key += `[${buffer.arrayStride},${buffer.stepMode ?? 'vertex'}`;
    for (const attr of buffer.attributes) {
      key += `(${attr.format},${attr.offset},${attr.shaderLocation})`;
    }
    key += ']';
  }

  key += '|T';
  for (const target of desc.targets ?? []) {
    key += `[${target.format},${target.writeMask ?? 0xf}`;
    if (target.blend) {
      const { color, alpha } = target.blend;
      key += `,${color.operation ?? 'add'},${color.srcFactor ?? 'one'},${color.dstFactor ?? 'zero'}`;
      key += `,${alpha.operation ?? 'add'},${alpha.srcFactor ?? 'one'},${alpha.dstFactor ?? 'zero'}`;
    }
    key += ']';
  }

  return key;
}

const shared = new WeakMap();

/**
 * The cache for pipelines outside the renderer's -- a bake, a mip chain, the
 * culling and clustering passes -- one per device. Every pipeline in the
 * engine is then a plain descriptor through a PipelineCache.
 */
export function sharedPipelines(device) {
  let cache = shared.get(device);
  if (!cache) shared.set(device, (cache = new PipelineCache(device)));
  return cache;
}
