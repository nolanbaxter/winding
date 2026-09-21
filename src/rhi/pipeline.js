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
    /** How many distinct pipelines exist. Watch this stop growing after load. */
    this.created = 0;
    /** Cache hits. If this is 0 in steady state, the key is unstable. */
    this.hits = 0;
  }

  get(desc) {
    const key = pipelineKey(desc);
    const existing = this.pipelines.get(key);
    if (existing) {
      this.hits++;
      return existing;
    }
    const pipeline = this.device.createRenderPipeline(gpuDescriptor(desc));
    this.pipelines.set(key, pipeline);
    this.created++;
    return pipeline;
  }

  /**
   * Compile off the main thread. Call during loading with every variant the
   * scene can use; by the first frame, get() is pure cache hits.
   */
  async warm(descs) {
    await Promise.all(descs.map(async (desc) => {
      const key = pipelineKey(desc);
      if (this.pipelines.has(key)) return;
      const pipeline = await this.device.createRenderPipelineAsync(gpuDescriptor(desc));
      // Another warm() may have landed first; keep one object per key so the
      // sort key's pipeline identity stays meaningful.
      if (!this.pipelines.has(key)) {
        this.pipelines.set(key, pipeline);
        this.created++;
      }
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
    fragment: (desc.targets?.length ?? 0) === 0 ? undefined : {
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
    ? `|D${d.format},${d.depthCompare},${d.depthWriteEnabled ? 1 : 0},${d.depthBias ?? 0}`
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
