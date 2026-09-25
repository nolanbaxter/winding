// Debug lines: segments, boxes, spheres and axes, drawn for one frame.
//
// Immediate mode. Call them every frame you want them seen; each frame's are
// drawn and then forgotten, so nothing has to be removed and nothing goes
// stale when what it marked moves:
//
//   engine.debug.box(min, max, [1, 1, 0]);
//   engine.debug.axes(position, 0.5);
//
// Drawn last, onto the screen itself, after tone mapping: a colour asked for
// is the colour shown, with no exposure, bloom or antialiasing in between.
// Colours are linear, as everywhere else in the engine. The scene's depth
// hides them where geometry is in front, unless `depthTest` is off.
//
// One pixel wide, which is the only width WebGPU draws lines at.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { DEPTH_FORMAT } from '../rhi/device.js';
import { grownCapacity } from '../core/grow.js';

/** A vertex: position (3 floats) and colour (4 bytes). */
const VERTEX_BYTES = 16;

/**
 * Segments per circle of a sphere. The chord drifts from the true circle by
 * r (1 - cos(pi / 32)), under half a percent of the radius.
 */
const CIRCLE_SEGMENTS = 32;

const WHITE = [1, 1, 1];

const SHADER = /* wgsl */ `
@group(0) @binding(0) var<uniform> viewProjection : mat4x4<f32>;

struct Out {
  @builtin(position) clip  : vec4<f32>,
  @location(0)       color : vec4<f32>,
};

@vertex
fn vs(@location(0) position : vec3<f32>, @location(1) color : vec4<f32>) -> Out {
  return Out(viewProjection * vec4<f32>(position, 1.0), color);
}

@fragment
fn fs(v : Out) -> @location(0) vec4<f32> {
  return v.color;
}
`;

export class DebugLines {
  static async create(rhi, pipelines) {
    const shader = await compileShader(rhi.device, SHADER, 'debug-lines.wgsl');
    const layout = rhi.device.createBindGroupLayout({
      label: 'debug-lines',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    });
    const descriptor = (depthTest) => ({
      label: depthTest ? 'debug-lines' : 'debug-lines:on-top',
      layout: createPipelineLayout(rhi.device, { 0: layout }, 'debug-lines'),
      shader,
      buffers: [{
        arrayStride: VERTEX_BYTES,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'unorm8x4' },
        ],
      }],
      targets: [{ format: rhi.viewFormat }],
      primitive: { topology: 'line-list' },
      // Equal passes too, so an edge drawn along a surface shows on it.
      depth: { format: DEPTH_FORMAT, depthCompare: depthTest ? 'greater-equal' : 'always', depthWriteEnabled: false },
    });
    const descriptors = { tested: descriptor(true), onTop: descriptor(false) };
    await pipelines.warm([descriptors.tested, descriptors.onTop]);
    return new DebugLines(rhi, pipelines, descriptors, layout);
  }

  constructor(rhi, pipelines, descriptors, layout) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    this._descriptors = descriptors;
    /** Whether the scene's depth hides lines behind geometry. A plain field. */
    this.depthTest = true;
    /** Vertices this frame: two a segment. */
    this.count = 0;
    this._floats = new Float32Array(256 * 4);
    this._bytes = new Uint8Array(this._floats.buffer);
    this._vertexBuffer = null;
    this._capacity = 0;
    this._uniform = createBuffer(rhi, {
      label: 'debug-lines', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._bindGroup = rhi.device.createBindGroup({
      label: 'debug-lines', layout, entries: [{ binding: 0, resource: { buffer: this._uniform } }],
    });
    this._execute = (pass) => {
      pass.setPipeline(this.pipelines.get(this.depthTest ? this._descriptors.tested : this._descriptors.onTop));
      pass.setBindGroup(0, this._bindGroup);
      pass.setVertexBuffer(0, this._vertexBuffer);
      pass.draw(this.count);
    };
  }

  /** A segment. Returns this, so calls chain. */
  line(from, to, color = WHITE) {
    this._vertex(from[0], from[1], from[2], color);
    this._vertex(to[0], to[1], to[2], color);
    return this;
  }

  /** An axis-aligned box's twelve edges. */
  box(min, max, color = WHITE) {
    const x = [min[0], max[0]];
    const y = [min[1], max[1]];
    const z = [min[2], max[2]];
    for (let a = 0; a < 2; a++) {
      for (let b = 0; b < 2; b++) {
        this.line([x[0], y[a], z[b]], [x[1], y[a], z[b]], color);
        this.line([x[a], y[0], z[b]], [x[a], y[1], z[b]], color);
        this.line([x[a], y[b], z[0]], [x[a], y[b], z[1]], color);
      }
    }
    return this;
  }

  /** Three great circles, one around each axis. */
  sphere(center, radius, color = WHITE) {
    const [cx, cy, cz] = center;
    for (let axis = 0; axis < 3; axis++) {
      const at = (a, b) => (axis === 0 ? [cx, cy + a, cz + b] : axis === 1 ? [cx + a, cy, cz + b] : [cx + a, cy + b, cz]);
      let [pu, pv] = [radius, 0];
      for (let i = 1; i <= CIRCLE_SEGMENTS; i++) {
        const angle = (i / CIRCLE_SEGMENTS) * 2 * Math.PI;
        const u = radius * Math.cos(angle);
        const v = radius * Math.sin(angle);
        this.line(at(pu, pv), at(u, v), color);
        [pu, pv] = [u, v];
      }
    }
    return this;
  }

  /** The world axes at a point: x red, y green, z blue, each `size` long. */
  axes(origin, size = 1) {
    const [x, y, z] = origin;
    this.line(origin, [x + size, y, z], [1, 0, 0]);
    this.line(origin, [x, y + size, z], [0, 1, 0]);
    this.line(origin, [x, y, z + size], [0, 0, 1]);
    return this;
  }

  /** Forget this frame's lines. The renderer calls it after each frame. */
  clear() {
    this.count = 0;
  }

  _vertex(x, y, z, color) {
    if ((this.count + 1) * 4 > this._floats.length) {
      const floats = new Float32Array(grownCapacity(this._floats.length / 4, this.count + 1) * 4);
      floats.set(this._floats);
      this._floats = floats;
      this._bytes = new Uint8Array(floats.buffer);
    }
    const f = this.count * 4;
    this._floats[f] = x;
    this._floats[f + 1] = y;
    this._floats[f + 2] = z;
    const b = f * 4 + 12;
    for (let c = 0; c < 3; c++) this._bytes[b + c] = Math.round(Math.min(Math.max(color[c], 0), 1) * 255);
    this._bytes[b + 3] = 255;
    this.count++;
  }

  /**
   * Upload this frame's lines and add the pass that draws them onto the
   * finished picture. Nothing, when there are none.
   */
  addPass(graph, { surface, depth, viewProjection }) {
    if (this.count === 0) return;
    const bytes = this.count * VERTEX_BYTES;
    if (bytes > this._capacity) {
      this._vertexBuffer?.destroy();
      this._capacity = grownCapacity(this._capacity, bytes);
      this._vertexBuffer = createBuffer(this.rhi, {
        label: 'debug-lines', size: this._capacity, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    this.rhi.queue.writeBuffer(this._vertexBuffer, 0, this._bytes, 0, bytes);
    this.rhi.queue.writeBuffer(this._uniform, 0, viewProjection);
    graph.addPass({
      name: 'debug-lines',
      color: [{ resource: surface }],
      depth: { resource: depth },
      execute: this._execute,
    });
  }

  destroy() {
    this._vertexBuffer?.destroy();
    this._uniform.destroy();
  }
}
