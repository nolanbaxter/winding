// The geometry arena: every primitive's vertices in one buffer, every
// primitive's indices in another.
//
// A buffer per primitive meant every batch rebound its own vertex and index
// buffers before drawing, a command apiece, for a hundred batches a pass. In
// one arena a pass binds both once, and a primitive is where it starts:
// `baseVertex` and `firstIndex`, which every draw call already takes. It is
// also what lets one draw reach the geometry of many primitives.
//
// The buffers are STORAGE as well as VERTEX and INDEX, for shaders that read
// them directly, and COPY_SRC, so growing can carry what is there across on
// the GPU.

import { grownCapacity, RangeAllocator } from '../core/grow.js';
import { storageCapacity, createBuffer } from '../rhi/buffer.js';
import { VERTEX_STRIDE_BYTES } from './vertex.js';

const INDEX_BYTES = 4;

export class GeometryArena {
  constructor(rhi, { vertexCapacity = 1 << 16, indexCapacity = 1 << 18 } = {}) {
    this.rhi = rhi;
    this.vertexCapacity = vertexCapacity;
    this.indexCapacity = indexCapacity;
    this.vertexBuffer = this._create('geometry-vertices', vertexCapacity * VERTEX_STRIDE_BYTES,
      GPUBufferUsage.VERTEX);
    this.indexBuffer = this._create('geometry-indices', indexCapacity * INDEX_BYTES, GPUBufferUsage.INDEX);
    this._vertices = new RangeAllocator();
    this._indices = new RangeAllocator();
    /** Bumped when either buffer is replaced, so bind groups naming them rebuild. */
    this.revision = 0;
  }

  _create(label, size, usage) {
    return createBuffer(this.rhi, {
      label,
      size,
      usage: usage | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  /**
   * One primitive's interleaved vertices (VERTEX_STRIDE_BYTES each) and its
   * indices, relative to its own first vertex. Returns where they landed.
   * Called at load, never during a frame.
   */
  allocate(vertices, indices) {
    const vertexCount = vertices.byteLength / VERTEX_STRIDE_BYTES;
    const baseVertex = this._place('vertex', vertexCount, VERTEX_STRIDE_BYTES);
    const firstIndex = this._place('index', indices.length, INDEX_BYTES);
    const queue = this.rhi.queue;
    queue.writeBuffer(this.vertexBuffer, baseVertex * VERTEX_STRIDE_BYTES, vertices.buffer ?? vertices, vertices.byteOffset ?? 0, vertices.byteLength);
    queue.writeBuffer(this.indexBuffer, firstIndex * INDEX_BYTES, indices.buffer, indices.byteOffset, indices.byteLength);
    return { baseVertex, firstIndex };
  }

  /** Give back what allocate() returned, for `vertexCount` vertices and `indexCount` indices. */
  free(baseVertex, vertexCount, firstIndex, indexCount) {
    this._vertices.free(baseVertex, vertexCount);
    this._indices.free(firstIndex, indexCount);
  }

  /** Room for `count` more vertices or indices, growing the buffer if they pass its end. */
  _place(kind, count, bytes) {
    const ranges = kind === 'vertex' ? this._vertices : this._indices;
    const written = ranges.end;
    const base = ranges.alloc(count);
    const capacity = kind === 'vertex' ? this.vertexCapacity : this.indexCapacity;
    if (ranges.end > capacity) {
      const grown = grownCapacity(capacity, ranges.end, storageCapacity(this.rhi, bytes), `${kind}es`);
      const old = kind === 'vertex' ? this.vertexBuffer : this.indexBuffer;
      const buffer = this._create(old.label, grown * bytes, kind === 'vertex' ? GPUBufferUsage.VERTEX : GPUBufferUsage.INDEX);
      // Only what has been written: the rest of the old buffer was never set.
      if (written > 0) {
        const encoder = this.rhi.device.createCommandEncoder({ label: 'geometry-grow' });
        encoder.copyBufferToBuffer(old, 0, buffer, 0, written * bytes);
        this.rhi.queue.submit([encoder.finish()]);
      }
      old.destroy();
      if (kind === 'vertex') { this.vertexBuffer = buffer; this.vertexCapacity = grown; }
      else { this.indexBuffer = buffer; this.indexCapacity = grown; }
      this.revision++;
    }
    return base;
  }

  /** Both buffers, bound for a pass's draws. */
  bind(pass) {
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
  }

  destroy() {
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
  }
}
