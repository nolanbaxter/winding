// GPU buffer creation.

export function createBuffer(rhi, { label, size, usage, data }) {
  const bytes = size ?? align4(data.byteLength);
  // WebGPU does not throw for this. It returns an invalid buffer -- and with
  // mappedAtCreation still hands back memory to write into -- so the upload
  // looks fine and the mesh simply never draws.
  const max = rhi.limits?.maxBufferSize ?? Infinity;
  if (bytes > max) {
    throw new RangeError(`createBuffer: ${label ?? 'buffer'} is ${bytes} bytes, past this device's ${max}`);
  }

  if (data) {
    // mappedAtCreation writes straight into the buffer's own memory with no
    // staging copy and no queue round-trip. Only available at creation, which
    // is why static geometry is uploaded this way and dynamic data is not.
    const buffer = rhi.device.createBuffer({ label, size: bytes, usage, mappedAtCreation: true });
    const dst = new Uint8Array(buffer.getMappedRange());
    dst.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
    return buffer;
  }

  return rhi.device.createBuffer({ label, size: bytes, usage });
}

/**
 * How many `bytesPerItem` items one storage buffer can hold and still be bound
 * whole on this device: the smaller of the binding limit and the buffer
 * limit. The ceiling for every array the shaders read as one binding.
 */
export function storageCapacity(rhi, bytesPerItem) {
  const limits = rhi.limits;
  if (!limits) return Infinity;
  return Math.floor(Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) / bytesPerItem);
}

// GPU buffer sizes must be a multiple of 4. Math rather than `& ~3`, which
// works in signed 32 bits and turns a 2 GiB size negative.
function align4(n) {
  return Math.ceil(n / 4) * 4;
}
