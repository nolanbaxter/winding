// GPU buffer creation.

export function createBuffer(rhi, { label, size, usage, data }) {
  if (data) {
    // mappedAtCreation writes straight into the buffer's own memory with no
    // staging copy and no queue round-trip. Only available at creation, which
    // is why static geometry is uploaded this way and dynamic data is not.
    const buffer = rhi.device.createBuffer({
      label,
      size: size ?? align4(data.byteLength),
      usage,
      mappedAtCreation: true,
    });
    const dst = new Uint8Array(buffer.getMappedRange());
    dst.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
    return buffer;
  }

  return rhi.device.createBuffer({ label, size, usage });
}

// GPU buffer sizes must be a multiple of 4.
function align4(n) {
  return (n + 3) & ~3;
}
