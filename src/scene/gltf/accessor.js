// Accessor decoding -- the layer where glTF's flexibility actually lives.
//
// An accessor says "read `count` elements of `type`, made of `componentType`,
// starting at this offset into this bufferView". Four things make that harder
// than it sounds, and all four appear in real exports:
//
//   1. byteStride    attributes can be interleaved, so elements are not adjacent
//   2. normalized    integers that stand in for floats in [0,1] or [-1,1]
//   3. sparse        a base array plus a list of index/value overrides
//   4. alignment     a DataView path is needed when a typed-array view would
//                    not be legally aligned
//
// Everything here returns Float32Array (attributes) or Uint32Array (indices),
// so nothing downstream ever branches on a source format again.

const COMPONENTS = {
  5120: { name: 'BYTE', bytes: 1, get: 'getInt8', scale: 1 / 127, signed: true },
  5121: { name: 'UNSIGNED_BYTE', bytes: 1, get: 'getUint8', scale: 1 / 255, signed: false },
  5122: { name: 'SHORT', bytes: 2, get: 'getInt16', scale: 1 / 32767, signed: true },
  5123: { name: 'UNSIGNED_SHORT', bytes: 2, get: 'getUint16', scale: 1 / 65535, signed: false },
  5125: { name: 'UNSIGNED_INT', bytes: 4, get: 'getUint32', scale: 1 / 4294967295, signed: false },
  5126: { name: 'FLOAT', bytes: 4, get: 'getFloat32', scale: 1, signed: true },
};

const TYPE_COMPONENT_COUNT = {
  SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16,
};

export function componentCountOf(type) {
  const n = TYPE_COMPONENT_COUNT[type];
  if (n === undefined) throw new Error(`glTF: unknown accessor type "${type}"`);
  return n;
}

/**
 * Read an accessor as float data, dequantizing normalized integers.
 *
 * Returns a Float32Array of accessor.count * componentCount, tightly packed
 * whatever the source stride was.
 */
export function readAccessorAsFloat32(json, buffers, accessorIndex) {
  const accessor = accessorAt(json, accessorIndex);
  const comp = componentInfo(accessor.componentType);
  const perElement = componentCountOf(accessor.type);
  const out = new Float32Array(accessor.count * perElement);

  readInto(out, json, buffers, accessor, comp, perElement, accessor.normalized === true);

  if (accessor.sparse) {
    applySparse(out, json, buffers, accessor, perElement, accessor.normalized === true);
  }
  return out;
}

/**
 * Read an accessor as unsigned integers, without dequantizing.
 *
 * `expectedType` defaults to SCALAR, which is what indices are. Joint indices
 * are VEC4 and must come through here rather than through the float reader:
 * they ADDRESS a palette, so a normalized read would turn joint 3 of 4 into
 * 0.75 and a float read of a u16 would be fine until a skeleton passed 2^24
 * joints. Neither is a number this can afford to be approximately right about.
 */
export function readAccessorAsUint32(json, buffers, accessorIndex, expectedType = 'SCALAR') {
  const accessor = accessorAt(json, accessorIndex);
  const comp = componentInfo(accessor.componentType);

  if (accessor.type !== expectedType) {
    throw new Error(
      `glTF: accessor ${accessorIndex} must be ${expectedType}, got ${accessor.type}`,
    );
  }
  if (comp.signed && comp.name !== 'FLOAT') {
    throw new Error(`glTF: accessor ${accessorIndex} uses signed ${comp.name} where an unsigned integer is required`);
  }
  // FLOAT fell through the test above, because FLOAT is signed and the clause
  // excluding it was written to let the float READER share this guard. Here it
  // is not survivable: the fast path views the buffer as Float32Array and
  // copies into a Uint32Array, which truncates every value toward zero. An
  // index of 2.0 reads as 2 and nothing looks wrong until one is 65535.9.
  // The spec allows neither -- indices are unsigned byte, short or int, and
  // joints are unsigned byte or short -- so this is a malformed file.
  if (comp.name === 'FLOAT') {
    throw new Error(
      `glTF: accessor ${accessorIndex} stores FLOAT where an unsigned integer is required`,
    );
  }

  const perElement = componentCountOf(expectedType);
  const out = new Uint32Array(accessor.count * perElement);
  readInto(out, json, buffers, accessor, comp, perElement, false);

  if (accessor.sparse) applySparse(out, json, buffers, accessor, perElement, false);
  return out;
}

function accessorAt(json, index) {
  const accessor = json.accessors?.[index];
  if (!accessor) throw new Error(`glTF: accessor ${index} does not exist`);
  return accessor;
}

function componentInfo(componentType) {
  const comp = COMPONENTS[componentType];
  if (!comp) throw new Error(`glTF: unknown componentType ${componentType}`);
  return comp;
}

/**
 * Core read. `out` is already the right length; this fills it.
 *
 * An accessor with no bufferView is legal and means "all zeros" -- that form
 * exists precisely so a sparse accessor can describe a mostly-empty array.
 */
function readInto(out, json, buffers, accessor, comp, perElement, normalized) {
  if (accessor.bufferView === undefined) return out;   // zeros, possibly + sparse

  const view = json.bufferViews?.[accessor.bufferView];
  if (!view) throw new Error(`glTF: bufferView ${accessor.bufferView} does not exist`);

  const buffer = buffers[view.buffer];
  if (!buffer) throw new Error(`glTF: buffer ${view.buffer} was not resolved`);

  // MAT2 and MAT3 pad each COLUMN to four bytes when the component is
  // smaller than that, so their elements are not `bytes * count` long and the
  // arithmetic below would read every matrix after the first from the wrong
  // place. Nothing in this engine can reach it -- the only matrices read are
  // inverse binds, which are MAT4, where the rule does not apply -- so this
  // refuses rather than implementing a layout no asset here uses. A file that
  // needs it gets a message naming exactly what is missing.
  if ((accessor.type === 'MAT2' || accessor.type === 'MAT3') && comp.bytes < 4) {
    throw new Error(
      `glTF: ${accessor.type} of ${comp.name} needs column padding, which this reader does not do`,
    );
  }

  const elementBytes = comp.bytes * perElement;
  const stride = view.byteStride ?? elementBytes;
  const start = buffer.byteOffset + (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const needed = stride * (accessor.count - 1) + elementBytes;

  // accessor.byteOffset is measured from the START of the bufferView, so the
  // view's own byteOffset (which is measured from the buffer) must not appear
  // on this side of the comparison.
  if ((accessor.byteOffset ?? 0) + needed > view.byteLength) {
    throw new Error(
      `glTF: accessor reads ${(accessor.byteOffset ?? 0) + needed} bytes from a ` +
      `${view.byteLength}-byte bufferView`,
    );
  }

  // Fast path: tightly packed AND legally aligned for a typed-array view, so
  // the whole run copies in one go instead of element by element.
  const packed = stride === elementBytes;
  const aligned = start % comp.bytes === 0;
  if (packed && aligned && !normalized) {
    const Source = typedArrayFor(comp);
    const src = new Source(buffer.buffer, start, accessor.count * perElement);
    out.set(src);
    return out;
  }

  // General path. A DataView reads at any alignment and any stride, which is
  // why it is the fallback rather than an error.
  const data = new DataView(buffer.buffer);
  const getter = comp.get;
  const scale = normalized ? comp.scale : 1;

  for (let e = 0; e < accessor.count; e++) {
    const base = start + e * stride;
    for (let c = 0; c < perElement; c++) {
      const raw = data[getter](base + c * comp.bytes, true);
      // Normalized signed types clamp at -1: the most negative value (-128,
      // -32768) would otherwise dequantize slightly past it.
      out[e * perElement + c] = normalized && comp.signed
        ? Math.max(raw * scale, -1)
        : raw * scale;
    }
  }
  return out;
}

/**
 * Sparse accessors: a base array with a list of overrides. Used by morph
 * targets and by exporters that compress mostly-identical arrays.
 */
function applySparse(out, json, buffers, accessor, perElement, normalized) {
  const { count, indices, values } = accessor.sparse;

  const indexComp = componentInfo(indices.componentType);
  const sparseIndices = new Uint32Array(count);
  readInto(
    sparseIndices, json, buffers,
    { bufferView: indices.bufferView, byteOffset: indices.byteOffset ?? 0, count },
    indexComp, 1, false,
  );

  const valueComp = componentInfo(accessor.componentType);
  // Match the destination's type: routing a sparse INDEX override through
  // Float32 would quietly lose precision above 2^24.
  const sparseValues = new out.constructor(count * perElement);
  readInto(
    sparseValues, json, buffers,
    { bufferView: values.bufferView, byteOffset: values.byteOffset ?? 0, count },
    valueComp, perElement, normalized,
  );

  for (let i = 0; i < count; i++) {
    const target = sparseIndices[i] * perElement;
    if (target + perElement > out.length) {
      throw new Error(`glTF: sparse index ${sparseIndices[i]} is outside the accessor`);
    }
    for (let c = 0; c < perElement; c++) out[target + c] = sparseValues[i * perElement + c];
  }
  return out;
}

function typedArrayFor(comp) {
  switch (comp.get) {
    case 'getInt8': return Int8Array;
    case 'getUint8': return Uint8Array;
    case 'getInt16': return Int16Array;
    case 'getUint16': return Uint16Array;
    case 'getUint32': return Uint32Array;
    default: return Float32Array;
  }
}
