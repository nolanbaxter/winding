// EXT_meshopt_compression and KHR_meshopt_compression.
//
// Both compress a bufferView, not an attribute: the view names a second byte
// range holding the compressed form, and decoding it gives back exactly the
// bytes the view describes. So decoding happens once, straight after the
// buffers resolve -- each compressed view is pointed at a fresh buffer of its
// decoded bytes -- and nothing downstream (accessors, sparse data, animation,
// skins) knows compression happened.
//
// Written from the two specifications' bitstream appendices, which define the
// format completely; there is no decoder library. KHR is EXT plus a second
// version of the attribute bitstream and a COLOR filter, and the rest is the
// same, so one decoder reads both, with the KHR-only parts refused under EXT.

export const MESHOPT_EXTENSIONS = ['EXT_meshopt_compression', 'KHR_meshopt_compression'];

/** Is this buffer a placeholder that only compressed views name? */
export function isFallbackBuffer(buffer) {
  return MESHOPT_EXTENSIONS.some((name) => buffer.extensions?.[name]?.fallback === true);
}

export function usesMeshopt(json) {
  return MESHOPT_EXTENSIONS.some((name) => json.extensionsUsed?.includes(name));
}

/**
 * Decode every compressed bufferView. Returns a new document and buffer list;
 * the originals are untouched. `maxBytes` bounds what a view may decode to,
 * the same bound every accessor gets -- a few compressed bytes can claim any
 * count, so the claim is refused before anything is allocated for it.
 */
export function decompressViews(json, buffers, maxBytes = Infinity) {
  const out = [...buffers];
  let changed = false;
  const views = (json.bufferViews ?? []).map((view, i) => {
    const name = MESHOPT_EXTENSIONS.find((n) => view.extensions?.[n]);
    if (!name) return view;
    changed = true;
    const bytes = decodeView(view, view.extensions[name], name === MESHOPT_EXTENSIONS[1], buffers, maxBytes, `bufferView ${i}`);
    out.push(bytes);
    return { ...view, buffer: out.length - 1, byteOffset: 0 };
  });
  return changed ? { json: { ...json, bufferViews: views }, buffers: out } : { json, buffers };
}

const MODES = ['ATTRIBUTES', 'TRIANGLES', 'INDICES'];
const FILTER_STRIDES = {
  NONE: () => true,
  OCTAHEDRAL: (s) => s === 4 || s === 8,
  QUATERNION: (s) => s === 8,
  EXPONENTIAL: (s) => s % 4 === 0,
  COLOR: (s) => s === 4 || s === 8,
};

function decodeView(view, ext, khr, buffers, maxBytes, where) {
  const fail = (rule) => { throw new Error(`glTF: ${where} ${rule}`); };
  const { byteStride: stride, count, mode } = ext;
  const filter = ext.filter ?? 'NONE';

  // The extension's validity rules, in its own order.
  if (!Number.isInteger(count) || count < 1) fail(`has meshopt count ${JSON.stringify(count)}`);
  if (!MODES.includes(mode)) fail(`has meshopt mode ${JSON.stringify(mode)}`);
  if (!(filter in FILTER_STRIDES) || (filter === 'COLOR' && !khr)) fail(`has meshopt filter ${JSON.stringify(filter)}`);
  if (view.byteStride !== undefined && view.byteStride !== stride) fail(`has byteStride ${view.byteStride} but its meshopt data has ${stride}`);
  if (stride * count !== view.byteLength) fail(`is ${view.byteLength} bytes but its meshopt data decodes to ${stride} x ${count}`);
  if (mode === 'ATTRIBUTES' && !(Number.isInteger(stride) && stride > 0 && stride % 4 === 0 && stride <= 256)) fail(`has meshopt byteStride ${stride}; attributes need a multiple of 4 up to 256`);
  if (mode !== 'ATTRIBUTES' && stride !== 2 && stride !== 4) fail(`has meshopt byteStride ${stride}; indices are 2 or 4 bytes`);
  if (mode === 'TRIANGLES' && count % 3 !== 0) fail(`has ${count} triangle indices, not a multiple of 3`);
  if (mode !== 'ATTRIBUTES' && filter !== 'NONE') fail(`filters indices with ${filter}`);
  if (!FILTER_STRIDES[filter](stride)) fail(`uses ${filter} on a ${stride}-byte stride`);
  if (stride * count > maxBytes) fail(`decodes to ${stride * count} bytes; this device's largest buffer is ${maxBytes}`);

  const source = buffers[ext.buffer];
  if (!source) fail(`names meshopt buffer ${ext.buffer}, which does not exist`);
  const start = ext.byteOffset ?? 0;
  if (!(Number.isInteger(start) && start >= 0 && Number.isInteger(ext.byteLength) && start + ext.byteLength <= source.byteLength)) {
    fail(`spans meshopt bytes ${start}..${start + ext.byteLength} of a ${source.byteLength}-byte buffer`);
  }
  const data = source.subarray(start, start + ext.byteLength);

  try {
    if (mode === 'TRIANGLES') return decodeTriangles(data, count, stride);
    if (mode === 'INDICES') return decodeIndexSequence(data, count, stride);
    const bytes = decodeAttributes(data, count, stride, khr);
    if (filter !== 'NONE') FILTERS[filter](bytes, count, stride);
    return bytes;
  } catch (e) {
    fail(`has meshopt data that ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// Mode 0: attributes. Elements in blocks of up to 256; within a block, each
// byte position is a plane of per-element deltas, stored in groups of 16 at
// 0, 1, 2, 4 or 8 bits each, a delta too big for its width stored whole after
// the group. Version 1 (KHR only) adds per-byte control of the widths and
// per-4-byte channels that delta 16- or 32-bit values instead of bytes.

// Bits per delta for each 2-bit group header: version 0, then version 1's
// control modes 0 and 1.
const WIDTHS_V0 = [0, 2, 4, 8];
const WIDTHS_V1 = [[0, 1, 2, 4], [1, 2, 4, 8]];
const GROUP = 16;
// The largest a group can be: 4-bit deltas plus all 16 stored whole. The tail
// is always at least this long, so a stream with fewer bytes left is short.
const GROUP_MAX_BYTES = 24;

export function decodeAttributes(data, count, stride, allowV1 = false) {
  const header = data[0];
  const version = header & 15;
  if ((header & 0xf0) !== 0xa0 || version > (allowV1 ? 1 : 0)) {
    throw new Error(`starts 0x${(header ?? 0).toString(16)}, not an attribute stream this extension defines`);
  }

  const tailBytes = stride + (version === 1 ? stride / 4 : 0);
  const tailPadded = Math.max(tailBytes, version === 1 ? 24 : 32);
  if (data.length - 1 < tailPadded) throw new Error('is too short to hold its tail');
  const tail = data.length - tailBytes;
  const last = data.slice(tail, tail + stride);   // the baseline, then each block's last element
  const channels = data.subarray(tail + stride);  // version 1: one mode byte per 4 bytes

  const out = new Uint8Array(count * stride);
  const blockMax = Math.min((8192 / stride) & ~(GROUP - 1), 256);
  const planes = new Uint8Array(blockMax * 4);
  let p = 1;

  for (let base = 0; base < count; base += blockMax) {
    const n = Math.min(blockMax, count - base);
    const aligned = (n + GROUP - 1) & ~(GROUP - 1);
    const control = p;
    if (version === 1) p += stride / 4;

    for (let k = 0; k < stride; k += 4) {
      const controlByte = version === 1 ? data[control + k / 4] : 0;
      for (let j = 0; j < 4; j++) {
        const plane = j * aligned;
        const ctrl = (controlByte >> (j * 2)) & 3;
        if (ctrl === 3) {                       // literal
          if (data.length - p < n) throw new Error('ends inside a block');
          planes.set(data.subarray(p, p + n), plane);
          p += n;
        } else if (ctrl === 2) {                // all zero
          planes.fill(0, plane, plane + aligned);
        } else {
          p = decodePlane(data, p, planes, plane, aligned, version === 1 ? WIDTHS_V1[ctrl] : WIDTHS_V0);
        }
      }

      const channel = version === 1 ? channels[k / 4] : 0;
      const kind = channel & 15;
      if (kind > 2 || (kind < 2 && channel >> 4)) throw new Error(`has channel mode 0x${channel.toString(16)}`);
      undelta(planes, aligned, n, out, base, stride, k, last, kind, channel >> 4);
    }
  }

  if (data.length - p !== tailPadded) throw new Error('has bytes left over before its tail');
  return out;
}

function decodePlane(data, p, out, o, size, widths) {
  const groups = size / GROUP;
  const header = p;
  p += (groups + 3) >> 2;
  for (let g = 0; g < groups; g++) {
    if (data.length - p < GROUP_MAX_BYTES) throw new Error('ends inside a block');
    const bits = widths[(data[header + (g >> 2)] >> ((g & 3) * 2)) & 3];
    p = decodeGroup(data, p, out, o + g * GROUP, bits);
  }
  return p;
}

// Specialised by width: this is the decoder's inner loop.
function decodeGroup(data, p, out, o, bits) {
  if (bits === 0) {
    for (let i = 0; i < GROUP; i++) out[o + i] = 0;
    return p;
  }
  if (bits === 8) {
    for (let i = 0; i < GROUP; i++) out[o + i] = data[p + i];
    return p + GROUP;
  }
  // Packed deltas: most significant first for 2 and 4 bits, least significant
  // first for 1 bit. All ones means the delta follows as a whole byte.
  let whole = p + 2 * bits;
  if (bits === 4) {
    for (let j = 0; j < 8; j++) {
      const byte = data[p + j];
      const hi = byte >> 4, lo = byte & 15;
      out[o++] = hi === 15 ? data[whole++] : hi;
      out[o++] = lo === 15 ? data[whole++] : lo;
    }
  } else if (bits === 2) {
    for (let j = 0; j < 4; j++) {
      const byte = data[p + j];
      for (let shift = 6; shift >= 0; shift -= 2) {
        const v = (byte >> shift) & 3;
        out[o++] = v === 3 ? data[whole++] : v;
      }
    }
  } else {
    for (let j = 0; j < 2; j++) {
      const byte = data[p + j];
      for (let shift = 0; shift < 8; shift++) out[o++] = (byte >> shift) & 1 ? data[whole++] : 0;
    }
  }
  return whole;
}

// Undo the deltas for bytes k..k+3 of every element in the block. Kind 0:
// four zigzagged byte deltas. Kind 1: two zigzagged 16-bit deltas. Kind 2: one
// 32-bit XOR, rotated left by `rotation` when it was encoded.
function undelta(planes, aligned, n, out, base, stride, k, last, kind, rotation) {
  let o = base * stride + k;
  if (kind === 0) {
    for (let j = 0; j < 4; j++) {
      let prev = last[k + j];
      for (let i = 0, w = o + j; i < n; i++, w += stride) {
        const d = planes[j * aligned + i];
        prev = (prev + ((d >> 1) ^ -(d & 1))) & 0xff;
        out[w] = prev;
      }
      last[k + j] = prev;
    }
  } else if (kind === 1) {
    for (let h = 0; h < 4; h += 2) {
      let prev = last[k + h] | (last[k + h + 1] << 8);
      for (let i = 0, w = o + h; i < n; i++, w += stride) {
        const d = planes[h * aligned + i] | (planes[(h + 1) * aligned + i] << 8);
        prev = (prev + ((d >> 1) ^ -(d & 1))) & 0xffff;
        out[w] = prev & 0xff;
        out[w + 1] = prev >> 8;
      }
      last[k + h] = prev & 0xff;
      last[k + h + 1] = prev >> 8;
    }
  } else {
    const r = (32 - rotation) & 31;
    let prev = last[k] | (last[k + 1] << 8) | (last[k + 2] << 16) | (last[k + 3] << 24);
    for (let i = 0, w = o; i < n; i++, w += stride) {
      const d = planes[i] | (planes[aligned + i] << 8) | (planes[2 * aligned + i] << 16) | (planes[3 * aligned + i] << 24);
      prev ^= (d << r) | (d >>> ((32 - r) & 31));
      out[w] = prev & 0xff;
      out[w + 1] = (prev >> 8) & 0xff;
      out[w + 2] = (prev >> 16) & 0xff;
      out[w + 3] = prev >>> 24;
    }
    last[k] = prev & 0xff;
    last[k + 1] = (prev >> 8) & 0xff;
    last[k + 2] = (prev >> 16) & 0xff;
    last[k + 3] = prev >>> 24;
  }
}

// ---------------------------------------------------------------------------
// Mode 1: triangles. One code byte per triangle, naming an edge from the last
// 16 edges and a third vertex that is the next new index, one of the last 16
// vertices, one either side of the last explicit index, or explicit; plus two
// codes for triangles that share nothing. Explicit indices are LEB128
// zigzagged deltas from the last one.

export function decodeTriangles(data, count, stride) {
  if (data[0] !== 0xe1) throw new Error(`starts 0x${(data[0] ?? 0).toString(16)}, not 0xe1`);
  if (data.length < 1 + count / 3 + 16) throw new Error('is too short for its triangle count');

  const out = stride === 2 ? new Uint16Array(count) : new Uint32Array(count);
  const edges = new Uint32Array(32);
  const verts = new Uint32Array(16);
  let edgeAt = 0, vertAt = 0, next = 0, last = 0;
  let code = 1;
  let p = 1 + count / 3;
  // The 16-byte table that ends the stream. A triangle reads at most 16
  // bytes, so starting at or before it can never read past the end.
  const table = data.length - 16;
  for (let j = 0; j < 16; j++) {
    const t = data[table + j];
    if (j < 14 ? t >> 4 === 15 || (t & 15) === 15 : t !== 0) throw new Error('has a malformed code table');
  }

  const edge = (a, b) => { edges[(edgeAt & 15) * 2] = a; edges[(edgeAt & 15) * 2 + 1] = b; edgeAt++; };
  const vertex = (v) => { verts[vertAt & 15] = v; vertAt++; };
  const recent = (back) => verts[(vertAt - back) & 15];
  const explicit = () => { last = (last + unzigzag(leb128())) >>> 0; return last; };
  const leb128 = () => {
    let v = 0;
    for (let shift = 0; shift < 35; shift += 7) {
      const byte = data[p++];
      v |= (byte & 127) << shift;
      if (byte < 128) break;
    }
    return v >>> 0;
  };

  for (let i = 0; i < count; i += 3) {
    if (p > table) throw new Error('runs into its code table');
    const tri = data[code++];
    let a, b, c;
    if (tri < 0xf0) {
      const e = ((edgeAt - 1 - (tri >> 4)) & 15) * 2;
      a = edges[e];
      b = edges[e + 1];
      const fc = tri & 15;
      if (fc === 0) { c = next++; vertex(c); }
      else if (fc < 13) c = recent(fc + 1);
      else if (fc < 15) { c = last = (last + (fc === 13 ? -1 : 1)) >>> 0; vertex(c); }
      else { c = explicit(); vertex(c); }
      edge(c, b);
      edge(a, c);
    } else {
      // 0xf0-0xfd look the second and third codes up in the table; 0xfe and
      // 0xff read them from the stream, a zero there restarting `next`.
      const aux = tri < 0xfe ? data[table + (tri & 15)] : data[p++];
      if (tri >= 0xfe && aux === 0) next = 0;
      const fb = aux >> 4, fc = aux & 15;
      a = tri === 0xff ? 0 : next++;
      b = fb === 0 ? next++ : fb < 15 ? recent(fb) : 0;
      c = fc === 0 ? next++ : fc < 15 ? recent(fc) : 0;
      if (tri === 0xff) a = explicit();
      if (fb === 15) b = explicit();
      if (fc === 15) c = explicit();
      vertex(a);
      if (fb === 0 || fb === 15) vertex(b);
      if (fc === 0 || fc === 15) vertex(c);
      edge(b, a);
      edge(c, b);
      edge(a, c);
    }
    out[i] = a;
    out[i + 1] = b;
    out[i + 2] = c;
  }

  if (p !== table) throw new Error('has bytes left over before its code table');
  return new Uint8Array(out.buffer);
}

// ---------------------------------------------------------------------------
// Mode 2: indices. Each is a LEB128 value whose low bit picks one of two
// running baselines and whose rest is a zigzagged delta from it.

export function decodeIndexSequence(data, count, stride) {
  if (data[0] !== 0xd1) throw new Error(`starts 0x${(data[0] ?? 0).toString(16)}, not 0xd1`);
  if (data.length < 1 + count + 4) throw new Error('is too short for its index count');

  const out = stride === 2 ? new Uint16Array(count) : new Uint32Array(count);
  const baselines = [0, 0];
  const end = data.length - 4;   // a 4-byte tail; an index is at most 5 bytes
  let p = 1;
  for (let i = 0; i < count; i++) {
    if (p >= end) throw new Error('ends early');
    let v = 0;
    for (let shift = 0; shift < 35; shift += 7) {
      const byte = data[p++];
      v |= (byte & 127) << shift;
      if (byte < 128) break;
    }
    const b = v & 1;
    baselines[b] = (baselines[b] + unzigzag(v >>> 1)) >>> 0;
    out[i] = baselines[b];
  }
  if (p !== end) throw new Error('has bytes left over before its tail');
  return new Uint8Array(out.buffer);
}

const unzigzag = (v) => (v & 1 ? ~(v >>> 1) : v >>> 1);

// ---------------------------------------------------------------------------
// Filters, applied in place after mode 0. Rounding is to nearest, halves away
// from zero, as the reference decoder does it.

const round = (v) => (v < 0 ? -Math.round(-v) : Math.round(v));

export const FILTERS = {
  // A unit vector from two octahedral coordinates; the third component holds
  // 1.0 at the precision used, and the fourth passes through.
  OCTAHEDRAL(bytes, count, stride) {
    const d = stride === 4 ? new Int8Array(bytes.buffer) : new Int16Array(bytes.buffer);
    const max = stride === 4 ? 127 : 32767;
    for (let i = 0; i < count * 4; i += 4) {
      let x = d[i], y = d[i + 1];
      const z = d[i + 2] - Math.abs(x) - Math.abs(y);
      const t = Math.min(z, 0);
      x += x >= 0 ? t : -t;
      y += y >= 0 ? t : -t;
      const s = max / Math.sqrt(x * x + y * y + z * z);
      d[i] = round(x * s);
      d[i + 1] = round(y * s);
      d[i + 2] = round(z * s);
    }
  },

  // A unit quaternion from its three smallest components, scaled by sqrt 2;
  // the fourth holds 1.0 at the precision used, over the index of the largest.
  QUATERNION(bytes, count) {
    const d = new Int16Array(bytes.buffer);
    for (let i = 0; i < count * 4; i += 4) {
      const s = Math.SQRT1_2 / (d[i + 3] | 3);
      const largest = d[i + 3] & 3;
      const x = d[i] * s, y = d[i + 1] * s, z = d[i + 2] * s;
      const w = Math.sqrt(Math.max(0, 1 - x * x - y * y - z * z));
      d[i + ((largest + 1) & 3)] = round(x * 32767);
      d[i + ((largest + 2) & 3)] = round(y * 32767);
      d[i + ((largest + 3) & 3)] = round(z * 32767);
      d[i + largest] = round(w * 32767);
    }
  },

  // m * 2^e, from a signed 8-bit exponent over a signed 24-bit mantissa.
  EXPONENTIAL(bytes, count, stride) {
    const d = new Int32Array(bytes.buffer);
    const f = new Float32Array(bytes.buffer);
    for (let i = 0; i < (count * stride) / 4; i++) f[i] = 2 ** (d[i] >> 24) * ((d[i] << 8) >> 8);
  },

  // RGBA from YCoCg; the alpha's top set bit gives the precision K.
  COLOR(bytes, count, stride) {
    const bits = stride * 2;
    const d = bits === 8 ? bytes : new Uint16Array(bytes.buffer);
    const max = 2 ** bits - 1;
    const signed = (v) => (v << (32 - bits)) >> (32 - bits);
    for (let i = 0; i < count * 4; i += 4) {
      const alpha = d[i + 3];
      const range = 2 ** (32 - Math.clz32(alpha)) - 1;
      const y = d[i], co = signed(d[i + 1]), cg = signed(d[i + 2]);
      const s = max / range;
      d[i] = round((y + co - cg) * s);
      d[i + 1] = round((y + cg) * s);
      d[i + 2] = round((y - co - cg) * s);
      d[i + 3] = round((((alpha << 1) & range) | (alpha & 1)) * s);
    }
  },
};
