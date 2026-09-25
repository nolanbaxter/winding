// Radiance .hdr (RGBE) decoding, for environment maps.
//
// The format Poly Haven and most HDRI libraries ship: a text header, then
// scanlines of four bytes a pixel -- a shared exponent and three mantissas,
// which is how it holds a sun and a shadow in 32 bits a pixel. Scanlines are
// usually run-length coded one channel at a time.
//
// Files come from strangers, so every length is checked before it is trusted:
// the header's size against the device, and every run against the scanline it
// writes into. A malformed file is an error that says what is wrong with it,
// never a read past the end.

/**
 * Decode an .hdr file to linear RGB floats, top row first.
 *
 * `maxDimension` is the device's texture limit: a map it could not hold is
 * refused here, before a byte of it is decoded.
 *
 * @returns {{ width: number, height: number, data: Float32Array }} RGB, 3 floats a pixel
 */
export function parseHDR(bytes, { maxDimension = Infinity } = {}) {
  let at = 0;
  const line = () => {
    const start = at;
    while (at < bytes.length && bytes[at] !== 0x0a) at++;
    if (at >= bytes.length) throw new Error('hdr: the header never ends');
    return String.fromCharCode(...bytes.subarray(start, at++));
  };

  const magic = line();
  if (magic !== '#?RADIANCE' && magic !== '#?RGBE') {
    throw new Error(`hdr: not a Radiance file (it starts ${JSON.stringify(magic.slice(0, 16))})`);
  }
  // EXPOSURE says the pixels were multiplied by it after capture; radiance is
  // the stored value divided by every one of them.
  let exposure = 1;
  for (let header = line(); header !== ''; header = line()) {
    if (header.startsWith('FORMAT=') && header !== 'FORMAT=32-bit_rle_rgbe') {
      throw new Error(`hdr: ${header} is not supported; only 32-bit_rle_rgbe is`);
    }
    if (header.startsWith('EXPOSURE=')) {
      const value = Number(header.slice(9));
      if (!(value > 0 && Number.isFinite(value))) throw new Error(`hdr: ${header} is not a positive number`);
      exposure *= value;
    }
  }

  // "-Y height +X width" is the standard orientation: rows from the top, left
  // to right. "+Y" is the same from the bottom. The rest are rotations nothing
  // in practice writes.
  const size = /^([-+])Y (\d+) \+X (\d+)$/.exec(line());
  if (!size) throw new Error('hdr: only -Y h +X w and +Y h +X w orientations are supported');
  const height = Number(size[2]);
  const width = Number(size[3]);
  if (!(width > 0 && height > 0)) throw new Error(`hdr: a ${width}x${height} image has no pixels`);
  if (width > maxDimension || height > maxDimension) {
    throw new Error(`hdr: a ${width}x${height} map is past this device's ${maxDimension}`);
  }
  const bottomUp = size[1] === '+';

  const data = new Float32Array(width * height * 3);
  const scanline = new Uint8Array(width * 4);
  const scale = 1 / exposure;
  for (let row = 0; row < height; row++) {
    at = readScanline(bytes, at, scanline, width);
    const y = bottomUp ? height - 1 - row : row;
    for (let x = 0; x < width; x++) {
      const e = scanline[x * 4 + 3];
      const o = (y * width + x) * 3;
      if (e === 0) { data[o] = data[o + 1] = data[o + 2] = 0; continue; }
      // Radiance's own conversion: the mantissa's bin centre, times 2^(e-136).
      const f = 2 ** (e - 136) * scale;
      data[o] = (scanline[x * 4] + 0.5) * f;
      data[o + 1] = (scanline[x * 4 + 1] + 0.5) * f;
      data[o + 2] = (scanline[x * 4 + 2] + 0.5) * f;
    }
  }
  return { width, height, data };
}

/**
 * One scanline into `out` as RGBE bytes, pixel by pixel. Returns where the
 * next one starts.
 *
 * New-style runs are marked by the bytes 2 2 and the width, and code each
 * channel on its own: a count over 128 repeats the next byte, anything else
 * is that many literal bytes. Anything else is flat pixels, with the old
 * style's 1 1 1 n meaning "the last pixel again", n shifted up by a byte for
 * each such marker in a row.
 */
function readScanline(bytes, at, out, width) {
  const need = (n) => {
    if (at + n > bytes.length) throw new Error('hdr: the pixel data ends early');
  };
  need(4);
  if (width >= 8 && width < 0x8000 && bytes[at] === 2 && bytes[at + 1] === 2
    && ((bytes[at + 2] << 8) | bytes[at + 3]) === width) {
    at += 4;
    for (let channel = 0; channel < 4; channel++) {
      for (let x = 0; x < width;) {
        need(1);
        let count = bytes[at++];
        if (count > 128) {
          count -= 128;
          need(1);
          if (x + count > width) throw new Error('hdr: a run is longer than its scanline');
          const value = bytes[at++];
          for (let k = 0; k < count; k++) out[(x++) * 4 + channel] = value;
        } else {
          if (count === 0 || x + count > width) throw new Error('hdr: a run is longer than its scanline');
          need(count);
          for (let k = 0; k < count; k++) out[(x++) * 4 + channel] = bytes[at++];
        }
      }
    }
    return at;
  }

  let shift = 0;
  for (let x = 0; x < width;) {
    need(4);
    if (bytes[at] === 1 && bytes[at + 1] === 1 && bytes[at + 2] === 1) {
      if (x === 0) throw new Error('hdr: a repeat with no pixel before it');
      const count = bytes[at + 3] << shift;
      if (x + count > width) throw new Error('hdr: a run is longer than its scanline');
      for (let k = 0; k < count; k++, x++) out.copyWithin(x * 4, (x - 1) * 4, x * 4);
      shift += 8;
    } else {
      out.set(bytes.subarray(at, at + 4), x * 4);
      x++;
      shift = 0;
    }
    at += 4;
  }
  return at;
}

const HALF_MAX = 65504;

/**
 * An f32 as the bits of the nearest f16, for an rgba16float upload. Values
 * past f16's largest finite, 65504, are held there rather than turned into
 * infinity: a sun that bright still reads as blinding, and an infinity would
 * poison every convolution that sums it.
 */
export function halfBits(value) {
  if (!(value > 0)) return 0;          // negatives are not radiance; NaN is not a value
  if (value >= HALF_MAX) return 0x7bff;
  if (value < 2 ** -24) return 0;
  if (value < 2 ** -14) return Math.round(value / 2 ** -24);   // subnormal
  // Just under a power of two, log2 can round up to it; the mantissa then
  // rounds to zero, which is the right answer: that power of two.
  let exponent = Math.floor(Math.log2(value));
  let mantissa = Math.round((value / 2 ** exponent - 1) * 1024);
  if (mantissa === 1024) { mantissa = 0; exponent++; }
  return ((exponent + 15) << 10) | mantissa;
}

/** RGB floats as RGBA half-float bits, alpha 1, ready for writeTexture. */
export function toHalfRGBA(rgb) {
  const pixels = rgb.length / 3;
  const out = new Uint16Array(pixels * 4);
  const one = halfBits(1);
  for (let i = 0; i < pixels; i++) {
    out[i * 4] = halfBits(rgb[i * 3]);
    out[i * 4 + 1] = halfBits(rgb[i * 3 + 1]);
    out[i * 4 + 2] = halfBits(rgb[i * 3 + 2]);
    out[i * 4 + 3] = one;
  }
  return out;
}
