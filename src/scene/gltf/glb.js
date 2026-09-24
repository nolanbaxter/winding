// glTF container parsing: .glb (binary) and .gltf (JSON).
//
// A .glb is a 12-byte header followed by length-prefixed chunks: a JSON chunk
// describing the scene, then usually one BIN chunk holding every buffer. It is
// the same data as a .gltf, just without a second network round trip for the
// .bin -- which is why it is what you actually ship.

const GLB_MAGIC = 0x46546c67;    // 'glTF', little-endian
const CHUNK_JSON = 0x4e4f534a;   // 'JSON'
const CHUNK_BIN = 0x004e4942;    // 'BIN\0'
const HEADER_BYTES = 12;
const CHUNK_HEADER_BYTES = 8;

/**
 * Returns { json, binary } where `binary` is the BIN chunk as a Uint8Array, or
 * null when there isn't one (plain .gltf, or a .glb with external buffers).
 *
 * Accepts either format and detects which by the magic number, so callers do
 * not have to care what extension the file had.
 */
export function parseContainer(source) {
  const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);

  if (bytes.byteLength < 4) throw new Error('glTF: file is too short to be valid');

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== GLB_MAGIC) {
    // Not binary: the whole file is the JSON document.
    return { json: JSON.parse(new TextDecoder().decode(bytes)), binary: null };
  }

  if (bytes.byteLength < HEADER_BYTES) throw new Error('glb: truncated header');

  const version = view.getUint32(4, true);
  if (version !== 2) {
    throw new Error(`glb: version ${version} is not supported (this reads glTF 2.0)`);
  }

  const declaredLength = view.getUint32(8, true);
  if (declaredLength > bytes.byteLength) {
    throw new Error(
      `glb: header declares ${declaredLength} bytes but the file has ${bytes.byteLength}`,
    );
  }

  let json = null;
  let binary = null;
  let offset = HEADER_BYTES;

  while (offset + CHUNK_HEADER_BYTES <= declaredLength) {
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    const dataStart = offset + CHUNK_HEADER_BYTES;

    if (dataStart + chunkLength > declaredLength) {
      throw new Error(`glb: chunk at ${offset} runs past the end of the file`);
    }

    if (chunkType === CHUNK_JSON) {
      if (json !== null) throw new Error('glb: more than one JSON chunk');
      json = JSON.parse(new TextDecoder().decode(bytes.subarray(dataStart, dataStart + chunkLength)));
    } else if (chunkType === CHUNK_BIN) {
      if (binary !== null) throw new Error('glb: more than one BIN chunk');
      binary = bytes.subarray(dataStart, dataStart + chunkLength);
    }
    // Unknown chunk types are skipped on purpose: the spec reserves them for
    // future use and requires readers to ignore what they don't recognize.

    // Chunk lengths are already 4-byte padded per spec, but rounding up costs
    // nothing and keeps one malformed exporter from desyncing the whole walk.
    // Math, not `& ~3`: that works in signed 32 bits, and a 2 GiB chunk turned
    // the offset negative.
    offset = dataStart + Math.ceil(chunkLength / 4) * 4;
  }

  if (json === null) throw new Error('glb: no JSON chunk');
  return { json, binary };
}

/**
 * Resolve every buffer the document references into a Uint8Array.
 *
 * Three sources, in the order the spec allows them:
 *   - no uri          -> the GLB BIN chunk (only valid for buffer 0)
 *   - data: uri       -> base64 inline
 *   - relative uri    -> fetched against baseURL
 */
export async function resolveBuffers(json, binary, { baseURL, fetchImpl = globalThis.fetch } = {}) {
  const buffers = json.buffers ?? [];

  return Promise.all(buffers.map(async (buffer, i) => {
    if (buffer.uri === undefined) {
      if (i !== 0 || binary === null) {
        throw new Error(`glTF: buffer ${i} has no uri and there is no BIN chunk for it`);
      }
      return binary;
    }

    if (buffer.uri.startsWith('data:')) return decodeDataURI(buffer.uri);

    if (!baseURL) {
      throw new Error(
        `glTF: buffer ${i} references "${buffer.uri}" but no baseURL was given to resolve it against`,
      );
    }
    const response = await fetchImpl(new URL(buffer.uri, baseURL));
    if (!response.ok) {
      throw new Error(`glTF: fetching buffer ${i} (${buffer.uri}) failed with ${response.status}`);
    }
    return new Uint8Array(await response.arrayBuffer());
  }));
}

function decodeDataURI(uri) {
  const comma = uri.indexOf(',');
  if (comma < 0) throw new Error('glTF: malformed data uri');

  const meta = uri.slice(5, comma);
  const payload = uri.slice(comma + 1);

  if (!meta.endsWith(';base64')) {
    // Percent-encoded text data uris are legal but nothing produces them for
    // binary buffers. Failing loudly beats returning plausible garbage.
    throw new Error('glTF: only base64 data uris are supported for buffers');
  }

  // atob exists in browsers and in Node 16+.
  const binary = atob(payload);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}
