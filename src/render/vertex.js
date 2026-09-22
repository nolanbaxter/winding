// The vertex format the renderer consumes.
//
// This lives in render/ rather than in the glTF importer because it is the
// RENDERER's contract: it describes what the PBR shader reads. The importer
// writes to match it, which makes scene/ depend on render/ -- downward, and
// therefore allowed. The other direction would have the renderer reaching up
// into the asset layer to find out what its own shader consumes.
//
// One definition, read by both the code that fills the buffer and the pipeline
// that binds it, so they cannot drift.
//
// ONE FORMAT, NOT A FAMILY. Every vertex carries a second UV set and a colour
// whether its asset has them or not, which costs 12 bytes a vertex -- 60 up
// from 48. The alternative is a vertex format per attribute combination, and
// that multiplies into the pipeline count, the shader permutations and the
// importer all at once. Carrying the fields is the cheaper mistake: an asset
// without them gets uv1 = uv0 and colour = white, both of which are identities
// downstream, so nothing branches at runtime either.
//
// The colour is unorm8x4 rather than four floats. glTF allows float, but
// vertex colours are authored at 8 bits per channel essentially always, and
// four floats would have made the stride 72 instead of 60.

export const VERTEX_STRIDE_FLOATS = 15;
export const VERTEX_STRIDE_BYTES = VERTEX_STRIDE_FLOATS * 4;   // 60

/** Float index of the packed unorm8x4 colour, for writers using a Uint32 view. */
export const VERTEX_COLOR_INDEX = 14;

/** Vertex colour of an asset that has none. Multiplies to identity. */
export const VERTEX_COLOR_WHITE = 0xffffffff;

export const VERTEX_BUFFER_LAYOUT = {
  arrayStride: VERTEX_STRIDE_BYTES,
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },    // position
    { shaderLocation: 1, offset: 12, format: 'float32x3' },   // normal
    { shaderLocation: 2, offset: 24, format: 'float32x2' },   // uv0
    { shaderLocation: 3, offset: 32, format: 'float32x4' },   // tangent + handedness
    { shaderLocation: 4, offset: 48, format: 'float32x2' },   // uv1
    { shaderLocation: 5, offset: 56, format: 'unorm8x4' },    // COLOR_0
  ],
};

/** Pack a float colour in [0,1] into the unorm8x4 the layout expects. */
export function packVertexColor(r, g, b, a) {
  return ((Math.max(0, Math.min(1, a)) * 255 + 0.5) << 24
    | (Math.max(0, Math.min(1, b)) * 255 + 0.5) << 16
    | (Math.max(0, Math.min(1, g)) * 255 + 0.5) << 8
    | (Math.max(0, Math.min(1, r)) * 255 + 0.5)) >>> 0;
}
