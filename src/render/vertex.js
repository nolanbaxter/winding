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

export const VERTEX_STRIDE_FLOATS = 12;
export const VERTEX_STRIDE_BYTES = VERTEX_STRIDE_FLOATS * 4;   // 48

export const VERTEX_BUFFER_LAYOUT = {
  arrayStride: VERTEX_STRIDE_BYTES,
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },    // position
    { shaderLocation: 1, offset: 12, format: 'float32x3' },   // normal
    { shaderLocation: 2, offset: 24, format: 'float32x2' },   // uv
    { shaderLocation: 3, offset: 32, format: 'float32x4' },   // tangent + handedness
  ],
};
