// Resolving weighted-blended OIT into the scene colour.
//
// Separate from pbr.js because it is a full-screen composite, not a surface
// shader: it binds two textures and no material, no lights and no geometry.
//
// accum holds sum(colour * alpha * w) and sum(alpha * w); reveal holds the
// product of (1 - alpha), which is exactly how much background survives. The
// weighted average of the colours is accum.rgb / accum.a, and it is composited
// over whatever is already there by (1 - reveal).
export const OIT_RESOLVE_SHADER = /* wgsl */ `
@group(0) @binding(0) var accumTex  : texture_2d<f32>;
@group(0) @binding(1) var revealTex : texture_2d<f32>;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index : u32) -> VertexOut {
  // The same full-screen triangle the post stack uses: three vertices with no
  // buffer, which beats a quad's two triangles and their shared edge.
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  var out : VertexOut;
  out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs(v : VertexOut) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(v.position.xy);
  let accum = textureLoad(accumTex, coord, 0);
  let reveal = textureLoad(revealTex, coord, 0).r;

  // Nothing blended covered this pixel. Returning zero with a zero alpha lets
  // the blend state leave the background exactly as it was, rather than
  // compositing a divide-by-almost-zero over it.
  if (reveal > 0.9999) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }

  // The average is weighted, so the weights divide back out. accum.a is the
  // sum of the weights and cannot be zero here -- reveal below 1 means at
  // least one fragment contributed -- but a fully transparent one contributes
  // a weight of zero, so the guard stays.
  let colour = accum.rgb / max(accum.a, 1e-5);
  return vec4<f32>(colour, 1.0 - reveal);
}
`;
