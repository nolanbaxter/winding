// Tangent generation.
//
// A normal map stores directions in TANGENT space -- a per-pixel basis built
// from the surface normal, the direction U increases across the surface, and
// the direction V does. Without a tangent the shader has a normal but no idea
// which way the texture is oriented, so normal mapping cannot work at all.
//
// glTF assets often ship without TANGENT because the exporter assumes the
// engine will derive it. Deriving it means solving, per triangle, the linear
// system that maps UV deltas to position deltas:
//
//   edge1 = T * du1 + B * dv1
//   edge2 = T * du2 + B * dv2
//
// Invert that 2x2 and you have T and B for the triangle. Accumulate per vertex
// across every triangle that touches it, then orthonormalize against the
// (interpolated, already-smoothed) normal.
//
// ponytail: this is the standard Lengyel derivation, not MikkTSpace. MikkTSpace
// is the de facto baking standard, and a normal map baked against it can show
// faint seams under this. Swap in a MikkTSpace port if that ever shows up on a
// real asset -- the interface here does not change.

const DEGENERATE_UV_EPSILON = 1e-12;

/**
 * @param positions Float32Array, 3 per vertex
 * @param normals   Float32Array, 3 per vertex (already final)
 * @param uvs       Float32Array, 2 per vertex
 * @param indices   Uint32Array, 3 per triangle
 * @returns Float32Array, 4 per vertex: xyz = tangent, w = handedness (+1/-1)
 */
export function generateTangents(positions, normals, uvs, indices) {
  const vertexCount = positions.length / 3;
  const tangents = new Float32Array(vertexCount * 4);

  // Accumulators. The bitangent is needed only to recover handedness at the
  // end, but it has to be summed alongside the tangent to stay consistent.
  const tanAccum = new Float32Array(vertexCount * 3);
  const bitanAccum = new Float32Array(vertexCount * 3);

  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];

    const p0 = i0 * 3, p1 = i1 * 3, p2 = i2 * 3;
    const e1x = positions[p1] - positions[p0];
    const e1y = positions[p1 + 1] - positions[p0 + 1];
    const e1z = positions[p1 + 2] - positions[p0 + 2];
    const e2x = positions[p2] - positions[p0];
    const e2y = positions[p2 + 1] - positions[p0 + 1];
    const e2z = positions[p2 + 2] - positions[p0 + 2];

    const t0 = i0 * 2, t1 = i1 * 2, t2 = i2 * 2;
    const du1 = uvs[t1] - uvs[t0];
    const dv1 = uvs[t1 + 1] - uvs[t0 + 1];
    const du2 = uvs[t2] - uvs[t0];
    const dv2 = uvs[t2 + 1] - uvs[t0 + 1];

    // Determinant of the UV matrix. Zero means the triangle has no area in UV
    // space -- collapsed or unwrapped onto a line -- and there is genuinely no
    // tangent to compute. Skipping leaves the vertex to its other triangles.
    const det = du1 * dv2 - du2 * dv1;
    if (Math.abs(det) < DEGENERATE_UV_EPSILON) continue;
    const r = 1 / det;

    const tx = (e1x * dv2 - e2x * dv1) * r;
    const ty = (e1y * dv2 - e2y * dv1) * r;
    const tz = (e1z * dv2 - e2z * dv1) * r;

    const bx = (e2x * du1 - e1x * du2) * r;
    const by = (e2y * du1 - e1y * du2) * r;
    const bz = (e2z * du1 - e1z * du2) * r;

    // Unrolled rather than looping over [p0, p1, p2]: that array would be a
    // fresh allocation for every triangle in the mesh.
    tanAccum[p0] += tx; tanAccum[p0 + 1] += ty; tanAccum[p0 + 2] += tz;
    tanAccum[p1] += tx; tanAccum[p1 + 1] += ty; tanAccum[p1 + 2] += tz;
    tanAccum[p2] += tx; tanAccum[p2 + 1] += ty; tanAccum[p2 + 2] += tz;

    bitanAccum[p0] += bx; bitanAccum[p0 + 1] += by; bitanAccum[p0 + 2] += bz;
    bitanAccum[p1] += bx; bitanAccum[p1 + 1] += by; bitanAccum[p1 + 2] += bz;
    bitanAccum[p2] += bx; bitanAccum[p2 + 1] += by; bitanAccum[p2 + 2] += bz;
  }

  for (let v = 0; v < vertexCount; v++) {
    const n = v * 3;
    const nx = normals[n], ny = normals[n + 1], nz = normals[n + 2];
    let tx = tanAccum[n], ty = tanAccum[n + 1], tz = tanAccum[n + 2];

    // Gram-Schmidt: remove whatever part of the accumulated tangent points
    // along the normal, leaving it in the surface plane where it belongs.
    const dot = nx * tx + ny * ty + nz * tz;
    tx -= nx * dot; ty -= ny * dot; tz -= nz * dot;

    let length = Math.hypot(tx, ty, tz);
    if (length < 1e-8) {
      // No usable tangent (isolated vertex, or every triangle degenerate).
      // Any perpendicular direction beats NaN, and flat-shaded geometry with
      // no normal map never looks at it.
      [tx, ty, tz] = perpendicularTo(nx, ny, nz);
      length = 1;
    }
    const inv = 1 / length;

    tangents[v * 4] = tx * inv;
    tangents[v * 4 + 1] = ty * inv;
    tangents[v * 4 + 2] = tz * inv;

    // Handedness: whether the bitangent should be cross(N, T) or its negative.
    // Mirrored UV islands flip it, which is exactly why glTF stores it per
    // vertex in w instead of assuming one convention for the whole mesh.
    const cx = ny * tz - nz * ty;
    const cy = nz * tx - nx * tz;
    const cz = nx * ty - ny * tx;
    const handedness = cx * bitanAccum[n] + cy * bitanAccum[n + 1] + cz * bitanAccum[n + 2];
    tangents[v * 4 + 3] = handedness < 0 ? -1 : 1;
  }

  return tangents;
}

/** Any unit vector perpendicular to n. Picks the axis n leans on least. */
function perpendicularTo(nx, ny, nz) {
  const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
  // Crossing with the least-aligned axis keeps the result well-conditioned.
  const [ux, uy, uz] = ax < ay && ax < az ? [1, 0, 0] : ay < az ? [0, 1, 0] : [0, 0, 1];
  const cx = ny * uz - nz * uy;
  const cy = nz * ux - nx * uz;
  const cz = nx * uy - ny * ux;
  const inv = 1 / (Math.hypot(cx, cy, cz) || 1);
  return [cx * inv, cy * inv, cz * inv];
}

/**
 * Flat normals, per the glTF rule that a mesh without NORMAL gets faceted
 * shading. That requires every triangle to own its vertices, so this returns
 * de-indexed geometry -- three unique vertices per face.
 */
export function unweldAndComputeFlatNormals(positions, indices, extraAttributes) {
  const triangleCount = indices.length / 3;
  const outPositions = new Float32Array(triangleCount * 9);
  const outNormals = new Float32Array(triangleCount * 9);
  const outExtras = extraAttributes.map(
    (attr) => new Float32Array(triangleCount * 3 * attr.components),
  );

  for (let t = 0; t < triangleCount; t++) {
    const triBase = t * 3;

    for (let corner = 0; corner < 3; corner++) {
      const source = indices[triBase + corner];
      const from = source * 3;
      const to = (t * 3 + corner) * 3;
      outPositions[to] = positions[from];
      outPositions[to + 1] = positions[from + 1];
      outPositions[to + 2] = positions[from + 2];

      for (let a = 0; a < extraAttributes.length; a++) {
        const { data, components } = extraAttributes[a];
        const f = source * components;
        const o = (t * 3 + corner) * components;
        for (let c = 0; c < components; c++) outExtras[a][o + c] = data[f + c];
      }
    }

    const b = t * 9;
    const e1x = outPositions[b + 3] - outPositions[b];
    const e1y = outPositions[b + 4] - outPositions[b + 1];
    const e1z = outPositions[b + 5] - outPositions[b + 2];
    const e2x = outPositions[b + 6] - outPositions[b];
    const e2y = outPositions[b + 7] - outPositions[b + 1];
    const e2z = outPositions[b + 8] - outPositions[b + 2];

    // Counter-clockwise winding puts the face normal along this cross product,
    // matching the engine's frontFace: 'ccw'.
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const length = Math.hypot(nx, ny, nz);
    if (length > 0) {
      nx /= length; ny /= length; nz /= length;
    } else {
      // A zero-area triangle has no normal to compute. Exporters emit them
      // routinely at welded UV seams and collapsed quads, so this is not an
      // exotic input. Leaving (0,0,0) would reach the shader's
      // normalize(tbn * tangentNormal) and hand it a basis containing a zero
      // column -- NaN, which then spreads through the lighting for every
      // fragment of that face. Any unit vector is as defensible as another
      // here, and +Y matches the up axis the rest of the engine assumes.
      // The tangent path twenty lines up already does exactly this.
      nx = 0; ny = 1; nz = 0;
    }

    for (let corner = 0; corner < 3; corner++) {
      const o = b + corner * 3;
      outNormals[o] = nx; outNormals[o + 1] = ny; outNormals[o + 2] = nz;
    }
  }

  const outIndices = new Uint32Array(triangleCount * 3);
  for (let i = 0; i < outIndices.length; i++) outIndices[i] = i;

  return { positions: outPositions, normals: outNormals, indices: outIndices, extras: outExtras };
}
