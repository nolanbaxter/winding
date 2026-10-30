// A mesh made from arrays, written as a .glb: what engine.createMesh loads.
//
// Written as a file and loaded, rather than uploaded straight: the loader
// already does everything a mesh needs -- flat normals where none are given
// (glTF's rule), tangents, the chunk boxes culling reads, a material shared
// with an identical one, its pipelines built before it is handed back -- and
// a generated mesh then comes back from a lost GPU device like any file does,
// read again from these bytes. The cost is a copy of the arrays, once.

const FLOAT = 5126, UNSIGNED_SHORT = 5123, UNSIGNED_INT = 5125;

/**
 * GLB bytes for one mesh: { positions, indices, normals, uvs, colors, material },
 * checked, every error naming `caller`.
 */
export function meshGLB({ positions, indices, normals, uvs, colors, material = {} } = {}, caller = 'createMesh') {
  const floats = (name, values, required = false) => {
    if (values === undefined || values === null) {
      if (required) throw new Error(`${caller}: ${name} is required: x, y, z for each vertex`);
      return null;
    }
    if (!(Array.isArray(values) || ArrayBuffer.isView(values))) throw new Error(`${caller}: ${name} must be an array of numbers`);
    const out = Float32Array.from(values);
    for (let i = 0; i < out.length; i++) {
      if (!Number.isFinite(out[i])) throw new Error(`${caller}: ${name} must be finite numbers, got ${values[i]} at ${i}`);
    }
    return out;
  };
  const p = floats('positions', positions, true);
  if (p.length === 0 || p.length % 3 !== 0) throw new Error(`${caller}: positions must be x, y, z for each vertex: a multiple of 3 numbers, got ${p.length}`);
  const vertexCount = p.length / 3;

  const check = (name, values, per) => {
    if (values !== null && values.length !== vertexCount * per) {
      throw new Error(`${caller}: ${name} must be ${per} numbers for each of the ${vertexCount} vertices, got ${values.length}`);
    }
    return values;
  };
  const n = check('normals', floats('normals', normals), 3);
  const t = check('uvs', floats('uvs', uvs), 2);
  const c = floats('colors', colors);
  if (c !== null && c.length !== vertexCount * 3 && c.length !== vertexCount * 4) {
    throw new Error(`${caller}: colors must be 3 or 4 numbers (RGB or RGBA) for each of the ${vertexCount} vertices, got ${c.length}`);
  }
  const colorComponents = c === null ? 0 : c.length / vertexCount;

  let ix = null;
  if (indices !== undefined && indices !== null) {
    if (!(Array.isArray(indices) || ArrayBuffer.isView(indices))) throw new Error(`${caller}: indices must be an array of whole numbers`);
    if (indices.length === 0 || indices.length % 3 !== 0) throw new Error(`${caller}: indices must be three a triangle: a multiple of 3, got ${indices.length}`);
    ix = vertexCount > 65535 ? new Uint32Array(indices.length) : new Uint16Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
      const v = indices[i];
      if (!(Number.isInteger(v) && v >= 0 && v < vertexCount)) throw new Error(`${caller}: indices must be whole numbers from 0 to ${vertexCount - 1}, got ${v} at ${i}`);
      ix[i] = v;
    }
  } else if (vertexCount % 3 !== 0) {
    throw new Error(`${caller}: without indices, every three positions are a triangle: ${vertexCount} vertices is not a whole number of them`);
  }

  const {
    color = [1, 1, 1, 1], metallic = 0, roughness = 1, emissive = [0, 0, 0], doubleSided = false,
  } = material ?? {};
  if (!(color?.length === 4 && [...color].every((v) => Number.isFinite(v) && v >= 0))) {
    throw new Error(`${caller}: material.color must be 4 finite numbers, 0 or more, got ${color}`);
  }
  if (!(metallic >= 0 && metallic <= 1)) throw new Error(`${caller}: material.metallic must be from 0 to 1, got ${metallic}`);
  if (!(roughness >= 0 && roughness <= 1)) throw new Error(`${caller}: material.roughness must be from 0 to 1, got ${roughness}`);
  if (!(emissive?.length === 3 && [...emissive].every((v) => Number.isFinite(v) && v >= 0))) {
    throw new Error(`${caller}: material.emissive must be 3 finite numbers, 0 or more, got ${emissive}`);
  }
  // Blended where anything is less than opaque: the colour's alpha, or a vertex's.
  let blend = color[3] < 1;
  if (colorComponents === 4) for (let i = 3; i < c.length && !blend; i += 4) blend = c[i] < 1;

  // The binary chunk: each array 4-byte aligned, a view and an accessor apiece.
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i += 3) {
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], p[i + k]); max[k] = Math.max(max[k], p[i + k]); }
  }
  const parts = [];
  const accessors = [];
  const add = (array, componentType, type, count, extra = {}) => {
    parts.push(array);
    accessors.push({ bufferView: parts.length - 1, componentType, count, type, ...extra });
    return accessors.length - 1;
  };
  const attributes = { POSITION: add(p, FLOAT, 'VEC3', vertexCount, { min, max }) };
  if (n !== null) attributes.NORMAL = add(n, FLOAT, 'VEC3', vertexCount);
  if (t !== null) attributes.TEXCOORD_0 = add(t, FLOAT, 'VEC2', vertexCount);
  if (c !== null) attributes.COLOR_0 = add(c, FLOAT, colorComponents === 4 ? 'VEC4' : 'VEC3', vertexCount);
  const primitive = { attributes, material: 0 };
  if (ix !== null) primitive.indices = add(ix, ix instanceof Uint32Array ? UNSIGNED_INT : UNSIGNED_SHORT, 'SCALAR', ix.length);

  const bufferViews = [];
  let length = 0;
  for (const part of parts) {
    bufferViews.push({ buffer: 0, byteOffset: length, byteLength: part.byteLength });
    length += Math.ceil(part.byteLength / 4) * 4;
  }
  const json = {
    asset: { version: '2.0', generator: 'Winding createMesh' },
    scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [primitive] }],
    materials: [{
      pbrMetallicRoughness: { baseColorFactor: [...color], metallicFactor: metallic, roughnessFactor: roughness },
      emissiveFactor: [...emissive], doubleSided: doubleSided === true, alphaMode: blend ? 'BLEND' : 'OPAQUE',
    }],
    buffers: [{ byteLength: length }], bufferViews, accessors,
  };

  const text = new TextEncoder().encode(JSON.stringify(json));
  const jsonLength = Math.ceil(text.length / 4) * 4;
  const glb = new Uint8Array(12 + 8 + jsonLength + 8 + length);
  const view = new DataView(glb.buffer);
  view.setUint32(0, 0x46546c67, true);              // 'glTF'
  view.setUint32(4, 2, true);
  view.setUint32(8, glb.length, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);             // 'JSON'
  glb.fill(0x20, 20, 20 + jsonLength);              // padded with spaces, as the spec says
  glb.set(text, 20);
  view.setUint32(20 + jsonLength, length, true);
  view.setUint32(24 + jsonLength, 0x004e4942, true); // 'BIN\0'
  let at = 28 + jsonLength;
  for (const part of parts) {
    glb.set(new Uint8Array(part.buffer, part.byteOffset, part.byteLength), at);
    at += Math.ceil(part.byteLength / 4) * 4;
  }
  return glb;
}
