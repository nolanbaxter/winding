// glTF 2.0 import.
//
// Produces plain CPU data -- typed arrays and descriptions, no GPU objects.
// That separation is deliberate: it keeps the importer independent of the RHI,
// makes it testable with no GPU at all, and leaves uploading as a decision the
// renderer makes rather than one the loader forces.
//
//   const model = await loadGLTF(bytes);
//   instantiate(model, entities, transforms);   // becomes entities
//
// Not handled: skins, cameras, morph targets, KHR extensions. Animations ARE
// read (see animation.js), but only their TRS channels -- a `weights` channel
// has no morph targets to drive and is dropped.
// Each is a real feature, none is needed to draw a static mesh, and every one
// of them is additive to this file rather than a rewrite of it.

import { DEBUG, assertFinite } from '../../core/assert.js';
import { NULL_HANDLE } from '../../core/handle.js';
import { VERTEX_STRIDE_FLOATS } from '../../render/vertex.js';
import { mat4Decompose } from '../../core/math/mat4.js';
import { parseContainer, resolveBuffers } from './glb.js';
import { readAccessorAsFloat32, readAccessorAsUint32 } from './accessor.js';
import { generateTangents, unweldAndComputeFlatNormals } from './tangents.js';

// The vertex format is the renderer's contract, defined in render/vertex.js.
// Re-exported here so importer callers do not have to know that, but there is
// exactly one definition and this file is not it.
import { readAnimations } from './animation.js';

export {
  VERTEX_STRIDE_FLOATS, VERTEX_STRIDE_BYTES, VERTEX_BUFFER_LAYOUT,
} from '../../render/vertex.js';

const MODE_TRIANGLES = 4;
const MODE_NAMES = {
  0: 'POINTS', 1: 'LINES', 2: 'LINE_LOOP', 3: 'LINE_STRIP',
  4: 'TRIANGLES', 5: 'TRIANGLE_STRIP', 6: 'TRIANGLE_FAN',
};

export const DEFAULT_MATERIAL = Object.freeze({
  name: 'default',
  baseColorFactor: Float32Array.from([1, 1, 1, 1]),
  metallic: 1,
  roughness: 1,
  emissive: Float32Array.from([0, 0, 0]),
  alphaMode: 'OPAQUE',
  alphaCutoff: 0.5,
  doubleSided: false,
  normalScale: 1,
  occlusionStrength: 1,
  textures: Object.freeze({ baseColor: -1, metallicRoughness: -1, normal: -1, occlusion: -1, emissive: -1 }),
});

/**
 * Load a .glb or .gltf.
 *
 * `baseURL` is only needed when the document references external buffer files;
 * a .glb with its BIN chunk needs nothing.
 */
export async function loadGLTF(source, options = {}) {
  const { json, binary } = parseContainer(source);

  const version = json.asset?.version;
  if (version === undefined) throw new Error('glTF: document has no asset.version');
  if (!version.startsWith('2.')) {
    throw new Error(`glTF: version ${version} is not supported (this reads 2.x)`);
  }

  const required = json.extensionsRequired ?? [];
  if (required.length > 0) {
    // Loading anyway would produce geometry that is wrong in a way nothing
    // reports -- Draco-compressed buffers read as noise, quantized meshes come
    // out the wrong size. Say so.
    throw new Error(`glTF: requires unsupported extensions: ${required.join(', ')}`);
  }

  const buffers = await resolveBuffers(json, binary, options);
  return buildModel(json, buffers);
}

function buildModel(json, buffers) {
  const materials = readMaterials(json);
  const meshes = (json.meshes ?? []).map((mesh, i) => ({
    name: mesh.name ?? `mesh_${i}`,
    primitives: (mesh.primitives ?? []).map((p) => buildPrimitive(json, buffers, p)),
  }));
  const nodes = readNodes(json);

  return {
    nodes,
    meshes,
    materials,
    animations: readAnimations(json, buffers),
    roots: findRoots(json, nodes),
    // The raw document, for subsystems that need what this view deliberately
    // drops -- image decoding needs the bufferViews, and extensions will need
    // the JSON. Exposed explicitly rather than re-parsed, and never read by
    // anything in this file.
    source: { json, buffers },
  };
}

// ------------------------------------------------------------- primitives

function buildPrimitive(json, buffers, primitive) {
  const mode = primitive.mode ?? MODE_TRIANGLES;
  if (mode !== MODE_TRIANGLES) {
    throw new Error(
      `glTF: primitive mode ${MODE_NAMES[mode] ?? mode} is not supported; ` +
      'the renderer draws triangle lists only',
    );
  }

  const attributes = primitive.attributes ?? {};
  if (attributes.POSITION === undefined) {
    throw new Error('glTF: primitive has no POSITION attribute');
  }

  let positions = readAccessorAsFloat32(json, buffers, attributes.POSITION);
  let vertexCount = positions.length / 3;

  let indices = primitive.indices !== undefined
    ? readAccessorAsUint32(json, buffers, primitive.indices)
    : sequentialIndices(vertexCount);

  let normals = attributes.NORMAL !== undefined
    ? readAccessorAsFloat32(json, buffers, attributes.NORMAL) : null;
  let uvs = attributes.TEXCOORD_0 !== undefined
    ? readAccessorAsFloat32(json, buffers, attributes.TEXCOORD_0) : null;
  let tangents = attributes.TANGENT !== undefined
    ? readAccessorAsFloat32(json, buffers, attributes.TANGENT) : null;

  checkLength(normals, vertexCount, 3, 'NORMAL');
  checkLength(uvs, vertexCount, 2, 'TEXCOORD_0');
  checkLength(tangents, vertexCount, 4, 'TANGENT');

  for (let i = 0; i < indices.length; i++) {
    if (indices[i] >= vertexCount) {
      throw new Error(`glTF: index ${indices[i]} is past the ${vertexCount}-vertex attribute array`);
    }
  }

  // Bounds come from the POSITION accessor when the exporter provided them --
  // the spec requires min/max there precisely so a loader can build a bounding
  // volume without touching the vertex data.
  const bounds = accessorBounds(json, attributes.POSITION) ?? computeBounds(positions);

  // Bounds are derived from every position, so six numbers stand in for the
  // whole vertex buffer: a NaN anywhere in it lands here.
  if (DEBUG) {
    assertFinite(bounds.min, 'glTF POSITION bounds');
    assertFinite(bounds.max, 'glTF POSITION bounds');
  }

  const hadUVs = uvs !== null;

  if (normals === null) {
    // The spec is explicit: a primitive with no NORMAL is flat-shaded, and any
    // supplied tangents are discarded. Flat shading needs per-face vertices, so
    // the geometry has to be de-indexed first.
    const extras = hadUVs ? [{ data: uvs, components: 2 }] : [];
    const unwelded = unweldAndComputeFlatNormals(positions, indices, extras);
    positions = unwelded.positions;
    normals = unwelded.normals;
    indices = unwelded.indices;
    if (hadUVs) uvs = unwelded.extras[0];
    tangents = null;
    vertexCount = positions.length / 3;
  }

  if (!hadUVs) {
    uvs = new Float32Array(vertexCount * 2);
  } else if (tangents === null) {
    tangents = generateTangents(positions, normals, uvs, indices);
  }

  if (tangents === null) {
    // No UVs means no tangent frame exists to compute. Fill with a valid unit
    // vector rather than zeros so a shader that samples it cannot produce NaN.
    tangents = new Float32Array(vertexCount * 4);
    for (let v = 0; v < vertexCount; v++) {
      tangents[v * 4] = 1;
      tangents[v * 4 + 3] = 1;
    }
  }

  return {
    vertices: interleave(positions, normals, uvs, tangents, vertexCount),
    indices,
    vertexCount,
    indexCount: indices.length,
    material: primitive.material ?? -1,
    bounds,
  };
}

function interleave(positions, normals, uvs, tangents, vertexCount) {
  const out = new Float32Array(vertexCount * VERTEX_STRIDE_FLOATS);
  for (let v = 0; v < vertexCount; v++) {
    const o = v * VERTEX_STRIDE_FLOATS;
    const p = v * 3, t = v * 2, g = v * 4;

    out[o] = positions[p]; out[o + 1] = positions[p + 1]; out[o + 2] = positions[p + 2];
    out[o + 3] = normals[p]; out[o + 4] = normals[p + 1]; out[o + 5] = normals[p + 2];
    out[o + 6] = uvs[t]; out[o + 7] = uvs[t + 1];
    out[o + 8] = tangents[g]; out[o + 9] = tangents[g + 1];
    out[o + 10] = tangents[g + 2]; out[o + 11] = tangents[g + 3];
  }
  return out;
}

function sequentialIndices(vertexCount) {
  const indices = new Uint32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) indices[i] = i;
  return indices;
}

function checkLength(array, vertexCount, components, name) {
  if (array !== null && array.length !== vertexCount * components) {
    throw new Error(
      `glTF: ${name} has ${array.length / components} entries but POSITION has ${vertexCount}`,
    );
  }
}

function accessorBounds(json, accessorIndex) {
  const accessor = json.accessors?.[accessorIndex];
  if (!accessor?.min || !accessor?.max) return null;
  return { min: Float32Array.from(accessor.min), max: Float32Array.from(accessor.max) };
}

function computeBounds(positions) {
  const min = Float32Array.from([Infinity, Infinity, Infinity]);
  const max = Float32Array.from([-Infinity, -Infinity, -Infinity]);
  for (let i = 0; i < positions.length; i += 3) {
    for (let c = 0; c < 3; c++) {
      if (positions[i + c] < min[c]) min[c] = positions[i + c];
      if (positions[i + c] > max[c]) max[c] = positions[i + c];
    }
  }
  return { min, max };
}

// --------------------------------------------------------------- materials

function readMaterials(json) {
  return (json.materials ?? []).map((material, i) => {
    const pbr = material.pbrMetallicRoughness ?? {};
    return {
      name: material.name ?? `material_${i}`,
      baseColorFactor: Float32Array.from(pbr.baseColorFactor ?? [1, 1, 1, 1]),
      metallic: pbr.metallicFactor ?? 1,
      roughness: pbr.roughnessFactor ?? 1,
      emissive: Float32Array.from(material.emissiveFactor ?? [0, 0, 0]),
      alphaMode: material.alphaMode ?? 'OPAQUE',
      alphaCutoff: material.alphaCutoff ?? 0.5,
      doubleSided: material.doubleSided === true,
      normalScale: material.normalTexture?.scale ?? 1,
      // How much the occlusion map is allowed to darken ambient light. Lives on
      // the texture reference in glTF, not on the material.
      occlusionStrength: material.occlusionTexture?.strength ?? 1,
      // Texture INDICES, not images. Decoding and uploading belong to the
      // texture system; resolving them here would drag the RHI into a file that
      // has no other reason to know a GPU exists.
      textures: {
        baseColor: pbr.baseColorTexture?.index ?? -1,
        metallicRoughness: pbr.metallicRoughnessTexture?.index ?? -1,
        normal: material.normalTexture?.index ?? -1,
        occlusion: material.occlusionTexture?.index ?? -1,
        emissive: material.emissiveTexture?.index ?? -1,
      },
    };
  });
}

// ------------------------------------------------------------------ nodes

function readNodes(json) {
  return (json.nodes ?? []).map((node, i) => {
    const position = new Float32Array(3);
    const rotation = Float32Array.from([0, 0, 0, 1]);
    const scale = Float32Array.from([1, 1, 1]);

    if (node.matrix) {
      // glTF allows either form. TransformStore holds TRS, so a matrix node is
      // decomposed here -- once, at load -- rather than every frame.
      if (node.matrix.length !== 16) {
        throw new Error(`glTF: node ${i} has a matrix with ${node.matrix.length} entries`);
      }
      const m = Float32Array.from(node.matrix);
      if (!mat4Decompose(position, rotation, scale, m)) {
        throw new Error(`glTF: node ${i} has a degenerate matrix that cannot be decomposed`);
      }
    } else {
      if (node.translation) position.set(node.translation);
      if (node.rotation) rotation.set(node.rotation);
      if (node.scale) scale.set(node.scale);
    }

    return {
      name: node.name ?? `node_${i}`,
      position,
      rotation,
      scale,
      children: node.children ?? [],
      mesh: node.mesh ?? -1,
    };
  });
}

function findRoots(json, nodes) {
  const scene = json.scenes?.[json.scene ?? 0];
  if (scene?.nodes) return scene.nodes.slice();

  // No scene declared: anything nothing else claims as a child is a root.
  const isChild = new Uint8Array(nodes.length);
  for (const node of nodes) for (const child of node.children) isChild[child] = 1;

  const roots = [];
  for (let i = 0; i < nodes.length; i++) if (!isChild[i]) roots.push(i);
  return roots;
}

// ------------------------------------------------------------ instantiate

/**
 * Turn a parsed model into live entities and transforms.
 *
 * Returns the entity handle for every node (indexed the same as model.nodes),
 * plus the subset that carry a mesh -- which is what the renderer needs and
 * what walking the tree again later would only have to rediscover.
 */
export function instantiate(model, entities, transforms, { parent = NULL_HANDLE } = {}) {
  const created = new Array(model.nodes.length).fill(NULL_HANDLE);
  const renderables = [];

  const visit = (nodeIndex, parentEntity) => {
    const node = model.nodes[nodeIndex];
    if (!node) throw new Error(`glTF: node ${nodeIndex} is referenced but does not exist`);
    if (created[nodeIndex] !== NULL_HANDLE) {
      // glTF node graphs must be forests. A repeated node means a cycle or a
      // diamond, and recursing into it would not terminate.
      throw new Error(`glTF: node ${nodeIndex} has more than one parent`);
    }

    const entity = entities.alloc();
    created[nodeIndex] = entity;
    transforms.add(entity, {
      position: node.position,
      rotation: node.rotation,
      scale: node.scale,
      parent: parentEntity,
    });

    if (node.mesh >= 0) renderables.push({ entity, mesh: node.mesh });
    for (const child of node.children) visit(child, entity);
  };

  for (const root of model.roots) visit(root, parent);
  return { entities: created, renderables };
}
