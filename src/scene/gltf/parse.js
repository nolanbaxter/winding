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
// Not handled: cameras, KHR extensions.
//
// Skins and morph targets both come through whole -- the joint list, the
// inverse bind matrices and the per-vertex influences (skin.js); the per-target
// vertex deltas and the weights that mix them (morph.js). Animations bring
// their TRS channels and their `weights` channels alike.
//
// What DEFORMS geometry is a vertex shader, a matrix palette and a bounds
// derivation, none of which live in the importer. What is here is the data
// those need, in the layout they want it, verified on its own.

import { DEBUG, assertFinite } from '../../core/assert.js';
import { NULL_HANDLE } from '../../core/handle.js';
import {
  VERTEX_STRIDE_FLOATS, VERTEX_COLOR_INDEX, VERTEX_COLOR_WHITE, packVertexColor,
} from '../../render/vertex.js';
import { mat4Decompose } from '../../core/math/mat4.js';
import { parseContainer, resolveBuffers } from './glb.js';
import { readAccessorAsFloat32, readAccessorAsUint32, componentCountOf } from './accessor.js';
import {
  readSkins, normalizeWeights, checkJointIndices, jointInfluenceRadii,
} from './skin.js';
import { readMorphTargets, morphWeightsFor } from './morph.js';
import { generateTangents, unweldAndComputeFlatNormals } from './tangents.js';

// The vertex format is the renderer's contract, defined in render/vertex.js.
// Re-exported here so importer callers do not have to know that, but there is
// exactly one definition and this file is not it.
import { readAnimations } from './animation.js';

export {
  VERTEX_STRIDE_FLOATS, VERTEX_STRIDE_BYTES, VERTEX_BUFFER_LAYOUT,
  VERTEX_COLOR_INDEX, VERTEX_COLOR_WHITE, packVertexColor,
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
  uvSets: Object.freeze({ baseColor: 0, metallicRoughness: 0, normal: 0, occlusion: 0, emissive: 0 }),
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
  const meshes = (json.meshes ?? []).map((mesh, i) => {
    const name = mesh.name ?? `mesh_${i}`;
    const primitives = (mesh.primitives ?? []).map(
      (p) => buildPrimitive(json, buffers, p, `mesh "${name}"`),
    );

    // Every primitive of a mesh is deformed by ONE set of weights, so they
    // must agree on how many targets there are. The spec requires it; an
    // exporter that broke it would leave half a face morphing.
    const targetCount = primitives[0]?.morph?.targetCount ?? 0;
    for (const primitive of primitives) {
      if ((primitive.morph?.targetCount ?? 0) !== targetCount) {
        throw new Error(
          `glTF: mesh "${name}" has primitives with different morph target counts`,
        );
      }
    }

    return {
      name,
      primitives,
      /** How many targets every primitive here carries. Zero means none. */
      targetCount,
      // The mesh's own default weights. A node instancing it may override
      // them, which is why these are kept rather than resolved here.
      weights: morphWeightsFor(mesh.weights, null, targetCount, `mesh "${name}"`),
    };
  });
  const nodes = readNodes(json);
  const skins = readSkins(json, buffers);

  // Joint indices are only meaningful against a particular skin, and which
  // skin that is comes from the NODE, not the mesh. So the range check cannot
  // happen while a primitive is being built -- it happens here, once the
  // pairing is known, and it walks each (mesh, skin) pair only once however
  // many nodes share it.
  const checked = new Set();
  for (const node of nodes) {
    if (node.skin < 0 || node.mesh < 0) continue;
    if (node.skin >= skins.length) {
      throw new Error(`glTF: node "${node.name}" names skin ${node.skin}, which does not exist`);
    }
    const key = `${node.mesh}:${node.skin}`;
    if (checked.has(key)) continue;
    checked.add(key);

    const skin = skins[node.skin];
    const jointCount = skin.joints.length;
    for (const primitive of meshes[node.mesh]?.primitives ?? []) {
      if (primitive.jointIndices === null) continue;
      checkJointIndices(primitive.jointIndices, jointCount, `mesh "${meshes[node.mesh].name}"`);

      // Accumulated onto the SKIN rather than the primitive, and maxed across
      // every mesh it drives. A skin is what a runtime bound is built from,
      // and one skin driving two meshes wants a sphere big enough for both --
      // conservative, which is the only direction a cull bound may err.
      jointInfluenceRadii(
        primitive.positions, primitive.jointIndices, primitive.jointWeights,
        primitive.vertexCount, skin.inverseBind, jointCount, skin.jointRadii,
      );
    }
  }

  // Morph weights, resolved once the mesh each node instances is known. A
  // node with no override inherits the mesh's defaults -- copied, not shared,
  // because two nodes instancing one mesh animate independently.
  for (const node of nodes) {
    if (node.mesh < 0) {
      node.weights = null;
      continue;
    }
    const mesh = meshes[node.mesh];
    if (!mesh) throw new Error(`glTF: node "${node.name}" names mesh ${node.mesh}, which does not exist`);
    node.weights = mesh.targetCount === 0
      ? null
      : morphWeightsFor(mesh.weights, node.weights, mesh.targetCount, `node "${node.name}"`);
  }

  return {
    nodes,
    meshes,
    materials,
    skins,
    animations: readAnimations(json, buffers, meshes),
    roots: findRoots(json, nodes),
    // The raw document, for subsystems that need what this view deliberately
    // drops -- image decoding needs the bufferViews, and extensions will need
    // the JSON. Exposed explicitly rather than re-parsed, and never read by
    // anything in this file.
    source: { json, buffers },
  };
}

// ------------------------------------------------------------- primitives

function buildPrimitive(json, buffers, primitive, label) {
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

  // A second UV set, which materials reference per texture through `texCoord`.
  // Baked occlusion on UV1 is the standard layout out of Blender and Max, and
  // sampling it with set 0 is wrong pixels rather than a missing texture.
  let uv1s = attributes.TEXCOORD_1 !== undefined
    ? readAccessorAsFloat32(json, buffers, attributes.TEXCOORD_1) : null;

  // COLOR_0 is VEC3 or VEC4, and normalized ubyte/ushort as often as float --
  // readAccessorAsFloat32 has already undone that. VEC3 means opaque.
  let colors = null;
  if (attributes.COLOR_0 !== undefined) {
    const raw = readAccessorAsFloat32(json, buffers, attributes.COLOR_0);
    const components = componentCountOf(json.accessors[attributes.COLOR_0].type);
    colors = new Float32Array(vertexCount * 4);
    for (let v = 0; v < vertexCount; v++) {
      const s = v * components;
      colors[v * 4] = raw[s];
      colors[v * 4 + 1] = raw[s + 1];
      colors[v * 4 + 2] = raw[s + 2];
      colors[v * 4 + 3] = components === 4 ? raw[s + 3] : 1;
    }
  }

  // Skinning influences. Present together or not at all: either is meaningless
  // alone, and a mesh with one of them is malformed rather than half-skinned.
  let jointIndices = null;
  let jointWeights = null;
  if (attributes.JOINTS_0 !== undefined || attributes.WEIGHTS_0 !== undefined) {
    if (attributes.JOINTS_0 === undefined || attributes.WEIGHTS_0 === undefined) {
      throw new Error('glTF: a primitive has one of JOINTS_0 / WEIGHTS_0 without the other');
    }
    // More than four influences per vertex. Taking the first four and
    // renormalizing is the usual graceful degradation, and it silently changes
    // how the mesh deforms -- so this refuses instead and names the limit,
    // which a re-export can satisfy.
    if (attributes.JOINTS_1 !== undefined) {
      throw new Error(
        'glTF: JOINTS_1 is present; this renderer supports four influences per vertex',
      );
    }
    jointIndices = readAccessorAsUint32(json, buffers, attributes.JOINTS_0, 'VEC4');
    jointWeights = readAccessorAsFloat32(json, buffers, attributes.WEIGHTS_0);
    checkLength(jointWeights, vertexCount, 4, 'WEIGHTS_0');
    if (jointIndices.length !== vertexCount * 4) {
      throw new Error(
        `glTF: JOINTS_0 has ${jointIndices.length / 4} entries but POSITION has ${vertexCount}`,
      );
    }
    normalizeWeights(jointWeights, vertexCount);
  }

  // Morph targets. Interleaved vertex-major by morph.js, which is what lets
  // the unweld below treat them as one more per-vertex attribute.
  let morph = readMorphTargets(json, buffers, primitive.targets, vertexCount, label);

  checkLength(normals, vertexCount, 3, 'NORMAL');
  checkLength(uvs, vertexCount, 2, 'TEXCOORD_0');
  checkLength(tangents, vertexCount, 4, 'TANGENT');
  checkLength(uv1s, vertexCount, 2, 'TEXCOORD_1');

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
    // the geometry has to be de-indexed first -- and every other attribute has
    // to come along, or it would end up indexed by the old vertex ids.
    const extras = [];
    if (hadUVs) extras.push({ data: uvs, components: 2 });
    if (uv1s !== null) extras.push({ data: uv1s, components: 2 });
    if (colors !== null) extras.push({ data: colors, components: 4 });
    if (jointWeights !== null) extras.push({ data: jointWeights, components: 4 });
    // Joint INDICES ride through the float path as whole numbers. Every index
    // a skin can address is far below 2^24, where a float32 is still exact, so
    // nothing is lost and the unweld needs no integer variant.
    if (jointIndices !== null) extras.push({ data: Float32Array.from(jointIndices), components: 4 });
    // One "attribute" of targetCount * stride floats. Vertex-major layout is
    // what makes that true -- see morph.js.
    if (morph !== null) {
      extras.push({ data: morph.deltas, components: morph.targetCount * morph.stride });
    }

    const unwelded = unweldAndComputeFlatNormals(positions, indices, extras);
    positions = unwelded.positions;
    normals = unwelded.normals;
    indices = unwelded.indices;

    let next = 0;
    if (hadUVs) uvs = unwelded.extras[next++];
    if (uv1s !== null) uv1s = unwelded.extras[next++];
    if (colors !== null) colors = unwelded.extras[next++];
    if (jointWeights !== null) jointWeights = unwelded.extras[next++];
    if (jointIndices !== null) jointIndices = Uint32Array.from(unwelded.extras[next++]);
    // The extent is a property of the deltas, not of how they are indexed, so
    // duplicating vertices cannot change it.
    if (morph !== null) morph = { ...morph, deltas: unwelded.extras[next++] };

    tangents = null;
    vertexCount = positions.length / 3;
  }

  if (!hadUVs) {
    uvs = new Float32Array(vertexCount * 2);
  } else if (tangents === null) {
    tangents = generateTangents(positions, normals, uvs, indices);
  }

  // A material may name texCoord 1 on an asset that only supplies set 0, which
  // is malformed but common. Falling back to set 0 renders it the way the
  // author almost certainly meant rather than with zeros.
  if (uv1s === null) uv1s = uvs;

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
    vertices: interleave(positions, normals, uvs, tangents, uv1s, colors, vertexCount),
    // Kept alongside the interleaved copy so a caller that wants triangle-exact
    // picking can retain a fraction of the memory rather than the whole vertex.
    positions,
    indices,
    vertexCount,
    indexCount: indices.length,
    material: primitive.material ?? -1,
    bounds,
    // Null unless the mesh is skinned. Kept beside the interleaved vertices
    // rather than inside them: only a skinned pipeline binds these, so they
    // become a second vertex buffer rather than 12 bytes on every static mesh.
    jointIndices,
    jointWeights,
    // Null unless the primitive has morph targets. The deltas are a storage
    // buffer the vertex shader indexes, not a vertex attribute: a vertex reads
    // every target, so they cannot be a per-vertex binding without one
    // attribute per target.
    morph,
  };
}

/**
 * Pack every attribute into one interleaved buffer in the renderer's layout.
 *
 * `uv1s` is never null by the time this runs -- it falls back to `uvs` -- but
 * `colors` may be, and an asset without vertex colours gets opaque white,
 * which multiplies to identity in the shader. That is what lets the format
 * stay single rather than becoming a family.
 */
function interleave(positions, normals, uvs, tangents, uv1s, colors, vertexCount) {
  const out = new Float32Array(vertexCount * VERTEX_STRIDE_FLOATS);
  // The colour is unorm8x4, so it is written through a second view of the same
  // memory rather than as a float.
  const packed = new Uint32Array(out.buffer);

  for (let v = 0; v < vertexCount; v++) {
    const o = v * VERTEX_STRIDE_FLOATS;
    const p = v * 3, t = v * 2, g = v * 4;

    out[o] = positions[p]; out[o + 1] = positions[p + 1]; out[o + 2] = positions[p + 2];
    out[o + 3] = normals[p]; out[o + 4] = normals[p + 1]; out[o + 5] = normals[p + 2];
    out[o + 6] = uvs[t]; out[o + 7] = uvs[t + 1];
    out[o + 8] = tangents[g]; out[o + 9] = tangents[g + 1];
    out[o + 10] = tangents[g + 2]; out[o + 11] = tangents[g + 3];
    out[o + 12] = uv1s[t]; out[o + 13] = uv1s[t + 1];
    packed[o + VERTEX_COLOR_INDEX] = colors === null
      ? VERTEX_COLOR_WHITE
      : packVertexColor(colors[g], colors[g + 1], colors[g + 2], colors[g + 3]);
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
      // Which UV set each texture samples. glTF puts `texCoord` on the texture
      // REFERENCE, so two maps on one material can disagree -- baked occlusion
      // on set 1 beside a base colour on set 0 is the usual shape. Dropping it
      // samples the wrong pixels with no error anywhere.
      uvSets: {
        baseColor: pbr.baseColorTexture?.texCoord ?? 0,
        metallicRoughness: pbr.metallicRoughnessTexture?.texCoord ?? 0,
        normal: material.normalTexture?.texCoord ?? 0,
        occlusion: material.occlusionTexture?.texCoord ?? 0,
        emissive: material.emissiveTexture?.texCoord ?? 0,
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
      // Which skin drives this node's mesh, or -1. The node carries it rather
      // than the mesh, because one mesh can be instanced under two skeletons.
      skin: node.skin ?? -1,
      // This instance's morph weights, still raw: their length can only be
      // checked against the mesh, and which mesh that is may be a node this
      // loop has not reached. buildModel resolves them.
      weights: node.weights ?? null,
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
