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
// Lights (KHR_lights_punctual) and cameras come through as data on the nodes
// that carry them, and the scene makes them live. A directional light joins
// the scene's others; whether it casts the shadow is Scene.sun's call -- the
// brightest does. KHR_materials_emissive_strength scales the emissive factor.
// No other extension is handled.
//
// A MALFORMED FILE IS REFUSED, with the rule it broke, rather than loaded into
// something quietly wrong: a NaN in a light colour spreads through bloom, an
// index into nothing drops a light without a word, a bufferView longer than
// its buffer reads whatever bytes come next as vertices.
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
import {
  readAccessorAsFloat32, readAccessorAsUint32, componentCountOf, checkAccessors,
} from './accessor.js';
import {
  readSkins, normalizeWeights, checkJointIndices, jointInfluenceRadii,
} from './skin.js';
import { readMorphTargets, morphWeightsFor } from './morph.js';
import {
  generateTangents, unweldAndComputeFlatNormals, perpendicularTo,
} from './tangents.js';

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

  const unsupported = (json.extensionsRequired ?? []).filter((e) => !SUPPORTED_EXTENSIONS.has(e));
  if (unsupported.length > 0) {
    // Loading anyway would produce geometry that is wrong in a way nothing
    // reports -- Draco-compressed buffers read as noise, quantized meshes come
    // out the wrong size. Say so.
    throw new Error(`glTF: requires unsupported extensions: ${unsupported.join(', ')}`);
  }

  // Before any resource is fetched or any array allocated. `maxBytes` is the
  // largest buffer the device can make -- the engine passes it -- and bounds
  // what the file may ask for; see checkAccessors.
  const maxBytes = options.maxBytes ?? Infinity;
  checkAccessors(json, maxBytes);

  const buffers = await resolveBuffers(json, binary, options);
  return buildModel(json, buffers, maxBytes);
}

function buildModel(json, buffers, maxBytes = Infinity) {
  const materials = readMaterials(json);
  const meshes = (json.meshes ?? []).map((mesh, i) => {
    const name = mesh.name ?? `mesh_${i}`;
    const primitives = (mesh.primitives ?? []).map(
      (p) => buildPrimitive(json, buffers, p, `mesh "${name}"`, maxBytes),
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
  const lights = readLights(json);
  const cameras = readCameras(json);
  for (const node of nodes) {
    if (node.light !== -1 && !isIndex(node.light, lights.length)) {
      throw new Error(`glTF: node "${node.name}" names light ${JSON.stringify(node.light)}, which does not exist`);
    }
    if (node.camera !== -1 && !isIndex(node.camera, cameras.length)) {
      throw new Error(`glTF: node "${node.name}" names camera ${JSON.stringify(node.camera)}, which does not exist`);
    }
  }
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
    lights,
    cameras,
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

function buildPrimitive(json, buffers, primitive, label, maxBytes = Infinity) {
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

  if (primitive.material !== undefined && !isIndex(primitive.material, json.materials?.length ?? 0)) {
    throw new Error(`glTF: ${label} names material ${primitive.material}, which does not exist`);
  }

  let indices = primitive.indices !== undefined
    ? readAccessorAsUint32(json, buffers, primitive.indices)
    : sequentialIndices(vertexCount);

  // A triangle list is three indices a triangle. A leftover one or two used to
  // pass, and the tangent pass then wrote NaN into the vertex they named.
  if (indices.length % 3 !== 0) {
    throw new Error(`glTF: ${label} has ${indices.length} indices, which is not a whole number of triangles`);
  }

  // The vertex buffer this becomes, sized now, before flat shading de-indexes
  // it -- one vertex per index -- and before interleaving widens every vertex.
  // Each step multiplies what the accessors alone could ask for, and a mesh
  // past the device's buffer limit cannot be drawn at any cost.
  const drawnVertices = attributes.NORMAL === undefined ? indices.length : vertexCount;
  const vertexBytes = drawnVertices * VERTEX_STRIDE_FLOATS * 4;
  if (vertexBytes > maxBytes) {
    throw new RangeError(`glTF: ${label} needs a ${vertexBytes}-byte vertex buffer, past the ${maxBytes} this device can hold`);
  }

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
    // The one attribute checkLength below did not cover: reads past the end
    // came back undefined and packed as transparent black.
    checkLength(raw, vertexCount, components, 'COLOR_0');
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
    // glTF 3.7.3.3: joints are unsigned byte or unsigned short, and the skin
    // buffer packs them as uint16x4 on that promise. An unsigned int accessor
    // is not valid glTF, and it would have been narrowed there without a word.
    if (json.accessors[attributes.JOINTS_0].componentType === 5125) {
      throw new Error('glTF: JOINTS_0 stores UNSIGNED_INT; joints are unsigned byte or unsigned short');
    }
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
  let morph = readMorphTargets(json, buffers, primitive.targets, vertexCount, label, {
    drawnVertices, maxBytes,
  });

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
    // No UVs means no tangent frame exists to compute. Any direction will do,
    // because without UVs nothing samples a normal map -- but it has to lie IN
    // THE SURFACE PLANE, and that is the part a constant gets wrong.
    //
    // This filled every vertex with (1,0,0) and called it "a valid unit vector
    // rather than zeros so a shader that samples it cannot produce NaN". It is
    // a unit vector, and it produces NaN anyway: on any face whose normal
    // points along X it is PARALLEL to the normal, so the shader's
    // cross(N, T) is the zero vector and normalize() of that is NaN. A NaN
    // fragment is then spread across a wide blocky area by the bloom chain and
    // tonemapped to black, so the symptom is nothing like the cause.
    //
    // An untextured cube hits it on two faces out of six, which makes it about
    // the most reachable shape there is. Nothing here caught it because every
    // asset in the repository has UVs.
    //
    // perpendicularTo is the same helper generateTangents already uses when
    // its own accumulation degenerates -- it crosses with the least-aligned
    // axis, which is well conditioned for every normal rather than for most.
    tangents = new Float32Array(vertexCount * 4);
    for (let v = 0; v < vertexCount; v++) {
      const n = v * 3;
      const [tx, ty, tz] = perpendicularTo(normals[n], normals[n + 1], normals[n + 2]);
      tangents[v * 4] = tx;
      tangents[v * 4 + 1] = ty;
      tangents[v * 4 + 2] = tz;
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

/**
 * Numbers the file writes in its JSON, checked where they enter. JSON has no
 * NaN, but 1e999 parses to Infinity and a string copies into a Float32Array
 * as NaN -- and either one reaching a shader is spread by bloom until other
 * models in the frame go black too.
 */
function finiteNumbers(values, length, what) {
  if (!Array.isArray(values) || values.length !== length || !values.every(Number.isFinite)) {
    throw new Error(`glTF: ${what} must be ${length} finite numbers, got [${values}]`);
  }
  return values;
}

function finiteNumber(value, fallback, what) {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value)) throw new Error(`glTF: ${what} must be a finite number, got ${value}`);
  return value;
}

function readMaterials(json) {
  return (json.materials ?? []).map((material, i) => {
    const pbr = material.pbrMetallicRoughness ?? {};
    const of = (field) => `material ${i} ${field}`;
    return {
      name: material.name ?? `material_${i}`,
      baseColorFactor: Float32Array.from(
        pbr.baseColorFactor === undefined ? [1, 1, 1, 1] : finiteNumbers(pbr.baseColorFactor, 4, of('baseColorFactor')),
      ),
      metallic: finiteNumber(pbr.metallicFactor, 1, of('metallicFactor')),
      roughness: finiteNumber(pbr.roughnessFactor, 1, of('roughnessFactor')),
      // KHR_materials_emissive_strength lifts the factor past 1, which the
      // core spec clamps it to. Blender writes it for any emission strength
      // above 1, so ignoring it dims every such glow with nothing reported.
      // Applied here, once: the shader already takes the factor as a float.
      emissive: Float32Array.from(
        material.emissiveFactor === undefined ? [0, 0, 0] : finiteNumbers(material.emissiveFactor, 3, of('emissiveFactor')),
      ).map(
        (v) => v * emissiveStrengthOf(material, i),
      ),
      alphaMode: material.alphaMode ?? 'OPAQUE',
      alphaCutoff: finiteNumber(material.alphaCutoff, 0.5, of('alphaCutoff')),
      doubleSided: material.doubleSided === true,
      normalScale: finiteNumber(material.normalTexture?.scale, 1, of('normalTexture.scale')),
      // How much the occlusion map is allowed to darken ambient light. Lives on
      // the texture reference in glTF, not on the material.
      occlusionStrength: finiteNumber(material.occlusionTexture?.strength, 1, of('occlusionTexture.strength')),
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
      const m = Float32Array.from(finiteNumbers(node.matrix, 16, `node ${i} matrix`));
      if (!mat4Decompose(position, rotation, scale, m)) {
        throw new Error(`glTF: node ${i} has a degenerate matrix that cannot be decomposed`);
      }
    } else {
      if (node.translation) position.set(finiteNumbers(node.translation, 3, `node ${i} translation`));
      if (node.rotation) rotation.set(finiteNumbers(node.rotation, 4, `node ${i} rotation`));
      if (node.scale) scale.set(finiteNumbers(node.scale, 3, `node ${i} scale`));
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
      // Indices into model.lights / model.cameras, or -1. A light is data on
      // a node in glTF, exactly as it is in the scene: the node is where it
      // is and which way it points.
      light: node.extensions?.KHR_lights_punctual?.light ?? -1,
      camera: node.camera ?? -1,
    };
  });
}

// ------------------------------------------------------- lights and cameras

/** Extensions this importer understands well enough to accept as required. */
const SUPPORTED_EXTENSIONS = new Set(['KHR_lights_punctual', 'KHR_materials_emissive_strength']);

/**
 * KHR_lights_punctual, in the terms scene.addLight uses.
 *
 * Almost nothing to translate, because the engine's lights were built to this
 * spec: intensity is candela for both, a spot's angles are half-angles from its
 * axis for both, it shines down its node's -Z for both, and the falloff the
 * spec recommends -- inverse-square times clamp(1 - (d/range)^4)^2 -- is the
 * shader's, term for term. So `range` IS the radius.
 *
 * A directional light has no position and no reach, just a direction (its
 * node's -Z) and a colour at an intensity (lux, as the sun's is).
 *
 * Every rule the extension states is checked here, because each one broken
 * used to become something wrong on screen: a two-number colour put NaN in
 * the light buffers, an inner cone past the outer one a hard edge, a negative
 * range a light that never shone.
 */
function readLights(json) {
  const block = json.extensions?.KHR_lights_punctual;
  if (block === undefined) return [];
  if (!Array.isArray(block.lights)) {
    throw new Error('glTF: KHR_lights_punctual has no lights array');
  }
  return block.lights.map((light, i) => {
    const where = `glTF: light ${i}`;
    if (light === null || typeof light !== 'object') throw new Error(`${where} is not an object`);
    if (light.type !== 'point' && light.type !== 'spot' && light.type !== 'directional') {
      throw new Error(`${where} has type ${JSON.stringify(light.type)}; KHR_lights_punctual defines point, spot and directional`);
    }
    const color = light.color ?? [1, 1, 1];
    if (!Array.isArray(color) || color.length !== 3 || !color.every((c) => Number.isFinite(c) && c >= 0)) {
      throw new Error(`${where} color must be three non-negative numbers, got ${JSON.stringify(color)}`);
    }
    const intensity = light.intensity ?? 1;
    if (!(Number.isFinite(intensity) && intensity >= 0)) {
      throw new Error(`${where} intensity must be a non-negative number, got ${intensity}`);
    }
    if (light.type === 'directional') {
      return { name: light.name ?? '', type: 'directional', color, intensity };
    }
    if (light.range !== undefined && !(Number.isFinite(light.range) && light.range > 0)) {
      throw new Error(`${where} range must be a positive number, got ${light.range}`);
    }
    const innerAngle = light.spot?.innerConeAngle ?? 0;
    const outerAngle = light.spot?.outerConeAngle ?? Math.PI / 4;
    if (light.type === 'spot' && !(innerAngle >= 0 && innerAngle < outerAngle && outerAngle <= Math.PI / 2)) {
      throw new Error(`${where} cone needs 0 <= inner < outer <= pi/2, got inner ${innerAngle} outer ${outerAngle}`);
    }
    return {
      name: light.name ?? '',
      type: light.type,
      color,
      intensity,
      radius: light.range ?? unboundedLightRadius(intensity, color),
      innerAngle,
      outerAngle,
    };
  });
}

/** A whole number that indexes into an array of `count`. */
function isIndex(value, count) {
  return Number.isInteger(value) && value >= 0 && value < count;
}

/** KHR_materials_emissive_strength's factor for a material, checked. */
function emissiveStrengthOf(material, i) {
  const strength = material.extensions?.KHR_materials_emissive_strength?.emissiveStrength ?? 1;
  if (!(Number.isFinite(strength) && strength >= 0)) {
    throw new Error(`glTF: material ${i} emissiveStrength must be a non-negative number, got ${strength}`);
  }
  return strength;
}

/**
 * How far a light with no `range` reaches.
 *
 * The spec says an absent range means infinite, and clustering cannot use one
 * -- a light that reaches everywhere is tested against every cell. So the
 * radius is where it stops mattering: a white surface facing the light gets
 * irradiance I / d^2 and reflects radiance I / (pi d^2), and past the distance
 * where that falls below 1/256 it is under one 8-bit step at exposure 1.
 * Solving for d gives sqrt(256 I / pi). Derived from the display, not picked.
 */
export function unboundedLightRadius(intensity, color) {
  const brightest = intensity * Math.max(color[0], color[1], color[2]);
  return brightest > 0
    ? TOE_STEEPENING * Math.sqrt((WINDOW_WORST * brightest) / (Math.PI * HALF_STEP_RADIANCE))
    : 0;
}

/**
 * The slopes below are the curves' slopes AT black. Just above it ACES
 * steepens -- its x^2 term is seventeen times its linear one there -- so at
 * the radius those slopes give, the worst pixel still moves 0.536 of a step,
 * not 0.5. Bisection against the real curves needs 1.0335 more reach, and
 * needs it at every intensity: a pixel depends only on I / d^2, so the
 * intensity scales out and one factor serves them all. Rounded up to 1.034,
 * because 1.0335 is the bisection's own bound and lands exactly on the half
 * step. The gltf test re-derives it against the curves, so a change to the
 * tonemap fails there.
 */
const TOE_STEEPENING = 1.034;

/**
 * The smallest radiance the display can show: half an 8-bit step, traced back
 * through the output pipeline at exposure 1. Near black the ACES curve in
 * render/post.js has slope b/e = 0.03/0.14 and the sRGB encode has slope
 * 12.92 (core/color.js), so half a step out, (0.5 / 255), is this much in.
 *
 * This used to be taken as 1/256 of radiance directly, as if the output were
 * linear. It is not: sRGB brightens the darks by 12.92x before ACES dims them
 * by 0.21x, so a light cut at that radiance still showed as byte 3.6.
 */
const HALF_STEP_RADIANCE = (0.5 / 255) / (12.92 * (0.03 / 0.14));

/**
 * How far the window pulls a light below plain inverse-square, at worst,
 * relative to the radiance at the radius: max over x = d/r of
 * (1 - (1 - x^4)^2) / x^2 = 2x^2 - x^6, reached at x^4 = 2/3, which is
 * (4/3) sqrt(2/3). A radius chosen so THIS stays under half a step makes the
 * cutoff invisible everywhere inside it, not just at the edge.
 */
const WINDOW_WORST = (4 / 3) * Math.sqrt(2 / 3);

/**
 * glTF cameras, in the terms the Camera constructor takes.
 *
 * Perspective drops two fields on purpose. `zfar`: the engine's perspective
 * projection has no far plane, so geometry past it is drawn rather than cut.
 * `aspectRatio`: the canvas decides the aspect, as it does for every camera
 * -- the spec lets a runtime do that, and stretching to a ratio the viewport
 * does not have would be the other option.
 *
 * Orthographic keeps `ymag` (half the view height) because an orthographic
 * Camera derives its height from distance; the scene places it at the distance
 * that shows exactly that.
 */
function readCameras(json) {
  return (json.cameras ?? []).map((camera, i) => {
    const name = camera.name ?? `camera_${i}`;
    const where = `glTF: camera ${i}`;
    if (camera.type === 'orthographic') {
      const o = camera.orthographic ?? {};
      // All four are required, and a box that ends before it starts has
      // nothing in it -- it used to load and then fail on every frame.
      if (!(Number.isFinite(o.xmag) && o.xmag !== 0 && Number.isFinite(o.ymag) && o.ymag !== 0)) {
        throw new Error(`${where} xmag and ymag must be non-zero numbers`);
      }
      if (!(Number.isFinite(o.znear) && o.znear >= 0 && Number.isFinite(o.zfar) && o.zfar > o.znear)) {
        throw new Error(`${where} needs 0 <= znear < zfar, got znear ${o.znear} zfar ${o.zfar}`);
      }
      const far = o.zfar;
      return {
        name,
        orthographic: true,
        // A zero znear is legal for orthographic glTF, and harmless to linear
        // depth -- but the light clusters are sliced logarithmically from the
        // near plane, and log(0) has no slices. A sliver of the box is free.
        near: o.znear > 0 ? o.znear : far * 1e-4,
        far,
        halfHeight: Math.abs(o.ymag),
      };
    }
    if (camera.type !== 'perspective') {
      throw new Error(`${where} has type ${JSON.stringify(camera.type)}; glTF defines perspective and orthographic`);
    }
    const p = camera.perspective ?? {};
    if (!(Number.isFinite(p.yfov) && p.yfov > 0 && p.yfov < Math.PI)) {
      throw new Error(`${where} yfov must be between 0 and pi, got ${p.yfov}`);
    }
    if (!(Number.isFinite(p.znear) && p.znear > 0)) {
      throw new Error(`${where} znear must be a positive number, got ${p.znear}`);
    }
    return {
      name,
      orthographic: false,
      fovY: p.yfov,
      near: p.znear,
    };
  });
}

function findRoots(json, nodes) {
  const index = json.scene ?? 0;
  const scene = json.scenes?.[index];

  // A DECLARED scene is authoritative, including when it is empty. `scenes:
  // [{}]` is legal and means a document whose contents are all referenced
  // rather than instantiated -- a library of meshes, which is a real way to
  // ship one. Falling through to orphan detection there loads every node in
  // the file, which is the opposite of what the document says.
  if (scene !== undefined) return (scene.nodes ?? []).slice();

  // An explicit `scene` that names nothing is malformed, and recovering from
  // it would mean loading something other than what was asked for.
  if (json.scene !== undefined) {
    throw new Error(`glTF: scene ${json.scene} is the default scene but does not exist`);
  }

  // No scene declared at all: the spec leaves the choice to the runtime, so
  // anything nothing else claims as a child is a root.
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
