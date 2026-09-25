// glTF animation channels.
//
// An animation is a set of CHANNELS, each pointing at one node and one of its
// properties, driven by a SAMPLER: a list of times and a matching list of
// values. This file turns the JSON into flat typed arrays and stops; evaluating
// them is animation.js, which knows nothing about glTF.
//
// Three of the four properties are TRS and have a fixed width. The fourth,
// `weights`, does not: it drives every morph target of the node's mesh at once,
// so one keyframe is as many floats as that mesh has targets. The accessor is
// SCALAR either way -- the file stores a flat run and leaves the reader to know
// where one keyframe ends. Which mesh a node instances is the only thing that
// says, so the node table has to be in hand before a weights channel can be
// read at all.

import { MATERIAL_TEXTURES } from './images.js';
import { readAccessorAsFloat32, componentCountOf } from './accessor.js';

/** glTF property name -> how many floats one keyframe holds. */
const PATH_COMPONENTS = { translation: 3, rotation: 4, scale: 3 };

/** The morph target count of the mesh a node instances, or 0. */
function targetCountOf(json, meshes, nodeIndex) {
  const mesh = json.nodes?.[nodeIndex]?.mesh;
  if (mesh === undefined) return 0;
  return meshes[mesh]?.targetCount ?? 0;
}

const INTERPOLATIONS = new Set(['LINEAR', 'STEP', 'CUBICSPLINE']);

export function readAnimations(json, buffers, meshes = []) {
  // One read per accessor, however many channels share it. Keyframes are
  // never written after import, so sharing the arrays is safe -- and without
  // it ten thousand channels naming one sampler read it ten thousand times.
  const reads = new Map();
  const read = (index) => {
    let values = reads.get(index);
    if (values === undefined) {
      values = readAccessorAsFloat32(json, buffers, index);
      reads.set(index, values);
    }
    return values;
  };

  return (json.animations ?? []).map((animation, a) => {
    const channels = [];
    // Pointers this engine has nothing to drive with, kept so a caller can see
    // what a clip does that will not show.
    const ignored = [];
    let duration = 0;

    for (const channel of animation.channels ?? []) {
      let path = channel.target?.path;
      let node = channel.target?.node;
      // KHR_animation_pointer: the target is a JSON pointer, and one naming a
      // node's TRS or weights becomes exactly the channel the core form would
      // have been. Anything else drives a light, a camera or a material.
      let target = null;
      if (path === 'pointer') {
        const pointer = channel.target.extensions?.KHR_animation_pointer?.pointer;
        target = resolvePointer(json, meshes, pointer, a);
        if (typeof target === 'string') {
          ignored.push(target);
          continue;
        }
        if (target.node !== undefined) ({ node, path } = target);
      }
      if (target === null || target.node !== undefined) {
        // A channel with no node is legal and targets nothing. One naming a
        // node that does not exist is not: it used to be kept, and the player
        // skipped it forever without a word.
        if (node === undefined) continue;
        if (!(Number.isInteger(node) && node >= 0 && node < (json.nodes?.length ?? 0))) {
          throw new Error(`glTF: animation ${a} targets node ${node}, which does not exist`);
        }
      }

      const morph = path === 'weights';
      // A weights channel on a node whose mesh has no targets drives nothing.
      // Legal, and the same non-event as a channel with no node. One weight of
      // the list, which only a pointer can name, is one float.
      const components = target?.components
        ?? (morph ? targetCountOf(json, meshes, node) : PATH_COMPONENTS[path]);
      if (!(components > 0)) continue;

      const sampler = animation.samplers?.[channel.sampler];
      if (!sampler) throw new Error(`glTF: animation ${a} channel references missing sampler ${channel.sampler}`);

      const times = read(sampler.input);
      const values = read(sampler.output);
      const interpolation = sampler.interpolation ?? 'LINEAR';
      if (!INTERPOLATIONS.has(interpolation)) {
        throw new Error(`glTF: animation ${a} uses interpolation ${JSON.stringify(interpolation)}; glTF defines LINEAR, STEP and CUBICSPLINE`);
      }
      // Keyframe times must strictly increase. The binary search that samples
      // them assumes it, and the clip's length is read off the last one.
      for (let k = 1; k < times.length; k++) {
        if (!(times[k] > times[k - 1])) {
          throw new Error(`glTF: animation ${a} keyframe times do not strictly increase (${times[k - 1]} then ${times[k]})`);
        }
      }

      // The output accessor's own type has to agree with the property being
      // driven, or the sampling below would read whatever happens to be next in
      // the buffer and produce plausible nonsense rather than an error.
      //
      // Weights are the exception, and only in how the width is spelled: the
      // accessor is SCALAR and the width comes from the mesh, so what is
      // checked is the total below rather than the element type here.
      const what = target?.pointer ?? path;
      const declared = componentCountOf(json.accessors[sampler.output].type);
      const expected = morph ? 1 : components;
      if (declared !== expected) {
        throw new Error(
          `glTF: animation ${a} drives ${what} (${expected} components) from a ${declared}-component accessor`,
        );
      }

      // CUBICSPLINE stores an in-tangent and an out-tangent either side of each
      // value, so a keyframe occupies three times the room.
      const perKey = interpolation === 'CUBICSPLINE' ? 3 : 1;
      if (values.length !== times.length * components * perKey) {
        throw new Error(
          `glTF: animation ${a} has ${times.length} times but ${values.length} values for ${interpolation} ${what}`,
        );
      }

      if (target?.kind !== undefined) {
        const { kind, index, field } = target;
        channels.push({
          node: -1, path: 'property', kind, index, field, key: `${kind}/${index}/${field}`, times, values, interpolation, components,
        });
      } else {
        // `offset` is where in the node's weights the first float goes: 0 for
        // the whole list, k for the one weight /nodes/i/weights/k names.
        channels.push({ node, path, offset: target?.offset ?? 0, times, values, interpolation, components });
      }
      if (times.length > 0) duration = Math.max(duration, times[times.length - 1]);
    }

    return { name: animation.name ?? `animation_${a}`, duration, channels, ignored };
  });
}

// ------------------------------------------------------ KHR_animation_pointer

/** Material pointer suffix -> the importer's field and its width. */
const MATERIAL_FIELDS = {
  'pbrMetallicRoughness/baseColorFactor': ['baseColorFactor', 4],
  'pbrMetallicRoughness/metallicFactor': ['metallic', 1],
  'pbrMetallicRoughness/roughnessFactor': ['roughness', 1],
  emissiveFactor: ['emissiveFactor', 3],
  'extensions/KHR_materials_emissive_strength/emissiveStrength': ['emissiveStrength', 1],
  alphaCutoff: ['alphaCutoff', 1],
  'normalTexture/scale': ['normalScale', 1],
  'occlusionTexture/strength': ['occlusionStrength', 1],
  'extensions/KHR_materials_ior/ior': ['ior', 1],
  'extensions/KHR_materials_specular/specularFactor': ['specular', 1],
  'extensions/KHR_materials_specular/specularColorFactor': ['specularColor', 3],
  'extensions/KHR_materials_clearcoat/clearcoatFactor': ['clearcoat', 1],
  'extensions/KHR_materials_clearcoat/clearcoatRoughnessFactor': ['clearcoatRoughness', 1],
  'extensions/KHR_materials_clearcoat/clearcoatNormalTexture/scale': ['clearcoatNormalScale', 1],
  'extensions/KHR_materials_sheen/sheenColorFactor': ['sheenColor', 3],
  'extensions/KHR_materials_sheen/sheenRoughnessFactor': ['sheenRoughness', 1],
  'extensions/KHR_materials_anisotropy/anisotropyStrength': ['anisotropyStrength', 1],
  'extensions/KHR_materials_anisotropy/anisotropyRotation': ['anisotropyRotation', 1],
  'extensions/KHR_materials_iridescence/iridescenceFactor': ['iridescence', 1],
  'extensions/KHR_materials_iridescence/iridescenceIor': ['iridescenceIor', 1],
  'extensions/KHR_materials_iridescence/iridescenceThicknessMinimum': ['iridescenceThicknessMinimum', 1],
  'extensions/KHR_materials_iridescence/iridescenceThicknessMaximum': ['iridescenceThicknessMaximum', 1],
  'extensions/KHR_materials_transmission/transmissionFactor': ['transmission', 1],
  'extensions/KHR_materials_volume/thicknessFactor': ['thickness', 1],
  'extensions/KHR_materials_volume/attenuationDistance': ['attenuationDistance', 1],
  'extensions/KHR_materials_volume/attenuationColor': ['attenuationColor', 3],
};
// KHR_texture_transform on each texture reference, in the slot order the
// importer keeps them: 'uv<slot>.<part>'.
MATERIAL_TEXTURES.forEach(({ path }, slot) => {
  const base = `${path}/extensions/KHR_texture_transform`;
  MATERIAL_FIELDS[`${base}/offset`] = [`uv${slot}.offset`, 2];
  MATERIAL_FIELDS[`${base}/rotation`] = [`uv${slot}.rotation`, 1];
  MATERIAL_FIELDS[`${base}/scale`] = [`uv${slot}.scale`, 2];
});

/** Light pointer suffix -> field, width, and which light types have it. */
const LIGHT_FIELDS = {
  color: ['color', 3, 'point spot directional'],
  intensity: ['intensity', 1, 'point spot directional'],
  range: ['range', 1, 'point spot'],
  'spot/innerConeAngle': ['innerAngle', 1, 'spot'],
  'spot/outerConeAngle': ['outerAngle', 1, 'spot'],
};

/**
 * Camera pointer suffix -> field, width, and which camera type has it. Not
 * here: perspective zfar and aspectRatio, and orthographic xmag, which the
 * importer drops too -- the engine's perspective has no far plane, and the
 * canvas decides the aspect (see readCameras in parse.js).
 */
const CAMERA_FIELDS = {
  'perspective/yfov': ['fovY', 1, 'perspective'],
  'perspective/znear': ['near', 1, 'perspective'],
  'orthographic/ymag': ['halfHeight', 1, 'orthographic'],
  'orthographic/znear': ['near', 1, 'orthographic'],
  'orthographic/zfar': ['far', 1, 'orthographic'],
};

/**
 * What a KHR_animation_pointer channel drives, or the pointer itself (a
 * string) when it is nothing this engine renders.
 *
 * A pointer into an array that does not have that element is an error, as a
 * node index past the end is for the core form: the file is wrong, and a clip
 * that quietly did less than it says is the thing the rest of the importer
 * refuses to hand over. A pointer to a property the engine has no use for --
 * a texture transform, an extension it does not implement, a field of a light
 * type that does not have it -- is a well-formed file asking for something
 * that cannot show, so the channel is set aside and listed on the clip.
 */
function resolvePointer(json, meshes, pointer, a) {
  if (typeof pointer !== 'string') {
    throw new Error(`glTF: animation ${a} has a pointer channel with no pointer`);
  }
  const at = (collection, index, count) => {
    const i = Number(index);
    if (!(i < count)) throw new Error(`glTF: animation ${a} points at ${pointer}, but there is no ${collection} ${i}`);
    return i;
  };

  let m = /^\/nodes\/(\d+)\/(translation|rotation|scale|weights)(?:\/(\d+))?$/.exec(pointer);
  if (m && (m[3] === undefined || m[2] === 'weights')) {
    const node = at('node', m[1], json.nodes?.length ?? 0);
    if (m[3] === undefined) return { node, path: m[2], pointer };
    const offset = Number(m[3]);
    if (!(offset < targetCountOf(json, meshes, node))) {
      throw new Error(`glTF: animation ${a} points at ${pointer}, but that node has no morph target ${offset}`);
    }
    return { node, path: 'weights', offset, components: 1, pointer };
  }

  m = /^\/materials\/(\d+)\/(.+)$/.exec(pointer);
  if (m && MATERIAL_FIELDS[m[2]]) {
    const [field, components] = MATERIAL_FIELDS[m[2]];
    return { kind: 'material', index: at('material', m[1], json.materials?.length ?? 0), field, components, pointer };
  }

  m = /^\/extensions\/KHR_lights_punctual\/lights\/(\d+)\/(.+)$/.exec(pointer);
  if (m && LIGHT_FIELDS[m[2]]) {
    const lights = json.extensions?.KHR_lights_punctual?.lights ?? [];
    const index = at('light', m[1], lights.length);
    const [field, components, types] = LIGHT_FIELDS[m[2]];
    if (!types.split(' ').includes(lights[index].type)) return pointer;
    return { kind: 'light', index, field, components, pointer };
  }

  m = /^\/cameras\/(\d+)\/(.+)$/.exec(pointer);
  if (m && CAMERA_FIELDS[m[2]]) {
    const index = at('camera', m[1], json.cameras?.length ?? 0);
    const [field, components, type] = CAMERA_FIELDS[m[2]];
    if (json.cameras[index].type !== type) return pointer;
    return { kind: 'camera', index, field, components, pointer };
  }

  return pointer;
}
