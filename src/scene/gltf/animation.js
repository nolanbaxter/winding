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

import { readAccessorAsFloat32, componentCountOf } from './accessor.js';

/** glTF property name -> how many floats one keyframe holds. */
const PATH_COMPONENTS = { translation: 3, rotation: 4, scale: 3 };

/** The morph target count of the mesh a node instances, or 0. */
function targetCountOf(json, meshes, nodeIndex) {
  const mesh = json.nodes?.[nodeIndex]?.mesh;
  if (mesh === undefined) return 0;
  return meshes[mesh]?.targetCount ?? 0;
}

export function readAnimations(json, buffers, meshes = []) {
  return (json.animations ?? []).map((animation, a) => {
    const channels = [];
    let duration = 0;

    for (const channel of animation.channels ?? []) {
      const path = channel.target?.path;
      const node = channel.target?.node;
      // A channel with no node is legal and targets nothing.
      if (node === undefined) continue;

      const morph = path === 'weights';
      // A weights channel on a node whose mesh has no targets drives nothing.
      // Legal, and the same non-event as a channel with no node.
      const components = morph ? targetCountOf(json, meshes, node) : PATH_COMPONENTS[path];
      if (!(components > 0)) continue;

      const sampler = animation.samplers?.[channel.sampler];
      if (!sampler) throw new Error(`glTF: animation ${a} channel references missing sampler ${channel.sampler}`);

      const times = readAccessorAsFloat32(json, buffers, sampler.input);
      const values = readAccessorAsFloat32(json, buffers, sampler.output);
      const interpolation = sampler.interpolation ?? 'LINEAR';

      // The output accessor's own type has to agree with the property being
      // driven, or the sampling below would read whatever happens to be next in
      // the buffer and produce plausible nonsense rather than an error.
      //
      // Weights are the exception, and only in how the width is spelled: the
      // accessor is SCALAR and the width comes from the mesh, so what is
      // checked is the total below rather than the element type here.
      const declared = componentCountOf(json.accessors[sampler.output].type);
      const expected = morph ? 1 : components;
      if (declared !== expected) {
        throw new Error(
          `glTF: animation ${a} drives ${path} (${expected} components) from a ${declared}-component accessor`,
        );
      }

      // CUBICSPLINE stores an in-tangent and an out-tangent either side of each
      // value, so a keyframe occupies three times the room.
      const perKey = interpolation === 'CUBICSPLINE' ? 3 : 1;
      if (values.length !== times.length * components * perKey) {
        throw new Error(
          `glTF: animation ${a} has ${times.length} times but ${values.length} values for ${interpolation} ${path}`,
        );
      }

      channels.push({ node, path, times, values, interpolation, components });
      if (times.length > 0) duration = Math.max(duration, times[times.length - 1]);
    }

    return { name: animation.name ?? `animation_${a}`, duration, channels };
  });
}
