// glTF animation channels.
//
// An animation is a set of CHANNELS, each pointing at one node and one of its
// TRS properties, driven by a SAMPLER: a list of times and a matching list of
// values. Nothing here is per-vertex -- that is skinning, which is separate and
// not implemented. This file turns the JSON into flat typed arrays and stops;
// evaluating them is animation.js, which knows nothing about glTF.
//
// Channels targeting `weights` are dropped, because morph targets are not
// imported and a weight channel with nothing to drive is not an error worth
// failing a load over.

import { readAccessorAsFloat32, componentCountOf } from './accessor.js';

/** glTF property name -> how many floats one keyframe holds. */
const PATH_COMPONENTS = { translation: 3, rotation: 4, scale: 3 };

export function readAnimations(json, buffers) {
  return (json.animations ?? []).map((animation, a) => {
    const channels = [];
    let duration = 0;

    for (const channel of animation.channels ?? []) {
      const path = channel.target?.path;
      const node = channel.target?.node;
      // A channel with no node is legal and targets nothing.
      if (node === undefined || !(path in PATH_COMPONENTS)) continue;

      const sampler = animation.samplers?.[channel.sampler];
      if (!sampler) throw new Error(`glTF: animation ${a} channel references missing sampler ${channel.sampler}`);

      const times = readAccessorAsFloat32(json, buffers, sampler.input);
      const values = readAccessorAsFloat32(json, buffers, sampler.output);
      const interpolation = sampler.interpolation ?? 'LINEAR';
      const components = PATH_COMPONENTS[path];

      // The output accessor's own type has to agree with the property being
      // driven, or the sampling below would read whatever happens to be next in
      // the buffer and produce plausible nonsense rather than an error.
      const declared = componentCountOf(json.accessors[sampler.output].type);
      if (declared !== components) {
        throw new Error(
          `glTF: animation ${a} drives ${path} (${components} components) from a ${declared}-component accessor`,
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
