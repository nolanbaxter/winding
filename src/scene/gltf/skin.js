// glTF skins: the joint list and the bind pose.
//
// A skin is two things. `joints` names the nodes that drive it, in the order
// the vertices' joint indices address -- so it is a mapping from "joint 3" to
// "node 17", and the ONLY thing that makes a vertex's integer meaningful.
// `inverseBindMatrices` takes a vertex from model space into each joint's
// local space, which is what lets the joint's current world matrix put it back
// somewhere else.
//
// The palette a shader eventually reads is jointWorld * inverseBind, per joint,
// per instance. None of that is here: this file turns JSON into flat arrays and
// stops, the same way animation.js does. The joints are node indices, and what
// maps a node index onto an entity is the instance, not the asset.

import { readAccessorAsFloat32 } from './accessor.js';

/** Column-major identity, for a skin that declares no inverse bind matrices. */
const IDENTITY = Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/**
 * Read every skin in the document.
 *
 * `inverseBindMatrices` is optional; the spec says an absent one means every
 * matrix is identity, which is a skeleton authored already in bind pose.
 */
export function readSkins(json, buffers) {
  return (json.skins ?? []).map((skin, i) => {
    const joints = Uint32Array.from(skin.joints ?? []);
    if (joints.length === 0) {
      throw new Error(`glTF: skin ${i} has no joints`);
    }

    for (const node of joints) {
      if (node >= (json.nodes?.length ?? 0)) {
        throw new Error(`glTF: skin ${i} names node ${node}, which does not exist`);
      }
    }

    let inverseBind;
    if (skin.inverseBindMatrices === undefined) {
      inverseBind = new Float32Array(joints.length * 16);
      for (let j = 0; j < joints.length; j++) inverseBind.set(IDENTITY, j * 16);
    } else {
      inverseBind = readAccessorAsFloat32(json, buffers, skin.inverseBindMatrices);
      if (inverseBind.length !== joints.length * 16) {
        throw new Error(
          `glTF: skin ${i} has ${joints.length} joints but ` +
          `${inverseBind.length / 16} inverse bind matrices`,
        );
      }
    }

    return {
      name: skin.name ?? `skin${i}`,
      joints,
      inverseBind,
      /** The node the skeleton hangs from. A hint; nothing here needs it. */
      skeleton: skin.skeleton ?? -1,
    };
  });
}

/**
 * Normalize a vertex's four influence weights so they sum to one.
 *
 * The spec requires this of the file and exporters get it wrong often enough
 * that validators check for it. An unnormalized set does not fail: it scales
 * the vertex toward or away from the origin by whatever the sum happens to be,
 * which reads as a mesh that inflates or collapses as it animates.
 *
 * A vertex with no influence at all is pinned to its first joint rather than
 * left at zero, because a zero palette sends it to the origin.
 */
export function normalizeWeights(weights, vertexCount) {
  for (let v = 0; v < vertexCount; v++) {
    const o = v * 4;
    const sum = weights[o] + weights[o + 1] + weights[o + 2] + weights[o + 3];
    if (sum > 0) {
      const inv = 1 / sum;
      weights[o] *= inv; weights[o + 1] *= inv;
      weights[o + 2] *= inv; weights[o + 3] *= inv;
    } else {
      weights[o] = 1; weights[o + 1] = 0; weights[o + 2] = 0; weights[o + 3] = 0;
    }
  }
  return weights;
}

/**
 * Every joint index a primitive uses must address a joint the skin declares.
 *
 * Out of range is not survivable downstream: the index reads past the palette,
 * which is a storage buffer, so it picks up whatever the next instance's
 * matrices are and drags the vertex somewhere arbitrary. Checked once here
 * rather than clamped per vertex on the GPU forever.
 */
export function checkJointIndices(joints, jointCount, label) {
  for (let i = 0; i < joints.length; i++) {
    if (joints[i] >= jointCount) {
      throw new Error(
        `glTF: ${label} uses joint ${joints[i]}, past the ${jointCount} the skin declares`,
      );
    }
  }
}
