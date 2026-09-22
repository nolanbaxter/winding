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
      /**
       * How far each joint's influence reaches, filled in once the meshes this
       * skin drives are known -- which is a pairing the node holds, not the
       * mesh. Zero here means no vertex uses that joint.
       */
      jointRadii: new Float32Array(joints.length),
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

/**
 * How far each joint's influence reaches, in that joint's own space.
 *
 * The number a skinned bounding box is built from at runtime. A skinned
 * vertex ends up at `sum(w_j * jointWorld_j * inverseBind_j) * v`, which is a
 * weighted average of its per-joint positions and therefore inside their
 * convex hull -- so a union of spheres, one per influencing joint, contains it.
 * The radius of joint j's sphere is the farthest any vertex it influences sits
 * from the joint's own origin, and `inverseBind_j * v` is exactly that vertex
 * in joint j's space.
 *
 * ANY nonzero weight counts. A vertex with 0.001 on a distant joint still
 * moves when that joint does, and a bound that excluded it would be a bound
 * that is sometimes wrong -- which for culling means geometry vanishing.
 *
 * Exact while joints are rigid, which is what a skeleton is. A joint with
 * scale inflates its own influence, and the sphere does not grow with it; the
 * same assumption the vertex shader makes about normals.
 */
export function jointInfluenceRadii(
  positions, jointIndices, jointWeights, vertexCount, inverseBind, jointCount, out = null,
) {
  const radii = out ?? new Float32Array(jointCount);

  for (let v = 0; v < vertexCount; v++) {
    const p = v * 3;
    const x = positions[p], y = positions[p + 1], z = positions[p + 2];
    const g = v * 4;

    for (let k = 0; k < 4; k++) {
      if (jointWeights[g + k] <= 0) continue;
      const j = jointIndices[g + k];
      const m = j * 16;

      // Column-major: the vertex through this joint's inverse bind matrix.
      const jx = inverseBind[m] * x + inverseBind[m + 4] * y + inverseBind[m + 8] * z + inverseBind[m + 12];
      const jy = inverseBind[m + 1] * x + inverseBind[m + 5] * y + inverseBind[m + 9] * z + inverseBind[m + 13];
      const jz = inverseBind[m + 2] * x + inverseBind[m + 6] * y + inverseBind[m + 10] * z + inverseBind[m + 14];

      const distance = Math.hypot(jx, jy, jz);
      if (distance > radii[j]) radii[j] = distance;
    }
  }
  return radii;
}
