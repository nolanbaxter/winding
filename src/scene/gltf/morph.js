// glTF morph targets: per-vertex deltas and the weights that mix them.
//
// A morph target is a sparse-in-spirit copy of the mesh expressed as a
// DIFFERENCE from it. The deformed vertex is
//
//     v' = v + sum_t( weight_t * delta_t(v) )
//
// which is the whole feature. Everything below is about the layout that sum
// reads from, because the sum itself is three lines of shader.
//
// TWO THINGS THE FILE FORMAT LEAVES OPEN, CLOSED HERE.
//
// First, targets need not agree on which attributes they carry: one may move
// positions only, the next positions and normals. A shader that had to ask per
// target would need a per-target descriptor and a branch inside its hot loop.
// So the stride is decided ONCE for the primitive -- the widest any of its
// targets needs -- and the narrower ones are padded with zeros, which are the
// identity of addition and therefore cost nothing but the bytes.
//
// Second, glTF says nothing about ordering. The layout here is VERTEX-MAJOR:
// every target's delta for vertex 0, then every target's delta for vertex 1.
// Target-major would be the obvious transcription of the file and is the wrong
// one twice over. A vertex shader touches ONE vertex and every target, so
// vertex-major puts everything it reads in one cache line and target-major
// strides across the whole array. And the flat-shading unweld reorders
// vertices, which vertex-major makes a copy of `targetCount * stride` floats
// per vertex -- the same operation every other attribute already gets -- where
// target-major would need a transpose.
//
// WHY THE STRIDE IS A FIELD AND UV1 IS NOT. render/vertex.js makes the
// opposite call: every vertex carries a second UV set whether it has one or
// not. The reason that is right there and wrong here is what the alternative
// costs. There, a narrower format means a second pipeline; here it means one
// integer and two comparisons. And the array this multiplies is the largest in
// the asset -- positions-only targets are the common case, and always storing
// normals and tangents would triple a face rig for nothing.

import { readAccessorAsFloat32, componentCountOf } from './accessor.js';
import { hypot3 } from '../../core/math/vec3.js';

/** Floats per vertex per target, by the widest attribute any target carries. */
export const MORPH_STRIDE_POSITION = 3;
export const MORPH_STRIDE_NORMAL = 6;
export const MORPH_STRIDE_TANGENT = 9;

/** The attributes a target may deform, in the order they occupy the stride. */
const TARGET_ATTRIBUTES = [
  { name: 'POSITION', type: 'VEC3', components: 3, offset: 0, stride: MORPH_STRIDE_POSITION },
  { name: 'NORMAL', type: 'VEC3', components: 3, offset: 3, stride: MORPH_STRIDE_NORMAL },
  // TANGENT deltas are VEC3, not VEC4: the handedness in w is a property of
  // the UV layout, which morphing does not change. The spec is explicit.
  { name: 'TANGENT', type: 'VEC3', components: 3, offset: 6, stride: MORPH_STRIDE_TANGENT },
];

/**
 * Read one primitive's morph targets into a single interleaved array.
 *
 * Returns null for a primitive with no targets, which is most of them.
 *
 * `extent[t]` is the farthest any vertex travels under target t at weight 1 --
 * the one number a conservative bound needs, computed here because this is the
 * only place the deltas are ever walked. Without it every frame would have to
 * re-derive it from the vertex data, which is the per-vertex CPU cost this
 * whole design exists to avoid.
 */
export function readMorphTargets(
  json, buffers, targets, vertexCount, label,
  { drawnVertices = vertexCount, maxBytes = Infinity } = {},
) {
  if (!Array.isArray(targets) || targets.length === 0) return null;

  const targetCount = targets.length;

  // The widest attribute any target carries decides the stride for all of them.
  let stride = 0;
  for (const target of targets) {
    for (const attribute of TARGET_ATTRIBUTES) {
      if (target[attribute.name] !== undefined && attribute.stride > stride) {
        stride = attribute.stride;
      }
    }
  }
  if (stride === 0) {
    // Every target is empty, or carries only attributes nothing can deform.
    // Drawing it as unmorphed geometry would be correct; saying so is better,
    // because it is a file that does not mean what it appears to.
    throw new Error(
      `glTF: ${label} has ${targetCount} morph targets, none of which deform ` +
      'POSITION, NORMAL or TANGENT',
    );
  }

  // Sized as drawn -- flat shading duplicates the deltas with the vertices --
  // before anything is allocated. A thousand targets all naming one accessor
  // is valid glTF, and it asked for gigabytes this way before any check ran.
  const bytes = drawnVertices * targetCount * stride * 4;
  if (bytes > maxBytes) {
    throw new RangeError(`glTF: ${label} has ${bytes} bytes of morph deltas, past the ${maxBytes} this device can hold`);
  }

  const deltas = new Float32Array(vertexCount * targetCount * stride);
  const extent = new Float32Array(targetCount);

  for (let t = 0; t < targetCount; t++) {
    const target = targets[t];

    for (const attribute of TARGET_ATTRIBUTES) {
      const accessorIndex = target[attribute.name];
      if (accessorIndex === undefined) continue;

      const declared = componentCountOf(json.accessors[accessorIndex].type);
      if (declared !== attribute.components) {
        throw new Error(
          `glTF: ${label} morph target ${t} gives ${attribute.name} as ` +
          `${json.accessors[accessorIndex].type}, which is not VEC3`,
        );
      }

      const values = readAccessorAsFloat32(json, buffers, accessorIndex);
      if (values.length !== vertexCount * attribute.components) {
        throw new Error(
          `glTF: ${label} morph target ${t} has ` +
          `${values.length / attribute.components} ${attribute.name} deltas but ` +
          `the primitive has ${vertexCount} vertices`,
        );
      }

      for (let v = 0; v < vertexCount; v++) {
        const to = (v * targetCount + t) * stride + attribute.offset;
        const from = v * attribute.components;
        deltas[to] = values[from];
        deltas[to + 1] = values[from + 1];
        deltas[to + 2] = values[from + 2];
      }

      if (attribute.offset === 0) {
        for (let v = 0; v < vertexCount; v++) {
          const from = v * 3;
          const distance = hypot3(values[from], values[from + 1], values[from + 2]);
          if (distance > extent[t]) extent[t] = distance;
        }
      }
    }
  }

  return { targetCount, stride, deltas, extent };
}

/**
 * The weights a morphed mesh starts at.
 *
 * glTF puts defaults on the MESH and lets a NODE override them, which is the
 * same split skins use for a different reason: the mesh is shared and the node
 * is the instance. An absent list means every weight is zero, which is the
 * undeformed mesh.
 *
 * A list of the wrong length is refused rather than padded. Padding would
 * leave targets at zero that the author set, which reads as a face that is
 * subtly wrong in a way nothing reports.
 */
export function morphWeightsFor(meshWeights, nodeWeights, targetCount, label) {
  const source = nodeWeights ?? meshWeights;
  if (source === undefined || source === null) return new Float32Array(targetCount);

  if (source.length !== targetCount) {
    throw new Error(
      `glTF: ${label} gives ${source.length} morph weights for ${targetCount} targets`,
    );
  }
  // JSON numbers: 1e999 parses to Infinity and a string copies in as NaN.
  if (!source.every(Number.isFinite)) {
    throw new Error(`glTF: ${label} has a morph weight that is not a finite number: [${source}]`);
  }
  return Float32Array.from(source);
}
