// The joint matrix palette.
//
// One buffer for the frame, holding every skinned instance's joint matrices
// end to end. A renderable's draw data carries where its own begin, which is
// what lets two characters in different poses share a batch and a draw call --
// the batch is (primitive, material, winding, skinned) and says nothing about
// pose.
//
// Each entry is jointWorld * inverseBind. Both halves already exist: the
// transform hierarchy composes joint world matrices every frame because joints
// are ordinary nodes, and the inverse bind matrices came from the file. So
// this is a multiply per joint per frame and no new state.

import { mat4Multiply, mat4MultiplyAffine } from '../core/math/mat4.js';
import { grownCapacity } from '../core/grow.js';
import { storageCapacity, createBuffer } from '../rhi/buffer.js';
import { handleIndex } from '../core/handle.js';

/** Bytes per joint matrix: a mat4x4 of f32. */
export const JOINT_BYTES = 64;

export class SkinPalette {
  constructor(rhi, capacity = 256) {
    this.rhi = rhi;
    this.capacity = capacity;
    this.data = new Float32Array(capacity * 16);
    this.buffer = createBuffer(rhi, {
      label: 'joint-palette',
      size: capacity * JOINT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    /** Start joint of each skin instance, by index into scene.skins. */
    this.offsets = new Uint32Array(64);
    this.jointCount = 0;
    /** Bumped when the buffer is replaced, so bind groups know to rebuild. */
    this.revision = 0;
  }

  /**
   * Rebuild every palette from the scene's current pose.
   *
   * Unconditional: a joint is a node like any other, so it moves when an
   * animation or the user moves it, and the transform store's `moved` flags
   * are cleared by the frame that consumed them. Tracking which skins changed
   * would mean reading those flags before the renderer does, which is a
   * coupling worth more than the multiply it saves.
   */
  update(scene) {
    const skins = scene.skins;
    if (skins.length === 0) {
      this.jointCount = 0;
      return;
    }

    let total = 0;
    for (const skin of skins) total += skin.joints.length;
    if (total > this.capacity) this._grow(total);
    if (this.offsets.length < skins.length) {
      this.offsets = new Uint32Array(grownCapacity(this.offsets.length, skins.length));
    }

    const world = scene.transforms.world;
    let joint = 0;
    for (let s = 0; s < skins.length; s++) {
      const { joints, inverseBind, affine } = skins[s];
      this.offsets[s] = joint;
      // A joint's world matrix is always affine; the inverse binds come from
      // the file and were checked at load. Both affine, the cheaper multiply
      // gives the same bits.
      const multiply = affine ? mat4MultiplyAffine : mat4Multiply;

      for (let j = 0; j < joints.length; j++) {
        // jointWorld * inverseBind, straight into the upload staging array.
        // The node's own transform does NOT appear: glTF says a skinned mesh's
        // node transform is ignored, because the joints place it entirely.
        multiply(
          this.data, world, inverseBind,
          joint * 16, handleIndex(joints[j]) * 16, j * 16,
        );
        joint++;
      }
    }

    this.jointCount = joint;
    this.rhi.queue.writeBuffer(this.buffer, 0, this.data, 0, joint * 16);
  }

  _grow(needed) {
    const capacity = grownCapacity(
      this.capacity, needed, storageCapacity(this.rhi, JOINT_BYTES), 'joints',
    );
    this.data = new Float32Array(capacity * 16);
    this.buffer.destroy();
    this.buffer = createBuffer(this.rhi, {
      label: 'joint-palette',
      size: capacity * JOINT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.capacity = capacity;
    this.revision++;
  }

  destroy() {
    this.buffer.destroy();
  }
}
