// Camera.
//
// Note what is absent: a far plane. The projection is reverse-Z with an
// infinite far distance, so there is no "draw distance" number
// to pick, tune, or get wrong. One less arbitrary value in the API.

import { DEBUG, assert } from '../core/assert.js';
import { vec3Create, vec3Copy, vec3Cross, vec3Normalize, vec3Sub } from '../core/math/vec3.js';
import {
  mat4Create, mat4Invert, mat4LookAt, mat4Multiply, mat4PerspectiveReverseZInfinite,
} from '../core/math/mat4.js';

export class Camera {
  constructor({ fovY = Math.PI / 3, near = 0.1 } = {}) {
    /** Vertical field of view in radians. Horizontal follows from the aspect. */
    this.fovY = fovY;
    /**
     * The only depth knob. Reverse-Z keeps precision near-uniform, so this can
     * sit much closer than the usual advice allows without z-fighting.
     */
    this.near = near;

    this.position = vec3Create(0, 0, 5);
    this.target = vec3Create(0, 0, 0);
    this.up = vec3Create(0, 1, 0);

    this.view = mat4Create();
    this.projection = mat4Create();
    this.viewProjection = mat4Create();
    /**
     * Screen back to view space. The clustered light grid needs it to turn
     * screen tiles into view-space froxels; nothing else does yet.
     */
    this.inverseProjection = mat4Create();
    this.aspect = 1;
  }

  /** Recompute from the current position/target/fov. Call once per frame. */
  update(aspect) {
    this.aspect = aspect;
    mat4LookAt(this.view, this.position, this.target, this.up);
    mat4PerspectiveReverseZInfinite(this.projection, this.fovY, aspect, this.near);
    // P * V: the view transform applies first, then the projection.
    mat4Multiply(this.viewProjection, this.projection, this.view);
    mat4Invert(this.inverseProjection, this.projection);
    return this;
  }

  /**
   * The world-space ray through a point on the canvas.
   *
   * `x`/`y` are in CSS pixels from the canvas's top-left, which is what a mouse
   * event already gives you after subtracting getBoundingClientRect(). Pass the
   * canvas's CSS size, not its backing-store size -- the two differ on a
   * high-DPI display and using the wrong one skews the ray.
   *
   * Built from fovY, aspect and the camera basis rather than by inverting the
   * view-projection. Same answer, no matrix inverse per call, and it stays
   * readable next to the projection it has to agree with -- an unprojection
   * derived from the matrix is only checkable by testing it, whereas this one
   * is the same construction read backwards.
   *
   * No allocation: both outputs are caller-provided.
   */
  rayFromScreen(x, y, width, height, outOrigin, outDirection) {
    // A zero size is the realistic way to get here with bad numbers: reading
    // getBoundingClientRect() on a canvas that is not laid out yet gives 0x0,
    // and the division below then produces NaN rather than failing.
    if (DEBUG) assert(width > 0 && height > 0, `rayFromScreen: canvas size is ${width}x${height}`);

    // Pixel -> normalized device coords. Y flips: screens count down, clip
    // space counts up.
    const ndcX = (x / width) * 2 - 1;
    const ndcY = 1 - (y / height) * 2;

    const tanHalf = Math.tan(this.fovY * 0.5);

    // The camera basis. `up` is the caller's hint, not necessarily perpendicular
    // to the view direction, so the real up is recovered from the cross product.
    vec3Sub(FORWARD, this.target, this.position);
    vec3Normalize(FORWARD, FORWARD);
    vec3Cross(RIGHT, FORWARD, this.up);
    vec3Normalize(RIGHT, RIGHT);
    vec3Cross(UP, RIGHT, FORWARD);

    const sx = ndcX * tanHalf * this.aspect;
    const sy = ndcY * tanHalf;
    for (let i = 0; i < 3; i++) {
      outDirection[i] = FORWARD[i] + RIGHT[i] * sx + UP[i] * sy;
    }
    vec3Normalize(outDirection, outDirection);

    // The eye, not the near plane: a caller comparing hit distances wants them
    // measured from the camera, and the near plane is a rendering concern.
    vec3Copy(outOrigin, this.position);
    return outDirection;
  }
}

// Scratch for rayFromScreen. Module-level because a camera is not re-entrant
// and this is the engine's standard way of keeping the frame loop allocation
// free -- even though picking is not in it.
const FORWARD = vec3Create();
const RIGHT = vec3Create();
const UP = vec3Create();
