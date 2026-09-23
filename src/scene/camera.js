// Camera.
//
// Note what is absent: a far plane. The projection is reverse-Z with an
// infinite far distance, so there is no "draw distance" number
// to pick, tune, or get wrong. One less arbitrary value in the API.

import { DEBUG, assert, assertFinite } from '../core/assert.js';
import { vec3Create, vec3Copy, vec3Cross, vec3Normalize, vec3Sub } from '../core/math/vec3.js';
import {
  mat4Create, mat4Invert, mat4LookAt, mat4Multiply, mat4PerspectiveReverseZInfinite,
} from '../core/math/mat4.js';

/**
 * How far back a sphere of `radius` has to sit to fill the view.
 *
 * Exported because TWO things frame: a bare Camera, and an OrbitController,
 * which owns its camera's position through yaw/pitch/distance and would
 * overwrite anything written directly. They need the same number and must not
 * each have their own idea of it -- the controller used to compute this
 * inline, without the aspect term, so a wide object on a portrait viewport ran
 * off both sides.
 *
 * THE TIGHTER HALF-ANGLE WINS. fovY is vertical; horizontally the frustum is
 * `atan(tan(fovY/2) * aspect)`, which on a portrait viewport is the smaller of
 * the two.
 *
 * Returns 0 for a degenerate radius, because there is no distance to derive
 * and the caller knows better what to do about it than this does.
 */
export function fitDistance(radius, { fovY, aspect = 1, margin = 1, near = 0 } = {}) {
  if (!(radius > 0)) return 0;
  const halfY = fovY * 0.5;
  // A non-positive aspect means update() has not run; square is the
  // conservative read, since it can only over-estimate the distance.
  const halfX = Math.atan(Math.tan(halfY) * (aspect > 0 ? aspect : 1));
  const fit = (radius / Math.sin(Math.min(halfY, halfX))) * margin;

  // The near plane must not cut the sphere: its nearest point is
  // `distance - radius` away and that has to clear `near`. Bites when framing
  // something smaller than the near plane.
  return Math.max(fit, radius + near);
}

/** Half the diagonal of an axis-aligned box: the radius that contains it. */
export function boundsRadius(min, max) {
  return 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
}

/** Scratch for frameBounds. Not re-entrant, and it never needs to be. */
const FRAME_DIRECTION = vec3Create();

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

  /**
   * Move the camera so a world-space box fills the view, keeping the direction
   * it is already looking from.
   *
   * The guess this removes: every example ever written picks a camera distance
   * by eye and adjusts until the model fits. That number is derivable from the
   * bounds and the field of view, and nothing else in this engine asks you to
   * remember a value it could have computed.
   *
   * FITS THE BOUNDING SPHERE, not the eight corners. The sphere is rotation
   * invariant, so orbiting does not change how much of the view the object
   * fills -- fitting corners makes an object breathe as it turns, growing and
   * shrinking with the projected silhouette. The same argument the shadow
   * cascades are sphere-fitted for, and the same reason.
   *
   * It also means `margin` can default to an exact fit. A box is strictly
   * inside its own sphere except at the eight corners, so a sphere that
   * exactly touches the frustum already leaves visible air on every side.
   *
   * THE TIGHTER HALF-ANGLE WINS. fovY is the vertical one; horizontally the
   * frustum is `atan(tan(fovY/2) * aspect)`. On a portrait viewport that is
   * the SMALLER of the two, so fitting to fovY alone would push a wide object
   * off both sides. Aspect defaults to the last one update() was given, which
   * is zero before the first frame -- pass it explicitly when framing during
   * setup.
   */
  frameBounds(min, max, { margin = 1, aspect = this.aspect } = {}) {
    if (DEBUG) {
      assertFinite(min, 'frameBounds min', 0, 3);
      assertFinite(max, 'frameBounds max', 0, 3);
    }

    const cx = (min[0] + max[0]) * 0.5;
    const cy = (min[1] + max[1]) * 0.5;
    const cz = (min[2] + max[2]) * 0.5;

    const fit = fitDistance(boundsRadius(min, max), {
      fovY: this.fovY, aspect, margin, near: this.near,
    });

    // Where the camera is now, relative to what it was looking at. Preserved
    // so framing is a zoom rather than a jump to some canonical angle.
    vec3Sub(FRAME_DIRECTION, this.position, this.target);
    let length = Math.hypot(FRAME_DIRECTION[0], FRAME_DIRECTION[1], FRAME_DIRECTION[2]);
    if (!(length > 0)) {
      // Degenerate: the camera is sitting on its own target and has no
      // direction to preserve. Looking down -Z is the convention everything
      // else here starts from.
      FRAME_DIRECTION[0] = 0; FRAME_DIRECTION[1] = 0; FRAME_DIRECTION[2] = 1;
      length = 1;
    }
    const inv = 1 / length;

    // A degenerate box has no distance to derive, so the only honest answer
    // is to look at it from wherever you already were.
    const distance = fit > 0 ? fit : length;

    this.target[0] = cx; this.target[1] = cy; this.target[2] = cz;
    this.position[0] = cx + FRAME_DIRECTION[0] * inv * distance;
    this.position[1] = cy + FRAME_DIRECTION[1] * inv * distance;
    this.position[2] = cz + FRAME_DIRECTION[2] * inv * distance;
    return this;
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
