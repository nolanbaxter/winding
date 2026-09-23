// Camera controllers.
//
// A viewer without a way to move the camera is not a viewer, so this is part of
// Tier 3 rather than something every example re-invents.
//
// Controllers mutate a Camera and nothing else. They do not know the engine
// exists, which is what lets you drive one from your own loop, from a test, or
// from an animation.

import { vec3Create } from '../core/math/vec3.js';
import { fitDistance, boundsRadius } from '../scene/camera.js';

const MIN_PITCH = -Math.PI / 2 + 0.01;   // never exactly straight down: at the
const MAX_PITCH = Math.PI / 2 - 0.01;    // pole the up vector becomes ambiguous

/** Pixels of movement before a press counts as a drag rather than a click. */
const DRAG_SLOP = 3;

/**
 * Drag to orbit, wheel to zoom, right-drag or shift-drag to pan.
 *
 * Call update(dt) once per rendered frame for the damping to run. Skipping it
 * still works, it just snaps instead of easing.
 */
export class OrbitController {
  constructor(camera, element, {
    distance = 6, yaw = 0, pitch = 0.3, target = [0, 0, 0],
    minDistance = 0.1, maxDistance = 1000,
    rotateSpeed = 0.005, zoomSpeed = 0.0015, panSpeed = 0.002,
    damping = 12,
  } = {}) {
    this.camera = camera;
    this.element = element;

    this.distance = distance;
    this.yaw = yaw;
    this.pitch = pitch;
    this.target = vec3Create(target[0], target[1], target[2]);

    // Where the camera is easing toward. Separating desired from actual is what
    // makes damping a two-line lerp instead of a state machine.
    this.desired = { distance, yaw, pitch, target: vec3Create(target[0], target[1], target[2]) };

    this.minDistance = minDistance;
    this.maxDistance = maxDistance;
    this.rotateSpeed = rotateSpeed;
    this.zoomSpeed = zoomSpeed;
    this.panSpeed = panSpeed;
    this.damping = damping;

    this._dragging = 0;   // 0 none, 1 orbit, 2 pan
    /**
     * True when the last press moved far enough to be a drag rather than a
     * click. A click handler that wants to select something reads this to tell
     * "released after orbiting" from "clicked on that object" -- the browser
     * fires a click event for both.
     */
    this.dragged = false;
    this._attach();
    this.update(0);       // place the camera before the first frame; nothing to forget
  }

  _attach() {
    const element = this.element;

    this._onPointerDown = (event) => {
      // Shift-drag and right-drag both pan; trackpad users have no right button
      // worth relying on.
      this._dragging = (event.button === 2 || event.shiftKey) ? 2 : 1;
      element.setPointerCapture(event.pointerId);
      this._lastX = event.clientX;
      this._lastY = event.clientY;
      this._downX = event.clientX;
      this._downY = event.clientY;
      this.dragged = false;
    };

    this._onPointerMove = (event) => {
      if (!this._dragging) return;
      // A few pixels of slop: a press almost never lands on the exact pixel it
      // started on, and treating a 1px wobble as a drag loses the click.
      if (Math.abs(event.clientX - this._downX) > DRAG_SLOP
        || Math.abs(event.clientY - this._downY) > DRAG_SLOP) {
        this.dragged = true;
      }
      const dx = event.clientX - this._lastX;
      const dy = event.clientY - this._lastY;
      this._lastX = event.clientX;
      this._lastY = event.clientY;

      if (this._dragging === 1) {
        this.desired.yaw -= dx * this.rotateSpeed;
        // PLUS dy, matching the yaw line above. clientY grows downward, so a
        // drag upward is negative and this lowers the camera -- which is what
        // grabbing the front of an object and pulling it up does: the
        // underside rotates toward you.
        //
        // It was minus, which raised the camera instead. That is the opposite
        // metaphor from the one the yaw line uses, so horizontal felt like
        // turning the object and vertical felt like the object was hinged
        // behind itself. Two conventions in one drag.
        this.desired.pitch = clamp(this.desired.pitch + dy * this.rotateSpeed, MIN_PITCH, MAX_PITCH);
      } else {
        // Pan in the camera's own plane, scaled by distance so the target
        // tracks the cursor at any zoom level.
        const scale = this.panSpeed * this.desired.distance;
        const cosYaw = Math.cos(this.desired.yaw);
        const sinYaw = Math.sin(this.desired.yaw);
        this.desired.target[0] -= (dx * cosYaw) * scale;
        this.desired.target[2] += (dx * sinYaw) * scale;
        this.desired.target[1] += dy * scale;
      }
    };

    this._onPointerUp = (event) => {
      this._dragging = 0;
      if (element.hasPointerCapture?.(event.pointerId)) element.releasePointerCapture(event.pointerId);
    };

    this._onWheel = (event) => {
      event.preventDefault();
      // Exponential, so one notch feels the same whether you are 1 unit or 100
      // units out. Linear zoom crawls when far and overshoots when close.
      this.desired.distance = clamp(
        this.desired.distance * Math.exp(event.deltaY * this.zoomSpeed),
        this.minDistance, this.maxDistance,
      );
    };

    this._onContextMenu = (event) => event.preventDefault();

    element.addEventListener('pointerdown', this._onPointerDown);
    element.addEventListener('pointermove', this._onPointerMove);
    element.addEventListener('pointerup', this._onPointerUp);
    element.addEventListener('pointercancel', this._onPointerUp);
    element.addEventListener('wheel', this._onWheel, { passive: false });
    element.addEventListener('contextmenu', this._onContextMenu);
  }

  /** @param dt seconds since the last frame; 0 snaps straight to the target */
  update(dt) {
    // Frame-rate independent exponential decay. The naive `x += (target - x) *
    // k` eases faster on a fast display, which makes the feel change with the
    // monitor. This does not.
    const t = dt > 0 ? 1 - Math.exp(-this.damping * dt) : 1;

    this.yaw += (this.desired.yaw - this.yaw) * t;
    this.pitch += (this.desired.pitch - this.pitch) * t;
    this.distance += (this.desired.distance - this.distance) * t;
    for (let i = 0; i < 3; i++) {
      this.target[i] += (this.desired.target[i] - this.target[i]) * t;
    }

    const cosPitch = Math.cos(this.pitch);
    this.camera.position[0] = this.target[0] + Math.sin(this.yaw) * cosPitch * this.distance;
    this.camera.position[1] = this.target[1] + Math.sin(this.pitch) * this.distance;
    this.camera.position[2] = this.target[2] + Math.cos(this.yaw) * cosPitch * this.distance;
    this.camera.target.set(this.target);
    return this;
  }

  /**
   * Adopt wherever the camera is now, instead of overwriting it.
   *
   * This controller OWNS the camera's position: update() rebuilds it from
   * yaw, pitch, distance and target every frame. So anything that moves the
   * camera directly -- a cutscene, a teleport, a saved viewpoint,
   * Camera.frameBounds -- works for exactly one frame and is then silently
   * undone. There was no way to hand control back.
   *
   * This is the inverse of what update() does, so the two agree by
   * construction: distance is the length of the offset, pitch is the angle it
   * makes with the horizontal plane, and yaw is its bearing in that plane.
   *
   * TWO PLACES IT CANNOT BE EXACT, both because the controller's coordinates
   * cannot express every pose:
   *
   *   Past the pitch clamp. Looking straight down is a pose this controller
   *   deliberately refuses -- at the pole the up vector is ambiguous -- so a
   *   camera aimed there is adopted at the nearest angle it can hold.
   *
   *   Outside the distance clamp, for the same reason.
   *
   * Both are applied immediately rather than on the next frame, because a
   * camera that moves one frame after you stopped touching it is harder to
   * explain than one that moves when you ask.
   *
   * Yaw is left alone when the camera is directly above or below its target,
   * where the bearing is genuinely undefined -- keeping the old one means no
   * spin when it comes back off the pole.
   */
  syncFromCamera() {
    const camera = this.camera;
    const dx = camera.position[0] - camera.target[0];
    const dy = camera.position[1] - camera.target[1];
    const dz = camera.position[2] - camera.target[2];
    const distance = Math.hypot(dx, dy, dz);

    this.target.set(camera.target);
    this.desired.target.set(camera.target);

    if (distance > 0) {
      // sin(pitch) = dy / distance, straight out of update()'s Y term. The
      // clamp is against floating point pushing the ratio past 1, not against
      // bad input.
      this.pitch = clamp(Math.asin(clamp(dy / distance, -1, 1)), MIN_PITCH, MAX_PITCH);

      // atan2(x, z), not the usual (y, x): update() puts sin(yaw) on X and
      // cos(yaw) on Z, and both carry the same positive scale, so it divides
      // out. Skipped at the pole, where both are zero and the answer would be
      // whatever the noise says.
      if (Math.hypot(dx, dz) > 1e-6) this.yaw = Math.atan2(dx, dz);

      this.distance = clamp(distance, this.minDistance, this.maxDistance);
    }

    this.desired.yaw = this.yaw;
    this.desired.pitch = this.pitch;
    this.desired.distance = this.distance;

    // Both actual AND desired, so the damping has nothing to travel: adopting
    // a pose should be instant, not a glide from wherever it used to be.
    return this.update(0);
  }

  /**
   * Frame an axis-aligned box: point at its centre and back off to fit it.
   *
   * Same signature as Camera.frameBounds on purpose, so anything that frames
   * can take either -- the controller owns the camera's position through
   * yaw/pitch/distance, so writing to the camera directly would be undone on
   * the next update(). Scene.frame relies on that.
   *
   * The fit itself comes from fitDistance rather than being worked out here.
   * It used to be inline and left out the aspect term, which pushed a wide
   * object off both sides of a portrait viewport, and defaulted to a fill of
   * 1.4 that meant nothing in particular. One definition now, and its default
   * is an exact fit -- a box is strictly inside its own bounding sphere except
   * at the corners, so exact already leaves air.
   */
  frameBounds(min, max, { margin = 1 } = {}) {
    for (let i = 0; i < 3; i++) this.desired.target[i] = (min[i] + max[i]) * 0.5;
    this.desired.distance = clamp(
      fitDistance(boundsRadius(min, max), {
        fovY: this.camera.fovY,
        aspect: this.camera.aspect,
        margin,
        near: this.camera.near,
      }),
      this.minDistance,
      this.maxDistance,
    );
    return this.update(0);
  }

  detach() {
    const element = this.element;
    element.removeEventListener('pointerdown', this._onPointerDown);
    element.removeEventListener('pointermove', this._onPointerMove);
    element.removeEventListener('pointerup', this._onPointerUp);
    element.removeEventListener('pointercancel', this._onPointerUp);
    element.removeEventListener('wheel', this._onWheel);
    element.removeEventListener('contextmenu', this._onContextMenu);
  }
}

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}
