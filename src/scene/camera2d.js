// A camera for 2D: one world unit is one CSS pixel, y points down, and the
// origin is the top-left -- the coordinates an HTML canvas, CSS and every
// image editor already use, so a sprite drawn at (10, 20) is where a canvas
// would put it, and a 16-pixel sprite is as big as a 16-pixel <img> on any
// screen. Snapping and edge smoothing work in the screen's own pixels.
//
//   const camera = new Camera2D();
//   engine.run({ scene, camera });
//
// A scene viewed through one draws its 2D content -- sprites and text -- in
// painter's order (by layer, then in the order added), with no lighting and no
// depth, and composites it as the browser composites a page: in sRGB, so a
// colour comes out exactly as authored and half-transparent edges blend the
// way they do in an image editor. See render/view2d.js.

import { mat4Create } from '../core/math/mat4.js';

export class Camera2D {
  constructor({
    position = [0, 0], zoom = 1, angle = 0, pivot = [0, 0],
    background = [0, 0, 0, 1], pixelSnap = false, ambient = [0, 0, 0], anchor, rotation,
  } = {}) {
    if (rotation !== undefined) throw new Error('Camera2D: rotation is now angle, in the same radians, as node.setAngle takes');
    if (anchor !== undefined) throw new Error('Camera2D: anchor is now pivot -- the point of the view placed at position, as a sprite\'s pivot is');
    /**
     * The world point at `pivot` of the view. With the default pivot that is
     * the view's top-left, as a canvas's scroll is; with [0.5, 0.5] it is the
     * centre, which is what a camera following something wants.
     */
    this.position = Float32Array.from(position);
    /** Where in the view `position` sits: [0, 0] top-left, [1, 1] bottom-right, as a sprite's pivot. */
    this.pivot = Float32Array.from(pivot);
    /** Screen pixels per world unit. 2 draws everything twice as large. */
    this.zoom = zoom;
    /** Radians, turning the view clockwise about its centre. */
    this.angle = angle;
    /** What the view is cleared to: sRGB, 0..1, as a CSS colour. */
    this.background = Float32Array.from(background);
    /**
     * For pixel art: put every sprite's corner on a whole screen pixel, the
     * view's own offset with it, and make a unit a whole number of screen
     * pixels -- zoom times the pixel ratio, rounded. On a 1.5x screen a texel
     * is then 2 pixels, not 1 and 2 by turns. For an unrotated view.
     */
    this.pixelSnap = pixelSnap;
    /**
     * The light on anything `lit` where no light reaches, linear, as a light's
     * colour is. Black, the default, shows lit things only where lights are.
     */
    this.ambient = Float32Array.from(ambient);

    /** World to the canvas's own pixels. */
    this.view = mat4Create();
    /** Canvas pixels to clip space. */
    this.projection = mat4Create();
    /** World to clip: projection times view. */
    this.viewProjection = mat4Create();
    /** The canvas, in its own pixels, and its pixels to a CSS pixel, as of the last update. */
    this.width = 1;
    this.height = 1;
    this.pixelRatio = 1;
  }

  /** Marks this as a 2D view: the renderer draws it with its 2D path. */
  get is2D() { return true; }

  /**
   * Recompute the matrices for a canvas `width` x `height` of its own pixels,
   * `pixelRatio` of them to a CSS pixel. The renderer calls it every frame;
   * `aspect` is ignored and is there so the call is the same as a 3D camera's.
   */
  update(aspect, width = this.width, height = this.height, pixelRatio = this.pixelRatio) {
    this.width = width;
    this.height = height;
    this.pixelRatio = pixelRatio;
    // Canvas pixels per world unit: a whole number of them when snapping, so
    // every texel of pixel art is the same size.
    const z = this.pixelSnap ? Math.max(1, Math.round(this.zoom * pixelRatio)) : this.zoom * pixelRatio;
    const c = Math.cos(this.angle), s = Math.sin(this.angle);
    // The world point at the centre of the view: `position` is at the pivot,
    // and the way from the pivot to the centre turns with the view.
    const ox = (0.5 - this.pivot[0]) * width / z, oy = (0.5 - this.pivot[1]) * height / z;
    const cx = this.position[0] + c * ox - s * oy;
    const cy = this.position[1] + s * ox + c * oy;
    // screen = R(-angle) (world - centre) zoom + size / 2: the view turns
    // clockwise, so the world on screen turns the other way.
    let tx = width / 2 - z * (c * cx + s * cy);
    let ty = height / 2 - z * (-s * cx + c * cy);
    if (this.pixelSnap) { tx = Math.round(tx); ty = Math.round(ty); }
    const v = this.view;
    v.fill(0);
    v[0] = z * c; v[1] = -z * s;
    v[4] = z * s; v[5] = z * c;
    v[10] = 1; v[12] = tx; v[13] = ty; v[15] = 1;
    const p = this.projection;
    p.fill(0);
    p[0] = 2 / width; p[5] = -2 / height; p[10] = 1;
    p[12] = -1; p[13] = 1; p[15] = 1;
    const m = this.viewProjection;
    m.fill(0);
    m[0] = p[0] * v[0]; m[1] = p[5] * v[1];
    m[4] = p[0] * v[4]; m[5] = p[5] * v[5];
    m[10] = 1;
    m[12] = p[0] * v[12] + p[12]; m[13] = p[5] * v[13] + p[13]; m[15] = 1;
    return this;
  }

  /** Where canvas pixel (x, y) -- the canvas's own pixels -- is in the world, as of the last update. */
  screenToWorld(x, y, out = new Float32Array(2)) {
    const v = this.view;
    const dx = x - v[12], dy = y - v[13];
    // The inverse of a rotation-and-scale is its transpose over the scale squared.
    const zz = v[0] * v[0] + v[1] * v[1];
    out[0] = (v[0] * dx + v[1] * dy) / zz;
    out[1] = (v[4] * dx + v[5] * dy) / zz;
    return out;
  }

  /** Where world point (x, y) lands on the canvas, in its own pixels, as of the last update. */
  worldToScreen(x, y, out = new Float32Array(2)) {
    const v = this.view;
    out[0] = v[0] * x + v[4] * y + v[12];
    out[1] = v[1] * x + v[5] * y + v[13];
    return out;
  }
}
