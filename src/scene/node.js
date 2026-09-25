// Node: a cursor over the scene's stores.
//
// A Node holds an entity handle and a scene reference. It is NOT where anything
// is stored -- positions still live in TransformStore's Float32Array columns,
// and a Node is just a comfortable way to reach one. Creating a thousand Nodes
// allocates a thousand small objects and moves no data; creating none is also
// fine, because the stores work perfectly well addressed by handle.
//
// Identity: two Nodes for the same entity are different JS objects. Compare
// `a.entity === b.entity`, not `a === b`.
//
// SETTERS, NOT PROPERTIES. `node.position.x = 5` is the idiom everyone expects
// and it cannot work here. The position column is shared memory; handing out a
// live view means a write that never marks the transform dirty, so the node
// silently does not move until something unrelated happens to touch it. The
// alternatives are a Proxy on every access or dirty-checking every transform
// every frame, and both give back what the propagation design just bought.

import {
  quatCreate, quatSetAxisAngle, quatFromEuler, quatNormalize, quatLookAlong,
} from '../core/math/quat.js';
import { NULL_HANDLE } from '../core/handle.js';

// Shared scratch. Node methods are called from user code, never concurrently,
// and a fresh quaternion per setRotation call would allocate in someone's loop.
const scratchQuat = quatCreate();

export class Node {
  constructor(scene, entity) {
    this.scene = scene;
    this.entity = entity;
  }

  get alive() {
    return this.scene.entities.alive(this.entity);
  }

  setPosition(x, y, z) {
    this.scene.transforms.setPosition(this.entity, x, y, z);
    return this;
  }

  setScale(x, y = x, z = x) {
    this.scene.transforms.setScale(this.entity, x, y, z);
    return this;
  }

  /** Takes a quaternion. For angles you can reason about, use the two below. */
  setRotation(q) {
    this.scene.transforms.setRotation(this.entity, q);
    return this;
  }

  /**
   * Face -Z along (x, y, z), upright: the way a spot shines, the way a
   * directional light's light travels, the way a followed camera looks. In
   * the parent's space, like every other setter here.
   *
   *   key.setDirection(-0.4, -0.7, -0.3);
   */
  setDirection(x, y, z) {
    quatLookAlong(scratchQuat, [x, y, z]);
    this.scene.transforms.setRotation(this.entity, scratchQuat);
    return this;
  }

  setRotationAxisAngle(axis, radians) {
    quatSetAxisAngle(scratchQuat, axis, radians);
    this.scene.transforms.setRotation(this.entity, scratchQuat);
    return this;
  }

  /** Radians, YXZ order. Converted to a quaternion immediately and not stored. */
  setRotationEuler(yaw, pitch, roll = 0) {
    quatFromEuler(scratchQuat, yaw, pitch, roll);
    // Normalizing costs almost nothing here and guards against a caller's
    // accumulated angles drifting the result off the unit sphere.
    quatNormalize(scratchQuat, scratchQuat);
    this.scene.transforms.setRotation(this.entity, scratchQuat);
    return this;
  }

  /** Pass null to detach to the scene root. */
  setParent(node) {
    this.scene.transforms.setParent(this.entity, node ? node.entity : NULL_HANDLE);
    return this;
  }

  /** The player for this instance. Only the Node scene.add() returned has one. */
  get animation() {
    return this.scene.playerFor(this);
  }

  /** Clip names this instance can play. */
  get animations() {
    return this.animation?.names ?? [];
  }

  /** Play a clip by name or index. Does nothing if there is no such clip. */
  play(nameOrIndex, options) {
    this.animation?.play(nameOrIndex, options);
    return this;
  }

  /** Stop every layer, or `{ layer }`; with `{ fade }` the clips fade out. */
  stop(options) {
    this.animation?.stop(options);
    return this;
  }

  /**
   * World-space position, as of the last Scene.update().
   *
   * Reading it straight after a setPosition returns the OLD value: the world
   * matrix has not been recomposed yet.
   */
  getWorldPosition(out) {
    const o = this.scene.transforms.worldOffset(this.entity);
    out[0] = this.scene.transforms.world[o + 12];
    out[1] = this.scene.transforms.world[o + 13];
    out[2] = this.scene.transforms.world[o + 14];
    return out;
  }

  /**
   * This node's morph target weights, or null if its mesh has none.
   *
   * A LIVE view, which is the one exception to the rule at the top of this
   * file: `node.weights[0] = 1` is meant to work. Nothing derived is cached
   * from it -- the renderer uploads the array every frame and the bounds pass
   * reads it every frame -- so there is no dirty flag for a direct write to
   * miss. See Scene.morphWeights.
   */
  get weights() {
    return this.scene.morphWeights(this.entity);
  }

  /**
   * Change this light's colour, brightness, reach or cone. Partial: only the
   * fields given change. Returns false if this node is not a light.
   *
   *   lamp.setLight({ intensity: 30 });
   *   torch.setLight({ color: colorFromHex('#ffb060'), outerAngle: 0.4 });
   *
   * Position and aim are not here, because they are not properties of the
   * light -- they are where its node is. Move the node.
   */
  setLight(changes) {
    return this.scene.setLight(this.entity, changes);
  }

  /** Nodes created for this entity's children, in the order the asset declared them. */
  children() {
    return this.scene.childrenOf(this);
  }

  destroy() {
    this.scene.remove(this);
  }
}
