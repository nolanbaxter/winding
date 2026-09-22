// Scene: entities, transforms, and the renderable list.
//
// Deliberately GPU-free. It stores references to primitives that already live
// on the GPU, but it never calls a WebGPU function, which is what keeps it
// testable under Node and what keeps "what is in the world" separate from "how
// it gets drawn".
//
// Renderables are stored as SoA columns because the renderer walks all of them
// every frame to update bounds and cull. The Node objects users hold are
// cursors over this (see node.js), not entries in it.

import { DEBUG, assert, assertFinite } from '../core/assert.js';
import { HandleAllocator, handleIndex, NULL_HANDLE } from '../core/handle.js';
import { TransformStore } from './transform.js';
import { Node } from './node.js';
import { updateWorldBounds, updateSkinBounds, applySkinBounds } from './bounds.js';
import { aabbRayDistance, rayTriangleDistance } from '../core/math/aabb.js';
import { AnimationPlayer } from './animation.js';
import { vec3Create, vec3TransformMat4, vec3TransformMat4Dir } from '../core/math/vec3.js';
import { mat4Create, mat4Copy, mat4Invert } from '../core/math/mat4.js';
import { grownCapacity, growArray } from '../core/grow.js';

const DEFAULT_CAPACITY = 4096;

export class Scene {
  constructor({ capacity = DEFAULT_CAPACITY, renderableCapacity = capacity, lightCapacity = 256 } = {}) {
    this.capacity = capacity;
    this.entities = new HandleAllocator(capacity);
    this.transforms = new TransformStore(capacity);

    // --- renderable columns -------------------------------------------------
    this.renderableCount = 0;
    this.renderableCapacity = renderableCapacity;
    /**
     * Bumped whenever the set of renderables changes. Batching is O(n log n)
     * and must not run on a scene that is merely moving.
     */
    this.revision = 0;
    this.renderableEntity = new Uint32Array(renderableCapacity);
    /** Which transform slot each renderable reads its world matrix from. */
    this.renderableMatrixSlot = new Uint32Array(renderableCapacity);
    this.renderableMaterial = new Uint16Array(renderableCapacity);
    /** Primitive descriptors: GPU buffers + index count. Held, never called. */
    this.renderablePrimitive = new Array(renderableCapacity);

    /** Index into this.skins, or -1: which palette a renderable reads. */
    this.renderableSkin = new Int32Array(renderableCapacity).fill(-1);

    this.localMin = new Float32Array(renderableCapacity * 3);
    this.localMax = new Float32Array(renderableCapacity * 3);
    this.worldMin = new Float32Array(renderableCapacity * 3);
    this.worldMax = new Float32Array(renderableCapacity * 3);

    // --- lighting -----------------------------------------------------------
    // Plain mutable fields: they are per-scene data the renderer reads when it
    // is handed this scene, not hidden state a function reaches for.
    this.sun = {
      direction: Float32Array.from([-0.35, -0.55, -0.45]),
      color: Float32Array.from([3.2, 3.0, 2.7]),
    };
    /** Set by the engine. Drives ambient light and the skybox. */
    this.environment = null;

    // --- punctual lights ----------------------------------------------------
    // Packed exactly as the GPU wants them, so uploading is one memcpy rather
    // than a per-light gather. Four vec4s each; see LIGHT_FLOATS below.
    this.lightCount = 0;
    this.lightCapacity = lightCapacity;
    this.lights = new Float32Array(lightCapacity * LIGHT_FLOATS);

    this._childrenOf = new Map();   // entity -> [entity], asset declaration order
    /** Resolved skin instances: joint ENTITIES plus the bind pose. */
    this.skins = [];
    this._pendingSkins = [];
    /** Root entity -> AnimationPlayer, for asset instances that have clips. */
    this._players = new Map();
  }

  /**
   * Instantiate a loaded asset. Returns the root Node.
   *
   * Synchronous on purpose: every slow step -- parsing, image decode, buffer
   * upload, pipeline compilation -- already happened in engine.load(). Adding
   * to a scene must never be the thing that stalls a frame.
   */
  add(asset, { parent = null } = {}) {
    const created = new Array(asset.nodes.length).fill(NULL_HANDLE);
    const roots = [];

    const visit = (nodeIndex, parentEntity) => {
      const node = asset.nodes[nodeIndex];
      if (DEBUG) assert(created[nodeIndex] === NULL_HANDLE, 'asset node graph is not a tree');

      const entity = this.entities.alloc();
      created[nodeIndex] = entity;
      this.transforms.add(entity, {
        position: node.position,
        rotation: node.rotation,
        scale: node.scale,
        parent: parentEntity,
      });

      if (node.mesh >= 0) {
        // A skin instance per (node, skin), because `created` is this
        // instance's node-to-entity map -- two copies of one character need
        // two palettes, which is the same reason the animation player is per
        // instance. Deferred until after the walk, since a joint node may not
        // have been visited yet.
        // Indexed against this.skins, which accumulates across every add() --
        // the pending list is only this call's tail of it.
        const skinIndex = node.skin >= 0 && asset.skins?.[node.skin] !== undefined
          ? this.skins.length + this._pendingSkins.push({ skin: asset.skins[node.skin], created }) - 1
          : -1;
        for (const primitive of asset.meshes[node.mesh].primitives) {
          // Only a mesh that HAS influences is skinned. A rigged mesh
          // instanced under a node with no skin renders static, which is what
          // the pairing living on the node means.
          // primitive.skinned, not primitive.jointIndices: by the time a
          // primitive reaches the scene it is the renderer's object, which
          // carries GPU buffers rather than the arrays they were built from.
          const skinned = skinIndex >= 0 && primitive.skinned ? skinIndex : -1;
          this._addRenderable(entity, primitive, skinned);
        }
      }

      const childEntities = [];
      for (const child of node.children) childEntities.push(visit(child, entity));
      if (childEntities.length > 0) this._childrenOf.set(entity, childEntities);

      return entity;
    };

    const parentEntity = parent ? parent.entity : NULL_HANDLE;
    for (const root of asset.roots) roots.push(visit(root, parentEntity));

    // Joints resolve now, not during the walk: a skin may name a node the walk
    // had not reached yet, and `created` is only complete once it is done.
    for (const pending of this._pendingSkins) {
      const { skin, created: map } = pending;
      const joints = new Uint32Array(skin.joints.length);
      for (let j = 0; j < skin.joints.length; j++) {
        const jointEntity = map[skin.joints[j]];
        if (jointEntity === undefined || jointEntity === NULL_HANDLE) {
          throw new Error(
            `Scene.add: skin "${skin.name}" names node ${skin.joints[j]}, which is not in the ` +
            'asset\'s default scene, so it has no entity to drive it',
          );
        }
        joints[j] = jointEntity;
      }
      this.skins.push({
        joints,
        inverseBind: skin.inverseBind,
        jointRadii: skin.jointRadii,
        // Recomputed each frame from the joints' world positions. A skinned
        // mesh's vertices move without its model matrix moving, so its bounds
        // cannot come from transforming a static box.
        boundsMin: new Float32Array(3),
        boundsMax: new Float32Array(3),
      });
    }
    this._pendingSkins.length = 0;
    // Multi-root assets get a wrapper so the caller always gets one handle back
    // and can move the whole thing with a single setPosition.
    let handle;
    if (roots.length === 1) {
      handle = roots[0];
    } else {
      handle = this.entities.alloc();
      this.transforms.add(handle, { parent: parentEntity });
      for (const root of roots) this.transforms.setParent(root, handle);
      this._childrenOf.set(handle, roots);
    }

    // `created` maps the asset's node indices onto THIS instance's entities,
    // which is the whole reason two copies of one asset can play the same clip
    // at different times. It is kept only when there is something to play.
    if (asset.animations?.length > 0) {
      this._players.set(handle, new AnimationPlayer(asset.animations, created, this.entities));
    }

    return new Node(this, handle);
  }

  _addRenderable(entity, primitive, skin = -1) {
    if (this.renderableCount >= this.renderableCapacity) {
      this._growRenderables(this.renderableCount + 1);
    }
    const i = this.renderableCount++;
    this.revision++;

    this.renderableEntity[i] = entity;
    this.renderableMatrixSlot[i] = handleIndex(entity);
    this.renderableMaterial[i] = primitive.materialId;
    this.renderablePrimitive[i] = primitive;
    this.renderableSkin[i] = skin;

    this.localMin.set(primitive.bounds.min, i * 3);
    this.localMax.set(primitive.bounds.max, i * 3);
    return i;
  }

  _growRenderables(needed) {
    const capacity = grownCapacity(this.renderableCapacity, needed);

    this.renderableEntity = growArray(this.renderableEntity, capacity);
    this.renderableMatrixSlot = growArray(this.renderableMatrixSlot, capacity);
    this.renderableMaterial = growArray(this.renderableMaterial, capacity);
    this.renderablePrimitive.length = capacity;

    this.renderableSkin = growArray(this.renderableSkin, capacity);
    this.localMin = growArray(this.localMin, capacity, 3);
    this.localMax = growArray(this.localMax, capacity, 3);
    // World bounds are recomputed from local every time they are read, so these
    // only need the room, not the contents.
    this.worldMin = growArray(this.worldMin, capacity, 3);
    this.worldMax = growArray(this.worldMax, capacity, 3);

    this.renderableCapacity = capacity;
  }

  /** An empty node, for grouping things you position together. */
  createNode({ parent = null } = {}) {
    const entity = this.entities.alloc();
    this.transforms.add(entity, { parent: parent ? parent.entity : NULL_HANDLE });
    return new Node(this, entity);
  }

  node(entity) {
    return new Node(this, entity);
  }

  childrenOf(node) {
    const children = this._childrenOf.get(node.entity);
    return children ? children.map((entity) => new Node(this, entity)) : [];
  }

  /**
   * Remove a node and everything under it.
   *
   * Renderables are swap-removed, so their order changes -- nothing may cache a
   * renderable index across a remove.
   */
  remove(node) {
    const doomed = [];
    const collect = (entity) => {
      doomed.push(entity);
      for (const child of this._childrenOf.get(entity) ?? []) collect(child);
    };
    collect(node.entity);

    const dying = new Set(doomed);
    for (let i = this.renderableCount - 1; i >= 0; i--) {
      if (!dying.has(this.renderableEntity[i])) continue;

      const last = --this.renderableCount;
      if (i !== last) {
        this.renderableEntity[i] = this.renderableEntity[last];
        this.renderableMatrixSlot[i] = this.renderableMatrixSlot[last];
        this.renderableMaterial[i] = this.renderableMaterial[last];
        this.renderablePrimitive[i] = this.renderablePrimitive[last];
        this.localMin.copyWithin(i * 3, last * 3, last * 3 + 3);
        this.localMax.copyWithin(i * 3, last * 3, last * 3 + 3);
        // World bounds move with their renderable too. Without this the
        // survivor inherits the deleted object's box and keeps it until it
        // happens to move: the GPU culls a visible mesh, and raycast returns
        // the wrong thing.
        this.worldMin.copyWithin(i * 3, last * 3, last * 3 + 3);
        this.worldMax.copyWithin(i * 3, last * 3, last * 3 + 3);
      }
      this.renderablePrimitive[last] = undefined;
      this.revision++;
    }

    for (const entity of doomed) {
      this.transforms.remove(entity);
      this._childrenOf.delete(entity);
      this._players.delete(entity);
      this.entities.free(entity);
    }
  }

  // ------------------------------------------------------------------ lights

  /**
   * Add a point or spot light. Returns its index.
   *
   * `radius` is where the light reaches exactly zero. Physical inverse-square
   * falloff never quite does, so without a cutoff every light would have to be
   * tested against every cluster in the scene -- the radius is what makes
   * clustering possible at all, not a shortcut.
   *
   *   scene.addLight({ position: [0, 3, 0], color: [1, 0.7, 0.4], intensity: 20, radius: 12 });
   *   scene.addLight({ position, direction, innerAngle: 0.3, outerAngle: 0.5, ... });
   */
  addLight({
    position = [0, 0, 0],
    color = [1, 1, 1],
    intensity = 1,
    radius = 10,
    direction = null,
    innerAngle = 0.2,
    outerAngle = 0.5,
  } = {}) {
    if (this.lightCount >= this.lightCapacity) {
      const capacity = grownCapacity(this.lightCapacity, this.lightCount + 1);
      this.lights = growArray(this.lights, capacity, LIGHT_FLOATS);
      this.lightCapacity = capacity;
    }
    const index = this.lightCount++;
    this._writeLight(index, position, color, intensity, radius, direction, innerAngle, outerAngle);
    return index;
  }

  _writeLight(index, position, color, intensity, radius, direction, innerAngle, outerAngle) {
    const o = index * LIGHT_FLOATS;
    const light = this.lights;

    light[o] = position[0]; light[o + 1] = position[1]; light[o + 2] = position[2];
    light[o + 3] = radius;

    light[o + 4] = color[0]; light[o + 5] = color[1]; light[o + 6] = color[2];
    light[o + 7] = intensity;

    if (direction) {
      const length = Math.hypot(direction[0], direction[1], direction[2]) || 1;
      light[o + 8] = direction[0] / length;
      light[o + 9] = direction[1] / length;
      light[o + 10] = direction[2] / length;

      // Frostbite's smooth cone: precomputing scale and offset turns the
      // per-pixel test into a multiply-add instead of two cosines.
      const cosOuter = Math.cos(outerAngle);
      const scale = 1 / Math.max(Math.cos(innerAngle) - cosOuter, 1e-4);
      light[o + 12] = scale;
      light[o + 13] = -cosOuter * scale;
      light[o + 14] = LIGHT_SPOT;
    } else {
      light[o + 8] = 0; light[o + 9] = -1; light[o + 10] = 0;
      light[o + 12] = 1; light[o + 13] = 0;
      light[o + 14] = LIGHT_POINT;
    }
    light[o + 11] = 0;
    light[o + 15] = 0;
  }

  setLightPosition(index, x, y, z) {
    const o = index * LIGHT_FLOATS;
    this.lights[o] = x; this.lights[o + 1] = y; this.lights[o + 2] = z;
  }

  setLightColor(index, r, g, b, intensity = this.lights[index * LIGHT_FLOATS + 7]) {
    const o = index * LIGHT_FLOATS + 4;
    this.lights[o] = r; this.lights[o + 1] = g; this.lights[o + 2] = b;
    this.lights[o + 3] = intensity;
  }

  /** Swap-remove, so light indices are not stable across a removal. */
  removeLight(index) {
    const last = --this.lightCount;
    if (index !== last) {
      this.lights.copyWithin(index * LIGHT_FLOATS, last * LIGHT_FLOATS, (last + 1) * LIGHT_FLOATS);
    }
  }

  /**
   * Recompose world matrices. Returns how many transforms were recomputed.
   *
   * With a job system it runs one depth level at a time across threads; without
   * one it is the same code on this thread. Identical results either way.
   */
  update(jobs = null) {
    return jobs?.parallel ? this.transforms.updateParallel(jobs) : this.transforms.update();
  }

  /** The AnimationPlayer for an asset instance, or null if it has no clips. */
  playerFor(node) {
    return this._players.get(node.entity) ?? null;
  }

  /**
   * Advance every playing clip.
   *
   * engine.run() calls this once per rendered frame, before composition, so
   * playing a clip is all you have to do. Driving renderFrame() yourself means
   * calling this yourself -- that is the deal the manual path makes everywhere
   * else too.
   *
   * Stepped by real elapsed time rather than the fixed simulation step, because
   * animation is presentation: it belongs with the camera controller, not with
   * the physics the accumulator exists to keep deterministic.
   */
  advanceAnimations(dt) {
    let playing = 0;
    for (const player of this._players.values()) {
      if (player.advance(dt, this.transforms)) playing++;
    }
    return playing;
  }

  /**
   * The nearest renderable a ray hits, or null.
   *
   * Two phases. Every world bounding box the ray enters is collected and sorted
   * by entry distance, then walked near to far; a primitive loaded with
   * `retainGeometry` is tested triangle by triangle, and one loaded without it
   * is taken at its box distance because that is the only answer available. The
   * walk stops as soon as the next box starts further away than the best hit so
   * far, which is what keeps an exact test off geometry that cannot win.
   *
   * Mixing the two in one scene is allowed and means what it looks like: an
   * un-retained object can shadow a retained one, because its box is all this
   * knows about it.
   *
   * Composes transforms and refreshes bounds first, because the alternative is
   * an API where the answer silently depends on whether you happened to render
   * since the last move. Both are no-ops on a settled scene.
   *
   * @param origin    vec3, world space
   * @param direction vec3, world space; normalized, or distances come back scaled
   * @returns `{ node, renderable, distance }`, or null
   */
  raycast(origin, direction, { maxDistance = Infinity } = {}) {
    // Unconditional, not DEBUG-only: this is a trust boundary, and a ray with a
    // NaN component is not merely wrong, it is INVISIBLY wrong. The slab test
    // derives no constraint from a NaN axis, so such a ray "hits" the first
    // renderable at distance zero. A zero-sized canvas is enough to produce one.
    assertFinite(origin, 'raycast origin', 0, 3);
    assertFinite(direction, 'raycast direction', 0, 3);

    this.update();
    updateWorldBounds(
      this.renderableCount, this.localMin, this.localMax, this.worldMin, this.worldMax,
      this.transforms.world, this.renderableMatrixSlot, this.transforms.moved,
    );
    // Picking has to see the pose too, or a click lands on where a character
    // was authored rather than where it is standing.
    if (this.skins.length > 0) {
      updateSkinBounds(this.skins, this.transforms.world);
      applySkinBounds(
        this.renderableCount, this.renderableSkin, this.skins, this.worldMin, this.worldMax,
      );
    }

    const candidates = [];

    for (let i = 0; i < this.renderableCount; i++) {
      const distance = aabbRayDistance(this.worldMin, this.worldMax, origin, direction, i * 3);
      // Not `>= 0`: a miss is -1, and a hit at exactly 0 means the origin is
      // already inside the box, which is a hit.
      if (distance < 0 || distance >= maxDistance) continue;
      candidates.push({ renderable: i, distance });
    }
    candidates.sort(byDistance);

    let bestDistance = maxDistance;
    let best = -1;

    for (const candidate of candidates) {
      if (candidate.distance >= bestDistance) break;

      const primitive = this.renderablePrimitive[candidate.renderable];
      // A skinned renderable is answered at its box even when its geometry was
      // retained. The triangles are the BIND POSE, and the narrow phase reaches
      // them by inverting the mesh node's matrix -- which a skinned mesh's
      // vertices do not follow at all. Testing them would not merely be
      // approximate, it would miss, and a posed character with retainGeometry
      // would become unpickable while its box said otherwise. Skinning the
      // triangles here would cost a palette blend per vertex per click.
      const skinned = this.renderableSkin[candidate.renderable] >= 0;
      if (skinned || primitive.positions === undefined || primitive.indices === undefined) {
        best = candidate.renderable;
        bestDistance = candidate.distance;
        continue;
      }

      const distance = this._triangleDistance(candidate.renderable, primitive, origin, direction);
      if (distance < 0 || distance >= bestDistance) continue;
      bestDistance = distance;
      best = candidate.renderable;
    }

    if (best < 0) return null;
    return {
      node: new Node(this, this.renderableEntity[best]),
      renderable: best,
      distance: bestDistance,
    };
  }

  /**
   * Nearest triangle of one renderable along a ray, or -1.
   *
   * The ray is pushed into local space rather than the triangles into world
   * space: one matrix inverse against however many vertices the primitive has.
   *
   * The transformed direction is deliberately left un-normalized. `M` is
   * linear, so `M(o + t*d)` and `o + t*d` share the same `t`, and the distance
   * comes back on the same scale as the box distances it is compared against.
   * Normalizing here would silently rescale it by the object's scale factor.
   */
  _triangleDistance(renderable, primitive, origin, direction) {
    const slot = this.renderableMatrixSlot[renderable];
    mat4Copy(PICK_WORLD, this.transforms.world, 0, slot * 16);
    // A scale of zero on any axis collapses the mesh to a plane or a point.
    // There is nothing to hit, and inverting would divide by zero.
    if (mat4Invert(PICK_INVERSE, PICK_WORLD) === null) return -1;

    vec3TransformMat4(LOCAL_ORIGIN, origin, PICK_INVERSE);
    vec3TransformMat4Dir(LOCAL_DIRECTION, direction, PICK_INVERSE);

    const { positions, indices } = primitive;
    let nearest = -1;

    for (let i = 0; i + 2 < indices.length; i += 3) {
      const distance = rayTriangleDistance(
        LOCAL_ORIGIN, LOCAL_DIRECTION, positions,
        indices[i] * 3, indices[i + 1] * 3, indices[i + 2] * 3,
      );
      if (distance < 0) continue;
      if (nearest < 0 || distance < nearest) nearest = distance;
    }
    return nearest;
  }

  /**
   * The nearest renderable under a point on the canvas.
   *
   * `x`/`y` are CSS pixels from the canvas's top-left, and `width`/`height` its
   * CSS size -- exactly what a pointer event plus getBoundingClientRect give you.
   */
  pick(camera, x, y, width, height, options) {
    camera.rayFromScreen(x, y, width, height, PICK_ORIGIN, PICK_DIRECTION);
    return this.raycast(PICK_ORIGIN, PICK_DIRECTION, options);
  }
}

// Scratch for pick(). A scene is not raycast re-entrantly, and the result is
// read before the next call.
const PICK_ORIGIN = vec3Create();
const PICK_DIRECTION = vec3Create();

/** Scratch for the narrow phase: the ray, pushed into one renderable's local space. */
const PICK_WORLD = mat4Create();
const PICK_INVERSE = mat4Create();
const LOCAL_ORIGIN = vec3Create();
const LOCAL_DIRECTION = vec3Create();

function byDistance(a, b) {
  return a.distance - b.distance;
}

/** positionRadius, colorIntensity, directionCone, coneFalloff -- four vec4s. */
export const LIGHT_FLOATS = 16;
export const LIGHT_POINT = 0;
export const LIGHT_SPOT = 1;
