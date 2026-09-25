// Winding -- a WebGPU renderer.
//
// Everything Tier 3 needs, in one import. Lower tiers are reachable from their
// own modules, and `engine.rhi` / `engine.renderer` get you there without
// leaving this one. Never a wall, always a floor.

export { Winding } from './app/engine.js';
export { OrbitController } from './app/controllers.js';
export { StatsOverlay } from './app/overlay.js';

export { Camera } from './scene/camera.js';
export { Scene } from './scene/scene.js';
export { Node } from './scene/node.js';

export { Environment } from './render/ibl.js';
export { parseHDR } from './render/hdr.js';

// Math is part of the public surface, not an engine internal. Tier 3 already
// demands it: getWorldPosition(out) wants a vec3, setRotationAxisAngle wants an
// axis, camera.position and camera.target are vectors you write into.
// Leaving it out of this barrel meant reaching past it into src/core/math.
export * from './core/math/vec3.js';
export * from './core/math/quat.js';
export * from './core/math/mat4.js';
export * from './core/math/aabb.js';

// Colour conversion, for the same reason the math is here: every colour this
// engine takes is linear and every colour a human picks is not, and leaving
// the conversion out meant every caller hand-rolling it or getting it wrong.
export * from './core/color.js';
