// Options a call does not know: said once in the console, with the nearest
// name it does know, and otherwise ignored.
//
// Before this, a misspelt option was simply dropped. `castShadows` for
// `castShadow` turned no shadows off and said nothing, and a test ran for
// weeks without testing what it was written to. A warning rather than a throw,
// because code that runs today with a stray option must keep running in 1.x;
// it costs one lookup per key, on calls that take options, never per frame of
// work.
//
// Every call's names are listed here, in one place, so test/docs.test.js can
// hold each list to the options table in docs/API.md: an option added to the
// code and the reference, but not here, fails there instead of warning at a
// user.

/** The names each call takes, by the name its messages use. */
export const OPTIONS = {
  create: ['label', 'powerPreference', 'onDeviceLost', 'onError', 'exposure', 'autoExposure', 'antialias', 'grading',
    'post', 'shadows', 'shadowDistance', 'lightDistance', 'ao', 'oit', 'taa', 'fog', 'dof', 'environment', 'maxDraws',
    'gpuTiming', 'workerCount', 'onDemand'],
  'create shadows': ['size', 'cascades', 'lambda', 'casterExtent', 'normalBias', 'depthBiasSlope', 'depthBiasConstant', 'localSize'],
  'create post': ['threshold', 'knee', 'filterRadius', 'strength', 'levels', 'antialias', 'grading'],
  createScene: ['environment', 'capacity', 'renderableCapacity', 'lightCapacity'],
  load: ['retainGeometry', 'baseURL', 'fetch'],
  loadTexture: ['srgb', 'pixelated', 'label', 'fetch'],
  createMesh: ['positions', 'indices', 'normals', 'uvs', 'colors', 'material'],
  'createMesh material': ['color', 'metallic', 'roughness', 'emissive', 'doubleSided'],
  loadLUT: ['fetch'],
  loadSplats: ['fetch'],
  loadEnvironment: ['fetch', 'size', 'irradianceSize', 'prefilterMips', 'label'],
  run: ['update', 'frame', 'hud'],
  renderFrame: ['hud', 'target'],
  createTarget: ['size', 'pixelated', 'label'],
  grading: ['whiteBalance', 'contrast', 'saturation', 'lut'],
  autoExposure: ['min', 'max', 'brighten', 'darken'],
  fog: ['visibility', 'height', 'scaleHeight', 'albedo'],
  dof: ['focusDistance', 'fStop', 'sensorHeight'],
  Scene: ['capacity', 'renderableCapacity', 'lightCapacity'],
  createNode: ['parent'],
  add: ['parent'],
  frame: ['margin', 'aspect'],
  raycast: ['maxDistance'],
  Camera: ['fovY', 'near', 'orthographic', 'far'],
  frameBounds: ['margin', 'aspect'],
  play: ['loop', 'speed', 'time', 'fade', 'layer', 'weight', 'join', 'sync'],
  stop: ['layer', 'fade'],
  addLight: ['type', 'position', 'direction', 'color', 'intensity', 'radius', 'innerAngle', 'outerAngle', 'parent', 'castShadow', 'size'],
  addEmitter: ['rate', 'lifetime', 'size', 'sizeEnd', 'speed', 'direction', 'spread', 'radius', 'acceleration', 'drag',
    'color', 'colorEnd', 'texture', 'blend', 'layer', 'position', 'parent'],
  addDecal: ['texture', 'size', 'color', 'position', 'parent'],
  addProbe: ['size', 'fade', 'position', 'parent'],
  addSplats: ['splats', 'position', 'parent'],
  addSprite: ['texture', 'position', 'parent', 'size', 'pixels', 'color', 'rect', 'pivot', 'angle', 'facing', 'blend',
    'cutoff', 'layer', 'animation', 'lit'],
  spriteSheet: ['columns', 'rows', 'count', 'first'],
  addText: ['font', 'position', 'parent', 'text', 'size', 'color', 'align', 'pivot', 'lineHeight', 'width', 'facing',
    'pixels', 'layer', 'lit', 'blend', 'stroke', 'strokeWidth'],
  addShape: ['position', 'parent', 'shape', 'size', 'radius', 'color', 'stroke', 'strokeWidth', 'pivot', 'layer', 'blend', 'lit'],
  addPath: ['position', 'parent', 'points', 'closed', 'color', 'stroke', 'strokeWidth', 'layer', 'blend', 'lit'],
  addTilemap: ['position', 'parent', 'tileset', 'tileSize', 'columns', 'rows', 'tiles', 'margin', 'spacing', 'layer',
    'color', 'pivot', 'blend', 'lit'],
  Camera2D: ['position', 'pivot', 'zoom', 'angle', 'background', 'pixelSnap', 'ambient'],
  OrbitController: ['distance', 'yaw', 'pitch', 'target', 'minDistance', 'maxDistance', 'rotateSpeed', 'zoomSpeed', 'panSpeed', 'damping'],
  StatsOverlay: ['interval', 'parent'],
  'benchmark.run': ['frames', 'warmup', 'update'],
  Environment: ['size', 'irradianceSize', 'prefilterMips', 'label', 'sky', 'map'],
  'Environment sky': ['ground', 'horizon', 'zenith', 'sun', 'sunColor', 'sunIntensity', 'glow'],
  parseHDR: ['maxDimension'],
};

// A kind's setX takes the names its addX does, less where it is, which is the node's.
for (const kind of ['Light', 'Emitter', 'Decal', 'Probe', 'Sprite', 'Text', 'Shape', 'Path', 'Tilemap']) {
  OPTIONS[`set${kind}`] = OPTIONS[`add${kind}`].filter((key) => key !== 'position' && key !== 'parent');
}

/**
 * Names a call takes that are not options to list: old names that throw
 * saying what they are called now (the call's own check, which says more than
 * a guess would), and the engine's own, which no user passes.
 */
const QUIET = {
  run: ['overlay'],
  renderFrame: ['overlay'],
  play: ['add'],
  Camera2D: ['rotation', 'anchor'],
  addText: ['anchor'],
  setText: ['anchor'],
  addSprite: ['rotation'],
  setSprite: ['rotation'],
  addProbe: ['blend', 'min', 'max'],
  setProbe: ['blend', 'min', 'max'],
  setLight: ['position', 'parent'],
  Environment: ['capture'],
};

/** Each caller and name warned about already: once each is enough, and a loop would flood the console. */
const warned = new Set();

/**
 * Warn about each name in `options` that `caller` does not take, once.
 * `path` names a nested object, as in `create: unknown option 'shadows.sise'`.
 * Anything that is not a plain object passes: the call's own checks say what
 * it should be.
 */
export function checkOptions(caller, options, path = '', list = caller) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) return;
  const known = OPTIONS[list];
  for (const key of Object.keys(options)) {
    if (known.includes(key) || options[key] === undefined || QUIET[list]?.includes(key)) continue;
    const id = `${caller} ${path}${key}`;
    if (warned.has(id)) continue;
    warned.add(id);
    const near = nearest(key, known);
    const hint = list.startsWith('set') && (key === 'position' || key === 'parent')
      ? ` (that is the node's: use node.set${key[0].toUpperCase()}${key.slice(1)})`
      : near ? ` (did you mean '${path}${near}'?)` : '';
    console.warn(`${caller}: unknown option '${path}${key}'${hint}; it is ignored`);
  }
}

/**
 * The known name closest to `key`, or null if none is close enough to be a
 * slip: within a third of its length in single-letter edits (at least one),
 * ignoring case, so `castShadows` finds `castShadow` and `colour` finds `color`.
 */
export function nearest(key, known) {
  let best = null;
  let bestDistance = Infinity;
  const lower = key.toLowerCase();
  for (const name of known) {
    const d = editDistance(lower, name.toLowerCase());
    if (d < bestDistance) { best = name; bestDistance = d; }
  }
  return bestDistance <= Math.max(1, Math.floor(key.length / 3)) ? best : null;
}

/** Levenshtein distance: insertions, deletions and substitutions, one row at a time. */
function editDistance(a, b) {
  let row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const next = [i];
    for (let j = 1; j <= b.length; j++) {
      next[j] = Math.min(row[j] + 1, next[j - 1] + 1, row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    row = next;
  }
  return row[b.length];
}
