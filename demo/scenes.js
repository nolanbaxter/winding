// The demo's scenes: the README's, live. Each is a plain function of the
// engine's public API -- read one to see how a scene like it is made.
//
// A scene's build(ctx) returns { scene, camera, frame?, hud?, orbit? }:
// `frame(dt, time)` runs once a frame before drawing, `hud` is run's hud
// option, and `orbit` is where a 3D camera starts, for the page's orbit
// controls. ctx carries the engine and a few helpers, below in index.html.

import { Camera, Camera2D, spriteSheet } from '../src/winding.js';
import { buildDemoGLB, buildFeatureGLB } from '../test/fixtures/demoModel.js';

const KHRONOS = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models';
const HELMET = `${KHRONOS}/DamagedHelmet/glTF-Binary/DamagedHelmet.glb`;
const SPONZA = `${KHRONOS}/Sponza/glTF/Sponza.gltf`;

/** A seeded generator, so a scene comes out the same every time. */
function seeded(seed) {
  return () => ((seed = (Math.imul(seed ^ (seed >>> 15), 0x2c1b3c6d) + 0x6d2b79f5) >>> 0) / 4294967296);
}

/** '#rrggbb' of an [r, g, b] 0..255, each channel nudged by up to `jitter`. */
function hex([r, g, b], jitter = 0, random = Math.random) {
  const n = (random() * 2 - 1) * jitter;
  return `#${[r, g, b].map((c) => Math.max(0, Math.min(255, Math.round(c + n))).toString(16).padStart(2, '0')).join('')}`;
}

/** A directional light from a premultiplied colour: its brightest channel is the intensity. */
function sun(scene, direction, premultiplied) {
  const intensity = Math.max(...premultiplied);
  return scene.addLight({ type: 'directional', direction, color: premultiplied.map((c) => c / intensity), intensity });
}

/** Where an orbit starts that puts the camera at `position`, looking at `target`. */
function orbitFrom(position, target) {
  const d = position.map((v, i) => v - target[i]);
  const distance = Math.hypot(...d);
  return { distance, yaw: Math.atan2(d[0], d[2]), pitch: Math.asin(d[1] / distance), target };
}

const camera3D = (near = 0.1) => new Camera({ fovY: Math.PI / 3, near });

/**
 * A 2D camera that fits a `width` x `height` stretch of the world to the
 * canvas, centred on `center`, whatever the window's size: its zoom set
 * every frame from the canvas.
 */
function fitted(ctx, width, height, center, options = {}) {
  const camera = new Camera2D({ pivot: [0.5, 0.5], position: center, ...options });
  const frame = () => {
    const { width: w, height: h, pixelRatio } = ctx.engine.gpu;
    camera.zoom = Math.min(w / pixelRatio / width, h / pixelRatio / height);
  };
  return { camera, frame };
}

export const SCENES = [
  {
    id: 'effects',
    title: 'Effects',
    blurb: 'Embers from GPU particles rise past a helmet, over a scorch decal. Fog and depth of field are on, and the wordmark is text in the scene.',
    async build(ctx) {
      const { engine } = ctx;
      engine.renderer.exposure = 0.55;
      engine.renderer.fog = { visibility: 250 };
      engine.renderer.dof = { focusDistance: 5, fStop: 1.2 };
      const scene = engine.createScene();
      scene.add(await ctx.load(buildFeatureGLB({ baseColorFactor: [0.62, 0.58, 0.52, 1], roughnessFactor: 0.8 }), 'floor-warm'))
        .setAxisAngle([1, 0, 0], -Math.PI / 2).setScale(30, 30, 1);
      const helmet = await ctx.load(HELMET);
      scene.add(helmet).setPosition(0, 1, 0);
      scene.add(helmet).setPosition(-5, 1, -8);
      scene.add(helmet).setPosition(4, 1, -15);
      sun(scene, [-0.4, -0.7, -0.45], [2.4, 2.2, 2.0]);

      const art = new OffscreenCanvas(256, 256);
      const g = art.getContext('2d');
      const burn = g.createRadialGradient(128, 128, 8, 128, 128, 128);
      burn.addColorStop(0, 'rgba(12, 8, 5, 0.97)');
      burn.addColorStop(0.62, 'rgba(24, 12, 6, 0.9)');
      burn.addColorStop(0.72, 'rgba(70, 30, 10, 0.75)');
      burn.addColorStop(1, 'rgba(70, 30, 10, 0)');
      g.fillStyle = burn;
      g.fillRect(0, 0, 256, 256);
      scene.addDecal({ texture: await ctx.texture(art), size: [4.4, 4.4, 1] }).setAxisAngle([1, 0, 0], -Math.PI / 2);
      scene.addEmitter({
        position: [0, 0.1, -0.6], rate: 160, lifetime: [1.5, 3], size: 0.12, sizeEnd: 0.02, speed: [0.4, 1.1],
        spread: 0.7, radius: 1.2, acceleration: [0, 0.6, 0], drag: 0.3, color: [14, 5, 1.2, 1], colorEnd: [4, 0.4, 0, 0],
      });
      // The wordmark: the logo as a sprite beside the word as text, one piece.
      const font = await ctx.font("700 64px ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace");
      const px = 0.32 / 54;
      const mark = [-70 * px, 2.25, 0];
      scene.addText({ font, text: 'Winding', size: 54 * px, pivot: [0, 0.5], position: mark });
      const side = 83.2 * px;
      scene.addSprite({ texture: await ctx.logo(), size: [side, side], pivot: [0.5 + (55.4 * px) / side, 0.5], position: mark });
      return { scene, camera: camera3D(), orbit: orbitFrom([1.2, 1.7, 4.8], [0, 1.25, 0]) };
    },
  },
  {
    id: 'sponza',
    title: 'Sponza',
    blurb: 'Crytek Sponza, sunlit, with cascaded shadows and image-based light. A large download the first time, from Khronos.',
    async build(ctx) {
      const scene = ctx.engine.createScene();
      scene.add(await ctx.load(SPONZA));
      sun(scene, [-0.55, -0.82, -0.12], [5.2, 4.7, 4.0]);
      return { scene, camera: camera3D(), orbit: { ...orbitFrom([-7, 0.6, 0], [8, 2.2, 0]), spin: false } };
    },
  },
  {
    id: 'lights',
    title: '576 lights',
    blurb: 'Clustered lighting: the view is cut into cells, each lists the lights that reach it, and a pixel shades with its cell\'s few.',
    async build(ctx) {
      ctx.engine.renderer.exposure = 0.5;
      const scene = ctx.engine.createScene();
      scene.add(await ctx.load(buildDemoGLB({ arms: 6 }), 'demo-model'));
      sun(scene, [-0.35, -0.55, -0.45], [0.05, 0.05, 0.07]);
      const N = 24;
      const lights = [];
      for (let x = 0; x < N; x++) for (let z = 0; z < N; z++) {
        const u = x / (N - 1), v = z / (N - 1), hue = u * 0.6 + v * 0.4;
        const home = [(u - 0.5) * 24, -1.1 + Math.sin(u * 9) * 0.25 + Math.cos(v * 7) * 0.25, (v - 0.5) * 24];
        const node = scene.addLight({
          position: home,
          color: [0.5 + 0.5 * Math.cos(hue * 6.283), 0.5 + 0.5 * Math.cos((hue + 0.33) * 6.283), 0.5 + 0.5 * Math.cos((hue + 0.66) * 6.283)],
          intensity: 2.6, radius: 1.35,
        });
        lights.push([node, home, u * 9 + v * 5]);
      }
      // Each light bobs on its own phase.
      const frame = (dt, time) => {
        for (const [node, [x, y, z], phase] of lights) node.setPosition(x, y + 0.3 * Math.sin(time * 1.3 + phase), z);
      };
      return { scene, camera: camera3D(), frame, orbit: orbitFrom([7.5, 2.2, 9], [-1, -0.5, -1]) };
    },
  },
  {
    id: 'helmets',
    title: '121 helmets',
    blurb: 'GPU-driven drawing: 121 helmets in two draw calls. A compute pass culls them and writes the counts; the CPU never learns which survived.',
    async build(ctx) {
      const scene = ctx.engine.createScene();
      const asset = await ctx.load(HELMET);
      sun(scene, [-0.35, -0.55, -0.45], [0.5, 0.5, 0.55]);
      const N = 11;
      for (let x = 0; x < N; x++) for (let z = 0; z < N; z++) scene.add(asset).setPosition((x - N / 2) * 2.6, 0, (z - N / 2) * 2.6);
      return { scene, camera: camera3D(), orbit: orbitFrom([9.5, 5.5, 13], [-1, -0.4, -1]) };
    },
  },
  {
    id: 'fox',
    title: 'Fox',
    blurb: 'Skinning and animation: the fox walks, then cross-fades to a run and back, every few seconds.',
    async build(ctx) {
      const scene = ctx.engine.createScene();
      const fox = scene.add(await ctx.load(`${KHRONOS}/Fox/glTF-Binary/Fox.glb`));
      const min = new Float32Array(3), max = new Float32Array(3);
      scene.bounds(min, max);
      const span = Math.max(max[0] - min[0], max[2] - min[2]);
      scene.add(await ctx.load(buildFeatureGLB({ baseColorFactor: [0.55, 0.55, 0.52, 1], roughnessFactor: 0.9 }), 'floor-grey'))
        .setAxisAngle([1, 0, 0], -Math.PI / 2)
        .setScale(span * 6 / 3.2, span * 6 / 3.2, 1)
        .setPosition((min[0] + max[0]) / 2, min[1], (min[2] + max[2]) / 2);
      sun(scene, [-0.45, -0.75, -0.5], [3.4, 3.2, 2.9]);
      const c = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
      fox.play('Walk');
      let running = false, next = 4;
      const frame = (dt, time) => {
        if (time < next) return;
        running = !running;
        fox.play(running ? 'Run' : 'Walk', { fade: 0.5 });
        next = time + (running ? 3 : 4);
      };
      return {
        scene, camera: camera3D(span * 0.01), frame,
        orbit: orbitFrom([c[0] + span * 0.78, c[1] + span * 0.26, c[2] + span * 0.52], [c[0], c[1] * 0.8, c[2]]),
      };
    },
  },
  {
    id: 'splats',
    title: 'Gaussian splats',
    blurb: 'A cactus captured with 3D Gaussian Splatting: 452,000 soft ellipsoids, culled and sorted back to front on the GPU whenever the view moves. Scan by Steam Studio (steam-studio.jp), CC0. A 14 MB download the first time.',
    async build(ctx) {
      const scene = ctx.engine.createScene();
      // The capture is y down, as the photographs were: turned upright.
      scene.addSplats({ splats: await ctx.splats('cactus.splat') }).setAxisAngle([1, 0, 0], Math.PI);
      return { scene, camera: camera3D(0.05), orbit: orbitFrom([0, 1.6, 3.2], [0, 0.7, 0]) };
    },
  },
  {
    id: 'materials',
    title: 'Materials',
    blurb: 'Khronos\'s toy car: a clear-coated body, sheen on the seat fabric, and a windscreen that transmits and refracts what\'s behind it.',
    async build(ctx) {
      const scene = ctx.engine.createScene();
      scene.add(await ctx.load(`${KHRONOS}/ToyCar/glTF/ToyCar.gltf`));
      const min = new Float32Array(3), max = new Float32Array(3);
      scene.bounds(min, max);
      const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
      const car = [(min[0] + max[0]) / 2, max[1] - span * 0.11, (min[2] + max[2]) / 2];
      sun(scene, [-0.45, -0.75, -0.5], [3.4, 3.2, 2.9]);
      return {
        scene, camera: camera3D(span * 0.005),
        orbit: orbitFrom([car[0], car[1] + span * 0.16, car[2] + span * 0.6], car),
      };
    },
  },
  {
    id: 'shadows',
    title: 'Alpha shadows',
    blurb: 'A cut-out lattice casts the shape of its holes, and a tinted pane a lighter shadow, as a sun swings round behind them.',
    async build(ctx) {
      const scene = ctx.engine.createScene();
      scene.add(await ctx.load(buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.78, 1], roughnessFactor: 0.9 }), 'floor-light'))
        .setAxisAngle([1, 0, 0], -Math.PI / 2).setScale(5, 5, 1);
      const tex = new OffscreenCanvas(256, 256);
      const t = tex.getContext('2d');
      t.fillStyle = '#c9803a';
      t.fillRect(0, 0, 256, 256);
      t.globalCompositeOperation = 'destination-out';
      for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) { t.beginPath(); t.arc(32 + x * 64, 32 + y * 64, 24, 0, Math.PI * 2); t.fill(); }
      const lattice = await dataURL(await tex.convertToBlob({ type: 'image/png' }));
      scene.add(await ctx.load(buildFeatureGLB({ baseColorFactor: [1, 1, 1, 1], alphaMode: 'MASK', alphaCutoff: 0.5, imageURI: lattice }), 'lattice'))
        .setScale(0.55, 0.55, 1).setPosition(-1, 0.88, -0.6);
      scene.add(await ctx.load(buildFeatureGLB({ baseColorFactor: [0.45, 0.75, 1, 0.45], alphaMode: 'BLEND', roughnessFactor: 0.1 }), 'pane'))
        .setScale(0.55, 0.55, 1).setPosition(1, 0.88, -0.6);
      const light = sun(scene, [0, -0.62, 0.6], [3.4, 3.2, 2.9]);
      const frame = (dt, time) => {
        const azimuth = Math.sin(time * 0.5) * 0.75;
        light.setDirection(Math.sin(azimuth) * 0.6, -0.62, Math.cos(azimuth) * 0.6);
      };
      return { scene, camera: camera3D(), frame, orbit: { ...orbitFrom([0, 2.2, 4.6], [0, 0.55, -0.1]), spin: false } };
    },
  },
  {
    id: 'hud',
    title: 'HUD over 3D',
    blurb: 'A second, 2D scene drawn over the finished 3D frame: shapes, paths and text, in exact colours and crisp at any size.',
    async build(ctx) {
      const { engine } = ctx;
      const scene = engine.createScene();
      scene.add(await ctx.load(HELMET));
      sun(scene, [-0.4, -0.7, -0.45], [3.4, 3.2, 2.9]);
      const font = await ctx.font('600 40px system-ui, sans-serif');
      const hud = engine.createScene();
      const ink = [0.93, 0.94, 0.96, 1], accent = [0.85, 0.47, 0.34, 1], glass = [0.04, 0.05, 0.07, 0.62];
      hud.addShape({ size: [158, 46], radius: 10, color: glass, stroke: [1, 1, 1, 0.14], strokeWidth: 1, pivot: [0, 0], position: [12, 12] });
      hud.addText({ font, text: 'HULL', size: 11, pivot: [0, 0], color: ink, position: [22, 19] });
      hud.addShape({ size: [138, 10], radius: 5, color: [1, 1, 1, 0.12], pivot: [0, 0], position: [22, 38] });
      const health = hud.addShape({ size: [138, 10], radius: 5, color: accent, pivot: [0, 0], position: [22, 38] });
      // The minimap, crosshair and label, anchored to the right, the middle
      // and the bottom: placed each frame, so they follow the window.
      const minimap = hud.createNode();
      hud.addShape({ shape: 'ellipse', size: [84, 84], color: glass, stroke: accent, strokeWidth: 2, parent: minimap });
      const route = [[-26, 18], [-14, 4], [-18, -12], [0, -22], [14, -8], [8, 8], [24, 14]];
      hud.addPath({ points: route, closed: false, color: [0, 0, 0, 0], stroke: [1, 1, 1, 0.55], strokeWidth: 2, parent: minimap });
      const marker = hud.addShape({ shape: 'ellipse', size: [8, 8], color: accent, stroke: ink, strokeWidth: 1.5, parent: minimap });
      hud.addPath({ points: [[0, -5], [4, 4], [0, 2], [-4, 4]], color: ink, position: [0, -30], parent: minimap });
      const crosshair = hud.createNode();
      hud.addShape({ shape: 'ellipse', size: [22, 22], color: [0, 0, 0, 0], stroke: [1, 1, 1, 0.75], strokeWidth: 1.5, parent: crosshair });
      for (const [x, y] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        hud.addPath({ points: [[x * 15, y * 15], [x * 21, y * 21]], closed: false, color: [0, 0, 0, 0], stroke: [1, 1, 1, 0.75], strokeWidth: 1.5, parent: crosshair });
      }
      const label = hud.createNode();
      hud.addText({ font, text: 'DAMAGED HELMET', size: 12, pivot: [0, 1], color: ink, position: [0, -4], parent: label });
      hud.addPath({ points: [[0, 0], [104, 0]], closed: false, color: [0, 0, 0, 0], stroke: accent, strokeWidth: 2, parent: label });

      const lengths = route.slice(1).map((p, i) => Math.hypot(p[0] - route[i][0], p[1] - route[i][1]));
      const total = lengths.reduce((a, b) => a + b);
      const along = (t) => {
        let d = t * total;
        for (let i = 0; i < lengths.length; i++) {
          if (d <= lengths[i]) { const k = d / lengths[i]; return [route[i][0] + (route[i + 1][0] - route[i][0]) * k, route[i][1] + (route[i + 1][1] - route[i][1]) * k]; }
          d -= lengths[i];
        }
        return route[route.length - 1];
      };
      const view = new Camera2D();
      const frame = (dt, time) => {
        const { width, height, pixelRatio } = engine.gpu;
        // Laid out 280 units tall, scaled as a whole to the canvas.
        view.zoom = height / pixelRatio / 280;
        const W = width / pixelRatio / view.zoom, H = 280;
        minimap.setPosition(W - 52, H - 52);
        crosshair.setPosition(W / 2, H / 2);
        label.setPosition(14, H - 12);
        const a = time * 0.8;
        hud.setShape(health, { size: [138 * (0.45 + 0.3 * (0.5 + 0.5 * Math.cos(a))), 10] });
        const [x, y] = along(0.5 - 0.5 * Math.cos(a));
        marker.setPosition(x, y);
      };
      return { scene, camera: camera3D(), frame, hud: { scene: hud, camera: view }, orbit: orbitFrom([0, 0.6, 2.4], [0, -0.05, 0]) };
    },
  },
  {
    id: 'dungeon',
    title: '2D dungeon',
    blurb: 'Pixel art from a tilemap, lit by two flickering torches and a staff, with embers from the same GPU particles 3D uses. Every texel is drawn in code.',
    kind: '2d',
    async build(ctx) {
      const random = seeded(7);
      const tileset = await ctx.pixels(64, 16, (set) => {
        for (let y = 0; y < 16; y++) {
          for (let x = 0; x < 16; x++) {
            const seam = x % 8 === 0 || y % 8 === 0;
            set(x, y, seam ? '#4a4e58' : hex([122, 126, 138], 12, random));
            const moss = random() < 0.35 + 0.3 * Math.sin(x * 0.7 + y * 0.4);
            set(16 + x, y, seam ? '#4a4e58' : moss ? hex([96, 138, 84], 14, random) : hex([122, 126, 138], 12, random));
            const row = Math.floor(y / 4), mortar = y % 4 === 3 || (x + (row % 2) * 4) % 8 === 7;
            set(32 + x, y, mortar ? '#4a3b33' : hex([168, 112, 86], 16, random));
            set(48 + x, y, y > 12 ? '#2a221d' : y > 10 ? '#5c4638' : mortar ? '#4a3b33' : hex([178, 122, 94], 16, random));
          }
        }
      });
      const torch = await ctx.pixels(32, 16, (set) => {
        for (let f = 0; f < 2; f++) {
          for (let y = 10; y < 16; y++) set(f * 16 + 7, y, '#4a4a52'), set(f * 16 + 8, y, '#33333a');
          set(f * 16 + 6, 10, '#4a4a52'); set(f * 16 + 9, 10, '#4a4a52');
          const flame = f === 0 ? [[7, 3], [6, 5], [6, 7], [6, 8], [8, 4], [8, 6], [9, 7], [9, 8], [7, 9], [8, 9]] : [[8, 3], [7, 4], [6, 6], [6, 8], [9, 5], [9, 7], [9, 8], [7, 9], [8, 9], [7, 2]];
          for (const [x, y] of flame) set(f * 16 + x, y, '#ff8a2a');
          for (let y = 5; y < 10; y++) set(f * 16 + 7, y, '#ffd35a'), set(f * 16 + 8, y, y > 6 ? '#ffd35a' : '#ff8a2a');
          set(f * 16 + 7, 8, '#fff4c2'); set(f * 16 + 8, 8, '#fff4c2');
        }
      });
      const hero = await ctx.pixels(64, 16, (set) => {
        for (let f = 0; f < 4; f++) {
          const bob = f === 1 || f === 2 ? 1 : 0, sway = f === 2 ? 1 : f === 0 ? -1 : 0;
          const o = f * 16;
          for (let y = 3 + bob; y < 14; y++) {
            const half = y < 7 + bob ? 3 : 4 + Math.min(1, y - 10 - bob);
            for (let x = 8 - half + (y > 10 ? sway : 0); x < 8 + half + (y > 10 ? sway : 0); x++) set(o + x, y, y < 7 + bob ? '#5b86d0' : '#4870b8');
          }
          set(o + 6, 5 + bob, '#f4e2c4'); set(o + 9, 5 + bob, '#f4e2c4');
          set(o + 6, 14, '#1d1d24'); set(o + 7, 14, '#1d1d24'); set(o + 9, 14, '#1d1d24'); set(o + 10, 14, '#1d1d24');
          for (let y = 6 + bob; y < 13; y++) set(o + 12 + sway, y, '#7a5a38');
          set(o + 12 + sway, 5 + bob, '#9fd7ff');
        }
      });
      const scene = ctx.engine.createScene();
      const columns = 16, rows = 11, mapW = columns * 16, mapH = rows * 16;
      const ids = [];
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < columns; x++) {
          const wall = y === 0 || x === 0 || x === columns - 1;
          ids.push(y === 1 && !wall ? 4 : wall ? 3 : random() < 0.3 ? 2 : 1);
        }
      }
      scene.addTilemap({ tileset, tileSize: [16, 16], columns, rows, tiles: ids, lit: true });
      const flames = [[Math.round(mapW * 0.27), 24], [Math.round(mapW * 0.73), 24]];
      const lights = flames.map(([x, y]) => {
        scene.addSprite({ texture: torch, position: [x, y + 2], animation: { frames: spriteSheet({ columns: 2 }), fps: 1 / 0.21 }, layer: 2 });
        scene.addEmitter({
          position: [x, y - 3], rate: 18, lifetime: [0.8, 1.6], size: 1.2, sizeEnd: 0.4, speed: [4, 12], direction: [0, -1], spread: 0.5,
          acceleration: [0, -8], color: [1, 0.75, 0.35, 1], colorEnd: [0.9, 0.25, 0, 0], layer: 3,
        });
        return scene.addLight({ position: [x, y], color: [1, 0.62, 0.3], radius: 110 });
      });
      scene.addSprite({ texture: hero, position: [mapW / 2, 90], lit: true, layer: 1, animation: { frames: spriteSheet({ columns: 4 }), fps: 12 / 1.68 } });
      scene.addLight({ position: [mapW / 2 + 4, 85], color: [0.45, 0.75, 1], intensity: 1.4, radius: 46 });
      const { camera, frame: fit } = fitted(ctx, mapW, mapH, [mapW / 2, mapH / 2], {
        pixelSnap: true, ambient: [0.12, 0.12, 0.2], background: [0.02, 0.02, 0.03, 1],
      });
      const frame = (dt, time) => {
        fit();
        lights.forEach((light, k) => scene.setLight(light, { intensity: 2.4 + 0.35 * Math.sin(time * 5.2 + k * 2.1) + 0.15 * Math.sin(time * 7.9 + k) }));
      };
      return { scene, camera, frame };
    },
  },
  {
    id: 'platformer',
    title: '2D platformer',
    blurb: 'A pixel-art tilemap with a pit and a ledge, hills as filled paths, clouds and coins as shapes, and a score and hearts over it all.',
    kind: '2d',
    async build(ctx) {
      const random = seeded(11);
      const tileset = await ctx.pixels(64, 16, (set) => {
        for (let y = 0; y < 16; y++) {
          for (let x = 0; x < 16; x++) {
            const blade = 3 + ((x * 7) % 3 === 0 ? 1 : 0);
            set(x, y, y < blade ? hex(y < 1 ? [132, 206, 92] : [96, 176, 70], 10, random) : random() < 0.06 ? '#8a6a4a' : hex([122, 84, 52], 10, random));
            set(16 + x, y, random() < 0.07 ? '#8a6a4a' : hex([118, 80, 50], 10, random));
            const edge = x === 0 || y === 0 ? '#c9ccd4' : x === 15 || y === 15 ? '#5d626e' : null;
            set(32 + x, y, edge ?? hex([146, 150, 160], 8, random));
            const plank = y % 5 === 4 || x === 0 || x === 15 || y === 0 || y === 15 || Math.abs(x - y) < 1 || Math.abs(x + y - 15) < 1;
            set(48 + x, y, plank ? '#6b4524' : hex([176, 122, 70], 10, random));
          }
        }
      });
      const heroArt = await ctx.pixels(16, 16, (set) => {
        for (let y = 2; y < 9; y++) for (let x = 2; x < 14; x++) if ((x - 7.5) ** 2 / 36 + (y - 8) ** 2 / 36 < 1) set(x, y, '#d8403a');
        for (const [x, y] of [[5, 4], [9, 3], [11, 6], [4, 7]]) set(x, y, '#fff4e8');
        for (let y = 9; y < 15; y++) for (let x = 4; x < 12; x++) set(x, y, '#f2dcc0');
        set(6, 11, '#2a1d18'); set(9, 11, '#2a1d18');
        for (let x = 4; x < 12; x++) set(x, 15, '#3a2a22');
      });
      const scene = ctx.engine.createScene();
      const columns = 27, rows = 15, ids = new Array(columns * rows).fill(0);
      const put = (x, y, id) => { ids[y * columns + x] = id; };
      for (let x = 0; x < columns; x++) {
        if (x === 9 || x === 10) continue;
        const top = x >= 20 ? 10 : 12;
        put(x, top, 1);
        for (let y = top + 1; y < rows; y++) put(x, y, 2);
      }
      for (let x = 13; x <= 17; x++) put(x, 8, 3);
      put(5, 11, 4); put(6, 11, 4); put(6, 10, 4);
      scene.addTilemap({ tileset, tileSize: [16, 16], columns, rows, tiles: ids });
      scene.addShape({ shape: 'ellipse', size: [70, 70], color: [1, 0.95, 0.6, 0.25], position: [352, 44], layer: -8 });
      scene.addShape({ shape: 'ellipse', size: [40, 40], color: [1, 0.93, 0.55, 1], position: [352, 44], layer: -8 });
      // Clouds, each a node drifting to the right and round again.
      const clouds = [];
      for (const [cx, cy, s] of [[70, 50, 1], [230, 34, 0.8], [300, 92, 0.6]]) {
        const cloud = scene.createNode().setPosition(cx, cy);
        for (const [dx, dy, w] of [[-14, 4, 26], [0, -4, 34], [16, 3, 28], [4, 8, 40]]) {
          scene.addShape({ shape: 'ellipse', size: [w * s, w * s * 0.7], color: [1, 1, 1, 0.95], position: [dx * s, dy * s], parent: cloud, layer: -7 });
        }
        clouds.push([cloud, cx, cy, 6 + s * 6]);
      }
      const hills = (base, height, waves, phase) => [
        ...Array.from({ length: 49 }, (_, i) => { const x = -10 + i * 9.2; return [x, base - height * (0.5 + 0.5 * Math.sin(x / 400 * Math.PI * 2 * waves + phase))]; }),
        [432, 240], [-10, 240],
      ];
      scene.addPath({ points: hills(170, 50, 2.2, 0.4), color: [0.62, 0.82, 0.8, 1], layer: -6 });
      scene.addPath({ points: hills(196, 36, 3.1, 2.1), color: [0.4, 0.68, 0.5, 1], stroke: [0.3, 0.55, 0.38, 1], strokeWidth: 1.5, layer: -5 });
      const coins = [];
      for (let i = 0; i < 6; i++) {
        const t = i / 5;
        coins.push([scene.addShape({ shape: 'ellipse', size: [7, 9], color: [1, 0.8, 0.2, 1], stroke: [0.72, 0.46, 0.08, 1], strokeWidth: 1.4, position: [120 + t * 70, 150 - Math.sin(t * Math.PI) * 34], layer: 1 }), i]);
      }
      const hero = scene.addSprite({ texture: heroArt, pivot: [0.5, 1], position: [74, 192], layer: 2 });
      const font = await ctx.font("700 48px ui-monospace, Consolas, 'Liberation Mono', monospace");
      scene.addText({ font, text: 'SCORE 004200', size: 11, pivot: [0, 0], color: [1, 1, 1, 1], position: [16, 12], layer: 10 });
      const heart = Array.from({ length: 40 }, (_, i) => {
        const a = (i / 40) * Math.PI * 2;
        return [16 * Math.sin(a) ** 3 * 0.34, -(13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)) * 0.34];
      });
      for (let i = 0; i < 3; i++) {
        scene.addPath({ points: heart, color: i < 2 ? [0.9, 0.2, 0.25, 1] : [0, 0, 0, 0], stroke: [1, 1, 1, 1], strokeWidth: 1.2, position: [336 + i * 16, 20], layer: 10 });
      }
      const { camera, frame: fit } = fitted(ctx, 400, 225, [208, 124], { pixelSnap: true, background: [0.55, 0.8, 0.98, 1] });
      const frame = (dt, time) => {
        fit();
        for (const [cloud, cx, cy, speed] of clouds) cloud.setPosition(((cx + time * speed + 60) % 520) - 60, cy);
        for (const [coin, i] of coins) coin.setScale(Math.abs(Math.cos(time * 2.4 + i * 0.5)) * 0.85 + 0.15, 1);
        hero.setPosition(74, 192 - Math.abs(Math.sin(time * 3)) * 3);
      };
      return { scene, camera, frame };
    },
  },
  {
    id: 'street',
    title: '2D night street',
    blurb: 'Three lamps throw spot-light cones onto lit buildings and cobbles. Windows glow, fireflies drift, and a moon hangs over the skyline.',
    kind: '2d',
    async build(ctx) {
      const W = 1600, H = 900;
      const random = seeded(23);
      const scene = ctx.engine.createScene();
      const art = new OffscreenCanvas(128, 128);
      const g = art.getContext('2d');
      const fade = g.createRadialGradient(64, 64, 0, 64, 64, 64);
      fade.addColorStop(0, 'rgba(255,255,255,0.9)');
      fade.addColorStop(0.35, 'rgba(255,255,255,0.35)');
      fade.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = fade;
      g.fillRect(0, 0, 128, 128);
      const glow = await ctx.texture(art);
      for (let i = 0; i < 90; i++) {
        const s = 1.5 + random() * 2;
        scene.addShape({ shape: 'ellipse', size: [s, s], color: [1, 1, 0.95, 0.4 + random() * 0.6], position: [random() * W, random() * 420], layer: -10 });
      }
      scene.addSprite({ texture: glow, size: [300, 300], color: [0.75, 0.82, 1, 0.35], position: [1300, 150], layer: -9, blend: 'additive' });
      scene.addShape({ shape: 'ellipse', size: [92, 92], color: [0.96, 0.94, 0.84, 1], position: [1300, 150], layer: -9 });
      const skyline = [[0, 900]];
      for (let x = 0; x <= W; x += 60 + random() * 60) {
        const top = 380 + random() * 140;
        skyline.push([x, top], [x + 50 + random() * 40, top]);
      }
      skyline.push([W, 900]);
      scene.addPath({ points: skyline, color: [0.07, 0.08, 0.15, 1], layer: -8 });
      const buildings = [[40, 300, 360, [0.5, 0.34, 0.28]], [360, 260, 420, [0.4, 0.42, 0.5]], [640, 330, 330, [0.55, 0.45, 0.32]], [990, 300, 400, [0.36, 0.4, 0.36]], [1310, 260, 350, [0.5, 0.36, 0.34]]];
      for (const [x, w, h, colour] of buildings) {
        const top = 720 - h;
        scene.addShape({ size: [w, h], color: [...colour, 1], pivot: [0, 0], position: [x, top], lit: true, layer: -5 });
        scene.addPath({ points: [[x - 14, top], [x + w / 2, top - 70], [x + w + 14, top]], color: [0.22, 0.2, 0.24, 1], lit: true, layer: -5 });
        for (let wy = top + 40; wy < 640; wy += 90) {
          for (let wx = x + 34; wx < x + w - 50; wx += 80) {
            const glowing = random() < 0.45;
            scene.addShape({ size: [34, 50], radius: 3, color: glowing ? [1, 0.8, 0.45, 1] : [0.12, 0.13, 0.2, 1], position: [wx + 17, wy + 25], lit: !glowing, layer: -4 });
          }
        }
      }
      const cobbles = await ctx.pixels(16, 16, (set) => {
        for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
          const seam = (y % 8 === 0) || ((x + (Math.floor(y / 8) % 2) * 8) % 16 === 0);
          set(x, y, seam ? '#2a2a30' : hex([104, 100, 108], 14, random));
        }
      });
      scene.addTilemap({ tileset: cobbles, tileSize: [16, 16], columns: 34, rows: 4, tiles: new Array(34 * 4).fill(1), position: [0, 720], lit: true, layer: -3 }).setScale(3, 3, 1);
      scene.addShape({ size: [W, 10], color: [0.42, 0.42, 0.46, 1], pivot: [0, 1], position: [0, 724], lit: true, layer: -2 });
      for (const x of [280, 820, 1360]) {
        scene.addShape({ size: [12, 300], radius: 3, color: [0.16, 0.17, 0.2, 1], pivot: [0.5, 1], position: [x, 722], lit: true, layer: 0 });
        scene.addPath({ points: [[x, 440], [x + 46, 440]], closed: false, color: [0, 0, 0, 0], stroke: [0.16, 0.17, 0.2, 1], strokeWidth: 8, lit: true, layer: 0 });
        scene.addSprite({ texture: glow, size: [150, 150], color: [1, 0.8, 0.45, 0.55], position: [x + 46, 452], blend: 'additive', layer: 1 });
        scene.addShape({ size: [30, 20], radius: 6, color: [1, 0.88, 0.6, 1], position: [x + 46, 452], layer: 1 });
        scene.addLight({ position: [x + 46, 462], direction: [0, 1], color: [1, 0.78, 0.45], intensity: 2.2, radius: 460, innerAngle: 0.25, outerAngle: 0.62 });
        const flies = scene.addEmitter({ position: [x + 46, 560], rate: 0.2, lifetime: 30, size: 5, sizeEnd: 5, speed: [4, 10], spread: Math.PI, radius: 120, drag: 0.2, color: [0.85, 1, 0.45, 0.9], layer: 2 });
        scene.burst(flies, 7);
      }
      scene.addLight({ position: [1300, 150], color: [0.55, 0.65, 1], intensity: 0.35, radius: 1500 });
      scene.addShape({ size: [220, 60], color: [0.34, 0.33, 0.36, 1], pivot: [0, 1], position: [1080, 722], lit: true, layer: 3 });
      const cat = [[0, 0], [30, 0], [34, -18], [30, -40], [34, -58], [30, -70], [24, -60], [14, -60], [8, -70], [4, -58], [8, -40], [2, -20], [-6, -6], [-26, 4], [-30, -2], [-8, -12]];
      scene.addPath({ points: cat, color: [0.03, 0.03, 0.05, 1], position: [1150, 662], layer: 4 });
      const eyes = [12, 24].map((dx) => scene.addShape({ shape: 'ellipse', size: [5, 4], color: [0.9, 0.95, 0.4, 1], position: [1150 + dx, 662 - 52], layer: 5 }));
      const { camera, frame: fit } = fitted(ctx, W, H, [W / 2, H / 2], { ambient: [0.05, 0.06, 0.12], background: [0.03, 0.04, 0.09, 1] });
      const frame = (dt, time) => {
        fit();
        // The cat blinks now and then.
        const blink = (time % 4.3) < 0.14 ? 0.15 : 1;
        for (const eye of eyes) eye.setScale(1, blink);
      };
      return { scene, camera, frame };
    },
  },
  {
    id: 'chart',
    title: '2D sea chart',
    blurb: 'Concave islands with beaches, each three paths; a dotted route; a star compass; names and a legend in text. Paths and text, crisp at any size.',
    kind: '2d',
    async build(ctx) {
      const random = seeded(5);
      const font = await ctx.font("600 48px Georgia, 'Times New Roman', serif");
      const scene = ctx.engine.createScene();
      const ink = [0.18, 0.14, 0.1, 1], paper = [0.96, 0.92, 0.82, 1];
      for (let row = 0; row < 14; row++) {
        const y0 = 40 + row * 64, phase = random() * 6;
        scene.addPath({
          points: Array.from({ length: 21 }, (_, i) => [i * 80, y0 + 6 * Math.sin(i * 0.9 + phase)]), closed: false,
          color: [0, 0, 0, 0], stroke: [1, 1, 1, 0.1], strokeWidth: 2, layer: -5,
        });
      }
      const outline = (r, a, b, scale) => Array.from({ length: 72 }, (_, i) => {
        const t = (i / 72) * Math.PI * 2;
        const k = r * scale * (1 + 0.2 * Math.sin(3 * t + a) + 0.1 * Math.sin(7 * t + b));
        return [Math.cos(t) * k, Math.sin(t) * k * 0.8];
      });
      for (const [name, x, y, r] of [['Winding Isle', 520, 360, 210], ['Cluster Key', 1130, 270, 130], ['Forward Rock', 1020, 640, 95]]) {
        const a = random() * 6, b = random() * 6;
        scene.addPath({ points: outline(r, a, b, 1), color: [0.93, 0.84, 0.62, 1], stroke: [1, 1, 1, 0.5], strokeWidth: 3, position: [x, y] });
        scene.addPath({ points: outline(r, a, b, 0.82), color: [0.42, 0.62, 0.36, 1], stroke: [0.3, 0.48, 0.27, 1], strokeWidth: 2, position: [x, y], layer: 1 });
        scene.addPath({ points: outline(r, a + 1, b, 0.4), color: [0.32, 0.5, 0.29, 1], position: [x, y], layer: 2 });
        scene.addText({ font, text: name, size: 24, color: paper, pivot: [0.5, 0.5], position: [x, y + r * 0.95], layer: 5 });
      }
      const route = (t) => {
        const [p0, p1, p2] = [[640, 470], [1350, 520], [1040, 610]];
        const u = 1 - t;
        return [u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]];
      };
      for (let i = 0; i <= 40; i++) scene.addShape({ shape: 'ellipse', size: [7, 7], color: paper, position: route(i / 40), layer: 3 });
      // The ship sails the route, and back.
      const ship = scene.createNode().setPosition(640, 462);
      scene.addPath({ points: [[-22, 0], [22, 0], [14, 12], [-14, 12]], color: [0.45, 0.28, 0.16, 1], stroke: ink, strokeWidth: 2, parent: ship, layer: 4 });
      scene.addPath({ points: [[0, -2], [0, -38], [18, -6]], color: paper, stroke: ink, strokeWidth: 2, parent: ship, layer: 4 });
      const star = Array.from({ length: 16 }, (_, i) => {
        const t = (i / 16) * Math.PI * 2 - Math.PI / 2, k = i % 4 === 0 ? 64 : i % 2 === 0 ? 40 : 14;
        return [Math.cos(t) * k, Math.sin(t) * k];
      });
      scene.addShape({ shape: 'ellipse', size: [150, 150], color: [0, 0, 0, 0], stroke: paper, strokeWidth: 2, position: [1440, 760], layer: 3 });
      scene.addPath({ points: star, color: paper, stroke: ink, strokeWidth: 2, position: [1440, 760], layer: 4 });
      scene.addText({ font, text: 'N', size: 26, color: paper, position: [1440, 666], layer: 4 });
      scene.addShape({ size: [196, 214], radius: 14, color: [...paper.slice(0, 3), 0.96], stroke: ink, strokeWidth: 2, pivot: [0, 0], position: [60, 616], layer: 6 });
      scene.addText({ font, text: 'Legend', size: 28, color: ink, pivot: [0, 0], position: [84, 632], layer: 7 });
      scene.addPath({ points: [[0, 0], [148, 0]], closed: false, color: [0, 0, 0, 0], stroke: [...ink.slice(0, 3), 0.35], strokeWidth: 1.5, position: [84, 674], layer: 7 });
      const row = (k) => 702 + k * 36;
      const icon = 104, label = 140;
      scene.addPath({ points: [[-22, 0], [22, 0], [14, 12], [-14, 12]], color: [0.45, 0.28, 0.16, 1], stroke: ink, strokeWidth: 3, position: [icon, row(0) + 2], layer: 7 }).setScale(0.55, 0.55);
      scene.addPath({ points: [[0, -2], [0, -38], [18, -6]], color: paper, stroke: ink, strokeWidth: 3, position: [icon, row(0) + 2], layer: 7 }).setScale(0.55, 0.55);
      for (const dx of [-12, 0, 12]) scene.addShape({ shape: 'ellipse', size: [7, 7], color: ink, position: [icon + dx, row(1)], layer: 7 });
      scene.addShape({ size: [34, 22], radius: 6, color: [0.42, 0.62, 0.36, 1], stroke: [0.93, 0.84, 0.62, 1], strokeWidth: 4, position: [icon, row(2)], layer: 7 });
      scene.addShape({ size: [34, 22], radius: 6, color: [0.32, 0.5, 0.29, 1], position: [icon, row(3)], layer: 7 });
      for (const [k, name] of ['Ship', 'Route', 'Island', 'Hills'].entries()) {
        scene.addText({ font, text: name, size: 21, color: ink, pivot: [0, 0.5], position: [label, row(k)], layer: 7 });
      }
      const { camera, frame: fit } = fitted(ctx, 1600, 900, [800, 450], { background: [0.16, 0.42, 0.56, 1] });
      const frame = (dt, time) => {
        fit();
        const t = 0.5 - 0.5 * Math.cos(time * 0.25);
        const [x, y] = route(t);
        ship.setPosition(x, y - 8 + Math.sin(time * 2) * 1.5);
      };
      return { scene, camera, frame };
    },
  },
];

/** An image as a data URL, which a glTF built in memory can hold. */
function dataURL(blob) {
  return new Promise((ok) => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(blob); });
}
