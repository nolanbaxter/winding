// The types, held to the docs' own examples. Never run, only checked, in CI:
//   npx -y -p typescript@7 tsc --noEmit --strict --target es2022 --module nodenext --lib es2022,dom test/types.ts
// Each @ts-expect-error is a mistake the types must refuse; tsc fails if one is let through.

import {
  Winding, Camera, Camera2D, OrbitController, StatsOverlay, Environment, spriteSheet, parseHDR,
  colorFromHex, vec3Create, quatCreate, quatSetAxisAngle, mat4Create, mat4Invert,
} from '../src/winding.js';
import type { Node, Texture } from '../src/winding.js';
import { Benchmark } from '../src/bench.js';

declare const canvas: HTMLCanvasElement;

export async function threeD(): Promise<void> {
  const engine = await Winding.create(canvas, { ao: true, shadows: { size: 4096 }, onDeviceLost: () => location.reload() });
  const studio = await engine.loadEnvironment('studio.hdr', { size: 256 });
  const scene = engine.createScene({ environment: studio });
  const helmet = await engine.load('helmet.glb', { retainGeometry: true });
  const node: Node = scene.add(helmet).setPosition(0, 0, -3).setEuler(0.5, 0);
  node.play('Walk', { fade: 0.3 });
  node.animation?.layer('upper', { mask: 'Spine' });

  const camera = new Camera({ fovY: 0.8 });
  camera.position.set([0, 2, 6]);
  const orbit = new OrbitController(camera, canvas, { distance: 4, target: [0, 1, 0] });
  orbit.desired.yaw += Math.PI / 2;
  const stats = new StatsOverlay(engine);

  const lamp = scene.addLight({ position: [0, 3, 0], color: [1, 0.7, 0.4], intensity: 20, radius: 8 });
  scene.setLight(lamp, { intensity: 30, castShadow: true });
  const light = scene.lightOf(lamp);
  if (light?.type === 'spot') light.outerAngle?.toFixed();
  scene.addEmitter({ rate: 200, lifetime: [0.4, 0.8], size: 0.05, speed: [2, 4], acceleration: [0, -9.81, 0] });
  scene.addSplats({ splats: await engine.loadSplats('room.ply') }).setAxisAngle([1, 0, 0], Math.PI);

  engine.renderer.resolution = 0.75;
  engine.renderer.autoExposure = { min: -2, darken: 2 };
  engine.renderer.autoExposure = true;
  engine.renderer.softShadows = false;
  engine.renderer.taa = true;
  engine.renderer.splatCull = 0;
  engine.renderer.fog = { visibility: 200, scaleHeight: 20 };
  engine.renderer.post.strength = 0.15;
  engine.grading = { whiteBalance: 5000, lut: await engine.loadLUT('film.cube') };
  engine.grading = null;
  engine.debug.axes([0, 0, 0], 0.5).box([-1, 0, -1], [1, 2, 1], [1, 1, 0]);

  const hit = scene.pick(camera, 10, 10, 100, 100);
  if (hit) hit.node.setScale(1.2);
  const min = vec3Create(), max = vec3Create();
  if (scene.bounds(min, max)) orbit.frameBounds(min, max, { margin: 1.2 });

  const sky = new Environment(engine.gpu, { sky: { zenith: [0.05, 0.08, 0.2], sunIntensity: 30 } });
  const map = parseHDR(new Uint8Array(0), { maxDimension: engine.gpu.limits.maxTextureDimension2D });
  engine.createScene({ environment: new Environment(engine.gpu, { map }) });
  engine.unload(sky);

  engine.run(scene, camera, { frame: (alpha, clock) => { orbit.update(clock.realDelta); stats.update(clock.realDelta); } });
  const report = await new Benchmark(engine).run(scene, camera, { frames: 300 });
  console.log(Benchmark.format(report), report.cpu[0].p95);

  const q = quatSetAxisAngle(quatCreate(), [0, 1, 0], 1);
  node.setRotation(q);
  const inverse: Float32Array | null = mat4Invert(mat4Create(), camera.view);
  void inverse;

  // @ts-expect-error -- lifetime and size are required
  scene.addEmitter({ rate: 10 });
  // @ts-expect-error -- a light's type cannot change
  scene.setLight(lamp, { type: 'point' });
  // @ts-expect-error -- antialias is on or off
  engine.renderer.post.antialias = 'fxaa';
  // @ts-expect-error -- renamed to gpu in 1.0
  engine.rhi.device;
}

export async function twoD(engine: Winding): Promise<void> {
  const scene = engine.createScene();
  const camera = new Camera2D({ pivot: [0.5, 0.5], zoom: 3, pixelSnap: true });
  const hero: Texture = await engine.loadTexture('hero.png', { pixelated: true });
  const player = scene.addSprite({ texture: hero, animation: { frames: spriteSheet({ columns: 8 }), fps: 10 }, pivot: [0.5, 1] });
  scene.setSprite(player, { color: colorFromHex('#e03a2f') });
  const frame: number | undefined = scene.spriteOf(player)?.frame;
  void frame;

  const font = await engine.loadFont('600 32px system-ui, sans-serif');
  const hud = engine.createScene();
  hud.addText({ font, text: 'HULL', size: 11, pivot: [0, 0], position: [22, 19] });
  const health = hud.addShape({ size: [138, 10], radius: 5, color: [0.85, 0.47, 0.34, 1], pivot: [0, 0], position: [22, 38] });
  scene.setShape(health, { size: [69, 10] });

  const tiles = await engine.loadTexture('tiles.png', { pixelated: true });
  const level = scene.addTilemap({ tileset: tiles, tileSize: [16, 16], columns: 100, rows: 40 });
  scene.setTile(level, 12, 3, 5 | 0x80000000);
  const id: number = scene.tileAt(level, 12, 3) & 0x1fffffff;
  void id;

  const hit = scene.pick(camera, 1, 2, 300, 200);
  if (hit?.tile) scene.setTile(hit.node, hit.tile[0], hit.tile[1], 0);
  const target = await engine.createTarget({ size: [160, 160] });
  hud.addSprite({ texture: target, pivot: [0, 0], position: [16, 16] });
  engine.renderFrame(scene, camera, { target });
  engine.run(scene, camera, { hud: { scene: hud } });

  // @ts-expect-error -- a sprite needs a texture
  scene.addSprite({ size: [1, 1] });
  // @ts-expect-error -- not a shape
  scene.addShape({ shape: 'triangle', size: [1, 1] });
  // @ts-expect-error -- a 2D pick has no options
  scene.pick(camera, 1, 2, 300, 200, { maxDistance: 1 });
}
