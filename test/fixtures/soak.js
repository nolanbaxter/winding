// Nothing builds up: everything an app does, over and over, with every GPU
// buffer and texture counted from creation to destroy(), every pipeline and
// shader module counted as made (a cache that never forgets one is the leak),
// and -- when the page can force a collection (Chrome with
// --js-flags=--expose-gc) -- every engine, scene, model, texture and font a
// round lets go of checked to have been collected.
//
// Each scenario runs a few times to warm up, the counts are taken, it runs
// `rounds` more times, and they are taken again. Before each count the page
// keeps drawing for a while, as an app does: what the renderer holds from the
// last frame, or gives back after a quiet spell, is let go by then.
//
// Used by the GPU suite (gpu.test.js) and by tools pages that run it longer.

import { buildFeatureGLB, buildRiggedGLB } from './demoModel.js';

/**
 * Its scenarios, by name: each a check of its own in gpu.test.js, so CI can
 * run just those a change reaches, and spread them across runners.
 */
export const SCENARIOS = {
  models: 'a viewer swapping models',
  skinnedModels: 'skinned models loaded and unloaded',
  scenes: 'a level made and dropped',
  spawning: 'spawning and despawning',
  features: 'every feature on, then off',
  resizing: 'a window resized',
  lightBurst: 'a burst of shadowed lights',
  lights: 'lights coming and going',
  effects: 'sprites, text, particles and decals',
  changingText: 'a score rewritten every frame',
  environments: 'environments loaded and unloaded',
  targets: 'a minimap target made and dropped',
  probes: 'reflection probes placed and removed',
  levels2D: 'a 2D game changing levels',
  engines: 'whole engines, on a canvas kept for the next',
};

export async function soak({ Winding, Camera, Camera2D, rounds = 4, only = null, settleFrames = 130 }) {
  const collect = typeof globalThis.gc === 'function' ? globalThis.gc : null;

  // --- the counters, on every device, while this runs ---
  const live = new Map();
  const ids = new WeakMap();
  let nextId = 1;
  const collected = new FinalizationRegistry((id) => live.delete(id));
  const made = { pipelines: 0, shaders: 0 };
  const P = GPUDevice.prototype;
  const saved = {};
  const patch = (owner, name, wrap) => { saved[name] ??= []; saved[name].push([owner, owner[name]]); owner[name] = wrap(owner[name]); };
  const track = (object, label, device) => {
    const id = nextId++;
    ids.set(object, id);
    live.set(id, { ref: new WeakRef(object), label: (label || '(none)').replace(/[0-9]+/g, '#'), device: new WeakRef(device) });
    collected.register(object, id, object);
  };
  patch(P, 'createBuffer', (f) => function (d) { const o = f.call(this, d); track(o, d.label, this); return o; });
  patch(P, 'createTexture', (f) => function (d) { const o = f.call(this, d); track(o, d.label, this); return o; });
  for (const name of ['createRenderPipeline', 'createRenderPipelineAsync', 'createComputePipeline', 'createComputePipelineAsync']) {
    patch(P, name, (f) => function (d) { made.pipelines++; return f.call(this, d); });
  }
  patch(P, 'createShaderModule', (f) => function (d) { made.shaders++; return f.call(this, d); });
  for (const C of [GPUBuffer, GPUTexture]) {
    patch(C.prototype, 'destroy', (f) => function () {
      const id = ids.get(this);
      if (id !== undefined) { live.delete(id); collected.unregister(this); }
      return f.call(this);
    });
  }
  patch(P, 'destroy', (f) => function () {
    for (const [id, e] of live) if (e.device.deref() === this) live.delete(id);
    return f.call(this);
  });

  const watched = [];
  const watch = (what, object) => { watched.push({ what, ref: new WeakRef(object) }); return object; };
  const settle = async () => { for (let i = 0; i < 4; i++) { collect?.(); await new Promise((r) => setTimeout(r, 20)); } };
  const count = async () => {
    await settle();
    const labels = new Map();
    let n = 0;
    for (const e of live.values()) {
      if (e.ref.deref() === undefined) continue;
      n++;
      labels.set(e.label, (labels.get(e.label) ?? 0) + 1);
    }
    return { n, labels, pipelines: made.pipelines, shaders: made.shaders };
  };

  // --- an engine, and what the scenarios share ---
  const canvas = Object.assign(document.createElement('canvas'), { width: 320, height: 180 });
  canvas.style.cssText = 'position:fixed;left:-10000px;width:320px;height:180px';
  document.body.appendChild(canvas);
  const keptCanvas = Object.assign(document.createElement('canvas'), { width: 160, height: 90 });
  keptCanvas.style.cssText = 'position:fixed;left:-10000px;width:160px;height:90px';
  document.body.appendChild(keptCanvas);
  const BOX = buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.8, 1] });
  const RIG = buildRiggedGLB();
  const image = Object.assign(document.createElement('canvas'), { width: 64, height: 64 });
  { const x = image.getContext('2d'); x.fillStyle = '#c84'; x.fillRect(8, 8, 48, 48); }
  // A 16 x 8 Radiance image, flat scanlines: a grey sky.
  const HDR = (() => {
    const head = new TextEncoder().encode(['#?RADIANCE', 'FORMAT=32-bit_rle_rgbe', '', '-Y 8 +X 16', ''].join('\n'));
    const out = new Uint8Array(head.length + 16 * 8 * 4);
    out.set(head);
    for (let i = head.length; i < out.length; i += 4) out.set([100, 110, 130, 128], i);
    return out;
  })();

  const engine = await Winding.create(canvas, { antialias: false });
  engine.gpu.resize(320, 180);
  const camera = new Camera({ fovY: 1, near: 0.1 });
  camera.position.set([0, 3, 8]);
  camera.target.set([0, 0, 0]);
  const frames = (scene, cam = camera, n = 3) => {
    for (let i = 0; i < n; i++) engine.renderFrame(scene, cam);
    return engine.gpu.device.queue.onSubmittedWorkDone();
  };
  const base = engine.createScene();
  const box = await engine.load(BOX.slice());
  base.add(box);
  base.addLight({ type: 'directional', direction: [-0.3, -1, -0.4], intensity: 3 });
  await frames(base, camera, 5);

  const scenarios = {
    // A viewer swapping models.
    async models() {
      const model = watch('unloaded model', await engine.load(BOX.slice()));
      const node = watch('removed node', base.add(model).setPosition(2, 0, 0));
      await frames(base);
      base.remove(node);
      engine.unload(model);
    },
    async skinnedModels() {
      const model = watch('unloaded skinned model', await engine.load(RIG.slice()));
      const node = base.add(model).setPosition(-2, 0, 0);
      await frames(base);
      base.remove(node);
      engine.unload(model);
    },
    // Levels: a new scene each time, the old one simply dropped.
    async scenes() {
      const scene = watch('dropped scene', engine.createScene());
      for (let i = 0; i < 20; i++) scene.add(box).setPosition(i - 10, 0, -3);
      scene.addLight({ type: 'directional', direction: [-0.3, -1, -0.4], intensity: 3 });
      scene.addLight({ type: 'point', position: [0, 2, 0], intensity: 5, range: 6, castShadow: true });
      await frames(scene);
    },
    // Spawning and despawning.
    async spawning() {
      const nodes = [];
      for (let i = 0; i < 50; i++) nodes.push(base.add(box).setPosition((i % 10) - 5, 0, -Math.floor(i / 10)));
      await frames(base);
      for (const n of nodes) base.remove(n);
    },
    // A settings menu: every feature on, then off.
    async features() {
      const r = engine.renderer;
      r.ao = true; r.taa = true; r.oit = true; r.dof = { focusDistance: 5, fStop: 2 }; r.post.strength = 0.1;
      r.autoExposure = true; r.fog = { color: [0.5, 0.5, 0.6], visibility: 50 }; r.softShadows = true;
      await frames(base, camera, 2);
      await r._pipelinesBuilt();
      await frames(base, camera, 3);
      r.ao = false; r.taa = false; r.oit = false; r.dof = null; r.post.strength = 0; r.autoExposure = null; r.fog = null;
      await frames(base, camera, 2);
    },
    async resizing() {
      for (const [w, h] of [[400, 225], [512, 288], [320, 180]]) { engine.gpu.resize(w, h); await frames(base, camera, 2); }
    },
    // A burst of shadowed lights, then a long quiet: their shadow layers go back.
    async lightBurst() {
      const lights = [];
      for (let i = 0; i < 12; i++) lights.push(base.addLight({ type: 'point', position: [i - 6, 2, 0], intensity: 4, range: 4, castShadow: true }));
      await frames(base);
      for (const l of lights) base.remove(l);
    },
    // Lights coming and going, shadows and all.
    async lights() {
      const lights = [];
      for (let i = 0; i < 6; i++) {
        lights.push(base.addLight({ type: i % 2 ? 'spot' : 'point', position: [i - 3, 2, 1], intensity: 4, range: 5, castShadow: true }));
      }
      await frames(base);
      for (const l of lights) base.remove(l);
    },
    // Sprites, text, particles and decals, with their textures and fonts.
    async effects() {
      const texture = watch('unloaded texture', await engine.loadTexture(image));
      const font = watch('unloaded font', await engine.loadFont('32px sans-serif'));
      const nodes = [
        base.addSprite({ texture, position: [0, 1, 0] }),
        base.addText({ font, text: `soak ${Math.random().toFixed(4)}`, size: 0.3 }),
        base.addEmitter({ rate: 200, lifetime: [0.2, 0.4], size: 0.05, speed: [1, 2] }),
        base.addDecal({ texture, size: [1, 1, 0.5] }),
      ];
      await frames(base, camera, 4);
      for (const n of nodes) base.remove(n);
      await frames(base, camera, 1);
      engine.unload(texture);
      engine.unload(font);
    },
    // A score rewritten every frame.
    async changingText() {
      const font = await engine.loadFont('24px sans-serif');
      const nodes = [];
      for (let i = 0; i < 10; i++) nodes.push(base.addText({ font, text: String(i), size: 0.2 }));
      for (let k = 0; k < 4; k++) {
        for (const n of nodes) base.setText(n, { text: String(Math.random()).slice(0, 6) });
        await frames(base, camera, 1);
      }
      for (const n of nodes) base.remove(n);
      await frames(base, camera, 1);
      engine.unload(font);
    },
    async environments() {
      const environment = watch('unloaded environment', await engine.loadEnvironment(HDR.slice(), { size: 16 }));
      const scene = engine.createScene({ environment });
      scene.add(box);
      scene.addLight({ type: 'directional', direction: [-0.3, -1, -0.4], intensity: 3 });
      await frames(scene);
      engine.unload(environment);
    },
    // A minimap: a target made, drawn into, unloaded.
    async targets() {
      const target = watch('unloaded target', await engine.createTarget({ size: [64, 64] }));
      engine.renderFrame(base, camera, { target });
      await frames(base, camera, 1);
      engine.unload(target);
    },
    async probes() {
      const probe = base.addProbe({ position: [0, 1, 0], size: [4, 3, 4] });
      await engine.captureProbes(base, [probe]);
      await frames(base);
      base.remove(probe);
    },
    // A 2D game changing levels.
    async levels2D() {
      const texture = watch('2D texture', await engine.loadTexture(image, { pixelated: true }));
      const scene = watch('dropped 2D scene', engine.createScene());
      for (let i = 0; i < 20; i++) scene.addSprite({ texture, position: [i * 10, 20, 0] });
      scene.addTilemap({ tileset: texture, tileSize: [16, 16], columns: 8, rows: 8, tiles: new Array(64).fill(0) });
      await frames(scene, new Camera2D({ zoom: 2 }));
      engine.unload(texture);
    },
    // A live editor, or a component remounting on the canvas it kept.
    async engines() {
      const other = watch('destroyed engine', await Winding.create(keptCanvas, { antialias: false }));
      const scene = other.createScene();
      scene.add(await other.load(BOX.slice()));
      scene.addLight({ type: 'directional', direction: [-0.3, -1, -0.4], intensity: 3 });
      for (let i = 0; i < 2; i++) other.renderFrame(scene, camera);
      await other.gpu.device.queue.onSubmittedWorkDone();
      other.destroy();
    },
  };

  const drain = async () => {
    for (let i = 0; i < settleFrames; i++) engine.renderFrame(base, camera);
    await engine.gpu.device.queue.onSubmittedWorkDone();
  };
  const rows = [];
  let failed = 0;
  try {
    for (const [name, run] of Object.entries(scenarios)) {
      if (only && !only.includes(name)) continue;
      try {
        for (let i = 0; i < 2; i++) await run();
        await drain();
        watched.length = 0;
        const a = await count();
        for (let i = 0; i < rounds; i++) await run();
        await drain();
        const b = await count();
        const objects = b.n - a.n;
        // A new device compiles its own: those die with it.
        const ownDevice = name === 'engines';
        const pipelines = ownDevice ? 0 : b.pipelines - a.pipelines;
        const shaders = ownDevice ? 0 : b.shaders - a.shaders;
        const grew = [...b.labels].filter(([k, v]) => v > (a.labels.get(k) ?? 0)).map(([k, v]) => `${k} +${v - (a.labels.get(k) ?? 0)}`);
        // One may outlive its round: a debugger attached to the page keeps the
        // last async call's locals. More than one is held by the engine.
        const kept = new Map();
        if (collect) for (const w of watched) if (w.ref.deref() !== undefined) kept.set(w.what, (kept.get(w.what) ?? 0) + 1);
        const held = [...kept].filter(([, k]) => k > 1);
        const ok = objects <= 0 && pipelines <= 0 && shaders <= 0 && held.length === 0;
        if (!ok) failed++;
        rows.push(`${ok ? 'ok' : 'LEAK'} ${name}: GPU objects ${objects >= 0 ? '+' : ''}${objects}, pipelines +${pipelines}, shaders +${shaders}`
          + `${grew.length ? ` (${grew.join(', ')})` : ''}${held.length ? `; still alive: ${held.map(([k, n]) => `${n} ${k}s`).join(', ')}` : ''}`);
      } catch (error) {
        failed++;
        rows.push(`FAIL ${name}: ${error.message}`);
      }
    }
  } finally {
    engine.destroy();
    canvas.remove();
    keptCanvas.remove();
    for (const name of Object.keys(saved)) for (const [owner, original] of saved[name].reverse()) owner[name] = original;
  }
  return { rows, failed, collected: collect !== null };
}
