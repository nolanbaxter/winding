// GPU smoke test. Run: node serve.js, then open http://localhost:8080/test/gpu.html
//
// The other ten suites run in Node, which means not one line of WGSL in this
// engine is checked by any of them. Every shader bug this codebase has actually
// produced -- a backtick inside a WGSL comment ending the JS template literal,
// `target` used as an identifier when WGSL reserves it, a bind group declared
// one way and used another -- was invisible to `npm test` and surfaced later as
// a confusing failure somewhere else. This is the suite that would have caught
// them, and it needs a real device, so it runs in a browser.
//
// It does not check that anything LOOKS right. It checks that every shader the
// engine can compile does compile, every pipeline permutation builds, and a
// full frame records and submits without the device complaining.

import { Winding, Camera } from '../src/winding.js';
import { shaderErrors } from '../src/rhi/shader.js';
import { NOT_BATCHED } from '../src/render/gpudriven.js';
import { buildDemoGLB, buildRiggedGLB } from './fixtures/demoModel.js';

const FRAMES = 30;

/** Small enough that adding anything at all forces GpuDriven to grow. */
const START_DRAWS = 8;

// Every (alphaMode, doubleSided) pipeline permutation, per variantKey():
// alphaMode in the low two bits, doubleSided at bit 2. The demo model uses two
// of the six, so without this the rest never compile and a mistake in any of
// them sits undetected until someone loads an asset that uses it.
const ALL_VARIANTS = [0, 1, 2, 4, 5, 6];

const results = [];

function check(name, ok, detail = '') {
  results.push({ name, ok, detail });
  console[ok ? 'log' : 'error'](`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? `\n      ${detail}` : ''}`);
}

/** Runs fn, and turns a throw into a failed check rather than a dead page. */
async function step(name, fn) {
  try {
    const detail = await fn();
    check(name, true, typeof detail === 'string' ? detail : '');
    return true;
  } catch (error) {
    check(name, false, error.message);
    return false;
  }
}

export async function run(canvas, onDone) {
  if (!navigator.gpu) {
    check('WebGPU is available', false, 'navigator.gpu is undefined -- this browser cannot run the test');
    return finish(onDone);
  }

  // Device errors are asynchronous and do not throw anywhere useful, so they
  // are collected and asserted at the end rather than caught at a call site.
  const deviceErrors = [];
  let engine;

  const created = await step('device and shaders come up', async () => {
    engine = await Winding.create(canvas, {
      // Deliberately far too small. Every container that sizes itself off this
      // has to grow during the run, which means the whole suite below doubles
      // as a test that growth does not break anything -- a stale bind group
      // naming a destroyed buffer would surface as a device error.
      maxDraws: START_DRAWS,
      onError: (error) => deviceErrors.push(String(error.message ?? error)),
      onDeviceLost: (info) => deviceErrors.push(`device lost: ${info.reason} ${info.message}`),
    });
  });
  if (!created) return finish(onDone);

  await step('every material pipeline permutation builds', async () => {
    await engine.renderer.ensureVariants(ALL_VARIANTS);
    const built = engine.renderer._pipelineByVariant.size;
    // Each material variant expands by winding AND by skinning: four
    // pipelines apiece, so nothing has to compile mid-frame whichever a scene
    // turns out to need.
    if (built !== ALL_VARIANTS.length * 4) {
      throw new Error(`${ALL_VARIANTS.length} variants built ${built} pipelines, expected ${ALL_VARIANTS.length * 4}`);
    }
    return `${ALL_VARIANTS.length} variants, ${built} pipelines`;
  });

  const scene = engine.createScene();
  const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
  camera.position.set([0, 2, 8]);

  await step('a glTF asset imports and adds to the scene', async () => {
    const asset = await engine.load(await buildDemoGLB({ arms: 6 }));
    scene.add(asset);
    return `${scene.renderableCount} renderables`;
  });

  // Lights, so the clustered path actually populates clusters instead of
  // running every shader against an empty light list.
  await step('lights register', async () => {
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      scene.addLight({
        position: [Math.cos(a) * 4, 0.5, Math.sin(a) * 4],
        color: [1, 1, 1], intensity: 10, radius: 5,
      });
    }
    scene.addLight({
      position: [0, 7, 0], direction: [0, -1, 0], color: [1, 1, 1],
      intensity: 100, radius: 16, innerAngle: 0.25, outerAngle: 0.5,
    });
    return '8 point + 1 spot';
  });

  await step(`${FRAMES} frames record and submit`, async () => {
    for (let i = 0; i < FRAMES; i++) {
      // Move something every frame so the transform, cull and cluster paths do
      // real work rather than taking their "nothing changed" branches.
      camera.position.set([Math.sin(i * 0.1) * 8, 2, Math.cos(i * 0.1) * 8]);
      engine.renderFrame(scene, camera);
    }
    // Errors are raised while the queue drains, which has not happened yet.
    await engine.rhi.device.queue.onSubmittedWorkDone();
  });

  await step('GPU pass timing reports a duration per pass', async () => {
    const profiler = engine.renderer.gpuTiming;
    if (!profiler.supported) return 'skipped: no timestamp-query on this adapter';

    // Chrome quantises timestamps to 65536ns, so most passes read as 0 on any
    // given frame. Only the mean over a few hundred frames means anything.
    const SAMPLES = 400;
    profiler.resetAverages();
    for (let i = 0; i < SAMPLES * 3 && profiler.samples < SAMPLES; i++) {
      camera.position.set([Math.sin(i * 0.02) * 8, 2, Math.cos(i * 0.02) * 8]);
      engine.renderFrame(scene, camera);
      await engine.rhi.device.queue.onSubmittedWorkDone();
    }
    if (profiler.samples === 0) throw new Error('no timing readback completed');

    for (const { name, ms } of profiler.average) {
      if (!Number.isFinite(ms) || ms < 0) throw new Error(`pass ${name} averaged ${ms}ms`);
    }
    if (profiler.averageTotalMs <= 0) {
      throw new Error('every pass averaged 0ms, which means the queries never landed');
    }

    console.log(`[gpu timing] ${profiler.samples} samples, ${profiler.averageTotalMs.toFixed(3)}ms total:`,
      profiler.average.map((p) => `${p.name}=${p.ms.toFixed(4)}`).join(' '));

    const top = profiler.slowest(3).map((p) => `${p.name} ${p.ms.toFixed(3)}ms`).join(', ');
    return `${profiler.average.length} passes over ${profiler.samples} frames, `
      + `${profiler.averageTotalMs.toFixed(2)}ms; slowest ${top}`;
  });

  await step('blended geometry is held out of the batched path', async () => {
    const gpu = engine.renderer.gpu;
    if (gpu.transparentCount === 0) throw new Error('the demo asset has no BLEND material to test with');

    for (let t = 0; t < gpu.transparentCount; t++) {
      const i = gpu.transparentItems[t];
      if (!engine.renderer.materials.isTransparent(scene.renderableMaterial[i])) {
        throw new Error(`renderable ${i} is in the transparent list but its material is not BLEND`);
      }
      if (gpu.itemBatch[i] !== NOT_BATCHED) {
        throw new Error(`renderable ${i} is blended but was still assigned batch ${gpu.itemBatch[i]}`);
      }
    }
    // The visible list is split at opaqueCount, so the two regions must
    // partition the scene exactly -- an overlap would corrupt one of them.
    if (gpu.opaqueCount + gpu.transparentCount !== scene.renderableCount) {
      throw new Error(`${gpu.opaqueCount} opaque + ${gpu.transparentCount} blended != ${scene.renderableCount} renderables`);
    }
    return `${gpu.transparentCount} blended, ${gpu.opaqueCount} batched`;
  });

  await step('blended draws are ordered back to front, and re-sort with the camera', async () => {
    const r = engine.renderer;
    const drawOrder = (x, z) => {
      camera.position.set([x, 2, z]);
      camera.target.set([0, 0.6, 0]);
      engine.renderFrame(scene, camera);
      return [...r.transparentList.payloads.slice(0, r.transparentList.count)];
    };
    const distance = (i) => {
      const o = i * 3;
      const c = (k) => (scene.worldMin[o + k] + scene.worldMax[o + k]) * 0.5 - camera.position[k];
      return Math.hypot(c(0), c(1), c(2));
    };

    const front = drawOrder(0, 9);
    for (let k = 1; k < front.length; k++) {
      if (distance(front[k - 1]) < distance(front[k])) {
        throw new Error(`draw order is not far-to-near: ${front.join(',')}`);
      }
    }

    // The decisive check. A list that happens to be sorted once could be
    // accidental; one that reverses when the camera crosses to the other side
    // is actually being sorted.
    const back = drawOrder(0, -9);
    if (front.join() !== [...back].reverse().join()) {
      throw new Error(`order did not reverse from the opposite side: ${front.join(',')} vs ${back.join(',')}`);
    }
    return `${front.length} panes, reversed from behind`;
  });

  await step('the scene grows past every starting capacity and still draws', async () => {
    // The one check that exercises growth on a real device. Every container
    // here starts smaller than what gets added, so buffers are destroyed and
    // recreated and every bind group naming them has to be rebuilt -- a stale
    // one is a validation error, which the device-error check below catches.
    const gpu = engine.renderer.gpu;
    const startCapacity = gpu.capacity;
    const startRenderables = scene.renderableCount;

    const asset = await engine.load(await buildDemoGLB({ arms: 3 }));
    for (let i = 0; i < 60; i++) {
      scene.add(asset).setPosition((i % 10) * 3 - 15, 0, Math.floor(i / 10) * 3 - 9);
    }
    for (let i = 0; i < 300; i++) {
      scene.addLight({ position: [i % 20, 1, (i / 20) | 0], color: [1, 1, 1], intensity: 1, radius: 2 });
    }

    for (let i = 0; i < 5; i++) engine.renderFrame(scene, camera);
    await engine.rhi.device.queue.onSubmittedWorkDone();

    if (scene.renderableCount <= startRenderables) throw new Error('nothing was added');
    if (gpu.capacity < scene.renderableCount) {
      throw new Error(`gpu capacity ${gpu.capacity} cannot hold ${scene.renderableCount} renderables`);
    }
    if (gpu.capacity <= START_DRAWS) {
      throw new Error(`gpu never grew past its starting ${START_DRAWS}`);
    }
    if (engine.renderer.clusters.lightCapacity < scene.lightCount) {
      throw new Error('the clustered light buffer did not grow with the scene');
    }
    if (gpu.opaqueCount + gpu.transparentCount !== scene.renderableCount) {
      throw new Error('the batch partition did not survive growth');
    }
    return `${scene.renderableCount} renderables, ${scene.lightCount} lights, gpu capacity ${gpu.capacity}`;
  });

  await step('the graph ordered every live pass', async () => {
    const { passes, executed, culled, edges } = engine.renderer.graph.stats;
    if (executed + culled !== passes) {
      throw new Error(`${passes} passes, ${executed} executed, ${culled} culled -- ${passes - executed - culled} unaccounted for`);
    }
    return `${executed} of ${passes} passes, ${edges} edges`;
  });

  await step('a skinned mesh draws, and bind pose matches the static mesh', async () => {
    // The strong check for step 2: with every joint at the origin and identity
    // inverse binds, the palette must be identity, so a skinned draw and an
    // unskinned draw of the same geometry produce the same vertices. Any error
    // in the multiply order, the joint-to-entity mapping or the vertex buffer
    // shows up before there is any animation to confuse it with.
    const rigged = await engine.load(buildRiggedGLB());
    const riggedNode = scene.add(rigged);
    engine.renderFrame(scene, camera);
    await engine.rhi.device.queue.onSubmittedWorkDone();

    const palette = engine.renderer.skinPalette;
    if (palette.jointCount !== 2) {
      throw new Error(`expected 2 joints in the palette, got ${palette.jointCount}`);
    }
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    for (let j = 0; j < 2; j++) {
      for (let k = 0; k < 16; k++) {
        if (Math.abs(palette.data[j * 16 + k] - identity[k]) > 1e-5) {
          throw new Error(`joint ${j} is not identity in bind pose at element ${k}`);
        }
      }
    }

    // A skinned batch, a skin vertex buffer, and a skinned pipeline.
    const gpu = engine.renderer.gpu;
    let skinnedBatches = 0;
    for (let b = 0; b < gpu.batchCount; b++) if (gpu.batchSkinned[b]) skinnedBatches++;
    if (skinnedBatches === 0) throw new Error('nothing batched as skinned');

    const primitive = rigged.meshes[0].primitives[0];
    if (!primitive.skinBuffer) throw new Error('the rigged primitive has no skin vertex buffer');

    // Pose a joint and confirm the bounds follow it. The mesh node never
    // moves, so a box transformed from the bind pose would be unchanged -- and
    // the character would be culled with its arm on screen.
    const riggedIndex = scene.renderableCount - 1;
    const beforeTop = scene.worldMax[riggedIndex * 3 + 1];

    const skin = scene.skins[scene.renderableSkin[riggedIndex]];
    scene.transforms.setPosition(skin.joints[1], 0, 12, 0);
    engine.renderFrame(scene, camera);
    await engine.rhi.device.queue.onSubmittedWorkDone();

    const afterTop = scene.worldMax[riggedIndex * 3 + 1];
    if (!(afterTop > beforeTop + 5)) {
      throw new Error(
        `bounds did not follow the joint: top was ${beforeTop.toFixed(2)}, now ${afterTop.toFixed(2)}`,
      );
    }

    riggedNode.destroy();
    return `${skinnedBatches} skinned batch, ${palette.jointCount} joints, `
      + `bind pose identity, bounds ${beforeTop.toFixed(1)} -> ${afterTop.toFixed(1)}`;
  });

  await step('order-independent transparency resolves into the scene', async () => {
    // A second engine, because oit is chosen at construction: the pipelines
    // and the resolve are built then, so nothing compiles mid-frame.
    const oitCanvas = document.createElement('canvas');
    oitCanvas.width = 256;
    oitCanvas.height = 256;
    document.body.appendChild(oitCanvas);

    const oitEngine = await Winding.create(oitCanvas, { oit: true });
    try {
      const oitScene = oitEngine.createScene();
      oitScene.add(await oitEngine.load(await buildDemoGLB({ arms: 3 })));
      const oitCamera = new Camera({ fovY: Math.PI / 3, near: 0.1 });
      oitCamera.position.set([0, 2, 8]);

      for (let i = 0; i < 5; i++) oitEngine.renderFrame(oitScene, oitCamera);
      await oitEngine.rhi.device.queue.onSubmittedWorkDone();

      const graph = oitEngine.renderer.graph;
      const names = [];
      for (let s = 0; s < graph._orderCount; s++) names.push(graph._passes[graph._order[s]].name);

      const oitAt = names.indexOf('oit');
      const resolveAt = names.indexOf('oit-resolve');
      if (oitAt < 0 || resolveAt < 0) throw new Error(`no oit passes in: ${names.join(', ')}`);
      if (resolveAt < oitAt) throw new Error('the resolve was ordered before the pass it reads');
      if (names.indexOf('forward:late') > oitAt) {
        throw new Error('blended geometry accumulated before the opaque depth existed');
      }
      if (names.indexOf('tonemap') < resolveAt) {
        throw new Error('tonemap ran before the resolve composited into the scene');
      }
      return `${names.length} passes, oit at ${oitAt}, resolve at ${resolveAt}`;
    } finally {
      oitEngine.destroy();
      oitCanvas.remove();
    }
  });

  check('no WGSL compilation errors', shaderErrors.length === 0, shaderErrors.join('\n'));
  check('no uncaptured device errors', deviceErrors.length === 0, deviceErrors.join('\n'));

  engine.destroy();
  return finish(onDone);
}

function finish(onDone) {
  const failed = results.filter((r) => r.ok === false);
  const summary = {
    passed: results.length - failed.length,
    failed: failed.length,
    results,
  };
  // Read by anything driving this page headlessly. The title is the cheapest
  // thing an automated runner can assert on without knowing the DOM.
  globalThis.__gpuTest = summary;
  document.title = failed.length === 0 ? `PASS ${summary.passed}` : `FAIL ${failed.length}`;
  console.log(`\n${summary.passed} passed, ${failed.length} failed`);
  onDone?.(summary);
  return summary;
}
