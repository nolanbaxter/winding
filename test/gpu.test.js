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
import { NOT_BATCHED, DRAW_DATA_BYTES } from '../src/render/gpudriven.js';
import {
  buildDemoGLB, buildRiggedGLB, buildMorphedGLB, buildFeatureGLB, twoToneImageURI,
} from './fixtures/demoModel.js';
import { PBR_SHADER } from '../src/render/shaders/pbr.js';

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

  await step('a morph target moves the vertex the shader reads', async () => {
    // The strong check for step 3, and it runs the ENGINE'S OWN WGSL: the
    // Morphed struct and applyMorph are sliced straight out of PBR_SHADER and
    // dropped into a compute shader, bound to the same delta buffer, the same
    // weight buffer and the same draw data the frame just used. What it cannot
    // share with the vertex shader is the entry point; everything the vertex
    // shader would read, it reads.
    //
    // A readback is the only way to see a vertex position at all -- the engine
    // deliberately never reads one back, and the swap chain is not COPY_SRC --
    // so this pays for one, once, on four vertices.
    const morphed = await engine.load(buildMorphedGLB());
    const morphNode = scene.add(morphed);
    const morphIndex = scene.renderableCount - 1;

    morphNode.weights[0] = 0.5;      // top edge up by 10 * 0.5
    morphNode.weights[1] = 0.25;     // right edge out by 4 * 0.25
    engine.renderFrame(scene, camera);
    await engine.rhi.device.queue.onSubmittedWorkDone();

    const device = engine.rhi.device;
    const store = engine.renderer.morph;
    const primitive = morphed.meshes[0].primitives[0];

    // 4 vertices * 2 targets * 3 floats.
    if (store.deltaCount < 24) {
      throw new Error(`the arena holds ${store.deltaCount} floats, expected at least 24`);
    }
    if (store.weightCount !== 2) {
      throw new Error(`gathered ${store.weightCount} weights, expected 2`);
    }
    if ((primitive.morphCountStride & 0xffff) !== 2 || (primitive.morphCountStride >>> 16) !== 3) {
      throw new Error(`count/stride packed as ${primitive.morphCountStride.toString(16)}`);
    }

    // The real struct and the real function, lifted out of the real shader.
    const slice = (source, from, until) => {
      const start = source.indexOf(from);
      const end = source.indexOf(until, start);
      if (start < 0 || end < 0) throw new Error(`cannot find ${from} in the shader`);
      return source.slice(start, end + until.length);
    };
    const drawStruct = slice(PBR_SHADER, 'struct DrawData {', '\n};');
    const morphFn = slice(PBR_SHADER, 'struct Morphed {', '\n  return out;\n}');

    const module = device.createShaderModule({
      label: 'morph-check',
      code: `
${drawStruct}

@group(0) @binding(0) var<storage, read> drawData     : array<DrawData>;
@group(0) @binding(1) var<storage, read> morphDeltas  : array<f32>;
@group(0) @binding(2) var<storage, read> morphWeights : array<f32>;
@group(0) @binding(3) var<storage, read_write> out    : array<f32>;
@group(0) @binding(4) var<uniform> which : vec4<u32>;

${morphFn}

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let draw = drawData[which.x];
  let m = applyMorph(draw, id.x, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  out[id.x * 3u] = m.position.x;
  out[id.x * 3u + 1u] = m.position.y;
  out[id.x * 3u + 2u] = m.position.z;
}
`,
    });

    const outBuffer = device.createBuffer({
      size: 4 * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const staging = device.createBuffer({
      size: 4 * 3 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const which = device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(which, 0, Uint32Array.from([morphIndex, 0, 0, 0]));

    const pipeline = device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: 'main' },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: engine.renderer.gpu.drawDataBuffer } },
        { binding: 1, resource: { buffer: store.deltaBuffer } },
        { binding: 2, resource: { buffer: store.weightBuffer } },
        { binding: 3, resource: { buffer: outBuffer } },
        { binding: 4, resource: { buffer: which } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(outBuffer, 0, staging, 0, staging.size);
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    outBuffer.destroy();
    staging.destroy();
    which.destroy();

    // The deltas alone, since the check fed a zero base position: vertex 0
    // moves under neither target, 1 only under the right-edge one, 2 only
    // under the top-edge one, and 3 under both.
    const expected = [
      0, 0, 0,
      1, 0, 0,        // 4 * 0.25
      0, 5, 0,        // 10 * 0.5
      1, 5, 0,
    ];
    for (let i = 0; i < expected.length; i++) {
      if (Math.abs(got[i] - expected[i]) > 1e-4) {
        throw new Error(
          `vertex ${Math.floor(i / 3)} axis ${i % 3}: expected ${expected[i]}, got ${got[i]}`,
        );
      }
    }

    // And the box grew to cover it -- 0.5 * 10 + 0.25 * 4 = 6 on every side.
    const top = scene.worldMax[morphIndex * 3 + 1];
    if (!(top > 2 + 5.9)) {
      throw new Error(`bounds did not follow the weights: top is ${top.toFixed(2)}`);
    }

    morphNode.destroy();
    return `${store.deltaCount} delta floats, 2 weights, vertex 3 at `
      + `(${got[9].toFixed(1)}, ${got[10].toFixed(1)}), bounds top ${top.toFixed(1)}`;
  });

  await step('a mesh that is both skinned and morphed gets both', async () => {
    // The two deformations meet in one vertex shader and one bounding box, and
    // neither the shader variant nor the batch key says anything about morphs.
    // So this is the check that they compose: a skinned pipeline reading morph
    // words, and a box that grew for the joint AND for the weight.
    const both = await engine.load(buildRiggedGLB({ morphed: true }));
    const bothNode = scene.add(both);
    const index = scene.renderableCount - 1;

    if (scene.renderableSkin[index] < 0) throw new Error('not registered as skinned');
    if (scene.renderableMorph[index] < 0) throw new Error('not registered as morphed');

    // Pose a joint, leaving every weight at zero.
    const skin = scene.skins[scene.renderableSkin[index]];
    scene.transforms.setPosition(skin.joints[1], 0, 12, 0);
    engine.renderFrame(scene, camera);
    await engine.rhi.device.queue.onSubmittedWorkDone();

    const posedTop = scene.worldMax[index * 3 + 1];
    const restingFront = scene.worldMax[index * 3 + 2];
    if (!(posedTop > 12)) {
      throw new Error(`bounds did not follow the joint: top is ${posedTop.toFixed(2)}`);
    }

    // Now the weight, on an axis the skeleton did not touch.
    bothNode.weights[0] = 1;
    engine.renderFrame(scene, camera);
    await engine.rhi.device.queue.onSubmittedWorkDone();

    const morphedFront = scene.worldMax[index * 3 + 2];
    if (Math.abs(morphedFront - restingFront - 3) > 0.01) {
      throw new Error(
        `the weight did not reach the posed box: front was ${restingFront.toFixed(2)}, `
        + `now ${morphedFront.toFixed(2)}, expected +3`,
      );
    }
    if (!(scene.worldMax[index * 3 + 1] > posedTop)) {
      throw new Error('the joint correction was lost when the morph one was applied');
    }

    // And the draw data carries both, on a batch the renderer calls skinned.
    const gpu = engine.renderer.gpu;
    const word = index * (DRAW_DATA_BYTES / 4);
    const targets = gpu.drawDataU32[word + 31] & 0xffff;
    if (targets !== 1) throw new Error(`draw data says ${targets} targets, expected 1`);

    let skinnedBatches = 0;
    for (let b = 0; b < gpu.batchCount; b++) if (gpu.batchSkinned[b]) skinnedBatches++;
    if (skinnedBatches === 0) throw new Error('nothing batched as skinned');

    bothNode.destroy();
    return `skinned batch with ${targets} target, box top ${posedTop.toFixed(1)}, `
      + `front ${restingFront.toFixed(1)} -> ${morphedFront.toFixed(1)}`;
  });

  await step('every material and geometry feature actually draws', async () => {
    // The gap this closes: between them, the rendered fixtures used ONE
    // material texture slot, one alpha mode that draws, positive scales only,
    // indexed geometry only, and supplied normals only. Everything else was
    // imported, tested on the CPU, compiled into a pipeline -- and never once
    // turned into a pixel. A constant tangent and a black emissive default
    // both shipped through that gap.
    //
    // So this renders one full-view quad per feature and reads the middle
    // pixel. No screen-space arithmetic, no screenshot to squint at: each
    // feature gets one unambiguous answer.
    const SIZE = 64;
    const MID = ((SIZE / 2) * SIZE + SIZE / 2) * 4;

    const probeCanvas = document.createElement('canvas');
    probeCanvas.width = SIZE;
    probeCanvas.height = SIZE;
    document.body.appendChild(probeCanvas);

    // A flat green sky, so "the quad was not drawn here" is unmistakable and
    // no lighting gradient can be mistaken for geometry.
    const SKY = [0, 1, 0];
    const probe = await Winding.create(probeCanvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
    });

    const shootWith = async (options, light) => {
      const scene = probe.createScene();
      const cam = new Camera({ fovY: 1.0, near: 0.1 });
      cam.position.set([0, 0, 2]);
      cam.target.set([0, 0, 0]);
      scene.add(await probe.load(buildFeatureGLB(options)));
      if (light) scene.addLight(light);
      probe.renderFrame(scene, cam);
      // Once per frame, before anything else awaits -- see rhi.readPixels.
      const pixels = await probe.rhi.readPixels();
      return [pixels[MID], pixels[MID + 1], pixels[MID + 2]];
    };
    const shoot = (options) => shootWith(options, null);

    const isSky = ([r, g, b]) => g > r && g > b;
    const red = ([r, g, b]) => r > g && r > b;
    const blue = ([r, g, b]) => b > r && b > g;
    const show = (p) => `rgb(${p.join(',')})`;

    const results = [];
    const expect = (name, pixel, ok) => {
      if (!ok) throw new Error(`${name}: got ${show(pixel)}`);
      results.push(name);
    };

    try {
      // The control. Everything below is only meaningful if this is a red quad
      // over a green sky.
      const control = await shoot({});
      expect('opaque', control, red(control));

      const sky = await shoot({ baseColorFactor: [0.9, 0.15, 0.1, 1], nodeScale: [0.001, 0.001, 1] });
      expect('sky', sky, isSky(sky));

      // MASK, both sides of the cutoff. The discard in the fragment shader had
      // never executed: the suite compiles all six material variants, and
      // compiling is not drawing.
      const cutOut = await shoot({ alphaMode: 'MASK', alphaCutoff: 0.5, baseColorFactor: [0.9, 0.15, 0.1, 0.2] });
      expect('MASK below cutoff discards', cutOut, isSky(cutOut));

      const kept = await shoot({ alphaMode: 'MASK', alphaCutoff: 0.5, baseColorFactor: [0.9, 0.15, 0.1, 0.9] });
      expect('MASK above cutoff draws', kept, red(kept));

      // COLOR_0, shipped in 0.3.0 and never rendered. White material, blue
      // vertices: if the multiply is dropped the quad comes back white.
      const vertexColour = await shoot({
        baseColorFactor: [1, 1, 1, 1],
        colors: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
      });
      expect('COLOR_0 tints the surface', vertexColour, blue(vertexColour));

      // TEXCOORD_1, also 0.3.0 and also never rendered. uv0 lands on the red
      // half of a two-pixel texture and uv1 on the blue half, so which set the
      // material asked for is the whole answer.
      const image = twoToneImageURI('#ff0000', '#0000ff');
      const LEFT = [0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5];    // wholly inside the red texel
      const RIGHT = [0.75, 0.5, 0.75, 0.5, 0.75, 0.5, 0.75, 0.5];   // wholly inside the blue one
      const onSet0 = await shoot({ imageURI: image, baseColorTexCoord: 0, uv0: LEFT, uv1: RIGHT });
      expect('texCoord 0 samples uv0', onSet0, red(onSet0));

      const onSet1 = await shoot({ imageURI: image, baseColorTexCoord: 1, uv0: LEFT, uv1: RIGHT });
      expect('texCoord 1 samples uv1', onSet1, blue(onSet1));

      // A negative scale flips the winding, which is why there is a mirrored
      // pipeline variant. Seven CPU checks and no draws until now: if the
      // front face were wrong the quad would be culled and the sky would show.
      const mirrored = await shoot({ nodeScale: [-1, 1, 1] });
      expect('a mirrored node still faces the camera', mirrored, red(mirrored));

      // Geometry the importer has to synthesise something for.
      const unindexed = await shoot({ indexed: false });
      expect('a non-indexed primitive draws', unindexed, red(unindexed));

      // The tangent bug's own case: no NORMAL means flat shading, and the
      // invented tangent frame used to be degenerate on X-facing geometry.
      const flat = await shoot({ includeNormals: false });
      expect('flat-shaded geometry is not NaN', flat, red(flat));

      // A NORMAL MAP, which nothing had ever sampled. The default is a flat
      // 1x1, so the whole tangent-space path -- the TBN basis, normalScale,
      // the handedness in tangent.w -- had only ever been fed a normal of
      // (0,0,1), which is the one input that makes the basis irrelevant.
      //
      // This is the code the black-cube bug lived next door to: an arbitrary
      // tangent is harmless while the map is flat, and becomes the surface
      // orientation the moment it is not.
      const flatMap = twoToneImageURI('#8080ff', '#8080ff');     // (0,0,1): no tilt
      const tiltedMap = twoToneImageURI('#ff8080', '#ff8080');   // hard tilt along +X
      const withFlat = await shoot({ normalImageURI: flatMap, baseColorFactor: [0.8, 0.8, 0.8, 1] });
      const withTilt = await shoot({ normalImageURI: tiltedMap, baseColorFactor: [0.8, 0.8, 0.8, 1] });
      const shift = Math.abs(withFlat[0] - withTilt[0])
        + Math.abs(withFlat[1] - withTilt[1])
        + Math.abs(withFlat[2] - withTilt[2]);
      if (shift < 12) {
        throw new Error(
          `a normal map changed nothing: flat ${show(withFlat)} vs tilted ${show(withTilt)}`,
        );
      }
      results.push('a normal map reorients the surface');

      // And scale 0 must put it back, which is the one knob on that texture.
      const cancelled = await shoot({
        normalImageURI: tiltedMap, normalScale: 0, baseColorFactor: [0.8, 0.8, 0.8, 1],
      });
      const residue = Math.abs(cancelled[0] - withFlat[0]) + Math.abs(cancelled[1] - withFlat[1]);
      if (residue > 6) {
        throw new Error(`normalScale 0 did not cancel the map: ${show(cancelled)} vs flat ${show(withFlat)}`);
      }
      results.push('normalScale 0 cancels it');

      // CLUSTERED LIGHTING, also never checked against a pixel. The suite
      // counts lights into the buffer and the cluster passes run, but whether
      // a punctual light ever reaches a surface was nobody's assertion.
      const unlit = await shootWith({ baseColorFactor: [0.6, 0.6, 0.6, 1] }, null);
      const lit = await shootWith({ baseColorFactor: [0.6, 0.6, 0.6, 1] }, {
        position: [0, 0, 1.2], color: [1, 1, 1], intensity: 40, radius: 6,
      });
      if (!(lit[0] > unlit[0] + 20)) {
        throw new Error(`a point light did not reach the surface: ${show(unlit)} -> ${show(lit)}`);
      }
      results.push('a point light reaches the surface');

      // Emissive with no texture. The default map was black, so this factor
      // used to be multiplied away entirely.
      const dim = await shoot({ baseColorFactor: [0.05, 0.05, 0.05, 1] });
      const glowing = await shoot({ baseColorFactor: [0.05, 0.05, 0.05, 1], emissiveFactor: [0.9, 0.1, 0.1] });
      if (!(glowing[0] > dim[0] + 30)) {
        throw new Error(`emissive factor did nothing: ${show(dim)} -> ${show(glowing)}`);
      }
      results.push('emissive without a map glows');
    } finally {
      probe.destroy();
      probeCanvas.remove();
    }

    return `${results.length} features drawn and read back`;
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
