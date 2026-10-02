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

import { EXTENSION_TEXTURES } from '../src/scene/gltf/images.js';
import { Winding, Camera, Camera2D } from '../src/winding.js';
import { Benchmark } from '../src/bench.js';
import { Environment } from '../src/render/ibl.js';
import { CLUSTER_Z, MAX_LIGHTS_PER_CLUSTER } from '../src/render/clustered.js';
import { shaderErrors } from '../src/rhi/shader.js';
import { NOT_BATCHED, DRAW_DATA_BYTES } from '../src/render/gpudriven.js';
import {
  buildDemoGLB, buildRiggedGLB, buildMorphedGLB, buildFeatureGLB, buildLodGLB, twoToneImageURI,
} from './fixtures/demoModel.js';
import { pbrShader } from '../src/render/shaders/pbr.js';

/** The forward shader with every extension texture bound, as a roomy device builds it. */
const PBR_SHADER = pbrShader(EXTENSION_TEXTURES.length);

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
  // A key light, so the frames below draw cascades as any lit scene does.
  scene.addLight({ type: 'directional', direction: [-0.35, -0.55, -0.45], intensity: 3.2 });
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
    await engine.gpu.device.queue.onSubmittedWorkDone();
  });

  await step('a benchmark times every CPU phase and GPU pass, and leaves nothing on', async () => {
    const renderer = engine.renderer;
    if (renderer.profiler !== null) throw new Error('a profiler was attached before anyone asked');
    if (renderer.gpuTiming.enabled) throw new Error('GPU timing was on by default');

    // Chrome quantises timestamps to 65536ns, so most passes read as 0 on any
    // given frame. Only the mean over a few hundred frames means anything.
    const report = await new Benchmark(engine).run(scene, camera, {
      frames: 400,
      update: (i) => camera.position.set([Math.sin(i * 0.02) * 8, 2, Math.cos(i * 0.02) * 8]),
    });

    if (renderer.profiler !== null) throw new Error('the benchmark stayed attached');
    if (renderer.gpuTiming.enabled) throw new Error('GPU timing stayed on after the run');

    // The renderer's side of the contract: every phase marked, in frame order.
    const expected = [
      'transforms', 'camera', 'bounds', 'draw data', 'skin + morph', 'batch sort',
      'transparent sort', 'lights', 'shadow fit', 'clusters + frame uniform',
      'graph build', 'encode', 'submit', 'frame (cpu)',
    ];
    const names = report.cpu.map((row) => row.name);
    if (names.join() !== expected.join()) throw new Error(`phases were: ${names.join(', ')}`);
    if (report.frames !== 400) throw new Error(`${report.frames} frames recorded, not 400`);
    if (!(report.wall?.median > 0)) throw new Error('no wall time');

    console.log(Benchmark.format(report));

    if (!renderer.gpuTiming.available) return `cpu only: no timestamp-query on this adapter`;
    if (report.gpuSamples === 0) throw new Error('no GPU timing readback completed');
    for (const { name, mean } of report.gpu) {
      if (!Number.isFinite(mean) || mean < 0) throw new Error(`pass ${name} averaged ${mean}ms`);
    }
    if (!report.gpu.some((row) => row.mean > 0)) {
      throw new Error('every pass averaged 0ms, which means the queries never landed');
    }
    const top = (rows, key) => [...rows].sort((a, b) => b[key] - a[key])[0];
    const cpuTop = top(report.cpu.filter((r) => r.name !== 'frame (cpu)'), 'median');
    const gpuTop = top(report.gpu, 'mean');
    return `${report.cpu.length - 1} cpu phases, ${report.gpu.length} gpu passes; `
      + `slowest ${cpuTop.name} ${cpuTop.median.toFixed(3)}ms / ${gpuTop.name} ${gpuTop.mean.toFixed(3)}ms`;
  });

  await step('every cluster lists exactly the lights that overlap it', async () => {
    // Light assignment runs from the lights: each finds the cells its sphere
    // can reach and tests only those. The claim is that the lists are the same
    // as testing every light against every cell, so that is what this does --
    // on the CPU, against the cluster boxes the GPU built, for every cell.
    const clusters = engine.renderer.clusters;
    const lightScene = engine.createScene();
    lightScene.add(await engine.load(await buildDemoGLB({ arms: 6 })));
    // Big and small, near and far, some straddling the near plane.
    for (let i = 0; i < 160; i++) {
      const a = i * 2.399;
      const r = 0.5 + (i % 7) * 1.3;
      lightScene.addLight({ position: [Math.cos(a) * (i % 13), (i % 5) - 1, Math.sin(a) * (i % 13) - 4], radius: r, intensity: 2 });
    }
    const cam = new Camera({ fovY: Math.PI / 3, near: 0.1 });
    cam.position.set([0, 2, 8]);
    engine.renderFrame(lightScene, cam);

    const device = engine.gpu.device;
    const read = async (buffer, bytes) => {
      const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const copy = staging.getMappedRange().slice(0);
      staging.destroy();
      return copy;
    };
    const cells = clusters.gridX * clusters.gridY * CLUSTER_Z;
    const bounds = new Float32Array(await read(clusters.boundsBuffer, cells * 32));
    const counts = new Uint32Array(await read(clusters.countBuffer, cells * 4));
    const indices = new Uint32Array(await read(clusters.indexBuffer, cells * MAX_LIGHTS_PER_CLUSTER * 4));

    const view = cam.view;
    const lights = [];
    for (let i = 0; i < lightScene.lightCount; i++) {
      const o = i * 16;
      const [x, y, z, r] = lightScene.lights.subarray(o, o + 4);
      lights.push({
        c: [
          view[0] * x + view[4] * y + view[8] * z + view[12],
          view[1] * x + view[5] * y + view[9] * z + view[13],
          view[2] * x + view[6] * y + view[10] * z + view[14],
        ],
        r,
      });
    }

    let pairs = 0;
    let ambiguous = 0;
    for (let cell = 0; cell < cells; cell++) {
      const min = bounds.subarray(cell * 8, cell * 8 + 3);
      const max = bounds.subarray(cell * 8 + 4, cell * 8 + 7);
      const listed = new Set(indices.subarray(cell * MAX_LIGHTS_PER_CLUSTER,
        cell * MAX_LIGHTS_PER_CLUSTER + Math.min(counts[cell], MAX_LIGHTS_PER_CLUSTER)));
      let expected = 0;
      for (let i = 0; i < lights.length; i++) {
        const { c, r } = lights[i];
        let d = 0;
        for (let k = 0; k < 3; k++) {
          const out = Math.max(min[k] - c[k], 0) + Math.max(c[k] - max[k], 0);
          d += out * out;
        }
        // f32 on the GPU, f64 here: a sphere grazing a box is a coin toss for
        // both, and saying which is right would be measuring the rounding.
        if (Math.abs(d - r * r) < 1e-3 * r * r) { ambiguous++; continue; }
        const overlaps = d < r * r;
        if (overlaps) expected++;
        if (overlaps && !listed.has(i) && counts[cell] <= MAX_LIGHTS_PER_CLUSTER) {
          throw new Error(`cell ${cell}: light ${i} overlaps it and is not listed`);
        }
        if (!overlaps && listed.has(i)) throw new Error(`cell ${cell}: light ${i} is listed and does not overlap it`);
      }
      pairs += expected;
    }
    if (pairs === 0) throw new Error('no light overlapped any cell, so this checked nothing');
    return `${pairs} light-cell pairs over ${cells} cells match a brute force (${ambiguous} grazing, skipped)`;
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
    await engine.gpu.device.queue.onSubmittedWorkDone();

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
    await engine.gpu.device.queue.onSubmittedWorkDone();

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
    await engine.gpu.device.queue.onSubmittedWorkDone();

    const afterTop = scene.worldMax[riggedIndex * 3 + 1];
    if (!(afterTop > beforeTop + 5)) {
      throw new Error(
        `bounds did not follow the joint: top was ${beforeTop.toFixed(2)}, now ${afterTop.toFixed(2)}`,
      );
    }
    // And the box the CULL SHADER reads. The upload was gated on the mesh
    // node moving, so this one stayed at the bind pose while the one above
    // was right -- and the GPU culled the character by its old box.
    const culledTop = engine.renderer.gpu.boundsData[riggedIndex * 12 + 5];
    if (culledTop !== Math.fround(afterTop)) {
      throw new Error(`the cull box top is ${culledTop.toFixed(2)}; the scene's is ${afterTop.toFixed(2)}`);
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
    await engine.gpu.device.queue.onSubmittedWorkDone();

    const device = engine.gpu.device;
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
    await engine.gpu.device.queue.onSubmittedWorkDone();

    const posedTop = scene.worldMax[index * 3 + 1];
    const restingFront = scene.worldMax[index * 3 + 2];
    if (!(posedTop > 12)) {
      throw new Error(`bounds did not follow the joint: top is ${posedTop.toFixed(2)}`);
    }

    // Now the weight, on an axis the skeleton did not touch.
    bothNode.weights[0] = 1;
    engine.renderFrame(scene, camera);
    await engine.gpu.device.queue.onSubmittedWorkDone();

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

    // WHERE is a fraction of the canvas, not a pixel index. This used to read
    // ((32 * 64) + 32) * 4 on the assumption that the canvas stayed 64x64 --
    // but the engine sizes its backing store from the CSS box, which is
    // 320x240 on this page, so "the middle" was pixel (32, 32) near the top-
    // left corner. Every check still passed because the quad fills the view,
    // and the one light check passed only because its radius covered the
    // corner too. A light aimed at the middle measured nothing.
    // `points` reads several places from the one frame, as a list.
    const shootWith = async (options, light, { orthographic = false, at = [0.5, 0.5], points = null } = {}) => {
      const scene = probe.createScene();
      const cam = new Camera({ fovY: 1.0, near: 0.1, orthographic });
      cam.position.set([0, 0, 2]);
      cam.target.set([0, 0, 0]);
      scene.add(await probe.load(buildFeatureGLB(options)));
      // A function gets the scene to arrange as it likes, lights and all. An
      // object is one light over the key light every other shot has.
      if (typeof light === 'function') {
        light(scene);
      } else {
        scene.addLight({ type: 'directional', direction: [-0.35, -0.55, -0.45], color: [1, 0.9375, 0.84375], intensity: 3.2 });
        if (light) scene.addLight(light);
      }
      probe.renderFrame(scene, cam);
      // Once per frame, before anything else awaits -- see rhi.readPixels.
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      const read = ([x, y]) => {
        const i = (Math.floor(y * height) * width + Math.floor(x * width)) * 4;
        return [pixels[i], pixels[i + 1], pixels[i + 2]];
      };
      return points ? points.map(read) : read(at);
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

      // KHR_texture_transform: the same red UVs, moved half a texture along u
      // by the material, land in the blue texel. The importer's matrix is
      // checked against the spec in Node; this is the shader applying it.
      const moved = await shoot({ imageURI: image, uv0: LEFT, uv1: LEFT, baseColorTransform: { offset: [0.5, 0] } });
      expect('a texture transform moves where the texture is sampled', moved, blue(moved));

      {
        // The material extensions, lit head-on by a point light so the
        // highlight sits in the middle. Red only: the sky is pure green, so red
        // is the light's alone.
        const lamp = (scene) => scene.addLight({ position: [0, 0, 1], color: [1, 1, 1], intensity: 3, radius: 4 });
        const dielectric = { baseColorFactor: [0.1, 0.1, 0.1, 1], metallicFactor: 0, roughnessFactor: 0.4 };
        const shootExt = (materialExtensions, extra = {}) => shootWith({ ...dielectric, materialExtensions, ...extra }, lamp);
        const [plain] = await shootExt(null);
        const [denser] = await shootExt({ KHR_materials_ior: { ior: 3 } });
        const [matte] = await shootExt({ KHR_materials_specular: { specularFactor: 0 } });
        const [cyan] = await shootExt({ KHR_materials_specular: { specularColorFactor: [0, 1, 1] } });
        // specularTexture's alpha is the strength: 0 everywhere, so no highlight,
        // through the extension binding rather than the factor.
        const [mapped] = await shootExt({ KHR_materials_specular: { specularTexture: { index: 0 } } }, {
          extraImageURIs: [twoToneImageURI('rgba(255,255,255,0)', 'rgba(255,255,255,0)')],
        });
        const specular = `red ${plain} plain, ${denser} at ior 3, ${matte} at specular 0, `
          + `${cyan} tinted cyan, ${mapped} with a zero specular map`;
        if (!(denser > plain + 20 && matte < plain - 20 && cyan < plain - 20 && Math.abs(mapped - matte) <= 2)) {
          throw new Error(`ior and specular: ${specular}`);
        }
        results.push(`ior and specular (${specular})`);

        // A clear coat over a fully rough base: the lamp's sharp highlight is
        // the coat's alone. A normal map tilting the coat 45 degrees moves the
        // highlight off the middle, so the coat's own normal is what it used.
        const roughBase = { baseColorFactor: [0.1, 0.1, 0.1, 1], metallicFactor: 0, roughnessFactor: 1 };
        const shootRough = (materialExtensions, extra = {}) => shootWith({ ...roughBase, materialExtensions, ...extra }, lamp);
        const bare = await shootRough(null);
        const [coated] = await shootRough({ KHR_materials_clearcoat: { clearcoatFactor: 1 } });
        const [tilted] = await shootRough({ KHR_materials_clearcoat: { clearcoatFactor: 1, clearcoatNormalTexture: { index: 0 } } }, {
          extraImageURIs: [twoToneImageURI('rgb(218,128,218)', 'rgb(218,128,218)')],
        });
        const coat = `red ${bare[0]} bare, ${coated} coated, ${tilted} with the coat's normal tilted`;
        if (!(coated > bare[0] + 60 && tilted < coated - 60)) throw new Error(`clearcoat: ${coat}`);
        results.push(`clearcoat (${coat})`);

        // Sheen in green, lit by the green sky: its rim brightens green and
        // takes its share of the base's red. Sheen roughness 0 from the
        // texture's alpha reflects nothing, and leaves the base alone.
        const sheenOf = (extra) => ({ KHR_materials_sheen: { sheenColorFactor: [0, 1, 0], sheenRoughnessFactor: 1, ...extra } });
        const sheen = await shootRough(sheenOf({}));
        const smooth = await shootRough(sheenOf({ sheenRoughnessTexture: { index: 0 } }), {
          extraImageURIs: [twoToneImageURI('rgba(255,255,255,0)', 'rgba(255,255,255,0)')],
        });
        const sheenText = `${show(bare)} bare, ${show(sheen)} with sheen, ${show(smooth)} at sheen roughness 0`;
        if (!(sheen[1] > bare[1] + 20 && sheen[0] < bare[0] && smooth.every((c, k) => c === bare[k]))) {
          throw new Error(`sheen: ${sheenText}`);
        }
        results.push(`sheen (${sheenText})`);

        // Anisotropy stretches the lamp's highlight along the tangent -- u,
        // which runs along x on this quad -- and a quarter turn stretches it
        // along y instead. Read beside the middle, a step along each axis.
        const glossy = { baseColorFactor: [0.05, 0.05, 0.05, 1], metallicFactor: 0, roughnessFactor: 0.3 };
        const beside = { points: [[0.62, 0.5], [0.5, 0.62]] };
        const stretched = async (rotation) => (await shootWith({
          ...glossy, materialExtensions: { KHR_materials_anisotropy: { anisotropyStrength: 1, anisotropyRotation: rotation } },
        }, lamp, beside)).map((p) => p[0]);
        const [isoX, isoY] = (await shootWith(glossy, lamp, beside)).map((p) => p[0]);
        const [alongX, acrossY] = await stretched(0);
        const [acrossX, alongY] = await stretched(Math.PI / 2);
        const aniso = `beside the highlight, x and y: ${isoX} ${isoY} plain, ${alongX} ${acrossY} along u, ${acrossX} ${alongY} turned`;
        if (!(alongX > isoX + 30 && acrossY < isoY - 15 && alongY > isoY + 15 && acrossX < isoX - 5)) {
          throw new Error(`anisotropy: ${aniso}`);
        }
        results.push(`anisotropy (${aniso})`);

        // A 400 nm film colours a white highlight; a film of 0 nm leaves it as
        // the plain surface has it. Red against blue, which the green sky adds
        // nothing to.
        const film = (nm) => shootWith({
          ...glossy, materialExtensions: { KHR_materials_iridescence: { iridescenceFactor: 1, iridescenceThicknessMaximum: nm } },
        }, lamp);
        const plainHighlight = await shootWith(glossy, lamp);
        const tinted = await film(400);
        const bareFilm = await film(0);
        const redBlue = ([r, , b]) => Math.abs(r - b);
        const irid = `${show(plainHighlight)} plain, ${show(tinted)} at 400 nm, ${show(bareFilm)} at 0 nm`;
        if (!(redBlue(tinted) > 30 && redBlue(bareFilm) <= 3 && bareFilm.every((c, k) => Math.abs(c - plainHighlight[k]) <= 8))) {
          throw new Error(`iridescence: ${irid}`);
        }
        results.push(`iridescence (${irid})`);

        // KHR_materials_unlit: the base colour, whatever the light.
        const unlitQuad = { baseColorFactor: [0.5, 0.25, 0.125, 1], materialExtensions: { KHR_materials_unlit: {} } };
        const unlitDark = await shootWith(unlitQuad, () => {});
        const unlitLit = await shootWith(unlitQuad, lamp);
        if (!(red(unlitDark) && unlitDark.every((c, k) => c === unlitLit[k]))) {
          throw new Error(`unlit: ${show(unlitDark)} in the dark, ${show(unlitLit)} under a lamp`);
        }
        results.push('an unlit surface ignores the light');
      }

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

      // The same, through an orthographic camera and well off-centre. The
      // froxel builder used to assume every ray passes through the eye, which
      // is only true in perspective: under ortho an edge froxel's box missed
      // the cell it stood for, and a light there lit nothing. At the centre
      // the two constructions agree, which is why this reads near the edge.
      const ortho = { orthographic: true, at: [0.8, 0.5] };
      // Distance 2, fovY 1: half height 2 tan(0.5) = 1.09, and at 4:3 half
      // width 1.46, so 80% across is x = 0.6 * 1.46 = 0.87 in world space. A
      // small radius keeps the light local to it -- and keeps its depth range,
      // 1.8 +- 0.5 from the camera, clear of depth 1, which is where every
      // fragment would look for its lights if view depth were read from clip.w
      // (always 1 under ortho) instead of measured along the view axis.
      const orthoLight = { position: [0.87, 0, 0.2], color: [1, 1, 1], intensity: 6, radius: 0.5 };
      const orthoUnlit = await shootWith({ baseColorFactor: [0.6, 0.6, 0.6, 1] }, null, ortho);
      const orthoLit = await shootWith({ baseColorFactor: [0.6, 0.6, 0.6, 1] }, orthoLight, ortho);
      if (!(orthoLit[0] > orthoUnlit[0] + 20)) {
        throw new Error(
          `an off-centre light did not reach the surface under ortho: ${show(orthoUnlit)} -> ${show(orthoLit)}`,
        );
      }
      results.push('an off-centre light lights an orthographic view');

      // A lamp that came IN THE FILE. The CPU suites prove the importer reads
      // it and the scene attaches it, each against hand-built data; this is
      // the one place engine.load's asset is what reaches scene.add.
      const fileLit = await shootWith({
        baseColorFactor: [0.6, 0.6, 0.6, 1],
        lamp: { translation: [0, 0, 0.2], light: { type: 'point', intensity: 6, range: 0.5 } },
      }, null);
      if (!(fileLit[0] > unlit[0] + 20)) {
        throw new Error(`a light imported from glTF did not reach the surface: ${show(unlit)} -> ${show(fileLit)}`);
      }
      results.push('a light imported from glTF lights the surface');

      // EVERY directional light lights, whatever else there is. The bright one
      // faces away from the quad and adds nothing; the dim one faces it, and
      // one that does not cast lights as well as one that does.
      const grey = { baseColorFactor: [0.6, 0.6, 0.6, 1] };
      const dark = await shootWith(grey, () => {});
      const fill = await shootWith(grey, (scene) => {
        scene.addLight({ type: 'directional', direction: [0, 0, 1], intensity: 20 });   // away
        scene.addLight({ type: 'directional', direction: [0, 0, -1], intensity: 2, castShadow: false });   // at it
      });
      if (!(fill[0] > dark[0] + 20)) {
        throw new Error(`a directional light that casts no shadow lit nothing: ${show(dark)} -> ${show(fill)}`);
      }
      results.push('a directional light that casts no shadow still lights');

      // A SHADOW IS CAST -- nothing checked this, and for as long as the
      // cascade near plane was clamped in front of a light eye at the world
      // origin, nothing on the sun's side of that origin cast one. The floor
      // is the feature quad at the origin; the blocker is a small quad in
      // front of it, turned away from the sun so the shadow pass (which culls
      // front faces) keeps it. The camera looks at the floor's centre from the
      // side, past the blocker, where its shadow falls.
      const shadowed = async (withBlocker, blocker = { baseColorFactor: [0.8, 0.8, 0.8, 1] }) => {
        const scene = probe.createScene();
        scene.addLight({ type: 'directional', direction: [0, 0, -1], intensity: 3 });
        scene.add(await probe.load(buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.8, 1] })));
        if (withBlocker) {
          scene.add(await probe.load(buildFeatureGLB({ ...blocker, nodeScale: [0.25, 0.25, 1] })))
            .setPosition(0, 0, 0.8).setAxisAngle([0, 1, 0], Math.PI);
        }
        const cam = new Camera({ fovY: 0.9, near: 0.05 });
        cam.position.set([2.5, 0.3, 2.5]);
        cam.target.set([0, 0, 0]);
        for (let i = 0; i < 3; i++) { probe.renderFrame(scene, cam); await probe.gpu.device.queue.onSubmittedWorkDone(); }
        probe.renderFrame(scene, cam);
        const pixels = await probe.gpu.readPixels();
        const { width, height } = probe.gpu;
        const i = (Math.floor(height / 2) * width + Math.floor(width / 2)) * 4;
        return [pixels[i], pixels[i + 1], pixels[i + 2]];
      };
      const open = await shadowed(false);
      const shaded = await shadowed(true);
      if (!(shaded[0] + shaded[1] + shaded[2] < (open[0] + open[1] + open[2]) * 0.8)) {
        throw new Error(`a caster above the floor cast no shadow: ${show(open)} -> ${show(shaded)}`);
      }
      results.push('a caster on the sun side of the origin casts a shadow');

      // ALPHA SHAPES THE SHADOW. A masked caster whose alpha is under its
      // cutoff is invisible, and cast a full shadow anyway: the shadow pass had
      // no fragment stage to test alpha with. A blended one cast none at all.
      const sum = (c) => c[0] + c[1] + c[2];
      const cutAway = await shadowed(true, { baseColorFactor: [0.8, 0.8, 0.8, 0.2], alphaMode: 'MASK', alphaCutoff: 0.5 });
      if (!(sum(cutAway) > sum(open) * 0.95)) {
        throw new Error(`a caster masked away still cast a shadow: ${show(open)} -> ${show(cutAway)}`);
      }
      results.push('a masked-away caster casts no shadow');
      const glass = await shadowed(true, { baseColorFactor: [0.8, 0.8, 0.8, 0.5], alphaMode: 'BLEND' });
      if (!(sum(glass) < sum(open) * 0.9 && sum(glass) > sum(shaded) * 1.1)) {
        throw new Error(`a half-transparent caster should cast a partial shadow: open ${show(open)}, glass ${show(glass)}, solid ${show(shaded)}`);
      }
      results.push('a half-transparent caster casts a partial shadow');

      // A SMOOTHER SURFACE HAS A BRIGHTER HIGHLIGHT. GGX's divisor floor used to
      // sit above what a smooth lobe divides by, which inverted exactly this:
      // at roughness 0.05 the peak was 62, below the 199 of roughness 0.2.
      // Camera and sun both straight down the quad's normal, so the centre
      // pixel is the peak of the lobe.
      const highlight = async (roughnessFactor, intensity) => {
        const scene = probe.createScene();
        scene.addLight({ type: 'directional', direction: [0, 0, -1], intensity });
        scene.add(await probe.load(buildFeatureGLB({
          baseColorFactor: [0.5, 0.5, 0.5, 1], metallicFactor: 1, roughnessFactor,
        })));
        const cam = new Camera({ fovY: 1.0, near: 0.1 });
        cam.position.set([0, 0, 2]);
        cam.target.set([0, 0, 0]);
        probe.renderFrame(scene, cam);
        const pixels = await probe.gpu.readPixels();
        const { width, height } = probe.gpu;
        const i = (Math.floor(height / 2) * width + Math.floor(width / 2)) * 4;
        return [pixels[i], pixels[i + 1], pixels[i + 2]];
      };
      const glossy = await highlight(0.05, 0.02);
      const rough = await highlight(0.2, 0.02);
      if (!(glossy[0] > rough[0] + 10)) {
        throw new Error(`a smoother highlight was not brighter: rough ${show(rough)}, glossy ${show(glossy)}`);
      }
      results.push('a smoother highlight is brighter');

      // And a near-mirror under a bright sun must come out white, not black.
      // The highlight now reaches far past what the half-float target holds;
      // where the store saturates this passes with or without the shader's
      // clamp, and where it rounds to infinity it passes only with it.
      const blown = await highlight(0.045, 50);
      if (!(blown[0] > 200 && blown[1] > 200 && blown[2] > 200)) {
        throw new Error(`a near-mirror under a bright sun did not come out white: ${show(blown)}`);
      }
      results.push('a mirror highlight saturates instead of overflowing');

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

  await step('two directional lights each cast their own shadow', async () => {
    // A blocker over the ground under two directional lights slanting in from
    // opposite sides: its two shadows fall either side of it, one per light,
    // each from that light's own cascades. Turning one light's castShadow off
    // must take away its shadow and leave the other.
    const ground = await engine.load(buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.8, 1], roughnessFactor: 0.9 }));
    const block = await engine.load(buildFeatureGLB({ baseColorFactor: [0.2, 0.2, 0.2, 1] }));
    const cam = new Camera({ fovY: Math.PI / 3, near: 0.1 });
    cam.position.set([0, 6, 0.01]);
    cam.target.set([0, 0, 0]);
    const shot = async (castRight, castLeft) => {
      const scene = engine.createScene();
      scene.add(ground).setAxisAngle([1, 0, 0], -Math.PI / 2).setScale(5, 5, 1);
      scene.add(block).setAxisAngle([1, 0, 0], Math.PI / 2).setScale(0.35, 0.35, 1).setPosition(0, 1, 0);
      scene.addLight({ type: 'directional', direction: [1, -1, 0], intensity: 5, castShadow: castRight });  // shadow at +x
      scene.addLight({ type: 'directional', direction: [-1, -1, 0], intensity: 5, castShadow: castLeft }); // shadow at -x
      engine.renderFrame(scene, cam);
      const pixels = await engine.gpu.readPixels();
      const { width, height } = engine.gpu;
      const at = ([x, y, z]) => {
        const m = cam.viewProjection;
        const w = m[3] * x + m[7] * y + m[11] * z + m[15];
        const px = Math.round(((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * width);
        const py = Math.round((0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5) * height);
        const i = (py * width + px) * 4;
        return 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
      };
      return { right: at([1, 0, 0]), left: at([-1, 0, 0]), casting: engine.renderer.shadows.shadowedCount };
    };
    // Each point against itself with nothing casting: a shadow removes one of
    // two lights there, and the tonemap keeps that a modest step.
    const none = await shot(false, false);
    const both = await shot(true, true);
    const one = await shot(true, false);
    if (none.casting !== 0 || both.casting !== 2 || one.casting !== 1) {
      throw new Error(`${none.casting}, ${both.casting} and ${one.casting} lights casting, expected 0, 2 and 1`);
    }
    if (!(both.right < none.right - 8 && both.left < none.left - 8)) {
      throw new Error(`both shadows should fall: right ${none.right.toFixed(0)} -> ${both.right.toFixed(0)}, left ${none.left.toFixed(0)} -> ${both.left.toFixed(0)}`);
    }
    if (Math.abs(one.left - none.left) > 2 || Math.abs(one.right - both.right) > 2) {
      throw new Error(`one light's shadow off: left ${one.left.toFixed(0)} (unshadowed ${none.left.toFixed(0)}), right ${one.right.toFixed(0)} (shadowed ${both.right.toFixed(0)})`);
    }
    return `right ${none.right.toFixed(0)} -> ${both.right.toFixed(0)}, left ${none.left.toFixed(0)} -> ${both.left.toFixed(0)}; the left light's off, only its shadow goes`;
  });


  await step('point and spot lights cast shadows when asked, and only where the caster is', async () => {
    // A blocker over the ground, one light above it and no sun, from straight
    // above. The ground under the blocker must darken when the light casts,
    // and the ground beside it must not change. The three cases are the three
    // shapes a view can take: a point light's cube, a narrow spot's single
    // frustum, and a spot wider than a cube face, which takes the cube.
    const ground = await engine.load(buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.8, 1], roughnessFactor: 0.9 }));
    const block = await engine.load(buildFeatureGLB({ baseColorFactor: [0.2, 0.2, 0.2, 1] }));
    const cam = new Camera({ fovY: Math.PI / 3, near: 0.1 });
    cam.position.set([0, 6, 0.01]);
    cam.target.set([0, 0, 0]);
    const brightness = async (light) => {
      const scene = engine.createScene();
      scene.add(ground).setAxisAngle([1, 0, 0], -Math.PI / 2).setScale(5, 5, 1);
      // Facing down, so the light sees its back: the shadow pass culls fronts.
      scene.add(block).setAxisAngle([1, 0, 0], Math.PI / 2).setScale(0.35, 0.35, 1).setPosition(0, 1, 0);
      scene.addLight({ position: [0, 3, 0], intensity: 40, radius: 12, ...light });
      engine.renderFrame(scene, cam);
      const pixels = await engine.gpu.readPixels();
      const { width, height } = engine.gpu;
      const at = (fx) => {
        const i = (Math.floor(height / 2) * width + Math.floor(fx * width)) * 4;
        return 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
      };
      return { under: at(0.5), beside: at(0.7), views: engine.renderer.shadows.localCount };
    };
    const cases = [
      ['point', { type: 'point' }, 6],
      ['narrow spot', { type: 'spot', direction: [0, -1, 0], outerAngle: 0.6, innerAngle: 0.3 }, 1],
      ['wide spot', { type: 'spot', direction: [0, -1, 0], outerAngle: 1.1, innerAngle: 0.5 }, 6],
    ];
    const report = [];
    for (const [name, light, expectedViews] of cases) {
      const lit = await brightness(light);
      const shadowed = await brightness({ ...light, castShadow: true });
      if (!(shadowed.under < lit.under - 20)) {
        throw new Error(`${name}: under the blocker ${lit.under.toFixed(0)} -> ${shadowed.under.toFixed(0)}, no shadow`);
      }
      if (Math.abs(shadowed.beside - lit.beside) > 2) {
        throw new Error(`${name}: beside the blocker ${lit.beside.toFixed(0)} -> ${shadowed.beside.toFixed(0)}, shadow where nothing casts`);
      }
      if (shadowed.views !== expectedViews || lit.views !== 0) {
        throw new Error(`${name}: ${shadowed.views} views casting and ${lit.views} not, expected ${expectedViews} and 0`);
      }
      report.push(`${name} ${lit.under.toFixed(0)} -> ${shadowed.under.toFixed(0)} (${expectedViews} views)`);
    }
    return report.join(', ');
  });

  await step('a point light keeps its shadow maps while nothing near it moves, and redraws when a caster does', async () => {
    // The cached frame must be the frame a fresh draw gives, pixel for pixel;
    // a caster that moves must take its shadow with it; and something moving
    // far outside the light's reach must not cost it a draw.
    const ground = await engine.load(buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.8, 1], roughnessFactor: 0.9 }));
    const block = await engine.load(buildFeatureGLB({ baseColorFactor: [0.2, 0.2, 0.2, 1] }));
    const cam = new Camera({ fovY: Math.PI / 3, near: 0.1 });
    cam.position.set([0, 6, 0.01]);
    cam.target.set([0, 0, 0]);
    const scene = engine.createScene();
    scene.add(ground).setAxisAngle([1, 0, 0], -Math.PI / 2).setScale(5, 5, 1);
    const blocker = scene.add(block).setAxisAngle([1, 0, 0], Math.PI / 2).setScale(0.35, 0.35, 1).setPosition(0, 1, 0);
    const far = scene.add(block).setPosition(40, 0, 0);
    scene.addLight({ type: 'point', position: [0, 3, 0], intensity: 40, radius: 8, castShadow: true });
    const frame = async () => {
      engine.renderFrame(scene, cam);
      const pixels = await engine.gpu.readPixels();
      return { pixels, drawn: engine.renderer.stats.shadowViewsDrawn };
    };
    const { width, height } = engine.gpu;
    const lum = (pixels, fx) => {
      const i = (Math.floor(height / 2) * width + Math.floor(fx * width)) * 4;
      return 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
    };
    const same = (a, b) => a.every((v, i) => v === b[i]);

    const first = await frame();
    const kept = await frame();
    far.setPosition(41, 0, 0);
    const farMoved = await frame();
    blocker.setPosition(1.2, 1, 0);
    const moved = await frame();
    const keptAgain = await frame();
    const report = `drawn ${first.drawn}, ${kept.drawn}, ${farMoved.drawn} (far caster moved), ${moved.drawn} (blocker moved), ${keptAgain.drawn}; `
      + `centre ${lum(first.pixels, 0.5).toFixed(0)} -> ${lum(moved.pixels, 0.5).toFixed(0)}`;
    if (first.drawn !== 6 || kept.drawn !== 0 || farMoved.drawn !== 0 || moved.drawn !== 6 || keptAgain.drawn !== 0) throw new Error(report);
    if (!same(first.pixels, kept.pixels) || !same(moved.pixels, keptAgain.pixels)) throw new Error(`${report}; a kept map drew a different frame`);
    if (!(lum(moved.pixels, 0.5) > lum(first.pixels, 0.5) + 20)) throw new Error(`${report}; the shadow did not leave the centre`);
    return report;
  });


  await step('an equirectangular map becomes the sky, the right way round', async () => {
    // Above the horizon: green on the half of the panorama centred on +X (the
    // image's middle, three.js's layout), red on the other half. Below: blue.
    // Looking each way must see what the map put there, through the bake.
    const width = 64, height = 32;
    const data = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const o = (y * width + x) * 3;
        const u = (x + 0.5) / width;
        if (y >= height / 2) data[o + 2] = 1;
        else if (u > 0.25 && u < 0.75) data[o + 1] = 1;
        else data[o] = 1;
      }
    }
    const environment = new Environment(engine.gpu, { map: { width, height, data } });
    if (environment.environment.width !== width / 4) {
      throw new Error(`a ${width}-wide map baked a ${environment.environment.width} cube, expected ${width / 4}`);
    }
    const scene = engine.createScene({ environment });
    const look = async (target) => {
      const cam = new Camera({ fovY: 0.5, near: 0.1 });
      cam.position.set([0, 0, 0]);
      cam.target.set(target);
      engine.renderFrame(scene, cam);
      const pixels = await engine.gpu.readPixels();
      const { width: w, height: h } = engine.gpu;
      const i = (Math.floor(h / 2) * w + Math.floor(w / 2)) * 4;
      return [pixels[i], pixels[i + 1], pixels[i + 2]];
    };
    const dominant = ([r, g, b]) => (r > g && r > b ? 'red' : g > b ? 'green' : 'blue');
    const seen = {
      '+X': dominant(await look([1, 0.4, 0])),
      '-X': dominant(await look([-1, 0.4, 0])),
      down: dominant(await look([0.01, -1, 0])),
    };
    environment.destroy();
    if (seen['+X'] !== 'green' || seen['-X'] !== 'red' || seen.down !== 'blue') {
      throw new Error(`looking +X, -X and down saw ${seen['+X']}, ${seen['-X']}, ${seen.down}`);
    }
    return 'green along +X, red behind, blue below';
  });


  await step('ambient occlusion darkens a corner and leaves open floor alone', async () => {
    // A floor meeting a wall, lit by the sky alone, rendered by an engine with
    // occlusion and one without. Where the wall meets the floor, the floor
    // sees half the sky and must come out darker. Open floor two radii from
    // anything sees all of it and must not change -- an integral that reads
    // the same as an unoccluded surface there is the whole claim of GTAO.
    const brightness = async (ao, points) => {
      const canvas = document.createElement('canvas');
      canvas.style.width = '320px';
      canvas.style.height = '240px';
      document.body.appendChild(canvas);
      const probe = await Winding.create(canvas, { ao });
      probe.gpu.resize(320, 240);
      const scene = probe.createScene();
      const quad = await probe.load(buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.8, 1], roughnessFactor: 1 }));
      scene.add(quad).setAxisAngle([1, 0, 0], -Math.PI / 2).setScale(4, 4, 1);
      scene.add(quad).setScale(4, 2, 1).setPosition(0, 3.2, -1.2);
      const cam = new Camera({ fovY: 1.0, near: 0.1 });
      cam.position.set([0, 1.2, 4]);
      cam.target.set([0, 0, -1]);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      const values = points.map(([x, y, z]) => {
        const m = cam.viewProjection;
        const w = m[3] * x + m[7] * y + m[11] * z + m[15];
        const px = Math.round(((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * width);
        const py = Math.round((0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5) * height);
        const i = (py * width + px) * 4;
        return 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
      });
      probe.destroy();
      canvas.remove();
      return values;
    };
    // At the foot of the wall, and open floor well in front of it.
    const points = [[0, 0, -1.1], [0, 0, 2.5]];
    const [cornerOff, openOff] = await brightness(false, points);
    const [cornerOn, openOn] = await brightness(true, points);
    if (!(cornerOn < cornerOff - 3)) {
      throw new Error(`the corner went ${cornerOff.toFixed(1)} -> ${cornerOn.toFixed(1)}; occlusion should darken it`);
    }
    if (Math.abs(openOn - openOff) > 1) {
      throw new Error(`open floor went ${openOff.toFixed(1)} -> ${openOn.toFixed(1)}; nothing occludes it`);
    }
    return `corner ${cornerOff.toFixed(0)} -> ${cornerOn.toFixed(0)}, open floor ${openOff.toFixed(0)} -> ${openOn.toFixed(0)}`;
  });


  await step('antialiasing, ambient occlusion and OIT switch on and off at runtime', async () => {
    // An engine made with all three off. Asked for later, each builds in the
    // background; a frame after that has it, and a frame after switching it
    // off does not. The tilted quad's edge shows the antialiasing.
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      antialias: false, post: { strength: 0 }, exposure: 4,
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
    });
    probe.gpu.device.pushErrorScope('validation');
    probe.gpu.resize(320, 240);
    const scene = probe.createScene();
    scene.add(await probe.load(buildFeatureGLB({ baseColorFactor: [1, 1, 1, 1], emissiveFactor: [1, 1, 1] })))
      .setAxisAngle([0, 0, 1], 0.3).setScale(0.5, 0.5, 1);
    scene.add(await probe.load(buildFeatureGLB({ baseColorFactor: [0.2, 0.4, 1, 0.5], alphaMode: 'BLEND' })))
      .setPosition(0.6, 0, 0.5).setScale(0.3, 0.3, 1);
    const cam = new Camera({ fovY: 1.0, near: 0.1 });
    cam.position.set([0, 0, 3]);
    cam.target.set([0, 0, 0]);
    const r = probe.renderer;
    const look = async () => {
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      let between = 0;
      for (let i = 0; i < pixels.length; i += 4) if (pixels[i + 1] > 25 && pixels[i + 1] < 230) between++;
      return { aa: r.post._antialiasing, ao: r._frameAO, oit: r._frameOIT, between };
    };
    const off = await look();
    r.post.antialias = true;
    r.ao = { radius: 0.2 };
    r.oit = true;
    const asked = await look();   // asks: builds start, this frame draws without them
    await Promise.all([r.post._fxaaBuilding, ...[...r._variantSets.values()].map((set) => set.building)]);
    const on = await look();
    r.post.antialias = false;
    r.ao = null;
    r.oit = false;
    const offAgain = await look();
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();
    const show = (s) => `aa ${s.aa}, ao ${s.ao}, oit ${s.oit}, ${s.between} px between`;
    const report = `made off: ${show(off)}; the frame that asked: ${show(asked)}; once built: ${show(on)}; off again: ${show(offAgain)}`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const is = (s, v) => s.aa === v && s.ao === v && s.oit === v;
    if (!(is(off, false) && is(on, true) && is(offAgain, false) && on.between > 50 && offAgain.between < on.between / 4)) throw new Error(report);
    return report;
  });

  await step('antialiasing softens a hard edge, and off leaves it hard', async () => {
    // A flat quad turned a little against a flat sky, unlit so the edge is
    // one colour meeting another. Without antialiasing every pixel along it
    // is one or the other; FXAA blends the ones the edge crosses.
    const SKY = [0, 0, 0];
    const blended = async (antialias) => {
      const canvas = document.createElement('canvas');
      canvas.style.width = '320px';
      canvas.style.height = '240px';
      document.body.appendChild(canvas);
      const probe = await Winding.create(canvas, {
        antialias,
        // No bloom, whose halo would blend the edge either way, and a quad bright
        // enough to sit well clear of the band counted as between.
        post: { strength: 0 },
        exposure: 4,
        environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
      });
      probe.gpu.resize(320, 240);
      const scene = probe.createScene();
      scene.add(await probe.load(buildFeatureGLB({ baseColorFactor: [1, 1, 1, 1], emissiveFactor: [1, 1, 1] })))
        .setAxisAngle([0, 0, 1], 0.3).setScale(0.5, 0.5, 1);
      const cam = new Camera({ fovY: 1.0, near: 0.1 });
      cam.position.set([0, 0, 3]);
      cam.target.set([0, 0, 0]);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      let between = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        const g = pixels[i + 1];
        if (g > 25 && g < 230) between++;
      }
      probe.destroy();
      canvas.remove();
      return between;
    };
    const hard = await blended(false);
    const soft = await blended(true);
    if (!(soft > hard * 3 && soft > 100)) {
      throw new Error(`pixels between the two colours: ${hard} without, ${soft} with antialiasing`);
    }
    return `${hard} edge pixels between the two colours without, ${soft} with`;
  });

  await step('a lower resolution draws the view smaller, and NIS brings it up sharper than a stretch', async () => {
    // The same hard edge as above, at full resolution, then at half: brought
    // up by NIS, and stretched bilinearly (the scaler's pipeline hidden for
    // the frame). Both leave some pixels between the two colours at twice the
    // scale; a stretch leaves a ramp, many of them halfway, and NIS a step.
    const SKY = [0, 0, 0];
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas, {
      antialias: false, post: { strength: 0 }, exposure: 4,
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
    });
    try {
      probe.gpu.resize(320, 240);
      const scene = probe.createScene();
      scene.add(await probe.load(buildFeatureGLB({ baseColorFactor: [1, 1, 1, 1], emissiveFactor: [1, 1, 1] })))
        .setAxisAngle([0, 0, 1], 0.3).setScale(0.5, 0.5, 1);
      const cam = new Camera({ fovY: 1.0, near: 0.1 });
      cam.position.set([0, 0, 3]);
      cam.target.set([0, 0, 0]);
      const r = probe.renderer;
      const frame = () => {
        probe.renderFrame(scene, cam);
        return probe.gpu.readPixels();
      };
      /** Pixels about halfway between the sky and the quad. */
      const between = (pixels) => {
        let n = 0;
        for (let i = 0; i < pixels.length; i += 4) if (pixels[i + 1] > 64 && pixels[i + 1] < 192) n++;
        return n;
      };
      const lit = (pixels) => {
        let n = 0;
        for (let i = 0; i < pixels.length; i += 4) if (pixels[i + 1] >= 128) n++;
        return n;
      };
      const full = await frame();
      r.resolution = 0.5;
      probe.renderFrame(scene, cam);
      await r.upscaler._scalerReady();
      const nis = await frame();
      if (!r.upscaler.scaled) throw new Error('the scaler never came up');
      if (r._scaled.width !== 160 || r._scaled.height !== 120) throw new Error(`drawn at ${r._scaled.width}x${r._scaled.height}, not 160x120`);
      const scaler = r.upscaler.scalePipeline;
      r.upscaler.scalePipeline = undefined;
      const stretched = await frame();
      r.upscaler.scalePipeline = scaler;
      r.resolution = 1;
      const back = await frame();
      if (r._scaled !== null) throw new Error('the scaled view outlived full resolution');

      // The quad covers the same area at every resolution.
      const area = [full, nis, stretched].map(lit);
      if (area.some((a) => Math.abs(a - area[0]) > area[0] * 0.03)) throw new Error(`lit pixels ${area.join(' / ')}`);
      const edges = [full, nis, stretched].map(between);
      if (!(edges[1] < edges[2] * 0.5)) throw new Error(`edge pixels: full ${edges[0]}, NIS ${edges[1]}, stretched ${edges[2]}`);
      if (between(back) !== edges[0]) throw new Error('full resolution did not come back as it was');
      let refused = '';
      r.resolution = 0.4;
      try { probe.renderFrame(scene, cam); } catch (error) { refused = error.message; }
      r.resolution = 1;
      if (!refused.startsWith('resolution must be from 0.5 to 1')) throw new Error(`0.4 was not refused: ${refused}`);
      return `edge pixels: full ${edges[0]}, NIS ${edges[1]}, stretched ${edges[2]}`;
    } finally {
      probe.destroy();
      canvas.remove();
    }
  });

  await step('splats draw back to front whatever order the file has them in, behind geometry in front', async () => {
    // Two opaque splats on the view axis, red in front of blue, written to
    // the file in both orders: the sort puts red over blue either way. Then
    // a white quad in front of both hides them.
    const SKY = [0, 0, 0];
    const canvas = document.createElement('canvas');
    canvas.style.width = '160px';
    canvas.style.height = '120px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas, {
      antialias: false, post: { strength: 0 },
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
    });
    const record = (z, rgb) => {
      const bytes = new Uint8Array(32);
      const view = new DataView(bytes.buffer);
      [0, 0, z, 0.4, 0.4, 0.4].forEach((v, i) => view.setFloat32(i * 4, v, true));
      bytes.set([...rgb, 255], 24);
      bytes.set([255, 128, 128, 128], 28);
      return bytes;
    };
    const file = (...records) => {
      const out = new Uint8Array(records.length * 32);
      records.forEach((r, i) => out.set(r, i * 32));
      return out;
    };
    try {
      probe.gpu.resize(160, 120);
      const red = record(1, [255, 0, 0]);
      const blue = record(-1, [0, 0, 255]);
      const cam = new Camera({ fovY: 0.8, near: 0.1 });
      cam.position.set([0, 0, 5]);
      cam.target.set([0, 0, 0]);
      const centre = async (scene) => {
        probe.renderFrame(scene, cam);
        const pixels = await probe.gpu.readPixels({ x: 80, y: 60, width: 1, height: 1 });
        return [...pixels.subarray(0, 3)];
      };
      const seen = [];
      for (const bytes of [file(red, blue), file(blue, red)]) {
        const splats = await probe.loadSplats(bytes);
        const scene = probe.createScene();
        scene.addSplats({ splats });
        seen.push(await centre(scene));
        // And behind a quad at z = 2, which hides them both.
        scene.add(await probe.load(buildFeatureGLB({ baseColorFactor: [1, 1, 1, 1], emissiveFactor: [1, 1, 1] })))
          .setPosition(0, 0, 2).setScale(0.5, 0.5, 1);
        seen.push(await centre(scene));
        probe.unload(splats);
        let refused = '';
        try { probe.renderFrame(scene, cam); } catch (error) { refused = error.message; }
        if (!/unloaded/.test(refused)) throw new Error(`an unloaded capture still drew: ${refused}`);
      }
      const [front, hidden, back, hidden2] = seen;
      for (const [name, [r, g, b]] of [['red first', front], ['blue first', back]]) {
        if (!(r > 150 && b < 30)) throw new Error(`${name} in the file: centre is ${r},${g},${b}, not red`);
      }
      for (const [r, g, b] of [hidden, hidden2]) {
        if (!(r > 150 && g > 150 && b > 150)) throw new Error(`the quad in front did not hide the splats: ${r},${g},${b}`);
      }
      return `centre ${front.join(',')} and ${back.join(',')}; behind the quad ${hidden.join(',')}`;
    } finally {
      probe.destroy();
      canvas.remove();
    }
  });

  await step('grading: saturation, a white balance that neutralises its light, and a .cube LUT', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
      post: { strength: 0 },
      antialias: false,
    });
    probe.gpu.device.pushErrorScope('validation');
    const { planckianXY } = await import('../src/render/grading.js');
    // A tungsten light's colour in linear sRGB, from its chromaticity, brightest channel 0.5.
    const [x, y] = planckianXY(3200);
    const X = x / y;
    const Z = (1 - x - y) / y;
    const rgb = [3.2404542 * X - 1.5371385 - 0.4985314 * Z, -0.9692660 * X + 1.8760108 + 0.0415560 * Z, 0.0556434 * X - 0.2040259 + 1.0572252 * Z];
    const peak = Math.max(...rgb);
    const lamp = [...rgb.map((c) => (0.5 * c) / peak), 1];
    const shot = async (color, grading) => {
      probe.grading = grading;
      const scene = probe.createScene();
      scene.add(await probe.load(buildFeatureGLB({ baseColorFactor: color, materialExtensions: { KHR_materials_unlit: {} } })));
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set([0, 0, 2]);
      cam.target.set([0, 0, 0]);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      const i = ((height >> 1) * width + (width >> 1)) * 4;
      return [pixels[i], pixels[i + 1], pixels[i + 2]];
    };
    const orange = [0.8, 0.4, 0.1, 1];
    const plain = await shot(orange, null);
    const grey = await shot(orange, { saturation: 0 });
    const tungsten = await shot(lamp, null);
    const balanced = await shot(lamp, { whiteBalance: 3200 });
    const cube = (f) => `LUT_3D_SIZE 2\n${[0, 1].flatMap((b) => [0, 1].flatMap((g) => [0, 1].map((r) => f(r, g, b).join(' ')))).join('\n')}`;
    const identity = await probe.loadLUT(cube((r, g, b) => [r, g, b]));
    const invert = await probe.loadLUT(cube((r, g, b) => [1 - r, 1 - g, 1 - b]));
    const same = await shot(orange, { lut: identity });
    const inverted = await shot(orange, { lut: invert });
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();
    const show = (p) => `rgb(${p.join(',')})`;
    const spread = (p) => Math.max(...p) - Math.min(...p);
    const report = `${show(plain)} as is, ${show(grey)} at saturation 0; a 3200 K light ${show(tungsten)}, `
      + `${show(balanced)} balanced to it; ${show(same)} through an identity LUT, ${show(inverted)} through an inverting one`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const ok = spread(grey) <= 1 && spread(tungsten) > 60 && spread(balanced) <= 4
      && same.every((c, k) => Math.abs(c - plain[k]) <= 1) && inverted.every((c, k) => Math.abs(c + plain[k] - 255) <= 2);
    if (!ok) throw new Error(report);
    return report;
  });

  await step('depth of field blurs by the lens: sharp at the focus distance, spread away from it', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
      post: { strength: 0 },
      antialias: false,
    });
    probe.gpu.device.pushErrorScope('validation');
    const quad = await probe.load(buildFeatureGLB({ baseColorFactor: [1, 1, 1, 1], materialExtensions: { KHR_materials_unlit: {} } }));
    const { lensCoefficients } = await import('../src/render/dof.js');
    // The quad's right edge along the middle row, the quad 5 m away.
    const edge = async (dof) => {
      probe.renderer.dof = dof;
      const scene = probe.createScene();
      scene.add(quad).setPosition(-1.6, 0, 0);
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set([0, 0, 5]);
      cam.target.set([0, 0, 0]);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      let between = 0;
      for (let x = 0; x < width; x++) {
        const v = pixels[((height >> 1) * width + x) * 4];
        if (v > 20 && v < 220) between++;
      }
      return between;
    };
    const lens = { fStop: 0.2, focusDistance: 5 };
    const none = await edge(null);
    const focused = await edge(lens);
    const near = { fStop: 0.2, focusDistance: 0.5 };
    const blurred = await edge(near);
    // What the lens says the quad's disc is, focused at 0.5: scale (1 - 0.5 / 5).
    const disc = lensCoefficients(near, 1, probe.gpu.height).scale * (1 - 0.5 / 5);
    const error = await probe.gpu.device.popErrorScope();
    probe.renderer.dof = null;
    probe.destroy();
    canvas.remove();
    const report = `edge pixels: ${none} without, ${focused} focused on it, ${blurred} focused at 0.5 m, `
      + `where the lens gives a ${disc.toFixed(0)} px disc`;
    if (error) throw new Error(`${report}; ${error.message}`);
    if (!(focused <= none + 2 && blurred > disc * 0.4 && blurred < disc * 1.6)) throw new Error(report);
    return report;
  });

  await step('text stays sharp magnified, lies in its plane when asked, and survives its atlas growing', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
      post: { strength: 0 },
      antialias: false,
    });
    probe.gpu.device.pushErrorScope('validation');
    const font = await probe.loadFont('128px sans-serif');
    const red = [1, 0, 0, 1];
    const shot = async (build, { from = [0, 0, 3] } = {}) => {
      const scene = probe.createScene();
      build(scene);
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set(from);
      cam.target.set([0, 0, 0]);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      let lit = 0;
      let partial = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i] > 200) lit++;
        else if (pixels[i] > 30) partial++;
      }
      const row = [...Array(width).keys()].map((x) => pixels[((height >> 1) * width + x) * 4]);
      return { lit, partial, row, green: pixels[((height >> 1) * width + (width >> 1)) * 4 + 1] };
    };
    // An I, three units tall three units away: far past its 128 px raster.
    const big = await shot((s) => s.addText({ font, text: 'I', size: 3, color: red }));
    // Its edges along the middle row: every pixel between dark and full.
    const ramp = big.row.filter((v) => v > 30 && v <= 200).length;
    const flat = await shot((s) => s.addText({ font, text: 'I', size: 1, color: red, facing: 'plane' })
      .setAxisAngle([0, 1, 0], Math.PI / 2));
    const facing = await shot((s) => s.addText({ font, text: 'I', size: 1, color: red })
      .setAxisAngle([0, 1, 0], Math.PI / 2));
    // Enough glyphs at 128 px to overflow a 512 atlas, then the I again.
    const before = big.lit;
    font.ensure('ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789');
    const atlas = font.texture.width;
    const after = (await shot((s) => s.addText({ font, text: 'I', size: 3, color: red }))).lit;
    const error = await probe.gpu.device.popErrorScope();
    font.destroy();
    probe.destroy();
    canvas.remove();

    const report = `a magnified I: ${big.lit} px full, ${ramp} px of edge across its middle row; `
      + `${flat.lit + flat.partial} px lying edge-on in its plane, ${facing.lit} px facing; `
      + `after the atlas grew to ${atlas}, ${after} px (was ${before})`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const ok = big.lit > 1000 && big.green < 20 && ramp <= 4
      && flat.lit + flat.partial === 0 && facing.lit > 100
      && atlas > 512 && after === before;
    if (!ok) throw new Error(report);
    return report;
  });

  await step('a decal paints the base colour in its box, is lit as the surface is, only from the side it faces, and the last added on top', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
      post: { strength: 0 },
    });
    probe.gpu.device.pushErrorScope('validation');
    // Red on the left of the image, clear on the right.
    const image = await probe.loadTexture(twoToneImageURI('#ff0000', 'rgba(255,0,0,0)'));
    const green = await probe.loadTexture(twoToneImageURI('#00ff00', 'rgba(0,255,0,0)'));
    const floor = await probe.load(buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0 }));
    const look = async ({ decal = true, fromBelow = false, intensity = 3, layers = [image] } = {}) => {
      const scene = probe.createScene();
      scene.add(floor).setAxisAngle([1, 0, 0], -Math.PI / 2).setScale(3, 3, 1);
      scene.addLight({ type: 'directional', direction: [0, -1, 0], intensity });
      // A 2 x 2 box, projecting straight down -- or straight up, from below.
      for (const texture of decal ? layers : []) {
        scene.addDecal({ texture, size: [2, 2, 1] })
          .setAxisAngle([1, 0, 0], fromBelow ? Math.PI / 2 : -Math.PI / 2);
      }
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set([0, 6, 0.01]);
      cam.target.set([0, 0, 0]);
      probe.renderFrame(scene, cam);
      // The first frame with decals starts their pipelines; draw once they are ready.
      await probe.renderer._variantSets.get(2)?.building;
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      const at = (x) => [0, 1, 2].map((c) => pixels[((height >> 1) * width + Math.floor(x * width)) * 4 + c]);
      // World x -0.44, +0.44 and +1.4: the red half, the clear half, outside the box.
      return { red: at(0.45), clear: at(0.55), outside: at(0.8) };
    };
    const bare = await look({ decal: false });
    const painted = await look();
    const dim = await look({ intensity: 1 });
    const below = await look({ fromBelow: true });
    // Two decals in one box: the one added last is on top, every frame,
    // whatever order the cluster pass listed them in.
    const greenLast = await look({ layers: [image, green] });
    const redLast = await look({ layers: [green, image] });
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();
    const show = (p) => `rgb(${p.join(',')})`;
    const report = `floor ${show(bare.red)}; painted ${show(painted.red)}, ${show(painted.clear)} where clear, `
      + `${show(painted.outside)} outside; ${show(dim.red)} under a dimmer light; ${show(below.red)} projected from below; `
      + `${show(greenLast.red)} with green added last, ${show(redLast.red)} with red`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const same = (a, b) => a.every((c, k) => Math.abs(c - b[k]) <= 2);
    const ok = painted.red[0] > painted.red[1] + 80 && same(painted.clear, bare.clear) && same(painted.outside, bare.outside)
      && dim.red[0] < painted.red[0] - 30 && same(below.red, bare.red)
      && greenLast.red[1] > greenLast.red[0] + 80 && redLast.red[0] > redLast.red[1] + 80;
    if (!ok) throw new Error(report);
    return report;
  });

  await step('particles follow their solved paths at any frame rate, and are drawn while they live', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
      post: { strength: 0 },
    });
    probe.gpu.device.pushErrorScope('validation');
    const cam = new Camera({ fovY: 1, near: 0.1 });
    cam.position.set([0, 0, 6]);
    cam.target.set([0, 0, 0]);

    /** Every live particle of one emitter, read back from the pool. */
    const read = async (node) => {
      const system = probe.renderer.particles;
      const ring = system.rings.get(node.entity);
      const bytes = ring.capacity * 32;
      const buffer = probe.gpu.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = probe.gpu.device.createCommandEncoder();
      encoder.copyBufferToBuffer(system.pool, ring.offset * 32, buffer, 0, bytes);
      probe.gpu.queue.submit([encoder.finish()]);
      await buffer.mapAsync(GPUMapMode.READ);
      const f = new Float32Array(buffer.getMappedRange().slice(0));
      buffer.unmap();
      buffer.destroy();
      const live = [];
      for (let i = 0; i < ring.capacity; i++) {
        const o = i * 8;
        if (f[o + 3] < f[o + 7]) live.push({ position: [f[o], f[o + 1], f[o + 2]], age: f[o + 3], velocity: [f[o + 4], f[o + 5], f[o + 6]] });
      }
      return live;
    };
    /** Burst 32 at time 0, then run `steps` frames of `dt`. */
    const flight = async (options, steps, dt) => {
      const scene = probe.createScene();
      const node = scene.addEmitter({ lifetime: 10, size: 0.1, ...options });
      scene.burst(node, 32);
      probe.renderFrame(scene, cam);
      for (let k = 0; k < steps; k++) {
        scene.advance(dt);
        probe.renderFrame(scene, cam);
      }
      return read(node);
    };
    const worst = (live, expect) => Math.max(...live.map((p) => Math.abs(p.position[1] - expect)));

    // Thrown up at 2 under gravity: y = 2t - 9.81 t^2 / 2 at t = 0.5.
    const thrown = await flight({ speed: 2, acceleration: [0, -9.81, 0] }, 5, 0.1);
    const fine = await flight({ speed: 2, acceleration: [0, -9.81, 0] }, 50, 0.01);
    const ballistic = 2 * 0.5 - 0.5 * 9.81 * 0.25;
    // Under drag 1 alone: y = 2 (1 - e^-t).
    const dragged = await flight({ speed: 2, drag: 1 }, 5, 0.1);
    const coasted = 2 * (1 - Math.exp(-0.5));
    // A cone of pi/4 about +Y, and births anywhere in a ball of radius 1.
    const coned = await flight({ speed: 1, spread: Math.PI / 4 }, 0, 0);
    const widest = Math.max(...coned.map((p) => Math.acos(Math.min(1, p.velocity[1] / Math.hypot(...p.velocity)))));
    const balled = await flight({ radius: 1 }, 0, 0);
    const farthest = Math.max(...balled.map((p) => Math.hypot(...p.position)));

    // A stream: rate 100 for half a second is 50 particles.
    const scene = probe.createScene();
    const stream = scene.addEmitter({ rate: 100, lifetime: 1, size: 0.3, speed: 0.5, color: [4, 4, 4, 1] });
    // The canvas first: its image lasts only until something else awaits.
    const lit = async () => {
      const pixels = await probe.gpu.readPixels();
      let n = 0;
      for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > 60) n++;
      return n;
    };
    for (let k = 0; k < 5; k++) { scene.advance(0.1); probe.renderFrame(scene, cam); }
    const drawn = await lit();
    const alive = (await read(stream)).length;
    scene.setEmitter(stream, { rate: 0 });
    for (let k = 0; k < 12; k++) { scene.advance(0.1); probe.renderFrame(scene, cam); }
    const after = await lit();
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();

    const report = `y off by ${worst(thrown, ballistic).toExponential(1)} thrown at 10 fps, `
      + `${worst(fine, ballistic).toExponential(1)} at 100 fps; ${worst(dragged, coasted).toExponential(1)} under drag; `
      + `widest ${(widest * 180 / Math.PI).toFixed(1)} deg in a 45 deg cone; farthest ${farthest.toFixed(3)} in a ball of 1; `
      + `${alive} alive at rate 100 for 0.5 s; ${drawn} px drawn, ${after} after their lifetime`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const ok = thrown.length === 32 && worst(thrown, ballistic) < 1e-4 && worst(fine, ballistic) < 1e-4
      && worst(dragged, coasted) < 1e-4 && widest <= Math.PI / 4 + 1e-3 && farthest <= 1 + 1e-4
      && alive === 50 && drawn > 50 && after === 0;
    if (!ok) throw new Error(report);
    return report;
  });

  await step('sprites face the camera, keep their pixel size, hide behind geometry, and blend in order', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
      post: { strength: 0 },
    });
    probe.gpu.device.pushErrorScope('validation');
    const white = await probe.loadTexture(twoToneImageURI('#ffffff', '#ffffff'));
    const halfClear = await probe.loadTexture(twoToneImageURI('rgba(255,255,255,0)', '#ffffff'));
    const wall = await probe.load(buildFeatureGLB({ baseColorFactor: [0.2, 0.2, 0.2, 1], materialExtensions: { KHR_materials_unlit: {} } }));
    const shot = async (build, { from = [0, 0, 0], to = [0, 0, -1] } = {}) => {
      const scene = probe.createScene();
      build(scene);
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set(from);
      cam.target.set(to);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      return {
        middle: [0, 1, 2].map((c) => pixels[((height >> 1) * width + (width >> 1)) * 4 + c]),
        at: (x) => [0, 1, 2].map((c) => pixels[((height >> 1) * width + Math.floor(x * width)) * 4 + c]),
        redRow: [...Array(width).keys()].filter((x) => pixels[((height >> 1) * width + x) * 4] > 100).length,
        lit: Array.from({ length: pixels.length / 4 }, (_, i) => pixels[i * 4] + pixels[i * 4 + 1] > 100).filter(Boolean).length,
      };
    };
    const red = [1, 0, 0, 1];
    const plain = await shot((s) => s.addSprite({ texture: white, color: red, position: [0, 0, -3] }));
    const pixels = await shot((s) => s.addSprite({ texture: white, color: red, position: [0, 0, -30], pixels: true, size: [40, 40] }));
    const hidden = await shot((s) => {
      s.addSprite({ texture: white, color: red, position: [0, 0, -5] });
      s.add(wall).setPosition(0, 0, -3);
    });
    const order = async (nearFirst) => (await shot((s) => {
      const near = () => s.addSprite({ texture: white, color: [0, 1, 0, 0.5], position: [0, 0, -2] });
      const far = () => s.addSprite({ texture: white, color: red, position: [0, 0, -4] });
      if (nearFirst) { near(); far(); } else { far(); near(); }
    })).middle;
    const [inOrder, reversed] = [await order(false), await order(true)];
    const additive = await shot((s) => {
      s.add(wall).setPosition(0, 0, -4);
      s.addSprite({ texture: white, color: red, position: [0, 0, -3], blend: 'additive' });
    });
    // Seen from above: facing the camera it shows its face, standing upright
    // it shows its edge.
    const above = { from: [0, 5, 0.001], to: [0, 0, 0] };
    const facing = await shot((s) => s.addSprite({ texture: white, color: red }), above);
    const upright = await shot((s) => s.addSprite({ texture: white, color: red, facing: 'upright' }), above);
    const cutout = await shot((s) => s.addSprite({ texture: halfClear, color: red, position: [0, 0, -2], blend: 'cutout' }));
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();

    const show = (p) => `rgb(${p.join(',')})`;
    const report = `${show(plain.middle)} plain; ${pixels.redRow} px wide at 40 px; ${show(hidden.middle)} behind a wall; `
      + `${show(inOrder)} and ${show(reversed)} for two alpha sprites either order; ${show(additive.middle)} added to grey; `
      + `${facing.lit} px facing vs ${upright.lit} upright from above; cutout ${show(cutout.at(0.45))} | ${show(cutout.at(0.55))}`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const ok = plain.middle[0] > 200 && plain.middle[1] < 20
      && Math.abs(pixels.redRow - 40) <= 1
      && Math.abs(hidden.middle[0] - hidden.middle[1]) <= 3
      && inOrder.every((c, k) => Math.abs(c - reversed[k]) <= 1) && inOrder[1] > 100 && inOrder[0] > 100
      && additive.middle[0] > additive.middle[1] + 60 && additive.middle[1] > 40
      && upright.lit * 4 < facing.lit
      && cutout.at(0.45)[0] < 20 && cutout.at(0.55)[0] > 200;
    if (!ok) throw new Error(report);
    return report;
  });

  await step('2D lands colours exactly, blends in sRGB, stacks by layer, keeps pixel art sharp, and moves one slot', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    probe.gpu.device.pushErrorScope('validation');
    const white = await probe.loadTexture(twoToneImageURI('#ffffff', '#ffffff'));
    const sharp = await probe.loadTexture(twoToneImageURI('#ff0000', '#0000ff'), { pixelated: true });
    const smooth = await probe.loadTexture(twoToneImageURI('#ff0000', '#0000ff'));
    const { width: W, height: H, pixelRatio: r } = probe.gpu;
    const scene = probe.createScene();
    // A unit is a CSS pixel: places and sizes are fractions of the view, in them.
    const box = (texture, x, y, w, h, options = {}) => scene.addSprite({
      texture, position: [x * W / r, y * H / r], size: [w * W / r, h * H / r], pivot: [0, 0], ...options,
    });
    const camera = new Camera2D({ background: [1, 1, 1, 1] });
    // rgb(128, 128, 128), as CSS writes it.
    const grey = box(white, 0.05, 0.1, 0.1, 0.2, { color: [128 / 255, 128 / 255, 128 / 255, 1] });
    box(white, 0.25, 0.1, 0.1, 0.2, { color: [0, 0, 0, 0.5] });
    // Red is added first but on a higher layer: it must still be on top.
    box(white, 0.45, 0.1, 0.2, 0.2, { color: [1, 0, 0, 1], layer: 1 });
    box(white, 0.55, 0.1, 0.2, 0.2, { color: [0, 1, 0, 1] });
    box(sharp, 0.05, 0.5, 0.4, 0.2);
    box(smooth, 0.55, 0.5, 0.4, 0.2);

    const shoot = async () => {
      probe.renderFrame(scene, camera);
      const pixels = await probe.gpu.readPixels();
      return (x, y) => {
        const i = (Math.floor(y * H) * W + Math.floor(x * W)) * 4;
        return [pixels[i], pixels[i + 1], pixels[i + 2]];
      };
    };
    const first = await shoot();
    const written = probe.stats.sprites2DWritten;
    grey.setPosition(0.05 * W / r, 0.75 * H / r);
    const second = await shoot();
    const movedWritten = probe.stats.sprites2DWritten;
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();

    const show = (p) => `rgb(${p.join(',')})`;
    const at = {
      grey: first(0.1, 0.2), half: first(0.3, 0.2), overlap: first(0.6, 0.2), green: first(0.7, 0.2),
      background: first(0.5, 0.95), sharp: first(0.05 + 0.4 * 0.45, 0.6), smooth: first(0.55 + 0.4 * 0.45, 0.6),
      left: second(0.1, 0.2), arrived: second(0.1, 0.85),
    };
    const report = `grey ${show(at.grey)}, half black over white ${show(at.half)}, overlap ${show(at.overlap)}, `
      + `green ${show(at.green)}, background ${show(at.background)}; just left of the middle of a 2-texel sprite: `
      + `pixelated ${show(at.sharp)}, smooth ${show(at.smooth)}; moved: ${show(at.left)} where it was, `
      + `${show(at.arrived)} where it went, ${written} then ${movedWritten} slots written`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const is = (p, rgb, tolerance = 0) => p.every((c, k) => Math.abs(c - rgb[k]) <= tolerance);
    // Blended in linear light, half black over white would be 188.
    const ok = is(at.grey, [128, 128, 128]) && is(at.half, [127, 127, 127], 1)
      && is(at.overlap, [255, 0, 0]) && is(at.green, [0, 255, 0]) && is(at.background, [255, 255, 255])
      && is(at.sharp, [255, 0, 0]) && at.smooth[2] > 40
      && is(at.left, [255, 255, 255]) && is(at.arrived, [128, 128, 128])
      && written === 6 && movedWritten === 1;
    if (!ok) throw new Error(report);
    return report;
  });

  await step('2D multiplies and screens, repeats, smooths turned edges, outlines text, joins long lines, and draws into targets', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    probe.gpu.device.pushErrorScope('validation');
    const { width: W, pixelRatio: r } = probe.gpu;
    const white = await probe.loadTexture(twoToneImageURI('#ffffff', '#ffffff'));
    const stripes = await probe.loadTexture(twoToneImageURI('#ff0000', '#0000ff'), { pixelated: true });
    const font = await probe.loadFont('64px sans-serif');
    const scene = probe.createScene();
    // A unit is a canvas pixel here.
    const camera = new Camera2D({ zoom: 1 / r, background: [1, 1, 1, 1] });
    const box = (x, y, w, h, options) => scene.addSprite({ texture: white, position: [x, y], size: [w, h], pivot: [0, 0], ...options });
    // Multiply: grey over red halves it. Screen: red over dark blue adds them, less their product.
    box(10, 10, 30, 20, { color: [1, 0, 0, 1] });
    box(10, 10, 30, 20, { color: [0.5, 0.5, 0.5, 1], blend: 'multiply' });
    box(50, 10, 30, 20, { color: [0, 0, 0.5, 1] });
    box(50, 10, 30, 20, { color: [0.5, 0, 0, 1], blend: 'screen' });
    // A red-and-blue image four times across: each texel 5 pixels.
    scene.addSprite({ texture: stripes, rect: [0, 0, 4, 1], size: [40, 10], pivot: [0, 0], position: [90, 10] });
    // A turned white square on black: its edge smoothed, as a shape's is.
    box(140, 5, 50, 50, { color: [0, 0, 0, 1] });
    box(165, 30, 30, 30, { pivot: [0.5, 0.5], angle: Math.PI / 5 });
    // White text outlined in red, on black.
    box(200, 5, 60, 70, { color: [0, 0, 0, 1] });
    scene.addText({ font, text: 'I', size: 64, color: [1, 1, 1, 1], stroke: [1, 0, 0, 1], strokeWidth: 3, position: [230, 40] });
    // A see-through line of 60 points, so four pieces: its joins no darker than its middles.
    const points = Array.from({ length: 60 }, (_, k) => [10 + k * 4, 100]);
    scene.addPath({ points, closed: false, strokeWidth: 8, stroke: [0, 0, 1, 0.5] });
    // A 2D scene drawn into a target, and a 3D one into another, both shown by sprites.
    const flat = await probe.createTarget({ size: [32, 32] });
    const inner = probe.createScene();
    inner.addShape({ size: [16, 16], color: [1, 0, 0, 1], pivot: [0, 0], position: [16, 16] });
    probe.renderFrame(inner, new Camera2D({ background: [0, 1, 0, 1] }), { target: flat });
    scene.addSprite({ texture: flat, size: [32, 32], pivot: [0, 0], position: [10, 130] });
    const deep = await probe.createTarget({ size: [64, 64] });
    const eye = new Camera({ fovY: 1, near: 0.1 });
    eye.target.set([0, 0.2, -1]);
    probe.renderFrame(probe.createScene(), eye, { target: deep });
    scene.addSprite({ texture: deep, size: [64, 64], pivot: [0, 0], position: [60, 130] });
    probe.renderFrame(scene, camera);
    const pixels = await probe.gpu.readPixels();
    const error = await probe.gpu.device.popErrorScope();
    let unloaded = '';
    probe.unload(flat);
    try { probe.renderFrame(inner, new Camera2D(), { target: flat }); } catch (e) { unloaded = e.message; }
    probe.destroy();
    canvas.remove();

    const at = (x, y) => [...pixels.subarray((y * W + x) * 4, (y * W + x) * 4 + 3)];
    const is = (p, rgb, tolerance = 2) => p.every((c, k) => Math.abs(c - rgb[k]) <= tolerance);
    let smooth = 0;
    for (let y = 5; y < 55; y++) for (let x = 140; x < 190; x++) { const v = at(x, y)[0]; if (v > 30 && v < 225) smooth++; }
    let outline = 0;
    for (let y = 5; y < 75; y++) for (let x = 200; x < 260; x++) { const [red, green] = at(x, y); if (red > 200 && green < 60) outline++; }
    const [join, middle, end] = [at(10 + 16 * 4, 100), at(10 + 8 * 4, 100), at(10 + 58 * 4, 100)];
    const inTarget = [at(12, 132), at(10 + 24, 130 + 24)];
    const sky = at(60 + 32, 130 + 20);
    const report = `multiply rgb(${at(20, 20)}), screen rgb(${at(60, 20)}); repeat ${[92, 97, 102, 127].map((x) => at(x, 15)[0] > 200 ? 'r' : 'b').join('')}; `
      + `${smooth} smoothed edge pixels, ${outline} outline pixels; line join rgb(${join}), middle rgb(${middle}), far end rgb(${end}); `
      + `2D target rgb(${inTarget[0]}) | rgb(${inTarget[1]}), 3D target rgb(${sky}); unloaded: ${unloaded}`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const ok = is(at(20, 20), [128, 0, 0]) && is(at(60, 20), [128, 0, 128])
      && report.includes('repeat rbrb') && smooth > 20 && outline > 20
      && is(join, middle, 1) && middle[2] > 200 && middle[0] > 100 && middle[0] < 160 && is(end, middle, 1)
      && is(inTarget[0], [0, 255, 0]) && is(inTarget[1], [255, 0, 0]) && sky.some((c) => c > 30)
      && /unloaded/.test(unloaded);
    if (!ok) throw new Error(report);
    return report;
  });

  await step('pixel art lands a texel to a pixel wherever it is, and a sheet frame never shows its neighbour', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    const { width: W, pixelRatio: r } = probe.gpu;
    // Five texels, five colours.
    const colours = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]];
    const image = new OffscreenCanvas(5, 1);
    const context = image.getContext('2d');
    colours.forEach((c, k) => { context.fillStyle = `rgb(${c.join(',')})`; context.fillRect(k, 0, 1, 1); });
    const strip = await probe.loadTexture(await image.convertToBlob(), { pixelated: true });
    const sheet = await probe.loadTexture(twoToneImageURI('#ff0000', '#0000ff'));
    const scene = probe.createScene();
    // A texel to a canvas pixel, so a unit is one here; ten strips, each a
    // tenth of a pixel further along, centred on their node.
    const camera = new Camera2D({ zoom: 1 / r, pixelSnap: true, background: [0, 0, 0, 1] });
    for (let k = 0; k < 10; k++) scene.addSprite({ texture: strip, position: [20 + k * 10.1, 10 + k * 4] });
    // The red frame of a red-and-blue sheet, smooth and enlarged: its right
    // edge is where filtering would reach the blue frame.
    scene.addSprite({ texture: sheet, rect: [0, 0, 0.5, 1], size: [40, 40], pivot: [0, 0], position: [200, 10] });
    probe.renderFrame(scene, camera);
    const pixels = await probe.gpu.readPixels();
    probe.destroy();
    canvas.remove();
    const at = (x, y) => [...pixels.subarray((y * W + x) * 4, (y * W + x) * 4 + 3)];
    const rows = [];
    for (let k = 0; k < 10; k++) {
      const y = 10 + k * 4;
      rows.push(Array.from({ length: 150 }, (_, x) => at(x, y)).filter((p) => p.some((c) => c > 0)).map((p) => p.join(',')));
    }
    const wanted = colours.map((c) => c.join(',')).join(' ');
    const wrong = rows.map((row, k) => [k, row.join(' ')]).filter(([, row]) => row !== wanted);
    const edge = at(238, 30);
    const report = `${10 - wrong.length} of 10 strips exactly one pixel a texel; a sheet frame's far edge rgb(${edge.join(',')})`;
    if (wrong.length > 0) throw new Error(`${report}; strip ${wrong[0][0]} drew ${wrong[0][1]}`);
    if (edge[2] > 2) throw new Error(report);
    return report;
  });

  await step('a tilemap draws its tiles, flipped as Tiled flips them, shows through where empty, and uploads one changed tile', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    probe.gpu.device.pushErrorScope('validation');
    // One tile, two texels: red on its left, blue on its right.
    const tileset = await probe.loadTexture(twoToneImageURI('#ff0000', '#0000ff'), { pixelated: true });
    const white = await probe.loadTexture(twoToneImageURI('#ffffff', '#ffffff'));
    const scene = probe.createScene();
    const camera = new Camera2D({ background: [1, 1, 1, 1] });
    const FLIP_X = 0x80000000;
    // Three tiles, each 40 x 20 pixels from (20, 20): plain, mirrored, empty.
    const map = scene.addTilemap({ tileset, tileSize: [2, 1], columns: 3, rows: 1, tiles: [1, (FLIP_X | 1) >>> 0, 0], position: [20, 20] });
    map.setScale(20, 20, 1);
    // Green on a lower layer, under the whole map.
    scene.addSprite({ texture: white, color: [0, 1, 0, 1], position: [20, 20], size: [120, 20], pivot: [0, 0], layer: -1 });

    const shoot = async () => {
      probe.renderFrame(scene, camera);
      const pixels = await probe.gpu.readPixels();
      // x in CSS pixels, read at the canvas pixel it lands on.
      const { width: W, pixelRatio: r } = probe.gpu;
      return (x) => [0, 1, 2].map((c) => pixels[(Math.floor(30 * r) * W + Math.floor(x * r)) * 4 + c]);
    };
    const first = await shoot();
    const uploaded = probe.stats.tiles2DWritten;
    scene.setTile(map, 2, 0, 1);
    const second = await shoot();
    const changed = probe.stats.tiles2DWritten;
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();

    const show = (p) => `rgb(${p.join(',')})`;
    const at = { plain: [first(30), first(50)], flipped: [first(70), first(90)], empty: first(120), set: second(110) };
    const report = `plain ${show(at.plain[0])} | ${show(at.plain[1])}, mirrored ${show(at.flipped[0])} | ${show(at.flipped[1])}, `
      + `empty ${show(at.empty)}; set later ${show(at.set)}; ${uploaded} then ${changed} tiles uploaded`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const is = (p, rgb) => p.every((c, k) => c === rgb[k]);
    const red = [255, 0, 0], blue = [0, 0, 255];
    const ok = is(at.plain[0], red) && is(at.plain[1], blue) && is(at.flipped[0], blue) && is(at.flipped[1], red)
      && is(at.empty, [0, 255, 0]) && is(at.set, red) && uploaded === 3 && changed === 1;
    if (!ok) throw new Error(report);
    return report;
  });

  await step('a HUD draws over the finished 3D frame: exact colours, round shapes, a one-pixel edge', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, { environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } } });
    probe.gpu.device.pushErrorScope('validation');
    const wall = await probe.load(buildFeatureGLB({ baseColorFactor: [0.5, 0.5, 0.5, 1], materialExtensions: { KHR_materials_unlit: {} } }));
    const scene = probe.createScene();
    scene.add(wall).setPosition(0, 0, -1).setScale(4, 4, 1);
    const camera = new Camera({ fovY: 1, near: 0.1 });
    camera.position.set([0, 0, 0]);
    camera.target.set([0, 0, -1]);
    const { width: W, height: H, pixelRatio: r } = probe.gpu;
    const cx = W >> 1, cy = H >> 1;
    // Laid out in canvas pixels, to read single ones back: a unit is a CSS pixel.
    const px = (v) => v / r;

    const hud = probe.createScene();
    const grey = 128 / 255;
    // Top-left, a bar with round ends, off the wall's middle.
    hud.addShape({ size: [px(100), px(40)], radius: px(20), color: [grey, grey, grey, 1], pivot: [0, 0], position: [px(20), px(20)] });
    // Centred on a pixel's centre, so its outer edge crosses one pixel's centre.
    hud.addShape({
      shape: 'ellipse', size: [px(60), px(60)], color: [0, 0, 1, 1], stroke: [1, 0, 0, 1], strokeWidth: px(6),
      position: [px(cx + 0.5), px(cy + 0.5)],
    });

    probe.renderFrame(scene, camera, { hud: { scene: hud } });
    const pixels = await probe.gpu.readPixels();
    const at = (x, y) => [0, 1, 2].map((c) => pixels[(y * W + x) * 4 + c]);
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();

    const show = (p) => `rgb(${p.join(',')})`;
    const wallColour = at(cx + 50, cy);
    const edge = [];
    for (let x = cx + 26; x <= cx + 36; x++) edge.push(at(x, cy));
    const is = (p, rgb) => p.every((c, k) => c === rgb[k]);
    const partial = edge.filter((p) => !is(p, [255, 0, 0]) && !is(p, wallColour)).length;
    const read = {
      // A capsule: straight along its top, round at its ends.
      bar: at(70, 40), corner: at(20, 20), barEdge: at(70, 20), centre: at(cx, cy), ring: at(cx + 27, cy),
    };
    const report = `3D wall ${show(wallColour)}; HUD bar ${show(read.bar)}, its rounded corner ${show(read.corner)}, `
      + `its top edge ${show(read.barEdge)}; circle ${show(read.centre)}, outline ${show(read.ring)}, `
      + `${partial} partial pixel across its outer edge`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const ok = wallColour[0] > 40 && is(read.bar, [128, 128, 128]) && is(read.corner, wallColour)
      && is(read.barEdge, [128, 128, 128]) && is(read.centre, [0, 0, 255]) && is(read.ring, [255, 0, 0]) && partial === 1;
    if (!ok) throw new Error(report);
    return report;
  });

  await step('picking a 2D view names what each pixel shows, and text wraps to its width', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    probe.gpu.device.pushErrorScope('validation');
    const white = await probe.loadTexture(twoToneImageURI('#ffffff', '#ffffff'));
    const font = await probe.loadFont('32px sans-serif');
    const scene = probe.createScene();
    const camera = new Camera2D({ background: [1, 1, 1, 1] });
    const ball = scene.addShape({ shape: 'ellipse', size: [100, 60], color: [1, 0, 0, 1], position: [80, 70] });
    ball.setAngle(0.5);
    const bar = scene.addSprite({ texture: white, color: [0, 1, 0, 1], size: [90, 30], position: [230, 70], layer: 1 });
    bar.setAngle(-0.6);
    // Three words, each about 40 pixels wide at 20, in a 70-pixel box: three lines.
    scene.addText({ font, text: 'wide wide wide', size: 20, width: 70, pivot: [0, 0], color: [0, 0, 1, 1], position: [20, 140] });

    probe.renderFrame(scene, camera);
    const pixels = await probe.gpu.readPixels();
    const { width: W, pixelRatio: r } = probe.gpu;
    const colour = (x, y) => [0, 1, 2].map((c) => pixels[(Math.floor(y * r) * W + Math.floor(x * r)) * 4 + c]);
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();

    // Every pure pixel on a grid, against what a pick there names. Edges
    // (blended colours) and text ink are skipped.
    const is = (p, rgb) => p.every((c, k) => c === rgb[k]);
    let checked = 0;
    const wrong = [];
    for (let y = 1; y < 240; y += 6) {
      for (let x = 1; x < 320; x += 6) {
        const p = colour(x, y);
        const expected = is(p, [255, 0, 0]) ? ball.entity : is(p, [0, 255, 0]) ? bar.entity : is(p, [255, 255, 255]) ? null : undefined;
        if (expected === undefined) continue;
        // At the pixel's centre, where the GPU decided its colour.
        const hit = scene.pick(camera, (Math.floor(x * r) + 0.5) / r, (Math.floor(y * r) + 0.5) / r, 320, 240);
        // White inside the text's block is the text's, by design.
        if (expected === null && hit !== null && scene.texts.has(hit.node.entity)) continue;
        checked++;
        if ((hit?.node.entity ?? null) !== expected) wrong.push(`(${x}, ${y}) shows rgb(${p}) and picks ${hit === null ? 'nothing' : hit.node.entity}`);
      }
    }
    // The text's ink: its height says how many lines it wrapped to.
    let top = Infinity, bottom = -Infinity, left = Infinity, right = -Infinity;
    for (let y = 130; y < 240; y++) {
      for (let x = 0; x < 200; x++) {
        const [red, , blue] = colour(x, y);
        if (blue - red > 100) { top = Math.min(top, y); bottom = Math.max(bottom, y); left = Math.min(left, x); right = Math.max(right, x); }
      }
    }
    const report = `${checked} pixels checked against a pick, ${wrong.length} wrong${wrong.length ? `: ${wrong.slice(0, 3).join('; ')}` : ''}; `
      + `wrapped text inked ${right - left + 1} x ${bottom - top + 1} px in a 70 px box`;
    if (error) throw new Error(`${report}; ${error.message}`);
    // Three lines at 20 px a line are more than 40 px tall; one would be under 25.
    if (!(wrong.length === 0 && checked > 1000 && bottom - top > 40 && right - left < 72)) throw new Error(report);
    return report;
  });

  await step('particles draw in a 2D view, in its colours, between the layers around them', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    probe.gpu.device.pushErrorScope('validation');
    const white = await probe.loadTexture(twoToneImageURI('#ffffff', '#ffffff'));
    const scene = probe.createScene();
    const camera = new Camera2D();
    // A still cloud of 40-pixel dots at (100, 100), on layer 1: under a green
    // card on layer 2 over its right half, over a blue one on layer 0.
    const cloud = scene.addEmitter({ lifetime: 60, size: 40, speed: 0, color: [1, 0, 0, 1], blend: 'alpha', layer: 1, position: [100, 100] });
    scene.addSprite({ texture: white, color: [0, 1, 0, 1], size: [60, 60], pivot: [0, 0.5], position: [100, 100], layer: 2 });
    scene.addSprite({ texture: white, color: [0, 0, 1, 1], size: [100, 100], position: [100, 100] });
    scene.burst(cloud, 40);
    scene.advance(0.016);
    probe.renderFrame(scene, camera);
    const pixels = await probe.gpu.readPixels();
    const { width: W, pixelRatio: r } = probe.gpu;
    const at = (x, y) => [0, 1, 2].map((c) => pixels[(Math.floor(y * r) * W + Math.floor(x * r)) * 4 + c]);
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();
    const show = (p) => `rgb(${p.join(',')})`;
    const read = { centre: at(97, 100), under: at(103, 100), rim: at(100 - 30, 100) };
    const report = `the cloud's middle ${show(read.centre)}, under the card ${show(read.under)}, past its rim ${show(read.rim)}`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const is = (p, rgb) => p.every((c, k) => c === rgb[k]);
    // Forty dots, each opaque at its middle: red. The card is on top; past the dots, the blue card below.
    if (!(is(read.centre, [255, 0, 0]) && is(read.under, [0, 255, 0]) && is(read.rim, [0, 0, 255]))) throw new Error(report);
    return report;
  });

  await step('debug lines draw over a 2D view, in its sRGB colours, and last one frame', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    probe.gpu.device.pushErrorScope('validation');
    const scene = probe.createScene();
    const camera = new Camera2D();
    const shoot = async () => {
      probe.renderFrame(scene, camera);
      const pixels = await probe.gpu.readPixels();
      const { width: W, pixelRatio: r } = probe.gpu;
      return (x, y) => [0, 1, 2].map((c) => pixels[(Math.floor(y * r) * W + Math.floor(x * r)) * 4 + c]);
    };
    // On pixel centres, so each lands on one row or column.
    probe.debug.box([40.5, 150.5], [120.5, 190.5], [1, 0, 1]);
    probe.debug.circle([200.5, 170.5], 20, [0.5, 0.5, 0.5]);
    const first = await shoot();
    const second = await shoot();
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();
    const show = (p) => `rgb(${p.join(',')})`;
    const read = { edge: first(80, 150), side: first(40, 170), inside: first(80, 170), circle: first(220, 170), gone: second(80, 150) };
    const report = `box edge ${show(read.edge)}, side ${show(read.side)}, inside ${show(read.inside)}; grey circle ${show(read.circle)}; `
      + `the next frame ${show(read.gone)}`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const is = (p, rgb) => p.every((c, k) => c === rgb[k]);
    // Grey 0.5 is 128 as sRGB; as linear light it would be 188.
    if (!(is(read.edge, [255, 0, 255]) && is(read.side, [255, 0, 255]) && is(read.inside, [0, 0, 0])
      && is(read.circle, [128, 128, 128]) && is(read.gone, [0, 0, 0]))) throw new Error(report);
    return report;
  });

  await step('a smooth sprite with clear pixels leaves no dark fringe, enlarged or shrunk', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    probe.gpu.device.pushErrorScope('validation');
    // White on the left, clear on the right. Clear pixels decode with black in
    // them, which filtering must not let through.
    const small = await probe.loadTexture(twoToneImageURI('#ffffff', 'rgba(255,255,255,0)'));
    const sheet = document.createElement('canvas');
    sheet.width = sheet.height = 64;
    const g = sheet.getContext('2d');
    g.fillStyle = '#ffffff';
    // 28, not 32: the edge off every mip's texel boundary, so levels average across it.
    g.fillRect(0, 0, 28, 64);
    const large = await probe.loadTexture(sheet.toDataURL('image/png'));
    const scene = probe.createScene();
    const camera = new Camera2D({ background: [1, 1, 1, 1] });
    scene.addSprite({ texture: small, size: [200, 40], pivot: [0, 0], position: [20, 20] });   // enlarged 100 times
    scene.addSprite({ texture: large, size: [8, 8], pivot: [0, 0], position: [20, 100] });     // shrunk 8 times: mip 3
    probe.renderFrame(scene, camera);
    const pixels = await probe.gpu.readPixels();
    const { width: W, pixelRatio: r } = probe.gpu;
    const darkest = (y, x0, x1) => {
      let low = 255;
      for (let x = Math.floor(x0 * r); x < Math.ceil(x1 * r); x++) low = Math.min(low, pixels[(Math.floor(y * r) * W + x) * 4]);
      return low;
    };
    const enlarged = darkest(40, 20, 220), shrunk = darkest(104, 20, 28);
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();
    const report = `darkest pixel over white: ${enlarged} enlarged, ${shrunk} shrunk (255 is none)`;
    if (error) throw new Error(`${report}; ${error.message}`);
    if (!(enlarged >= 253 && shrunk >= 253)) throw new Error(report);
    return report;
  });

  await step("a tileset's margin and gaps are stepped over, and a diagonal flip swaps a tile's axes as Tiled's does", async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    probe.gpu.device.pushErrorScope('validation');
    // Two 2x2 tiles in a 1-texel margin with a 1-texel gap, all of which is
    // magenta: tile 1 is red, green / blue, white; tile 2 yellow.
    const sheet = document.createElement('canvas');
    sheet.width = 7;
    sheet.height = 4;
    const g = sheet.getContext('2d');
    g.fillStyle = '#ff00ff';
    g.fillRect(0, 0, 7, 4);
    for (const [x, y, c] of [[1, 1, '#ff0000'], [2, 1, '#00ff00'], [1, 2, '#0000ff'], [2, 2, '#ffffff'], [4, 1, '#ffff00'], [5, 1, '#ffff00'], [4, 2, '#ffff00'], [5, 2, '#ffff00']]) {
      g.fillStyle = c;
      g.fillRect(x, y, 1, 1);
    }
    const tileset = await probe.loadTexture(sheet.toDataURL('image/png'), { pixelated: true });
    const scene = probe.createScene();
    const camera = new Camera2D();
    const DIAGONAL = 0x20000000;
    const map = scene.addTilemap({ tileset, tileSize: [2, 2], margin: 1, spacing: 1, columns: 3, rows: 1, tiles: [1, DIAGONAL | 1, 2], position: [20, 100] });
    map.setScale(20, 20, 1);
    probe.renderFrame(scene, camera);
    const pixels = await probe.gpu.readPixels();
    const { width: W, pixelRatio: r } = probe.gpu;
    const at = (x, y) => [0, 1, 2].map((c) => pixels[(Math.floor(y * r) * W + Math.floor(x * r)) * 4 + c]);
    let magenta = 0;
    for (let y = 100; y < 140; y++) for (let x = 20; x < 140; x++) { const [red, green, blue] = at(x, y); if (red > 200 && blue > 200 && green < 50) magenta++; }
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();

    // Each tile is 40 pixels; its quarters' centres are 10 in from its corners.
    const quarters = (x) => [at(x + 10, 110), at(x + 30, 110), at(x + 10, 130), at(x + 30, 130)];
    const plain = quarters(20), flipped = quarters(60), second = at(120, 120);
    const show = (ps) => ps.map((p) => `rgb(${p.join(',')})`).join(' ');
    const report = `plain ${show(plain)}; diagonal ${show(flipped)}; second tile ${show([second])}; ${magenta} magenta pixels`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const R = [255, 0, 0], G = [0, 255, 0], B = [0, 0, 255], Wh = [255, 255, 255];
    const same = (ps, expected) => ps.every((p, i) => p.every((c, k) => c === expected[i][k]));
    if (!(same(plain, [R, G, B, Wh]) && same(flipped, [R, B, G, Wh]) && same([second], [[255, 255, 0]]) && magenta === 0)) throw new Error(report);
    return report;
  });

  await step('2D paths fill concave outlines and stroke lines exactly; 2D lights light only what is lit', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const probe = await Winding.create(canvas);
    probe.gpu.device.pushErrorScope('validation');
    const white = await probe.loadTexture(twoToneImageURI('#ffffff', '#ffffff'));
    const scene = probe.createScene();
    const camera = new Camera2D();
    // The left half: a lit white floor under one light; a strip of it unlit.
    scene.addSprite({ texture: white, size: [160, 200], pivot: [0, 0], lit: true });
    scene.addSprite({ texture: white, size: [160, 20], pivot: [0, 0], position: [0, 210] });
    // On a pixel's centre, so the pixels read below are exact distances from it.
    scene.addLight({ position: [80.5, 100.5], radius: 60 });
    // Bottom right: a lit strip under a spot aimed right, in a cone 0.3 wide.
    scene.addSprite({ texture: white, size: [140, 56], pivot: [0, 0], position: [170, 180], lit: true });
    scene.addLight({ position: [175.5, 207.5], direction: [1, 0], radius: 200, innerAngle: 0.2, outerAngle: 0.3 });
    // The right half: an L filled red, its notch empty; a blue line 6 wide.
    scene.addPath({ points: [[0, 0], [10, 0], [10, 30], [30, 30], [30, 40], [0, 40]], color: [1, 0, 0, 1], position: [200, 20] });
    scene.addPath({ points: [[0, 0], [100, 0]], closed: false, color: [0, 0, 0, 0], stroke: [0, 0, 1, 1], strokeWidth: 6, position: [200, 150] });

    const shoot = async () => {
      probe.renderFrame(scene, camera);
      const pixels = await probe.gpu.readPixels();
      const { width: W, pixelRatio: r } = probe.gpu;
      return (x, y) => [0, 1, 2].map((c) => pixels[(Math.floor(y * r) * W + Math.floor(x * r)) * 4 + c]);
    };
    const dark = await shoot();
    camera.ambient.set([0.2, 0.2, 0.2]);
    const dim = await shoot();
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();

    const show = (p) => `rgb(${p.join(',')})`;
    const read = {
      centre: dark(80.5, 100.5), half: dark(110.5, 100.5), beyond: dark(150.5, 100.5), unlit: dark(80.5, 220.5),
      ambient: dim(150.5, 100.5), ell: dark(205.5, 25.5), foot: dark(225.5, 55.5), notch: dark(225.5, 35.5),
      line: dark(250.5, 152.5), past: dark(250.5, 153.5),
      aimed: dark(240.5, 207.5), aside: dark(240.5, 234.5), behind: dark(171.5, 207.5),
    };
    const report = `light: ${show(read.centre)} at the light, ${show(read.half)} half its radius out, ${show(read.beyond)} past it, `
      + `${show(read.ambient)} past it under a 0.2 ambient, ${show(read.unlit)} unlit; path: ${show(read.ell)} and ${show(read.foot)} `
      + `inside the L, ${show(read.notch)} in its notch; line ${show(read.line)} inside, ${show(read.past)} past its edge; `
      + `spot ${show(read.aimed)} along its aim, ${show(read.aside)} outside its cone, ${show(read.behind)} behind it`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const is = (p, rgb, tolerance = 0) => p.every((c, k) => Math.abs(c - rgb[k]) <= tolerance);
    // Half the radius out: (1 - 0.5^2)^2 = 0.5625 in linear light, 198 in sRGB. Ambient 0.2 is 124.
    const ok = is(read.centre, [255, 255, 255]) && is(read.half, [198, 198, 198], 1) && is(read.beyond, [0, 0, 0])
      && is(read.ambient, [124, 124, 124], 1) && is(read.unlit, [255, 255, 255])
      && is(read.ell, [255, 0, 0]) && is(read.foot, [255, 0, 0]) && is(read.notch, [0, 0, 0])
      && is(read.line, [0, 0, 255]) && is(read.past, [0, 0, 0])
      // 65 of its 200 out along its aim: (1 - 65^2 / 200^2)^2 = 0.80 linear, 231 sRGB.
      && is(read.aimed, [231, 231, 231], 1) && is(read.aside, [0, 0, 0]) && is(read.behind, [0, 0, 0]);
    if (!ok) throw new Error(report);
    return report;
  });

  await step('a reflection probe captures the room the right way round, and a mirror reflects it', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
    });
    probe.gpu.device.pushErrorScope('validation');
    const unlit = (color) => probe.load(buildFeatureGLB({
      baseColorFactor: color, materialExtensions: { KHR_materials_unlit: {} },
    }));
    const [red, green, blue] = await Promise.all([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]].map(unlit));
    // A red wall two units along +X, facing back at the origin; green on it
    // above the middle, blue on it toward +Z.
    const room = () => {
      const scene = probe.createScene();
      const facing = (node) => node.setAxisAngle([0, 1, 0], -Math.PI / 2);
      facing(scene.add(red)).setPosition(2, 0, 0);
      facing(scene.add(green)).setPosition(1.9, 0.8, 0).setScale(0.15, 0.15, 1);
      facing(scene.add(blue)).setPosition(1.9, 0, 0.8).setScale(0.15, 0.15, 1);
      return scene;
    };

    // 1. The capture itself, read back from the probe array: face 0 is +X.
    const scene = room();
    scene.addProbe({ size: [6, 6, 6] });
    await probe.captureProbes(scene);
    const set = probe.renderer._probeSets.get(scene);
    const size = set.size;
    const readback = probe.gpu.device.createBuffer({ size: size * size * 8 * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = probe.gpu.device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture: set.texture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
      { buffer: readback, bytesPerRow: size * 8, rowsPerImage: size }, [size, size, 2],
    );
    probe.gpu.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    const half = (h) => {
      const e = (h >> 10) & 31;
      const m = h & 1023;
      return (e === 0 ? m / 1024 * 2 ** -14 : (1 + m / 1024) * 2 ** (e - 15)) * (h & 0x8000 ? -1 : 1);
    };
    const texel = (face, u, v) => {
      const i = ((face * size + Math.floor(v * size)) * size + Math.floor(u * size)) * 4;
      return [0, 1, 2].map((c) => +half(halves[i + c]).toFixed(2));
    };
    // cubeDirection(+X, u, v) = (1, -v', -u') with u', v' in [-1, 1]: up is
    // small v, and +Z is small u.
    const middle = texel(0, 0.5, 0.5);
    const above = texel(0, 0.5, 0.5 - 0.4 * 0.5);
    const towardZ = texel(0, 0.5 - 0.4 * 0.5, 0.5);
    const behind = texel(1, 0.5, 0.5);
    const face = `+X face: ${middle} middle, ${above} above, ${towardZ} toward +Z; -X face ${behind}`;
    const is = (p, c) => p[c] > 0.5 && p.every((x, k) => k === c || x < 0.2);
    if (!(is(middle, 0) && is(above, 1) && is(towardZ, 2) && behind.every((x) => x < 0.05))) {
      throw new Error(`the capture is not the right way round: ${face}`);
    }

    // 2. A mirror floor under the room reflects the wall once the probe is
    // captured -- and the sky, black, where no probe's box reaches.
    const floorMirror = await probe.load(buildFeatureGLB({ baseColorFactor: [1, 1, 1, 1], metallicFactor: 1, roughnessFactor: 0 }));
    const reflection = async (probeBox) => {
      const s = room();
      s.add(floorMirror).setAxisAngle([1, 0, 0], -Math.PI / 2).setPosition(0, -1, 0).setScale(1, 1, 1);
      if (probeBox) await probe.captureProbes(s, [s.addProbe(probeBox)]);
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set([-1.5, 0.2, 0]);
      cam.target.set([1, -1, 0]);
      probe.renderFrame(s, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      // Where the floor is, below the middle of the view.
      const i = (Math.floor(height * 0.8) * width + (width >> 1)) * 4;
      return [pixels[i], pixels[i + 1], pixels[i + 2]];
    };
    const sky = await reflection(null);
    const captured = await reflection({ size: [6, 6, 6] });
    const elsewhere = await reflection({ position: [13, 0, 0], size: [6, 6, 6] });
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();
    const show = (p) => `rgb(${p.join(',')})`;
    const report = `${face}; the mirror floor: ${show(sky)} with no probe, ${show(captured)} inside a captured one, `
      + `${show(elsewhere)} when its box is elsewhere`;
    if (error) throw new Error(`${report}; ${error.message}`);
    if (!(captured[0] > sky[0] + 60 && captured[0] > captured[1] + 60 && Math.abs(elsewhere[0] - sky[0]) <= 3)) {
      throw new Error(report);
    }
    return report;
  });

  await step('a double-sided surface casts whichever side faces the light', async () => {
    // A quad above a floor, lit from above, FACING the light. The shadow pass
    // draws back faces, so single-sided it casts nothing -- the limit the
    // README states -- and double-sided it must cast.
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
    });
    const floor = await probe.load(buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0 }));
    const dark = async (doubleSided) => {
      const quad = await probe.load(buildFeatureGLB({ metallicFactor: 0, doubleSided }));
      const scene = probe.createScene();
      scene.add(floor).setAxisAngle([1, 0, 0], -Math.PI / 2).setScale(4, 4, 1);
      scene.add(quad).setAxisAngle([1, 0, 0], -Math.PI / 2).setScale(0.35, 0.35, 1).setPosition(0, 1, 0);
      scene.addLight({ type: 'directional', direction: [1, -1, 0], intensity: 3 });
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set([0, 6, 0.01]);
      cam.target.set([0, 0, 0]);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      let count = 0;
      for (let i = 0; i < pixels.length; i += 4) if (pixels[i + 1] < 60) count++;
      return count;
    };
    const single = await dark(false);
    const double = await dark(true);
    probe.destroy();
    canvas.remove();
    const report = `shadow pixels with the lit side up: ${single} single-sided, ${double} double-sided`;
    if (!(single === 0 && double > 500)) throw new Error(report);
    return report;
  });

  await step('levels of detail switch at their coverage, and cast only the level shown', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
    });
    probe.gpu.device.pushErrorScope('validation');
    // The quad's sphere: radius hypot(3.2, 3.2) / 2 = 2.263. At fovY 1 its
    // coverage is 2.263 / tan(0.5) / d = 4.14 / d: 1.04 at 4, 0.35 at 12,
    // 0.10 at 40, and 0.04 at 100, below the last level's 0.05.
    const lod = await probe.load(buildLodGLB([0.5, 0.2, 0.05]));
    const middle = async (distance) => {
      const scene = probe.createScene();
      scene.add(lod);
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set([0, 0, distance]);
      cam.target.set([0, 0, 0]);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      const i = ((height >> 1) * width + (width >> 1)) * 4;
      return [pixels[i], pixels[i + 1], pixels[i + 2]];
    };
    const levels = [];
    for (const d of [4, 12, 40, 100]) levels.push(await middle(d));

    // A small copy in front of a lit wall, the light a little to the side so
    // its shadow falls clear of it. Turned to face the wall: the shadow pass
    // draws back faces, so a single-sided quad casts only with its back to the
    // light. The same quad whose levels all need more coverage than it has
    // shows no level, and must cast no shadow either.
    const wall = await probe.load(buildFeatureGLB({ baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0 }));
    const hidden = await probe.load(buildLodGLB([5, 4, 3]));
    const shadowAt = async (asset) => {
      const scene = probe.createScene();
      scene.add(wall).setPosition(0, 0, -2).setScale(4, 4, 1);
      scene.add(asset).setAxisAngle([0, 1, 0], Math.PI).setScale(0.3, 0.3, 1);
      scene.addLight({ type: 'directional', direction: [0.3, 0, -1], intensity: 3 });
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set([0, 0, 4]);
      cam.target.set([0, 0, 0]);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      const at = (x) => pixels[((height >> 1) * width + Math.floor(x * width)) * 4 + 1];
      // Where the shadow falls on the wall, and the same spot mirrored.
      return { shadow: at(0.603), clear: at(0.397) };
    };
    const cast = await shadowAt(lod);
    const none = await shadowAt(hidden);
    const error = await probe.gpu.device.popErrorScope();
    probe.destroy();
    canvas.remove();

    const show = (p) => `rgb(${p.join(',')})`;
    const report = `${levels.map(show).join(' -> ')} at 4, 12, 40 and 100 m; wall ${cast.clear} lit, `
      + `${cast.shadow} in the shown level's shadow, ${none.shadow} behind a group showing none`;
    if (error) throw new Error(`${report}; ${error.message}`);
    const only = (p, c) => p[c] > 200 && p.every((v, k) => k === c || v < 30);
    const ok = only(levels[0], 0) && only(levels[1], 1) && only(levels[2], 2) && levels[3].every((v) => v < 30)
      && cast.shadow < cast.clear - 30 && Math.abs(none.shadow - none.clear) <= 3;
    if (!ok) throw new Error(report);
    return report;
  });

  await step('debug lines draw their exact colour, hide behind geometry unless asked not to, and last one frame', async () => {
    const canvas = document.createElement('canvas');
    canvas.style.width = '320px';
    canvas.style.height = '240px';
    document.body.appendChild(canvas);
    const SKY = [0, 0, 0];
    const probe = await Winding.create(canvas, {
      environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } },
    });
    const quad = await probe.load(buildFeatureGLB({ baseColorFactor: [0.2, 0.2, 0.2, 1] }));
    // The line lies along y = 0, which lands on the boundary between the two
    // middle rows; whichever the rasterizer picks, one of them holds it.
    const row = async ({ withQuad = false, draw = true, depthTest = true } = {}) => {
      const scene = probe.createScene();
      if (withQuad) scene.add(quad);
      const cam = new Camera({ fovY: 1, near: 0.1 });
      cam.position.set([0, 0, 2]);
      cam.target.set([0, 0, 0]);
      probe.debug.depthTest = depthTest;
      if (draw) probe.debug.line([-1, 0, -1], [1, 0, -1], [1, 0, 0]);
      probe.renderFrame(scene, cam);
      const pixels = await probe.gpu.readPixels();
      const { width, height } = probe.gpu;
      const at = (y) => { const i = (y * width + (width >> 1)) * 4; return [pixels[i], pixels[i + 1], pixels[i + 2]]; };
      const [a, b] = [at((height >> 1) - 1), at(height >> 1)];
      return a[0] >= b[0] ? a : b;
    };
    const alone = await row();
    const hidden = await row({ withQuad: true });
    const onTop = await row({ withQuad: true, depthTest: false });
    const gone = await row({ draw: false });
    probe.destroy();
    canvas.remove();
    const show = (p) => `rgb(${p.join(',')})`;
    const report = `${show(alone)} alone, ${show(hidden)} behind a quad, ${show(onTop)} with depthTest off, `
      + `${show(gone)} the frame after`;
    const red = (p) => p[0] === 255 && p[1] === 0 && p[2] === 0;
    if (!(red(alone) && !red(hidden) && red(onTop) && gone[0] === 0)) throw new Error(report);
    return report;
  });

  await step('fog thickens with distance, takes its colour from the light, and pools below its height', async () => {
    // An unlit white quad straight ahead, which fog can only dim toward its
    // own colour. That colour is never given: under a black sky and a red
    // light it must come out red, and under a blue sky with no light, blue.
    const shots = async (sky, lights, cases) => {
      const canvas = document.createElement('canvas');
      canvas.style.width = '320px';
      canvas.style.height = '240px';
      document.body.appendChild(canvas);
      const probe = await Winding.create(canvas, {
        environment: { sky: { ground: sky, horizon: sky, zenith: sky, sunIntensity: 0, glow: 0 } },
      });
      const quad = await probe.load(buildFeatureGLB({
        baseColorFactor: [1, 1, 1, 1], materialExtensions: { KHR_materials_unlit: {} },
      }));
      const results = {};
      for (const [name, { fog, distance = 2, quad: drawQuad = true }] of Object.entries(cases)) {
        probe.renderer.fog = fog ?? null;
        const scene = probe.createScene();
        for (const light of lights) scene.addLight(light);
        const cam = new Camera({ fovY: 1, near: 0.1 });
        cam.position.set([0, 0, 0]);
        cam.target.set([0, 0, -1]);
        if (drawQuad) scene.add(quad).setPosition(0, 0, -distance).setScale(distance, distance, 1);
        probe.renderFrame(scene, cam);
        const pixels = await probe.gpu.readPixels();
        const { width, height } = probe.gpu;
        const at = (y) => { const i = (Math.floor(y * height) * width + (width >> 1)) * 4; return [pixels[i], pixels[i + 1], pixels[i + 2]]; };
        results[name] = { middle: at(0.5), top: at(0.02) };
      }
      probe.destroy();
      canvas.remove();
      return results;
    };
    const BLACK = [0, 0, 0];
    const red = [{ type: 'directional', direction: [0, -1, 0], color: [1, 0, 0], intensity: 4 * Math.PI }];
    const r = await shots(BLACK, red, {
      clear: { distance: 20 },
      near: { fog: { visibility: 20 }, distance: 2 },
      far: { fog: { visibility: 20 }, distance: 20 },
      layerHere: { fog: { visibility: 20, scaleHeight: 2 }, distance: 20 },
      layerBelow: { fog: { visibility: 20, height: -10, scaleHeight: 2 }, distance: 20 },
      skyUniform: { fog: { visibility: 20 }, quad: false },
      skyLayer: { fog: { visibility: 20, height: -10, scaleHeight: 2 }, quad: false },
    });
    const blue = await shots([0, 0, 1], [], { far: { fog: { visibility: 20 }, distance: 20 } });
    const gb = (p) => (p[1] + p[2]) / 2;
    const show = (p) => `rgb(${p.join(',')})`;
    const report = `white quad ${show(r.clear.middle)} clear; `
      + `${show(r.near.middle)} at 2 m and ${show(r.far.middle)} at 20 m in 20 m fog; `
      + `${show(r.layerHere.middle)} in a layer at eye height, ${show(r.layerBelow.middle)} with it 10 m below; `
      + `sky above ${show(r.skyUniform.top)} in uniform fog, ${show(r.skyLayer.top)} over a layer below; `
      + `${show(blue.far.middle)} under a blue sky`;
    const ok = gb(r.clear.middle) > 200
      && gb(r.near.middle) > gb(r.far.middle) + 60 && r.far.middle[0] > gb(r.far.middle) + 60
      && gb(r.layerBelow.middle) > gb(r.layerHere.middle) + 60
      && r.skyUniform.top[0] > 60 && r.skyUniform.top[0] > r.skyLayer.top[0] + 40
      && blue.far.middle[2] > blue.far.middle[0] + 60;
    if (!ok) throw new Error(report);
    return report;
  });

  await step('transmission shows the scene behind: tinted, blurred, bent and absorbed', async () => {
    // A glowing red quad a unit behind a glass one, in the dark, so every red
    // value in the middle came through the glass. Along one row, just past the
    // red quad's edge, is where a volume's refraction shows: the view bends
    // toward the axis inside it, onto the quad.
    const SKY = [0, 0, 0];
    const EDGE = [0.58, 0.62];   // inside the red quad, and just outside it
    const shots = async (engineOptions, cases) => {
      const canvas = document.createElement('canvas');
      canvas.style.width = '320px';
      canvas.style.height = '240px';
      document.body.appendChild(canvas);
      const probe = await Winding.create(canvas, {
        environment: { sky: { ground: SKY, horizon: SKY, zenith: SKY, sunIntensity: 0, glow: 0 } }, ...engineOptions,
      });
      probe.gpu.device.pushErrorScope('validation');
      const results = {};
      for (const [name, { glass, back = 1, lamp = false }] of Object.entries(cases)) {
        const scene = probe.createScene();
        const cam = new Camera({ fovY: 1, near: 0.1 });
        cam.position.set([0, 0, 2]);
        cam.target.set([0, 0, 0]);
        if (back > 0) {
          scene.add(await probe.load(buildFeatureGLB({ baseColorFactor: [1, 0, 0, 1], emissiveFactor: [1, 0, 0], metallicFactor: 0 })))
            .setPosition(0, 0, -1).setScale(back, back, 1);
        }
        if (glass) {
          scene.add(await probe.load(buildFeatureGLB({
            metallicFactor: 0, roughnessFactor: 0.05, ...glass, baseColorFactor: glass.baseColorFactor ?? [1, 1, 1, 1],
          })));
        }
        if (lamp) scene.addLight({ position: [0, 0, -0.5], color: [1, 1, 1], intensity: 3, radius: 4 });
        probe.renderFrame(scene, cam);
        const pixels = await probe.gpu.readPixels();
        const { width, height } = probe.gpu;
        const red = (x, y = 0.5) => pixels[(Math.floor(y * height) * width + Math.floor(x * width)) * 4];
        results[name] = { middle: red(0.5), edge: EDGE.map((x) => red(x)) };
      }
      const error = await probe.gpu.device.popErrorScope();
      probe.destroy();
      canvas.remove();
      if (error) throw new Error(`transmission with ${JSON.stringify(engineOptions)}: ${error.message}`);
      return results;
    };
    const glass = (extensions, extra = {}) => ({
      materialExtensions: { KHR_materials_transmission: { transmissionFactor: 1 }, ...extensions }, ...extra,
    });
    const r = await shots({}, {
      none: {},
      opaque: { glass: {} },
      clear: { glass: glass({}) },
      tinted: { glass: glass({}, { baseColorFactor: [0.2, 1, 1, 1] }) },
      sharp: { glass: glass({}), back: 0.03 },
      rough: { glass: glass({}, { roughnessFactor: 0.8 }), back: 0.03 },
      roughAtIor1: { glass: glass({ KHR_materials_ior: { ior: 1 } }, { roughnessFactor: 0.8 }), back: 0.03 },
      absorbing: { glass: glass({ KHR_materials_volume: { thicknessFactor: 0.5, attenuationDistance: 0.25, attenuationColor: [0.5, 1, 1] } }) },
      thin: { glass: glass({}), back: 0.3 },
      thick: { glass: glass({ KHR_materials_volume: { thicknessFactor: 2 } }), back: 0.3 },
      lampThrough: { glass: glass({}, { roughnessFactor: 0.5 }), back: 0, lamp: true },
      lampOpaque: { glass: { roughnessFactor: 0.5 }, back: 0, lamp: true },
    });
    const report = `red behind ${r.none.middle}: glass ${r.opaque.middle} opaque, ${r.clear.middle} clear, `
      + `${r.tinted.middle} tinted; small target ${r.sharp.middle} sharp, ${r.rough.middle} rough, `
      + `${r.roughAtIor1.middle} rough at ior 1; ${r.absorbing.middle} absorbed; past the edge `
      + `${r.thin.edge[1]} thin, ${r.thick.edge[1]} thick; lamp behind ${r.lampOpaque.middle} -> ${r.lampThrough.middle}`;
    const ok = r.opaque.middle < 20
      && Math.abs(r.clear.middle - r.none.middle) <= 8
      && r.tinted.middle < r.clear.middle - 40
      && r.rough.middle < r.sharp.middle - 100 && Math.abs(r.roughAtIor1.middle - r.sharp.middle) <= 8
      && r.absorbing.middle < r.clear.middle - 40
      && r.thin.edge[0] > 200 && r.thin.edge[1] < 20 && r.thick.edge[1] > 200
      && r.lampThrough.middle > r.lampOpaque.middle + 100;
    if (!ok) throw new Error(report);

    // Its own pass beside ambient occlusion's two targets, and beside OIT.
    for (const options of [{ ao: true }, { oit: true }]) {
      const { clear } = await shots(options, { clear: { glass: glass({}) } });
      if (Math.abs(clear.middle - r.clear.middle) > 8) throw new Error(`with ${JSON.stringify(options)}: ${clear.middle}`);
    }
    return report;
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
      await oitEngine.gpu.device.queue.onSubmittedWorkDone();

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

  await step('unload gives back ids and buffers, and only once no scene draws the asset', async () => {
    const asset = await engine.load(await buildDemoGLB({ arms: 2 }));
    if (asset.materialIds.length === 0) throw new Error('the demo asset has no materials to give back');
    const node = scene.add(asset);
    let refused = false;
    try { engine.unload(asset); } catch { refused = true; }
    if (!refused) throw new Error('unloaded an asset a scene still draws');

    node.destroy();
    engine.unload(asset);
    engine.unload(asset);   // twice is harmless
    // Drawing after the free is the real check: a batch still naming a
    // destroyed buffer shows up as the device error asserted at the end.
    engine.renderFrame(scene, camera);
    await engine.gpu.device.queue.onSubmittedWorkDone();

    const again = await engine.load(await buildDemoGLB({ arms: 2 }));
    if (!again.materialIds.every((id) => asset.materialIds.includes(id))) {
      throw new Error(`reload took ids ${again.materialIds} rather than the freed ${asset.materialIds}`);
    }
    engine.unload(again);
    return `${asset.materialIds.length} material ids came back and were reused`;
  });

  await step('two scenes the same size keep their own draw lists', async () => {
    // Two renderables each, so under per-scene counters both reported
    // revision 2 -- and the renderer, which compared revision alone, kept the
    // first scene's sorted batches for the second.
    const asset = await engine.load(await buildDemoGLB({ arms: 2 }));
    if (asset.meshes.length < 2) throw new Error('the demo needs two meshes for this');
    const meshA = { name: 'a', primitives: [asset.meshes[0].primitives[0]] };
    const meshB = { name: 'b', primitives: [asset.meshes[1].primitives[0]] };
    const at = (x) => ({ position: [x, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], children: [] });
    const same = engine.createScene();     // one mesh twice: one batch
    same.add({ meshes: [meshA], nodes: [{ ...at(-2), mesh: 0 }, { ...at(2), mesh: 0 }], roots: [0, 1] });
    const mixed = engine.createScene();    // two meshes: two batches
    mixed.add({ meshes: [meshA, meshB], nodes: [{ ...at(-2), mesh: 0 }, { ...at(2), mesh: 1 }], roots: [0, 1] });

    const counts = [];
    for (const s of [same, mixed, same, mixed]) {
      engine.renderFrame(s, camera);
      const { batchList, gpu } = engine.renderer;
      if (batchList.count !== gpu.batchCount) {
        throw new Error(`drew ${batchList.count} batches of the ${gpu.batchCount} this scene has`);
      }
      counts.push(gpu.batchCount);
    }
    await engine.gpu.device.queue.onSubmittedWorkDone();
    return `batches per frame: ${counts.join(', ')}`;
  });

  await step('a load that fails part way gives back what it had built', async () => {
    const realEnsure = engine.renderer.ensureVariants;
    engine.renderer.ensureVariants = () => Promise.reject(new Error('forced'));
    const freeBefore = engine.renderer.materials._free.length;
    try {
      await engine.load(await buildDemoGLB({ arms: 2 }));
      throw new Error('the forced failure did not surface');
    } catch (error) {
      if (error.message !== 'forced') throw error;
    } finally {
      engine.renderer.ensureVariants = realEnsure;
    }
    const returned = engine.renderer.materials._free.length - freeBefore;
    if (returned !== 3) throw new Error(`${returned} material ids came back, expected the demo's 3`);
    return 'buffers, textures and 3 material ids freed';
  });

  await step('the device is asked for what the adapter has, not the defaults', async () => {
    const { adapter, limits } = engine.gpu;
    for (const name of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxTextureDimension2D']) {
      if (limits[name] !== adapter.limits[name]) {
        throw new Error(`${name}: device ${limits[name]}, adapter ${adapter.limits[name]}`);
      }
    }
    // A cube this small has five levels; asking for six made an invalid texture.
    const small = new Environment(engine.gpu, { size: 16 });
    const mips = small.prefilterMips;
    small.destroy();
    if (mips !== 5) throw new Error(`a 16 cube kept ${mips} prefilter levels`);
    let refused = false;
    try { engine.unload({ engine: {}, meshes: [], materialIds: [] }); } catch { refused = true; }
    if (!refused) throw new Error("unloaded another engine's asset");
    return `maxTextureDimension2D ${limits.maxTextureDimension2D}, maxBufferSize ${limits.maxBufferSize}`;
  });

  await step('a destroyed engine refuses work by name, including a load in flight', async () => {
    const doomedCanvas = document.createElement('canvas');
    document.body.appendChild(doomedCanvas);
    const doomed = await Winding.create(doomedCanvas);
    const inFlight = doomed.load(await buildDemoGLB({ arms: 1 }));
    doomed.destroy();
    doomedCanvas.remove();
    let message = '';
    try { await inFlight; } catch (error) { message = error.message; }
    if (!/destroyed/.test(message)) throw new Error(`a load in flight ${message ? `threw "${message}"` : 'resolved'}`);
    for (const call of [() => doomed.createScene(), () => doomed.renderFrame(null, camera), () => doomed.run(null, camera)]) {
      let threw = '';
      try { call(); } catch (error) { threw = error.message; }
      if (!/destroyed/.test(threw)) throw new Error(`a call on a destroyed engine ${threw ? `threw "${threw}"` : 'went through'}`);
    }
    return 'load, createScene, renderFrame and run all refuse';
  });

  await step('an engine that never ran shuts down when its canvas leaves', async () => {
    // No loop to notice, so this is the resize observer's job -- and a hidden
    // page delivers no observations at all, so there is nothing to test there.
    if (document.visibilityState === 'hidden') return 'not run: the page is hidden';
    const idleCanvas = document.createElement('canvas');
    document.body.appendChild(idleCanvas);
    const idle = await Winding.create(idleCanvas);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    idleCanvas.remove();
    for (let i = 0; i < 20 && !idle.gpu.destroyed; i++) await new Promise((r) => setTimeout(r, 50));
    if (!idle.gpu.destroyed) {
      idle.destroy();
      throw new Error('still alive a second after its canvas left');
    }
    return 'destroyed by its resize observer';
  });

  await step('a running engine whose canvas leaves the page shuts down', async () => {
    // What a live editor's reload does: the page is rewritten in place, the
    // old canvas drops out of the document and nothing tells the engine.
    // The loop is driven by hand, so a hidden tab's paused rAF cannot stall it.
    const frames = [];
    const realRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (cb) => frames.push(cb);
    const goneCanvas = document.createElement('canvas');
    document.body.appendChild(goneCanvas);
    const gone = await Winding.create(goneCanvas);
    try {
      const goneScene = gone.createScene();
      let drawn = 0;
      gone.run(goneScene, new Camera({ near: 0.1 }), { frame: () => drawn++ });
      const tick = (ms) => frames.splice(0).forEach((cb) => cb(ms));
      tick(16);
      if (drawn !== 1 || gone.gpu.destroyed) throw new Error('a connected canvas did not draw');
      goneCanvas.remove();
      tick(32);
      if (drawn !== 1) throw new Error('drew into a detached canvas');
      if (!gone.gpu.destroyed) throw new Error('the device was kept alive');
      if (frames.length) throw new Error('the loop asked for another frame');
      gone.destroy();   // the page's own teardown may still call it
      return 'destroyed on the first frame after removal';
    } finally {
      globalThis.requestAnimationFrame = realRaf;
      gone.destroy();
      goneCanvas.remove();
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
