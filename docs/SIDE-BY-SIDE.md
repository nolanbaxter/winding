# Hello world, side by side

The same small program written four times: once in Winding and once in each of Three.js, Babylon.js
and PlayCanvas. Each one:

- draws to a `<canvas>` that fills the window, through WebGPU, and keeps up when the window resizes;
- loads a glTF model;
- lights it with a sun that casts shadows, plus light from the surroundings, tone-mapped;
- gives you a camera you can drag to orbit and scroll to zoom.

Each is written as its own documentation shows, and each was run in Chrome to check that it works
as printed. The page is `<canvas></canvas>` with `width: 100%; height: 100%` on a page with no
margin, plus an import map or bundler that resolves the imports.

**Versions when this was written (12 October 2026):**

| Engine | Package | Version |
|---|---|---|
| Winding | `winding-engine` | 1.7.0 |
| Three.js | `three` | 0.186.1 (r186) |
| Babylon.js | `@babylonjs/core`, `@babylonjs/loaders` | 9.30.0 |
| PlayCanvas | `playcanvas` | 2.23.2 |

Newer versions of any of them may differ. Each engine's own docs are the authority on its current
API.

## Line count

| | Winding | Three.js | Babylon.js | PlayCanvas |
|---|---|---|---|---|
| Lines, blank lines excluded | 11 | 37 | 21 | 25 |
| Imports | 1 | 4 | 2 | 2 |

Line count is a rough measure. Most of the difference comes from what each engine does by default,
not from how wordy it is; the table after the code shows where.

## Winding

```js
import { Winding, Camera, OrbitController } from 'winding-engine';

const canvas = document.querySelector('canvas');
const engine = await Winding.create(canvas);
const scene = engine.createScene();
const camera = new Camera({ fovY: Math.PI / 3, near: 0.1 });

scene.add(await engine.load('model.glb'));
scene.addLight({ type: 'directional', direction: [-0.4, -0.8, -0.4], intensity: 3 });

const orbit = new OrbitController(camera, canvas, { distance: 4, target: [0, 1, 0] });
engine.run(scene, camera, {
  frame: (alpha, clock) => orbit.update(clock.realDelta),
});
```

The engine sizes the canvas and follows its resizes. It bakes a sky and lights the scene from it,
and a directional light casts shadows unless told not to. Nothing has to be turned on, and nothing
has to be re-done when the window changes.

## Three.js

```js
import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const canvas = document.querySelector('canvas');
const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
await renderer.init();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xa0b8d0);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment()).texture;
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 1000);
camera.position.set(0, 2.2, 3.8);

const gltf = await new GLTFLoader().loadAsync('model.glb');
gltf.scene.traverse((object) => {
  if (object.isMesh) object.castShadow = object.receiveShadow = true;
});
scene.add(gltf.scene);

const sun = new THREE.DirectionalLight(0xffffff, 3);
sun.position.set(4, 8, 4);
sun.castShadow = true;
scene.add(sun);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 1, 0);
controls.enableDamping = true;
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
```

Compared with Winding:

- **Shadows take three switches:** on the renderer, on the light, and on every mesh. The last one
  means walking the loaded model.
- **Surrounding light is built by hand:** a `RoomEnvironment` scene, prefiltered by a
  `PMREMGenerator`. Without it, metal and glossy surfaces have nothing to reflect.
- **Resizing is yours:** the camera's aspect, its projection matrix and the renderer's size.
- **The sun is a position, not a direction.** Its shadow covers a fixed box, 10 m across by
  default; a larger scene has to set `sun.shadow.camera`'s bounds itself.
- Orbit controls, the glTF loader and the room environment are addons, imported separately.

## Babylon.js

```js
import { WebGPUEngine, Scene, ArcRotateCamera, DirectionalLight, ShadowGenerator, Vector3, ImportMeshAsync } from '@babylonjs/core';
import '@babylonjs/loaders/glTF/index.js';

const canvas = document.querySelector('canvas');
const engine = new WebGPUEngine(canvas, { antialias: true });
await engine.initAsync();

const scene = new Scene(engine);
scene.createDefaultEnvironment({ createGround: false });
scene.imageProcessingConfiguration.toneMappingEnabled = true;
const camera = new ArcRotateCamera('camera', -Math.PI / 2, 1.2, 4, new Vector3(0, 1, 0), scene);
camera.minZ = 0.1;
camera.attachControl(canvas, true);

const { meshes } = await ImportMeshAsync('model.glb', scene);

const sun = new DirectionalLight('sun', new Vector3(-0.4, -0.8, -0.4), scene);
sun.intensity = 3;
const shadows = new ShadowGenerator(2048, sun);
for (const mesh of meshes) {
  shadows.addShadowCaster(mesh);
  mesh.receiveShadows = true;
}

engine.runRenderLoop(() => scene.render());
addEventListener('resize', () => engine.resize());
```

Compared with Winding:

- **Shadows come from a `ShadowGenerator`,** with each mesh added as a caster and set to receive.
- **`createDefaultEnvironment()`** gives the image lighting and a skybox in one call. It downloads
  its environment texture from Babylon's own asset server.
- **Every object takes a name** as its first argument (`'camera'`, `'sun'`).
- **The glTF loader is a separate package,** registered by importing it for its side effect.
- Tone mapping is off by default and is turned on through the scene's image processing.
- `ArcRotateCamera` is the orbit camera, placed by two angles and a radius.

## PlayCanvas

```js
import * as pc from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';

const canvas = document.querySelector('canvas');
const device = await pc.createGraphicsDevice(canvas, { deviceTypes: ['webgpu'], antialias: true });
const app = new pc.Application(canvas, { graphicsDevice: device });
app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
app.setCanvasResolution(pc.RESOLUTION_AUTO);
addEventListener('resize', () => app.resizeCanvas());
app.scene.ambientLight = new pc.Color(0.35, 0.35, 0.4);

const asset = await new Promise((resolve, reject) => {
  app.assets.loadFromUrl('model.glb', 'container', (error, asset) => (error ? reject(error) : resolve(asset)));
});
app.root.addChild(asset.resource.instantiateRenderEntity({ castShadows: true, receiveShadows: true }));

const sun = new pc.Entity('sun');
sun.addComponent('light', { type: 'directional', intensity: 3, castShadows: true });
sun.setEulerAngles(40, 45, 0);
app.root.addChild(sun);

const camera = new pc.Entity('camera');
camera.addComponent('camera', { clearColor: new pc.Color(0.63, 0.72, 0.82), toneMapping: pc.TONEMAP_ACES });
camera.setPosition(0, 2.2, 3.8);
camera.addComponent('script');
app.root.addChild(camera);
const controls = camera.script.create(CameraControls);
controls.focusPoint = new pc.Vec3(0, 1, 0);

app.start();
```

Compared with Winding:

- **Everything is an entity with components:** the camera, the light and the model each get added
  to `app.root`.
- **Loading uses a callback,** wrapped in a promise here.
- **Shadows are flags** on the light and on the loaded model.
- **No image lighting without an asset.** PlayCanvas lights from its surroundings only once you
  load an environment atlas, so this uses a flat ambient colour instead, and the scene is lit
  less fully than in the other three.
- The orbit camera is `CameraControls`, a script shipped in the package (`scripts/esm/`) and
  attached through a script component.
- The light is aimed by rotating its entity: a directional light shines down the entity's −Y.

## What each one does for you

| | Winding | Three.js | Babylon.js | PlayCanvas |
|---|---|---|---|---|
| Canvas sized and resized | yes | you write it | a resize listener | two calls and a resize listener |
| Light from the surroundings | yes | you build it | one call | needs an asset |
| Sun shadows | on by default | three switches | a generator | flags |
| Shadow area fitted | to the view | you set the box | to the casters | to the view |
| Tone mapping | on by default | one line | one line | camera option |
| Orbit camera | built in | addon | built in | script in the package |
| glTF loader | built in | addon | separate package | built in |
| WebGL fallback | none | automatic | a different engine class | add `'webgl2'` to the list |

The last row is where Winding does less. It draws with WebGPU only, so a page that has to reach
browsers without it should check [`Winding.supported()`](API.md#winding-supported) first and show
something else. The other three can draw through WebGL instead.
