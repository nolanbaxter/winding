# Winding API reference

Every public call, option and setting, with what it does, what it takes, what it returns and what
it throws. Look a call up in the [index](#index) below, or search this page for its name. The
[README](../README.md) says what the engine is and how it's built; this page is how to use it.

```js
import { Winding, Camera, Camera2D } from 'winding-engine';
```

TypeScript types come with the package, so an editor shows these calls' options as you type. The
option shapes have names you can import, such as `SpriteOptions`, `Texture` and `Hit`:
`import type { SpriteOptions } from 'winding-engine'`.

<a id="conventions"></a>
## Conventions

These hold everywhere, so one learned is one learned for good.

- **Every kind of thing works the same way.** `scene.addX(options)` makes one and returns its
  [`Node`](#node). `scene.setX(node, changes)` changes any of the options `addX` took, leaving the
  rest. `scene.xOf(node)` returns a copy of its options, or `null` if the node has none. It goes
  with its node: [`scene.remove(node)`](#scene-remove) or [`node.destroy()`](#node-destroy). The
  kinds: sprites, text, shapes, paths, tilemaps, emitters, decals, lights and reflection probes.
- **Everything is placed by its node.** A node's position, rotation and scale place what hangs off
  it, and a node parented to another follows it. `position` and `parent` are options of every
  `addX`.
- **Two numbers mean 2D.** `[x, y]` or `(x, y)` wherever a 3D call takes three: z is 0 for a
  position or direction, and 1 for a scale.
- **`pivot` is the point of a thing placed at its node, measured from its top-left**: `[0, 0]` is
  the top-left, `[1, 1]` the bottom-right, `[0.5, 0.5]` the centre. The same in 3D and 2D, for
  sprites, text, shapes, tilemaps and the 2D camera.
- **Angles are radians.** In a 2D view a positive angle turns clockwise, as CSS `rotate()` does.
- **Colours are `[r, g, b, a]`, 0 to 1** — `[r, g, b]` for a light. In a 3D view they're linear
  light, as lighting works, so a colour past 1 glows through bloom. In a 2D view they're sRGB, as
  CSS and image editors work, so `rgb(128, 128, 128)` is `[128 / 255, 128 / 255, 128 / 255, 1]`
  and lands on screen as 128. The same sprite's `color` means linear light when a 3D camera draws
  it and sRGB when a 2D camera does.
- **Units.** A 3D world unit is a metre, as in glTF. A 2D unit is a CSS pixel, the same size on
  any screen.
- **Errors are thrown, and name the call.** A bad option throws at the call that took it, with
  the option's name and what it should be, rather than drawing something quietly wrong.
- **Nothing needs setting up or tearing down in order.** There's no `init()`, and the engine grows
  what it needs as you add to a scene.

<a id="contents"></a>
## Contents

[Winding](#winding) · [Renderer settings](#renderer-settings) · [Debug lines](#debug-lines) · [Scene](#scene) · [Node](#node) · [Camera](#camera) · [Animation](#animation) · [Lights](#lights) · [Particles](#particles) · [Gaussian splats](#splats) · [Decals](#decals) · [Reflection probes](#reflection-probes) · [Sprites](#sprites) · [Text](#text) · [Shapes](#shapes) · [Paths](#paths) · [Tilemaps](#tilemaps) · [Camera2D](#camera2d) · [The 2D view](#the-2d-view) · [OrbitController](#orbitcontroller) · [StatsOverlay](#statsoverlay) · [Benchmark](#benchmark) · [Environment and HDR](#environment-and-hdr) · [Colour helpers](#colour-helpers) · [Math](#math) · [Renamed in 1.0](#renamed)

<a id="index"></a>
## Index

Every call, setting and entry, A to Z by its name.

**A** &nbsp; [`aabb`](#math-aabb) · [`scene.add()`](#scene-add) · [`scene.addDecal()`](#scene-adddecal) · [`scene.addEmitter()`](#scene-addemitter) · [`scene.addLight()`](#scene-addlight) · [`scene.addPath()`](#scene-addpath) · [`scene.addProbe()`](#scene-addprobe) · [`scene.addShape()`](#scene-addshape) · [`scene.addSplats()`](#scene-addsplats) · [`scene.addSprite()`](#scene-addsprite) · [`scene.addText()`](#scene-addtext) · [`scene.addTilemap()`](#scene-addtilemap) · [`scene.advance()`](#scene-advance) · [`node.alive`](#node-alive) · [`camera.ambient`](#camera2d-ambient) · [`camera.angle`](#camera2d-angle) · [`node.animation`](#node-animation) · [`node.animations`](#node-animations) · [`engine.renderer.post.antialias`](#post-antialias) · [`engine.renderer.ao`](#renderer-ao) · [`engine.renderer.autoExposure`](#renderer-autoexposure) · [`engine.debug.axes()`](#debug-axes)

**B** &nbsp; [`camera.background`](#camera2d-background) · [`new Benchmark()`](#benchmark) · [`scene.bounds()`](#scene-bounds) · [`engine.debug.box()`](#debug-box) · [`scene.burst()`](#scene-burst)

**C** &nbsp; [`Camera properties`](#camera-properties) · [`new Camera()`](#camera) · [`new Camera2D()`](#camera2d) · [`engine.captureProbes()`](#engine-captureprobes) · [`node.children()`](#node-children) · [`scene.childrenOf()`](#scene-childrenof) · [`engine.debug.circle()`](#debug-circle) · [`engine.clock`](#engine-clock) · [`colorFromBytes()`](#color-colorfrombytes) · [`colorFromHex()`](#color-colorfromhex) · [`Winding.create()`](#winding-create) · [`scene.createNode()`](#scene-createnode) · [`engine.createScene()`](#engine-createscene) · [`engine.createTarget()`](#engine-createtarget)

**D** &nbsp; [`engine.debug`](#engine-debug) · [`scene.decalOf()`](#scene-decalof) · [`engine.debug.depthTest`](#debug-depthtest) · [`engine.destroy()`](#engine-destroy) · [`node.destroy()`](#node-destroy) · [`overlay.destroy()`](#statsoverlay-destroy) · [`orbit.detach()`](#orbitcontroller-detach) · [`engine.renderer.dof`](#renderer-dof) · [`orbit.dragged`](#orbitcontroller-dragged) · [`engine.renderer.drawSkybox` (renamed)](#renderer-drawskybox)

**E** &nbsp; [`scene.emitterOf()`](#scene-emitterof) · [`engine.environment`](#engine-environment) · [`scene.environment`](#scene-environment) · [`new Environment()`](#environment) · [`engine.renderer.exposure`](#renderer-exposure)

**F** &nbsp; [`engine.renderer.post.filterRadius`](#post-filterradius) · [`engine.renderer.fog`](#renderer-fog) · [`camera.follow()`](#camera-follow) · [`Benchmark.format()`](#benchmark-format) · [`engine.fps`](#engine-fps) · [`scene.frame()`](#scene-frame) · [`orbit.frameBounds()`](#orbitcontroller-framebounds) · [`camera.frameBounds()`](#camera-framebounds)

**G** &nbsp; [`engine.gpu`](#engine-gpu) · [`engine.grading`](#engine-grading) · [`engine.renderer.post.grading`](#post-grading)

**H** &nbsp; [`HUD`](#view2d-hud)

**I** &nbsp; [`engine.invalidate()`](#engine-invalidate) · [`camera.is2D`](#camera2d-is2d)

**K** &nbsp; [`engine.renderer.post.knee`](#post-knee)

**L** &nbsp; [`engine.renderer.lightDistance`](#renderer-lightdistance) · [`engine.renderer.post.levels`](#post-levels) · [`scene.lightOf()`](#scene-lightof) · [`engine.debug.line()`](#debug-line) · [`linearToSrgb()`](#color-lineartosrgb) · [`lit`](#view2d-lighting) · [`engine.load()`](#engine-load) · [`engine.loadEnvironment()`](#engine-loadenvironment) · [`engine.loadFont()`](#engine-loadfont) · [`engine.loadLUT()`](#engine-loadlut) · [`engine.loadSplats()`](#engine-loadsplats) · [`engine.loadTexture()`](#engine-loadtexture)

**M** &nbsp; [`mat4`](#math-mat4)

**N** &nbsp; [`scene.node()`](#scene-node)

**O** &nbsp; [`engine.renderer.oit`](#renderer-oit) · [`engine.onDemand`](#engine-ondemand) · [`new OrbitController()`](#orbitcontroller) · [`camera.orthographicHalfHeight()`](#camera-orthographichalfheight)

**P** &nbsp; [`Painter's order`](#view2d-order) · [`parseHDR()`](#parsehdr) · [`scene.particlesActive`](#scene-particlesactive) · [`scene.pathOf()`](#scene-pathof) · [`scene.pick()`](#scene-pick) · [`scene.pick()`](#scene-pick-2d) · [`camera.pivot`](#camera2d-pivot) · [`engine.gpu.pixelRatio`](#engine-gpu-pixelratio) · [`camera.pixelSnap`](#camera2d-pixelsnap) · [`node.play()`](#node-play) · [`camera.position`](#camera2d-position) · [`scene.probeOf()`](#scene-probeof)

**Q** &nbsp; [`quat`](#math-quat)

**R** &nbsp; [`scene.raycast()`](#scene-raycast) · [`camera.rayFromScreen()`](#camera-rayfromscreen) · [`engine.gpu.readPixels()`](#engine-gpu-readpixels) · [`scene.remove()`](#scene-remove) · [`engine.renderer`](#engine-renderer) · [`engine.renderFrame()`](#engine-renderframe) · [`benchmark.report()`](#benchmark-report) · [`engine.renderer.post.requestedLevels` (renamed)](#post-requestedlevels) · [`engine.renderer.resolution`](#renderer-resolution) · [`engine.rhi` (renamed)](#engine-rhi) · [`camera.rotation` (renamed)](#camera2d-rotation) · [`engine.run()`](#engine-run) · [`benchmark.run()`](#benchmark-run)

**S** &nbsp; [`new Scene()`](#scene-constructor) · [`camera.screenToWorld()`](#camera2d-screentoworld) · [`node.setAngle()`](#node-setangle) · [`node.setAxisAngle()`](#node-setaxisangle) · [`scene.setDecal()`](#scene-setdecal) · [`node.setDirection()`](#node-setdirection) · [`scene.setEmitter()`](#scene-setemitter) · [`node.setEuler()`](#node-seteuler) · [`scene.setLight()`](#scene-setlight) · [`node.setParent()`](#node-setparent) · [`scene.setPath()`](#scene-setpath) · [`node.setPosition()`](#node-setposition) · [`scene.setProbe()`](#scene-setprobe) · [`node.setRotation()`](#node-setrotation) · [`node.setScale()`](#node-setscale) · [`scene.setShape()`](#scene-setshape) · [`scene.setSprite()`](#scene-setsprite) · [`scene.setText()`](#scene-settext) · [`scene.setTile()`](#scene-settile) · [`scene.setTilemap()`](#scene-settilemap) · [`scene.setTiles()`](#scene-settiles) · [`engine.renderer.shadowDistance`](#renderer-shadowdistance) · [`engine.renderer.shadows`](#renderer-shadows) · [`scene.shapeOf()`](#scene-shapeof) · [`engine.skippedFrames`](#engine-skippedframes) · [`engine.renderer.skybox`](#renderer-skybox) · [`engine.renderer.softShadows`](#renderer-softshadows) · [`engine.debug.sphere()`](#debug-sphere) · [`scene.splatsOf()`](#scene-splatsof) · [`scene.spriteOf()`](#scene-spriteof) · [`spriteSheet()`](#spritesheet) · [`sRGB colours`](#view2d-colour) · [`srgbToLinear()`](#color-srgbtolinear) · [`benchmark.start()`](#benchmark-start) · [`engine.stats`](#engine-stats) · [`new StatsOverlay()`](#statsoverlay) · [`engine.stop()`](#engine-stop) · [`node.stop()`](#node-stop) · [`engine.renderer.post.strength`](#post-strength) · [`orbit.syncFromCamera()`](#orbitcontroller-syncfromcamera)

**T** &nbsp; [`engine.renderer.taa`](#renderer-taa) · [`scene.textOf()`](#scene-textof) · [`engine.renderer.post.threshold`](#post-threshold) · [`scene.tileAt()`](#scene-tileat) · [`scene.tilemapOf()`](#scene-tilemapof)

**U** &nbsp; [`engine.unload()`](#engine-unload) · [`scene.update()`](#scene-update) · [`camera.update()`](#camera2d-update) · [`camera.update()`](#camera-update) · [`orbit.update()`](#orbitcontroller-update) · [`overlay.update()`](#statsoverlay-update)

**V** &nbsp; [`vec3`](#math-vec3) · [`camera.view`](#camera2d-matrices)

**W** &nbsp; [`node.weights`](#node-weights) · [`camera.width`](#camera2d-size) · [`node.worldPosition()`](#node-worldposition) · [`camera.worldToScreen()`](#camera2d-worldtoscreen)

**Z** &nbsp; [`camera.zoom`](#camera2d-zoom)

<a id="winding"></a>
## Winding

<a id="winding-create"></a>
### `Winding.create(canvas, options)` → `Promise<Winding>`

Creates an engine on a `<canvas>`: requests the WebGPU device, compiles the renderer's pipelines, and bakes the default environment. The canvas is sized, configured and watched for resizes for you.

| Option | Default | Meaning |
|---|---|---|
| `label` | `'winding'` | Label of the GPU device, shown in WebGPU error messages. |
| `powerPreference` | `'high-performance'` | Passed to `requestAdapter`: `'high-performance'` or `'low-power'`. |
| `onDeviceLost` | `null` | `(detail) => {}` when the GPU goes away. `detail` is `{ reason, message, recoverable, action: 'reload' }`. Without it the loss is logged to the console. The engine destroys itself either way. |
| `onError` | `null` | `(error) => {}` for uncaptured WebGPU errors. Without it they go to `console.error`. |
| `exposure` | `1` | Starting value of [`renderer.exposure`](#renderer-exposure). |
| `autoExposure` | `null` | Starting value of [`renderer.autoExposure`](#renderer-autoexposure). |
| `antialias` | `true` | FXAA after the tonemap. Same as `post.antialias`; see [`post.antialias`](#post-antialias). |
| `grading` | `null` | Starting colour grading; see [`engine.grading`](#engine-grading). |
| `post` | `{}` | Bloom and post settings: `threshold`, `knee`, `filterRadius`, `strength`, `levels`, `antialias`, `grading`. See [Renderer settings](#renderer-settings). Values here win over the top-level `antialias` and `grading`. |
| `shadows` | `{}` | Shadow maps. See the table below; some fields can change later through [`renderer.shadows`](#renderer-shadows). |
| `shadowDistance` | `null` | Starting value of [`renderer.shadowDistance`](#renderer-shadowdistance). |
| `lightDistance` | `null` | Starting value of [`renderer.lightDistance`](#renderer-lightdistance). |
| `ao` | `false` | Ambient occlusion: `true`, or `{ radius }` in world units. Starting value of [`renderer.ao`](#renderer-ao). |
| `oit` | `false` | Starting value of [`renderer.oit`](#renderer-oit). |
| `taa` | `false` | Starting value of [`renderer.taa`](#renderer-taa). |
| `fog` | `null` | Starting value of [`renderer.fog`](#renderer-fog). |
| `dof` | `null` | Starting value of [`renderer.dof`](#renderer-dof). |
| `environment` | `{}` | Settings for the default [`Environment`](#environment) (`size`, `irradianceSize`, `prefilterMips`, `sky`, `map`, `label`). Settings only, not an `Environment` instance. |
| `maxDraws` | `4096` | Starting capacity of the draw lists. They grow as needed. |
| `gpuTiming` | `false` | Time every pass on the GPU. Read results from `engine.renderer.gpuTiming`, and switch at any time with `engine.renderer.gpuTiming.enabled`. Does nothing on a device without `timestamp-query`. |
| `workerCount` | `min(cores - 1, 7)` | Transform workers. Workers only run when the page is cross-origin isolated (COOP + COEP); otherwise work runs on the main thread and this is ignored. `0` runs inline. |
| `onDemand` | `true` | Starting value of [`engine.onDemand`](#engine-ondemand). Pass `false` to draw every frame. |

`shadows` options:

| Option | Default | Meaning |
|---|---|---|
| `size` | `2048` | Texels on a side of each directional shadow cascade. |
| `cascades` | `4` | Cascades per shadow-casting directional light, 1 to 4. |
| `lambda` | `0.7` | How cascades split the range: `0` uniform, `1` logarithmic. |
| `casterExtent` | `4` | How far behind each cascade casters are still caught, as a multiple of the cascade's radius. |
| `normalBias` | `1.5` | Offset along the surface normal at lookup, in texels. Raise it for shadow acne. |
| `depthBiasSlope` | `-2` | Slope-scaled depth bias while drawing the map. Fixed at creation. |
| `depthBiasConstant` | `-1` | Constant depth bias while drawing the map. Fixed at creation. |
| `localSize` | `512` | Texels on a side of each point or spot shadow view. A point light uses six. |

Returns: the engine.
Throws: if `options.environment` is an `Environment` instance; `'WebGPU is not available…'` when `navigator.gpu` is missing; `'No WebGPU adapter…'`; `'Could not get a webgpu context from the canvas.'`; when the GPU allows fewer storage buffers per stage than the renderer reads; `RangeError` when `shadows.cascades` is outside 1 to 4, `shadows.size` is past the device limit, or `shadows.localSize` is outside 5 to the device limit; `RangeError` when an environment size is past the device limit.

```js
import { Winding, Camera } from 'winding-engine';

const engine = await Winding.create(canvas, {
  ao: true,
  shadows: { size: 4096 },
  onDeviceLost: () => location.reload(),
});
```

Notes: a `shadows.shadowDistance` is ignored; use the top-level `shadowDistance`.

<a id="engine-createscene"></a>
### `engine.createScene(options)` → `Scene`

Creates an empty scene lit by an environment. Every scene you draw comes from here.

| Option | Default | Meaning |
|---|---|---|
| `environment` | `engine.environment` | The [`Environment`](#environment) that lights the scene and draws its background, such as one from [`engine.loadEnvironment`](#engine-loadenvironment). Several scenes can share one. |
| `capacity` | `4096` | Starting node capacity (a `Scene` option). |
| `renderableCapacity` | `capacity` | Starting capacity for drawn meshes (a `Scene` option). |
| `lightCapacity` | `256` | Starting light capacity (a `Scene` option). |

Returns: a `Scene` with `scene.environment` set.
Throws: `'createScene: this engine was destroyed'`; `'createScene: that Environment belongs to another engine'`.

```js
const studio = await engine.loadEnvironment('studio.hdr');
const scene = engine.createScene({ environment: studio });
```

<a id="engine-load"></a>
### `engine.load(source, options)` → `Promise<Model>`

Loads a `.glb` or `.gltf` and returns a model that `scene.add()` takes. All the slow work (download, decode, upload, pipeline compiles) finishes before it resolves, so adding the model never stalls a frame.

`source` is a URL string, an `ArrayBuffer` or a `Uint8Array`.

| Option | Default | Meaning |
|---|---|---|
| `retainGeometry` | `false` | Keep positions and indices (and skin weights and morph deltas) on the CPU, so `scene.raycast` hits triangles instead of bounding boxes. Costs about 12 bytes a vertex plus 4 an index for as long as the model lives. |
| `baseURL` | the URL of `source` | What relative buffer and image URLs in a `.gltf` resolve against. Bytes without a `baseURL` cannot fetch anything. |
| `fetch` | `globalThis.fetch` | Replaces `fetch` for every download: `source` when it is a URL, and the buffers and images the file names. Use it to refuse, rewrite or restrict URLs, above all from files you did not write. Every loader that downloads takes this option. |

Returns: a model object: `{ nodes, roots, meshes, materials, materialIds, animations, skins, lights, cameras, textures, source, engine }`. Pass the whole object to `scene.add()`.
Throws: `'load: <url> returned <status>'` for a failed download; `'load: this engine was destroyed'` if the engine is destroyed before or during the load; errors from the glTF parser (a malformed file, or one past the device's buffer limit). A failed load frees everything it had made.

```js
const helmet = await engine.load('helmet.glb', { retainGeometry: true });
scene.add(helmet);
```

Notes: every call allocates, the same file included. Free a model you no longer need with [`engine.unload`](#engine-unload). An image larger than the device's biggest texture is left out, with a warning in the console, and its material uses its factor alone. A relative `baseURL` is taken against the page, as a relative link in it would be.

<a id="engine-unload"></a>
### `engine.unload(asset)` → `void`

Frees the GPU memory of anything a load call returned: a model from [`load`](#engine-load), a texture from [`loadTexture`](#engine-loadtexture), a font from [`loadFont`](#engine-loadfont), a LUT from [`loadLUT`](#engine-loadlut), an environment from [`loadEnvironment`](#engine-loadenvironment), splats from [`loadSplats`](#engine-loadsplats), or a target from [`createTarget`](#engine-createtarget). Calling it twice does nothing.

Throws: `'unload: mesh "<name>" is still in a scene; remove it first'` for a model any scene still draws; `'unload: this asset was loaded by another engine'`; `'unload: this is not something load, loadTexture, loadFont, loadLUT, loadEnvironment, loadSplats or createTarget returned'`.

```js
scene.remove(helmetNode);
engine.unload(helmet);
```

Notes: only models are checked for use. A texture, font, LUT, environment or splats is freed at once, so stop using it first (remove its sprites, text or splat nodes, clear `engine.grading`, move scenes to another environment).

<a id="engine-loadtexture"></a>
### `engine.loadTexture(source, options)` → `Promise<Texture>`

Loads an image as a mipmapped GPU texture, for sprites (`scene.addSprite({ texture })`).

`source` is a URL string, a `Blob`, an `ImageBitmap`, or anything `createImageBitmap` takes.

| Option | Default | Meaning |
|---|---|---|
| `srgb` | `true` | The image holds colour. Pass `false` for data such as masks. |
| `pixelated` | `false` | For pixel art: enlarged, each texel stays a hard square, like CSS `image-rendering: pixelated`. |
| `label` | `'texture'` | GPU label, for error messages. |
| `fetch` | `globalThis.fetch` | Replaces `fetch` for downloading a URL `source`. |

Returns: `{ texture, view, width, height, pixelated }`.
Throws: `'loadTexture: <url> returned <status>'`; `'loadTexture: this engine was destroyed'`; decode errors from `createImageBitmap`.

```js
const pin = await engine.loadTexture('pin.png', { pixelated: true });
scene.addSprite({ texture: pin, position: [0, 1, 0] });
```

Notes: the pixels are used as authored, with no colour conversion or premultiplied alpha. Free with [`engine.unload`](#engine-unload).

<a id="engine-loadfont"></a>
### `engine.loadFont(css)` → `Promise<Font>`

Makes a font for `scene.addText` from any CSS font the page can use, rasterised as a distance field at the pixel size it names. Waits for a web font to load first.

Returns: a `Font`.
Throws: `'loadFont: the font needs a size in pixels, like '64px sans-serif'; got '<css>''` when `css` has no `px` size; `'loadFont: this engine was destroyed'`.

```js
const font = await engine.loadFont('64px Inter');
scene.addText({ font, text: 'Gate 3', size: 0.4 });
```

Notes: pick a size near the one the text is mostly seen at. It stays sharp above it and down to about an eighth of it. Free with [`engine.unload`](#engine-unload).

<a id="engine-loadlut"></a>
### `engine.loadLUT(source, options)` → `Promise<LUT>`

Loads a 3D colour lookup table from an Adobe `.cube` file, for [`engine.grading`](#engine-grading).

`source` is a URL, or the file's text. Text containing `LUT_3D_SIZE` is parsed directly; anything else is fetched.

| Option | Default | Meaning |
|---|---|---|
| `fetch` | `globalThis.fetch` | Replaces `fetch` for downloading `source`. |

Returns: `{ size, data, domainMin, domainMax, texture, view }`.
Throws: `'loadLUT: <url> returned <status>'`; `'loadLUT: this engine was destroyed'`; `'parseCube: a 1D LUT; only 3D LUTs are supported'`; `'parseCube: LUT_3D_SIZE must be 2 or more…'`; `'parseCube: a size N LUT has N³ entries, and this has …'`; `'parseCube: cannot read the line …'`; `'parseCube: DOMAIN_MAX must be above DOMAIN_MIN'`.

```js
engine.grading = { lut: await engine.loadLUT('film.cube') };
```

Notes: free with [`engine.unload`](#engine-unload) after taking it out of `engine.grading`.

<a id="engine-loadenvironment"></a>
### `engine.loadEnvironment(source, options)` → `Promise<Environment>`

Loads a Radiance `.hdr` panorama and bakes it into an [`Environment`](#environment): ambient light, reflections and background. Give it to [`engine.createScene`](#engine-createscene).

`source` is a URL, an `ArrayBuffer` or a `Uint8Array`.

| Option | Default | Meaning |
|---|---|---|
| `fetch` | `globalThis.fetch` | Replaces `fetch` for downloading `source`. |
| `size` | the power of two at or below map width / 4 | Edge of the baked cube, in texels. A power of two. Smaller costs less memory and gives a softer background; the lighting is the same. |
| `irradianceSize` | `32` | Edge of the diffuse-light cube. A power of two. |
| `prefilterMips` | `6` | Roughness levels of the reflection cube, capped by the cube's mip count. |
| `label` | `'env'` | GPU label prefix. |

Returns: an `Environment`.
Throws: `'loadEnvironment: <url> returned <status>'`; `'loadEnvironment: this engine was destroyed'`; the [`parseHDR`](#parsehdr) errors (including a map past the device's texture limit); the [`Environment`](#environment) errors.

```js
const studio = await engine.loadEnvironment('studio.hdr', { size: 256 });
const scene = engine.createScene({ environment: studio });
```

<a id="engine-loadsplats"></a>
### `engine.loadSplats(source, { fetch })` → `Promise<Splats>`

Loads a Gaussian splat capture for [`scene.addSplats`](#scene-addsplats). It reads four formats, told apart by their bytes, not their names:

| Format | What it is |
|---|---|
| `.ply` | As 3D Gaussian Splatting training writes it: binary, with `f_dc`, `opacity`, `scale` and `rot` properties. |
| `.splat` | 32 bytes a splat, the compact web format. No harmonics. |
| `.spz` | Niantic's, versions 1 to 3: gzipped and quantised, about a tenth of a `.ply`. Unpacked by the browser's own `DecompressionStream`, and turned from its y-up axes to the `.ply`'s, so it lands as its `.ply` would. |
| `.sog` | PlayCanvas's, version 2: a zip of lossless WebP images and `meta.json`, about a fifteenth of a `.ply`. A loose `meta.json` and its images are not read: zip them. |

A capture with higher spherical harmonics (a `.ply`'s `f_rest_*`, and the same in a `.spz` or `.sog`, to degree 3) keeps them, so colour changes with the view as it did in the photographs: a sheen, a reflection. They are kept as half floats, 20 to 92 bytes a splat by degree.

`source` is a URL, an `ArrayBuffer` or a `Uint8Array`. `fetch` replaces `fetch` for downloading `source`.

Returns: `{ count, min, max, degree }`: how many splats, the corners of the box around their centres in the capture's units, and the degree of its harmonics, 0 to 3. Free it with [`engine.unload`](#engine-unload) once no scene draws it.
Throws: `'loadSplats: <url> returned <status>'`; `'splats: a .ply in '<format>' format; only binary_little_endian is read'`; `'splats: this .ply is not a splat capture: no <properties>'`; `'splats: the .ply is cut short: …'`; `'splats: not a .ply, and <n> bytes is not a whole number of 32-byte .splat records'`; `'splats: splat <i> has a position that is not a number'`; `'splats: a compressed .ply (from SuperSplat) is not read; …'`; `'splats: a .spz of version 4, compressed with zstd, which browsers cannot unpack; …'`; `'splats: gzipped, but not a .spz'`; `'splats: the .spz is cut short: …'`; `'splats: a .sog of version <n>; version 2 is read …'`; `'splats: the .sog has no <file>'`; `'splats: <image> has <n> pixels, for <count> splats'`; `'splats: a zip64 .sog is not read'`; `'splats: <n> splats need <bytes> bytes in one buffer, past this device's <limit>; …'`.

```js
const room = await engine.loadSplats('room.ply');
```

Notes:
- Every call allocates, the same file included. One `Splats` can be drawn by many nodes, in many scenes.
- A `.spz` is read as Niantic's spec has it: stored y up (RUB), and turned to the `.ply`'s axes, so it needs the same half turn about x a `.ply` does. Some tools write a `.spz` without that conversion, in the `.ply`'s axes as they are -- PlayCanvas's `splat-transform` does -- and the header does not say which. Such a capture comes out the other way up: it stands upright without the half turn.
- A `.spz` saved antialiased (Mip-Splatting) is drawn as any other; its antialiasing is not applied.
- A `.spz` with harmonics of degree 4 keeps the first three degrees.

<a id="engine-run"></a>
### `engine.run(scene, camera, { update, frame, hud })` → `void`

Starts the frame loop on `requestAnimationFrame`, drawing `scene` through `camera`: a `Camera` for 3D, or a `Camera2D` for a 2D view. Simulation runs at a fixed rate and drawing at the display's rate.

| Option | Default | Meaning |
|---|---|---|
| `update` | none | `(dt, elapsed) => {}` at a fixed 60 Hz (`dt` is `engine.clock.fixedDt`), zero or more times per drawn frame. Put simulation here. |
| `frame` | none | `(alpha, clock) => {}` once per frame, before drawing. `alpha` in [0, 1) is how far this frame sits between the last two simulation steps; interpolate with it. |
| `hud` | `null` | `{ scene, camera }`: a 2D scene drawn over every frame. `camera` must be a `Camera2D`; a plain one is used if left out. See [HUD](#view2d-hud). |

Throws: `'run: this engine was destroyed'`; `'run: takes (scene, camera, { update, frame, hud }), the scene first, from createScene()'`; `'run: already running; call stop() first'`; `'run: hud needs { scene, camera }, the scene from createScene()'`; `'run: the hud's camera must be a Camera2D'`; `'run: overlay is now hud, with the same { scene, camera }'`.

```js
const orbit = new OrbitController(camera, canvas);
engine.run(scene, camera, {
  update: (dt) => { x += speed * dt; ship.setPosition(x, 0, 0); },
  frame: (alpha, clock) => orbit.update(clock.realDelta),
  hud: { scene: hud },
});
```

Notes: animations and particles advance on real time each frame, before `update`. Without `update`, `engine.clock.elapsed` does not advance. With [`engine.onDemand`](#engine-ondemand) on, a frame where nothing changed is skipped. If the canvas leaves the document or the device is lost, the loop destroys the engine.

<a id="engine-stop"></a>
### `engine.stop()` → `void`

Stops the loop that [`engine.run`](#engine-run) started. Call `run` again to restart it. Safe to call when not running.

```js
engine.stop();
```

<a id="engine-invalidate"></a>
### `engine.invalidate()` → `void`

Makes the next frame of [`engine.run`](#engine-run) draw even if nothing seems to have changed. Only matters while [`engine.onDemand`](#engine-ondemand) is on.

```js
paint(canvasTexture);          // your own GPU writes: run cannot see them
engine.invalidate();
```

Notes: `run` already notices changes to the scene (assigning `scene.environment` included), the camera, the canvas size, debug lines and every setting under [Renderer settings](#renderer-settings), the live [`shadows`](#renderer-shadows) fields included. Call this for changes it cannot see, such as drawing into a texture a sprite shows.

<a id="engine-renderframe"></a>
### `engine.renderFrame(scene, camera, { hud, target })` → `void`

Draws one frame now. Use it when you own the loop instead of calling [`engine.run`](#engine-run), or to draw into a target.

| Option | Default | Meaning |
|---|---|---|
| `hud` | `null` | `{ scene, camera }`, as `run` takes it. |
| `target` | `null` | A target from [`engine.createTarget`](#engine-createtarget) to draw into, in place of the canvas. |

Throws: `'renderFrame: this engine was destroyed'`; `"renderFrame: target must be one this engine's createTarget made"`; `'renderFrame: that target was unloaded'`; `'Renderer: the scene has no environment'`; the `hud` errors listed under `run`, named `renderFrame`; setting errors such as a bad [`renderer.fog`](#renderer-fog), [`renderer.dof`](#renderer-dof) or [`engine.grading`](#engine-grading).

```js
function tick() {
  engine.renderFrame(scene, camera);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
```

Notes: it does not move the scene on, skip idle frames, or update `engine.fps`. Call [`scene.advance(dt)`](#scene-advance) yourself.

<a id="engine-createtarget"></a>
### `engine.createTarget({ size, pixelated, label })` → `Promise<Target>`

A texture to draw a scene into, with [`renderFrame`](#engine-renderframe)'s `target`. A sprite shows it as it would a loaded image: a minimap, a screen in a room, a split screen, or pixel art drawn small and shown large. A 2D or a 3D scene can be drawn into it.

| Option | Default | Meaning |
|---|---|---|
| `size` | required | `[width, height]` in the target's own pixels. |
| `pixelated` | `false` | Shown enlarged with hard edges, as [`loadTexture`](#engine-loadtexture)'s is. |
| `label` | `'target'` | GPU label, for error messages. |

Returns: `{ texture, view, width, height, pixelated }`, which a sprite takes as its `texture`.

Throws: `'createTarget: size must be [width, height], whole pixels from 1 to N, …'`; `'createTarget: this engine was destroyed'`.

```js
const map = await engine.createTarget({ size: [160, 160] });
hud.addSprite({ texture: map, pivot: [0, 0], position: [16, 16] });
// whenever the map should change -- every frame, or when something on it moves:
engine.renderFrame(world, overhead, { target: map });
```

Notes:
- A Camera2D drawing into a target counts its pixels one to a unit at zoom 1: its pixel ratio is 1.
- Colours are kept as the canvas keeps them: a 3D scene is tonemapped, and a sprite shows the target's pixels as it would a loaded image's.
- Drawing into a target makes the next frame of `run` draw, even with [`onDemand`](#engine-ondemand) on, since what shows the target has changed.
- There are no debug lines in a target, and no mipmaps: a target drawn much smaller than its size shimmers.
- Particle emitters draw in one scene at a time: those of a scene drawn into a target take over from the canvas scene's for that frame.
- Free it with [`engine.unload`](#engine-unload).

<a id="engine-captureprobes"></a>
### `engine.captureProbes(scene, probeNodes)` → `Promise<void>`

Renders each reflection probe's view of the scene (six faces) and prefilters it, so nearby surfaces reflect it. Captures all of the scene's probes, or only the probe nodes you pass.

Throws: `'captureProbes: this engine was destroyed'`; `'captureProbes: this node is not a reflection probe'`; `'captureProbes: the scene has no environment'`.

```js
const hall = scene.addProbe({ position: [0, 2, 0], size: [10, 4, 16] });
await engine.captureProbes(scene);
// later, after the hall's contents change:
await engine.captureProbes(scene, [hall]);
```

Notes: this costs six scene renders per probe, so do it at load time or after what a probe sees has changed, not every frame. A probe that moves shows nothing until captured again.

<a id="engine-destroy"></a>
### `engine.destroy()` → `void`

Stops the loop and frees everything the engine owns: workers, renderer, its default environment and the GPU device. Calling it again does nothing.

```js
engine.destroy();
```

Notes: afterwards every method that uses the GPU throws `'<method>: this engine was destroyed'`. Environments from [`engine.loadEnvironment`](#engine-loadenvironment) die with the device. The engine also destroys itself when the device is lost or the canvas leaves the document.

<a id="engine-grading"></a>
### `engine.grading` → `object | null`

Colour grading, read and set at any time. The same object as [`post.grading`](#post-grading). Setting `undefined` or `null` turns it off.

| Field | Default | Meaning |
|---|---|---|
| `whiteBalance` | off | Colour temperature in kelvin that comes out white, 1667 to 25000. `3200` undoes tungsten orange; `10000` undoes overcast blue. |
| `contrast` | `1` | Contrast about middle grey, in stops. Must be above 0. |
| `saturation` | `1` | `0` is grey, `1` unchanged. Must be 0 or more. |
| `lut` | none | A LUT from [`engine.loadLUT`](#engine-loadlut), applied after the tonemap. |

Throws (on the next frame): `'whiteBalance: a colour temperature from 1667 K to 25000 K, got …'`; `'grading: contrast must be positive, got …'`; `'grading: saturation must be 0 or more, got …'`.

```js
engine.grading = { whiteBalance: 5000, contrast: 1.1, saturation: 0.9 };
engine.grading = null;
```

Notes: grading is part of the 3D post chain. A 2D view (`Camera2D`) does not use it.

<a id="engine-stats"></a>
### `engine.stats` → `object`

Counts from the last frame drawn. Read only. The first five fields start at 0; the others appear once a frame of that kind has set them.

| Field | Set by | Meaning |
|---|---|---|
| `renderables` | 3D | Meshes in the scene. |
| `draws` | 3D | Indirect draw batches (one per mesh part and material). |
| `recomposed` | 3D, 2D | Transforms recomputed this frame. |
| `transparent` | 3D | Blended objects this frame. |
| `transparentDraws` | 3D | Draw calls for blended objects. |
| `shadowViews` | 3D | Point and spot shadow views. |
| `shadowViewsDrawn` | 3D | Of those, how many were redrawn (the rest were cached). |
| `cascadesDrawn` | 3D | Directional shadow cascades redrawn. |
| `sprites2D` | 2D | Sprites and glyphs in the 2D view. |
| `sprites2DWritten` | 2D | Of those, how many were re-uploaded. |
| `tiles2DWritten` | 2D | Tile-map tiles re-uploaded. |
| `emitters` | 2D | Particle emitters drawn in the 2D view. |
| `hudSprites` | HUD | Sprites and glyphs in the HUD. |
| `hudSpritesWritten` | HUD | Of those, how many were re-uploaded. |
| `splats` | 3D | Splats in the clouds drawn, before culling. |

```js
console.log(`${engine.stats.draws} draws, ${engine.stats.renderables} meshes`);
```

Notes: CPU time per phase is in `engine.renderer.timing` (`transforms`, `upload`, `graph`, `encode`, `total`, in milliseconds). There is no visible-object count: culling runs on the GPU and is never read back.

<a id="engine-debug"></a>
### `engine.debug` → `DebugLines`

Lines drawn for one frame, for debugging. See [Debug lines](#debug-lines).

```js
engine.debug.axes([0, 0, 0], 1);
```

<a id="engine-fps"></a>
### `engine.fps` → `number`

Frames per second of the [`engine.run`](#engine-run) loop, updated every half second. `0` until `run` has gone half a second. Read only.

Notes: skipped idle frames count, so this is the loop's rate, not the number of frames drawn.

<a id="engine-skippedframes"></a>
### `engine.skippedFrames` → `number`

How many frames `run` skipped because nothing had changed. Read it to check that a still scene is idle.

<a id="engine-ondemand"></a>
### `engine.onDemand` → `boolean`

Default `true` (from the `onDemand` option). When on, `run` skips a frame that would draw exactly what the last one drew, so a still scene costs the GPU nothing. Set `false` to draw every frame. See [`engine.invalidate`](#engine-invalidate).

```js
engine.onDemand = false;
```

<a id="engine-clock"></a>
### `engine.clock` → `Clock`

The fixed-step clock `run` drives.

| Field | Meaning |
|---|---|
| `fixedDt` | Simulation step in seconds, `1 / 60`. Writable: change it to change the `update` rate. |
| `elapsed` | Simulated seconds so far. Only advances when `run` has an `update`. |
| `realDelta` | Real seconds since the last frame, capped at 0.25. |
| `alpha` | How far between the last two steps this frame is, in [0, 1). |
| `frame` | Frames begun. |

```js
engine.run(scene, camera, { frame: (alpha, clock) => orbit.update(clock.realDelta) });
```

<a id="engine-renderer"></a>
### `engine.renderer` → `Renderer`

The renderer. Its live settings are listed under [Renderer settings](#renderer-settings).

<a id="engine-environment"></a>
### `engine.environment` → `Environment`

The default environment, made from the `environment` option of [`Winding.create`](#winding-create). Scenes use it unless given another. Freed by [`engine.destroy`](#engine-destroy).

<a id="engine-gpu"></a>
### `engine.gpu` → `Device`

The GPU layer. `engine.gpu.device` is the raw `GPUDevice`; `engine.gpu.width` and `engine.gpu.height` are the canvas size in pixels.

<a id="engine-gpu-pixelratio"></a>
### `engine.gpu.pixelRatio` → `number`

Canvas pixels per CSS pixel: 2 on most phones, 1.5 at 150% Windows scaling. Measured from the canvas, so it is exact after rounding. Falls back to `devicePixelRatio` while the canvas has no CSS width. Read only.

```js
const x = event.offsetX * engine.gpu.pixelRatio;
```

<a id="engine-gpu-readpixels"></a>
### `engine.gpu.readPixels({ x, y, width, height })` → `Promise<Uint8Array>`

The canvas's pixels as RGBA bytes, row by row from the top-left: the whole canvas, or the region given in canvas pixels. Given only `x` and `y`, the region runs to the canvas's far edges. For screenshots and tests.

```js
engine.renderFrame(scene, camera);
const pixels = await engine.gpu.readPixels();
const at = (x, y) => pixels.subarray((y * engine.gpu.width + x) * 4, (y * engine.gpu.width + x) * 4 + 4);
```

Notes: call it straight after `renderFrame`, with nothing awaited between. The browser shows a frame and hands the canvas a fresh, blank one as soon as the page waits for anything, so a read after an `await` returns zeros. For the same reason, read once a frame and index into the result, rather than reading pixel by pixel.

<a id="renderer-settings"></a>
## Renderer settings

Plain fields on `engine.renderer` and `engine.renderer.post`. Set them at any time; the next frame uses them, and [`engine.run`](#engine-run) notices the change. They apply to 3D views. A 2D view (`Camera2D`) draws without them.

<a id="renderer-exposure"></a>
### `engine.renderer.exposure` → `number`

Default `1` (the `exposure` option). Multiplies the scene's linear colour before the tonemap. `2` is one stop brighter. With [`autoExposure`](#renderer-autoexposure) on, it applies on top, as exposure compensation.

```js
engine.renderer.exposure = 0.5;
```

<a id="renderer-autoexposure"></a>
### `engine.renderer.autoExposure` → `object | null`

Default `null` (off; the `autoExposure` option). Exposure chosen from the image, as an eye or a camera adapts: it opens up in a dark room and stops down outside, easing from one to the other over time. Assign `true` for the defaults, or an object; `null` turns it off.

| Field | Default | Meaning |
|---|---|---|
| `min` | `-8` | The lowest exposure it will choose, in stops. |
| `max` | `8` | The highest, in stops. At least `min`. |
| `brighten` | `3` | How fast the image brightens, going somewhere darker, in stops a second. Above 0. |
| `darken` | `1` | How fast it darkens, going somewhere brighter, in stops a second. Above 0. |

It measures the scene each frame, before exposure, and puts its average brightness at middle grey, leaving out the darkest and brightest tenth and black: a lamp in shot or a dark corner does not swing it. [`exposure`](#renderer-exposure) still applies on top, so `exposure = 2` keeps everything a stop brighter than auto exposure would.

Throws (on the next frame): `'autoExposure: true, { min, max, brighten, darken }, or null to turn it off, got …'`; `'autoExposure: min must be a finite number of stops, got …'`, and the same for `max`; `'autoExposure: min must be at most max, got …'`; `'autoExposure: brighten must be a positive number of stops a second, got …'`, and the same for `darken`.

```js
engine.renderer.autoExposure = true;
engine.renderer.autoExposure = { min: -2, max: 4, darken: 2 };
engine.renderer.exposure = 1.5;   // half a stop over what auto exposure picks
```

Notes:
- The first frame after it is turned on takes the right exposure at once; after that it eases.
- It adapts only on frames drawn to the canvas. A frame drawn into a [target](#engine-createtarget) uses the canvas's exposure and does not move it.
- With [`onDemand`](#engine-ondemand), `run` keeps drawing while the exposure is still easing, even if nothing moved, and rests once it has settled.
- It costs two small compute passes, about 0.1 ms at 1280x720 on integrated graphics (Intel Iris Xe).

<a id="renderer-resolution"></a>
### `engine.renderer.resolution` → `number`

Default `1`. The share of the canvas's width and height the 3D view is drawn at, from `0.5` to `1`. NVIDIA Image Scaling then brings it up to the canvas size and sharpens it. At `0.75` the scene shades 44% fewer pixels, so on a GPU-bound scene the frame is cheaper. The scaler itself costs about 2 ms at 1920x1080 on integrated graphics (Intel Iris Xe), and less on a discrete GPU.

Throws (on the next frame): `'resolution must be from 0.5 to 1, …'`.

```js
engine.renderer.resolution = 0.75;
```

Notes: only frames drawn to the canvas are scaled. A [target](#engine-createtarget) keeps the size it was made at. The HUD and the stats overlay are drawn at the canvas's full resolution, after scaling, so text stays crisp. [Debug lines](#debug-lines) are drawn into the 3D view, so they are scaled with it. Sprites sized in pixels keep their size on screen. The first time a lower resolution is set, the scaler builds in the background, and frames are stretched without sharpening until it is ready, as with [`ao`](#renderer-ao).

<a id="renderer-fog"></a>
### `engine.renderer.fog` → `object | null`

Default `null` (no fog). Exponential height fog. Its colour is not chosen: it scatters the environment's light and the directional lights' light, times `albedo`.

| Field | Default | Meaning |
|---|---|---|
| `visibility` | required | Metres at which a dark object fades to 2% contrast, measured at `height`. Must be above 0. |
| `height` | `0` | Height where the density matches `visibility`. |
| `scaleHeight` | uniform | Metres over which the density falls by a factor of e going up. Leave out for fog that is the same at every height. |
| `albedo` | `[1, 1, 1]` | Share of light the fog scatters rather than absorbs, per channel. Three numbers, 0 or more. |

Throws (on the next frame): `'fog: visibility must be a positive number of metres…'`; `'fog: height must be a finite number…'`; `'fog: scaleHeight must be positive, or left out for uniform fog…'`; `'fog: albedo must be three non-negative numbers…'`.

```js
engine.renderer.fog = { visibility: 200, scaleHeight: 20 };
```

Notes: shadows inside the fog are not modelled.

<a id="renderer-dof"></a>
### `engine.renderer.dof` → `object | null`

Default `null` (off). Depth of field from a real lens: the focal length comes from the camera's `fovY` on a sensor of `sensorHeight`.

| Field | Default | Meaning |
|---|---|---|
| `focusDistance` | required | Distance in focus, in world units (metres). Must be past the lens's focal length. |
| `fStop` | required | Aperture. Lower is blurrier. Must be above 0. |
| `sensorHeight` | `0.024` | Sensor height in metres (full frame). |

Throws (on the next frame): `'dof: focusDistance must be positive…'`; `'dof: fStop must be positive…'`; `'dof: sensorHeight must be positive…'`; `'dof: focusDistance must be past the lens's focal length…'`.

```js
engine.renderer.dof = { focusDistance: 3, fStop: 1.8 };
```

Notes: skipped for an orthographic camera.

<a id="renderer-ao"></a>
### `engine.renderer.ao` → `{ radius } | null`

Default `null` (off; the `ao` option). Ambient occlusion. `radius` is how far occlusion reaches, in world units. A `null` radius means a thirty-second of the scene's bounding radius, worked out each frame. Assign `{ radius }` to turn it on and `null` to turn it off.

```js
engine.renderer.ao = { radius: 0.5 };
engine.renderer.ao.radius = 1;
engine.renderer.ao = null;
```

Throws (on the next frame): `'ao: radius must be positive, or null to fit the scene, …'`.

Notes: the first time it is turned on, its pipelines build in the background. Frames draw without it until they are ready, a fraction of a second. After that, switching costs nothing. The same holds for [`oit`](#renderer-oit) and [`post.antialias`](#post-antialias).

<a id="renderer-taa"></a>
### `engine.renderer.taa` → `boolean`

Default `false` (the `taa` option). Temporal antialiasing, in place of FXAA on frames drawn to the canvas. The camera is moved a fraction of a pixel each frame, a different fraction each time, and each frame is blended with the ones before it, so over sixteen frames every pixel has been sampled at sixteen points inside it: close to supersampling, spread over time.

```js
engine.renderer.taa = true;
```

What it does better than FXAA:
- Edges come out much nearer a supersampled frame: on Sponza at 1280x720, the error at edges against 16x supersampling fell from 9.3 to 6.2 with the camera still, and from 9.8 to 9.3 panning.
- Thin and shiny things stop crawling as the camera moves, which FXAA cannot fix: it sees one frame at a time.
- The grain of soft shadows ([`size`](#scene-addlight)) is turned a little each frame and averages away.

What it costs:
- About 1.2 ms a frame more than FXAA at 1280x720 on integrated graphics (Intel Iris Xe): 0.6 for its own pass, the rest for material textures read half a mip sharper, which keeps surfaces from going soft.
- Surfaces inside an object are still a little softer than with FXAA (2.7 against 2.2 in that measure).
- Two images of history, 8 bytes a pixel each: about 15 MB at 1280x720.

Notes:
- Where a surface was last frame is worked out from the depth buffer and the camera, so it is exact for everything that the camera's own movement moves. Something that moves by itself is kept from smearing by holding the history to the colours around each pixel in this frame; on Sponza a box crossing the view at 7 pixels a frame left no trail. Particles, splats and sprites are treated the same way.
- A still view keeps drawing for 24 frames while the picture settles; then [`run`](#engine-run) rests as it does without TAA.
- Only frames drawn to the canvas use it: a [target](#engine-createtarget) keeps FXAA.
- The first time it is turned on, its pass builds in the background, and frames use FXAA until it is ready.

<a id="renderer-oit"></a>
### `engine.renderer.oit` → `boolean`

Default `false` (the `oit` option). Weighted-blended order-independent transparency for blended materials, in place of back-to-front sorting. Good for smoke and foliage; sorting stays exact for separate glass objects.

```js
engine.renderer.oit = true;
```

Notes: builds in the background the first time, like [`ao`](#renderer-ao).

<a id="renderer-softshadows"></a>
### `engine.renderer.softShadows` → `boolean`

Default `true`. Whether lights given a [`size`](#scene-addlight) cast soft shadows. Set it `false` to draw every light's shadow with the plain edge -- on a slow device, say, or from a graphics setting -- without changing any light: each keeps its size, and soft shadows come back when it is set `true` again. Off, the soft-shadow code is compiled out of the shaders, so it costs nothing at all.

```js
engine.renderer.softShadows = false;
```

<a id="renderer-shadows"></a>
### `engine.renderer.shadows` → `ShadowMaps`

The shadow maps. Three of the [`shadows` options](#winding-create) can change at any time as fields here: `lambda`, `casterExtent` and `normalBias`.

```js
engine.renderer.shadows.normalBias = 2.5;   // less acne
```

Notes: `size`, `cascades`, `localSize` and the two depth biases are fixed at creation. The textures and pipelines are built with them.

<a id="renderer-skybox"></a>
### `engine.renderer.skybox` → `boolean`

Default `true`. Whether the environment is drawn as the background. Turning it off shows black behind the scene and changes nothing about lighting.

```js
engine.renderer.skybox = false;
```

<a id="renderer-shadowdistance"></a>
### `engine.renderer.shadowDistance` → `number | null`

Default `null`. How far from the camera directional shadows reach, in world units. `null` fits the whole scene: the distance to its farthest point, rounded up to a power of two so shadows stay stable. Set a number to spend the shadow map's resolution on a shorter range.

```js
engine.renderer.shadowDistance = 40;
```

<a id="renderer-lightdistance"></a>
### `engine.renderer.lightDistance` → `number | null`

Default `null`. How far along the view point and spot lights are resolved, in world units. `null` uses the scene's farthest depth from the camera each frame. Set a number to pin it.

```js
engine.renderer.lightDistance = 100;
```

<a id="post-threshold"></a>
### `engine.renderer.post.threshold` → `number`

Default `1.2`. Linear brightness (brightest channel) where bloom starts. Only light brighter than white blooms at the default.

<a id="post-knee"></a>
### `engine.renderer.post.knee` → `number`

Default `0.6`. Width of the soft ramp around `threshold`, as a fraction of it, so bloom fades in instead of popping on.

<a id="post-filterradius"></a>
### `engine.renderer.post.filterRadius` → `number`

Default `1`. Blur radius of each bloom upsample, in texels. Larger is a wider, softer halo.

<a id="post-strength"></a>
### `engine.renderer.post.strength` → `number`

Default `0.06`. How much of the image is bloom, 0 to 1 (clamped). Bloom is mixed in, not added, so raising it moves light into the halo without brightening the image.

```js
engine.renderer.post.strength = 0.15;
```

<a id="post-levels"></a>
### `engine.renderer.post.levels` → `number`

Default `5` (the `post.levels` option). How many times bloom halves the image, 1 to 6. More is a wider halo. Values outside that are clamped each frame, and a frame uses fewer if the canvas is too small.

`engine.renderer.post.levelsDrawn` is the count the last frame drew. Read only.

```js
const engine = await Winding.create(canvas, { post: { levels: 6 } });
engine.renderer.post.levels = 3;
```

<a id="post-antialias"></a>
### `engine.renderer.post.antialias` → `boolean`

Default `true` (the `antialias` option). FXAA after the tonemap. Off writes the tonemapped image straight to the screen.

```js
engine.renderer.post.antialias = false;
```

Notes: an engine created with `antialias: false` builds FXAA the first time it is turned on. Frames draw without it until it is ready, as with [`ao`](#renderer-ao).

<a id="post-grading"></a>
### `engine.renderer.post.grading` → `object | null`

Default `null`. The colour grading object. Same as [`engine.grading`](#engine-grading), which is the usual way to set it.

<a id="debug-lines"></a>
## Debug lines

`engine.debug` draws lines for exactly one frame. Call it every frame you want them seen; nothing needs removing. Every call returns `engine.debug`, so calls chain.

In 3D, lines are drawn after the tonemap, straight onto the screen, with no exposure, bloom or antialiasing. Colours are linear, like every 3D colour, but clamped to 0–1 here (they cannot glow). The scene's depth hides them behind geometry unless [`depthTest`](#debug-depthtest) is off.

Through a `Camera2D` they are drawn over the 2D view with no depth, and colours are sRGB, like CSS. Points can be `[x, y]`.

Colours are `[r, g, b]`; a fourth value is ignored. The default is white. Lines are one pixel wide.

<a id="debug-line"></a>
### `engine.debug.line(from, to, color)` → `DebugLines`

A segment from `from` to `to`. Points are `[x, y, z]`, or `[x, y]` at z = 0.

```js
engine.debug.line([0, 0, 0], [0, 2, 0], [1, 0, 0]);
```

<a id="debug-box"></a>
### `engine.debug.box(min, max, color)` → `DebugLines`

With `[x, y, z]` corners, the twelve edges of an axis-aligned box. With `[x, y]` corners, a rectangle's four edges.

```js
engine.debug.box([-1, 0, -1], [1, 2, 1], [1, 1, 0]);
engine.debug.box([10, 10], [110, 60], [1, 0.5, 0]);   // 2D rectangle
```

<a id="debug-sphere"></a>
### `engine.debug.sphere(center, radius, color)` → `DebugLines`

Three circles, one around each axis, at `center` (`[x, y, z]`, or `[x, y]` at z = 0).

```js
engine.debug.sphere([0, 1, 0], 0.5, [0, 1, 1]);
```

<a id="debug-circle"></a>
### `engine.debug.circle(center, radius, color)` → `DebugLines`

A circle in the x-y plane (a 2D view's plane) around `center`, `[x, y]` or `[x, y, z]`.

```js
engine.debug.circle([160, 120], 24, [0, 1, 0]);
```

<a id="debug-axes"></a>
### `engine.debug.axes(origin, size)` → `DebugLines`

The world axes at `origin` (`[x, y, z]`, or `[x, y]` at z = 0): x red, y green, z blue, each `size` long (default `1`).

```js
engine.debug.axes([0, 0, 0], 0.5);
```

<a id="debug-depthtest"></a>
### `engine.debug.depthTest` → `boolean`

Default `true`. Whether scene geometry hides lines behind it. Set `false` to draw them on top. Has no effect in a 2D view.

```js
engine.debug.depthTest = false;
```

<a id="scene"></a>
## Scene

A scene holds everything in the world: models, lights, emitters, decals, probes, and the 2D kinds. Everything in it is a [`Node`](#node). Every kind follows one pattern: `scene.addX(options)` returns a Node, `scene.setX(node, changes)` changes some options, `scene.xOf(node)` returns a copy of the options or `null`, and [`scene.remove(node)`](#scene-remove) or [`node.destroy()`](#node-destroy) removes it.

<a id="scene-constructor"></a>
### `new Scene(options)` → `Scene`

Makes an empty scene with no lights. You normally get one from [`engine.createScene(options)`](#engine-createscene), which passes these options through and sets [`scene.environment`](#scene-environment).

| Option | Default | Meaning |
|---|---|---|
| `capacity` | `4096` | Nodes to make room for up front. |
| `renderableCapacity` | `capacity` | Mesh primitives to make room for up front. |
| `lightCapacity` | `256` | Point and spot lights to make room for up front. |

All three are starting sizes. Each store grows when it fills.

Throws: if the node count ever needs more than 2^24 slots (`HandleAllocator: ... exceeds the 24-bit index space`).

```js
const scene = engine.createScene({ capacity: 20000 });
```

<a id="scene-environment"></a>
### `scene.environment`

The `Environment` that gives the scene its ambient light, reflections and background. [`engine.createScene`](#engine-createscene) sets it: to the engine's own environment, or to the one you pass as `createScene({ environment })`. A scene with no lights is lit by its environment alone.

<a id="scene-createnode"></a>
### `scene.createNode(options)` → `Node`

Makes an empty node. Use it to group things you move together, or as a mount point for a camera or light.

| Option | Default | Meaning |
|---|---|---|
| `parent` | `null` | Node to attach it to. `null` puts it at the scene root. |

Returns: the new Node, at the origin of its parent's space.

Throws: `'createNode: parent was removed'`; `'createNode: parent is a node of another scene'`. Every `addX` that takes a `parent` checks it the same way, named for itself.

```js
const mount = scene.createNode({ parent: car });
mount.setPosition(0, 2, 6);
```

<a id="scene-node"></a>
### `scene.node(entity)` → `Node`

Wraps an entity handle (a number, as in `node.entity`) in a Node. It does not check that the entity is alive; check [`node.alive`](#node-alive).

<a id="scene-childrenof"></a>
### `scene.childrenOf(node)` → `Node[]`

The node's direct children, oldest first. Returns `[]` for a node that is no longer alive. Same as [`node.children()`](#node-children).

<a id="scene-add"></a>
### `scene.add(asset, options)` → `Node`

Puts a loaded glTF model into the scene. `asset` is what [`engine.load`](#engine-load) returned. Synchronous: all the slow work already happened in `engine.load`. Add the same asset as many times as you like; each copy is independent.

| Option | Default | Meaning |
|---|---|---|
| `parent` | `null` | Node to attach the model to. |

Returns: the model's root Node. A file with several root nodes gets one extra wrapper node, so you always get one Node to move the whole model by. Only this Node can play the model's clips (see [`node.play`](#node-play)).

Throws:
- `Scene.add: this asset was unloaded; load it again`, if the asset was passed to `engine.unload`.
- If the file's node graph is not a tree (a node with more than one parent).
- If a skin names a joint node that is not in the file's default scene.

Nothing is left in the scene when it throws.

```js
const robot = scene.add(await engine.load('robot.glb'));
robot.setPosition(0, 0, -3);
```

Notes: lights in the file become lights in the scene (see [Lights](#lights)); a directional one casts shadows by default. Cameras in the file become [`Camera`](#camera) objects that already follow their nodes, pushed onto `scene.cameras` in the order added.

<a id="scene-remove"></a>
### `scene.remove(node)`

Removes a node and everything under it: meshes, lights, sprites, emitters, decals, probes, text, and every child. Does nothing for a node that is already gone.

```js
scene.remove(robot);
```

Notes: a camera imported with the model is dropped from `scene.cameras`. A camera you made that [follows](#camera-follow) a removed node stops following and stays where it was. Same as [`node.destroy()`](#node-destroy).

<a id="scene-update"></a>
### `scene.update()` → `number`

Recomputes world matrices from the positions, rotations and scales you set. [`engine.run`](#engine-run) calls it every frame; call it yourself only if you need world positions (for example [`node.worldPosition`](#node-worldposition)) before the next frame.

Returns: how many transforms were recomputed.

<a id="scene-advance"></a>
### `scene.advance(dt)` → `void`

Moves the scene on by `dt` seconds: every playing clip, every sprite animation, and every emitter's clock (the GPU catches the particles up on the next frame). [`engine.run`](#engine-run) calls it once per frame with the real elapsed time, for the scene and its HUD. Call it yourself only when you drive frames with [`renderFrame`](#engine-renderframe). A negative or `NaN` `dt` moves nothing.

```js
scene.advance(clock.realDelta);
engine.renderFrame(scene, camera);
```

<a id="scene-bounds"></a>
### `scene.bounds(outMin, outMax)` → `boolean`

Writes the world-space box around every mesh and splat cloud in the scene into `outMin` and `outMax` (each a 3-element array). Brings transforms up to date first.

Returns: `true`, or `false` if the scene has no meshes or splats, in which case `outMin` and `outMax` are left alone.

```js
const min = vec3Create(), max = vec3Create();
if (scene.bounds(min, max)) console.log(min, max);
```

Notes: meshes from [`scene.add`](#scene-add) count, and splat clouds from [`scene.addSplats`](#scene-addsplats), by the box around their centres. Sprites, text, particles and lights do not.

<a id="scene-frame"></a>
### `scene.frame(camera, options)` → `boolean`

Moves a camera so the whole scene fills the view, keeping the direction it already looks from. It is [`scene.bounds`](#scene-bounds) followed by [`camera.frameBounds`](#camera-framebounds).

| Option | Default | Meaning |
|---|---|---|
| `margin` | `1` | Multiplies the fitting distance. `1` fits exactly; `1.2` leaves more room. |
| `aspect` | `camera.aspect` | Width over height of the view. |

Returns: `true`, or `false` if the scene has no meshes or splats, having moved nothing.

```js
scene.frame(camera, { aspect: canvas.clientWidth / canvas.clientHeight });
```

Notes: `camera.aspect` is `1` until the camera's first [`update`](#camera-update), so pass `aspect` when you frame during setup. With an `OrbitController`, frame through the controller's `frameBounds` instead; it owns the camera's position.

<a id="scene-raycast"></a>
### `scene.raycast(origin, direction, options)` → `{ node, renderable, distance } | null`

Finds the nearest mesh a ray hits. Brings transforms and bounds up to date first, so the answer is right even if nothing has rendered since the last move.

| Option | Default | Meaning |
|---|---|---|
| `maxDistance` | `Infinity` | Ignore hits this far or further. |

`origin` and `direction` are world-space 3-element arrays. Normalize `direction`, or `distance` comes back scaled by its length.

Returns: `{ node, renderable, distance }` for the nearest hit, or `null`. `node` is the Node that carries the mesh. `renderable` is an index into the scene's mesh list and changes when anything is removed, so do not keep it.

Throws: if `origin` or `direction` has a non-finite component (`raycast origin: non-finite at ...`).

```js
const hit = scene.raycast([0, 5, 0], [0, -1, 0], { maxDistance: 20 });
if (hit) console.log('ground at', 5 - hit.distance);
```

Notes: the test is exact, triangle by triangle and as skinned or morphed, only for models loaded with `engine.load(url, { retainGeometry: true })`. Other models are hit at their bounding box. Only meshes are hit; for sprites and 2D things use [`scene.pick`](#scene-pick) with a `Camera2D`. A mesh drawn in levels of detail is hit by its finest level: a ray has no distance to choose one by.

<a id="scene-pick"></a>
### `scene.pick(camera, x, y, width, height, options)` → `{ node, renderable, distance } | null`

Finds the nearest mesh under a point on the canvas. `x`, `y` are CSS pixels from the canvas's top-left; `width`, `height` are the canvas's CSS size. That is what a pointer event and `getBoundingClientRect()` give you. With a 3D [`Camera`](#camera) it builds the ray with [`camera.rayFromScreen`](#camera-rayfromscreen) and calls [`scene.raycast`](#scene-raycast); `options` are `raycast`'s.

Returns: as [`scene.raycast`](#scene-raycast).

Throws: if `width` or `height` is not positive, or the ray is not finite.

```js
canvas.addEventListener('click', (e) => {
  const r = canvas.getBoundingClientRect();
  const hit = scene.pick(camera, e.clientX - r.left, e.clientY - r.top, r.width, r.height);
  if (hit) hit.node.setScale(1.2);
});
```

Notes: with a `Camera2D` it returns what the 2D view shows there instead, topmost first: `{ node, point }`, plus `tile: [column, row]` for a tilemap, or `null`. See [`scene.pick` with a Camera2D](#scene-pick-2d).

<a id="node"></a>
## Node

A Node is a handle to one thing in a scene. It stores nothing itself: `node.scene` is its scene and `node.entity` its entity handle. Two Node objects can refer to the same thing, so compare `a.entity === b.entity`, not `a === b`.

Transforms are set through methods, never by writing to properties. Positions, rotations and scales are in the parent's space. The setters return the node, so they chain:

```js
lamp.setPosition(2, 3, 0).setDirection(0, -1, -0.5);
```

The transform setters throw on a non-finite value (`setPosition: non-finite in (…)`), before writing anything, so a caught throw leaves the node where it was.

<a id="node-alive"></a>
### `node.alive` → `boolean`

`false` once the node has been removed. A removed node's methods must not be used.

<a id="node-setposition"></a>
### `node.setPosition(x, y, z = 0)` → `Node`

Places the node in its parent's space. `z` defaults to 0, so `setPosition(x, y)` places a 2D node.

<a id="node-setangle"></a>
### `node.setAngle(radians)` → `Node`

Turns the node about Z, in the screen's plane: the rotation for 2D. With a `Camera2D`, whose y points down, a positive angle turns clockwise, as CSS `rotate()` does. Replaces the node's whole rotation.

<a id="node-setscale"></a>
### `node.setScale(x, y, z)` → `Node`

One number scales every axis. Two numbers set x and y and leave z at 1, for a 2D node: `setScale(-1, 1)` mirrors it. Three are used as given.

```js
tree.setScale(2);          // 2, 2, 2
sprite.setScale(-1, 1);    // -1, 1, 1
box.setScale(1, 2, 0.5);
```

<a id="node-setrotation"></a>
### `node.setRotation(q)` → `Node`

Sets the rotation from a quaternion `[x, y, z, w]`. For angles, use [`setAxisAngle`](#node-setaxisangle), [`setEuler`](#node-seteuler) or [`setDirection`](#node-setdirection).

<a id="node-setdirection"></a>
### `node.setDirection(x, y, z = 0)` → `Node`

Turns the node so its -Z points along `(x, y, z)`, keeping +Y as upright as it can. That is the way a spot light shines, a directional light's light travels, and a followed camera looks. The vector need not be unit length. `setDirection(x, y)` aims across a 2D view. Pointing straight up or down has no upright answer; it then takes the shortest turn from -Z.

```js
key.setDirection(-0.4, -0.7, -0.3);
```

<a id="node-setaxisangle"></a>
### `node.setAxisAngle(axis, radians)` → `Node`

Sets the rotation to `radians` about `axis`.

Throws: if `axis` is not unit length (`quatSetAxisAngle: axis must be normalized`).

```js
decal.setAxisAngle([1, 0, 0], -Math.PI / 2);
```

<a id="node-seteuler"></a>
### `node.setEuler(yaw, pitch, roll = 0)` → `Node`

Sets the rotation from angles in radians, applied in YXZ order: yaw about Y, pitch about X, roll about Z. The angles are not stored.

<a id="node-setparent"></a>
### `node.setParent(node)` → `Node`

Attaches the node to another. `null` moves it to the scene root. Its position, rotation and scale are kept as numbers, now in the new parent's space, so it moves in the world if the parents differ.

Throws: `setParent: would create a cycle in the transform hierarchy`, if the new parent is the node itself or below it; `setParent: parent was removed`; `setParent: parent is a node of another scene`.

<a id="node-worldposition"></a>
### `node.worldPosition(out)` → `out`

Writes the node's world position into `out` (a 3-element array) and returns it.

```js
const p = node.worldPosition(vec3Create());
```

Notes: the value is as of the last [`scene.update()`](#scene-update). Straight after `setPosition` it returns the old position until the scene updates.

<a id="node-children"></a>
### `node.children()` → `Node[]`

The node's direct children, oldest first; `[]` if the node is gone. Same as [`scene.childrenOf(node)`](#scene-childrenof).

<a id="node-destroy"></a>
### `node.destroy()`

Removes the node and everything under it. Same as [`scene.remove(node)`](#scene-remove). Returns nothing.

<a id="camera"></a>
## Camera

<a id="camera"></a>
### `new Camera(options)` → `Camera`

A 3D camera, perspective by default. It looks from `position` toward `target`. [`engine.run`](#engine-run) calls its [`update`](#camera-update) each frame.

| Option | Default | Meaning |
|---|---|---|
| `fovY` | `Math.PI / 3` | Vertical field of view, radians. The horizontal one follows from the aspect. |
| `near` | `0.1` | Nearest distance drawn. Precision is near-uniform, so this can be small without z-fighting. |
| `orthographic` | `false` | Parallel projection: size on screen does not change with distance. |
| `far` | `1000` | Where an orthographic view ends. Used only when `orthographic` is true. |

A perspective camera has no far plane: it draws to infinity, and there is no draw distance to set. Only an orthographic camera has `far`.

An orthographic camera shows a height of `2 * distance * tan(fovY / 2)`, where `distance` is from `position` to `target`: what a perspective camera would see at the target. Move the camera closer or further to zoom.

```js
const camera = new Camera({ fovY: 0.8 });
camera.position.set([0, 2, 6]);
camera.target.set([0, 1, 0]);
```

<a id="camera-properties"></a>
### Camera properties

| Property | Default | Meaning |
|---|---|---|
| `position` | `[0, 0, 5]` | Where the camera is. A `Float32Array`; write into it. |
| `target` | `[0, 0, 0]` | The point it looks at. |
| `up` | `[0, 1, 0]` | Which way is up. Need not be exactly perpendicular to the view. |
| `fovY` | from options | Vertical field of view, radians. |
| `near` | from options | Nearest distance drawn. |
| `orthographic` | from options | Switch at any time; a camera switched to orthographic gets `far = 1000` if it had none. |
| `far` | from options | Orthographic only. |
| `aspect` | `1` | Read only: the aspect from the last `update`. |
| `following` | `null` | Read only: the node set by [`follow`](#camera-follow). |
| `view`, `projection`, `viewProjection`, `inverseProjection` | | Read only: matrices from the last `update`. |

<a id="camera-follow"></a>
### `camera.follow(node)` → `Camera`

Makes the camera ride a node: every `update` takes its position and aim from the node's world transform. The node's -Z is the view direction and its +Y is up; scale is ignored. `null` stops following and leaves the camera where it was.

```js
const mount = scene.createNode({ parent: car });
mount.setPosition(0, 2, 6);
camera.follow(mount);   // a chase camera
```

Notes: while following, `position`, `target` and `up` are overwritten every update, and an `OrbitController` on this camera stands aside. To hand control back, call `follow(null)` then `controller.syncFromCamera()`. The distance from `position` to `target` is kept, which sets an orthographic camera's view height. If the node is removed, the camera stops following and stays put.

<a id="camera-framebounds"></a>
### `camera.frameBounds(min, max, options)` → `Camera`

Moves the camera so a world-space box fills the view, keeping the direction it already looks from. It fits the box's bounding sphere, so the fit does not change as you orbit.

| Option | Default | Meaning |
|---|---|---|
| `margin` | `1` | Multiplies the fitting distance. `1` fits the sphere exactly, which already leaves some air around the box. |
| `aspect` | `camera.aspect` | Width over height. `camera.aspect` is `1` before the first `update`, so pass it during setup. |

Throws: if `min` or `max` has a non-finite component.

```js
camera.frameBounds([-1, 0, -1], [1, 2, 1], { aspect: 16 / 9, margin: 1.1 });
```

Notes: an orthographic camera's `far` is pushed out if the box would not fit. See also [`scene.frame`](#scene-frame).

<a id="camera-orthographichalfheight"></a>
### `camera.orthographicHalfHeight()` → `number`

Half the world height an orthographic camera shows: `distance * tan(fovY / 2)`, with `distance` from `position` to `target`.

<a id="camera-update"></a>
### `camera.update(aspect)` → `Camera`

Recomputes the matrices from `position`, `target`, `up`, `fovY` and the followed node. `aspect` is the view's width over height. [`engine.run`](#engine-run) calls it every frame; call it yourself only when you drive frames yourself.

<a id="camera-rayfromscreen"></a>
### `camera.rayFromScreen(x, y, width, height, outOrigin, outDirection)` → `outDirection`

The world-space ray through a point on the canvas. `x`, `y` are CSS pixels from the canvas's top-left; `width`, `height` the canvas's CSS size, not its backing-store size. Writes the ray into `outOrigin` and `outDirection` (3-element arrays) and returns `outDirection`.

A perspective ray starts at the camera's position and its direction is unit length. An orthographic ray starts on the camera's plane under the point and points straight down the view.

Throws: if `width` or `height` is not positive.

```js
const origin = vec3Create(), dir = vec3Create();
camera.rayFromScreen(x, y, rect.width, rect.height, origin, dir);
```

Notes: the aspect is `width / height`, so the ray is right before the first frame and straight after a resize. [`scene.pick`](#scene-pick) does this and the raycast in one call.

<a id="animation"></a>
## Animation

Models with clips play them through the Node [`scene.add`](#scene-add) returned. [`engine.run`](#engine-run) advances them every frame.

<a id="node-play"></a>
### `node.play(nameOrIndex, options)` → `Node`

Starts a clip, by name or by index into [`node.animations`](#node-animations). Without a fade, the new clip replaces what the layer was playing. The pose changes on the next [`scene.advance`](#scene-advance).

| Option | Default | Meaning |
|---|---|---|
| `loop` | `true` | Repeat. A clip that does not loop stops on its last frame. |
| `speed` | `1` | Playback rate. Negative plays backwards; a clip that does not loop then stops at its start. |
| `time` | `0` | Where to start, in seconds. |
| `fade` | `0` | Seconds to cross-fade from what the layer is playing. On the base layer with nothing playing, the clip starts at full weight. On another layer it fades the layer in. |
| `layer` | `'base'` | Which layer to play on. Make other layers first with `node.animation.layer(name)`. |
| `weight` | `1` | The clip's weight within its layer, when fully in. |
| `join` | `false` | Join the clips already playing on the layer instead of replacing them. Move weights with `node.animation.setWeight`. |
| `sync` | `false` | Share one clock with the layer's other synced clips, measured in cycles, so clips of different lengths stay in step. A synced clip loops, and one joining a group starts at the group's place, ignoring `time`. |

Returns: the node. Does nothing if there is no such clip, or the node has no player.

Throws: `AnimationPlayer: no layer named "..."; make it with layer() first`, for an unknown `layer`. A `RangeError` if `weight` is negative or not finite.

```js
const model = scene.add(asset);
model.play('Walk');
model.play('Run', { fade: 0.3 });   // cross-fade over 0.3 s
```

<a id="node-stop"></a>
### `node.stop(options)` → `Node`

Stops every layer, or one.

| Option | Default | Meaning |
|---|---|---|
| `layer` | every layer | The layer to stop. |
| `fade` | `0` | Seconds to fade out. Without a fade the pose stays where it is. With one, a layer above the base hands the nodes back to the layers beneath. |

Throws: for an unknown `layer`.

```js
model.stop({ layer: 'upper', fade: 0.2 });
```

<a id="node-animation"></a>
### `node.animation` → `AnimationPlayer | null`

The model instance's player, or `null` (not the returned root of a model, or no clips). Use it for layers, weights and root motion.

| Member | Meaning |
|---|---|
| `names` | Clip names. |
| `play(nameOrIndex, options)` | As [`node.play`](#node-play), but returns `false` if there is no such clip. |
| `stop(options)` | As [`node.stop`](#node-stop). |
| `layer(name, { mask, weight, additive })` | Makes a layer, or changes one; layers apply in the order made. `mask`: a node name or list of names, each with everything under it; `null` clears it. `weight` scales the layer. `additive: true` adds each clip's change from its first keyframe instead of covering the pose beneath. Throws for an additive base layer, a negative weight, or a mask name no node has. |
| `setWeight(nameOrIndex, weight, { layer = 'base', fade = 0 })` | Moves a playing clip's weight, at once or over `fade` seconds. Returns `false` if the clip is not playing on that layer. Weight 0 keeps it playing, silent. |
| `rootMotion({ node, vertical = false, apply = true })` | Moves the instance by the travel of one node (default: the highest node below the instance that a clip translates), and holds that node in place. `vertical: true` includes height. `apply: false` only measures, into `motion`. `rootMotion(null)` turns it off. Throws if it cannot choose the node; name it with `node`. |
| `motion` | `{ position, yaw }`: what root motion moved the instance by in the last advance. |
| `clip`, `time`, `speed`, `loop`, `finished` | The base layer's newest clip. `time`, `speed` and `loop` can be set. `finished` is true once a clip that does not loop has reached its end. |

```js
const player = model.animation;
player.layer('upper', { mask: 'Spine' });
model.play('Wave', { layer: 'upper', fade: 0.2 });
```

<a id="node-animations"></a>
### `node.animations` → `string[]`

The clip names this instance can play; `[]` if it has none.

<a id="node-weights"></a>
### `node.weights` → `Float32Array | null`

The morph target weights of the mesh on this node, or `null` if it has none. A live array: write to it directly.

```js
face.weights[0] = 1;   // full smile
```

Notes: the weights belong to the node that carries the mesh, which is often a child of the model's root, not the root itself. A clip that animates the weights overwrites what you write.

<a id="lights"></a>
## Lights

A light is a node. Its position and aim come from its node's transform, so it moves, parents and animates like anything else. Change its colour, brightness, reach and cone with [`scene.setLight`](#scene-setlight). A new scene has no lights; its [environment](#scene-environment) lights it until you add one.

<a id="scene-addlight"></a>
### `scene.addLight(options)` → `Node`

Adds a point, spot or directional light.

| Option | Default | Meaning |
|---|---|---|
| `type` | `'point'` (`'spot'` with a `direction`) | `'point'`, `'spot'` or `'directional'`. |
| `position` | `[0, 0, 0]` | In the parent's space. `[x, y]` places it in a 2D view. A directional light's position does not matter. |
| `direction` | `null` | Aims the node's -Z this way: where a spot shines and where a directional light's light travels. `[x, y]` aims across a 2D view. After this the node's rotation carries the aim. |
| `color` | `[1, 1, 1]` | Linear RGB, each 0 or more. |
| `intensity` | `1` | Multiplies the colour. 0 or more. |
| `radius` | `10` | Where a point or spot light fades to exactly zero. Above 0. Not used by a directional light. |
| `innerAngle` | `0.2` | Spot only: radians from the axis where the cone starts to fade. |
| `outerAngle` | `0.5` | Spot only: radians from the axis where the cone reaches zero. `0 ≤ innerAngle ≤ outerAngle ≤ π/2`; equal angles give a hard edge. |
| `parent` | `null` | Node to attach it to. The light then follows and aims with its parent. |
| `castShadow` | `true` for directional, `false` otherwise | Whether it casts shadows. |
| `size` | `0` | How large the light itself is, which softens its shadows: a point or spot light's radius in world units (a bulb, 0.05; a window, 0.5); a directional light's angle across, in radians (the sun, 0.0093). `0` casts the plain edge. 0 or more. |

Returns: the light's Node.

Throws: `'addLight: type must be point, spot or directional, got …'`; `'addLight: color must be 3 finite numbers, 0 or more, got …'`; `'addLight: intensity must be 0 or more, got …'`; `'addLight: radius must be positive, got …'`; `'addLight: size must be 0 or more, got …'`; `'addLight: angles need 0 <= innerAngle <= outerAngle <= PI/2, got …'`.

```js
const lamp = scene.addLight({ position: [0, 3, 0], color: [1, 0.7, 0.4], intensity: 20, radius: 8 });
const sun = scene.addLight({ type: 'directional', direction: [-0.4, -0.8, -0.4], intensity: 3 });
const torch = scene.addLight({ direction: [0, 0, -1], parent: hand });   // a spot that aims where the hand aims
```

Notes:
- A directional light with no `direction` shines along -Z. Give it one, or turn its node.
- Passing `direction` without `type` makes a spot. For a directional light, say `type: 'directional'`.
- Shadows are drawn from a closed mesh's back faces, which keeps its lit faces free of acne. A mesh with an open edge -- a plane, a roof of one sheet -- is drawn whole, so it casts whichever side faces the light, as a double-sided material does.
- Soft shadows: a light with a `size` casts percentage-closer soft shadows (PCSS). Its penumbra widens with the gap between the caster and where the shadow falls, as a real light's does: sharp where a chair leg meets the floor, soft under a table top. The width is capped at 64 shadow-map texels, and the edge is a fine per-pixel grain, not bands. [`renderer.softShadows`](#renderer-softshadows) turns it off for every light at once. It costs only while some light that casts has a size: the scene's shaders are built a second way, with it compiled in, the first time one does, and frames show the plain edge until they are ready. It is not cheap: on integrated graphics (Intel Iris Xe), a sun the sun's size took Sponza's forward pass at 1280x720 from 5.7 to 8.8 ms, most of it 8 depth reads and 12 filtered taps for every pixel it lights.
- Shadows, by type: a directional light gets up to four cascaded shadow maps that follow the camera. A spot gets one shadow view down its cone, or six (like a point light) when `outerAngle` is wider than 45 degrees. A point light gets six, one per cube face. A point or spot light draws no shadow maps while its radius sphere is off screen. Maps are redrawn only when something within the light's reach moves.
- 2D: through a `Camera2D`, point and spot lights also light every sprite, shape, path, text or tilemap made with `lit: true`, fading to nothing at `radius` (in the view's units). Give `position: [x, y]` and, for a spot, `direction: [x, y]`. Directional lights and shadows do not apply in 2D. Where no light reaches, the camera's `ambient` lights it.

  ```js
  scene.addLight({ position: [120, 80], radius: 90, color: [1, 0.7, 0.4], intensity: 1.5 });
  scene.addLight({ position: [40, 60], direction: [1, 0], radius: 200, outerAngle: 0.4 });
  ```

<a id="scene-setlight"></a>
### `scene.setLight(node, changes)`

Changes what a light is, without moving it: the same names [`addLight`](#scene-addlight) takes. Only the fields given change. Move or aim it through its node.

| Change | Applies to |
|---|---|
| `color`, `intensity`, `castShadow`, `size` | every type |
| `radius` | point, spot |
| `innerAngle`, `outerAngle` | spot |

Throws: `'setLight: this node is not a light'`; `'setLight: <type, position, direction or parent> can't change here; …'`; `'setLight: a <type> light has no <field>'` for a field the table does not list for its type; the value errors of [`addLight`](#scene-addlight), named `setLight`. A spot's angles are checked together, so changing one alone must still fit the other.

```js
scene.setLight(lamp, { intensity: 30, castShadow: true });
lamp.setPosition(2, 3, 0);   // moving is the node's job
```

Notes: a light cannot change type; remove it and add another.

<a id="scene-lightof"></a>
### `scene.lightOf(node)` → `object | null`

The light's settings, or `null` if the node is not a light. A copy: change it through [`setLight`](#scene-setlight).

Returns: `{ type, color, intensity, castShadow, size }`, plus `radius` for a point or spot, plus `innerAngle` and `outerAngle` for a spot. Position and direction are not included; they are the node's.

```js
if (scene.lightOf(lamp).castShadow) console.log('casts');
```

<a id="particles"></a>
## Particles

Emitters are simulated on the GPU. Once born, a particle lives in world space, so a moving emitter leaves a trail. [`engine.run`](#engine-run) advances them.

<a id="scene-addemitter"></a>
### `scene.addEmitter(options)` → `Node`

Adds a particle emitter as a node. Particles leave along its `direction`, turned by the node's rotation, and are drawn as quads facing the camera.

| Option | Default | Meaning |
|---|---|---|
| `rate` | `0` | Particles per second. `0` for bursts only (see [`scene.burst`](#scene-burst)). |
| `lifetime` | required | Seconds. One number, or `[min, max]` for each particle to pick from. Above 0. |
| `size` | required | Width at birth, in world units. One number. |
| `sizeEnd` | `size` | Width at death. |
| `speed` | `0` | Speed at birth. One number, or `[min, max]`. |
| `direction` | `[0, 1, 0]` | In the node's space. `[x, y]` in a 2D view. Must not be zero. |
| `spread` | `0` | Radians off `direction` a particle may leave at: 0 is a line, `Math.PI` every way. |
| `radius` | `0` | Particles are born anywhere in this sphere around the node. 0 is a point. |
| `acceleration` | `[0, 0, 0]` | World space, units per second squared: gravity, if you want it. `[x, y]` in a 2D view. |
| `drag` | `0` | Share of speed lost per second, as a rate. |
| `color` | `[1, 1, 1, 1]` | At birth. Linear in 3D, and may exceed 1 to glow. sRGB in a 2D view. |
| `colorEnd` | `color` | At death. |
| `texture` | `null` | From [`engine.loadTexture`](#engine-loadtexture). Without one, a soft round dot. |
| `blend` | `'additive'` | `'additive'` (needs no draw order) or `'alpha'`, whose particles are sorted far to near on the GPU every frame in 3D. |
| `layer` | `0` | 2D only: its place in the painter's order with sprites. |
| `position` | `[0, 0, 0]` | In the parent's space. |
| `parent` | `null` | Node to attach it to. |

Returns: the emitter's Node.

Throws (all start `addEmitter:`): `lifetime` or `size` missing; `size` not a single number; `size` or `sizeEnd` negative; `lifetime` not positive; `speed` negative, or a range with min above max; `rate`, `radius` or `drag` negative or not finite; `direction` zero or the wrong length; `spread` outside 0 to pi; `color`, `colorEnd` or `acceleration` the wrong length or not finite; `texture` not from `engine.loadTexture`; `blend` not `'additive'` or `'alpha'`; `layer` not finite.

```js
const sparks = scene.addEmitter({
  rate: 200, lifetime: [0.4, 0.8], size: 0.05, sizeEnd: 0, speed: [2, 4], spread: 0.4,
  acceleration: [0, -9.81, 0], color: [4, 2, 0.5, 1],
});
```

<a id="scene-setemitter"></a>
### `scene.setEmitter(node, changes)`

Changes an emitter's options: the same names [`addEmitter`](#scene-addemitter) takes, checked the same way. Particles already alive live on.

Throws: `setEmitter: this node has no emitter`, or as `addEmitter` for a bad value.

```js
scene.setEmitter(sparks, { rate: 0 });   // stop emitting; the last sparks finish
```

<a id="scene-emitterof"></a>
### `scene.emitterOf(node)` → `object | null`

The emitter's options as [`addEmitter`](#scene-addemitter) took them, or `null`. A copy: change it through [`setEmitter`](#scene-setemitter).

<a id="scene-burst"></a>
### `scene.burst(node, count)`

Emits `count` particles at once, on the next frame.

Throws: `burst: this node has no emitter`; `burst: count must be a whole number, got ...` for a negative or fractional count.

```js
const puff = scene.addEmitter({ lifetime: 1, size: 0.2, speed: [1, 2], spread: Math.PI });
scene.burst(puff, 50);
```

<a id="scene-particlesactive"></a>
### `scene.particlesActive` → `boolean`

Whether a particle may still be alive, or one is about to be born.

<a id="splats"></a>
## Gaussian splats

A capture made by 3D Gaussian Splatting: up to millions of soft, coloured ellipsoids fitted to photographs. They are sorted back to front on the GPU whenever the camera, the cloud or the canvas has moved, and blended over the scene. Load one with [`engine.loadSplats`](#engine-loadsplats).

<a id="scene-addsplats"></a>
### `scene.addSplats({ splats, position, parent })` → `Node`

Adds a capture as a node. The node's position, rotation and scale place it, and one capture can be added any number of times. `position` defaults to `[0, 0, 0]`.

Splats are lit by nothing: they show the colour the capture saw, from where the camera sees them when the capture has harmonics, decoded from sRGB, fogged by [`renderer.fog`](#renderer-fog) by the distance to each one, and tonemapped with the rest of the frame. Geometry in front hides them. They write no depth, so they hide nothing and cast no shadows. They are drawn straight after opaque geometry, so sprites, particles and blended surfaces in front of a capture show over it; blended or transmissive surfaces behind one show over it too. [Depth of field](#renderer-dof) works from depth, so it blurs splats as whatever geometry is behind them, or as far away where there is none. They count in [`scene.bounds`](#scene-bounds) and [`scene.frame`](#scene-frame) by the box around their centres. They are not picked or raycast.

Throws: `'addSplats: splats must be what engine.loadSplats returned'`; `'addSplats: these splats were unloaded'`. A frame drawing splats unloaded since they were added throws `'addSplats: these splats were unloaded; remove the node first'`.

```js
const room = await engine.loadSplats('room.ply');
// Captures usually come with y down, as the photographs were: turn them upright.
scene.addSplats({ splats: room }).setAxisAngle([1, 0, 0], Math.PI);
```

Notes: a capture's harmonics are worked out once a visible splat whenever it is sorted again, so a still view barely pays for them: for 205,000 splats of degree 3 on integrated graphics (Intel Iris Xe), about 1.4 ms in the frames the view moves, and 0.16 ms in the draw otherwise. Two captures are each sorted on their own and drawn farther one first, so where two overlap they do not interleave. On integrated graphics (Intel Iris Xe, a million splats, 1280x720) the draw costs about 10 ms for a capture seen whole, and more up close, where splats fill the screen; [`renderer.resolution`](#renderer-resolution) cuts that part. The sort costs about 2 ms, and nothing while the view is still.

<a id="scene-splatsof"></a>
### `scene.splatsOf(node)` → `{ splats } | null`

The capture a node draws, or `null`.

<a id="decals"></a>
## Decals

<a id="scene-adddecal"></a>
### `scene.addDecal(options)` → `Node`

Projects a texture onto whatever surfaces lie inside a box, along the box's -Z: a scorch mark, a poster, a puddle. It changes the surface's base colour before lighting, so it is lit, shadowed and fogged with the surface. Surfaces facing away from it are not painted. Later decals paint over earlier ones.

| Option | Default | Meaning |
|---|---|---|
| `texture` | required | From [`engine.loadTexture`](#engine-loadtexture). Its alpha is how much it covers. |
| `size` | required | The box: `[width, height, depth]`, width and height across the image and depth along the projection, in the node's units. Scaled by the node. |
| `color` | `[1, 1, 1, 1]` | Multiplies the texture. Its alpha scales the cover. |
| `position` | `[0, 0, 0]` | In the parent's space. |
| `parent` | `null` | Node to attach it to. |

Returns: the decal's Node.

Throws: `addDecal: texture must be one engine.loadTexture returned`; `addDecal: size must be [width, height, depth], all positive, ...`; `addDecal: color must be 4 finite numbers, ...`.

```js
const scorch = scene.addDecal({ texture: burn, size: [2, 2, 0.5] });
scorch.setPosition(0, 0.01, 0).setAxisAngle([1, 0, 0], -Math.PI / 2);   // project downward
```

<a id="scene-setdecal"></a>
### `scene.setDecal(node, changes)`

Changes a decal's `texture`, `size` or `color`, checked as in [`addDecal`](#scene-adddecal).

Throws: `setDecal: this node has no decal`, or as `addDecal` for a bad value.

<a id="scene-decalof"></a>
### `scene.decalOf(node)` → `{ texture, size, color } | null`

The decal's settings, or `null`. Change them through [`setDecal`](#scene-setdecal).

<a id="reflection-probes"></a>
## Reflection probes

A probe makes the surfaces inside a box reflect the scene as seen from the probe, instead of the sky. Nothing shows until it is captured with [`engine.captureProbes`](#engine-captureprobes).

<a id="scene-addprobe"></a>
### `scene.addProbe(options)` → `Node`

Adds a probe as a node. Its box is centred on the node and stays square to the world: turning or scaling the node does not turn or scale it. Moving the node moves the box, and the probe then shows nothing until it is captured again.

| Option | Default | Meaning |
|---|---|---|
| `size` | required | The box, `[width, height, depth]` in world units. |
| `fade` | `0` | How far in from the box's faces the probe fades in. 0 is a hard edge; more hides the seam between neighbouring probes. |
| `position` | `[0, 0, 0]` | In the parent's space. |
| `parent` | `null` | Node to attach it to. |

Returns: the probe's Node.

Throws: `addProbe: a probe is a node now -- give its box as size, ...` if given the old `min`/`max`; `addProbe: size must be [width, height, depth], all positive, ...`; `addProbe: fade must be 0 or more, ...`; `addProbe: blend is now fade, …` for the old name.

```js
const hall = scene.addProbe({ position: [0, 2, 0], size: [10, 4, 16] });
await engine.captureProbes(scene);
```

Notes: where boxes nest, the smaller one wins. Capturing renders the scene six times per probe, so do it once the room is loaded, and again when it changes.

<a id="scene-setprobe"></a>
### `scene.setProbe(node, changes)`

Changes a probe's `size` or `fade`. The probe shows nothing until it is captured again.

Throws: `setProbe: this node is not a reflection probe`, or as `addProbe` for a bad value.

<a id="scene-probeof"></a>
### `scene.probeOf(node)` → `{ size, fade, captured } | null`

The probe's settings and whether it is captured now, or `null` if the node is not a probe.

<a id="sprites"></a>
## Sprites

<a id="scene-addsprite"></a>
### `scene.addSprite(options)` → `Node`

Adds a textured quad as a node. It moves, parents and is removed like any other node. Both 3D views and 2D views ([`Camera2D`](#camera2d)) draw it.

| Option | Default | Meaning |
|---|---|---|
| `texture` | required | A texture from `engine.loadTexture`. |
| `position` | `[0, 0, 0]` | Where the node goes. `[x, y]` places it in 2D. |
| `parent` | `null` | A node to hang it off. |
| `size` | see Notes | `[width, height]`. 3D: world units, or pixels with `pixels: true`. 2D: CSS pixels at zoom 1. |
| `pixels` | `false` | 3D only. `size` is in screen pixels, so the sprite stays one size on screen. The node's scale is then ignored. |
| `color` | `[1, 1, 1, 1]` | Multiplies the texture. 3D: linear, may exceed 1 to glow. 2D: sRGB 0..1, as in CSS. |
| `rect` | `[0, 0, 1, 1]` | `[u0, v0, u1, v1]`: the part of the texture to show. Past 0..1 the image repeats: `[0, 0, 4, 1]` shows it four times across, for a scrolling background or a parallax strip. Ignored while `animation` is set. |
| `pivot` | `[0.5, 0.5]` | The point of the image placed at the node, 0..1 from its top-left. `[0.5, 1]` is its bottom middle, for something standing on the ground. |
| `angle` | `0` | Radians. In 2D it turns the quad clockwise, on top of the node's angle. In 3D it turns the quad in its own plane (counter-clockwise as seen, since y points up). |
| `facing` | `'camera'` | 3D only. `'camera'` turns to face the camera every way. `'upright'` turns about Y only, for trees and people. `'plane'` does not turn: the quad lies in the node's own x-y plane, like a sign. |
| `blend` | `'alpha'` | `'alpha'`, `'additive'` (adds light), `'multiply'` (darkens: shadows, tints), `'screen'` (lightens: glows), or `'cutout'` (each pixel drawn fully or not at all, by `cutoff`). |
| `cutoff` | `0.5` | For `'cutout'`: pixels with alpha below this are dropped. 0..1. |
| `layer` | `0` | 2D only. Higher layers draw over lower ones. See [painter's order](#view2d-order). |
| `animation` | `null` | `{ frames, fps = 12, loop = true }`. `frames` is a list of rects, as [`spriteSheet`](#spritesheet) makes. A non-looping animation stops on its last frame. |
| `lit` | `false` | 2D only. Lit by the scene's lights and the camera's ambient. See [lighting](#view2d-lighting). |

Returns: the new `Node`.

Throws:
- `addSprite: texture must be one engine.loadTexture returned` if `texture` is missing or has no view or size.
- `addSprite: size must be 2 finite numbers` / `size must be positive`.
- `addSprite: color must be 4 finite numbers`, and the same for `rect` (4) and `pivot` (2).
- `addSprite: facing is 'camera', 'upright' or 'plane'` for any other `facing`.
- `addSprite: blend is 'alpha', 'additive', 'multiply', 'screen' or 'cutout'` for any other `blend`.
- `addSprite: cutoff must be between 0 and 1`.
- `addSprite: angle must be a finite number`, and the same for `layer`.
- `addSprite: rotation is now angle, …` for the old name.
- `addSprite: animation.frames must be a list of rects` if `frames` is not a non-empty array; each frame must be 4 finite numbers.
- `addSprite: animation.fps must be positive`.

```js
import { spriteSheet } from 'winding-engine';

const hero = await engine.loadTexture('hero.png', { pixelated: true });
const player = scene.addSprite({
  texture: hero,
  animation: { frames: spriteSheet({ columns: 8 }), fps: 10 },
  pivot: [0.5, 1],
  position: [160, 120],
});
```

Notes:

| | 3D view | 2D view |
|---|---|---|
| Quad | Faces the camera, per `facing`. | Flat on screen. `facing` and `pixels` are ignored. |
| Default size | 1 unit wide at its frame's aspect, or the frame's size in pixels with `pixels: true`. The frame is its first animation frame, or its `rect`. | The current frame's own size in texels. |
| Units | World units, or pixels with `pixels: true`. | A unit is a CSS pixel. |
| Colour | Linear light. Unlit, fogged, and may glow through bloom. | sRGB, blended as a browser blends a page. |
| Order | Cutouts first, then additive, then alpha, multiply and screen sorted back to front. `layer` is ignored. | By `layer`, then in the order added. |
| Scale | Scaled by the node (unless `pixels`). | Scaled by the node. A negative x scale (`node.setScale(-1, 1)`) mirrors it. |

- `engine.run` advances animations. If you call `engine.renderFrame` yourself, call [`scene.advance(dt)`](#scene-advance) each frame.
- In 3D, sprites draw after opaque geometry and before glass. An alpha sprite in front of glass is drawn over by it.
- In 2D, a sprite's edge is smoothed over a screen pixel, as a shape's is, so a turned one isn't jagged; one on whole pixels (with [`pixelSnap`](#camera2d-pixelsnap)) is unchanged. A `'cutout'` keeps its hard edge.
- A frame of a sheet samples only inside its own rect, so smooth filtering never shows an edge of the next frame. Mipmaps still average neighbouring frames when a sheet is drawn much smaller than its size: leave a few texels between frames if that shows.
- A [target](#engine-createtarget) is a texture too: a sprite shows what was drawn into it.
- Remove it with `scene.remove(node)` or `node.destroy()`.

<a id="scene-setsprite"></a>
### `scene.setSprite(node, changes)`

Changes a sprite's options. Takes the same names as [`scene.addSprite`](#scene-addsprite); options you leave out keep their values.

Returns: nothing.
Throws: `setSprite: this node has no sprite`, or any error `addSprite` throws for the merged options, named `setSprite:`.

```js
scene.setSprite(player, { animation: { frames: spriteSheet({ columns: 8, first: 8, count: 8 }), fps: 10 } });
```

Notes:
- Passing `animation` (including `null`) restarts from its first frame.
- If you never gave `size`, a new `texture` gets a new default size. A `size` you gave is kept.

<a id="scene-spriteof"></a>
### `scene.spriteOf(node)` → `object | null`

A sprite's current options, or `null` if the node has no sprite.

Returns: a copy with `texture`, `size`, `color`, `rect`, `pivot`, `angle`, `facing`, `blend`, `cutoff`, `pixels`, `layer`, `lit` and `animation` (`{ frames, fps, loop }` or `null`), plus `frame` (the frame shown) and `time` (seconds the animation has run).

```js
const { frame } = scene.spriteOf(player);
```

Notes: a copy all the way down: changing it changes nothing. Change the sprite through [`scene.setSprite`](#scene-setsprite).

<a id="spritesheet"></a>
### `spriteSheet({ columns, rows, count, first })` → `number[][]`

The frames of a sprite sheet laid out in a grid, as rects in reading order: left to right, then top to bottom. Use it for `animation.frames`, or pick one as a sprite's `rect`. Exported from `'winding-engine'`.

| Option | Default | Meaning |
|---|---|---|
| `columns` | required | Frames across. A whole number above zero. |
| `rows` | `1` | Frames down. A whole number above zero. |
| `count` | `columns * rows` | How many frames. Stops short of a last row that is not full. |
| `first` | `0` | Frames to skip from the start. |

Returns: an array of `[u0, v0, u1, v1]`, one per frame.
Throws:
- `spriteSheet: columns and rows must be whole numbers above zero`.
- `spriteSheet: frames A to B are not all in a C x R sheet` if `first` or `count` is not a whole number, `count` is below 1, `first` is negative, or `first + count` is past the last frame.

```js
const run = spriteSheet({ columns: 6, rows: 4, first: 6, count: 6 });   // the second row
scene.addSprite({ texture: hero, rect: run[0] });
```

<a id="text"></a>
## Text

<a id="scene-addtext"></a>
### `scene.addText(options)` → `Node`

Adds a string as a node. Each glyph is a quad drawn from a distance field, so text stays sharp at any size. Both 3D and 2D views draw it.

| Option | Default | Meaning |
|---|---|---|
| `font` | required | A font from `engine.loadFont`. |
| `position` | `[0, 0, 0]` | Where the node goes. `[x, y]` places it in 2D. |
| `parent` | `null` | A node to hang it off. |
| `text` | `''` | The string. Turned into a string with `String()`. `\n` starts a new line. |
| `size` | required | The font's em. 3D: world units, or pixels with `pixels: true`. 2D: CSS pixels at zoom 1. |
| `color` | `[1, 1, 1, 1]` | 3D: linear, may exceed 1 to glow. 2D: sRGB 0..1. |
| `align` | `'left'` | `'left'`, `'center'` or `'right'`: each line within the block. |
| `pivot` | `[0.5, 0.5]` | The point of the block placed at the node, 0..1. `[0, 0]` is the block's top-left. |
| `lineHeight` | the font's | Distance between baselines, in ems. The font's ascent plus descent by default. |
| `width` | `Infinity` | Lines wrap between words to fit it, in the same units as `size`. The block is then this wide, so `align` and `pivot` work within it. A word wider than it overflows. |
| `facing` | `'camera'` | 3D only. `'camera'`, `'upright'` or `'plane'`, as for [sprites](#scene-addsprite). |
| `pixels` | `false` | 3D only. `size` is in screen pixels, and the node's scale is ignored. |
| `layer` | `0` | 2D only. See [painter's order](#view2d-order). |
| `lit` | `false` | 2D only. See [lighting](#view2d-lighting). |
| `blend` | `'alpha'` | How it meets what is under it: `'alpha'`, `'additive'` (adds light), `'multiply'` (darkens; white changes nothing) or `'screen'` (lightens; black changes nothing). |
| `stroke` | `[0, 0, 0, 1]` | The outline's colour. |
| `strokeWidth` | `0` | The outline's width, in the units of `size`, drawn outside the letters so they keep their weight. At most about a ninth of an em for a 64px font (see Notes). 0 means no outline. |

Returns: the new `Node`.

Throws:
- `addText: anchor is now pivot, and [0, 0] is the block's top-left, as a sprite's is` if `anchor` is passed at all.
- `addText: width must be positive`.
- `addText: font must be one engine.loadFont returned`.
- `addText: size must be positive` (also for a missing or non-finite size).
- `addText: color must be 4 finite numbers`.
- `addText: facing is 'camera', 'upright' or 'plane'`.
- `addText: pivot must be [x, y]`.
- `addText: layer must be a finite number`.
- `addText: align is 'left', 'center' or 'right'` for any other `align`.
- `addText: blend is 'alpha', 'additive', 'multiply' or 'screen'`.
- `addText: stroke must be 4 finite numbers`.
- `addText: strokeWidth must be 0 to N for this font at this size, …` past what the font holds.

```js
const font = await engine.loadFont('600 32px system-ui, sans-serif');
scene.addText({ font, text: 'Gate 3', size: 0.4, parent: gate });                        // 3D, world units
hud.addText({ font, text: 'HULL', size: 11, pivot: [0, 0], position: [22, 19] });       // 2D, CSS pixels
```

Notes:
- In 3D text is unlit and fogged; in 2D it follows `lit`.
- Lines break on single spaces, on `\n`, and between any two Chinese or Japanese characters—but not before a closing mark such as `。` or after an opening one, as a browser breaks them—and between the words of Thai, Lao, Khmer and Burmese, which are written without spaces.
- A character is what a reader counts as one: a letter and its accents, a flag, an emoji and its skin tone are each one glyph. An emoji is drawn in the text's colour, not its own.
- Scripts whose letters join, change shape or reorder—Arabic, Hebrew, the Indic scripts, Thai and the rest—are drawn a word at a time, shaped by the browser as it draws text. A line whose first letter is right-to-left (Arabic, Hebrew) runs right to left: its words in reverse, each read as written. This is not the full bidirectional algorithm: numbers and Latin words inside it stay in place as words, and `align` is not reversed. Latin ligatures are not formed.
- Pairs are kerned as the font kerns them ("AV" sits closer than "AH").
- The outline is drawn in the font's distance field, which reaches 7 texels past each letter at its raster size: `strokeWidth` is at most 7/64 of an em for a font loaded at 64px, so 2.2 pixels for 20px text. Load the font larger for a wider outline. An outline is drawn outside the letters, where a shape's is inside its edge: inside, it would eat thin strokes.
- The font is rasterised at the size its CSS names. Text stays sharp above that size and down to about an eighth of it; smaller shimmers. Load the font near the size it is mostly seen at.
- `anchor` was renamed to `pivot`. The old name throws rather than being ignored.

<a id="scene-settext"></a>
### `scene.setText(node, changes)`

Changes a text's options, such as its string. Takes the same names as [`scene.addText`](#scene-addtext); options you leave out keep their values.

Returns: nothing.
Throws: `setText: this node has no text`, or any error `addText` throws for the merged options, named `setText:`.

```js
scene.setText(score, { text: `Score ${points}` });
```

Notes: every call lays the text out again and rebuilds the 2D draw list. Fine for a score; avoid it on many texts every frame.

<a id="scene-textof"></a>
### `scene.textOf(node)` → `object | null`

A text's current state, or `null` if the node has no text.

Returns: the options as `addText` and `setText` last took them — `font`, `text`, `size`, `pivot`, `width` and the rest — as a copy.

```js
const { align, pivot } = scene.textOf(label);
```

Notes: change it through [`scene.setText`](#scene-settext).

<a id="shapes"></a>
## Shapes

<a id="scene-addshape"></a>
### `scene.addShape(options)` → `Node`

Adds a rectangle or an ellipse as a node, for a 2D view. It is computed per pixel from its distance to the edge, so it is round and smooth at any size. A 3D camera does not draw it.

| Option | Default | Meaning |
|---|---|---|
| `position` | `[0, 0, 0]` | Where the node goes. `[x, y]` is enough. |
| `parent` | `null` | A node to hang it off. |
| `shape` | `'rect'` | `'rect'` or `'ellipse'`. A circle is an ellipse as wide as tall. |
| `size` | required | `[width, height]` in units (CSS pixels at zoom 1). |
| `radius` | `0` | A rect's corner radius. Drawn no larger than half the shorter side, so half the height makes a capsule; the radius asked for is kept, so a shape that shrinks and grows back gets its corner back. |
| `color` | `[1, 1, 1, 1]` | The fill, sRGB 0..1. Alpha 0 for an outline alone. |
| `stroke` | `[0, 0, 0, 1]` | The outline's colour. Drawn inside the edge, like a CSS border. |
| `strokeWidth` | `0` | The outline's width. 0 means no outline. |
| `pivot` | `[0.5, 0.5]` | The point of the shape placed at the node. `[0, 0]` is its top-left. |
| `layer` | `0` | See [painter's order](#view2d-order). |
| `blend` | `'alpha'` | How it meets what is under it: `'alpha'`, `'additive'` (adds light), `'multiply'` (darkens; white changes nothing) or `'screen'` (lightens; black changes nothing). |
| `lit` | `false` | See [lighting](#view2d-lighting). |

Returns: the new `Node`.
Throws:
- `addShape: shape is 'rect' or 'ellipse'`.
- `addShape: size must be 2 finite numbers` / `size must be positive`.
- `addShape: radius must be 0 or more`, and the same for `strokeWidth`.
- `addShape: color must be 4 finite numbers`, and the same for `stroke` (4) and `pivot` (2).
- `addShape: layer must be a finite number`.
- `addShape: blend is 'alpha' or 'additive'`.

```js
scene.addShape({ size: [120, 12], radius: 6, color: [0.9, 0.2, 0.2, 1] });
scene.addShape({ shape: 'ellipse', size: [30, 30], color: [0, 0, 0, 0], stroke: [1, 1, 1, 1], strokeWidth: 2 });
```

Notes: placed, turned (`node.setAngle`) and scaled by its node. Under a non-uniform scale, the corner radius and outline scale by the smaller axis.

<a id="scene-setshape"></a>
### `scene.setShape(node, changes)`

Changes a shape's options. Takes the same names as [`scene.addShape`](#scene-addshape); options you leave out keep their values.

Returns: nothing.
Throws: `setShape: this node has no shape`, or any error `addShape` throws for the merged options, named `setShape:`.

```js
scene.setShape(health, { size: [138 * hp, 10] });
```

<a id="scene-shapeof"></a>
### `scene.shapeOf(node)` → `object | null`

A shape's current options, or `null` if the node has no shape.

Returns: a copy with `shape`, `size`, `radius` (as asked), `color`, `stroke`, `strokeWidth`, `pivot`, `layer`, `blend` and `lit`.

```js
const [width] = scene.shapeOf(health).size;
```

Notes: change it through [`scene.setShape`](#scene-setshape).

<a id="paths"></a>
## Paths

<a id="scene-addpath"></a>
### `scene.addPath(options)` → `Node`

Adds a polygon to fill, a line to stroke, or both, as a node, for a 2D view. Computed per pixel from the distance to its segments, so it is smooth at any size. A 3D camera does not draw it.

| Option | Default | Meaning |
|---|---|---|
| `position` | `[0, 0, 0]` | Where the node goes. `[x, y]` is enough. |
| `parent` | `null` | A node to hang it off. |
| `points` | required | A list of `[x, y]`, in order, in the node's units. At least 2. |
| `closed` | `true` | Whether the last point joins the first. Only a closed path fills. A crossing or concave outline fills by the nonzero rule, as a canvas does. |
| `color` | `[1, 1, 1, 1]` | The fill, sRGB 0..1. Alpha 0 for a line alone. |
| `stroke` | `[0, 0, 0, 1]` | The line's colour. Centred on the path, with round joins and ends. |
| `strokeWidth` | `0` | The line's width. 0 means no line. |
| `layer` | `0` | See [painter's order](#view2d-order). |
| `blend` | `'alpha'` | How it meets what is under it: `'alpha'`, `'additive'` (adds light), `'multiply'` (darkens; white changes nothing) or `'screen'` (lightens; black changes nothing). |
| `lit` | `false` | See [lighting](#view2d-lighting). |

Returns: the new `Node`.
Throws:
- `addPath: points must be a list of [x, y]` if `points` is not an array.
- `addPath: point N must be [x, y]` for a point that is not 2 finite numbers.
- `addPath: a path needs at least 2 points`.
- `addPath: color must be 4 finite numbers`, and the same for `stroke`.
- `addPath: strokeWidth must be 0 or more`.
- `addPath: layer must be a finite number`.
- `addPath: blend is 'alpha', 'additive', 'multiply' or 'screen'`.

```js
scene.addPath({ points: [[0, 0], [60, 20], [0, 40]], color: [1, 0.8, 0, 1] });   // a triangle
scene.addPath({ points: route, closed: false, color: [0, 0, 0, 0], stroke: [1, 1, 1, 1], strokeWidth: 3 });
```

Notes:
- There is no `pivot`. The points are measured from the node.
- An open path with `strokeWidth: 0` draws nothing.
- A closed path's pixels test every segment: hundreds of points are fine; thousands are slow.
- An open path longer than 16 segments is drawn in pieces of 16, each covering only its own stretch, so a line across the screen costs what its pixels do. A see-through line is blended once where two pieces meet, but twice where it crosses a stretch of itself more than 16 segments away.

<a id="scene-setpath"></a>
### `scene.setPath(node, changes)`

Changes a path's options. Takes the same names as [`scene.addPath`](#scene-addpath). The points are kept unless you pass `points`.

Returns: nothing.
Throws: `setPath: this node has no path`, or any error `addPath` throws for the merged options, named `setPath:`.

```js
scene.setPath(trail, { points: history });
```

Notes: new points rebuild the 2D draw list. A colour change does not.

<a id="scene-pathof"></a>
### `scene.pathOf(node)` → `object | null`

A path's current options, or `null` if the node has no path.

Returns: the options as `addPath` and `setPath` took them, as a copy: `points` as a list of `[x, y]`, `closed`, `color`, `stroke`, `strokeWidth`, `layer`, `blend` and `lit`.

```js
const { points } = scene.pathOf(trail);   // [[x, y], ...]
```

Notes: what it returns can be passed straight back to [`scene.setPath`](#scene-setpath).

<a id="tilemaps"></a>
## Tilemaps

<a id="scene-addtilemap"></a>
### `scene.addTilemap(options)` → `Node`

Adds a grid of tiles from one tileset image, as a node, for a 2D view. It draws as one quad whatever its size, and changing a tile uploads that tile alone. A 3D camera does not draw it.

| Option | Default | Meaning |
|---|---|---|
| `position` | `[0, 0, 0]` | Where the node goes. With the default `pivot`, the map's top-left. |
| `parent` | `null` | A node to hang it off. |
| `tileset` | required | A texture from `engine.loadTexture`: tiles in a grid, read left to right, then top to bottom. |
| `tileSize` | required | `[width, height]` of a tile, in the tileset's texels, and in units on screen (before the node's scale). Whole numbers. |
| `columns` | required | The map's width in tiles. A whole number above zero. |
| `rows` | required | The map's height in tiles. A whole number above zero. |
| `tiles` | all empty | `columns * rows` ids, row by row from the top-left. See Notes. |
| `margin` | `0` | Texels around the tileset's grid, as Tiled names it. A whole number. |
| `spacing` | `0` | Texels between the tileset's tiles, as Tiled names it. A whole number. |
| `layer` | `0` | See [painter's order](#view2d-order). |
| `color` | `[1, 1, 1, 1]` | Multiplies every tile, sRGB 0..1. |
| `pivot` | `[0, 0]` | The point of the map placed at the node. `[0, 0]` is its top-left. |
| `blend` | `'alpha'` | How it meets what is under it: `'alpha'`, `'additive'` (adds light), `'multiply'` (darkens; white changes nothing) or `'screen'` (lightens; black changes nothing). |
| `lit` | `false` | See [lighting](#view2d-lighting). |

Returns: the new `Node`.
Throws:
- `addTilemap: tileset must be one engine.loadTexture returned`.
- `addTilemap: margin and spacing must be whole texels, 0 or more`.
- `addTilemap: tileSize must be whole texels inside the W x H tileset` if a side is not a whole number above zero, or a tile plus two margins is larger than the tileset.
- `addTilemap: columns and rows must be whole numbers above zero`.
- `addTilemap: tiles holds N ids; a C x R map needs M` if `tiles` is the wrong length.
- `addTilemap: tile N is not in the tileset, which holds T (ids start at 1; 0 is empty)` for an id that is not a whole 32-bit number, or whose tile (flip bits aside) is past the tileset's last.
- `addTilemap: tile N flips no tile; an empty tile is 0` for flip bits on tile 0.
- `addTilemap: layer must be a finite number`.
- `addTilemap: color must be 4 finite numbers`, and the same for `pivot` (2).
- On the first frame that draws it: `addTilemap: a C x R map is past this device's N tiles a side` if `columns` or `rows` is larger than the GPU's largest texture side.

```js
const tiles = await engine.loadTexture('tiles.png', { pixelated: true });
const map = scene.addTilemap({ tileset: tiles, tileSize: [16, 16], columns: 100, rows: 40, tiles: level });
scene.setTile(map, 12, 3, 0);   // break a block
```

Notes:
- Tile ids: `0` is empty, `1` is the tileset's first tile, `2` the next, and so on.
- The top three bits flip a tile as Tiled does: horizontally (2^31), vertically (2^30) and diagonally (2^29). Tiled's layer data works as is when the map uses one tileset starting at id 1. For a later tileset, subtract its `firstgid - 1` from the id and keep the flip bits.
- The tileset holds as many tiles as fit its grid: whole tiles across times whole tiles down, after `margin` and `spacing`.
- Zoomed out, a tilemap is filtered, not shimmering: it draws from a copy of its tileset with each tile in a cell of its own and its edges stretched out, mipped down to tiles two texels across, so no tile takes in its neighbour. The copy is made the first time the map is drawn, about four times the tileset's size. Tiles under four texels across, and tilesets whose copy would pass 8192 texels, are read as they are, with no mipmaps.

<a id="scene-settilemap"></a>
### `scene.setTilemap(node, changes)`

Changes a tilemap's options. Takes the same names as [`scene.addTilemap`](#scene-addtilemap). Its tiles are kept unless you pass `tiles`, and you must pass them if `columns` or `rows` changes.

Returns: nothing.
Throws: `setTilemap: this node has no tilemap`, or any error `addTilemap` throws for the merged options, named `setTilemap:`. Changing the size without new `tiles` throws the `tiles holds N ids` error.

```js
scene.setTilemap(map, { color: [0.6, 0.6, 0.8, 1] });
```

Notes: every call uploads the whole map again. To change tiles, use [`scene.setTile`](#scene-settile) or [`scene.setTiles`](#scene-settiles).

<a id="scene-tilemapof"></a>
### `scene.tilemapOf(node)` → `object | null`

A tilemap's current options, or `null` if the node has no tilemap.

Returns: a copy with `tileset`, `tileSize`, `margin`, `spacing`, `columns`, `rows`, `tiles` (a `Uint32Array`), `layer`, `color`, `pivot` and `lit`.

```js
const { columns, rows } = scene.tilemapOf(map);
```

Notes: `tiles` is a copy. Read single tiles with [`scene.tileAt`](#scene-tileat) and change them with [`scene.setTile`](#scene-settile).

<a id="scene-settile"></a>
### `scene.setTile(node, x, y, id)`

Sets the tile at column `x`, row `y`. `id` follows the same rules as `addTilemap`'s `tiles`, flip bits included.

Returns: nothing.
Throws: `setTile: this node has no tilemap`; `setTile: (x, y) is not on the C x R map` if `x` or `y` is not a whole number on the map; `setTile: tile N is not in the tileset…` as for `addTilemap`. A refused call writes nothing.

```js
scene.setTile(map, 12, 3, 5 | 0x80000000);   // tile 5, flipped horizontally
```

<a id="scene-settiles"></a>
### `scene.setTiles(node, x, y, width, tiles)`

Sets a block of tiles. `tiles` holds rows of `width` ids; the block's top-left is at column `x`, row `y`.

Returns: nothing.
Throws: `setTiles: this node has no tilemap`; `setTiles: a W-wide block of N at (x, y) is not inside the C x R map` if `x`, `y` or `width` is not a whole number, `width` is not above zero, `tiles.length` is not a multiple of `width`, or the block runs off the map; `setTiles: tile N is not in the tileset…` as for `addTilemap`. A refused call writes nothing.

```js
scene.setTiles(map, 10, 5, 3, [1, 1, 1,
                                 2, 2, 2]);
```

<a id="scene-tileat"></a>
### `scene.tileAt(node, x, y)` → `number`

The id at column `x`, row `y`, flip bits included. Fractional `x` and `y` are floored.

Returns: the id, or `0` for an empty tile or a position off the map.
Throws: `tileAt: this node has no tilemap`.

```js
const id = scene.tileAt(map, 12, 3) & 0x1fffffff;   // without the flip bits
```

<a id="camera2d"></a>
## Camera2D

A camera for 2D. One unit is one CSS pixel, y points down, and the origin is the top-left, as in an HTML canvas and CSS. A scene viewed through one is drawn by [the 2D view](#view2d-order).

<a id="camera2d"></a>
### `new Camera2D(options)`

Creates a 2D camera. Pass it to `engine.run` in place of a 3D camera, or as a HUD's camera.

| Option | Default | Meaning |
|---|---|---|
| `position` | `[0, 0]` | The world point placed at `pivot` of the view. |
| `pivot` | `[0, 0]` | Where in the view `position` sits: `[0, 0]` top-left, `[1, 1]` bottom-right. `[0.5, 0.5]` centres `position`, for a camera that follows something. |
| `zoom` | `1` | Screen pixels per world unit. 2 draws everything twice as large. |
| `angle` | `0` | Radians. Turns the view clockwise about its centre. |
| `background` | `[0, 0, 0, 1]` | What the view is cleared to, sRGB 0..1. |
| `pixelSnap` | `false` | For pixel art: puts every item's corner on a whole screen pixel, and makes a unit a whole number of screen pixels. See [`camera.pixelSnap`](#camera2d-pixelsnap). |
| `ambient` | `[0, 0, 0]` | The light on `lit` things where no light reaches, linear. See [lighting](#view2d-lighting). |

Throws: `Camera2D: anchor is now pivot -- the point of the view placed at position, as a sprite's pivot is` if `anchor` is passed.

```js
import { Camera2D } from 'winding-engine';

const camera = new Camera2D({ pivot: [0.5, 0.5], zoom: 3, pixelSnap: true });
const at = new Float32Array(3);
engine.run(scene, camera, { frame() { player.worldPosition(at); camera.position.set([at[0], at[1]]); } });
```

Notes: every option can be changed later through the property of the same name. The renderer reads them each frame.

<a id="camera2d-position"></a>
### `camera.position` → `Float32Array(2)`

The world point at `pivot` of the view. Write into it to scroll: `camera.position[0] += 4`.

<a id="camera2d-pivot"></a>
### `camera.pivot` → `Float32Array(2)`

Where in the view `position` sits, 0..1 from its top-left.

<a id="camera2d-zoom"></a>
### `camera.zoom` → `number`

Screen pixels per world unit.

<a id="camera2d-angle"></a>
### `camera.angle` → `number`

Radians, turning the view clockwise about its centre. The world on screen turns the other way.

<a id="camera2d-background"></a>
### `camera.background` → `Float32Array(4)`

The clear colour, sRGB 0..1. Ignored when the camera draws a [HUD](#view2d-hud), which is drawn over the frame without clearing.

<a id="camera2d-pixelsnap"></a>
### `camera.pixelSnap` → `boolean`

For pixel art. Puts the view's offset and each item's corner on whole screen pixels, so a texel never lands between them and nothing flickers as it moves. And makes a unit a whole number of screen pixels: `zoom` × the pixel ratio, rounded (at least 1). On a 1.5× screen, zoom 1 draws a texel as 2 screen pixels, not as 1 and 2 by turns. For an unrotated view.

Notes: this is the one exception to "a unit is a CSS pixel": at a fractional pixel ratio, a snapped view is drawn a little larger or smaller than the CSS size. `screenToWorld` and `pick` follow it.

<a id="camera2d-ambient"></a>
### `camera.ambient` → `Float32Array(3)`

Light on `lit` things where no light reaches, linear. Black shows lit things only where lights reach.

<a id="camera2d-matrices"></a>
### `camera.view`, `camera.projection`, `camera.viewProjection` → `Float32Array(16)`

Read-only matrices, recomputed by [`camera.update`](#camera2d-update). `view` maps world to canvas pixels, `projection` maps canvas pixels to clip space, and `viewProjection` is the two combined.

<a id="camera2d-size"></a>
### `camera.width`, `camera.height`, `camera.pixelRatio` → `number`

The canvas size in its own (device) pixels, and device pixels per CSS pixel, as of the last update. They start at `1`.

<a id="camera2d-is2d"></a>
### `camera.is2D` → `true`

Always `true`. The renderer and `scene.pick` use it to take the 2D path.

<a id="camera2d-update"></a>
### `camera.update(aspect, width, height, pixelRatio)` → `Camera2D`

Recomputes the matrices for a canvas `width` × `height` of its own pixels, `pixelRatio` of them to a CSS pixel. The renderer calls it every frame, so you rarely need to. `aspect` is ignored; it is there so the call matches a 3D camera's.

| Argument | Default | Meaning |
|---|---|---|
| `aspect` | — | Ignored. |
| `width` | `camera.width` | Canvas width in its own pixels. |
| `height` | `camera.height` | Canvas height in its own pixels. |
| `pixelRatio` | `camera.pixelRatio` | Canvas pixels per CSS pixel. |

Returns: the camera.

```js
camera.update(0, canvas.width, canvas.height, devicePixelRatio);
```

<a id="camera2d-screentoworld"></a>
### `camera.screenToWorld(x, y, out)` → `Float32Array(2)`

Where canvas pixel (`x`, `y`) is in the world, as of the last update.

| Argument | Default | Meaning |
|---|---|---|
| `x`, `y` | required | A point in the canvas's own pixels (device pixels, not CSS pixels). |
| `out` | a new `Float32Array(2)` | Where to write the result. |

Returns: `out`.

```js
const r = canvas.getBoundingClientRect();
const world = camera.screenToWorld((e.clientX - r.left) * camera.width / r.width,
                                   (e.clientY - r.top) * camera.height / r.height);
```

Notes: a pointer event gives CSS pixels. Scale them as above. [`scene.pick`](#scene-pick-2d) does this for you.

<a id="camera2d-worldtoscreen"></a>
### `camera.worldToScreen(x, y, out)` → `Float32Array(2)`

Where world point (`x`, `y`) lands on the canvas, in its own pixels, as of the last update.

| Argument | Default | Meaning |
|---|---|---|
| `x`, `y` | required | A world point. |
| `out` | a new `Float32Array(2)` | Where to write the result. |

Returns: `out`.

```js
const [sx, sy] = camera.worldToScreen(200, 120);   // canvas pixels
```

<a id="the-2d-view"></a>
## The 2D view

What a [`Camera2D`](#camera2d) draws: sprites, text, shapes, paths, tilemaps and particle emitters. There is no depth and none of the 3D passes (no shadows, fog, bloom or tonemapping).

<a id="view2d-order"></a>
### Painter's order

Everything draws by `layer`, lowest first. Within a layer, things draw in the order they were added. `setX` keeps a thing's place; to bring something to the front, give it a higher `layer`.

```js
scene.addTilemap({ tileset, tileSize: [16, 16], columns: 40, rows: 20, tiles, layer: 0 });
scene.addSprite({ texture: hero, layer: 1 });
```

Notes: kinds do not matter. A shape on layer 2 draws over a tilemap on layer 1, whichever was added first. Emitters ([`scene.addEmitter`](#scene-addemitter)) take a place in the same order by their `layer`.

<a id="view2d-colour"></a>
### sRGB colours

A 2D view composites in sRGB, as a browser composites a page. Colours you give (`color`, `stroke`, `background`) are sRGB 0..1, like CSS, and an opaque colour lands on screen exactly as authored. Half-transparent edges blend as they do in an image editor.

```js
scene.addShape({ size: [40, 40], color: [1, 0.5, 0, 1] });   // CSS rgb(255, 128, 0)
```

Notes: 3D colours are linear light. The same numbers look different in a 3D view and a 2D view. A colour texture loaded with the default `srgb: true` shows as it looks in an image viewer.

<a id="view2d-lighting"></a>
### Lighting: `lit` and ambient

By default nothing in a 2D view is lit: the colour is the colour. Anything with `lit: true` is instead multiplied by the camera's [`ambient`](#camera2d-ambient) plus every point and spot light in the scene ([`scene.addLight`](#scene-addlight)).

```js
const camera = new Camera2D({ ambient: [0.08, 0.08, 0.12] });
scene.addTilemap({ tileset, tileSize: [16, 16], columns, rows, tiles, lit: true });
scene.addLight({ position: [200, 120], color: [1, 0.8, 0.5], intensity: 2, radius: 160 });
```

Notes:
- A light's x and y place it; its z is ignored. `radius` is in world units (CSS pixels), where its light fades to nothing.
- A spot aims along its direction's x and y. Pass `direction: [x, y]`. A spot aimed straight out of the screen lights nothing.
- Directional lights do not light a 2D view.
- The lighting is done in linear light, so a white light at intensity 1 shows the colour exactly as authored.
- Every lit pixel tests every light. Dozens of lights are fine; hundreds are slow.

<a id="view2d-hud"></a>
### HUD

`engine.run(scene, camera, { hud: { scene, camera } })` draws a second, 2D scene over every frame ([`engine.run`](#engine-run)). It is drawn after tonemapping, bloom and antialiasing, so its colours land exactly and its text stays crisp. It works over a 3D view or a 2D one.

| Field | Default | Meaning |
|---|---|---|
| `scene` | required | A scene from `engine.createScene()`. |
| `camera` | a plain `Camera2D` | A `Camera2D` to view it through. |

Throws: `run: hud needs { scene, camera }, the scene from createScene()`; `run: the hud's camera must be a Camera2D`.

```js
const hud = engine.createScene();
const hudCamera = new Camera2D();
const health = hud.addShape({ size: [138, 10], radius: 5, color: [0.85, 0.47, 0.34, 1], pivot: [0, 0], position: [22, 38] });
engine.run(scene, camera, { hud: { scene: hud, camera: hudCamera } });
```

Notes:
- The HUD is not cleared; the camera's `background` is ignored.
- `run` advances the HUD scene with the main one.
- Emitters in a HUD scene are not drawn.
- Pass your own camera if you want to [pick](#scene-pick-2d) in the HUD; the default one is private.
- `engine.renderFrame(scene, camera, { hud })` takes the same object.

<a id="scene-pick-2d"></a>
### `scene.pick(camera2D, x, y, width, height)` → `{ node, point, tile? } | null`

The topmost thing a 2D view shows under a point on the canvas. Use it for clicks and hovers.

| Argument | Meaning |
|---|---|
| `camera2D` | The `Camera2D` the scene is drawn through. |
| `x`, `y` | CSS pixels from the canvas's top-left. |
| `width`, `height` | The canvas's CSS size. |

Returns: `{ node, point }`, where `point` is the world point `[x, y]`. For a tilemap it also has `tile: [column, row]`. `null` if nothing is hit.

```js
canvas.addEventListener('pointerdown', (e) => {
  const r = canvas.getBoundingClientRect();
  const hit = scene.pick(camera, e.clientX - r.left, e.clientY - r.top, r.width, r.height);
  if (hit?.tile) scene.setTile(hit.node, hit.tile[0], hit.tile[1], 0);
});
```

What counts as a hit, per kind, tested topmost first (the reverse of [painter's order](#view2d-order)):

| Kind | Hit where |
|---|---|
| Sprite | Anywhere in its quad, transparent pixels included. |
| Text | Anywhere in its block. |
| Shape | Inside its outline (its rounded rect or ellipse), even with a clear fill. |
| Path | Inside a closed path by the nonzero rule, even with a clear fill, or within half `strokeWidth` of its line. |
| Tilemap | On a tile that is not empty (id not 0). |

Notes:
- It uses the camera as of its last frame. Before the first frame, call [`camera.update`](#camera2d-update) yourself.
- Node transforms are brought up to date first, so a node moved since the last frame is hit where it is now.
- Emitters are never hit.
- With a 3D camera, `scene.pick` casts a ray instead; see [`scene.pick`](#scene-pick).

<a id="orbitcontroller"></a>
## OrbitController

<a id="orbitcontroller"></a>
### `new OrbitController(camera, element, options)` → `OrbitController`

Mouse and touch control for a 3D `Camera`: drag to orbit, wheel to zoom, right-drag or shift-drag to pan. It listens on `element` and only changes the camera, so it works with or without `engine.run`. It sets the element's CSS `touch-action` to `none`, so a touch browser gives it the finger instead of scrolling the page, and puts it back on `detach`.

| Option | Default | Meaning |
|---|---|---|
| `distance` | `6` | Distance from the target. |
| `yaw` | `0` | Angle around the vertical axis, radians. `0` looks from +Z. |
| `pitch` | `0.3` | Angle above the horizontal, radians. Kept just short of straight up or down, however it is set. |
| `target` | `[0, 0, 0]` | Point the camera orbits and looks at. |
| `minDistance` | `0.1` | Closest zoom. |
| `maxDistance` | `1000` | Farthest zoom. |
| `rotateSpeed` | `0.005` | Radians per pixel dragged. |
| `zoomSpeed` | `0.0015` | Zoom per wheel unit (exponential, so it feels the same near and far). |
| `panSpeed` | `0.002` | Pan per pixel, scaled by distance. |
| `damping` | `12` | How fast the camera eases to where it is going. Higher is snappier. |

Each option is also a writable field of the same name. `yaw`, `pitch`, `distance` and `target` are where the camera is now; `desired` (`{ distance, yaw, pitch, target }`) is where it is easing to. To move the camera from code, write `desired`.

```js
const orbit = new OrbitController(camera, canvas, { distance: 4, target: [0, 1, 0] });
orbit.desired.yaw += Math.PI / 2;   // eases a quarter turn
```

Notes: the camera is placed immediately. The controller owns the camera's position: anything that moves the camera directly is overwritten on the next `update`, unless you call [`syncFromCamera`](#orbitcontroller-syncfromcamera). It does nothing while the camera follows a node.

<a id="orbitcontroller-update"></a>
### `orbit.update(dt)` → `OrbitController`

Eases the camera toward `desired` and writes its position and target. Call once per frame with the frame's time in seconds. `0` snaps straight there. If you never call it, input still works but snaps instead of easing.

```js
engine.run(scene, camera, { frame: (alpha, clock) => orbit.update(clock.realDelta) });
```

<a id="orbitcontroller-syncfromcamera"></a>
### `orbit.syncFromCamera()` → `OrbitController`

Adopts the camera's current position and target, instead of overwriting them. Call it after moving the camera yourself (a cutscene, a saved view, `camera.frameBounds`).

```js
camera.position.set([4, 3, 4]);
orbit.syncFromCamera();
```

Notes: takes effect at once, with no easing. A pose past the pitch or distance limits is clamped to the nearest one the controller can hold.

<a id="orbitcontroller-framebounds"></a>
### `orbit.frameBounds(min, max, { margin })` → `OrbitController`

Points at the centre of an axis-aligned box (`[x, y, z]` corners) and backs off until it fits the view. `margin` defaults to `1`, an exact fit; larger leaves more room. Snaps at once.

```js
orbit.frameBounds([-1, 0, -1], [1, 2, 1], { margin: 1.2 });
```

<a id="orbitcontroller-dragged"></a>
### `orbit.dragged` → `boolean`

`true` when the last press moved more than 3 pixels, so it was a drag, not a click. Read it in a click handler to ignore the click that ends an orbit.

```js
canvas.addEventListener('click', (event) => {
  if (orbit.dragged) return;
  pick(event);
});
```

<a id="orbitcontroller-detach"></a>
### `orbit.detach()` → `void`

Removes the controller's event listeners from `element`. The camera stays where it is.

<a id="statsoverlay"></a>
## StatsOverlay

<a id="statsoverlay"></a>
### `new StatsOverlay(engine, options)` → `StatsOverlay`

A small fixed box in the page's bottom-left corner showing fps, canvas size, object and draw counts, pipeline count, render graph passes, and CPU time per phase.

| Option | Default | Meaning |
|---|---|---|
| `interval` | `0.5` | Seconds between refreshes. |
| `parent` | `document.body` | Element the box is added to. |

The box is `overlay.element`, a `<div>` you can restyle.

```js
const stats = new StatsOverlay(engine);
engine.run(scene, camera, { frame: (alpha, clock) => stats.update(clock.realDelta) });
```

Notes: `pipelines` should stop climbing once loading is done. If it keeps rising, shaders are compiling mid-frame.

<a id="statsoverlay-update"></a>
### `overlay.update(dt)` → `void`

Adds `dt` seconds and refreshes the text once `interval` has passed. Call once per frame.

<a id="statsoverlay-destroy"></a>
### `overlay.destroy()` → `void`

Removes the box from the page.

<a id="benchmark"></a>
## Benchmark

Opt-in timing. It lives in its own module, never imported by the engine, and until it attaches the
renderer's profiling hook costs a null check.

```js
import { Benchmark } from 'winding-engine/bench.js';
```

<a id="benchmark"></a>
### `new Benchmark(engine)`

A benchmark for one engine. It records every CPU phase of a frame, every GPU pass where the device
has timestamp queries, and the wall time.

<a id="benchmark-run"></a>
### `benchmark.run(scene, camera, options)` → `Promise<report>`

Renders `frames` frames, one at a time, and reports on them. It waits for the GPU after every
frame, which is what makes the wall time mean something; it measures a frame, not the throughput of
a pipelined loop.

| Option | Default | Meaning |
|---|---|---|
| `frames` | `300` | frames to record |
| `warmup` | `30` | frames rendered first and not recorded: the first frames compile pipelines and grow buffers |
| `update` | `null` | `update(i)`, run untimed before each frame, to move the camera or animate the scene |

Returns: a report, as [`benchmark.report()`](#benchmark-report) describes.

```js
const report = await new Benchmark(engine).run(scene, camera, { frames: 300 });
console.log(Benchmark.format(report));
```

<a id="benchmark-start"></a>
### `benchmark.start()` and `benchmark.stop()`

Record every frame the engine renders, however it's driven — your own loop or `engine.run` --
between the two. Starting again while running starts the recording over. `stop()` puts GPU timing
back as it was before the first `start()`.

<a id="benchmark-report"></a>
### `benchmark.report()` → `report`

Everything recorded so far, summarised, as plain data safe to `JSON.stringify`:

```js
{ frames,
  cpu:  [{ name, mean, median, p95, max, share }],   // in frame order; they sum to the frame
  gpu:  [{ name, mean, share }] | null,              // null without timestamp queries
  wall: { mean, median, p95, max } | null }          // null unless from run()
```

Times are milliseconds. `share` is the fraction of the frame the phase or pass took.

<a id="benchmark-format"></a>
### `Benchmark.format(report)` → `string`

A report as a table to print, ending with what the frame is bound by: CPU, GPU, or waiting for the
display.

<a id="environment-and-hdr"></a>
## Environment and HDR

<a id="environment"></a>
### `new Environment(gpu, options)` → `Environment`

A baked lighting environment: diffuse ambient light, reflections, and the background. Baked once, from a procedural sky or from an equirectangular HDR map. Most code gets one from [`engine.loadEnvironment`](#engine-loadenvironment) or uses `engine.environment`; construct one directly for a custom sky.

| Option | Default | Meaning |
|---|---|---|
| `size` | `128`; with a map, the power of two at or below map width / 4 | Edge of the sky cube and reflection cube, in texels. A power of two. |
| `irradianceSize` | `32` | Edge of the diffuse-light cube. A power of two. |
| `prefilterMips` | `6` | Roughness levels for reflections, capped by the cube's mip count. |
| `label` | `'env'` | GPU label prefix. |
| `sky` | the default sky | Procedural sky settings, merged over the defaults below. Ignored when `map` is given. |
| `map` | `null` | An equirectangular panorama, `{ width, height, data }` with `data` as linear RGB floats, as [`parseHDR`](#parsehdr) returns. Its centre faces +X and its top row is straight up. |

`sky` fields (all colours linear):

| Field | Default | Meaning |
|---|---|---|
| `ground` | `[0.10, 0.09, 0.08]` | Colour below the horizon. |
| `horizon` | `[0.62, 0.66, 0.74]` | Colour at the horizon. |
| `zenith` | `[0.16, 0.30, 0.60]` | Colour straight up. |
| `sun` | `[0.35, 0.55, 0.45]` | Direction toward the sun disc (normalised for you). |
| `sunColor` | `[1.0, 0.93, 0.80]` | Colour of the disc and its glow. |
| `sunIntensity` | `60` | Brightness of the disc. `0` removes it. |
| `glow` | `0.5` | Brightness of the halo around the disc. `0` removes it. |

Properties: `size` (cube edge), `prefilterMips`, `sky` (the merged settings). Method: `destroy()`.

Throws: `'Environment: map needs width, height and width * height * 3 floats of RGB'`; `RangeError` `'Environment: a WxH map is past this device's N'`; `RangeError` `'Environment: size N is past this device's M'`.

```js
import { Environment } from 'winding-engine';

const dusk = new Environment(engine.gpu, {
  sky: { zenith: [0.05, 0.08, 0.2], sun: [0.9, 0.1, 0], sunIntensity: 30 },
});
const scene = engine.createScene({ environment: dusk });
```

Notes: the sky lights the scene only through ambient light and reflections; its sun disc casts no shadow. For sunlight and shadows, add a directional light shining along the opposite of `sun`. Settings are read once at bake time; changing `sky` afterwards does nothing. An environment belongs to the engine whose `gpu` made it. Free it with [`engine.unload`](#engine-unload) once no scene uses it.

<a id="parsehdr"></a>
### `parseHDR(bytes, { maxDimension })` → `{ width, height, data }`

Decodes a Radiance `.hdr` (RGBE) file to linear RGB floats, top row first. [`engine.loadEnvironment`](#engine-loadenvironment) calls it for you; use it directly to inspect or edit a map before building an [`Environment`](#environment).

`bytes` is the file, as a `Uint8Array` or an `ArrayBuffer`.

| Option | Default | Meaning |
|---|---|---|
| `maxDimension` | `Infinity` | Refuse a map wider or taller than this before decoding it. Pass `engine.gpu.limits.maxTextureDimension2D`. |

Returns: `{ width, height, data }`, with `data` a `Float32Array` of 3 floats a pixel. Any `EXPOSURE=` header is divided out.
Throws: `'hdr: not a Radiance file…'`; `'hdr: the header never ends'`; `'hdr: FORMAT=… is not supported; only 32-bit_rle_rgbe is'`; `'hdr: EXPOSURE=… is not a positive number'`; `'hdr: only -Y h +X w and +Y h +X w orientations are supported'`; `'hdr: a WxH image has no pixels'`; `'hdr: a WxH map is past this device's N'`; `'hdr: the pixel data ends early'`; `'hdr: a run is longer than its scanline'`; `'hdr: a repeat with no pixel before it'`.

```js
const bytes = new Uint8Array(await (await fetch('sky.hdr')).arrayBuffer());
const map = parseHDR(bytes, { maxDimension: engine.gpu.limits.maxTextureDimension2D });
const sky = new Environment(engine.gpu, { map, size: 256 });
```


<a id="colour-helpers"></a>
## Colour helpers

3D colours in Winding are linear light. Colour pickers, hex codes and CSS values are sRGB. These convert sRGB to linear for 3D. 2D colours (anything seen through a `Camera2D`) are already sRGB and need no conversion. Alpha is never converted.

A light's colour may exceed 1: convert the colour, then multiply by the brightness you want.

<a id="color-srgbtolinear"></a>
### `srgbToLinear(c)` → `number`

One channel, sRGB (0–1) to linear.

```js
srgbToLinear(0.5);   // 0.214
```

<a id="color-lineartosrgb"></a>
### `linearToSrgb(c)` → `number`

One channel, linear back to sRGB (0–1).

```js
linearToSrgb(0.214);   // 0.5
```

<a id="color-colorfromhex"></a>
### `colorFromHex(hex)` → `[r, g, b, a]`

A hex colour as linear RGBA. Takes 3, 4, 6 or 8 digits, with or without `#`. Alpha is `1` when not given.

Throws: `'colorFromHex: "<hex>" is not a 3, 4, 6 or 8 digit hex colour'`, for a number as well: `0xff0000` is written `'#ff0000'`.

```js
colorFromHex('#e03a2f');     // [0.745, 0.042, 0.028, 1]
colorFromHex('#e03a2f80');   // alpha 0.502
```

<a id="color-colorfrombytes"></a>
### `colorFromBytes(r, g, b, a)` → `[r, g, b, a]`

0–255 sRGB channels, as a colour picker gives them, as linear RGBA. `a` defaults to `255`.

```js
const sun = colorFromBytes(255, 240, 220).slice(0, 3).map((c) => c * 3);
```

Notes: returns a plain array, so it survives `JSON.stringify`.

<a id="math"></a>
## Math

Exported from `'winding-engine'`. Plain functions over typed arrays. Every function that produces a vector, quaternion or matrix writes into `out` first and returns it. Passing the same array as `out` and an input is safe. Only the `*Create` functions allocate.

<a id="math-vec3"></a>
### vec3

A vec3 is a `Float32Array(3)` (any array of 3 numbers works as input).

| Function | What it does |
|---|---|
| `vec3Create(x = 0, y = 0, z = 0)` | A new `Float32Array(3)`. Allocates. |
| `vec3Set(out, x, y, z)` | Sets `out` to `x, y, z`. |
| `vec3Copy(out, a)` | Copies `a` into `out`. |
| `vec3Add(out, a, b)` | `a + b`. |
| `vec3Sub(out, a, b)` | `a - b`. |
| `vec3Mul(out, a, b)` | Component-wise `a * b`. |
| `vec3Scale(out, a, s)` | `a * s`. |
| `vec3ScaleAndAdd(out, a, b, s)` | `a + b * s`. |
| `vec3Negate(out, a)` | `-a`. |
| `vec3Dot(a, b)` | The dot product. Returns a number. |
| `vec3Cross(out, a, b)` | The cross product `a × b`. |
| `vec3LengthSq(a)` | Squared length. Cheaper than `vec3Length` for comparisons. |
| `hypot3(x, y, z)` | `sqrt(x² + y² + z²)`: a faster `Math.hypot` for three numbers. |
| `vec3Length(a)` | Length. |
| `vec3DistanceSq(a, b)` | Squared distance between two points. |
| `vec3Normalize(out, a)` | Unit length. A zero vector gives zero, not NaN. |
| `vec3Lerp(out, a, b, t)` | Linear blend: `a` at 0, `b` at 1. |
| `vec3Min(out, a, b)` | Component-wise minimum. |
| `vec3Max(out, a, b)` | Component-wise maximum. |
| `vec3TransformMat4(out, a, m)` | Transforms a point (w = 1), with perspective divide. |
| `vec3TransformMat4Dir(out, a, m)` | Transforms a direction (w = 0): no translation. |
| `vec3TransformQuat(out, a, q)` | Rotates by a quaternion. |

<a id="math-quat"></a>
### quat

A quaternion is a `Float32Array(4)`, `[x, y, z, w]`.

| Function | What it does |
|---|---|
| `quatCreate()` | A new identity quaternion. Allocates. |
| `quatIdentity(out)` | Sets `out` to the identity. |
| `quatCopy(out, a)` | Copies `a` into `out`. |
| `quatSetAxisAngle(out, axis, rad)` | A turn of `rad` radians about a unit `axis`, right-hand rule. |
| `quatMultiply(out, a, b)` | `a * b`: `b` applies first, then `a`. |
| `quatDot(a, b)` | The 4D dot product. Returns a number. |
| `quatConjugate(out, a)` | The conjugate: the inverse, for a unit quaternion. |
| `quatNormalize(out, a)` | Unit length. A zero quaternion gives the identity. |
| `quatFromEuler(out, yaw, pitch, roll)` | From radians, YXZ order: roll about Z, then pitch about X, then yaw about Y. There is no inverse. |
| `quatFromMat4(out, m, mOff = 0)` | The rotation of a matrix's upper 3×3, which must have no scale. |
| `quatSlerp(out, a, b, t)` | Spherical blend at constant angular speed, the short way round. |
| `quatFromTo(out, from, to)` | The shortest turn taking unit vector `from` onto unit vector `to`. |
| `quatLookAlong(out, direction, up = [0, 1, 0])` | Points -Z along `direction` with +Y as upright as it can be. `direction` need not be unit length. |

<a id="math-mat4"></a>
### mat4

A matrix is a column-major `Float32Array(16)`. The optional `*Off` arguments address a matrix or vector inside a larger array.

| Function | What it does |
|---|---|
| `mat4Create()` | A new identity matrix. Allocates. |
| `mat4Identity(out)` | Sets `out` to the identity. |
| `mat4Copy(out, a, outOff = 0, aOff = 0)` | Copies `a` into `out`. |
| `mat4GetTranslation(out, m)` | Writes `m`'s translation into vec3 `out`. |
| `mat4Multiply(out, a, b, outOff = 0, aOff = 0, bOff = 0)` | `a * b`: applied to a vector, `b` happens first. |
| `mat4MultiplyAffine(out, a, b, outOff = 0, aOff = 0, bOff = 0)` | `a * b` for matrices whose bottom row is `0, 0, 0, 1` (translation, rotation, scale). Faster; wrong for projections. |
| `mat4FromQuatPosScale(out, q, pos, scale, outOff = 0, qOff = 0, posOff = 0, scaleOff = 0)` | Builds translation × rotation × scale. |
| `mat4Invert(out, a)` | The inverse. Returns `null`, leaving `out` untouched, if the determinant is exactly zero. |
| `mat4LookAt(out, eye, center, up)` | A right-handed view matrix looking from `eye` toward `center`, down its -Z. |
| `mat4NormalMatrix(out, m, outOff = 0, mOff = 0)` | The inverse transpose of the upper 3×3, written as a WGSL `mat3x3` (3 columns padded to 4 floats). Returns `false` for a singular matrix, else `true`. |
| `mat4Decompose(outPos, outRot, outScale, m, mOff = 0)` | Splits into translation, rotation and scale. Shear is lost. Returns `false`, leaving the outputs untouched, if an axis has zero scale. |
| `mat4OrthographicReverseZ(out, left, right, bottom, top, near, far)` | An orthographic projection for WebGPU, reverse-Z: `near` maps to depth 1, `far` to 0. |
| `mat4PerspectiveReverseZInfinite(out, fovYRadians, aspect, near)` | A perspective projection for WebGPU, reverse-Z with no far plane. Use with `depthCompare: 'greater'` and a depth clear of 0. |

<a id="math-aabb"></a>
### aabb

A box is a pair of vec3s, `min` and `max`. An empty box has `min` at `+Infinity` and `max` at `-Infinity`.

| Function | What it does |
|---|---|
| `aabbTransform(outMin, outMax, min, max, m, mOff = 0, outOff = 0, inOff = 0)` | The smallest axis-aligned box holding the box transformed by `m`. An empty box stays empty. Returns nothing. |
| `aabbBoundingSphere(outCenter, min, max)` | Writes the box's centre into `outCenter` and returns the radius of the sphere around it. |
| `aabbUnion(min, max, otherMin, otherMax)` | Grows `min`/`max` in place to hold the other box. Returns nothing. |
| `aabbSetEmpty(min, max)` | Makes the box empty. Returns nothing. |
| `aabbRayDistance(min, max, origin, direction, boundsOff = 0)` | Distance along a ray to where it enters the box: `0` if the origin is inside, `-1` for a miss. Does not check the ray is finite. |
| `rayTriangleDistance(origin, direction, positions, a, b, c)` | Distance along a ray to a triangle, from either side, or `-1` for a miss. `a`, `b`, `c` index flat xyz `positions`, so pass `index * 3`. The distance is in units of `direction`'s length. |

<a id="renamed"></a>
## Renamed in 1.0

Names that changed in 1.0, for code written against 0.x. A renamed option, and each of the four properties with an entry here, throws saying what it is called now; a renamed method is simply gone, and calling it says it is not a function.

| Was | Is |
|---|---|
| `engine.run({ scene, camera, update, frame, overlay })` | [`engine.run(scene, camera, { update, frame, hud })`](#engine-run) |
| `overlay` (in `run` and `renderFrame`) | `hud` |
| `engine.stats.overlay2D`, `overlay2DWritten` | [`hudSprites`, `hudSpritesWritten`](#engine-stats) |
| a sprite's `rotation` | [`angle`](#scene-addsprite) |
| `addReflectionProbe`, `setReflectionProbe`, `reflectionProbeOf`, `engine.captureReflectionProbes` | [`addProbe`](#scene-addprobe), [`setProbe`](#scene-setprobe), [`probeOf`](#scene-probeof), [`engine.captureProbes`](#engine-captureprobes) |
| a probe's `min` and `max`; its `blend` | `position` and `size`; `fade` |
| `play({ add })` | [`play({ join })`](#node-play) |
| `scene.advanceAnimations(dt)`, `scene.advanceParticles(dt)` | [`scene.advance(dt)`](#scene-advance) |
| `node.setRotationAxisAngle`, `node.setRotationEuler`, `node.getWorldPosition` | [`setAxisAngle`](#node-setaxisangle), [`setEuler`](#node-seteuler), [`worldPosition`](#node-worldposition) |
| `node.setLight(changes)` | [`scene.setLight(node, changes)`](#scene-setlight) |
| `scene.playerFor(node)` | [`node.animation`](#node-animation) |
| text's `anchor`; Camera2D's `anchor` | `pivot`, from the top-left |
| an emitter's `size: [birth, death]` | `size` and `sizeEnd` |

<a id="engine-rhi"></a>
### `engine.rhi`

Now [`engine.gpu`](#engine-gpu). Reading or assigning `engine.rhi` throws `'engine.rhi is now engine.gpu'`.

<a id="camera2d-rotation"></a>
### `camera.rotation` (Camera2D)

Now [`camera.angle`](#camera2d-angle). Reading or assigning it throws.

<a id="renderer-drawskybox"></a>
### `engine.renderer.drawSkybox`

Now [`engine.renderer.skybox`](#renderer-skybox). Reading or assigning it throws.

<a id="post-requestedlevels"></a>
### `engine.renderer.post.requestedLevels`

Now [`engine.renderer.post.levels`](#post-levels); what the last frame drew is `post.levelsDrawn`. Reading or assigning it throws.
