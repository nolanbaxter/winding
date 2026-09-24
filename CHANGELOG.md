# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Faster math where it runs per object, same answers.** Each replacement
  is checked against the code it replaced -- bit for bit where it can be,
  float32 where the inputs are -- and measured in place, old and new
  alternating:
  - World matrices compose with an affine multiply (36 multiplies instead of
    64; every matrix built from translation, rotation and scale is affine):
    359 -> 302 us for 10,000 transforms with 1,000 moving roots. Skin
    palettes use it too, when the file's inverse binds are affine, which is
    checked at load rather than assumed.
  - The box transform behind every moved object's bounds is unrolled:
    45 -> 28 us for 1,700.
  - Math.hypot, ten times slower than a square root of squares in V8 for a
    guard float32 data never needs, is gone from all 22 call sites.
  - Slerp takes a square root where it took sin(acos(c)).
  - Sort-key checks build their messages only when they fail; they were
    formatted on every call, 33 us per 1,700 keys, with DEBUG shipped on.

- **Nothing standing on a floor cast a sun shadow.** Each cascade's light
  view has its eye at the world origin, and its near plane was clamped to at
  least 0.01 in front of it -- a perspective habit an orthographic box does
  not need. Everything on the sun's side of the origin was clipped out of
  the map, which for a floor at y = 0 under an overhead sun is every object
  on it. Shadows appeared only for geometry below the origin, so whether a
  scene had them depended on where it sat. No test checked that a shadow is
  ever cast; one does now, on the GPU, and a Node check covers the depth
  range. Found by two audits independently.
- **Glossy highlights kept as little as 5% of their light.** GGX guarded its
  divisor with a floor of 1e-7, above what a smooth lobe divides by, so
  every surface smoother than roughness 0.116 had its highlight capped: at
  the roughness floor the peak was 41 instead of 77,625, and a smoother
  surface came out DIMMER than a rougher one. The floor is now far below
  anything the clamped roughness reaches. The shader's output is also held
  to 65504, the half-float target's largest value, which such a highlight
  now exceeds. GPU check: a smoother highlight is brighter (fails with the
  old floor).
- **Shadows crawled when the camera turned.** The default shadow range was
  the scene's farthest point along the VIEW direction, recomputed every
  frame, so turning in place resized every cascade (15.0 -> 16.0 over three
  degrees) and moved the texel-snapping grid with it. The range is now
  radial -- from where the camera is, not where it looks -- and rounded up to
  a power of two, so walking changes it only at the doublings; the cost is
  up to twice the resolution-limiting distance, and pinning shadowDistance
  trades it back. The slice sphere's radius is now computed in the camera's
  own frame, so turning leaves it identical to the last bit rather than
  within a rounding error. Light clustering keeps the tight view depth.
- **Destroying nodes could corrupt other nodes.** The scene kept a second
  record of the hierarchy, a map of children, next to the transform store's
  parent links, and it went stale: createNode({ parent }), add(asset,
  { parent }) and setParent never wrote to it, and remove() never unlinked a
  removed child from its parent's list. So destroying a child and then its
  parent threw halfway through, or -- if the slot had been reused -- wiped an
  unrelated node's transform; children made those three ways survived their
  parent and jumped to the origin; and a node reparented away died with its
  old parent. The map is gone and children are derived from the parent
  links, listed in the order they were added.
- **Destroying a node twice** could erase whatever reused its slot. A dead
  node is now ignored.
- **A failed add() broke every later one.** A skin naming a joint outside the
  default scene threw after the walk, leaving the half-built instance in the
  scene with no handle and its pending skin to throw again on every later
  add(). add() is now all or nothing.
- **Removed assets kept their skin palettes and morph weights**, which were
  still multiplied and uploaded every frame -- a hundred add/remove cycles,
  a hundred palettes. Each is now owned by its node and removed with it.
- **The glTF importer trusted what it was given.** It now refuses, naming the
  rule broken, what used to load into something quietly wrong:
  - light colours that are not three numbers (they put NaN in the light
    buffers), negative intensity or range, spot cones outside
    0 <= inner < outer <= pi/2, unknown light types, a malformed
    KHR_lights_punctual block;
  - node light or camera indices, and primitive material indices, that name
    nothing (lights and cameras vanished; a material fell back to whatever
    was registered first);
  - cameras whose near and far planes cannot make a view (they loaded, then
    failed every frame);
  - a negative or non-numeric emission strength;
  - a bufferView longer than its buffer (it read on into the next chunk as
    vertex data), in both the accessor and the image readers;
  - a COLOR_0 shorter than POSITION, and an index count that is not a whole
    number of triangles (tangents wrote NaN into the leftover vertex);
  - animation channels targeting a node that does not exist, unknown
    interpolation, and keyframe times that do not strictly increase.
- **Lights without a range were cut off at 40% of their visible reach.** The
  radius assumed radiance reaches the display linearly; through sRGB and the
  ACES tonemap the cut still showed. It is now derived from both curves, so
  the fade that ends a light never moves a pixel by half an 8-bit step.

## [0.9.1] - 2026-09-24

**Faster, measured.** No API changes. Every number below comes from the
opt-in benchmark, with old and new alternating in one page -- runs a minute
apart differ by more than most of these changes.

### Changed

- **Lights find their clusters, instead of every cluster testing every
  light.** Assignment was cells x lights sphere tests however small the
  lights, 14% of the GPU frame on a 200-light scene. Now one workgroup per
  light projects its bounding box to the tiles and depth slices it can
  reach and runs the same exact sphere-box test on those alone, appending
  with an atomic. The lists are the same -- a new GPU suite step checks
  every cell's list against a brute force (70,668 pairs) -- and the image
  differs from before by rounding only (17 pixels, one level each).
  Measured in one page, alternating:

  | GPU ms for assignment      | before | now   |
  |----------------------------|--------|-------|
  | 200 lights, 1,700 objects  | 0.31   | 0.07  |
  | 1,000 small lights         | 1.75   | 0.18  |
  | 1 light filling the view   | 0.035  | 0.17  |

  The last row is the price: a handful of huge point lights now costs up to
  0.14ms more. Four shapes were measured and this one's worst case is the
  least bad; the others lost up to 1.65ms somewhere. Also: in a cell past
  its 64-light cap, which lights drop can now vary between frames.

- **Blended objects that draw alike are one instanced call.** Transparent
  geometry was one draw per object, on the reasoning that instancing would
  lose the back-to-front order. Not for NEIGHBOURS in that order: a GPU
  blends a draw's instances in sequence, and the draw data is already laid
  out sorted, so a run of adjacent objects with the same mesh, material and
  variant is one call of that many instances -- the same image. Measured on
  100 copies of the demo scene (272 visible panes), alternating blocks in
  one page: 272 calls became 1, the late forward pass went 0.55ms -> 0.35ms
  of GPU time and the whole GPU frame 2.17ms -> 1.95ms. CPU time did not
  move: Chrome's JavaScript side of a draw call is cheaper than the timer.
  Buffers are also no longer rebound for every object.

- **The render graph compiles only when the frame changes.** It was
  re-deriving the order, the live passes, every load and store op and the
  texture aliasing on every frame, for a frame that is almost always the
  same one. `compile()` now writes a signature of the declaration -- every
  pass's attachments, reads, writes and clears, every resource's kind and
  descriptor -- and reuses the last result when it matches. Nothing is
  cached by remembering to invalidate: the signature is everything compile
  reads, so a resize, a toggle or a new pass cannot match. Measured on the
  demo scene with the benchmark, cache on and off in alternating 100-frame
  blocks of one page, eight each: graph build 0.28ms -> 0.20ms with no
  overlap between the blocks, CPU frame 1.06ms -> 0.97ms. 989 of 990 frames
  reused the compile.

## [0.9.0] - 2026-09-24

**Lights and cameras you attach -- from code or from the file -- a sun that is
not special, a camera without perspective, and a way to measure all of it.**

Breaking: `addLight` returns a node, and the index-based light calls are
gone. A light index was never stable -- removal swap-compacts the array -- so
anything holding one could end up pointing at a different light. And
`scene.sun` is a node too: `scene.sun.direction.set(...)` and
`scene.sun.color.set(...)` become `scene.sun.setDirection(x, y, z)` and
`scene.sun.setLight({ color, intensity })`.

### Changed

- **The sun is a directional light, and not a special one.** It was a plain
  `{ direction, color }` object the renderer read, the one light that was
  not a node and the only kind there could be one of. Now `addLight({ type:
  'directional' })` makes a node like any other light, and EVERY directional
  light lights the scene. The one thing only one can have is the shadow map,
  and it goes to the brightest by luminance -- the shadow you would see --
  so `scene.sun` is derived from the lights there are, not a slot to fill. A
  new scene still starts with one, as a light node you can aim, recolour,
  parent (a day cycle is one rotating node) or destroy.
- **GPU pass timing is off by default** (`gpuTiming: false`). It stamped
  every pass of every frame for a result nothing in the engine read. It can
  now be switched at runtime (`renderer.gpuTiming.enabled`), which is what
  the benchmark does for the length of a run.
- **Lights are scene nodes.** `scene.addLight(...)` returns a `Node` and
  accepts a `parent`, so a light on a moving object follows it with no
  per-frame bookkeeping -- the viewer example used to call
  `setLightPosition` on twelve lights every frame. A spot aims down its
  node's -Z (glTF `KHR_lights_punctual`), so it turns with its parent too.
  `setLightPosition`, `setLightColor` and `removeLight(index)` are
  removed: move the node, call `light.setLight({ color, intensity, radius,
  innerAngle, outerAngle })` for the rest, and `light.destroy()` to remove
  it. Destroying a parent takes its lights with it.

### Added

- **Orthographic camera.** `new Camera({ orthographic: true })`. What it
  shows is derived from the distance to its target -- the height a
  perspective camera with the same `fovY` sees there -- so orbit zoom,
  `frameBounds`, `syncFromCamera` and picking need nothing new, and
  switching projection keeps the model the same size. It is the one camera
  with a `far` (default 1000, linear depth, so generosity is free), which
  `frameBounds` pushes out if what it frames would not fit. A perspective
  camera still has no `far` at all.
- `quatFromTo(out, from, to)`, the shortest rotation between two
  directions.
- **`camera.follow(node)`.** The camera rides a node: position and aim come
  from its world transform, looking down -Z with +Y up. A chase camera is a
  node parented to the car; first person is one parented to the head. The
  node is in charge -- an `OrbitController` on the camera stands aside while
  it follows; `follow(null)` and `syncFromCamera()` hand control back. A
  destroyed node ends the follow and leaves the camera where it was.
- **glTF lights.** `KHR_lights_punctual` point and spot lights import onto
  their nodes, so a lamp animated in Blender plays with the clip. The
  engine's lights already matched the spec -- candela, half-angle cones, -Z,
  and the same windowed falloff -- so `range` is the radius. An absent range
  (infinite, per the spec) becomes the distance where the light falls below
  one 8-bit step on a white surface: `sqrt(256 I / pi)`. Directional lights
  come in too and join the scene's others. A file that lists the extension as
  required now loads.
- **Benchmarking, opt-in** (`src/bench.js`, not imported by the engine).
  `new Benchmark(engine).run(scene, camera)` times every CPU phase of a
  frame -- thirteen of them, which partition the frame so they sum to its
  total -- every GPU pass, and the wall time, and `Benchmark.format` says
  whether the frame was CPU-bound, GPU-bound or waiting for the display.
  Until a benchmark attaches, the renderer's hook is a null check per phase.
- **`node.setDirection(x, y, z)`** faces a node's -Z along a direction,
  upright, and **`quatLookAlong`** is the rotation behind it. Upright
  matters: the shortest turn, which `addLight({ direction })` used, rolls
  the node when the target is up and to the side at once.
- **`KHR_materials_emissive_strength`.** Blender writes it for any emission
  strength above 1; ignored, those glows came out dimmer with nothing said.
- **glTF cameras.** Perspective and orthographic cameras import as
  `scene.cameras`, each already following its node. `zfar` is dropped for
  perspective (there is no far plane) and the canvas decides the aspect;
  orthographic `ymag` is honoured exactly.

### Fixed

- **Removing a renderable handed its skin and morph to another one.** The
  swap-remove moved entity, material and bounds down into the gap but left
  the skin and morph columns behind, so the survivor wore the deleted object's
  skin palette and morph weights.

Each of these was invisible under perspective and would have broken the
orthographic camera on arrival.

- **Froxels assumed every ray passes through the eye.** The cluster builder
  scaled one near-plane point out to each slice. Under ortho an off-centre
  froxel landed nowhere near its cell, and lights there lit nothing. Now two
  unprojections of the same pixel, correct under any projection.
- **Fragment view depth came from `clip.w`**, which is the view depth only
  because a perspective matrix has -1 in its w row. Under ortho it is 1
  everywhere, so every fragment read the lights and shadow cascade for depth
  1. Now measured along the camera's view axis.
- **Shadow cascades were fitted as a pyramid**, leaving the near corners of an
  orthographic view outside every cascade.
- **The frustum extractor rejected any finite projection.** Leaving the far
  plane out only ever keeps more, and the rasteriser clips it, so it is left
  out for ortho too rather than asserted away.
- **The GPU feature matrix read the wrong pixel.** It indexed a 64x64 canvas
  the engine had resized to its 320x240 CSS box, so "the middle" was near the
  top-left corner. The quad fills the view, so every check passed anyway, and
  the one lighting check passed only because its light's radius reached the
  corner. It now addresses pixels by fraction of the real size, and a new
  check lights an off-centre spot under ortho -- verified to fail against
  either of the first two fixes reverted.

## [0.8.0] - 2026-09-23

**Numbers you used to have to guess.** A camera distance, and a colour in a
space nobody tells you about. Both were derivable all along.

Minor rather than patch because two behaviours change: orbiting tilts the
other way, and `OrbitController.frameBounds` takes `margin` where it took
`fill`, with a different default.

### Fixed

- **Orbiting tilted the wrong way.** `clientY` grows downward, so a drag
  upward is a negative delta -- and the pitch line subtracted it, raising the
  camera. The yaw line directly above subtracts its delta, which sends the
  camera the other way and turns the object with your hand.

  So horizontal was "grab the object" and vertical was "grab the camera": two
  conventions in one gesture, which reads as the model being hinged behind
  itself. Grab the front of a ball and pull up and its underside rotates
  toward you, so the camera goes down. Reported from a page someone was
  actually using, which is the only way this was ever going to be found.

- **`OrbitController.frameBounds` left the aspect out**, so a wide object on
  a portrait viewport ran off both sides -- by nearly a factor of two. It
  computed the distance inline rather than sharing the one definition.

- **The package shipped whatever was lying around.** `files` listed
  `examples` and `DESIGN.md`, both of which `.gitignore` says in as many
  words are "Not published". They are untracked, so the tarball depended on
  who published it: CI, from a clean clone, produced 59 files; a local publish
  produced 62, including files in no version control anywhere. `files` is
  `["src"]` now.

### Added

- **Camera framing.** `scene.frame(camera)` -- or a controller; the same
  method serves either -- points at everything in the scene and backs off far
  enough to see it.

  ```js
  scene.frame(controller);          // no distance to pick
  ```

  A sphere of radius r fills a frustum of half-angle a at `r / sin(a)`, and
  the scene already computes those bounds every frame for culling. The number
  was always there, just never offered. It fits the bounding SPHERE rather
  than the eight corners, so orbiting does not change how much of the view the
  object fills -- fitting corners makes an object breathe as it turns, which
  is the same reason the shadow cascades are sphere-fitted.

  `Camera.frameBounds`, `Scene.bounds`, and `fitDistance` / `boundsRadius`
  for anything that wants the arithmetic on its own.

- **`OrbitController.syncFromCamera()`**, which adopts wherever the camera is
  instead of overwriting it. The controller rebuilds position from yaw, pitch,
  distance and target every frame, so a cutscene, a teleport, a saved
  viewpoint or `Camera.frameBounds` all worked for exactly one frame and were
  then silently undone. There was no way to hand control back; now there is.

  It is the exact inverse of what `update()` does, so the two agree by
  construction. It cannot be exact past the pitch clamp or outside the
  distance clamp -- those are poses the controller deliberately refuses -- so
  it adopts the nearest it can hold, immediately rather than a frame later.

- **sRGB to linear.** `colorFromHex`, `colorFromBytes`, `srgbToLinear`,
  `linearToSrgb`.

  Every colour this engine takes is linear -- base colour, emissive, light
  colour, sun colour, the sky -- and every colour a human has is not. Typing a
  picked value straight in is not slightly off: sRGB 0.35 is linear 0.10, so
  it arrives roughly three times too bright and washed toward white with
  nothing reported. I made that mistake writing an example, picked
  `[0.85, 0.35, 0.25]` for red and got salmon, and a user asked why.

  Alpha is deliberately NOT converted, which is what a hand-rolled version
  gets wrong: the transfer function is for colour channels, and curving alpha
  makes transparency subtly wrong in a way that reads as a blending bug.

  Note that a light EMITS, so its colour is radiance that routinely exceeds 1
  -- the default sun is `[3.2, 3.0, 2.7]`. `#ffffff` converts to `[1,1,1]`,
  which is a correct white and a dim sun.

### Changed

- The 0.7.2 notes listed `examples/cdn.html` as an addition to the package.
  It is gitignored and shipped to nobody; the entry now says so.


## [0.7.3] - 2026-09-23

**Rendered pixels are something a check can read now**, and the first thing
they showed was a scene mixup that had nothing to do with what I was looking
for.

### Fixed

- **A second scene drew the first one's geometry.** `GpuDriven` decided its
  cached batches were stale by comparing `scene.revision` alone. Revision
  counts changes WITHIN a scene and starts at zero in every scene, so two
  scenes one `add()` old each both report 1 -- and rendering the second kept
  the first's batches: its primitives, its materials, and therefore its
  pipelines.

  `createScene()` is public and multi-scene was deliberately fixed once
  already, so this is a menu behind a game, or a preview beside a main view,
  drawing the wrong thing. Scenes carry an `id` now and staleness is a
  (scene, revision) pair, because a revision alone never identified anything.

### Added

- **`rhi.readPixels()`**, returning RGBA bytes from the frame just drawn. The
  swap chain is now `COPY_SRC`, which costs a lazy-clear optimisation and buys
  the only way to ask what colour a pixel is -- a WebGPU canvas has no working
  `toDataURL`, so without it the answer did not exist at any price.

  Two things it undoes, both found by using it. The swap chain is
  `bgra8unorm` nearly everywhere, so blue and red are swapped back: a method
  promising RGBA that returns BGRA is a bug that reads as a rendering bug. And
  it works ONCE per frame -- the await inside lets the browser present, so a
  second call copies from a texture that has just been handed over and
  cleared. There is deliberately no `readPixel(x, y)`, because that shape
  invites exactly that.

- **A feature-matrix check**: one full-view quad per feature, rendered, with
  the middle pixel read back. Fourteen of them -- opaque, sky, MASK either
  side of its cutoff, COLOR_0, texCoord 0 and 1, a mirrored node, a
  non-indexed primitive, flat-shaded geometry, a normal map, `normalScale` 0
  cancelling it, a point light reaching a surface, and emissive with no map.

  Between them the rendered fixtures had used ONE material texture slot, one
  alpha mode that draws, positive scales only, indexed geometry only and
  supplied normals only. Everything else was imported, unit-tested, compiled
  into a pipeline, and never turned into a pixel. The suite's strongest claim
  was "no device errors, no WGSL errors", and a NaN normal and a black default
  texture both produce neither.


## [0.7.2] - 2026-09-23

**Two bugs that only untextured geometry could reach.** Every asset in this
repository has UVs and a full set of maps, so neither had ever been rendered
into existence. Both were found by drawing one cube with no textures on it.

### Fixed

- **An untextured face whose normal points along X shaded to NaN, and the
  bloom chain spread it across the screen as black.** A mesh with no UVs got
  the SAME tangent on every vertex -- `(1,0,0)` -- which the code called "a
  valid unit vector rather than zeros so a shader that samples it cannot
  produce NaN". It is a unit vector, and it produces NaN anyway: on any face
  whose normal points along X it is PARALLEL to that normal, so `cross(N, T)`
  is the zero vector and `normalize()` of that is NaN.

  The symptom looked nothing like the cause -- a black slab several times the
  size of the object, because bloom had smeared two faces' worth of NaN through
  its mip chain. An untextured cube hits it on two faces out of six.

  Fixed with `perpendicularTo`, which was already in `tangents.js` doing this
  exact job for `generateTangents`, and whose least-aligned-axis trick
  `brdf.js` already used for IBL sampling. The pattern was in the codebase
  twice and missing from the third place that needed it.

- **Emissive without a texture did not glow.** The absent-emissive default was
  a black 1x1 and the shader multiplies it by `emissiveFactor`, so the factor
  was multiplied away and a material that asked to glow rendered dark. glTF
  says an absent texture reads as 1.0 on every channel, which is the premise
  the whole design rests on: absent maps point at defaults and the factors do
  the scaling, instead of a shader variant per combination of present maps.
  Emissive was the one slot breaking its own rule. The black default had no
  other user and is gone.

### Added

- **A sky you can set.** `Environment` takes a `sky`: `ground`, `horizon`,
  `zenith`, `sun` direction, `sunColor`, `sunIntensity` and `glow`. They were
  three constants in a shader.

  ```js
  await Winding.create(canvas, {
    environment: { sky: { ground: [0.02, 0.02, 0.03], zenith: [0.05, 0.07, 0.12] } },
  });
  ```

  Substituted into the shader source rather than uploaded as a uniform,
  because they are BAKE-TIME values: the cubemap is generated once when an
  Environment is constructed and read every frame after. A uniform would add a
  buffer and a binding to describe numbers that never change again.

- **`renderer.drawSkybox`**, which turns the background off without touching
  the lighting. Setting the sky to black would do both, because the same
  cubemap is the ambient term -- the sky IS the light.

- **A CDN smoke page** -- the smallest thing that imports the engine from
  jsDelivr and draws one cube. It is what found both bugs above. NOT SHIPPED:
  it lives in `examples/`, which is deliberately untracked and unpublished,
  so this entry describes how the bugs were found rather than something you
  can open. (Corrected in 0.7.4; the original wording listed it as an addition
  to the package.)

### Changed

- `skyRadiance` moved from `brdf.js` to `ibl.js`, next to its only caller. It
  is not a BRDF, and every shader that included that file was compiling it as
  dead code.


## [0.7.1] - 2026-09-23

**The package is `winding-engine`.** 0.7.0 could not be published: npm's
typosquatting filter refuses `winding` as too similar to `bindings`, a
long-established package that loads native `.node` files. The filter is worth
having, so the package took a name rather than an argument.

Nothing else changed. The project is still Winding, the repository is still
`winding`, and the wordmark still carries no extension -- three.js makes the
same split, with a `three.js` repository publishing a package called `three`.

A note for anyone checking a name before they build on it: **unregistered is
not publishable.** `npm view winding` returned a clean 404 right up until the
registry rejected the upload, because the similarity check only runs on PUT.
The only way to know is to try, and a rejected publish costs nothing.

### Changed

- `package.json` name, and every install line in the README, now say
  `winding-engine`. Deep imports move with it:
  `import { GpuProfiler } from 'winding-engine/render/timing.js'`.


## [0.7.0] - 2026-09-23

**Installable from somewhere other than this repository.** No build step and no
dependencies, so the files that get published are the files that are here --
what was missing was the packaging metadata, an install section, and one bug
that only appears when the engine is served from another origin.

Minor: new public exports and a package surface, nothing existing changed shape.

### Added

- **npm packaging.** `exports`, `files`, `repository`, `keywords`,
  `sideEffects` and the `unpkg`/`jsdelivr` entry fields. 200 kB packed, 62
  files.

  `exports` maps `./*` on purpose, so `import 'winding/render/timing.js'`
  works. A package with one entry point would put a wall exactly where the
  design says there isn't one -- dropping a tier is additive, and that has to
  survive the package boundary or it was never true.

- **An Install section** covering the CDN, npm and import-map routes, with the
  version pinned in every example. `@latest` re-resolves on every page load,
  which means a release nobody has seen can change what a page runs.

### Fixed

- **A cross-origin worker threw instead of loading.** `new Worker(url)` refuses
  a cross-origin script -- it throws, it does not degrade -- and an engine
  loaded from a CDN is cross-origin by definition.

  The case it broke was the good one. Workers are only spawned on a page that
  set COOP and COEP, because that is what `SharedArrayBuffer` needs, so the
  better-configured the host page, the harder the failure: a page that did
  everything right got an exception out of `Winding.create`, while a page that
  did nothing quietly ran inline and worked.

  Cross-origin now loads through a same-origin blob module that imports the
  real URL, because a module's own imports go through CORS where the Worker
  constructor never does. Same-origin is untouched -- the shim costs a fetch
  hop to get around a restriction that is not there.

  The specifier is built with `JSON.stringify`, not hand-quoted. It is pasted
  into a module that gets executed, and a URL may legally contain a quote.

### Changed

- The README's opening example imports `'winding'` rather than a repo-relative
  path, and the test counts it quotes are current again (399 Node, 16 GPU).


## [0.6.2] - 2026-09-22

A race in the job system, and the reason nobody had seen it.

The parallel suite had never run a worker. It spawned them correctly and
dispatched in the same synchronous run, so the readiness message could not
have arrived yet and every dispatch took the inline branch -- fourteen of
them, across every release so far. Nothing failed, because the serial
fallback is bit-identical to the parallel path by design; that identity is
what makes results useless as evidence about which one ran.

Making it run parallel found a real bug in the first suite execution.

Patch rather than minor: `ready()` and `stats` are additions, but nothing
that existed changes shape for a caller. The cursor layout is internal, and
the one caller of `runChunks` outside this file is the worker entry module,
which is in this repository.

### Fixed

- **A worker could run one dispatch's chunks with another's arguments.** The
  chunk cursor was a plain counter reset per dispatch and claimed with an
  atomicAdd, which made a claim anonymous. A thread descheduled between
  reading its dispatch's arguments and claiming could wake into the NEXT
  dispatch, take its chunks, run them with the PREVIOUS dispatch's arguments,
  and then credit the new dispatch's completion counter -- so the dispatch
  returned believing work was done that its own handler never touched.

  `TransformStore.updateParallel` dispatches once per depth level, back to
  back, with a different base each time. That is exactly the shape that
  triggers it: a range of the order array composed against the wrong level,
  leaving world matrices at whatever was in the buffer. Silent.

  The cursor now carries the epoch in its high 16 bits and is claimed by
  compare-exchange, so a claim belongs to a dispatch and a stale thread takes
  nothing rather than stealing live work. It also counts chunks rather than
  items, which means it can no longer be advanced past the end at all.

  Found only because the fix below made the suite run parallel for the first
  time. Reproduced at 3 failures in 8 runs, confirmed by draining between
  dispatches (8 in 8), and verified after the fix at 40 in 40 -- and 0 in 40
  with the tag check removed.

- **The parallel job suite had never run a worker.** It spawned them
  correctly and then dispatched in the same synchronous run, so
  `readyCount` -- which a worker raises by POSTING A MESSAGE, and a message
  cannot reach a thread that has not returned to its event loop -- was still
  zero every time. Fourteen dispatches, every one on the inline branch.

  Nothing failed, and nothing could have: the serial fallback is
  bit-identical to the parallel path by design, which is what makes results
  useless as evidence about which one ran. The atomics, the cursor claiming
  and the epoch wakeup were the largest unverified surface in the codebase,
  and the suite reported them green while one of them was broken.

### Added

- **`JobSystem.ready()`**, resolving true once every worker has installed its
  handlers and false if dispatch will run inline anyway. Bounded by the same
  deadline a lost worker gets mid-dispatch, because it is the same question.
  The engine deliberately does not await it -- a worker that has not checked
  in just means the next few dispatches run on the calling thread -- but
  anything that must know the parallel path was exercised has no other way to
  ask.

- **`JobSystem.stats`**, counting dispatches by branch. Whether a dispatch
  went parallel was otherwise unobservable, which is precisely how this went
  unnoticed for the life of the project.

- **Proof of parallelism in the suite**, rather than an assumption of it. Each
  thread signs the items it processed, and the check asserts more than one
  signature appears, that no item is unsigned, and that the dispatching thread
  is among them. The job it uses spends real time per chunk on purpose: with a
  trivial handler the dispatching thread can legally claim every chunk before
  a worker finishes waking, which would make the assertion flaky rather than
  wrong.

  Measured across the change: before, `{parallel: 0, inline: 1}` and one
  thread; after, `{parallel: 1}` with four threads taking 2000 items each and
  8ms of wall time against ~32ms of serial work.

### Changed

- The chunk cursor counts chunks rather than items, so its value is a chunk
  index and no longer overshoots. A dispatch with more than 65535 chunks has
  its chunk size raised to fit the index beside the epoch tag -- a chunk size
  is a performance hint, and doing more work per claim cannot make a result
  wrong, where refusing the caller's number would.

## [0.6.1] - 2026-09-22

Four files that loaded wrong rather than failing. All four were in the
deferred ledger rather than newly found -- the morph work produced almost no
patch debt of its own, because the two real bugs it turned up were fixed in
the commits that found them.

Every one of these is the same shape: a malformed or unusual document that
produced plausible geometry instead of an error. Nothing threw, nothing
logged, and the result looked like an asset problem.

### Fixed

- **A node with two parents was instantiated twice in release builds.**
  `Scene.add` guarded the forest rule with a DEBUG-only assert. Past it, a
  diamond built the subtree twice and every downstream map -- the
  node-to-entity table, the animation player's, the skin's joint resolution --
  kept only the second copy. A clip then drove one of the two and the other
  sat frozen. Now an unconditional throw naming the node; one comparison per
  node is not a reason to ship that.

- **Float index accessors truncated silently.** FLOAT is signed, and the
  signed guard excluded it so the float reader could share it. The fast path
  then viewed the buffer as `Float32Array` and copied into a `Uint32Array`,
  truncating toward zero. An index of 2.0 read as 2, and nothing looked wrong
  until one was 65535.9. Refused now, for indices and joints alike.

- **A declared but empty default scene loaded the whole document.**
  `"scenes": [{}]` is legal and means a document whose contents are all
  referenced rather than instantiated -- a mesh library, which is a real way
  to ship one. The check was `scene?.nodes`, so an empty scene fell through
  to orphan detection and instantiated every node in the file. A declared
  scene is now authoritative whether or not it has nodes, and an explicit
  `scene` index that names nothing is an error rather than a silent fallback.

- **MAT2/MAT3 accessors of small components read at the wrong stride.** glTF
  pads each column to four bytes there, so the element is not
  `bytes * count` long and every matrix after the first came from the wrong
  place. Nothing in this engine can reach it -- the only matrices read are
  MAT4 inverse binds, where the rule does not apply -- so it is refused with a
  message naming what is missing rather than implementing a layout no asset
  here uses.

### Changed

- `scene.js` no longer imports `DEBUG` or `assert`; the one use it had is now
  unconditional.


## [0.6.0] - 2026-09-22

**Morph targets**, the last deformation glTF describes that this engine did
not do. Four steps, each verified before the next depended on it.

Minor rather than patch because the vertex path and the animation sampler
both changed behaviour, but no binary layout moved: `DrawData` is still 128
bytes -- the three morph words sit in padding `paletteOffset` already left --
and the vertex format is untouched, because deltas are a storage buffer the
shader indexes rather than an attribute per target.

Two bugs came out of building it, and both predate it. One had been live
since skinning shipped in 0.5.0.

### Added

- **Morph targets.** Per-target vertex deltas and per-instance weights, read,
  stored, animated, deformed and bounded. `targets`, `mesh.weights`,
  `node.weights` and `weights` animation channels all come through.

  One vertex shader does both deformations, in the spec's order: morph first,
  since a target is authored against the bind pose, then the joint palette.
  Neither adds a pipeline permutation -- the target count is draw data and
  zero means the loop does not run, so a static mesh runs the same shader and
  pays a compare.

  Weights are per instance and handed out live, so `node.weights[0] = 0.4`
  works and two faces sharing a mesh are still one batch and one draw call.
  That is the one live view `node.js` allows: the rule against them exists
  because a write that skips a setter leaves a stale dirty flag, and weights
  have no hierarchy and no flag.

  Deltas are stored VERTEX-MAJOR in one engine-wide arena. Target-major is
  the obvious transcription of the file and wrong twice: a vertex shader reads
  one vertex and every target, and the flat-shading unweld reorders vertices.
  The stride is decided once per primitive at the widest any of its targets
  needs, so a positions-only rig stores three floats per target rather than
  nine.

  Bounds pad by `sum(|weight| * extent)`. The absolute value is load-bearing:
  glTF does not bound weights to [0,1], a negative weight is how "the opposite
  of this expression" is authored, and a signed sum would SHRINK the box and
  cull geometry that is on screen.

- **A GPU check that verifies a vertex position.** The morph check slices the
  `Morphed` struct and `applyMorph` out of `PBR_SHADER` into a compute shader
  bound to the same delta buffer, weight buffer and draw data the frame just
  used, and reads four vertices back. Every other GPU check verifies the
  absence of errors; this one verifies arithmetic, and it is what found the
  fix below.

### Fixed

- **`GpuDriven._grow` left `drawDataU32` viewing the old buffer.** Half of
  `DrawData` is u32, so every integer write after a grow went into a detached
  array -- silently, because an index inside the old length writes where
  nothing is uploaded from and an index past it writes nowhere at all. Every
  u32 field read as zero on the GPU.

  Live since skinning shipped in 0.5.0: any skinned instance added after the
  draw buffer grew used palette offset 0, which is a second character wearing
  the first one's pose. Both views are now allocated in one place.

- **A four-target `weights` channel was slerped like a quaternion.** The
  sampler chose between lerp and slerp by component count, which was the same
  question as "is this a rotation" until morph targets existed. Four
  independent sliders were swept through a sphere and normalized to unit
  length -- every number wrong, nothing thrown. It asks the path now.

- **The README claimed glTF import handled "Not skins or morph targets"** in
  the same sentence that listed skins as supported. A fragment left from
  before skinning landed.

### Changed

- Skinned and morphed renderables are picked at their bounding box, even with
  `retainGeometry`. The retained triangles are the undeformed mesh, so a hit
  on them reports where a vertex was authored rather than where it is.
  Unconditional rather than "when a weight is nonzero": precision that
  switches on and off as a clip plays is a worse contract than precision that
  is honestly coarse.


## [0.5.0] - 2026-09-22

**Skinning.** The largest single item the engine was missing, done in four
steps, each verified before the next depended on it.

Minor because two binary layouts moved again: `DRAW_DATA_BYTES` is 128 where
it was 112, and the transparent sort key is now 15 depth / 5 pipeline / 12
material. The vertex format did not change -- influences ride in a second
buffer that only skinned pipelines bind.

### Added

- **Skinned meshes deform.** `skins`, `JOINTS_0`, `WEIGHTS_0` and
  `inverseBindMatrices` are read; a per-instance joint palette is built each
  frame from matrices the transform hierarchy already composes; and a skinned
  vertex path applies it. Two characters in different poses still share a
  batch and a draw call, because the palette offset is per-instance draw data
  rather than per batch.

  The mesh node's own transform is deliberately absent from the skinned path.
  glTF 3.7.3.3 says the joints place a skinned mesh entirely, and applying it
  as well moves the character twice.

- **Bounds that follow the pose.** A skinned mesh's vertices move without its
  model matrix moving, so transforming a static local box gives the box the
  character had when it was AUTHORED -- raise an arm and geometry sits outside
  its own bounds, frustum-culled with the arm on screen. A skinned vertex is a
  weighted average of its per-joint positions, so a union of spheres contains
  every vertex the skin can produce: one per joint, at its current world
  position, sized by how far its influence reached in bind pose. Derived
  rather than inflated by a factor, and the cost is per joint rather than per
  vertex.

- **Skinned shadow casters**, so a character's shadow deforms with it.

### Fixed

- **The shadow pass read draw data at the wrong stride.** Introduced by this
  release's own second step: the shadow shader declared `DrawData` without
  `paletteOffset`, and WGSL sizes a struct from its members, so it read 112
  bytes out of a buffer written at 128. Instance 0 landed correctly and every
  one after it was misaligned. There is now a check that every shader reading
  `DrawData` declares the same struct, and that the struct they agree on is
  the stride the CPU writes.

- **Triangle-exact picking missed on skinned meshes.** Posed box, bind-pose
  triangles, reached through a matrix skinned vertices do not follow -- so a
  posed character with `retainGeometry` became unpickable while its box said
  otherwise. Skinned renderables are answered at their box, like any primitive
  whose geometry was not retained.

- A backtick inside a WGSL comment terminates the template literal and the
  file fails to parse pointing at whatever word followed. It has happened
  twice; there is now a check that no shader source contains one.

### Changed

- `variantKey` takes a fourth argument, `skinned`. Each material variant
  expands into four pipelines -- two windings times two skinning -- all
  compiled at load, because `render()` is never allowed to create one.
- `TRANSPARENT_PIPELINE_BITS` is 5 and `TRANSPARENT_DEPTH_BITS` is 15, where
  they were 4 and 16. Three alpha modes times two sidedness times two windings
  times two skinning is 24 pipelines, past what 4 bits holds; the bit came
  from depth, which had 65536 buckets for an ordering that cannot resolve
  better than a pixel.
- `ShadowMaps.addPasses` takes the joint palette.
- The Limitations section drops "resource aliasing is idle", which was an
  accurate note about the default frame and never a limitation.

348 checks under Node, 14 on a real device.

## [0.4.0] - 2026-09-22

One defect and one feature, both about transparency and shape.

**Minor because `CLUSTER_X` and `CLUSTER_Y` are gone.** Nothing else was
removed and no signature broke. At 16:9 the froxel grid is byte-for-byte what
it was, so the default viewport renders identically.

### Added

- **Weighted-blended order-independent transparency**, behind `{ oit: true }`.
  Blended geometry accumulates into two targets -- depth TESTED against the
  opaque pass and never written -- and one full-screen resolve composites it
  over the scene.

  A second path, deliberately, not a replacement. The sorted path is EXACT for
  separated convex objects and has no answer at all for interpenetrating ones,
  because no single per-object order exists there; this needs no order and is
  approximate everywhere. Neither is better. Architectural glass wants the
  sorted one, smoke and foliage want this one, and most scenes this engine
  draws are the case the default already gets right.

  The weight function is where the approximation lives, and it is the part of
  McGuire's paper that carries constants tuned against a normalised view depth.
  Reverse-Z means the depth buffer value is already 1 at the near plane falling
  toward 0, with no far plane to normalise against, so it is used directly and
  the cubic is all that survives. The range is derived: `rgba16float` holds
  65504, so the cap is that over `OIT_LAYER_BUDGET`. Past that depth complexity
  the sum saturates and near layers stop dominating, which shows as
  transparency flattening rather than as anything breaking.

### Fixed

- **The froxel grid was 16 by 9 whatever the viewport was.** Tile SIZE always
  came from the real resolution, so coverage was never wrong -- the cells were
  just stretched. A portrait phone canvas got them about 3x taller than wide,
  an ultrawide the same on the other axis, and a stretched cell overlaps
  proportionally more light spheres. So `MAX_LIGHTS_PER_CLUSTER` is reached at
  a fraction of the light count a 16:9 monitor manages, and overflow past that
  cap is dropped with nothing reported: lights wink out in busy regions on a
  phone and look fine on a desktop.

  The tile COUNT is a real budget -- it sizes the index buffer -- so it stays
  fixed at 144, which is what 16 by 9 was. The arrangement is derived:
  `x = round(sqrt(tiles * aspect))`, `y = floor(tiles / x)`. y floors rather
  than rounds because `x * y <= 144` is what the buffer depends on, and
  rounding both lands at 15 x 10 for 3:2. Portrait gets 9 x 16 and square
  12 x 12, both with froxel aspect 1.00.

### Changed

- **`onDeviceLost` carries enough to act on**: `reason`, `message`,
  `recoverable`, and `action`. The default console path now says the engine
  does not rebuild GPU state and that reloading is the route back, rather than
  printing a reason and stopping.

  The README entry said "the callback fires; rebuilding the GPU state is on
  you", which describes an unbuilt feature rather than a decision. Rebuilding
  means holding a CPU copy of every GPU resource for the process lifetime, and
  the expensive part is the textures' CONTENTS -- either every decoded image
  stays resident, which is the gigabyte `load()` releases on purpose, or every
  asset is fetched and decoded again, which is `load()`. A driver reset, a GPU
  hang, an out-of-memory and a reclaimed background tab all want the same
  response, and the callback now names it.

- `CLUSTER_X` and `CLUSTER_Y` are replaced by `CLUSTER_TILES` and
  `clusterGridFor(aspect)`. The live shape is `clusters.gridX` / `gridY`.
  `CLUSTER_COUNT` is unchanged at 3456.

323 checks under Node, 13 on a real device.

## [0.3.1] - 2026-09-22

A patch: nothing was removed and no signature broke. But **two of these change
what you see**, both by making something correct that was not:

- Shadows and clustered lights now cover your scene rather than the first 60
  world units of it. If your scene was roughly Sponza-sized nothing moves.
- Rough metal stops speckling.

### Fixed

- **`shadowDistance` and `lightDistance` were a flat 60 world units.** True of
  Sponza and of nothing an order of magnitude either side of it: a 5 km terrain
  got shadows that stop 60 units from the camera with no boundary, and a 5 cm
  part got all four cascades collapsed into the first metre of a space
  thousands of times its size. The culler maintained world bounds every frame
  that answer the question exactly, and nothing read them.

  Both now default to the scene's far corner in view depth, which is how far
  there is anything to shadow or light. Pass a number to pin either. The union
  is recomputed only when something moved or the contents changed, so a settled
  scene never pays for it, and the floor is the smallest LEGAL value rather
  than a chosen one -- both consumers require strictly more than the near
  plane, so with an empty scene twice near is the minimum that satisfies them.

- **Four options reached `Winding.create` and stopped there.** `shadows`,
  `post`, `powerPreference` and the two ranges above were accepted and never
  forwarded, so the shadow map's size and cascade count, the bloom settings and
  the GPU preference were unreachable without constructing a `Renderer`
  yourself -- in a file whose header explains that dropping a tier is meant to
  be additive.

- **The IBL convolutions sampled the environment at LOD 0.** Karis' "solving
  the bright dots": a fixed 64 samples over a full-resolution source
  undersamples it, because each sample stands for a cone of directions and
  reads a single texel. With the energy concentrated in a few texels -- the
  procedural sky's own 60x sun, and any captured HDR far worse -- a sample
  either hits it and blows the estimate up or misses and loses it, and
  neighbouring output texels disagree frame to frame. That is the speckle that
  swims across rough metal.

  Each sample now reads the mip whose texels cover its own solid angle. The
  pdf is `cos/pi` for the cosine-weighted irradiance and `D*NoH/(4*VoH)` for
  GGX; roughness 0 is pinned to level 0, since a mirror has no cone.

- **`generateMipmaps` only ever reduced array layer 0.** Correct for a 2D
  texture and wrong for a cubemap, whose six faces are six layers -- five of
  them were left undefined below the base level. Surfaced by the fix above,
  which needed the source cube to have mips at all.

### Added

- `unionWorldBounds` and `farthestViewDepth` in `scene/bounds.js`, which are
  what the derived ranges are built from.
- `Renderer.create` takes `shadowDistance`. Both it and `lightDistance` accept
  `null`, which is the new default and means "derive from the scene".

319 checks under Node, 12 on a real device.

## [0.3.0] - 2026-09-22

The three items the 0.2.0 audits left open, all of them cases where the engine
silently did something other than what it said.

**No API was removed, but two binary layouts moved.** `VERTEX_STRIDE_BYTES` is
60 where it was 48 and `MATERIAL_BYTES` is 64 where it was 48, so anything that
wrote a vertex or material buffer by hand needs rebuilding. Assets loaded
through `engine.load` are unaffected.

### Fixed

- **Mirrored node transforms rendered inside-out.** glTF 3.7.4 requires the
  triangle winding to reverse when a node's global transform has a negative
  determinant. `frontFace` was fixed at `ccw`, so mirroring one chair to make
  its pair -- the everyday case in furniture, architecture and product assets
  -- drew both with the faces that should be visible culled and the ones that
  should not kept. `mat4Decompose` had always DETECTED the mirror and parked it
  in `scale.x`; nothing downstream consumed it.

  Mirroring belongs to the instance and a pipeline belongs to the material, so
  it could not simply join `variantKey`. It is a fourth variant bit the
  registry never sets and the renderer ORs in per batch; winding joins the
  batch key, because one indirect draw has one front face; the sort key's
  pipeline id doubles so two pipelines do not claim to be one state bucket;
  and `ensureVariants` compiles both windings at load, since render() is never
  allowed to create a pipeline. Read from the WORLD matrix, because a mirroring
  parent mirrors everything under it and two mirrors cancel.

- **A second UV set was never read.** `texCoord` sits on the texture REFERENCE
  rather than the material, so one material's maps can disagree about which set
  they sample -- baked occlusion on set 1 beside a base colour on set 0 is the
  ordinary export out of Blender and Max. Those materials sampled the wrong
  pixels. Not a missing texture, which is noticeable; wrong pixels, which is
  not.

- **`COLOR_0` was dropped.** Core spec, no extension behind it, and not on the
  importer's list of things deliberately unhandled. Assets carrying their tint
  in vertex colours rendered uniformly white.

- **The batch tables were sized per renderable and indexed per batch.**
  `batchCount <= renderableCount` is a bound, but it is the loosest one in the
  engine -- being loose is the point of batching. At a 256-byte alignment the
  per-batch uniform buffer was 512 bytes per RENDERABLE, which passes the
  default `maxBufferSize` somewhere above 524,288 renderables. `createBuffer`
  does not throw for that; it returns an invalid buffer and the frame goes
  black. So the real ceiling was 32x lower than the 2^24 entities the README
  advertised as a derived hard limit. Sized by batch now, and growing
  separately: 14 batches is 7 KB where 677 renderables was 347 KB.

### Changed

- **The vertex is 60 bytes, up from 48.** Every vertex carries a second UV set
  and a colour whether its asset has them or not. The alternative is a vertex
  format per attribute combination, which multiplies into the pipeline count,
  the shader permutations and the importer at once -- on top of the winding
  variants above. Carrying the fields is the cheaper mistake, and it pays
  twice: an asset without them gets `uv1 = uv0` and colour = white, both
  identities downstream, so nothing branches at runtime either.

  The colour is `unorm8x4` rather than four floats. glTF allows float, but
  vertex colours are authored at 8 bits per channel essentially always, and
  four floats would have made the stride 72.

- **`MATERIAL_BYTES` is 64**, carrying five bits that say which UV set each
  map samples. The shader picks with a `select` rather than a branch: both
  sets are interpolated anyway, so choosing between them is free and uniform
  across the quad.

- `variantKey` takes a third argument, `mirrored`, defaulting to false.

### Added

- `VARIANT_MIRRORED` and `VARIANT_DOUBLE_SIDED`, the named variant bits.
- `packVertexColor`, `VERTEX_COLOR_INDEX` and `VERTEX_COLOR_WHITE`, for writing
  the packed colour through a `Uint32` view of the vertex buffer.
- `uvSetMask` and the five `UV_SET_*` bits.

315 checks under Node, 12 on a real device.

## [0.2.0] - 2026-09-22

Three of the README's limitations, then two audits and the fixes they found.

**If you are upgrading, four of these change what you SEE**, not just what is
correct underneath. Double-sided back faces shade instead of going black;
lights beyond `lightDistance` light things; `NEAREST`-filtered textures stop
mipping, so pixel art goes sharp; and a texture past the device's size limit
now throws where it used to render untextured in silence. If a scene looks
different after this release, it is one of those four.

### Removed

Five exports, each with no caller anywhere in the engine or its tests:

- `UniformRing` and its whole per-frame allocator. `rhi/buffer.js` went from
  131 lines to 26 and stopped advertising a subsystem nothing used.
- `SUN_DIRECTION`, documented for an analytic key light that does not exist,
  and a hand-copy of three literals from the WGSL beside it.
- `DRAW_BYTES`, a second name for `DRAW_DATA_BYTES`.
- `opaqueDepthBucket`, which filled a sort-key field that is always zero.
- `GROUP_PASS`, renamed `GROUP_RESERVED`. Nothing ever bound at that rate; the
  shadow cascade matrices the header claimed lived there are in `GROUP_FRAME`.

`RenderGraph` no longer takes `maxPasses` or `maxResources`, and `GpuProfiler`
no longer takes `maxPasses`. See Changed.

### Changed

- **`sampleClip` and `AnimationPlayer` take the scene's handle allocator.**
  `sampleClip(clip, time, transforms, entityOf, entities)` and
  `new AnimationPlayer(clips, entityOf, entities)`. Required, not optional --
  liveness cannot be answered without it. See Fixed.
- **`TRANSPARENT_PIPELINE_BITS` is 4 and `TRANSPARENT_MATERIAL_BITS` is 12**,
  where they were 8 and 8. See Fixed.
- **Two-phase occlusion culling.** Whatever was drawn last frame is drawn
  first, the depth pyramid is built from that, and a second cull tests
  everything else against it before a second pass draws what it newly admits.
  Nothing is a frame stale any more, so an object that becomes visible appears
  on the frame it does instead of popping in on the next. `lastViewProjection`
  is gone. Both phases share one indirect buffer, one visible list and one
  batch-info buffer, each doubled, so this costs one compute dispatch and one
  set of indirect draws rather than a second copy of the machinery.
- **The render graph versions resources written more than once.** A read edged
  from every writer regardless of declaration order, which is right for a
  single-writer resource and wrong for a sequence: the depth pyramid reads
  depth between the two forward passes, and edging it from both put it after a
  pass that depends on it. A resource written several times is a sequence of
  values, and a read means the one current where it was declared.
- **The render graph has no pass or resource ceiling.** It was 32, and the
  default frame's pass count is a function of resolution -- 31 at 1080p,
  exactly 32 at 1440p and 4K, and 33 at 5K, where it threw every frame. The
  capacity is deleted rather than raised: a frame declares what it needs, and
  the pools extend to that and are reused. `GpuProfiler` follows the same rule
  and sizes its query set from the frame, so its first frame is untimed.
- **`createScene()` no longer touches the job system**, and the compose job is
  registered once per engine. See Fixed.
- **`ClusteredLights.update` and `createTexture2D` throw** on inputs that used
  to fail silently: a non-positive near, a `lightDistance` at or inside it, and
  a texture past `maxTextureDimension2D`.

### Added


- **Triangle-exact picking**, behind `engine.load(src, { retainGeometry: true })`.
  `scene.raycast` keeps the bounding-box test as a broad phase, sorts the
  boxes the ray enters by entry distance, and walks them near to far, stopping
  as soon as the next box begins further away than the best hit so far. The
  narrow phase inverts the model matrix and pushes the RAY into local space
  rather than the triangles into world space, and deliberately does not
  renormalize the transformed direction, so the distance comes back on the
  same scale as the box distances. Primitives loaded without the flag are
  still answered at their box, so mixing the two degrades per object.
- **Per-pass GPU timing** on any adapter with `timestamp-query`, which was
  already requested and unused. The render graph owns it: it knows the pass
  list and the order, so every pass gets its `timestampWrites` without a call
  site being instrumented by hand. Readback is unsynchronised through a ring
  of staging buffers, because mapping a buffer the GPU may still be writing
  would stall the frame being measured. Read `renderer.gpuTiming.average`, not
  a single frame: Chrome quantises these timestamps to 65536ns, so most passes
  read as exactly 0 on any given frame and only the mean over hundreds of them
  means anything.

### Fixed

Two audits, one for stubbed-out or wrong code and one for whether the engine is
general purpose or fitted to the two demo assets. Everything below came from
reading; three ad-hoc detector scripts produced nothing but false positives.

**Wrong pixels**

- **The depth pyramid was not conservative.** `fsFirst` documented its 2x2
  gather as one that "always covers the full footprint". The scale lands in
  [1, 2), so an output texel owns a source interval shorter than two texels
  that still straddles THREE of them when unaligned -- 75% of the row at 1920
  wide. Missing a texel from a MIN raises the result, and under reverse-Z
  higher is nearer, so the pyramid claimed a nearer occluder than existed and
  the cull test deleted visible geometry. Now a 3x3 gather clamped to the
  footprint's own last texel.
- **Double-sided back faces shaded black.** `cullMode: 'none'` was set, but the
  fragment shader built its TBN from the interpolated normal with no
  `front_facing` flip, so a back face arrived with the normal of the front it
  was authored as and `NoL` clamped to zero. Every leaf, curtain and thin
  panel. The normal and bitangent now flip together; the tangent does not,
  since it follows the UV's u axis.
- **`lightDistance` was a hard cutoff**, in a file that says three times it is
  not. The exponential mapping ends the final cell exactly there, so a light
  past it sat outside every cluster box and was added to none -- while the
  fragment shader clamps depths beyond `lightDistance` INTO that cell. The last
  slice now reaches as far as the farthest light does, derived from the light
  list each frame.
- **glTF's `NEAREST` and `LINEAR` minFilters built a mip chain.** Those two
  mean no mip chain at all. They now pin both LOD clamps to 0.
- **An animation channel could drive a recycled entity.** The 0.1.1 guard
  tested the transform SLOT and never the generation, and handles recycle
  last-in-first-out, so the next `alloc()` took the slot back and marked it
  live. The guard caught "freed" and missed "freed and reused", which is the
  common case. It is now one call to `HandleAllocator.alive()`, which is the
  only thing that compares generations.
- **Blended geometry past 255 materials reordered the frame.** The transparent
  sort key's material field was 8 bits while `MaterialRegistry` guards 4096
  from the opaque key, and nothing minted against the narrower one. The
  missing width was next door: pipeline ids are a dense index over
  `variantKey`, which has at most six values, so 8 bits carried 3 bits of
  information. Now 16/4/12, with the material width defined AS
  `OPAQUE_MATERIAL_BITS` and three module-load asserts so the two cannot drift.

**Crashes, leaks and NaN**

- **The render graph threw every frame at 5K** and had zero headroom at 1440p
  and 4K. See Changed.
- **Resizing leaked every texture it passed through.** The graph's pool is
  keyed by descriptor and was emptied only by a `destroy()` nothing called, so
  every distinct surface size a window drag passes through stayed resident.
  Measured over 120 widths: 120 textures, 1.84 GB. Now one. `PostStack`'s bind
  group cache had the same shape, and with the pool destroying textures it
  stopped being merely a leak.
- **Three NaN paths with no guard**: zeroing `sun.direction`, which is the
  natural way to say "no sun", produced NaN cascade matrices;
  `log(lightDistance / near)` was unguarded in three different ways; and a
  zero-area triangle in a mesh without normals left a zero normal that turned
  the whole face NaN in the shader. Exporters emit collapsed triangles
  routinely, and the tangent path twenty lines away already handled it.
- **Eight `destroy()` methods had no caller** and `Renderer` had none at all,
  so the teardown path existed on paper. `Engine.destroy()` also destroyed an
  `Environment` that `create()` documents as shareable, and a lost device left
  `_running` true forever so every later `run()` threw "already running". The
  decoded `ImageBitmap`s were never closed either -- over a gigabyte on Sponza.
- **A texture past `maxTextureDimension2D`** returned an invalid texture with
  no exception, after which the upload, the mip generation and every draw
  binding it were silent no-ops.

**Silent state corruption**

- **`createScene()` hijacked the previous scene.** Each call re-registered the
  compose job with a closure over the new scene and re-pointed the shared
  buffers, so with two scenes the workers composed one scene's columns for
  every frame of the other. `setSharedData` now publishes an owner, and
  `updateParallel` republishes when the owner is not itself -- a condition a
  per-store revision cannot express.
- **`mat4Decompose` wrote `outPos` before the guard** that documents the
  outputs as untouched on a degenerate matrix, so a caller trusting it got a
  transform mixed from two different matrices.
- **`createPipelineLayout`'s range check was DEBUG-only and ran after** the
  loop that ignores out-of-range keys, so in release a fifth bind group was
  dropped in silence.
- **`compileShaderSync` had no `.catch()`** on `getCompilationInfo()`, so a
  device lost mid-load left `shaderErrors` empty -- which the GPU suite reads
  as clean, meaning a compile failure could report as a pass.
- **`serve.js` tested `startsWith(ROOT)` with no separator**, so a sibling
  directory whose name merely extends the root's was served.

**Documentation that described something else**

`hzb.js` still said two-phase culling was "not done here"; `drawlist.js`
claimed opaque draws sort near-to-far for early-Z, which they never have;
`quat.js` said there is deliberately no `quatFromEuler` fifty lines above
exporting one; `assert.js` said every call site is DEBUG-guarded;
`bindgroups.js` described a four-rate model that is three; `material.js`
documented a texture slot named `orm` that does not exist, so anyone following
it got a flat material with no error; `mat4Invert` promised null on a singular
matrix when it only tests for an exactly-zero determinant; `pbr.js`'s
`instance_index` note is false on the transparent path. Four arithmetic counts
in comments were wrong where the code was right.

302 checks under Node, 12 on a real device.

## [0.1.1] - 2026-09-21

A correctness pass over the renderer's maths. Every entry below is a bug that
was in 0.1.0; nothing here changes an API.

Three of them are the same mistake: **a quantity defined against view-space
depth, consumed as radial distance from the eye.** Those agree only along the
view axis and diverge by 1/cos away from it, so each one was correct in the
middle of the frame and wrong toward the edges.

### Fixed

- **Clustered lighting read the wrong depth slice.** The compute pass builds
  froxels as bands of constant view z, but the fragment shader picked its slice
  with `length(world - cameraPosition)`. Fragments near the edges of the screen
  read a light list built for a band further away and lost the lights that
  should have reached them, which showed as hard stair-stepped cutoffs in the
  falloff. Now uses `1.0 / clip.w`, which is exactly `-viewZ` under this
  projection.
- **Shadow cascade selection had the same mismatch.** `cascadeSplits` are
  fitted as depths on the CPU, but `selectCascade` compared them against radial
  distance. Cascades nest outward, so this cost shadow resolution at the edges
  of the frame rather than producing holes.
- **Blended draws sorted by radial distance** where `transparentDepthBucket`
  documents, and computes with, view depth.
- **The frame uniform was uploaded before the values it copies were
  recomputed.** `shadows.update()` and `clusters.update()` ran *after* the
  upload, so the fragment shader read the previous frame's cascade matrices,
  splits, texel sizes, cluster tile size and light count while the shadow map
  was rasterised with the current ones. Shadows lagged a frame behind a moving
  camera, one frame after any resize used the wrong tile grid, and the very
  first frame read cascade matrices that were still zero.
- **Unused cascades were skipped with a sentinel of `Infinity`**, but
  `selectCascade` tests `viewDepth < split` and every depth is below infinity,
  so the sentinel selected the cascade it was meant to skip — an identity
  matrix and a texture layer that was never allocated. Only reachable with
  fewer than four cascades.
- **The shadow map's hardware depth bias had the wrong sign for reverse-Z.**
  A larger depth value is nearer, so the positive bias pushed blockers toward
  the light, which is the acne-worsening direction. Masked in practice by
  front-face culling and the normal-offset bias.
- **The pipeline cache key omitted `depthBiasSlopeScale`**, so two pipelines
  differing only in slope scale collapsed onto one cache entry.
- **An animation channel naming a node outside the asset's default scene drove
  entity 0.** The node-to-entity map is pre-filled with `NULL_HANDLE`, which is
  `0`, and the guard tested for `undefined`. Channels pointing at freed handles
  are now skipped too.
- **Swap-removing a renderable did not relocate its world bounds**, so the
  surviving object inherited the deleted one's box and kept it until it next
  moved: the GPU culled a visible mesh, and `raycast` returned the wrong thing.

### Changed

- `sliceFor` and the WGSL cluster lookup name their argument `depth` rather
  than `distance`. The old wording is what the first bug above was written
  from.

## [0.1.0] - 2026-09-21

First public release.

### Added

- Reverse-Z depth with an infinite far plane, carried through the whole engine:
  no far distance in the camera API, a five-plane frustum because the sixth is
  degenerate, `depth32float` cleared to 0 and compared with `greater`.
- GPU-driven rendering. Renderables batch by primitive and material; a compute
  pass does frustum and occlusion culling and writes indirect draw arguments,
  using an atomic as the allocator so no prefix sum is needed.
- Hierarchical-Z occlusion from the previous frame's depth, min-reduced, which
  is *max* under reverse-Z.
- A render graph that derives execution order, load/store ops, texture
  lifetimes and dead-pass elimination from declared reads and writes.
- Clustered forward lighting over a froxel grid with exponential Z slicing.
- Cascaded shadow maps: sphere-fitted cascades, texel snapping, normal-offset
  bias, front-face culling.
- PBR with Cook-Torrance GGX and split-sum IBL, the BRDF term as an analytic
  polynomial rather than a lookup texture.
- Bloom and ACES tonemapping in an HDR post stack.
- glTF 2.0 import: geometry, materials, images, and animation including all
  three interpolation modes.
- Transparency: blended geometry leaves the GPU-driven path and is sorted
  back-to-front on the CPU.
- Picking against the world bounds the culler already maintains.
- Growth on demand for every container, so the capacity arguments are starting
  sizes rather than budgets.
- A job system: atomic-cursor parallel-for across workers, degrading to an
  identical inline path without cross-origin isolation.
- 261 checks under Node, plus a browser suite that boots the engine on a real
  device and verifies what WGSL cannot be verified without one.

[Unreleased]: https://github.com/nolanbaxter/winding/compare/v0.9.1...HEAD
[0.9.1]: https://github.com/nolanbaxter/winding/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/nolanbaxter/winding/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/nolanbaxter/winding/compare/v0.7.3...v0.8.0
[0.7.3]: https://github.com/nolanbaxter/winding/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/nolanbaxter/winding/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/nolanbaxter/winding/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/nolanbaxter/winding/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/nolanbaxter/winding/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/nolanbaxter/winding/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/nolanbaxter/winding/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/nolanbaxter/winding/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/nolanbaxter/winding/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/nolanbaxter/winding/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/nolanbaxter/winding/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nolanbaxter/winding/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/nolanbaxter/winding/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nolanbaxter/winding/releases/tag/v0.1.0
