# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/nolanbaxter/winding/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/nolanbaxter/winding/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nolanbaxter/winding/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/nolanbaxter/winding/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nolanbaxter/winding/releases/tag/v0.1.0
