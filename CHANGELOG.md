# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Three of the README's limitations, worked through in order of what they cost
to fix rather than what they cost to describe.

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

### Changed

- **Occlusion culling runs in two phases and is no longer a frame stale.**
  What was drawn last frame is drawn first, the depth pyramid is built from
  that, and a second cull tests everything else against it before a second
  pass draws whatever it newly admits. An object that becomes visible now
  appears on the frame it does instead of popping in on the next, because the
  pyramid and the matrix it is projected with both belong to this frame.
  `lastViewProjection` is gone. The two phases share one indirect buffer, one
  visible list and one batch-info buffer, each doubled, so this costs one
  compute dispatch and one set of indirect draws rather than a second copy of
  the machinery.
- **The render graph versions resources that are written more than once.** A
  read edged from every writer regardless of declaration order, which is right
  for a single-writer resource and wrong for a sequence: the depth pyramid
  reads depth between the two forward passes, and edging it from both put it
  after a pass that depends on it. A resource written several times is a
  sequence of values and a read means the one current where it was declared.
  Single-writer resources behave exactly as before, which is the case the rule
  was written for.

### Fixed

- The aliasing limitation was described as never triggering "because every
  transient target is a distinct size". Measured, the reason is different: the
  bloom chain does contain same-size pairs, but `bloom{i}` and `bloom-up{i}`
  overlap in lifetime by construction, so there is nothing to share. Adding a
  three-pass separable blur aliases immediately. The pass is idle here, not
  speculative, and its own tests already covered it.

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

[Unreleased]: https://github.com/nolanbaxter/winding/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/nolanbaxter/winding/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nolanbaxter/winding/releases/tag/v0.1.0
