# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
