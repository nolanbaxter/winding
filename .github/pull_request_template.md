## What this changes

<!-- What it does and why. For a fix, how to see the bug before it. -->

## Checklist

- [ ] Against `develop`, not `main`
- [ ] A test that fails without this change
- [ ] `npm test` passes
- [ ] The GPU test page passes (`npm run test:gpu`, then `/test/gpu.html`), if this touches `src/render` or `src/rhi`
- [ ] `docs/API.md` updated for anything a user can see
- [ ] A line in `CHANGELOG.md` under `[Unreleased]`
- [ ] For a speed-up: before and after, measured in one page
- [ ] No new dependencies, no build step
