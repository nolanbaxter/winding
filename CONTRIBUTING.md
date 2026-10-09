# Contributing to Winding

Thank you for helping. Winding is a WebGPU engine for the browser with no dependencies and no
build step, and every change keeps it that way: plain ES modules, served as they are.

## Branches

- **`develop`** is where work lands. Branch from it, and open pull requests against it.
- **`main`** is released code only. A release is a pull request from `develop` into `main` that
  moves the changelog's `[Unreleased]` section under a version and bumps `package.json`; merging
  it tags the release and publishes it to npm. Nothing else reaches `main`.

## Running the tests

```bash
npm test
```

The Node suites: about 750 checks, no browser, no install (there is nothing to install). CI runs
them on Node 20, 22 and 24 -- the ones your change reaches (see below).

```bash
npm run test:gpu
```

Serves the repository at <http://localhost:8080>; open `/test/gpu.html` in a browser with WebGPU.
It draws real frames and reads them back, and it is the only thing that tests WGSL and what
reaches the screen. CI runs it as well, in headless Chrome on a software GPU
(`npm run test:gpu:headless` does the same on your machine), so a shader that fails to compile
fails the pull request. A software GPU is not your GPU, though: run the page on real hardware
before any pull request that touches `src/render` or `src/rhi`, and say in the pull request that
it passed.

### What CI runs

Only what a change reaches. `.github/affected.js` reads the change and picks the Node suites whose
imports reach a changed file, the type check if declarations changed, and the GPU checks that run a
changed function -- from `test/gpu-map.json`, which records, for every function in `src/` the GPU
suite runs, which checks run it. Those are split across up to four runners. A change to what every
check sets up, to the GPU suite itself, or to CI runs all of it; docs, the changelog and benchmarks
run nothing of it. `node .github/affected.js <base> <head>` shows the plan for any range.

The map stays usable as the code moves -- line numbers are carried from its commit through git --
and a check it does not know always runs. Remake it after adding or reshaping GPU checks, or once a
release, on a machine with a GPU and `src/` committed:

```bash
npm run test:gpu:map
```

## What a change includes

- **A test that fails without it.** A bug fix starts from a script or a frame that shows the bug;
  the fix comes with a test that fails on the old code. Node tests live in `test/*.test.js`, GPU
  steps in `test/gpu.test.js`.
- **The API reference.** `docs/API.md` is updated in the same change as anything a user can see:
  a new call, a changed option or default, a new error message, behaviour a fix changes. A new
  entry gets an `<a id>` anchor and a line in the index. `npm test` checks that every public method
  has an entry and every link lands; options, defaults and wording are checked by you.
- **The changelog.** A line under `## [Unreleased]` in `CHANGELOG.md`, in Added, Changed or Fixed.
- **The README**, when a feature or a limitation changes.

## Versions

Winding follows [semantic versioning](https://semver.org) against `docs/API.md`:

- **patch** (1.2.x): fixes to documented behaviour, performance, docs;
- **minor** (1.x.0): additions, and renames that keep the old name working with a warning;
- **major** (x.0.0): anything that breaks code written against the API reference.

## Performance

A change made for speed comes with a measurement: the old code and the new, timed in one page in
alternating blocks (`src/bench.js` times every pass), since separate runs drift by more than most
changes. A change that measures no better is left out, however sensible it looked.

## Style

Match the code around you. Comments say why, in full sentences. Names are short and real --
`resolution`, not `renderScaleFactorValue`, and not an invented word. No new dependencies.

## Reporting

- **Bugs:** open an issue with the bug form. The browser, GPU and the GPU test page's result make
  most reports answerable at once.
- **Questions and ideas:** [Discussions](https://github.com/nolanbaxter/winding/discussions).
- **Security issues:** privately, as [SECURITY.md](SECURITY.md) says, not in an issue.

By taking part you agree to the [code of conduct](CODE_OF_CONDUCT.md).
