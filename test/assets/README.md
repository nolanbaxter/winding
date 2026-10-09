# Test assets

Models and other files for trying the engine against: tests, benchmarks and examples load them from
here. They are not published -- the npm package carries `src/` and the API reference only -- and
none of the engine depends on them. Models built in code live beside this, in `test/fixtures/`.

Every file is listed below with where it came from and the terms it is under. A file here with no
entry is a mistake.

| File | Size | What it is | Origin and licence |
|---|---|---|---|
| `walker.glb` | 28 KB | A geometric robot, 6 ft tall, built from boxes, spheres and cylinders in four greys. glTF 2.0, binary embedded; no textures, skins or animation. | Made for this project. MIT, as the repository. |

## walker.glb

- **Scale and placement:** metres, origin at the feet, facing −Z, standing with its arms down.
- **Structure:** 48 nodes, 30 of them drawn parts, sharing 5 meshes and 4 materials (`shell`,
  `mid`, `dark`, `visor`).
- **Joints:** 18 named pivot nodes the parts hang from, so it can be posed or animated node by node:
  `walker`, `hips`, `spine`, `neck`, and `shoulder`, `elbow`, `wrist`, `leg`, `hip`, `knee` and
  `ankle`, each `.R` and `.L`.
