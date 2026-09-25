// Builds a .glb in memory so the demo exercises the REAL import path.
//
// The point is not the geometry -- it is that these bytes go through
// parseContainer -> accessors -> interleave -> tangent generation -> node
// decomposition -> instantiate, exactly as a file from Blender would. A demo
// that hand-fed vertex arrays to the GPU would prove none of it.
//
// Pass ?model=<url> on the page to load an actual .glb instead.

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

/** Unit cube: 24 vertices so each face gets its own normal and UV square. */
function cubeAttributes() {
  const faces = [
    { normal: [0, 0, 1], corners: [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]] },
    { normal: [0, 0, -1], corners: [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]] },
    { normal: [1, 0, 0], corners: [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]] },
    { normal: [-1, 0, 0], corners: [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]] },
    { normal: [0, 1, 0], corners: [[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]] },
    { normal: [0, -1, 0], corners: [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]] },
  ];
  const uvCorners = [[0, 1], [1, 1], [1, 0], [0, 0]];

  const positions = new Float32Array(24 * 3);
  const normals = new Float32Array(24 * 3);
  const uvs = new Float32Array(24 * 2);
  const indices = new Uint16Array(36);

  let v = 0, n = 0, t = 0, i = 0;
  faces.forEach((face, f) => {
    face.corners.forEach((corner, c) => {
      positions[v++] = corner[0] * 0.5;
      positions[v++] = corner[1] * 0.5;
      positions[v++] = corner[2] * 0.5;
      normals[n++] = face.normal[0];
      normals[n++] = face.normal[1];
      normals[n++] = face.normal[2];
      uvs[t++] = uvCorners[c][0];
      uvs[t++] = uvCorners[c][1];
    });
    const base = f * 4;
    indices[i++] = base; indices[i++] = base + 1; indices[i++] = base + 2;
    indices[i++] = base; indices[i++] = base + 2; indices[i++] = base + 3;
  });

  return { positions, normals, uvs, indices };
}

function packBuffer(arrays) {
  let total = 0;
  const views = arrays.map((a) => {
    const byteOffset = total;
    total += (a.byteLength + 3) & ~3;          // bufferViews must stay 4-aligned
    return { byteOffset, byteLength: a.byteLength };
  });
  const bytes = new Uint8Array(total);
  arrays.forEach((a, k) => {
    bytes.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), views[k].byteOffset);
  });
  return { bytes, views };
}

function pad4(bytes, fill) {
  const size = (bytes.length + 3) & ~3;
  if (size === bytes.length) return bytes;
  const out = new Uint8Array(size).fill(fill);
  out.set(bytes);
  return out;
}

function encodeGLB(json, binary) {
  const jsonChunk = pad4(new TextEncoder().encode(JSON.stringify(json)), 0x20);
  const binChunk = pad4(binary, 0);
  const total = 12 + 8 + jsonChunk.length + 8 + binChunk.length;

  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonChunk.length, true);
  view.setUint32(16, CHUNK_JSON, true);
  out.set(jsonChunk, 20);

  const o = 20 + jsonChunk.length;
  view.setUint32(o, binChunk.length, true);
  view.setUint32(o + 4, CHUNK_BIN, true);
  out.set(binChunk, o + 8);
  return out;
}

/**
 * A PNG generated at runtime and embedded in the GLB.
 *
 * Encoding a real PNG matters: it means the demo goes through the actual image
 * path -- Blob, createImageBitmap, sRGB texture, mip generation -- instead of
 * handing the GPU a texture the loader never had to decode.
 */
async function texturePNG(size = 256) {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d');

  const cells = 8;
  const cell = size / cells;
  for (let y = 0; y < cells; y++) {
    for (let x = 0; x < cells; x++) {
      const dark = (x + y) % 2 === 0;
      ctx.fillStyle = dark ? '#cfd6e0' : '#6d7a8c';
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }

  // An asymmetric mark, so a flipped or rotated UV mapping is obvious rather
  // than hidden by the checker's symmetry.
  ctx.fillStyle = '#e8613c';
  ctx.beginPath();
  ctx.moveTo(size * 0.5, size * 0.18);
  ctx.lineTo(size * 0.74, size * 0.62);
  ctx.lineTo(size * 0.26, size * 0.62);
  ctx.closePath();
  ctx.fill();

  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return new Uint8Array(await blob.arrayBuffer());
}

/** A large floor quad, facing +Y. Shadows need something to land on. */
function groundAttributes(half = 14) {
  return {
    positions: Float32Array.from([
      -half, 0, -half, half, 0, -half, half, 0, half, -half, 0, half,
    ]),
    normals: Float32Array.from([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    uvs: Float32Array.from([0, 0, 8, 0, 8, 8, 0, 8]),
    // Counter-clockwise seen from above, matching frontFace 'ccw'.
    indices: Uint16Array.from([0, 3, 2, 0, 2, 1]),
  };
}

/**
 * A three-level hierarchy: one hub, `arms` spokes parented to it, and a small
 * cube parented to each spoke. Rotating only the hub moves all 13 nodes, which
 * is the transform hierarchy doing its job where you can see it.
 *
 * Node 0 deliberately uses `matrix` form rather than TRS, so the decomposition
 * path runs in the demo too.
 */
export async function buildDemoGLB({ arms = 6, glass = 3 } = {}) {
  const cube = cubeAttributes();
  const ground = groundAttributes();
  const png = await texturePNG();
  // Keyframes for the demo clip: rise, return, dip, return. Five keys, because
  // the last has to repeat the first for the loop to wrap without a jerk.
  const animTimes = Float32Array.from([0, 0.5, 1, 1.5, 2]);
  const animBob = Float32Array.from([
    0, 1.6, 0,
    0, 2.5, 0,
    0, 1.6, 0,
    0, 0.9, 0,
    0, 1.6, 0,     // equal to the first, or the wrap is a visible jerk
  ]);

  const { bytes, views } = packBuffer([
    cube.positions, cube.normals, cube.uvs, cube.indices, png,
    ground.positions, ground.normals, ground.uvs, ground.indices,
    animTimes, animBob,
  ]);

  const nodes = [{
    name: 'hub',
    // Identity, in matrix form: exercises mat4Decompose on load.
    matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    children: [],
    mesh: 0,
  }];

  // The floor is a sibling root, so spinning the hub does not spin the ground.
  const groundNode = {
    name: 'ground', translation: [0, -1.6, 0], mesh: 1,
  };

  for (let a = 0; a < arms; a++) {
    const angle = (a / arms) * Math.PI * 2;
    const spoke = nodes.length;
    nodes[0].children.push(spoke);

    nodes.push({
      name: `spoke_${a}`,
      translation: [Math.cos(angle) * 3, 0, Math.sin(angle) * 3],
      rotation: [0, Math.sin(angle / 2), 0, Math.cos(angle / 2)],
      scale: [0.7, 0.7, 0.7],
      mesh: 0,
      children: [nodes.length + 1],
    });

    nodes.push({
      name: `tip_${a}`,
      translation: [0, 1.6, 0],
      scale: [0.45, 0.45, 0.45],
      mesh: 0,
    });
  }

  const glassNodes = [];
  for (let g = 0; g < glass; g++) {
    glassNodes.push({
      name: `glass_${g}`,
      translation: [(g - (glass - 1) / 2) * 1.1, 0.6, g * 1.3 - 1.3],
      scale: [2.4, 2.4, 0.06],
      mesh: 2,
    });
  }

  const json = {
    asset: { version: '2.0', generator: 'demo' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [
      {
        bufferView: 0, componentType: 5126, count: 24, type: 'VEC3',
        min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5],
      },
      { bufferView: 1, componentType: 5126, count: 24, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 24, type: 'VEC2' },
      { bufferView: 3, componentType: 5123, count: 36, type: 'SCALAR' },
      {
        bufferView: 5, componentType: 5126, count: 4, type: 'VEC3',
        min: [-14, 0, -14], max: [14, 0, 14],
      },
      { bufferView: 6, componentType: 5126, count: 4, type: 'VEC3' },
      { bufferView: 7, componentType: 5126, count: 4, type: 'VEC2' },
      { bufferView: 8, componentType: 5123, count: 6, type: 'SCALAR' },
      { bufferView: 9, componentType: 5126, count: 5, type: 'SCALAR' },
      { bufferView: 10, componentType: 5126, count: 5, type: 'VEC3' },
    ],
    // No TANGENT on purpose: the importer derives it from the UVs.
    meshes: [
      {
        name: 'cube',
        primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 0 }],
      },
      {
        name: 'ground',
        primitives: [{ attributes: { POSITION: 4, NORMAL: 5, TEXCOORD_0: 6 }, indices: 7, material: 1 }],
      },
      {
        name: 'glass',
        primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 2 }],
      },
    ],
    // The PNG rides in the same BIN chunk as the geometry, referenced by a
    // bufferView -- which is exactly how a real .glb ships its textures.
    images: [{ name: 'checker', bufferView: 4, mimeType: 'image/png' }],
    samplers: [{ magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 10497 }],
    textures: [{ source: 0, sampler: 0 }],
    materials: [
      {
        name: 'demo',
        pbrMetallicRoughness: {
          baseColorTexture: { index: 0 },
          baseColorFactor: [1, 1, 1, 1],
          metallicFactor: 0,
          roughnessFactor: 0.6,
        },
      },
      {
        name: 'ground',
        pbrMetallicRoughness: {
          baseColorFactor: [0.42, 0.44, 0.47, 1],
          metallicFactor: 0,
          roughnessFactor: 0.9,
        },
      },
      {
        // The whole point of the glass panes: overlapping BLEND surfaces are
        // the only thing that shows whether draw order is actually being
        // sorted, because a wrong order still renders -- it just looks wrong.
        name: 'glass',
        alphaMode: 'BLEND',
        doubleSided: true,
        pbrMetallicRoughness: {
          baseColorFactor: [0.35, 0.72, 0.9, 0.4],
          metallicFactor: 0,
          roughnessFactor: 0.1,
        },
      },
    ],
    // One channel per tip node. They share a sampler, which is normal in real
    // assets and exercises the case where several channels read one curve.
    animations: [{
      name: 'bob',
      samplers: [{ input: 8, output: 9, interpolation: 'LINEAR' }],
      channels: nodes
        .map((node, i) => ({ node, i }))
        .filter(({ node }) => node.name?.startsWith('tip_'))
        .map(({ i }) => ({ sampler: 0, target: { node: i, path: 'translation' } })),
    }],
    nodes: [...nodes, groundNode, ...glassNodes],
    scenes: [{ nodes: [0, nodes.length, ...glassNodes.map((_, g) => nodes.length + 1 + g)] }],
    scene: 0,
  };

  return encodeGLB(json, bytes);
}

/**
 * A rigged quad: two joints, both at the origin with identity inverse binds.
 *
 * Bind pose on purpose. With an identity palette a skinned draw must produce
 * exactly the vertices an unskinned one would, which makes it the strongest
 * check available before any animation exists -- an error in the multiply
 * order, the joint-to-entity map or the vertex buffer shows up as a deformed
 * or vanished mesh rather than as something subtly off.
 */
export function buildRiggedGLB({ morphed = false } = {}) {
  const positions = Float32Array.from([
    -1, 0, 0, 1, 0, 0, -1, 2, 0, 1, 2, 0,
  ]);
  const normals = Float32Array.from([
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
  ]);
  const uvs = Float32Array.from([0, 1, 1, 1, 0, 0, 1, 0]);
  const indices = Uint16Array.from([0, 1, 2, 2, 1, 3]);
  // Bottom edge to joint 0, top edge to joint 1.
  const joints = Uint16Array.from([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
  const weights = Float32Array.from([
    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
  ]);
  const inverseBind = Float32Array.from([
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
  ]);

  // One morph target when asked for: the top edge forward by 3, on an axis
  // neither the skeleton nor the other check touches, so the two corrections
  // to the bounds stay distinguishable.
  const target = Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3]);

  const arrays = [positions, indices, normals, uvs, joints, weights, inverseBind];
  if (morphed) arrays.push(target);
  const { bytes, views } = packBuffer(arrays);

  return encodeGLB({
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [
      { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3', min: [-1, 0, 0], max: [1, 2, 0] },
      { bufferView: 1, componentType: 5123, count: 6, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: 4, type: 'VEC3' },
      { bufferView: 3, componentType: 5126, count: 4, type: 'VEC2' },
      { bufferView: 4, componentType: 5123, count: 4, type: 'VEC4' },
      { bufferView: 5, componentType: 5126, count: 4, type: 'VEC4' },
      { bufferView: 6, componentType: 5126, count: 2, type: 'MAT4' },
      ...(morphed ? [{ bufferView: 7, componentType: 5126, count: 4, type: 'VEC3' }] : []),
    ],
    materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.8, 0.3, 0.3, 1] } }],
    meshes: [{
      name: 'rigged',
      weights: morphed ? [0] : undefined,
      primitives: [{
        attributes: { POSITION: 0, NORMAL: 2, TEXCOORD_0: 3, JOINTS_0: 4, WEIGHTS_0: 5 },
        indices: 1,
        material: 0,
        targets: morphed ? [{ POSITION: 7 }] : undefined,
      }],
    }],
    skins: [{ joints: [1, 2], inverseBindMatrices: 6 }],
    nodes: [
      { name: 'rigged-root', mesh: 0, skin: 0, children: [1, 2] },
      { name: 'joint-hip' },
      { name: 'joint-chest' },
    ],
    scenes: [{ nodes: [0] }],
    scene: 0,
  }, bytes);
}

/**
 * The same quad with two morph targets and no skin.
 *
 * Target 0 raises the top edge by 10, target 1 pushes the right edge out by 4.
 * Deliberately asymmetric in both the axis and the vertices they touch, so a
 * transposed index or a swapped target lands somewhere the check can see.
 */
export function buildMorphedGLB() {
  const positions = Float32Array.from([
    -1, 0, 0, 1, 0, 0, -1, 2, 0, 1, 2, 0,
  ]);
  const normals = Float32Array.from([
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
  ]);
  const uvs = Float32Array.from([0, 1, 1, 1, 0, 0, 1, 0]);
  const indices = Uint16Array.from([0, 1, 2, 2, 1, 3]);

  // Top edge up.
  const target0 = Float32Array.from([0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 10, 0]);
  // Right edge out.
  const target1 = Float32Array.from([0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0]);

  const { bytes, views } = packBuffer([positions, indices, normals, uvs, target0, target1]);

  return encodeGLB({
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors: [
      { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3', min: [-1, 0, 0], max: [1, 2, 0] },
      { bufferView: 1, componentType: 5123, count: 6, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: 4, type: 'VEC3' },
      { bufferView: 3, componentType: 5126, count: 4, type: 'VEC2' },
      { bufferView: 4, componentType: 5126, count: 4, type: 'VEC3' },
      { bufferView: 5, componentType: 5126, count: 4, type: 'VEC3' },
    ],
    materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.3, 0.6, 0.8, 1] } }],
    meshes: [{
      name: 'morphed',
      weights: [0, 0],
      primitives: [{
        attributes: { POSITION: 0, NORMAL: 2, TEXCOORD_0: 3 },
        indices: 1,
        material: 0,
        targets: [{ POSITION: 4 }, { POSITION: 5 }],
      }],
    }],
    nodes: [{ name: 'morphed-root', mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  }, bytes);
}

/**
 * One quad, filling the view, with exactly one feature turned on.
 *
 * Every other fixture here is a scene. This is a probe: the point is that a
 * check can render it and read the middle pixel, so each feature gets a single
 * unambiguous answer instead of a screenshot somebody has to squint at.
 *
 * It exists because the rendered fixtures between them used ONE material
 * texture slot, one alpha mode that draws, positive scales only, indexed
 * geometry only and supplied normals only. Everything outside that had been
 * imported and tested on the CPU, compiled into a pipeline, and never once
 * turned into a pixel -- which is how a constant tangent and a black emissive
 * default both shipped.
 */
export function buildFeatureGLB({
  // A KHR_lights_punctual light on its own node: { translation, light }.
  lamp = null,
  metallicFactor,
  roughnessFactor,
  baseColorFactor = [0.9, 0.15, 0.1, 1],
  alphaMode,
  alphaCutoff,
  emissiveFactor,
  colors = null,
  includeNormals = true,
  indexed = true,
  uv0 = null,
  uv1 = null,
  baseColorTexCoord,
  // KHR_texture_transform on the base colour texture: { offset, rotation, scale }.
  baseColorTransform = null,
  imageURI = null,
  normalImageURI = null,
  normalScale,
  nodeScale,
  // The material's extensions object, as the file would carry it.
  materialExtensions = null,
  doubleSided = false,
  // Images after the base colour and normal ones, for extension textures.
  extraImageURIs = [],
} = {}) {
  const S = 1.6;
  const corners = [[-S, -S], [S, -S], [S, S], [-S, S]];
  const uvCorners = [[0, 0], [1, 0], [1, 1], [0, 1]];
  // A non-indexed primitive is the same quad written out as six vertices,
  // which is exactly what the importer's sequentialIndices path has to cope
  // with and what nothing had ever drawn.
  const order = indexed ? [0, 1, 2, 3] : [0, 1, 2, 0, 2, 3];

  const positions = Float32Array.from(order.flatMap((i) => [...corners[i], 0]));
  const normals = Float32Array.from(order.flatMap(() => [0, 0, 1]));
  const uvs = Float32Array.from(order.flatMap((i) => (uv0 ? uv0.slice(i * 2, i * 2 + 2) : uvCorners[i])));
  const count = order.length;

  const arrays = [positions, normals, uvs];
  const accessors = [
    { componentType: 5126, count, type: 'VEC3', min: [-S, -S, 0], max: [S, S, 0] },
    { componentType: 5126, count, type: 'VEC3' },
    { componentType: 5126, count, type: 'VEC2' },
  ];
  const attributes = { POSITION: 0, TEXCOORD_0: 2 };
  if (includeNormals) attributes.NORMAL = 1;
  let next = 3;

  let indicesAccessor;
  if (indexed) {
    arrays.push(Uint16Array.from([0, 1, 2, 0, 2, 3]));
    accessors.push({ componentType: 5123, count: 6, type: 'SCALAR' });
    indicesAccessor = next++;
  }
  if (colors) {
    arrays.push(Float32Array.from(order.flatMap((i) => colors.slice(i * 4, i * 4 + 4))));
    accessors.push({ componentType: 5126, count, type: 'VEC4' });
    attributes.COLOR_0 = next++;
  }
  if (uv1) {
    arrays.push(Float32Array.from(order.flatMap((i) => uv1.slice(i * 2, i * 2 + 2))));
    accessors.push({ componentType: 5126, count, type: 'VEC2' });
    attributes.TEXCOORD_1 = next++;
  }

  const { bytes, views } = packBuffer(arrays);
  accessors.forEach((a, i) => { a.bufferView = i; });

  const pbr = { baseColorFactor };
  if (metallicFactor !== undefined) pbr.metallicFactor = metallicFactor;
  if (roughnessFactor !== undefined) pbr.roughnessFactor = roughnessFactor;
  if (imageURI) {
    pbr.baseColorTexture = { index: 0 };
    if (baseColorTexCoord !== undefined) pbr.baseColorTexture.texCoord = baseColorTexCoord;
    if (baseColorTransform) pbr.baseColorTexture.extensions = { KHR_texture_transform: baseColorTransform };
  }

  const material = { name: 'probe', pbrMetallicRoughness: pbr };
  if (normalImageURI) {
    material.normalTexture = { index: imageURI ? 1 : 0 };
    if (normalScale !== undefined) material.normalTexture.scale = normalScale;
  }
  if (alphaMode) material.alphaMode = alphaMode;
  if (alphaCutoff !== undefined) material.alphaCutoff = alphaCutoff;
  if (emissiveFactor) material.emissiveFactor = emissiveFactor;
  if (materialExtensions) material.extensions = materialExtensions;
  if (doubleSided) material.doubleSided = true;

  const primitive = { attributes, material: 0 };
  if (indexed) primitive.indices = indicesAccessor;

  const node = { name: 'probe', mesh: 0 };
  if (nodeScale) node.scale = nodeScale;

  const json = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors,
    materials: [material],
    meshes: [{ name: 'probe', primitives: [primitive] }],
    nodes: [node],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  if (lamp) {
    json.extensionsUsed = ['KHR_lights_punctual'];
    json.extensions = { KHR_lights_punctual: { lights: [lamp.light] } };
    json.nodes.push({ name: 'lamp', translation: lamp.translation, extensions: { KHR_lights_punctual: { light: 0 } } });
    json.scenes[0].nodes.push(1);
  }
  if (materialExtensions) json.extensionsUsed = [...(json.extensionsUsed ?? []), ...Object.keys(materialExtensions)];
  const uris = [imageURI, normalImageURI, ...extraImageURIs].filter(Boolean);
  if (uris.length > 0) {
    json.images = uris.map((uri) => ({ uri }));
    json.samplers = [{ magFilter: 9728, minFilter: 9728 }];   // NEAREST, so halves stay crisp
    json.textures = uris.map((_, i) => ({ source: i, sampler: 0 }));
  }

  return encodeGLB(json, bytes);
}

/** A 2x1 PNG as a data URI: left half one colour, right half another. */
export function twoToneImageURI(left, right) {
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 1;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = left;
  ctx.fillRect(0, 0, 1, 1);
  ctx.fillStyle = right;
  ctx.fillRect(1, 0, 1, 1);
  return canvas.toDataURL('image/png');
}

/**
 * One quad at three levels of detail (MSFT_lod), each level an unlit colour
 * -- red, green, blue, finest first -- so which level drew is one pixel's
 * question. `coverage` is MSFT_screencoverage, one value a level.
 */
export function buildLodGLB(coverage = [0.5, 0.2, 0.01]) {
  const S = 1.6;
  const positions = Float32Array.from([-S, -S, 0, S, -S, 0, S, S, 0, -S, S, 0]);
  const normals = Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const uvs = Float32Array.from([0, 0, 1, 0, 1, 1, 0, 1]);
  const indices = Uint16Array.from([0, 1, 2, 0, 2, 3]);
  const { bytes, views } = packBuffer([positions, normals, uvs, indices]);
  const accessors = [
    { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3', min: [-S, -S, 0], max: [S, S, 0] },
    { bufferView: 1, componentType: 5126, count: 4, type: 'VEC3' },
    { bufferView: 2, componentType: 5126, count: 4, type: 'VEC2' },
    { bufferView: 3, componentType: 5123, count: 6, type: 'SCALAR' },
  ];
  const colours = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]];
  const json = {
    asset: { version: '2.0' },
    extensionsUsed: ['MSFT_lod', 'KHR_materials_unlit'],
    buffers: [{ byteLength: bytes.length }],
    bufferViews: views.map((v) => ({ buffer: 0, ...v })),
    accessors,
    materials: colours.map((baseColorFactor) => ({
      pbrMetallicRoughness: { baseColorFactor }, extensions: { KHR_materials_unlit: {} },
    })),
    meshes: colours.map((_, m) => ({
      primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: m }],
    })),
    nodes: [
      { name: 'high', mesh: 0, extensions: { MSFT_lod: { ids: [1, 2] } }, extras: { MSFT_screencoverage: coverage } },
      { name: 'medium', mesh: 1 },
      { name: 'low', mesh: 2 },
    ],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  return encodeGLB(json, bytes);
}
