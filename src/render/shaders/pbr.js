// Physically based surface shader: metallic-roughness and the glTF material
// extensions, normal mapped, lit by directional and clustered punctual lights
// plus the prebaked environment.
//
// Everything in here happens in LINEAR light. The only encode is the -srgb
// swap-chain view at the very end, which is why there is no
// pow(x, 1/2.2) anywhere below.

import { BRDF_WGSL } from './brdf.js';
import { MAX_LIGHTS_PER_CLUSTER, CLUSTER_COUNT, DECAL_INDEX_BASE } from '../clustered.js';
import { MATERIAL_WGSL, EXTENSION_BINDING } from '../material.js';
import { EXTENSION_TEXTURES, CORE_TEXTURE_COUNT } from '../../scene/gltf/images.js';
import { SHEEN_ALBEDO, SHEEN_TABLE_SIZE } from '../sheen.js';
import { FOG_WGSL } from '../fog.js';

/**
 * Depth complexity the OIT weighting stays well-behaved at.
 *
 * The accumulation target is rgba16float, whose largest finite value is 65504.
 * Dividing that by this budget is the largest weight a single fragment may
 * carry, so this many fully-weighted layers can sum without saturating. Past
 * it the sum clips and the nearest layers stop dominating, which shows as
 * transparency that flattens rather than as anything breaking.
 */
export const OIT_LAYER_BUDGET = 64;

/**
 * The per-frame uniform, as WGSL: the surface shader's, and the sprites' too,
 * which read its camera and fog.
 */
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  viewProjection : mat4x4<f32>,             //   0
  cameraPosition : vec4<f32>,               //  64
  environment    : vec4<f32>,               //  80  x = prefiltered mip count
  cascadeInfo    : vec4<f32>,               //  96  x = cascades per casting light, 0 for none
  cascadeSplits  : vec4<f32>,               // 112  view DEPTH each cascade ends at
  cascadeTexel   : vec4<f32>,               // 128  world size of one texel, per cascade
  shadowParams   : vec4<f32>,               // 144  x = normal bias, y = map size, z = camera near, w = local map size
  clusterGrid    : vec4<u32>,               // 160  x, y, z cells; w = light count
  clusterDepth   : vec4<f32>,               // 176  x slice scale, y bias, zw tile size
  cameraForward  : vec4<f32>,               // 192  world-space view axis; w = directional count
  // Fog (render/fog.js): x extinction at y height, z 1 / scale height; x = 0 for none.
  fog            : vec4<f32>,               // 208
  fogAlbedo      : vec4<f32>,               // 224
  fogLight       : vec4<f32>,               // 240  the directional lights' inscattered radiance
  probeInfo      : vec4<f32>,               // 256  x = reflection probes to consult
  // Written once, not per frame: the sheen albedo table (render/sheen.js),
  // four entries a vec4 because a uniform array's stride is 16.
  sheenAlbedo    : array<vec4<f32>, ${SHEEN_ALBEDO.length / 4}>,   // 272
};
`;

/** Each extension texture kind's index, for sampleExtension. */
const KIND = Object.fromEntries(EXTENSION_TEXTURES.map(({ slot }, k) => [slot, `${k}u`]));

/**
 * The forward shader, for a device that binds `extensionSlots` extension
 * textures a material (see extensionSlotCount in material.js).
 */
export const pbrShader = (extensionSlots) => /* wgsl */ `
${BRDF_WGSL}
${FOG_WGSL}

// Pipeline-overridable constant: set per variant, so the MASK variant gets its
// own compiled pipeline and opaque geometry never pays for the discard.
override USE_ALPHA_MASK : bool = false;

// Whether the material extensions' layers are compiled in (VARIANT_EXTENDED
// in material.js). Off, every branch below that reads them folds away, and a
// surface without them does not pay for their registers.
override EXTENSIONS : bool = true;

// Whether reflection probes are read (render/probes.js). Off unless the scene
// has captured probes: carrying the lookup cost Sponza, which has none, 6% of
// its forward time -- 6.16 ms against 5.81 at 1280x720.
override PROBES : bool = true;

// Whether decals are painted (render/decals.js). Off unless the scene has
// some, for the same reason as PROBES.
override DECALS : bool = true;

/**
 * The ambient term of the fragment shade() last ran, for fsAO to write out
 * beside the colour. Private to the invocation, so it is one fragment's.
 */
var<private> ambientOut : vec3<f32>;

/** The smallest normal f32: below it a squared length has no direction left. */
const F32_MIN_NORMAL : f32 = 1.17549435e-38;

${FRAME_WGSL}

/** A decal (render/decals.js). */
struct Decal {
  worldToDecal : mat4x4<f32>,   // its box onto [-1, 1] on every axis
  color        : vec4<f32>,
  facing       : vec4<f32>,     // xyz: its +Z in world space; w = its layer in decalMaps
};

/** A reflection probe (render/probes.js). */
struct Probe {
  boxMin   : vec4<f32>,   // w = how far in from its faces it fades in; 0 for a hard edge
  boxMax   : vec4<f32>,   // w = its layer in probeMaps
  position : vec4<f32>,   // where it captured from
};

/** A directional light. Packed by Scene.refreshLights. */
struct Directional {
  direction : vec4<f32>,   // the way its light travels; w = shadow slot + 1, 0 for none
  color     : vec4<f32>,   // colour times intensity
};

struct Light {
  positionRadius : vec4<f32>,
  colorIntensity : vec4<f32>,
  directionCone  : vec4<f32>,   // w = shadow views: 0, 1 (a spot's), or 6 (a cube)
  coneFalloff    : vec4<f32>,   // x scale, y offset, z type (0 point, 1 spot), w = first shadow layer + 1
};

/** A point or spot light's shadow view. params.x is tan(half the field of view). */
struct LocalView {
  viewProjection : mat4x4<f32>,
  params         : vec4<f32>,
};

/**
 * Per-object data, indexed by the shader rather than bound per draw.
 *
 * In a STORAGE buffer this is 112 bytes. As a dynamic uniform it had to sit on
 * a 256-byte boundary, so 144 of every 256 bytes was padding: 69% waste,
 * measured. Storage has no such rule, and the index can come from a compute
 * shader instead of from the CPU.
 */
struct DrawData {
  model         : mat4x4<f32>,    //   0
  normalMatrix  : mat3x3<f32>,    //  64   occupies 48 bytes
  // Where this instance's joint matrices begin. Zero and unread for anything
  // not skinned.
  paletteOffset : u32,            // 112
  // Where this PRIMITIVE's morph deltas begin, as a float index into one
  // engine-wide arena.
  morphBase     : u32,            // 116
  // Where this INSTANCE's weights begin. Two faces sharing a mesh share
  // morphBase and differ here, which is what lets them share a draw call.
  morphWeights  : u32,            // 120
  // Target count in the low 16 bits, floats per target per vertex in the high
  // 16. Packed because splitting them would push this struct to 144 bytes for
  // twelve bits; see packMorphCountStride in render/morph.js.
  morphCount    : u32,            // 124
};                                // 128, exactly: mat3x3 aligns the struct to 16

/** The only thing still bound per draw: where this batch's slice begins. */
struct Batch {
  firstVisible : u32,
};

${MATERIAL_WGSL}

@group(0) @binding(0) var<uniform> frame        : Frame;
@group(0) @binding(1) var          irradiance   : texture_cube<f32>;
@group(0) @binding(2) var          prefiltered  : texture_cube<f32>;
@group(0) @binding(3) var          envSampler   : sampler;
@group(0) @binding(4) var          shadowMap    : texture_depth_2d_array;
// A comparison sampler does the depth test and the bilinear blend in one fetch,
// so each PCF tap returns an already-filtered occlusion fraction.
@group(0) @binding(5) var          shadowSampler: sampler_comparison;
@group(0) @binding(15) var         localShadowMap : texture_depth_2d_array;
@group(0) @binding(16) var<storage, read> localViews : array<LocalView>;
@group(0) @binding(6) var<storage, read> lights         : array<Light>;
@group(0) @binding(7) var<storage, read> clusterIndices : array<u32>;
@group(0) @binding(8) var<storage, read> clusterCounts  : array<u32>;
@group(0) @binding(9) var<storage, read> drawData       : array<DrawData>;
// Written by the cull compute shader: the compacted list of surviving objects,
// grouped into one contiguous slice per batch.
@group(0) @binding(10) var<storage, read> visibleItems  : array<u32>;
// Every skinned instance's joint matrices, end to end. One buffer for the
// frame, indexed by draw.paletteOffset + the vertex's joint.
@group(0) @binding(11) var<storage, read> palette       : array<mat4x4<f32>>;
// Every morphed primitive's target deltas, end to end. Static after load.
@group(0) @binding(12) var<storage, read> morphDeltas   : array<f32>;
// Every morphed instance's weights, rebuilt each frame.
@group(0) @binding(13) var<storage, read> morphWeights  : array<f32>;
// Every directional light.
@group(0) @binding(14) var<storage, read> directionals  : array<Directional>;
// Their cascade matrices: slot * cascades + cascade, the same index as the
// layer of shadowMap they were drawn into.
@group(0) @binding(17) var<storage, read> cascadeViews  : array<mat4x4<f32>>;
// The opaque scene, mipped, for transmissive surfaces (render/transmission.js).
@group(0) @binding(18) var         behindMap : texture_2d<f32>;
// Reflection probes' prefiltered cubes, and their boxes, smallest first.
@group(0) @binding(19) var         probeMaps : texture_cube_array<f32>;
@group(0) @binding(20) var<storage, read> probes : array<Probe>;
// Every decal texture, a layer each, and the decals, in the order added.
@group(0) @binding(21) var         decalMaps : texture_2d_array<f32>;
@group(0) @binding(22) var<storage, read> decals : array<Decal>;

@group(2) @binding(0) var<uniform> material     : Material;
@group(2) @binding(1) var          baseColorMap : texture_2d<f32>;
@group(2) @binding(2) var          normalMap    : texture_2d<f32>;
@group(2) @binding(3) var          mrMap        : texture_2d<f32>;
@group(2) @binding(4) var          occlusionMap : texture_2d<f32>;
@group(2) @binding(5) var          emissiveMap  : texture_2d<f32>;
@group(2) @binding(6) var          surfSampler  : sampler;
${Array.from({ length: extensionSlots }, (_, i) => `@group(2) @binding(${EXTENSION_BINDING + i}) var extensionMap${i} : texture_2d<f32>;`).join('\n')}

@group(3) @binding(0) var<uniform> batch : Batch;

struct VertexOut {
  @builtin(position) clip     : vec4<f32>,
  @location(0)       world    : vec3<f32>,
  @location(1)       normal   : vec3<f32>,
  @location(2)       tangent  : vec3<f32>,
  @location(3)       bitangent: vec3<f32>,
  @location(4)       uv       : vec2<f32>,
  @location(5)       uv1      : vec2<f32>,
  @location(6)       color    : vec4<f32>,
  // How much the model matrix stretches each axis, for KHR_materials_volume,
  // whose thickness is in the mesh's own units.
  @location(7) @interpolate(flat) modelScale : vec3<f32>,
};


/**
 * The morphed vertex: the authored one plus every target's delta, weighted.
 *
 * Applied BEFORE skinning, which is the order glTF specifies and the only one
 * that makes sense -- a target is authored against the bind pose, so it has to
 * move the vertex while the vertex is still in it.
 *
 * Zero targets costs one comparison, which is what lets every static mesh in
 * the engine share this vertex shader instead of a variant of it.
 *
 * Per PASS, not per frame: a morphed vertex is deformed again for the early
 * draw, the late draw and each shadow cascade. The ceiling is a rig with many
 * targets active at once, and the answer if it is ever reached is to compact
 * the nonzero weights on the CPU -- a change to what the weight buffer holds,
 * not to any of this.
 */
struct Morphed {
  position : vec3<f32>,
  normal   : vec3<f32>,
  tangent  : vec3<f32>,
};

fn applyMorph(
  draw : DrawData, vertex : u32,
  position : vec3<f32>, normal : vec3<f32>, tangent : vec3<f32>,
) -> Morphed {
  var out : Morphed;
  out.position = position;
  out.normal = normal;
  out.tangent = tangent;

  let count = draw.morphCount & 0xffffu;
  if (count == 0u) { return out; }

  let stride = draw.morphCount >> 16u;
  // Vertex-major: every target's delta for this vertex sits together, so the
  // loop below walks forward through memory instead of across the array.
  var o = draw.morphBase + vertex * count * stride;

  for (var t = 0u; t < count; t = t + 1u) {
    let w = morphWeights[draw.morphWeights + t];
    // A weight of zero is the resting state of most targets of most rigs, and
    // skipping it skips the reads, which are what the loop actually costs.
    if (w != 0.0) {
      out.position = out.position
        + w * vec3<f32>(morphDeltas[o], morphDeltas[o + 1u], morphDeltas[o + 2u]);
      // A stride wider than 3 means every target of this primitive carries
      // normals; wider than 6, tangents. Decided once at import, so this is
      // uniform across the draw rather than a branch that diverges.
      if (stride > 3u) {
        out.normal = out.normal
          + w * vec3<f32>(morphDeltas[o + 3u], morphDeltas[o + 4u], morphDeltas[o + 5u]);
      }
      if (stride > 6u) {
        out.tangent = out.tangent
          + w * vec3<f32>(morphDeltas[o + 6u], morphDeltas[o + 7u], morphDeltas[o + 8u]);
      }
    }
    o = o + stride;
  }
  return out;
}

@vertex
fn vs(
  @builtin(instance_index) instance : u32,
  @builtin(vertex_index)   vertex   : u32,
  @location(0) position : vec3<f32>,
  @location(1) normal   : vec3<f32>,
  @location(2) uv       : vec2<f32>,
  @location(3) tangent  : vec4<f32>,
  @location(4) uv1      : vec2<f32>,
  @location(5) color    : vec4<f32>,
) -> VertexOut {
  var out : VertexOut;

  // For the batched draws: instance_index counts from 0 within the draw, and
  // batch.firstVisible turns it into an index into the frame-wide visible list.
  // WebGPU only allows a non-zero firstInstance behind an optional feature,
  // which is why the base arrives in a uniform rather than in the arguments.
  //
  // The blended draws use this same shader and do the opposite: they are DIRECT
  // draws, which may set firstInstance freely, so they pass the absolute slot
  // there and bind a firstVisible of 0. Both end up indexing the same list.
  let draw = drawData[visibleItems[batch.firstVisible + instance]];
  let m = applyMorph(draw, vertex, position, normal, tangent.xyz);

  let world = draw.model * vec4<f32>(m.position, 1.0);
  out.world = world.xyz;
  out.clip = frame.viewProjection * world;
  out.modelScale = vec3<f32>(length(draw.model[0].xyz), length(draw.model[1].xyz), length(draw.model[2].xyz));

  // Normals use the inverse-transpose; tangents do NOT. A tangent is a
  // direction along the surface, so it transforms like a position delta and
  // the model matrix is correct for it.
  out.normal = normalize(draw.normalMatrix * m.normal);
  out.tangent = normalize((draw.model * vec4<f32>(m.tangent, 0.0)).xyz);

  // tangent.w is the handedness the importer computed per vertex, which is
  // what keeps mirrored UV islands from lighting inside out.
  out.bitangent = cross(out.normal, out.tangent) * tangent.w;
  out.uv = uv;
  out.uv1 = uv1;
  out.color = color;
  return out;
}

/**
 * An extension texture, by kind. Its rows in uvTransforms name its UV set, its
 * transform and the binding it shares with any kind reading the same image. A
 * kind with no texture reads 1 on every channel, as glTF says an absent one
 * does -- so the factor alone stands.
 *
 * A switch because WebGPU cannot index an array of textures. The binding is
 * uniform across the draw, so the branch is too.
 */
fn sampleExtension(kind : u32, uv0 : vec2<f32>, uv1 : vec2<f32>) -> vec4<f32> {
  let row = (${CORE_TEXTURE_COUNT}u + kind) * 2u;
  let slot = i32(material.uvTransforms[row + 1u].w);
  let p = vec3<f32>(select(uv0, uv1, material.uvTransforms[row].w > 0.5), 1.0);
  let uv = vec2<f32>(dot(material.uvTransforms[row].xyz, p), dot(material.uvTransforms[row + 1u].xyz, p));
  switch slot {
${Array.from({ length: extensionSlots }, (_, i) => `    case ${i}: { return textureSample(extensionMap${i}, surfSampler, uv); }`).join('\n')}
    default: { return vec4<f32>(1.0); }
  }
}

/** Whether a kind has a texture: its binding, -1 for none. */
fn extensionBound(kind : u32) -> bool {
  return material.uvTransforms[(${CORE_TEXTURE_COUNT}u + kind) * 2u + 1u].w >= 0.0;
}

/**
 * A tangent-space normal into world space, around the geometric normal.
 *
 * The bitangent flips with the normal on a back face, to keep the basis
 * right-handed. The tangent does not: it follows the UV's u axis, which does
 * not reverse.
 *
 * Every normalize here guards its length first. An interpolated tangent can
 * reach zero -- a file's zero TANGENT, or opposite tangents meeting inside
 * one triangle -- and so can a flat normal-map texel, and normalize() of
 * zero is NaN. D3D's min() happens to swallow one; Vulkan and Metal hand it
 * to bloom, which spreads it across the frame. Below the smallest normal
 * f32 there is no direction left to recover, so such a fragment keeps its
 * geometric normal.
 */
fn mapNormal(v : VertexOut, geometric : vec3<f32>, facing : f32, tangentNormal : vec3<f32>) -> vec3<f32> {
  let tangentLength = dot(v.tangent, v.tangent);
  let bitangentLength = dot(v.bitangent, v.bitangent);
  if (tangentLength <= F32_MIN_NORMAL || bitangentLength <= F32_MIN_NORMAL) { return geometric; }
  let tbn = mat3x3<f32>(
    v.tangent * inverseSqrt(tangentLength),
    v.bitangent * inverseSqrt(bitangentLength) * facing,
    geometric,
  );
  let mapped = tbn * tangentNormal;
  let mappedLength = dot(mapped, mapped);
  if (mappedLength <= F32_MIN_NORMAL) { return geometric; }
  return mapped * inverseSqrt(mappedLength);
}

/**
 * KHR_materials_sheen's directional albedo E(NoV, roughness), bilinear in the
 * table: NoV at bin centres, roughness from 0 to 1.
 */
fn sheenAlbedo(NoV : f32, roughness : f32) -> f32 {
  let last = ${SHEEN_TABLE_SIZE - 1}.0;
  let x = clamp(NoV * ${SHEEN_TABLE_SIZE}.0 - 0.5, 0.0, last);
  let y = clamp(roughness, 0.0, 1.0) * last;
  let x0 = u32(x);
  let y0 = u32(y);
  let x1 = min(x0 + 1u, ${SHEEN_TABLE_SIZE - 1}u);
  let y1 = min(y0 + 1u, ${SHEEN_TABLE_SIZE - 1}u);
  let fx = x - f32(x0);
  return mix(
    mix(sheenEntry(x0, y0), sheenEntry(x1, y0), fx),
    mix(sheenEntry(x0, y1), sheenEntry(x1, y1), fx),
    y - f32(y0),
  );
}

fn sheenEntry(x : u32, y : u32) -> f32 {
  let i = y * ${SHEEN_TABLE_SIZE}u + x;
  return frame.sheenAlbedo[i / 4u][i % 4u];
}

/** The Charlie sheen distribution, as KHR_materials_sheen gives it. */
fn charlieDistribution(NoH : f32, roughness : f32) -> f32 {
  let inverse = 1.0 / max(roughness * roughness, 1e-7);
  let sin2h = max(1.0 - NoH * NoH, 0.0);
  return (2.0 + inverse) * pow(sin2h, inverse * 0.5) / (2.0 * PI);
}

/** The spec's fit for the Charlie visibility's lambda. */
fn charlieL(x : f32, alphaG : f32) -> f32 {
  let t = (1.0 - alphaG) * (1.0 - alphaG);
  return mix(21.5473, 25.3245, t) / (1.0 + mix(3.82987, 3.32435, t) * pow(x, mix(0.19823, 0.16801, t)))
    + mix(-1.97760, -1.27393, t) * x + mix(-4.32054, -4.85967, t);
}

fn charlieLambda(cosTheta : f32, alphaG : f32) -> f32 {
  if (cosTheta < 0.5) { return exp(charlieL(cosTheta, alphaG)); }
  return exp(2.0 * charlieL(0.5, alphaG) - charlieL(1.0 - cosTheta, alphaG));
}

/**
 * The Charlie visibility -- the spec's, not its cheaper Ashikhmin option,
 * which reflects up to half again more light than arrives at grazing angles.
 */
fn charlieVisibility(NoV : f32, NoL : f32, roughness : f32) -> f32 {
  let alphaG = max(roughness * roughness, 1e-7);
  return 1.0 / ((1.0 + charlieLambda(NoV, alphaG) + charlieLambda(NoL, alphaG)) * (4.0 * NoV * NoL));
}

/**
 * GGX stretched along a tangent: alpha t along it, alpha b across. Written
 * as the reference viewer writes it, which stays finite as either alpha
 * shrinks. With the two equal it is distributionGGX exactly.
 */
fn distributionGGXAnisotropic(NoH : f32, ToH : f32, BoH : f32, at : f32, ab : f32) -> f32 {
  let a2 = at * ab;
  let f = vec3<f32>(ab * ToH, at * BoH, a2 * NoH);
  let w2 = a2 / max(dot(f, f), 1e-30);
  return a2 * w2 * w2 / PI;
}

/** Height-correlated Smith visibility for the stretched lobe; holds 1/(4 NoL NoV). */
fn visibilitySmithGGXAnisotropic(
  NoL : f32, NoV : f32, BoV : f32, ToV : f32, ToL : f32, BoL : f32, at : f32, ab : f32,
) -> f32 {
  let lambdaV = NoL * length(vec3<f32>(at * ToV, ab * BoV, NoV));
  let lambdaL = NoV * length(vec3<f32>(at * ToL, ab * BoL, NoL));
  return clamp(0.5 / max(lambdaV + lambdaL, 1e-5), 0.0, 1.0);
}

// ---- KHR_materials_iridescence: Belcour and Barla's thin film, in the
// form the spec gives it.

const XYZ_TO_REC709 = mat3x3<f32>(
   3.2404542, -0.9692660,  0.0556434,
  -1.5371385,  1.8760108, -0.2040259,
  -0.4985314,  0.0415560,  1.0572252,
);

/** The spec's Gaussian fit of the eye's response, in Fourier space. */
fn evalSensitivity(OPD : f32, shift : vec3<f32>) -> vec3<f32> {
  let phase = 2.0 * PI * OPD * 1.0e-9;
  let val = vec3<f32>(5.4856e-13, 4.4201e-13, 5.2481e-13);
  let pos = vec3<f32>(1.6810e+06, 1.7953e+06, 2.2084e+06);
  let spread = vec3<f32>(4.3278e+09, 9.3046e+09, 6.6121e+09);
  var xyz = val * sqrt(2.0 * PI * spread) * cos(pos * phase + shift) * exp(-phase * phase * spread);
  xyz.x = xyz.x + 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x)
    * exp(-4.5282e+09 * phase * phase);
  return XYZ_TO_REC709 * (xyz / 1.0685e-7);
}

fn iorToFresnel0(transmitted : vec3<f32>, incident : f32) -> vec3<f32> {
  let r = (transmitted - vec3<f32>(incident)) / (transmitted + vec3<f32>(incident));
  return r * r;
}

/**
 * The film's Fresnel at a view angle, over a base whose head-on reflectance
 * is baseF0, from air.
 *
 * As the film thins to nothing its index is eased to the outside's over the
 * first 30 nm, as the reference viewer does, so a film of 0 nm leaves the
 * base's own Fresnel rather than a colour the formula has no business making.
 */
fn iridescentFresnel(filmIorIn : f32, cosTheta1 : f32, thickness : f32, baseF0 : vec3<f32>) -> vec3<f32> {
  let filmIor = mix(1.0, filmIorIn, smoothstep(0.0, 30.0, thickness));
  let sinTheta2Sq = (1.0 / (filmIor * filmIor)) * (1.0 - cosTheta1 * cosTheta1);
  let cosTheta2Sq = 1.0 - sinTheta2Sq;
  if (cosTheta2Sq < 0.0) { return vec3<f32>(1.0); }   // total internal reflection
  let cosTheta2 = sqrt(cosTheta2Sq);

  // First interface, air to film.
  let R0 = iorToFresnel0(vec3<f32>(filmIor), 1.0).x;
  let R12 = R0 + (1.0 - R0) * pow(clamp(1.0 - cosTheta1, 0.0, 1.0), 5.0);
  let T121 = 1.0 - R12;
  let phi12 = select(0.0, PI, filmIor < 1.0);
  let phi21 = PI - phi12;

  // Second interface, film to base. The base's index from its F0, clamped
  // short of 1, where it would be infinite.
  let sqrtF0 = sqrt(clamp(baseF0, vec3<f32>(0.0), vec3<f32>(0.9999)));
  let baseIor = (vec3<f32>(1.0) + sqrtF0) / (vec3<f32>(1.0) - sqrtF0);
  let R1 = iorToFresnel0(baseIor, filmIor);
  let R23 = R1 + (vec3<f32>(1.0) - R1) * pow(1.0 - cosTheta2, 5.0);
  let phi23 = select(vec3<f32>(0.0), vec3<f32>(PI), baseIor < vec3<f32>(filmIor));

  let OPD = 2.0 * filmIor * thickness * cosTheta2;
  let phi = vec3<f32>(phi21) + phi23;

  let R123 = clamp(R12 * R23, vec3<f32>(1e-5), vec3<f32>(0.9999));
  let r123 = sqrt(R123);
  let Rs = T121 * T121 * R23 / (vec3<f32>(1.0) - R123);
  var I = R12 + Rs;
  var Cm = Rs - T121;
  for (var m = 1; m <= 2; m = m + 1) {
    Cm = Cm * r123;
    I = I + Cm * 2.0 * evalSensitivity(f32(m) * OPD, f32(m) * phi);
  }
  return max(I, vec3<f32>(0.0));
}

/**
 * The environment's radiance along a direction at a prefilter level, as seen
 * from a point: the reflection probes whose boxes hold it, then the sky for
 * whatever they leave.
 *
 * Each probe is read box-projected -- the ray from the point runs to the
 * box's wall, and the probe is sampled toward that hit from where it
 * captured -- so what it shows lines up with the room rather than sitting at
 * infinity. Probes come smallest box first, and each takes its weight of what
 * the ones before it left: nested boxes resolve to the tighter, and a probe
 * fading at its edge hands the rest on.
 */
fn environmentRadiance(position : vec3<f32>, direction : vec3<f32>, lod : f32) -> vec3<f32> {
  var radiance = vec3<f32>(0.0);
  var left = 1.0;
  let count = select(0u, u32(frame.probeInfo.x), PROBES);
  for (var i = 0u; i < count && left > 0.0; i = i + 1u) {
    let probe = probes[i];
    let inside = min(position - probe.boxMin.xyz, probe.boxMax.xyz - position);
    let depth = min(min(inside.x, inside.y), inside.z);
    if (depth < 0.0) { continue; }
    let weight = select(1.0, clamp(depth / probe.boxMin.w, 0.0, 1.0), probe.boxMin.w > 0.0);
    // Where the ray leaves the box: the nearest of the far planes it heads for.
    let safe = select(direction, vec3<f32>(1e-6), abs(direction) < vec3<f32>(1e-6));
    let far = max((probe.boxMax.xyz - position) / safe, (probe.boxMin.xyz - position) / safe);
    let hit = position + direction * min(min(far.x, far.y), far.z);
    let sample = textureSampleLevel(probeMaps, envSampler, hit - probe.position.xyz, i32(probe.boxMax.w), lod).rgb;
    radiance = radiance + sample * weight * left;
    left = left * (1.0 - weight);
  }
  if (left > 0.0) {
    radiance = radiance + textureSampleLevel(prefiltered, envSampler, direction, lod).rgb * left;
  }
  return radiance;
}

/**
 * The base colour with every decal whose box holds this point painted over
 * it, in the order they were added. A surface facing away from a decal is
 * not painted by it: nothing projected along its -Z can reach one.
 *
 * Sampled with gradients carried from the world position's, taken before
 * the loop -- a derivative inside it would be in control flow that is not
 * uniform -- so a decal filters as a texture on the surface would.
 */
fn applyDecals(albedo : vec3<f32>, position : vec3<f32>, normal : vec3<f32>, dx : vec3<f32>, dy : vec3<f32>, cluster : u32) -> vec3<f32> {
  var colour = albedo;
  // Only the decals the cluster pass put in this fragment's cell. They arrive
  // there in whatever order the GPU ran them, and a later decal paints over an
  // earlier one, so the list is walked smallest index first: n passes, each
  // taking the least index above the last. The lists are short.
  let count = min(clusterCounts[${CLUSTER_COUNT}u + cluster], ${MAX_LIGHTS_PER_CLUSTER}u);
  let base = ${DECAL_INDEX_BASE}u + cluster * ${MAX_LIGHTS_PER_CLUSTER}u;
  var previous = -1;
  for (var n = 0u; n < count; n = n + 1u) {
    var next = 0x7fffffff;
    for (var k = 0u; k < count; k = k + 1u) {
      let candidate = i32(clusterIndices[base + k]);
      if (candidate > previous && candidate < next) { next = candidate; }
    }
    previous = next;
    let i = u32(next);
    let decal = decals[i];
    let p = (decal.worldToDecal * vec4<f32>(position, 1.0)).xyz;
    if (any(abs(p) > vec3<f32>(1.0)) || dot(normal, decal.facing.xyz) <= 0.0) { continue; }
    let flip = vec2<f32>(0.5, -0.5);
    let uv = p.xy * flip + vec2<f32>(0.5);
    let gx = (decal.worldToDecal * vec4<f32>(dx, 0.0)).xy * flip;
    let gy = (decal.worldToDecal * vec4<f32>(dy, 0.0)).xy * flip;
    let texel = textureSampleGrad(decalMaps, surfSampler, uv, i32(decal.facing.w), gx, gy) * decal.color;
    colour = mix(colour, texel.rgb, clamp(texel.a, 0.0, 1.0));
  }
  return colour;
}

fn maxChannel(c : vec3<f32>) -> f32 {
  return max(c.r, max(c.g, c.b));
}

/**
 * A fragment's surface, gathered once and then lit by every light and the
 * environment alike.
 */
struct Surface {
  position       : vec3<f32>,
  n              : vec3<f32>,
  view           : vec3<f32>,
  NoV            : f32,
  roughness      : f32,
  // Base colour less the metal: a metal has no diffuse term at all.
  diffuse        : vec3<f32>,
  // Specular reflectance head-on and at grazing, dielectric and metal mixed.
  f0             : vec3<f32>,
  f90            : f32,
  // The dielectric layer's own head-on reflectance and weight, which is what
  // decides how much light reaches the diffuse beneath it.
  dielectricF0   : vec3<f32>,
  specularWeight : f32,
  // KHR_materials_sheen: black for none. Scaling is what the layer leaves the
  // surface beneath it, 1 - max(sheenColor) * E(NoV).
  sheenColor     : vec3<f32>,
  sheenRoughness : f32,
  sheenScaling   : f32,
  // KHR_materials_clearcoat: 0 for none. The coat's own normal and its
  // Fresnel at the view, which is also what it takes from everything beneath.
  coat           : f32,
  coatRoughness  : f32,
  coatN          : vec3<f32>,
  coatNoV        : f32,
  coatFresnel    : f32,
  // KHR_materials_anisotropy: 0 for none. The stretched direction and the one
  // across it, and the roughness along it.
  anisotropy     : f32,
  anisotropicT   : vec3<f32>,
  anisotropicB   : vec3<f32>,
  alphaT         : f32,
  // KHR_materials_iridescence: 0 for none. The film's Fresnel at the view,
  // over the dielectric and over the dielectric and metal as mixed.
  iridescence    : f32,
  iridescenceDielectric : vec3<f32>,
  iridescenceF   : vec3<f32>,
  // KHR_materials_transmission and _volume: 0 for none. The roughness the
  // light passing through is blurred by, what the volume lets through along
  // the path, and the scene behind as seen along it.
  transmission   : f32,
  transmissionRoughness : f32,
  transmittance  : vec3<f32>,
  behind         : vec3<f32>,
};

/**
 * What one light gives a surface, radiance already attenuated and shadowed.
 *
 * KHR_materials_specular's layering: the dielectric reflects weight * F and
 * passes the diffuse beneath 1 - weight * max(F) -- by its largest channel,
 * so a tinted specular cannot light one channel of the diffuse past 1. The
 * metal reflects its own colour. With weight 1 and f0 grey this is the plain
 * metallic workflow.
 */
fn surfaceLight(s : Surface, l : vec3<f32>, NoL : f32, radiance : vec3<f32>) -> vec3<f32> {
  let h = normalize(s.view + l);
  let NoH = max(dot(s.n, h), 0.0);
  let x = pow(clamp(1.0 - dot(s.view, h), 0.0, 1.0), 5.0);
  // An iridescent film swaps its own Fresnel in, as far as its strength.
  // Every extension term below is behind EXTENSIONS, like the gathering that
  // sets it: a plain pipeline must not carry what it can never use.
  var F = s.f0 + (vec3<f32>(s.f90) - s.f0) * x;
  if (EXTENSIONS) { F = mix(F, s.iridescenceF, s.iridescence); }
  // The visibility term already holds the 1/(4 NoL NoV) denominator.
  var DV : f32;
  if (EXTENSIONS && s.anisotropy > 0.0) {
    let ab = s.roughness * s.roughness;
    DV = distributionGGXAnisotropic(NoH, dot(s.anisotropicT, h), dot(s.anisotropicB, h), s.alphaT, ab)
      * visibilitySmithGGXAnisotropic(NoL, s.NoV, dot(s.anisotropicB, s.view), dot(s.anisotropicT, s.view),
          dot(s.anisotropicT, l), dot(s.anisotropicB, l), s.alphaT, ab);
  } else {
    DV = distributionGGX(NoH, s.roughness) * visibilitySmithGGX(s.NoV, NoL, s.roughness);
  }
  let specular = DV * F;
  var reflectance = s.specularWeight * (s.dielectricF0 + (1.0 - s.dielectricF0) * x);
  // What transmits passes through instead of scattering back: the diffuse
  // term gives up that share.
  var diffuse = s.diffuse;
  if (EXTENSIONS) {
    reflectance = mix(reflectance, s.iridescenceDielectric, s.iridescence);
    diffuse = diffuse * (1.0 - s.transmission);
  }
  var colour = ((1.0 - maxChannel(reflectance)) * diffuse / PI + specular) * NoL;

  // Sheen over the base, which keeps what the sheen does not take.
  if (EXTENSIONS && maxChannel(s.sheenColor) > 0.0) {
    colour = s.sheenColor * charlieDistribution(NoH, s.sheenRoughness)
      * charlieVisibility(s.NoV, NoL, s.sheenRoughness) * NoL
      + colour * s.sheenScaling;
  }

  // The coat over both, by its own normal. Its Fresnel is the spec's, taken at
  // the view, so what it reflects is exactly what everything beneath loses.
  if (EXTENSIONS && s.coat > 0.0) {
    let coatNoL = clamp(dot(s.coatN, l), 0.0, 1.0);
    let coatSpecular = distributionGGX(max(dot(s.coatN, h), 0.0), s.coatRoughness)
      * visibilitySmithGGX(s.coatNoV, coatNoL, s.coatRoughness) * coatNoL;
    colour = colour * (1.0 - s.coat * s.coatFresnel) + vec3<f32>(s.coat * s.coatFresnel * coatSpecular);
  }
  return colour * radiance;
}

/**
 * What a light behind a transmissive surface sends through it to the eye.
 *
 * A thin wall's microfacets pass light as they would reflect its mirror image
 * through the surface, so the lobe is the specular one about that mirror --
 * at the transmission roughness, tinted by the base colour, less what the
 * dielectric reflects, and less what a volume absorbs on the way.
 */
fn surfaceTransmission(s : Surface, l : vec3<f32>, radiance : vec3<f32>) -> vec3<f32> {
  let mirrored = l - 2.0 * dot(l, s.n) * s.n;
  let NoL = max(dot(s.n, mirrored), 0.0);
  let h = normalize(mirrored + s.view);
  let x = pow(clamp(1.0 - dot(s.view, h), 0.0, 1.0), 5.0);
  let dielectric = maxChannel(mix(
    s.specularWeight * (s.dielectricF0 + (1.0 - s.dielectricF0) * x), s.iridescenceDielectric, s.iridescence));
  // At the base's own floor: a perfectly smooth lobe is a delta no light can hit.
  let r = max(s.transmissionRoughness, 0.045);
  let btdf = distributionGGX(max(dot(s.n, h), 0.0), r) * visibilitySmithGGX(s.NoV, NoL, r);
  var colour = s.transmission * (1.0 - dielectric) * s.diffuse * s.transmittance * btdf * NoL;
  if (maxChannel(s.sheenColor) > 0.0) { colour = colour * s.sheenScaling; }
  if (s.coat > 0.0) { colour = colour * (1.0 - s.coat * s.coatFresnel); }
  return colour * radiance;
}

/**
 * The environment's light on a surface: the prebaked irradiance and the
 * prefiltered reflection, split the same way as surfaceLight, with the
 * split-sum's scale and bias standing in for Fresnel.
 */
fn surfaceAmbient(s : Surface) -> vec3<f32> {
  let irradianceSample = textureSample(irradiance, envSampler, s.n).rgb;
  // An anisotropic surface reflects along a normal bent toward the one the
  // stretched lobe implies -- the reference viewer's stand-in for sampling
  // the lobe, bent further the stronger and the smoother it is.
  var reflectN = s.n;
  if (EXTENSIONS && s.anisotropy > 0.0) {
    let across = cross(s.anisotropicB, s.view);
    let bend = 1.0 - s.anisotropy * (1.0 - s.roughness);
    let bent = mix(cross(across, s.anisotropicB), s.n, bend * bend * bend * bend);
    let bentLength = dot(bent, bent);
    if (bentLength > F32_MIN_NORMAL) { reflectN = bent * inverseSqrt(bentLength); }
  }
  let reflected = reflect(-s.view, reflectN);
  let maxMip = max(frame.environment.x - 1.0, 0.0);
  // Perceptual roughness indexes the mip chain directly, because that is how
  // ibl.js spaced the levels when it baked them.
  let prefilteredSample = environmentRadiance(s.position, reflected, s.roughness * maxMip);
  let ab = envBRDFApprox(s.NoV, s.roughness);
  // A film's Fresnel is already the view's, so it takes the split sum's
  // shadowing, ab.x + ab.y, and nothing else.
  var fresnel = s.f0 * ab.x + vec3<f32>(s.f90 * ab.y);
  var reflectance = s.specularWeight * (s.dielectricF0 * ab.x + vec3<f32>(ab.y));
  var diffuseLight = irradianceSample;
  if (EXTENSIONS) {
    fresnel = mix(fresnel, s.iridescenceF * (ab.x + ab.y), s.iridescence);
    reflectance = mix(reflectance, s.iridescenceDielectric * (ab.x + ab.y), s.iridescence);
    // A transmissive surface's diffuse share is partly the scene behind it.
    diffuseLight = mix(diffuseLight, s.behind, s.transmission);
  }
  let specular = prefilteredSample * fresnel;
  var colour = (1.0 - maxChannel(reflectance)) * diffuseLight * s.diffuse + specular;

  // The sheen reflects E of the environment, read at its own roughness from
  // the GGX-prefiltered cube -- the stand-in Filament uses, since a Charlie-
  // prefiltered cube would be a second bake for one extension.
  if (EXTENSIONS && maxChannel(s.sheenColor) > 0.0) {
    let sheenLight = environmentRadiance(s.position, reflected, s.sheenRoughness * maxMip);
    colour = s.sheenColor * sheenLight * sheenAlbedo(s.NoV, s.sheenRoughness) + colour * s.sheenScaling;
  }

  // The coat: the split sum without its Fresnel, which the layering applies.
  if (EXTENSIONS && s.coat > 0.0) {
    let coatAB = envBRDFApprox(s.coatNoV, s.coatRoughness);
    let coatLight = environmentRadiance(s.position, reflect(-s.view, s.coatN), s.coatRoughness * maxMip);
    colour = colour * (1.0 - s.coat * s.coatFresnel)
      + s.coat * s.coatFresnel * coatLight * (coatAB.x + coatAB.y);
  }
  return colour;
}

/** A texture slot's UV, through its KHR_texture_transform. */
fn transformUV(slot : u32, uv : vec2<f32>) -> vec2<f32> {
  let p = vec3<f32>(uv, 1.0);
  return vec2<f32>(dot(material.uvTransforms[slot * 2u].xyz, p), dot(material.uvTransforms[slot * 2u + 1u].xyz, p));
}

/**
 * A normal map's tangent-space xy, from the transformed texture's axes back
 * to the mesh's. Its UVs are A uv + t, so the texture's own u and v axes lie
 * along the columns of A^-1 in the mesh's UV space -- a rotated normal map
 * encodes rotated slopes, and without this a bump lights from the wrong side.
 * The identity leaves it exactly as sampled.
 */
fn untransformNormal(row : u32, xy : vec2<f32>) -> vec2<f32> {
  let r0 = material.uvTransforms[row].xy;
  let r1 = material.uvTransforms[row + 1u].xy;
  let det = r0.x * r1.y - r0.y * r1.x;
  if (abs(det) < F32_MIN_NORMAL) { return xy; }
  return vec2<f32>(r1.y * xy.x - r0.y * xy.y, -r1.x * xy.x + r0.x * xy.y) / det;
}

/**
 * Which cascade covers this distance. -1 means beyond the shadow distance,
 * where everything is simply lit.
 *
 * Written as an if-chain rather than a loop indexing cascadeSplits[i]: dynamic
 * indexing into a vec4 is legal WGSL but compiles to a scratch-memory round
 * trip on some drivers, and there are only ever four.
 */
fn selectCascade(viewDepth : f32) -> i32 {
  if (viewDepth < frame.cascadeSplits.x) { return 0; }
  if (viewDepth < frame.cascadeSplits.y) { return 1; }
  if (viewDepth < frame.cascadeSplits.z) { return 2; }
  if (viewDepth < frame.cascadeSplits.w) { return 3; }
  return -1;
}

fn cascadeTexelSize(cascade : i32) -> f32 {
  if (cascade == 0) { return frame.cascadeTexel.x; }
  if (cascade == 1) { return frame.cascadeTexel.y; }
  if (cascade == 2) { return frame.cascadeTexel.z; }
  return frame.cascadeTexel.w;
}

fn cascadeSplit(cascade : i32) -> f32 {
  if (cascade == 0) { return frame.cascadeSplits.x; }
  if (cascade == 1) { return frame.cascadeSplits.y; }
  if (cascade == 2) { return frame.cascadeSplits.z; }
  return frame.cascadeSplits.w;
}

/**
 * How far apart this cascade's PCF taps are, in its own texels.
 *
 * What makes a seam at a split is that two things jump there: the blur, three
 * texels wide, and the normal offset, a texel and a half -- and the next
 * cascade's texel is about twice the size. So the spacing grows across the
 * slice, from 1 where it begins to the ratio of the two texel sizes at the
 * split: the blur and the offset arrive at the split already the next
 * cascade's, and nothing jumps. No band width to choose, and still nine taps.
 * The price is that each cascade softens toward its far end instead of
 * holding its sharpness and then stepping.
 *
 * Each tap is a bilinear 2x2 comparison, so taps up to two texels apart leave
 * no gaps. The ratio is usually about two but not always (2.9 into a long last
 * slice); past two, the gaps fall where the spacing is widest -- the far end of
 * a slice, where the whole penumbra is two or three pixels on screen and
 * anything inside it is below one.
 */
fn tapSpacing(cascade : i32, viewDepth : f32) -> f32 {
  if (cascade + 1 >= i32(frame.cascadeInfo.x)) { return 1.0; }
  let start = select(cascadeSplit(cascade - 1), frame.shadowParams.z, cascade == 0);
  let end = cascadeSplit(cascade);
  let along = clamp((viewDepth - start) / max(end - start, 1e-6), 0.0, 1.0);
  let ratio = cascadeTexelSize(cascade + 1) / cascadeTexelSize(cascade);
  return mix(1.0, ratio, along);
}

/**
 * How much of a casting directional light reaches this point, through its
 * cascades. 1 is fully lit.
 *
 * The bias here is a NORMAL OFFSET rather than a depth offset: the lookup
 * position is pushed along the surface normal by roughly one shadow texel
 * before being projected. A depth bias moves the comparison and detaches the
 * shadow from the object's feet (peter-panning); moving the sample point
 * sideways along the surface fixes acne without that, because a surface's own
 * texel genuinely is what the offset lands in.
 *
 * Surfaces facing away from the light are skipped entirely -- they are already
 * dark from N.L, and their shadow lookups are the noisiest ones there are.
 */
fn directionalVisibility(slot : i32, worldPosition : vec3<f32>, normal : vec3<f32>, NoL : f32, viewDepth : f32) -> f32 {
  if (NoL <= 0.0) { return 1.0; }

  let cascade = selectCascade(viewDepth);
  if (cascade < 0) { return 1.0; }
  let layer = slot * i32(frame.cascadeInfo.x) + cascade;

  // Scale the offset with how glancing the light is: a surface almost edge-on
  // needs far more, which is the same reason the map itself uses slope-scaled
  // hardware bias.
  let slope = clamp(1.0 - NoL, 0.0, 1.0);
  let spacing = tapSpacing(cascade, viewDepth);
  let offset = cascadeTexelSize(cascade) * spacing * frame.shadowParams.x * (1.0 + slope * 2.0);
  let biased = worldPosition + normal * offset;

  let lightClip = cascadeViews[layer] * vec4<f32>(biased, 1.0);
  let ndc = lightClip.xyz / lightClip.w;

  // NDC y is up, texture v is down.
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

  // Outside its own cascade the clamped sampler would smear the edge texel
  // across the whole world. Treat it as lit; the next cascade covers it.
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z <= 0.0) {
    return 1.0;
  }

  // 3x3 PCF. Nine taps of hardware-filtered comparisons, which is a soft edge
  // about three texels wide -- enough to hide the staircase without the cost
  // of a real soft-shadow kernel.
  let texel = spacing / frame.shadowParams.y;
  var total = 0.0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      total = total + textureSampleCompareLevel(
        shadowMap, shadowSampler,
        uv + vec2<f32>(f32(x), f32(y)) * texel,
        layer, ndc.z,
      );
    }
  }
  return total / 9.0;
}

/**
 * How much of a point or spot light reaches this point, through its own
 * shadow maps. 1 is fully lit, and so is a light that casts none.
 *
 * A cube's face is the major axis of the direction from the light, in the
 * order shadows.js lays them out: +x -x +y -y +z -z. The rest is the sun's
 * lookup: a normal offset of about a texel -- a texel at this distance, since
 * a perspective view's texels grow with it -- then 3x3 PCF.
 */
fn localVisibility(light : Light, worldPosition : vec3<f32>, normal : vec3<f32>, NoL : f32) -> f32 {
  let first = i32(light.coneFalloff.w) - 1;
  if (first < 0) { return 1.0; }

  let fromLight = worldPosition - light.positionRadius.xyz;
  var layer = first;
  var along = 0.0;
  if (light.directionCone.w > 1.5) {
    let a = abs(fromLight);
    if (a.x >= a.y && a.x >= a.z) {
      layer = first + select(1, 0, fromLight.x > 0.0);
      along = a.x;
    } else if (a.y >= a.z) {
      layer = first + select(3, 2, fromLight.y > 0.0);
      along = a.y;
    } else {
      layer = first + select(5, 4, fromLight.z > 0.0);
      along = a.z;
    }
  } else {
    along = dot(fromLight, light.directionCone.xyz);
  }

  let view = localViews[layer];
  let size = frame.shadowParams.w;
  let slope = clamp(1.0 - NoL, 0.0, 1.0);
  let texelWorld = 2.0 * max(along, 0.0) * view.params.x / size;
  let biased = worldPosition + normal * texelWorld * frame.shadowParams.x * (1.0 + slope * 2.0);

  let clip = view.viewProjection * vec4<f32>(biased, 1.0);
  if (clip.w <= 0.0) { return 1.0; }
  let ndc = clip.xyz / clip.w;
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return 1.0; }

  let texel = 1.0 / size;
  var total = 0.0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      total = total + textureSampleCompareLevel(
        localShadowMap, shadowSampler,
        uv + vec2<f32>(f32(x), f32(y)) * texel,
        layer, ndc.z,
      );
    }
  }
  return total / 9.0;
}

/**
 * Which light cluster a fragment falls in.
 *
 * Must match sliceFor() in clustered.js exactly. The two derive the same
 * mapping from opposite ends -- the compute pass inverts it to find a slice's
 * depth range, this applies it forwards -- and if they disagree, lights are
 * assigned to cells nothing looks up.
 */
fn clusterFor(fragCoord : vec2<f32>, viewDepth : f32) -> u32 {
  let tileX = min(u32(fragCoord.x / frame.clusterDepth.z), frame.clusterGrid.x - 1u);
  let tileY = min(u32(fragCoord.y / frame.clusterDepth.w), frame.clusterGrid.y - 1u);

  // Exponential in depth: uniform slices would spend almost every cell on the
  // far half of the view, where perspective makes them enormous.
  let raw = log(max(viewDepth, 1e-4)) * frame.clusterDepth.x + frame.clusterDepth.y;
  let slice = u32(clamp(raw, 0.0, f32(frame.clusterGrid.z - 1u)));

  return (slice * frame.clusterGrid.y + tileY) * frame.clusterGrid.x + tileX;
}

/**
 * The skinned vertex path.
 *
 * glTF 3.7.3.3 is explicit that a skinned mesh's own node transform is
 * IGNORED: the joints place it entirely, because each joint matrix already
 * carries its node's world transform. So draw.model does not appear here.
 * Using it as well would apply the skeleton root's transform twice.
 *
 * Each palette entry is jointWorld * inverseBind, built on the CPU from
 * matrices the transform hierarchy already composes. draw.paletteOffset is
 * where this INSTANCE's joints begin, which is what lets two characters in
 * different poses share a batch and a draw call.
 */
@vertex
fn vsSkinned(
  @builtin(instance_index) instance : u32,
  @builtin(vertex_index)   vertex   : u32,
  @location(0) position : vec3<f32>,
  @location(1) normal   : vec3<f32>,
  @location(2) uv       : vec2<f32>,
  @location(3) tangent  : vec4<f32>,
  @location(4) uv1      : vec2<f32>,
  @location(5) color    : vec4<f32>,
  @location(6) joints   : vec4<u32>,
  @location(7) weights  : vec4<f32>,
) -> VertexOut {
  var out : VertexOut;

  let draw = drawData[visibleItems[batch.firstVisible + instance]];
  let base = draw.paletteOffset;
  // Morph, then skin. A target is authored against the bind pose, so it has to
  // move the vertex before the joints take it out of that pose.
  let m = applyMorph(draw, vertex, position, normal, tangent.xyz);

  // Linear blend skinning: the weighted sum of matrices, applied once. Summing
  // the MATRICES and transforming once is not the same as transforming four
  // times and summing -- it is, for an affine transform, and it is one matrix
  // multiply instead of four.
  let skin = palette[base + joints.x] * weights.x
           + palette[base + joints.y] * weights.y
           + palette[base + joints.z] * weights.z
           + palette[base + joints.w] * weights.w;

  let world = skin * vec4<f32>(m.position, 1.0);
  out.world = world.xyz;
  out.clip = frame.viewProjection * world;
  out.modelScale = vec3<f32>(length(skin[0].xyz), length(skin[1].xyz), length(skin[2].xyz));

  // The blended matrix's upper 3x3 for normals rather than its inverse
  // transpose. Exact while the joints are rigid, which is what a skeleton is;
  // it skews normals under non-uniform joint scale, which almost nothing
  // authors and every real-time skinning path accepts.
  let skin3 = mat3x3<f32>(skin[0].xyz, skin[1].xyz, skin[2].xyz);
  out.normal = normalize(skin3 * m.normal);
  out.tangent = normalize(skin3 * m.tangent);
  out.bitangent = cross(out.normal, out.tangent) * tangent.w;

  out.uv = uv;
  out.uv1 = uv1;
  out.color = color;
  return out;
}

/**
 * The surface, then the fog between it and the eye. Everything that shades --
 * opaque, blended, transmissive, OIT -- comes through here, so they all sit
 * in the same fog. The ambient term the occlusion pass takes its share of is
 * dimmed with the rest.
 */
fn shade(v : VertexOut, frontFacing : bool) -> vec4<f32> {
  let colour = shadeSurface(v, frontFacing);
  if (frame.fog.x <= 0.0) { return colour; }
  let toSurface = v.world - frame.cameraPosition.xyz;
  let distance = length(toSurface);
  let direction = toSurface / max(distance, 1e-6);
  let through = exp(-fogDepth(frame.fog, frame.cameraPosition.xyz, direction, distance));
  let inscatter = frame.fogAlbedo.rgb * fogMeanRadiance(irradiance, envSampler) + frame.fogLight.rgb;
  ambientOut = ambientOut * through;
  return vec4<f32>(min(colour.rgb * through + inscatter * (1.0 - through), vec3<f32>(65504.0)), colour.a);
}

fn shadeSurface(v : VertexOut, frontFacing : bool) -> vec4<f32> {
  // Which UV set each map samples, one bit apiece. A select rather than a
  // branch: both sets are interpolated already, so picking between them is
  // free and uniform across the quad, where a branch would not be.
  let uvSets = u32(material.uvSets);
  let uvBaseColor = transformUV(0u, select(v.uv, v.uv1, (uvSets & 1u) != 0u));
  let uvMetallicRoughness = transformUV(1u, select(v.uv, v.uv1, (uvSets & 2u) != 0u));
  let uvNormal = transformUV(2u, select(v.uv, v.uv1, (uvSets & 4u) != 0u));
  let uvOcclusion = transformUV(3u, select(v.uv, v.uv1, (uvSets & 8u) != 0u));
  let uvEmissive = transformUV(4u, select(v.uv, v.uv1, (uvSets & 16u) != 0u));

  // COLOR_0 multiplies base colour, per the spec. An asset without it carries
  // opaque white, so this costs those nothing and needs no variant.
  let sampled = textureSample(baseColorMap, surfSampler, uvBaseColor)
    * material.baseColor * v.color;

  if (USE_ALPHA_MASK) {
    if (sampled.a < material.alphaCutoff) { discard; }
  }

  // KHR_materials_unlit: the base colour is the whole answer -- no lights, no
  // environment, no emission. No ambient either, so occlusion leaves it be.
  if (material.unlit > 0.5) {
    ambientOut = vec3<f32>(0.0);
    return vec4<f32>(min(sampled.rgb, vec3<f32>(65504.0)), sampled.a);
  }

  // glTF puts roughness in G and metallic in B. Occlusion is its own texture,
  // even though exporters usually pack it into R of this one.
  let mr = textureSample(mrMap, surfSampler, uvMetallicRoughness);
  let roughness = clamp(mr.g * material.roughness, 0.045, 1.0);
  let metallic  = clamp(mr.b * material.emissive.w, 0.0, 1.0);

  // The spec's blend: strength 0 disables the map entirely rather than
  // multiplying ambient by zero.
  let occlusionSample = textureSample(occlusionMap, surfSampler, uvOcclusion).r;
  let occlusion = 1.0 + material.occlusionStrength * (occlusionSample - 1.0);

  // Tangent-space normal into world space.
  //
  // The geometric normal is flipped for a back face. A double-sided material
  // draws both windings from one set of vertices, so a back face arrives with
  // the normal of the front it was authored as -- pointing away from the eye.
  // Left alone, NoL clamps to zero and every back face shades ambient-only,
  // which is the whole leaf on a tree, the inside of a curtain, or the reverse
  // of any thin panel. Single-sided geometry never reaches here with a back
  // face, since the pipeline culls it, so this costs those nothing.
  //
  // The bitangent flips with the normal to keep the basis right-handed. The
  // tangent does not: it follows the UV's u axis, which does not reverse.
  let facing = select(-1.0, 1.0, frontFacing);
  var tangentNormal = (textureSample(normalMap, surfSampler, uvNormal).xyz * 2.0 - 1.0)
                    * vec3<f32>(material.normalScale, material.normalScale, 1.0);
  tangentNormal = vec3<f32>(untransformNormal(4u, tangentNormal.xy), tangentNormal.z);
  let geometric = normalize(v.normal) * facing;
  let n = mapNormal(v, geometric, facing, tangentNormal);

  let view = normalize(frame.cameraPosition.xyz - v.world);
  let NoV = max(dot(n, view), 1e-4);

  // View-space depth, shared by cascade selection and the cluster lookup.
  // Both are defined against planes of constant z on the CPU side, so neither
  // may use radial distance.
  //
  // Measured along the view axis rather than read from clip.w. That used to be
  // 1 / v.clip.w, which is -viewZ only because a PERSPECTIVE projection has -1
  // in its w row; an orthographic one has w = 1 everywhere, so every fragment
  // would have claimed to sit at depth 1 and read the lights and shadow
  // cascade for that slice. The dot product is the same number in both.
  let viewDepth = dot(v.world - frame.cameraPosition.xyz, frame.cameraForward.xyz);
  // The cell decals and punctual lights are both listed by; see clusterFor.
  let cluster = clusterFor(v.clip.xy, viewDepth);

  // A dielectric reflects ((ior - 1) / (ior + 1))^2 head-on -- 4% at glTF's
  // default 1.5 -- tinted and weighted by KHR_materials_specular, whose
  // strength is the texture's alpha and whose colour its rgb. A metal reflects
  // its own colour and has no diffuse term at all.
  // Decals paint the base colour, before anything is lit.
  var albedo = sampled.rgb;
  let worldDx = dpdx(v.world);
  let worldDy = dpdy(v.world);
  if (DECALS) { albedo = applyDecals(albedo, v.world, geometric, worldDx, worldDy, cluster); }
  let r = (material.ior - 1.0) / (material.ior + 1.0);
  // The factors hold either way; only their textures need the extended shader.
  var specularColor = material.specularColor.rgb;
  var specularStrength = material.specular;
  if (EXTENSIONS) {
    specularColor = specularColor * sampleExtension(${KIND.specularColor}, v.uv, v.uv1).rgb;
    specularStrength = specularStrength * sampleExtension(${KIND.specular}, v.uv, v.uv1).a;
  }
  var s : Surface;
  s.position = v.world;
  s.n = n;
  s.view = view;
  s.NoV = NoV;
  s.roughness = roughness;
  s.diffuse = albedo * (1.0 - metallic);
  s.dielectricF0 = min(r * r * specularColor, vec3<f32>(1.0));
  s.specularWeight = specularStrength;
  s.f0 = mix(s.dielectricF0 * s.specularWeight, albedo, metallic);
  s.f90 = mix(s.specularWeight, 1.0, metallic);

  // KHR_materials_anisotropy: direction from the texture's red and green,
  // strength from its blue -- (1, 0) at full strength without one, which the
  // absent texture's white would not give -- turned by the rotation, then
  // into world space along the tangent frame.
  if (EXTENSIONS && material.anisotropyStrength > 0.0) {
    var direction = vec2<f32>(1.0, 0.0);
    var strength = material.anisotropyStrength;
    if (extensionBound(${KIND.anisotropy})) {
      let texel = sampleExtension(${KIND.anisotropy}, v.uv, v.uv1).rgb;
      let d = texel.rg * 2.0 - 1.0;
      if (dot(d, d) > F32_MIN_NORMAL) { direction = normalize(d); }
      strength = strength * texel.b;
    }
    let c = cos(material.anisotropyRotation);
    let sn = sin(material.anisotropyRotation);
    direction = mat2x2<f32>(c, sn, -sn, c) * direction;
    let tangentLength = dot(v.tangent, v.tangent);
    let bitangentLength = dot(v.bitangent, v.bitangent);
    if (tangentLength > F32_MIN_NORMAL && bitangentLength > F32_MIN_NORMAL) {
      let t = v.tangent * inverseSqrt(tangentLength) * direction.x
        + v.bitangent * inverseSqrt(bitangentLength) * facing * direction.y;
      let b = cross(geometric, t);
      if (dot(t, t) > F32_MIN_NORMAL && dot(b, b) > F32_MIN_NORMAL) {
        s.anisotropy = strength;
        s.anisotropicT = normalize(t);
        s.anisotropicB = normalize(b);
        let a = roughness * roughness;
        s.alphaT = mix(a, 1.0, strength * strength);
      }
    }
  }

  // KHR_materials_iridescence: strength from the texture's red, thickness
  // between the two bounds by its green. Taken once, at the view -- per light
  // it would be the costliest thing in the shader.
  if (EXTENSIONS && material.iridescence > 0.0) {
    s.iridescence = material.iridescence * sampleExtension(${KIND.iridescence}, v.uv, v.uv1).r;
    let thickness = mix(material.iridescenceThickness.x, material.iridescenceThickness.y,
      sampleExtension(${KIND.iridescenceThickness}, v.uv, v.uv1).g);
    s.iridescenceDielectric = iridescentFresnel(material.iridescenceIor, NoV, thickness, s.dielectricF0 * s.specularWeight);
    s.iridescenceF = mix(s.iridescenceDielectric,
      iridescentFresnel(material.iridescenceIor, NoV, thickness, albedo), metallic);
  }

  // KHR_materials_transmission, from the texture's red, and _volume's
  // thickness from its green. The opaque scene is read where the view ray
  // leaves the surface: straight through a thin wall, or bent by the index of
  // refraction and carried the thickness through a volume. The blur is the
  // reference viewer's: roughness, scaled to nothing as the index falls to 1,
  // where a thin wall bends nothing.
  if (EXTENSIONS && material.transmission > 0.0) {
    s.transmission = material.transmission * sampleExtension(${KIND.transmission}, v.uv, v.uv1).r;
    s.transmissionRoughness = roughness * clamp(material.ior * 2.0 - 2.0, 0.0, 1.0);
    s.transmittance = vec3<f32>(1.0);
    var exit = v.world;
    let thickness = material.thickness * sampleExtension(${KIND.thickness}, v.uv, v.uv1).g;
    if (thickness > 0.0) {
      // An index of 0 is the spec's infinity: the ray goes straight in.
      let eta = select(0.0, 1.0 / material.ior, material.ior > 0.0);
      let refracted = refract(-view, n, eta);
      if (dot(refracted, refracted) > F32_MIN_NORMAL) {
        let ray = normalize(refracted) * thickness * v.modelScale;
        exit = v.world + ray;
        // Beer's law, as the spec writes it: c ^ (x / d).
        if (material.attenuationDistance > 0.0) {
          s.transmittance = pow(material.attenuationColor.rgb, vec3<f32>(length(ray) / material.attenuationDistance));
        }
      }
    }
    let clip = frame.viewProjection * vec4<f32>(exit, 1.0);
    let screen = vec2<f32>(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
    let lod = log2(f32(textureDimensions(behindMap).x)) * s.transmissionRoughness;
    s.behind = textureSampleLevel(behindMap, envSampler, screen, lod).rgb * s.transmittance;
  }

  // KHR_materials_sheen: colour from the texture's rgb, roughness from its
  // alpha. Skipped unless the factor asks for any, which also keeps the
  // texture reads uniform.
  if (EXTENSIONS && maxChannel(material.sheenColor.rgb) > 0.0) {
    s.sheenColor = material.sheenColor.rgb * sampleExtension(${KIND.sheenColor}, v.uv, v.uv1).rgb;
    s.sheenRoughness = material.sheenRoughness * sampleExtension(${KIND.sheenRoughness}, v.uv, v.uv1).a;
    // E can pass 1 at the lowest roughness and the most grazing view, where
    // the spec's visibility fit overshoots; nothing beneath goes negative.
    s.sheenScaling = max(1.0 - maxChannel(s.sheenColor) * sheenAlbedo(NoV, s.sheenRoughness), 0.0);
  }

  // KHR_materials_clearcoat: strength from the texture's red, roughness from
  // its green. Without its own normal map the coat takes none: the spec keeps
  // the base's bumps under a smooth coat.
  if (EXTENSIONS && material.clearcoat > 0.0) {
    s.coat = material.clearcoat * sampleExtension(${KIND.clearcoat}, v.uv, v.uv1).r;
    s.coatRoughness = clamp(
      material.clearcoatRoughness * sampleExtension(${KIND.clearcoatRoughness}, v.uv, v.uv1).g, 0.045, 1.0);
    s.coatN = geometric;
    if (extensionBound(${KIND.clearcoatNormal})) {
      let row = (${CORE_TEXTURE_COUNT}u + ${KIND.clearcoatNormal}) * 2u;
      var coatTangent = (sampleExtension(${KIND.clearcoatNormal}, v.uv, v.uv1).xyz * 2.0 - 1.0)
        * vec3<f32>(material.clearcoatNormalScale, material.clearcoatNormalScale, 1.0);
      coatTangent = vec3<f32>(untransformNormal(row, coatTangent.xy), coatTangent.z);
      s.coatN = mapNormal(v, geometric, facing, coatTangent);
    }
    s.coatNoV = max(dot(s.coatN, view), 1e-4);
    s.coatFresnel = 0.04 + 0.96 * pow(1.0 - s.coatNoV, 5.0);
  }

  // ---- directional lights ----
  // Every one alike: the same BRDF, and its cascades if it casts. A count
  // stored as its VALUE, not its bits: small integers reinterpreted as f32 are
  // subnormals, which a backend may flush to zero -- and every directional
  // light would vanish with nothing reported. The slot likewise.
  //
  // Shadows attenuate the DIRECT term only. Ambient comes from the whole sky,
  // which no shadow map says anything about -- multiplying it too is the usual
  // cause of shadowed areas going implausibly black.
  var direct = vec3<f32>(0.0);
  let directionalCount = u32(frame.cameraForward.w);
  for (var di = 0u; di < directionalCount; di = di + 1u) {
    let dl = directionals[di];
    let dL = -dl.direction.xyz;
    let dNoL = dot(n, dL);
    let slot = i32(dl.direction.w) - 1;
    if (dNoL <= 0.0) {
      // From behind: only what a transmissive surface lets through, shadowed
      // as its far side would be.
      if (EXTENSIONS && s.transmission > 0.0) {
        var through = 1.0;
        if (slot >= 0) { through = directionalVisibility(slot, v.world, -n, -dNoL, viewDepth); }
        direct = direct + surfaceTransmission(s, dL, dl.color.rgb * through);
      }
      continue;
    }
    var shadow = 1.0;
    if (slot >= 0) { shadow = directionalVisibility(slot, v.world, n, dNoL, viewDepth); }
    if (shadow <= 0.0) { continue; }
    direct = direct + surfaceLight(s, dL, dNoL, dl.color.rgb * shadow);
  }

  // ---- clustered punctual lights ----
  // Only the lights the compute pass put in THIS fragment's cell are touched,
  // so a scene with hundreds costs a handful per pixel.
  //
  // DEPTH, not radial distance. buildClusters slices the frustum with planes at
  // constant view z, so a fragment has to pick its slice with the same
  // quantity. Radial distance agrees only on the view axis and grows by 1/cos
  // away from it, which sends edge-of-screen fragments to a slice further away
  // than the froxel they are actually in: they then read a light list built for
  // somewhere else, and the lights that should reach them are simply absent.
  //
  // Lights append themselves concurrently, so a crowded cell's count can run
  // past what its slice of the index list holds. Only that many are real.
  let lightCount = min(clusterCounts[cluster], ${MAX_LIGHTS_PER_CLUSTER}u);
  let clusterBase = cluster * ${MAX_LIGHTS_PER_CLUSTER}u;

  for (var li = 0u; li < lightCount; li = li + 1u) {
    let light = lights[clusterIndices[clusterBase + li]];

    let toLight = light.positionRadius.xyz - v.world;
    let lightDistance = length(toLight);
    if (lightDistance >= light.positionRadius.w) { continue; }

    let l = toLight / max(lightDistance, 1e-4);
    let lightNoL = dot(n, l);
    // From behind, only a transmissive surface has anything to show.
    let through = lightNoL <= 0.0;
    if (through && !(EXTENSIONS && s.transmission > 0.0)) { continue; }

    // Inverse-square, windowed so it reaches exactly zero at the radius.
    // Physical falloff never does, and a light that is merely very dim at its
    // cutoff pops visibly when a cluster boundary drops it.
    let ratio = lightDistance / light.positionRadius.w;
    let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    let attenuation = (window * window) / max(lightDistance * lightDistance, 1e-4);

    var cone = 1.0;
    if (light.coneFalloff.z > 0.5) {
      // The scale and offset were precomputed on the CPU, so the cone test is
      // a multiply-add rather than two cosines per pixel.
      let alignment = dot(-l, light.directionCone.xyz);
      cone = clamp(alignment * light.coneFalloff.x + light.coneFalloff.y, 0.0, 1.0);
      cone = cone * cone;
    }
    if (cone <= 0.0) { continue; }
    if (through) {
      let behindShadow = localVisibility(light, v.world, -n, -lightNoL);
      direct = direct + surfaceTransmission(s, l,
        light.colorIntensity.rgb * light.colorIntensity.a * attenuation * cone * behindShadow);
      continue;
    }
    let shadow = localVisibility(light, v.world, n, lightNoL);
    if (shadow <= 0.0) { continue; }

    direct = direct + surfaceLight(s, l, lightNoL,
      light.colorIntensity.rgb * light.colorIntensity.a * attenuation * cone * shadow);
  }

  // ---- ambient, from the prebaked environment ----
  let ambient = surfaceAmbient(s) * occlusion;
  ambientOut = min(ambient, vec3<f32>(65504.0));

  // A coat dims what shines through it as it dims everything else beneath.
  var emissive = textureSample(emissiveMap, surfSampler, uvEmissive).rgb * material.emissive.rgb;
  if (EXTENSIONS) { emissive = emissive * (1.0 - s.coat * s.coatFresnel); }

  // Linear HDR, deliberately not clamped to 1. Bloom needs to know a highlight
  // was at 60x white, not that it was clipped; the post stack tonemaps once at
  // the end. Exposure lives there too, for the same reason.
  //
  // Clamped to 65504 only, which is the largest finite value the rgba16float
  // target holds. A near-mirror metal under a bright sun goes past it -- GGX
  // peaks near 78,000 at the roughness floor. Chrome on Windows saturates the
  // store on its own (measured: no change without this, even at 5,000x);
  // a backend that rounds to infinity instead would hand bloom's blur an
  // infinity to spread as NaN. One min per fragment, so it stays.
  return vec4<f32>(min(direct + ambient + emissive, vec3<f32>(65504.0)), sampled.a);
}

@fragment
fn fs(v : VertexOut, @builtin(front_facing) frontFacing : bool) -> @location(0) vec4<f32> {
  return shade(v, frontFacing);
}

struct AoOut {
  @location(0) color   : vec4<f32>,
  @location(1) ambient : vec4<f32>,
};

/** The same shade, and its ambient term on its own for the occlusion pass (ao.js). */
@fragment
fn fsAO(v : VertexOut, @builtin(front_facing) frontFacing : bool) -> AoOut {
  let color = shade(v, frontFacing);
  return AoOut(color, vec4<f32>(ambientOut, 1.0));
}

// Weighted-blended order-independent transparency (McGuire and Bavoil 2013).
//
// The opt-in second transparency path. The sorted one is EXACT for separated
// convex objects and has no answer at all for interpenetrating ones, because
// no single per-object order exists there. This is approximate everywhere and
// needs no order, which makes the two different tools rather than one being
// better: architectural glass wants the sorted path, smoke and foliage want
// this one.
//
// Two targets, resolved in one pass afterwards:
//   accum  sum of colour * alpha * w, and of alpha * w
//   reveal product of (1 - alpha), which is how much background survives
struct OitOut {
  @location(0) accum  : vec4<f32>,
  @location(1) reveal : f32,
};

/**
 * How much a fragment counts, by depth.
 *
 * The whole approximation lives here: nearer fragments are weighted more, so
 * the result resembles a correct back-to-front composite without anything
 * being sorted. McGuire's paper picks its constants against a normalised view
 * depth; this engine has reverse-Z, where the depth buffer value is already
 * 1 at the near plane and falls to 0 with no far plane to normalise against,
 * so it is used directly and the cubic is all that remains of the tuning.
 *
 * The range is derived rather than chosen. The accumulation target is
 * rgba16float, whose largest finite value is 65504, so the budget is that
 * divided by the depth complexity this stays well-behaved at. Past that the
 * sum saturates and the nearest layers stop dominating.
 *
 * That budget is per fragment, and the colour is HDR -- up to 65504 itself,
 * a sun glint on smooth glass -- so the weight is also capped to keep
 * colour * alpha * w inside it. It assumed colour <= 1, and one glint filled
 * the target on its own: too dim where the store saturates, infinity spread
 * by bloom as NaN where it does not. For colour <= 1 the cap never engages,
 * and scaling w leaves a fragment's own accum.rgb / accum.a exact.
 */
fn oitWeight(colour : vec4<f32>, depth : f32) -> f32 {
  let budget = 65504.0 / ${OIT_LAYER_BUDGET}.0;
  let w = colour.a * clamp(depth * depth * depth * budget, 1.0, budget);
  let largest = max(max(colour.r, max(colour.g, colour.b)) * colour.a, colour.a);
  return min(w, budget / max(largest, 1e-6));
}

@fragment
fn fsOIT(v : VertexOut, @builtin(front_facing) frontFacing : bool) -> OitOut {
  let colour = shade(v, frontFacing);
  // builtin(position).z in a fragment is already the value that would go to the
  // depth buffer, so under reverse-Z it is 1 at the near plane and falls
  // toward 0. No division and no unprojection.
  let w = oitWeight(colour, clamp(v.clip.z, 0.0, 1.0));

  var out : OitOut;
  out.accum = vec4<f32>(colour.rgb * colour.a, colour.a) * w;
  out.reveal = colour.a;
  return out;
}
`;

/** Frame uniform size in bytes. Matches the struct above. */
export const FRAME_BYTES = 272;
