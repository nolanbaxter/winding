// Physically based surface shader: metallic-roughness, normal mapped, lit by
// one analytic sun plus the prebaked environment.
//
// Everything in here happens in LINEAR light. The only encode is the -srgb
// swap-chain view at the very end, which is why there is no
// pow(x, 1/2.2) anywhere below.

import { BRDF_WGSL } from './brdf.js';
import { MAX_LIGHTS_PER_CLUSTER } from '../clustered.js';

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

export const PBR_SHADER = /* wgsl */ `
${BRDF_WGSL}

// Pipeline-overridable constant: set per variant, so the MASK variant gets its
// own compiled pipeline and opaque geometry never pays for the discard.
override USE_ALPHA_MASK : bool = false;

struct Frame {
  viewProjection : mat4x4<f32>,             //   0
  cameraPosition : vec4<f32>,               //  64  w = exposure
  sunDirection   : vec4<f32>,               //  80  w = prefiltered mip count
  sunColor       : vec4<f32>,               //  96  w = active cascade count
  cascades       : array<mat4x4<f32>, 4>,   // 112
  cascadeSplits  : vec4<f32>,               // 368  view DEPTH each cascade ends at
  cascadeTexel   : vec4<f32>,               // 384  world size of one texel, per cascade
  shadowParams   : vec4<f32>,               // 400  x = normal bias, y = map size
  clusterGrid    : vec4<u32>,               // 416  x, y, z cells; w = light count
  clusterDepth   : vec4<f32>,               // 432  x slice scale, y bias, zw tile size
};                                          // 448

struct Light {
  positionRadius : vec4<f32>,
  colorIntensity : vec4<f32>,
  directionCone  : vec4<f32>,
  coneFalloff    : vec4<f32>,   // x scale, y offset, z type (0 point, 1 spot)
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
  model        : mat4x4<f32>,     //  0
  normalMatrix : mat3x3<f32>,     // 64   occupies 48 bytes
};                                // 112

/** The only thing still bound per draw: where this batch's slice begins. */
struct Batch {
  firstVisible : u32,
};

struct Material {
  baseColor         : vec4<f32>,  //  0
  emissive          : vec4<f32>,  // 16   w = metallic
  roughness         : f32,        // 32
  normalScale       : f32,        // 36
  alphaCutoff       : f32,        // 40
  occlusionStrength : f32,        // 44
  // Bit per texture slot: set means that map samples UV set 1. Carried as f32
  // because the rest of the struct is, and a u32 here would move every offset.
  uvSets            : f32,        // 48
};                                // 64

@group(0) @binding(0) var<uniform> frame        : Frame;
@group(0) @binding(1) var          irradiance   : texture_cube<f32>;
@group(0) @binding(2) var          prefiltered  : texture_cube<f32>;
@group(0) @binding(3) var          envSampler   : sampler;
@group(0) @binding(4) var          shadowMap    : texture_depth_2d_array;
// A comparison sampler does the depth test and the bilinear blend in one fetch,
// so each PCF tap returns an already-filtered occlusion fraction.
@group(0) @binding(5) var          shadowSampler: sampler_comparison;
@group(0) @binding(6) var<storage, read> lights         : array<Light>;
@group(0) @binding(7) var<storage, read> clusterIndices : array<u32>;
@group(0) @binding(8) var<storage, read> clusterCounts  : array<u32>;
@group(0) @binding(9) var<storage, read> drawData       : array<DrawData>;
// Written by the cull compute shader: the compacted list of surviving objects,
// grouped into one contiguous slice per batch.
@group(0) @binding(10) var<storage, read> visibleItems  : array<u32>;

@group(2) @binding(0) var<uniform> material     : Material;
@group(2) @binding(1) var          baseColorMap : texture_2d<f32>;
@group(2) @binding(2) var          normalMap    : texture_2d<f32>;
@group(2) @binding(3) var          mrMap        : texture_2d<f32>;
@group(2) @binding(4) var          occlusionMap : texture_2d<f32>;
@group(2) @binding(5) var          emissiveMap  : texture_2d<f32>;
@group(2) @binding(6) var          surfSampler  : sampler;

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
};

@vertex
fn vs(
  @builtin(instance_index) instance : u32,
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

  let world = draw.model * vec4<f32>(position, 1.0);
  out.world = world.xyz;
  out.clip = frame.viewProjection * world;

  // Normals use the inverse-transpose; tangents do NOT. A tangent is a
  // direction along the surface, so it transforms like a position delta and
  // the model matrix is correct for it.
  out.normal = normalize(draw.normalMatrix * normal);
  out.tangent = normalize((draw.model * vec4<f32>(tangent.xyz, 0.0)).xyz);

  // tangent.w is the handedness the importer computed per vertex, which is
  // what keeps mirrored UV islands from lighting inside out.
  out.bitangent = cross(out.normal, out.tangent) * tangent.w;
  out.uv = uv;
  out.uv1 = uv1;
  out.color = color;
  return out;
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

fn cascadeMatrix(cascade : i32, p : vec4<f32>) -> vec4<f32> {
  if (cascade == 0) { return frame.cascades[0] * p; }
  if (cascade == 1) { return frame.cascades[1] * p; }
  if (cascade == 2) { return frame.cascades[2] * p; }
  return frame.cascades[3] * p;
}

/**
 * How much of the sun reaches this point. 1 is fully lit.
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
fn sunVisibility(worldPosition : vec3<f32>, normal : vec3<f32>, NoL : f32, viewDepth : f32) -> f32 {
  if (NoL <= 0.0) { return 1.0; }

  let cascade = selectCascade(viewDepth);
  if (cascade < 0) { return 1.0; }

  // Scale the offset with how glancing the light is: a surface almost edge-on
  // needs far more, which is the same reason the map itself uses slope-scaled
  // hardware bias.
  let slope = clamp(1.0 - NoL, 0.0, 1.0);
  let offset = cascadeTexelSize(cascade) * frame.shadowParams.x * (1.0 + slope * 2.0);
  let biased = worldPosition + normal * offset;

  let lightClip = cascadeMatrix(cascade, vec4<f32>(biased, 1.0));
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
  let texel = 1.0 / frame.shadowParams.y;
  var total = 0.0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      total = total + textureSampleCompareLevel(
        shadowMap, shadowSampler,
        uv + vec2<f32>(f32(x), f32(y)) * texel,
        cascade, ndc.z,
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

fn shade(v : VertexOut, frontFacing : bool) -> vec4<f32> {
  // Which UV set each map samples, one bit apiece. A select rather than a
  // branch: both sets are interpolated already, so picking between them is
  // free and uniform across the quad, where a branch would not be.
  let uvSets = u32(material.uvSets);
  let uvBaseColor = select(v.uv, v.uv1, (uvSets & 1u) != 0u);
  let uvMetallicRoughness = select(v.uv, v.uv1, (uvSets & 2u) != 0u);
  let uvNormal = select(v.uv, v.uv1, (uvSets & 4u) != 0u);
  let uvOcclusion = select(v.uv, v.uv1, (uvSets & 8u) != 0u);
  let uvEmissive = select(v.uv, v.uv1, (uvSets & 16u) != 0u);

  // COLOR_0 multiplies base colour, per the spec. An asset without it carries
  // opaque white, so this costs those nothing and needs no variant.
  let sampled = textureSample(baseColorMap, surfSampler, uvBaseColor)
    * material.baseColor * v.color;

  if (USE_ALPHA_MASK) {
    if (sampled.a < material.alphaCutoff) { discard; }
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
  let tangentNormal = (textureSample(normalMap, surfSampler, uvNormal).xyz * 2.0 - 1.0)
                    * vec3<f32>(material.normalScale, material.normalScale, 1.0);
  let tbn = mat3x3<f32>(
    normalize(v.tangent),
    normalize(v.bitangent) * facing,
    normalize(v.normal) * facing,
  );
  let n = normalize(tbn * tangentNormal);

  let view = normalize(frame.cameraPosition.xyz - v.world);
  let NoV = max(dot(n, view), 1e-4);

  // View-space depth, shared by cascade selection and the cluster lookup.
  // Both are defined against planes of constant z on the CPU side, so neither
  // may use radial distance. The projection has -1 in its w row, making clip.w
  // exactly -viewZ; the fragment's builtin position carries the reciprocal.
  let viewDepth = 1.0 / v.clip.w;

  // Dielectrics reflect ~4% head-on; metals reflect their own colour and have
  // no diffuse term at all. That single split is the whole metallic workflow.
  let albedo = sampled.rgb;
  let f0 = mix(vec3<f32>(0.04), albedo, metallic);
  let diffuseColor = albedo * (1.0 - metallic);

  // ---- analytic sun ----
  let l = normalize(frame.sunDirection.xyz);
  let h = normalize(view + l);
  let NoL = max(dot(n, l), 0.0);
  let NoH = max(dot(n, h), 0.0);
  let VoH = max(dot(view, h), 0.0);

  let d = distributionGGX(NoH, roughness);
  let vis = visibilitySmithGGX(NoV, NoL, roughness);
  let f = fresnelSchlick(VoH, f0);

  // vis already contains the 1/(4 NoL NoV) denominator.
  let specular = d * vis * f;
  let kd = (vec3<f32>(1.0) - f);

  // Shadows attenuate the DIRECT term only. Ambient comes from the whole sky,
  // which the sun's shadow map says nothing about -- multiplying it too is the
  // usual cause of shadowed areas going implausibly black.
  let visibility = sunVisibility(v.world, n, NoL, viewDepth);
  var direct = (kd * diffuseColor / PI + specular) * frame.sunColor.rgb * NoL * visibility;

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
  // The projection is [.., -1] in the w row, so clip.w is exactly -viewZ, and
  // the fragment's builtin position carries its reciprocal. No extra uniform.
  let cluster = clusterFor(v.clip.xy, viewDepth);
  let lightCount = clusterCounts[cluster];
  let clusterBase = cluster * ${MAX_LIGHTS_PER_CLUSTER}u;

  for (var li = 0u; li < lightCount; li = li + 1u) {
    let light = lights[clusterIndices[clusterBase + li]];

    let toLight = light.positionRadius.xyz - v.world;
    let lightDistance = length(toLight);
    if (lightDistance >= light.positionRadius.w) { continue; }

    let l = toLight / max(lightDistance, 1e-4);
    let lightNoL = dot(n, l);
    if (lightNoL <= 0.0) { continue; }

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

    let lh = normalize(view + l);
    let lNoH = max(dot(n, lh), 0.0);
    let lVoH = max(dot(view, lh), 0.0);

    let lf = fresnelSchlick(lVoH, f0);
    let lightSpecular = distributionGGX(lNoH, roughness)
                      * visibilitySmithGGX(NoV, lightNoL, roughness) * lf;
    let lightKd = vec3<f32>(1.0) - lf;

    direct = direct
      + (lightKd * diffuseColor / PI + lightSpecular)
      * light.colorIntensity.rgb * light.colorIntensity.a
      * attenuation * cone * lightNoL;
  }

  // ---- ambient, from the prebaked environment ----
  let irradianceSample = textureSample(irradiance, envSampler, n).rgb;
  let ambientDiffuse = irradianceSample * diffuseColor;

  let reflected = reflect(-view, n);
  let maxMip = max(frame.sunDirection.w - 1.0, 0.0);
  // Perceptual roughness indexes the mip chain directly, because that is how
  // ibl.js spaced the levels when it baked them.
  let prefilteredSample = textureSampleLevel(
    prefiltered, envSampler, reflected, roughness * maxMip).rgb;

  let ab = envBRDFApprox(NoV, roughness);
  let ambientSpecular = prefilteredSample * (f0 * ab.x + vec3<f32>(ab.y));

  let ambient = (ambientDiffuse + ambientSpecular) * occlusion;

  let emissive = textureSample(emissiveMap, surfSampler, uvEmissive).rgb * material.emissive.rgb;

  // Linear HDR, deliberately unclamped. Bloom needs to know a highlight was at
  // 60x white, not that it was clipped to 1; the post stack tonemaps once at
  // the end. Exposure lives there too, for the same reason.
  return vec4<f32>(direct + ambient + emissive, sampled.a);
}

@fragment
fn fs(v : VertexOut, @builtin(front_facing) frontFacing : bool) -> @location(0) vec4<f32> {
  return shade(v, frontFacing);
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
 */
fn oitWeight(alpha : f32, depth : f32) -> f32 {
  let maxWeight = 65504.0 / ${OIT_LAYER_BUDGET}.0;
  return alpha * clamp(depth * depth * depth * maxWeight, 1.0, maxWeight);
}

@fragment
fn fsOIT(v : VertexOut, @builtin(front_facing) frontFacing : bool) -> OitOut {
  let colour = shade(v, frontFacing);
  // builtin(position).z in a fragment is already the value that would go to the
  // depth buffer, so under reverse-Z it is 1 at the near plane and falls
  // toward 0. No division and no unprojection.
  let w = oitWeight(colour.a, clamp(v.clip.z, 0.0, 1.0));

  var out : OitOut;
  out.accum = vec4<f32>(colour.rgb * colour.a, colour.a) * w;
  out.reveal = colour.a;
  return out;
}
`;

/** Frame uniform size in bytes. Matches the struct above. */
export const FRAME_BYTES = 448;
