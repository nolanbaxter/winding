// Shared BRDF and sampling code, as a WGSL source fragment.
//
// Exported as a string rather than a .wgsl file because two shaders need it --
// the IBL prefilter and the surface shader must use the SAME GGX distribution
// or the prefiltered environment answers a question the lighting is not asking,
// and the error shows up as ambient that is subtly too bright or too dim at
// grazing angles. One definition, concatenated into both.
//
// No preprocessor: an #include system for one shared file would be more
// machinery than the problem has.

export const BRDF_WGSL = /* wgsl */ `
const PI = 3.14159265359;

// GGX / Trowbridge-Reitz normal distribution. "roughness" is glTF's perceptual
// roughness; squaring it once here is what makes the parameter feel linear.
fn distributionGGX(NoH : f32, roughness : f32) -> f32 {
  let a  = roughness * roughness;
  let a2 = a * a;
  let d  = NoH * NoH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-7);
}

// Height-correlated Smith visibility (Heitz 2014). This is G / (4 NoL NoV), so
// the 4 NoL NoV denominator of the Cook-Torrance specular term is already
// folded in and must not be applied again.
fn visibilitySmithGGX(NoV : f32, NoL : f32, roughness : f32) -> f32 {
  let a  = roughness * roughness;
  let a2 = a * a;
  let lambdaV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  let lambdaL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / max(lambdaV + lambdaL, 1e-5);
}

fn fresnelSchlick(cosTheta : f32, f0 : vec3<f32>) -> vec3<f32> {
  let f = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
  return f0 + (vec3<f32>(1.0) - f0) * f;
}

// Van der Corput radical inverse: bit-reverse an integer into [0,1). Paired
// with a linear sequence it gives the Hammersley set -- a low-discrepancy
// sequence that converges far faster than uniform random for this integral.
fn radicalInverseVdC(input : u32) -> f32 {
  var bits = input;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i : u32, count : u32) -> vec2<f32> {
  return vec2<f32>(f32(i) / f32(count), radicalInverseVdC(i));
}

// Sample a half-vector from the GGX distribution around the normal. Importance
// sampling: draw samples where the distribution is large instead of uniformly,
// so a few dozen samples land where thousands of uniform ones would be needed.
fn importanceSampleGGX(xi : vec2<f32>, n : vec3<f32>, roughness : f32) -> vec3<f32> {
  let a = roughness * roughness;

  let phi = 2.0 * PI * xi.x;
  let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
  let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

  let h = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

  // Any basis around n will do; crossing with the axis n leans on least keeps
  // it well-conditioned.
  let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.999);
  let tangentX = normalize(cross(up, n));
  let tangentY = cross(n, tangentX);

  return normalize(tangentX * h.x + tangentY * h.y + n * h.z);
}

// Analytic fit to the split-sum BRDF integral (Lazarov). The usual approach
// bakes this into a 2D lookup texture; the polynomial is accurate to well
// under a perceptible difference and removes a texture, a bind slot, and a
// generation pass. Returns (scale, bias) for F0.
fn envBRDFApprox(NoV : f32, roughness : f32) -> vec2<f32> {
  let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
  return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// Direction for a point on a cubemap face. WebGPU uses the D3D/Vulkan face
// layout, and getting a sign wrong here mirrors the environment in a way that
// only shows up as reflections moving the wrong way.
fn cubeDirection(face : u32, uv : vec2<f32>) -> vec3<f32> {
  let c = uv * 2.0 - 1.0;
  let u = c.x;
  let v = c.y;
  switch face {
    case 0u: { return normalize(vec3<f32>( 1.0,   -v,   -u)); }  // +X
    case 1u: { return normalize(vec3<f32>(-1.0,   -v,    u)); }  // -X
    case 2u: { return normalize(vec3<f32>(   u,  1.0,    v)); }  // +Y
    case 3u: { return normalize(vec3<f32>(   u, -1.0,   -v)); }  // -Y
    case 4u: { return normalize(vec3<f32>(   u,   -v,  1.0)); }  // +Z
    default: { return normalize(vec3<f32>(  -u,   -v, -1.0)); }  // -Z
  }
}

// Procedural sky. Linear HDR values -- the sun is far brighter than 1.0, which
// is the point of storing the environment in a float format.
fn skyRadiance(dir : vec3<f32>) -> vec3<f32> {
  let sunDirection = normalize(vec3<f32>(0.35, 0.55, 0.45));
  let height = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

  let ground  = vec3<f32>(0.10, 0.09, 0.08);
  let horizon = vec3<f32>(0.62, 0.66, 0.74);
  let zenith  = vec3<f32>(0.16, 0.30, 0.60);

  var color = mix(horizon, zenith, smoothstep(0.5, 1.0, height));
  color = mix(ground, color, smoothstep(0.47, 0.53, height));

  let toSun = max(dot(dir, sunDirection), 0.0);
  let disc  = pow(toSun, 900.0) * 60.0;
  let glow  = pow(toSun, 8.0) * 0.5;
  return color + vec3<f32>(1.0, 0.93, 0.80) * (disc + glow);
}
`;

/** The same sun direction the sky shader uses, for the analytic key light. */
export const SUN_DIRECTION = (() => {
  const length = Math.hypot(0.35, 0.55, 0.45);
  return Float32Array.from([0.35 / length, 0.55 / length, 0.45 / length]);
})();
