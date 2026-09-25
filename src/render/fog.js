// Exponential height fog, integrated exactly along each view ray.
//
// The medium's density falls off with height, e^(-(y - height) / scaleHeight),
// so it pools in valleys and thins with altitude; without a scale height it is
// the same everywhere. That density integrates along a straight ray in closed
// form, so no marching: one exp per pixel, in the surface shader (so blended,
// transmissive and opaque surfaces agree) and in the sky, which is a ray to
// infinity.
//
// Nothing about its colour is chosen. What the fog scatters toward the eye is
// the light arriving at it: the environment's mean radiance, and each
// directional light's share, isotropically -- times the medium's albedo,
// white unless it absorbs. Shadows inside the fog are not modelled; a ray
// through a shadowed valley scatters as if lit.

/**
 * Meteorological visibility: the distance at which a dark object's contrast
 * against the horizon falls to 2% (Koschmieder). It is what "you can see
 * 200 m" means, and it fixes the extinction: e^(-sigma V) = 0.02.
 */
export const VISIBILITY_CONTRAST = 0.02;

/**
 * The shader's fog coefficients from the options, checked:
 *   visibility   metres, at `height`; required
 *   height       where the density is the one visibility gives; default 0
 *   scaleHeight  metres over which the density falls by e; omit for uniform fog
 *   albedo       the share of light the medium scatters rather than absorbs
 * Returns { extinction, height, inverseScaleHeight, albedo }.
 */
export function fogCoefficients({ visibility, height = 0, scaleHeight = Infinity, albedo = [1, 1, 1] }) {
  if (!(visibility > 0 && Number.isFinite(visibility))) {
    throw new Error(`fog: visibility must be a positive number of metres, got ${visibility}`);
  }
  if (!Number.isFinite(height)) throw new Error(`fog: height must be a finite number, got ${height}`);
  if (!(scaleHeight > 0)) throw new Error(`fog: scaleHeight must be positive, or left out for uniform fog, got ${scaleHeight}`);
  if (!(albedo?.length === 3 && albedo.every((a) => a >= 0 && Number.isFinite(a)))) {
    throw new Error(`fog: albedo must be three non-negative numbers, got ${albedo}`);
  }
  return {
    extinction: -Math.log(VISIBILITY_CONTRAST) / visibility,
    height,
    inverseScaleHeight: 1 / scaleHeight,
    albedo,
  };
}

/**
 * WGSL. Reads its coefficients as arguments, so the surface shader and the
 * sky can hold them in their own uniforms:
 *   fog.x  extinction at fog.y height, per metre; 0 for no fog
 *   fog.z  1 / scale height; 0 for uniform fog
 */
export const FOG_WGSL = /* wgsl */ `
/**
 * The largest exponent whose exp() is still finite in f32 is about 88.7.
 * Exponents are held under it, so a ray deep in a dense layer gives a very
 * large optical depth rather than an infinity a backend may mishandle.
 */
const FOG_EXPONENT_LIMIT = 80.0;

/**
 * Optical depth from the eye along a unit direction, over a distance, or to
 * infinity when the distance is negative. The density integral of
 * sigma e^(-(y - h) k) along y = eye.y + t dy is closed-form; x = dy k d.
 */
fn fogDepth(fog : vec4<f32>, eye : vec3<f32>, direction : vec3<f32>, distance : f32) -> f32 {
  let atEye = fog.x * exp(clamp(-(eye.y - fog.y) * fog.z, -FOG_EXPONENT_LIMIT, FOG_EXPONENT_LIMIT));
  if (distance < 0.0) {
    // To infinity: finite only rising through a thinning layer.
    if (fog.z > 0.0 && direction.y > 0.0) { return atEye / (direction.y * fog.z); }
    return 1e30;
  }
  let x = direction.y * fog.z * distance;
  // (1 - e^-x) / x, which is 1 at x = 0 and needs its series there.
  var spread = 1.0 - 0.5 * x;
  if (abs(x) > 1e-3) { spread = (1.0 - exp(min(-x, FOG_EXPONENT_LIMIT))) / x; }
  return min(atEye * distance * spread, 1e30);
}

/**
 * The environment's mean radiance, from its irradiance at the six axes:
 * irradiance averaged over the sphere is pi times the mean radiance, and six
 * antipodal samples average every spherical harmonic through order two
 * exactly -- which is where a cosine-convolved environment keeps its energy.
 */
fn fogMeanRadiance(irradianceMap : texture_cube<f32>, mapSampler : sampler) -> vec3<f32> {
  let sum = textureSampleLevel(irradianceMap, mapSampler, vec3<f32>(1.0, 0.0, 0.0), 0.0).rgb
    + textureSampleLevel(irradianceMap, mapSampler, vec3<f32>(-1.0, 0.0, 0.0), 0.0).rgb
    + textureSampleLevel(irradianceMap, mapSampler, vec3<f32>(0.0, 1.0, 0.0), 0.0).rgb
    + textureSampleLevel(irradianceMap, mapSampler, vec3<f32>(0.0, -1.0, 0.0), 0.0).rgb
    + textureSampleLevel(irradianceMap, mapSampler, vec3<f32>(0.0, 0.0, 1.0), 0.0).rgb
    + textureSampleLevel(irradianceMap, mapSampler, vec3<f32>(0.0, 0.0, -1.0), 0.0).rgb;
  return sum / (6.0 * 3.14159265359);
}
`;

/**
 * Pack the fog for a uniform: 12 floats, three vec4s -- the coefficients, the
 * albedo, and the directional lights' inscattered radiance (albedo times
 * each light's colour over 4 pi, the isotropic phase function). Zeros for no
 * fog. `directionals` is Scene's packed array, DIRECTIONAL_FLOATS apiece.
 */
export function packFog(out, offset, fog, directionals, directionalCount, stride) {
  out.fill(0, offset, offset + 12);
  if (fog === null || fog === undefined) return;
  const { extinction, height, inverseScaleHeight, albedo } = fogCoefficients(fog);
  out[offset] = extinction;
  out[offset + 1] = height;
  out[offset + 2] = inverseScaleHeight;
  out[offset + 4] = albedo[0];
  out[offset + 5] = albedo[1];
  out[offset + 6] = albedo[2];
  for (let d = 0; d < directionalCount; d++) {
    for (let c = 0; c < 3; c++) {
      out[offset + 8 + c] += albedo[c] * directionals[d * stride + 4 + c] / (4 * Math.PI);
    }
  }
}
