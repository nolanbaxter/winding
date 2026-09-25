// KHR_materials_sheen's directional albedo: how much light a sheen layer
// reflects, by view angle and sheen roughness -- E(NoV, r) in the spec, which
// scales the layer beneath by 1 - max(sheenColor) * E and weights the sheen's
// share of the environment.
//
// The spec says to look E up. This is the table, integrated from the same BRDF
// the shader evaluates -- the Charlie distribution and the spec's Charlie
// visibility. Baked rather than integrated at startup, which takes about
// 350 ms.

/** Entries along each axis: NoV at bin centres, roughness from 0 to 1. */
export const SHEEN_TABLE_SIZE = 16;

const mix = (a, b, t) => a + (b - a) * t;

/** The spec's fit for the Charlie visibility's lambda, in its own variables. */
function charlieL(x, alphaG) {
  const t = (1 - alphaG) * (1 - alphaG);
  return mix(21.5473, 25.3245, t) / (1 + mix(3.82987, 3.32435, t) * x ** mix(0.19823, 0.16801, t))
    + mix(-1.97760, -1.27393, t) * x + mix(-4.32054, -4.85967, t);
}

function charlieLambda(cosTheta, alphaG) {
  return Math.abs(cosTheta) < 0.5
    ? Math.exp(charlieL(cosTheta, alphaG))
    : Math.exp(2 * charlieL(0.5, alphaG) - charlieL(1 - cosTheta, alphaG));
}

/**
 * E(NoV, roughness): the sheen BRDF times NoL over the hemisphere, by the
 * midpoint rule in (theta, phi) with `steps` of each.
 */
export function sheenAlbedo(NoV, roughness, steps = 128) {
  const alphaG = Math.max(roughness * roughness, 1e-7);
  const inverse = 1 / alphaG;
  const vx = Math.sqrt(1 - NoV * NoV);
  const lambdaV = charlieLambda(NoV, alphaG);
  const dTheta = Math.PI / 2 / steps;
  const dPhi = 2 * Math.PI / steps;
  let sum = 0;
  for (let i = 0; i < steps; i++) {
    const theta = (i + 0.5) * dTheta;
    const NoL = Math.cos(theta);
    const sinTheta = Math.sin(theta);
    const V = 1 / ((1 + lambdaV + charlieLambda(NoL, alphaG)) * 4 * NoV * NoL);
    for (let j = 0; j < steps; j++) {
      const phi = (j + 0.5) * dPhi;
      const hx = vx + sinTheta * Math.cos(phi);
      const hy = sinTheta * Math.sin(phi);
      const hz = NoV + NoL;
      const cos2h = (hz * hz) / (hx * hx + hy * hy + hz * hz);
      const D = (2 + inverse) * Math.max(1 - cos2h, 0) ** (inverse * 0.5) / (2 * Math.PI);
      sum += D * V * NoL * sinTheta;
    }
  }
  return sum * dTheta * dPhi;
}

/** Where entry (x, y) of the table was integrated. */
export function sheenTablePoint(x, y) {
  return [(x + 0.5) / SHEEN_TABLE_SIZE, y / (SHEEN_TABLE_SIZE - 1)];
}

/**
 * The table, row by roughness, NoV along each row; see sheenTablePoint.
 * test/render.test.js integrates it again, and prints the table to paste here
 * when the two disagree.
 */
export const SHEEN_ALBEDO = Float32Array.of(
  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
  1.6027, 0.4504, 0.1248, 0.0308, 0.0065, 0.0011, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
  1.1805, 0.5966, 0.3191, 0.1705, 0.0889, 0.0446, 0.0212, 0.0095, 0.0039, 0.0014, 0.0005, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000,
  0.9514, 0.5853, 0.3816, 0.2525, 0.1665, 0.1084, 0.0691, 0.0429, 0.0256, 0.0147, 0.0080, 0.0040, 0.0018, 0.0007, 0.0002, 0.0000,
  0.8295, 0.5647, 0.4059, 0.2972, 0.2184, 0.1598, 0.1156, 0.0824, 0.0575, 0.0393, 0.0260, 0.0165, 0.0098, 0.0052, 0.0024, 0.0008,
  0.7657, 0.5547, 0.4228, 0.3286, 0.2570, 0.2008, 0.1560, 0.1200, 0.0911, 0.0683, 0.0503, 0.0359, 0.0246, 0.0158, 0.0091, 0.0043,
  0.7372, 0.5568, 0.4411, 0.3564, 0.2902, 0.2365, 0.1922, 0.1551, 0.1241, 0.0986, 0.0774, 0.0595, 0.0444, 0.0317, 0.0211, 0.0123,
  0.7320, 0.5692, 0.4632, 0.3845, 0.3219, 0.2701, 0.2262, 0.1886, 0.1563, 0.1290, 0.1057, 0.0854, 0.0675, 0.0516, 0.0376, 0.0253,
  0.7433, 0.5900, 0.4894, 0.4141, 0.3536, 0.3028, 0.2591, 0.2210, 0.1877, 0.1592, 0.1344, 0.1123, 0.0923, 0.0742, 0.0576, 0.0424,
  0.7658, 0.6168, 0.5188, 0.4451, 0.3854, 0.3349, 0.2911, 0.2525, 0.2183, 0.1888, 0.1628, 0.1393, 0.1179, 0.0980, 0.0796, 0.0623,
  0.7953, 0.6471, 0.5496, 0.4763, 0.4166, 0.3660, 0.3217, 0.2824, 0.2474, 0.2170, 0.1901, 0.1657, 0.1432, 0.1221, 0.1023, 0.0836,
  0.8275, 0.6779, 0.5798, 0.5060, 0.4460, 0.3949, 0.3501, 0.3101, 0.2744, 0.2433, 0.2157, 0.1905, 0.1672, 0.1453, 0.1247, 0.1051,
  0.8575, 0.7057, 0.6066, 0.5323, 0.4718, 0.4202, 0.3749, 0.3345, 0.2982, 0.2666, 0.2385, 0.2129, 0.1891, 0.1668, 0.1457, 0.1256,
  0.8801, 0.7267, 0.6271, 0.5526, 0.4920, 0.4403, 0.3949, 0.3542, 0.3177, 0.2860, 0.2578, 0.2320, 0.2081, 0.1857, 0.1644, 0.1443,
  0.8897, 0.7368, 0.6380, 0.5643, 0.5043, 0.4532, 0.4082, 0.3680, 0.3319, 0.3005, 0.2725, 0.2471, 0.2234, 0.2012, 0.1802, 0.1603,
  0.8806, 0.7319, 0.6360, 0.5646, 0.5065, 0.4570, 0.4135, 0.3744, 0.3393, 0.3089, 0.2818, 0.2571, 0.2342, 0.2127, 0.1923, 0.1730,
);
