// sRGB to linear, because every colour this engine takes is LINEAR and every
// colour a human picks is not.
//
// This exists for one mistake, and it is not a small one. Everything in the
// renderer works in linear light -- base colour, emissive, light colour, the
// sky -- and the only encode is the -srgb swap-chain view at the very end.
// A colour picker, a hex code, a CSS value and the number in a design file are
// all sRGB. Typing one into `baseColorFactor` is not slightly off: sRGB 0.35
// is linear 0.10, so a colour arrives roughly three times too bright and
// washed toward white, with nothing reported anywhere.
//
// I made exactly that error writing an example, picked `[0.85, 0.35, 0.25]`
// by eye for red, and got salmon. The numbers looked dim written down and
// encode to rgb(237, 160, 137) on screen. That is the whole reason this file
// is here rather than in a README.
//
// ALPHA IS NOT CONVERTED, which is the part a hand-rolled version gets wrong.
// The sRGB transfer function applies to colour channels only; alpha is already
// linear in both spaces, and putting it through the curve makes every
// transparency subtly wrong in a way that looks like a blending bug.
//
// LIGHT COLOURS ARE NOT SURFACE COLOURS. A surface reflects at most all of the
// light it receives, so its colour belongs in [0,1]. A light EMITS, and its
// colour is radiance that routinely exceeds 1 -- the default sun is
// [3.2, 3.0, 2.7]. Converting #ffffff gives [1,1,1], which is a correct white
// and a dim sun. Multiply it by the brightness you want.

/** One channel, sRGB to linear. The definition everything else is built on. */
export function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** One channel, linear back to sRGB. What the swap chain does on write. */
export function linearToSrgb(c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}

/**
 * A hex colour as linear RGBA, ready for baseColorFactor or a light.
 *
 *     colorFromHex('#e03a2f')     // [0.745, 0.042, 0.028, 1]
 *     colorFromHex('#e03a2f80')   // ... with alpha 0.502, unconverted
 *
 * Accepts 3, 4, 6 and 8 digits, with or without the leading #, because those
 * are the forms that get pasted. Anything else throws rather than resolving to
 * black -- a silently black material is the failure this file exists to stop,
 * and it would be perverse to introduce a second way to reach it.
 */
export function colorFromHex(hex) {
  const digits = String(hex).replace(/^#/, '');
  if (!/^[0-9a-fA-F]+$/.test(digits) || ![3, 4, 6, 8].includes(digits.length)) {
    throw new Error(`colorFromHex: "${hex}" is not a 3, 4, 6 or 8 digit hex colour`);
  }

  // #rgb and #rgba are shorthand for doubled digits: #e32 is #ee3322.
  const wide = digits.length <= 4
    ? digits.split('').map((d) => d + d).join('')
    : digits;

  const byte = (i) => parseInt(wide.slice(i * 2, i * 2 + 2), 16);
  return colorFromBytes(byte(0), byte(1), byte(2), wide.length === 8 ? byte(3) : 255);
}

/**
 * 0-255 channels as linear RGBA. The form a colour picker hands you.
 *
 * A plain array rather than a Float32Array on purpose. This is a value
 * computed once and handed somewhere, not a column that gets mutated -- and a
 * plain array survives JSON.stringify, which a Float32Array does not. That
 * matters: writing one straight into a glTF's `baseColorFactor` is the most
 * likely thing anyone does with it.
 */
export function colorFromBytes(r, g, b, a = 255) {
  return [
    srgbToLinear(r / 255),
    srgbToLinear(g / 255),
    srgbToLinear(b / 255),
    a / 255,            // linear already; see the header
  ];
}
