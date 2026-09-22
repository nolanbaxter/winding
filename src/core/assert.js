// Development-only validation. Nothing fails silently.
//
// Almost every call site guards with `if (DEBUG)`. Build with
// `--define:DEBUG=false` and any minifier removes the dead branch entirely.
//
// Two things do NOT go through that guard. Errors that must fire in release --
// capacity exhaustion, allocation failure, a camera near of zero -- throw
// directly instead of coming here at all. And a trust boundary, where the
// input came from outside the engine, calls assertFinite unguarded on purpose:
// Scene.raycast does this, because a NaN ray is not merely wrong, it reports a
// hit at distance zero on whatever it looks at first.

export const DEBUG = true;

export function assert(cond, msg) {
  if (!cond) throw new Error(msg ?? 'assertion failed');
}

/**
 * Catch NaN or Infinity where it appears, not ten systems downstream.
 *
 * One NaN in a position propagates through composition into every child, the
 * bounds and the cull, and surfaces as geometry that is simply not there, with
 * nothing pointing back at the source.
 *
 * Offset and length match the mat4 idiom, so a caller can check one slice of a
 * shared column without allocating a view.
 */
export function assertFinite(array, what, offset = 0, length = array.length - offset) {
  for (let i = 0; i < length; i++) {
    if (!Number.isFinite(array[offset + i])) {
      throw new Error(`${what}: non-finite at [${offset + i}] (${array[offset + i]})`);
    }
  }
}
