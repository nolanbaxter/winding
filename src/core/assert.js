// Development-only validation. Nothing fails silently.
//
// Every call site guards with `if (DEBUG)`. Build with `--define:DEBUG=false`
// and any minifier removes the dead branch entirely.
//
// Errors that must fire in release too (capacity exhaustion, allocation
// failure) throw directly instead of going through here.

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
