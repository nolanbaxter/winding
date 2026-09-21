// Shared memory allocation.
//
// Parallel work over the engine's SoA columns needs memory both sides can see
// at once, so the columns are SharedArrayBuffers.
//
// SharedArrayBuffer only exists on a cross-origin isolated page (COOP + COEP),
// and when it is missing the constructor is simply absent -- no error, no hint.
// So every allocation goes through here, falls back to a plain ArrayBuffer, and
// the job system checks `sharedMemoryAvailable` to decide whether to run jobs
// in parallel or inline on the calling thread.
//
// The fallback is not a degraded mode with different behaviour. Single-threaded
// execution over a plain buffer produces bit-identical results; it is only
// slower. That matters because it means a bug can never be "only in the
// parallel path" without also being a bug in the ordering.

/**
 * True when SharedArrayBuffer is usable. False on a page without COOP/COEP,
 * which includes any static host that cannot set response headers.
 */
export const sharedMemoryAvailable = (() => {
  if (typeof SharedArrayBuffer === 'undefined') return false;
  // Node has SharedArrayBuffer unconditionally; browsers gate it on isolation.
  if (typeof globalThis.crossOriginIsolated === 'boolean') return globalThis.crossOriginIsolated;
  return true;
})();

export function sharedBuffer(byteLength) {
  return sharedMemoryAvailable
    ? new SharedArrayBuffer(byteLength)
    : new ArrayBuffer(byteLength);
}

export function sharedFloat32Array(length) {
  return new Float32Array(sharedBuffer(length * 4));
}

export function sharedInt32Array(length) {
  return new Int32Array(sharedBuffer(length * 4));
}

export function sharedUint32Array(length) {
  return new Uint32Array(sharedBuffer(length * 4));
}

export function sharedUint8Array(length) {
  return new Uint8Array(sharedBuffer(length));
}

/**
 * Whether this thread is allowed to block in Atomics.wait.
 *
 * The browser's main thread is not: blocking it would freeze the page, so the
 * call throws. Workers may. This is why the job system spins on the main thread
 * instead of waiting, and why it keeps jobs short enough that spinning is
 * cheaper than the alternative anyway.
 */
export const canBlock = typeof window === 'undefined';
