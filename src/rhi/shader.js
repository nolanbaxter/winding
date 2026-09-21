// Shader modules.
//
// The one thing this adds over device.createShaderModule: WGSL compilation
// errors DO NOT THROW. createShaderModule always returns a module, and a
// broken one fails later as a confusing pipeline-creation error pointing at
// the wrong line. compile() surfaces the real diagnostic instead.

import { DEBUG } from '../core/assert.js';

let nextShaderId = 1;

/**
 * Errors reported by compileShaderSync, which cannot throw at its call site
 * because the diagnostic arrives after it has already returned the module.
 * The GPU smoke test asserts this is empty; nothing else reads it.
 */
export const shaderErrors = [];

export class Shader {
  constructor(module, label) {
    this.module = module;
    this.label = label;
    /** Stable identity for pipeline cache keys -- GPU objects are not comparable. */
    this.id = nextShaderId++;
  }
}

/**
 * Create a shader module and report compilation problems where they happened.
 *
 * Async because getCompilationInfo() is, and it is awaited at load time --
 * which is exactly when you want to hear about a broken shader.
 */
export async function compileShader(device, code, label = 'shader') {
  const shader = new Shader(device.createShaderModule({ label, code }), label);

  const problem = report(await shader.module.getCompilationInfo(), code, label);
  if (problem) throw new Error(problem);

  return shader;
}

/**
 * The same diagnostics, for call sites that cannot await.
 *
 * Mipmap generation and IBL setup build their pipelines inside synchronous
 * functions well down the image-loading path. Threading async up those chains
 * just to read a diagnostic is a worse trade than reporting it late: the module
 * comes back immediately, and any error lands in shaderErrors and the console
 * the moment the driver has it. What is NOT acceptable is the third option --
 * calling createShaderModule bare, which is how a reserved keyword in this
 * codebase once surfaced as a pipeline error pointing at the wrong thing.
 */
export function compileShaderSync(device, code, label = 'shader') {
  const shader = new Shader(device.createShaderModule({ label, code }), label);

  shader.module.getCompilationInfo().then((info) => {
    const problem = report(info, code, label);
    if (problem) {
      shaderErrors.push(problem);
      console.error(problem);
    }
  });

  return shader;
}

/** Warnings straight to the console, errors as a formatted message, null if clean. */
function report(info, code, label) {
  for (const m of info.messages) {
    if (m.type === 'warning') {
      console.warn(`${label}:${m.lineNum}:${m.linePos} ${m.message}`);
    }
  }

  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length === 0) return null;

  const detail = errors
    .map((m) => `  ${label}:${m.lineNum}:${m.linePos}  ${m.message}\n${sourceLine(code, m.lineNum)}`)
    .join('\n');
  return `WGSL compilation failed:\n${detail}`;
}

function sourceLine(code, lineNum) {
  if (!DEBUG || !lineNum) return '';
  const line = code.split('\n')[lineNum - 1];
  return line === undefined ? '' : `    | ${line}`;
}
