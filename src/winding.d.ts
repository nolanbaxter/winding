// Types for winding-engine. Written by hand from docs/API.md, which says what
// every call does; this says only what it takes and returns. test/docs.test.js
// fails if a public method or getter is missing here.

// The WebGPU objects a few fields hand out. Empty here, they merge with the
// full types from lib.dom or @webgpu/types where a project has them, and stand
// alone where it has not, so these types need neither.
declare global {
  interface GPUDevice {}
  interface GPUTexture {}
  interface GPUTextureView {}
  interface GPUSupportedLimits {}
  interface GPUError {}
}

// ------------------------------------------------------------------- shapes

/** Any array of numbers works as input; outputs are Float32Arrays. */
export type Vec2 = ArrayLike<number>;
/** `[x, y, z]`, or `[x, y]` where a call says two numbers mean 2D. */
export type Vec3 = ArrayLike<number>;
export type Vec4 = ArrayLike<number>;
/** `[x, y, z, w]`. */
export type Quat = ArrayLike<number>;
/** A column-major 4x4 matrix. */
export type Mat4 = ArrayLike<number>;
/** `[r, g, b, a]`, 0 to 1: linear light in 3D, sRGB in 2D. `[r, g, b]` for a light. */
export type Color = ArrayLike<number>;
/** `[u0, v0, u1, v1]`. */
export type Rect = ArrayLike<number>;
export type Fetch = typeof globalThis.fetch;

/** What `addX` took, less where it goes: `setX` changes the rest. */
export type Changes<T> = Partial<Omit<T, 'position' | 'parent'>>;

export type Blend = 'alpha' | 'additive' | 'multiply' | 'screen';
export type Facing = 'camera' | 'upright' | 'plane';

// ------------------------------------------------------------------ assets

/** From `engine.load`. Pass the whole object to `scene.add`. */
export interface Model {
  readonly nodes: readonly unknown[];
  readonly roots: readonly number[];
  readonly meshes: readonly unknown[];
  readonly materials: readonly unknown[];
  readonly materialIds: readonly number[];
  readonly animations: readonly unknown[];
  readonly skins: readonly unknown[];
  readonly lights: readonly unknown[];
  readonly cameras: readonly unknown[];
  readonly textures: readonly unknown[];
  readonly source: string | null;
  readonly engine: Winding;
}

/** From `engine.loadTexture`; `engine.createTarget` returns one too. */
export interface Texture {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly width: number;
  readonly height: number;
  readonly pixelated: boolean;
}

/** From `engine.createTarget`: a texture to draw into. */
export interface Target extends Texture {}

/** From `engine.loadFont`. */
export interface Font {}

/** From `engine.loadLUT`. */
export interface LUT {
  readonly size: number;
  readonly data: Float32Array;
  readonly domainMin: Float32Array | number[];
  readonly domainMax: Float32Array | number[];
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
}

/** From `engine.loadSplats`. */
export interface Splats {
  readonly count: number;
  /** The degree of its spherical harmonics, 0 to 3: 0 is one colour from every side. */
  readonly degree: number;
  readonly min: Float32Array;
  readonly max: Float32Array;
}

/** Linear RGB floats, 3 a pixel, top row first. */
export interface HDRMap {
  width: number;
  height: number;
  data: Float32Array;
}

// ------------------------------------------------------------------ engine

export interface ShadowOptions {
  size?: number;
  cascades?: 1 | 2 | 3 | 4;
  lambda?: number;
  casterExtent?: number;
  normalBias?: number;
  depthBiasSlope?: number;
  depthBiasConstant?: number;
  localSize?: number;
}

export interface PostOptions {
  threshold?: number;
  knee?: number;
  filterRadius?: number;
  strength?: number;
  levels?: number;
  antialias?: boolean;
  grading?: Grading | null;
}

export interface DeviceLost {
  reason: string;
  message: string;
  recoverable: boolean;
  action: 'reload';
}

/** Auto exposure's settings. Ranges in stops, speeds in stops a second. */
export interface AutoExposure {
  min?: number;
  max?: number;
  brighten?: number;
  darken?: number;
}

export interface WindingOptions {
  label?: string;
  powerPreference?: 'high-performance' | 'low-power';
  onDeviceLost?: ((detail: DeviceLost) => void) | null;
  onError?: ((error: GPUError) => void) | null;
  exposure?: number;
  autoExposure?: AutoExposure | boolean | null;
  antialias?: boolean;
  grading?: Grading | null;
  post?: PostOptions;
  shadows?: ShadowOptions;
  shadowDistance?: number | null;
  lightDistance?: number | null;
  ao?: boolean | { radius: number | null };
  oit?: boolean;
  taa?: boolean;
  fog?: Fog | null;
  dof?: DepthOfField | null;
  /** Settings for the default environment, not an `Environment`. */
  environment?: EnvironmentOptions;
  maxDraws?: number;
  gpuTiming?: boolean;
  workerCount?: number;
  onDemand?: boolean;
}

export interface Grading {
  whiteBalance?: number;
  contrast?: number;
  saturation?: number;
  lut?: LUT;
}

export interface Fog {
  visibility: number;
  height?: number;
  scaleHeight?: number;
  albedo?: Vec3;
}

export interface DepthOfField {
  focusDistance: number;
  fStop: number;
  sensorHeight?: number;
}

export interface Hud {
  scene: Scene;
  camera?: Camera2D;
}

export interface Clock {
  fixedDt: number;
  readonly elapsed: number;
  readonly realDelta: number;
  readonly alpha: number;
  readonly frame: number;
}

export interface RunOptions {
  update?: (dt: number, elapsed: number) => void;
  frame?: (alpha: number, clock: Clock) => void;
  hud?: Hud | null;
}

export interface Stats {
  readonly renderables: number;
  readonly draws: number;
  readonly recomposed: number;
  readonly transparent: number;
  readonly transparentDraws: number;
  readonly shadowViews?: number;
  readonly shadowViewsDrawn?: number;
  readonly cascadesDrawn?: number;
  readonly sprites2D?: number;
  readonly sprites2DWritten?: number;
  readonly tiles2DWritten?: number;
  readonly emitters?: number;
  readonly hudSprites?: number;
  readonly hudSpritesWritten?: number;
  readonly splats?: number;
}

export interface Device {
  /** The raw WebGPU device. */
  readonly device: GPUDevice;
  readonly limits: GPUSupportedLimits & { readonly [limit: string]: number };
  /** The canvas size in pixels. */
  readonly width: number;
  readonly height: number;
  readonly pixelRatio: number;
  readPixels(region?: { x?: number; y?: number; width?: number; height?: number }): Promise<Uint8Array>;
}

export interface ShadowMaps {
  lambda: number;
  casterExtent: number;
  normalBias: number;
  readonly size: number;
  readonly cascades: number;
  readonly localSize: number;
}

export interface Post {
  threshold: number;
  knee: number;
  filterRadius: number;
  strength: number;
  levels: number;
  readonly levelsDrawn: number;
  antialias: boolean;
  grading: Grading | null;
}

export interface Renderer {
  exposure: number;
  autoExposure: AutoExposure | boolean | null;
  resolution: number;
  fog: Fog | null;
  dof: DepthOfField | null;
  ao: { radius: number | null } | null;
  oit: boolean;
  softShadows: boolean;
  taa: boolean;
  readonly shadows: ShadowMaps;
  skybox: boolean;
  shadowDistance: number | null;
  lightDistance: number | null;
  readonly post: Post;
  /** CPU milliseconds per phase of the last frame. */
  readonly timing: { transforms: number; upload: number; graph: number; encode: number; total: number };
  readonly gpuTiming: { enabled: boolean; readonly supported: boolean };
}

export interface DebugLines {
  depthTest: boolean;
  line(from: Vec3, to: Vec3, color?: Color): DebugLines;
  box(min: Vec3, max: Vec3, color?: Color): DebugLines;
  sphere(center: Vec3, radius: number, color?: Color): DebugLines;
  circle(center: Vec3, radius: number, color?: Color): DebugLines;
  axes(origin: Vec3, size?: number): DebugLines;
}

export interface SceneOptions {
  capacity?: number;
  renderableCapacity?: number;
  lightCapacity?: number;
}

export interface CreateSceneOptions extends SceneOptions {
  environment?: Environment;
}

export class Winding {
  static create(canvas: HTMLCanvasElement, options?: WindingOptions): Promise<Winding>;
  private constructor();

  readonly gpu: Device;
  readonly renderer: Renderer;
  readonly environment: Environment;
  readonly clock: Clock;
  readonly fps: number;
  readonly skippedFrames: number;
  onDemand: boolean;
  get stats(): Stats;
  get debug(): DebugLines;
  get grading(): Grading | null;
  set grading(value: Grading | null | undefined);
  /** @deprecated Now `engine.gpu`; reading it throws. */
  get rhi(): never;
  set rhi(value: never);

  createScene(options?: CreateSceneOptions): Scene;
  load(source: string | ArrayBuffer | Uint8Array, options?: { retainGeometry?: boolean; baseURL?: string; fetch?: Fetch }): Promise<Model>;
  loadTexture(source: string | ImageBitmapSource, options?: { srgb?: boolean; pixelated?: boolean; label?: string; fetch?: Fetch }): Promise<Texture>;
  loadFont(css: string): Promise<Font>;
  loadLUT(source: string, options?: { fetch?: Fetch }): Promise<LUT>;
  loadEnvironment(
    source: string | ArrayBuffer | Uint8Array,
    options?: { fetch?: Fetch; size?: number; irradianceSize?: number; prefilterMips?: number; label?: string },
  ): Promise<Environment>;
  loadSplats(source: string | ArrayBuffer | Uint8Array, options?: { fetch?: Fetch }): Promise<Splats>;
  unload(asset: Model | Texture | Font | LUT | Environment | Splats | Target): void;
  run(scene: Scene, camera: Camera | Camera2D, options?: RunOptions): void;
  stop(): void;
  invalidate(): void;
  renderFrame(scene: Scene, camera: Camera | Camera2D, options?: { hud?: Hud | null; target?: Target | null }): void;
  createTarget(options: { size: Vec2; pixelated?: boolean; label?: string }): Promise<Target>;
  captureProbes(scene: Scene, probeNodes?: Node[]): Promise<void>;
  destroy(): void;
}

// ------------------------------------------------------------------- scene

export interface Hit {
  node: Node;
  renderable: number;
  distance: number;
}

export interface Hit2D {
  node: Node;
  point: [number, number];
  tile?: [number, number];
}

export interface Placed {
  position?: Vec3;
  parent?: Node | null;
}

export interface LightOptions extends Placed {
  type?: 'point' | 'spot' | 'directional';
  direction?: Vec3 | null;
  color?: Vec3;
  intensity?: number;
  radius?: number;
  innerAngle?: number;
  outerAngle?: number;
  castShadow?: boolean;
  /** The light's own size, for soft shadows: a radius for a point or spot, an angle across for a directional. */
  size?: number;
}

export interface Light {
  type: 'point' | 'spot' | 'directional';
  color: number[];
  intensity: number;
  castShadow: boolean;
  size: number;
  radius?: number;
  innerAngle?: number;
  outerAngle?: number;
}

export interface EmitterOptions extends Placed {
  rate?: number;
  lifetime: number | [number, number];
  size: number;
  sizeEnd?: number;
  speed?: number | [number, number];
  direction?: Vec3;
  spread?: number;
  radius?: number;
  acceleration?: Vec3;
  drag?: number;
  color?: Color;
  colorEnd?: Color;
  texture?: Texture | null;
  blend?: 'additive' | 'alpha';
  layer?: number;
}

export interface DecalOptions extends Placed {
  texture: Texture;
  size: Vec3;
  color?: Color;
}

export interface ProbeOptions extends Placed {
  size: Vec3;
  fade?: number;
}

export interface SpriteAnimation {
  frames: Rect[];
  fps?: number;
  loop?: boolean;
}

export interface SpriteOptions extends Placed {
  texture: Texture;
  size?: Vec2;
  pixels?: boolean;
  color?: Color;
  rect?: Rect;
  pivot?: Vec2;
  angle?: number;
  facing?: Facing;
  blend?: Blend | 'cutout';
  cutoff?: number;
  layer?: number;
  animation?: SpriteAnimation | null;
  lit?: boolean;
}

export interface Sprite extends Required<Omit<SpriteOptions, 'position' | 'parent' | 'animation'>> {
  animation: Required<SpriteAnimation> | null;
  /** The frame shown. */
  frame: number;
  /** Seconds the animation has run. */
  time: number;
}

export interface TextOptions extends Placed {
  font: Font;
  size: number;
  text?: unknown;
  color?: Color;
  align?: 'left' | 'center' | 'right';
  pivot?: Vec2;
  lineHeight?: number;
  width?: number;
  facing?: Facing;
  pixels?: boolean;
  layer?: number;
  lit?: boolean;
  blend?: Blend;
  stroke?: Color;
  strokeWidth?: number;
}

export interface ShapeOptions extends Placed {
  shape?: 'rect' | 'ellipse';
  size: Vec2;
  radius?: number;
  color?: Color;
  stroke?: Color;
  strokeWidth?: number;
  pivot?: Vec2;
  layer?: number;
  blend?: Blend;
  lit?: boolean;
}

export interface PathOptions extends Placed {
  points: Vec2[];
  closed?: boolean;
  color?: Color;
  stroke?: Color;
  strokeWidth?: number;
  layer?: number;
  blend?: Blend;
  lit?: boolean;
}

export interface TilemapOptions extends Placed {
  tileset: Texture;
  tileSize: Vec2;
  columns: number;
  rows: number;
  tiles?: ArrayLike<number>;
  margin?: number;
  spacing?: number;
  layer?: number;
  color?: Color;
  pivot?: Vec2;
  blend?: Blend;
  lit?: boolean;
}

export interface FrameOptions {
  margin?: number;
  aspect?: number;
}

export class Scene {
  constructor(options?: SceneOptions);

  get environment(): Environment;
  set environment(value: Environment);
  /** Cameras from added models, in the order added. */
  readonly cameras: Camera[];
  get particlesActive(): boolean;

  createNode(options?: { parent?: Node | null }): Node;
  node(entity: number): Node;
  childrenOf(node: Node): Node[];
  add(asset: Model, options?: { parent?: Node | null }): Node;
  remove(node: Node): void;
  update(): number;
  advance(dt: number): void;
  bounds(outMin: Vec3, outMax: Vec3): boolean;
  frame(camera: Camera, options?: FrameOptions): boolean;
  raycast(origin: Vec3, direction: Vec3, options?: { maxDistance?: number }): Hit | null;
  pick(camera: Camera, x: number, y: number, width: number, height: number, options?: { maxDistance?: number }): Hit | null;
  pick(camera: Camera2D, x: number, y: number, width: number, height: number): Hit2D | null;

  addLight(options?: LightOptions): Node;
  setLight(node: Node, changes: Pick<LightOptions, 'color' | 'intensity' | 'castShadow' | 'size' | 'radius' | 'innerAngle' | 'outerAngle'>): void;
  lightOf(node: Node): Light | null;

  addEmitter(options: EmitterOptions): Node;
  setEmitter(node: Node, changes: Changes<EmitterOptions>): void;
  emitterOf(node: Node): Omit<EmitterOptions, 'position' | 'parent'> | null;
  burst(node: Node, count: number): void;

  addSplats(options: { splats: Splats } & Placed): Node;
  splatsOf(node: Node): { splats: Splats } | null;

  addDecal(options: DecalOptions): Node;
  setDecal(node: Node, changes: Changes<DecalOptions>): void;
  decalOf(node: Node): { texture: Texture; size: number[]; color: number[] } | null;

  addProbe(options: ProbeOptions): Node;
  setProbe(node: Node, changes: Changes<ProbeOptions>): void;
  probeOf(node: Node): { size: number[]; fade: number; captured: boolean } | null;

  addSprite(options: SpriteOptions): Node;
  setSprite(node: Node, changes: Changes<SpriteOptions>): void;
  spriteOf(node: Node): Sprite | null;

  addText(options: TextOptions): Node;
  setText(node: Node, changes: Changes<TextOptions>): void;
  textOf(node: Node): Omit<TextOptions, 'position' | 'parent'> | null;

  addShape(options: ShapeOptions): Node;
  setShape(node: Node, changes: Changes<ShapeOptions>): void;
  shapeOf(node: Node): Required<Omit<ShapeOptions, 'position' | 'parent'>> | null;

  addPath(options: PathOptions): Node;
  setPath(node: Node, changes: Changes<PathOptions>): void;
  pathOf(node: Node): Required<Omit<PathOptions, 'position' | 'parent'>> & { points: [number, number][] } | null;

  addTilemap(options: TilemapOptions): Node;
  setTilemap(node: Node, changes: Changes<TilemapOptions>): void;
  tilemapOf(node: Node): Required<Omit<TilemapOptions, 'position' | 'parent' | 'blend'>> & { tiles: Uint32Array } | null;
  setTile(node: Node, x: number, y: number, id: number): void;
  setTiles(node: Node, x: number, y: number, width: number, tiles: ArrayLike<number>): void;
  tileAt(node: Node, x: number, y: number): number;
}

// -------------------------------------------------------------------- node

export interface PlayOptions {
  loop?: boolean;
  speed?: number;
  time?: number;
  fade?: number;
  layer?: string;
  weight?: number;
  join?: boolean;
  sync?: boolean;
}

export interface StopOptions {
  layer?: string;
  fade?: number;
}

export interface AnimationPlayer {
  readonly names: string[];
  play(nameOrIndex: string | number, options?: PlayOptions): boolean;
  stop(options?: StopOptions): void;
  layer(name: string, options?: { mask?: string | string[] | null; weight?: number; additive?: boolean }): void;
  setWeight(nameOrIndex: string | number, weight: number, options?: { layer?: string; fade?: number }): boolean;
  rootMotion(options: { node?: string | number; vertical?: boolean; apply?: boolean } | null): void;
  readonly motion: { position: Float32Array; yaw: number };
  readonly clip: unknown;
  time: number;
  speed: number;
  loop: boolean;
  readonly finished: boolean;
}

export class Node {
  private constructor();
  readonly scene: Scene;
  readonly entity: number;
  get alive(): boolean;
  get animation(): AnimationPlayer | null;
  get animations(): string[];
  get weights(): Float32Array | null;

  setPosition(x: number, y: number, z?: number): this;
  setAngle(radians: number): this;
  setScale(x: number, y?: number, z?: number): this;
  setRotation(q: Quat): this;
  setDirection(x: number, y: number, z?: number): this;
  setAxisAngle(axis: Vec3, radians: number): this;
  setEuler(yaw: number, pitch: number, roll?: number): this;
  setParent(node: Node | null): this;
  worldPosition<T extends Vec3>(out: T): T;
  children(): Node[];
  play(nameOrIndex: string | number, options?: PlayOptions): this;
  stop(options?: StopOptions): this;
  destroy(): void;
}

// ----------------------------------------------------------------- cameras

export class Camera {
  constructor(options?: { fovY?: number; near?: number; orthographic?: boolean; far?: number });
  readonly position: Float32Array;
  readonly target: Float32Array;
  readonly up: Float32Array;
  fovY: number;
  near: number;
  orthographic: boolean;
  far: number;
  readonly aspect: number;
  readonly following: Node | null;
  readonly view: Float32Array;
  readonly projection: Float32Array;
  readonly viewProjection: Float32Array;
  readonly inverseProjection: Float32Array;

  follow(node: Node | null): this;
  frameBounds(min: Vec3, max: Vec3, options?: FrameOptions): this;
  orthographicHalfHeight(): number;
  update(aspect: number): this;
  rayFromScreen<T extends Vec3>(x: number, y: number, width: number, height: number, outOrigin: Vec3, outDirection: T): T;
}

export class Camera2D {
  constructor(options?: {
    position?: Vec2;
    pivot?: Vec2;
    zoom?: number;
    angle?: number;
    background?: Color;
    pixelSnap?: boolean;
    ambient?: Vec3;
  });
  readonly position: Float32Array;
  readonly pivot: Float32Array;
  zoom: number;
  angle: number;
  readonly background: Float32Array;
  pixelSnap: boolean;
  readonly ambient: Float32Array;
  readonly view: Float32Array;
  readonly projection: Float32Array;
  readonly viewProjection: Float32Array;
  readonly width: number;
  readonly height: number;
  readonly pixelRatio: number;
  get is2D(): true;
  /** @deprecated Now `camera.angle`; reading it throws. */
  get rotation(): never;
  set rotation(value: never);

  update(aspect?: number, width?: number, height?: number, pixelRatio?: number): this;
  screenToWorld<T extends Vec2 = Float32Array>(x: number, y: number, out?: T): T;
  worldToScreen<T extends Vec2 = Float32Array>(x: number, y: number, out?: T): T;
}

// ------------------------------------------------------------------ helpers

export interface OrbitPose {
  distance: number;
  yaw: number;
  pitch: number;
  target: Float32Array;
}

export interface OrbitOptions {
  distance?: number;
  yaw?: number;
  pitch?: number;
  target?: Vec3;
  minDistance?: number;
  maxDistance?: number;
  rotateSpeed?: number;
  zoomSpeed?: number;
  panSpeed?: number;
  damping?: number;
}

export class OrbitController {
  constructor(camera: Camera, element: HTMLElement, options?: OrbitOptions);
  distance: number;
  yaw: number;
  pitch: number;
  target: Float32Array;
  minDistance: number;
  maxDistance: number;
  rotateSpeed: number;
  zoomSpeed: number;
  panSpeed: number;
  damping: number;
  /** Where the camera is easing to. Write it to move the camera from code. */
  desired: OrbitPose;
  readonly dragged: boolean;

  update(dt: number): this;
  syncFromCamera(): this;
  frameBounds(min: Vec3, max: Vec3, options?: { margin?: number }): this;
  detach(): void;
}

export class StatsOverlay {
  constructor(engine: Winding, options?: { interval?: number; parent?: HTMLElement });
  readonly element: HTMLDivElement;
  update(dt: number): void;
  destroy(): void;
}

export function spriteSheet(options: { columns: number; rows?: number; count?: number; first?: number }): number[][];

// -------------------------------------------------------------- environment

export interface Sky {
  ground?: Vec3;
  horizon?: Vec3;
  zenith?: Vec3;
  sun?: Vec3;
  sunColor?: Vec3;
  sunIntensity?: number;
  glow?: number;
}

export interface EnvironmentOptions {
  size?: number;
  irradianceSize?: number;
  prefilterMips?: number;
  label?: string;
  sky?: Sky;
  map?: HDRMap | null;
}

export class Environment {
  constructor(gpu: Device, options?: EnvironmentOptions);
  readonly size: number;
  readonly prefilterMips: number;
  readonly sky: Required<Sky>;
  destroy(): void;
}

export function parseHDR(bytes: Uint8Array | ArrayBuffer, options?: { maxDimension?: number }): HDRMap;

// ------------------------------------------------------------------- colour

export function srgbToLinear(c: number): number;
export function linearToSrgb(c: number): number;
export function colorFromHex(hex: string): [number, number, number, number];
export function colorFromBytes(r: number, g: number, b: number, a?: number): [number, number, number, number];

// --------------------------------------------------------------------- math

export function vec3Create(x?: number, y?: number, z?: number): Float32Array;
export function vec3Set<T extends Vec3>(out: T, x: number, y: number, z: number): T;
export function vec3Copy<T extends Vec3>(out: T, a: Vec3): T;
export function vec3Add<T extends Vec3>(out: T, a: Vec3, b: Vec3): T;
export function vec3Sub<T extends Vec3>(out: T, a: Vec3, b: Vec3): T;
export function vec3Mul<T extends Vec3>(out: T, a: Vec3, b: Vec3): T;
export function vec3Scale<T extends Vec3>(out: T, a: Vec3, s: number): T;
export function vec3ScaleAndAdd<T extends Vec3>(out: T, a: Vec3, b: Vec3, s: number): T;
export function vec3Negate<T extends Vec3>(out: T, a: Vec3): T;
export function vec3Dot(a: Vec3, b: Vec3): number;
export function vec3Cross<T extends Vec3>(out: T, a: Vec3, b: Vec3): T;
export function vec3LengthSq(a: Vec3): number;
export function hypot3(x: number, y: number, z: number): number;
export function vec3Length(a: Vec3): number;
export function vec3DistanceSq(a: Vec3, b: Vec3): number;
export function vec3Normalize<T extends Vec3>(out: T, a: Vec3): T;
export function vec3Lerp<T extends Vec3>(out: T, a: Vec3, b: Vec3, t: number): T;
export function vec3Min<T extends Vec3>(out: T, a: Vec3, b: Vec3): T;
export function vec3Max<T extends Vec3>(out: T, a: Vec3, b: Vec3): T;
export function vec3TransformMat4<T extends Vec3>(out: T, a: Vec3, m: Mat4): T;
export function vec3TransformMat4Dir<T extends Vec3>(out: T, a: Vec3, m: Mat4): T;
export function vec3TransformQuat<T extends Vec3>(out: T, a: Vec3, q: Quat): T;

export function quatCreate(): Float32Array;
export function quatIdentity<T extends Quat>(out: T): T;
export function quatCopy<T extends Quat>(out: T, a: Quat): T;
export function quatSetAxisAngle<T extends Quat>(out: T, axis: Vec3, rad: number): T;
export function quatMultiply<T extends Quat>(out: T, a: Quat, b: Quat): T;
export function quatDot(a: Quat, b: Quat): number;
export function quatConjugate<T extends Quat>(out: T, a: Quat): T;
export function quatNormalize<T extends Quat>(out: T, a: Quat): T;
export function quatFromEuler<T extends Quat>(out: T, yaw: number, pitch: number, roll: number): T;
export function quatFromMat4<T extends Quat>(out: T, m: Mat4, mOff?: number): T;
export function quatSlerp<T extends Quat>(out: T, a: Quat, b: Quat, t: number): T;
export function quatFromTo<T extends Quat>(out: T, from: Vec3, to: Vec3): T;
export function quatLookAlong<T extends Quat>(out: T, direction: Vec3, up?: Vec3): T;

export function mat4Create(): Float32Array;
export function mat4Identity<T extends Mat4>(out: T): T;
export function mat4Copy<T extends Mat4>(out: T, a: Mat4, outOff?: number, aOff?: number): T;
export function mat4GetTranslation<T extends Vec3>(out: T, m: Mat4): T;
export function mat4Multiply<T extends Mat4>(out: T, a: Mat4, b: Mat4, outOff?: number, aOff?: number, bOff?: number): T;
export function mat4MultiplyAffine<T extends Mat4>(out: T, a: Mat4, b: Mat4, outOff?: number, aOff?: number, bOff?: number): T;
export function mat4FromQuatPosScale<T extends Mat4>(
  out: T, q: Quat, pos: Vec3, scale: Vec3, outOff?: number, qOff?: number, posOff?: number, scaleOff?: number,
): T;
export function mat4Invert<T extends Mat4>(out: T, a: Mat4): T | null;
export function mat4LookAt<T extends Mat4>(out: T, eye: Vec3, center: Vec3, up: Vec3): T;
export function mat4NormalMatrix(out: Mat4, m: Mat4, outOff?: number, mOff?: number): boolean;
export function mat4Decompose(outPos: Vec3, outRot: Quat, outScale: Vec3, m: Mat4, mOff?: number): boolean;
export function mat4OrthographicReverseZ<T extends Mat4>(
  out: T, left: number, right: number, bottom: number, top: number, near: number, far: number,
): T;
export function mat4PerspectiveReverseZInfinite<T extends Mat4>(out: T, fovYRadians: number, aspect: number, near: number): T;

export function aabbTransform(
  outMin: Vec3, outMax: Vec3, min: Vec3, max: Vec3, m: Mat4, mOff?: number, outOff?: number, inOff?: number,
): void;
export function aabbBoundingSphere(outCenter: Vec3, min: Vec3, max: Vec3): number;
export function aabbUnion(min: Vec3, max: Vec3, otherMin: Vec3, otherMax: Vec3): void;
export function aabbSetEmpty(min: Vec3, max: Vec3): void;
export function aabbRayDistance(min: Vec3, max: Vec3, origin: Vec3, direction: Vec3, boundsOff?: number): number;
export function rayTriangleDistance(origin: Vec3, direction: Vec3, positions: ArrayLike<number>, a: number, b: number, c: number): number;
