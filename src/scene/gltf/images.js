// glTF image decoding.
//
// Kept separate from parse.js on purpose: parse.js is pure data and runs under
// Node with no GPU and no DOM, which is what makes it testable. Decoding needs
// createImageBitmap, so it lives here and is called only by code that already
// assumes a browser.
//
// Nothing here touches the GPU either -- uploading is render/textures.js. The
// split means image decode can move to a worker later without the uploader
// knowing.

import { checkViewInBuffer } from './accessor.js';

/**
 * Two flags on createImageBitmap that are wrong by default for textures.
 *
 * colorSpaceConversion: 'none' -- the browser will otherwise convert the image
 * into the display's colour space, which silently alters every pixel of a
 * normal map or a roughness map. Texture bytes must arrive exactly as authored;
 * whether they are sRGB is decided by the texture FORMAT, not by the decoder.
 *
 * premultiplyAlpha: 'none' -- glTF stores straight alpha. Premultiplying darkens
 * the colour of every partially transparent texel, which shows up as a dark
 * fringe around cutouts.
 */
const BITMAP_OPTIONS = { colorSpaceConversion: 'none', premultiplyAlpha: 'none' };

/**
 * Decode every image a material samples.
 *
 * Returns an array parallel to json.images, with null where an image could not
 * be decoded. A broken image is not fatal -- the material falls back to its
 * factor, which is a far better outcome than refusing to show the model.
 */
export async function decodeImages(
  json, buffers, { baseURL, fetchImpl = globalThis.fetch, maxDimension = Infinity } = {},
) {
  const images = json.images ?? [];
  const used = usedImages(json);

  return Promise.all(images.map(async (image, index) => {
    // Nothing samples it, so decoding it only costs memory -- and a file can
    // declare as many images as it likes.
    if (!used.has(index)) return null;
    try {
      const blob = await imageBlob(image, json, buffers, { baseURL, fetchImpl, index });
      // Sized from the header, before decoding. createImageBitmap decodes the
      // whole image first, so a 30000-pixel PNG of a few kilobytes cost gigabytes
      // before the texture's own size check ever ran.
      const size = imageSize(new Uint8Array(await blob.arrayBuffer()));
      if (size === null) throw new Error('is not PNG, JPEG or WebP');
      if (size.width > maxDimension || size.height > maxDimension) {
        throw new Error(`is ${size.width}x${size.height}, past this device's ${maxDimension}`);
      }
      return await createImageBitmap(blob, BITMAP_OPTIONS);
    } catch (error) {
      console.warn(`glTF: image ${index} (${image.name ?? image.uri ?? 'embedded'}) failed: ${error.message}`);
      return null;
    }
  }));
}

async function imageBlob(image, json, buffers, { baseURL, fetchImpl, index }) {
  if (image.bufferView !== undefined) {
    // Embedded in the BIN chunk, which is how .glb ships textures.
    const view = json.bufferViews?.[image.bufferView];
    if (!view) throw new Error(`bufferView ${image.bufferView} does not exist`);

    const buffer = buffers[view.buffer];
    if (!buffer) throw new Error(`buffer ${view.buffer} was not resolved`);
    checkViewInBuffer(view, buffer, image.bufferView);
    const start = buffer.byteOffset + (view.byteOffset ?? 0);
    const bytes = new Uint8Array(buffer.buffer, start, view.byteLength);

    if (!image.mimeType) throw new Error('embedded image has no mimeType');
    return new Blob([bytes], { type: image.mimeType });
  }

  if (image.uri === undefined) throw new Error(`image ${index} has neither uri nor bufferView`);

  // fetch handles data: URIs natively, so base64 needs no special case here.
  if (image.uri.startsWith('data:')) return (await fetchImpl(image.uri)).blob();

  if (!baseURL) throw new Error(`external image "${image.uri}" needs a baseURL to resolve against`);
  const response = await fetchImpl(new URL(image.uri, baseURL));
  if (!response.ok) throw new Error(`fetch returned ${response.status}`);
  return response.blob();
}

// ------------------------------------------------------------- pure helpers
// Exported separately because they are the error-prone part and, unlike the
// decode above, they can be tested without a browser.

const GLTF_NEAREST = 9728;
const GLTF_LINEAR = 9729;
const GLTF_NEAREST_MIPMAP_NEAREST = 9984;
const GLTF_LINEAR_MIPMAP_NEAREST = 9985;
const GLTF_NEAREST_MIPMAP_LINEAR = 9986;
const GLTF_LINEAR_MIPMAP_LINEAR = 9987;

const GLTF_CLAMP_TO_EDGE = 33071;
const GLTF_MIRRORED_REPEAT = 33648;

/**
 * Every texture a material can sample: its slot's name, where the reference
 * sits in the material's JSON, and whether the image holds colour. The core
 * five first, in the order their UV sets and transforms are kept, then the
 * material extensions'. One list, so the decoder, the uploader, the importer
 * and the animation pointers cannot disagree about what a material samples.
 *
 * The sRGB column is the extension specs': colours are sRGB, numbers are not.
 */
export const MATERIAL_TEXTURES = [
  { slot: 'baseColor', path: 'pbrMetallicRoughness/baseColorTexture', srgb: true },
  { slot: 'metallicRoughness', path: 'pbrMetallicRoughness/metallicRoughnessTexture', srgb: false },
  { slot: 'normal', path: 'normalTexture', srgb: false },
  { slot: 'occlusion', path: 'occlusionTexture', srgb: false },
  { slot: 'emissive', path: 'emissiveTexture', srgb: true },
  { slot: 'specular', path: 'extensions/KHR_materials_specular/specularTexture', srgb: false },
  { slot: 'specularColor', path: 'extensions/KHR_materials_specular/specularColorTexture', srgb: true },
  { slot: 'clearcoat', path: 'extensions/KHR_materials_clearcoat/clearcoatTexture', srgb: false },
  { slot: 'clearcoatRoughness', path: 'extensions/KHR_materials_clearcoat/clearcoatRoughnessTexture', srgb: false },
  { slot: 'clearcoatNormal', path: 'extensions/KHR_materials_clearcoat/clearcoatNormalTexture', srgb: false },
  { slot: 'sheenColor', path: 'extensions/KHR_materials_sheen/sheenColorTexture', srgb: true },
  { slot: 'sheenRoughness', path: 'extensions/KHR_materials_sheen/sheenRoughnessTexture', srgb: false },
  { slot: 'anisotropy', path: 'extensions/KHR_materials_anisotropy/anisotropyTexture', srgb: false },
  { slot: 'iridescence', path: 'extensions/KHR_materials_iridescence/iridescenceTexture', srgb: false },
  { slot: 'iridescenceThickness', path: 'extensions/KHR_materials_iridescence/iridescenceThicknessTexture', srgb: false },
  { slot: 'transmission', path: 'extensions/KHR_materials_transmission/transmissionTexture', srgb: false },
  { slot: 'thickness', path: 'extensions/KHR_materials_volume/thicknessTexture', srgb: false },
];
export const CORE_TEXTURE_COUNT = 5;
/** The extensions' textures, which share the bindings after the core five. */
export const EXTENSION_TEXTURES = MATERIAL_TEXTURES.slice(CORE_TEXTURE_COUNT);

/** The texture reference at a MATERIAL_TEXTURES path, or undefined. */
export function textureReference(material, path) {
  return path.split('/').reduce((node, key) => node?.[key], material);
}

/** Every image some material samples, by index. */
export function usedImages(json) {
  const used = new Set();
  for (const material of json.materials ?? []) {
    for (const { path } of MATERIAL_TEXTURES) {
      const image = textureImageIndex(json, textureReference(material, path)?.index);
      if (image >= 0) used.add(image);
    }
  }
  return used;
}

/**
 * Width and height from an image's header, or null if it is none of the three
 * formats glTF allows: PNG and JPEG in the core spec, WebP by EXT_texture_webp.
 * Enough of each header to find the size and nothing more.
 */
export function imageSize(bytes) {
  const u16be = (i) => (bytes[i] << 8) | bytes[i + 1];
  const u32be = (i) => ((bytes[i] << 24) >>> 0) + (bytes[i + 1] << 16) + (bytes[i + 2] << 8) + bytes[i + 3];
  const u24le = (i) => bytes[i] | (bytes[i + 1] << 8) | (bytes[i + 2] << 16);
  const ascii = (i, n) => String.fromCharCode(...bytes.subarray(i, i + n));

  // PNG: the signature, then IHDR -- always the first chunk.
  if (bytes.length >= 24 && bytes[0] === 0x89 && ascii(1, 3) === 'PNG' && ascii(12, 4) === 'IHDR') {
    return { width: u32be(16), height: u32be(20) };
  }

  // JPEG: walk the segments to the first start-of-frame. C4, C8 and CC share
  // its range and are not frames.
  if (bytes[0] === 0xff && bytes[1] === 0xd8) {
    let i = 2;
    while (i + 9 < bytes.length) {
      if (bytes[i] !== 0xff) return null;
      const marker = bytes[i + 1];
      if (marker === 0xff) { i++; continue; }                  // fill byte
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { width: u16be(i + 7), height: u16be(i + 5) };
      }
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) { i += 2; continue; }   // no length
      i += 2 + u16be(i + 2);
    }
    return null;
  }

  // WebP: a RIFF container around one of three bitstreams.
  if (bytes.length >= 30 && ascii(0, 4) === 'RIFF' && ascii(8, 4) === 'WEBP') {
    const chunk = ascii(12, 4);
    if (chunk === 'VP8 ') return { width: (bytes[26] | (bytes[27] << 8)) & 0x3fff, height: (bytes[28] | (bytes[29] << 8)) & 0x3fff };
    if (chunk === 'VP8L') {
      const b = bytes.subarray(21, 25);
      return {
        width: 1 + (((b[1] & 0x3f) << 8) | b[0]),
        height: 1 + (((b[3] & 0x0f) << 10) | (b[2] << 2) | ((b[1] & 0xc0) >> 6)),
      };
    }
    if (chunk === 'VP8X') return { width: 1 + u24le(24), height: 1 + u24le(27) };
  }
  return null;
}

/** The image index a texture points at, or -1. */
export function textureImageIndex(json, textureIndex) {
  if (textureIndex === undefined || textureIndex < 0) return -1;
  const texture = json.textures?.[textureIndex];
  if (!texture) return -1;
  return texture.source ?? -1;
}

export function textureSamplerIndex(json, textureIndex) {
  if (textureIndex === undefined || textureIndex < 0) return -1;
  return json.textures?.[textureIndex]?.sampler ?? -1;
}

/**
 * Translate a glTF sampler into a GPUSamplerDescriptor.
 *
 * glTF folds the mip filter into minFilter as six combined enums; WebGPU keeps
 * minFilter and mipmapFilter separate. Unspecified filters mean "the client
 * chooses", and trilinear repeat is the choice that surprises nobody.
 */
export function samplerDescriptor(sampler = {}) {
  const magFilter = sampler.magFilter === GLTF_NEAREST ? 'nearest' : 'linear';

  let minFilter = 'linear';
  let mipmapFilter = 'linear';
  // glTF's two unsuffixed minFilters mean NO mip chain, not "some mip mode".
  // WebGPU has no way to say that in the filter fields, so it is said with the
  // LOD clamps instead. Mapping them to a mipmapFilter and leaving the clamps
  // open -- which is what this did -- gives the pixel-art and UI-atlas assets
  // that bother to ask for NEAREST a full chain and LOD selection, so they
  // blur and swim at distance, which is the exact thing they asked not to do.
  let mipped = true;
  switch (sampler.minFilter) {
    case GLTF_NEAREST:
      minFilter = 'nearest'; mipmapFilter = 'nearest'; mipped = false; break;
    case GLTF_LINEAR:
      minFilter = 'linear'; mipmapFilter = 'nearest'; mipped = false; break;
    case GLTF_NEAREST_MIPMAP_NEAREST:
      minFilter = 'nearest'; mipmapFilter = 'nearest'; break;
    case GLTF_LINEAR_MIPMAP_NEAREST:
      minFilter = 'linear'; mipmapFilter = 'nearest'; break;
    case GLTF_NEAREST_MIPMAP_LINEAR:
      minFilter = 'nearest'; mipmapFilter = 'linear'; break;
    case GLTF_LINEAR_MIPMAP_LINEAR:
    default:
      minFilter = 'linear'; mipmapFilter = 'linear'; break;
  }

  return {
    magFilter,
    minFilter,
    mipmapFilter,
    addressModeU: addressMode(sampler.wrapS),
    addressModeV: addressMode(sampler.wrapT),
    addressModeW: 'repeat',
    // Pinning both clamps to 0 is WebGPU's way of saying "level 0 only".
    ...(mipped ? {} : { lodMinClamp: 0, lodMaxClamp: 0 }),
    // Anisotropy is only legal when all three filters are linear; asking for it
    // alongside a nearest filter is a validation error, not a silent downgrade.
    // It is also meaningless without mips.
    maxAnisotropy: mipped && magFilter === 'linear' && minFilter === 'linear' && mipmapFilter === 'linear'
      ? 16 : 1,
  };
}

function addressMode(wrap) {
  switch (wrap) {
    case GLTF_CLAMP_TO_EDGE: return 'clamp-to-edge';
    case GLTF_MIRRORED_REPEAT: return 'mirror-repeat';
    default: return 'repeat';
  }
}

/**
 * Which texture slots a material uses, and whether each holds colour.
 *
 * The sRGB decision belongs to the SLOT, not the image: base colour and
 * emissive are authored in sRGB, everything else holds raw numbers. The same
 * image used in two slots therefore needs two GPU textures, which is why the
 * upload cache keys on both.
 */
export function materialTextureSlots(material) {
  const textures = material.textures ?? {};
  return MATERIAL_TEXTURES.map(({ slot, srgb }) => ({ slot, texture: textures[slot] ?? -1, srgb }));
}
