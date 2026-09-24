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
 * Decode every image the document declares.
 *
 * Returns an array parallel to json.images, with null where an image could not
 * be decoded. A broken image is not fatal -- the material falls back to its
 * factor, which is a far better outcome than refusing to show the model.
 */
export async function decodeImages(json, buffers, { baseURL, fetchImpl = globalThis.fetch } = {}) {
  const images = json.images ?? [];

  return Promise.all(images.map(async (image, index) => {
    try {
      const blob = await imageBlob(image, json, buffers, { baseURL, fetchImpl, index });
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
  return [
    { slot: 'baseColor', texture: textures.baseColor ?? -1, srgb: true },
    { slot: 'emissive', texture: textures.emissive ?? -1, srgb: true },
    { slot: 'normal', texture: textures.normal ?? -1, srgb: false },
    { slot: 'metallicRoughness', texture: textures.metallicRoughness ?? -1, srgb: false },
    { slot: 'occlusion', texture: textures.occlusion ?? -1, srgb: false },
  ];
}
