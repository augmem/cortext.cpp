"use strict";

const path = require("path");

const supportedNativeTargets = new Set([
  "darwin-arm64",
  "darwin-x64",
  "linux-arm64",
  "linux-x64",
  "win32-arm64",
  "win32-x64",
]);

function platformTag() {
  return `${process.platform}-${process.arch}`;
}

const candidates = [
  process.env.CORTEXT_NODE_ADDON_PATH,
  supportedNativeTargets.has(platformTag())
    ? path.join(__dirname, "prebuilds", platformTag(), "cortext.node")
    : null,
  path.join(__dirname, "cortext.node"),
  path.join(__dirname, "build", "ffi-release", "bindings", "javascript", "cortext.node"),
  path.join(__dirname, "..", "..", "build", "ffi-release", "bindings", "javascript", "cortext.node"),
].filter(Boolean);

let native = null;
let lastError = null;

for (const candidate of candidates) {
  try {
    native = require(candidate);
    break;
  } catch (err) {
    lastError = err;
  }
}

if (!native) {
  const tried = candidates.map((candidate) => `  - ${candidate}`).join("\n");
  const supported = [...supportedNativeTargets].sort().join(", ");
  const detail = lastError && lastError.message ? `\nLast error: ${lastError.message}` : "";
  throw new Error(
    `Could not load cortext Node addon for ${platformTag()}. ` +
      `Bundled packages support: ${supported}.\nTried:\n${tried}${detail}`
  );
}

function normalizeMedia(media, mediaMimeType, options) {
  if (media && typeof media === "object" && "data" in media && "mimetype" in media) {
    const resolvedOptions =
      options === undefined && mediaMimeType && typeof mediaMimeType === "object"
        ? mediaMimeType
        : options;
    return [media.data, media.mimetype, resolvedOptions];
  }
  return [media, mediaMimeType, options];
}

class Cortext extends native.NativeCortext {
  processTextJson(text, sourceId, options) {
    return super.processTextJson(text, sourceId, options);
  }

  processText(text, sourceId, options) {
    return JSON.parse(this.processTextJson(text, sourceId, options));
  }

  processAudioJson(pcm, sourceId, options) {
    return super.processAudioJson(pcm, sourceId, options);
  }

  processAudio(pcm, sourceId, options) {
    return JSON.parse(this.processAudioJson(pcm, sourceId, options));
  }

  processAudioWithMediaJson(pcm, sourceId, media, mediaMimeType, options) {
    const [mediaData, mimetype, resolvedOptions] = normalizeMedia(
      media,
      mediaMimeType,
      options
    );
    return super.processAudioWithMediaJson(
      pcm,
      sourceId,
      mediaData,
      mimetype,
      resolvedOptions
    );
  }

  processAudioWithMedia(pcm, sourceId, media, mediaMimeType, options) {
    return JSON.parse(
      this.processAudioWithMediaJson(pcm, sourceId, media, mediaMimeType, options)
    );
  }

  processImageJson(data, width, height, channels, sourceId, options) {
    return super.processImageJson(data, width, height, channels, sourceId, options);
  }

  processImage(data, width, height, channels, sourceId, options) {
    return JSON.parse(
      this.processImageJson(data, width, height, channels, sourceId, options)
    );
  }

  processImageWithMediaJson(data, width, height, channels, sourceId, media, mediaMimeType, options) {
    const [mediaData, mimetype, resolvedOptions] = normalizeMedia(
      media,
      mediaMimeType,
      options
    );
    return super.processImageWithMediaJson(
      data,
      width,
      height,
      channels,
      sourceId,
      mediaData,
      mimetype,
      resolvedOptions
    );
  }

  processImageWithMedia(data, width, height, channels, sourceId, media, mediaMimeType, options) {
    return JSON.parse(
      this.processImageWithMediaJson(
        data,
        width,
        height,
        channels,
        sourceId,
        media,
        mediaMimeType,
        options
      )
    );
  }

  embedText(text) {
    return JSON.parse(this.embedTextJson(text)).embedding;
  }

  embedAudio(pcm) {
    return JSON.parse(this.embedAudioJson(pcm)).embedding;
  }

  embedImage(data, width, height, channels) {
    return JSON.parse(this.embedImageJson(data, width, height, channels)).embedding;
  }

  consolidate() {
    return JSON.parse(this.consolidateJson());
  }
}

module.exports = {
  NativeCortext: native.NativeCortext,
  Cortext,
  version: native.version,
  lastError: native.lastError,
};
