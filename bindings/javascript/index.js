"use strict";

const path = require("path");

const candidates = [
  process.env.CORTEXT_NODE_ADDON_PATH,
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
const detail = lastError && lastError.message ? `\nLast error: ${lastError.message}` : "";
throw new Error(`Could not load cortext Node addon. Tried:\n${tried}${detail}`);
}

class Cortext extends native.NativeCortext {
  processText(text, sourceId) {
    return JSON.parse(this.processTextJson(text, sourceId));
  }

  processAudio(pcm, sourceId) {
    return JSON.parse(this.processAudioJson(pcm, sourceId));
  }

  processImage(data, width, height, channels, sourceId) {
    return JSON.parse(this.processImageJson(data, width, height, channels, sourceId));
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
