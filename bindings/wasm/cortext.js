export class CortextWasm {
  constructor(module) {
    this.module = module;
  }

  static async create(moduleFactory, options = {}) {
    const module = await moduleFactory(options);
    return new CortextWasm(module);
  }

  mkdirp(path) {
    const parts = path.split("/").filter(Boolean);
    let current = "";
    for (const part of parts) {
      current += `/${part}`;
      try {
        this.module.FS.mkdir(current);
      } catch (err) {
        if (!err || err.code !== "EEXIST") {
          throw err;
        }
      }
    }
  }

  writeFile(path, bytes) {
    const slash = path.lastIndexOf("/");
    if (slash > 0) {
      this.mkdirp(path.slice(0, slash));
    }
    this.module.FS.writeFile(path, bytes);
  }

  async writeModelFile(file, {
    directory = "/models/AIST-87M-GGUF",
    filename = "AIST-87M_q8_0.gguf",
  } = {}) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    this.writeFile(`${directory}/${filename}`, bytes);
  }

  createContext({
    focus = 0.5,
    sensitivity = 0.5,
    stability = 0.5,
    dbPath = ":memory:",
  } = {}) {
    const dbPathPtr = this.#writeString(dbPath);
    try {
      const handle = this.module._cortext_create(
        focus,
        sensitivity,
        stability,
        dbPathPtr,
      );
      if (!handle) {
        throw new Error(this.lastError() || "Failed to create Cortext context");
      }
      return handle;
    } finally {
      this.module._free(dbPathPtr);
    }
  }

  freeContext(handle) {
    if (handle) {
      this.module._cortext_free(handle);
    }
  }

  reset(handle) {
    this.#checkStatus(this.module._cortext_reset(handle));
  }

  flush(handle) {
    this.#checkStatus(this.module._cortext_flush(handle));
  }

  consolidate(handle) {
    this.#checkStatus(this.module._cortext_consolidate(handle));
  }

  version() {
    return this.module.UTF8ToString(this.module._cortext_version());
  }

  lastError() {
    const ptr = this.module._cortext_last_error();
    return ptr ? this.module.UTF8ToString(ptr) : "";
  }

  embedText(handle, text) {
    const textPtr = this.#writeString(text);
    try {
      return this.#readJson(this.module._cortext_embed_text_json(handle, textPtr));
    } finally {
      this.module._free(textPtr);
    }
  }

  processText(handle, text, sourceId = "browser") {
    const textPtr = this.#writeString(text);
    const sourcePtr = this.#writeString(sourceId);
    try {
      return this.#readJson(
        this.module._cortext_process_text_json(handle, textPtr, sourcePtr),
      );
    } finally {
      this.module._free(textPtr);
      this.module._free(sourcePtr);
    }
  }

  embedAudio(handle, samples) {
    const sampleArray = samples instanceof Float32Array
      ? samples
      : new Float32Array(samples);
    const bytes = sampleArray.length * sampleArray.BYTES_PER_ELEMENT;
    const pcmPtr = this.module._malloc(bytes);
    if (!pcmPtr) {
      throw new Error("Failed to allocate audio buffer");
    }
    try {
      this.module.HEAPF32.set(sampleArray, pcmPtr / sampleArray.BYTES_PER_ELEMENT);
      return this.#readJson(
        this.module._cortext_embed_audio_json(handle, pcmPtr, sampleArray.length),
      );
    } finally {
      this.module._free(pcmPtr);
    }
  }

  embedImage(handle, pixels, width, height, channels) {
    const pixelArray = pixels instanceof Uint8Array ? pixels : new Uint8Array(pixels);
    const dataPtr = this.module._malloc(pixelArray.length);
    if (!dataPtr) {
      throw new Error("Failed to allocate image buffer");
    }
    try {
      this.module.HEAPU8.set(pixelArray, dataPtr);
      return this.#readJson(
        this.module._cortext_embed_image_json(
          handle,
          dataPtr,
          width,
          height,
          channels,
        ),
      );
    } finally {
      this.module._free(dataPtr);
    }
  }

  #writeString(value) {
    const size = this.module.lengthBytesUTF8(value) + 1;
    const ptr = this.module._malloc(size);
    if (!ptr) {
      throw new Error("Failed to allocate string");
    }
    this.module.stringToUTF8(value, ptr, size);
    return ptr;
  }

  #readJson(ptr) {
    if (!ptr) {
      throw new Error(this.lastError() || "Cortext returned null JSON");
    }
    try {
      return JSON.parse(this.module.UTF8ToString(ptr));
    } finally {
      this.module._cortext_string_free(ptr);
    }
  }

  #checkStatus(status) {
    if (status !== 0) {
      throw new Error(this.lastError() || `Cortext call failed with status ${status}`);
    }
  }
}
