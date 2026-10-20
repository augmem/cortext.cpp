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

  // The late-interaction reranker is required with the AIST model; the engine
  // looks for it beside the model (models/AIST-87M-GGUF/AIST-87M-LI.gguf).
  async writeLateInteractionModelFile(file, {
    directory = "/models/AIST-87M-GGUF",
    filename = "AIST-87M-LI.gguf",
  } = {}) {
    await this.writeModelFile(file, { directory, filename });
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

  // Process options (every process call):
  //   includeEmbedding (default true), retention ("natural" | "durable" |
  //   "boundary" | "ephemeral" or 0..3), evidenceBudgetBytes (whole bytes;
  //   0 or absent = whole memories), referenceMs (ms since the Unix epoch;
  //   absent/null = unset; read only by ephemeral calls) and asOf (hide
  //   memories stored after referenceMs).
  processText(handle, text, sourceId = "browser", options = {}) {
    const processArgs = this.#processArgs(options);
    let textPtr = 0;
    let sourcePtr = 0;
    try {
      textPtr = this.#writeString(text);
      sourcePtr = this.#writeString(sourceId);
      return this.#readJson(
        this.module._cortext_wasm_process_text_json(
          handle,
          textPtr,
          sourcePtr,
          ...processArgs,
        ),
      );
    } finally {
      this.module._free(textPtr);
      this.module._free(sourcePtr);
    }
  }

  // Text with caller media: the text is embedded and media = { data,
  // mimetype } (Uint8Array bytes) is stored as the payload. Only text/*
  // media is used as text; empty or absent media stores the text itself.
  processTextWithMedia(handle, text, sourceId = "browser", media = null,
                       options = {}) {
    const processArgs = this.#processArgs(options);
    let textPtr = 0;
    let sourcePtr = 0;
    let mediaArgs = null;
    try {
      textPtr = this.#writeString(text);
      sourcePtr = this.#writeString(sourceId);
      mediaArgs = this.#writeMedia(media);
      return this.#readJson(
        this.module._cortext_wasm_process_text_with_media_json(
          handle,
          textPtr,
          sourcePtr,
          ...mediaArgs.args,
          ...processArgs,
        ),
      );
    } finally {
      this.module._free(textPtr);
      this.module._free(sourcePtr);
      mediaArgs?.free();
    }
  }

  // Audio: 16 kHz mono float32 PCM. options.media optionally stores caller
  // media instead of the PCM. Browser builds without audio support throw.
  processAudio(handle, samples, sourceId = "browser", options = {}) {
    const processArgs = this.#processArgs(options);
    const sampleArray = samples instanceof Float32Array
      ? samples
      : new Float32Array(samples);
    let pcmPtr = 0;
    let sourcePtr = 0;
    let mediaArgs = null;
    try {
      pcmPtr = this.module._malloc(
        Math.max(sampleArray.length * sampleArray.BYTES_PER_ELEMENT, 1),
      );
      if (!pcmPtr) {
        throw new Error("Failed to allocate audio buffer");
      }
      this.module.HEAPF32.set(sampleArray, pcmPtr / sampleArray.BYTES_PER_ELEMENT);
      sourcePtr = this.#writeString(sourceId);
      mediaArgs = this.#writeMedia(options.media);
      return this.#readJson(
        this.module._cortext_wasm_process_audio_json(
          handle,
          pcmPtr,
          sampleArray.length,
          sourcePtr,
          ...mediaArgs.args,
          ...processArgs,
        ),
      );
    } finally {
      this.module._free(pcmPtr);
      this.module._free(sourcePtr);
      mediaArgs?.free();
    }
  }

  // Image: row-major RGB/RGBA bytes. options.media optionally stores caller
  // media (for example the original image/png bytes) instead of the pixels.
  processImage(handle, pixels, width, height, channels, sourceId = "browser",
               options = {}) {
    const processArgs = this.#processArgs(options);
    const pixelArray = this.#checkedPixels(pixels, width, height, channels);
    let dataPtr = 0;
    let sourcePtr = 0;
    let mediaArgs = null;
    try {
      dataPtr = this.module._malloc(pixelArray.length);
      if (!dataPtr) {
        throw new Error("Failed to allocate image buffer");
      }
      this.module.HEAPU8.set(pixelArray, dataPtr);
      sourcePtr = this.#writeString(sourceId);
      mediaArgs = this.#writeMedia(options.media);
      return this.#readJson(
        this.module._cortext_wasm_process_image_json(
          handle,
          dataPtr,
          width,
          height,
          channels,
          sourcePtr,
          ...mediaArgs.args,
          ...processArgs,
        ),
      );
    } finally {
      this.module._free(dataPtr);
      this.module._free(sourcePtr);
      mediaArgs?.free();
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
    const pixelArray = this.#checkedPixels(pixels, width, height, channels);
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

  #checkedPixels(pixels, width, height, channels) {
    const pixelArray = pixels instanceof Uint8Array ? pixels : new Uint8Array(pixels);
    if (!Number.isInteger(width) || !Number.isInteger(height)
        || !Number.isInteger(channels) || width <= 0 || height <= 0
        || channels <= 0) {
      throw new TypeError("width, height, and channels must be positive integers");
    }
    if (width > 0x7fffffff || height > 0x7fffffff || channels > 0x7fffffff) {
      throw new RangeError("image dimensions exceed the native int range");
    }
    const expected = width * height * channels;
    if (!Number.isSafeInteger(expected)) {
      throw new RangeError("image dimensions overflow addressable memory");
    }
    if (pixelArray.length < expected) {
      throw new RangeError(
        "image buffer is smaller than width * height * channels",
      );
    }
    return pixelArray;
  }

  // Trailing scalar arguments shared by every cortext_wasm_process_*_json:
  // include_embedding, retention, evidence_budget_bytes, has_reference_ms,
  // reference_ms, as_of.
  #processArgs({
    includeEmbedding = true,
    retention = "natural",
    evidenceBudgetBytes = 0,
    referenceMs = null,
    asOf = false,
  } = {}) {
    if (!Number.isInteger(evidenceBudgetBytes) || evidenceBudgetBytes < 0
        || evidenceBudgetBytes > 0xffffffff) {
      throw new RangeError(
        "evidenceBudgetBytes must be a whole number of bytes below 2^32",
      );
    }
    const hasReference = referenceMs !== null && referenceMs !== undefined;
    if (hasReference && (!Number.isSafeInteger(referenceMs) || referenceMs < 0)) {
      throw new RangeError(
        "referenceMs must be a whole, non-negative number of ms since the epoch",
      );
    }
    return [
      includeEmbedding ? 1 : 0,
      this.#retentionValue(retention),
      evidenceBudgetBytes,
      hasReference ? 1 : 0,
      hasReference ? referenceMs : 0,
      asOf ? 1 : 0,
    ];
  }

  // media = { data: Uint8Array | ArrayBuffer | number[], mimetype: string }.
  // Returns the (data, size, mimetype) pointer arguments and their release.
  #writeMedia(media) {
    if (media === null || media === undefined) {
      return { args: [0, 0, 0], free() {} };
    }
    const bytes = media.data instanceof Uint8Array
      ? media.data
      : new Uint8Array(media.data ?? []);
    if (bytes.length === 0) {
      return { args: [0, 0, 0], free() {} };
    }
    if (typeof media.mimetype !== "string" || media.mimetype === "") {
      throw new TypeError("media.mimetype must be a non-empty string");
    }
    const dataPtr = this.module._malloc(bytes.length);
    if (!dataPtr) {
      throw new Error("Failed to allocate media buffer");
    }
    let mimePtr = 0;
    try {
      mimePtr = this.#writeString(media.mimetype);
    } catch (err) {
      this.module._free(dataPtr);
      throw err;
    }
    this.module.HEAPU8.set(bytes, dataPtr);
    const module = this.module;
    return {
      args: [dataPtr, bytes.length, mimePtr],
      free() {
        module._free(dataPtr);
        module._free(mimePtr);
      },
    };
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

  #retentionValue(retention) {
    const values = { natural: 0, durable: 1, boundary: 2, ephemeral: 3 };
    if (typeof retention === "number" && Number.isInteger(retention)
        && retention >= 0 && retention <= 3) {
      return retention;
    }
    if (typeof retention === "string"
        && Object.prototype.hasOwnProperty.call(values, retention)) {
      return values[retention];
    }
    throw new TypeError(
      "retention must be natural, durable, boundary, or ephemeral",
    );
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
