# Cortext Bindings

The bindings are thin language wrappers over `include/cortext/capi.h`. They do
not implement memory logic themselves; all processing, embedding, retrieval,
consolidation, reset, and storage behavior lives in the native library.

## Shared Surface

All bindings expose the same v1 concepts:

- `Config`: Focus, Sensitivity, Stability, mechanism toggles, and signal-filter
  toggles.
- `processText` / `processAudio` / `processImage`: run the memory pipeline and
  return a JSON-compatible context object.
- `Media` + `processAudioWithMedia` / `processImageWithMedia`: process
  canonical PCM/pixels while optionally storing caller-provided source media
  bytes instead of the canonical processing representation.
- Process JSON payloads include the processed signal `embedding` and
  `embedding_dimension` by default. Use each binding's process options
  (`includeEmbedding: false`, `OmitEmbedding`, or `include_embedding=False`) to
  omit them without a second processing path.
- `embedText` / `embedAudio` / `embedImage`: embed-only calls that do not store
  signals, retrieve memories, or mutate processor state.
- `consolidate`: explicit shallow graph consolidation.
- `flush`: commit buffered writes.
- `reset`: reset volatile processor state while keeping durable memory rows.
- `version` and `lastError`.

`source_id` values are opaque provenance strings. They are not command channels
or mode selectors.

The C ABI also exposes synchronous external DB/object-store callback
constructors for embedders that own storage. Python wraps those callbacks as
`DBProvider` and `ObjectStoreProvider`; Go, Node, Dart, and WASM currently use
the native SQLite/object-store path.

## Build Native Library

The default FFI build uses CMake:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

For the Node addon:

```bash
cmake --preset ffi-release-node
cmake --build --preset ffi-release-node --target cortext cortext_node
```

For the browser WebAssembly bundle:

```bash
./build-wasm.sh
```

The WASM wrapper is in `bindings/wasm`, and the generated Emscripten module is
written to `build-wasm/dist/wasm`.

Set `CORTEXT_LIBRARY_PATH` when a binding should load a specific shared library
instead of searching `build/ffi-release` and `zig-out/lib`.

## Packages

- `bindings/python`: Python via `ctypes`.
- `bindings/go`: Go via `cgo`.
- `bindings/javascript`: Node-API addon with TypeScript declarations.
- `bindings/dart`: Dart via `dart:ffi`.
- `bindings/wasm`: Browser ES module over the exported C ABI.

See each package README for language-specific commands and examples.
