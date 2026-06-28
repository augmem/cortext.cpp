# Cortext

Cortext is an embedding-first memory engine for applications that need durable,
queryable context over long-running streams. It ingests text, audio, and image
signals, writes source-backed memories to SQLite, maintains working memory,
builds graph associations, retrieves compact context packets, and runs explicit
shallow consolidation over stored embeddings.

The current v1 branch is a hard-cut runtime: Cortext does **not** ship an
internal decoder stack, provider registry, semantic extractor, summarizer,
static label bank, fact layer, or mode-selected deep consolidation path. Git
history preserves those experiments; the release surface is the smaller
embedding/graph engine documented here.

## What It Provides

- Native C++20 API and a stable C ABI.
- FFI bindings for Python, Go, JavaScript/TypeScript, and Dart.
- Text, audio, and image processing calls that can store durable memory.
- Text, audio, and image embed-only calls that do not mutate memory state.
- Working-memory and long-term retrieval packets returned as JSON through the
  C ABI.
- Explicit `Consolidate()` / `cortext_consolidate_json()` shallow replay.
- `Reset()` / `cortext_reset()` for volatile processor-state reset without
  deleting durable memories.
- SQLite metadata storage plus sqlite-objstore payload storage by default.
- External database/object-store callback seams for embedders that own storage.

## Runtime Model

The retained engine is built around a configured embedding model. The preferred
release model is `augmem/AIST-87M` in the local GGUF layout under
`models/AIST-87M-GGUF/`; embedding-only fallback assets may be available for
development builds. Audio inputs use 16 kHz mono float32 PCM. Image inputs use
row-major RGB/RGBA bytes with explicit width, height, and channel count.

`source_id` is opaque provenance. Cortext uses it for exact same-source grouping
and hydration, not for hidden behavior switches.

## Build And Test

Core native builds require a C++20 compiler and CMake. SQLite is built from the
bundled `third_party/sqlite` source tree.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build -R cortext_tests --output-on-failure
```

Audio and image ingestion/embedding support is enabled by default and is part
of the supported Cortext build contract. The AIST ggml kernel backend is built
from bundled source by default; use `CORTEXT_USE_SYSTEM_GGML=ON` only when you
intentionally want to link against a preinstalled ggml. Unsupported text-only
builds must opt out explicitly with `CORTEXT_ENABLE_AUDIO=OFF` or
`CORTEXT_ENABLE_IMAGE=OFF` plus `CORTEXT_ALLOW_UNSUPPORTED_TEXT_ONLY_BUILD=ON`.

OpenTelemetry support is enabled by default. Use
`CORTEXT_DISABLE_OPENTELEMETRY=ON` only when you want a no-op telemetry build
without the OpenTelemetry dependency.

The CI release gate runs the model-free suite directly:

```bash
./build/tests/cortext_tests '~[aist]' --reporter compact
```

Examples are optional:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCORTEXT_BUILD_EXAMPLES=ON
cmake --build build -j
./build/examples/topical_chat_analysis/cortext_topical_chat_analysis --help
```

## C++ Quickstart

```cpp
#include <cortext/cortext.hpp>

#include <iostream>

int main()
{
  cortext::Cortext::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto engine = cortext::Cortext::Create(cfg, "memory.db", "models");

  auto context = engine->ProcessText("Bailey likes tennis balls.", "chat/main");
  for (const auto &memory : context.retrieved_memory) {
    std::cout << memory.id << " " << memory.source_id << "\n";
  }

  auto embedding = engine->EmbedText("embed without storing");
  std::cout << "embedding dims: " << embedding.size() << "\n";

  engine->Consolidate();
  engine->Flush();
}
```

## C ABI And FFI

The C ABI lives in `include/cortext/capi.h`. JSON-returning helpers allocate
strings owned by Cortext; callers must release them with `cortext_string_free`.

Primary calls:

| Purpose | C ABI |
|---|---|
| create | `cortext_create_with_config` |
| process text/audio/image | `cortext_process_*_json` |
| embed text/audio/image only | `cortext_embed_*_json` |
| explicit shallow consolidation | `cortext_consolidate_json` |
| commit buffered writes | `cortext_flush` |
| reset volatile state | `cortext_reset` |
| diagnostics | `cortext_last_error`, `cortext_version` |

Bindings live under `bindings/`:

- `bindings/python`: `ctypes`
- `bindings/go`: `cgo`
- `bindings/javascript`: Node-API plus TypeScript declarations
- `bindings/dart`: `dart:ffi`
- `bindings/wasm`: browser ES-module wrapper over the WebAssembly C ABI

Build the shared library for FFI consumers with CMake:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

Node's native addon uses the Node-enabled CMake preset:

```bash
cmake --preset ffi-release-node
cmake --build --preset ffi-release-node --target cortext cortext_node
```

## WebAssembly

The browser build uses Emscripten and emits an ES module plus a `.wasm` payload:

```bash
./build-wasm.sh
```

Outputs:

```text
build-wasm/dist/wasm/cortext.js
build-wasm/dist/wasm/cortext.wasm
```

The generated module exports the public C ABI and `malloc/free`. A small
JavaScript wrapper lives in `bindings/wasm/cortext.js`, and a browser demo lives
in `examples/web/`.

Cortext still needs the AIST GGUF embedding model. For demos, either select the
model file in the browser UI or embed the model directory into the virtual
filesystem at build time:

```bash
./build-wasm.sh -DCORTEXT_WASM_PRELOAD_MODELS_DIR="$PWD/models/AIST-87M-GGUF"
```

Serve the repository root after building:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/examples/web/`.

## Repository Layout

- `include/`, `src/`: public headers and C++ implementation.
- `src/operations/`: memory pipeline operations.
- `tests/`: Catch2 test suite.
- `examples/`: benchmarks, demos, and smoke tests.
- `bindings/`: Python, Go, JavaScript/TypeScript, and Dart FFI packages.
- `docs/paper/`: manuscript source and generated markdown.
- `models/`: local model assets.
- `third_party/`: vendored native dependencies.

## Documentation

The design paper is generated at
`docs/paper/_manuscript/index.md` from the source sections in
`docs/paper/sections/`. Release-facing implementation details are concentrated
in:

- `docs/paper/sections/7_consolidation.qmd`
- `docs/paper/sections/9_experimental.qmd`
- `docs/paper/sections/10_implementation.qmd`
- `docs/paper/sections/11_optimization.qmd`

Regenerate the paper with:

```bash
QUARTO_DISABLE_GIT=1 QUARTO_DISABLE_GITHUB=1 quarto render docs/paper
```

If Quarto tries to read GitHub metadata from an SSH remote in a non-authenticated
environment, temporarily point `origin` at the HTTPS URL for the render and then
restore it.
