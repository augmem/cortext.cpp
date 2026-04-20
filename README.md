# Cortext

Cortext is a C++20 context and memory engine for LLM-adjacent applications. It processes text, audio, and image signals, maintains working memory plus long-term retrieval state in SQLite, and can run explicit consolidation passes that summarize, label, and connect memories.

The repository centers on the native library in `src/` and `include/`. It also includes:
- a Catch2 test suite
- optional chat, benchmark, and analysis binaries under `examples/`
- FFI bindings under `bindings/`
- experiment harnesses under `scripts/`
- the paper/docs source that describes the current algorithms and reported results

## Why Cortext Exists

Cortext began for a personal reason. Three years ago, my father-in-law was diagnosed with dementia. Since then, I have been focused on using my background in AI and machine learning to build systems that help people with memory loss preserve continuity, confidence, and independence.

The same architecture also happens to be useful for long-horizon LLM memory. But the primary motivation is human: Cortext is designed to process real-time information from a wearable device through a hub that can deliver gentle nudges to help someone remember context, reduce confusion, and avoid the humiliation and frustration that memory loss can create.

## What Cortext Provides

- memory-aware processing over streaming text, audio, and image inputs
- working-memory tracking plus long-term retrieval state in SQLite
- explicit consolidation passes for summary, label, and relation generation
- native C++ API plus C ABI for bindings
- optional local model backends for embeddings and deep consolidation

## Repository Layout

- `src/` and `include/`: core engine implementation and public headers
- `tests/`: Catch2 test suite built into the `cortext_tests` target
- `examples/`: optional binaries for chat UI, benchmarking, telemetry smoke tests, and topical-chat analysis
- `bindings/`: Python, Go, and JavaScript bindings over the C ABI
- `scripts/` and `tools/`: experiment harnesses and offline utilities
- `docs/paper/sections/`: manuscript source
- `docs/paper/_manuscript/index.md`: generated manuscript output
- `models/` and `third_party/`: local runtime assets and vendored dependencies

The rendered manuscript source of truth is here:
- [docs/paper/_manuscript/index.md](docs/paper/_manuscript/index.md)

## Build Requirements

Core native builds require:
- a C++20 toolchain
- CMake
- `pkg-config`
- SQLite development headers

Some features rely on optional local runtimes or model assets under `models/` and `third_party/`, including:
- ONNX Runtime
- LiteRT-LM
- onnxruntime-genai
- sherpa-onnx
- `llama.cpp` GGUF support for EmbeddingGemma and Liquid deep-consolidation backends

## Quickstart Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build -R cortext_tests --output-on-failure
```

Examples are off by default. To build them:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCORTEXT_BUILD_EXAMPLES=ON
cmake --build build -j
./build/examples/topical_chat_analysis/cortext_topical_chat_analysis --help
```

The Dear ImGui chat example also requires desktop dependencies such as `glfw3`, OpenGL, and `CURL`.

## Minimal C++ Example

```cpp
#include <cortext/cortext.hpp>
#include <iostream>

int main() {
  cortext::Cortext::Config cfg;
  cfg.focus = 0.7;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.8;

  auto engine = cortext::Cortext::Create(cfg, ":memory:", "models");

  auto ctx = engine->ProcessText("Hello from Cortext", "chat/user");
  if (ctx.should_interrupt) {
    for (const auto& memory : ctx.retrieved_memory) {
      std::cout << memory.id << " " << memory.source_id << "\n";
    }
  }

  engine->Consolidate();
  engine->Flush();
}
```

Primary public entrypoints:
- C++ API: `include/cortext/cortext.hpp`
- C API: `include/cortext/capi.h`

## Foreign-Language Integration

For Python, Go, JavaScript, TypeScript, and Dart consumers, use the dedicated FFI release preset. It builds a shared library and disables the heaviest optional backends by default so bindings do not need the full research stack on every install.

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

The C API includes:
- `cortext_config_init()` and `cortext_create_with_config()` for binding-safe configuration
- `cortext_version()` for runtime version checks
- `cortext_last_error()` for per-thread error reporting
- `cortext_*_json()` helpers that return the full `Context` as UTF-8 JSON
- `cortext_string_free()` to release JSON strings returned by the library

Repository-local bindings live under `bindings/`:
- `bindings/python`: pure `ctypes` wrapper over the JSON C ABI
- `bindings/go`: `cgo` wrapper with raw JSON and decoded `map[string]any` helpers
- `bindings/javascript`: Node.js addon plus TypeScript declarations
- `bindings/dart`: `dart:ffi` wrapper with generated bindings and JSON helpers

After building with `ffi-release`, quick smoke commands are:

```bash
PYTHONPATH=bindings/python python3 -c "import cortext; print(cortext.version())"
(cd bindings/go && go test .)
(cd bindings/javascript && npm run build && node -e "const { version } = require('./'); console.log(version())")
(cd bindings/dart && dart pub get && dart test)
```

## Important CMake Options

- `BUILD_TESTING=ON|OFF`: build the Catch2 suite
- `CORTEXT_BUILD_EXAMPLES=ON|OFF`: build binaries under `examples/`
- `CORTEXT_BUILD_NODE_BINDINGS=ON|OFF`: build the Node.js addon under `bindings/javascript`
- `BUILD_WASM=ON|OFF`: configure the WebAssembly build
- `CORTEXT_ENABLE_EMBEDDINGGEMMA=ON|OFF`: enable the EmbeddingGemma encoder path
- `CORTEXT_DISABLE_LITERT=ON|OFF`: disable LiteRT-LM-backed extractor/summarizer paths
- `CORTEXT_DISABLE_OGA=ON|OFF`: disable onnxruntime-genai-backed Phi-4 paths
- `CORTEXT_DISABLE_SHERPA_ONNX=ON|OFF`: disable sherpa-onnx audio integration

## Deep Consolidation Backends

Deep consolidation backend selection stays internal, but you can override it with environment variables:

- `CORTEXT_DEEP_LLM_BACKEND=auto|gemma|lfm2|mixed`
- `CORTEXT_LFM2_SUMMARIZER_MODEL=/abs/path/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`
- `CORTEXT_LFM2_EXTRACT_MODEL=/abs/path/LFM2.5-350M-Q4_K_M.gguf`
- `CORTEXT_LLAMA_CPP_LOG_LEVEL=none|error|warn|info|debug`

Current behavior:
- `auto` prefers the mixed path when both stacks are available
- the Liquid `llama.cpp` summarizer path now prefers `LFM2.5-1.2B-Instruct-GGUF`
- the Liquid extractor path continues to prefer `LFM2.5-350M-GGUF`
- if the preferred Liquid summarizer is not present, the resolver falls back to `LFM2.5-350M-GGUF`, then the older pinned `LFM2-2.6B-Transcript` summarizer
- the mixed path uses Gemma/LiteRT-LM for summarization and Liquid/`llama.cpp` for extraction

## Experiments And Docs

Run the long-horizon harness with:

```bash
python scripts/run_memory_harness.py --max-conversations 2 --max-turns 360 --max-total 720 --no-multi
```

When algorithms or results change, update the paper sources under `docs/paper/sections/` and regenerate the manuscript:

```bash
QUARTO_DISABLE_GIT=1 QUARTO_DISABLE_GITHUB=1 quarto render docs/paper
```

`docs/paper/_manuscript/index.md` is the generated manuscript source of truth used by the paper build.

## Release Status

Cortext is currently released as an `alpha`.

The current alpha is focused on proving the core long-horizon memory architecture, retrieval behavior, consolidation pipeline, and local inference stack well enough to ship publicly and iterate with users.

## v1 Direction

The planned `v1` direction is to harden the runtime and inference stack in two specific ways:

- move the event-driven system to `stateforward/sml.cpp`
- use that transition to improve runtime structure and memory safety
- move inference onto `stateforward/emel.cpp` when that library is complete and ready for production use

Those changes are intentionally deferred until `v1`. The current alpha remains focused on shipping, stabilizing behavior, and collecting real-world feedback before making that architectural transition.

## License

Cortext is licensed under the Apache License, Version 2.0. See `LICENSE`.

## Third-Party Licensing

This repository includes or depends on third-party code, model assets, and runtime components that may be licensed under terms other than Apache-2.0.

The Apache-2.0 license applies to Cortext source code in this repository unless otherwise noted. Third-party components retain their own licenses, including material under `third_party/`, `models/`, and any bundled external assets.

Before redistributing binaries, model bundles, or packaged releases, verify the license terms for every included dependency and model artifact.
