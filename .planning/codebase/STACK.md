# Technology Stack

**Analysis Date:** 2026-04-07

## Languages

**Primary:**
- C++20 - Core engine, public API, examples, and tests live in `CMakeLists.txt`, `src/`, `include/`, `examples/`, and `tests/`.
- C - Low-level integration code and embedded extensions live in `CMakeLists.txt`, `src/capi.cpp`, `include/cortext/capi.h`, `third_party/sqlite-vec/sqlite-vec.c`, and `third_party/sqlite-objstore/src/*.c`.

**Secondary:**
- Python 3.10+ - FFI package metadata is in `bindings/python/pyproject.toml`; experiment and data/model tooling live in `scripts/*.py` and `tools/**/*.py`.
- Go 1.24 - Go bindings are declared in `bindings/go/go.mod`.
- JavaScript/TypeScript - Node addon packaging lives in `bindings/javascript/package.json` and native addon source lives in `bindings/javascript/src/addon.cpp`.
- Objective-C++ - macOS voice chat integration lives in `examples/chat/voice_session.mm`.
- CMake - Build orchestration lives in `CMakeLists.txt`, `CMakePresets.json`, `tests/CMakeLists.txt`, `examples/**/CMakeLists.txt`, and `cmake/*.cmake`.

## Runtime

**Environment:**
- Native host runtime on macOS/Linux is the default build path in `CMakeLists.txt` and `CMakePresets.json`.
- Optional WebAssembly runtime is configured through `CMakePresets.json` and `cmake/EmscriptenToolchain.cmake`.

**Package Manager:**
- CMake + FetchContent + git submodules - native dependency resolution is defined in `CMakeLists.txt` and `.gitmodules`.
- Binding-specific package managers:
  - `setuptools>=68` in `bindings/python/pyproject.toml`
  - Go modules in `bindings/go/go.mod`
  - npm in `bindings/javascript/package.json`
- Lockfile: missing at the repo root and in `bindings/`.

## Frameworks

**Core:**
- CMake 3.16+ - primary native build system in `CMakeLists.txt`; presets require CMake 3.21+ in `CMakePresets.json`.
- SQLite 3 - primary persistence layer via system `sqlite3` on native builds and vendored `third_party/sqlite` for WASM in `CMakeLists.txt` and `include/cortext/store/sqlite_store.hpp`.
- Eigen 3.4.0 - numeric/vector math dependency fetched in `CMakeLists.txt`.
- nlohmann/json v3.12.0 - JSON handling for C API responses, generator schemas, and tests in `CMakeLists.txt`, `src/capi.cpp`, and `src/generator/json_decoder.cpp`.

**Testing:**
- Catch2 v3.5.3 - unit/integration test framework fetched in `tests/CMakeLists.txt`.
- CTest - test registration and execution live in `CMakeLists.txt` and `tests/CMakeLists.txt`.

**Build/Dev:**
- Bazel/Bazelisk - required to build LiteRT-LM in `CMakeLists.txt` and `scripts/build_litert.sh`.
- pkg-config - required to discover system SQLite on native builds in `CMakeLists.txt`.
- Zlib - required for ONNX Runtime and ImageBind tokenizer gzip handling in `CMakeLists.txt` and `src/encoder/imagebind.cpp`.
- Emscripten - optional WASM toolchain in `cmake/EmscriptenToolchain.cmake`.

## Key Dependencies

**Critical:**
- `opentelemetry-cpp` v1.24.0 - tracing/metrics/logging API dependency fetched in `CMakeLists.txt` and used in `src/telemetry/telemetry.cpp`.
- ONNX Runtime - local inference runtime for ImageBind and optional Gemma ORT paths in `CMakeLists.txt`, `src/encoder/imagebind.cpp`, and `src/encoder/embeddinggemma.cpp`.
- `onnxruntime-genai` - Phi-4 extractor/summarizer backend vendored via `third_party/onnxruntime-genai` and linked in `CMakeLists.txt`, `src/extractor/phi4_extractor.cpp`, and `include/cortext/summarizer/phi4_summarizer.hpp`.
- LiteRT-LM - Gemma extractor/summarizer backend vendored via `third_party/litert-lm` and linked in `CMakeLists.txt`, `src/extractor/gemma_extractor.cpp`, and `src/summarizer/gemma_summarizer.cpp`.
- `llama.cpp` system libraries (`llama`, `ggml`, `ggml-base`) - GGUF inference path for Liquid/LFM2 and optional EmbeddingGemma GGUF in `CMakeLists.txt`, `src/deep_llm/deep_llm_factory.cpp`, and `src/encoder/embeddinggemma.cpp`.
- `sherpa-onnx` - offline ASR/TTS integration in `CMakeLists.txt`, `src/audio/sherpa_onnx.cpp`, and `include/cortext/audio/sherpa_onnx.hpp`.

**Infrastructure:**
- `sqlite-vec` - embedded vector index for 256-dim embeddings in `CMakeLists.txt`, `src/store/schema.cpp`, and `src/store/extension_loader.cpp`.
- `sqlite-objstore` - blob/object payload storage in `CMakeLists.txt`, `src/store/schema.cpp`, `src/store/extension_loader.cpp`, and `src/operations/memory_storage.cpp`.
- Node.js headers / N-API v8 - optional Node addon build path in `CMakeLists.txt` and `bindings/javascript/src/addon.cpp`.
- Desktop UI stack for chat example:
  - Dear ImGui `v1.91.6-docking` in `examples/chat/CMakeLists.txt`
  - ImPlot commit `4707b24` in `examples/chat/CMakeLists.txt`
  - `glfw3`, `OpenGL`, and `CURL` in `examples/chat/CMakeLists.txt`
  - `openai-cpp` v0.1.3 and `whisper.cpp` v1.8.4 in `examples/chat/CMakeLists.txt`

## Configuration

**Environment:**
- Build toggles are controlled through CMake options in `CMakeLists.txt` and presets in `CMakePresets.json`.
- Runtime model discovery is driven by the `models_dir` argument on `cortext::Cortext::Create()` in `include/cortext/cortext.hpp` and by encoder/backend resolution in `src/encoder/text_encoder_factory.hpp` and `src/deep_llm/deep_llm_factory.cpp`.
- Important runtime overrides are read from environment variables in:
  - `src/deep_llm/deep_llm_factory.cpp` for `CORTEXT_DEEP_LLM_BACKEND`, `CORTEXT_LFM2_SUMMARIZER_MODEL`, and `CORTEXT_LFM2_EXTRACT_MODEL`
  - `src/encoder/embeddinggemma.cpp` and `src/encoder/text_encoder_factory.hpp` for `CORTEXT_EMBEDDINGGEMMA_BACKEND`
  - `src/store/extension_loader.cpp` for `SQLITE_VEC_PATH`
  - `examples/chat/main.cpp` for `OPENAI_*`, `CORTEXT_MODELS_DIR`, `CORTEXT_CHAT_DB`, `CORTEXT_CHAT_SETTINGS`, and `CORTEXT_*` knob/env settings
- `.env` files: not detected by filename in the repo root during this scan.

**Build:**
- Root build graph: `CMakeLists.txt`
- Presets: `CMakePresets.json`
- LiteRT helper scripts: `cmake/litert_append_build.cmake`, `cmake/litert_install.cmake`, and `scripts/build_litert.sh`
- CI build recipe: `.github/workflows/build.yml`
- Binding manifests: `bindings/python/pyproject.toml`, `bindings/go/go.mod`, and `bindings/javascript/package.json`

## Model and Runtime Assets

**Bundled local model directories:**
- ImageBind ONNX encoders in `models/imagebind/`
- EmbeddingGemma ONNX export in `models/embeddinggemma-300m-onnx/`
- EmbeddingGemma LiteRT export in `models/embeddinggemma-300m-litert/`
- EmbeddingGemma GGUF in `models/llama_cpp/`
- Gemma 3n LiteRT models in `models/gemma3n-e2b-litert/`
- Gemma 3n ONNX assets in `models/gemma-3n/onnx/`
- Phi-4 multimodal ONNX assets in `models/phi4-mm-cpu/`
- Liquid/LFM2 GGUF assets in `models/LFM2-1.2B-Extract-GGUF/`, `models/LFM2-2.6B-Transcript-GGUF/`, `models/LFM2.5-1.2B-Instruct-GGUF/`, and `models/LFM2.5-350M-GGUF/`
- sherpa-onnx ASR/TTS assets in `models/sherpa-onnx/`
- whisper.cpp asset in `models/whisper.cpp/`

**Model resolution rules implemented in code:**
- Preferred text encoder resolution is implemented in `src/encoder/text_encoder_factory.hpp`.
- ImageBind requires ONNX model files plus BPE merges; BPE fallback includes `third_party/imagebind_assets/bpe/bpe_simple_vocab_16e6.txt.gz` in `src/encoder/imagebind.cpp`.
- Deep LLM backend selection and model fallback order are implemented in `src/deep_llm/deep_llm_factory.cpp`.

## Platform Requirements

**Development:**
- C++20-capable compiler, CMake, pkg-config, and SQLite development headers are required by `README.md`, `CMakeLists.txt`, and `.github/workflows/build.yml`.
- Bazel/Bazelisk is required when LiteRT-LM is enabled in `CMakeLists.txt` and `scripts/build_litert.sh`.
- Native chat example additionally needs `glfw3`, OpenGL, and libcurl per `examples/chat/CMakeLists.txt`.
- macOS voice/chat extras rely on Cocoa, Foundation, AVFoundation, IOKit, and CoreVideo in `examples/chat/CMakeLists.txt` and `examples/chat/voice_session.mm`.

**Production:**
- No hosted deployment target is defined in the repo.
- The shipping artifact is a native shared/static library plus optional examples/bindings built locally from `CMakeLists.txt`.
- CI only verifies Linux native build/test on GitHub Actions in `.github/workflows/build.yml`.

---

*Stack analysis: 2026-04-07*
