<!-- GSD:project-start source:PROJECT.md -->
## Project

**cortext**

cortext is the memory engine powering augmem.ai for augmenting human and LLM memory. It is a brownfield C++ system that ingests multimodal signals, persists durable memory traces, and resurfaces relevant context for applications, agents, analyses, and realtime chat experiences.

**Core Value:** Important context should resurface at the right time for humans and models without requiring manual memory management.

### Constraints

- **Tech stack**: C++20 with CMake, SQLite, and local model runtimes — the current engine architecture is already in production use and should be evolved, not replaced casually
- **API stability**: Public headers in `include/` and the C API require explicit approval before breaking changes — bindings and examples depend on them
- **Research traceability**: Algorithm and experiment changes must be reflected in `docs/paper/sections/` and the generated manuscript — this repo treats paper evidence as part of the product record
- **Performance**: On-device latency matters, especially for speech and realtime interaction paths — augmem.ai needs memory augmentation that feels live, not batch-oriented
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++20 - Core engine, public API, examples, and tests live in `CMakeLists.txt`, `src/`, `include/`, `examples/`, and `tests/`.
- C - Low-level integration code and embedded extensions live in `CMakeLists.txt`, `src/capi.cpp`, `include/cortext/capi.h`, `third_party/sqlite-vec/sqlite-vec.c`, and `third_party/sqlite-objstore/src/*.c`.
- Python 3.10+ - FFI package metadata is in `bindings/python/pyproject.toml`; experiment and data/model tooling live in `scripts/*.py` and `tools/**/*.py`.
- Go 1.24 - Go bindings are declared in `bindings/go/go.mod`.
- JavaScript/TypeScript - Node addon packaging lives in `bindings/javascript/package.json` and native addon source lives in `bindings/javascript/src/addon.cpp`.
- Objective-C++ - macOS voice chat integration lives in `examples/chat/voice_session.mm`.
- CMake - Build orchestration lives in `CMakeLists.txt`, `CMakePresets.json`, `tests/CMakeLists.txt`, `examples/**/CMakeLists.txt`, and `cmake/*.cmake`.
## Runtime
- Native host runtime on macOS/Linux is the default build path in `CMakeLists.txt` and `CMakePresets.json`.
- Optional WebAssembly runtime is configured through `CMakePresets.json` and `cmake/EmscriptenToolchain.cmake`.
- CMake + FetchContent + git submodules - native dependency resolution is defined in `CMakeLists.txt` and `.gitmodules`.
- Binding-specific package managers:
- Lockfile: missing at the repo root and in `bindings/`.
## Frameworks
- CMake 3.16+ - primary native build system in `CMakeLists.txt`; presets require CMake 3.21+ in `CMakePresets.json`.
- SQLite 3 - primary persistence layer via system `sqlite3` on native builds and vendored `third_party/sqlite` for WASM in `CMakeLists.txt` and `include/cortext/store/sqlite_store.hpp`.
- Eigen 3.4.0 - numeric/vector math dependency fetched in `CMakeLists.txt`.
- nlohmann/json v3.12.0 - JSON handling for C API responses, generator schemas, and tests in `CMakeLists.txt`, `src/capi.cpp`, and `src/generator/json_decoder.cpp`.
- Catch2 v3.5.3 - unit/integration test framework fetched in `tests/CMakeLists.txt`.
- CTest - test registration and execution live in `CMakeLists.txt` and `tests/CMakeLists.txt`.
- Bazel/Bazelisk - required to build LiteRT-LM in `CMakeLists.txt` and `scripts/build_litert.sh`.
- pkg-config - required to discover system SQLite on native builds in `CMakeLists.txt`.
- Zlib - required for ONNX Runtime integration in `CMakeLists.txt` and `src/encoder/embeddinggemma.cpp`.
- Emscripten - optional WASM toolchain in `cmake/EmscriptenToolchain.cmake`.
## Key Dependencies
- `opentelemetry-cpp` v1.24.0 - tracing/metrics/logging API dependency fetched in `CMakeLists.txt` and used in `src/telemetry/telemetry.cpp`.
- ONNX Runtime - optional local inference runtime for Gemma ORT paths in `CMakeLists.txt` and `src/encoder/embeddinggemma.cpp`.
- `onnxruntime-genai` - Phi-4 extractor/summarizer backend vendored via `third_party/onnxruntime-genai` and linked in `CMakeLists.txt`, `src/extractor/phi4_extractor.cpp`, and `include/cortext/summarizer/phi4_summarizer.hpp`.
- LiteRT-LM - Gemma extractor/summarizer backend vendored via `third_party/litert-lm` and linked in `CMakeLists.txt`, `src/extractor/gemma_extractor.cpp`, and `src/summarizer/gemma_summarizer.cpp`.
- `llama.cpp` system libraries (`llama`, `ggml`, `ggml-base`) - GGUF inference path for Liquid/LFM2 and optional EmbeddingGemma GGUF in `CMakeLists.txt`, `src/deep_llm/deep_llm_factory.cpp`, and `src/encoder/embeddinggemma.cpp`.
- `sherpa-onnx` - offline ASR/TTS integration in `CMakeLists.txt`, `src/audio/sherpa_onnx.cpp`, and `include/cortext/audio/sherpa_onnx.hpp`.
- `sqlite-vec` - embedded vector index for 256-dim embeddings in `CMakeLists.txt`, `src/store/schema.cpp`, and `src/store/extension_loader.cpp`.
- `sqlite-objstore` - blob/object payload storage in `CMakeLists.txt`, `src/store/schema.cpp`, `src/store/extension_loader.cpp`, and `src/operations/memory_storage.cpp`.
- Node.js headers / N-API v8 - optional Node addon build path in `CMakeLists.txt` and `bindings/javascript/src/addon.cpp`.
- Desktop UI stack for chat example:
## Configuration
- Build toggles are controlled through CMake options in `CMakeLists.txt` and presets in `CMakePresets.json`.
- Runtime model discovery is driven by the `models_dir` argument on `cortext::Cortext::Create()` in `include/cortext/cortext.hpp` and by encoder/backend resolution in `src/encoder/text_encoder_factory.hpp` and `src/deep_llm/deep_llm_factory.cpp`.
- Important runtime overrides are read from environment variables in:
- `.env` files: not detected by filename in the repo root during this scan.
- Root build graph: `CMakeLists.txt`
- Presets: `CMakePresets.json`
- LiteRT helper scripts: `cmake/litert_append_build.cmake`, `cmake/litert_install.cmake`, and `scripts/build_litert.sh`
- CI build recipe: `.github/workflows/build.yml`
- Binding manifests: `bindings/python/pyproject.toml`, `bindings/go/go.mod`, and `bindings/javascript/package.json`
## Model and Runtime Assets
- EmbeddingGemma ONNX export in `models/embeddinggemma-300m-onnx/`
- EmbeddingGemma LiteRT export in `models/embeddinggemma-300m-litert/`
- EmbeddingGemma GGUF in `models/llama_cpp/`
- Gemma 3n LiteRT models in `models/gemma3n-e2b-litert/`
- Gemma 3n ONNX assets in `models/gemma-3n/onnx/`
- Phi-4 multimodal ONNX assets in `models/phi4-mm-cpu/`
- Liquid/LFM2 GGUF assets in `models/LFM2-1.2B-Extract-GGUF/`, `models/LFM2-2.6B-Transcript-GGUF/`, `models/LFM2.5-1.2B-Instruct-GGUF/`, and `models/LFM2.5-350M-GGUF/`
- sherpa-onnx ASR/TTS assets in `models/sherpa-onnx/`
- whisper.cpp asset in `models/whisper.cpp/`
- Preferred text encoder resolution is implemented in `src/encoder/text_encoder_factory.hpp`.
- Deep LLM backend selection and model fallback order are implemented in `src/deep_llm/deep_llm_factory.cpp`.
## Platform Requirements
- C++20-capable compiler, CMake, pkg-config, and SQLite development headers are required by `README.md`, `CMakeLists.txt`, and `.github/workflows/build.yml`.
- Bazel/Bazelisk is required when LiteRT-LM is enabled in `CMakeLists.txt` and `scripts/build_litert.sh`.
- Native chat example additionally needs `glfw3`, OpenGL, and libcurl per `examples/chat/CMakeLists.txt`.
- macOS voice/chat extras rely on Cocoa, Foundation, AVFoundation, IOKit, and CoreVideo in `examples/chat/CMakeLists.txt` and `examples/chat/voice_session.mm`.
- No hosted deployment target is defined in the repo.
- The shipping artifact is a native shared/static library plus optional examples/bindings built locally from `CMakeLists.txt`.
- CI only verifies Linux native build/test on GitHub Actions in `.github/workflows/build.yml`.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Use lower_snake_case for C++ source and header files. Public/private pairs usually mirror each other, for example `src/operations/focus.cpp` with `include/cortext/operations/focus.hpp`, and tests follow the same stem with `.test.cpp`, for example `tests/operations_focus.test.cpp`.
- Keep subsystem prefixes in filenames so related files sort together: `src/store/schema.cpp`, `src/store/facts.cpp`, `tests/store.test.cpp`, `tests/store_extensions.test.cpp`.
- Internal-only helpers are explicit in the filename instead of hidden behind umbrella headers, for example `src/operations/meta_learning_internal.hpp`, `src/operations/constructive_recall_internal.hpp`, and `src/operations/eviction_ablation.hpp`.
- In the library and most tests, use PascalCase for functions and methods, including file-local helpers: `NowMillis`, `ParseDbOperation`, `LoadObjstorePayload`, `InitializeCoreSchema`, `SeedEmbeddingV2`, and `IOperation::Execute`.
- Preserve local style when a file already uses a different convention. Some tests and scripts use lower_snake_case helpers such as `create_temp_db`, `cleanup_temp_db`, `parse_metrics`, and `run_case` in `tests/store.test.cpp` and `scripts/run_memory_harness.py`.
- Operation classes expose work through `Execute (OperationContext &, Transaction &) const` as defined in `include/cortext/processor/operation.hpp`.
- Use lower_snake_case for locals, parameters, and fields: `observed_cosine`, `ctx_window_size`, `db_path_`, `had_value_`, `weight_relevance_prior`, and `interrupt_gate_blocked_no_store`.
- Member fields commonly end with `_`, especially RAII wrappers and private state, for example `name_`, `old_value_`, `db_path_`, and `impl_`.
- Constants use `k` + PascalCase, for example `kEmbeddingDim`, `kHourMs`, `kHumanHalfLifeSeconds`, and `kCoverageGainFloorBase`.
- Boolean names read like predicates or state flags: `focus_priors_initialized`, `should_interrupt`, `interrupt_aborted`, `reinforcement_enabled`.
- Use PascalCase for classes, structs, and interfaces: `Cortext`, `SignalProcessor`, `SQLiteConnection`, `TempDatabase`, `ScopedEnvVar`, and `IOperation`.
- Namespace names are lower-case subsystem names, usually nested under `cortext`: `cortext::operations`, `cortext::store`, `cortext::testing`, `cortext::telemetry`, and `cortext::internal`.
- Test doubles follow the same type naming as production code: `TestEncoder`, `CapturingSummarizer`, `KeywordEncoder`, and `TriggerBoundaryOp`.
## Code Style
- No repo-level formatter config was detected. `.clang-format`, `.clang-tidy`, and `.editorconfig` are not present at the repository root.
- The dominant library style in `src/` and `include/` uses 2-space indentation, opening braces on the next line, and spaces before parentheses: see `src/store.cpp`, `src/cortext.cpp`, `src/operations/focus.cpp`, and `include/cortext/cortext.hpp`.
- Tests in `tests/operations_focus.test.cpp`, `tests/store.test.cpp`, and `tests/integration_consolidation.test.cpp` generally follow the same Allman-style formatting as the library.
- Some example and app-facing code uses a different local dialect with tighter spacing and same-line braces. Preserve the local file style when editing `examples/topical_chat_analysis/main.cpp` and `tests/chat_chunk_diagnostics.test.cpp` instead of normalizing them to the core style.
- Formatting is enforced mostly by review and by matching nearby code, not by a checked-in formatter.
- Compiler warnings are the practical style gate. `CMakeLists.txt` enables `-Wall -Wextra -Wpedantic` for non-MSVC builds and `/W4` for MSVC.
- `CORTEXT_WARNINGS_AS_ERRORS` defaults to `ON` in `CMakeLists.txt`, so library changes should be written as warning-clean by default.
- Sanitizers are opt-in quality checks in `CMakeLists.txt`: `CORTEXT_ENABLE_ASAN`, `CORTEXT_ENABLE_UBSAN`, and `CORTEXT_ENABLE_MSAN`.
## Import Organization
- Public headers are included with the installed-style prefix, for example `<cortext/processor.hpp>`, `<cortext/store/sqlite_store.hpp>`, and `<cortext/operations/focus.hpp>`.
- Internal-only test coverage sometimes reaches into non-public code with relative includes when there is no public seam. Examples: `tests/deep_llm_factory.test.cpp` includes `../src/deep_llm/deep_llm_factory.hpp`, and `tests/chat_chunk_diagnostics.test.cpp` includes `../examples/chat/chunk_diagnostics.hpp`.
- There are no alias macros or umbrella headers acting as barrel files. Include the exact header you need.
## Error Handling
- Throw typed or standard exceptions for hard failures in low-level components. `src/store.cpp` throws `StoreError` on SQLite prepare/open failures, and `tests/deep_llm_factory.test.cpp` expects `std::runtime_error` for invalid backend overrides.
- Catch exceptions at system boundaries when the code can degrade gracefully, then emit telemetry instead of crashing. `LoadObjstorePayload` and `LoadSignalBlobs` in `src/cortext.cpp` catch `std::exception` and log warnings before returning `false`.
- Use `std::optional`, empty containers, or boolean return values for absence and best-effort behavior, for example `Context::ProcessorOutput` fields in `include/cortext/cortext.hpp` and the `bool` returns in `src/cortext.cpp`.
- Mark intentionally unused parameters explicitly with `(void)tx;` in operation implementations such as `src/operations/focus.cpp`.
## Logging
- Library code uses the wrapper in `include/cortext/telemetry/telemetry.hpp` and `src/telemetry/telemetry.cpp` instead of ad hoc `std::cout` logging.
- Emit structured event names and attributes rather than interpolated strings. Examples: `telemetry::LogDebug ("cortext.focus.init", {...})` and `telemetry::LogWarn ("Failed to load signal blobs", {...})` in `src/operations/focus.cpp` and `src/cortext.cpp`.
- Telemetry is safe to call even when no SDK provider is installed. `tests/telemetry_noop_by_default.test.cpp` verifies the no-op path.
- Example binaries and scripts may still print to stdout or log files directly, for example `examples/topical_chat_analysis/main.cpp` and `scripts/run_memory_harness.py`.
## Comments
- Use `/// @brief` comments for public headers and reusable helpers where callers need contract-level guidance, as seen in `include/cortext/cortext.hpp`, `include/cortext/store/schema.hpp`, and `include/cortext/processor/operation.hpp`.
- Use short `//` comments for rationale, algorithm references, schema blocks, and test setup notes. Good examples are `src/store/schema.cpp`, `tests/formula_validation.test.cpp`, `tests/regression_behavior.test.cpp`, and `tests/operations_threshold.test.cpp`.
- Prefer comments that explain why a step exists or which paper/spec rule it traces back to. Avoid line-by-line narration.
- Doxygen-style `/// @brief`, `/// @param`, and `/// @return` comments are the main documentation pattern for C++ APIs and test helpers, not block comments or generated doc annotations from another toolchain.
## Function Design
- Keep file-local helpers narrow and focused in anonymous namespaces. `src/store.cpp` splits query work into `ParseDbOperation`, `PrepareStatement`, `BindParameters`, and `FetchResultRow` instead of a single large function.
- Complex workflows are composed from small operation objects rather than one monolith. `src/cortext.cpp` wires many `IOperation` implementations together instead of embedding the algorithm logic inline.
- Favor explicit domain objects over long primitive parameter lists for processing code: operations receive `OperationContext &` and `Transaction &`, and high-level APIs use `Cortext::Config`, `SignalProcessor::Config`, and typed structs in `include/cortext/cortext.hpp`.
- Helper functions in tests and benchmarks often accept plain values when seeding deterministic state, for example `SeedMemoryV2`, `SeedSignalV2`, `MakeSignal`, and `SeedLongTermMemory`.
- Pure helpers return plain values or `std::optional` where absence is meaningful, for example `NowMillis`, `ToMillis`, `FindModelPath`, and `Context::ProcessTextAt`.
- Stateful operations usually mutate context and transaction state rather than returning values. Follow the `IOperation` contract unless you are writing a pure helper outside the pipeline.
## Module Design
- Public surface area lives under `include/cortext/` and is marked with `CORTEXT_EXPORT` when needed, for example `include/cortext/cortext.hpp`.
- Keep internal implementation details in `src/` or internal headers such as `src/deep_llm/deep_llm_factory.hpp` and `include/cortext/internal/cancellation.hpp`.
- Do not widen the public API accidentally. Tests are willing to include internal headers directly when coverage needs it.
- Not used. Headers are imported directly by subsystem path, for example `include/cortext/operations/focus.hpp` and `include/cortext/store/sqlite_store.hpp`.
- When adding a new operation or subsystem, create a matching header/source pair and include it explicitly from the call sites that need it.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- `include/cortext/cortext.hpp` exposes a small facade API while `src/cortext.cpp` owns wiring, backend selection, and memory hydration.
- `include/cortext/processor.hpp`, `include/cortext/processor/operation.hpp`, and `include/cortext/processor/operation_set.hpp` define a sequential operation pipeline executed by `src/signal_processor.cpp`.
- `include/cortext/store/store.hpp` abstracts persistence while `include/cortext/store/sqlite_store.hpp`, `src/store.cpp`, and `src/store/schema.cpp` implement a SQLite-first runtime with migrations and nested transactions.
## Layers
- Purpose: Stable entrypoints for native and FFI consumers.
- Location: `include/cortext/cortext.hpp`, `include/cortext/capi.h`, `src/capi.cpp`
- Contains: `cortext::Cortext`, the `Context` DTO, config structs, C ABI wrappers, JSON serialization helpers.
- Depends on: `SignalProcessor`, encoder/extractor/summarizer factories, store implementation.
- Used by: `examples/topical_chat_analysis/main.cpp`, `examples/chat/main.cpp`, `bindings/python/cortext/__init__.py`, `bindings/go/cortext.go`, `bindings/javascript/src/addon.cpp`, tests such as `tests/cortext.test.cpp`.
- Purpose: Build the runtime graph and choose local model backends.
- Location: `src/cortext.cpp`, `src/encoder/text_encoder_factory.hpp`, `src/deep_llm/deep_llm_factory.cpp`
- Contains: `Cortext::Impl`, operation-pipeline assembly, text encoder selection, deep summarizer/extractor selection, context hydration.
- Depends on: operations, processor, store, telemetry, encoder/summarizer/extractor implementations.
- Used by: `Cortext::Create()` in `src/cortext.cpp`.
- Purpose: Execute one signal through the algorithm stack while mutating long-lived processor state.
- Location: `include/cortext/processor.hpp`, `include/cortext/processor/processor_context.hpp`, `include/cortext/processor/operation_context.hpp`, `src/signal_processor.cpp`
- Contains: `SignalProcessor`, `ProcessorContext`, `OperationContext`, transaction-scoped output assembly, state load/persist helpers.
- Depends on: `Store`, `IOperation`, `Signal`, telemetry, Eigen.
- Used by: `src/cortext.cpp`, tests such as `tests/signal_processor.test.cpp` and `tests/operation_context.test.cpp`.
- Purpose: Implement the actual cognitive/memory algorithms as small pipeline steps.
- Location: `include/cortext/operations/*.hpp`, `src/operations/*.cpp`
- Contains: scoring, thresholding, retrieval, graph construction, consolidation, working-memory, neuromodulation, accumulation, storage, and feedback steps.
- Depends on: `OperationContext`, `ProcessorContext`, `Store`, occasionally encoder/extractor/summarizer interfaces.
- Used by: `BuildPipelineRoot()` in `src/cortext.cpp`.
- Purpose: Own schema, migrations, SQL execution, transaction nesting, and low-level content storage.
- Location: `include/cortext/store/*.hpp`, `src/store.cpp`, `src/store/schema.cpp`, `src/store/facts.cpp`, `src/store/extension_loader.cpp`
- Contains: `Store`, `Transaction`, `SQLiteStore`, schema migrations, sqlite extension loading, fact queries/helpers.
- Depends on: SQLite C API, bundled sqlite extensions, telemetry.
- Used by: `SignalProcessor`, `Cortext` hydration, store-focused tests such as `tests/store.test.cpp` and `tests/migration_core.test.cpp`.
- Purpose: Encoders and local inference adapters for summarization, extraction, audio, and deep consolidation.
- Location: `include/cortext/encoder/*.hpp`, `include/cortext/extractor/*.hpp`, `include/cortext/summarizer/*.hpp`, `include/cortext/audio/*.hpp`, `src/encoder/*.cpp`, `src/extractor/*.cpp`, `src/summarizer/*.cpp`, `src/audio/*.cpp`, `src/deep_llm/*.cpp`
- Contains: `Encoder`, `Extractor`, `Summarizer` interfaces plus EmbeddingGemma, Gemma, Phi-4, sherpa-onnx, and llama.cpp-backed implementations.
- Depends on: model assets under `models/`, third-party runtimes configured in `CMakeLists.txt`.
- Used by: the composition layer in `src/cortext.cpp` and targeted tests like `tests/gemma_extractor.test.cpp` and `tests/phi4_summarizer.test.cpp`.
- Purpose: Optional binaries for manual use, experiments, telemetry smoke tests, and research sweeps.
- Location: `examples/`, `tools/`, `scripts/`
- Contains: chat UI, benchmark programs, topical-chat analysis, sqlite telemetry smoke test, offline label/text tools, Python/bash experiment harnesses.
- Depends on: `cortext::cortext`, and in several cases private headers under `src/`.
- Used by: local development and experiment workflows, not by the core library.
## Data Flow
- Long-lived adaptive state lives in `ProcessorContext` in `include/cortext/processor/processor_context.hpp`.
- Transaction-scoped per-signal state lives in `OperationContext` in `include/cortext/processor/operation_context.hpp`.
- Persisted state, memories, signals, embeddings, associations, and accumulators live in the v2 SQLite schema created by `src/store/schema.cpp`.
## Key Abstractions
- Purpose: Main user-facing object for processing and consolidation.
- Examples: `include/cortext/cortext.hpp`, `src/cortext.cpp`
- Pattern: Pimpl facade with backend composition hidden in `Cortext::Impl`.
- Purpose: One atomic algorithm step in the processing chain.
- Examples: `include/cortext/processor/operation.hpp`, `include/cortext/processor/operation_set.hpp`, `src/operations/threshold.cpp`, `src/operations/graph_retrieval.cpp`
- Pattern: Command-style interface executed sequentially by `OperationSet`.
- Purpose: Carry EWMAs, thresholds, priors, recent context, consolidation timers, working-memory state, and blender weights across signals.
- Examples: `include/cortext/processor/processor_context.hpp`, `src/signal_processor.cpp`
- Pattern: Mutable state bag persisted to the database between runs.
- Purpose: Isolate SQL execution from the rest of the library.
- Examples: `include/cortext/store/store.hpp`, `include/cortext/store/sqlite_store.hpp`
- Pattern: Interface + SQLite implementation with nested transactions/savepoints.
- Purpose: Hide model-specific runtime details from the processing pipeline.
- Examples: `include/cortext/encoder/encoder.hpp`, `include/cortext/extractor/extractor.hpp`, `include/cortext/summarizer/summarizer.hpp`
- Pattern: Runtime-selected strategy objects passed into `SignalProcessor::Config`.
## Entry Points
- Location: `src/cortext.cpp`
- Triggers: `Cortext::Create()` from C++ callers and `cortext_create_with_config()` from `src/capi.cpp`
- Responsibilities: Open store, run migrations, choose encoder and deep LLM backends, build pipeline root, create `SignalProcessor`.
- Location: `include/cortext/cortext.hpp`, `src/cortext.cpp`
- Triggers: Text/audio/image calls from examples, tests, and bindings.
- Responsibilities: Encode input, execute processor, hydrate memory results, return `Context`.
- Location: `include/cortext/capi.h`, `src/capi.cpp`
- Triggers: Go, Python, JavaScript bindings.
- Responsibilities: C-compatible lifecycle, status/error handling, JSON serialization.
- Location: `tests/CMakeLists.txt`
- Triggers: `cortext_tests` executable and `ctest`.
- Responsibilities: Build a single Catch2 binary that exercises both public APIs and internal/private subsystems.
- Location: `examples/chat/main.cpp`, `examples/topical_chat_analysis/main.cpp`, `examples/otel_sqlite_smoketest/main.cpp`, `examples/benchmark/*.cpp`
- Triggers: Optional `CORTEXT_BUILD_EXAMPLES=ON` builds.
- Responsibilities: Manual UI, telemetry analysis, smoke tests, and research benchmarking.
## Error Handling
- Store and schema code throw `StoreError`-derived exceptions from `include/cortext/store/store.hpp` and log failures in `src/store.cpp` / `src/store/schema.cpp`.
- `src/cortext.cpp` and `src/signal_processor.cpp` catch selected failures around hydration/state restore and log warnings through telemetry instead of crashing the caller.
- `src/capi.cpp` wraps all public C functions in exception-catching helpers and exposes details via `cortext_last_error()`.
## Cross-Cutting Concerns
- Public consumers should prefer `include/cortext/*`, but examples and some tools deliberately include private headers from `src/` by adding `${PROJECT_SOURCE_DIR}/src` or `${CMAKE_SOURCE_DIR}/src` in `examples/*/CMakeLists.txt` and `tools/*/CMakeLists.txt`.
- Tests are intentionally white-box. `tests/CMakeLists.txt` defines `CORTEXT_TESTING=1`, enabling helpers such as `DebugHydrateForTest()` in `include/cortext/cortext.hpp`.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->


# Repository Guidelines

## Project Structure & Module Organization
- `src/` + `include/`: C++ core engine and public headers.
- `tests/`: Catch2 unit/integration tests (`cortext_tests` target).
- `examples/`: runnable demos and analysis tooling (e.g., `examples/topical_chat_analysis`).
- `scripts/` + `tools/`: experiment harnesses and generators (e.g., `scripts/run_memory_harness.py`).
- `docs/paper/sections/`: manuscript source; `docs/paper/_manuscript/` is generated output.
- `models/` + `third_party/`: runtime assets (EmbeddingGemma, LiteRT, sqlite extensions).

## Build, Test, and Development Commands
- Configure/build:
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
  cmake --build build -j
  ```
- Run tests:
  ```bash
  ctest --test-dir build -R cortext_tests --output-on-failure
  ```
- Run topical chat analysis:
  ```bash
  ./build/examples/topical_chat_analysis/cortext_topical_chat_analysis --help
  ```
- Long-horizon harness example:
  ```bash
  python scripts/run_memory_harness.py --max-conversations 2 --max-turns 360 --max-total 720 --no-multi
  ```

## Coding Style & Naming Conventions
- Match local style in the file you touch (e.g., 2-space indentation, braces on new lines).
- Prefer existing naming patterns (PascalCase helpers, lower_snake locals) over introducing new conventions.
- Pre-release rule: breaking changes are fine, but do not leave unused/deprecated code behind.

## Testing Guidelines
- Add tests when changing algorithms or thresholds (interrupts, retrieval, consolidation).
- Use `examples/topical_chat_analysis` for end-to-end validation before large sweeps.
- Keep outputs deterministic where possible; prefer fixed seeds when adding new metrics.
- Run long sweeps with `nohup` (or equivalent) so they survive terminal/session disconnects.
- Avoid `sleep` to poll background commands; for long runs, watch output with `tail -f` or check `nohup.out`/snapshot logs directly.

## Docs & Experiment Reporting
- Always update `docs/paper/sections/` when algorithms change or experiment results are produced.
- Regenerate the manuscript:
  ```bash
  QUARTO_DISABLE_GIT=1 QUARTO_DISABLE_GITHUB=1 quarto render docs/paper
  ```
- `docs/paper/_manuscript/index.md` is the generated source of truth for the compiled paper.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and descriptive (e.g., “Add interrupt precision/recall metrics”).
- PRs should include a brief summary, test commands run, and any updated experiment logs/paths.

## Repository-Specific Notes
- Behavior should derive from the three knobs (F/S/T) wherever possible.
- Consolidation/labeling uses gemma-3n-e2b via LiteRT; embeddings use the configured text encoder.
- Do not modify the public API surface (public headers in `include/`, C API) without explicit approval.

## SML / stateforward Rules
- For new realtime orchestration work, treat `docs/rules/sml.rules.md` as binding.
- Follow the RTC actor model and no-queue invariant. Do not use `sml::process_queue`, `sml::defer_queue`, mailboxes, or post-for-later mechanisms.
- Keep dispatch run-to-completion, deterministic, single-writer per actor, allocation-free during dispatch, and provably bounded.
- Do not call an actor's own `process_event` from guards, actions, or entry/exit handlers.
- Model internal multi-step flows with `sml::completion<TEvent>`, anonymous transitions, and/or entry actions. Keep anonymous/completion chains acyclic or statically bounded.
- Treat transient handoff data as event payload, not context. Use explicit events and `sml::completion<TEvent>` for per-dispatch or cross-state handoff data; reserve context for persistent actor-owned runtime state only.
- Do not use completion transitions or anonymous transitions as data-plane iteration loops. Bulk numeric/data iteration belongs in bounded allocation-free kernels inside a single transition phase.
- Keep guards pure predicates of `(event, context)` with no side effects.
- Keep actions bounded, non-blocking, and allocation-free during dispatch.
- Do not put runtime branching (`if`, `else`, `switch`, `?:`) in actions or in functions called from actions. Express runtime control flow with explicit guards, choice states, and transitions.
- Do not emulate branching with loop constructs, handler tables, or runtime-indexed dispatch arrays.
- Inject time through event payloads; do not read wall-clock time in guards or actions.
- Keep publicly exposed events small and immutable. Internal-only synchronous handoff events may carry mutable references/pointers only within the same RTC chain and must never escape via public APIs.
- Use constructor dependency injection with a component-local context aggregate.
- Use `visit_current_states` or `is(...)` for state inspection.
- Define explicit behavior for unexpected events and use `sml::unexpected_event` rather than silent drops.
- Keep tracing deterministic, bounded, and allocation-free.
- Reproduce reported SML bugs with a failing unit test before fixing them.
- `planum.cpp` should follow the same rules; its local copy lives at `third_party/planum.cpp/docs/sml.rules.md`.
