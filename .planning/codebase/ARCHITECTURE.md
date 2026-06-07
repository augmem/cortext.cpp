# Architecture

**Analysis Date:** 2026-04-07

## Pattern Overview

**Overall:** Facade-driven C++ library over a fixed operation pipeline with SQLite-backed persistent state.

**Key Characteristics:**
- `include/cortext/cortext.hpp` exposes a small facade API while `src/cortext.cpp` owns wiring, backend selection, and memory hydration.
- `include/cortext/processor.hpp`, `include/cortext/processor/operation.hpp`, and `include/cortext/processor/operation_set.hpp` define a sequential operation pipeline executed by `src/signal_processor.cpp`.
- `include/cortext/store/store.hpp` abstracts persistence while `include/cortext/store/sqlite_store.hpp`, `src/store.cpp`, and `src/store/schema.cpp` implement a SQLite-first runtime with migrations and nested transactions.

## Layers

**Public API Layer:**
- Purpose: Stable entrypoints for native and FFI consumers.
- Location: `include/cortext/cortext.hpp`, `include/cortext/capi.h`, `src/capi.cpp`
- Contains: `cortext::Cortext`, the `Context` DTO, config structs, C ABI wrappers, JSON serialization helpers.
- Depends on: `SignalProcessor`, encoder/extractor/summarizer factories, store implementation.
- Used by: `examples/topical_chat_analysis/main.cpp`, `examples/chat/main.cpp`, `bindings/python/cortext/__init__.py`, `bindings/go/cortext.go`, `bindings/javascript/src/addon.cpp`, tests such as `tests/cortext.test.cpp`.

**Composition Layer:**
- Purpose: Build the runtime graph and choose local model backends.
- Location: `src/cortext.cpp`, `src/encoder/text_encoder_factory.hpp`, `src/deep_llm/deep_llm_factory.cpp`
- Contains: `Cortext::Impl`, operation-pipeline assembly, text encoder selection, deep summarizer/extractor selection, context hydration.
- Depends on: operations, processor, store, telemetry, encoder/summarizer/extractor implementations.
- Used by: `Cortext::Create()` in `src/cortext.cpp`.

**Signal Processing Layer:**
- Purpose: Execute one signal through the algorithm stack while mutating long-lived processor state.
- Location: `include/cortext/processor.hpp`, `include/cortext/processor/processor_context.hpp`, `include/cortext/processor/operation_context.hpp`, `src/signal_processor.cpp`
- Contains: `SignalProcessor`, `ProcessorContext`, `OperationContext`, transaction-scoped output assembly, state load/persist helpers.
- Depends on: `Store`, `IOperation`, `Signal`, telemetry, Eigen.
- Used by: `src/cortext.cpp`, tests such as `tests/signal_processor.test.cpp` and `tests/operation_context.test.cpp`.

**Operation Layer:**
- Purpose: Implement the actual cognitive/memory algorithms as small pipeline steps.
- Location: `include/cortext/operations/*.hpp`, `src/operations/*.cpp`
- Contains: scoring, thresholding, retrieval, graph construction, consolidation, working-memory, neuromodulation, accumulation, storage, and feedback steps.
- Depends on: `OperationContext`, `ProcessorContext`, `Store`, occasionally encoder/extractor/summarizer interfaces.
- Used by: `BuildPipelineRoot()` in `src/cortext.cpp`.

**Persistence Layer:**
- Purpose: Own schema, migrations, SQL execution, transaction nesting, and low-level content storage.
- Location: `include/cortext/store/*.hpp`, `src/store.cpp`, `src/store/schema.cpp`, `src/store/facts.cpp`, `src/store/extension_loader.cpp`
- Contains: `Store`, `Transaction`, `SQLiteStore`, schema migrations, sqlite extension loading, fact queries/helpers.
- Depends on: SQLite C API, bundled sqlite extensions, telemetry.
- Used by: `SignalProcessor`, `Cortext` hydration, store-focused tests such as `tests/store.test.cpp` and `tests/migration_core.test.cpp`.

**Model Backend Layer:**
- Purpose: Encoders and local inference adapters for summarization, extraction, audio, and deep consolidation.
- Location: `include/cortext/encoder/*.hpp`, `include/cortext/extractor/*.hpp`, `include/cortext/summarizer/*.hpp`, `include/cortext/audio/*.hpp`, `src/encoder/*.cpp`, `src/extractor/*.cpp`, `src/summarizer/*.cpp`, `src/audio/*.cpp`, `src/deep_llm/*.cpp`
- Contains: `Encoder`, `Extractor`, `Summarizer` interfaces plus EmbeddingGemma, Gemma, Phi-4, sherpa-onnx, and llama.cpp-backed implementations.
- Depends on: model assets under `models/`, third-party runtimes configured in `CMakeLists.txt`.
- Used by: the composition layer in `src/cortext.cpp` and targeted tests like `tests/gemma_extractor.test.cpp` and `tests/phi4_summarizer.test.cpp`.

**Application Layer:**
- Purpose: Optional binaries for manual use, experiments, telemetry smoke tests, and research sweeps.
- Location: `examples/`, `tools/`, `scripts/`
- Contains: chat UI, benchmark programs, topical-chat analysis, sqlite telemetry smoke test, offline label/text tools, Python/bash experiment harnesses.
- Depends on: `cortext::cortext`, and in several cases private headers under `src/`.
- Used by: local development and experiment workflows, not by the core library.

## Data Flow

**Online Signal Processing:**

1. A caller invokes `Cortext::ProcessText`, `Cortext::ProcessAudio`, or `Cortext::ProcessImage` from `src/cortext.cpp`.
2. `src/cortext.cpp` encodes raw input through a selected `Encoder` and constructs a modality-agnostic `Signal` from `include/cortext/signal.hpp`.
3. `SignalProcessor::Process()` in `src/signal_processor.cpp` opens a transaction, creates an `OperationContext`, executes the `OperationSet`, and persists long-lived processor state.
4. Output IDs from retrieval/storage are converted into `Cortext::Context`, and `src/cortext.cpp` hydrates working memory and retrieved memories from SQLite before returning to the caller.

**Consolidation Flow:**

1. `Cortext::Consolidate()` in `src/cortext.cpp` constructs a synthetic consolidation signal.
2. The same processor/store stack runs consolidation-related operations such as `ConsolidationGate`, `ConsolidationCluster`, `ConsolidationShallow`, `ConsolidationSummarize`, `EnqueueExtractionJobs`, and `ProcessExtractionResults`.
3. Summaries, labels, relations, and graph edges are written back through the store and returned as a normal `Context`.

**FFI Flow:**

1. Foreign-language wrappers call the C ABI in `include/cortext/capi.h`.
2. `src/capi.cpp` converts C data to `cortext::Cortext` calls.
3. JSON-returning helpers serialize `Cortext::Context` and release ownership via `cortext_string_free()`.

**State Management:**
- Long-lived adaptive state lives in `ProcessorContext` in `include/cortext/processor/processor_context.hpp`.
- Transaction-scoped per-signal state lives in `OperationContext` in `include/cortext/processor/operation_context.hpp`.
- Persisted state, memories, signals, embeddings, associations, and accumulators live in the v2 SQLite schema created by `src/store/schema.cpp`.

## Key Abstractions

**Facade Runtime (`Cortext`):**
- Purpose: Main user-facing object for processing and consolidation.
- Examples: `include/cortext/cortext.hpp`, `src/cortext.cpp`
- Pattern: Pimpl facade with backend composition hidden in `Cortext::Impl`.

**Pipeline Step (`IOperation`):**
- Purpose: One atomic algorithm step in the processing chain.
- Examples: `include/cortext/processor/operation.hpp`, `include/cortext/processor/operation_set.hpp`, `src/operations/threshold.cpp`, `src/operations/graph_retrieval.cpp`
- Pattern: Command-style interface executed sequentially by `OperationSet`.

**Long-Lived Adaptive State (`ProcessorContext`):**
- Purpose: Carry EWMAs, thresholds, priors, recent context, consolidation timers, working-memory state, and blender weights across signals.
- Examples: `include/cortext/processor/processor_context.hpp`, `src/signal_processor.cpp`
- Pattern: Mutable state bag persisted to the database between runs.

**Persistence Boundary (`Store` / `Transaction`):**
- Purpose: Isolate SQL execution from the rest of the library.
- Examples: `include/cortext/store/store.hpp`, `include/cortext/store/sqlite_store.hpp`
- Pattern: Interface + SQLite implementation with nested transactions/savepoints.

**Backend Interfaces (`Encoder`, `Extractor`, `Summarizer`):**
- Purpose: Hide model-specific runtime details from the processing pipeline.
- Examples: `include/cortext/encoder/encoder.hpp`, `include/cortext/extractor/extractor.hpp`, `include/cortext/summarizer/summarizer.hpp`
- Pattern: Runtime-selected strategy objects passed into `SignalProcessor::Config`.

## Entry Points

**Core Library Creation:**
- Location: `src/cortext.cpp`
- Triggers: `Cortext::Create()` from C++ callers and `cortext_create_with_config()` from `src/capi.cpp`
- Responsibilities: Open store, run migrations, choose encoder and deep LLM backends, build pipeline root, create `SignalProcessor`.

**Processing API:**
- Location: `include/cortext/cortext.hpp`, `src/cortext.cpp`
- Triggers: Text/audio/image calls from examples, tests, and bindings.
- Responsibilities: Encode input, execute processor, hydrate memory results, return `Context`.

**C API Surface:**
- Location: `include/cortext/capi.h`, `src/capi.cpp`
- Triggers: Go, Python, JavaScript bindings.
- Responsibilities: C-compatible lifecycle, status/error handling, JSON serialization.

**Test Runner:**
- Location: `tests/CMakeLists.txt`
- Triggers: `cortext_tests` executable and `ctest`.
- Responsibilities: Build a single Catch2 binary that exercises both public APIs and internal/private subsystems.

**Example Applications:**
- Location: `examples/chat/main.cpp`, `examples/topical_chat_analysis/main.cpp`, `examples/otel_sqlite_smoketest/main.cpp`, `examples/benchmark/*.cpp`
- Triggers: Optional `CORTEXT_BUILD_EXAMPLES=ON` builds.
- Responsibilities: Manual UI, telemetry analysis, smoke tests, and research benchmarking.

## Error Handling

**Strategy:** Internal C++ code throws exceptions; the C API converts failures to numeric status codes plus thread-local error strings.

**Patterns:**
- Store and schema code throw `StoreError`-derived exceptions from `include/cortext/store/store.hpp` and log failures in `src/store.cpp` / `src/store/schema.cpp`.
- `src/cortext.cpp` and `src/signal_processor.cpp` catch selected failures around hydration/state restore and log warnings through telemetry instead of crashing the caller.
- `src/capi.cpp` wraps all public C functions in exception-catching helpers and exposes details via `cortext_last_error()`.

## Cross-Cutting Concerns

**Logging:** `include/cortext/telemetry/telemetry.hpp` and `src/telemetry/telemetry.cpp` provide OpenTelemetry-based spans, metrics, and logs used across store, processing, and examples.

**Validation:** Input validation happens at API boundaries (`src/capi.cpp`, modality-specific processing methods in `src/cortext.cpp`) and in backend factories such as `src/encoder/text_encoder_factory.hpp` and `src/deep_llm/deep_llm_factory.cpp`.

**Authentication:** Not applicable inside the core library. External auth is only relevant to optional example code such as `examples/chat/main.cpp`, which integrates an OpenAI client separately from the core runtime.

**Boundary Notes:**
- Public consumers should prefer `include/cortext/*`, but examples and some tools deliberately include private headers from `src/` by adding `${PROJECT_SOURCE_DIR}/src` or `${CMAKE_SOURCE_DIR}/src` in `examples/*/CMakeLists.txt` and `tools/*/CMakeLists.txt`.
- Tests are intentionally white-box. `tests/CMakeLists.txt` defines `CORTEXT_TESTING=1`, enabling helpers such as `DebugHydrateForTest()` in `include/cortext/cortext.hpp`.

---

*Architecture analysis: 2026-04-07*
