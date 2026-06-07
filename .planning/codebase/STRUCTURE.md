# Codebase Structure

**Analysis Date:** 2026-04-07

## Directory Layout

```text
cortext/
├── src/                    # Core library implementation and private wiring
├── include/cortext/       # Public headers for the library and C API
├── tests/                 # Catch2 white-box and integration tests
├── examples/              # Optional binaries for chat, analysis, smoke tests, and benches
├── bindings/              # Python, Go, and JavaScript wrappers over the C API
├── scripts/               # Experiment harnesses, dataset prep, and reporting scripts
├── tools/                 # Optional CLI utilities built with examples enabled
├── docs/paper/            # Quarto paper source and generated manuscript
├── data/                  # Research datasets, label-bank assets, and benchmark inputs
├── models/                # Local model/runtime assets discovered at runtime
├── third_party/           # Vendored native dependencies and runtimes
├── cmake/                 # Toolchain and helper CMake modules
├── CMakeLists.txt         # Root build graph for library, deps, tests, examples, bindings
└── .planning/codebase/    # Generated codebase reference docs
```

## Directory Purposes

**`src/`:**
- Purpose: Private implementation of the `cortext` library.
- Contains: top-level facade code in `src/cortext.cpp`, processor/store/runtime code, and subsystem folders such as `src/operations/`, `src/store/`, `src/encoder/`, `src/extractor/`, `src/summarizer/`, `src/audio/`, `src/deep_llm/`, `src/telemetry/`.
- Key files: `src/cortext.cpp`, `src/signal_processor.cpp`, `src/store.cpp`, `src/store/schema.cpp`, `src/capi.cpp`.

**`include/cortext/`:**
- Purpose: Public headers installed for consumers.
- Contains: top-level APIs plus public subsystem contracts.
- Key files: `include/cortext/cortext.hpp`, `include/cortext/capi.h`, `include/cortext/processor.hpp`, `include/cortext/store/store.hpp`, `include/cortext/store/sqlite_store.hpp`.

**`tests/`:**
- Purpose: Single-binary Catch2 suite covering unit, integration, and regression behavior.
- Contains: `*.test.cpp` files plus test fixtures/helpers.
- Key files: `tests/CMakeLists.txt`, `tests/test_helpers.hpp`, `tests/integration_chat_e2e.test.cpp`, `tests/store.test.cpp`, `tests/signal_processor.test.cpp`.

**`examples/`:**
- Purpose: Opt-in apps and research executables built only when `CORTEXT_BUILD_EXAMPLES=ON`.
- Contains: interactive UI, benchmark programs, topical-chat analysis, telemetry smoke tests.
- Key files: `examples/chat/main.cpp`, `examples/topical_chat_analysis/main.cpp`, `examples/otel_sqlite_smoketest/main.cpp`, `examples/benchmark/CMakeLists.txt`.

**`bindings/`:**
- Purpose: Language-specific wrappers over the C API.
- Contains: Python `ctypes`, Go `cgo`, and Node-API wrappers.
- Key files: `bindings/python/cortext/__init__.py`, `bindings/go/cortext.go`, `bindings/javascript/src/addon.cpp`.

**`scripts/`:**
- Purpose: Dataset preparation, experiment automation, downloads, and report generation.
- Contains: Python and shell workflows for long-horizon studies and label-bank generation.
- Key files: `scripts/run_memory_harness.py`, `scripts/run_topical_chat_sweep.sh`, `scripts/render_experiment_tables.py`.

**`tools/`:**
- Purpose: Small opt-in developer utilities linked against the library.
- Contains: text embedding and label-bank generation executables plus supporting Python helpers.
- Key files: `tools/text_embedder/main.cpp`, `tools/label_bank_generator/main.cpp`, `tools/CMakeLists.txt`.

**`docs/paper/`:**
- Purpose: Paper/manuscript source that mirrors algorithm and schema design.
- Contains: sectioned Quarto sources, diagrams, bibliography, and generated manuscript output.
- Key files: `docs/paper/index.qmd`, `docs/paper/diagrams/entity-relationship.qmd`, `docs/paper/sections/10_implementation.qmd`, `docs/paper/_manuscript/index.md`.

**`data/`:**
- Purpose: Checked-in experiment inputs, centroids, label-bank artifacts, and prepared corpora.
- Contains: `npy`, `jsonl`, archives, and metadata for emotion, chat, and evaluation datasets.
- Key files: `data/label_bank/metadata.json`, `data/audio_emotion/metadata.json`, `data/topical_chat/train.jsonl`.

**`models/`:**
- Purpose: Runtime model directories resolved by encoder/extractor/summarizer factories.
- Contains: EmbeddingGemma, Gemma, Liquid/LFM2, sherpa-onnx, whisper, and Phi-4 assets.
- Key files: runtime directories such as `models/embeddinggemma-300m-onnx`, `models/gemma3n-e2b-litert`, `models/LFM2.5-1.2B-Instruct-GGUF`.

**`third_party/`:**
- Purpose: Vendored native dependencies referenced from `CMakeLists.txt`.
- Contains: `onnxruntime`, `onnxruntime-genai`, `litert-lm`, `sqlite`, `sqlite-vec`, `sqlite-objstore`, `sherpa-onnx`.
- Key files: `third_party/litert-lm`, `third_party/onnxruntime-genai`, `third_party/sqlite-objstore/CMakeLists.txt`.

## Key File Locations

**Entry Points:**
- `CMakeLists.txt`: Root target graph and dependency wiring.
- `src/cortext.cpp`: Library composition root and main runtime implementation.
- `src/capi.cpp`: C ABI entrypoints and JSON marshaling.
- `examples/chat/main.cpp`: Interactive chat application entrypoint.
- `examples/topical_chat_analysis/main.cpp`: Analysis executable entrypoint.
- `examples/otel_sqlite_smoketest/main.cpp`: Telemetry/store smoke test entrypoint.

**Configuration:**
- `CMakePresets.json`: Named configure/build presets including FFI-focused builds.
- `CMakeLists.txt`: Feature flags for tests, examples, WASM, and optional runtimes.
- `examples/chat/chat_memory.settings.json`: Example-local UI/runtime settings.
- `docs/paper/_quarto.yml`: Paper build configuration.

**Core Logic:**
- `include/cortext/cortext.hpp`: Main public API.
- `include/cortext/processor.hpp`: Processing orchestrator API.
- `include/cortext/processor/processor_context.hpp`: Long-lived adaptive runtime state.
- `src/signal_processor.cpp`: Transactional signal execution and state persistence.
- `src/store/schema.cpp`: Canonical v2 schema and migrations.
- `src/operations/`: Algorithm implementations. Add new per-signal algorithm steps here.

**Testing:**
- `tests/CMakeLists.txt`: Test target definition.
- `tests/test_helpers.hpp`: Shared fixtures, in-memory encoders, and schema seeding helpers.
- `tests/integration_*.test.cpp`: Cross-subsystem behavior tests.

## Naming Conventions

**Files:**
- Library implementation files use lower_snake_case names such as `src/signal_processor.cpp`, `src/store/schema.cpp`, `src/operations/graph_retrieval.cpp`.
- Public headers mirror subsystem and filename, for example `include/cortext/store/sqlite_store.hpp` and `include/cortext/operations/interrupt_gate.hpp`.
- Tests use `<subject>.test.cpp`, for example `tests/operations_threshold.test.cpp` and `tests/integration_consolidation.test.cpp`.

**Directories:**
- Subsystems use lower_snake_case directories matching namespace or responsibility, for example `src/deep_llm`, `src/operations`, `include/cortext/summarizer`.
- Example and tool directories use lower_snake_case binary names, for example `examples/topical_chat_analysis`, `tools/label_bank_generator`.

## Where to Add New Code

**New Feature:**
- Primary code: place public API in `include/cortext/` and implementation in the matching `src/` subsystem.
- Tests: add a focused `tests/<feature>.test.cpp`; use `tests/test_helpers.hpp` when the feature touches storage or encoding.

**New Operation/Algorithm Step:**
- Implementation: add `include/cortext/operations/<name>.hpp` and `src/operations/<name>.cpp`.
- Wiring: register it in `BuildPipelineRoot()` inside `src/cortext.cpp`.
- Tests: add `tests/operations_<name>.test.cpp` or extend a nearby integration test if the step only makes sense in sequence.

**New Persistence Logic:**
- Schema/migrations: update `src/store/schema.cpp`.
- Store helpers: add code in `src/store/` and public contracts in `include/cortext/store/` only if the API truly needs to be public.
- Tests: update `tests/migration_core.test.cpp`, `tests/store.test.cpp`, or `tests/state_persistence.test.cpp`.

**New Public API Surface:**
- C++ API: extend `include/cortext/cortext.hpp` and implement in `src/cortext.cpp`.
- C ABI: mirror the capability in `include/cortext/capi.h` and `src/capi.cpp`.
- Bindings: propagate the change to `bindings/python/cortext/__init__.py`, `bindings/go/cortext.go`, and `bindings/javascript/src/addon.cpp` only after the C API is in place.

**New Example or Benchmark:**
- Example app: add a new `examples/<name>/` directory with its own `CMakeLists.txt`, then include it from `examples/CMakeLists.txt`.
- Benchmark variant: add a new `examples/benchmark/<name>.cpp` and register a target in `examples/benchmark/CMakeLists.txt`.

**Utilities:**
- Shared helpers for the library: prefer `src/core/`, `include/cortext/core/`, or the owning subsystem instead of creating a generic catch-all.
- Offline scripts: place automation in `scripts/`.
- Small compiled developer tools: place them in `tools/<tool>/`.

## Special Directories

**`build/`:**
- Purpose: Generated CMake, dependency, test, and binary output trees.
- Generated: Yes
- Committed: No

**`.planning/codebase/`:**
- Purpose: Generated architectural reference docs for GSD workflows.
- Generated: Yes
- Committed: Project-dependent, currently present in the working tree.

**`docs/paper/_manuscript/`:**
- Purpose: Generated manuscript output from Quarto.
- Generated: Yes
- Committed: Yes

**`models/`:**
- Purpose: Runtime-discovered local models used by encoder and deep LLM factories.
- Generated: No
- Committed: Yes

**`data/`:**
- Purpose: Research datasets, prepared corpora, centroids, and evaluation artifacts.
- Generated: Mixed
- Committed: Yes

**`objects/`:**
- Purpose: Repository-local object-store data used by the sqlite object-storage workflow.
- Generated: Yes in practice
- Committed: Yes in the current tree

## Practical Placement Rules

- Put new library code in `src/` and `include/cortext/`; do not add new core logic under `examples/`.
- Keep examples, tools, and bindings thin. If logic is useful outside one executable, move it into the library first.
- Treat `include/cortext/` as the supported external boundary. Reaching into `src/` is already done by examples/tests, but avoid extending that pattern for new reusable features.
- When you add a subsystem file under `src/`, update the `add_library(cortext ...)` source list in `CMakeLists.txt`.
- When you add a new optional executable, gate it behind the existing example/tool structure instead of expanding the default core build.

---

*Structure analysis: 2026-04-07*
