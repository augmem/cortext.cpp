# Coding Conventions

**Analysis Date:** 2026-04-07

## Naming Patterns

**Files:**
- Use lower_snake_case for C++ source and header files. Public/private pairs usually mirror each other, for example `src/operations/focus.cpp` with `include/cortext/operations/focus.hpp`, and tests follow the same stem with `.test.cpp`, for example `tests/operations_focus.test.cpp`.
- Keep subsystem prefixes in filenames so related files sort together: `src/store/schema.cpp`, `src/store/facts.cpp`, `tests/store.test.cpp`, `tests/store_extensions.test.cpp`.
- Internal-only helpers are explicit in the filename instead of hidden behind umbrella headers, for example `src/operations/meta_learning_internal.hpp`, `src/operations/constructive_recall_internal.hpp`, and `src/operations/eviction_ablation.hpp`.

**Functions:**
- In the library and most tests, use PascalCase for functions and methods, including file-local helpers: `NowMillis`, `ParseDbOperation`, `LoadObjstorePayload`, `InitializeCoreSchema`, `SeedEmbeddingV2`, and `IOperation::Execute`.
- Preserve local style when a file already uses a different convention. Some tests and scripts use lower_snake_case helpers such as `create_temp_db`, `cleanup_temp_db`, `parse_metrics`, and `run_case` in `tests/store.test.cpp` and `scripts/run_memory_harness.py`.
- Operation classes expose work through `Execute (OperationContext &, Transaction &) const` as defined in `include/cortext/processor/operation.hpp`.

**Variables:**
- Use lower_snake_case for locals, parameters, and fields: `observed_cosine`, `ctx_window_size`, `db_path_`, `had_value_`, `weight_relevance_prior`, and `interrupt_gate_blocked_no_store`.
- Member fields commonly end with `_`, especially RAII wrappers and private state, for example `name_`, `old_value_`, `db_path_`, and `impl_`.
- Constants use `k` + PascalCase, for example `kEmbeddingDim`, `kHourMs`, `kHumanHalfLifeSeconds`, and `kCoverageGainFloorBase`.
- Boolean names read like predicates or state flags: `focus_priors_initialized`, `should_interrupt`, `interrupt_aborted`, `reinforcement_enabled`.

**Types:**
- Use PascalCase for classes, structs, and interfaces: `Cortext`, `SignalProcessor`, `SQLiteConnection`, `TempDatabase`, `ScopedEnvVar`, and `IOperation`.
- Namespace names are lower-case subsystem names, usually nested under `cortext`: `cortext::operations`, `cortext::store`, `cortext::testing`, `cortext::telemetry`, and `cortext::internal`.
- Test doubles follow the same type naming as production code: `TestEncoder`, `CapturingSummarizer`, `KeywordEncoder`, and `TriggerBoundaryOp`.

## Code Style

**Formatting:**
- No repo-level formatter config was detected. `.clang-format`, `.clang-tidy`, and `.editorconfig` are not present at the repository root.
- The dominant library style in `src/` and `include/` uses 2-space indentation, opening braces on the next line, and spaces before parentheses: see `src/store.cpp`, `src/cortext.cpp`, `src/operations/focus.cpp`, and `include/cortext/cortext.hpp`.
- Tests in `tests/operations_focus.test.cpp`, `tests/store.test.cpp`, and `tests/integration_consolidation.test.cpp` generally follow the same Allman-style formatting as the library.
- Some example and app-facing code uses a different local dialect with tighter spacing and same-line braces. Preserve the local file style when editing `examples/topical_chat_analysis/main.cpp` and `tests/chat_chunk_diagnostics.test.cpp` instead of normalizing them to the core style.

**Linting:**
- Formatting is enforced mostly by review and by matching nearby code, not by a checked-in formatter.
- Compiler warnings are the practical style gate. `CMakeLists.txt` enables `-Wall -Wextra -Wpedantic` for non-MSVC builds and `/W4` for MSVC.
- `CORTEXT_WARNINGS_AS_ERRORS` defaults to `ON` in `CMakeLists.txt`, so library changes should be written as warning-clean by default.
- Sanitizers are opt-in quality checks in `CMakeLists.txt`: `CORTEXT_ENABLE_ASAN`, `CORTEXT_ENABLE_UBSAN`, and `CORTEXT_ENABLE_MSAN`.

## Import Organization

**Order:**
1. Put the paired public header or the closest project header first, for example `#include "cortext/operations/focus.hpp"` in `src/operations/focus.cpp` and `#include "cortext/cortext.hpp"` in `src/cortext.cpp`.
2. Follow with related project headers and local/internal headers, often grouped by subsystem, for example `meta_learning_internal.hpp` plus `cortext/core/...` and `cortext/processor/...` includes in `src/operations/focus.cpp`.
3. Put third-party and standard library includes last, for example `<Eigen/Dense>`, `<chrono>`, `<memory>`, and `<vector>` in `src/cortext.cpp` and `src/store.cpp`.

**Path Aliases:**
- Public headers are included with the installed-style prefix, for example `<cortext/processor.hpp>`, `<cortext/store/sqlite_store.hpp>`, and `<cortext/operations/focus.hpp>`.
- Internal-only test coverage sometimes reaches into non-public code with relative includes when there is no public seam. Examples: `tests/deep_llm_factory.test.cpp` includes `../src/deep_llm/deep_llm_factory.hpp`, and `tests/chat_chunk_diagnostics.test.cpp` includes `../examples/chat/chunk_diagnostics.hpp`.
- There are no alias macros or umbrella headers acting as barrel files. Include the exact header you need.

## Error Handling

**Patterns:**
- Throw typed or standard exceptions for hard failures in low-level components. `src/store.cpp` throws `StoreError` on SQLite prepare/open failures, and `tests/deep_llm_factory.test.cpp` expects `std::runtime_error` for invalid backend overrides.
- Catch exceptions at system boundaries when the code can degrade gracefully, then emit telemetry instead of crashing. `LoadObjstorePayload` and `LoadSignalBlobs` in `src/cortext.cpp` catch `std::exception` and log warnings before returning `false`.
- Use `std::optional`, empty containers, or boolean return values for absence and best-effort behavior, for example `Context::ProcessorOutput` fields in `include/cortext/cortext.hpp` and the `bool` returns in `src/cortext.cpp`.
- Mark intentionally unused parameters explicitly with `(void)tx;` in operation implementations such as `src/operations/focus.cpp`.

## Logging

**Framework:** `cortext::telemetry`

**Patterns:**
- Library code uses the wrapper in `include/cortext/telemetry/telemetry.hpp` and `src/telemetry/telemetry.cpp` instead of ad hoc `std::cout` logging.
- Emit structured event names and attributes rather than interpolated strings. Examples: `telemetry::LogDebug ("cortext.focus.init", {...})` and `telemetry::LogWarn ("Failed to load signal blobs", {...})` in `src/operations/focus.cpp` and `src/cortext.cpp`.
- Telemetry is safe to call even when no SDK provider is installed. `tests/telemetry_noop_by_default.test.cpp` verifies the no-op path.
- Example binaries and scripts may still print to stdout or log files directly, for example `examples/topical_chat_analysis/main.cpp` and `scripts/run_memory_harness.py`.

## Comments

**When to Comment:**
- Use `/// @brief` comments for public headers and reusable helpers where callers need contract-level guidance, as seen in `include/cortext/cortext.hpp`, `include/cortext/store/schema.hpp`, and `include/cortext/processor/operation.hpp`.
- Use short `//` comments for rationale, algorithm references, schema blocks, and test setup notes. Good examples are `src/store/schema.cpp`, `tests/formula_validation.test.cpp`, `tests/regression_behavior.test.cpp`, and `tests/operations_threshold.test.cpp`.
- Prefer comments that explain why a step exists or which paper/spec rule it traces back to. Avoid line-by-line narration.

**JSDoc/TSDoc:**
- Doxygen-style `/// @brief`, `/// @param`, and `/// @return` comments are the main documentation pattern for C++ APIs and test helpers, not block comments or generated doc annotations from another toolchain.

## Function Design

**Size:** 
- Keep file-local helpers narrow and focused in anonymous namespaces. `src/store.cpp` splits query work into `ParseDbOperation`, `PrepareStatement`, `BindParameters`, and `FetchResultRow` instead of a single large function.
- Complex workflows are composed from small operation objects rather than one monolith. `src/cortext.cpp` wires many `IOperation` implementations together instead of embedding the algorithm logic inline.

**Parameters:** 
- Favor explicit domain objects over long primitive parameter lists for processing code: operations receive `OperationContext &` and `Transaction &`, and high-level APIs use `Cortext::Config`, `SignalProcessor::Config`, and typed structs in `include/cortext/cortext.hpp`.
- Helper functions in tests and benchmarks often accept plain values when seeding deterministic state, for example `SeedMemoryV2`, `SeedSignalV2`, `MakeSignal`, and `SeedLongTermMemory`.

**Return Values:** 
- Pure helpers return plain values or `std::optional` where absence is meaningful, for example `NowMillis`, `ToMillis`, `FindModelPath`, and `Context::ProcessTextAt`.
- Stateful operations usually mutate context and transaction state rather than returning values. Follow the `IOperation` contract unless you are writing a pure helper outside the pipeline.

## Module Design

**Exports:** 
- Public surface area lives under `include/cortext/` and is marked with `CORTEXT_EXPORT` when needed, for example `include/cortext/cortext.hpp`.
- Keep internal implementation details in `src/` or internal headers such as `src/deep_llm/deep_llm_factory.hpp` and `include/cortext/internal/cancellation.hpp`.
- Do not widen the public API accidentally. Tests are willing to include internal headers directly when coverage needs it.

**Barrel Files:** 
- Not used. Headers are imported directly by subsystem path, for example `include/cortext/operations/focus.hpp` and `include/cortext/store/sqlite_store.hpp`.
- When adding a new operation or subsystem, create a matching header/source pair and include it explicitly from the call sites that need it.

---

*Convention analysis: 2026-04-07*
