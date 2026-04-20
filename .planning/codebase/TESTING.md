# Testing Patterns

**Analysis Date:** 2026-04-07

## Test Framework

**Runner:**
- Catch2 `v3.5.3`
- Config: `tests/CMakeLists.txt`
- The test target is `cortext_tests`, linked against `Catch2::Catch2WithMain` in `tests/CMakeLists.txt`.
- CTest runs the full suite as a single entry named `cortext_tests` because Catch2 discovery was intentionally avoided in `tests/CMakeLists.txt`.

**Assertion Library:**
- Catch2 macros and matchers: `REQUIRE`, `CHECK`, `REQUIRE_THROWS_AS`, `Catch::Approx`, and `Catch::Matchers::WithinAbs` in files such as `tests/store.test.cpp`, `tests/operations_focus.test.cpp`, and `tests/state_persistence.test.cpp`.

**Run Commands:**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build -R cortext_tests --output-on-failure
./build/examples/topical_chat_analysis/cortext_topical_chat_analysis --help
python scripts/run_memory_harness.py --max-conversations 2 --max-turns 360 --max-total 720 --no-multi
```
- No dedicated watch-mode or coverage target is configured in `tests/CMakeLists.txt` or the root `CMakeLists.txt`.

## Test File Organization

**Location:**
- Tests live in the flat `tests/` directory, not co-located with implementation files.
- Support helpers are centralized in `tests/test_helpers.hpp`.
- Current breadth is large: `tests/` contains about 75 `*.test.cpp` files and about 463 `TEST_CASE` blocks.

**Naming:**
- Use `<subsystem>.test.cpp` for unit-style coverage: `tests/store.test.cpp`, `tests/operation_set.test.cpp`, `tests/operations_focus.test.cpp`.
- Prefix multi-step pipeline tests with `integration_`: `tests/integration_consolidation.test.cpp`, `tests/integration_chat_e2e.test.cpp`, `tests/integration_working_memory_manual.test.cpp`.
- Use dedicated names for regression and formula locks: `tests/formula_validation.test.cpp`, `tests/regression_behavior.test.cpp`, `tests/operations_adherence_fixes.test.cpp`.

**Structure:**
```text
tests/
├── test_helpers.hpp
├── operation_set.test.cpp
├── store.test.cpp
├── operations_*.test.cpp
├── integration_*.test.cpp
├── regression_behavior.test.cpp
└── formula_validation.test.cpp
```

## Test Structure

**Suite Organization:**
```cpp
TEST_CASE ("UpdateFocus EWMA toward observed cosine and clamping",
           "[operations][focus]")
{
  Signal s;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  OperationContext ctx (s, pctx, cfg);
  InitializeFocusPriors init;
  init.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.weight_relevance > 0.0);
}
```

**Patterns:**
- Use descriptive sentence-style `TEST_CASE` names plus tag groups that match the subsystem, for example `[operations][focus]`, `[integration][e2e][chat]`, `[store][wal]`, `[schema][migration]`, and `[deep_llm][integration]`.
- Use `SECTION` to cover branches inside a single fixture when the setup cost is low, as in `tests/store.test.cpp`, `tests/regression_behavior.test.cpp`, and `tests/chat_chunk_diagnostics.test.cpp`.
- Operation tests usually create `Signal`, `ProcessorContext`, `SignalProcessor::Config`, then call `cortext::testing::RequireEncoder (cfg)` and build an `OperationContext`. See `tests/operations_focus.test.cpp`, `tests/operations_stability.test.cpp`, and `tests/operations_streaming_pacing.test.cpp`.
- Database-backed tests usually create an in-memory store with `SQLiteStore::Create (":memory:")` and then initialize real schema with `cortext::testing::InitializeCoreSchema (*store)`. See `tests/operations_graph_build.test.cpp`, `tests/operations_graph_retrieval.test.cpp`, and `tests/operations_emotion.test.cpp`.
- Teardown is mostly RAII. Temp resources clean themselves up through helpers like `TempDatabase` in `tests/store.test.cpp`, `TempDir` in `tests/deep_llm_factory.test.cpp`, and `ScopedEnvVar` in `tests/test_helpers.hpp`.

## Mocking

**Framework:** Handwritten test doubles and fakes

**Patterns:**
```cpp
class TestEncoder : public Encoder
{
public:
  void EncodeText (const std::string &, std::vector<float> &out) override
  {
    out.assign (256, 0.0f);
    out[0] = 1.0f;
  }
};
```

**What to Mock:**
- Mock model-facing dependencies with tiny deterministic implementations instead of using a mocking library. Examples: `TestEncoder` in `tests/test_helpers.hpp`, `CapturingSummarizer` in `tests/integration_consolidation.test.cpp`, and `KeywordEncoder` / `StubSummarizer` / `KeywordExtractor` in `tests/integration_chat_e2e.test.cpp`.
- Fake environment-driven behavior with scoped env-var wrappers such as `ScopedEnvVar` in `tests/test_helpers.hpp` and `tests/deep_llm_factory.test.cpp`.
- Fake pipeline steps by creating small `IOperation` subclasses in the test file, for example `RecordOrderOp` in `tests/operation_set.test.cpp` and `TriggerBoundaryOp` in `tests/state_persistence.test.cpp`.

**What NOT to Mock:**
- Do not mock the SQLite layer for persistence, migration, or retrieval logic. Tests usually hit a real in-memory SQLite database through `cortext::SQLiteStore`.
- Do not mock the operation pipeline when validating algorithm interactions. Integration tests build real `OperationSet` pipelines and execute them end-to-end in `tests/integration_chat_e2e.test.cpp` and `tests/integration_working_memory_manual.test.cpp`.

## Fixtures and Factories

**Test Data:**
```cpp
cortext::testing::InitializeCoreSchema (*store);
cortext::testing::SeedEmbeddingV2 (*store, 10LL, embedding, 1000LL);
cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1");
cortext::testing::SeedSignalV2 (*store, 30LL, 10LL, "chat", 2000LL, 0.5);
```

**Location:**
- Shared schema/data helpers live in `tests/test_helpers.hpp`: `InitializeCoreSchema`, `SeedEmbeddingV2`, `SeedMemoryV2`, `SeedMemoryV2Extended`, `SeedSignalV2`, `GetInt64`, `GetDouble`, `RequireEncoder`, and `GetNullTransaction`.
- File-specific builders usually live in an anonymous namespace at the top of the test file, for example `MakeSignal`, `MakeEmbedding`, and `SeedMemory` in `tests/integration_consolidation.test.cpp`.
- Complex test-specific resource wrappers stay local to the test file when they are not broadly reusable, for example `TempDatabase` in `tests/store.test.cpp` and `TempDir` in `tests/deep_llm_factory.test.cpp`.

## Coverage

**Requirements:** None enforced by tooling

**View Coverage:**
```bash
ctest --test-dir build -R cortext_tests --output-on-failure
```
- There is no coverage report target, lcov configuration, or CI coverage threshold in the checked-in CMake files.
- Practical coverage is breadth-oriented: operations, store, migrations, persistence, chat pipeline behavior, regressions, and formula locks all have direct tests in `tests/`.

## Test Types

**Unit Tests:**
- Unit tests target focused algorithmic behavior or a single component boundary. Examples: `tests/operations_focus.test.cpp`, `tests/operations_threshold.test.cpp`, `tests/core_knobs.test.cpp`, `tests/json_schema_constraint.test.cpp`, and `tests/telemetry_noop_by_default.test.cpp`.
- These tests usually avoid filesystem or network dependencies and rely on `GetNullTransaction`, deterministic embeddings, and hand-built state.

**Integration Tests:**
- Integration tests validate multi-step processing, persistence, or retrieval across the real store and real pipeline pieces. Examples: `tests/integration_consolidation.test.cpp`, `tests/integration_chat_e2e.test.cpp`, `tests/integration_working_memory_manual.test.cpp`, `tests/state_persistence.test.cpp`, and `tests/store_extensions.test.cpp`.
- Integration coverage commonly uses real schema migrations, real `OperationSet` composition, and real SQL queries against in-memory SQLite.

**E2E Tests:**
- There is no separate browser/UI E2E framework.
- The closest E2E coverage is full-pipeline conversational processing in `tests/integration_chat_e2e.test.cpp` plus the standalone executable `examples/topical_chat_analysis/main.cpp`.

## Benchmark and Validation Workflows

**Snapshot/Scenario Validation:**
- `examples/topical_chat_analysis/main.cpp` is the main full-pipeline validation executable. It links `cortext::cortext`, emits metrics/logs, and is the binary targeted by the repo’s documented analysis commands.
- `scripts/run_topical_chat_snapshot.sh` runs one deterministic scenario and captures `config.json`, `run.log`, `summary.csv`, `perf.json`, `git_status.txt`, and `git_diff.patch` under `logs/topical_chat_snapshots/...`.
- `scripts/run_topical_chat_sweep.sh` runs multiple F/S/T knob combinations and writes a `summary.csv` with retrieval, interruption, consolidation, and performance metrics.

**Harness/Batch Validation:**
- `scripts/run_memory_harness.py` automates baseline, multi-participant, and sweep runs against `build/examples/topical_chat_analysis/cortext_topical_chat_analysis`. Each case writes `config.json`, `run.log`, `status.json`, `returncode.txt`, and feeds a cumulative `summary.csv`.
- The harness records git state with `git_rev.txt`, `git_status.txt`, and `git_diff.patch`, so experiment results remain traceable to the exact checkout.
- Determinism controls are built into the shell runners: `--seed`, `--deterministic`, and `--synthetic-start-ms` are threaded through `scripts/run_topical_chat_snapshot.sh` and `scripts/run_topical_chat_sweep.sh`.

**Benchmarks:**
- Bench binaries are declared in `examples/benchmark/CMakeLists.txt`. The repo keeps many focused validation executables rather than one benchmark binary only.
- Representative examples include `examples/benchmark/storage_human_sleep_token_bench.cpp`, `examples/benchmark/storage_human_sleep_consolidation_bench.cpp`, `examples/benchmark/bitemporal_retrieval_bench.cpp`, and `examples/benchmark/meta_learning_ablation_bench.cpp`.
- Bench code usually uses synthetic deterministic encoders and real schema/store access, as shown in `examples/benchmark/storage_human_sleep_token_bench.cpp`.

## Common Patterns

**Async / Long-Running Testing:**
```cpp
auto out = processor.Process (MakeTextSignal (encoder, text, ts, source_id));
ts += 4000000;  // Synthetic idle gap to trigger deferred behavior
processor.Process (MakeTextSignal (encoder, "idle", ts, "cortext/idle"));
```
- Most “async” behavior is tested synchronously with synthetic timestamps, explicit idle gaps, stop tokens, or repeated process calls rather than threads or sleeps. See `tests/integration_chat_e2e.test.cpp`, `tests/operations_threshold.test.cpp`, and `tests/operations_streaming_pacing.test.cpp`.
- For long external validations, the repo relies on standalone scripts and harnesses instead of embedding those runs in Catch2.

**Error Testing:**
```cpp
REQUIRE_THROWS_AS (store->Execute ("INVALID SQL QUERY", {}),
                   cortext::StoreError);
```
- Error paths are verified directly with exception assertions in `tests/store.test.cpp`, `tests/phi4_extractor.test.cpp`, `tests/phi4_summarizer.test.cpp`, and `tests/deep_llm_factory.test.cpp`.
- Soft-failure and no-op contracts are validated explicitly, for example `tests/telemetry_noop_by_default.test.cpp` and empty-input checks in `tests/phi4_summarizer.test.cpp`.

## How Changes Are Typically Verified

**Algorithm or Threshold Change:**
- Add or update the closest focused operation test in `tests/operations_*.test.cpp`.
- If the behavior is formula-driven, update `tests/formula_validation.test.cpp` and `tests/regression_behavior.test.cpp`.
- Run `ctest --test-dir build -R cortext_tests --output-on-failure`.

**Pipeline, Retrieval, Consolidation, or Persistence Change:**
- Update/add an integration test in `tests/integration_consolidation.test.cpp`, `tests/integration_chat_e2e.test.cpp`, `tests/state_persistence.test.cpp`, or a neighboring integration file.
- Re-run the same `ctest` command after rebuilding.
- Validate behavior with `examples/topical_chat_analysis/main.cpp` directly or through `scripts/run_topical_chat_snapshot.sh` / `scripts/run_memory_harness.py` when the change affects long-horizon behavior or emitted metrics.

**Performance or Long-Horizon Behavior Change:**
- Prefer the benchmark binaries in `examples/benchmark/` or the harness/sweep scripts in `scripts/`.
- Preserve determinism and capture the generated logs and CSV summaries so runs remain comparable.

---

*Testing analysis: 2026-04-07*
