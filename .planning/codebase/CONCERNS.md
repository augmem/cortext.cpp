# Codebase Concerns

**Analysis Date:** 2026-04-07

## Tech Debt

**Core orchestration is concentrated in a few very large files:**
- Issue: The main runtime path is spread across several multi-thousand-line files that mix policy, persistence, telemetry, and feature toggles. `src/operations/graph_retrieval.cpp` handles seed selection, fact retrieval, graph expansion, diversification, metacognition, reconstruction writes, and debug logging in one unit. `src/signal_processor.cpp` and `src/cortext.cpp` both aggregate many responsibilities instead of acting as thin coordinators.
- Files: `src/operations/graph_retrieval.cpp`, `src/signal_processor.cpp`, `src/cortext.cpp`
- Impact: Small behavior changes are hard to isolate, regression scope is large, and debugging usually requires understanding several unrelated branches in the same file.
- Fix approach: Split each file by pipeline stage. Treat retrieval seeding, fact-layer scoring, graph expansion, and post-selection reinforcement as separate translation units with narrow helpers and local tests.

**Chat/demo code is effectively a second application inside the repo:**
- Issue: The example app contains production-like concerns including OpenAI streaming, OpenTelemetry export, persistence, prompt assembly, UI state, and voice control. `examples/chat/main.cpp` and `examples/chat/chat_window.cpp` are both large enough to behave like application roots rather than examples. `examples/chat/voice_session.mm` adds a large Objective-C++ concurrency surface on top.
- Files: `examples/chat/main.cpp`, `examples/chat/chat_window.cpp`, `examples/chat/voice_session.mm`, `examples/chat/streaming_client.cpp`
- Impact: Demo changes are expensive to review, operationally risky, and likely to break independently of the core library.
- Fix approach: Move chat runtime, persistence, streaming, telemetry, and voice backend code into reusable modules under `src/` or `examples/chat/` subdirectories with separate ownership boundaries and tests.

**Schema evolution remains append-only and centralized:**
- Issue: `src/store/schema.cpp` keeps the entire schema history in a single compiled migration list, including the full v2 schema and later `ALTER TABLE` additions. The model is forward-only and there is no downgrade, compatibility shim, or data backfill validation beyond “statement applied successfully”.
- Files: `src/store/schema.cpp`
- Impact: Schema changes are high-stakes because a migration mistake lands in the main bootstrap path and is hard to correct for already-created databases.
- Fix approach: Keep migrations small, introduce migration validation checks, and add database fixture coverage for upgrade paths from every published schema version.

**Build logic is heavily duplicated across optional backends:**
- Issue: `CMakeLists.txt` contains repeated local-inference setup for Gemma, OGA, sherpa-onnx, and LiteRT-LM. The same external dependency families are configured in multiple places with similar but not identical assumptions.
- Files: `CMakeLists.txt`
- Impact: Build fixes can diverge by feature flag combination, and platform-specific failures are hard to reproduce reliably.
- Fix approach: Consolidate backend bootstrap into shared CMake functions and reduce duplicate ORT configuration branches.

**Current churn is concentrated in demo and memory paths:**
- Issue: The working tree is currently dirty in the chat/demo stack and memory orchestration path, including `examples/chat/main.cpp`, `examples/chat/chat_window.cpp`, `src/cortext.cpp`, `src/operations/consolidation_summarize.cpp`, `src/operations/working_memory.cpp`, and the `third_party/sherpa-onnx` submodule. New untracked voice-related files are also present.
- Files: `examples/chat/main.cpp`, `examples/chat/chat_window.cpp`, `examples/chat/chat_window.hpp`, `examples/chat/voice_backend_bench.cpp`, `examples/chat/voice_session.cpp`, `examples/chat/voice_session.hpp`, `examples/chat/voice_session.mm`, `src/cortext.cpp`, `src/operations/consolidation_summarize.cpp`, `src/operations/working_memory.cpp`, `third_party/sherpa-onnx`
- Impact: The most volatile code overlaps with the least isolated runtime surface, which raises merge risk and makes root-cause analysis harder.
- Fix approach: Land chat/voice work in smaller slices, keep the submodule state pinned and clean, and separate demo-only changes from library behavior changes.

## Known Bugs

**Voice chat is unavailable outside the macOS implementation path:**
- Symptoms: Non-macOS builds instantiate `VoiceSession` successfully but `IsSupported()` returns `false`, `HasSpeakerAttribution()` returns `false`, and `Start()` emits an error that voice chat is only implemented for macOS builds.
- Files: `examples/chat/voice_session.cpp`, `examples/chat/voice_session.mm`
- Trigger: Build or run the chat example on a platform that does not compile the Objective-C++ implementation.
- Workaround: Use text-only chat, or build on macOS with the required sherpa/whisper dependencies present.

**CI/test failures are reported as one coarse test result:**
- Symptoms: The test target is discovered as a single `cortext_tests` CTest entry instead of many individual cases.
- Files: `tests/CMakeLists.txt`
- Trigger: Any test failure in `cortext_tests`.
- Workaround: Run the binary directly with Catch2 filters while debugging, because CTest output does not preserve per-case granularity.

**Chat stream state can grow without an upper bound during long or malformed responses:**
- Symptoms: The streaming client accumulates `raw_body`, `buffer`, and `full_content` for the entire request, and there is no overall transfer timeout.
- Files: `examples/chat/streaming_client.cpp`
- Trigger: Very long responses, servers that never terminate the stream, or SSE payloads that keep producing content without a clean finish.
- Workaround: Cancel the request manually; there is no built-in hard cap on buffered response size.

## Security Considerations

**Dynamic SQLite extension loading is enabled by default and accepts environment-driven paths:**
- Risk: `TryLoadDynamicExtensions()` enables extension loading on the connection, accepts `SQLITE_VEC_PATH`, and otherwise probes relative library paths under the repo. That is a broader attack surface than the embedded extension path.
- Files: `src/store/extension_loader.cpp`, `CMakeLists.txt`
- Current mitigation: Embedded `sqlite-vec` and `sqlite-objstore` are compiled in by default, and extension loading is disabled again after probing.
- Recommendations: Default `CORTEXT_ENABLE_DYNAMIC_EXTENSIONS` to `OFF`, log every attempted dynamic load, and restrict override paths to trusted absolute locations in non-dev builds.

**The chat example writes raw telemetry and conversation-adjacent data to disk:**
- Risk: The example always installs a stdout/file log exporter and writes to `examples/chat/logs.txt`. The same runtime also persists settings next to the database path by default. This can expose prompts, retrieved-memory previews, voice diagnostics, and operational metadata in a local checkout.
- Files: `examples/chat/main.cpp`, `examples/chat/chat_window.cpp`
- Current mitigation: None beyond local filesystem access controls.
- Recommendations: Make file logging opt-in, redact user/assistant content from default logs, and store settings/logs in an explicit application data directory rather than the repo tree.

**Silent exception handling hides storage and retrieval failures:**
- Risk: Several paths catch and suppress exceptions with no escalation. Examples include startup directory creation in `examples/chat/main.cpp`, graph expansion/fetch branches in `src/operations/graph_retrieval.cpp`, cleanup in `src/store.cpp`, and payload hydration fallbacks in `src/cortext.cpp`.
- Files: `examples/chat/main.cpp`, `src/operations/graph_retrieval.cpp`, `src/store.cpp`, `src/cortext.cpp`
- Current mitigation: Some paths emit telemetry warnings, but several catch-all blocks intentionally ignore the error.
- Recommendations: Replace `catch (...) {}` with structured error reporting and make storage/bootstrap failures explicit when they affect correctness.

## Performance Bottlenecks

**Retrieval performs several expensive database passes per query:**
- Problem: `src/operations/graph_retrieval.cpp` issues a seed KNN query, fact seed query, fact evidence query, stale fact query, recursive graph expansion, summary association query, summary label query, embedding fetch, and optional reconstruction reads/writes in one execution path.
- Files: `src/operations/graph_retrieval.cpp`
- Cause: The retrieval stack combines semantic search, fact provenance, recursive graph traversal, context diversification, and reconstruction ledger updates synchronously.
- Improvement path: Cache stable joins, push more ranking work into precomputed tables, add query-level profiling thresholds, and separate fact-layer expansion from the latency-sensitive retrieval hot path.

**Fact lifecycle maintenance scales by repeated per-record recomputation and re-embedding:**
- Problem: `MaintainFactLifecycle()` sweeps up to 256 facts per call, and `RecomputeFactLifecycle()` may recompute evidence aggregates, contradictions, cache rows, and text embeddings individually.
- Files: `src/store/facts.cpp`
- Cause: The lifecycle pass is row-oriented and recomputes derived state on demand rather than batching updates.
- Improvement path: Batch aggregate queries, only refresh embeddings when text changes, and move maintenance to a background/maintenance command instead of piggybacking on interactive paths.

**The chat database explorer runs many live queries on refresh:**
- Problem: `RefreshDatabaseExplorer()` issues multiple `COUNT(*)` queries plus recent-row fetches for memories, signals, associations, episodes, facts, and evictions every refresh.
- Files: `examples/chat/main.cpp`
- Cause: The database explorer is built directly on synchronous store calls with no throttling or memoization.
- Improvement path: Refresh on demand, cache snapshots, and move expensive explorer queries off the UI thread.

**The chat stream path retains full payloads and full responses in memory:**
- Problem: `StreamingChatClient::Stream()` keeps both the raw SSE body and the full accumulated assistant response.
- Files: `examples/chat/streaming_client.cpp`
- Cause: The streaming path is optimized for convenience and diagnostics, not bounded memory usage.
- Improvement path: Cap retained raw bodies, stream tokens directly to consumers, and keep only the final assistant content needed by the caller.

## Fragile Areas

**Storage bootstrap depends on optional SQLite capabilities lining up exactly:**
- Files: `src/store/schema.cpp`, `src/store/extension_loader.cpp`, `src/store.cpp`
- Why fragile: The schema assumes `objstore()` and `vec0()` are available when migrations run. Build flags, dynamic loading behavior, and platform-specific extension registration must all align for the database to initialize cleanly.
- Safe modification: Treat storage changes as end-to-end changes across CMake, extension loading, schema migration, and store tests.
- Test coverage: `tests/store.test.cpp`, `tests/store_extensions.test.cpp`, and `tests/migration_core.test.cpp` cover happy paths, but they do not exercise every feature-flag combination from `CMakeLists.txt`.

**Graph retrieval has multiple silent fallback branches:**
- Files: `src/operations/graph_retrieval.cpp`
- Why fragile: Expansion, summary-association fetches, label fetches, and embedding fetches all have `catch (...)` fallbacks. That keeps the system alive, but it also means result quality can degrade with little visibility.
- Safe modification: Add one change at a time, instrument every fallback, and compare retrieval metrics before and after edits.
- Test coverage: `tests/operations_graph_retrieval.test.cpp` and `tests/operations_graph_retrieval_bitemporal.test.cpp` cover core behavior, but they do not assert telemetry emitted on degraded paths.

**Voice handling combines threads, atomics, platform APIs, and model backends in one file:**
- Files: `examples/chat/voice_session.mm`, `examples/chat/voice_session.hpp`, `examples/chat/voice_session.cpp`
- Why fragile: The implementation manages capture, ASR, diarization, TTS, playback cancellation, and speaker tracking across multiple threads with shared mutable state.
- Safe modification: Isolate backend-specific code paths, add deterministic harnesses around queue/cancel behavior, and avoid mixing UI state changes with audio-thread work.
- Test coverage: No automated tests target `examples/chat/voice_session.mm` or `examples/chat/voice_session.cpp`.

**Prompt assembly and memory injection logic live inside the example runtime loop:**
- Files: `examples/chat/main.cpp`, `examples/chat/chat_window.cpp`
- Why fragile: Retrieved-memory filtering, chunk diagnostics, prompt shaping, idle consolidation control, and persistence are coupled inside the chat app event loop.
- Safe modification: Extract prompt building and memory filtering into testable helpers before changing retrieval or prompt policy.
- Test coverage: `tests/integration_chat_e2e.test.cpp`, `tests/chat_chunk_diagnostics.test.cpp`, `tests/chat_metrics_state.test.cpp`, and `tests/chat_stream_usage.test.cpp` cover parts of the chat stack, but not the full UI/runtime loop.

## Scaling Limits

**Feature-flag combinations are likely to outpace practical validation:**
- Current capacity: The repo supports many optional stacks: embedded SQLite extensions, Gemma ORT, OGA, sherpa-onnx, LiteRT-LM, and platform-specific voice backends.
- Limit: The matrix in `CMakeLists.txt` is larger than the exercised test/build matrix visible in `tests/CMakeLists.txt`.
- Scaling path: Define a supported subset of build profiles and automate those exact combinations in CI before adding more optional paths.

**Interactive retrieval latency will rise with memory and fact volume:**
- Current capacity: Retrieval currently depends on repeated KNN, fact, and association queries plus memory hydration from `objstore`.
- Limit: As `memories`, `fact_assertions`, `fact_evidence`, and `memory_reconstructions` grow, the multi-pass retrieval strategy in `src/operations/graph_retrieval.cpp` and hydration in `src/cortext.cpp` become more expensive.
- Scaling path: Add persisted ranking summaries, background reconstruction maintenance, and latency budgets per retrieval phase.

## Dependencies at Risk

**Heavy external ML/runtime dependencies are a build and upgrade risk:**
- Risk: The project depends on vendored or downloaded ONNX Runtime, onnxruntime-genai, sherpa-onnx, LiteRT-LM, and Bazel-driven artifacts. Several of these are bootstrapped via `ExternalProject_Add`, downloaded archives, or submodules.
- Impact: Version drift, platform-specific ABI issues, and broken upstream builds can block local development even when Cortext code is unchanged.
- Migration plan: Minimize supported combinations, pin versions aggressively, and add repeatable bootstrap scripts per platform.

**`third_party/sherpa-onnx` is currently dirty:**
- Risk: The submodule is not at a clean recorded state.
- Impact: Reproducible builds and future upgrades are harder because local behavior may not match the committed repository state.
- Migration plan: Commit or discard the submodule delta explicitly and document the expected sherpa revision in the main repo.

## Missing Critical Features

**The example application does not have end-to-end coverage for its newest runtime surfaces:**
- Problem: The test suite includes chat-oriented tests, but there are no automated tests for `examples/chat/main.cpp`, `examples/chat/chat_window.cpp`, `examples/chat/streaming_client.cpp`, or the voice session files.
- Blocks: Safe iteration on UI behavior, settings persistence, streaming edge cases, and voice interaction.

**There is no dedicated maintenance path for long-running fact and reconstruction cleanup:**
- Problem: Fact lifecycle and reconstruction growth are maintained inline rather than through a dedicated maintenance command or background job.
- Blocks: Predictable operational behavior on large persistent databases.

## Test Coverage Gaps

**Demo runtime and UI paths are mostly untested:**
- What's not tested: The actual example app event loop, OpenTelemetry/file logging setup, persisted settings load/save flow, database explorer refresh behavior, and UI state transitions.
- Files: `examples/chat/main.cpp`, `examples/chat/chat_window.cpp`
- Risk: Demo regressions can ship unnoticed because the covered chat tests exercise helper behavior rather than the running application.
- Priority: High

**Voice and platform-specific audio behavior are untested:**
- What's not tested: ASR/TTS queue behavior, cancellation races, diarization fallback behavior, and backend switching.
- Files: `examples/chat/voice_session.mm`, `examples/chat/voice_session.cpp`, `examples/chat/voice_session.hpp`
- Risk: Race conditions and platform-only failures are likely to surface first in manual runs.
- Priority: High

**Build/bootstrap paths for optional backends are not verified by unit tests:**
- What's not tested: ExternalProject downloads, Bazel/LiteRT bootstrap, OGA header-copy path, and dynamic extension fallback behavior under different CMake flags.
- Files: `CMakeLists.txt`, `src/store/extension_loader.cpp`
- Risk: Configuration regressions can break clean-room builds without affecting already-prepared developer machines.
- Priority: Medium

---

*Concerns audit: 2026-04-07*
