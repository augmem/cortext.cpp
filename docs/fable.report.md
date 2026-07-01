# Fable Review Report

Date: 2026-07-01.
Scope: full first-party tree (`src/`, `include/`, `tests/`, `bindings/`, `scripts/`, `tools/`, `.github/`, `docs/paper/`, build system, third_party hygiene).
Method: six parallel subsystem reviews (core runtime, retrieval/consolidation, storage, API/bindings, tests/CI, paper/eval), each reading code rather than pattern-matching, followed by my own verification of every load-bearing claim. All findings below cite file:line. Ideas already tried and rejected in `docs/paper/sections/9_experimental.qmd`, the hard-cut removal lists, and the 18-arm mechanism sweep were filtered out of the recommendations (Section 9 lists what was excluded and why).

Status update, 2026-07-01: Section 2.1 / E1 / P0.1 is addressed. The graph-retrieval temporal term now uses bounded age decay in `src/operations/graph_retrieval.cpp`, the multi-month monotonic-decay regression test passes, and the 9-probe MSC Gemma4-12B-AWQ A/B is recorded in `docs/paper/sections/9_experimental.qmd`. The A/B was not a quality win, so it closes the correctness bug but not the broader sufficiency-gap work.

## 1. Executive summary

The v1 hard cut is real and mostly clean. Replay parity held up, the memory-safety patterns that caused the old reconsolidation UAF are correctly fixed, the C ABI fundamentals are solid, and the operations layer survived a hostile read with no critical defect. The problems that remain cluster in five places:

1. ~~**The temporal recency term in graph retrieval is dead code with an overflow inversion.**~~ **Resolved 2026-07-01.** The correctness bug is fixed and covered by a regression test. The follow-up MSC Gemma A/B showed a small performance improvement but a quality regression, so the temporal fix should not be claimed as a sufficiency-gap win.
2. **The engine grows disk without bound.** WAL auto-checkpointing is disabled by default and object-store blobs are never garbage-collected after eviction. For a system whose pitch is "runs for a year on a stream," both are release-blocking product defects even though no test catches them.
3. **CI does not gate what the release claims.** The zig-smoke job fails on every run (removed `-Dllama` option), native CI downloads the 141 MB model and then excludes every test that uses it, sanitizer options exist but no job runs them, and the paper's replay-parity evidence is enforced nowhere.
4. **The hard cut left debris.** A 12-file dead label-classifier script family, a tool that crashes on import, a large undocumented `CORTEXT_*` env-var ablation surface baked into the release binary, stale adapter-era claims in the paper index, and two referenced-but-nonexistent paper sections.
5. **The eval evidence is strong but attackable, and the cheapest reinforcements use artifacts you already have.** n=9 probes, one judge family, and a holistic winner criterion are the three angles a skeptic hits first. All three have zero- or low-cost fixes.

Sections 2 through 8 are the audit. Section 9 is the envelope-pushing part. Section 10 is the prioritized plan.

## 2. Verified correctness bugs

### 2.1 Temporal recency scoring is inert, and inverts for old memories (High, resolved 2026-07-01)

`src/operations/graph_retrieval.cpp:369-380` computes `age_ms` (uint64 milliseconds since the memory's start) and passes it as the `int rank` parameter of `RetrievalGraphExpandedRagTemporalRankScore` (`include/cortext/core/knobs.hpp:1120-1137`), which is a rank-position decay `TemporalWeight / (max(0,rank)+offset)` with offset in [1.5, 2.6]. Three compounding defects, all verified:

* **Wrong unit.** For any memory older than one second, `age_ms >= 1000`, so the term is at most `TemporalWeight/1000`.
* **Double weighting.** The helper already multiplies by `RetrievalGraphExpandedRagTemporalWeight` (knobs.hpp:1135), and every call site multiplies by `temporal_weight` again (graph_retrieval.cpp:605, 795, 904, 1064, 1115, 1201, 1309). Effective term: `TemporalWeight^2 / age_ms`. With TemporalWeight in [0.06, 0.30], that is under 1e-4 for age >= 15 minutes. The recency signal contributes nothing to ranking.
* **Signed overflow inversion.** `age_ms` (uint64) narrows to `int`. Past ~24.85 days the value wraps; when it lands negative, `max(0, rank) = 0` and the memory receives the maximum temporal score. Bands of month-plus-old memories get a spurious recency boost over genuinely recent ones.

Consequence: on the exact long-horizon corpora being judged, recent-but-slightly-less-similar context loses to old high-similarity memories, and stale memories intermittently jump the queue. Fix: compute a bounded age decay in [0,1] (or pass a real rank position), apply the weight once, and add a unit test asserting monotone decay across a multi-month age range. Because the term is currently near-zero, fixing it changes ranking behavior and must be validated with a frozen-probe replay A/B before release.

Resolution, 2026-07-01: implemented bounded exponential age decay in `TemporalScore`, added `Graph retrieval temporal score decays across multi-month ages`, and ran the 9-probe MSC Gemma4-12B-AWQ baseline/fix A/B. The local A/B completed 9/9 judgments per arm with no compaction baseline; it slightly improved replay performance but reduced Cortext row wins from 3/9 to 2/9. This item is closed as a correctness bug, while larger sufficiency work remains.

### 2.2 Engine and pipeline

| # | Finding | Where | Severity |
|---|---|---|---|
| B1 | `ProcessAudio`/`ProcessImage` (and replay variants) skip the null/dimension validation their `Embed*` siblings enforce; NULL memcpy or `width=-1` size-wrap | `src/cortext.cpp:2033-2035`, `2178-2182`, `2464-2466`, `2540-2544` (contrast `2309-2312`, `2338-2346`) | Medium |
| B2 | `StreamingTextProbeSession` holds a raw `Cortext*`; its own guard reads through the dangling pointer after the engine is destroyed | `src/cortext.cpp:2686`, guards at `2702`, `2773`, `2811` | Medium |
| B3 | `~StopCallback` does not synchronize with an in-flight callback; a stop can `sqlite3_interrupt` a later unrelated query on the shared connection | `include/cortext/stop_token.hpp:101-107`, `145-151`; consumer `src/cortext.cpp:1452-1462` | Medium* |
| B4 | `StopCallback` ctor checks `stop_requested` outside the mutex; a concurrent `request_stop()` can silently miss the callback | `include/cortext/stop_token.hpp:135-142` | Low* |
| B5 | `NormalizeProbeEmbedding` passes NaN through unchanged (`norm > 0` is false for NaN); poisons the aggregated probe vector | `src/cortext.cpp:2653-2666` | Low |
| B6 | Streaming accumulator silently discards all prior chunks if the embedding dimension changes mid-session | `src/cortext.cpp:2799-2805` | Low |
| B7 | Engine mutable state (`signal_filter`, `processor`) has no synchronization and no documented single-thread contract, yet a cross-thread cancellation API is offered | `src/cortext.cpp:1285`, `1294` | Low |

\* B3/B4 are unreachable from in-tree code today (no in-tree `StopSource` ever calls `request_stop()`); they are live for any external embedder using the public cancellation API. Fix both by matching `std::stop_callback` semantics: register under the lock, and block destruction until an executing callback returns.

### 2.3 Retrieval and consolidation

| # | Finding | Where | Severity |
|---|---|---|---|
| R1 | Temporal scoring (Section 2.1) | graph_retrieval.cpp:369-380 + 7 call sites | Resolved 2026-07-01 (was High) |
| R2 | `GetRetrievedMemoryEmbeddings` keyed by `embedding_id` collapses candidates sharing one id; disagrees with per-candidate consumers in five operations | `graph_retrieval.cpp:1381`, `1503-1509` | Low |
| R3 | `max_clusters` cap in DBSCAN labeling checks an always-empty container; dead check, output capped by accident downstream | `src/operations/consolidation_cluster.cpp:180-183` (vs `:206-261`) | Low |
| R4 | Warm-loader trusts `current_memory_embeddings` without the reconstruction-freshness check the read path uses; the two paths should share one freshness rule | `src/signal_processor.cpp:1476-1512` vs `constructive_recall_internal.cpp:537-603` | Low |

The June stale-surface regression (reconsolidation appending without advancing the current surface) is verified fixed, and the reconsolidation UAF fix is present and correct: candidates are deep-copied before any cache mutation (`reconsolidation.cpp:562-587`). No sibling UAF was found anywhere in the operations layer after a deliberate hunt for the same pattern.

### 2.4 Operations layer

The sweep found no critical defect. The numeric guards hold everywhere they were checked (cosine/norm/softmax/entropy/tau denominators all clamped or guarded). Remaining items:

| # | Finding | Where | Severity |
|---|---|---|---|
| O1 | Unchecked `std::any_cast<double>` / `<long long>` on DB rows, unlike the defensive `get_double` helpers used by sibling files on the same table; a storage-class surprise aborts the signal transaction | `src/operations/stability_feedback.cpp:45-46`, `src/operations/influence.cpp:68-71` | Low-Medium (latent) |
| O2 | Hysteresis update is a self-referential EWMA: `Ewma(h, h, alpha)` is identically `h`, so the alpha_T smoothing is a mathematical no-op and the step only clamps to the band | `src/operations/threshold.cpp:165`, `192-195` | Low (but check intent) |
| O3 | Dead fallback branch in uncertainty (metrics can never be empty) and an unreachable ternary in influence | `src/operations/uncertainty.cpp:311-319`, `influence.cpp:76-79` | Info |
| O4 | Type-punned BLOB decodes dereference `reinterpret_cast` pointers instead of `memcpy` like the other decoders; UB under strict aliasing | `src/signal_processor.cpp:918`, `1255-1256` | Low |
| O5 | 64-bit DB counters truncated to `int`; `signals_processed` gates modulo checkpoints and an `== 0` freshness test, so a wrap misbehaves | `src/signal_processor.cpp:1094-1095`, `1157-1161`, `1207` | Low |

O2 deserves a decision, not just a fix: either the smoothing was intended (restore it by tracking a separate band prior) or the clamp alone is the desired behavior (delete the no-op EWMA and the misleading alpha).

## 3. Durability and storage

These matter more than usual because the product premise is a long-lived local database.

| # | Finding | Where | Severity |
|---|---|---|---|
| S1 | WAL auto-checkpoint disabled (`PRAGMA wal_autocheckpoint = 0`); the in-loop checkpoint is env-gated (`CORTEXT_FOREGROUND_WAL_CHECKPOINT`) and only above 256 MB; default path checkpoints only on `Flush()`/forced consolidation. A client that streams without flushing grows the `-wal` without bound and pays full-WAL replay at crash recovery | `src/store.cpp:455-456`; `src/signal_processor.cpp:38-40`, `454-489`, `2194-2204` | High |
| S2 | Object-store blobs are never garbage-collected. Eviction deletes memories/associations/signals/orphan embeddings but no code path calls the existing delete primitive; `objstore_data` grows monotonically forever | `src/operations/memory_strength.cpp:642-729` (deletes), `src/store/object_store.cpp:110-121` (unused primitive) | High |
| S3 | `SQLiteStore`'s statement cache and transaction stack have no lock and no documented single-writer contract; concurrent use of one instance is UB (use-after-finalize of cached `sqlite3_stmt*`) | `include/cortext/store/sqlite_store.hpp:214-223`; `src/store.cpp:722-752`, `803-843` | High (conditional) |
| S4 | Accumulator (boundary/episode/pacing) state persisted only on `Flush()`, and `~Cortext()` is defaulted so it never flushes; crash loses all per-source accumulation since the last flush while state/WM checkpoint every signal. Contradicts the comment at signal_processor.cpp:2389 | `src/signal_processor.cpp:2159-2205` vs `2297`, `2952-2958`; `src/cortext.cpp:1874` | Medium |
| S5 | `Checkpoint(full=true)` uses RESTART, never TRUNCATE; WAL file stays at high-water mark | `src/store.cpp:1079-1080` | Medium |
| S6 | `memory_evictions` audit table (plus 3 indexes) grows without bound; no retention anywhere | `src/store/schema.cpp:472-499`; writer `memory_strength.cpp:620-638` | Medium |
| S7 | Two-process concurrent first migration hits a PRIMARY KEY conflict (applied-set read outside the deferred tx); two processes sharing one DB also clobber the singleton `state` row last-writer-wins | `src/store/schema.cpp:47-49`, `868-884`; `signal_processor.cpp:2461` | Medium |
| S8 | Model-pin freshness keyed only on `COUNT(*) FROM embeddings`; a DB whose base embeddings were evicted but which still holds `current_memory_embeddings` vectors can be silently re-pinned to a different encoder, violating the README's own mixing-spaces invariant | `src/cortext.cpp:1240-1274`; `schema.cpp:641-647` | Medium |
| S9 | `GetWalStatus() const` actually performs a passive checkpoint; `INSERT OR REPLACE INTO state` resets unlisted columns (currently only dead ones) | `src/store.cpp:1101-1113`; `signal_processor.cpp:2461-2510` | Low |

Recommended package (fixes S1/S2/S5/S6 together): switch full checkpoints to `SQLITE_CHECKPOINT_TRUNCATE`, run an unconditional periodic passive checkpoint (drop the env gate and the 256 MB floor; make it a baked default per the turnkey rule), collect `blob_id`s during eviction and delete unreferenced ones reusing the existing `NOT EXISTS` pattern from memory_strength.cpp:716-729, and prune `memory_evictions` with the same incremental pattern as closed-WM rows. For S4: persist accumulators per-source inside the per-signal transaction (upsert, not DELETE-all-reinsert), or at minimum flush in the destructor. For S3/S7: document and enforce single-writer (a mutex around store bookkeeping is cheap; a `BEGIN IMMEDIATE` migration fixes the TOCTOU).

## 4. Build, CI, and tests

### 4.1 CI is not guarding the release

| # | Finding | Where | Severity |
|---|---|---|---|
| C1 | zig-smoke passes removed `-Dllama` option; `zig build -Dllama=false` exits 1 (reproduced with vendored Zig 0.15.2). The job fails on every run | `.github/workflows/build.yml:80,83,86` | Critical |
| C2 | Both native jobs download the 141 MB AIST model from a mutable HF `main` ref on every push, then run `'~[aist]'` so no model test executes. Zero encoder coverage in CI, plus a network flake surface | `CMakeLists.txt:116-126`; `build.yml:25,28,42,43` | Critical |
| C3 | ASan/UBSan/MSan options are fully wired but no preset and no CI job enables them, in a codebase with a fixed UAF in its history | `CMakeLists.txt:93-95`, `649-665` | High |
| C4 | The paper's regression evidence (replay parity, filtered suites) is local-only; nothing in CI enforces any of it | `9_experimental.qmd:50-63` vs `.github/` | High |
| C5 | Dead `-DCORTEXT_DISABLE_SHERPA_ONNX=ON` in CI (option no longer exists); same bug class as C1, silently ignored | `build.yml:22,41` | Low |
| C6 | `FetchContent_Populate` for GGML is deprecated (CMake 3.30+, future hard error under CMP0169) | `CMakeLists.txt:380-383` | Low |

The model fetch itself has good hygiene: pinned sha256 + size verified, atomic replace (`scripts/download_aist_model.py:38-90`). Pin `--revision` to a commit SHA instead of `main` to make it reproducible rather than merely fail-closed.

### 4.2 Test-suite integrity

| # | Finding | Where | Severity |
|---|---|---|---|
| T1 | Six operations wired into the per-signal pipeline have zero test references: `synaptic_tagging`, `neuromodulators`, `consolidation_gate`, `signal_metrics_persistence`, `accumulator_scores`, `accumulator_reset`. Weak-indirect-only: `storage_pressure`, `retrieval_trace_state`, `eviction_policy_override`, `drift_accumulation` | `src/cortext.cpp` registration vs `tests/` | High |
| T2 | Can't-fail tests: `operations_adherence_fixes.test.cpp:67,105,147` assert nothing (SUCCEED after "no crash"), and `integration_working_memory_manual.test.cpp` has zero REQUIRE/CHECK across the file yet runs in the automated suite. Under the project's own no-skip-tests rule, these should assert or not exist | tests/ as cited | Medium |
| T3 | Non-hermetic suite: dozens of undocumented `CORTEXT_*` env toggles (see 4.3) are read by shipping code and nothing scrubs the environment; `thread_config.test.cpp:26-80` mutates globals with raw setenv and can leak on failure | tests/, src/ | High |
| T4 | Temp-DB collisions: `store.test.cpp:30` names by 1-second wall clock; `operations_memory_storage.test.cpp:63` uses unseeded `rand()` | tests/ as cited | Medium |
| T5 | All 486 cases registered as a single opaque CTest entry; no per-case reporting or sharding | `tests/CMakeLists.txt:119-124` | Medium |
| T6 | The planum submodule is pinned to a stub commit containing only a README, so the configure-time gate for `audio_planum_bridge.test.cpp` can never pass: a permanently dead test | `.gitmodules`; `tests/CMakeLists.txt:94-98` | Medium |

### 4.3 The hidden knob surface (turnkey violation)

Shipping code reads dozens of undocumented env vars that gate core algorithm branches: `CORTEXT_BOUNDARY_DISABLE_NATURAL/PRESSURE/SURPRISAL`, `CORTEXT_FLASHBULB_DISABLE_*`, `CORTEXT_DISABLE_CONSTRUCTIVE_RECALL`, `CORTEXT_DISABLE_META_LEARNING`, `CORTEXT_DISABLE_INTERRUPT_ACCEPT`, `CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION`, `CORTEXT_DISABLE_CURRENT_MEMORY_SURFACE_WRITES`, `CORTEXT_RECONSTRUCTION_*`, `CORTEXT_AIST_SHADOW_ONLY`, and more. These are eval-harness ablation hooks that leaked into the release binary as ambient escape hatches. They contradict the stated design philosophy (baked defaults, hard checks, no escape hatches), make every test result environment-dependent, and mean a stray shell variable silently changes production behavior.

Recommendation: compile them out of release builds behind a single `CORTEXT_EXPERIMENT_HOOKS` build flag that is OFF by default (the ablation harness turns it on), keep the handful of genuinely operational ones (thread counts, journal mode) documented in README, and add one test asserting a release build ignores the ablation set. This is a hard cut of the same kind the project already did once; finish it.

## 5. Hard-cut completeness

* **Dead script family (12 files).** The offline label-classifier pipeline (`scripts/run_label_classifier_study.sh`, `label_classifier_lib.py`, `train/eval/probe_label_classifier.py`, `build_label_training_data.py`, `generate_openai_label_data.py`, `build_name_priors.py`, `build_wordnet_label_index.py`, `prepare_wnut17_label_data.py`, `sample_label_examples.py`, `collect_label_supervision_data.sh`) has zero inbound references and targets data/model paths that no longer exist. The paper says these workflows were removed under the hard-cut rule; the scripts contradict that. Delete them (git history preserves them).
* **Broken tool.** `tools/chat_replay_human_label_harness.py:25` imports `frozen_chat_replay_retrieval_eval`, which does not exist anywhere; it crashes on invocation. Delete or fix.
* **Portability.** `scripts/run_msc_frontier_judge.sh:29-30` and `tools/judge_chat_replay_live_run.py:249-251` hard-code `/shared/augmem/.env`, `/shared/orgnet/.env`, `/shared/personaplex/.env` for credentials. Replace with a documented `CORTEXT_EVAL_ENV_FILE` or a repo-local `.env` convention. (No literal secrets anywhere in the tree; all credentials are env-sourced. Verified.)
* **Submodule/vendor conflict.** `.gitmodules` declares `sqlite`, `sqlite-vec`, and `sqlite-objstore` as submodules, but all three are fully vendored (2,439 tracked files) and the build compiles the vendored paths. A well-meaning `git submodule update --init` can convert them to gitlinks and clobber the source the build depends on. Remove the stale `.gitmodules` entries and record the vendored upstream versions in a `third_party/VERSIONS` note.
* **Attribution gap.** No root NOTICE/THIRD-PARTY file consolidates the Apache-2.0 attribution obligations for statically linked sqlite-vec, sqlite-objstore, and blake3. One generated NOTICE file closes it.

## 6. Paper and README

* **Missing sections.** `@sec-pattern-separation` is referenced (`4_dynamic_thresholding.qmd:482`, `6_advanced_cognitive.qmd:655`) but no section defines it, and `sparse_key`/`index_store`/`procedural_store` appear in the write path and state maps (`appendices.qmd:14,24`) with their defining math never written. `@sec-activity` (`6_advanced_cognitive.qmd:585`) should repoint at `@sec-consolidation`. These are the two render warnings.
* **Stale adapter-era text.** `index.qmd:47` still claims consolidation produces "summary nodes linked by typed semantic relations," which the hard-cut consolidation explicitly does not do (`10_implementation.qmd:90-91`). `index.qmd:33` still lists metacognition, a removed subsystem. `index.qmd:9` dates the paper December 2025 while headline evals are June 2026.
* **Record the negative results.** The 18-arm mechanism sweep (ACT-R gate arms all failing, all_gates harmful, metacognitive cut, temporal and predictive kept) is documented nowhere in the manuscript. That is the single biggest re-proposal hazard for any future contributor or agent: the strongest reason this report does not recommend activation-gating ideas lives only in private session notes. Add a short "Negative results" subsection to `9_experimental.qmd`. Note the retained ACT-R-inspired activation *ledger* (`10_implementation.qmd:111`) explicitly, so the kept trace is not conflated with the rejected gates.
* **Framing risk.** README's headline "won 21 of 27 blind judgments" counts 3 repetitions x 9 probes; the denominator implies more independence than exists. Lead with probe-majority plus the CI (the protocol is already disclosed, this is presentation). The sufficiency caveat in README:36-39 is honest and should stay.
* **Artifacts are unverifiable from a clone.** `eval_runs/` is 80 GB and gitignored, but README and the paper cite artifact paths inside it. Publish the small aggregate judge JSONs (and ideally the per-judgment `*.rows.jsonl`) as release assets or a slim data repo, and link them from the paper.

## 7. API, bindings, DX

| # | Finding | Where | Severity |
|---|---|---|---|
| A1 | Node `lastError()` throws on the happy path: `cortext_last_error()` returns nullptr when empty and it is fed straight to `napi_create_string_utf8` | `bindings/javascript/src/addon.cpp:873-881` | Medium |
| A2 | Go reads the thread-local error in a separate cgo call; goroutine thread migration loses the real message intermittently | `bindings/go/cortext.go:397-411` | Medium |
| A3 | WASM binding is a reduced subset (no processAudio/Image, no media, no options, consolidate returns void) while `bindings/README.md:9` claims all bindings expose the same v1 concepts | `bindings/wasm/cortext.js` | Medium |
| A4 | `embed*` returns `{embedding, dimension}` in WASM but a bare vector in Node/Python/Go/Dart; same-named methods disagree across the two JS surfaces | `bindings/wasm/cortext.js:96-157` vs `index.js:122-132` | Low-Medium |
| A5 | Error strings are not UTF-8 sanitized at the C boundary; Python/Dart strict decoders throw on the very failure path they exist to report | `capi.cpp:1628`; `bindings/python/__init__.py:520` | Low |
| A6 | Knob range [0,1] documented but neither validated at construction nor documented as clamped (silent per-op clamp). The turnkey rule says hard checks: reject or document | `capi.h:224-226`; `cortext.cpp:112-114` | Low |
| A7 | Binding quickstarts omit the model-bootstrap prerequisite; first constructor throws an opaque "create failed" whose explanation is only reachable via `lastError()` (which in Node itself throws, A1) | `bindings/*/README.md` | Low |

Otherwise the C ABI is in good shape: consistent null checks and error codes, thread-local diagnostics, `struct_size` versioning, and no leaks found in any binding glue. Worth stating in release notes.

Two surface-design notes for v1.x, both ABI-relevant: `Context` exposes fourteen `interrupt_gate_*` debug fields and `ProcessorOutput.metrics` is an `unordered_map<int,double>` keyed by an enum cast (`include/cortext/cortext.hpp:50`, `120-137`). Both read as internal telemetry frozen into the public struct. Consider moving diagnostics behind one opaque JSON/trace accessor before the ABI hardens further.

## 8. Performance

Ordered by expected impact on steady-state latency; none of these change behavior.

1. **Per-signal rollback snapshot deep-copies every rebuildable cache** (`src/signal_processor.cpp:2079`). Dominant per-signal copy cost; a mutation journal or copy-on-write removes it. The error path also redundantly reloads predictive/suppression IDs it just restored (`:2112-2113`).
2. **Second source-expansion pass scans the whole surface cache per seed**, O(seeds x N) on the hot read path (`graph_retrieval.cpp:433-483`, called from `:1076-1127`). Pass 1 already uses the sorted source index; extend it to associations so pass 2 binary-searches. This partially undoes the win of the latency-flattening commit as the corpus grows.
3. **N+1 query patterns**: per-candidate `SELECT kind` in hydration (`cortext.cpp:1699`), up to three sequential resolve queries per candidate (`:1588-1620`), per-slot and per-record loads in `LoadWorkingMemory` (`signal_processor.cpp:1662-1789`), per-event SELECT+UPDATE pairs in `memory_strength.cpp:207,406`, and a per-signal created_at lookup in the interrupt gate (`interrupt_gate.cpp:238-262`).
4. **`PersistAnchor` rewrites every soft-anchor row on every signal** regardless of dirtiness (`soft_anchor.cpp:598-601`).
5. **Per-memory `std::getenv` in hydration** (`cortext.cpp:761`, `813`) and three ~25-attribute debug-log blocks built unconditionally per signal in working_memory.cpp. Cache the flags; gate log construction on level.
6. Retrieval write amplification: the constructive-recall loop inserts embedding + reconstruction rows and rewrites the current surface for top candidates on every retrieval (`graph_retrieval.cpp:1389-1502`). Intrinsic to the design, but profile `GraphRetrieve.reconstruction_versions` against the expansion phases before optimizing anything else.

## 9. Pushing the envelope

What this section deliberately does not propose, because the project already tried or removed it: decoder backends, adapter registries, semantic batch consolidation, static taxonomies, fact tables, label banks/bucket graphs, STM shadow promotion, persistent confidence-monitoring state, ACT-R activation gates (all sweep arms failed), all-gates configurations (harmful), front-loaded reinforcement, fixed co-retrieval increments, candidate-diversity interrupt scaling, and the host fallback text path. The ColBERT reranker and the sml.cpp/emel.cpp moves are already on the roadmap and not repeated here. Consolidation-family changes below carry the standing rule: verdicts come from long-corpus stress runs, not short replays.

### 9.1 Close the sufficiency gap without spending the token budget

The MSC result is a 98% token reduction with sufficiency 4.41 against RAG's 4.67. The gap is the compression cost, and there is enormous headroom: Cortext spends ~1k tokens where baselines spend 16k to 49k. Three levers, in order of confidence:

**E1. ~~Fix the temporal term (Section 2.1).~~ Resolved 2026-07-01.** The correctness fix and 9-probe MSC Gemma A/B are complete. The A/B was negative for quality, so continue sufficiency work with E2/E3 rather than treating temporal scoring as a won lever.

**E2. Adjacent-turn hydration for winners.** For the top 2-3 ranked memories, emit their immediately-adjacent same-`source_id` turns as hydration context rather than ranked competitors. The machinery exists (`SourceExpansionRowsFromSurface`, the turn-source expansion at `graph_retrieval.cpp:1217-1326`); today those neighbors compete inside the same capped top-k and get dropped. This is the classic fetch-surrounding-context move, done purely with provenance and timestamps, no semantics. Cost: maybe 200-400 extra tokens per packet against a 48k headroom. This is the single most promising sufficiency lever in the codebase.

**E3. Let retrieval consume soft anchors.** Formation is always-on and durable (`soft_anchor.cpp` maintains centroids plus `soft_anchor_links`), but ranking never reads them; the paper already marks consumption as the sanctioned next step (`6_advanced_cognitive.qmd:412-423`). Boost or expand candidates sharing a durable anchor with the query's nearest anchor. Embedding-only, uses state you already pay to maintain.

**E4. Sufficiency-elastic packets.** Per-retrieval, when the ranked list shows low margin or high score entropy (both computed today for the anchor path), widen hydration: more winners, deeper adjacent-turn context, up to a bounded ceiling (say 4k tokens). When confidence is high, stay at ~800. This is stateless per-call policy, not the removed persistent confidence-monitoring machinery, and it is exactly the closed-loop philosophy applied to the packet itself: the knob the system has never modulated is its own output budget. Judge with the existing blind harness; the prediction is sufficiency approaching the compaction baseline (4.63+) at a still-95%+ token reduction.

### 9.2 Scale the memory, not just the window

**E5. Matryoshka two-tier retrieval.** AIST truncation already normalizes correctly (the tested unit). Store a truncated prefix (64 or 128 dims) alongside full vectors; scan the prefix tier for coarse top-k, rerank the survivors at full dimension. This converts the remaining O(N) surface scans into cheap prefix scans, cuts the working set several-fold, and is the native pre-stage for the roadmap's post-ship ColBERT reranker (coarse prefix -> full dense -> late interaction). It also raises the ceiling on how many memories a device-class deployment can hold, which matters for the care use case more than benchmark wins do.

**E6. Wire generation embeddings into the influence blend.** The lambda path is implemented and dormant (`5_reinforcement.qmd:242-260`): nothing feeds the assistant's own responses back in. Embedding the host application's reply and passing it as the generation embedding closes the last open loop: memory influence gets credited by what the agent actually said, not just what was retrieved. Cheap API addition (`ProcessGeneration` or an optional arg), zero new subsystems, and it directly sharpens the usage signals that feedback knobs consume. Long-horizon judged.

### 9.3 Turn the release evidence into standing infrastructure

**E7. Replay-diff gate plus drift canary in CI.** Replay parity with system GGML is proven (`retr_diffs=0`, `rank_diffs=0`). Freeze the dense 578-message replay and its 383 probes as a golden artifact; a CI job with `CORTEXT_USE_SYSTEM_GGML=ON` asserts zero diffs and prints the diff otherwise. Add a canary that embeds a fixed probe set and compares hashes across builds: that is precisely the query-vector numeric drift that produced June's residual rank flips, detected in minutes instead of a forensic week. Together with the sanitizer job (C3) this is the highest-leverage CI money in the repo.

**E8. Strengthen the eval where a skeptic attacks first.** All five use artifacts that already exist:

| Attack | Cheap counter |
|---|---|
| n=9 probes, CI [0.52, 0.96] | Re-judge the already-saved MSC replay at `--probe-stride 100` (40-90 probes); replay cost already paid, only judge calls scale |
| One judge family (and the compaction baseline is the judge's own model) | Re-judge the saved `*.rows.jsonl` packets with a second judge family; report Cohen's kappa |
| Winner is the judge's holistic pick, and Cortext wins with lower sufficiency (concision bias) | Zero cost: recompute winners as argmax of the defined composite from per-packet scores already in the JSON; report both. Lean on the 128k length-controlled ablation |
| Raw-wins denominator (3 reps x 9 probes) | Zero cost: lead all tables with probe-majority plus bootstrap CI |
| Single dataset, multimodality never judged (`--max-media-per-system 0` everywhere) | The LongMemEval harness already exists (`scripts/run_longmemeval_study.sh`, `prepare_longmemeval.py`, `score_longmemeval.py`); run it. Add one media-enabled probe set to exercise the multimodal claim |

**E9. Own the local-first story.** The pieces exist and are undersold: an 87M multimodal encoder, a single SQLite file, a WASM build, and a care-context mission that is fundamentally a privacy story. A hosted demo where the entire memory engine runs in the browser tab (model preloaded into the virtual FS, nothing leaves the device) is a differentiator no RAG-stack competitor can copy cheaply, and it exercises the WASM binding gap (A3) into closure. The multi-process findings (S7) become the one real constraint to document: one writer per database, enforced with a lease, is the honest v1 contract.

## 10. Prioritized plan

**P0, before tagging the release:**
1. ~~Temporal scoring fix + frozen-replay A/B (2.1).~~ Done for the bug fix plus 9-probe MSC Gemma A/B on 2026-07-01; quality did not improve, and results are recorded in Section 9 of the paper.
2. WAL checkpoint defaults + TRUNCATE (S1/S5) and objstore GC on eviction (S2). Two days.
3. Fix zig-smoke (`-Dllama`) and drop the dead sherpa flag (C1/C5). Minutes.
4. Stop fetching the model in native CI; add the dedicated `[aist]` job with sha-keyed cache; add the ASan/UBSan job (C2/C3). Half a day.
5. Node `lastError()` one-liner (A1); ProcessAudio/Image input validation (B1). Hours.
6. Accumulator persistence per-signal or destructor flush (S4). Half a day.

**P1, release week:**
7. Compile ablation env hooks out of release builds; document the operational few (4.3). One day.
8. Delete the dead script family and the broken harness tool; fix the sibling `.env` paths (Section 5). Hours.
9. Tests for the six untested pipeline ops; make the can't-fail tests assert or delete them; env scrub in the test main; unique temp DBs (T1-T4). Two to three days.
10. Paper: negative-results subsection, pattern-separation section or inline definition, repoint `@sec-activity`, fix index.qmd stale lines, probe-majority-first tables (Section 6). One day.
11. NOTICE file; remove stale `.gitmodules` entries; publish aggregate eval JSONs (Sections 5, 6). Hours.
12. Single-writer contract documented + store mutex + `BEGIN IMMEDIATE` migrations (S3/S7). One day.

**P2, post-ship (ordered by expected quality-per-effort):**
13. E2 adjacent-turn hydration, then E4 elastic packets, judged on the existing blind harness.
14. E7 replay-diff gate + drift canary.
15. E8 eval strengthening runs (stride-100 re-judge, second judge, composite recompute, LongMemEval).
16. E3 soft-anchor consumption; E6 generation embeddings (long-horizon judged).
17. E5 Matryoshka tiering, as the pre-stage for the roadmap ColBERT reranker.
18. Performance items in Section 8 order, profile-first.

## Limitations

This was a static review. No sanitizer runs were executed, severities reflect code reading rather than reproduction (except C1, which was reproduced), and the storage growth findings were verified in source but not measured on a long run. The temporal-scoring bug has since been fixed and screened on the 9-probe MSC Gemma A/B; that screen was negative for quality, so the result should be treated as a correctness closure, not as release evidence for improved sufficiency.
