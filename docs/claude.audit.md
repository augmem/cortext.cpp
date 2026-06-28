# Cortext Code Audit — Code Smell, Slop & Duplication

**Date:** 2026-06-24
**Scope:** `src/` and `include/` (~44k LOC C++20), excluding generated files (`src/data/embedded_centroid_vectors.cpp`) and `third_party/`.
**Method:** Six subsystem reviewers read every in-scope file directly (two largest files via dedicated sub-readers), cross-checked against mechanical scans (grep/AST-ish duplication detection, dead-symbol analysis, formatting checks). Every `file:line` claim below was read or grep-verified.

## How to read this

Severity: **HIGH** (load-bearing logic duplicated, god-functions/objects, silent-degrade error paths, dead public surface) · **MED** (real maintenance hazard, localized) · **LOW** (cosmetic / nit).

Categories: duplication · dead-code · long-function · god-object · complexity · magic-number · error-handling · naming · abstraction · formatting.

**Overall health:** The codebase is disciplined in the obvious dimensions — zero `TODO`/`FIXME`/`HACK` markers, no commented-out code, no skipped tests, consistent telemetry usage, warnings-as-errors. The problems are structural and concentrated: (1) pervasive copy-paste of small utilities and decode logic that *already have shared homes*, (2) a handful of mega-functions and two blackboard god-objects, (3) parallel implementations that can silently diverge, and (4) measurable dead surface in `knobs.hpp`. None of this is a correctness emergency; all of it taxes every future change.

---

## 1. Cross-cutting issues (highest leverage)

These repeat across many files. Fixing them once removes the most duplication and the most drift risk.

### 1.1 — HIGH · duplication · Utilities re-defined per-file despite shared homes existing

| Helper | Canonical home | File-local copies | Notes |
|---|---|---|---|
| `Clamp01(double)` | `core/utils.hpp:72` (`core::Clamp01`) | **8** | `signal_filter.cpp:17`, `soft_anchor.cpp:45`, `cortext.cpp:112`, `predictive.cpp:291`, `competition.cpp:79`, `interrupt_gate.cpp:35`, `embedding_prediction_error.cpp:28`, `serial_position_apply.cpp:18`. Three differ in NaN handling — exactly the drift that causes bugs. |
| `EnvFlag(const char*)` | none (should be shared) | **8** | `neuromodulator_internal.hpp:14`, `emotion.cpp:19`, `cortext.cpp:118`, `boundary.cpp:20`, `constructive_recall_internal.cpp:33`, `accumulator.cpp:30`, `meta_learning_internal.cpp:72`, `graph_retrieval.cpp:315`. Byte-identical (lowercase + match `1/true/yes/on`). |
| `ElapsedMillis(SteadyClock::time_point)` | none | **8** | `accumulator.cpp:23`, `interrupt_gate.cpp:28`, `stability.cpp:23`, `predictive.cpp:59`, `memory_strength.cpp`, `reconsolidation.cpp`, `memory_storage.cpp`, `competition.cpp`. Byte-identical. |
| `LowerEnv(const char*)` | none | 2 + 5 inline | `object_store.cpp:27`, `extension_loader.cpp:28`, plus 5 IIFE lambdas inlined in `store.cpp` resolvers. |

**Fix:** add `core::Clamp01`'s NaN guard once and delete the 8 copies; create `core/env_utils.hpp` (`EnvFlag`, `LowerEnv`, `EnvInt`) and `core/timing.hpp` (`ElapsedMillis`); include them everywhere. Lowest-risk, highest-count cleanup in the repo.

### 1.2 — HIGH · duplication · Hand-rolled `std::any` decoding everywhere

`store::AnyToLongLong` exists in `store/utils.hpp`, yet **19 files** hand-write `type() == typeid(long long) / int / double` ladders to decode SQL row values: `emotion.cpp`, `reconsolidation.cpp`, `memory_strength.cpp` (its bespoke `GetDouble`/`GetInt64`/`GetInt64Maybe` trio), `consolidation*.cpp`, `constructive_recall_internal.cpp`, `meta_learning_internal.cpp`, `graph_build.cpp`, `interrupt_gate.cpp`, `stability.cpp`, `storage_pressure.cpp`, `emotion_cascade.cpp`, `cortext.cpp` (`get_s/get_ll/get_dbl/get_blob` defined 3× at lines 483, 571, 944), `signal_processor.cpp` (`ExtractInt64/Double/String` + six BLOB decoders), `capi.cpp`, `store.cpp`. There is **no `AnyToDouble`** companion, which is why the double paths are all open-coded.

**Fix:** add `store::AnyToDouble` and a `store/row_accessors.hpp` with `get_s/ll/dbl/blob`; route every site through them. Removes dozens of drift-prone branches and the leftover `[[maybe_unused]]` accessors in `cortext.cpp`.

### 1.3 — HIGH · duplication · Repeated SQL fragments with no helper

- `SELECT last_insert_rowid() AS id` + int/long decode: copy-pasted in **5 files** (`memory_storage.cpp`, `consolidation_shallow.cpp` ×2, `reconsolidation.cpp`, `constructive_recall_internal.cpp` ×2, `signal_processor.cpp`). → one `LastInsertId(tx) -> long long` / `InsertReturningId(tx, sql, params)`.
- `INSERT OR REPLACE INTO associations (...) VALUES (?,?,'<type>',?,?)`: **~8×** across `graph_build.cpp`'s five edge builders → `InsertEdge(tx, src, dst, type, weight, now_ts)`.
- Orphan-embedding `DELETE ... WHERE NOT EXISTS(memories) AND NOT EXISTS(signals) AND NOT EXISTS(memory_reconstructions)`: verbatim in `signal_processor.cpp:298` and `:318`.
- Savepoint `ROLLBACK TO SAVEPOINT` + `RELEASE SAVEPOINT` pair and root `COMMIT`/`ROLLBACK`: 4 sites in `store.cpp` (`561`, `699`, `846-904`, `906-968`); `Commit`/`CommitRootTransaction` and `Rollback`/`RollbackRootTransaction` are near-identical → `FinishRoot(bool)` + `FinishSavepoint(bool,name)`.

### 1.4 — HIGH · magic-number · Embedding dimension `256` hardcoded, no canonical constant

There is **no `core::kEmbeddingDim`**. The value `256` is hardcoded in `graph_build.cpp:58` (`constexpr int kEmbeddingDim = 256`), `consolidation.cpp:173` (`int emb_dim = 256; // Default assumption`), `embedded_centroid_vectors.cpp:16`, and `cortext.cpp:186` (`RetrievalEmbeddingView` truncation). `graph_retrieval.cpp` by contrast *derives* the dim from `signal.embedding.size()`. **A model with a different dim silently produces an empty graph** (`DecodeFloatBlob` fails → `continue`) with no error — the most dangerous instance. **Fix:** define one `core::kEmbeddingDim`, derive from config/store where possible, and assert/log on mismatch instead of silently dropping.

### 1.5 — HIGH · duplication · Re-implemented statistical / embedding primitives

`core::CosineSimilarity` exists (`algorithms.hpp:56`) and is used correctly by `consolidation_cluster.cpp`/`graph_build.cpp`, but `graph_retrieval.cpp:58` defines a private `Cosine` and `soft_anchor.cpp:106` a `Map01Cosine`. Beyond cosine, the same primitives are re-derived repeatedly with no shared home:

- **Windowed mean/variance** over `recent_scores`: `sensitivity.cpp:234`, `stability.cpp:203`, `uncertainty.cpp:61`.
- **Max-subtracted softmax + normalized entropy:** `focus_spread.cpp:30-119` is re-implemented *in full* as a fallback inside `uncertainty.cpp:85-159` — two copies of the operation's core math that can silently diverge.
- **Max-cosine-to-window (redundancy/novelty):** `sensitivity_feedback.cpp:15`, `uncertainty.cpp:162`, `interrupt_gate.cpp:77`.
- **L2-normalize-in-place (`1e-9f` floor):** `write_gate.cpp`, `accumulator.cpp`, `coherence.cpp`, `soft_anchor.cpp`.

**Fix:** an `operations/detail/` math module (`WindowedVariance`, `FocusSpreadEntropy`, `MaxCosineToWindow`, `NormalizeInPlace`). Make `uncertainty` call the canonical `focus_spread` function rather than re-derive it.

### 1.6 — HIGH · god-object · The two context blackboards

- **`operation_context.hpp` (1257 lines):** one per-signal scratchpad with **89 fields exposed via ~164 trivial getter/setter methods** across ~20 subsystems. The `contract_tags.hpp` families already define the natural grouping (e.g. `tags::MniGateDiagnostics` ↔ ~16 scalar accessors at `366-455`+`501-560`; `tags::SerialPositionPolicy` ↔ 7 pairs at `675-744`), yet each scalar is exposed individually.
- **`processor_context.hpp` (959 lines):** an all-public `struct` mixing ~150 state fields, nested container/record classes, and **21 non-trivial methods (`595-940`)** that implement a full retrieval-surface index (insert/remove/swap-erase/reindex). Its own doc-comment admits it "could be refactored into smaller sub-structs."

**Fix:** collapse each contract-tag family into a single struct accessor; extract `RetrievalSurfaceCache` and `AssociationFanout` from `processor_context` into standalone classes held by value. Highest-leverage structural refactor in the repo. Also note the six `…ByEmbedding`/`…ByMemory` method twins (`processor_context.hpp:823-940`) differ only in which index map they consult → one `Resolve(map,key)` helper.

### 1.7 — MED · error-handling · Silent-degrade contradicts the project's "fail loud" philosophy

Several paths swallow failures the codebase elsewhere insists on surfacing: `SqlObjectTransaction::Exists` returns `false` for *any* unrecognized `StoreError` (`object_store.cpp:272`); `graph_build` no-ops the entire edge stage when `store == nullptr` with no telemetry (`:402`, `:440`); `boundary.cpp`/`soft_anchor` add-residual mismatches `return` silently (`aist_gguf_encoder.cpp:1418` does the same). `ExecuteDirect` logs telemetry on prepare/step failures but **not** on bind/fetch failures (`store.cpp:1016`, `:1032`), while `ApplySingleMigration` double-logs the *same* failure (`schema.cpp:60` and `:79`). **Fix:** standardize — one log per failure at one layer; never silently convert an error to a boolean/empty result.

### 1.8 — HIGH · dead-code · 31 unused knob functions in `knobs.hpp`

31 `inline` knob functions are defined and have **zero callers anywhere** (not src, tests, nor other knobs — grep-verified). A further ~42 are referenced **only by tests**, never by production code. The dead set clusters almost entirely in retrieval features — the signature of an abandoned/stubbed feature line. Full list in [Appendix A](#appendix-a--dead--test-only-knob-functions). **Fix:** delete the 31 dead functions (and their helper structs); decide whether the 42 test-only knobs are aspirational or removable.

---

## 2. Subsystem findings

### 2.1 Memory operations (`src/operations/`)

**`competition.cpp`** — *HIGH duplication:* the two RIF-recovery loops (`106-172` memory-id, `179-245` embedding-id) are ~65 lines each, near byte-identical, differing only by column name → extract `RecoverSuppressed(...)`. *MED magic-number:* `1e-9` "still active" threshold inlined in SQL at `:157`/`:230` while `constants::kNormEpsilon` exists.

**`memory_strength.cpp`** — *HIGH duplication:* the 4-attribute `LogDebug("cortext.memory_strength", …)` early-return block is copy-pasted **5×** (`451`, `466`, `499`, `560`, `737`) → a `log_and_return(eviction_count)` lambda. *MED long-function:* `UpdateMemoryStrength::Execute` (`153-746`, ~590 lines) does feedback-update + a 7-stage eviction pipeline → split into `ApplyFeedbackUpdates` / `RunEvictionPass`. *MED:* `GetDouble`/`GetInt64`/`GetInt64Maybe` (`90-149`) — see §1.2.

**`memory_storage.cpp`** — *MED long-function:* `Execute` (`87-556`, ~470 lines) → extract `AssembleContentBlob` (`218-297`), `InsertSignalRows` (`391-449`). *LOW abstraction:* `SourceOriginFor()` returns constant `"source"`; `SourcePriorReliability` is a one-line pass-through → inline. *LOW formatting:* tab/space corruption at `504-513`.

**`working_memory.cpp`** — *MED long-function:* `Execute` (`81-603`, ~520 lines) → extract `SelectEvictionIndex` (`456-509`), `ChunkIntoSlot` (`295-345`). *MED duplication:* three ~12-attribute `LogDebug("cortext.working_memory", …)` blocks (`229`, `349`, `578`). *LOW naming:* `alpha`/`beta` reused with two unrelated meanings (`196` vs `281`). *LOW slop:* telemetry logs each weight twice (`w_alpha` and `alpha`, …) at `367`/`595`.

**`reconsolidation.cpp`** — *MED duplication ×3:* metadata decode ladders (`649-689`, `428-449` — §1.2); `WriteEmbeddingInPlaceOrFork` writes the delete+insert+refresh sequence twice (`174-183` vs `200-210`) → `WriteCurrentEmbedding(...)`; primary-reconstruction tail (`754-792`) duplicates `WriteNeighborUpdates` (`503-533`). *LOW dead-code:* stale `(void)max_drift;` at `:893` (it's used at `:898`).

**`consolidation.cpp`** — *HIGH duplication:* the consolidation **score formula** (`?1*strength - ?2*redundancy + …`) is written out **3×** — SELECT projection (`87-93`), WHERE (`102-107`), fallback CTE (`131-136`). Any formula change needs 3 edits → build one `score_expr` string or compute in C++. *LOW dead-code:* misleading `(void)now_ts;` at `:44` (used at `112`/`158`). *LOW magic-number:* `emb_dim = 256` (§1.4).

**`consolidation_cluster.cpp`** — *HIGH duplication:* cluster-build body duplicated between the normal loop (`214-261`) and forced-fallback (`319-359`) → `BuildCluster(...)`. *LOW:* header doc-comment claims "greedy / running-mean" but the impl is DBSCAN (`90-184`) — doc/code mismatch. O(n²) `neighbors_of` (acceptable at bounded n; add a comment).

**`consolidation_shallow.cpp`** — *MED duplication:* `last_insert_rowid` decode ×2 (`146`, `171` — §1.3); the two cluster-member branches (`236-255` vs `256-299`) share the UPDATE+INSERT writes → `LinkDerivedSource(...)`. *LOW dead-code:* `centroid_vec` is a third redundant copy of the same vector (`190-202`).

**`consolidation_gate.cpp`, `synaptic_tagging.cpp`** — clean, no findings.

### 2.2 Attention / signal operations (`src/operations/`)

Mega-`Execute` functions dominate; most also carry the §1.1/§1.5 duplication.

- **`boundary.cpp`** — *HIGH long-function:* `Execute` (`57-466`, ~410 lines). *MED dead-code:* the timeout branch (`timeout_trigger` `:290`, `SetBoundaryType("timeout")` `:366`) is permanently unreachable — comment at `:317` confirms "no hard timeouts." Remove it.
- **`interrupt_gate.cpp`** — *HIGH long-function:* `ComputeMniGateDecision::Execute` (`121-686`, ~565 lines); the MNI-defaults block is copy-pasted **5×** (`195`, `215`, `306`, `321`, `372`) → `SetMniDefaults(...)`. *MED dead-code:* intermediate vector `A` (`574-579`) only copied into `ids_to_record` → record directly. `F_eff`/`S_eff` (`139`) are telemetry-only.
- **`soft_anchor.cpp`** — *HIGH long-function:* `Execute` (`288-617`, ~330 lines, 4-deep nesting) → `ScoreCandidates`/`DecideHypothesis`/`PromoteOrUpdateAnchor`. *MED magic-number:* multi-view layout `1536`/`768` as 4 bare literals (`86-90`). *LOW:* `top_contra` hardcoded `0.0` (`:428`) — placeholder or missing computation.
- **`threshold.cpp`** — *HIGH long-function:* `Execute` (`49-244`, 9 phases + 40-attr telemetry). *MED duplication:* `delta_t`/`tau_rate` derivation duplicated in `UpdateRateState::Execute` (`121-128`/`138-144` vs `253-269`) → `ComputeRateTau(...)`.
- **`sensitivity.cpp`** — *HIGH long-function:* `Execute` (`108-285`). *MED magic-number:* valence remap `0.9`/`1.8` (`:185`). *LOW:* `delta_T_sens` non-`const` but never reassigned.
- **`predictive.cpp`** — *HIGH complexity:* `DecayActivePreActivation` (`94-288`) is two structurally identical halves (memory-id `113-199` / embedding-id `201-285`) → parameterize on key kind. `1e-6` floor repeated ~6×.
- **`uncertainty.cpp`** — *HIGH long-function:* `Execute` (`24-352`); the focus-spread re-derivation is §1.5's worst case.
- **`accumulator.cpp`** — *HIGH duplication:* post-reset init block duplicated between new-source (`117-146`) and post-reset (`218-247`) paths → `InitUnit(...)`.
- **`write_gate.cpp`** — *MED complexity:* `e_rep` selection (`106-122`) has a dead first branch assigning an empty vector.
- **`stability.cpp`, `coherence.cpp`** — *MED:* long `Execute` + the windowed-variance / `std::any` duplication of §1.2/§1.5.
- Clean: `effective_focus.cpp`, `precision.cpp`, `stability_feedback.cpp`, `accumulator_reset.cpp`, `accumulator_scores.cpp`, `spike_bypass.cpp`, `focus.cpp`.
- *MED abstraction (4 files):* `focus_feedback.cpp`, `sensitivity_feedback.cpp`, `stability_feedback.cpp`, `influence.cpp` share the "iterate used events → `memory_id>0 ? "...WHERE memory_id=?" : "...WHERE embedding_id=?"`" skeleton (~8 sites) → `UpdateMemoryColumn(tx, e, col, value)` + `ForEachUsedEvent(...)`.

### 2.3 Core / processor headers

- **`operation_context.hpp` / `processor_context.hpp`** — god-objects (§1.6). *MED:* the six by-embedding/by-memory method twins. *LOW:* `SetMetric(…, double value_0_to_100)` (`768`) clamps to `[-1,1]` — the param name is a stale lie. *LOW:* `operation_context.hpp:4` includes the whole `processor.hpp` only for `Config` — layering inversion; hoist `Config` to its own header.
- **`knobs.hpp` (4028 lines)** — *HIGH:* 31 dead functions (§1.8). *MED boilerplate:* ~340 functions of one shape; the 4-line `f=FocusBias(F); s=SensitivityBias(S); …` prelude recurs ~43× verbatim → a `KnobInputs` struct. *MED duplication:* `MemoryTraceTauMultipliers` and `MemoryTraceWeights` (`2266`/`2307`) are byte-identical struct shapes; `*ScoringWeights` structs store both `_raw` and normalized copies with hand-written normalization → a `Normalize(std::array)` helper. *LOW:* terse spec-symbol names (`NCtx`, `WRet`, `TauDt`) mixed with descriptive ones.
- **`operation_set.hpp`** — *MED duplication:* two demangle helpers (`28-50` & `115-134`). *LOW:* instrument-and-time block copy-pasted between `DynamicOperationSet::Execute` and `OperationSet::ExecuteOne` → `RunInstrumented(...)`.
- **`contract_tags.hpp`** — *LOW:* 5 write-only tags (`Violation`, `InterruptAborted`, `DriftAccumSnapshot`, `MniGateDiagnostics`, `StoredSignalId`) appear only in `Satisfies<>`, never `Requires<>` — they add no ordering constraint. Document or drop.
- **`accumulator_state.hpp`** — *LOW duplication:* `Reset` vs `ResetForNextUnit` share ~30 assignment lines → `ResetCommon()` + `SeedWith(embedding)`.
- Clean: `operation_fork.hpp`, `operation.hpp`, `algorithms.hpp`, `utils.hpp`, `sparse.hpp`, `constants.hpp`.

### 2.4 Store layer

- **`store.cpp` (1159 lines)** — *HIGH complexity:* transaction lifecycle spread across overlapping near-duplicate methods (§1.3). *MED duplication:* 5 env-override resolvers open-code the same getenv+lowercase IIFE (§1.1). *MED error-handling:* bind/fetch failures skip telemetry (§1.7). *MED magic-number:* `PRAGMA mmap_size = 268435456`, page-size `512`/`65536` inline (`:412`, `:437`). *LOW:* redundant `if(size>=cap){while(size>=cap)…}` (`735`); `GetWalStatus() const` mutates the WAL and `WalAutoCheckpointPages() const` uses `const_cast` (`1113`) — the `const` is wrong.
- **`schema.cpp` (887 lines)** — *MED long-function:* `GetCoreMigrations` is one 700-line function (migration 0 alone ~370 lines of inline SQL) → split per-migration. *MED redundancy:* several indexes/views are created twice across migrations (`idx_memories_embedding` `400`+`593`; `idx_associations_*` `422`+`729`/`739`; `recent_retrievals` view `459`+`779`) — harmless (`IF NOT EXISTS`) but churny. *LOW:* double-logging on migration failure; manual `typeid` branching in `GetAppliedMigrations` (use `AnyToLongLong`).
- **`object_store.cpp` (430 lines)** — *MED error-handling:* `Exists` swallows all non-transaction `StoreError` as `false` (§1.7). *LOW:* `put_ids_` set is dead in the default direct-SQLite path. No SQL-injection (all bound params).
- **`extension_loader.cpp`** — *LOW:* `LowerEnv` byte-copy (§1.1); `[[maybe_unused]]` helpers mix always-compiled and conditional code.
- **`schema_helpers.hpp`** — *MED abstraction:* four `…Defaults` column strings differ only by a literal kind (`'LONG_TERM'`/`'WORKING'`/…) with manually-synced placeholder counts → bind the kind as a parameter.
- *Note:* the per-row prepare/bind/step/finalize is correctly centralized in `ExecuteDirect`/`BindParameters`/`FetchResultRow` — good. No SQL-injection found anywhere; the one identifier interpolation (`ColumnExists` PRAGMA) is correctly quote-escaped.

### 2.5 Orchestration & C API

- **`cortext.cpp` (2677 lines)** — *HIGH god-function:* `Impl::HydrateContext` (`1461-1694`, ~233 lines) → `PopulateOutputScalars` / `BuildOrderedCandidateIds` / `HydrateRetrievedMemories`. *HIGH duplication:* **8 near-identical `Process*` ingress bodies** (`ProcessText 1833`, `ProcessTextAt 1889`, `ProcessAudio 1945`, `ProcessImage 2011`, three `ReplayIngress::*At`, `ProcessTextEmbeddingAt 1348`) — ~450 lines of copy-paste; `ProcessText` vs `ProcessTextAt` differ by *one line* → a shared `RunIngress(...)` + thin wrappers. *HIGH duplication:* `get_s/ll/dbl/blob` defined 3× (§1.2). *MED:* the `catch(std::exception){LogWarn} catch(...){LogWarn}` pair copy-pasted ~8×; per-memory `getenv` in the hydrate loop (`SourceBlobsDisabled()` at `:711`) → cache as `static const bool`; 8 `Create` overloads + 8 constructors = ~130 lines of forwarding → collapse to one options struct. *MED dead-code:* `HydrateMemory` sets 14 metrics to literal defaults (`660-675`) that are already the struct's defaults; unused `get_dbl`. *MED formatting:* tab corruption at `1518-1537`, `1602-1673`.
- **`signal_processor.cpp` (3012 lines)** — *HIGH god-functions:* `PersistState` (`2408-2576`) and `LoadState` (`1079-1328`) hand-maintain **~70-column** INSERT/loader as three parallel lists kept in sync by comments → drive from a `{column, value}` vector. `PersistWorkingMemory` (`2579-2947`, ~368 lines), `Process` (`2054-2274`, ~220 lines with the timing block repeated ~10× → `TimedStep(...)` + three near-identical catch blocks). *MED:* `LoadWorkingMemory` has an N+1 query pattern (`1665`, `1776`, `1749`); six BLOB decoders share the same dispatch (§1.2); blender metric count `12` hardcoded in 3 places (`1312`, `2441`, `1280`).
- **`capi.cpp` (1583 lines)** — **In good shape.** Error-wrapping is already factored into `invoke_status_only`/`invoke_json`/`invoke_embedding_json` templates; no exception escapes the C boundary; `char*` ownership is consistent (`malloc`/`cortext_string_free`). *MED:* the 5 `create` functions inline a repeated NULL-check + `struct_size` + try/catch prologue → an `invoke_create(fn)` template.
- **`signal_filter.cpp`/`.hpp`, `signal.hpp`, `cortext.hpp`** — clean; well-factored, named constants, shared `DecideAdaptive`.

### 2.6 Models / encoder / audio

- **`aist_gguf_encoder.cpp` (4273 lines)** — the single biggest file; **needs splitting** into ~5-6 files (GGUF I/O, DSP, image preprocessing, two inference backends, tokenizer). *HIGH duplication:* GGUF header parse hand-rolled **4×** (`945`, `1585`, `3066`, `3917`) → `ReadGgufHeader` + `ForEachTensorRecord`. `PreprocessAistImageNHWC` vs `…HWC` (`686` vs `743`) are identical bicubic resamplers differing only in the output index. **Parallel host/ggml implementations** of the same model (projection block ×3, BERT stack ×2, image/audio blocks ×2, block tables duplicated verbatim `1506` vs `2430`) — model-architecture edits must be mirrored or results silently diverge between fallback and kernel paths. *MED dead-code:* `NativeRuntime::EmbeddingLookup` (`3792`) never called; `full_text_graph_error_` (`2908`) never assigned; unused `<iostream>`/`<functional>`/`<numeric>` includes. *Note:* ggml handle management is **leak-free** (RAII + `unique_ptr` on all traced throw paths); only the triplicated destructor boilerplate is a smell → a single RAII `GgmlCtx` wrapper.
- **`graph_retrieval.cpp` (1180 lines)** — *HIGH long-function:* `Execute` (`386-1178`, ~790 lines) → extract seed-load / fanout / source-expansion / reconstruction. *HIGH duplication:* "row → Candidate" block ×4 (`566`, `664`, `845`, `945`); score-combination expression ×5 → `RowToCandidate` / `ScoreCandidate`. *MED:* inconsistent try/catch on optional-feature queries; manual `VALUES(?)…` IN-list with an O(n) param copy; boost factors `0.95`/`0.90` and `output_limit*4` as bare literals.
- **`graph_build.cpp` (457 lines)** — *HIGH:* `kEmbeddingDim=256` silent-empty-graph (§1.4). *MED duplication:* `INSERT…associations` ×8 → `InsertEdge`; `BuildCoOccurrenceEdges` vs `BuildContradictionEdges` near-identical; `typeid` dispatch vs `AnyToLongLong`. *LOW:* `Add` wrapper adds nothing; null-store no-op without telemetry; stale "V2 schema" comment (`398`).
- **`detect_memory_usage.cpp` (395 lines)** — *MED:* `ResolveMemoryIdForEmbedding` does up to 3 queries per embedding in a per-candidate loop — N+1 (`159-205`/`273-298`) → batch with `WHERE embedding_id IN (...)`. *LOW:* long `CreateReinforcementEdges`; two parallel candidate loops (records vs legacy `retrieved`).
- **`constructive_recall_internal.cpp` (818 lines)** — *MED:* `EnvFlag`/`AnyToInt64` duplication (§1.1/§1.2); two `AppendReconstruction*` repeat insert+`last_insert_rowid`. *LOW:* long `LoadCurrentEmbeddingImpl`; undocumented `-1` sentinels.
- **`meta_learning_internal.cpp` (508 lines)** — *MED:* `EnvFlag`/`GetDouble`/`GetInt64` duplication. *LOW:* positional 9-field brace-init in `GetFamilySpec` (use designated initializers); logistic pre-activation `z = …` computed identically in `Evaluate` and `UpdateRowForFamily` → `Preactivation(...)`.
- **`telemetry.cpp` (546 lines)** — *MED error-handling (verify):* `MakeStableStringView` (`220-239`) returns views into a thread_local 64-slot round-robin pool; a record with >64 string attributes could wrap and overwrite a string still referenced in the same emission. Confirm the OTel consumer copies eagerly; otherwise size the pool to max-attrs-per-record. 2048-byte silent truncation is undocumented. *LOW:* Log* family is 10 thin overloads.
- **`thread_config.cpp`** — *MED duplication:* `GetEmbedThreadCount`/`GetInferThreadCount` byte-identical except env-var name → `ThreadCountFromEnv(var)`. *LOW:* `cores/3` default unexplained.
- **`data/centroids.cpp`** — *LOW:* `ComputeArousal`/`ComputeGoalAlignment`/`ComputeViolation` identical except centroid pair → `BipolarScore(emb, hi, lo)`.
- **`neuromodulators.cpp`** — *LOW:* `/1000.0` ms→s bare literal. Otherwise clean.
- **`sherpa_onnx.cpp`** — *LOW:* the `#if !defined(CORTEXT_ENABLE_SHERPA_ONNX)` throw-guard repeated in 4 methods → `CORTEXT_SHERPA_REQUIRE()` macro. Resource handling clean (RAII).
- Clean, no findings: `embedding_model_pin.cpp`, `ggml_support.hpp` (LOW: 8-branch log-level ladder), `planum_bridge.cpp/.hpp`, `text_encoder_factory.hpp`, `internal/cancellation.cpp`.

---

## 3. Prioritized remediation roadmap

Ordered by (impact ÷ risk). Each item is mechanical and independently shippable.

1. **Consolidate copy-pasted utilities** (§1.1, §1.2) — `core::Clamp01` NaN guard + delete 8 copies; new `env_utils.hpp`/`timing.hpp`/`row_accessors.hpp`; add `store::AnyToDouble`. *Touches ~25 files, near-zero behavioral risk, removes the most duplication.*
2. **Single source of truth for formulas/SQL** (§1.3, 2.1) — consolidation score (3×→1), RIF-recovery loop, `memory_strength` summary log (5×→1), `InsertEdge`, `LastInsertId`. *Removes silent-divergence risk on load-bearing logic.*
3. **Embedding-dim constant + fail-loud** (§1.4, §1.7) — `core::kEmbeddingDim`; replace silent `continue`/`false`/null no-ops with logged errors.
4. **Delete dead code** (§1.8, Appendix A) — 31 dead knobs, `NativeRuntime::EmbeddingLookup`, `full_text_graph_error_`, boundary timeout branch, `HydrateMemory` default-metric block, unused includes.
5. **Decompose the worst god-functions** — `RunIngress` (cortext.cpp 8×), data-driven `PersistState`/`LoadState`, `HydrateContext`, `graph_retrieval::Execute`, the §2.2 mega-`Execute`s. *Higher effort; do behind tests.*
6. **Extract the context blackboards** (§1.6) and **split `aist_gguf_encoder.cpp`** — largest structural wins, highest effort; schedule deliberately.
7. **Formatting pass** — `clang-format` the tab-corrupted ranges in `cortext.cpp`, `signal_processor.cpp`, `memory_storage.cpp`. (Repo has no checked-in `.clang-format`; adopting one would prevent recurrence.)

---

## Appendix A — Dead & test-only knob functions

**31 with zero callers anywhere** (delete candidates):
`AssociationBoost`, `BoundaryWeightDrift`, `BoundaryWeightGap`, `LabelSalienceFallback`, `MinEdgeWeight`, `RetrievalBaseLevelAvailabilityCountSaturation`, `RetrievalBaseLevelAvailabilityTauSeconds`, `RetrievalBaseLevelAvailabilityWeight`, `RetrievalDiversificationWeights`, `RetrievalDurableSourceSetBaseWeight`, `RetrievalDurableSourceSetMinScore`, `RetrievalDurableSourceSetWeight`, `RetrievalEmotionWeight`, `RetrievalEvidenceBlendMaxMembers`, `RetrievalEvidenceBlendTemperature`, `RetrievalEvidenceBlendTieMargin`, `RetrievalGraphDepth`, `RetrievalGraphExpandedRagRelationWeight`, `RetrievalGraphExpandedRagTemporalWindow`, `RetrievalGraphExpansionRowLimit`, `RetrievalPartialMatchContradictionSaturation`, `RetrievalPartialMatchModalityMismatchWeight`, `RetrievalPartialMatchPenaltyWeight`, `RetrievalPartialMatchSourceMismatchWeight`, `RetrievalRecentInhibitionTauSeconds`, `RetrievalRecentInhibitionWeight`, `RetrievalSourceContradictionPenalty`, `RetrievalTemporalRankTauSeconds`, `RetrievalTemporalRankWeight`, `RetrievalTextQueryWMChars`, `RetrievalTextQueryWMSlots`.

These cluster in retrieval features (`BaseLevelAvailability`, `PartialMatch`, `RecentInhibition`, `TemporalRank`, `DurableSourceSet`, `EvidenceBlend`, `GraphExpandedRag`) — an abandoned/stubbed retrieval line. Verify against `docs/paper/sections/` before deleting in case any are documented-but-unwired.

**~42 referenced only by tests** (decide aspirational vs removable): `CheckIntervalTokens`, `ConsolidationRate`, `DerivedSourceFallbackEdgeWeight`, `FlashbulbThreshold`, `GraphBuildSequentialTauSeconds`, `GraphDepth`, `IdleRequiredSeconds`, `LabelCooccurrenceEdgeWeight`, `LabelFrequencyThreshold`, `MaxResults`, `MaxWaitTokens`, `MemoryUsageCacheDuration`, `MemoryUsageThreshold`, `MinEpisodesForConcept`, `RLSWindowN`, `RetrievalCandidateBlendScoringWeights`, `RetrievalContextMix`, `RetrievalContextReinstatementAlpha`, `RetrievalDurableSourceSupportSaturationCount`, `RetrievalFocusBias`, `RetrievalGraphExpansionEvidenceCounts`, `RetrievalMemoryAffectScoringWeights`, `RetrievalPredictiveBonusWeight`, `RetrievalPressureGateLowScale`, `RetrievalPressureRampLowScale`, `RetrievalProceduralSeedFanout`, `RetrievalProceduralSeedMinScore`, `RetrievalProceduralSeedReserveCount`, `RetrievalRouteTokenMinChars`, `RetrievalRoutineRecencyAdjustmentPolicy`, `RetrievalSeedFallbackSourceConfidence`, `RetrievalSensitivityBias`, `RetrievalSourceBackedBoostFloor`, `RetrievalSourceConfidenceThreshold`, `RetrievalSourceFreshnessTauSeconds`, `RetrievalSourceFreshnessWeight`, `RetrievalStabilityBias`, `RetrievalStalePenaltyStrongMultiplier`, `RetrievalThreshold`, `RetrievalTokenOverlapQueryWeight`, `RetrievalVectorDistanceScore`, `TargetPrecision`.

*(Dead-knob lists are grep-derived; a few may be reached via macro/alias — confirm with a compile after deletion. `core::` helpers used only by other knobs, e.g. `BiasMid`, `AffectSensitivityBias`, are **not** in these lists.)*
