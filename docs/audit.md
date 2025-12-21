# Audit Log — Manuscript Conformance (In Progress)
Date: 2025-12-19

## Scope
- Manuscript: `docs/paper/_manuscript/index.md`
- ER schema: `docs/paper/diagrams/entity-relationship.qmd`
- Code focus: `src/operations/*`, `include/cortext/core/knobs.hpp`, `include/cortext/processor/*`, `src/cortext.cpp`, `src/signal_processor.cpp`, `src/store/schema.cpp`

## Audit Log
- 2025-12-19: Started deep audit. Reviewed manuscript sections (Focus/Sensitivity/Stability, Metrics, Threshold, Accumulator/Boundary, Write gate, Uncertainty, Interrupt gate, Graph, Working Memory, Metacognition, Consolidation, Appendix rules). Traced pipeline order in `src/cortext.cpp` and compared per-step computations.
- 2025-12-19: Continued audit. Reviewed serial position effects, reconsolidation, retrieval competition, predictive pre-activation, emotional consolidation/cascade, consolidation scheduling, episode handling, and state persistence details.
- 2025-12-19: Completed deep audit and cross-checked all manuscript headings against code; no additional sections found beyond the findings listed below.
- 2025-12-19: Resumed audit to incorporate additional spec-finalized checks; added further deviations (uncertainty normalization edge cases, accumulator reset timing, cold-start timestamps, emotional consolidation gating, etc.).
- 2025-12-19: Updated extraction JSON handling to accept `labels` as the preferred payload key (with `entities` as legacy fallback) for schema/prompt consistency with the manuscript.
- 2025-12-19: Enforced labels-only extraction payloads (clean break): removed `entities` schema/parse fallback and renamed tool to `extract_labels_and_relations`.
- 2025-12-19: Re-checked semantic extraction section vs code: manuscript specifies labels/tags only; code still required label `type` and model-provided salience. Marked for removal with salience computed from embeddings instead.
- 2025-12-19: Implemented label-only extraction payloads (string arrays), computed label salience from embeddings, and updated manuscript/diagram references (label_frequency_threshold, no label types).
- 2025-12-20: Completed Session API + JSON-schema constrained label extraction. Enforced `labels` as required with `minItems=1`, added constrained decoding retries for Gemma, and tightened prompts to always emit at least one label. Added constraint unit tests (labels required, relation required fields) and updated Gemma integration test to assert strict schema conformance. Tests run: `cmake --build build --target cortext_tests`, `./build/tests/cortext_tests -s "[constraint]"`, `./build/tests/cortext_tests -s "[extractor][gemma][integration]"`.
- 2025-12-21: Updated Alpha schedules to manuscript constants, removed rate_target EWMA/scaling (base_rate only), normalized drift_acc by 1/2, and set structural coherence fallback to 0.5 when context < 2.
- 2025-12-21: Fixed boundary coherence drop (/2 clamp) with drift spike cold-start guard and applied write-gate refractory for all Δt with spike-bypass force-write.
- 2025-12-21: Updated uncertainty blending/normalization to use structural coherence and weighted novelty-surprise, corrected interrupt-gate coverage/dup threshold, added mood decay with Δt_mood + last_mood_ts + centered e_t, and fixed consolidation interval/rate units.
- 2025-12-21: Ran full test baseline after conformance changes; 15 failures remain (mostly migration/state tests missing encoder injection and uncertainty fallback test mismatch). Tracked in `docs/plans/conformance.plan.md` for follow-up fixes.
- 2025-12-21: Fixed migration/state persistence tests to use a test encoder (MakeConfig/RequireEncoder) now that encoder is mandatory when EmbeddingGemma is off.
- 2025-12-21: Rebuilt tests and ran `./build/tests/cortext_tests`; all tests passed (386 test cases, 2196 assertions).
- 2025-12-21: Switched Table‑1 metrics to use raw `F` (not `F_eff`) for relevance/mismatch/rarity/utility/coverage/salience/contradiction; metrics tests pass.
- 2025-12-21: PersistSignalMetrics now updates all signals in the current memory (not just the latest row); memory_storage tests pass.
- 2025-12-21: Drift spike now uses pre-update `eta_prev` baseline captured in `ComputeCoherence`.
- 2025-12-21: `write_rate_window_` now records only write events and trims by time window (`w_rate_seconds(T)`).
- 2025-12-21: Graph retrieval now enforces `graph_depth(T)` + `min_edge_weight(F)`, skips when `memory_stream` is empty, adds `implies`/`similar_to` edges, and uses non-saturating reinforcement increments.
- 2025-12-21: Implemented accumulator coherence window (`acc_signals_window`), persisted it, removed duplicate coherence pass, and added immediate accumulator reset after flush/spike (pre-interrupt gate).
- 2025-12-21: Switched `recent_memory_centroids` to `e_rep` with `win_mem_ctx(T)`, stopped capping `memory_stream`, and wired structural coherence into `context.SetCoherence` for persistence/interrupt gating.
- 2025-12-21: Rebuilt tests and ran `./build/tests/cortext_tests`; all tests passed (386 test cases, 2196 assertions).
- 2025-12-21: Updated RLS blending to use weight-space `blender_state` in composite scoring (no stale coefficient-path), fixed covariance update formula, and removed coefficient-RLS persistence.
- 2025-12-21: Interrupt gate now uses `candidate_star` values (rel/novelty/overlap) instead of max across candidates.
- 2025-12-21: Working memory benefit now uses `relevance_to_task(μ_acc, task_context)` (task relevance from context embeddings) instead of signal relevance.
- 2025-12-21: Metacognitive monitoring now computes FOK (from uncertainty) and retrieval strength (from usage events) each step and updates processor state.
- 2025-12-21: Memory storage now concatenates per-signal blobs for memory content and persists `drift_acc` into `memories.drift_mag` (mem_elapsed derivable from start/end timestamps).
- 2025-12-21: Serial position multiplier now modulates memory strength updates (applied to delta strength).
- 2025-12-21: Applied label frequency threshold during extraction; labels/relations below threshold are skipped.
- 2025-12-21: Streaming pacing now enforces `adjacent_window(F)` using time since last retrieval check.
- 2025-12-21: Persisted `theta_target`, set cold-start threshold defaults from priors, and mark priors initialized after loading state.
- 2025-12-21: Persisted additional processor state (focus/sensitivity/stability weights, metacognition, consolidation, delta_half_life_adj) and load/apply them on startup.
- 2025-12-21: Reconsolidation now decays lability over `τ_labile` using stored lability state/timestamps.
- 2025-12-21: Influence feedback now neutralizes generation terms when no generation embeddings are available.
- 2025-12-21: Rebuilt and ran `./build/tests/cortext_tests`; all tests passed (386 test cases, 2196 assertions).
- 2025-12-21: Aligned memory-usage detection with interrupt-gate selection (removed cache-based similarity heuristic), added selected-candidate tracking, and moved usage detection to post-gate.
- 2025-12-21: Emotional consolidation now tags stored memories (not usage-gated), applies flashbulb threshold, and persists emotional fields on write.
- 2025-12-21: Stability update now computes observed retention from active memories, updates retention_ema, and maintains retention history each step.
- 2025-12-21: Consolidation scheduling now enforces idle + not-accumulating requirement; clustering switched to density-based (DBSCAN-style).
- 2025-12-21: Composite score now applies control-weight multipliers and score_gain; focus spread now respects attention_width.
- 2025-12-21: Influence feedback now updates per-memory mean_influence.
- 2025-12-21: Rebuilt and ran `./build/tests/cortext_tests`; all tests passed (384 test cases, 2190 assertions).
- 2025-12-21: Clarified control-weight usage and attention_width effect in `docs/paper/sections/3_structural_metrics.qmd` (composite scoring + focus spread).

## Findings (Draft, Ongoing)

### Formula deviations
- (Resolved 2025-12-21) Alpha schedules (`AlphaT/AlphaF/AlphaS`) use `max(...)` with uncertainty floor; spec is linear. Also `α_min_T`/`α_span_T` constants differ from manuscript. (See `include/cortext/core/knobs.hpp`.)
- (Resolved 2025-12-21) `rate_target` prior deviates: code uses `base_rate * (0.5 + 1.5S)`; spec is `base_rate(S)` only. (`src/operations/sensitivity.cpp`.)
- (Resolved 2025-12-21) `rate_target` is dynamically updated via EWMA of observed (signal‑rate) in code; spec defines it as a knob‑derived constant `base_rate(S)` with no update rule. (`docs/paper/_manuscript/index.md`, `src/operations/sensitivity.cpp`.)
- (Resolved 2025-12-21) Boundary coherence drop formula missing `/2` and clamp; drift spike lacks cold-start guard. (`src/operations/boundary.cpp`.)
- (Resolved 2025-12-21) `drift_acc` accumulates raw drift (no `/2` normalization). (`include/cortext/processor/accumulator_state.hpp`.)
- (Resolved 2025-12-21) Structural coherence default is `1.0` when context <2; spec says `0.5`. (`src/operations/coherence.cpp`.)
- (Resolved 2025-12-21) Write refractory applied only when `dt_write < 3 * tau`; spec applies exponential form for all `dt`. (`src/operations/write_gate.cpp`.)
- (Resolved 2025-12-21) Spike bypass does **not** force write; code still requires `S_window > θ_memory`. (`src/operations/write_gate.cpp`.)
- (Resolved 2025-12-21) Uncertainty novelty_surprise blend uses simple average when both present; spec requires weighted blend `normalize([S, 1−T])`. (`src/operations/uncertainty.cpp`.)
- (Resolved 2025-12-21) Uncertainty weight normalization: when weights sum to ~0 (e.g., F=S=0, T=1), spec says `normalize` returns uniform weights; code falls back to `u_raw=1−maturity` instead of averaging available metrics. (`docs/paper/_manuscript/index.md`, `src/operations/uncertainty.cpp`.)
- (Resolved 2025-12-21) Uncertainty uses a recomputed `coherence_complement` and drops the term entirely when `|recent_context|<2`; spec says use `coherence_struct_t` (with 0.5 fallback), so coherence should still contribute. (`docs/paper/_manuscript/index.md`, `src/operations/uncertainty.cpp`.)
- (Resolved 2025-12-21) Coverage gain for interrupt MU uses `sim_ctx - redundancy` instead of `1 - redundancy`. (`src/operations/interrupt_gate.cpp`.)
- (Resolved 2025-12-21) Duplicate suppression threshold is reversed: code lerps 0.88→0.96 with F; spec is 0.96→0.88. (`src/operations/interrupt_gate.cpp`.)
- (Resolved 2025-12-21) Mood integration ignores time decay and centering: `λ_mood` is constant (no `Δt_mood`), `last_mood_ts` is unused, and `e_t` is not centered (`p_c - 1/6`). (`src/operations/sensitivity.cpp`, `include/cortext/core/knobs.hpp`.)
- (Resolved 2025-12-21) Consolidation interval and extraction interval compare **ms timestamps** to **second** thresholds (missing ×1000 conversion). (`src/operations/consolidation.cpp`.)
- (Resolved 2025-12-21) Consolidation rate trigger compares `m_rate` (writes/min) against `core::ConsolidationRate` (computed from seconds interval without ×60); spec’s trigger uses the write‑rate target (`rate_target`) and expects consistent units. (`docs/paper/_manuscript/index.md`, `src/operations/consolidation.cpp`, `include/cortext/core/knobs.hpp`.)

### Missing / divergent behavior
- (Resolved 2025-12-21) **Accumulator coherence window missing**: `acc_signals_window` not implemented; coherence computed vs `mu_acc` instead of raw mean cosine over window. (`src/operations/coherence.cpp`, `include/cortext/processor/accumulator_state.hpp`.)
- (Resolved 2025-12-21) `reset_accumulator()` does **not** clear `eta_acc` or `coherence_prev` (spec says clear). (`include/cortext/processor/accumulator_state.hpp`.)
- (Resolved 2025-12-21) Flush reset is deferred: `ComputeWriteGate` only sets `n_signals=0` and does not clear `mu_acc`/`drift_acc`/etc. before the interrupt gate; spec requires immediate `reset_accumulator()` on `should_flush` (even when write is rejected). (`docs/paper/_manuscript/index.md`, `src/operations/accumulator_reset.cpp`, `src/cortext.cpp`.)
- (Resolved 2025-12-21) `ComputeCoherence` runs twice per signal (before and after accumulator update), updating `eta_acc` twice and using updated `mu_acc` when it should use pre-append window. (`src/cortext.cpp`.)
- (Resolved 2025-12-21) Boundary gap detection uses `last_signal_ts` updated **before** boundary; thus gap is often zero. Spec says compute gap before update. (`src/operations/boundary.cpp`, `src/operations/accumulator.cpp`.)
- (Resolved 2025-12-21) `recent_memory_centroids` stores `mu_acc` and length `NCtx(T)`; spec says store `e_rep` and size `win_mem_ctx(T)=lerp(4,32,T)`. (`src/operations/write_gate.cpp`, `include/cortext/core/knobs.hpp`.)
- (Resolved 2025-12-21) `memory_stream` is capped to `NCtx(T)`; spec treats it as the full stream of written memories for focus_spread/uncertainty kNN queries. (`src/operations/write_gate.cpp`.)
- Structural coherence is never stored in `context.SetCoherence`, so `signals.coherence` persists as 0 and interrupt coherence penalty is wrong. (`src/operations/signal_metrics_persistence.cpp`, `src/operations/interrupt_gate.cpp`.)
- (Resolved 2025-12-21) Core metric formulas use `F_eff` (effective focus) instead of raw `F` for relevance/mismatch/rarity/utility/coverage/salience/contradiction; spec uses `F` in Table 1. (`src/operations/metrics.cpp`.)
- (Resolved 2025-12-21) RLS blending: `FitMetricWeightsRLS` updates `blender_state`, but `ComputeCompositeScore` ignores it and uses `rls_coefficients` that are never updated. Weight adaptation effectively no-op. (`src/operations/blend.cpp`.)
- (Resolved 2025-12-21) RLS covariance update appears incorrect (scalar applied where matrix multiply expected). (`src/operations/blend.cpp`.)
- (Resolved 2025-12-21) Per-signal metrics persistence updates only the latest signal row per source+timestamp; earlier signals in same memory never get metrics. (`src/operations/signal_metrics_persistence.cpp`.)
- (Resolved 2025-12-21) Graph retrieval should skip retrieval when `memory_stream` is empty; code queries embeddings regardless. (`src/operations/graph_retrieval.cpp`.)
- (Resolved 2025-12-21) Graph depth should use `graph_depth(T)`; code uses `GraphDepth(F)`. (`include/cortext/core/knobs.hpp`, `src/operations/graph_retrieval.cpp`.)
- (Resolved 2025-12-21) Graph traversal ignores `min_edge_weight(F)`; no filtering in traversal. (`src/operations/graph_retrieval.cpp`.)
- (Resolved 2025-12-21) Missing edge types in graph build: `implies` and `similar_to` are not created in automated graph build. (`src/operations/graph_build.cpp`.)
- (Resolved 2025-12-21) Reinforcement edges saturate immediately (`MIN(weight + 1.0, 1.0)`), not frequency-scaled. (`src/operations/graph_retrieval.cpp`.)
- (Resolved 2025-12-21) Interrupt gate uses max relevance/novelty across candidates instead of `candidate_star` values; overlap_star is max across candidates. Spec uses candidate_star values. (`src/operations/interrupt_gate.cpp`.)
- (Resolved 2025-12-21) Interrupt gate coherence penalty uses `context.GetCoherence()` which is never set to structural coherence. (`src/operations/interrupt_gate.cpp`, `src/operations/coherence.cpp`.)
- Interrupt gate coverage_gain uses `max(0, sim_ctx − redundancy)`; spec defines `coverage_gain = 1 − redundancy` (Appendix B/C). (`src/operations/interrupt_gate.cpp`, `docs/paper/_manuscript/index.md`.)
- (Resolved 2025-12-21) Working memory benefit uses signal relevance instead of `relevance_to_task(μ_acc, task_context)`. (`src/operations/working_memory.cpp`.)
- (Resolved 2025-12-21) Metacognitive FOK/retrieval strength never computed; thresholds output but no FOK state updates. (`src/operations/metacognitive.cpp`.)
- (Resolved 2025-12-21) Emotional consolidation fields (`flashbulb_threshold`, `half_life_bonus`, `detail_suppression`, `gist_components`) are stored but never used to adjust decay or retrieval. (`src/operations/emotion*.cpp`, `src/operations/memory_strength.cpp`.)
- (Resolved 2025-12-21) Emotional consolidation is executed only for **used** memories during the main loop; spec describes consolidation‑time tagging based on stored memory metadata (not usage‑gated). (`docs/paper/_manuscript/index.md`, `src/operations/emotion.cpp`.)
- (Resolved 2025-12-21) `flashbulb_threshold` is never applied; any memory passing θ_intensity/θ_arousal is marked `flashbulb=1`, regardless of the separate flashbulb threshold in the spec. (`docs/paper/_manuscript/index.md`, `src/operations/emotion.cpp`, `include/cortext/core/knobs.hpp`.)
- (Resolved 2025-12-21) Emotion probability fallback: when all cos ≤ 0, spec uses uniform distribution; code leaves zeros. (`src/operations/sensitivity.cpp`.)
- (Resolved 2025-12-21) Observed retention/retention_ema missing: `SetObservedRetentionSeconds` is never called, `retention_ema` is never updated, and active-memory mean age is not computed per step. Stability update relies only on preloaded history. (`src/operations/stability.cpp`, `src/signal_processor.cpp`, `include/cortext/processor/operation_context.hpp`.)
- (Resolved 2025-12-21) Memory content spec says concatenated blobs; storage uses last blob only for memory-level content. (`src/operations/memory_storage.cpp`.)
- (Resolved 2025-12-21) Accumulator metadata for memory storage missing `mem_elapsed` and normalized `drift_acc`; `drift_mag` is not stored in memories. (`src/operations/memory_storage.cpp`.)
- (Resolved 2025-12-21) Serial-position effects computed but **never applied** to any scoring/strength update; `serial_position_multiplier` is output-only. (`src/operations/serial_position*.cpp`.)
- (Resolved 2025-12-21) `attention_width`, `weight_relevance`, `mismatch_weight`, `coverage_gain_floor`, and sensitivity control weights (`weight_surprise`, `weight_valence`, `weight_arousal`, `emotion_gain`, `score_gain`) are computed but unused in scoring/gating. (Multiple files.)
- (Resolved 2025-12-21) Sensitivity feedback uses novelty of the **current signal** for all used memories and redundancy from global kNN, instead of per-memory novelty vs recent context and per-memory redundancy vs context. (`src/operations/sensitivity_feedback.cpp`.)
- (Resolved 2025-12-21) Memory-usage detection treats cached retrievals as “retrieved” for subsequent signals; `used` is based on similarity threshold, not interrupt injection, so `retrieved_count`/`used_count` diverge from spec. (`src/operations/detect_memory_usage.cpp`, `src/operations/memory_strength.cpp`.)
- (Resolved 2025-12-21) Consolidation scheduling ignores `is_accumulating_memory` and relaxes idle requirement when interval trigger fires (spec requires idle). (`src/operations/consolidation.cpp`.)
- (Resolved 2025-12-21) Consolidation clustering uses greedy single-linkage instead of DBSCAN/k-means (spec calls out density-based or k-means). (`src/operations/consolidation_cluster.cpp`.)
- (Resolved 2025-12-21) Entity frequency threshold is never applied to extraction results. (`include/cortext/core/knobs.hpp`, `src/operations/process_extraction_results.cpp`.)
- (Resolved 2025-12-19) Extraction schema now uses label-only payloads and computes salience from embeddings; docs updated to match. (`docs/paper/_manuscript/index.md`, `src/operations/process_extraction_results.cpp`, `src/extractor/*`.)
- (Resolved 2025-12-21) `write_rate_window_` records **signal** timestamps, not write events, so rate_target updates use signal rate rather than memory-write rate (spec says write_rate is memory writes). (`src/signal_processor.cpp`.)
- (Resolved 2025-12-21) `write_rate_window_` capacity uses `lerp(10,60,T)` count, not `w_rate_seconds(T)=lerp(60,300,T)` seconds. (`src/signal_processor.cpp`, `include/cortext/processor/processor_context.hpp`.)
- (Resolved 2025-12-21) Reconsolidation ignores `τ_labile` and does not decay lability over time; `current_lability` is fixed to susceptibility. (`src/operations/reconsolidation.cpp`.)
- (Resolved 2025-12-21) `theta_target` is not tracked; persisted as `theta_dynamic` by default. (`src/signal_processor.cpp`.)
- (Resolved 2025-12-21) Persisted focus/sensitivity/stability state is overwritten: `LoadState` does not set `*_priors_initialized`, so `InitializeFocusPriors` / `InitializeSensitivityPriors` / `InitializeStabilityPriors` re-seed and clobber loaded values. (Multiple files.)
- (Resolved 2025-12-21) Cold-start threshold state does not follow Appendix A defaults: `theta_dynamic` is initialized to a fixed 0.2 (not `θ_prior(F,S,T)`), and `hysteresis` starts at a fixed 0.05 rather than `base_band(T)` until `UpdateThreshold` runs. (`include/cortext/processor/processor_context.hpp`, `src/operations/threshold.cpp`.)
- Cold-start `last_rate_timestamp` should initialize to `now_ms()` per Appendix A; code defaults to `0`, affecting early Δt/rate estimates. (`include/cortext/processor/processor_context.hpp`, `docs/paper/_manuscript/index.md`.)
- (Resolved 2025-12-21) Drift spike uses **post‑EWMA** `eta_acc` as baseline because `ComputeCoherence` updates `eta_acc` before `DetectBoundary`. Spec uses `eta_prev` (pre‑update) for spike calculation. (`src/operations/coherence.cpp`, `src/operations/boundary.cpp`.)
- Accumulator `coherence_prev` initializes to `1.0` in code; spec initializes to `0` and resets to `0` at boundaries. (`include/cortext/processor/accumulator_state.hpp`, `docs/paper/_manuscript/index.md`.)
- (Resolved 2025-12-21) `adjacent_window(F)` is specified in streaming pacing but never used in implementation. (`docs/paper/_manuscript/index.md`, `include/cortext/core/knobs.hpp`, `src/operations/streaming_pacing.cpp`.)
- (Resolved 2025-12-21) Influence tracking uses recent input embeddings as generation trajectory; spec applies drift on **generation embeddings when available** and otherwise should skip/neutralize that term. (`src/operations/influence.cpp`.)

### Persistence / schema gaps
- (Resolved 2025-12-21) State fields required by manuscript are not persisted or loaded: `coverage_gain_floor`, `mismatch_weight`, `weight_surprise`, `weight_valence`, `weight_arousal`, `emotion_gain`, `score_gain`, `drift_weight`, `retention_ema`, `last_retrieval_ts`. Save uses constants; load reads only `weight_novelty`. (`src/signal_processor.cpp`.) (Resolved 2025-12-21: `last_mood_ts` now persisted/loaded.)
- (Resolved 2025-12-21) `acc_signals_window` exists in schema but not in `AccumulatorState`, and is neither saved nor loaded. (`docs/paper/diagrams/entity-relationship.qmd`, `src/signal_processor.cpp`.)
- (Resolved 2025-12-21) `signals.coherence` field persists `context.GetCoherence()` which is never set; should store structural coherence. (`src/operations/signal_metrics_persistence.cpp`, `src/operations/coherence.cpp`.)
- `retention_ema` is persisted in schema but never updated or read for stability updates. (`src/operations/stability.cpp`, `src/signal_processor.cpp`.)
- (Resolved 2025-12-21) `mean_influence` is never updated; `last_mood_ts` is updated but needs verification in downstream usage. (`src/operations/influence.cpp`, `src/operations/stability_feedback.cpp`.)
- (Resolved 2025-12-21) `last_retrieval_ts` was updated in memory but **not persisted**; `delta_half_life_adj` was always persisted as `0.0` (ignores computed adjustments). (`src/signal_processor.cpp`, `src/operations/stability_feedback.cpp`.)
- (Resolved 2025-12-21) Metacognitive state fields in schema (`fok_state`, `retrieval_strength`, `metacognitive_confidence`) were never persisted or updated. (`src/store/schema.cpp`, `src/operations/metacognitive.cpp`.)

## Next Checks (Planned)
- None — audit complete.

## Notes
This file will be updated as the audit continues.
