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

## Findings (Draft, Ongoing)

### Formula deviations
- Alpha schedules (`AlphaT/AlphaF/AlphaS`) use `max(...)` with uncertainty floor; spec is linear. Also `α_min_T`/`α_span_T` constants differ from manuscript. (See `include/cortext/core/knobs.hpp`.)
- `rate_target` prior deviates: code uses `base_rate * (0.5 + 1.5S)`; spec is `base_rate(S)` only. (`src/operations/sensitivity.cpp`.)
- `rate_target` is dynamically updated via EWMA of observed (signal‑rate) in code; spec defines it as a knob‑derived constant `base_rate(S)` with no update rule. (`docs/paper/_manuscript/index.md`, `src/operations/sensitivity.cpp`.)
- Boundary coherence drop formula missing `/2` and clamp; drift spike lacks cold-start guard. (`src/operations/boundary.cpp`.)
- `drift_acc` accumulates raw drift (no `/2` normalization). (`include/cortext/processor/accumulator_state.hpp`.)
- Structural coherence default is `1.0` when context <2; spec says `0.5`. (`src/operations/coherence.cpp`.)
- Write refractory applied only when `dt_write < 3 * tau`; spec applies exponential form for all `dt`. (`src/operations/write_gate.cpp`.)
- Spike bypass does **not** force write; code still requires `S_window > θ_memory`. (`src/operations/write_gate.cpp`.)
- Uncertainty novelty_surprise blend uses simple average when both present; spec requires weighted blend `normalize([S, 1−T])`. (`src/operations/uncertainty.cpp`.)
- Uncertainty weight normalization: when weights sum to ~0 (e.g., F=S=0, T=1), spec says `normalize` returns uniform weights; code falls back to `u_raw=1−maturity` instead of averaging available metrics. (`docs/paper/_manuscript/index.md`, `src/operations/uncertainty.cpp`.)
- Uncertainty uses a recomputed `coherence_complement` and drops the term entirely when `|recent_context|<2`; spec says use `coherence_struct_t` (with 0.5 fallback), so coherence should still contribute. (`docs/paper/_manuscript/index.md`, `src/operations/uncertainty.cpp`.)
- Coverage gain for interrupt MU uses `sim_ctx - redundancy` instead of `1 - redundancy`. (`src/operations/interrupt_gate.cpp`.)
- Duplicate suppression threshold is reversed: code lerps 0.88→0.96 with F; spec is 0.96→0.88. (`src/operations/interrupt_gate.cpp`.)
- Mood integration ignores time decay and centering: `λ_mood` is constant (no `Δt_mood`), `last_mood_ts` is unused, and `e_t` is not centered (`p_c - 1/6`). (`src/operations/sensitivity.cpp`, `include/cortext/core/knobs.hpp`.)
- Consolidation interval and extraction interval compare **ms timestamps** to **second** thresholds (missing ×1000 conversion). (`src/operations/consolidation.cpp`.)
- Consolidation rate trigger compares `m_rate` (writes/min) against `core::ConsolidationRate` (computed from seconds interval without ×60); spec’s trigger uses the write‑rate target (`rate_target`) and expects consistent units. (`docs/paper/_manuscript/index.md`, `src/operations/consolidation.cpp`, `include/cortext/core/knobs.hpp`.)

### Missing / divergent behavior
- **Accumulator coherence window missing**: `acc_signals_window` not implemented; coherence computed vs `mu_acc` instead of raw mean cosine over window. (`src/operations/coherence.cpp`, `include/cortext/processor/accumulator_state.hpp`.)
- `reset_accumulator()` does **not** clear `eta_acc` or `coherence_prev` (spec says clear). (`include/cortext/processor/accumulator_state.hpp`.)
- Flush reset is deferred: `ComputeWriteGate` only sets `n_signals=0` and does not clear `mu_acc`/`drift_acc`/etc. before the interrupt gate; spec requires immediate `reset_accumulator()` on `should_flush` (even when write is rejected). (`docs/paper/_manuscript/index.md`, `src/operations/write_gate.cpp`.)
- `ComputeCoherence` runs twice per signal (before and after accumulator update), updating `eta_acc` twice and using updated `mu_acc` when it should use pre-append window. (`src/cortext.cpp`.)
- Boundary gap detection uses `last_signal_ts` updated **before** boundary; thus gap is often zero. Spec says compute gap before update. (`src/operations/boundary.cpp`, `src/operations/accumulator.cpp`.)
- `recent_memory_centroids` stores `mu_acc` and length `NCtx(T)`; spec says store `e_rep` and size `win_mem_ctx(T)=lerp(4,32,T)`. (`src/operations/write_gate.cpp`.)
- `memory_stream` is capped to `NCtx(T)`; spec treats it as the full stream of written memories for focus_spread/uncertainty kNN queries. (`src/operations/write_gate.cpp`, `src/operations/focus_spread.cpp`, `src/operations/uncertainty.cpp`.)
- Structural coherence is never stored in `context.SetCoherence`, so `signals.coherence` persists as 0 and interrupt coherence penalty is wrong. (`src/operations/signal_metrics_persistence.cpp`, `src/operations/interrupt_gate.cpp`.)
- Core metric formulas use `F_eff` (effective focus) instead of raw `F` for relevance/mismatch/rarity/utility/coverage/salience/contradiction; spec uses `F` in Table 1. (`src/operations/metrics.cpp`.)
- RLS blending: `FitMetricWeightsRLS` updates `blender_state`, but `ComputeCompositeScore` ignores it and uses `rls_coefficients` that are never updated. Weight adaptation effectively no-op. (`src/operations/blend.cpp`.)
- RLS covariance update appears incorrect (scalar applied where matrix multiply expected). (`src/operations/blend.cpp`.)
- Per-signal metrics persistence updates only the latest signal row per source+timestamp; earlier signals in same memory never get metrics. (`src/operations/signal_metrics_persistence.cpp`.)
- Graph retrieval should skip retrieval when `memory_stream` is empty; code queries embeddings regardless. (`src/operations/graph_retrieval.cpp`.)
- Graph depth should use `graph_depth(T)`; code uses `GraphDepth(F)`. (`include/cortext/core/knobs.hpp`, `src/operations/graph_retrieval.cpp`.)
- Graph traversal ignores `min_edge_weight(F)`; no filtering in traversal. (`src/operations/graph_retrieval.cpp`.)
- Missing edge types in graph build: `implies` and `similar_to` are not created in automated graph build. (`src/operations/graph_build.cpp`.)
- Reinforcement edges saturate immediately (`MIN(weight + 1.0, 1.0)`), not frequency-scaled. (`src/operations/graph_retrieval.cpp`.)
- Interrupt gate uses max relevance/novelty across candidates instead of `candidate_star` values; overlap_star is max across candidates. Spec uses candidate_star values. (`src/operations/interrupt_gate.cpp`.)
- Interrupt gate coherence penalty uses `context.GetCoherence()` which is never set to structural coherence. (`src/operations/interrupt_gate.cpp`.)
- Interrupt gate coverage_gain uses `max(0, sim_ctx − redundancy)`; spec defines `coverage_gain = 1 − redundancy` (Appendix B/C). (`src/operations/interrupt_gate.cpp`, `docs/paper/_manuscript/index.md`.)
- Working memory benefit uses signal relevance instead of `relevance_to_task(μ_acc, task_context)`. (`src/operations/working_memory.cpp`.)
- Metacognitive FOK/retrieval strength never computed; thresholds output but no FOK state updates. (`src/operations/metacognitive.cpp`.)
- Emotional consolidation fields (`flashbulb_threshold`, `half_life_bonus`, `detail_suppression`, `gist_components`) are stored but never used to adjust decay or retrieval. (`src/operations/emotion*.cpp`, `src/operations/memory_strength.cpp`.)
- Emotional consolidation is executed only for **used** memories during the main loop; spec describes consolidation‑time tagging based on stored memory metadata (not usage‑gated). (`docs/paper/_manuscript/index.md`, `src/operations/emotion.cpp`.)
- `flashbulb_threshold` is never applied; any memory passing θ_intensity/θ_arousal is marked `flashbulb=1`, regardless of the separate flashbulb threshold in the spec. (`docs/paper/_manuscript/index.md`, `src/operations/emotion.cpp`, `include/cortext/core/knobs.hpp`.)
- Emotion probability fallback: when all cos ≤ 0, spec uses uniform distribution; code leaves zeros. (`src/operations/sensitivity.cpp`.)
- Observed retention/retention_ema missing: `SetObservedRetentionSeconds` is never called, `retention_ema` is never updated, and active-memory mean age is not computed per step. Stability update relies only on preloaded history. (`src/operations/stability.cpp`, `src/signal_processor.cpp`, `include/cortext/processor/operation_context.hpp`.)
- Memory content spec says concatenated blobs; storage uses last blob only for memory-level content. (`src/operations/memory_storage.cpp`.)
- Accumulator metadata for memory storage missing `mem_elapsed` and normalized `drift_acc`; `drift_mag` is not stored in memories. (`src/operations/memory_storage.cpp`.)
- Serial-position effects computed but **never applied** to any scoring/strength update; `serial_position_multiplier` is output-only. (`src/operations/serial_position*.cpp`.)
- `attention_width`, `weight_relevance`, `mismatch_weight`, `coverage_gain_floor`, and sensitivity control weights (`weight_surprise`, `weight_valence`, `weight_arousal`, `emotion_gain`, `score_gain`) are computed but unused in scoring/gating. (Multiple files.)
- Sensitivity feedback uses novelty of the **current signal** for all used memories and redundancy from global kNN, instead of per-memory novelty vs recent context and per-memory redundancy vs context. (`src/operations/sensitivity_feedback.cpp`.)
- Memory-usage detection treats cached retrievals as “retrieved” for subsequent signals; `used` is based on similarity threshold, not interrupt injection, so `retrieved_count`/`used_count` diverge from spec. (`src/operations/detect_memory_usage.cpp`, `src/operations/memory_strength.cpp`.)
- Consolidation scheduling ignores `is_accumulating_memory` and relaxes idle requirement when interval trigger fires (spec requires idle). (`src/operations/consolidation.cpp`.)
- Consolidation clustering uses greedy single-linkage instead of DBSCAN/k-means (spec calls out density-based or k-means). (`src/operations/consolidation_cluster.cpp`.)
- Entity frequency threshold is never applied to extraction results. (`include/cortext/core/knobs.hpp`, `src/operations/process_extraction_results.cpp`.)
- (Resolved 2025-12-19) Extraction schema now uses label-only payloads and computes salience from embeddings; docs updated to match. (`docs/paper/_manuscript/index.md`, `src/operations/process_extraction_results.cpp`, `src/extractor/*`.)
- `write_rate_window_` records **signal** timestamps, not write events, so rate_target updates use signal rate rather than memory-write rate (spec says write_rate is memory writes). (`src/signal_processor.cpp`, `src/operations/sensitivity.cpp`.)
- `write_rate_window_` capacity uses `lerp(10,60,T)` count, not `w_rate_seconds(T)=lerp(60,300,T)` seconds. (`src/signal_processor.cpp`, `include/cortext/core/knobs.hpp`.)
- Reconsolidation ignores `τ_labile` and does not decay lability over time; `current_lability` is fixed to susceptibility. (`src/operations/reconsolidation.cpp`.)
- `theta_target` is not tracked; persisted as `theta_dynamic` by default. (`src/signal_processor.cpp`.)
- Persisted focus/sensitivity/stability state is overwritten: `LoadState` does not set `*_priors_initialized`, so `InitializeFocusPriors` / `InitializeSensitivityPriors` / `InitializeStabilityPriors` re-seed and clobber loaded values. (Multiple files.)
- Cold-start threshold state does not follow Appendix A defaults: `theta_dynamic` is initialized to a fixed 0.2 (not `θ_prior(F,S,T)`), and `hysteresis` starts at a fixed 0.05 rather than `base_band(T)` until `UpdateThreshold` runs. (`include/cortext/processor/processor_context.hpp`, `src/operations/threshold.cpp`.)
- Cold-start `last_rate_timestamp` should initialize to `now_ms()` per Appendix A; code defaults to `0`, affecting early Δt/rate estimates. (`include/cortext/processor/processor_context.hpp`, `docs/paper/_manuscript/index.md`.)
- Drift spike uses **post‑EWMA** `eta_acc` as baseline because `ComputeCoherence` updates `eta_acc` before `DetectBoundary`. Spec uses `eta_prev` (pre‑update) for spike calculation. (`src/cortext.cpp`, `src/operations/coherence.cpp`, `src/operations/boundary.cpp`.)
- Accumulator `coherence_prev` initializes to `1.0` in code; spec initializes to `0` and resets to `0` at boundaries. (`include/cortext/processor/accumulator_state.hpp`, `docs/paper/_manuscript/index.md`.)
- `adjacent_window(F)` is specified in streaming pacing but never used in implementation. (`docs/paper/_manuscript/index.md`, `include/cortext/core/knobs.hpp`, `src/operations/streaming_pacing.cpp`.)
- Influence tracking uses recent input embeddings as generation trajectory; spec applies drift on **generation embeddings when available** and otherwise should skip/neutralize that term. (`src/operations/influence.cpp`.)

### Persistence / schema gaps
- State fields required by manuscript are not persisted or loaded: `coverage_gain_floor`, `mismatch_weight`, `weight_surprise`, `weight_valence`, `weight_arousal`, `emotion_gain`, `score_gain`, `drift_weight`, `retention_ema`, `last_mood_ts`, `last_retrieval_ts`. Save uses constants; load reads only `weight_novelty`. (`src/signal_processor.cpp`.)
- `acc_signals_window` exists in schema but not in `AccumulatorState`, and is neither saved nor loaded. (`docs/paper/diagrams/entity-relationship.qmd`, `src/signal_processor.cpp`.)
- `signals.coherence` field persists `context.GetCoherence()` which is never set; should store structural coherence. (`src/operations/signal_metrics_persistence.cpp`.)
- `retention_ema` is persisted in schema but never updated or read for stability updates. (`src/operations/stability.cpp`, `src/signal_processor.cpp`.)
- `mean_influence` and `last_mood_ts` are never updated; `last_retrieval_ts` is updated in memory but **not persisted**; `delta_half_life_adj` is always persisted as `0.0` (ignores computed adjustments). (`src/signal_processor.cpp`, `src/operations/influence.cpp`, `src/operations/stability_feedback.cpp`.)
- Metacognitive state fields in schema (`fok_state`, `retrieval_strength`, `metacognitive_confidence`) are never persisted or updated. (`src/store/schema.cpp`, `src/operations/metacognitive.cpp`.)

## Next Checks (Planned)
- None — audit complete.

## Notes
This file will be updated as the audit continues.
