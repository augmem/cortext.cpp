# Plan: Operations Conformance Alignment (Phase 0 Complete)

## Goal
Systematically align `src/operations` (plus persistence/tests) with:
- `docs/paper/_manuscript/index.md`
- `docs/paper/diagrams/entity-relationship.qmd`

## Phase 0: Inventory + Findings (Complete)

### Traceability Snapshot (doc → ops → persistence/tests)
- **Sections 3–5 (metrics + scoring):** `coherence.cpp`, `focus_spread.cpp`, `drift_accumulation.cpp`, `embedding_prediction_error.cpp`, `metrics.cpp`, `uncertainty.cpp` → `signals` metrics columns → `tests/operations_*metrics*.test.cpp`, `tests/formula_validation.test.cpp`
- **Sections 4–6 (adaptation, thresholds, accumulation):** `focus.cpp`, `sensitivity.cpp`, `stability.cpp`, `threshold.cpp`, `accumulator.cpp`, `boundary.cpp`, `spike_bypass.cpp`, `write_gate.cpp`, `streaming_pacing.cpp` → `state`, `accumulators`, `memories`, `signals` → `tests/operations_*threshold*.test.cpp`, `tests/operations_accumulator.test.cpp`, `tests/operations_write_gate.test.cpp`
- **Section 7 (reinforcement/decay/competition):** `memory_strength.cpp`, `competition.cpp`, `influence.cpp` → `memories`, `associations` → `tests/operations_memory_strength.test.cpp`, `tests/operations_competition.test.cpp`
- **Section 8 (advanced cognition):** `working_memory.cpp`, `metacognitive.cpp`, `reconsolidation.cpp`, `predictive.cpp`, `serial_position*.cpp`, `emotion*.cpp` → `memories`, `signals`, `state` → `tests/operations_working_memory.test.cpp`, `tests/operations_serial_position*.test.cpp`, `tests/operations_emotion*.test.cpp`
- **Section 9 (consolidation + graph):** `consolidation*.cpp`, `process_extraction_results.cpp`, `graph_build.cpp`, `graph_retrieval.cpp` → `memories`, `associations` → `tests/operations_consolidation*.test.cpp`, `tests/operations_graph_*.test.cpp`
- **Section 10 (interrupt gate + streaming):** `interrupt_gate.cpp`, `streaming_pacing.cpp`, `write_gate.cpp` → `state`, `accumulators` → `tests/operations_interrupt_gate.test.cpp`, `tests/operations_streaming_pacing.test.cpp`

### Gaps Found (Prioritized)
1. **Episodes table not populated.** `episodes` exists in schema but no inserts/updates; `memories.episode_id` is set to `episode_start_ts` without a corresponding `episodes` row.
2. **Signals embed/blob persistence incomplete.** `signals.embedding_id` uses memory embedding placeholder, and `signals.blob_id` is never set (accumulator notes “later” but never stores).
3. **Recent context load query invalid.** `LoadRecentContext` orders by `seq_order`, which does not exist in the `recent_context` view.
4. **Observed retention history table missing.** Loader reads `observed_retention_history` (not in schema); spec expects retention history buffer derived from signals/memories.
5. **Metric definitions drift from manuscript.**
   - **Novelty/Rarity:** computed as `1 - relevance` vs. Appendix B (max-cos novelty and mean-cos μ_sim/rarity).
   - **Relevance fallback:** empty context yields `0`, but spec expects `map01(cos)=0.5` baseline.
   - **Drift:** uses split of recent context with `KNeighbors`; spec requires lagged centroid drift with `k_ctx(T)`.
   - **Utility:** uses delta-to-mean score; spec requires ΔSSE based on prediction error.
6. **Focus spread uses recent_context instead of memory_stream kNN.** Spec requires kNN over memory_stream; code uses recent signals.
7. **Threshold update units + fallbacks.** `delta_t` uses milliseconds but formula assumes seconds; `observed_p90` fallback should be `theta_prior` when recent_scores empty (not 0).
8. **Pipeline order diverges from Appendix D.** `UpdateFocus/UpdateSensitivity` run before structural metrics/uncertainty, but they depend on `u_t`.
9. **Graph edges diverge from ER diagram.** Edge type `co_occurs_with` (vs. `co_occurs`) and weights stored as raw cosine/drift, not clamped to `[0,1]`. (Resolved in Phase 6)
10. **Emotion cascade threshold hardcoded.** `EmotionCascade` uses `emotional_intensity >= 0.5` instead of `ThetaIntensity(S)`.
11. **State fields exist but are unused/persisted.** `rho_hat_prev` and `write_rate_timestamps` are in schema/spec but unused in logic.

---

## Phase 1: Schema + Persistence Conformance (Complete)
- [x] Implement **episode lifecycle**: insert `episodes` rows on new episode start, update `end_ts`, `boundary_type`, and `centroid`; use real `episode_id` FK in `memories` and `accumulators`.
- [x] Fix `LoadRecentContext` query to order by `timestamp` (and align view columns).
- [x] Remove or replace `LoadObservedRetentionHistory` by deriving `retention_history` from `memories.last_used/last_access` or `signals`.
- [x] Persist **per-signal embeddings**: insert one row per signal into `embeddings`, set `signals.embedding_id` accordingly.
- [x] Persist **per-signal blobs**: store payload per signal in accumulator/memory storage, fill `signals.blob_id`, remove placeholder logic.
- [x] Persist/restore `write_rate_timestamps` (or explicitly remove from spec + schema if deprecated).
- [x] Persist/restore `rho_hat_prev` (bias-corrected rate state) per Appendix A.
- [x] Update tests: `tests/store.test.cpp`, `tests/migration_core.test.cpp`, `tests/state_persistence.test.cpp`, `tests/operations_memory_storage.test.cpp`, `tests/operations_accumulator.test.cpp`.
- [x] Build and run tests for this phase (required for completion).

## Phase 2: Core Signal Metrics + Scoring (Sections 3–5) (Complete)
- [x] Implement Appendix B novelty/μ_sim/rarity and relevance fallbacks for empty context.
- [x] Update drift to `k_ctx(T)` lagged centroid drift; ensure `drift_mag_t ∈ [0,2]` and scaling `(drift_mag_t/2)×(1−T)`.
- [x] Rework focus spread to query **memory_stream** kNN (not recent_context). Add/maintain memory_stream buffer (or DB-backed kNN) with Appendix C fallbacks.
- [x] Replace utility’s delta-score proxy with **ΔSSE** from prediction error (per Section 3.1.4).
- [x] Align uncertainty’s novelty component and focus-spread entropy to the updated definitions.
- [x] Update tests: `tests/formula_validation.test.cpp`, `tests/operations_metrics.test.cpp`, `tests/operations_focus_spread.test.cpp`, `tests/operations_uncertainty.test.cpp`, `tests/operations_embedding_prediction_error.test.cpp`.
- [x] Build and run tests for this phase (required for completion).

## Phase 3: Thresholding + Homeostatic Control (Section 6)
- [x] Convert `delta_t` in threshold/rate control to **seconds** (units per manuscript).
- [x] Use `theta_prior` as `observed_p90` fallback when `recent_scores` is empty.
- [x] Wire `rho_hat_prev` into rate estimation (persist/restore in state).
- [x] Validate ESS/reliability and homeostatic deltas vs Section 6 formulas and Appendix A defaults.
- [x] Update tests: `tests/operations_threshold.test.cpp`, `tests/operations_sensitivity_update.test.cpp`.
- [x] Build and run tests for this phase (required for completion).

## Phase 4: Reinforcement, Decay, Competition (Section 7)
- [x] Validate memory strength/decay/redundancy formulas vs Section 7 definitions.
- [x] Ensure connectivity/influence/sustained_influence updates align with graph usage and persistence.
- [x] Update tests: `tests/operations_memory_strength.test.cpp`, `tests/operations_influence_feedback.test.cpp`, `tests/operations_stability_feedback.test.cpp`, `tests/operations_adherence_fixes.test.cpp`, `tests/implicit_feedback.test.cpp`.
- [x] Build and run tests for this phase (required for completion).

## Phase 5: Advanced Cognitive Processes (Section 8)
- [x] Align working memory gating and metadata persistence with Section 8.1/8.2.
- [x] Update emotional cascade threshold to `ThetaIntensity(S)` (and arousal if available).
- [x] Verify reconsolidation, predictive pre-activation, serial position, and metacognition against manuscript formulas.
- [x] Update tests: `tests/operations_working_memory.test.cpp`, `tests/operations_serial_position*.test.cpp`, `tests/operations_emotion*.test.cpp`, `tests/operations_metacognitive.test.cpp`, `tests/operations_predictive.test.cpp`.
- [x] Build and run tests for this phase (required for completion).

## Phase 6: Consolidation + Graph Integration (Section 9)
- [x] Align association edge types to ER (`co_occurs`, `has_label`, etc.).
- [x] Normalize association weights to `[0,1]` (map01(cos), clamp) per ER diagram.
- [x] Verify consolidation scoring, clustering, and extraction sequencing vs manuscript.
- [x] Update tests: `tests/operations_consolidation*.test.cpp`, `tests/operations_graph_build.test.cpp`, `tests/operations_graph_retrieval.test.cpp`, `tests/operations_reconsolidation.test.cpp`, `tests/operations_emotion_cascade.test.cpp`, `tests/integration_consolidation.test.cpp`.
- [x] Build and run tests for this phase (required for completion).

## Phase 7: Streaming Integration + Interrupt Gate (Section 10)
- [x] Verify interrupt gate inputs/outputs use updated memory_stream/recent_memory_centroids.
- [x] Align write gate order with Appendix D (post-threshold, post-accumulator).
- [x] Update tests: `tests/operations_interrupt_gate.test.cpp`, `tests/operations_streaming_pacing.test.cpp`, `tests/operations_write_gate.test.cpp`, `tests/integration_consolidation.test.cpp`.
- [x] Build and run tests for this phase (required for completion).

## Phase 8: End-to-End Validation
- [x] Reorder pipeline in `src/cortext.cpp` to match Appendix D (structural metrics + uncertainty before focus/sensitivity updates).
- [x] Update operation ordering tests: `tests/operation_set.test.cpp`, `tests/signal_processor.test.cpp`.
- [x] Extend integration/regression tests: `tests/integration_consolidation.test.cpp`, `tests/regression_behavior.test.cpp`.
- [x] Run full test suite and update fixtures/goldens as needed.
- [x] Build and run tests for this phase (required for completion).

## Phase 9: Audit Findings Backlog (Resolved)

### TODO (Phase 9 formula deviations)
- [x] Align alpha schedules (AlphaT/AlphaF/AlphaS) to manuscript constants; remove uncertainty floors.
- [x] Make `rate_target` a constant `base_rate(S)` (no scaling, no EWMA update).
- [x] Fix boundary coherence drop (/2 + clamp) and add drift spike cold-start guard.
- [x] Normalize `drift_acc` by `/2` and set structural coherence fallback to `0.5`.
- [x] Apply write refractory exponential for all `Δt`; spike bypass must force write.
- [x] Fix uncertainty blending (weighted normalize of `[S, 1−T]`), uniform weight fallback, and always use structural coherence.
- [x] Correct interrupt-gate coverage gain (`1−redundancy`) and duplicate suppression threshold direction.
- [x] Implement mood integration decay with `Δt_mood`, update `last_mood_ts`, and center `e_t`.
- [x] Fix consolidation interval unit mismatches and rate-trigger units.

### Formula deviations
- [x] Align `AlphaT/AlphaF/AlphaS` schedules to spec (remove max-floor, fix `α_min_T`/`α_span_T` constants). (`include/cortext/core/knobs.hpp`)
- [x] Remove `rate_target` EWMA updates; make it a constant `base_rate(S)` (no `0.5+1.5S` scaling). (`src/operations/sensitivity.cpp`)
- [x] Fix boundary coherence drop formula (missing `/2` + clamp) and add drift spike cold-start guard. (`src/operations/boundary.cpp`)
- [x] Normalize `drift_acc` by `/2` as specified. (`include/cortext/processor/accumulator_state.hpp`)
- [x] Set structural coherence fallback to `0.5` when context < 2. (`src/operations/coherence.cpp`)
- [x] Apply write refractory exponential for all `dt`; spike bypass must force write (no threshold gating). (`src/operations/write_gate.cpp`)
- [x] Fix uncertainty blend/normalization: weighted `normalize([S,1−T])`, uniform fallback when weights ~0, and always use structural coherence (with 0.5 fallback). (`src/operations/uncertainty.cpp`)
- [x] Fix interrupt gate coverage gain (`1−redundancy`) and duplicate suppression threshold direction (0.96→0.88). (`src/operations/interrupt_gate.cpp`)
- [x] Apply mood integration: time decay via `Δt_mood`, update `last_mood_ts`, center `e_t` as `p_c − 1/6`. (`src/operations/sensitivity.cpp`, `include/cortext/core/knobs.hpp`)
- [x] Consolidation interval math: compare seconds vs ms consistently, and align rate-trigger units with `rate_target`. (`src/operations/consolidation.cpp`, `include/cortext/core/knobs.hpp`)

### Missing / divergent behavior
- [x] Implement accumulator coherence window `acc_signals_window` (raw mean cosine over window vs `mu_acc`). (`src/operations/coherence.cpp`, `include/cortext/processor/accumulator_state.hpp`)
- [x] `reset_accumulator()` must clear `eta_acc` and `coherence_prev`; apply immediate reset on `should_flush` before interrupt gate. (`include/cortext/processor/accumulator_state.hpp`, `src/operations/accumulator_reset.cpp`, `src/cortext.cpp`)
- [x] Prevent double `ComputeCoherence` updates per signal; use pre-append window for ETA. (`src/cortext.cpp`)
- [x] Boundary gap detection must use pre-update `last_signal_ts`. (`src/operations/boundary.cpp`, `src/operations/accumulator.cpp`)
- [x] Store `recent_memory_centroids` as `e_rep` with `win_mem_ctx(T)` length; stop capping `memory_stream` to `NCtx(T)`. (`src/operations/write_gate.cpp`, `include/cortext/core/knobs.hpp`)
- [x] Persist structural coherence into `context.SetCoherence` so interrupt gate penalties use it. (`src/operations/coherence.cpp`, `src/operations/signal_metrics_persistence.cpp`)
- [x] Use raw `F` (not `F_eff`) for Table‑1 metrics (relevance/mismatch/rarity/utility/coverage/salience/contradiction). (`src/operations/metrics.cpp`)
- [x] RLS blending: use weight-space `blender_state` (avoid stale `rls_coefficients`) and fix covariance update. (`src/operations/blend.cpp`)
- [x] Persist metrics for **all** signals in a memory, not just the latest source+timestamp. (`src/operations/signal_metrics_persistence.cpp`)
- [x] Graph retrieval: skip when `memory_stream` empty; use `graph_depth(T)` (not `GraphDepth(F)`); enforce `min_edge_weight(F)`; add missing edge types `implies`/`similar_to`; avoid weight saturation on reinforcement. (`src/operations/graph_retrieval.cpp`, `src/operations/graph_build.cpp`, `include/cortext/core/knobs.hpp`)
- [x] Interrupt gate should use `candidate_star` values (not max across candidates) for relevance/novelty/overlap. (`src/operations/interrupt_gate.cpp`)
- [x] Working memory benefit must use `relevance_to_task(μ_acc, task_context)`. (`src/operations/working_memory.cpp`)
- [x] Metacognitive FOK/retrieval strength must be computed and updated each step. (`src/operations/metacognitive.cpp`)
- [x] Emotional consolidation: apply flashbulb threshold, use stored fields (`half_life_bonus`, `detail_suppression`, `gist_components`), and perform consolidation-time tagging (not usage-gated). (`src/operations/emotion*.cpp`, `src/operations/memory_strength.cpp`)
- [x] Emotion probability fallback must be uniform when all cos ≤ 0. (`src/operations/sensitivity.cpp`)
- [x] Update observed retention/retention_ema each step; compute active-memory mean age. (`src/operations/stability.cpp`, `src/signal_processor.cpp`)
- [x] Memory content should concatenate blobs (not last blob only). (`src/operations/memory_storage.cpp`)
- [x] Store accumulator metadata: `mem_elapsed`, normalized `drift_acc`, `drift_mag`. (`src/operations/memory_storage.cpp`)
- [x] Apply serial-position multiplier to scoring/strength updates (not output-only). (`src/operations/serial_position*.cpp`)
- [x] Use computed control weights (`attention_width`, `weight_relevance`, `mismatch_weight`, `coverage_gain_floor`, `weight_surprise`, `weight_valence`, `weight_arousal`, `emotion_gain`, `score_gain`). (Multiple files)
- [x] Sensitivity feedback must use per-memory novelty vs context and per-memory redundancy. (`src/operations/sensitivity_feedback.cpp`)
- [x] Memory-usage detection should reflect interrupt injection vs retrieval; align `retrieved_count`/`used_count`. (`src/operations/detect_memory_usage.cpp`, `src/operations/memory_strength.cpp`)
- [x] Consolidation scheduling must respect `is_accumulating_memory` and idle requirement; clustering should be density‑based or k‑means (spec). (`src/operations/consolidation*.cpp`)
- [x] Apply label frequency threshold to extraction results. (`include/cortext/core/knobs.hpp`, `src/operations/process_extraction_results.cpp`)
- [x] `write_rate_window_` must track **write** events and use seconds capacity `w_rate_seconds(T)` (not count). (`src/signal_processor.cpp`, `include/cortext/processor/processor_context.hpp`)
- [x] Reconsolidation must incorporate `τ_labile` decay; `current_lability` cannot be fixed. (`src/operations/reconsolidation.cpp`)
- [x] Track and persist `theta_target` (not default to `theta_dynamic`). (`src/signal_processor.cpp`)
- [x] Prevent loaded priors from being clobbered: set `*_priors_initialized` on load. (Multiple files)
- [x] Cold-start defaults: `theta_dynamic=θ_prior(F,S,T)`, `hysteresis=base_band(T)`, `last_rate_timestamp=now_ms()`. (`include/cortext/processor/processor_context.hpp`, `src/operations/threshold.cpp`)
- [x] Use pre‑update `eta_prev` baseline for drift spikes. (`src/operations/coherence.cpp`, `src/operations/boundary.cpp`)
- [x] Initialize/reset `coherence_prev` to `0`. (`include/cortext/processor/accumulator_state.hpp`)
- [x] Implement `adjacent_window(F)` in streaming pacing. (`src/operations/streaming_pacing.cpp`)
- [x] Influence tracking should use generation embeddings when available; otherwise neutralize. (`src/operations/influence.cpp`)
- [x] Fix baseline test failures after encoder requirement change: migration/state tests must set a test encoder (or adjust constructor expectations) when EmbeddingGemma is disabled. (`tests/migration_core.test.cpp`, `tests/state_persistence.test.cpp`)
- [x] Update uncertainty fallback test to reflect structural-coherence-first path (weights normalize to uniform, no maturity fallback when coherence present). (`tests/operations_uncertainty.test.cpp`)

### Persistence / schema gaps
- [x] Persist missing state fields: `coverage_gain_floor`, `mismatch_weight`, `weight_surprise`, `weight_valence`, `weight_arousal`, `emotion_gain`, `score_gain`, `drift_weight`, `retention_ema`, `last_mood_ts`, `last_retrieval_ts`. (`src/signal_processor.cpp`)
- [x] Add `acc_signals_window` to `AccumulatorState` and persist/load it. (`src/signal_processor.cpp`, `include/cortext/processor/accumulator_state.hpp`)
- [x] Persist structural coherence to `signals.coherence` (not `context.GetCoherence()` default). (`src/operations/signal_metrics_persistence.cpp`)
- [x] Update and persist `mean_influence` (last_mood_ts/last_retrieval_ts/delta_half_life_adj handled). (`src/operations/influence.cpp`)
- [x] Persist and update metacognitive fields (`fok_state`, `retrieval_strength`, `metacognitive_confidence`). (`src/store/schema.cpp`, `src/operations/metacognitive.cpp`)

---

## Notes
- Use the traceability snapshot to decide whether to **update code** or **revise tests** when mismatches appear.
- Any new fields added for conformance must be persisted and covered by tests.
