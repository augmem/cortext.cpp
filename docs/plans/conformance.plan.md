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
- **Section 9 (consolidation + graph):** `consolidation*.cpp`, `process_extraction_results.cpp`, `graph_build.cpp`, `graph_retrieval.cpp`, `concept_detection.cpp` → `memories`, `associations` → `tests/operations_consolidation*.test.cpp`, `tests/operations_graph_*.test.cpp`
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
9. **Graph edges diverge from ER diagram.** Edge type `co_occurs_with` (vs. `co_occurs`) and weights stored as raw cosine/drift, not clamped to `[0,1]`.
10. **Emotion cascade threshold hardcoded.** `EmotionCascade` uses `emotional_intensity >= 0.5` instead of `ThetaIntensity(S)`.
11. **State fields exist but are unused/persisted.** `rho_hat_prev` and `write_rate_timestamps` are in schema/spec but unused in logic.

---

## Phase 1: Schema + Persistence Conformance
- [ ] Implement **episode lifecycle**: insert `episodes` rows on new episode start, update `end_ts`, `boundary_type`, and `centroid`; use real `episode_id` FK in `memories` and `accumulators`.
- [ ] Fix `LoadRecentContext` query to order by `timestamp` (and align view columns).
- [ ] Remove or replace `LoadObservedRetentionHistory` by deriving `retention_history` from `memories.last_used/last_access` or `signals`.
- [ ] Persist **per-signal embeddings**: insert one row per signal into `embeddings`, set `signals.embedding_id` accordingly.
- [ ] Persist **per-signal blobs**: store payload per signal in accumulator/memory storage, fill `signals.blob_id`, remove placeholder logic.
- [ ] Persist/restore `write_rate_timestamps` (or explicitly remove from spec + schema if deprecated).
- [ ] Persist/restore `rho_hat_prev` (bias-corrected rate state) per Appendix A.
- [ ] Update tests: `tests/store.test.cpp`, `tests/migration_core.test.cpp`, `tests/state_persistence.test.cpp`, `tests/operations_memory_storage.test.cpp`, `tests/operations_accumulator.test.cpp`.

## Phase 2: Core Signal Metrics + Scoring (Sections 3–5)
- [ ] Implement Appendix B novelty/μ_sim/rarity and relevance fallbacks for empty context.
- [ ] Update drift to `k_ctx(T)` lagged centroid drift; ensure `drift_mag_t ∈ [0,2]` and scaling `(drift_mag_t/2)×(1−T)`.
- [ ] Rework focus spread to query **memory_stream** kNN (not recent_context). Add/maintain memory_stream buffer (or DB-backed kNN) with Appendix C fallbacks.
- [ ] Replace utility’s delta-score proxy with **ΔSSE** from prediction error (per Section 3.1.4).
- [ ] Align uncertainty’s novelty component and focus-spread entropy to the updated definitions.
- [ ] Update tests: `tests/formula_validation.test.cpp`, `tests/operations_metrics.test.cpp`, `tests/operations_focus_spread.test.cpp`, `tests/operations_uncertainty.test.cpp`, `tests/operations_embedding_prediction_error.test.cpp`.

## Phase 3: Thresholding + Homeostatic Control (Section 6)
- [ ] Convert `delta_t` in threshold/rate control to **seconds** (units per manuscript).
- [ ] Use `theta_prior` as `observed_p90` fallback when `recent_scores` is empty.
- [ ] Wire `rho_hat_prev` into rate estimation (persist/restore in state).
- [ ] Validate ESS/reliability and homeostatic deltas vs Section 6 formulas and Appendix A defaults.
- [ ] Update tests: `tests/operations_threshold.test.cpp`, `tests/operations_*_feedback.test.cpp`, `tests/implicit_feedback.test.cpp`.

## Phase 4: Reinforcement, Decay, Competition (Section 7)
- [ ] Validate memory strength/decay/redundancy formulas vs Section 7 definitions.
- [ ] Ensure connectivity/influence/sustained_influence updates align with graph usage and persistence.
- [ ] Update tests: `tests/operations_memory_strength.test.cpp`, `tests/operations_competition.test.cpp`, `tests/operations_influence_feedback.test.cpp`.

## Phase 5: Advanced Cognitive Processes (Section 8)
- [ ] Align working memory gating and metadata persistence with Section 8.1/8.2.
- [ ] Update emotional cascade threshold to `ThetaIntensity(S)` (and arousal if available).
- [ ] Verify reconsolidation, predictive pre-activation, serial position, and metacognition against manuscript formulas.
- [ ] Update tests: `tests/operations_working_memory.test.cpp`, `tests/operations_serial_position*.test.cpp`, `tests/operations_emotion*.test.cpp`, `tests/operations_metacognitive.test.cpp`, `tests/operations_predictive.test.cpp`.

## Phase 6: Consolidation + Graph Integration (Section 9)
- [ ] Align association edge types to ER (`co_occurs`, `has_label`, etc.).
- [ ] Normalize association weights to `[0,1]` (map01(cos), clamp) per ER diagram.
- [ ] Verify consolidation scoring, clustering, and extraction sequencing vs manuscript.
- [ ] Update tests: `tests/operations_consolidation*.test.cpp`, `tests/operations_graph_build.test.cpp`, `tests/operations_graph_retrieval.test.cpp`.

## Phase 7: Streaming Integration + Interrupt Gate (Section 10)
- [ ] Verify interrupt gate inputs/outputs use updated memory_stream/recent_memory_centroids.
- [ ] Align write gate order with Appendix D (post-threshold, post-accumulator).
- [ ] Update tests: `tests/operations_interrupt_gate.test.cpp`, `tests/operations_streaming_pacing.test.cpp`, `tests/operations_write_gate.test.cpp`.

## Phase 8: End-to-End Validation
- [ ] Reorder pipeline in `src/cortext.cpp` to match Appendix D (structural metrics + uncertainty before focus/sensitivity updates).
- [ ] Update operation ordering tests: `tests/operation_set.test.cpp`, `tests/signal_processor.test.cpp`.
- [ ] Extend integration/regression tests: `tests/integration_consolidation.test.cpp`, `tests/regression_behavior.test.cpp`.
- [ ] Run full test suite and update fixtures/goldens as needed.

---

## Notes
- Use the traceability snapshot to decide whether to **update code** or **revise tests** when mismatches appear.
- Any new fields added for conformance must be persisted and covered by tests.
