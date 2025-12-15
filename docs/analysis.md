# Exhaustive Gap Analysis: Cortext Operations vs algorithms.md

This document provides a comprehensive evaluation of every algorithm, schema requirement, and persistence mechanism specified in `docs/algorithms.md` against the actual implementation.

---

## Executive Summary

| Category | Coverage | Critical Gaps |
|----------|----------|---------------|
| Algorithms 1-13 | ~98% | Alg 7 bootstrap coefficients (P3) |
| Algorithms 14-19 | **~95%** | ~~Alg 14/16 - RESOLVED~~ |
| Algorithms 20-27 | **~98%** | ~~Alg 24 gate_threshold - RESOLVED~~, ~~Alg 26 zones - RESOLVED~~ |
| Schema/Persistence | ~95% | Graph tables complete, processor_state expanded, signal_metrics linked |
| Section 0 (Knobs) | **~98%** | ~~Section 0.4 u_raw - RESOLVED~~ |

**Phase 0 Status: COMPLETE** - All critical formula errors have been resolved.
**Phase 1 Status: COMPLETE** - Schema and persistence gaps resolved.
**Phase 2 Status: COMPLETE** - Missing algorithm features implemented (u_raw, Alg 24, Alg 26).

---

## ALGORITHMS 1-13: DETAILED GAP ANALYSIS

### Algorithm 1: Focus Priors - **COMPLETE** ✓
- File: `src/operations/focus.cpp` lines 14-42
- All 4 priors correctly computed: `weight_relevance_prior`, `coverage_gain_floor_prior`, `mismatch_weight_prior`, `attention_width_prior`

### Algorithm 2: Dynamic Focus Update - **COMPLETE** ✓
- File: `src/operations/focus.cpp` lines 44-93
- EWMA with α_F schedule, context window maintenance correct

### Algorithm 3: Sensitivity Priors - **COMPLETE** ✓
- File: `src/operations/sensitivity.cpp` lines 13-41
- All 9 priors correctly computed

### Algorithm 4: Dynamic Sensitivity Update - **COMPLETE** ✓
- File: `src/operations/sensitivity.cpp` lines 93-227
- Emotion projections, β(S) temperature, ΔThreshold_emotion, ΔThreshold_sensitivity all correct

### Algorithm 5: Stability Priors - **COMPLETE** ✓
- File: `src/operations/stability.cpp` lines 10-39
- All 6 priors correctly computed

### Algorithm 6: Dynamic Stability Update - **COMPLETE** ✓
- File: `src/operations/stability.cpp` lines 41-109
- Z-score, EWMA with α_T, half-life targeting all correct

### Algorithm 7: Metric Weight Blending - **MOSTLY COMPLETE** ⚠️
- File: `src/operations/blend.cpp`
- **GAP #1:** Bootstrap coefficients use ad-hoc heuristics (lines 55-112) instead of formal `sigmoid(a_F[i]×F + a_S[i]×S + a_T[i]×T + b_i)` structure
- **GAP #2:** N parameter not implemented (`lerp(64, 512, T)` for RLS window)
- RLS forgetting φ(T) = 0.90 + 0.09T is correct (line 296)
- Blending with confidence correct (lines 358-360)

### Algorithm 8: Adaptive Threshold Evolution - **COMPLETE** ✓
- File: `src/operations/threshold.cpp` lines 49-189
- Bayesian blend, rate control (EMA + ESS), homeostasis, all deltas integrated correctly

### Algorithm 10: Coherence - **COMPLETE** ✓
- File: `src/operations/coherence.cpp` lines 36-63
- `coherence_t = 1 - clamp(var(cos_sims), 0, 1)` correct

### Algorithm 11: Focus Spread - **COMPLETE** ✓
- File: `src/operations/focus_spread.cpp` lines 13-76
- Softmax entropy normalized by `log(k)` correct

### Algorithm 12: Trajectory Drift / Episode Boundary - **COMPLETE** ✓
- File: `src/operations/boundary.cpp` lines 37-114
- Drift threshold `lerp(0.10, 0.35, T)` correct, boundary triggering correct

### Algorithm 13: Logprob Surprise - **COMPLETE** ✓
- File: `src/operations/logprob_surprise.cpp` lines 12-50
- Both formulas correct: `logprob_surprisal = clamp(mean_nll/5, 0, 1)` and `surprise = 1 - (1-drift)*(1-logprob_surprisal)`

---

## ALGORITHMS 14-27: DETAILED GAP ANALYSIS

### Algorithm 14: Memory Strength Adjustment - **COMPLETE** ✓
- File: `src/operations/memory_strength.cpp` lines 80-102
- Correctly implements: `strength += S × EWMA(use_frequency) + F × influence_factor − λ(T)`
- Combined with Algorithm 18 influence-weighted update in single SQL statement
- No serial position multiplier incorrectly applied (previous analysis was outdated)

### Algorithm 15: Focus Feedback Adjustment - **COMPLETE** ✓
- File: `src/operations/focus_feedback.cpp`
- αF=0.10, βF=0.05, attention_width narrowing/widening correct

### Algorithm 16: Sensitivity Feedback Adjustment - **COMPLETE** ✓
- File: `src/operations/sensitivity_feedback.cpp` lines 40-141
- `ComputeRedundancy()` function (lines 40-85) performs kNN similarity lookup via sqlite-vec
- Correctly computes: `redundancy = 1 - mean_sim` where sim = 1/(1+dist) for L2 distances
- Main loop applies: `adjustment = η × (novelty_reward × contextual_gain - redundancy)`
- Previous analysis was outdated; redundancy computation is now fully implemented

### Algorithm 17: Stability Feedback Adjustment - **COMPLETE** ✓
- File: `src/operations/stability_feedback.cpp`
- ΔHalfLife_adj_t correctly passed to Algorithm 6

### Algorithm 18: Influence-Weighted Update - **COMPLETE** ✓
- File: `src/operations/memory_strength.cpp` (combined with Alg 14)
- `influence_factor = (used_count/retrieved_count) × clamp(contextual_gain, -1, +1)` correct

### Algorithm 19: Logprob + Embedding Influence Feedback - **COMPLETE** ✓
- File: `src/operations/influence.cpp`
- λ₁=0.5, λ₂=0.4, λ₃=0.3 correct
- drift_contribution, sustained_influence EWMA, all modulations correct

### Algorithm 20: Memory Reconsolidation - **COMPLETE** ✓
- File: `src/operations/reconsolidation.cpp`
- τ_labile, drift_magnitude, lability_susceptibility correct
- Embedding blending and uncertainty bump correct
- Note: ripple_decay not used yet (graph traversal future feature)

### Algorithm 21: Retrieval Competition & Interference - **COMPLETE** ✓
- File: `src/operations/competition.cpp`
- All 6 knob-derived parameters correct, RIF recovery, lateral inhibition correct

### Algorithm 22: Predictive Pre-activation - **COMPLETE** ✓
- File: `src/operations/predictive.cpp`
- trajectory_samples, prediction_conf_threshold, pre_activation_decay, surprise_sensitivity correct
- Note: prediction_horizon not applied (future depth-based feature)

### Algorithm 23: Emotional Consolidation Tags - **COMPLETE** ✓
- File: `src/operations/emotion.cpp`
- All derived parameters correct, emotional_tags table schema correct

### Algorithm 24: Working Memory Gates - **COMPLETE** ✓
- File: `src/operations/working_memory.cpp`
- `WMGateThreshold(F) = lerp(0.1, 0.4, F)` added to `knobs.hpp` and used for gate decision
- Higher Focus (F=1) results in stricter gating (threshold=0.4)
- Lower Focus (F=0) results in permissive gating (threshold=0.1)
- Note: rehearsal_rate, slot_dedication_strength not yet implemented (P3)

### Algorithm 25: Metacognitive Monitoring - **COMPLETE** ✓
- File: `src/operations/metacognitive.cpp`
- All 7 derived parameters correct, TOT/unknown detection correct

### Algorithm 26: Serial Position Effects - **COMPLETE** ✓
- Files: `src/operations/serial_position.cpp`, `serial_position_apply.cpp`
- Zone classification via `SerialDetermineZone()` using signal-level rarity
- `von_restorff_multiplier` applied for distinctive zone items
- `middle_suppression` applied for middle zone items
- Primacy/recency zones retain original boosting logic

### Algorithm 27: MNI Interrupt Gate - **COMPLETE** ✓
- File: `src/operations/interrupt_gate.cpp`
- All knob-derived thresholds, MU computation, gate logic correct

### Section 5.5: Adaptive Threshold Modulation - **COMPLETE** ✓
- File: `src/operations/precision.cpp`
- retrieval_precision, target_precision, κ = 0.10×(1-T), ΔThreshold_precision correct

---

## SCHEMA & PERSISTENCE: DETAILED GAP ANALYSIS

### memory_feedback Table - **CRITICAL GAPS**
Current columns (schema.cpp Migration 1):
- embedding_id, retrieved_count, used_count, contextual_gain, use_frequency, last_used, strength, original_embedding, lability_state, lability_ts

**MISSING:**
| Column | Type | Purpose |
|--------|------|---------|
| `episode_id` | INTEGER | Episode id at last update |
| `mean_influence` | REAL | Average impact across cycles |
| `stability` | REAL | Per-memory stability multiplier (persistent) |

### processor_state Table - **CRITICAL GAPS**
Currently persists only 12 fields. Missing 18+ fields:

**Missing Focus Priors (Alg 1):**
- coverage_gain_floor_prior
- mismatch_weight_prior
- attention_width_prior

**Missing Sensitivity Priors (Alg 3):**
- base_rate_prior, weight_novelty_prior, weight_surprise_prior
- weight_valence_prior, weight_arousal_prior, weight_emotion_prior
- emotion_gain_prior, score_gain_prior, rate_target_prior

**Missing Sensitivity Dynamic (Alg 4):**
- weight_novelty (dynamic)

**Missing Stability Priors (Alg 5):**
- hysteresis_band_prior, half_life_prior, rate_decay_prior
- periphery_half_life_prior, salience_half_life_prior, drift_weight_prior

**Missing Stability Dynamic (Alg 6):**
- rate_decay, periphery_half_life, salience_half_life

**Missing Threshold State (Alg 8):**
- m_rate, rate_ticks (partial), dt_ema, last_rate_timestamp

### signal_metrics Table - **GAP**
**MISSING:** `embedding_id INTEGER` (FK to memory if written, else NULL)
- Cannot correlate signal metrics with stored memories
- RLS weight fitting cannot measure success against actual writes

### Graph Integration Tables - **CRITICAL: NOT IMPLEMENTED**
Algorithms 30-33 require these tables which DO NOT EXIST:

**graph_nodes Table:**
| Column | Type | Purpose |
|--------|------|---------|
| id | TEXT PRIMARY KEY | Node identifier |
| type | TEXT NOT NULL | (entity, goal) |
| label | TEXT | Human-readable name |
| embedding_id | TEXT | FK to embeddings (nullable) |
| metadata | JSONB | Additional attributes |
| created_at | INTEGER | Unix time |

**graph_edges Table:**
| Column | Type | Purpose |
|--------|------|---------|
| source_id | TEXT | Origin node |
| target_id | TEXT | Destination node |
| edge_type | TEXT | (co_occurs_with, implies, derived_from) |
| weight | REAL | Relationship strength |
| decay_rate | REAL | Knob-derived via T |
| last_reinforced | INTEGER | Unix time |

### Section 0.4: Uncertainty Computation - **COMPLETE** ✓
- File: `src/operations/uncertainty.cpp`
- Primary u_raw estimator fully implemented with weighted blend of:
  - `var_recent_scores` (weighted by F)
  - `focus_spread_entropy` (weighted by S)
  - `coherence_complement` (weighted by 1-T)
  - `novelty_surprise_spikes` (fusion of novelty + surprise, weighted by S)
- Fallback `u_raw = 1 - maturity(t)` used when structural metrics unavailable

---

## TODO: IMPLEMENTATION ROADMAP

### Phase 0: Critical Formula Errors (P0) - **COMPLETE** ✓

All Phase 0 items have been resolved:

- [x] **Alg 14 serial position** - No issue found; previous analysis was incorrect
  - File: `src/operations/memory_strength.cpp` lines 80-102
  - Formula correctly implements `strength += S × EWMA(use_frequency) + F × influence_factor − λ(T)`
  - No `sp_mult` variable exists in this file

- [x] **Alg 16 redundancy computation** - Fully implemented
  - File: `src/operations/sensitivity_feedback.cpp` lines 40-141
  - `ComputeRedundancy()` performs kNN similarity lookup via sqlite-vec
  - Correctly integrates redundancy penalty into weight_novelty adjustment

---

### Phase 1: Schema & Persistence (P1) - **COMPLETE** ✓

- [x] **Add graph_nodes and graph_edges tables**
  - File: `src/operations/graph_schema.cpp` (Migration 10 - already existed)
  - [x] `graph_nodes` table with columns: node_id, type, label, embedding_id, metadata, created_at
  - [x] `graph_edges` table with columns: source_id, target_id, edge_type, weight, decay_rate, last_reinforced
  - [x] Indexes on source_id, target_id, edge_type

- [x] **Extend memory_feedback table**
  - File: `src/store/schema.cpp` (Migration 7)
  - [x] Add `episode_id INTEGER` column
  - [x] Add `mean_influence REAL` column
  - [x] Add `stability REAL` column

- [x] **Expand processor_state persistence**
  - File: `src/store/schema.cpp` (Migration 8)
  - [x] Add Focus priors: coverage_gain_floor_prior, mismatch_weight_prior, attention_width_prior
  - [x] Add Sensitivity priors: base_rate_prior, weight_novelty_prior, weight_surprise_prior, weight_valence_prior, weight_arousal_prior, weight_emotion_prior, emotion_gain_prior, score_gain_prior, rate_target_prior
  - [x] Add Sensitivity dynamic: weight_novelty
  - [x] Add Stability priors: hysteresis_band_prior, half_life_prior, rate_decay_prior, periphery_half_life_prior, salience_half_life_prior, drift_weight_prior
  - [x] Add Stability dynamic: rate_decay, periphery_half_life, salience_half_life
  - [x] Add Threshold state: m_rate, rate_ticks, dt_ema, last_rate_timestamp
  - File: `src/signal_processor.cpp`
  - [x] Update LoadProcessorState for 21 new columns
  - [x] Update PersistProcessorState for 21 new columns

- [x] **Add embedding_id FK to signal_metrics**
  - File: `src/operations/signal_metrics_persistence.cpp`
  - [x] Column already existed in schema
  - [x] Updated INSERT to include embedding_id from OperationContext

---

### Phase 2: Missing Algorithm Features (P2) - **COMPLETE** ✓

- [x] **Implement u_raw primary estimator**
  - File: `src/operations/uncertainty.cpp`
  - [x] Implement weighted blend of variance component
  - [x] Implement focus_spread_entropy component
  - [x] Implement coherence_complement component
  - [x] Implement novelty_surprise_spikes component (fusion of novelty + surprise)
  - [x] Primary estimator now used when metrics available, fallback when not
  - [x] Tests exist for uncertainty computation

- [x] **Implement Alg 26 zone modulations**
  - File: `src/operations/serial_position_apply.cpp`
  - [x] Apply `von_restorff_multiplier` for distinctive items
  - [x] Apply `middle_suppression` for middle-zone items
  - [x] Zone classification via `SerialDetermineZone()` using signal-level rarity
  - [x] Tests added for zone classification and modulation effects

- [x] **Connect gate_threshold in Alg 24**
  - File: `src/operations/working_memory.cpp`
  - [x] Added `WMGateThreshold(F) = lerp(0.1, 0.4, F)` to `knobs.hpp`
  - [x] Replaced `T_dynamic` with `gate_threshold` in gate decision
  - [x] Tests added for Focus-derived gate threshold behavior

---

### Phase 3: Enhancements (P3) - **COMPLETE** ✓

- [x] **Implement Alg 7 formal bootstrap coefficients**
  - File: `src/operations/blend.cpp`
  - [x] Define coefficient matrices (c_F, c_S, c_T, d_i) derived from original heuristics
  - [x] Replace ad-hoc heuristics with `sigmoid(c_F[i]×F + c_S[i]×S + c_T[i]×T + d_i)`
  - [x] Implement N parameter: `RLSWindowN(T) = lerp(64, 512, T)` in `knobs.hpp`
  - [x] Add coefficient-space RLS storage (12 metrics × 4 coefficients)
  - [x] Add persistence for RLS coefficients and covariance matrix (Migration 9)
  - [x] All existing tests pass

- [x] **Persist observation windows**
  - File: `src/signal_processor.cpp`, `src/store/schema.cpp`
  - [x] `recent_context` window already persisted (Migration 5)
  - [x] `recent_scores` window already persisted (Migration 5)
  - [x] Added `observed_retention_history` table (Migration 9)
  - [x] Added `LoadObservedRetentionHistory()` and `PersistObservedRetentionHistory()`

- [x] **Persist emotion state across restarts**
  - File: `src/signal_processor.cpp`, `src/operations/sensitivity.cpp`
  - [x] Added `emotion_intensity_ewma`, `valence_ewma`, `arousal_ewma` to ProcessorContext
  - [x] Added columns to processor_state table (Migration 9)
  - [x] Applied EWMA smoothing with α = lerp(0.05, 0.30, S) in sensitivity.cpp
  - [x] Updated Load/PersistProcessorState for emotion state

---

## FILES TO MODIFY

| File | Phase | Changes Required |
|------|-------|------------------|
| ~~`src/operations/memory_strength.cpp`~~ | ~~P0~~ | ✓ Complete - No changes needed |
| ~~`src/operations/sensitivity_feedback.cpp`~~ | ~~P0~~ | ✓ Complete - Redundancy implemented |
| ~~`src/store/schema.cpp`~~ | ~~P1~~ | ✓ Complete - Migration 7 (memory_feedback), Migration 8 (processor_state) |
| ~~`src/signal_processor.cpp`~~ | ~~P1~~ | ✓ Complete - Load/Persist for 21 new processor_state columns |
| ~~`src/operations/signal_metrics_persistence.cpp`~~ | ~~P1~~ | ✓ Complete - embedding_id included in INSERT |
| ~~`src/operations/uncertainty.cpp`~~ | ~~P2~~ | ✓ Complete - u_raw primary estimator with novelty_surprise_spikes |
| ~~`src/operations/serial_position_apply.cpp`~~ | ~~P2~~ | ✓ Complete - Zone modulations (von_restorff, middle_suppression) |
| ~~`src/operations/working_memory.cpp`~~ | ~~P2~~ | ✓ Complete - gate_threshold = lerp(0.1, 0.4, F) |
| ~~`include/cortext/core/knobs.hpp`~~ | ~~P2/P3~~ | ✓ Complete - WMGateThreshold, RLSWindowN added |
| ~~`src/operations/blend.cpp`~~ | ~~P3~~ | ✓ Complete - Formal bootstrap coefficients, coefficient-space RLS |
| ~~`src/signal_processor.cpp`~~ | ~~P3~~ | ✓ Complete - Persist observation windows, emotion state, RLS coefficients |
| ~~`src/operations/sensitivity.cpp`~~ | ~~P3~~ | ✓ Complete - EWMA smoothing for emotion state |
| ~~`include/cortext/processor/processor_context.hpp`~~ | ~~P3~~ | ✓ Complete - RLS coefficients, emotion EWMA fields |

---

## VERIFICATION CHECKLIST

After all phases complete, verify:

- [ ] All 27 algorithms compute formulas per spec
- [ ] All knob-derived parameters match Section 0.2
- [ ] All EWMA schedules match Section 0.3
- [ ] Uncertainty integrates 4 structural signals (Section 0.4)
- [ ] memory_feedback has 8+ columns per spec
- [ ] processor_state has 30+ columns per spec
- [ ] signal_metrics has embedding_id FK
- [ ] graph_nodes and graph_edges tables exist
- [ ] Boot order persists/restores all state correctly
- [ ] All existing tests pass
- [ ] New tests cover gap fixes
