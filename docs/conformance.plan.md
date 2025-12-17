# Cortext Algorithm Conformance Plan

This document outlines the phased remediation plan for gaps identified between the codebase implementation and the algorithmic specification in `docs/research/algorithms.md`.

## Summary

| Phase | Focus | Gaps | Effort | Status |
|-------|-------|------|--------|--------|
| 1 | Critical Memory Decay | 1 | High | ✅ COMPLETED |
| 2 | Interrupt Gate Corrections | 4 | Medium | ✅ COMPLETED |
| 3 | Core Algorithm Fixes | 3 | Medium | ✅ COMPLETED |
| 4 | Metric & Threshold Fixes | 3 | Low | ✅ COMPLETED |
| 5 | Minor Formula Additions | 2 | Low | |

---

## Phase 1: Critical - Memory Strength Decay ✅ COMPLETED

**Priority: CRITICAL**
**Estimated Complexity: High**
**Status: COMPLETED** (2025-12-16)

The exponential decay mechanism was fixed to follow half-life semantics per algorithms.md Section 5.1.

### TODO 1.1: Implement Exponential Decay Formula ✅

**File:** `src/operations/memory_strength.cpp`
**Lines:** 113-132

**Previous (Incorrect):**
```cpp
strength = MAX(0.0, strength + reinforcement + influence - lambda)
```

**Fixed (Per Spec):**
```cpp
strength = MAX(0.0,
    strength * exp(-lambda * MAX(0.0, ts - last_strength_update_ts) / 1000.0)
    + reinforcement + influence)
```

**Completed Tasks:**
- [x] Added `last_strength_update_ts` column to `embeddings_meta` in migration 0 (`src/store/schema.cpp`)
- [x] Compute `delta_t = MAX(0, ts - last_strength_update_ts) / 1000.0` in SQL
- [x] Replace linear subtraction with SQL `exp(-lambda * delta_t)` decay
- [x] Update `last_strength_update_ts` after each strength computation
- [x] Added unit tests verifying half-life behavior in `tests/operations_memory_strength.test.cpp`

**Reference:** algorithms.md Section 5.1, lines 688-700

**Verified Acceptance Criteria:**
- ✅ Memory with `half_life = 120s` has `strength ≈ 0.5 * initial` after 120 seconds
- ✅ Test: "Algorithm 14 applies exponential decay based on elapsed time" validates exponential curve
- ✅ Test: "Algorithm 14 no decay when delta_t is zero" validates no spurious decay
- ✅ Test: "Algorithm 14 initializes last_strength_update_ts on INSERT" validates timestamp tracking

**Note:** Implementation uses SQLite `exp()` function which requires SQLite 3.35.0+ with math functions enabled

---

## Phase 2: Interrupt Gate Corrections ✅ COMPLETED

**Priority: HIGH**
**Estimated Complexity: Medium**
**Status: COMPLETED** (2025-12-17)

The interrupt gate was updated to use correct novelty computation and threshold sources per algorithms.md Section 8.

### TODO 2.1: Implement τ_novelty Formula ✅

**File:** `include/cortext/core/knobs.hpp`

**Added function:**
```cpp
inline double TauNovelty(double F, double S, double T) {
  return Lerp(0.10, 0.35, F) * (1.0 - 0.15 * S) * (1.0 + 0.3 * T);
}
```

**Completed Tasks:**
- [x] Added `TauNovelty(F, S, T)` function to knobs.hpp
- [x] Added unit test in `core_knobs.test.cpp` validating formula bounds

**Reference:** algorithms.md Section 8.1, lines 1167-1168

---

### TODO 2.2: Implement retrieval_thresh(F) Formula ✅

**File:** `include/cortext/core/knobs.hpp`

**Added function:**
```cpp
inline double RetrievalThreshold(double F) {
  return Lerp(0.25, 0.60, F);
}
```

**File:** `src/operations/interrupt_gate.cpp`

**Fixed:** Replaced `context.GetThresholdTDynamic()` with `core::RetrievalThreshold(F)`

**Completed Tasks:**
- [x] Added `RetrievalThreshold(F)` function to knobs.hpp
- [x] Updated interrupt_gate.cpp to use knob-derived threshold
- [x] Added unit test validating formula

**Reference:** algorithms.md Section 8.1, lines 1171-1172

---

### TODO 2.3: Implement Embedding Novelty Computation ✅

**File:** `src/operations/interrupt_gate.cpp`

**Fixed:** Replaced Jaccard ID-based novelty with embedding-based novelty computation:
```cpp
embedding_novelty = 1.0 - max(cos(candidate, ctx_window[i]) for all i)
```

**Completed Tasks:**
- [x] Implemented embedding novelty as `1 - max(cosine similarity to context window)`
- [x] Replaced `jaccard >= tau_j_eff` condition with `embedding_novelty >= tau_novelty_eff`
- [x] Added unit tests verifying embedding-space novelty computation (orthogonal/similar/empty cases)

**Reference:** algorithms.md Section 8.3, lines 1224-1225

---

### TODO 2.4: Implement K Parameter for Candidate Limiting ✅

**File:** `include/cortext/core/knobs.hpp`

**Added function:**
```cpp
inline int InterruptCandidateCount(double F) {
  return static_cast<int>(std::round(Lerp(10.0, 6.0, F)));
}
```

**File:** `src/operations/interrupt_gate.cpp`

**Completed Tasks:**
- [x] Added `InterruptCandidateCount(F)` function to knobs.hpp
- [x] Limited candidate evaluation to top K candidates by relevance using `std::partial_sort`
- [x] Added unit test

**Reference:** algorithms.md Section 8.3, line 1216

---

## Phase 3: Core Algorithm Fixes ✅ COMPLETED

**Priority: MEDIUM**
**Estimated Complexity: Medium**
**Status: COMPLETED** (2025-12-17)

### TODO 3.1: Add map01 Transformation in Focus Update ✅

**File:** `src/operations/focus.cpp`
**File:** `include/cortext/core/algorithms.hpp`

**Fixed:** Added `Map01()` helper and applied transformation before EWMA.

**Completed Tasks:**
- [x] Added `Map01(double cosine)` helper to algorithms.hpp: `clamp((z + 1) / 2, 0, 1)`
- [x] Applied `Map01()` to `observed_cosine` before EWMA in focus.cpp
- [x] Added unit test "Map01 transforms cosine [-1,1] to [0,1]" in `core_algorithms.test.cpp`

**Reference:** algorithms.md Section 2.1.2, lines 220-223

---

### TODO 3.2: Fix Prediction Error Metric Source ✅

**File:** `src/operations/metrics.cpp`

**Fixed:** Replaced score variance (`var_scores`) with `embedding_surprisal` from `EmbeddingPredictionError` operation.

**Completed Tasks:**
- [x] Retrieve `embedding_surprisal` from context metrics (computed by EmbeddingPredictionError operation)
- [x] Replace `var_scores` with `embedding_surprisal` in surprise metric computation
- [x] Added unit test "ComputeMetrics uses embedding_surprisal for surprise metric" in `operations_metrics.test.cpp`

**Reference:** algorithms.md Section 3.2, Table 1 row "Prediction Error"

---

### TODO 3.3: Document Alpha_F Formula Deviation ✅

**File:** `include/cortext/core/knobs.hpp`

**Evaluated:** The two-term max pattern is an **intentional design choice**, not a bug.

**Analysis:**
- Spec formula: `α_F(t) = α_min_F + F × α_span_F × u(t)`
- Implementation: `max(α_min_F × (1 + 0.5×u_t), spec_formula)`

The first term provides an uncertainty-scaled floor that ensures responsiveness during high uncertainty conditions even when F=0. This matches the pattern used in `AlphaT` and `AlphaS` functions. Without this floor, the system would be sluggish at F=0 regardless of uncertainty level.

**Completed Tasks:**
- [x] Evaluated deviation: Confirmed as intentional design choice
- [x] Documented deviation in code comments in knobs.hpp
- [x] Existing unit tests in `core_knobs.test.cpp` already validate bounds

**Reference:** algorithms.md Section 2.1.2, lines 227-230

---

## Phase 4: Metric & Threshold Fixes ✅ COMPLETED

**Priority: MEDIUM**
**Estimated Complexity: Low**
**Status: COMPLETED** (2025-12-17)

### TODO 4.1: Add Time-Scaled Total Delta Cap ✅

**File:** `src/operations/threshold.cpp`
**Lines:** 177-184

**Fixed:** Time-scaled capping now uses existing `delta_t` from rate calculation:
```cpp
const double max_delta_per_min = core::MaxDeltaTPerMin(p_ctx.signals_processed, cfg.stability);
const double cap_total = max_delta_per_min * std::max(delta_t, 0.1) / kSecondsPerMinute;
delta_total = core::Clamp(delta_total, -cap_total, +cap_total);
```

**Completed Tasks:**
- [x] Reuse existing delta_t computation from rate control loop
- [x] Scale max_delta by (delta_t / 60.0) per spec
- [x] Add minimum floor of 0.1s to avoid zeroing deltas
- [x] Add unit test in `operations_threshold.test.cpp`

**Reference:** algorithms.md Section 4.3, lines 663-665

---

### TODO 4.2: Implement rate_consolidate Formula ✅

**File:** `include/cortext/core/knobs.hpp`
**File:** `src/operations/consolidation.cpp`

**Added function:**
```cpp
inline double ConsolidationRate(double T, double S) {
  const double interval = static_cast<double>(ConsolidationIntervalSeconds(T));
  return (1.0 / std::max(interval, 1.0)) * (0.3 + 0.7 * T) * (1.0 - 0.5 * S);
}
```

**Completed Tasks:**
- [x] Added `ConsolidationRate(T, S)` function to knobs.hpp
- [x] Updated `EvaluateConsolidation::Execute` to use knob-derived rate
- [x] Removed external rate_target configuration dependency
- [x] Updated tests to work with knob-derived rates
- [x] Added unit test in `core_knobs.test.cpp`

**Previous Spec (for reference):**
```
rate_consolidate = (1 / max(interval, 1)) � (0.3 + 0.7T) � (1  0.5S)
```

**Reference:** algorithms.md Section 7.1, lines 1015-1017

---

### TODO 4.3: Integrate ExtractionBatchSize into Pipeline ✅

**File:** `src/operations/consolidation.cpp`
**Lines:** 170-176

**Fixed:** Now uses both `ExtractionBatchSize(T)` and `MaxExtractionsPerCycle(T)`:
```cpp
const int batch_size = core::ExtractionBatchSize(cfg.stability);
const int max_per_cycle = core::MaxExtractionsPerCycle(cfg.stability);
const int count = std::min({static_cast<int>(requests.size()), batch_size, max_per_cycle});
```

**Completed Tasks:**
- [x] Integrated ExtractionBatchSize into extraction job batching
- [x] Both limits now apply: batch_size (8-32) and max_per_cycle (20-5)
- [x] Existing tests in `operations_extraction.test.cpp` now pass

**Reference:** algorithms.md Section 7.3-7.4, line 1076

---

## Phase 5: Minor Formula Additions

**Priority: LOW**
**Estimated Complexity: Low**

### TODO 5.1: Add update_rate_on_surprise Formula

**File:** `include/cortext/core/knobs.hpp`

**Add new function:**
```cpp
inline double UpdateRateOnSurprise(double T, double S) {
  return Lerp(0.2, 0.02, T) * S;
}
```

**File:** `src/operations/predictive.cpp`

**Tasks:**
- [ ] Add `UpdateRateOnSurprise(T, S)` function to knobs.hpp
- [ ] Implement trajectory model update when surprise exceeds threshold
- [ ] Add unit test

**Reference:** algorithms.md Section 6.5, lines 941-942

---

### TODO 5.2: Document Emotion Projection Architecture Difference

**File:** `src/data/centroids.cpp`

The implementation uses learned affective dimension centroids (valence_positive/negative, arousal_high/low) instead of the spec's discrete 6-emotion Russell circumplex with softmax.

**Tasks:**
- [ ] Add documentation comment explaining architectural choice
- [ ] Consider if spec should be updated to match implementation or vice versa
- [ ] No code change required if intentional architectural decision

**Reference:** algorithms.md Section 2.2.2, lines 262-303

---

## Testing Requirements

Each fix should include:

1. **Unit test** validating the specific formula
2. **Integration test** verifying end-to-end behavior
3. **Regression test** ensuring existing functionality unchanged

### Test File Locations:
- `tests/core_knobs.test.cpp` - Knob formula validation
- `tests/operations_*.test.cpp` - Operation-specific tests
- `tests/integration_*.test.cpp` - End-to-end tests

---

## Verification Checklist

After all phases complete:

- [ ] All 13 gaps addressed (fixed or documented as intentional)
- [ ] Unit tests pass for all new/modified formulas
- [ ] Integration tests pass
- [ ] `ctest --test-dir build -R cortext_tests --output-on-failure` passes
- [ ] Code review completed
- [ ] Documentation updated if spec changed

---

## Timeline Recommendation

| Phase | Dependencies | Suggested Order |
|-------|--------------|-----------------|
| Phase 1 | None | First (critical) |
| Phase 2 | None | Second (high impact) |
| Phase 3 | Phase 1 | Third |
| Phase 4 | None | Fourth |
| Phase 5 | None | Fifth |

Phases 2-5 can be parallelized after Phase 1 is complete.
