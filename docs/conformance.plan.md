# Cortext Algorithm Conformance Plan

This document outlines the phased remediation plan for gaps identified between the codebase implementation and the algorithmic specification in `docs/research/algorithms.md`.

## Summary

| Phase | Focus | Gaps | Effort |
|-------|-------|------|--------|
| 1 | Critical Memory Decay | 1 | High |
| 2 | Interrupt Gate Corrections | 4 | Medium |
| 3 | Core Algorithm Fixes | 3 | Medium |
| 4 | Metric & Threshold Fixes | 3 | Low |
| 5 | Minor Formula Additions | 2 | Low |

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

## Phase 2: Interrupt Gate Corrections

**Priority: HIGH**
**Estimated Complexity: Medium**

The interrupt gate uses incorrect novelty computation and threshold sources.

### TODO 2.1: Implement �_novelty Formula

**File:** `include/cortext/core/knobs.hpp`

**Add new function:**
```cpp
inline double TauNovelty(double F, double S, double T) {
  return Lerp(0.10, 0.35, F) * (1.0 - 0.15 * S) * (1.0 + 0.3 * T);
}
```

**Tasks:**
- [ ] Add `TauNovelty(F, S, T)` function to knobs.hpp
- [ ] Add unit test in `core_knobs.test.cpp` validating formula bounds

**Reference:** algorithms.md Section 8.1, lines 1167-1168

---

### TODO 2.2: Implement retrieval_thresh(F) Formula

**File:** `include/cortext/core/knobs.hpp`

**Add new function:**
```cpp
inline double RetrievalThreshold(double F) {
  return Lerp(0.25, 0.60, F);
}
```

**File:** `src/operations/interrupt_gate.cpp`
**Lines:** 198-200

**Current (Incorrect):**
```cpp
const double retrieval_thresh = context.GetThresholdTDynamic();
```

**Required:**
```cpp
const double retrieval_thresh = core::RetrievalThreshold(cfg.focus);
```

**Tasks:**
- [ ] Add `RetrievalThreshold(F)` function to knobs.hpp
- [ ] Update interrupt_gate.cpp to use knob-derived threshold
- [ ] Add unit test validating formula

**Reference:** algorithms.md Section 8.1, lines 1171-1172

---

### TODO 2.3: Implement Embedding Novelty Computation

**File:** `src/operations/interrupt_gate.cpp`
**Lines:** 250-270

**Current (Incorrect):**
Uses Jaccard index on ID sets: `jaccard = 1 - |A)B|/|A*B|`

**Required (Per Spec):**
```cpp
embedding_novelty = 1.0 - max(cos(candidate.embedding, ctx_window[i].embedding) for all i)
```

**Tasks:**
- [ ] Add `ComputeEmbeddingNovelty()` function that computes `1 - max(cosine similarity to context)`
- [ ] Replace `jaccard >= tau_j_eff` condition with `embedding_novelty >= tau_novelty_eff`
- [ ] Keep Jaccard as secondary duplicate check if desired
- [ ] Add unit test verifying embedding-space novelty computation

**Reference:** algorithms.md Section 8.3, lines 1224-1225

---

### TODO 2.4: Implement K Parameter for Candidate Limiting

**File:** `include/cortext/core/knobs.hpp`

**Add new function:**
```cpp
inline int InterruptCandidateCount(double F) {
  return static_cast<int>(std::round(Lerp(10.0, 6.0, F)));
}
```

**File:** `src/operations/interrupt_gate.cpp`

**Tasks:**
- [ ] Add `InterruptCandidateCount(F)` function to knobs.hpp
- [ ] Limit candidate evaluation to top K candidates by relevance
- [ ] Add unit test

**Reference:** algorithms.md Section 8.3, line 1216

---

## Phase 3: Core Algorithm Fixes

**Priority: MEDIUM**
**Estimated Complexity: Medium**

### TODO 3.1: Add map01 Transformation in Focus Update

**File:** `src/operations/focus.cpp`
**Line:** 76

**Current (Incorrect):**
```cpp
p_ctx.weight_relevance = core::Ewma(p_ctx.weight_relevance, observed_cosine, alpha_f);
```

**Required:**
```cpp
// Transform cosine [-1, 1] to [0, 1]
const double mapped = core::Clamp((observed_cosine + 1.0) / 2.0, 0.0, 1.0);
p_ctx.weight_relevance = core::Ewma(p_ctx.weight_relevance, mapped, alpha_f);
```

**Tasks:**
- [ ] Add `Map01(double cosine)` helper to algorithms.hpp: `clamp((z + 1) / 2, 0, 1)`
- [ ] Apply `Map01()` to `observed_cosine` before EWMA
- [ ] Add unit test verifying transformation

**Reference:** algorithms.md Section 2.1.2, lines 220-223

---

### TODO 3.2: Fix Prediction Error Metric Source

**File:** `src/operations/metrics.cpp`
**Lines:** 116-122

**Current (Incorrect):**
Uses score variance (`var_scores`) as proxy for prediction error.

**Required:**
Use embedding surprisal from `embedding_prediction_error.cpp`:
```cpp
const double surprisal = p_ctx.embedding_surprisal;  // Already computed
const double prediction_error = surprisal * S * (1.0 - T);
```

**Tasks:**
- [ ] Retrieve `embedding_surprisal` from processor context (already computed)
- [ ] Replace `var_scores` with `embedding_surprisal` in metric computation
- [ ] Add unit test verifying correct metric source

**Reference:** algorithms.md Section 3.2, Table 1 row "Prediction Error"

---

### TODO 3.3: Fix Alpha_F Formula (Optional)

**File:** `include/cortext/core/knobs.hpp`
**Lines:** 177-184

**Current (Deviation):**
```cpp
double term1 = kAlphaMinF * (1.0 + 0.5 * u_t);
double term2 = kAlphaMinF + F * kAlphaSpanF * u_t;
return std::max(term1, term2);
```

**Spec says:**
```cpp
return kAlphaMinF + F * kAlphaSpanF * u_t;  // Single formula, no max
```

**Tasks:**
- [ ] Evaluate if two-term max is intentional design choice or bug
- [ ] If bug: simplify to single formula per spec
- [ ] If intentional: document deviation in code comments
- [ ] Add/update unit test

**Reference:** algorithms.md Section 2.1.2, lines 227-230

---

## Phase 4: Metric & Threshold Fixes

**Priority: MEDIUM**
**Estimated Complexity: Low**

### TODO 4.1: Add Time-Scaled Total Delta Cap

**File:** `src/operations/threshold.cpp`
**Line:** 180

**Current (Incorrect):**
```cpp
delta_total = core::Clamp(delta_total, -max_delta, +max_delta);
```

**Required:**
```cpp
const double delta_t_seconds = (now_ts - p_ctx.last_threshold_ts) / 1000.0;
const double cap_total = max_delta * (delta_t_seconds / 60.0);
delta_total = core::Clamp(delta_total, -cap_total, +cap_total);
```

**Tasks:**
- [ ] Compute time delta since last threshold update
- [ ] Scale `max_delta` by `(delta_t / 60.0)` as per spec
- [ ] Add unit test verifying time-scaled capping

**Reference:** algorithms.md Section 4.3, lines 663-665

---

### TODO 4.2: Fix rate_consolidate Formula (Optional)

**File:** `src/operations/consolidation.cpp`
**Lines:** 20-24

**Current:**
Simple rate comparison: `m_rate < 0.5 * rate_target`

**Spec says:**
```
rate_consolidate = (1 / max(interval, 1)) � (0.3 + 0.7T) � (1  0.5S)
```

**Tasks:**
- [ ] Add `ConsolidationRate(T, S, interval)` to knobs.hpp
- [ ] Integrate into `CheckRateTrigger()` or document as intentional simplification

**Reference:** algorithms.md Section 7.1, lines 1015-1017

---

### TODO 4.3: Use extraction_batch_size in Pipeline

**File:** `src/operations/consolidation.cpp`
**Line:** 169

**Current:**
Uses `MaxExtractionsPerCycle(T)` with formula `lerp(20, 5, T)`

**Spec says:**
`extraction_batch_size = round(lerp(8, 32, T))`

**Tasks:**
- [ ] Verify which formula is correct for the use case
- [ ] Either update consolidation.cpp to use `ExtractionBatchSize(T)` or document deviation
- [ ] Ensure consistent usage across extraction pipeline

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
