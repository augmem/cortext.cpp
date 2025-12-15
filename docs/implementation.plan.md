# Cortext Implementation Plan

## Overview

This document outlines a phased implementation plan to complete the Cortext architecture, addressing discrepancies identified between the paper specification (`docs/research/paper.md`) and the current codebase.

**Current Status**: ~90% complete (Phases 1, 2 & 3 done)
**Target**: Full alignment with paper specification

---

## Phase 1: Critical Formula Corrections ✅ COMPLETE

**Priority**: CRITICAL
**Estimated Complexity**: Low
**Dependencies**: None
**Status**: COMPLETE (2024-12-15)

These fixes correct mathematical discrepancies that affect system behavior.

### 1.1 Mood Magnitude Normalization ✅

**File**: `src/operations/sensitivity.cpp`
**Line**: ~284
**Issue**: Missing √6 normalization causes ΔThreshold_mood to be ~2.45× stronger than specified.

**Paper Formula (Section 4.2.4)**:
```
m_norm ← ‖M_t‖ / √6
ΔThreshold_mood_t ← −κ_mood × clamp(m_norm, 0, 1)
```

**Fixed Code**:
```cpp
const double magnitude = std::sqrt(magnitude_sq);
const double m_norm = core::Clamp(magnitude / std::sqrt(6.0), 0.0, 1.0);
const double delta_T_mood = -kappa_mood * m_norm;
```

- [x] Add `/ std::sqrt(6.0)` after magnitude calculation
- [x] Add `core::Clamp(magnitude / std::sqrt(6.0), 0.0, 1.0)` before multiplication
- [x] Add unit test validating mood threshold modulation bounds
- [x] Update `operations_mood.test.cpp` with edge case at max mood state

---

### 1.2 Working Memory Base Capacity Formula ✅

**File**: `include/cortext/core/knobs.hpp`
**Lines**: 394-398
**Issue**: Capacity ranges [3-11] instead of paper-specified [2-6].

**Paper Formula (Section 8.1)**:
```
base_capacity = round(lerp(5, 3, S) + lerp(−1, 1, F))
```

**Fixed Code**:
```cpp
const double cap = Lerp(5.0, 3.0, S) + Lerp(-1.0, 1.0, F);
```

- [x] Change `Lerp(9.0, 5.0, S)` to `Lerp(5.0, 3.0, S)`
- [x] Change `Lerp(-2.0, 2.0, F)` to `Lerp(-1.0, 1.0, F)`
- [x] Update `core_knobs.test.cpp` with boundary tests (S=0,F=0→4; S=1,F=1→4; S=0,F=1→6; S=1,F=0→2)
- [x] Verify capacity range is [2, 6] at extreme knob values

---

### 1.3 Trajectory Drift Threshold ✅ (NO CHANGE NEEDED)

**File**: `src/operations/boundary.cpp`
**Line**: ~88
**Issue**: ~~Episode boundaries trigger at much higher drift than specified.~~

**Paper Formula (Section 5.1.3)**:
```
drift_threshold ← lerp(0.10, 0.35, T)
```

**Actual Code** (already correct):
```cpp
const double threshold = core::Lerp(constants::kGainMedium, constants::kWeightHigh, cfg.stability);
// Where kGainMedium = 0.10 and kWeightHigh = 0.35
// This already implements lerp(0.10, 0.35, T)
```

- [x] ~~Change bounds from `(0.35, 0.65)` to `(0.10, 0.35)`~~ Already correct
- [x] ~~Add constant definitions~~ Already uses `kGainMedium` and `kWeightHigh`
- [x] No test updates needed - existing tests pass
- [x] No changes required - implementation matches paper

---

### 1.4 Prediction Horizon Inversion ✅

**File**: `src/operations/predictive.cpp` (was incorrectly noted as knobs.hpp)
**Issue**: Higher Focus should increase prediction horizon, but code inverts this.

**Paper Formula (Section 8.5)**:
```
prediction_horizon = round(lerp(2, 8, F))
```

**Fixed Code**:
```cpp
inline int PredictionHorizon(double F)
{
  // prediction_horizon = round(lerp(2, 8, F))
  // Higher Focus = longer prediction horizon
  return std::max(1, static_cast<int>(
    std::round(core::Lerp(kPredictionHorizonMin, kPredictionHorizonMax, F))));
}
// Where kPredictionHorizonMin = 2.0 and kPredictionHorizonMax = 8.0
```

- [x] Change `Lerp(5.0, 1.0, F)` to `Lerp(2.0, 8.0, F)`
- [x] Rename function from `TrajectorySamples` to `PredictionHorizon` for clarity
- [x] Update all call sites to use new function name
- [x] Add test verifying higher F yields longer horizon

---

## Phase 2: Streaming Integration ✅ COMPLETE

**Priority**: HIGH
**Estimated Complexity**: Medium
**Dependencies**: Phase 1
**Status**: COMPLETE (2024-12-15)

These features enable proper interrupt pacing during streaming generation.

### 2.1 Drift Accumulation Tracking ✅

**Files**:
- `src/operations/drift_accumulation.cpp` (NEW)
- `include/cortext/operations/drift_accumulation.hpp` (NEW)
- `include/cortext/processor/processor_context.hpp`

**Issue**: `drift_accum` and `drift_at_last_interrupt` declared but never used.

**Paper Formula (Section 10.4)**:
```
drift_accum += dist(x_t, x_{last_check})
```

- [x] Create new operation `UpdateDriftAccumulation` in `src/operations/`
- [x] Compute drift increment: `dist(current_embedding, last_check_embedding)`
- [x] Add `drift_accum_snapshot` to OperationContext for cross-operation access
- [x] Reset `drift_accum` after interrupt or pacing check
- [x] Store `last_pacing_check_embedding` in ProcessorContext
- [x] Add operation to pipeline after `ComputeCoherence`

---

### 2.2 Streaming Pacing Gate ✅

**Files**:
- `src/operations/streaming_pacing.cpp` (NEW)
- `include/cortext/operations/streaming_pacing.hpp` (NEW)
- `include/cortext/core/knobs.hpp`
- `src/operations/graph_retrieval.cpp`

**Issue**: No pacing logic exists - retrieval always triggered.

**Paper Formula (Section 10.4)**:
```
pacing_thresh(S) = lerp(0.5, 0.1, S)
if drift_accum > pacing_thresh(S): trigger_check()

max_wait_drift(F) = lerp(2.0, 0.5, F)
```

- [x] Create `StreamingPacingThreshold(S)` knob function returning `Lerp(0.5, 0.1, S)`
- [x] Create `MaxWaitDrift(F)` knob function returning `Lerp(2.0, 0.5, F)`
- [x] Implement `CheckStreamingPacing` operation
- [x] Gate retrieval operations behind pacing check
- [x] Add `should_check_retrieval` flag to OperationContext
- [x] Force check when `drift_accum > max_wait_drift(F)`
- [x] Add tests for pacing behavior (`operations_streaming_pacing.test.cpp`)

---

### 2.3 Refractory Dynamics Correction ✅

**File**: `src/operations/interrupt_gate.cpp`
**Lines**: ~230-240
**Issue**: Uses signal tick count instead of drift accumulation.

**Paper Formula (Section 10)**:
```
M_refrac = 1.0 + k_refrac × exp(−Δ / τ_refrac)
where Δ = cumulative_drift_since_last_interrupt
```

**Fixed Code**:
```cpp
const double Delta = std::max(0.0, p_ctx.drift_accum - p_ctx.drift_at_last_interrupt);
```

- [x] Change `Delta` to use `drift_accum - drift_at_last_interrupt`
- [x] Update `drift_at_last_interrupt` when interrupt occurs
- [x] Reset tracking on successful interrupt
- [x] Existing tests continue to pass (drift-based delta integrates cleanly)
- [x] Comment updated to document semantic change

---

### 2.4 Interrupt Gate Additional Parameters ✅ (NO CHANGE NEEDED)

**File**: `include/cortext/core/knobs.hpp`
**Issue**: ~~Missing knob functions for streaming parameters.~~

**Actual Code** (already correct):
```cpp
inline int MaxResults(double F) {
  return static_cast<int>(std::round(Lerp(64.0, 4.0, Clamp(F, 0.0, 1.0))));
}

inline int AdjacentWindow(double F) {
  return static_cast<int>(std::round(Lerp(8.0, 1.0, Clamp(F, 0.0, 1.0))));
}
```

- [x] ~~Add `MaxResults(F)` returning `round(lerp(64, 4, F))`~~ Already exists
- [x] ~~Add `AdjacentWindow(F)` returning `round(lerp(8, 1, F))`~~ Already exists
- [x] ~~Use `MaxResults` to limit retrieval candidate count~~ Already used
- [x] ~~Use `AdjacentWindow` for boundary-aware grouping~~ Already used

---

## Phase 3: Consolidation Pipeline ✅ COMPLETE

**Priority**: HIGH
**Estimated Complexity**: High
**Dependencies**: Phase 1
**Status**: COMPLETE (2024-12-15)

These features implement the memory consolidation system.

### 3.1 Memory Clustering Operation ✅

**File**: `src/operations/consolidation_cluster.cpp`
**Issue**: No clustering algorithm exists - candidates marked but never grouped.

**Paper Formula (Section 9.3)**:
```
cluster_i = {m_j | cos(m_j, μ_i) > merge_threshold}
μ_i = centroid(cluster_i)
```

- [x] Create `ConsolidationClusterParams` struct with knob-derived parameters
- [x] Add `MergeThreshold(F)` knob function returning `Lerp(0.85, 0.95, F)`
- [x] Implement greedy clustering on marked candidates (single-linkage style)
- [x] Compute cluster centroids via running mean embedding
- [x] Store clusters temporarily via `context.SetConsolidationClusters()`
- [x] Add minimum cluster size gate: `MinClusterSize(F)` = `round(lerp(3, 10, F))`
- [x] Add unit tests in `tests/operations_consolidation_cluster.test.cpp`

---

### 3.2 Cluster Summarization ✅

**File**: `src/operations/consolidation_summarize.cpp`
**Issue**: `consolidation_summaries` table never populated.

**Paper Formula (Section 9.3)**:
```
summary.embedding = μ_i
summary.text = summarize(cluster_i.text)
summary.metadata.sources = [m.id for m in cluster_i]
```

- [x] Create summary embedding from cluster centroid
- [x] Implement extractive summarization (select most representative memory via cosine similarity)
- [x] Insert summary into `consolidation_summaries` table (with centroid BLOB)
- [x] Insert source mappings into `consolidation_sources` table
- [x] Update `cluster_id` in `embeddings_meta` for source embeddings
- [x] Create vec_embeddings entry for summary centroid
- [x] Queue ExtractionRequests for clusters meeting `MinClusterSizeForExtraction(F)`
- [x] Clear processed candidates from `consolidation_candidates`

---

### 3.3 Semantic Extraction Implementation ✅

**File**: `src/operations/consolidation.cpp` (EnqueueExtractionJobs)
**Issue**: Currently a no-op.

**Paper Formula (Section 9.4)**:
```
extraction_batch_size = round(lerp(8, 32, T))
entity_frequency_threshold = round(lerp(5, 15, T))
extraction_interval = lerp(300, 3600, T)
max_extractions_per_cycle = round(lerp(20, 5, T))
```

- [x] Define extraction interface/callback for external LLM integration (`ExtractionCallback`)
- [x] Create `ExtractionRequest` struct with summary_id, text, source_texts, cluster_size
- [x] Create `ExtractionResult` struct with entities and relations
- [x] Add `ExtractionIntervalSeconds(T)` knob function returning `Lerp(300, 3600, T)`
- [x] Add `EntityFrequencyThreshold(T)` knob function returning `round(Lerp(5, 15, T))`
- [x] Implement batch queuing respecting `MaxExtractionsPerCycle(T)`
- [x] Add extraction interval tracking in ProcessorContext (`last_extraction_ts`)
- [x] Implement `ProcessExtractionResults` operation to populate tables
- [x] Insert entities into `extraction_entities` table
- [x] Insert relations into `extraction_relations` table
- [x] Update `entity_index` for name→node_id mapping

---

### 3.4 Consolidation Pipeline Integration ✅

**File**: `src/cortext.cpp` (pipeline construction)

- [x] Add `ConsolidationCluster` operation after `ConsolidationGate`
- [x] Add `ConsolidationSummarize` operation after clustering
- [x] Add `EnqueueExtractionJobs` operation after summarization
- [x] Add `ProcessExtractionResults` operation after extraction
- [x] Gate all operations behind `ConsolidationGate` (via `GetConsolidationShouldStart()`)
- [x] Update CMakeLists.txt with new source files
- [x] Add unit tests for ConsolidationCluster operation

**Pipeline Order**:
```
EvaluateConsolidation → ConsolidationGate → ConsolidationCluster →
ConsolidationSummarize → EnqueueExtractionJobs → ProcessExtractionResults →
BuildGraphFromConsolidation
```

---

## Phase 4: Knowledge Graph Enhancement

**Priority**: MEDIUM
**Estimated Complexity**: Medium
**Dependencies**: Phase 3

These features complete the knowledge graph construction.

### 4.1 Co-occurrence Edge Construction

**File**: `src/operations/graph_build.cpp`
**Issue**: Missing co-occurrence edges from embedding similarity.

**Paper Formula (Section 9.5.1)**:
```
for (m_i, m_j) in cluster.sources:
  cos_sim ← cos(m_i.embedding, m_j.embedding)
  if cos_sim > lerp(0.85, 0.95, F):
    create_edge(m_i, m_j, 'co_occurs_with', cos_sim)
```

- [ ] TODO: Add `CoOccurrenceThreshold(F)` knob function
- [ ] TODO: Iterate source pairs within each cluster
- [ ] TODO: Compute pairwise cosine similarities
- [ ] TODO: Create `co_occurs_with` edges above threshold
- [ ] TODO: Set edge weight to similarity value
- [ ] TODO: Add test for co-occurrence edge creation

---

### 4.2 Causal/Temporal Edge Construction

**File**: `src/operations/graph_build.cpp`
**Issue**: Missing temporal causality edges.

**Paper Formula (Section 9.5.1)**:
```
temporal_order ← sort_by_timestamp(cluster.sources)
for i in range(len(temporal_order) − 1):
  m_i, m_j ← temporal_order[i], temporal_order[i+1]
  drift_vec ← m_j.embedding − m_i.embedding
  drift_mag ← ‖drift_vec‖
  if drift_mag > lerp(0.15, 0.35, T):
    create_edge(m_i, m_j, 'causes', drift_mag)
```

- [ ] TODO: Add `CausalDriftThreshold(T)` knob function returning `Lerp(0.15, 0.35, T)`
- [ ] TODO: Sort cluster sources by timestamp
- [ ] TODO: Compute sequential drift magnitudes
- [ ] TODO: Create `causes` edges above drift threshold
- [ ] TODO: Set edge weight to drift magnitude
- [ ] TODO: Add `implies` edge type for weaker correlations
- [ ] TODO: Add test for causal edge construction

---

### 4.3 Contradiction and Reinforcement Edges

**File**: `src/operations/graph_build.cpp`
**Issue**: Missing contradiction and reinforcement edge types.

- [ ] TODO: Implement `contradicts` edge for strong negative similarity
- [ ] TODO: Define contradiction threshold: `cos_sim < -0.5`
- [ ] TODO: Implement `reinforces` edge from joint retrieval frequency
- [ ] TODO: Track co-retrieval counts in `memory_feedback` table
- [ ] TODO: Create reinforcement edges when co-retrieval > threshold
- [ ] TODO: Add edge decay based on time since last reinforcement

---

### 4.4 Concept Node Generation

**File**: New `src/operations/concept_detection.cpp`
**Issue**: Concept nodes (emergent centroids) not implemented.

**Paper Description (Section 9.5)**:
> Concept Nodes: Emergent centroids representing recurrent topics

- [ ] TODO: Define concept detection criteria (cluster recurrence, cross-episode presence)
- [ ] TODO: Implement concept centroid computation from related summaries
- [ ] TODO: Create concept nodes in `graph_nodes` with type='concept'
- [ ] TODO: Link concept nodes to related entities and summaries
- [ ] TODO: Add concept node embeddings to vec_embeddings

---

### 4.5 Emotional Cascade Propagation

**File**: `src/operations/emotion.cpp`
**Issue**: Emotional tags stored but not propagated through memory network.

**Paper Formula (Section 8.7)**:
```
cascade_radius = round(lerp(1, 5, S))
cascade_decay = lerp(0.7, 0.3, S)
```

- [ ] TODO: Create `PropagateEmotionalCascade` operation
- [ ] TODO: Query graph for memories within cascade_radius hops
- [ ] TODO: Apply decayed emotional intensity to neighbors
- [ ] TODO: Update `emotional_tags` for affected memories
- [ ] TODO: Scale half_life_bonus by cascade_decay per hop
- [ ] TODO: Add test for cascade propagation depth and decay

---

## Phase 5: Testing and Validation

**Priority**: HIGH
**Estimated Complexity**: Medium
**Dependencies**: Phases 1-4

### 5.1 Formula Validation Tests

- [ ] TODO: Create `tests/formula_validation.test.cpp`
- [ ] TODO: Test all knob-derived functions against paper formulas
- [ ] TODO: Verify boundary conditions (F=0, F=1, S=0, S=1, T=0, T=1)
- [ ] TODO: Test composite combinations (high F + low S, etc.)
- [ ] TODO: Compare numerical outputs against paper examples

---

### 5.2 Integration Tests

- [ ] TODO: Create `tests/integration_consolidation.test.cpp`
- [ ] TODO: Test full consolidation flow: trigger → cluster → summarize → extract
- [ ] TODO: Verify graph construction from consolidation output
- [ ] TODO: Test streaming pacing with simulated rapid signals
- [ ] TODO: Test interrupt gate with refractory dynamics

---

### 5.3 Regression Tests

- [ ] TODO: Capture baseline behavior metrics before Phase 1 changes
- [ ] TODO: Verify write rate stability after formula corrections
- [ ] TODO: Verify episode boundary frequency changes appropriately
- [ ] TODO: Monitor working memory capacity under various knob settings

---

## Phase 6: Documentation Updates

**Priority**: LOW
**Estimated Complexity**: Low
**Dependencies**: Phases 1-5

### 6.1 Algorithm Documentation Sync

- [ ] TODO: Update `docs/algorithms.md` to match implementation
- [ ] TODO: Document any intentional deviations from paper
- [ ] TODO: Add implementation notes for complex operations
- [ ] TODO: Update section references to match paper numbering

---

### 6.2 Database Schema Documentation

- [ ] TODO: Verify `docs/diagrams/database.md` reflects final schema
- [ ] TODO: Add documentation for new tables/columns from Phase 3-4
- [ ] TODO: Update relationship cardinalities if changed
- [ ] TODO: Add query examples for common operations

---

### 6.3 API Documentation

- [ ] TODO: Document new knob functions added in each phase
- [ ] TODO: Update operation pipeline documentation
- [ ] TODO: Add streaming integration guide
- [ ] TODO: Document extraction callback interface

---

## Implementation Timeline

| Phase | Description | Complexity | Dependencies | Status |
|-------|-------------|------------|--------------|--------|
| **Phase 1** | Critical Formula Corrections | Low | None | ✅ COMPLETE |
| **Phase 2** | Streaming Integration | Medium | Phase 1 | ✅ COMPLETE |
| **Phase 3** | Consolidation Pipeline | High | Phase 1 | ✅ COMPLETE |
| **Phase 4** | Knowledge Graph Enhancement | Medium | Phase 3 | Ready |
| **Phase 5** | Testing and Validation | Medium | Phases 1-4 | Blocked |
| **Phase 6** | Documentation Updates | Low | Phases 1-5 | Blocked |

---

## Success Criteria

### Phase 1 Complete When: ✅ DONE
- [x] All formula corrections pass unit tests (787 assertions, 188 test cases)
- [x] Mood threshold modulation bounded to expected range (√6 normalization)
- [x] WM capacity within [2, 6] range (verified at all extreme knob values)
- [x] Episode boundaries trigger at correct drift thresholds (already correct)

### Phase 2 Complete When: ✅ DONE
- [x] `drift_accum` actively tracked and used (via UpdateDriftAccumulation operation)
- [x] Streaming pacing gates retrieval appropriately (CheckStreamingPacing operation)
- [x] Refractory dynamics use drift instead of ticks (interrupt_gate.cpp fixed)
- [x] All tests pass (814 assertions in 198 test cases)

### Phase 3 Complete When: ✅ DONE
- [x] `consolidation_summaries` populated from clusters
- [x] `consolidation_sources` tracks source→summary mapping
- [x] Extraction interface defined and callable (`ExtractionCallback`)
- [x] Full consolidation pipeline executes without errors
- [x] All tests pass (818 assertions in 202 test cases)

### Phase 4 Complete When:
- [ ] All 6 edge types constructable
- [ ] Concept nodes detected and created
- [ ] Emotional cascades propagate through graph
- [ ] Graph-augmented retrieval benefits from new edges

### Phase 5 Complete When:
- [ ] 100% formula coverage in validation tests
- [ ] Integration tests pass for all new features
- [ ] No regressions in existing functionality

### Phase 6 Complete When:
- [ ] All documentation matches implementation
- [ ] No undocumented deviations from paper
- [ ] API fully documented for external consumers

---

## Notes

- Phases 2 and 3 can proceed in parallel after Phase 1
- Phase 4 depends on Phase 3 consolidation tables being populated
- Consider feature flags for gradual rollout of streaming pacing
- Extraction interface should support both sync and async implementations
