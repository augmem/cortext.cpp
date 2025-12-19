# Legacy Code Removal Plan

This plan outlines the steps to remove legacy database tables and operations, aligning the codebase with the V2 schema and `algorithms.md`.

## Legacy Tables to Remove

These tables (in `src/store/schema.cpp`) will be removed.

| Table | Action | Replacement / Notes |
|-------|--------|---------------------|
| `consolidation_summaries` | Delete | `MEMORIES` (kind='ASSOCIATION') |
| `consolidation_sources` | Delete | `ASSOCIATIONS` (edge_type='derived_from') |
| `extraction_entities` | Delete | `MEMORIES` (kind='LABEL') |
| `extraction_relations` | Delete | `ASSOCIATIONS` |
| `entity_index` | Delete | `MEMORIES.label` field |
| `goal_nodes` | Delete | Removed feature |
| `graph_nodes` | Delete | `MEMORIES` (kind='LABEL') |
| `graph_edges` | Delete | `ASSOCIATIONS` |
| `consolidation_candidates` | **Deleted** | ~~Refactor to compute & pass in-memory via `OperationContext`~~ (Phase 2 complete) |
| `memory_feedback` | Delete | `MEMORIES` (merged columns) |

---

## Phased Execution Checklist

### Phase 1: Feature Deletion & Cleanup
Target: Remove unused files and features to clear the path.

- [x] **Delete Goal Alignment Feature**
    - [x] Remove `src/operations/goal_alignment.cpp`
    - [x] Remove `include/cortext/operations/goal_alignment.hpp`
    - [x] Remove `src/operations/goal_alignment_fallback.cpp`
    - [x] Remove `include/cortext/operations/goal_alignment_fallback.hpp`
    - [x] Remove `tests/operations_goal_alignment.test.cpp`
- [x] **Delete Obsolete Schema Helpers**
    - [x] Remove `src/operations/graph_schema.cpp`
    - [x] Remove `include/cortext/operations/graph_schema.hpp`
    - [x] Remove `tests/operations_graph_schema.test.cpp`

### Phase 2: Core Refactoring (In-Memory Candidates)
Target: Replace `consolidation_candidates` table with `OperationContext` passing.

- [x] **Update `OperationContext`**
    - [x] Add `struct ConsolidationCandidate { long long embedding_id; double score; Eigen::VectorXf embedding; };`
    - [x] Add `std::vector<ConsolidationCandidate> candidates_` member.
    - [x] Add `SetConsolidationCandidates(...)` and `GetConsolidationCandidates()` methods.
- [x] **Refactor `ScoreConsolidation` (`src/operations/consolidation.cpp`)**
    - [x] Remove `INSERT INTO consolidation_candidates`.
    - [x] Update logic to collect qualifying candidates into `std::vector<ConsolidationCandidate>`.
    - [x] Call `context.SetConsolidationCandidates(candidates)`.
    - [x] Update connectivity query to use `ASSOCIATIONS` table (or temporary stub if needed before full migration).
- [x] **Refactor `ConsolidationCluster` (`src/operations/consolidation_cluster.cpp`)**
    - [x] Remove `SELECT ... FROM consolidation_candidates`.
    - [x] Update logic to read from `context.GetConsolidationCandidates()`.
- [x] **Remove `consolidation_candidates` table**
    - [x] Remove from `src/store/schema.cpp`
    - [x] Remove `DELETE FROM consolidation_candidates` from `consolidation_summarize.cpp`
    - [x] Update tests (`integration_consolidation.test.cpp`)

### Phase 3: Operation Migration (Graph & Extractions)
Target: Switch all remaining operations to use V2 tables (`MEMORIES`, `ASSOCIATIONS`).

- [x] **Refactor `ConsolidationSummarize` (`src/operations/consolidation_summarize.cpp`)**
    - [x] Remove `INSERT INTO consolidation_summaries`.
    - [x] Remove `INSERT INTO consolidation_sources`.
    - [x] Ensure it writes local sources to `ASSOCIATIONS` (derived_from).
    - [x] Remove `DELETE FROM consolidation_candidates` statement (done in Phase 2).
- [x] **Refactor `ProcessExtractionResults` (`src/operations/process_extraction_results.cpp`)**
    - [x] Replace `extraction_entities` INSERTs with `MEMORIES` (kind='LABEL') INSERTs.
    - [x] Remove `entity_index` INSERTs.
    - [x] Replace `extraction_relations` INSERTs with `ASSOCIATIONS` INSERTs.
- [x] **Rewrite `GraphBuild` (`src/operations/graph_build.cpp`)**
    - [x] Remove node creation logic (now uses MEMORIES with cluster_id).
    - [x] Rewrite edge creation to INSERT into `ASSOCIATIONS`.
    - [x] Remove `entity_index` dependency.
    - [x] Update tests to use V2 schema (`operations_graph_build.test.cpp`).

### Phase 4: Query Updates
Target: Update readers to look at V2 tables.

- [x] **Update `GraphRetrieval` (`src/operations/graph_retrieval.cpp`)**
    - [x] Rewrite CTEs to traverse `ASSOCIATIONS`.
    - [x] Update KNN query to use `k = ?` syntax for sqlite-vec compatibility.
    - [x] Create reinforcement edges in `ASSOCIATIONS` instead of `graph_edges`.
- [x] **Update `ConceptDetection` (`src/operations/concept_detection.cpp`)**
    - [x] Query `MEMORIES` (kind='LABEL') instead of `extraction_entities`.
    - [x] Query `ASSOCIATIONS` instead of `consolidation_sources`.
    - [x] Create concept nodes as `MEMORIES` (kind='LABEL') with 'generalizes' edges in `ASSOCIATIONS`.
- [x] **Update `SignalProcessor` (`src/signal_processor.cpp`)**
    - [x] Remove `ComputeObservedRetention` query to `memory_feedback` (legacy table).
- [x] **Update Misc Operations**
    - [x] `emotion_cascade.cpp`: Update graph traversal to use `ASSOCIATIONS` with memory_ids.
    - [x] `reconsolidation.cpp`: Update graph traversal to use `ASSOCIATIONS` with memory_ids.
- [x] **Update `ScoreConsolidation` (`src/operations/consolidation.cpp`)**
    - [x] Update connectivity query to count edges from `ASSOCIATIONS` instead of `graph_edges`.
- [x] **Update Tests**
    - [x] `operations_graph_retrieval.test.cpp`: Update to use `ASSOCIATIONS` instead of `graph_edges`.
    - [x] `operations_reconsolidation.test.cpp`: Update ripple tests to use `ASSOCIATIONS` and seed memories properly.

### Phase 5: Schema Removal
Target: Drop the legacy tables.

- [x] **Update `src/store/schema.cpp`**
    - [x] Remove definitions for all 9 legacy tables listed above.
    - [x] Remove legacy indices.

### Phase 6: Test Updates & Verification
Target: Ensure system stability.

- [x] **Update Tests**
    - [x] `operations_consolidation.test.cpp`: Verify in-memory candidate passing.
    - [x] `operations_graph_build.test.cpp`: Verify `ASSOCIATIONS` usage.
    - [x] `operations_extraction.test.cpp`: Verify `MEMORIES`/`ASSOCIATIONS` usage.
    - [x] `integration_consolidation.test.cpp`: Fix any broken integration flows.
    - [x] `operations_reconsolidation.test.cpp`: Replace `graph_edges` with `associations`.
    - [x] `operations_concept_detection.test.cpp`: Replace legacy tables with V2 schema.
    - [x] `operations_emotion_cascade.test.cpp`: Replace `graph_nodes`/`graph_edges` with V2 schema.
- [x] **Verification**
    - [x] Run full test suite: All 371 tests pass (1858 assertions)
