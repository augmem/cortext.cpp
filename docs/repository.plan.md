# Repository Pattern Refactoring Plan

## Objective
Decouple business logic from data access by implementing a **Repository / DAO Pattern**. This will move raw SQL strings out of `src/operations` and into a dedicated `src/store/repositories` layer, enabling strongly-typed C++ logic, improved testability, and compiler-checked refactoring.

## Strategy
*   **Data Transfer Objects (DTOs)**: simple C++ structs representing database rows (e.g., `MemoryState`, `GraphNode`).
*   **Repositories**: Classes that encapsulate SQL execution and return DTOs (e.g., `MemoryRepository::GetActive()`).
*   **Pure Logic**: Operations perform read-modify-write cycles using C++ math instead of complex SQL update strings.

---

## Phase 1: Foundation & Proof of Concept
**Goal**: Establish the pattern with the most critical/complex operation (`memory_strength.cpp`) to prove the benefits.

### 1.1 Scaffold Directory Structure
*   [ ] Create `include/cortext/store/models.hpp` for shared DTOs.
*   [ ] Create `include/cortext/store/repositories/` directory.
*   [ ] Create `src/store/repositories/` directory.

### 1.2 Define Core DTOs
*   [ ] Define `MemoryState` struct in `models.hpp`:
    ```cpp
    struct MemoryState {
        int64_t id;
        double strength;
        double use_frequency;
        double last_access;
        double lability_state;
        // ...
    };
    ```

### 1.3 Create MemoryRepository
*   [ ] Implement `MemoryRepository` class:
    *   Constructor taking `Store*` or `Transaction*`.
    *   `std::vector<MemoryState> GetActiveMemories(double strength_cutoff)`
    *   `std::vector<MemoryState> GetByIds(const std::vector<int64_t>& ids)`
    *   `void UpdateMemoryStrength(int64_t id, double strength, double freq, int64_t last_access)`
    *   `void RecordFeedback(int64_t id, bool used, int64_t timestamp)`

### 1.4 Refactor `MemoryStrength` Operation (POC)
*   [ ] Rewrite `src/operations/memory_strength.cpp`:
    *   **Remove**: The massive SQL `UPDATE` string containing exponential decay math.
    *   **Implement**:
        1.  `repo.GetActiveMemories()` to load state.
        2.  C++ implementation of the decay/boost math (move from SQL to `cortext/core/math_utils` or local logic).
        3.  Loop through results, apply math, call `repo.UpdateMemoryStrength()`.

---

## Phase 2: Core Entities & State
**Goal**: Migrate operations that mutate the primary state of the system (embeddings, feedback, gates).

### 2.1 Interrupt & Gating Repositories
*   [ ] Create `SystemRepository` for `system_state` table queries (e.g., `metrics`).
*   [ ] Refactor `GenerateInterruptGate` and `ComputeMniGateDecision` to use repositories.
    *   Note: These involve vector distances; repositories should handle specific vector query wrappers if standard SQL isn't enough (e.g., specific `sqlite-vec` calls).

### 2.2 Reconsolidation Repository
*   [ ] Expand `MemoryRepository` to handle lability fields.
*   [ ] Refactor `src/operations/reconsolidation.cpp` logic:
    *   Move "Ripple Propagation" graph queries into a `GraphRepository` (see Phase 3).
    *   Move the drift/blend math into C++.
    *   Use repository methods for the `DELETE` + `INSERT` vector update pattern.

### 2.3 Feedback Mechanisms
*   [ ] Refactor `FocusFeedback`, `SensitivityFeedback`, `StabilityFeedback`:
    *   Implement `SystemRepository::GetMetric(type)` and `UpdateMetric()`.
    *   Move logic like "narrowing/widening gain" calculating into C++.

---

## Phase 3: Graph & Structural Operations
**Goal**: Migrate complex graph building and episode alignment logic. This removes "stringly-typed" ID generation from business logic.

### 3.1 Graph Operations
*   [ ] Create `GraphRepository`:
    *   `void AddNode(const GraphNode& node)`
    *   `void AddEdge(const GraphEdge& edge)`
    *   `std::vector<GraphNeighbor> GetNeighbors(int64_t embedding_id, int depth)`
*   [ ] Refactor `src/operations/graph_build.cpp`:
    *   Move logic for generating `summary:<id>` and `entity:<name>` IDs into a helper `cortext::core::IdFactory`.
    *   Replace raw SQL inserts with `repo.AddNode()/AddEdge()`.
    *   Move co-occurrence checks (cosine similarity thresholds) into explicit C++ logic loop.

### 3.2 Episodic Storage
*   [ ] Create `EpisodeRepository`:
    *   `void CreateEpisode(...)`
    *   `Episode GetLastEpisode()`
*   [ ] Refactor `src/operations/boundary.cpp` to use this repository.

---

## Phase 4: Utilities, Cleanup & Optimization
**Goal**: Finalize the separation and clean up technical debt.

### 4.1 Centralize Initialization
*   [ ] Create `src/store/schema_init.cpp` (or enhance `SchemaRegistry`).
    *   Move all `CREATE TABLE IF NOT EXISTS` statements from individual operations (like `graph_build.cpp`) into this central initialization step.
    *   Ensure operations assume tables exist.

### 4.2 Utility Migration
*   [ ] Identify redundant math utils in `boundary.cpp`, `interrupt_gate.cpp`.
*   [ ] Move unique normalization/centroid logic to `cortext::core::algorithms`.

### 4.3 Validation
*   [ ] Verify no raw `tx.Execute("UPDATE...")` remains in `src/operations/`.
*   [ ] Ensure `src/store/repositories/` is the **only** place allowed to have SQL strings.

## Benefits Checklist
*   [ ] **Testability**: Can we mock `MemoryRepository` to test decay logic without a DB?
*   [ ] **Safety**: Does changing `MemoryState::strength` cause a compile error in the update logic (forcing us to fix it)?
*   [ ] **Clarity**: Is the math visible in C++ rather than hidden in a SQL string?
