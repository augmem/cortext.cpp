# Plan: Fix Audit Action Items

## Overview

This plan addresses the critical issues identified in the comprehensive alignment audit. The fixes are grouped by severity and dependency order.

---

## Critical Fixes (Breaking Issues)

### 1. Fix `recent_context` view query (Runtime Error)

**File:** `src/signal_processor.cpp`
**Line:** 544

**Problem:** Query uses non-existent `seq_order` column
```cpp
// Current:
"SELECT embedding FROM recent_context ORDER BY seq_order ASC"

// Fix:
"SELECT embedding FROM recent_context ORDER BY timestamp ASC"
```

**Test:** Run existing signal processor tests to verify context loading works.

---

### 2. Fix memory_feedback table reference (Runtime Error)

**File:** `src/signal_processor.cpp`
**Lines:** 32-34 (inside `ComputeObservedRetention`)

**Problem:** Queries non-existent v1 table `memory_feedback`
```cpp
// Current:
"SELECT mf.last_used FROM memory_feedback mf "
"JOIN embeddings e ON e.embedding_id = mf.embedding_id "
"WHERE e.strength >= ? AND mf.last_used > 0"

// Fix - use v2 memories table:
"SELECT last_used FROM memories "
"WHERE strength >= ? AND last_used > 0"
```

**Test:** Run consolidation and memory strength tests.

---

### 3. Remove/Fix observed_retention_history loading (Runtime Error)

**File:** `src/signal_processor.cpp`
**Lines:** 589-609 (`LoadObservedRetentionHistory` function)

**Problem:** Tries to load from non-existent table, catches exception silently.

**Fix:** Make it a no-op that returns empty since this is transient data derived from memories. The comment at line 1084 already acknowledges this.

```cpp
void LoadObservedRetentionHistory(Store* /*store*/, ProcessorContext& /*ctx*/) {
  // v2 schema: observed_retention_history is transient, computed from memories.last_used
  // No persistence needed - the history rebuilds naturally during signal processing
}
```

**Test:** Verify stability operations work without pre-loaded history.

---

### 4. Fix blob tracking in accumulator (Critical for Working Memory)

**Files:**
- `src/operations/accumulator.cpp` - Lines 17-30 (`CreateSignalRecord`)
- `include/cortext/operations/accumulator.hpp` - Add Store* parameter

**Problem:** Comment says "blob_id will be populated later" but it never is. This breaks working memory content hydration.

**Spec alignment:**
- algorithms.md Section 6.1.1: `memory.blob_ids ← [blob_1, blob_2, ..., blob_n]` - each signal tracks its blob
- database-v2.md: "SIGNALS owns both embedding_id and blob_id" and "Each signal has one content blob"

**Solution: Store blob at accumulation time** (aligns with spec - signals own their blobs)

**Part A - Update CreateSignalRecord signature and body (lines 17-30):**
```cpp
SignalRecord
CreateSignalRecord (const Signal &signal, double score, int serial_position,
                    Store* store)  // Add store parameter
{
  SignalRecord rec;
  rec.embedding = signal.embedding;
  rec.timestamp = signal.timestamp;
  rec.modality = signal.modality;
  rec.mime = signal.mimetype;
  rec.score = score;
  rec.serial_position = serial_position;

  // Store payload to objstore and capture blob_id (v2: SIGNALS owns blob_id)
  if (signal.payload && !signal.payload->empty() && store != nullptr) {
    auto blob_rows = store->Execute("SELECT objstore_put(?1) AS id",
                                    { *signal.payload });
    if (!blob_rows.empty() && blob_rows[0].count("id") != 0) {
      rec.blob_id = BlobFromAny(blob_rows[0].at("id"));
    }
  }

  return rec;
}
```

**Part B - Update call sites** (lines 66, 92, 110):
- Get Store* from registry: `auto* store = registry.Get<Store>("store");`
- Pass to CreateSignalRecord

**Part C - Update memory_storage.cpp:**
- Remove duplicate blob storage (lines 124-134) since blobs are now stored at accumulation
- The acc.signals already have blob_ids, just use them in INSERT

**Test:** Create test that verifies working memory slots have non-empty blob_ids after signal processing.

---

## Moderate Fixes (Consistency Issues)

### 5. Fix emotional cascade threshold (Section 6.7 alignment)

**File:** `src/operations/emotion_cascade.cpp`
**Line:** 62

**Problem:** Uses hardcoded `0.5` instead of knob-derived `ThetaIntensity(S)`

```cpp
// Current:
"WHERE flashbulb = 1 AND emotional_intensity >= 0.5 "

// Fix - use parameterized threshold:
"WHERE flashbulb = 1 AND emotional_intensity >= ?2 "
// And pass: { recent_window_ts, core::ThetaIntensity(cfg.sensitivity) }
```

**Also update function signature (line 52):**
```cpp
LoadEmotionalSources(Store *store, long long recent_window_ts, double S)
```

**Note:** Arousal threshold cannot be checked in SQL since arousal is not stored per-memory in v2 schema.

**Test:** Verify emotional cascade uses sensitivity-derived thresholds.

---

### 6. Update stale comments referencing v1 tables

**Files and lines:**
- `src/operations/memory_strength.cpp:117` - "memory_feedback" → "memories table"
- `src/operations/reconsolidation.cpp:202, 293` - "memory_feedback" → "memories table"
- `src/operations/predictive.cpp:194` - "memory_feedback" → "memories table"

---

## Legacy Code Removal (Required)

### 7. Rename objstore to blobs for documentation consistency

**File:** `src/store/schema.cpp`
**Line:** 155

Change: `"CREATE VIRTUAL TABLE IF NOT EXISTS objstore USING objstore()"`
To: `"CREATE VIRTUAL TABLE IF NOT EXISTS blobs USING objstore()"`

Then update all references from `objstore` to `blobs` throughout codebase.

### 8. Remove legacy v1 tables and migrate all references

This is a full migration involving 8 legacy tables with ~150+ references.

**v2 Schema Mapping (from database-v2.md):**

| v1 Table | v2 Replacement |
|----------|----------------|
| `graph_nodes` | `MEMORIES` with `kind='LABEL'` |
| `graph_edges` | `ASSOCIATIONS` table |
| `consolidation_summaries` | `MEMORIES` with `kind='ASSOCIATION'` |
| `consolidation_sources` | `ASSOCIATIONS` with `edge_type='derived_from'` |
| `extraction_entities` | `MEMORIES.label` field |
| `extraction_relations` | `ASSOCIATIONS` |
| `entity_index` | `MEMORIES.label` field |
| `goal_nodes` | Remove (application concern, not memory storage) |

**Note:** `consolidation_candidates` is a v2 table - keep it.

---

#### 8.1 Migrate `graph_build.cpp`

**Current:** Creates `graph_nodes`, `graph_edges`, `entity_index` entries
**v2 Migration:**
- `graph_nodes` type='summary' → INSERT INTO `memories` with `kind='ASSOCIATION'`
- `graph_nodes` type='entity' → INSERT INTO `memories` with `kind='LABEL'`, set `label=name`
- `graph_nodes` type='concept' → INSERT INTO `memories` with `kind='LABEL'`, set `label=name`
- `graph_edges` → INSERT INTO `associations` with appropriate `edge_type`
- `entity_index` → Use `memories.label` field instead

**Lines to change:** 271-278, 280-288, 290-293, 298-383, 416, 424

---

#### 8.2 Migrate `graph_retrieval.cpp`

**Current:** Queries `graph_edges` for expansion, `graph_nodes` for embedding lookup
**v2 Migration:**
- `graph_edges` → Query `associations` table
- `graph_nodes` → Query `memories` where `kind='LABEL'`
- Reinforcement INSERT → Use `associations`

**Lines to change:** 56, 195-199, 221, 241

---

#### 8.3 Migrate `concept_detection.cpp`

**Current:** Queries `consolidation_sources`, `extraction_entities`, `consolidation_summaries`, `graph_nodes`
**v2 Migration:**
- `consolidation_sources` → Query `associations` with `edge_type='derived_from'`
- `extraction_entities` → Query `memories` where `kind='LABEL'`
- `consolidation_summaries` → Query `memories` where `kind='ASSOCIATION'`
- `graph_nodes` → Query `memories` where `kind='LABEL'`
- INSERT `graph_nodes` → INSERT INTO `memories` with `kind='LABEL'`
- INSERT `graph_edges` → INSERT INTO `associations`

**Lines to change:** 54-61, 114, 149, 217, 267-288

---

#### 8.4 Migrate `consolidation_summarize.cpp`

**Current:** Inserts into `consolidation_summaries`, `consolidation_sources`
**v2 Migration:**
- `consolidation_summaries` → INSERT INTO `memories` with `kind='ASSOCIATION'`
- `consolidation_sources` → INSERT INTO `associations` with `edge_type='derived_from'`

**Lines to change:** 215-227, 290

---

#### 8.5 Migrate `consolidation.cpp`

**Current:** Queries `graph_edges` for connectivity metric
**v2 Migration:**
- `graph_edges` → Query `associations` table

**Lines to change:** 224

---

#### 8.6 Migrate `reconsolidation.cpp`

**Current:** Queries `graph_edges` for related memories
**v2 Migration:**
- `graph_edges` → Query `associations` table

**Lines to change:** 113

---

#### 8.7 Migrate `emotion_cascade.cpp`

**Current:** Queries `graph_edges` for cascade propagation
**v2 Migration:**
- `graph_edges` → Query `associations` table

**Lines to change:** 156-161

---

#### 8.8 Migrate `goal_alignment.cpp`

**Current:** Queries `goal_nodes`, `graph_nodes`, `graph_edges`
**v2 Migration:**
- Remove `goal_nodes` (application concern)
- `graph_nodes` → Query `memories` where `kind='LABEL'`
- `graph_edges` → Query `associations`

**Lines to change:** 38, 58-65, 114

---

#### 8.9 Migrate `process_extraction_results.cpp`

**Current:** Inserts into `extraction_entities`, `entity_index`, `extraction_relations`
**v2 Migration:**
- `extraction_entities` → INSERT INTO `memories` with `kind='LABEL'`, set `label`
- `entity_index` → Use `memories.label` (no separate table)
- `extraction_relations` → INSERT INTO `associations`

**Lines to change:** 128-150

---

#### 8.10 Remove legacy table definitions

**File:** `src/store/schema.cpp`
**Lines to remove:** 465-536 (entire legacy section)

---

#### 8.11 Update all tests

Each test needs to seed data using v2 tables instead of legacy tables:
- Replace `graph_nodes` INSERTs with `memories` INSERTs
- Replace `graph_edges` INSERTs with `associations` INSERTs
- Replace `consolidation_*` INSERTs with `memories`/`associations`
- Remove `operations_graph_schema.test.cpp` if it only tests legacy tables

---

#### 8.12 Update headers and scripts

- Update documentation comments in headers
- Update `scripts/analyze_memory_db.sh` queries

### 9. Remove LoadObservedRetentionHistory function entirely

**File:** `src/signal_processor.cpp`
**Lines:** 589-609

Instead of making it a no-op, remove the function entirely and remove its call site.

---

## Implementation Order

### Phase 1: Critical Runtime Fixes
1. **Fix seq_order query** (`signal_processor.cpp:544`) - unblocks runtime
2. **Fix memory_feedback reference** (`signal_processor.cpp:32-34`) - unblocks consolidation
3. **Remove LoadObservedRetentionHistory** (`signal_processor.cpp:589-609`) - remove dead code

### Phase 2: Working Memory Fix
4. **Fix blob tracking in accumulator** - critical for WM content hydration
5. **Update memory_storage.cpp** - remove duplicate blob storage

### Phase 3: Algorithm Alignment
6. **Fix emotional threshold** (`emotion_cascade.cpp`) - use knob-derived threshold
7. **Update stale comments** - trivial cleanup

### Phase 4: Schema/Table Naming
8. **Rename objstore to blobs** - schema and all references

### Phase 5: Legacy Code Removal (Major)
9. **Migrate operations to v2 ASSOCIATIONS/MEMORIES**
10. **Update all tests**
11. **Remove legacy table definitions from schema.cpp**
12. **Update scripts and documentation**

---

## Files to Modify

### Phase 1-2: Critical Fixes
| File | Lines | Change |
|------|-------|--------|
| `src/signal_processor.cpp` | 32-34 | Update SQL to query `memories` table |
| `src/signal_processor.cpp` | 544 | Change `seq_order` to `timestamp` |
| `src/signal_processor.cpp` | 589-609 | Remove `LoadObservedRetentionHistory` function |
| `src/operations/accumulator.cpp` | 17-30, 66, 92, 110 | Store blob at accumulation, populate blob_id |
| `src/operations/memory_storage.cpp` | 124-134 | Remove duplicate blob storage |

### Phase 3: Algorithm Alignment
| File | Lines | Change |
|------|-------|--------|
| `src/operations/emotion_cascade.cpp` | 52, 62 | Use `ThetaIntensity(S)` instead of 0.5 |
| `src/operations/memory_strength.cpp` | 117 | Update comment |
| `src/operations/reconsolidation.cpp` | 202, 293 | Update comments |
| `src/operations/predictive.cpp` | 194 | Update comment |

### Phase 4: Schema Naming
| File | Lines | Change |
|------|-------|--------|
| `src/store/schema.cpp` | 155 | Rename `objstore` to `blobs` |
| Multiple files | - | Update all `objstore` references |

### Phase 5: Legacy Removal (9 operations + 11 tests + headers)
| File Type | Files | Change |
|-----------|-------|--------|
| Operations | 9 files | Migrate to ASSOCIATIONS/MEMORIES |
| Tests | 11 files | Update to v2 schema |
| Headers | 4 files | Update documentation |
| Schema | schema.cpp:465-536 | Remove legacy table definitions |
| Scripts | analyze_memory_db.sh | Update queries |

---

## Test Plan

1. Run `ctest --test-dir build -R cortext_tests --output-on-failure` after each change
2. Specifically run:
   - `./build/tests/cortext_tests "[signal_processor]"`
   - `./build/tests/cortext_tests "[memory_strength]"`
   - `./build/tests/cortext_tests "[consolidation]"`
   - `./build/tests/cortext_tests "[working_memory]"`
   - `./build/tests/cortext_tests "[emotion]"`
3. Build and run chat example to verify working memory content hydration
