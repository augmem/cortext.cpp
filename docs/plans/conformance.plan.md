# Plan: Fix Audit Action Items

## Overview

This plan addresses the remaining issues identified in the comprehensive alignment audit. Many items from the original audit have been completed as part of the legacy code removal (see `remove-leagcy.plan.md`).

**Status:** Updated after legacy table migration completed (Phase 1-6 of remove-leagcy.plan.md)

---

## Critical Fixes (Breaking Issues)

### 1. Fix `recent_context` view query (Runtime Error)

**File:** `src/signal_processor.cpp`
**Line:** 473

**Problem:** Query uses non-existent `seq_order` column
```cpp
// Current:
"SELECT embedding FROM recent_context ORDER BY seq_order ASC"

// Fix:
"SELECT embedding FROM recent_context ORDER BY timestamp ASC"
```

**Test:** Run existing signal processor tests to verify context loading works.

---

### 2. Remove/Fix observed_retention_history loading (Runtime Error)

**File:** `src/signal_processor.cpp`
**Lines:** 518-538 (`LoadObservedRetentionHistory` function)

**Problem:** Tries to load from non-existent `observed_retention_history` table, catches exception silently.

**Fix:** Make it a no-op that returns empty since this is transient data derived from memories.

```cpp
void LoadObservedRetentionHistory(Store& /*store*/, ProcessorContext& /*ctx*/) {
  // v2 schema: observed_retention_history is transient, computed from memories.last_used
  // No persistence needed - the history rebuilds naturally during signal processing
}
```

**Also remove call site at line 714.**

**Test:** Verify stability operations work without pre-loaded history.

---

### 3. Fix blob tracking in accumulator (Critical for Working Memory)

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

### 4. Fix emotional cascade threshold (Section 6.7 alignment)

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

### 5. Update stale comments referencing v1 tables

**Files and lines:**
- `src/operations/memory_strength.cpp:117` - "memory_feedback" → "memories table"
- `src/operations/reconsolidation.cpp:202, 293` - "memory_feedback" → "memories table"
- `src/operations/predictive.cpp:194` - "memory_feedback" → "memories table"

---

## Optional Cleanup

### 6. Rename objstore to blobs for documentation consistency

**File:** `src/store/schema.cpp`
**Line:** 107

Change: `"CREATE VIRTUAL TABLE IF NOT EXISTS objstore USING objstore()"`
To: `"CREATE VIRTUAL TABLE IF NOT EXISTS blobs USING objstore()"`

Then update all references from `objstore` to `blobs` throughout codebase.

**Note:** This is optional cosmetic cleanup. The current `objstore` name works correctly.

---

## Completed Items (Removed from Plan)

The following items were completed as part of the legacy code removal plan:

- **Section 2 (memory_feedback reference)** - Already fixed, no `memory_feedback` references remain
- **Section 8 (Legacy v1 table migrations)** - All 9 subsections completed:
  - 8.1-8.7: All operations migrated to V2 schema (ASSOCIATIONS, MEMORIES)
  - 8.8: `goal_alignment.cpp` removed entirely (file deleted)
  - 8.9: `process_extraction_results.cpp` migrated
  - 8.10: Legacy table definitions removed from schema.cpp
  - 8.11: All tests updated to V2 schema
  - 8.12: Headers updated
- **`operations_graph_schema.test.cpp`** - Removed (file deleted)
- **`consolidation_candidates` table** - Now uses in-memory passing via OperationContext

---

## Implementation Order

### Phase 1: Critical Runtime Fixes
1. **Fix seq_order query** (`signal_processor.cpp:473`) - unblocks runtime
2. **Remove LoadObservedRetentionHistory** (`signal_processor.cpp:518-538`) - remove dead code

### Phase 2: Working Memory Fix
3. **Fix blob tracking in accumulator** - critical for WM content hydration
4. **Update memory_storage.cpp** - remove duplicate blob storage

### Phase 3: Algorithm Alignment
5. **Fix emotional threshold** (`emotion_cascade.cpp:62`) - use knob-derived threshold
6. **Update stale comments** - trivial cleanup

### Phase 4: Optional Cleanup
7. **Rename objstore to blobs** - schema and all references (optional)

---

## Files to Modify

### Phase 1: Critical Fixes
| File | Lines | Change |
|------|-------|--------|
| `src/signal_processor.cpp` | 473 | Change `seq_order` to `timestamp` |
| `src/signal_processor.cpp` | 518-538, 714 | Remove `LoadObservedRetentionHistory` function and call |

### Phase 2: Working Memory Fix
| File | Lines | Change |
|------|-------|--------|
| `src/operations/accumulator.cpp` | 17-30, 66, 92, 110 | Store blob at accumulation, populate blob_id |
| `src/operations/memory_storage.cpp` | 124-134 | Remove duplicate blob storage |

### Phase 3: Algorithm Alignment
| File | Lines | Change |
|------|-------|--------|
| `src/operations/emotion_cascade.cpp` | 52, 62 | Use `ThetaIntensity(S)` instead of 0.5 |
| `src/operations/memory_strength.cpp` | 117 | Update comment |
| `src/operations/reconsolidation.cpp` | 202, 293 | Update comments |
| `src/operations/predictive.cpp` | 194 | Update comment |

### Phase 4: Optional Cleanup
| File | Lines | Change |
|------|-------|--------|
| `src/store/schema.cpp` | 107 | Rename `objstore` to `blobs` |
| Multiple files | - | Update all `objstore` references |

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
