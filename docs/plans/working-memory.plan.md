# Plan: Expose Working Memory in Cortext API

## Goal

Replace local chat history with Cortext's working memory system. The LLM should only see what working memory has decided to keep active - naturally bounded by capacity, with decay, chunking, and intelligent eviction.

## API Design

Split `Context::memories` into two separate vectors using the same `Memory` type:

```cpp
struct Context {
  std::vector<Memory> working_memory;    // Active WM slots (conversation context)
  std::vector<Memory> retrieved_memory;  // Long-term store query results (injected context)
  bool should_interrupt = false;
  ProcessorOutput output;
};
```

Both use `Memory` struct - WM slots populate `HistoricalMetrics` with their aggregated signal metrics.

---

## Files to Modify

### 1. `include/cortext/processor/processor_context.hpp`
Add fields to WMSlot (line ~318):
```cpp
struct WMSlot {
  // existing fields...
  std::string source_id;                             // e.g., "chat/user", "chat/assistant"
  std::vector<std::vector<unsigned char>> blob_ids;  // objstore refs
  std::string modality;                              // "text" | "audio" | "image"
};
```

### 2. `include/cortext/processor/accumulator_state.hpp`
Add blob tracking:
```cpp
struct AccumulatorState {
  // existing fields...
  std::vector<std::vector<unsigned char>> blob_ids; // objstore refs for all signals
  std::string primary_modality;                      // "text" | "audio" | "image"
};
```
Update `Reset()` to clear these fields.

### 3. `src/operations/accumulator.cpp` (or memory_storage.cpp)
When a signal is stored to objstore, track the blob_id in the accumulator:
```cpp
acc.blob_ids.push_back(blob_id);
if (acc.primary_modality.empty()) {
  acc.primary_modality = signal.modality;
}
```

### 4. `src/operations/working_memory.cpp`
**Already has per-source accumulators** (`accumulator_states` map keyed by `source_id`).

Changes needed:
- **Line ~171-186 (chunking loop)**: Add source_id match check:
  ```cpp
  if (slot.source_id != signal.source_id)
    continue;  // Only chunk into same-source slots
  ```
- **Line ~191-232 (chunking)**: Append blob_ids when chunking:
  ```cpp
  for (const auto& bid : acc.blob_ids) {
    slot.blob_ids.push_back(bid);
  }
  ```
- **Line ~328-352 (new slot creation)**: Set source_id and blob_ids:
  ```cpp
  slot.source_id = signal.source_id;
  slot.blob_ids = acc.blob_ids;
  slot.modality = acc.primary_modality;
  ```

### 5. `include/cortext/cortext.hpp`
Update `Context` struct:
```cpp
struct Context {
  std::vector<Memory> working_memory;    // NEW: Active WM slots
  std::vector<Memory> retrieved_memory;  // RENAMED from 'memories'
  bool should_interrupt = false;
  ProcessorOutput output;
};
```

### 6. `src/cortext.cpp`
Add working memory hydration:
- After signal processing, iterate `processor_context_->wm_slots`
- For each slot, load and join blobs from objstore:
  - Text: concatenate with newlines
  - Audio/Image: join as sequence (or just use first for now)
- Convert each `WMSlot` to `Memory`:
  - `content` ← joined blob content
  - `source_id` ← `slot.source_id`
  - `modality` ← `slot.modality`
  - `timestamp` ← `slot.last_ts`
  - `metrics.composite_score` ← `slot.s_avg`
  - `metrics.salience` ← `slot.s_max`
  - `metrics.arousal` ← `slot.s_arousal_avg`

### 7. `examples/chat/main.cpp`
Replace local history with working memory:
- Remove `std::vector<ChatMessage> history`
- Modify `BuildOpenAIMessages` to take `std::vector<Memory>` (working memory)
- Build conversation from `ctx.working_memory` ordered by timestamp
- Keep `ctx.retrieved_memory` for system prompt injection (when `should_interrupt`)

---

## Implementation Order

### Phase 1: Data structures
- `processor_context.hpp`: Add `source_id`, `blob_ids`, `modality` to WMSlot
- `accumulator_state.hpp`: Add `blob_ids`, `primary_modality`

### Phase 2: Blob tracking in accumulator
- `memory_storage.cpp`: When storing to objstore, add blob_id to accumulator
- `accumulator_state.hpp`: Update `Reset()` to clear blob tracking

### Phase 3: Working memory slot creation
- `working_memory.cpp`:
  - Add source_id match check in chunking loop
  - Copy blob_ids and source_id to slot on create/chunk

### Phase 4: API changes
- `cortext.hpp`: Split `memories` → `working_memory` + `retrieved_memory`
- `cortext.cpp`:
  - Hydrate `working_memory` from wm_slots (load blobs, join content)
  - Rename existing retrieval to `retrieved_memory`

### Phase 5: Chat example
- Remove local `history` vector
- Build messages from `ctx.working_memory` sorted by timestamp
- Use `ctx.retrieved_memory` for system prompt injection

### Phase 6: Tests
- Update tests using `Context::memories`
- Add working memory exposure tests

---

## Key Considerations

1. **Ordering**: WM slots need to be sorted by timestamp for conversation order
2. **Source identification**: Must track whether content is user/assistant
3. **Backward compatibility**: Existing code using `ctx.memories` needs updating
4. **No cross-source chunking**: Prevent user/assistant content from chunking together to maintain clean role separation

## Future Enhancement (Out of Scope)

Embedding source_id into the semantic space could make source similarity part of the chunking decision naturally. This would be a more elegant solution but requires architectural changes to the embedding pipeline.

---

## Test Plan

1. Run existing tests after each phase
2. Verify chat example builds and runs
3. Check telemetry logs show WM content being used for messages
4. Test conversation flow with multiple turns
