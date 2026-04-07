# Requirements: cortext

**Defined:** 2026-04-07
**Core Value:** Important context should resurface at the right time for humans and models without requiring manual memory management.

## v1 Requirements

### Core Processing

- [ ] **CORE-01**: Engine can process user signals through a low-latency online path without blocking on consolidation or long-running background work
- [ ] **CORE-02**: Engine can run consolidation, extraction, graph refresh, and maintenance asynchronously against persisted memory state
- [ ] **CORE-03**: Existing public C++ and C APIs continue to work through internal runtime and pipeline refactors

### Retrieval & Evidence

- [ ] **RET-01**: Retrieval uses hybrid ranking signals so memory recall is not limited to dense semantic similarity alone
- [ ] **RET-02**: Applications can inspect retrieved memories with provenance and score breakdowns sufficient to explain why they surfaced
- [ ] **RET-03**: Retrieval supports configurable hydration budgets and filters for different human and LLM workflows
- [ ] **RET-04**: Engine preserves source evidence needed to justify summaries, labels, and fact-derived memories

### Memory Lifecycle

- [ ] **LIFE-01**: Applications can scope memories by namespace, session, or owner to prevent unrelated memory pollution
- [ ] **LIFE-02**: Engine can supersede or invalidate memories and facts without destructively deleting historical evidence by default
- [ ] **LIFE-03**: Applications can perform explicit lifecycle operations such as correction, retraction, and deletion on stored memories
- [ ] **LIFE-04**: Temporal conflicts are handled so newer facts can override stale ones without corrupting historical context

### Voice & Speaker Safety

- [ ] **VOICE-01**: Realtime voice ingress supports low-latency transcription with partials, endpointing, and on-device execution
- [ ] **VOICE-02**: Voice ingress can retrieve against current context without durably retaining the heard utterance
- [ ] **VOICE-03**: Voice ingestion uses speaker-safe attribution or confidence gating so unrelated room speech does not silently pollute durable memory
- [ ] **VOICE-04**: Realtime assistant voice supports reply playback and interruption/barge-in behavior suitable for live conversation

### Evaluation & Integrations

- [ ] **EVAL-01**: Engine exposes repeatable benchmarks for retrieval quality, latency, and speech pipeline performance
- [ ] **EVAL-02**: Examples and integrations can inject, inspect, and surface retrieved memory context for both human-facing and LLM-facing workflows
- [ ] **EVAL-03**: Algorithm and experiment changes are reflected in paper sections and the generated manuscript

## v2 Requirements

### Advanced Memory Surfaces

- **ADV-01**: Engine can use privacy/context policies to gate which memories may surface in a given interaction
- **ADV-02**: Engine can augment recall through richer graph expansion and cross-modal memory surfaces beyond the current baseline
- **ADV-03**: Engine can adapt write behavior and memory budgets dynamically based on workload and interaction mode

## Out of Scope

| Feature | Reason |
|---------|--------|
| Managed cloud memory backend as primary architecture | cortext is currently local-first and engine-first |
| Generic augmem.ai application UI work unrelated to engine behavior | This roadmap is for the engine powering the product, not the full app surface |
| Breaking public API redesign without explicit approval | Existing bindings and consumers depend on current public surfaces |
| Perfect overlapping-speaker diarization across arbitrary room audio in v1 | Too much scope and risk for the current roadmap |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 1 | Pending |
| CORE-02 | Phase 1 | Pending |
| CORE-03 | Phase 1 | Pending |
| RET-01 | Phase 2 | Pending |
| RET-02 | Phase 2 | Pending |
| RET-03 | Phase 2 | Pending |
| RET-04 | Phase 2 | Pending |
| LIFE-01 | Phase 3 | Pending |
| LIFE-02 | Phase 3 | Pending |
| LIFE-03 | Phase 3 | Pending |
| LIFE-04 | Phase 3 | Pending |
| VOICE-01 | Phase 4 | Pending |
| VOICE-02 | Phase 4 | Pending |
| VOICE-03 | Phase 4 | Pending |
| VOICE-04 | Phase 4 | Pending |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-07*
*Last updated: 2026-04-07 after initialization*
