# Roadmap: cortext

## Overview

This roadmap hardens cortext as the local-first memory engine behind augmem.ai by first separating interactive and background work, then making retrieval explainable, then adding scoped lifecycle correctness, then landing speaker-safe realtime voice, and finally closing the loop with repeatable evaluation and integration surfaces. Advanced memory surfaces remain a v2 direction until these v1 correctness and latency foundations are stable.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Two-Speed Core Runtime** - Split low-latency online processing from durable background maintenance without breaking public APIs.
- [ ] **Phase 2: Explainable Hybrid Retrieval** - Improve recall quality and expose provenance, score breakdowns, and evidence-preserving retrieval surfaces.
- [ ] **Phase 3: Scoped Lifecycle & Temporal Correctness** - Add namespace, correction, supersession, and historical-trace behavior for durable memory.
- [ ] **Phase 4: Realtime Voice & Speaker-Safe Ingestion** - Ship low-latency on-device voice flows that avoid unsafe durable writes.
- [ ] **Phase 5: Evaluation & Integration Surfaces** - Make quality measurable, integrations inspectable, and research artifacts match shipped behavior.

## Phase Details

### Phase 1: Two-Speed Core Runtime
**Goal**: Applications can use the stable public APIs while the engine separates low-latency ingestion/retrieval from durable background maintenance.
**Depends on**: Nothing (first phase)
**Requirements**: CORE-01, CORE-02, CORE-03
**Success Criteria** (what must be TRUE):
  1. Existing C++ and C API callers can process signals successfully through the refactored engine without changing their integration code.
  2. Interactive processing returns without waiting for consolidation, extraction, graph refresh, or maintenance work to finish inline.
  3. Background maintenance can run asynchronously against persisted state and its results become visible to later retrievals.
**Plans**: 3 plans

### Phase 2: Explainable Hybrid Retrieval
**Goal**: Applications can recall the right memories with hybrid ranking and understand exactly why they surfaced.
**Depends on**: Phase 1
**Requirements**: RET-01, RET-02, RET-03, RET-04
**Success Criteria** (what must be TRUE):
  1. Retrieval returns useful memories using hybrid signals instead of relying on dense semantic similarity alone.
  2. Applications can inspect each surfaced memory with provenance and score breakdowns that explain its ranking.
  3. Applications can request different hydration budgets and filters for human and LLM workflows and receive bounded results that honor those constraints.
  4. Summaries, labels, and fact-derived memories can be traced back to preserved source evidence.
**Plans**: 3 plans

### Phase 3: Scoped Lifecycle & Temporal Correctness
**Goal**: Applications can keep memories scoped, correct them explicitly, and trust newer truths to outrank stale ones without destroying history.
**Depends on**: Phase 2
**Requirements**: LIFE-01, LIFE-02, LIFE-03, LIFE-04
**Success Criteria** (what must be TRUE):
  1. Applications can write and query memories by namespace, session, or owner without unrelated memory leaking across scopes.
  2. Applications can correct, retract, invalidate, or delete stored memories through explicit lifecycle operations.
  3. Historical evidence remains available by default even when memories or facts are superseded or invalidated.
  4. When facts conflict, newer valid facts surface ahead of stale ones without corrupting the prior record.
**Plans**: 3 plans

### Phase 4: Realtime Voice & Speaker-Safe Ingestion
**Goal**: Applications can run live voice memory flows with on-device streaming, safe non-durable retrieval, and confidence-gated durable writes.
**Depends on**: Phase 3
**Requirements**: VOICE-01, VOICE-02, VOICE-03, VOICE-04
**Success Criteria** (what must be TRUE):
  1. Realtime voice ingress produces low-latency partial and final transcriptions with endpointing on device.
  2. Voice interactions can retrieve against the current spoken context without durably retaining the heard utterance.
  3. Durable voice-derived memories are only written when speaker attribution is confident enough to avoid unrelated room speech polluting memory.
  4. Assistant voice playback supports interruption and barge-in behavior suitable for live conversation.
**Plans**: 3 plans

### Phase 5: Evaluation & Integration Surfaces
**Goal**: Developers can verify engine quality, integrate cortext cleanly into human and LLM workflows, and keep research artifacts aligned with shipped behavior.
**Depends on**: Phase 4
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Developers can run repeatable benchmarks for retrieval quality, latency, and speech-pipeline performance and compare runs reliably.
  2. Examples and integrations can inject memory, inspect retrieval context, and surface engine outputs for both human-facing and LLM-facing workflows.
  3. Algorithm changes and experiment results from shipped phases are reflected in paper sections and the generated manuscript.
**Plans**: 2 plans

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Two-Speed Core Runtime | 0/3 | Not started | - |
| 2. Explainable Hybrid Retrieval | 0/3 | Not started | - |
| 3. Scoped Lifecycle & Temporal Correctness | 0/3 | Not started | - |
| 4. Realtime Voice & Speaker-Safe Ingestion | 0/3 | Not started | - |
| 5. Evaluation & Integration Surfaces | 0/2 | Not started | - |
