# Requirements: cortext

**Defined:** 2026-04-07
**Core Value:** Relevant memory should be formed and resurfaced on device, in realtime, and in a way that feels biologically plausible rather than manually scripted.

## v1 Requirements

### Audio Pipeline

- [ ] **AUD-00**: Cortext remains audio-engine agnostic and accepts normalized audio inputs plus metadata rather than speech-runtime-specific implementation types
- [x] **AUD-01**: Engine can ingest live audio through a low-latency on-device pipeline suitable for realtime memory augmentation
- [ ] **AUD-02**: Audio pipeline supports partials, endpointing, and finalized utterance boundaries without requiring cloud services
- [ ] **AUD-03**: Audio-derived memory can retrieve against current context without durably retaining the heard utterance
- [ ] **AUD-04**: Audio pipeline can safely emit durable writes only when modality- and speaker-confidence thresholds are satisfied

### Speaker Attribution

- [ ] **SPK-00**: A separate speech front-end module can sit in front of Cortext and provide speaker metadata without leaking front-end implementation details into Cortext public APIs
- [ ] **SPK-01**: Engine can assign stable anonymous speaker identities for room audio on device
- [ ] **SPK-02**: Speaker attribution is fast enough for live interaction and materially better suited to the target device than the current Sherpa diarization path
- [ ] **SPK-03**: Applications can tag or resolve anonymous speaker streams into semantic roles such as self, assistant, or other
- [ ] **SPK-04**: Speaker-attribution failures degrade safely rather than silently polluting durable memory

### Runtime & Orchestration

- [x] **RUN-01**: New multimodal realtime features use `stateforward/sml.cpp` state machines for orchestration
- [ ] **RUN-02**: New audio/speaker components integrate into the existing engine without breaking the public C++ or C APIs
- [ ] **RUN-03**: Realtime modality features expose explicit lifecycle states that can be inspected and debugged in examples and benchmarks

### Multimodal Expansion

- [ ] **MM-01**: Engine can use a low-latency multimodal embedding path for text, audio, and image memory augmentation
- [ ] **MM-02**: Multimodal embeddings integrate with current Cortext ingestion and retrieval flows rather than living as a separate side system
- [ ] **MM-03**: Cross-modal recall can surface relevant memory across text, audio, and image inputs
- [ ] **MM-04**: Multimodal memory behavior continues to respect the engine’s biologically inspired controls and memory semantics

### Evaluation & Surfaces

- [ ] **EVAL-01**: Engine exposes repeatable benchmarks for audio latency, speaker attribution quality, and multimodal retrieval behavior
- [ ] **EVAL-02**: Human-facing and LLM-facing integrations can inspect and surface multimodal memory context during live interaction
- [ ] **EVAL-03**: New modality work is reflected in paper sections and generated manuscript updates

## v2 Requirements

### Advanced Memory Surfaces

- **ADV-01**: Engine can apply richer privacy and context gating to multimodal memory surfacing
- **ADV-02**: Engine can support more advanced graph expansion and adaptive write behavior across modalities
- **ADV-03**: Engine can optimize more of the speech stack under a unified `ggml`-native runtime beyond the initial speaker stack

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cloud-first speech or memory processing | The engine direction is explicitly on-device and local-first |
| Perfect arbitrary-room diarization in the first milestone | The first milestone should land a strong practical speaker stack, not solve every edge case |
| App-level augmem.ai UI work unrelated to engine behavior | This roadmap is for the engine, not the full product shell |
| Breaking public APIs as part of modality expansion | Existing integrations must keep working |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUD-00 | Phase 1 | Pending |
| AUD-01 | Phase 1 | Complete |
| AUD-02 | Phase 1 | Pending |
| AUD-03 | Phase 3 | Pending |
| AUD-04 | Phase 3 | Pending |
| SPK-00 | Phase 2 | Pending |
| SPK-01 | Phase 2 | Pending |
| SPK-02 | Phase 2 | Pending |
| SPK-03 | Phase 2 | Pending |
| SPK-04 | Phase 2 | Pending |
| RUN-01 | Phase 1 | Complete |
| RUN-02 | Phase 1 | Pending |
| RUN-03 | Phase 1 | Pending |
| MM-01 | Phase 4 | Pending |
| MM-02 | Phase 4 | Pending |
| MM-03 | Phase 4 | Pending |
| MM-04 | Phase 4 | Pending |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0

---
*Requirements defined: 2026-04-07*
*Last updated: 2026-04-07 after audio-front-end boundary refinement*
