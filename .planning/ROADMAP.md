# Roadmap: cortext

## Overview

This roadmap reorients cortext around its actual next frontier: multimodal memory augmentation, not more text-only iteration. Milestone 1 focuses on the audio processing pipeline, but keeps a clean boundary: Cortext remains audio-engine agnostic while a separate speech front-end/submodule handles realtime audio control, speaker attribution, and related runtime logic before passing normalized inputs and metadata into the engine. From there, the roadmap expands into unified multimodal embeddings, cross-modal recall, and the evaluation surfaces needed to prove the engine in real use.

## Milestones

### Milestone 1: Audio Processing Pipeline
Focus the first milestone on realtime on-device audio memory augmentation, speaker-safe ingestion, a separate speech front-end boundary, and explicit state-machine-driven orchestration.

Phases in this milestone:
- Phase 1: Audio Runtime & State Machine Foundation
- Phase 2: GGML Speaker Stack
- Phase 3: Audio Memory Ingestion & Safety

### Milestone 2: Multimodal Memory Expansion
Extend the engine from audio-first work into unified multimodal embeddings, cross-modal retrieval, and evaluation surfaces.

Phases in this milestone:
- Phase 4: OmniEmbed Multimodal Integration
- Phase 5: Multimodal Evaluation & Surfaces

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Audio Front-End Scaffolding & Boundary Contract** - Scaffold `planum.cpp`, define the speech front-end boundary into Cortext, and establish the `stateforward/sml.cpp` / `co_sm` actor skeleton without treating Phase 1 as the full audio-runtime implementation phase.
- [ ] **Phase 2: GGML Speech Front-End** - Build the separate `ggml` speech front-end/submodule for speaker attribution and related realtime audio intelligence.
- [ ] **Phase 3: Audio Memory Ingestion & Safety** - Connect audio and speaker outputs to Cortext memory augmentation with safe retrieval-only and durable-write behavior.
- [ ] **Phase 4: OmniEmbed Multimodal Integration** - Integrate the low-latency multimodal embedding path for text, audio, and image memory augmentation.
- [ ] **Phase 5: Multimodal Evaluation & Surfaces** - Make multimodal quality measurable and expose multimodal memory surfaces for humans, LLMs, and paper-backed evaluation.

## Phase Details

### Phase 1: Audio Front-End Scaffolding & Boundary Contract
**Goal**: `planum.cpp` is scaffolded as the separate audio front-end, the `planum.cpp -> cortext` contract is defined, and the realtime actor skeleton is established with `stateforward/sml.cpp` / `co_sm` while Cortext stays audio-engine agnostic and continues to expose stable public APIs.
**Depends on**: Nothing (first phase)
**Primary repo**: `planum.cpp`
**Secondary repo**: `cortext`
**Boundary owned by**: Cortext ingestion contract for normalized audio + metadata
**Requirements**: AUD-00, AUD-01, AUD-02, RUN-01, RUN-02, RUN-03
**Success Criteria** (what must be TRUE):
  1. `planum.cpp` has the repo/module scaffolding for the audio front-end and actor layout required for later runtime implementation work.
  2. Cortext continues to accept normalized audio inputs and metadata without taking on speech-runtime-specific types or responsibilities.
  3. The realtime audio front-end skeleton is modeled through explicit `stateforward/sml.cpp` / `co_sm` state machines rather than ad hoc control flow.
  4. Existing public C++ and C API consumers continue to work while the new audio runtime is introduced behind the front-end boundary.
  5. Example, test, and benchmark tooling have stable scaffolding hooks for later audio/speaker implementation work.
**Plans**: 7 plans
Plans:
- [ ] `01-01-PLAN.md` — Bootstrap `planum.cpp` scaffold build/test/example/benchmark hooks
- [ ] `01-02-PLAN.md` — Add the locked `co_sm` session actor skeleton and lifecycle tests
- [ ] `01-03-PLAN.md` — Create scaffold-only `audio` and `segmentation` actor landing zones
- [ ] `01-04-PLAN.md` — Create the `signaling` actor landing zone and a benchmark lifecycle probe
- [ ] `01-05-PLAN.md` — Define the normalized `planum.cpp -> cortext` perception contract and contract tests
- [ ] `01-06-PLAN.md` — Wire the session scaffold to the synthetic contract sink plus example/benchmark hooks
- [ ] `01-07-PLAN.md` — Add the private Cortext bridge and a deterministic Cortext-side smoke hook

### Phase 2: GGML Speech Front-End
**Goal**: A separate `ggml` speech front-end/submodule can provide stable anonymous speakers and related realtime audio intelligence in a way that composes cleanly with Cortext.
**Depends on**: Phase 1
**Primary repo**: `planum.cpp`
**Secondary repo**: None
**Boundary owned by**: `planum.cpp` speech event outputs conforming to the Phase 1 contract
**Requirements**: SPK-00, SPK-01, SPK-02, SPK-03, SPK-04
**Success Criteria** (what must be TRUE):
  1. A separate speech front-end module exists in front of Cortext and can emit speaker metadata without leaking front-end implementation details into Cortext public APIs.
  2. The speech front-end can attribute room audio to stable anonymous speaker streams such as `speaker:1`, `speaker:2`, and `speaker:3` on device.
  3. The new `ggml`-based speaker stack is benchmarked against the current Sherpa/Sortformer/whisper-based baselines and demonstrates a better fit for the target on-device performance constraints.
  4. Applications can resolve anonymous speaker streams into semantic roles such as self, assistant, or other without rewriting the Cortext core pipeline.
  5. Low-confidence speaker attribution fails safely and is visible to applications and benchmarks.
**Plans**: 3 plans

### Phase 3: Audio Memory Ingestion & Safety
**Goal**: Audio and speaker outputs feed the memory engine in a way that supports live retrieval, optional retention, and safe durable writes.
**Depends on**: Phase 2
**Primary repo**: Shared (`planum.cpp` + `cortext`)
**Secondary repo**: None
**Boundary owned by**: `planum.cpp` emits confidence/retention metadata; `cortext` owns memory semantics and write policy application
**Requirements**: AUD-03, AUD-04
**Success Criteria** (what must be TRUE):
  1. Audio interactions can retrieve memory context live without necessarily storing the heard utterance durably.
  2. Durable audio-derived memory writes are gated on modality and speaker confidence rather than happening blindly.
  3. Live audio memory flows can distinguish retrieval-only behavior from durable retention behavior in examples and benchmarks.
**Plans**: 2 plans

### Phase 4: OmniEmbed Multimodal Integration
**Goal**: The engine can use the low-latency multimodal embedding path across text, audio, and image without splitting multimodal memory into a side system.
**Depends on**: Phase 3
**Primary repo**: `cortext`
**Secondary repo**: `planum.cpp` only if speech-side adapter changes are required
**Boundary owned by**: Cortext multimodal ingestion/retrieval semantics
**Requirements**: MM-01, MM-02, MM-03, MM-04
**Success Criteria** (what must be TRUE):
  1. A low-latency multimodal embedding path is integrated for text, audio, and image inputs.
  2. Multimodal embeddings plug into current Cortext ingestion and retrieval flows instead of bypassing them.
  3. Cross-modal recall can surface relevant memory across text, audio, and image interactions.
  4. Multimodal behavior continues to respect the engine’s biologically inspired memory controls and semantics.
**Plans**: 3 plans

### Phase 5: Multimodal Evaluation & Surfaces
**Goal**: Developers can measure multimodal engine quality and expose multimodal memory augmentation cleanly to humans, LLMs, and the paper/evaluation workflow.
**Depends on**: Phase 4
**Primary repo**: Shared (`cortext` + `planum.cpp`)
**Secondary repo**: None
**Boundary owned by**: Shared benchmark and integration surfaces, with Cortext owning memory outcomes and `planum.cpp` owning speech front-end measurements
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Developers can run repeatable benchmarks for audio latency, speaker attribution quality, and multimodal retrieval behavior.
  2. Human-facing and LLM-facing integrations can inspect and surface multimodal memory context during live interaction.
  3. Multimodal engine changes and experiment results are reflected in the paper sections and generated manuscript.
**Plans**: 2 plans

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Audio Front-End Scaffolding & Boundary Contract | 4/7 | In progress | 01, 02, 03, 05 |
| 2. GGML Speech Front-End | 0/3 | Not started | - |
| 3. Audio Memory Ingestion & Safety | 0/2 | Not started | - |
| 4. OmniEmbed Multimodal Integration | 0/3 | Not started | - |
| 5. Multimodal Evaluation & Surfaces | 0/2 | Not started | - |
