# cortext

## What This Is

cortext is a biologically inspired memory augmentation engine powering augmem.ai. It is a high-performance, realtime, on-device system that helps humans and LLMs form, retain, and resurface context across text, audio, and image modalities.

## Core Value

Relevant memory should be formed and resurfaced on device, in realtime, and in a way that feels biologically plausible rather than manually scripted.

## Requirements

### Validated

- ✓ Process text signals into durable memory with retrieval, consolidation, and working-memory behavior — existing
- ✓ Run the core memory engine locally with SQLite-backed persistence and F/S/T-driven behavior — existing
- ✓ Support multimodal engine surfaces and local model backends across examples, benchmarks, and integrations — existing
- ✓ Provide experimental and paper-backed evaluation flows for text-centric memory behavior — existing

### Active

- [ ] Define a clean front-end boundary so speech/audio runtime logic lives outside Cortext core
- [ ] Build a high-performance audio processing pipeline as the first multimodal milestone
- [ ] Add a custom on-device `ggml` speaker/diarization stack in a separate speech front-end submodule suitable for realtime room audio
- [ ] Use `stateforward/sml.cpp` to orchestrate new realtime modality features with explicit state machines
- [ ] Integrate low-latency multimodal embeddings so text, audio, and image memory can share a unified augmentation path
- [ ] Expand memory augmentation quality and observability beyond text-first workflows

### Out of Scope

- Managed cloud memory service as the primary architecture — cortext remains local-first and engine-first
- Generic augmem.ai product/UI work that does not improve the engine itself — that belongs above the engine boundary
- Breaking the public C++ headers or C API without explicit approval — existing consumers depend on them

## Context

- cortext is already strong on the text-memory side. The next milestone is not proving text memory again; it is moving the memory engine into the other supported modalities.
- The repository already contains local model runtimes, voice experiments, multimodal ingestion paths, benchmarks, and paper infrastructure. The bottleneck is now modality expansion and realtime on-device behavior, especially for audio.
- cortext should remain audio-engine agnostic. Speaker attribution, diarization, VAD, endpointing, and realtime audio control should sit in a front-end layer before Cortext rather than inside its public API surface.
- The likely shape is a separate speech front-end submodule that produces normalized audio events, speaker metadata, confidence, and retention policy inputs for Cortext.
- New feature work should use `stateforward/sml.cpp` for explicit state-machine-driven orchestration rather than ad hoc control flow.
- In the `stateforward/sml.cpp` actor model, transient orchestration and handoff data should move through explicit events / `sml::completion<TEvent>` rather than being mirrored into actor context.
- Future `planum.cpp` naming should be domain-first: prefer `planum::processor`, `planum::audio`, `planum::segmentation`, and `planum::signaling` over `session` terminology or a redundant `runtime` namespace.
- The desired direction is a cohesive on-device stack: biologically inspired memory behavior, low-latency multimodal inference, and safe realtime augmentation for both humans and LLMs.

## Constraints

- **Tech stack**: C++20, CMake, SQLite, and local model runtimes remain the core substrate — the engine should evolve from the existing codebase, not restart elsewhere
- **Realtime performance**: New modality work must target low-latency on-device execution, especially in the audio path
- **Engine boundary**: Cortext should consume normalized modality inputs and metadata, not own speech-runtime-specific implementation details
- **State orchestration**: New realtime feature flows should leverage `stateforward/sml.cpp` for explicit lifecycle/state transitions
- **API stability**: Public headers in `include/` and the C API require explicit approval before breaking changes
- **Research traceability**: Algorithm and experiment changes must continue to update `docs/paper/sections/` and the generated manuscript

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat cortext as a biologically inspired memory augmentation engine, not just a text-memory library | The product scope is broader than text and should cover human and LLM memory across modalities | — Pending |
| Make the first milestone audio-first | Text experiments are already mature enough that the highest-value next step is the audio pipeline | — Pending |
| Keep Cortext audio-engine agnostic | Speech runtime details should sit in front of the memory engine, not inside its public boundary | — Pending |
| Build the custom speaker/diarization stack around `ggml` in a separate speech front-end submodule | Current local diarization options are too slow or fragmented for the desired on-device realtime path, and the speech stack should be replaceable without reshaping Cortext | — Pending |
| Use `stateforward/sml.cpp` for new feature orchestration | Realtime multimodal flows need explicit and debuggable state transitions | — Pending |
| Use domain-first names in `planum.cpp` (`processor`, `audio`, `segmentation`, `signaling`) | `session` is product-ambiguous and `runtime` is redundant under the `planum` namespace | — Pending |
| Preserve F/S/T as the main behavioral control surface | The engine remains biologically inspired and those knobs continue to define behavior | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -> still the right priority?
3. Audit Out of Scope -> reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-07 after audio-front-end boundary refinement*
