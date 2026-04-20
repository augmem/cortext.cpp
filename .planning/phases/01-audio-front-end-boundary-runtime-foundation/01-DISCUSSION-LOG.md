# Phase 1 Discussion Log

**Date:** 2026-04-07
**Phase:** 1
**Phase Name:** Audio Front-End Scaffolding & Boundary Contract

## Decisions Captured

1. Cortext remains audio-engine agnostic.
2. `planum.cpp` is the separate speech/audio front-end repo/submodule that sits in front of Cortext.
3. `planum.cpp` is strictly responsible for auditory/language processing concerns such as diarization, segmentation, STT, audio/runtime state, and its own confidence signals.
4. `planum.cpp` must not own retention or memory semantics.
5. The boundary into Cortext carries perception events and metadata, not memory policy.
6. `stateforward/sml.cpp` rules copied from `emel.cpp` are binding for this work.
7. The main `planum.cpp` actor uses `co_sm`, which is still RTC in the local `stateforward/sml.cpp` fork.
8. Locked main actor states: `inactive`, `activating`, `listening`, `segmenting`, `endpointing`, `signaling`, `degraded`, `errored`.
9. Actor organization should follow the folder-based, namespaced `emel.cpp/src` pattern.
10. Phase 1 is scaffolding-first, not the phase that fully implements the audio runtime.

## Explicit Rejections

- Rejected pushing retention metadata into `planum.cpp`.
- Rejected using `idle` as the top-level inactive state name.
- Rejected making Phase 1 the full runtime implementation phase.

## Open Discretion Left for Planning

- Exact folder names for contract/event types
- Exact scaffold artifact set in `planum.cpp`
- How much placeholder/example code is useful without drifting into implementation

---

*Captured from discuss-phase conversation on 2026-04-07*
