# Phase 1: Audio Front-End Scaffolding & Boundary Contract - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 scaffolds `planum.cpp` as the separate audio front-end that sits in front of Cortext. The scope is the repo/module structure, actor layout, `stateforward/sml.cpp` / `co_sm` orchestration skeleton, and the boundary contract from `planum.cpp` into Cortext. This phase is intentionally scaffolding-first, not the phase that fully implements segmentation, diarization, or STT behavior.

</domain>

<decisions>
## Implementation Decisions

### Repo Boundary
- **D-01:** `planum.cpp` is the primary repo for Phase 1 work and lives as a separate repo/submodule in front of Cortext.
- **D-02:** Cortext remains audio-engine agnostic. Speech-engine-specific implementation details must not leak into Cortext public APIs.
- **D-03:** `ProcessAudio(...)` stays speaker-agnostic; speaker/runtime intelligence belongs in `planum.cpp` before ingestion into Cortext.

### Front-End Contract
- **D-04:** `planum.cpp` is strictly an auditory/language front-end: diarization, segmentation, STT, audio/runtime state, and confidence about its own outputs.
- **D-05:** `planum.cpp` must not own retention policy, memory write eligibility, retrieve-vs-retain behavior, or Cortext memory semantics.
- **D-06:** The `planum.cpp -> cortext` contract should carry perception events such as normalized audio segments/chunks, timestamps, segment/turn identifiers, speaker ids or speaker confidence, transcript text and transcript confidence, runtime/endpoint state, and degraded/error status.
- **D-07:** Cortext decides retrieve/retain/write behavior and all memory semantics downstream of the front-end contract.

### Runtime Orchestration
- **D-08:** New realtime orchestration in `planum.cpp` uses `stateforward/sml.cpp`.
- **D-09:** The main Phase 1 actor uses `co_sm`; this is still RTC in the `stateforward/sml.cpp` fork and must not introduce mailbox/queue semantics.
- **D-10:** The main actor state set is locked to: `inactive`, `activating`, `listening`, `segmenting`, `endpointing`, `signaling`, `degraded`, `errored`.

### Actor / Source Layout
- **D-11:** Actor layout follows the folder-based, namespaced pattern used in `../stateforward/emel/emel.cpp/src`.
- **D-12:** Phase 1 should scaffold actor folders/files rather than fully implement the machines.
- **D-13:** Expected actor decomposition is around audio runtime, segmentation, signaling, and top-level orchestration, with speaker-specific machines deferred to a later phase.

### the agent's Discretion
- Exact naming of scaffold-only headers/source files inside the locked actor/folder pattern
- Whether the boundary types live under a dedicated `contract/`, `events/`, or `api/` area inside `planum.cpp`
- How much placeholder logic to include in scaffold examples, as long as it does not turn the phase into full implementation

</decisions>

<specifics>
## Specific Ideas

- `planum.cpp` is the new repo name and is currently a private GitHub repo/submodule.
- The broader product direction is multimodal memory augmentation, but this phase is about audio-front-end scaffolding only.
- Future repos may follow the same naming pattern, e.g. `planum.go`, `planum.py`, `cortext.cpp`.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning
- `.planning/PROJECT.md` — Project framing, architecture direction, and active requirements for the audio-first milestone
- `.planning/REQUIREMENTS.md` — Phase 1 requirement ids (`AUD-00`, `AUD-01`, `AUD-02`, `RUN-01`, `RUN-02`, `RUN-03`)
- `.planning/ROADMAP.md` — Phase ownership and milestone sequencing
- `.planning/STATE.md` — Current focus and recent architectural decisions

### SML / actor rules
- `AGENTS.md` — Repository-wide rules plus the binding `SML / stateforward Rules` section
- `docs/rules/sml.rules.md` — Local copy of the canonical SML rules to follow in Cortext
- `third_party/planum.cpp/docs/sml.rules.md` — Matching SML rules copied into `planum.cpp`
- `../stateforward/emel/emel.cpp/AGENTS.md` — Upstream actor/modeling conventions informing the local rules
- `../stateforward/emel/emel.cpp/docs/rules/sml.rules.md` — Canonical SML constraints source

### Existing code / integration
- `include/cortext/cortext.hpp` — Current Cortext public ingestion surface, including `ProcessAudio(...)`
- `examples/chat/voice_session.hpp` — Existing example-level voice callback/event shape that can inform the boundary contract

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `examples/chat/voice_session.hpp`: already models partial/final transcript callbacks, speaker ids, and runtime-facing events that can inform the `planum.cpp` contract.

### Established Patterns
- `../stateforward/emel/emel.cpp/src`: folder-based and namespaced actor layout with per-actor `context.hpp`, `events.hpp`, `guards.hpp`, `actions.hpp`, and `sm.hpp`.
- `include/cortext/cortext.hpp`: Cortext public APIs stay stable and should not absorb speech-engine-specific types.

### Integration Points
- `planum.cpp` will sit in front of Cortext and hand normalized perception events into the Cortext ingestion layer.
- Phase 1 must define the boundary cleanly enough that later speaker/runtime implementation can happen in `planum.cpp` without pushing that complexity into Cortext.

</code_context>

<deferred>
## Deferred Ideas

- Full segmentation/endpointing implementation details
- The custom `ggml` speaker stack
- Detailed diarization behavior
- Full STT backend implementation
- Cortext-side memory policy changes driven by audio metadata

</deferred>

---

*Phase: 01-audio-front-end-boundary-runtime-foundation*
*Context gathered: 2026-04-07*
