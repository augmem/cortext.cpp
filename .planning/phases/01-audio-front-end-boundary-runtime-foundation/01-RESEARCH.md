# Phase 1: Audio Front-End Scaffolding & Boundary Contract - Research

**Researched:** 2026-04-07
**Domain:** `planum.cpp` scaffolding, `stateforward/sml.cpp` actor organization, and the `planum.cpp -> cortext` boundary for normalized audio perception events
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `planum.cpp` is the primary repo for Phase 1 work and lives as a separate repo/submodule in front of Cortext.
- **D-02:** Cortext remains audio-engine agnostic. Speech-engine-specific implementation details must not leak into Cortext public APIs.
- **D-03:** `ProcessAudio(...)` stays speaker-agnostic; speaker/runtime intelligence belongs in `planum.cpp` before ingestion into Cortext.
- **D-04:** `planum.cpp` is strictly an auditory/language front-end: diarization, segmentation, STT, audio/runtime state, and confidence about its own outputs.
- **D-05:** `planum.cpp` must not own retention policy, memory write eligibility, retrieve-vs-retain behavior, or Cortext memory semantics.
- **D-06:** The `planum.cpp -> cortext` contract should carry perception events such as normalized audio segments/chunks, timestamps, segment/turn identifiers, speaker ids or speaker confidence, transcript text and transcript confidence, runtime/endpoint state, and degraded/error status.
- **D-07:** Cortext decides retrieve/retain/write behavior and all memory semantics downstream of the front-end contract.
- **D-08:** New realtime orchestration in `planum.cpp` uses `stateforward/sml.cpp`.
- **D-09:** The main Phase 1 actor uses `co_sm`; this is still RTC in the `stateforward/sml.cpp` fork and must not introduce mailbox/queue semantics.
- **D-10:** The main actor state set is locked to: `inactive`, `activating`, `listening`, `segmenting`, `endpointing`, `signaling`, `degraded`, `errored`.
- **D-11:** Actor layout follows the folder-based, namespaced pattern used in `../stateforward/emel/emel.cpp/src`.
- **D-12:** Phase 1 should scaffold actor folders/files rather than fully implement the machines.
- **D-13:** Expected actor decomposition is around audio runtime, segmentation, signaling, and top-level orchestration, with speaker-specific machines deferred to a later phase.

### Claude's Discretion
- Exact naming of scaffold-only headers/source files inside the locked actor/folder pattern
- Whether the boundary types live under a dedicated `contract/`, `events/`, or `api/` area inside `planum.cpp`
- How much placeholder logic to include in scaffold examples, as long as it does not turn the phase into full implementation

### Deferred Ideas (OUT OF SCOPE)
- Full segmentation/endpointing implementation details
- The custom `ggml` speaker stack
- Detailed diarization behavior
- Full STT backend implementation
- Cortext-side memory policy changes driven by audio metadata
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AUD-00 | Cortext remains audio-engine agnostic and accepts normalized audio inputs plus metadata rather than speech-runtime-specific implementation types | Use a contract-first `planum::contract::perception_event` boundary and keep Cortext changes in a private adapter instead of `include/` or the C API. [VERIFIED: local codebase grep] |
| AUD-01 | Engine can ingest live audio through a low-latency on-device pipeline suitable for realtime memory augmentation | Phase 1 should only scaffold the actor tree and handoff path; low-latency runtime work is deferred, but the top-level `co_sm` scaffold and child actors must exist now so later audio work lands in the front-end repo instead of Cortext. [VERIFIED: local codebase grep][CITED: https://boost-ext.github.io/sml/user_guide.html] |
| AUD-02 | Audio pipeline supports partials, endpointing, and finalized utterance boundaries without requiring cloud services | The contract should explicitly represent partial transcript, finalized segment, endpoint, degraded, and error events even though the algorithms behind them are not implemented in Phase 1. [VERIFIED: local codebase grep] |
| RUN-01 | New multimodal realtime features use `stateforward/sml.cpp` state machines for orchestration | The scaffold should use `co_sm` plus folder-based `context.hpp`, `events.hpp`, `guards.hpp`, `actions.hpp`, and `sm.hpp` components modeled after `emel.cpp`. [VERIFIED: local codebase grep] |
| RUN-02 | New audio/speaker components integrate into the existing engine without breaking the public C++ or C APIs | Phase 1 should route finalized transcript events through existing `Cortext::ProcessTextAt(...)` in a private bridge and leave `include/cortext/cortext.hpp` and `include/cortext/capi.h` unchanged. [VERIFIED: local codebase grep] |
| RUN-03 | Realtime modality features expose explicit lifecycle states that can be inspected and debugged in examples and benchmarks | The main actor should expose the locked state set and support inspection through `is(...)` / `visit_current_states(...)`, with example or test hooks proving the state skeleton before audio algorithms exist. [VERIFIED: local codebase grep][CITED: https://boost-ext.github.io/sml/user_guide.html] |
</phase_requirements>

## Summary

Phase 1 should establish `planum.cpp` as a contract-first front-end scaffold, not a partial speech runtime. The evidence is consistent across the phase context, the roadmap, the current `planum.cpp` submodule stub, and the copied SML rules: the repo already exists only as a placeholder, the main actor states are locked, the actor organization is expected to mirror `emel.cpp`, and the binding runtime choice is `co_sm` with RTC semantics preserved in the local fork. [VERIFIED: local codebase grep][CITED: https://boost-ext.github.io/sml/user_guide.html]

The cleanest Phase 1 boundary is a small, explicit `perception_event` contract in `planum.cpp` that carries transcript text, timing, segment identity, speaker metadata, runtime state, and degraded/error metadata, but never retention or write-policy decisions. That recommendation follows directly from the locked decisions and from the current Cortext code: `ProcessTextAt(...)` already accepts explicit timestamps, while the existing `ProcessAudio(...)` path takes raw PCM, generates timestamps internally, and is therefore the wrong Phase 1 seam for segmented speaker-aware front-end output. [VERIFIED: local codebase grep]

The minimum Cortext work in this phase is a private bridge, not a public API change. That bridge should translate finalized transcript-bearing `perception_event`s into existing Cortext ingestion calls, ignore or surface partial/state-only events without memory writes, and keep all retrieve/retain/write policy downstream in Cortext or the application layer. [VERIFIED: local codebase grep]

**Primary recommendation:** Scaffold `planum.cpp` around one `co_sm` top-level session actor, three stub child actors, and a dedicated `contract/` area, then add only a private Cortext adapter that consumes finalized transcript events through existing timestamped text ingestion. [VERIFIED: local codebase grep]

## Standard Stack

### Core
| Library / Component | Version / Pin | Purpose | Why Standard |
|---------|---------|---------|--------------|
| C++20 [VERIFIED: local codebase grep] | required by Cortext and by `co_sm` usage in the local `sml.cpp` fork [VERIFIED: local codebase grep] | Language level for `planum.cpp` scaffold and actor code [VERIFIED: local codebase grep] | Cortext already builds as C++20, and `co_sm` is explicitly documented as the C++20 utility path. [VERIFIED: local codebase grep] |
| CMake [VERIFIED: local codebase grep] | Cortext floor `3.16`; `emel.cpp` floor `3.20` [VERIFIED: local codebase grep] | Build/test scaffold for the new front-end repo [VERIFIED: local codebase grep] | Using `3.20+` in `planum.cpp` keeps the scaffold aligned with the `emel.cpp` reference repo rather than dragging in a lower baseline just for Phase 1. [VERIFIED: local codebase grep] |
| `stateforward/sml.cpp` `co_sm` [VERIFIED: local codebase grep] | local fork, repo-pinned in sibling checkout [VERIFIED: local codebase grep] | Top-level orchestration wrapper for the Phase 1 session actor [VERIFIED: local codebase grep] | The fork exposes synchronous `process_event(...)`, an async path, and scheduler contracts that require FIFO ordering, single-consumer dispatch, and run-to-completion. [VERIFIED: local codebase grep] |
| `emel.cpp` folder-based actor pattern [VERIFIED: local codebase grep] | local reference repo [VERIFIED: local codebase grep] | File layout, namespaces, and transition-table style [VERIFIED: local codebase grep] | The phase context explicitly locks this pattern, and `emel.cpp` already demonstrates the expected `context/events/guards/actions/sm` component split. [VERIFIED: local codebase grep] |

### Supporting
| Library / Component | Version / Pin | Purpose | When to Use |
|---------|---------|---------|-------------|
| `Cortext::ProcessTextAt(...)` [VERIFIED: local codebase grep] | existing public API [VERIFIED: local codebase grep] | Minimum Phase 1 ingest path for finalized transcript events [VERIFIED: local codebase grep] | Use for transcript-bearing finalized segments because it already accepts explicit timestamps and does not require new public API surface. [VERIFIED: local codebase grep] |
| `Cortext::ProcessAudio(...)` [VERIFIED: local codebase grep] | existing public API [VERIFIED: local codebase grep] | Legacy/raw PCM ingest path [VERIFIED: local codebase grep] | Keep available, but do not make it the new front-end boundary because it works on PCM plus generated timestamps rather than normalized segmented perception events. [VERIFIED: local codebase grep] |
| `examples/chat/voice_session.hpp` callback shapes [VERIFIED: local codebase grep] | current example code [VERIFIED: local codebase grep] | Source material for event taxonomy only [VERIFIED: local codebase grep] | Use its partial/final/speaker ideas as input, but do not let `std::function` callbacks become the canonical repo-to-repo contract. [VERIFIED: local codebase grep] |
| Catch2 v3.5.3 in Cortext [VERIFIED: local codebase grep] | fetched in `tests/CMakeLists.txt` [VERIFIED: local codebase grep] | Cortext-side bridge and API-regression validation [VERIFIED: local codebase grep] | Existing Cortext tests can validate “no public API break” immediately while `planum.cpp` adds its own scaffold tests. [VERIFIED: local codebase grep] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Dedicated `contract/perception_event.hpp` | Example-local callback wiring | Faster to sketch, but it would freeze an example callback API instead of a clean front-end boundary. [VERIFIED: local codebase grep] |
| Private Cortext bridge under non-public source paths | New speech-aware Cortext public API types | That would violate the locked agnostic boundary and risk breaking headers/bindings. [VERIFIED: local codebase grep] |
| `co_sm` top-level actor | Ad hoc threaded controller or mailbox | The copied SML rules explicitly prohibit queues/mailboxes and require RTC deterministic dispatch. [VERIFIED: local codebase grep][CITED: https://boost-ext.github.io/sml/user_guide.html] |

**Bootstrap:**
```bash
git submodule update --init --recursive third_party/planum.cpp
cmake -S third_party/planum.cpp -B third_party/planum.cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build third_party/planum.cpp/build -j
```

**Version verification:** No new third-party package recommendation is required for Phase 1; the work is based on existing repo-pinned submodules, the local `stateforward` fork, and Cortext’s current public/API surface. [VERIFIED: local codebase grep]

## Architecture Patterns

### Recommended Project Structure
```text
third_party/planum.cpp/
├── CMakeLists.txt
├── README.md
├── docs/
│   └── sml.rules.md
├── include/
│   └── planum/
│       ├── contract/
│       │   ├── ids.hpp
│       │   ├── perception_event.hpp
│       │   ├── runtime_state.hpp
│       │   └── sink.hpp
│       └── front_end.hpp
├── src/
│   └── planum/
│       ├── runtime/
│       │   ├── session/
│       │   │   ├── context.hpp
│       │   │   ├── events.hpp
│       │   │   ├── guards.hpp
│       │   │   ├── actions.hpp
│       │   │   └── sm.hpp
│       │   ├── audio/
│       │   │   ├── context.hpp
│       │   │   ├── events.hpp
│       │   │   ├── guards.hpp
│       │   │   ├── actions.hpp
│       │   │   └── sm.hpp
│       │   ├── segmentation/
│       │   │   ├── context.hpp
│       │   │   ├── events.hpp
│       │   │   ├── guards.hpp
│       │   │   ├── actions.hpp
│       │   │   └── sm.hpp
│       │   └── signaling/
│       │       ├── context.hpp
│       │       ├── events.hpp
│       │       ├── guards.hpp
│       │       ├── actions.hpp
│       │       └── sm.hpp
│       └── support/
│           ├── fixed_string.hpp
│           ├── time.hpp
│           └── tracing.hpp
├── tests/
│   ├── runtime_session.test.cpp
│   ├── contract_perception_event.test.cpp
│   └── cortext_bridge_contract.test.cpp
└── examples/
    └── runtime_smoke/
        └── main.cpp
```

The important part is not the exact names above; it is the separation: `contract/` for repo-to-repo types, `runtime/<actor>/` for SML actors, and a single top-level actor folder that owns the locked state set. That structure fits the locked `emel.cpp/src` pattern and keeps later diarization/STT work from leaking into the contract surface. [VERIFIED: local codebase grep]

### Pattern 1: One Top-Level `co_sm` Session Actor
**What:** Use one `planum::runtime::session::sm` as the inspectable lifecycle owner with the locked states `inactive`, `activating`, `listening`, `segmenting`, `endpointing`, `signaling`, `degraded`, and `errored`. [VERIFIED: local codebase grep]

**When to use:** Immediately in Phase 1; this is the scaffold that gives later audio/runtime work a home without implementing the runtime itself. [VERIFIED: local codebase grep]

**Example:**
```cpp
// Source style: ../stateforward/emel/emel.cpp/src/emel/gbnf/sampler/token_parser/sm.hpp
namespace sml = boost::sml;

struct model {
  auto operator()() const {
    return sml::make_transition_table(
      sml::state<activating> <= *sml::state<inactive> + sml::event<event::activate>
                 / action::begin_activation
    , sml::state<listening> <= sml::state<activating> + sml::completion<event::activate_runtime>
                 [ guard::activation_ok{} ] / action::mark_listening
    , sml::state<errored> <= sml::state<activating> + sml::completion<event::activate_runtime>
                 [ guard::activation_failed{} ] / action::publish_error
    , sml::state<errored> <= sml::state<inactive> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
  }
};
```

This is the right pattern because the local rules require explicit transitions, explicit unexpected-event handling, and no mailbox/process-queue semantics. `co_sm` keeps the top-level wrapper compatible with those rules while still exposing synchronous `process_event(...)` and `visit_current_states(...)`. [VERIFIED: local codebase grep][CITED: https://boost-ext.github.io/sml/user_guide.html]

### Pattern 2: Contract-First Boundary, Sink-Second
**What:** Define repo boundary types before any runtime implementation code. The minimum useful Phase 1 contract is one event family plus one sink interface:

```cpp
namespace planum::contract {

enum class event_kind : uint8_t {
  partial_transcript,
  final_transcript,
  segment_opened,
  segment_closed,
  endpoint_reached,
  runtime_state_changed,
  degraded,
  error,
};

enum class runtime_state : uint8_t {
  inactive,
  activating,
  listening,
  segmenting,
  endpointing,
  signaling,
  degraded,
  errored,
};

struct perception_event {
  event_kind kind;
  runtime_state state;
  uint64_t emitted_at_ms;
  uint64_t segment_start_ms;
  uint64_t segment_end_ms;
  std::string_view stream_id;
  std::string_view segment_id;
  std::string_view turn_id;
  std::string_view transcript_text;
  float transcript_confidence;
  std::string_view speaker_id;
  float speaker_confidence;
  uint32_t degraded_code;
  uint32_t error_code;
};

struct sink {
  virtual ~sink() = default;
  virtual void accept(const perception_event& ev) = 0;
};

}  // namespace planum::contract
```

**When to use:** Before writing any speaker, endpointing, or STT logic. The contract is the Phase 1 deliverable; runtime algorithms are not. [VERIFIED: local codebase grep]

**Why this shape:** The phase context already identifies the payload categories that must cross the boundary, and the current example voice session already demonstrates the need for partial/final transcript text, speaker identity, runtime state, and errors. What is intentionally absent here is memory policy: no `retain`, no `force_write`, no `retrieve_only`, and no Cortext-specific policy enum. [VERIFIED: local codebase grep]

### Pattern 3: Private Cortext Bridge
**What:** Add a non-public Cortext bridge, for example under `src/audio/planum_bridge.hpp` and `src/audio/planum_bridge.cpp`, that translates `planum::contract::perception_event` into existing engine calls. [VERIFIED: local codebase grep]

**When to use:** In Plan 3 of this phase, after the contract exists and before any live runtime implementation begins. [VERIFIED: local codebase grep]

**Bridge rules:**
- Finalized transcript events call `Cortext::ProcessTextAt(...)` with a deterministic timestamp and a derived `source_id`. [VERIFIED: local codebase grep]
- Partial transcript events do not perform memory writes in Phase 1; they are for front-end signaling/UI or telemetry only. [VERIFIED: local codebase grep]
- Runtime-state, degraded, and error events do not mutate memory semantics in Phase 1. [VERIFIED: local codebase grep]
- `include/cortext/cortext.hpp` and `include/cortext/capi.h` remain unchanged. [VERIFIED: local codebase grep]

### Recommended Three-Plan Breakdown

#### Plan 1: `planum.cpp` Repo Scaffold & Actor Skeleton
- Add `CMakeLists.txt`, namespace roots, `include/planum/`, `src/planum/runtime/`, and `tests/`. [VERIFIED: local codebase grep]
- Add the top-level `session` actor as a `co_sm` wrapper with the locked state set only. [VERIFIED: local codebase grep]
- Add stub `audio`, `segmentation`, and `signaling` actor folders with empty or no-op machines so later phases have fixed landing zones. [VERIFIED: local codebase grep]
- Add smoke tests proving state inspection, explicit unexpected-event handling, and RTC/no-queue behavior at the scaffold layer. [VERIFIED: local codebase grep]

#### Plan 2: Boundary Contract & Contract Tests
- Add `include/planum/contract/` with ids, enums, event structs, and sink interface. [VERIFIED: local codebase grep]
- Define the event taxonomy for `partial_transcript`, `final_transcript`, `endpoint_reached`, `runtime_state_changed`, `degraded`, and `error`. [VERIFIED: local codebase grep]
- Add tests that prove the contract carries transcript/speaker/timing metadata but no memory-policy fields. [VERIFIED: local codebase grep]
- Add one runtime-smoke example that emits fake events through the contract without using a real backend. [VERIFIED: local codebase grep]

#### Plan 3: Cortext Private Bridge & Example Hook
- Add a private Cortext bridge that consumes the contract and routes finalized transcript events through existing `ProcessTextAt(...)`. [VERIFIED: local codebase grep]
- Add bridge tests in Cortext proving no changes to public headers, C API, or bindings are required. [VERIFIED: local codebase grep]
- Replace example-local ad hoc voice handoff with the bridge hook or a thin adapter around it, while keeping real STT/diarization implementation deferred. [VERIFIED: local codebase grep]

### Anti-Patterns to Avoid
- **Making `voice_session.hpp` the contract:** It is an example callback surface with `std::function` callbacks and backend/model-path fields, not a stable front-end boundary. [VERIFIED: local codebase grep]
- **Using `ProcessAudio(...)` as the Phase 1 seam:** It is a raw PCM ingest API with internally generated timestamps, which is not the normalized segmented front-end contract the phase has locked. [VERIFIED: local codebase grep]
- **Adding speaker/runtime types to `include/cortext/` or the C API:** That would push speech-front-end concerns into the stable engine surface. [VERIFIED: local codebase grep]
- **Implementing real diarization, endpointing, or STT kernels now:** That would convert a scaffold phase into a runtime phase and collapse the roadmap separation between Phases 1 and 2. [VERIFIED: local codebase grep]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Top-level audio-runtime orchestration | Custom controller thread, mailbox, or `post-for-later` queue [VERIFIED: local codebase grep] | `co_sm` plus explicit events and RTC dispatch [VERIFIED: local codebase grep] | The copied SML rules forbid queues and self-dispatch, and the local `co_sm` wrapper preserves synchronous `process_event(...)`. [VERIFIED: local codebase grep] |
| Repo-to-repo boundary | Example callback API or raw backend config struct [VERIFIED: local codebase grep] | Dedicated `contract/perception_event.hpp` + `contract/sink.hpp` [VERIFIED: local codebase grep] | The example callback shape already mixes backend paths and app callbacks with transcript output; that is the wrong abstraction boundary. [VERIFIED: local codebase grep] |
| Cortext integration | New public speech-aware API family [VERIFIED: local codebase grep] | Private bridge that reuses `ProcessTextAt(...)` [VERIFIED: local codebase grep] | Public header/C API stability is a repo rule, and timestamped text ingestion already exists. [VERIFIED: local codebase grep] |
| Memory policy in front-end | `retain_input`, `force_write`, `retrieve_only`, or write-threshold flags in `planum.cpp` events [VERIFIED: local codebase grep] | Keep only perception/confidence/runtime metadata upstream; keep memory policy downstream in Cortext [VERIFIED: local codebase grep] | The locked decisions explicitly forbid giving `planum.cpp` ownership of retention or write semantics. [VERIFIED: local codebase grep] |

**Key insight:** Phase 1 is not about building speech intelligence; it is about building the container that keeps later speech intelligence out of Cortext core. [VERIFIED: local codebase grep]

## Common Pitfalls

### Pitfall 1: Boundary Drift Into Memory Policy
**What goes wrong:** The front-end contract starts carrying flags like “retain this,” “write this,” or “retrieve-only,” and `planum.cpp` quietly becomes the owner of Cortext memory semantics. [VERIFIED: local codebase grep]
**Why it happens:** The current example integration already computes app-level flags such as reply enablement and retain-vs-retrieve behavior near the voice path, so it is easy to promote those fields into the new boundary by accident. [VERIFIED: local codebase grep]
**How to avoid:** Keep the contract limited to perception facts plus confidence/runtime metadata; treat all memory policy as downstream-only. [VERIFIED: local codebase grep]
**Warning signs:** Contract structs begin to mention “retain,” “write,” “policy,” “storage,” or Cortext-specific write thresholds. [VERIFIED: local codebase grep]

### Pitfall 2: Example Contract Becomes Product Contract
**What goes wrong:** `VoiceSessionConfig` or a small variation of it becomes the de facto boundary because it already exists. [VERIFIED: local codebase grep]
**Why it happens:** It is the only current place in Cortext that models partial/final transcripts and speaker ids in one header. [VERIFIED: local codebase grep]
**How to avoid:** Use it only as an input to the event taxonomy; make the real boundary its own neutral `contract/` namespace. [VERIFIED: local codebase grep]
**Warning signs:** Boundary types contain backend names, model paths, `std::function` callbacks, or playback/TTS controls. [VERIFIED: local codebase grep]

### Pitfall 3: Scaffold Creep
**What goes wrong:** Phase 1 starts implementing actual endpointing, STT, speaker attribution, or audio device plumbing. [VERIFIED: local codebase grep]
**Why it happens:** Audio runtime work is adjacent to the scaffold, and an empty actor tree can feel “unfinished” unless the scope line is held. [VERIFIED: local codebase grep]
**How to avoid:** Allow only no-op actions, synthetic events, and smoke examples in Phase 1. Real backend logic belongs in later phases. [VERIFIED: local codebase grep]
**Warning signs:** New model assets, long-running loops in actions, backend-specific configs in contract headers, or audio capture threads appearing in the scaffold plan. [VERIFIED: local codebase grep]

### Pitfall 4: Violating the SML Contract in the Name of Convenience
**What goes wrong:** Actions start branching, actor context stores dispatch-local scratch, or a queue/mailbox gets introduced to simplify deferred work. [VERIFIED: local codebase grep]
**Why it happens:** Audio runtimes often tempt developers toward async buffering and callback-driven re-entry. [ASSUMED]
**How to avoid:** Keep work phase-bounded, model transitions explicitly, use typed completions for internal phase progression, and keep any buffering outside the actor dispatch contract. [VERIFIED: local codebase grep][CITED: https://boost-ext.github.io/sml/user_guide.html]
**Warning signs:** `std::queue`, stored callbacks, self-dispatch, `if`-heavy actions, or context fields that mirror the current event payload. [VERIFIED: local codebase grep]

## Code Examples

Verified patterns from official and repo sources:

### Synchronous `co_sm` Wrapper
```cpp
// Source: ../stateforward/sml/sml.cpp/README.md
utility::co_sm<my_fsm> sm{};
sm.process_event(my_event{});
sm.visit_current_states([](auto state) {});
```

The local fork explicitly documents `process_event(...)` as the synchronous path “same as regular sm,” which is the right Phase 1 default. [VERIFIED: local codebase grep]

### Destination-First Transition Table Layout
```cpp
// Source style: ../stateforward/emel/emel.cpp/src/emel/gbnf/sampler/token_parser/sm.hpp
return sml::make_transition_table(
  sml::state<listening> <= *sml::state<inactive> + sml::event<event::activate>
             / action::begin_activation
, sml::state<segmenting> <= sml::state<listening> + sml::completion<event::chunk_runtime>
             [ guard::speech_detected{} ] / action::open_segment
, sml::state<errored> <= sml::state<segmenting> + sml::unexpected_event<sml::_>
             / action::on_unexpected
);
```

The destination-first row style is not cosmetic here; it is part of the locked `emel.cpp` reference pattern the phase expects. [VERIFIED: local codebase grep]

### Minimal Private Cortext Bridge
```cpp
// Source anchor: include/cortext/cortext.hpp
if (ev.kind == planum::contract::event_kind::final_transcript &&
    !ev.transcript_text.empty()) {
  const std::string source_id = build_source_id(ev.stream_id, ev.speaker_id);
  (void)cortext.ProcessTextAt(std::string(ev.transcript_text), source_id, ev.segment_end_ms);
}
```

This is the minimum useful bridge because `ProcessTextAt(...)` already exists, already accepts timestamps, and does not require public API changes. [VERIFIED: local codebase grep]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Example-local voice callbacks plus ad hoc source-id derivation in `examples/chat/main.cpp` [VERIFIED: local codebase grep] | Dedicated front-end contract in `planum.cpp` plus a private Cortext bridge [VERIFIED: local codebase grep] | Locked in the 2026-04-07 planning rewrite and Phase 1 context refresh [VERIFIED: local codebase grep] | Keeps speech runtime concerns out of the engine core and out of the public API. [VERIFIED: local codebase grep] |
| Raw PCM ingest via `ProcessAudio(...)` as the only audio-facing Cortext API [VERIFIED: local codebase grep] | Normalized perception-event handoff with transcript/speaker/timing/state metadata [VERIFIED: local codebase grep] | Locked in the updated modality-first roadmap for Phase 1 [VERIFIED: local codebase grep] | Lets `planum.cpp` own diarization/segmentation/runtime state while Cortext stays memory-engine focused. [VERIFIED: local codebase grep] |
| Implicit or example-state-driven runtime flow [VERIFIED: local codebase grep] | Explicit inspectable `co_sm` lifecycle states with RTC rules copied from `emel.cpp` [VERIFIED: local codebase grep] | Locked in the current phase context [VERIFIED: local codebase grep] | Makes later runtime behavior debuggable without implementing it yet. [VERIFIED: local codebase grep] |

**Deprecated/outdated:**
- Treating the current chat example’s voice callback wiring as the architectural template for the repo boundary is outdated for the new modality-first direction. [VERIFIED: local codebase grep]
- Treating raw `ProcessAudio(...)` ingest as the primary speech boundary is outdated for this phase because the phase has locked the front-end contract to normalized perception events plus metadata. [VERIFIED: local codebase grep]

## Assumptions Log

> List all claims tagged `[ASSUMED]` in this research. The planner and discuss-phase use this
> section to identify decisions that need user confirmation before execution.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Audio runtime work will tempt developers toward async buffering and callback-style re-entry if the scaffold is not kept strict. | Common Pitfalls | Medium — it affects how aggressively the plan should front-load guardrails and tests. |

## Open Questions (RESOLVED)

1. **Where should the private Cortext bridge live in Phase 1?**
   - Resolution: Land the bridge as `src/audio/planum_bridge.hpp` and `src/audio/planum_bridge.cpp`, which keeps it private to Cortext source paths while still allowing white-box tests and source-level smoke targets to include it. Public headers and the C API remain unchanged. [VERIFIED: local codebase grep]
   - Planning impact: Phase 1 bridge work stays in `src/` and `examples/chat/` only; no files under `include/cortext/` or `include/cortext/capi.h` are touched. [VERIFIED: local codebase grep]

2. **How should partial transcript events behave in Phase 1?**
   - Resolution: Partial events are represented in `planum.cpp` contract types and may be emitted through synthetic sink/example/benchmark scaffolds, but the Phase 1 Cortext bridge treats them as explicit no-write events. They do not call `ProcessTextAt(...)`, do not call `ProcessAudio(...)`, and do not introduce retrieval-only memory semantics ahead of Phase 3. [VERIFIED: local codebase grep]
   - Planning impact: Phase 1 proves contract shape and observability for partials without changing Cortext memory behavior. [VERIFIED: local codebase grep]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `third_party/planum.cpp` submodule [VERIFIED: local codebase grep] | Primary repo for Phase 1 [VERIFIED: local codebase grep] | ✓ [VERIFIED: local codebase grep] | present as repo stub with `README.md` and `docs/sml.rules.md` only [VERIFIED: local codebase grep] | none — Phase 1 work lands there [VERIFIED: local codebase grep] |
| `../stateforward/sml/sml.cpp` checkout [VERIFIED: local codebase grep] | `co_sm` reference and fork semantics [VERIFIED: local codebase grep] | ✓ [VERIFIED: local codebase grep] | local checkout present [VERIFIED: local codebase grep] | use copied rules plus vendored header path if direct sibling access is unavailable later |
| `cmake` [VERIFIED: local codebase grep] | Build scaffold [VERIFIED: local codebase grep] | ✓ [VERIFIED: local codebase grep] | `4.0.3` [VERIFIED: local codebase grep] | none needed [VERIFIED: local codebase grep] |
| `ctest` [VERIFIED: local codebase grep] | Test scaffold [VERIFIED: local codebase grep] | ✓ [VERIFIED: local codebase grep] | `4.0.3` [VERIFIED: local codebase grep] | none needed [VERIFIED: local codebase grep] |
| C++20 compiler [VERIFIED: local codebase grep] | Cortext validation and future `planum.cpp` build [VERIFIED: local codebase grep] | ✓ [VERIFIED: local codebase grep] | `Apple clang 16.0.0` [VERIFIED: local codebase grep] | `g++` also resolves to Apple clang on this machine [VERIFIED: local codebase grep] |

**Missing dependencies with no fallback:**
- None for planning. [VERIFIED: local codebase grep]

**Missing dependencies with fallback:**
- `planum.cpp` currently has no build or test scaffold; Phase 1 itself must create that scaffold before any runtime implementation work can proceed. [VERIFIED: local codebase grep]

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Mixed: Catch2 `v3.5.3` exists in Cortext; no test framework detected yet in `third_party/planum.cpp`. [VERIFIED: local codebase grep] |
| Config file | `tests/CMakeLists.txt` in Cortext; none detected yet in `third_party/planum.cpp`. [VERIFIED: local codebase grep] |
| Quick run command | `ctest --test-dir build -R cortext_tests --output-on-failure` [VERIFIED: local codebase grep] |
| Full suite command | `ctest --test-dir build --output-on-failure` [VERIFIED: local codebase grep] |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUD-00 | Contract carries normalized perception metadata without speech-runtime types entering Cortext public APIs. [VERIFIED: local codebase grep] | unit + API surface regression [VERIFIED: local codebase grep] | `ctest --test-dir build -R cortext_tests --output-on-failure` plus future `planum.cpp` contract tests [VERIFIED: local codebase grep] | ❌ Wave 0 |
| AUD-01 | `planum.cpp` has the actor/build scaffold needed for later low-latency runtime work. [VERIFIED: local codebase grep] | unit + smoke [VERIFIED: local codebase grep] | future `ctest --test-dir third_party/planum.cpp/build --output-on-failure` | ❌ Wave 0 |
| AUD-02 | Contract and state skeleton represent partials, endpointing, and finalized boundaries. [VERIFIED: local codebase grep] | unit state-machine tests [VERIFIED: local codebase grep] | future `ctest --test-dir third_party/planum.cpp/build --output-on-failure` | ❌ Wave 0 |
| RUN-01 | Realtime orchestration uses SML/co_sm rather than ad hoc control flow. [VERIFIED: local codebase grep] | unit state-machine tests [VERIFIED: local codebase grep] | future `ctest --test-dir third_party/planum.cpp/build --output-on-failure` | ❌ Wave 0 |
| RUN-02 | Cortext public C++ and C APIs remain unchanged while bridge code is added privately. [VERIFIED: local codebase grep] | unit + regression [VERIFIED: local codebase grep] | `ctest --test-dir build -R cortext_tests --output-on-failure` [VERIFIED: local codebase grep] | ❌ Wave 0 |
| RUN-03 | Lifecycle states are inspectable in tests/examples. [VERIFIED: local codebase grep] | unit + smoke [VERIFIED: local codebase grep] | future `ctest --test-dir third_party/planum.cpp/build --output-on-failure` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `ctest --test-dir build -R cortext_tests --output-on-failure` for Cortext-side work, plus the future `planum.cpp` scaffold test target once created. [VERIFIED: local codebase grep]
- **Per wave merge:** `ctest --test-dir build --output-on-failure` plus the full `planum.cpp` test target once created. [VERIFIED: local codebase grep]
- **Phase gate:** Cortext regression tests green, `planum.cpp` scaffold tests green, and one runtime-smoke example proving state inspection without real backend logic. [VERIFIED: local codebase grep]

### Wave 0 Gaps
- [ ] `third_party/planum.cpp/CMakeLists.txt` — build and test root for the primary Phase 1 repo. [VERIFIED: local codebase grep]
- [ ] `third_party/planum.cpp/tests/runtime_session.test.cpp` — locked-state skeleton and `unexpected_event` coverage. [VERIFIED: local codebase grep]
- [ ] `third_party/planum.cpp/tests/contract_perception_event.test.cpp` — contract-shape assertions and “no memory policy fields” checks. [VERIFIED: local codebase grep]
- [ ] `tests/planum_bridge_contract.test.cpp` or equivalent Cortext-side bridge test — proves no public API break. [VERIFIED: local codebase grep]
- [ ] Example smoke harness under `third_party/planum.cpp/examples/runtime_smoke/` — proves inspectable state transitions using synthetic events only. [VERIFIED: local codebase grep]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no — this phase does not add an auth surface or hosted service boundary. [VERIFIED: local codebase grep] | none in scope for Phase 1. [VERIFIED: local codebase grep] |
| V3 Session Management | no — the “session” here is an actor lifecycle, not a user session boundary. [VERIFIED: local codebase grep] | none in scope for Phase 1. [VERIFIED: local codebase grep] |
| V4 Access Control | no — no multi-tenant or permissioned surface is introduced by the scaffold itself. [VERIFIED: local codebase grep] | keep the bridge private and non-public. [VERIFIED: local codebase grep] |
| V5 Input Validation | yes — the new contract is an external ingest boundary and must validate event kind, timestamps, ids, and optional fields before bridging into Cortext. [VERIFIED: local codebase grep] | typed enums, explicit error/degraded events, and existing Cortext invalid-parameter/error-code discipline. [VERIFIED: local codebase grep] |
| V6 Cryptography | no — no cryptographic feature is introduced by this scaffold phase. [VERIFIED: local codebase grep] | none in scope for Phase 1. [VERIFIED: local codebase grep] |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed transcript/speaker/timing metadata at the repo boundary [VERIFIED: local codebase grep] | Tampering | Validate contract enums and required identifiers, route invalid events to explicit `error` / `degraded` states, and do not pass unchecked metadata through the bridge. [VERIFIED: local codebase grep] |
| Front-end contract quietly acquires memory-policy authority [VERIFIED: local codebase grep] | Elevation of Privilege | Keep retain/retrieve/write policy out of `planum.cpp` types and confine those decisions to Cortext/application code. [VERIFIED: local codebase grep] |
| Unexpected runtime events corrupt actor state progression [VERIFIED: local codebase grep] | Denial of Service | Add explicit `unexpected_event` handling and inspectable `degraded` / `errored` states in every scaffolded actor. [VERIFIED: local codebase grep][CITED: https://boost-ext.github.io/sml/user_guide.html] |
| Queue or callback re-entry reintroduces nondeterministic control flow [VERIFIED: local codebase grep] | Denial of Service | Use RTC `co_sm` dispatch, avoid mailboxes/process queues, and never store callbacks for later re-entry. [VERIFIED: local codebase grep] |

## Sources

### Primary (HIGH confidence)
- Local planning context:
  - `.planning/phases/01-audio-front-end-boundary-runtime-foundation/01-CONTEXT.md`
  - `.planning/ROADMAP.md`
  - `.planning/REQUIREMENTS.md`
  - `.planning/PROJECT.md`
- Local Cortext code and API surface:
  - `include/cortext/cortext.hpp`
  - `include/cortext/capi.h`
  - `src/cortext.cpp`
  - `examples/chat/voice_session.hpp`
  - `examples/chat/main.cpp`
  - `tests/CMakeLists.txt`
- Local `planum.cpp` and `stateforward` references:
  - `third_party/planum.cpp/README.md`
  - `third_party/planum.cpp/docs/sml.rules.md`
  - `../stateforward/emel/emel.cpp/AGENTS.md`
  - `../stateforward/emel/emel.cpp/src/emel/gbnf/sampler/token_parser/sm.hpp`
  - `../stateforward/emel/emel.cpp/src/emel/text/encoders/{context,events,actions,guards,any}.hpp`
  - `../stateforward/sml/sml.cpp/README.md`
  - `../stateforward/sml/sml.cpp/include/boost/sml/utility/co_sm.hpp`
  - `../stateforward/sml/sml.cpp/test/ft/co_sm.cpp`

### Secondary (MEDIUM confidence)
- Boost.SML user guide: https://boost-ext.github.io/sml/user_guide.html
- Boost.SML tutorial: https://boost-ext.github.io/sml/tutorial.html
- Boost.SML dispatch table source: https://github.com/boost-ext/sml/blob/v1.1.13/include/boost/sml/utility/dispatch_table.hpp

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - almost every recommendation is constrained by locked decisions plus local code/repo evidence. [VERIFIED: local codebase grep]
- Architecture: HIGH - the actor layout, state set, and boundary direction are directly specified in the phase context and reinforced by the `emel.cpp` reference pattern. [VERIFIED: local codebase grep]
- Pitfalls: MEDIUM - most pitfalls are directly observable from current code and rules, but one pitfall includes a predictive failure-mode assumption. [VERIFIED: local codebase grep]

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 for phase planning; refresh sooner if `planum.cpp` gains its own build/test scaffold or the boundary decisions change.
