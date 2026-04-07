---
phase: 01-audio-front-end-boundary-runtime-foundation
plan: 06
subsystem: runtime
tags: [planum.cpp, sml, contract, runtime, benchmark, example]
requires:
  - "01-02"
  - "01-04"
  - "01-05"
provides:
  - "Session `co_sm` sink wiring that emits normalized synthetic contract events"
  - "Targeted sink-emission coverage for lifecycle, transcript, endpoint, degraded, and error events"
  - "Example and benchmark seams that expose both lifecycle state and captured contract output"
affects: [phase-01-runtime, phase-01-contract-boundary, phase-01-observability]
tech-stack:
  added: []
  patterns:
    - "Bounded SML actions publish typed synthetic `PerceptionEvent`s through a non-owning contract sink"
    - "Synthetic examples and probes use fixed-size capture sinks to inspect seam output without backend logic"
key-files:
  created:
    - third_party/planum.cpp/tests/runtime_contract_sink.test.cpp
    - third_party/planum.cpp/examples/runtime_smoke/main.cpp
  modified:
    - third_party/planum.cpp/src/planum/runtime/session/context.hpp
    - third_party/planum.cpp/src/planum/runtime/session/events.hpp
    - third_party/planum.cpp/src/planum/runtime/session/actions.hpp
    - third_party/planum.cpp/src/planum/runtime/session/sm.hpp
    - third_party/planum.cpp/tests/CMakeLists.txt
    - third_party/planum.cpp/examples/CMakeLists.txt
    - third_party/planum.cpp/benchmarks/runtime_state_probe.cpp
key-decisions:
  - "Keep sink ownership outside the session actor by storing only a non-owning `planum::contract::Sink*` in context, with a local null sink preserving existing scaffold call sites."
  - "Emit lifecycle and perception facts from explicit bounded SML actions instead of callbacks, backend hooks, or deferred dispatch."
  - "Pin the runtime smoke example output path to the plan verification path so automation can execute the seam without ad hoc path discovery."
patterns-established:
  - "Session lifecycle transitions now publish `runtime_state_changed` events whenever the locked actor enters a new lifecycle state."
  - "Synthetic transcript, endpoint, degraded, and error probes can be exercised from examples, benchmarks, and tests through the same contract sink seam."
requirements-completed: [AUD-02, RUN-03]
duration: 7min
completed: 2026-04-07
---

# Phase 1 Plan 06: Audio Front-End Boundary Runtime Foundation Summary

**Synthetic session lifecycle transitions now emit normalized contract events through `planum::contract::Sink`, with runtime smoke and benchmark probes exposing both state and seam output**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-07T21:26:30Z
- **Completed:** 2026-04-07T21:33:04Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Threaded `planum::contract::Sink` through the session actor context and emitted synthetic `PerceptionEvent`s for lifecycle state changes plus partial/final/endpoint/degraded/error probes.
- Added `runtime_contract_sink` coverage that proves sink emission behavior and guards against backend or policy creep at the front-end boundary.
- Added a deterministic `runtime_smoke` example and upgraded `runtime_state_probe` so both observe current session state and captured sink output without any live audio backend.

## Task Commits

Each task was committed atomically in the nested `third_party/planum.cpp` repo:

1. **Task 1: Thread `planum::contract::sink` through the session actor** - `03f4d5b` (test), `2d445f7` (feat)
2. **Task 2: Extend the example and benchmark hooks to observe the session/contract seam** - `6bc56cb` (feat)

## Files Created/Modified

- `third_party/planum.cpp/src/planum/runtime/session/context.hpp` - Adds the non-owning sink seam to persistent session context without introducing backend ownership.
- `third_party/planum.cpp/src/planum/runtime/session/events.hpp` - Defines typed synthetic transcript, endpoint, degraded, and error probe events for the contract seam.
- `third_party/planum.cpp/src/planum/runtime/session/actions.hpp` - Emits bounded `PerceptionEvent`s directly from explicit SML actions and lifecycle transitions.
- `third_party/planum.cpp/src/planum/runtime/session/sm.hpp` - Wires the sink-aware session actor constructors and transition rows for lifecycle plus synthetic seam output.
- `third_party/planum.cpp/tests/runtime_contract_sink.test.cpp` - Verifies lifecycle state emission and synthetic contract-event mappings through a capture sink.
- `third_party/planum.cpp/tests/CMakeLists.txt` - Registers the focused `runtime_contract_sink` executable and test.
- `third_party/planum.cpp/examples/runtime_smoke/main.cpp` - Demonstrates inspectable session state plus captured contract events through a deterministic synthetic session run.
- `third_party/planum.cpp/examples/CMakeLists.txt` - Materializes the `runtime_smoke` target and aligns its output path with the plan verification command.
- `third_party/planum.cpp/benchmarks/runtime_state_probe.cpp` - Extends the probe to capture and print sink-emitted contract events alongside lifecycle snapshots.

## Decisions Made

- Used explicit event payload structs carrying ids, transcript text, and status metadata so the seam stays normalized and backend-free.
- Preserved the existing zero-argument `session::sm` construction path by injecting a local null sink instead of forcing every scaffold caller to own a sink immediately.
- Kept both the example and benchmark fully synthetic, using fixed-size in-memory capture buffers rather than callbacks, queues, or dynamic runtime infrastructure.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added test-target wiring outside the listed file set**
- **Found during:** Task 1
- **Issue:** The plan required `cmake --build ... --target planum_tests` and `ctest -R runtime_contract_sink`, but `third_party/planum.cpp/tests/CMakeLists.txt` had no target or test registration for the new sink coverage.
- **Fix:** Added `runtime_contract_sink` to the nested test CMake and made `planum_tests` depend on it.
- **Files modified:** `third_party/planum.cpp/tests/CMakeLists.txt`
- **Verification:** `cmake --build third_party/planum.cpp/build --target planum_tests -j && ctest --test-dir third_party/planum.cpp/build --output-on-failure -R runtime_contract_sink`
- **Committed in:** `03f4d5b`

**2. [Rule 3 - Blocking] Added example-target wiring and output-path alignment outside the listed file set**
- **Found during:** Task 2
- **Issue:** The plan required building and executing `runtime_smoke`, but `third_party/planum.cpp/examples/CMakeLists.txt` was still an empty custom target and the default executable path did not match the plan’s verification command.
- **Fix:** Materialized the `runtime_smoke` executable target, pre-created its output directory, and pinned the runtime output path to `build/examples/runtime_smoke/runtime_smoke`.
- **Files modified:** `third_party/planum.cpp/examples/CMakeLists.txt`
- **Verification:** `cmake --build third_party/planum.cpp/build --target runtime_smoke runtime_state_probe -j && third_party/planum.cpp/build/examples/runtime_smoke/runtime_smoke && third_party/planum.cpp/build/benchmarks/runtime_state_probe`
- **Committed in:** `6bc56cb`

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both deviations were required to satisfy the plan’s explicit verification commands while keeping the runtime scaffold synthetic and queue-free.

## Issues Encountered

- The nested `planum.cpp` build had a stale `build/examples/runtime_smoke` executable from the earlier flat output layout. After switching to the plan’s expected nested path, I removed the generated artifact and re-ran the build so the executable could be emitted into the new directory.
- `third_party/planum.cpp` still has a pre-existing untracked `docs/` directory. It is unrelated to this plan and was left untouched.

## Threat Flags

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The session scaffold now exposes the normalized contract seam directly, so later runtime work can reuse the same sink path for private Cortext bridging or richer actor interactions.
- Examples, benchmarks, and tests all have deterministic visibility into both lifecycle state and emitted contract events before any live STT, endpointing, or diarization logic lands.

## Self-Check: PASSED

- Found `.planning/phases/01-audio-front-end-boundary-runtime-foundation/01-06-SUMMARY.md`
- Found task commit `03f4d5b`
- Found task commit `2d445f7`
- Found task commit `6bc56cb`

---
*Phase: 01-audio-front-end-boundary-runtime-foundation*
*Completed: 2026-04-07*
