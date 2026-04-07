---
phase: 01-audio-front-end-boundary-runtime-foundation
plan: 02
subsystem: runtime
tags: [sml, co_sm, audio-front-end, state-machine, testing]
requires:
  - phase: 01-01
    provides: "Standalone planum.cpp runtime target and scaffold test hooks"
provides:
  - "Locked `co_sm` session actor with explicit top-level lifecycle states"
  - "Session actor structure split across context, events, guards, actions, and machine headers"
  - "Targeted `runtime_session` behavior coverage for state inspection and unexpected-event handling"
affects: [phase-01-actors, phase-01-signaling, phase-01-benchmarks]
tech-stack:
  added: []
  patterns: ["Context-owning `co_sm` wrapper for runtime actors", "Destination-first locked lifecycle transition tables", "Explicit `unexpected_event<T>` self-handling for external control probes"]
key-files:
  created:
    - third_party/planum.cpp/src/planum/runtime/session/context.hpp
    - third_party/planum.cpp/src/planum/runtime/session/events.hpp
    - third_party/planum.cpp/src/planum/runtime/session/guards.hpp
    - third_party/planum.cpp/src/planum/runtime/session/actions.hpp
    - third_party/planum.cpp/src/planum/runtime/session/sm.hpp
    - third_party/planum.cpp/tests/runtime_session.test.cpp
  modified:
    - third_party/planum.cpp/CMakeLists.txt
    - third_party/planum.cpp/tests/CMakeLists.txt
key-decisions:
  - "Wrap the session `co_sm` in a non-movable actor class that owns its injected context so SML dependency references stay stable."
  - "Model unexpected control probes with explicit `unexpected_event<unexpected_probe>` self-transitions per locked state instead of introducing queue-like fallback behavior."
patterns-established:
  - "Top-level runtime actors in `planum.cpp` use the `context/events/guards/actions/sm` split with a context-owning wrapper."
  - "Locked lifecycle behavior is proven through direct `is(...)` inspection plus `visit_current_states(...)` coverage in a dedicated runtime test."
requirements-completed: [AUD-01, RUN-01, RUN-03]
duration: 7min
completed: 2026-04-07
---

# Phase 1 Plan 02: Audio Front-End Boundary Runtime Foundation Summary

**Locked `co_sm` session lifecycle with explicit inactive/activating/listening/segmenting/endpointing/signaling/degraded/errored states and targeted behavior coverage**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-07T21:03:57Z
- **Completed:** 2026-04-07T21:10:54Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Added the session actor scaffold files under `third_party/planum.cpp/src/planum/runtime/session/` with persistent context, typed lifecycle events, and pure resume guards.
- Implemented the top-level `co_sm` session machine with the locked lifecycle state set and explicit unexpected-event handling.
- Added `runtime_session` coverage that proves initial-state inspection, lifecycle progression, and explicit handling of unexpected external events.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define the session actor structure and lifecycle events** - `a9be843` (feat)
2. **Task 2: RED - add failing `co_sm` lifecycle coverage** - `496d84c` (test)
3. **Task 2: GREEN - implement the locked session machine** - `5cd2b05` (feat)

## Files Created/Modified
- `third_party/planum.cpp/src/planum/runtime/session/context.hpp` - Persistent actor-owned counters for bounded scaffold actions.
- `third_party/planum.cpp/src/planum/runtime/session/events.hpp` - Locked lifecycle event set and typed locked-state vocabulary.
- `third_party/planum.cpp/src/planum/runtime/session/guards.hpp` - Pure guards for degraded-state resume routing.
- `third_party/planum.cpp/src/planum/runtime/session/actions.hpp` - Bounded scaffold-only counter actions for lifecycle and unexpected-event transitions.
- `third_party/planum.cpp/src/planum/runtime/session/sm.hpp` - Context-owning `co_sm` wrapper and the locked top-level transition table.
- `third_party/planum.cpp/tests/runtime_session.test.cpp` - Targeted lifecycle, inspection, and unexpected-event behavior verification.
- `third_party/planum.cpp/CMakeLists.txt` - Exposed `src/` and `stateforward/sml.cpp` headers to the scaffold runtime target.
- `third_party/planum.cpp/tests/CMakeLists.txt` - Registered the `runtime_session` executable and CTest entry.

## Decisions Made
- Used a wrapper class around `co_sm<model>` instead of a plain alias so the actor owns its context and can expose `process_event`, `is(...)`, `visit_current_states(...)`, and context inspection without leaking dependency-injection details.
- Kept actions strictly bounded counter increments and modeled all runtime control flow in the transition table so the scaffold remains RTC-safe and queue-free.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Extended the scaffold build graph to expose session sources and `stateforward/sml.cpp` headers**
- **Found during:** Task 2 (RED - add failing `co_sm` lifecycle coverage)
- **Issue:** The Plan 1 scaffold target only exposed `include/`, so the new session actor headers and `co_sm` includes could not compile in the planned runtime test.
- **Fix:** Added `src/` plus the sibling `stateforward/sml.cpp/include` path to `planum_runtime`, and registered `runtime_session` in the `planum_tests` aggregate target.
- **Files modified:** `third_party/planum.cpp/CMakeLists.txt`, `third_party/planum.cpp/tests/CMakeLists.txt`
- **Verification:** `cmake -S third_party/planum.cpp -B third_party/planum.cpp/build -DCMAKE_BUILD_TYPE=Debug && cmake --build third_party/planum.cpp/build --target planum_tests -j && ctest --test-dir third_party/planum.cpp/build --output-on-failure -R runtime_session`
- **Committed in:** `496d84c`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The deviation was required to compile the planned `co_sm` session actor and did not expand runtime scope beyond the locked scaffold.

## Issues Encountered
- The initial scaffold expected `co_sm::is(...)` to accept raw state tags; the wrapper needed to pass SML state wrappers instead.
- `unexpected_event<T>` handling in the fork is best proven through state/counter effects rather than `process_event(...)` return values, so the runtime test was tightened around explicit machine behavior.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- The top-level runtime owner is now fixed and inspectable, so later Phase 1 plans can attach child actors and signaling hooks without reopening the lifecycle vocabulary.
- The test harness now has a dedicated runtime-state machine target for future actor and benchmark work.

## Self-Check: PASSED

- Found `.planning/phases/01-audio-front-end-boundary-runtime-foundation/01-02-SUMMARY.md`
- Found task commit `a9be843`
- Found task commit `496d84c`
- Found task commit `5cd2b05`

---
*Phase: 01-audio-front-end-boundary-runtime-foundation*
*Completed: 2026-04-07*
