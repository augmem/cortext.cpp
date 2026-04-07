---
phase: 01-audio-front-end-boundary-runtime-foundation
plan: 05
subsystem: api
tags: [planum, contract, audio-front-end, testing, boundary]
requires:
  - phase: 01
    provides: "Standalone `planum.cpp` scaffold and runnable test harness"
provides:
  - "Canonical `planum::contract` perception boundary headers for normalized transcript, timing, speaker, lifecycle, degraded, and error facts"
  - "Output-only sink seam for downstream consumers of normalized perception events"
  - "Targeted contract regression coverage that guards against retention, backend, and callback drift"
affects: [phase-01-bridge, phase-02-speaker-stack, phase-03-audio-memory]
tech-stack:
  added: []
  patterns: ["Contract-first repo boundary in `planum.cpp`", "Header-shape regression test that inspects forbidden contract drift terms"]
key-files:
  created:
    - third_party/planum.cpp/include/planum/contract/event_kind.hpp
    - third_party/planum.cpp/include/planum/contract/ids.hpp
    - third_party/planum.cpp/include/planum/contract/runtime_state.hpp
    - third_party/planum.cpp/include/planum/contract/perception_event.hpp
    - third_party/planum.cpp/include/planum/contract/sink.hpp
    - third_party/planum.cpp/tests/contract_perception_event.test.cpp
  modified:
    - third_party/planum.cpp/tests/CMakeLists.txt
key-decisions:
  - "Keep the contract payload non-owning with `std::string_view`-backed ids and metadata so the boundary stays allocation-light and policy-free."
  - "Represent sink output as `Accept(const PerceptionEvent&) noexcept` so downstream consumers observe facts without gaining mutation or ownership hooks."
patterns-established:
  - "Boundary types in `include/planum/contract/` carry perception facts only; memory authority stays downstream in Cortext."
  - "Contract tests compile the public headers and read them as text to block retention, backend, and callback drift."
requirements-completed: [AUD-00, AUD-02]
duration: 3min
completed: 2026-04-07
---

# Phase 1 Plan 05: Audio Front-End Boundary Runtime Foundation Summary

**Normalized `planum.cpp` perception-event headers with transcript, timing, speaker, lifecycle, degraded/error metadata plus an output-only sink seam and drift-guard tests**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-07T21:01:23Z
- **Completed:** 2026-04-07T21:03:54Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Defined the canonical `planum::contract` event taxonomy, id wrappers, runtime lifecycle enum, and normalized `PerceptionEvent` payload.
- Added an output-only `Sink` seam so downstream consumers accept boundary facts without callback-heavy or mutable contract surface.
- Added a dedicated `contract_perception_event` test that proves required metadata and explicitly rejects retention, backend, and callback drift.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define the normalized contract types** - `5c6bc8e` (feat)
2. **Task 2: Add the sink seam and contract-shape tests** - `0017824` (test), `13f1847` (feat)

## Files Created/Modified
- `third_party/planum.cpp/include/planum/contract/event_kind.hpp` - Canonical boundary event taxonomy.
- `third_party/planum.cpp/include/planum/contract/ids.hpp` - Lightweight stream, segment, turn, speaker, and timestamp identifiers.
- `third_party/planum.cpp/include/planum/contract/runtime_state.hpp` - Locked inspectable front-end lifecycle states.
- `third_party/planum.cpp/include/planum/contract/perception_event.hpp` - Normalized perception payload with transcript, timing, speaker, degraded, and error metadata.
- `third_party/planum.cpp/include/planum/contract/sink.hpp` - Output-only seam for downstream boundary consumers.
- `third_party/planum.cpp/tests/contract_perception_event.test.cpp` - Targeted contract test covering taxonomy, metadata shape, and forbidden field drift.
- `third_party/planum.cpp/tests/CMakeLists.txt` - Minimal test target wiring so the dedicated contract test participates in `planum_tests`.

## Decisions Made
- Used `std::string_view` inside the contract payload instead of owning strings so the boundary remains lightweight and does not imply policy or storage ownership.
- Kept degraded and error metadata present on the canonical payload rather than splitting additional event structs, which keeps the seam uniform while still allowing event-kind-specific interpretation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added minimal CTest wiring for the new contract executable**
- **Found during:** Task 2 (Add the sink seam and contract-shape tests)
- **Issue:** The existing `third_party/planum.cpp/tests/CMakeLists.txt` only built `repo_scaffold_smoke`, so the new `contract_perception_event.test.cpp` would never build or run under the plan’s verification command.
- **Fix:** Registered a dedicated `contract_perception_event` executable and test entry in `third_party/planum.cpp/tests/CMakeLists.txt`.
- **Files modified:** `third_party/planum.cpp/tests/CMakeLists.txt`
- **Verification:** `cmake --build third_party/planum.cpp/build --target planum_tests -j && ctest --test-dir third_party/planum.cpp/build --output-on-failure -R contract_perception_event`
- **Committed in:** `0017824`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The extra test registration was required to make the plan’s explicit verification runnable. No contract or runtime scope was expanded.

## Issues Encountered
- The initial RED run exposed that `third_party/planum.cpp/build` had not been configured in this checkout yet. A one-time `cmake -S third_party/planum.cpp -B third_party/planum.cpp/build -DCMAKE_BUILD_TYPE=Debug` resolved that environment issue before the intended missing-sink failure.
- `third_party/planum.cpp` contained pre-existing untracked paths (`docs/`, `src/planum/runtime/session/`). They were left untouched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- The `planum.cpp -> cortext` seam is now concrete and test-guarded, so later bridge work can bind to one normalized event payload instead of example callbacks.
- Future speech runtime work can emit partials, finals, endpoint, degraded, and error events without reopening the contract shape.

## Self-Check: PASSED

- Found `.planning/phases/01-audio-front-end-boundary-runtime-foundation/01-05-SUMMARY.md`
- Found task commit `5c6bc8e`
- Found task commit `0017824`
- Found task commit `13f1847`

---
*Phase: 01-audio-front-end-boundary-runtime-foundation*
*Completed: 2026-04-07*
