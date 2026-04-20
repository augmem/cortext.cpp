---
phase: 01
slug: audio-front-end-boundary-runtime-foundation
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-07
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Mixed: `ctest` in both repos, Catch2 in Cortext, lightweight repo-local smoke tests in `planum.cpp` |
| **Config file** | `tests/CMakeLists.txt` in Cortext; `third_party/planum.cpp/tests/CMakeLists.txt` plus top-level `third_party/planum.cpp/CMakeLists.txt` |
| **Quick run command** | `ctest --test-dir third_party/planum.cpp/build --output-on-failure -R "repo_scaffold_smoke|runtime_session|contract_perception_event|runtime_contract_sink"` |
| **Full suite command** | `ctest --test-dir third_party/planum.cpp/build --output-on-failure && ctest --test-dir build -R cortext_tests --output-on-failure` |
| **Estimated runtime** | ~90 seconds |

---

## Sampling Rate

- **After every task commit:** Run `ctest --test-dir third_party/planum.cpp/build --output-on-failure -R "repo_scaffold_smoke|runtime_session|contract_perception_event|runtime_contract_sink"` or `ctest --test-dir build -R cortext_tests --output-on-failure` for bridge work
- **After every plan wave:** Run `ctest --test-dir third_party/planum.cpp/build --output-on-failure`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | AUD-01 | T-01-01 | Repo scaffold builds without backend/runtime creep | build | `cmake -S third_party/planum.cpp -B third_party/planum.cpp/build -DCMAKE_BUILD_TYPE=Debug && cmake --build third_party/planum.cpp/build --target planum_runtime -j` | ✅ | ⬜ pending |
| 01-01-02 | 01 | 1 | RUN-01 | T-01-02 / T-01-03 | README and smoke test keep Phase 1 scaffold-only and RTC-oriented | smoke | `cmake --build third_party/planum.cpp/build --target planum_runtime planum_tests -j && ctest --test-dir third_party/planum.cpp/build --output-on-failure -R repo_scaffold_smoke` | ✅ | ⬜ pending |
| 01-02-01 | 02 | 2 | RUN-01 | T-01-05 | Locked lifecycle vocabulary exists before machine wiring | structure | `test -f third_party/planum.cpp/src/planum/runtime/session/context.hpp && test -f third_party/planum.cpp/src/planum/runtime/session/events.hpp && test -f third_party/planum.cpp/src/planum/runtime/session/guards.hpp && rg -n "inactive|activating|listening|segmenting|endpointing|signaling|degraded|errored" third_party/planum.cpp/src/planum/runtime/session/*.hpp` | ✅ | ⬜ pending |
| 01-02-02 | 02 | 2 | RUN-03 | T-01-04 / T-01-06 | Session actor stays within the locked state graph and handles unexpected events explicitly | unit | `cmake --build third_party/planum.cpp/build --target planum_tests -j && ctest --test-dir third_party/planum.cpp/build --output-on-failure -R runtime_session` | ✅ | ⬜ pending |
| 01-03-01 | 03 | 2 | AUD-01 | T-01-07 | Audio landing zone exists without backend/device ownership | structure | `test -f third_party/planum.cpp/src/planum/runtime/audio/context.hpp && test -f third_party/planum.cpp/src/planum/runtime/audio/events.hpp && test -f third_party/planum.cpp/src/planum/runtime/audio/guards.hpp && test -f third_party/planum.cpp/src/planum/runtime/audio/actions.hpp && test -f third_party/planum.cpp/src/planum/runtime/audio/sm.hpp && rg -n "namespace planum::runtime::audio|struct context|struct model" third_party/planum.cpp/src/planum/runtime/audio/*.hpp` | ✅ | ⬜ pending |
| 01-03-02 | 03 | 2 | AUD-01 | T-01-08 / T-01-09 | Segmentation landing zone exists without endpoint/runtime implementation creep | structure | `test -f third_party/planum.cpp/src/planum/runtime/segmentation/context.hpp && test -f third_party/planum.cpp/src/planum/runtime/segmentation/events.hpp && test -f third_party/planum.cpp/src/planum/runtime/segmentation/guards.hpp && test -f third_party/planum.cpp/src/planum/runtime/segmentation/actions.hpp && test -f third_party/planum.cpp/src/planum/runtime/segmentation/sm.hpp && rg -n "namespace planum::runtime::segmentation|struct context|struct model" third_party/planum.cpp/src/planum/runtime/segmentation/*.hpp` | ✅ | ⬜ pending |
| 01-04-01 | 04 | 3 | AUD-01 | T-01-11 | Signaling landing zone exists without queue or transport semantics | structure | `test -f third_party/planum.cpp/src/planum/runtime/signaling/context.hpp && test -f third_party/planum.cpp/src/planum/runtime/signaling/events.hpp && test -f third_party/planum.cpp/src/planum/runtime/signaling/guards.hpp && test -f third_party/planum.cpp/src/planum/runtime/signaling/actions.hpp && test -f third_party/planum.cpp/src/planum/runtime/signaling/sm.hpp && rg -n "namespace planum::runtime::signaling|struct context|struct model" third_party/planum.cpp/src/planum/runtime/signaling/*.hpp` | ✅ | ⬜ pending |
| 01-04-02 | 04 | 3 | RUN-03 | T-01-10 / T-01-12 | Benchmark-facing probe can inspect locked session states through synthetic events | benchmark smoke | `cmake --build third_party/planum.cpp/build --target runtime_state_probe -j && third_party/planum.cpp/build/benchmarks/runtime_state_probe` | ✅ | ⬜ pending |
| 01-05-01 | 05 | 2 | AUD-00 | T-01-13 | Contract types carry normalized perception metadata only | structure | `test -f third_party/planum.cpp/include/planum/contract/perception_event.hpp && rg -n "partial|final|endpoint|runtime|degraded|error|speaker|segment|turn" third_party/planum.cpp/include/planum/contract/*.hpp` | ✅ | ⬜ pending |
| 01-05-02 | 05 | 2 | AUD-02 | T-01-14 / T-01-15 | Contract taxonomy and no-policy exclusions are proven by tests | unit | `cmake --build third_party/planum.cpp/build --target planum_tests -j && ctest --test-dir third_party/planum.cpp/build --output-on-failure -R contract_perception_event` | ✅ | ⬜ pending |
| 01-06-01 | 06 | 4 | AUD-02 | T-01-16 / T-01-17 | Session emits synthetic contract events through sink without backend or policy logic | unit | `cmake --build third_party/planum.cpp/build --target planum_tests -j && ctest --test-dir third_party/planum.cpp/build --output-on-failure -R runtime_contract_sink` | ✅ | ⬜ pending |
| 01-06-02 | 06 | 4 | RUN-03 | T-01-18 | Example and benchmark hooks can inspect states plus sink output through the synthetic seam | example + benchmark smoke | `cmake --build third_party/planum.cpp/build --target runtime_smoke runtime_state_probe -j && third_party/planum.cpp/build/examples/runtime_smoke/runtime_smoke && third_party/planum.cpp/build/benchmarks/runtime_state_probe` | ✅ | ⬜ pending |
| 01-07-01 | 07 | 3 | RUN-02 | T-01-19 / T-01-20 | Private bridge maps only finalized transcript events and leaves public APIs unchanged | unit | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build --target cortext_tests -j && ctest --test-dir build -R cortext_tests --output-on-failure` | ✅ | ⬜ pending |
| 01-07-02 | 07 | 3 | RUN-02 | T-01-21 | Deterministic Cortext-side smoke target compiles without live audio/runtime dependencies | build | `cmake --build build --target cortext_planum_bridge_smoke -j` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure is covered by Plan 01. No separate pre-phase Wave 0 is required.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-07
