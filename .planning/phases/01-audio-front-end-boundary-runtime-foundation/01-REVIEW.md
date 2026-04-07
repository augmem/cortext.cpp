---
phase: 1
review_depth: standard
review_type: advisory
status: issues_found
files_reviewed: 42
findings:
  critical: 0
  warning: 3
  info: 0
  total: 3
---

# Findings

## Warning

1. `third_party/planum.cpp/CMakeLists.txt:13` points `PLANUM_STATEFORWARD_INCLUDE_DIR` at `../../../stateforward/sml/sml.cpp/include`, which resolves three directories above `third_party/planum.cpp` and skips the repo's `third_party/` tree. From this checkout, that expands outside the repository, so the advertised standalone `planum.cpp` build cannot find `<stateforward/sml.hpp>` unless the dependency happens to exist next to the repo. This is a concrete build break for the new scaffold targets that include `src/planum/runtime/session/sm.hpp`.

2. `third_party/planum.cpp/include/planum/contract/ids.hpp:11`, `third_party/planum.cpp/include/planum/contract/perception_event.hpp:13`, and `third_party/planum.cpp/include/planum/contract/sink.hpp:14` define the front-end contract entirely in terms of non-owning `std::string_view` fields, but `Sink::Accept` does not document any lifetime restriction. The current tests and examples immediately copy `PerceptionEvent` objects by value (`third_party/planum.cpp/tests/runtime_contract_sink.test.cpp:22`, `third_party/planum.cpp/examples/runtime_smoke/main.cpp:22`, `third_party/planum.cpp/benchmarks/runtime_state_probe.cpp:22`), which also copies the views. That is safe only because the fixtures use string literals; with real ASR buffers or transient diagnostics, downstream consumers can retain dangling pointers and read freed memory.

3. `third_party/planum.cpp/src/planum/runtime/session/sm.hpp:165` only installs `sml::unexpected_event` handling for the synthetic `unexpected_probe` type. Real out-of-order runtime events such as `segment_complete` while inactive or `reset` while listening are still just rejected by the machine without any explicit accounting or policy hook. That conflicts with the repo's SML rule that unexpected events must have explicit behavior rather than silent drops, and it leaves the `unexpected_event_count` metric blind to the operational cases that matter.

# Residual risks and testing gaps

- The audio, segmentation, and signaling actor files are still placeholders, so this review could only assess naming, contract shape, and build wiring there, not runtime behavior.
- None of the scoped tests exercise non-literal transcript/speaker/detail storage, so the string lifetime hazard is currently untested.
- I did not run builds or tests; this report is source-only and advisory, as requested.
