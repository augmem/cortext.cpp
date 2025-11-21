# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.3
)
FetchContent_MakeAvailable(Catch2)

add_executable(cortext_tests
  cortext.test.cpp
  store.test.cpp
  store_extensions.test.cpp
  graph.test.cpp
  core_algorithms.test.cpp
  core_knobs.test.cpp
  operation_context.test.cpp
  operation_set.test.cpp
  operations_uncertainty.test.cpp
  operations_focus.test.cpp
  operations_focus_spread.test.cpp
  operations_boundary.test.cpp
  operations_stability.test.cpp
  operations_threshold.test.cpp
  operations_blend.test.cpp
  operations_metrics.test.cpp
  operations_memory_strength.test.cpp
  operations_focus_feedback.test.cpp
  operations_sensitivity_feedback.test.cpp
  operations_stability_feedback.test.cpp
  operations_influence_feedback.test.cpp
  operations_reconsolidation.test.cpp
  operations_competition.test.cpp
  operations_predictive.test.cpp
  operations_emotion.test.cpp
  operations_interrupt_gate.test.cpp
  operations_working_memory.test.cpp
  operations_metacognitive.test.cpp
  operations_serial_position.test.cpp
  operations_serial_position_apply.test.cpp
  operations_consolidation.test.cpp
  cortext.test.cpp
  signal_processor.test.cpp
  operations_adherence_fixes.test.cpp
)

target_link_libraries(cortext_tests PRIVATE cortext Catch2::Catch2WithMain)
target_compile_definitions(cortext_tests PRIVATE CORTEXT_TESTING=1)

include(Catch)
catch_discover_tests(cortext_tests)
