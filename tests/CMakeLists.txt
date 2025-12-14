# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)

include(FetchContent)

# Use a block to prevent Catch2's own tests from being added to ctest
block(SCOPE_FOR VARIABLES)
  set(BUILD_TESTING OFF)
  set(CATCH_BUILD_TESTING OFF)
  
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.3
  )
  FetchContent_MakeAvailable(Catch2)
endblock()

# Now include Catch2 cmake modules for test discovery
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

add_executable(cortext_tests
  cortext.test.cpp
  imagebind_encoder.test.cpp
  data.test.cpp
  operations_centroids.test.cpp
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
  operations_logprob_surprise.test.cpp
  operations_precision_modulation.test.cpp
  operations_graph_schema.test.cpp
  operations_graph_build.test.cpp
  operations_graph_retrieval.test.cpp
  operations_goal_alignment.test.cpp
  operations_consolidation.test.cpp
  migration_core.test.cpp
  cortext.test.cpp
  signal_processor.test.cpp
  operations_adherence_fixes.test.cpp
  memory_feedback_insert.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/telemetry_noop_by_default.test.cpp
)

target_link_libraries(cortext_tests PRIVATE cortext Catch2::Catch2WithMain)
target_compile_definitions(cortext_tests PRIVATE CORTEXT_TESTING=1)

# Ensure the ONNX Runtime dylib can be found when Catch2 runs test discovery
# during the build (macOS uses @rpath + LC_RPATH entries).
if(CORTEXT_ENABLE_IMAGEBIND_ORT)
  if(APPLE)
    # cortext_tests lives in <build>/tests; ORT is installed to
    # <build>/third_party/onnxruntime-install/lib
    set_property(TARGET cortext_tests APPEND PROPERTY BUILD_RPATH
      "@loader_path/../third_party/onnxruntime-install/lib"
    )
  elseif(UNIX)
    set_property(TARGET cortext_tests APPEND PROPERTY BUILD_RPATH
      "$ORIGIN/../third_party/onnxruntime-install/lib"
    )
  endif()
endif()

include(Catch)
catch_discover_tests(cortext_tests)
