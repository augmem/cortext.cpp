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
  data.test.cpp
  operations_centroids.test.cpp
  store.test.cpp
  store_extensions.test.cpp
  object_store.test.cpp
  core_algorithms.test.cpp
  core_knobs.test.cpp
  thread_config.test.cpp
  operation_context.test.cpp
  operation_set.test.cpp
  operation_set_typed.test.cpp
  operations_uncertainty.test.cpp
  operations_focus.test.cpp
  operations_focus_spread.test.cpp
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
  operations_meta_learning.test.cpp
  operations_constructive_recall.test.cpp
  operations_emotion.test.cpp
  operations_mood.test.cpp
  operations_sensitivity.test.cpp
  operations_sensitivity_update.test.cpp
  centroids.test.cpp
  operations_interrupt_gate.test.cpp
  operations_working_memory.test.cpp
  operations_serial_position.test.cpp
  operations_serial_position_apply.test.cpp
  operations_embedding_prediction_error.test.cpp
  operations_precision_modulation.test.cpp

  operations_graph_build.test.cpp
  operations_graph_retrieval.test.cpp
  operations_emotion_cascade.test.cpp

  operations_consolidation.test.cpp
  operations_consolidation_cluster.test.cpp
	  operations_write_gate.test.cpp
  operations_memory_storage.test.cpp
  operations_soft_anchor.test.cpp
  operations_detect_memory_usage.test.cpp
  operations_streaming_pacing.test.cpp
  operations_accumulator.test.cpp
  migration_core.test.cpp
  cortext.test.cpp
  signal_processor.test.cpp
  signal_filter.test.cpp
  operations_adherence_fixes.test.cpp
  # memory_feedback_insert.test.cpp removed - v1 table no longer exists
  state_persistence.test.cpp
  implicit_feedback.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/telemetry_noop_by_default.test.cpp
  formula_validation.test.cpp
  integration_working_memory_manual.test.cpp
  regression_behavior.test.cpp
  aist_gguf_encoder.test.cpp
)

# Asset- and submodule-dependent tests are gated at configure time: a test
# is either compiled in (and must pass) or absent. No runtime skips.

# The planum bridge test needs the planum.cpp submodule headers.
if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/planum.cpp/include/planum/contract/perception_event.hpp)
  target_sources(cortext_tests PRIVATE
    audio_planum_bridge.test.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/planum_bridge.cpp
  )
endif()

# AIST encoder integration needs the model under models/ and the default
# tri-modal build contract.
if((CORTEXT_FETCH_AIST_MODEL OR EXISTS ${CMAKE_SOURCE_DIR}/models/AIST-87M-GGUF)
   AND CORTEXT_ENABLE_AUDIO
   AND CORTEXT_ENABLE_IMAGE)
  target_sources(cortext_tests PRIVATE
    aist_gguf_encoder_integration.test.cpp
  )
  target_compile_definitions(cortext_tests PRIVATE CORTEXT_TESTS_HAVE_AIST_MODEL=1)
endif()

target_link_libraries(cortext_tests PRIVATE cortext Catch2::Catch2WithMain nlohmann_json::nlohmann_json)
target_compile_definitions(cortext_tests PRIVATE CORTEXT_TESTING=1)
target_include_directories(cortext_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/planum.cpp/include
)

include(Catch)

# Catch2 test discovery is collapsing many cases into a single CTest entry.
# Run the full suite as one CTest test to keep CI green.
add_test(NAME cortext_tests COMMAND $<TARGET_FILE:cortext_tests>)
set_tests_properties(cortext_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
