# Benchmark executable with new source files

add_executable(cortext_text_encoder_bench
  text_encoder_bench.cpp
)


add_executable(cortext_fact_store_bench
  fact_store_bench.cpp
)

add_executable(cortext_bitemporal_retrieval_bench
  bitemporal_retrieval_bench.cpp
)

add_executable(cortext_bitemporal_robustness_bench
  bitemporal_robustness_bench.cpp
)

add_executable(cortext_bitemporal_ablation_bench
  bitemporal_ablation_bench.cpp
)

add_executable(cortext_bitemporal_lifecycle_bench
  bitemporal_lifecycle_bench.cpp
)

add_executable(cortext_eviction_ablation_bench
  eviction_ablation_bench.cpp
)

add_executable(cortext_memory_system_ablation_bench
  memory_system_ablation_bench.cpp
)

add_executable(cortext_predictive_ablation_bench
  predictive_ablation_bench.cpp
)

add_executable(cortext_metacognitive_ablation_bench
  metacognitive_ablation_bench.cpp
)

add_executable(cortext_constructive_recall_ablation_bench
  constructive_recall_ablation_bench.cpp
)

add_executable(cortext_neuromodulator_ablation_bench
  neuromodulator_ablation_bench.cpp
)

add_executable(cortext_procedural_ablation_bench
  procedural_ablation_bench.cpp
)

add_executable(cortext_meta_learning_ablation_bench
  meta_learning_ablation_bench.cpp
)

add_executable(cortext_full_operation_ablation_bench
  full_operation_ablation_bench.cpp
)

add_executable(cortext_full_unit_ablation_bench
  full_unit_ablation_bench.cpp
)

add_executable(cortext_boundary_fact_extension_ablation_bench
  boundary_fact_extension_ablation_bench.cpp
)

add_executable(cortext_full_boundary_fact_unit_ablation_bench
  full_boundary_fact_unit_ablation_bench.cpp
)

add_executable(cortext_subcomponent_matrix_bench
  subcomponent_matrix_bench.cpp
)

add_executable(cortext_label_persistence_ablation_bench
  label_persistence_ablation_bench.cpp
)

add_executable(cortext_storage_eviction_budget_bench
  storage_eviction_budget_bench.cpp
)

add_executable(cortext_storage_eviction_sweep_bench
  storage_eviction_sweep_bench.cpp
)

add_executable(cortext_live_loop_steady_state_bench
  live_loop_steady_state_bench.cpp
)

add_executable(cortext_storage_consolidation_e2e_bench
  storage_consolidation_e2e_bench.cpp
)

add_executable(cortext_storage_human_sleep_consolidation_bench
  storage_human_sleep_consolidation_bench.cpp
)

add_executable(cortext_storage_human_sleep_token_bench
  storage_human_sleep_token_bench.cpp
)

add_executable(cortext_decay_surface_ablation_bench
  decay_surface_ablation_bench.cpp
)

add_executable(cortext_resurfacing_pressure_ablation_bench
  resurfacing_pressure_ablation_bench.cpp
)

add_executable(cortext_resurfacing_horizon_bench
  resurfacing_horizon_bench.cpp
)

add_executable(cortext_resurfacing_knob_sweep_bench
  resurfacing_knob_sweep_bench.cpp
)

add_executable(cortext_preference_update_ablation_bench
  preference_update_ablation_bench.cpp
)




add_executable(cortext_summarization_retrieval_ablation_bench
  summarization_retrieval_ablation_bench.cpp
)

add_executable(cortext_summarization_storage_pressure_ablation_bench
  summarization_storage_pressure_ablation_bench.cpp
)

add_executable(cortext_consolidation_label_pressure_ablation_bench
  summarization_storage_pressure_ablation_bench.cpp
)

add_executable(cortext_signal_filter_bench
  signal_filter_bench.cpp
)

add_executable(cortext_real_multimodal_episode_bench
  real_multimodal_episode_bench.cpp
)

add_executable(cortext_modality_agnostic_graph_bench
  modality_agnostic_graph_bench.cpp
)

add_executable(cortext_audio_video_accumulator_bench
  audio_video_accumulator_bench.cpp
)

add_executable(cortext_label_centroid_bench
  label_centroid_bench.cpp
)

add_executable(cortext_chat_replay_memory_eval
  chat_replay_memory_eval.cpp
)

add_executable(cortext_chat_replay_live_run
  chat_replay_live_run.cpp
)

add_executable(cortext_chat_replay_conversation_memory_bakeoff
  chat_replay_conversation_memory_bakeoff.cpp
)

add_executable(cortext_chat_replay_raw_speech_memory_eval
  chat_replay_raw_speech_memory_eval.cpp
  src/audio_loader.cpp
)

add_executable(cortext_label_vector_db_bench
  label_vector_db_bench.cpp
)

if(NOT CORTEXT_DISABLE_LITERT AND NOT EMSCRIPTEN)
  add_executable(cortext_gemma3n_multimodal_label_bench
    gemma3n_multimodal_label_bench.cpp
  )
endif()

# Include directories for local headers
target_include_directories(cortext_text_encoder_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_fact_store_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_bitemporal_retrieval_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_bitemporal_robustness_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_bitemporal_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_bitemporal_lifecycle_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_eviction_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_memory_system_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_predictive_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_metacognitive_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_constructive_recall_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_neuromodulator_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_procedural_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_meta_learning_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_full_operation_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_full_unit_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_boundary_fact_extension_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_full_boundary_fact_unit_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_subcomponent_matrix_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_label_persistence_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_storage_eviction_budget_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_storage_eviction_sweep_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_live_loop_steady_state_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_storage_consolidation_e2e_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_storage_human_sleep_consolidation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_storage_human_sleep_token_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_decay_surface_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_resurfacing_pressure_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_resurfacing_horizon_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_resurfacing_knob_sweep_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_preference_update_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_summarization_retrieval_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_summarization_storage_pressure_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_consolidation_label_pressure_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_signal_filter_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_real_multimodal_episode_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_modality_agnostic_graph_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_audio_video_accumulator_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_label_centroid_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_chat_replay_memory_eval PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_chat_replay_live_run PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_chat_replay_conversation_memory_bakeoff PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_chat_replay_raw_speech_memory_eval PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(cortext_chat_replay_conversation_memory_bakeoff PRIVATE
  CORTEXT_TESTING=1
)
if(TARGET cortext_gemma3n_multimodal_label_bench)
  target_include_directories(cortext_gemma3n_multimodal_label_bench SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${CMAKE_BINARY_DIR}/third_party/litert-lm-install/include"
  )
endif()

# Link dependencies
target_link_libraries(cortext_text_encoder_bench PRIVATE
  cortext
)
target_link_libraries(cortext_fact_store_bench PRIVATE
  cortext
)
target_link_libraries(cortext_bitemporal_retrieval_bench PRIVATE
  cortext
)
target_link_libraries(cortext_bitemporal_robustness_bench PRIVATE
  cortext
)
target_link_libraries(cortext_bitemporal_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_bitemporal_lifecycle_bench PRIVATE
  cortext
)
target_link_libraries(cortext_eviction_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_memory_system_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_predictive_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_metacognitive_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_constructive_recall_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_neuromodulator_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_procedural_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_meta_learning_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_full_operation_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_full_unit_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_boundary_fact_extension_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_full_boundary_fact_unit_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_subcomponent_matrix_bench PRIVATE
  cortext
)
target_link_libraries(cortext_label_persistence_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_storage_eviction_budget_bench PRIVATE
  cortext
)
target_link_libraries(cortext_storage_eviction_sweep_bench PRIVATE
  cortext
)
target_link_libraries(cortext_live_loop_steady_state_bench PRIVATE
  cortext
)
target_link_libraries(cortext_storage_consolidation_e2e_bench PRIVATE
  cortext
)
target_link_libraries(cortext_storage_human_sleep_consolidation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_storage_human_sleep_token_bench PRIVATE
  cortext
)
target_link_libraries(cortext_decay_surface_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_resurfacing_pressure_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_resurfacing_horizon_bench PRIVATE
  cortext
)
target_link_libraries(cortext_resurfacing_knob_sweep_bench PRIVATE
  cortext
)
target_link_libraries(cortext_preference_update_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_summarization_retrieval_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_summarization_storage_pressure_ablation_bench PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_consolidation_label_pressure_ablation_bench PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_signal_filter_bench PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_real_multimodal_episode_bench PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_modality_agnostic_graph_bench PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_audio_video_accumulator_bench PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_label_centroid_bench PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_chat_replay_memory_eval PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_chat_replay_live_run PRIVATE
  cortext
  nlohmann_json::nlohmann_json
  sqlite3
)
target_link_libraries(cortext_chat_replay_conversation_memory_bakeoff PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_chat_replay_raw_speech_memory_eval PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_label_vector_db_bench PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
if(TARGET cortext_gemma3n_multimodal_label_bench)
  target_link_libraries(cortext_gemma3n_multimodal_label_bench PRIVATE
    cortext
    nlohmann_json::nlohmann_json
    litert_lm
  )
  target_compile_features(cortext_gemma3n_multimodal_label_bench PRIVATE cxx_std_20)
  if(APPLE)
    set_property(TARGET cortext_gemma3n_multimodal_label_bench APPEND PROPERTY BUILD_RPATH
      "@loader_path/../../third_party/litert-lm-install/lib"
    )
  elseif(UNIX)
    set_property(TARGET cortext_gemma3n_multimodal_label_bench APPEND PROPERTY BUILD_RPATH
      "$ORIGIN/../../third_party/litert-lm-install/lib"
    )
  endif()
endif()
