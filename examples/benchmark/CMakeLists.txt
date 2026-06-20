# Benchmark executable with new source files

# sqlite_store.hpp includes <sqlite3.h>; resolve it the same way the
# library does (pkg-config on native).
if(SQLITE3_INCLUDE_DIRS)
  include_directories(${SQLITE3_INCLUDE_DIRS})
endif()

add_executable(cortext_text_encoder_bench
  text_encoder_bench.cpp
)


add_executable(cortext_eviction_ablation_bench
  eviction_ablation_bench.cpp
)

add_executable(cortext_predictive_ablation_bench
  predictive_ablation_bench.cpp
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

add_executable(cortext_cognitive_mechanism_ablation_bench
  cognitive_mechanism_ablation_bench.cpp
)

add_executable(cortext_soar_cue_rarity_julie_bench
  soar_cue_rarity_julie_bench.cpp
)

add_executable(cortext_meta_learning_ablation_bench
  meta_learning_ablation_bench.cpp
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

add_executable(cortext_storage_human_sleep_token_bench
  storage_human_sleep_token_bench.cpp
)

add_executable(cortext_decay_surface_ablation_bench
  decay_surface_ablation_bench.cpp
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

add_executable(cortext_chat_replay_live_run
  chat_replay_live_run.cpp
)

# Include directories for local headers
target_include_directories(cortext_text_encoder_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_eviction_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_predictive_ablation_bench PRIVATE
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
target_include_directories(cortext_cognitive_mechanism_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_soar_cue_rarity_julie_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_meta_learning_ablation_bench PRIVATE
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
target_include_directories(cortext_storage_human_sleep_token_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_decay_surface_ablation_bench PRIVATE
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
target_include_directories(cortext_chat_replay_live_run PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
# Link dependencies
target_link_libraries(cortext_text_encoder_bench PRIVATE
  cortext
)
target_link_libraries(cortext_eviction_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_predictive_ablation_bench PRIVATE
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
target_link_libraries(cortext_cognitive_mechanism_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_soar_cue_rarity_julie_bench PRIVATE
  cortext
)
target_link_libraries(cortext_meta_learning_ablation_bench PRIVATE
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
target_link_libraries(cortext_storage_human_sleep_token_bench PRIVATE
  cortext
)
target_link_libraries(cortext_decay_surface_ablation_bench PRIVATE
  cortext
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
target_link_libraries(cortext_chat_replay_live_run PRIVATE
  cortext
  nlohmann_json::nlohmann_json
  ${SQLITE3_LIBRARIES}
)
if(SQLITE3_LIBRARY_DIRS)
  target_link_directories(cortext_chat_replay_live_run PRIVATE ${SQLITE3_LIBRARY_DIRS})
endif()
