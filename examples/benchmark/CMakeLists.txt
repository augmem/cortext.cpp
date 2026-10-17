# Benchmark executable with new source files

add_executable(cortext_text_encoder_bench
  text_encoder_bench.cpp
)



add_executable(cortext_neuromodulator_ablation_bench
  neuromodulator_ablation_bench.cpp
)


add_executable(cortext_live_loop_steady_state_bench
  live_loop_steady_state_bench.cpp
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
target_include_directories(cortext_neuromodulator_ablation_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_live_loop_steady_state_bench PRIVATE
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
  ${PROJECT_SOURCE_DIR}/src
)
# Link dependencies
target_link_libraries(cortext_text_encoder_bench PRIVATE
  cortext
)
target_link_libraries(cortext_neuromodulator_ablation_bench PRIVATE
  cortext
)
target_link_libraries(cortext_live_loop_steady_state_bench PRIVATE
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
)
