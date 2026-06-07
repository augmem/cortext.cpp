add_executable(cortext_aist_audio_embedder
  main.cpp
  ${CMAKE_SOURCE_DIR}/examples/benchmark/src/audio_loader.cpp
)

target_link_libraries(cortext_aist_audio_embedder PRIVATE cortext nlohmann_json::nlohmann_json)
target_compile_features(cortext_aist_audio_embedder PRIVATE cxx_std_20)
target_include_directories(cortext_aist_audio_embedder PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/examples/benchmark
)

if (MSVC)
  target_compile_options(cortext_aist_audio_embedder PRIVATE /W4)
else()
  target_compile_options(cortext_aist_audio_embedder PRIVATE -Wall -Wextra -Wpedantic)
endif()
