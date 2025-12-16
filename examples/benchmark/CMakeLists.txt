# Benchmark executable with new source files
add_executable(cortext_benchmark
  main.cpp
  src/audio_loader.cpp
  src/output_formatter.cpp
  src/oga_runner.cpp
  src/litert_runner.cpp
  src/gemma3n_audio.cpp
)

# Include directories for local headers
target_include_directories(cortext_benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(cortext_benchmark PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)

# Add OGA include directories and link (needed for oga_runner.cpp)
if(NOT CORTEXT_DISABLE_OGA)
  target_include_directories(cortext_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-genai/src
  )
  target_link_libraries(cortext_benchmark PRIVATE onnxruntime-genai)
endif()

# Optional LiteRT-LM support
option(BENCHMARK_ENABLE_LITERT "Enable LiteRT-LM backend in benchmark" OFF)

if(BENCHMARK_ENABLE_LITERT)
  message(STATUS "Benchmark: LiteRT-LM backend enabled")
  target_compile_definitions(cortext_benchmark PRIVATE BENCHMARK_ENABLE_LITERT=1)

  # TODO: Add LiteRT-LM library when available
  # find_package(LiteRT REQUIRED)
  # target_link_libraries(cortext_benchmark PRIVATE litert::lm)
endif()

# Ensure ONNX Runtime dylib can be found at runtime
if(NOT CORTEXT_DISABLE_OGA)
  if(APPLE)
    set_property(TARGET cortext_benchmark APPEND PROPERTY BUILD_RPATH
      "@loader_path/../../third_party/onnxruntime-install/lib"
      "@loader_path/../../ort-download/${ORT_PACKAGE}/lib"
    )
  elseif(UNIX)
    set_property(TARGET cortext_benchmark APPEND PROPERTY BUILD_RPATH
      "$ORIGIN/../../third_party/onnxruntime-install/lib"
      "$ORIGIN/../../ort-download/${ORT_PACKAGE}/lib"
    )
  endif()
endif()
