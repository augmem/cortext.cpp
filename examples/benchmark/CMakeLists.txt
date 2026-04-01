# Benchmark executable with new source files
add_executable(cortext_benchmark
  main.cpp
  src/audio_loader.cpp
  src/output_formatter.cpp
  src/oga_runner.cpp
  src/litert_runner.cpp
  src/gemma3n_audio.cpp
)

add_executable(cortext_text_encoder_bench
  text_encoder_bench.cpp
)

add_executable(cortext_deep_llm_bench
  deep_llm_bench.cpp
)

# Include directories for local headers
target_include_directories(cortext_benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_text_encoder_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(cortext_deep_llm_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(cortext_benchmark PRIVATE
  cortext
  nlohmann_json::nlohmann_json
)
target_link_libraries(cortext_text_encoder_bench PRIVATE
  cortext
)
target_link_libraries(cortext_deep_llm_bench PRIVATE
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

# Optional LiteRT-LM support (C++ API)
option(BENCHMARK_ENABLE_LITERT "Enable LiteRT-LM backend in benchmark" OFF)

if(BENCHMARK_ENABLE_LITERT)
  message(STATUS "Benchmark: LiteRT-LM backend enabled (C++ API)")

  # LiteRT-LM build output directory (contains lib + headers from Bazel build)
  set(LITERT_BUILD_DIR "${CMAKE_SOURCE_DIR}/third_party/litert-lm-build")

  if(NOT EXISTS "${LITERT_BUILD_DIR}/lib")
    message(FATAL_ERROR
      "LiteRT-LM not built. Run: ./scripts/build_litert.sh")
  endif()

  # Check for Bazel-copied external headers
  if(NOT EXISTS "${LITERT_BUILD_DIR}/include/absl")
    message(FATAL_ERROR
      "Abseil headers not found. Re-run: ./scripts/build_litert.sh")
  endif()
  if(NOT EXISTS "${LITERT_BUILD_DIR}/include/litert")
    message(FATAL_ERROR
      "LiteRT headers not found. Re-run: ./scripts/build_litert.sh")
  endif()

  # Find the C++ shared library (exports all C++ symbols, bundles all dependencies)
  find_library(LITERT_LM_CPP
    NAMES litert_lm_cpp
    PATHS "${LITERT_BUILD_DIR}/lib"
    NO_DEFAULT_PATH
  )

  # Fall back to C API shared library if C++ variant not found
  if(NOT LITERT_LM_CPP)
    find_library(LITERT_LM_CPP
      NAMES litert_lm_shared
      PATHS "${LITERT_BUILD_DIR}/lib"
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT LITERT_LM_CPP)
    message(FATAL_ERROR "LiteRT-LM shared library not found in ${LITERT_BUILD_DIR}/lib")
  endif()

  message(STATUS "Found LiteRT-LM: ${LITERT_LM_CPP}")
  message(STATUS "Using Bazel-cached Abseil and LiteRT headers from ${LITERT_BUILD_DIR}/include")

  # Add include directories for C++ API headers
  # The build directory contains: absl/, litert/, runtime/, engine/, etc.
  target_include_directories(cortext_benchmark PRIVATE
    "${LITERT_BUILD_DIR}/include"
  )

  # Link the C++ shared library (Abseil is bundled in the shared library)
  target_link_libraries(cortext_benchmark PRIVATE
    ${LITERT_LM_CPP}
  )

  # Require C++20 for LiteRT-LM
  target_compile_features(cortext_benchmark PRIVATE cxx_std_20)

  # Define macro
  target_compile_definitions(cortext_benchmark PRIVATE BENCHMARK_ENABLE_LITERT=1)

  # Add rpath for LiteRT runtime libraries
  # Executable is at build/examples/benchmark, library is at third_party/litert-lm-build/lib
  if(APPLE)
    set_property(TARGET cortext_benchmark APPEND PROPERTY BUILD_RPATH
      "@loader_path/../../../third_party/litert-lm-build/lib"
    )
  elseif(UNIX)
    set_property(TARGET cortext_benchmark APPEND PROPERTY BUILD_RPATH
      "$ORIGIN/../../../third_party/litert-lm-build/lib"
    )
  endif()
endif()

# Audio pipeline benchmark (TTS -> ASR -> EmbeddingGemma / ImageBind)
if(CORTEXT_ENABLE_EMBEDDINGGEMMA AND NOT CORTEXT_DISABLE_SHERPA_ONNX)
  add_executable(cortext_audio_pipeline_bench
    audio_pipeline_bench.cpp
  )

  target_include_directories(cortext_audio_pipeline_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(cortext_audio_pipeline_bench PRIVATE
    cortext
    nlohmann_json::nlohmann_json
  )

  # Ensure runtime libraries are discoverable
  if(APPLE)
    set_property(TARGET cortext_audio_pipeline_bench APPEND PROPERTY BUILD_RPATH
      "@loader_path/../../third_party/onnxruntime-install/lib"
      "@loader_path/../../third_party/litert-lm-install/lib"
    )
  elseif(UNIX)
    set_property(TARGET cortext_audio_pipeline_bench APPEND PROPERTY BUILD_RPATH
      "$ORIGIN/../../third_party/onnxruntime-install/lib"
      "$ORIGIN/../../third_party/litert-lm-install/lib"
    )
  endif()
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
