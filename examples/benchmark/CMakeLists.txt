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

# Optional LiteRT-LM support (C++ API)
option(BENCHMARK_ENABLE_LITERT "Enable LiteRT-LM backend in benchmark" OFF)

if(BENCHMARK_ENABLE_LITERT)
  message(STATUS "Benchmark: LiteRT-LM backend enabled (C++ API)")

  # LiteRT-LM build output directory
  set(LITERT_BUILD_DIR "${CMAKE_SOURCE_DIR}/third_party/litert-lm-build")

  if(NOT EXISTS "${LITERT_BUILD_DIR}/lib")
    message(FATAL_ERROR
      "LiteRT-LM not built. Run: ./scripts/build_litert.sh")
  endif()

  # Fetch Abseil (required for LiteRT-LM C++ headers)
  include(FetchContent)
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_BUILD_TESTING OFF)
  FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20240722.0  # Match LiteRT-LM's Abseil version
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(abseil-cpp)
  message(STATUS "Fetched Abseil 20240722.0")

  # Find the shared library (bundles all dependencies)
  find_library(LITERT_LM_SHARED
    NAMES litert_lm_shared
    PATHS "${LITERT_BUILD_DIR}/lib"
    NO_DEFAULT_PATH
  )

  if(NOT LITERT_LM_SHARED)
    message(FATAL_ERROR "LiteRT-LM shared library not found in ${LITERT_BUILD_DIR}/lib")
  endif()

  message(STATUS "Found LiteRT-LM: ${LITERT_LM_SHARED}")

  # Add include directories for C++ API headers
  target_include_directories(cortext_benchmark PRIVATE
    "${LITERT_BUILD_DIR}/include"
    "${CMAKE_SOURCE_DIR}/third_party/litert-lm"  # For runtime/ headers
  )

  # Link the shared library and Abseil components
  target_link_libraries(cortext_benchmark PRIVATE
    ${LITERT_LM_SHARED}
    absl::status
    absl::statusor
    absl::strings
    absl::time
    absl::log
    absl::any_invocable
    absl::nullability
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
