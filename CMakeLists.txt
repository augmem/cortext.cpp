cmake_minimum_required(VERSION 3.16)
project(cortext VERSION 0.1.0 LANGUAGES C CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD STREQUAL "")
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTING "Build tests" ON)
option(CORTEXT_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(CORTEXT_BUILD_EXAMPLES "Build examples" OFF)
option(CORTEXT_BUILD_NODE_BINDINGS "Build Node.js addon bindings" OFF)
option(BUILD_WASM "Build WebAssembly version" OFF)
option(CORTEXT_WASM_EXPORT_ALL "Export all functions in WASM build" ON)
option(CORTEXT_EMBED_VEC "Embed sqlite-vec extension" ON)
option(CORTEXT_EMBED_OBJSTORE "Embed sqlite-objstore extension" ON)
option(CORTEXT_ENABLE_DYNAMIC_EXTENSIONS "Enable dynamic extension loading fallback" ON)
option(CORTEXT_DISABLE_IMAGEBIND_ORT "Disable ONNX Runtime for ImageBind embedder" OFF)
option(CORTEXT_ENABLE_GEMMA_ORT "Enable ONNX Runtime for Gemma text generator" OFF)
option(CORTEXT_DISABLE_OGA "Disable onnxruntime-genai for Phi-4 Extractor/Summarizer" OFF)
option(CORTEXT_DISABLE_LITERT "Disable LiteRT-LM for Gemma Extractor/Summarizer" OFF)
option(CORTEXT_DISABLE_SHERPA_ONNX "Disable sherpa-onnx ASR/TTS integration" OFF)
option(CORTEXT_ENABLE_EMBEDDINGGEMMA "Build EmbeddingGemma encoder" ON)

# Sanitizer options for catching memory bugs during development
option(CORTEXT_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(CORTEXT_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(CORTEXT_ENABLE_MSAN "Enable MemorySanitizer (Clang only)" OFF)

set(CORTEXT_NODE_EXECUTABLE "" CACHE FILEPATH "Path to the node executable used for Node.js addon builds")
set(CORTEXT_NODE_INCLUDE_DIR "" CACHE PATH "Path to Node.js headers used for Node.js addon builds")

if(APPLE)
  set(CORTEXT_ENABLE_ORT_COREML ON CACHE BOOL "Enable CoreML EP in ONNX Runtime build" FORCE)
else()
  set(CORTEXT_ENABLE_ORT_COREML OFF CACHE BOOL "Enable CoreML EP in ONNX Runtime build" FORCE)
endif()

if(EMSCRIPTEN AND BUILD_TESTING)
  message(STATUS "Disabling tests for WebAssembly builds (not supported).")
  set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
endif()

include(FetchContent)

# Use a block to scope BUILD_TESTING variable for third-party dependencies
# This prevents Eigen tests from being added to ctest
block(SCOPE_FOR VARIABLES)
  set(BUILD_TESTING OFF)
  set(EIGEN_BUILD_TESTING OFF)
  set(EIGEN_TESTING OFF)
  
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(eigen)

  # nlohmann/json for generator JSON schema handling
  # Note: v3.12.0 required to match LiteRT-LM's nlohmann/json version
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
  )
  FetchContent_MakeAvailable(nlohmann_json)
endblock()

#
# OpenTelemetry dependency (native builds only).
#
# cortext links against the OpenTelemetry API only. Applications are responsible
# for installing SDK providers/exporters as needed.
#
if(NOT EMSCRIPTEN)
  # Temporarily disable BUILD_TESTING while configuring the dependency to avoid
  # adding its tests into this build, then restore.
  set(_cortext_saved_build_testing "${BUILD_TESTING}")
  set(BUILD_TESTING OFF CACHE BOOL "Disable tests while configuring deps" FORCE)

  # Keep the dependency lightweight (API only is linked, but the upstream build
  # still supports optional components).
  set(WITH_EXAMPLES OFF CACHE BOOL "Disable OTel examples" FORCE)
  set(WITH_OTLP_FILE ON CACHE BOOL "Enable OTLP file exporters" FORCE)
  set(WITH_OTLP_GRPC ON CACHE BOOL "Enable OTLP gRPC exporters" FORCE)
  set(WITH_PROMETHEUS OFF CACHE BOOL "Disable Prometheus exporter" FORCE)
  set(WITH_JAEGER OFF CACHE BOOL "Disable Jaeger exporter" FORCE)
  set(WITH_ZIPKIN OFF CACHE BOOL "Disable Zipkin exporter" FORCE)

  FetchContent_Declare(
    opentelemetry_cpp
    GIT_REPOSITORY https://github.com/open-telemetry/opentelemetry-cpp.git
    GIT_TAG v1.24.0
  )
  FetchContent_MakeAvailable(opentelemetry_cpp)

  set(BUILD_TESTING "${_cortext_saved_build_testing}" CACHE BOOL "Build tests" FORCE)
endif()


if(NOT EMSCRIPTEN)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(SQLITE3 REQUIRED sqlite3)
endif()

add_library(cortext
    src/store.cpp
    src/store/extension_loader.cpp
    src/store/schema.cpp
    src/store/facts.cpp
    src/processor/operation_context.cpp
    src/signal_processor.cpp
    src/cortext.cpp
    src/core/thread_config.cpp
    src/capi.cpp
    src/telemetry/telemetry.cpp
    src/encoder/imagebind.cpp
    src/data/centroids.cpp
    src/data/embedded_centroid_vectors.cpp
    src/operations/centroids.cpp
    src/operations/focus.cpp
    src/operations/recent_context.cpp
    src/operations/sensitivity.cpp
    src/operations/uncertainty.cpp
    src/operations/threshold.cpp
    src/operations/coherence.cpp
    src/operations/stability.cpp
    src/operations/blend.cpp
    src/operations/metrics.cpp
    src/operations/effective_focus.cpp
    src/operations/focus_spread.cpp
    src/operations/boundary.cpp
    src/operations/memory_strength.cpp
    src/operations/focus_feedback.cpp
    src/operations/sensitivity_feedback.cpp
    src/operations/stability_feedback.cpp
    src/operations/influence.cpp
    src/operations/reconsolidation.cpp
    src/operations/competition.cpp
    src/operations/predictive.cpp
    src/operations/emotion.cpp
    src/operations/working_memory.cpp
    src/operations/metacognitive.cpp
    src/operations/serial_position.cpp
    src/operations/serial_position_apply.cpp
    src/operations/interrupt_gate.cpp
    src/operations/consolidation.cpp
    src/operations/consolidation_cluster.cpp
    src/operations/consolidation_gate.cpp
  src/operations/consolidation_shallow.cpp
  src/operations/label_bank.cpp
    src/operations/consolidation_summarize.cpp
    src/operations/process_extraction_results.cpp
    src/operations/embedding_prediction_error.cpp
    src/operations/precision.cpp
    src/operations/constructive_recall_internal.cpp
    src/operations/meta_learning_internal.cpp
    src/operations/graph_build.cpp
    src/operations/graph_retrieval.cpp
    src/operations/temporal_retrieval.cpp
    src/operations/eviction_ablation.cpp
    src/operations/retrieval_debug_state.cpp
    src/operations/neuromodulators.cpp
    # Phase 4: Knowledge Graph Enhancement
    src/operations/emotion_cascade.cpp
    src/operations/signal_metrics_persistence.cpp
    src/operations/write_gate.cpp
    src/operations/memory_storage.cpp
    src/operations/detect_memory_usage.cpp
    src/operations/synaptic_tagging.cpp
    src/operations/drift_accumulation.cpp
    src/operations/streaming_pacing.cpp
    # Section 4.4: Accumulator operations
    src/operations/accumulator_scores.cpp
    src/operations/accumulator.cpp
    src/operations/accumulator_reset.cpp
    src/operations/spike_bypass.cpp
    # Extractor component
    src/extractor/phi4_extractor.cpp
    src/extractor/gemma_extractor.cpp
    # Summarizer component
    src/summarizer/phi4_summarizer.cpp
    src/summarizer/gemma_summarizer.cpp
    # Deep consolidation LLM backends
    src/deep_llm/deep_llm_factory.cpp
    src/deep_llm/lfm2_llama_backend.cpp
    # JSON schema parser for constrained decoding
    src/generator/json_decoder.cpp
    # Audio preprocessing
    src/audio/gemma_audio.cpp
)

set(CORTEXT_LLAMA_CPP_AVAILABLE OFF)
if(NOT EMSCRIPTEN)
  find_path(CORTEXT_LLAMA_INCLUDE_DIR
    NAMES llama.h
    PATHS /opt/homebrew/include /usr/local/include
  )
  find_library(CORTEXT_LLAMA_LIBRARY
    NAMES llama libllama
    PATHS /opt/homebrew/lib /usr/local/lib
  )
  find_library(CORTEXT_GGML_LIBRARY
    NAMES ggml libggml
    PATHS /opt/homebrew/lib /usr/local/lib
  )
  find_library(CORTEXT_GGML_BASE_LIBRARY
    NAMES ggml-base libggml-base
    PATHS /opt/homebrew/lib /usr/local/lib
  )

  if(CORTEXT_LLAMA_INCLUDE_DIR AND CORTEXT_LLAMA_LIBRARY)
    set(CORTEXT_LLAMA_CPP_AVAILABLE ON)
    target_compile_definitions(cortext PRIVATE
      CORTEXT_ENABLE_LLAMA_CPP=1
      CORTEXT_GGML_BACKEND_HEADER_PATH=\"${CORTEXT_LLAMA_INCLUDE_DIR}/ggml-backend.h\"
      CORTEXT_LLAMA_HEADER_PATH=\"${CORTEXT_LLAMA_INCLUDE_DIR}/llama.h\"
    )
    target_link_libraries(cortext PRIVATE ${CORTEXT_LLAMA_LIBRARY})
    if(CORTEXT_GGML_LIBRARY)
      target_link_libraries(cortext PRIVATE ${CORTEXT_GGML_LIBRARY})
    endif()
    if(CORTEXT_GGML_BASE_LIBRARY)
      target_link_libraries(cortext PRIVATE ${CORTEXT_GGML_BASE_LIBRARY})
    endif()
  else()
    message(STATUS "llama.cpp library not found; GGUF generation backends disabled")
  endif()
endif()

if(CORTEXT_ENABLE_EMBEDDINGGEMMA)
  target_compile_definitions(cortext PUBLIC CORTEXT_ENABLE_EMBEDDINGGEMMA=1)
  target_sources(cortext PRIVATE src/encoder/embeddinggemma.cpp)
  if(CORTEXT_LLAMA_CPP_AVAILABLE)
    target_compile_definitions(cortext PRIVATE CORTEXT_ENABLE_EMBEDDINGGEMMA_LLAMA_CPP=1)
  else()
    message(STATUS "llama.cpp library not found; EmbeddingGemma llama.cpp backend disabled")
  endif()
else()
  set(CORTEXT_DISABLE_SHERPA_ONNX ON CACHE BOOL "" FORCE)
endif()

if(NOT CORTEXT_DISABLE_SHERPA_ONNX)
  target_sources(cortext PRIVATE src/audio/sherpa_onnx.cpp)
endif()
add_library(cortext::cortext ALIAS cortext)

# Public include dirs
target_include_directories(cortext
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite/src
)

# Embed sqlite-vec sources when enabled
set(CORTEXT_EMBED_SOURCES "")
if(CORTEXT_EMBED_VEC)
  list(APPEND CORTEXT_EMBED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-vec/sqlite-vec.c
  )
  # Provide sqlite-vec.h from template into build dir and add includes
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/third_party/sqlite-vec)
  # Populate template variables for sqlite-vec.h
  set(VERSION_MAJOR 0)
  set(VERSION_MINOR 0)
  set(VERSION_PATCH 0)
  set(VERSION "0.0.0")
  set(DATE "")
  set(SOURCE "")
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-vec/sqlite-vec.h.tmpl
    ${CMAKE_BINARY_DIR}/third_party/sqlite-vec/sqlite-vec.h
  )
  target_include_directories(cortext PRIVATE
    ${CMAKE_BINARY_DIR}/third_party/sqlite-vec
  )
  target_compile_definitions(cortext PRIVATE CORTEXT_EMBED_VEC=1 SQLITE_VEC_OMIT_FS=1)
  # Silence non-portable warning noise in embedded third-party C sources
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-vec/sqlite-vec.c
      PROPERTIES COMPILE_OPTIONS "-w"
    )
  endif()
endif()

if(CORTEXT_EMBED_SOURCES)
  target_sources(cortext PRIVATE ${CORTEXT_EMBED_SOURCES})
endif()

if(CORTEXT_EMBED_OBJSTORE)
  set(CORTEXT_OBJSTORE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/objstore.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/backend_registry.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/backend_fs_common.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/backend_portable.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/objstore_vtab.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/objstore_txn.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/object_manager.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/blake3_hash.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/third_party/blake3/blake3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/third_party/blake3/blake3_dispatch.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/third_party/blake3/blake3_portable.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/backend_opfs.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/backend_vfs.c
  )
  if(EMSCRIPTEN)
    # WASM builds rely on OPFS/VFS backends only.
  else()
    list(APPEND CORTEXT_OBJSTORE_SRCS
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/backend_file.c
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/src/backend_sqlite.c
    )
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    list(APPEND CORTEXT_OBJSTORE_SRCS
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/third_party/blake3/blake3_neon.c
    )
  endif()
  target_sources(cortext PRIVATE ${CORTEXT_OBJSTORE_SRCS})
  target_include_directories(cortext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-objstore/third_party/blake3
  )
  target_compile_definitions(cortext PRIVATE
    CORTEXT_EMBED_OBJSTORE=1
    BLAKE3_NO_SSE2=1
    BLAKE3_NO_SSE41=1
    BLAKE3_NO_AVX2=1
    BLAKE3_NO_AVX512=1
  )
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    set_source_files_properties(
      ${CORTEXT_OBJSTORE_SRCS}
      PROPERTIES COMPILE_OPTIONS "-w"
    )
  endif()
endif()

# Dynamic extension loading flag
if(CORTEXT_ENABLE_DYNAMIC_EXTENSIONS AND NOT EMSCRIPTEN)
  target_compile_definitions(cortext PRIVATE CORTEXT_ENABLE_DYNAMIC_EXTENSIONS=1)
endif()

# Ensure extensions are compiled as part of SQLite core to avoid sqlite3_api duplication
target_compile_definitions(cortext PRIVATE SQLITE_CORE=1)

# Link dependencies (SQLite handling differs per target)
target_link_libraries(cortext PUBLIC Eigen3::Eigen)
target_link_libraries(cortext PRIVATE nlohmann_json::nlohmann_json)
if(EMSCRIPTEN)
  # Use a block to prevent SQLite tests from being added to ctest
  block(SCOPE_FOR VARIABLES)
    set(BUILD_TESTING OFF)
    add_subdirectory(third_party/sqlite)
  endblock()
  target_link_libraries(cortext PRIVATE sqlite3)
  target_compile_definitions(cortext PRIVATE SQLITE_OMIT_LOAD_EXTENSION=1)
  target_sources(cortext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/wasm/auto_extensions.cpp)
else()
  target_include_directories(cortext PRIVATE ${SQLITE3_INCLUDE_DIRS})
  target_link_libraries(cortext PRIVATE ${SQLITE3_LIBRARIES})
endif()

#
# OpenTelemetry linkage (native only): API-only.
#
if(NOT EMSCRIPTEN)
  # opentelemetry-cpp headers can emit some warnings on AppleClang/Clang.
  # When CORTEXT_WARNINGS_AS_ERRORS=ON, those become hard errors. Suppress only
  # for the translation unit that includes OTel headers.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/telemetry.cpp
      PROPERTIES
        COMPILE_OPTIONS "-Wno-gnu-zero-variadic-macro-arguments;-Wno-deprecated-declarations"
    )
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/telemetry.cpp
      PROPERTIES
        COMPILE_OPTIONS "-Wno-deprecated-declarations"
    )
  endif()
  target_link_libraries(cortext PRIVATE opentelemetry-cpp::api)
endif()

# Optional ONNX Runtime integration for ImageBind embedder
if(NOT CORTEXT_DISABLE_IMAGEBIND_ORT AND NOT EMSCRIPTEN)
  include(ExternalProject)
  find_package(ZLIB REQUIRED)

  # Expose this feature macro to dependents (tests, downstream consumers).
  target_compile_definitions(cortext PUBLIC CORTEXT_ENABLE_IMAGEBIND_ORT=1)

  # Build ONNX Runtime from vendored source when needed.
  set(CORTEXT_ORT_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
  if(NOT CORTEXT_ORT_BUILD_TYPE)
    set(CORTEXT_ORT_BUILD_TYPE "Release")
  endif()

  set(CORTEXT_ORT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/cmake")
  set(CORTEXT_ORT_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/onnxruntime-build")
  set(CORTEXT_ORT_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/onnxruntime-install")
  set(CORTEXT_ORT_LIB_PATH
      "${CORTEXT_ORT_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}onnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}")

  set(CORTEXT_ORT_USE_COREML OFF)
  if(APPLE AND CORTEXT_ENABLE_ORT_COREML)
    set(CORTEXT_ORT_USE_COREML ON)
  endif()

  # Use headers from the ORT install tree.
  target_include_directories(cortext PRIVATE
    ${CORTEXT_ORT_INSTALL_DIR}/include
  )

  # If a prebuilt library exists, prefer it. Otherwise build/install into the
  # build tree.
  if(NOT EXISTS "${CORTEXT_ORT_LIB_PATH}")
    ExternalProject_Add(onnxruntime_ep
      SOURCE_DIR "${CORTEXT_ORT_SOURCE_DIR}"
      BINARY_DIR "${CORTEXT_ORT_BINARY_DIR}"
      INSTALL_DIR "${CORTEXT_ORT_INSTALL_DIR}"
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CORTEXT_ORT_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${CORTEXT_ORT_INSTALL_DIR}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        # Prevent ORT from picking up Homebrew/system Protobuf configs,
        # which can reference Abseil targets we aren't providing here.
        -DCMAKE_DISABLE_FIND_PACKAGE_Protobuf=ON
        -DCMAKE_DISABLE_FIND_PACKAGE_protobuf=ON
        # Prevent ORT from picking up Homebrew/system ONNX config, which can
        # pull in system Protobuf transitively.
        -DCMAKE_DISABLE_FIND_PACKAGE_ONNX=ON
        -DCMAKE_DISABLE_FIND_PACKAGE_onnx=ON
        # Force ORT to use its bundled flatbuffers instead of system/homebrew.
        -DCMAKE_DISABLE_FIND_PACKAGE_Flatbuffers=ON
        -DCMAKE_DISABLE_FIND_PACKAGE_flatbuffers=ON
        -Donnxruntime_BUILD_SHARED_LIB=ON
        -Donnxruntime_BUILD_UNIT_TESTS=OFF
        -Donnxruntime_ENABLE_PYTHON=OFF
        -Donnxruntime_USE_COREML=${CORTEXT_ORT_USE_COREML}
        -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
      BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CORTEXT_ORT_BUILD_TYPE} -- -j8
      BUILD_BYPRODUCTS "${CORTEXT_ORT_LIB_PATH}"
      USES_TERMINAL_CONFIGURE TRUE
      USES_TERMINAL_BUILD TRUE
      USES_TERMINAL_INSTALL TRUE
    )
  else()
    add_custom_target(onnxruntime_ep)
  endif()

  add_library(onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${CORTEXT_ORT_LIB_PATH}")
  add_dependencies(onnxruntime onnxruntime_ep)

  target_link_libraries(cortext PRIVATE onnxruntime ZLIB::ZLIB)
  add_dependencies(cortext onnxruntime_ep)
  if(APPLE AND CORTEXT_ENABLE_ORT_COREML)
    set(CORTEXT_ORT_COREML_PROVIDER_LIB
        "${CORTEXT_ORT_BINARY_DIR}/lib/libonnxruntime_providers_coreml.a")
    if(EXISTS "${CORTEXT_ORT_COREML_PROVIDER_LIB}")
      target_link_libraries(cortext PRIVATE "${CORTEXT_ORT_COREML_PROVIDER_LIB}")
    endif()
  endif()
endif()

# Optional ONNX Runtime integration for Gemma text generator
# Shares the same ORT build as ImageBind if both are enabled
if(CORTEXT_ENABLE_GEMMA_ORT)
  target_compile_definitions(cortext PUBLIC CORTEXT_ENABLE_GEMMA_ORT=1)

  if(CORTEXT_DISABLE_IMAGEBIND_ORT)
    # Build ORT if not already built for ImageBind
    include(ExternalProject)
    find_package(ZLIB REQUIRED)

    set(CORTEXT_ORT_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
    if(NOT CORTEXT_ORT_BUILD_TYPE)
      set(CORTEXT_ORT_BUILD_TYPE "Release")
    endif()

    set(CORTEXT_ORT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/cmake")
    set(CORTEXT_ORT_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/onnxruntime-build")
    set(CORTEXT_ORT_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/onnxruntime-install")
    set(CORTEXT_ORT_LIB_PATH
        "${CORTEXT_ORT_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}onnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}")

    set(CORTEXT_ORT_USE_COREML OFF)
    if(APPLE AND CORTEXT_ENABLE_ORT_COREML)
      set(CORTEXT_ORT_USE_COREML ON)
    endif()

    target_include_directories(cortext PRIVATE
      ${CORTEXT_ORT_INSTALL_DIR}/include
    )

    if(NOT EXISTS "${CORTEXT_ORT_LIB_PATH}")
      ExternalProject_Add(onnxruntime_ep
        SOURCE_DIR "${CORTEXT_ORT_SOURCE_DIR}"
        BINARY_DIR "${CORTEXT_ORT_BINARY_DIR}"
        INSTALL_DIR "${CORTEXT_ORT_INSTALL_DIR}"
        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=${CORTEXT_ORT_BUILD_TYPE}
          -DCMAKE_INSTALL_PREFIX=${CORTEXT_ORT_INSTALL_DIR}
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5
          -DCMAKE_DISABLE_FIND_PACKAGE_Protobuf=ON
          -DCMAKE_DISABLE_FIND_PACKAGE_protobuf=ON
          -DCMAKE_DISABLE_FIND_PACKAGE_ONNX=ON
          -DCMAKE_DISABLE_FIND_PACKAGE_onnx=ON
          -DCMAKE_DISABLE_FIND_PACKAGE_Flatbuffers=ON
          -DCMAKE_DISABLE_FIND_PACKAGE_flatbuffers=ON
          -Donnxruntime_BUILD_SHARED_LIB=ON
          -Donnxruntime_BUILD_UNIT_TESTS=OFF
          -Donnxruntime_ENABLE_PYTHON=OFF
          -Donnxruntime_USE_COREML=${CORTEXT_ORT_USE_COREML}
          -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CORTEXT_ORT_BUILD_TYPE} -- -j8
        BUILD_BYPRODUCTS "${CORTEXT_ORT_LIB_PATH}"
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE
      )
    else()
      add_custom_target(onnxruntime_ep)
    endif()

    add_library(onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${CORTEXT_ORT_LIB_PATH}")
    add_dependencies(onnxruntime onnxruntime_ep)

    target_link_libraries(cortext PRIVATE onnxruntime ZLIB::ZLIB)
    add_dependencies(cortext onnxruntime_ep)
    if(APPLE AND CORTEXT_ENABLE_ORT_COREML)
      set(CORTEXT_ORT_COREML_PROVIDER_LIB
          "${CORTEXT_ORT_BINARY_DIR}/lib/libonnxruntime_providers_coreml.a")
      if(EXISTS "${CORTEXT_ORT_COREML_PROVIDER_LIB}")
        target_link_libraries(cortext PRIVATE "${CORTEXT_ORT_COREML_PROVIDER_LIB}")
      endif()
    endif()
  endif()
endif()

# EmbeddingGemma ONNX encoder (uses ORT + SentencePiece from LiteRT-LM)
if(CORTEXT_ENABLE_EMBEDDINGGEMMA)
  set(CORTEXT_SENTENCEPIECE_INCLUDE_DIR "${CMAKE_BINARY_DIR}/bazel-output/external/sentencepiece")
  set(CORTEXT_SENTENCEPIECE_PROTO_DIR "${CORTEXT_SENTENCEPIECE_INCLUDE_DIR}/builtin_pb")
  set(CORTEXT_SENTENCEPIECE_LIB_DIR "${CMAKE_BINARY_DIR}/bazel-output/execroot/litert_lm/bazel-out/darwin_arm64-opt/bin/external/sentencepiece")
  set(CORTEXT_SENTENCEPIECE_PROCESSOR_LIB "${CORTEXT_SENTENCEPIECE_LIB_DIR}/libsentencepiece_processor.a")
  set(CORTEXT_SENTENCEPIECE_MODEL_PROTO_LIB "${CORTEXT_SENTENCEPIECE_LIB_DIR}/libsentencepiece_model_proto.a")
  set(CORTEXT_SENTENCEPIECE_PROTO_LIB "${CORTEXT_SENTENCEPIECE_LIB_DIR}/libsentencepiece_proto.a")

  foreach(sp_path
      "${CORTEXT_SENTENCEPIECE_INCLUDE_DIR}"
      "${CORTEXT_SENTENCEPIECE_PROTO_DIR}"
      "${CORTEXT_SENTENCEPIECE_PROCESSOR_LIB}"
      "${CORTEXT_SENTENCEPIECE_MODEL_PROTO_LIB}"
      "${CORTEXT_SENTENCEPIECE_PROTO_LIB}")
    if(NOT EXISTS "${sp_path}")
      message(FATAL_ERROR "EmbeddingGemma requires SentencePiece artifact: ${sp_path}")
    endif()
  endforeach()

  target_include_directories(cortext PRIVATE
    "${CORTEXT_SENTENCEPIECE_INCLUDE_DIR}"
    "${CORTEXT_SENTENCEPIECE_PROTO_DIR}"
  )
  target_link_libraries(cortext PRIVATE
    "${CORTEXT_SENTENCEPIECE_PROCESSOR_LIB}"
    "${CORTEXT_SENTENCEPIECE_MODEL_PROTO_LIB}"
    "${CORTEXT_SENTENCEPIECE_PROTO_LIB}"
    protobuf::libprotobuf
  )
  if(TARGET absl::status)
    target_link_libraries(cortext PRIVATE absl::status)
  endif()
  if(TARGET absl::statusor)
    target_link_libraries(cortext PRIVATE absl::statusor)
  endif()

  if((NOT CORTEXT_DISABLE_IMAGEBIND_ORT AND NOT EMSCRIPTEN) OR CORTEXT_ENABLE_GEMMA_ORT)
    target_compile_definitions(cortext PUBLIC CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT=1)
  endif()
endif()

# Optional onnxruntime-genai (OGA) integration for Phi-4 Extractor/Summarizer
# Enabled by default; set CORTEXT_DISABLE_OGA=ON to disable
if(CORTEXT_DISABLE_OGA)
  target_compile_definitions(cortext PUBLIC CORTEXT_DISABLE_OGA=1)
endif()

if(NOT CORTEXT_DISABLE_OGA AND NOT EMSCRIPTEN)
  # On Apple Silicon, explicitly set CMAKE_OSX_ARCHITECTURES if not already set
  # This is needed for OGA's ortlib.cmake to detect the correct architecture
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" AND NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architecture" FORCE)
  endif()

  # Use the same ORT library as ImageBind when enabled
  # This avoids version mismatch issues between downloaded ORT and built-from-source ORT
  if(NOT CORTEXT_DISABLE_IMAGEBIND_ORT)
    # Use the ORT built from third_party/onnxruntime source (for ImageBind)
    # OGA expects headers directly in include/, but ORT installs to include/onnxruntime/
    # Copy headers to the expected location if not already there
    if(NOT EXISTS "${CORTEXT_ORT_INSTALL_DIR}/include/onnxruntime_c_api.h")
      file(GLOB ORT_HEADERS_TO_COPY "${CORTEXT_ORT_INSTALL_DIR}/include/onnxruntime/*.h")
      if(ORT_HEADERS_TO_COPY)
        file(COPY ${ORT_HEADERS_TO_COPY} DESTINATION "${CORTEXT_ORT_INSTALL_DIR}/include/")
        message(STATUS "Copied ORT headers to ${CORTEXT_ORT_INSTALL_DIR}/include/")
      endif()
    endif()
    set(ORT_HOME "${CORTEXT_ORT_INSTALL_DIR}" CACHE PATH "ONNX Runtime home" FORCE)
    message(STATUS "Using ONNX Runtime from ImageBind build: ${ORT_HOME}")
  else()
    # Download ORT from GitHub releases (CPU-only) when ImageBind ORT is disabled
    set(ORT_VERSION "1.23.0")
    if(APPLE)
      if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(ORT_PACKAGE "onnxruntime-osx-arm64-${ORT_VERSION}")
      else()
        set(ORT_PACKAGE "onnxruntime-osx-x86_64-${ORT_VERSION}")
      endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(ORT_PACKAGE "onnxruntime-linux-aarch64-${ORT_VERSION}")
      else()
        set(ORT_PACKAGE "onnxruntime-linux-x64-${ORT_VERSION}")
      endif()
    else()
      message(FATAL_ERROR "Unsupported platform for OGA")
    endif()

    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PACKAGE}.tgz")
    set(ORT_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/ort-download")

    if(NOT EXISTS "${ORT_DOWNLOAD_DIR}/${ORT_PACKAGE}")
      message(STATUS "Downloading ONNX Runtime ${ORT_VERSION}...")
      file(DOWNLOAD ${ORT_URL} "${ORT_DOWNLOAD_DIR}/ort.tgz" SHOW_PROGRESS)
      file(ARCHIVE_EXTRACT INPUT "${ORT_DOWNLOAD_DIR}/ort.tgz" DESTINATION "${ORT_DOWNLOAD_DIR}")
    endif()

    # Set ORT_HOME for OGA - must be set as CACHE FORCE to propagate to subdirectory
    set(ORT_HOME "${ORT_DOWNLOAD_DIR}/${ORT_PACKAGE}" CACHE PATH "ONNX Runtime home" FORCE)
    message(STATUS "Using downloaded ONNX Runtime from: ${ORT_HOME}")
  endif()

  # Save and disable BUILD_TESTING for OGA subdirectory
  set(_cortext_saved_testing_for_oga "${BUILD_TESTING}")

  # Configure OGA build options (must be set as CACHE FORCE before add_subdirectory)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
  set(ENABLE_MODEL_BENCHMARK OFF CACHE BOOL "" FORCE)
  # CPU-only build, no CUDA/DML/ROCm
  set(USE_CUDA OFF CACHE BOOL "" FORCE)
  set(USE_DML OFF CACHE BOOL "" FORCE)
  set(USE_ROCM OFF CACHE BOOL "" FORCE)
  # Enable llguidance for JSON schema constrained decoding
  set(USE_GUIDANCE ON CACHE BOOL "" FORCE)

  add_subdirectory(third_party/onnxruntime-genai EXCLUDE_FROM_ALL)

  # Restore BUILD_TESTING
  set(BUILD_TESTING "${_cortext_saved_testing_for_oga}" CACHE BOOL "Build tests" FORCE)

  # Add OGA include directories
  target_include_directories(cortext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-genai/src
  )
  target_link_libraries(cortext PRIVATE onnxruntime-genai)
endif()

# Optional sherpa-onnx integration for ASR/TTS
# Enabled by default; set CORTEXT_DISABLE_SHERPA_ONNX=ON to disable
if(CORTEXT_DISABLE_SHERPA_ONNX)
  target_compile_definitions(cortext PUBLIC CORTEXT_DISABLE_SHERPA_ONNX=1)
endif()

if(NOT CORTEXT_DISABLE_SHERPA_ONNX AND NOT EMSCRIPTEN)
  # Configure sherpa-onnx to be a lightweight library build.
  set(SHERPA_ONNX_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_CHECK OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_WEBSOCKET OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_PORTAUDIO OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_BINARY OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_TTS ON CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_C_API ON CACHE BOOL "" FORCE)

  # Prefer sherpa-onnx to download its own ORT if our build isn't ready yet.
  if(DEFINED CORTEXT_ORT_LIB_PATH AND EXISTS "${CORTEXT_ORT_LIB_PATH}")
    set(ENV{SHERPA_ONNXRUNTIME_INCLUDE_DIR} "${CORTEXT_ORT_INSTALL_DIR}/include")
    set(ENV{SHERPA_ONNXRUNTIME_LIB_DIR} "${CORTEXT_ORT_INSTALL_DIR}/lib")
    set(SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE ON CACHE BOOL "" FORCE)
  else()
    set(SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
  endif()

  add_subdirectory(third_party/sherpa-onnx EXCLUDE_FROM_ALL)
  target_include_directories(cortext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sherpa-onnx)
  target_link_libraries(cortext PRIVATE sherpa-onnx-cxx-api)
  target_compile_definitions(cortext PUBLIC CORTEXT_ENABLE_SHERPA_ONNX=1)
endif()

# Optional LiteRT-LM integration for Gemma Extractor/Summarizer
# Enabled by default; set CORTEXT_DISABLE_LITERT=ON to disable
if(CORTEXT_DISABLE_LITERT)
  target_compile_definitions(cortext PUBLIC CORTEXT_DISABLE_LITERT=1)
endif()

if(NOT CORTEXT_DISABLE_LITERT AND NOT EMSCRIPTEN)
  include(ExternalProject)

  target_sources(cortext PRIVATE
    src/extractor/json_schema_constraint.cpp
  )

  # Detect platform for Bazel config
  if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
      set(LITERT_BAZEL_CONFIG "macos_arm64")
    else()
      set(LITERT_BAZEL_CONFIG "macos_x86_64")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(LITERT_BAZEL_CONFIG "linux_arm64")
    else()
      set(LITERT_BAZEL_CONFIG "linux_x86_64")
    endif()
  else()
    message(FATAL_ERROR "Unsupported platform for LiteRT-LM")
  endif()

  set(LITERT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-lm")
  set(LITERT_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/litert-lm-install")
  set(LITERT_LIB_PATH "${LITERT_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}litert_lm_cpp${CMAKE_SHARED_LIBRARY_SUFFIX}")

  # Check for bazel/bazelisk
  find_program(BAZEL_EXECUTABLE NAMES bazelisk bazel)
  if(NOT BAZEL_EXECUTABLE)
    message(FATAL_ERROR "Bazel not found. Install with: brew install bazelisk")
  endif()

  # Append custom BUILD target for C++ shared library if not already added
  set(LITERT_CUSTOM_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/litert-lm-BUILD.bazel")
  set(LITERT_ROOT_BUILD "${LITERT_SOURCE_DIR}/BUILD")

  if(NOT EXISTS "${LITERT_LIB_PATH}")
    ExternalProject_Add(litert_ep
      SOURCE_DIR "${LITERT_SOURCE_DIR}"
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E echo "Appending custom BUILD target..." &&
        ${CMAKE_COMMAND} -DLITERT_CUSTOM_BUILD=${LITERT_CUSTOM_BUILD}
          -DLITERT_ROOT_BUILD=${LITERT_ROOT_BUILD}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/litert_append_build.cmake
      BUILD_COMMAND ${BAZEL_EXECUTABLE}
        --output_base=${CMAKE_BINARY_DIR}/bazel-output
        build //:liblitert_lm_cpp.so
        //runtime/engine:litert_lm_shared
        --config=${LITERT_BAZEL_CONFIG}
        --compilation_mode=opt
        --verbose_failures
      BUILD_IN_SOURCE TRUE
      INSTALL_COMMAND ${CMAKE_COMMAND}
        -DLITERT_SOURCE_DIR=${LITERT_SOURCE_DIR}
        -DLITERT_INSTALL_DIR=${LITERT_INSTALL_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/litert_install.cmake
      BUILD_BYPRODUCTS "${LITERT_LIB_PATH}"
      USES_TERMINAL_BUILD TRUE
    )
  else()
    add_custom_target(litert_ep)
  endif()

  add_library(litert_lm SHARED IMPORTED GLOBAL)
  set_target_properties(litert_lm PROPERTIES IMPORTED_LOCATION "${LITERT_LIB_PATH}")
  add_dependencies(litert_lm litert_ep)

  target_include_directories(cortext PRIVATE ${LITERT_INSTALL_DIR}/include)
  # PRIVATE link - LiteRT-LM symbols are hidden via visibility settings
  target_link_libraries(cortext PRIVATE litert_lm)
  add_dependencies(cortext litert_ep)

  # Set RPATH for runtime library discovery
  if(APPLE)
    set_property(TARGET cortext APPEND PROPERTY BUILD_RPATH
      "@loader_path/../third_party/litert-lm-install/lib"
    )
    list(APPEND CMAKE_BUILD_RPATH "${LITERT_INSTALL_DIR}/lib")
  elseif(UNIX)
    set_property(TARGET cortext APPEND PROPERTY BUILD_RPATH
      "$ORIGIN/../third_party/litert-lm-install/lib"
    )
  endif()

  # Ensure LiteRT accelerator dylibs are present in the install lib dir on macOS.
  if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
      set(LITERT_ACCEL_DIR "${LITERT_SOURCE_DIR}/prebuilt/macos_arm64")
    else()
      set(LITERT_ACCEL_DIR "${LITERT_SOURCE_DIR}/prebuilt/macos_x86_64")
    endif()
    file(GLOB LITERT_ACCEL_LIBS "${LITERT_ACCEL_DIR}/*.dylib")
    if(NOT EMSCRIPTEN)
      list(FILTER LITERT_ACCEL_LIBS EXCLUDE REGEX "WebGpuAccelerator")
    endif()
    set(LITERT_ACCEL_CLEANUP_COMMANDS "")
    if(NOT EMSCRIPTEN)
      list(APPEND LITERT_ACCEL_CLEANUP_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E rm -f
          "${LITERT_INSTALL_DIR}/lib/libLiteRtWebGpuAccelerator.dylib"
      )
    endif()
    if(LITERT_ACCEL_LIBS)
      set(LITERT_ACCEL_SENTINEL "${LITERT_INSTALL_DIR}/lib/.litert_accel_copied")
      add_custom_command(
        OUTPUT "${LITERT_ACCEL_SENTINEL}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LITERT_INSTALL_DIR}/lib"
        ${LITERT_ACCEL_CLEANUP_COMMANDS}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LITERT_ACCEL_LIBS} "${LITERT_INSTALL_DIR}/lib/"
        COMMAND ${CMAKE_COMMAND} -E touch "${LITERT_ACCEL_SENTINEL}"
        DEPENDS ${LITERT_ACCEL_LIBS}
      )
      add_custom_target(litert_accel ALL DEPENDS "${LITERT_ACCEL_SENTINEL}")
      add_dependencies(cortext litert_accel)
    endif()
  endif()
endif()

# Use C++20 features
target_compile_features(cortext PUBLIC cxx_std_20)
target_compile_definitions(cortext PUBLIC CORTEXT_VERSION="${PROJECT_VERSION}")

# Set hidden visibility by default to avoid symbol collisions with dependencies
# (e.g., protobuf from LiteRT-LM vs OpenTelemetry). Only symbols marked with
# CORTEXT_EXPORT will be visible to consumers.
# Only apply when building as shared library - static libs don't have visibility.
if(NOT EMSCRIPTEN AND BUILD_SHARED_LIBS)
  set_target_properties(cortext PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  target_compile_definitions(cortext PRIVATE CORTEXT_BUILDING_SHARED)
endif()

# Sanitizer instrumentation (must be applied after target is defined)
if(CORTEXT_ENABLE_ASAN AND NOT EMSCRIPTEN)
  target_compile_options(cortext PUBLIC -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(cortext PUBLIC -fsanitize=address)
  message(STATUS "AddressSanitizer enabled for cortext")
endif()

if(CORTEXT_ENABLE_UBSAN AND NOT EMSCRIPTEN)
  target_compile_options(cortext PUBLIC -fsanitize=undefined)
  target_link_options(cortext PUBLIC -fsanitize=undefined)
  message(STATUS "UndefinedBehaviorSanitizer enabled for cortext")
endif()

if(CORTEXT_ENABLE_MSAN AND NOT EMSCRIPTEN)
  target_compile_options(cortext PUBLIC -fsanitize=memory -fno-omit-frame-pointer)
  target_link_options(cortext PUBLIC -fsanitize=memory)
  message(STATUS "MemorySanitizer enabled for cortext")
endif()

if(BUILD_TESTING)
  target_compile_definitions(cortext PRIVATE CORTEXT_TESTING=1)
endif()

if(EMSCRIPTEN)
  target_compile_options(cortext PRIVATE
    -O3
    -s WASM=1
    $<IF:$<BOOL:${CORTEXT_WARNINGS_AS_ERRORS}>, -Werror, >
  )
  target_link_options(cortext PRIVATE
    -O3
    -s WASM=1
    $<IF:$<BOOL:${CORTEXT_WASM_EXPORT_ALL}>, -s EXPORT_ALL=1, >
    -s ALLOW_MEMORY_GROWTH=1
    -s MODULARIZE=1
    -s EXPORT_ES6=1
  )
elseif(MSVC)
  target_compile_options(cortext PRIVATE /W4 $<IF:$<BOOL:${CORTEXT_WARNINGS_AS_ERRORS}>,/WX,>)
else()
  target_compile_options(cortext PRIVATE -Wall -Wextra -Wpedantic $<IF:$<BOOL:${CORTEXT_WARNINGS_AS_ERRORS}>, -Werror, >)
endif()

include(GNUInstallDirs)

if(NOT EMSCRIPTEN)
  install(TARGETS cortext
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT cortext
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT cortext
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT cortext
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT cortext
  )
endif()

if(CORTEXT_BUILD_NODE_BINDINGS AND NOT EMSCRIPTEN)
  if(NOT CORTEXT_NODE_EXECUTABLE)
    find_program(_cortext_node_executable NAMES node REQUIRED)
    set(CORTEXT_NODE_EXECUTABLE "${_cortext_node_executable}" CACHE FILEPATH
      "Path to the node executable used for Node.js addon builds" FORCE
    )
  endif()

  if(NOT CORTEXT_NODE_INCLUDE_DIR)
    execute_process(
      COMMAND "${CORTEXT_NODE_EXECUTABLE}" -p "require('path').dirname(require('path').dirname(process.execPath))"
      OUTPUT_VARIABLE CORTEXT_NODE_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
    set(CORTEXT_NODE_INCLUDE_DIR "${CORTEXT_NODE_PREFIX}/include/node")
  endif()

  if(NOT EXISTS "${CORTEXT_NODE_INCLUDE_DIR}/node_api.h")
    message(FATAL_ERROR "Node.js headers not found under ${CORTEXT_NODE_INCLUDE_DIR}")
  endif()

  add_library(cortext_node MODULE
    bindings/javascript/src/addon.cpp
  )
  target_compile_features(cortext_node PRIVATE cxx_std_20)
  target_include_directories(cortext_node PRIVATE
    "${CORTEXT_NODE_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  target_link_libraries(cortext_node PRIVATE cortext)
  target_compile_definitions(cortext_node PRIVATE NAPI_VERSION=8)
  set_target_properties(cortext_node PROPERTIES
    PREFIX ""
    OUTPUT_NAME "cortext"
    SUFFIX ".node"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/javascript"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/javascript"
  )

  if(APPLE)
    target_link_options(cortext_node PRIVATE "-undefined" "dynamic_lookup")
    set_property(TARGET cortext_node APPEND PROPERTY BUILD_RPATH
      "@loader_path/../.."
    )
  elseif(UNIX)
    set_property(TARGET cortext_node APPEND PROPERTY BUILD_RPATH
      "$ORIGIN/../.."
    )
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(CORTEXT_BUILD_EXAMPLES AND NOT EMSCRIPTEN)
  add_subdirectory(examples)
  add_subdirectory(tools)
endif()
