cmake_minimum_required(VERSION 3.16)
project(cortext VERSION 1.3.1 LANGUAGES C CXX)

set(CORTEXT_TARGET_IOS OFF)
if(IOS OR CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_OSX_SYSROOT MATCHES "iphone")
  set(CORTEXT_TARGET_IOS ON)
endif()

set(CORTEXT_TARGET_AARCH64 OFF)
if(CMAKE_OSX_ARCHITECTURES MATCHES "(^|;)arm64($|;)")
  set(CORTEXT_TARGET_AARCH64 ON)
elseif(CMAKE_OSX_ARCHITECTURES)
  set(CORTEXT_TARGET_AARCH64 OFF)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(CORTEXT_TARGET_AARCH64 ON)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD STREQUAL "")
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTING "Build tests" ON)
option(CORTEXT_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(CORTEXT_BUILD_EXAMPLES "Build examples" OFF)
option(CORTEXT_BUILD_TOOLS "Build command-line tools" OFF)
option(CORTEXT_BUILD_NODE_BINDINGS "Build Node.js addon bindings" OFF)
option(BUILD_WASM "Build WebAssembly version" OFF)
option(CORTEXT_WASM_EXPORT_ALL "Export all functions in WASM build" OFF)
option(CORTEXT_EMBED_VEC "Embed sqlite-vec extension" ON)
option(CORTEXT_EMBED_OBJSTORE "Embed sqlite-objstore extension" ON)
option(CORTEXT_ENABLE_DYNAMIC_EXTENSIONS "Enable dynamic extension loading fallback" ON)
option(CORTEXT_ENABLE_AUDIO "Build Cortext audio ingestion and embedding support" ON)
option(CORTEXT_ENABLE_IMAGE "Build Cortext image ingestion and embedding support" ON)
option(CORTEXT_ALLOW_UNSUPPORTED_TEXT_ONLY_BUILD
       "Allow unsupported builds with Cortext audio or image support disabled" OFF)
option(CORTEXT_FETCH_GGML
       "Fetch and build bundled ggml when CORTEXT_GGML_SOURCE_DIR is unset" ON)
option(CORTEXT_USE_SYSTEM_GGML
       "Use preinstalled ggml headers/libraries instead of bundled ggml" OFF)
# Default ON: link Git AIST shards into libcortext; assemble full GGUF at load.
# Opt out with -DCORTEXT_EMBED_AIST_MODEL=OFF (thin native + external model path).
# MSVC cannot .incbin shards — configure fails unless you opt out.
set(_cortext_embed_aist_default ON)
if(MSVC OR EMSCRIPTEN OR BUILD_WASM)
  # WASM uses CORTEXT_WASM_PRELOAD_MODEL_ASSETS_DIR; MSVC needs Clang or OFF.
  set(_cortext_embed_aist_default OFF)
endif()
option(CORTEXT_EMBED_AIST_MODEL
       "Link AIST model shards into libcortext (default ON; set OFF to opt out)"
       ${_cortext_embed_aist_default})
# Network fetch is a build-time bootstrap when shards/embed are unavailable.
# MSVC cannot .incbin, so default fetch ON there so a stock MSVC configure can
# still obtain a full GGUF for runtime discovery.
set(_cortext_fetch_aist_default OFF)
if(MSVC)
  set(_cortext_fetch_aist_default ON)
endif()
option(CORTEXT_FETCH_AIST_MODEL
       "Download the AIST GGUF model during the build when local assets are missing"
       ${_cortext_fetch_aist_default})
option(CORTEXT_DISABLE_OPENTELEMETRY "Disable OpenTelemetry integration" OFF)
option(CORTEXT_OPENTELEMETRY_EXPORTERS
       "Build optional OpenTelemetry exporter dependencies" OFF)
option(CORTEXT_EXPERIMENT_HOOKS
       "Enable experimental environment-variable ablation hooks" OFF)
set(CORTEXT_GGML_SOURCE_DIR "" CACHE PATH
    "Optional local ggml checkout")
set(CORTEXT_GGML_GIT_REPOSITORY "https://github.com/ggml-org/ggml.git"
    CACHE STRING "Repository used when CORTEXT_FETCH_GGML=ON")
set(CORTEXT_GGML_GIT_TAG "v0.9.5"
    CACHE STRING "Commit/tag used when CORTEXT_FETCH_GGML=ON")
set(CORTEXT_GGML_OPENMP_DEFAULT ON)
if(BUILD_SHARED_LIBS)
  set(CORTEXT_GGML_OPENMP_DEFAULT OFF)
endif()
option(CORTEXT_GGML_OPENMP
       "Enable OpenMP in the bundled ggml backend" ${CORTEXT_GGML_OPENMP_DEFAULT})
set(CORTEXT_WASM_PRELOAD_MODEL_ASSETS_DIR "" CACHE PATH
    "Optional model asset root to embed into the browser WASM bundle at /models")
set(CORTEXT_MODEL_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models" CACHE PATH
    "Directory where Cortext build-time model assets are stored")
set(CORTEXT_AIST_MODEL_QUANT "q8_0" CACHE STRING
    "AIST GGUF quantization to download by default: q8_0, q5_1, or all")
set_property(CACHE CORTEXT_AIST_MODEL_QUANT PROPERTY STRINGS q8_0 q5_1 all)

if(DEFINED CORTEXT_DISABLE_AUDIO AND CORTEXT_DISABLE_AUDIO)
  set(CORTEXT_ENABLE_AUDIO OFF CACHE BOOL
      "Build Cortext audio ingestion and embedding support" FORCE)
endif()

if(DEFINED CORTEXT_DISABLE_IMAGE AND CORTEXT_DISABLE_IMAGE)
  set(CORTEXT_ENABLE_IMAGE OFF CACHE BOOL
      "Build Cortext image ingestion and embedding support" FORCE)
endif()

if((NOT CORTEXT_ENABLE_AUDIO OR NOT CORTEXT_ENABLE_IMAGE)
   AND NOT CORTEXT_ALLOW_UNSUPPORTED_TEXT_ONLY_BUILD)
  message(FATAL_ERROR
    "Cortext builds require audio and image support by default. "
    "Use -DCORTEXT_ENABLE_AUDIO=OFF or -DCORTEXT_ENABLE_IMAGE=OFF only with "
    "-DCORTEXT_ALLOW_UNSUPPORTED_TEXT_ONLY_BUILD=ON.")
endif()

if(CORTEXT_ALLOW_UNSUPPORTED_TEXT_ONLY_BUILD
   AND (NOT CORTEXT_ENABLE_AUDIO OR NOT CORTEXT_ENABLE_IMAGE))
  message(WARNING
    "Configuring an unsupported Cortext build with audio or image support "
    "disabled. Public audio/image APIs will report build-time disabled errors.")
endif()

# Sanitizer options for catching memory bugs during development
option(CORTEXT_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(CORTEXT_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(CORTEXT_ENABLE_MSAN "Enable MemorySanitizer (Clang only)" OFF)

set(CORTEXT_NODE_EXECUTABLE "" CACHE FILEPATH "Path to the node executable used for Node.js addon builds")
set(CORTEXT_NODE_INCLUDE_DIR "" CACHE PATH "Path to Node.js headers used for Node.js addon builds")
set(CORTEXT_NODE_LIBRARY "" CACHE FILEPATH
    "Path to the Node.js import library used for Windows addon builds")

if(EMSCRIPTEN AND BUILD_TESTING)
  message(STATUS "Disabling tests for WebAssembly builds (not supported).")
  set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
endif()

if(BUILD_WASM AND NOT EMSCRIPTEN)
  message(FATAL_ERROR "BUILD_WASM requires the Emscripten toolchain. Use cmake --preset wasm.")
endif()

include(FetchContent)

if(NOT CORTEXT_AIST_MODEL_QUANT MATCHES "^(q8_0|q5_1|all)$")
  message(FATAL_ERROR
    "CORTEXT_AIST_MODEL_QUANT must be one of: q8_0, q5_1, all")
endif()

set(CORTEXT_EMBEDDED_AIST_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/aist_embed")
if(CORTEXT_EMBED_AIST_MODEL)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR
      "CORTEXT_EMBED_AIST_MODEL is not supported with MSVC (.incbin). "
      "Use Clang/GCC, or set -DCORTEXT_EMBED_AIST_MODEL=OFF and provide "
      "CORTEXT_AIST_MODEL_PATH / models/ at runtime.")
  endif()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  enable_language(ASM)
  set(_cortext_embed_stamp "${CORTEXT_EMBEDDED_AIST_GEN_DIR}/cortext_embed_aist.enabled")
  set(_cortext_embed_asm "${CORTEXT_EMBEDDED_AIST_GEN_DIR}/aist_embedded_blobs.S")
  set(_cortext_embed_meta "${CORTEXT_EMBEDDED_AIST_GEN_DIR}/aist_embedded_meta.h")
  set(_cortext_embed_table "${CORTEXT_EMBEDDED_AIST_GEN_DIR}/aist_embedded_table.inc")
  # Prefer tracked Git shards (no LFS). Full .gguf is never required in the tree.
  file(GLOB _cortext_aist_shard_files
       "${CMAKE_CURRENT_SOURCE_DIR}/models/AIST-87M-GGUF/chunks/AIST-87M_q8_0.gguf.part-*")
  add_custom_command(
    OUTPUT
      "${_cortext_embed_stamp}"
      "${_cortext_embed_asm}"
      "${_cortext_embed_meta}"
      "${_cortext_embed_table}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_embedded_aist.py"
            --out-dir "${CORTEXT_EMBEDDED_AIST_GEN_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_embedded_aist.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/shard_model.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/models/manifest.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/models/mdbr-leaf-ir/vocab.txt"
      ${_cortext_aist_shard_files}
    COMMENT "Linking Git AIST shards into libcortext (assemble at load)"
    VERBATIM
  )
  add_custom_target(cortext_aist_embedded_assets
    DEPENDS "${_cortext_embed_stamp}" "${_cortext_embed_asm}" "${_cortext_embed_meta}")
elseif(CORTEXT_FETCH_AIST_MODEL)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_custom_target(cortext_aist_model ALL
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_aist_model.py"
            --output-dir "${CORTEXT_MODEL_ASSETS_DIR}"
            --quant "${CORTEXT_AIST_MODEL_QUANT}"
    COMMENT "Ensuring Cortext AIST GGUF model is present on disk"
    VERBATIM
  )
endif()

# Temporarily disable third-party tests while configuring dependencies.
set(_cortext_saved_dependency_build_testing "${BUILD_TESTING}")
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_TESTING OFF)
if(DEFINED CMAKE_Fortran_COMPILER)
  set(_cortext_saved_fortran_compiler "${CMAKE_Fortran_COMPILER}")
else()
  set(_cortext_saved_fortran_compiler "__UNSET__")
endif()
set(CMAKE_Fortran_COMPILER "NOTFOUND" CACHE FILEPATH "Disable Fortran while configuring Eigen" FORCE)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)
if(_cortext_saved_fortran_compiler STREQUAL "__UNSET__")
  unset(CMAKE_Fortran_COMPILER CACHE)
else()
  set(CMAKE_Fortran_COMPILER "${_cortext_saved_fortran_compiler}" CACHE FILEPATH "Fortran compiler" FORCE)
endif()

# nlohmann/json for C API JSON serialization and test fixtures.
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(nlohmann_json)

# hnswlib provides the private sparse candidate route used by graph retrieval.
# Keep the exact commit pinned: SQLite and Cortext remain authoritative for
# eligibility and final ranking; this dependency only proposes bounded IDs.
FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
  GIT_TAG d9b3608c83d83b46c96e25088cb1d729b29dcfe9
)
FetchContent_MakeAvailable(hnswlib)
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          "-DHNSWLIB_SOURCE_DIR=${hnswlib_SOURCE_DIR}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PatchHnswlibPrefetch.cmake"
  RESULT_VARIABLE _cortext_hnswlib_patch_result
)
if(NOT _cortext_hnswlib_patch_result EQUAL 0)
  message(FATAL_ERROR "Failed to prepare the pinned hnswlib source")
endif()
set(BUILD_TESTING "${_cortext_saved_dependency_build_testing}")

#
# OpenTelemetry dependency (native builds only).
#
# Cortext links against the OpenTelemetry API by default. Applications remain
# responsible for installing telemetry SDK exporters as needed. Set
# CORTEXT_DISABLE_OPENTELEMETRY=ON for a dependency-free no-op telemetry build.
#
if(NOT EMSCRIPTEN AND NOT CORTEXT_DISABLE_OPENTELEMETRY AND NOT CORTEXT_TARGET_IOS)
  # Temporarily disable BUILD_TESTING while configuring the dependency to avoid
  # adding its tests into this build, then restore.
  set(_cortext_saved_build_testing "${BUILD_TESTING}")
  set(BUILD_TESTING OFF CACHE BOOL "Disable tests while configuring deps" FORCE)

  # Keep the default dependency lightweight: Cortext links only the OTel API.
  # Exporter SDK plumbing belongs to embedding applications and is opt-in.
  if(CORTEXT_OPENTELEMETRY_EXPORTERS)
    set(WITH_API_ONLY OFF CACHE BOOL "Build OpenTelemetry SDK/exporters" FORCE)
  else()
    set(WITH_API_ONLY ON CACHE BOOL "Build only the OpenTelemetry API" FORCE)
  endif()
  set(WITH_BENCHMARK OFF CACHE BOOL "Disable OTel benchmarks" FORCE)
  set(WITH_FUNC_TESTS OFF CACHE BOOL "Disable OTel functional tests" FORCE)
  set(WITH_EXAMPLES OFF CACHE BOOL "Disable OTel examples" FORCE)
  set(WITH_OTLP_FILE ${CORTEXT_OPENTELEMETRY_EXPORTERS}
      CACHE BOOL "Enable OTLP file exporters" FORCE)
  set(WITH_OTLP_GRPC ${CORTEXT_OPENTELEMETRY_EXPORTERS}
      CACHE BOOL "Enable OTLP gRPC exporters" FORCE)
  set(WITH_OTLP_HTTP OFF CACHE BOOL "Disable OTLP HTTP exporters" FORCE)
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

add_subdirectory(third_party/sqlite EXCLUDE_FROM_ALL)

add_library(cortext
    src/store.cpp
    src/store/extension_loader.cpp
    src/store/object_store.cpp
    src/store/schema.cpp
    src/internal/cancellation.cpp
    src/processor/operation_context.cpp
    src/signal_processor.cpp
    src/signal_filter.cpp
    src/cortext.cpp
    src/core/thread_config.cpp
    src/capi.cpp
    src/telemetry/telemetry.cpp
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
    src/operations/serial_position.cpp
    src/operations/serial_position_apply.cpp
    src/operations/interrupt_gate.cpp
    src/operations/consolidation.cpp
    src/operations/consolidation_cluster.cpp
    src/operations/consolidation_gate.cpp
    src/operations/consolidation_shallow.cpp
    src/operations/embedding_prediction_error.cpp
    src/operations/precision.cpp
    src/operations/constructive_recall_internal.cpp
    src/operations/meta_learning_internal.cpp
    src/operations/graph_build.cpp
    src/operations/graph_retrieval.cpp
    src/operations/sparse_retrieval_route_internal.cpp
    src/operations/sparse_retrieval_route_sqlite_internal.cpp
    src/operations/eviction_policy_override.cpp
    src/operations/storage_pressure.cpp
    src/operations/retrieval_trace_state.cpp
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
    src/operations/soft_anchor.cpp
    # Section 4.4: Accumulator operations
    src/operations/accumulator_scores.cpp
    src/operations/accumulator.cpp
    src/operations/accumulator_reset.cpp
    src/operations/spike_bypass.cpp
    # Embedding-model pin (precomputed vectors hard-fail on encoder mismatch)
    src/models/embedding_model_pin.cpp
    # Baked-in AIST model materialization (payload optional via CORTEXT_EMBED_AIST_MODEL)
    src/models/aist_embedded_model.cpp
)

if(CORTEXT_EMBED_AIST_MODEL)
  target_sources(cortext PRIVATE "${_cortext_embed_asm}")
  target_include_directories(cortext PRIVATE "${CORTEXT_EMBEDDED_AIST_GEN_DIR}")
  target_compile_definitions(cortext PRIVATE CORTEXT_HAS_EMBEDDED_AIST_MODEL=1)
  add_dependencies(cortext cortext_aist_embedded_assets)
  set_source_files_properties("${_cortext_embed_asm}" PROPERTIES
    LANGUAGE ASM
    COMPILE_FLAGS "-x assembler-with-cpp")
  message(STATUS "Cortext: AIST shards linked into libcortext (assemble at load; default ON)")
else()
  target_compile_definitions(cortext PRIVATE CORTEXT_HAS_EMBEDDED_AIST_MODEL=0)
  message(STATUS "Cortext: AIST embed opted out (CORTEXT_EMBED_AIST_MODEL=OFF)")
endif()

if(TARGET cortext_aist_model)
  add_dependencies(cortext cortext_aist_model)
endif()

if(WIN32)
  target_link_libraries(cortext PRIVATE shell32)
endif()

function(cortext_resolve_ggml_source_dir input_dir output_dir)
  if(EXISTS "${input_dir}/include/ggml.h"
     AND EXISTS "${input_dir}/include/ggml-backend.h"
     AND EXISTS "${input_dir}/CMakeLists.txt")
    set(${output_dir} "${input_dir}" PARENT_SCOPE)
    return()
  endif()
  message(FATAL_ERROR
    "CORTEXT_GGML_SOURCE_DIR must point to a ggml source directory: "
    "${input_dir}")
endfunction()

set(CORTEXT_GGML_AVAILABLE OFF)
if(NOT EMSCRIPTEN)
  set(_cortext_ggml_header_path "")
  if(CORTEXT_USE_SYSTEM_GGML)
    set(CORTEXT_GGML_INCLUDE_DIR "${CORTEXT_GGML_INCLUDE_DIR}" CACHE PATH
        "Path to the directory containing ggml.h and ggml-backend.h")
    set(CORTEXT_GGML_LIBRARY "${CORTEXT_GGML_LIBRARY}" CACHE FILEPATH
        "Path to libggml")
    set(CORTEXT_GGML_BASE_LIBRARY "${CORTEXT_GGML_BASE_LIBRARY}" CACHE FILEPATH
        "Path to libggml-base")
    set(CORTEXT_GGML_CPU_LIBRARY "${CORTEXT_GGML_CPU_LIBRARY}" CACHE FILEPATH
        "Path to libggml-cpu")
    set(CORTEXT_GGML_BLAS_LIBRARY "${CORTEXT_GGML_BLAS_LIBRARY}" CACHE FILEPATH
        "Path to libggml-blas")
    find_path(CORTEXT_GGML_INCLUDE_DIR
      NAMES ggml.h ggml-backend.h
      PATHS /opt/homebrew/include /usr/local/include
    )
    find_library(CORTEXT_GGML_LIBRARY
      NAMES ggml libggml
      PATHS /opt/homebrew/lib /usr/local/lib
    )
    find_library(CORTEXT_GGML_BASE_LIBRARY
      NAMES ggml-base libggml-base
      PATHS /opt/homebrew/lib /usr/local/lib
    )
    find_library(CORTEXT_GGML_CPU_LIBRARY
      NAMES ggml-cpu libggml-cpu
      PATHS /opt/homebrew/lib /usr/local/lib
    )
    find_library(CORTEXT_GGML_BLAS_LIBRARY
      NAMES ggml-blas libggml-blas
      PATHS /opt/homebrew/lib /usr/local/lib
    )

    if(NOT CORTEXT_GGML_INCLUDE_DIR
       OR NOT CORTEXT_GGML_LIBRARY
       OR NOT CORTEXT_GGML_CPU_LIBRARY)
      message(FATAL_ERROR
        "CORTEXT_USE_SYSTEM_GGML=ON but system ggml was not found. "
        "Set CORTEXT_GGML_INCLUDE_DIR/CORTEXT_GGML_LIBRARY/"
        "CORTEXT_GGML_CPU_LIBRARY, or leave CORTEXT_USE_SYSTEM_GGML=OFF "
        "to build bundled ggml.")
    endif()

    set(CORTEXT_GGML_AVAILABLE ON)
    set(_cortext_ggml_header_path "${CORTEXT_GGML_INCLUDE_DIR}/ggml-backend.h")
    target_include_directories(cortext PRIVATE "${CORTEXT_GGML_INCLUDE_DIR}")
    target_link_libraries(cortext PRIVATE
      ${CORTEXT_GGML_LIBRARY}
      ${CORTEXT_GGML_CPU_LIBRARY}
    )
    if(CORTEXT_GGML_BASE_LIBRARY)
      target_link_libraries(cortext PRIVATE ${CORTEXT_GGML_BASE_LIBRARY})
    endif()
    if(CORTEXT_GGML_BLAS_LIBRARY)
      target_link_libraries(cortext PRIVATE ${CORTEXT_GGML_BLAS_LIBRARY})
    endif()
    if(APPLE AND CORTEXT_GGML_BLAS_LIBRARY)
      find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
      target_link_libraries(cortext PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
    message(STATUS "Using system ggml backend from ${CORTEXT_GGML_INCLUDE_DIR}")
  elseif(CORTEXT_GGML_SOURCE_DIR OR CORTEXT_FETCH_GGML)
    set(_cortext_saved_build_shared_libs "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

    set(GGML_STATIC ON CACHE BOOL "" FORCE)
    set(GGML_OPENMP ${CORTEXT_GGML_OPENMP} CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
    set(GGML_ALL_WARNINGS_3RD_PARTY OFF CACHE BOOL "" FORCE)
    set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

    if(CORTEXT_GGML_SOURCE_DIR)
      cortext_resolve_ggml_source_dir("${CORTEXT_GGML_SOURCE_DIR}"
                                      _cortext_ggml_source_dir)
    else()
      FetchContent_Declare(
        cortext_ggml_source
        GIT_REPOSITORY "${CORTEXT_GGML_GIT_REPOSITORY}"
        GIT_TAG "${CORTEXT_GGML_GIT_TAG}"
      )
      FetchContent_GetProperties(cortext_ggml_source)
      if(NOT cortext_ggml_source_POPULATED)
        FetchContent_Populate(cortext_ggml_source)
      endif()
      cortext_resolve_ggml_source_dir("${cortext_ggml_source_SOURCE_DIR}"
                                      _cortext_ggml_source_dir)
    endif()

    add_subdirectory("${_cortext_ggml_source_dir}"
                     "${CMAKE_BINARY_DIR}/_deps/ggml"
                     EXCLUDE_FROM_ALL)

    set(BUILD_SHARED_LIBS "${_cortext_saved_build_shared_libs}" CACHE BOOL
        "Build shared libraries" FORCE)

    if(BUILD_SHARED_LIBS)
      foreach(_cortext_ggml_target ggml ggml-base ggml-cpu)
        if(TARGET ${_cortext_ggml_target})
          set_target_properties(${_cortext_ggml_target}
            PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
      endforeach()
    endif()

    if(TARGET ggml AND TARGET ggml-cpu)
      set(CORTEXT_GGML_AVAILABLE ON)
      set(_cortext_ggml_header_path
          "${_cortext_ggml_source_dir}/include/ggml-backend.h")
      target_include_directories(cortext PRIVATE
        "${_cortext_ggml_source_dir}/include"
      )
      target_link_libraries(cortext PRIVATE ggml ggml-cpu)
      if(TARGET ggml-base)
        target_link_libraries(cortext PRIVATE ggml-base)
      endif()
      message(STATUS "Using bundled ggml backend from ${_cortext_ggml_source_dir}")
    else()
      message(FATAL_ERROR
        "ggml was added but did not define expected ggml/ggml-cpu targets.")
    endif()
  else()
    message(STATUS
      "ggml library not found and CORTEXT_FETCH_GGML=OFF; "
      "GGUF kernel backend disabled")
  endif()

  if(CORTEXT_GGML_AVAILABLE)
    target_compile_definitions(cortext PRIVATE
      CORTEXT_ENABLE_GGML=1
      CORTEXT_GGML_BACKEND_HEADER_PATH=\"${_cortext_ggml_header_path}\"
    )
  endif()
endif()

if(CORTEXT_ENABLE_AUDIO AND NOT CORTEXT_GGML_AVAILABLE)
  message(FATAL_ERROR
    "Cortext audio support requires AIST ggml kernel ops. Keep "
    "CORTEXT_FETCH_GGML=ON, set CORTEXT_GGML_SOURCE_DIR to a local ggml "
    "checkout, provide CORTEXT_GGML_* cache entries, or "
    "opt out explicitly with -DCORTEXT_ENABLE_AUDIO=OFF "
    "-DCORTEXT_ALLOW_UNSUPPORTED_TEXT_ONLY_BUILD=ON.")
endif()

if(APPLE)
  find_library(CORTEXT_ACCELERATE_FRAMEWORK Accelerate)
  if(CORTEXT_ACCELERATE_FRAMEWORK)
    target_link_libraries(cortext PUBLIC ${CORTEXT_ACCELERATE_FRAMEWORK})
  endif()
endif()

# The AIST GGUF encoder is Cortext's required embedding model; its native
# runtime always builds.
target_sources(cortext PRIVATE src/models/aist_gguf_encoder.cpp)

add_library(cortext::cortext ALIAS cortext)

# Public include dirs
target_include_directories(cortext
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite/src
)

if(CORTEXT_ENABLE_AUDIO)
  target_compile_definitions(cortext PUBLIC CORTEXT_ENABLE_AUDIO=1)
  target_compile_definitions(cortext PRIVATE CORTEXT_REQUIRE_AIST_GGML_KERNELS=1)
else()
  target_compile_definitions(cortext PUBLIC CORTEXT_DISABLE_AUDIO=1)
endif()

if(CORTEXT_ENABLE_IMAGE)
  target_compile_definitions(cortext PUBLIC CORTEXT_ENABLE_IMAGE=1)
else()
  target_compile_definitions(cortext PUBLIC CORTEXT_DISABLE_IMAGE=1)
endif()

if(CORTEXT_EXPERIMENT_HOOKS)
  target_compile_definitions(cortext PUBLIC CORTEXT_EXPERIMENT_HOOKS=1)
endif()

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
  # Silence non-portable warning noise in embedded third-party C sources,
  # and enable sqlite-vec's SIMD distance kernels per architecture (matches
  # upstream release builds). The SIMD kernels reorder float summation, so
  # distances can differ from the scalar path in the last ulp.
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    set(CORTEXT_VEC_OPTIONS "-w")
    set(CORTEXT_VEC_DEFS "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      list(APPEND CORTEXT_VEC_OPTIONS "-mavx")
      list(APPEND CORTEXT_VEC_DEFS "SQLITE_VEC_ENABLE_AVX")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      list(APPEND CORTEXT_VEC_DEFS "SQLITE_VEC_ENABLE_NEON")
    endif()
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-vec/sqlite-vec.c
      PROPERTIES
        COMPILE_OPTIONS "${CORTEXT_VEC_OPTIONS}"
        COMPILE_DEFINITIONS "${CORTEXT_VEC_DEFS}"
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
  if(CORTEXT_TARGET_AARCH64)
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
  if(EMSCRIPTEN)
    target_compile_definitions(cortext PRIVATE
      OBJSTORE_HAVE_BACKEND_FILE=0
      OBJSTORE_HAVE_BACKEND_SQLITE=0
      OBJSTORE_HAVE_BACKEND_OPFS=1
      OBJSTORE_HAVE_BACKEND_VFS=1
    )
  endif()
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
target_include_directories(cortext SYSTEM PRIVATE ${hnswlib_SOURCE_DIR})
get_target_property(CORTEXT_NLOHMANN_INCLUDE_DIRS nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
if(CORTEXT_NLOHMANN_INCLUDE_DIRS)
  target_include_directories(cortext BEFORE PRIVATE ${CORTEXT_NLOHMANN_INCLUDE_DIRS})
endif()
if(EMSCRIPTEN)
  target_compile_definitions(cortext PRIVATE SQLITE_OMIT_LOAD_EXTENSION=1)
  target_sources(cortext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/wasm/auto_extensions.cpp)
elseif(CORTEXT_TARGET_IOS)
  target_compile_definitions(cortext PRIVATE SQLITE_OMIT_LOAD_EXTENSION=1)
endif()
target_link_libraries(cortext PUBLIC sqlite3)

#
# OpenTelemetry linkage (native only): API-only by default, no-op when disabled.
#
if(NOT EMSCRIPTEN AND NOT CORTEXT_DISABLE_OPENTELEMETRY AND NOT CORTEXT_TARGET_IOS)
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
else()
  target_compile_definitions(cortext PRIVATE CORTEXT_DISABLE_OPENTELEMETRY=1)
endif()

# Use C++20 features
target_compile_features(cortext PUBLIC cxx_std_20)
target_compile_definitions(cortext PUBLIC CORTEXT_VERSION="${PROJECT_VERSION}")

# Set hidden visibility by default. Only symbols marked with CORTEXT_EXPORT
# will be visible to consumers.
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

if(EMSCRIPTEN)
  target_compile_options(cortext PRIVATE
    -O3
    $<IF:$<BOOL:${CORTEXT_WARNINGS_AS_ERRORS}>, -Werror, >
  )
elseif(MSVC)
  target_compile_options(cortext PRIVATE /W4 $<IF:$<BOOL:${CORTEXT_WARNINGS_AS_ERRORS}>,/WX,>)
else()
  target_compile_options(cortext PRIVATE -Wall -Wextra -Wpedantic $<IF:$<BOOL:${CORTEXT_WARNINGS_AS_ERRORS}>, -Werror, >)
endif()

if(EMSCRIPTEN AND BUILD_WASM)
  set(_cortext_wasm_exported_functions
    "_malloc"
    "_free"
    "_cortext_config_init"
    "_cortext_create"
    "_cortext_create_with_config"
    "_cortext_create_with_store_callbacks"
    "_cortext_create_with_store_and_object_callbacks"
    "_cortext_create_with_config_and_object_callbacks"
    "_cortext_free"
    "_cortext_process_text"
    "_cortext_process_audio"
    "_cortext_process_image"
    "_cortext_embed_text_json"
    "_cortext_embed_audio_json"
    "_cortext_embed_image_json"
    "_cortext_consolidate"
    "_cortext_flush"
    "_cortext_reset"
    "_cortext_version"
    "_cortext_last_error"
    "_cortext_string_free"
    "_cortext_process_text_json"
    "_cortext_process_text_with_media_json"
    "_cortext_process_text_with_media_json_with_options"
    "_cortext_wasm_process_text_json"
    "_cortext_process_audio_json"
    "_cortext_process_image_json"
    "_cortext_consolidate_json"
  )
  list(JOIN _cortext_wasm_exported_functions "\",\"" _cortext_wasm_exports_joined)
  set(_cortext_wasm_exports_json "[\"${_cortext_wasm_exports_joined}\"]")

  set(_cortext_wasm_runtime_methods
    "FS"
    "ccall"
    "cwrap"
    "UTF8ToString"
    "stringToUTF8"
    "lengthBytesUTF8"
    "addFunction"
    "removeFunction"
  )
  list(JOIN _cortext_wasm_runtime_methods "\",\"" _cortext_wasm_runtime_joined)
  set(_cortext_wasm_runtime_json "[\"${_cortext_wasm_runtime_joined}\"]")

  add_executable(cortext_wasm src/wasm/module.cpp)
  target_compile_features(cortext_wasm PRIVATE cxx_std_20)
  target_link_libraries(cortext_wasm PRIVATE cortext)
  set_target_properties(cortext_wasm PROPERTIES
    OUTPUT_NAME "cortext"
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist/wasm"
  )
  target_link_options(cortext_wasm PRIVATE
    -O3
    --no-entry
    -sWASM=1
    -sALLOW_MEMORY_GROWTH=1
    -sALLOW_TABLE_GROWTH=1
    -sFORCE_FILESYSTEM=1
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createCortextModule
    -sENVIRONMENT=web,worker
    "-sEXPORTED_FUNCTIONS=${_cortext_wasm_exports_json}"
    "-sEXPORTED_RUNTIME_METHODS=${_cortext_wasm_runtime_json}"
  )
  if(CORTEXT_WASM_EXPORT_ALL)
    target_link_options(cortext_wasm PRIVATE -sEXPORT_ALL=1)
  endif()
  if(CORTEXT_WASM_PRELOAD_MODEL_ASSETS_DIR)
    target_link_options(cortext_wasm PRIVATE
      --preload-file "${CORTEXT_WASM_PRELOAD_MODEL_ASSETS_DIR}@/models"
    )
  endif()
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

  execute_process(
    COMMAND "${CORTEXT_NODE_EXECUTABLE}" -p "require('path').dirname(require('path').dirname(process.execPath))"
    OUTPUT_VARIABLE CORTEXT_NODE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(NOT CORTEXT_NODE_INCLUDE_DIR)
    set(CORTEXT_NODE_INCLUDE_DIR "${CORTEXT_NODE_PREFIX}/include/node")
  endif()

  if(NOT EXISTS "${CORTEXT_NODE_INCLUDE_DIR}/node_api.h")
    message(FATAL_ERROR "Node.js headers not found under ${CORTEXT_NODE_INCLUDE_DIR}")
  endif()

  # N-API glue for optional Node addon builds consumed by standalone
  # language packages (e.g. github.com/augmem/cortext.ts). Language-facing
  # package sources live outside this engine repository.
  add_library(cortext_node MODULE
    ffi/node/addon.cpp
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
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ffi/node"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ffi/node"
  )

  if(WIN32)
    if(NOT CORTEXT_NODE_LIBRARY)
      find_file(_cortext_node_library
        NAMES node.lib
        PATHS "${CORTEXT_NODE_PREFIX}"
        PATH_SUFFIXES "" x64 lib
        NO_DEFAULT_PATH
      )
      set(CORTEXT_NODE_LIBRARY "${_cortext_node_library}" CACHE FILEPATH
        "Path to the Node.js import library used for Windows addon builds"
        FORCE
      )
    endif()
    if(NOT CORTEXT_NODE_LIBRARY OR NOT EXISTS "${CORTEXT_NODE_LIBRARY}")
      message(FATAL_ERROR
        "Node.js import library node.lib was not found under "
        "${CORTEXT_NODE_PREFIX} (searched the prefix, x64, and lib). "
        "Set CORTEXT_NODE_LIBRARY to the node.lib path."
      )
    endif()
    target_link_libraries(cortext_node PRIVATE "${CORTEXT_NODE_LIBRARY}")
  elseif(APPLE)
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
  target_compile_definitions(cortext PRIVATE CORTEXT_TESTING=1)
  add_subdirectory(tests)
endif()

if(CORTEXT_BUILD_EXAMPLES AND NOT EMSCRIPTEN)
  add_subdirectory(examples)
endif()

if(CORTEXT_BUILD_TOOLS AND NOT EMSCRIPTEN)
  add_subdirectory(tools)
endif()
