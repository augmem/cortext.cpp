cmake_minimum_required(VERSION 3.16)
project(cortext LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTING "Build tests" ON)
option(CORTEXT_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(BUILD_WASM "Build WebAssembly version" OFF)
option(CORTEXT_WASM_EXPORT_ALL "Export all functions in WASM build" ON)
option(CORTEXT_EMBED_VEC "Embed sqlite-vec extension" ON)
option(CORTEXT_EMBED_GRAPH "Embed sqlite-graph extension" ON)
option(CORTEXT_EMBED_OBJSTORE "Embed sqlite-objstore extension" ON)
option(CORTEXT_ENABLE_DYNAMIC_EXTENSIONS "Enable dynamic extension loading fallback" ON)
option(CORTEXT_ENABLE_IMAGEBIND_ORT "Enable ONNX Runtime for ImageBind embedder" OFF)

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
endblock()


if(NOT EMSCRIPTEN)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(SQLITE3 REQUIRED sqlite3)
endif()

add_library(cortext
    src/store.cpp
    src/store/extension_loader.cpp
    src/processor/operation_context.cpp
    src/signal_processor.cpp
    src/cortext.cpp
    src/capi.cpp
    src/encoder/imagebind.cpp
    src/operations/focus.cpp
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
    src/operations/consolidation_gate.cpp
    src/operations/logprob_surprise.cpp
    src/operations/precision.cpp
    src/operations/graph_schema.cpp
    src/operations/graph_build.cpp
    src/operations/graph_retrieval.cpp
    src/operations/goal_alignment.cpp
)
add_library(cortext::cortext ALIAS cortext)

# Public include dirs
target_include_directories(cortext
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite/src
)

# Embed sqlite-vec/sqlite-graph sources when enabled
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
if(CORTEXT_EMBED_GRAPH)
  file(GLOB CORTEXT_GRAPH_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-graph/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-graph/src/cypher/*.c
  )
  list(APPEND CORTEXT_EMBED_SOURCES ${CORTEXT_GRAPH_SRCS})
  target_include_directories(cortext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-graph/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite-graph/src
  )
  target_compile_definitions(cortext PRIVATE CORTEXT_EMBED_GRAPH=1)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    set_source_files_properties(
      ${CORTEXT_GRAPH_SRCS}
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

# Optional ONNX Runtime integration for ImageBind embedder
if(CORTEXT_ENABLE_IMAGEBIND_ORT)
  target_compile_definitions(cortext PRIVATE CORTEXT_ENABLE_IMAGEBIND_ORT=1)
  target_include_directories(cortext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include
  )
  find_library(ONNXRUNTIME_LIB NAMES onnxruntime
    HINTS
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/build/Release
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/build/Debug
    PATHS ENV ONNXRUNTIME_LIBDIR
  )
  if(ONNXRUNTIME_LIB)
    target_link_libraries(cortext PRIVATE ${ONNXRUNTIME_LIB})
  else()
    message(WARNING "ONNX Runtime library not found; CORTEXT_ENABLE_IMAGEBIND_ORT will have no effect (fallback to zeros).")
  endif()
endif()

# Use C++17 features
target_compile_features(cortext PUBLIC cxx_std_17)

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
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
