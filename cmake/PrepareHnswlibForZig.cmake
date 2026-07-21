if(NOT DEFINED HNSWLIB_SOURCE_DIR)
  message(FATAL_ERROR "HNSWLIB_SOURCE_DIR is required")
endif()
if(NOT DEFINED HNSWLIB_OUTPUT_DIR)
  message(FATAL_ERROR "HNSWLIB_OUTPUT_DIR is required")
endif()

# Zig's package directory is a content-addressed shared cache. Copy only the
# header library into this build step's declared output, then apply the same
# checked patch used by CMake without mutating or racing the package cache.
file(REMOVE_RECURSE "${HNSWLIB_OUTPUT_DIR}/hnswlib")
file(MAKE_DIRECTORY "${HNSWLIB_OUTPUT_DIR}")
file(COPY "${HNSWLIB_SOURCE_DIR}/hnswlib"
     DESTINATION "${HNSWLIB_OUTPUT_DIR}")

get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
file(RELATIVE_PATH _output_relative "${_repo_root}" "${HNSWLIB_OUTPUT_DIR}")
# git apply interprets --directory as a Git path even on Windows.
string(REPLACE "\\" "/" _output_relative "${_output_relative}")
set(_patch "${CMAKE_CURRENT_LIST_DIR}/hnswlib-prefetch-bounds.patch")
execute_process(
  COMMAND git apply --check --unsafe-paths
          "--directory=${_output_relative}" "${_patch}"
  WORKING_DIRECTORY "${_repo_root}"
  RESULT_VARIABLE _apply_check
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT _apply_check EQUAL 0)
  message(FATAL_ERROR
    "Pinned Zig hnswlib source does not match the prefetch bounds patch")
endif()
execute_process(
  COMMAND git apply --unsafe-paths
          "--directory=${_output_relative}" "${_patch}"
  WORKING_DIRECTORY "${_repo_root}"
  RESULT_VARIABLE _apply_result
)
if(NOT _apply_result EQUAL 0)
  message(FATAL_ERROR "Failed to patch the build-local Zig hnswlib copy")
endif()
