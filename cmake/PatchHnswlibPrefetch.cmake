if(NOT DEFINED HNSWLIB_SOURCE_DIR)
  message(FATAL_ERROR "HNSWLIB_SOURCE_DIR is required")
endif()

set(_patch "${CMAKE_CURRENT_LIST_DIR}/hnswlib-prefetch-bounds.patch")
execute_process(
  COMMAND git -C "${HNSWLIB_SOURCE_DIR}" apply --check "${_patch}"
  RESULT_VARIABLE _apply_check
  OUTPUT_QUIET
  ERROR_QUIET
)
if(_apply_check EQUAL 0)
  execute_process(
    COMMAND git -C "${HNSWLIB_SOURCE_DIR}" apply "${_patch}"
    RESULT_VARIABLE _apply_result
  )
  if(NOT _apply_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply the hnswlib prefetch bounds patch")
  endif()
else()
  execute_process(
    COMMAND git -C "${HNSWLIB_SOURCE_DIR}" apply --reverse --check "${_patch}"
    RESULT_VARIABLE _reverse_check
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT _reverse_check EQUAL 0)
    message(FATAL_ERROR
      "Pinned hnswlib source does not match the prefetch bounds patch")
  endif()
endif()
