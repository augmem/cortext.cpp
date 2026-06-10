# Reconstructed helper (original was untracked on the Mac — .gitignore *.cmake).
# Appends the custom liblitert_lm_cpp.so target from litert-lm-BUILD.bazel to the
# litert-lm root BUILD file, idempotently.
if(NOT DEFINED LITERT_CUSTOM_BUILD OR NOT DEFINED LITERT_ROOT_BUILD)
  message(FATAL_ERROR "LITERT_CUSTOM_BUILD and LITERT_ROOT_BUILD must be defined")
endif()

file(READ "${LITERT_CUSTOM_BUILD}" _custom_content)

if(EXISTS "${LITERT_ROOT_BUILD}")
  file(READ "${LITERT_ROOT_BUILD}" _root_content)
else()
  set(_root_content "")
endif()

string(FIND "${_root_content}" "liblitert_lm_cpp.so" _already_present)
if(_already_present EQUAL -1)
  file(APPEND "${LITERT_ROOT_BUILD}" "\n${_custom_content}\n")
  message(STATUS "Appended custom litert_lm_cpp BUILD target to ${LITERT_ROOT_BUILD}")
else()
  message(STATUS "Custom litert_lm_cpp BUILD target already present in ${LITERT_ROOT_BUILD}")
endif()
