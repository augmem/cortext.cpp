# Reconstructed helper (original was untracked on the Mac — .gitignore *.cmake).
# Installs the bazel-built litert-lm shared library and the headers cortext
# consumes into LITERT_INSTALL_DIR (lib/ and include/).
if(NOT DEFINED LITERT_SOURCE_DIR OR NOT DEFINED LITERT_INSTALL_DIR)
  message(FATAL_ERROR "LITERT_SOURCE_DIR and LITERT_INSTALL_DIR must be defined")
endif()

file(MAKE_DIRECTORY "${LITERT_INSTALL_DIR}/lib")
file(MAKE_DIRECTORY "${LITERT_INSTALL_DIR}/include")

# The shared library lands in bazel-bin at the workspace root (BUILD_IN_SOURCE).
set(_lib "${LITERT_SOURCE_DIR}/bazel-bin/liblitert_lm_cpp.so")
if(NOT EXISTS "${_lib}")
  file(GLOB_RECURSE _lib_candidates "${LITERT_SOURCE_DIR}/bazel-bin/*litert_lm_cpp*")
  list(LENGTH _lib_candidates _n)
  if(_n EQUAL 0)
    message(FATAL_ERROR "liblitert_lm_cpp library not found under ${LITERT_SOURCE_DIR}/bazel-bin")
  endif()
  list(GET _lib_candidates 0 _lib)
endif()
file(COPY "${_lib}" DESTINATION "${LITERT_INSTALL_DIR}/lib")
message(STATUS "Installed ${_lib} -> ${LITERT_INSTALL_DIR}/lib")

# Prebuilt runtime dependencies of liblitert_lm_cpp.so (constraint provider etc.)
file(GLOB _prebuilt "${LITERT_SOURCE_DIR}/prebuilt/linux_x86_64/lib*.so")
foreach(_p ${_prebuilt})
  file(COPY "${_p}" DESTINATION "${LITERT_INSTALL_DIR}/lib")
endforeach()

# Source headers consumed via #include "runtime/..." paths.
file(GLOB_RECURSE _headers RELATIVE "${LITERT_SOURCE_DIR}"
  "${LITERT_SOURCE_DIR}/runtime/*.h"
  "${LITERT_SOURCE_DIR}/runtime/*.inc"
)
foreach(_h ${_headers})
  get_filename_component(_dir "${_h}" DIRECTORY)
  file(COPY "${LITERT_SOURCE_DIR}/${_h}" DESTINATION "${LITERT_INSTALL_DIR}/include/${_dir}")
endforeach()

# Generated proto headers and external repo headers (litert, absl, sentencepiece)
# from the bazel execroot/output tree, if present.
set(_bazel_bin "${LITERT_SOURCE_DIR}/bazel-bin")
file(GLOB_RECURSE _gen_headers RELATIVE "${_bazel_bin}" "${_bazel_bin}/runtime/*.pb.h")
foreach(_h ${_gen_headers})
  get_filename_component(_dir "${_h}" DIRECTORY)
  file(COPY "${_bazel_bin}/${_h}" DESTINATION "${LITERT_INSTALL_DIR}/include/${_dir}")
endforeach()

# litert external repo: source headers plus generated ones (build_config.h etc.)
foreach(_litert_root
    "${LITERT_SOURCE_DIR}/../../build/linux/bazel-output/external/litert/litert"
    "${LITERT_SOURCE_DIR}/../../build/linux/bazel-output/execroot/litert_lm/bazel-out/k8-opt/bin/external/litert/litert")
  if(EXISTS "${_litert_root}")
    file(COPY "${_litert_root}" DESTINATION "${LITERT_INSTALL_DIR}/include")
  endif()
endforeach()

set(_external "${LITERT_SOURCE_DIR}/bazel-litert-lm/external")
foreach(_repo litert litert_c "absl" "com_google_absl" sentencepiece)
  if(EXISTS "${_external}/${_repo}")
    file(GLOB_RECURSE _ext_headers RELATIVE "${_external}/${_repo}"
      "${_external}/${_repo}/*.h" "${_external}/${_repo}/*.inc")
    foreach(_h ${_ext_headers})
      get_filename_component(_dir "${_h}" DIRECTORY)
      file(COPY "${_external}/${_repo}/${_h}" DESTINATION "${LITERT_INSTALL_DIR}/include/${_dir}")
    endforeach()
  endif()
endforeach()

message(STATUS "LiteRT-LM install complete at ${LITERT_INSTALL_DIR}")
