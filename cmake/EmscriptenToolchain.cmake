# Resolve and delegate to the Emscripten SDK toolchain. Keeping this thin
# wrapper lets CMakePresets.json stay stable across local shells and CI.

set(_cortext_emscripten_toolchain "")

if(DEFINED ENV{EMSDK} AND EXISTS "$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")
  set(_cortext_emscripten_toolchain "$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")
elseif(DEFINED ENV{EMSCRIPTEN} AND EXISTS "$ENV{EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake")
  set(_cortext_emscripten_toolchain "$ENV{EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake")
else()
  find_program(_cortext_emcc NAMES emcc)
  if(_cortext_emcc)
    get_filename_component(_cortext_emcc_dir "${_cortext_emcc}" DIRECTORY)
    if(EXISTS "${_cortext_emcc_dir}/cmake/Modules/Platform/Emscripten.cmake")
      set(_cortext_emscripten_toolchain "${_cortext_emcc_dir}/cmake/Modules/Platform/Emscripten.cmake")
    endif()
  endif()
endif()

if(NOT _cortext_emscripten_toolchain)
  message(FATAL_ERROR "Could not locate Emscripten.cmake. Source emsdk_env.sh or put emcc on PATH.")
endif()

include("${_cortext_emscripten_toolchain}")
