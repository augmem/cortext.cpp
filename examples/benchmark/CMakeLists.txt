add_executable(cortext_benchmark main.cpp)

target_link_libraries(cortext_benchmark PRIVATE cortext nlohmann_json::nlohmann_json)

# Ensure ONNX Runtime dylib can be found at runtime
if(CORTEXT_ENABLE_GEMMA_ORT)
  if(APPLE)
    set_property(TARGET cortext_benchmark APPEND PROPERTY BUILD_RPATH
      "@loader_path/../../third_party/onnxruntime-install/lib"
    )
  elseif(UNIX)
    set_property(TARGET cortext_benchmark APPEND PROPERTY BUILD_RPATH
      "$ORIGIN/../../third_party/onnxruntime-install/lib"
    )
  endif()
endif()
