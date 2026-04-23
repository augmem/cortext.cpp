include(FetchContent)

option(CORTEXT_CHAT_ENABLE_OTLP_GRPC
  "Enable OTLP gRPC exporters for the chat example"
  OFF)

# --- Dependencies ---

# Dear ImGui (not a CMake project - we'll add sources directly)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.6-docking
)
FetchContent_MakeAvailable(imgui)

# ImPlot (pairs with Dear ImGui; sources added directly)
FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG 4707b24
)
FetchContent_Populate(implot)

# GLFW for windowing
find_package(glfw3 3.3 REQUIRED)

# OpenGL for rendering
find_package(OpenGL REQUIRED)

# openai-cpp (header-only)
FetchContent_Declare(
  openai_cpp
  GIT_REPOSITORY https://github.com/olrea/openai-cpp.git
  GIT_TAG v0.1.3
)
FetchContent_Populate(openai_cpp)

FetchContent_Declare(
  whisper_cpp
  GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
  GIT_TAG v1.8.4
)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(whisper_cpp)

find_package(CURL REQUIRED)

# --- Target ---

add_executable(cortext_chat
  main.cpp
  context_tab.cpp
  streaming_client.cpp
  imgui_app.cpp
  chat_window.cpp
)

add_executable(cortext_chat_profile
  profile.cpp
)

add_executable(cortext_voice_backend_bench
  voice_backend_bench.cpp
)

add_executable(cortext_planum_bridge_smoke
  planum_bridge_smoke.cpp
  ${CMAKE_SOURCE_DIR}/src/audio/planum_bridge.cpp
)

# Add ImGui sources (Dear ImGui is not a CMake project)
target_sources(cortext_chat PRIVATE
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  ${implot_SOURCE_DIR}/implot.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
)

target_include_directories(cortext_chat PRIVATE
  ${openai_cpp_SOURCE_DIR}/include
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${implot_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(cortext_chat PRIVATE
  cortext::cortext
  CURL::libcurl
  nlohmann_json::nlohmann_json
  glfw
  OpenGL::GL
)

if(NOT CORTEXT_DISABLE_OPENTELEMETRY)
  target_link_libraries(cortext_chat PRIVATE
    opentelemetry-cpp::api
    opentelemetry-cpp::trace
    opentelemetry-cpp::metrics
    opentelemetry-cpp::logs
  )
endif()

if(CORTEXT_CHAT_ENABLE_OTLP_GRPC AND NOT CORTEXT_DISABLE_OPENTELEMETRY)
  target_link_libraries(cortext_chat PRIVATE
    opentelemetry-cpp::otlp_grpc_exporter
    opentelemetry-cpp::otlp_grpc_metrics_exporter
    opentelemetry-cpp::otlp_grpc_log_record_exporter
  )
endif()

target_link_libraries(cortext_chat_profile PRIVATE
  cortext::cortext
)

target_include_directories(cortext_planum_bridge_smoke PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/third_party/planum.cpp/include
)
target_link_libraries(cortext_planum_bridge_smoke PRIVATE
  cortext::cortext
)

target_compile_features(cortext_voice_backend_bench PRIVATE cxx_std_20)
target_compile_features(cortext_planum_bridge_smoke PRIVATE cxx_std_20)
target_include_directories(cortext_voice_backend_bench PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/sherpa-onnx
  ${whisper_cpp_SOURCE_DIR}/include
)
target_link_libraries(cortext_voice_backend_bench PRIVATE
  sherpa-onnx-cxx-api
  whisper
)

target_compile_features(cortext_chat PRIVATE cxx_std_20)
target_compile_features(cortext_chat_profile PRIVATE cxx_std_20)
target_compile_definitions(cortext_chat PRIVATE
  CORTEXT_CHAT_ENABLE_OTLP_GRPC=$<BOOL:${CORTEXT_CHAT_ENABLE_OTLP_GRPC}>
)

if(APPLE AND NOT CORTEXT_DISABLE_SHERPA_ONNX)
  target_sources(cortext_chat PRIVATE
    voice_session.mm
  )
  target_include_directories(cortext_chat PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/sherpa-onnx
    ${whisper_cpp_SOURCE_DIR}/include
  )
  target_link_libraries(cortext_chat PRIVATE
    sherpa-onnx-cxx-api
    whisper
  )
else()
  target_sources(cortext_chat PRIVATE
    voice_session.cpp
  )
endif()

target_include_directories(cortext_chat_profile PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

# Ensure the runtime loader can find vendored dylibs on macOS.
if(APPLE)
  set(ORT_LIB_DIR "${CMAKE_BINARY_DIR}/third_party/onnxruntime-install/lib")
  set(LITERT_LIB_DIR "${CMAKE_BINARY_DIR}/third_party/litert-lm-install/lib")
  set_target_properties(cortext_chat PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR};${LITERT_LIB_DIR}"
  )
  set_target_properties(cortext_chat_profile PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR};${LITERT_LIB_DIR}"
  )
  set_target_properties(cortext_voice_backend_bench PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR};${LITERT_LIB_DIR}"
  )
  set_target_properties(cortext_planum_bridge_smoke PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR};${LITERT_LIB_DIR}"
  )
  # macOS frameworks needed by GLFW
  find_library(COCOA_LIBRARY Cocoa REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  find_library(AVFOUNDATION_LIBRARY AVFoundation REQUIRED)
  find_library(IOKIT_LIBRARY IOKit REQUIRED)
  find_library(COREVIDEO_LIBRARY CoreVideo REQUIRED)
  target_link_libraries(cortext_chat PRIVATE
    ${COCOA_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${AVFOUNDATION_LIBRARY}
    ${IOKIT_LIBRARY}
    ${COREVIDEO_LIBRARY}
  )
endif()

# Keep warnings reasonable for an example.
if(MSVC)
  target_compile_options(cortext_chat PRIVATE /W4)
  target_compile_options(cortext_chat_profile PRIVATE /W4)
  target_compile_options(cortext_voice_backend_bench PRIVATE /W4)
else()
  target_compile_options(cortext_chat PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cortext_chat_profile PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cortext_voice_backend_bench PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cortext_planum_bridge_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Suppress warnings in ImGui source files
set_source_files_properties(
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  ${implot_SOURCE_DIR}/implot.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
  PROPERTIES COMPILE_OPTIONS "-w"
)
