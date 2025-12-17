cmake_minimum_required(VERSION 3.24)

include(FetchContent)

# --- Dependencies ---

# Dear ImGui (not a CMake project - we'll add sources directly)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.6-docking
)
FetchContent_MakeAvailable(imgui)

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

# nlohmann/json (required by openai-cpp headers)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

find_package(CURL REQUIRED)

# --- Target ---

add_executable(cortext_chat
  main.cpp
  context_tab.cpp
  streaming_client.cpp
  imgui_app.cpp
  chat_window.cpp
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
)

target_include_directories(cortext_chat PRIVATE
  ${openai_cpp_SOURCE_DIR}/include
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(cortext_chat PRIVATE
  cortext::cortext
  CURL::libcurl
  nlohmann_json::nlohmann_json
  glfw
  OpenGL::GL
  opentelemetry-cpp::api
  opentelemetry-cpp::trace
  opentelemetry-cpp::metrics
  opentelemetry-cpp::logs
  opentelemetry-cpp::otlp_grpc_exporter
  opentelemetry-cpp::otlp_grpc_metrics_exporter
  opentelemetry-cpp::otlp_grpc_log_record_exporter
)

target_compile_features(cortext_chat PRIVATE cxx_std_17)

# Ensure the runtime loader can find vendored dylibs on macOS.
if(APPLE)
  set(ORT_LIB_DIR "${CMAKE_BINARY_DIR}/third_party/onnxruntime-install/lib")
  set(LITERT_LIB_DIR "${CMAKE_BINARY_DIR}/third_party/litert-lm-install/lib")
  set_target_properties(cortext_chat PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR};${LITERT_LIB_DIR}"
  )
  # macOS frameworks needed by GLFW
  find_library(COCOA_LIBRARY Cocoa REQUIRED)
  find_library(IOKIT_LIBRARY IOKit REQUIRED)
  find_library(COREVIDEO_LIBRARY CoreVideo REQUIRED)
  target_link_libraries(cortext_chat PRIVATE
    ${COCOA_LIBRARY}
    ${IOKIT_LIBRARY}
    ${COREVIDEO_LIBRARY}
  )
endif()

# Keep warnings reasonable for an example.
if(MSVC)
  target_compile_options(cortext_chat PRIVATE /W4)
else()
  target_compile_options(cortext_chat PRIVATE -Wall -Wextra -Wpedantic)
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
  PROPERTIES COMPILE_OPTIONS "-w"
)
