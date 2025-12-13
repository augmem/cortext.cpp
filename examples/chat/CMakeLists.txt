cmake_minimum_required(VERSION 3.24)

include(FetchContent)

# --- Dependencies ---

# FTXUI
FetchContent_Declare(
  ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
  GIT_TAG v6.1.9
)
FetchContent_MakeAvailable(ftxui)

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
)

target_link_libraries(cortext_chat PRIVATE
  cortext::cortext
  CURL::libcurl
  nlohmann_json::nlohmann_json
  ftxui::component
  ftxui::dom
  ftxui::screen
)

target_include_directories(cortext_chat PRIVATE
  ${openai_cpp_SOURCE_DIR}/include
)

target_compile_features(cortext_chat PRIVATE cxx_std_17)

# Ensure the runtime loader can find the vendored ONNX Runtime dylib on macOS.
# (The ONNX Runtime dylib uses an @rpath install_name.)
if(APPLE AND CORTEXT_ENABLE_IMAGEBIND_ORT)
  set(ORT_LIB_DIR "${CMAKE_BINARY_DIR}/third_party/onnxruntime-install/lib")
  set_target_properties(cortext_chat PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR}"
  )
endif()

# Keep warnings reasonable for an example.
if(MSVC)
  target_compile_options(cortext_chat PRIVATE /W4)
else()
  target_compile_options(cortext_chat PRIVATE -Wall -Wextra -Wpedantic)
endif()
