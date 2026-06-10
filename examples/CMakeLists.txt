# examples/CMakeLists.txt

# The ImGui chat demo needs desktop dependencies; skip it cleanly when they
# are absent (e.g. headless CI or servers) instead of failing the configure.
find_package(glfw3 3.3 QUIET)
if(glfw3_FOUND)
  add_subdirectory(chat)
else()
  message(STATUS "examples/chat skipped: glfw3 >= 3.3 not found")
endif()
add_subdirectory(otel_sqlite_smoketest)
add_subdirectory(benchmark)
add_subdirectory(topical_chat_analysis)
