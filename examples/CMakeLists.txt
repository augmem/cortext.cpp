# examples/CMakeLists.txt

# The ImGui chat demo needs desktop dependencies; skip it cleanly when they
# are absent (e.g. headless CI or servers) instead of failing the configure.
find_package(glfw3 3.3 QUIET)
if(glfw3_FOUND AND NOT CORTEXT_DISABLE_OPENTELEMETRY)
  add_subdirectory(chat)
elseif(CORTEXT_DISABLE_OPENTELEMETRY)
  message(STATUS "examples/chat skipped: OpenTelemetry disabled")
else()
  message(STATUS "examples/chat skipped: glfw3 >= 3.3 not found")
endif()
if(NOT CORTEXT_DISABLE_OPENTELEMETRY)
  add_subdirectory(otel_sqlite_smoketest)
else()
  message(STATUS "examples/otel_sqlite_smoketest skipped: OpenTelemetry disabled")
endif()
add_subdirectory(benchmark)
add_subdirectory(topical_chat_analysis)
