if(NOT CORTEXT_DISABLE_OPENTELEMETRY AND CORTEXT_OPENTELEMETRY_EXPORTERS)
  add_subdirectory(otel_sqlite_smoketest)
else()
  message(STATUS
    "examples/otel_sqlite_smoketest skipped: OpenTelemetry exporters disabled")
endif()
add_subdirectory(benchmark)
add_subdirectory(topical_chat_analysis)
