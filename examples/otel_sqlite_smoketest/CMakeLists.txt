cmake_minimum_required(VERSION 3.24)

add_executable(cortext_otel_sqlite_smoketest
  main.cpp
)

target_link_libraries(cortext_otel_sqlite_smoketest PRIVATE
  cortext::cortext
)

if(NOT CORTEXT_DISABLE_OPENTELEMETRY)
  target_link_libraries(cortext_otel_sqlite_smoketest PRIVATE
    opentelemetry-cpp::api
    opentelemetry-cpp::trace
    opentelemetry-cpp::logs
    opentelemetry-cpp::otlp_file_exporter
    opentelemetry-cpp::otlp_file_log_record_exporter
  )
endif()

target_compile_features(cortext_otel_sqlite_smoketest PRIVATE cxx_std_17)
