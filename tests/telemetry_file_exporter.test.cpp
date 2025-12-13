#include <catch2/catch_test_macros.hpp>

#include <cortext/telemetry/telemetry.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string MakeTempPath() {
  const auto dir = std::filesystem::temp_directory_path();
  const auto p = dir / ("cortext_otel_" + std::to_string(std::rand()) + ".log");
  return p.string();
}

} // namespace

TEST_CASE("telemetry: file exporter writes when configured") {
#if defined(CORTEXT_ENABLE_OTEL)
  const std::string path = MakeTempPath();
  std::remove(path.c_str());

  // Prefer file exporter by leaving OTEL_*_EXPORTER unset or set to none.
  unsetenv("OTEL_TRACES_EXPORTER");
  unsetenv("OTEL_METRICS_EXPORTER");
  unsetenv("OTEL_LOGS_EXPORTER");
  setenv("CORTEXT_OTEL_FILE_PATH", path.c_str(), 1);
  setenv("CORTEXT_OTEL_FILE_SIGNALS", "traces,metrics,logs", 1);

  REQUIRE(cortext::telemetry::InitializeFromEnv());

  {
    cortext::telemetry::ScopedSpan span("cortext.test.file_exporter");
    span.SetAttribute("test.key", "value");
  }
  cortext::telemetry::AddCounter("cortext.test.counter", 1);
  cortext::telemetry::RecordHistogram("cortext.test.hist", 0.5);
  cortext::telemetry::LogInfo("hello", {cortext::telemetry::Attribute::String("component", "test")});
  cortext::telemetry::ForceFlush();

  REQUIRE(std::filesystem::exists(path));
  REQUIRE(std::filesystem::file_size(path) > 0);

  // OTLP file exporter writes JSONL. Ensure we got at least one JSON line
  // (not the human-readable ostream format).
  {
    std::ifstream in(path);
    REQUIRE(in.is_open());
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty()) break;
    }
    REQUIRE(!line.empty());
    REQUIRE(line.rfind("{", 0) == 0);
    REQUIRE(line.find("\"") != std::string::npos);
  }
#else
  SUCCEED();
#endif
}


