#include <cortext/store/sqlite_store.hpp>
#include <cortext/telemetry/telemetry.hpp>

#include <opentelemetry/exporters/otlp/otlp_file_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_file_log_record_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_log_record_exporter_options.h>

#include <opentelemetry/logs/provider.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/trace/provider.h>

#include <opentelemetry/sdk/logs/logger_provider.h>
#include <opentelemetry/sdk/logs/simple_log_record_processor.h>
#include <opentelemetry/sdk/resource/resource.h>
#include <opentelemetry/sdk/trace/simple_processor.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string GetEnv(const char *name, const std::string &fallback = {}) {
  const char *v = std::getenv(name);
  if (!v) {
    return fallback;
  }
  return std::string(v);
}

std::string MakeSiblingPath(const std::string &path, const std::string &suffix) {
  const std::string ext = ".jsonl";
  if (path.size() >= ext.size() && path.compare(path.size() - ext.size(), ext.size(), ext) == 0) {
    return path.substr(0, path.size() - ext.size()) + suffix + ext;
  }
  return path + suffix;
}

void InstallOtelProviders(const std::string &jsonl_path) {
  namespace trace_sdk = opentelemetry::sdk::trace;
  namespace logs_sdk = opentelemetry::sdk::logs;
  namespace resource_sdk = opentelemetry::sdk::resource;
  namespace otlp_exporter = opentelemetry::exporter::otlp;
  const std::string trace_path = MakeSiblingPath(jsonl_path, ".traces");
  const std::string log_path = MakeSiblingPath(jsonl_path, ".logs");
  auto resource = resource_sdk::Resource::Create({{"service.name", "cortext_otel_sqlite_smoketest"}});
  otlp_exporter::OtlpFileClientFileSystemOptions trace_fs;
  trace_fs.file_pattern = trace_path;
  trace_fs.alias_pattern = trace_path + ".latest";
  trace_fs.flush_interval = std::chrono::milliseconds(50);
  trace_fs.flush_count = 1;
  otlp_exporter::OtlpFileExporterOptions trace_options;
  trace_options.backend_options = trace_fs;
  auto trace_exporter = otlp_exporter::OtlpFileExporterFactory::Create(trace_options);
  std::vector<std::unique_ptr<trace_sdk::SpanProcessor>> processors;
  processors.push_back(std::unique_ptr<trace_sdk::SpanProcessor>(new trace_sdk::SimpleSpanProcessor(std::move(trace_exporter))));
  auto tracer_provider = opentelemetry::nostd::shared_ptr<opentelemetry::trace::TracerProvider>(
      new trace_sdk::TracerProvider(std::move(processors), resource));
  opentelemetry::trace::Provider::SetTracerProvider(tracer_provider);
  otlp_exporter::OtlpFileClientFileSystemOptions log_fs;
  log_fs.file_pattern = log_path;
  log_fs.alias_pattern = log_path + ".latest";
  log_fs.flush_interval = std::chrono::milliseconds(50);
  log_fs.flush_count = 1;
  otlp_exporter::OtlpFileLogRecordExporterOptions log_options;
  log_options.backend_options = log_fs;
  auto log_exporter = otlp_exporter::OtlpFileLogRecordExporterFactory::Create(log_options);
  auto log_processor = std::unique_ptr<logs_sdk::LogRecordProcessor>(
      new logs_sdk::SimpleLogRecordProcessor(std::move(log_exporter)));
  auto logger_provider = opentelemetry::nostd::shared_ptr<opentelemetry::logs::LoggerProvider>(
      new logs_sdk::LoggerProvider(std::move(log_processor), resource));
  opentelemetry::logs::Provider::SetLoggerProvider(logger_provider);
}

} // namespace

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  const std::filesystem::path out_dir = GetEnv("CORTEXT_OTEL_OUT_DIR", "/tmp");
  const std::filesystem::path jsonl_path = out_dir / "otel.jsonl";
  try {
    std::filesystem::create_directories(out_dir);
  } catch (...) {
  }
  InstallOtelProviders(jsonl_path.string());
  auto store = cortext::SQLiteStore::Create(":memory:");
  if (!store) {
    std::cerr << "Failed to create SQLiteStore\n";
    return 1;
  }
  try {
    (void)store->Execute("SELCT 1");
  } catch (const std::exception &) {
  }
  try {
    (void)store->Execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT UNIQUE);");
    (void)store->Execute("INSERT INTO t(v) VALUES('x');");
    (void)store->Execute("INSERT INTO t(v) VALUES('x');");
  } catch (const std::exception &) {
  }
  const auto provider = opentelemetry::trace::Provider::GetTracerProvider();
  auto *sdk_provider = dynamic_cast<opentelemetry::sdk::trace::TracerProvider *>(provider.get());
  if (sdk_provider) {
    (void)sdk_provider->ForceFlush();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  std::cout << "Wrote trace/log JSONL under: " << out_dir.string() << "\n";
  return 0;
}
