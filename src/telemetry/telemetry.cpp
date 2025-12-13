#include "cortext/telemetry/telemetry.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(CORTEXT_ENABLE_OTEL)
#include <opentelemetry/context/runtime_context.h>
#include <opentelemetry/common/attribute_value.h>
#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/nostd/string_view.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_options.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>
#include <opentelemetry/sdk/resource/resource.h>
#include <opentelemetry/sdk/trace/batch_span_processor.h>
#include <opentelemetry/sdk/trace/batch_span_processor_options.h>
#include <opentelemetry/sdk/trace/samplers/always_off.h>
#include <opentelemetry/sdk/trace/samplers/always_on.h>
#include <opentelemetry/sdk/trace/samplers/parent.h>
#include <opentelemetry/sdk/trace/samplers/trace_id_ratio.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/trace/provider.h>
#include <opentelemetry/trace/scope.h>
#include <opentelemetry/trace/span.h>

#if __has_include(<opentelemetry/exporters/ostream/span_exporter.h>)
#include <opentelemetry/exporters/ostream/span_exporter.h>
#define CORTEXT_HAVE_OSTREAM_TRACE_EXPORTER 1
#endif
#if __has_include(<opentelemetry/exporters/ostream/metric_exporter.h>)
#include <opentelemetry/exporters/ostream/metric_exporter.h>
#define CORTEXT_HAVE_OSTREAM_METRIC_EXPORTER 1
#endif
#if __has_include(<opentelemetry/exporters/ostream/log_record_exporter.h>)
#include <opentelemetry/exporters/ostream/log_record_exporter.h>
#define CORTEXT_HAVE_OSTREAM_LOG_EXPORTER 1
#endif

// Logs are optional (and the C++ API surface has evolved). We compile logs
// support only if the relevant headers are present.
#if __has_include(<opentelemetry/logs/provider.h>) \
    && __has_include(<opentelemetry/logs/logger.h>) \
    && __has_include(<opentelemetry/sdk/logs/logger_provider.h>) \
    && __has_include(<opentelemetry/sdk/logs/batch_log_record_processor.h>)
#include <opentelemetry/logs/logger.h>
#include <opentelemetry/logs/provider.h>
#include <opentelemetry/sdk/logs/batch_log_record_processor.h>
#include <opentelemetry/sdk/logs/logger_provider.h>
#define CORTEXT_HAVE_OTEL_LOGS_API 1
#include <opentelemetry/sdk/logs/read_write_log_record.h>
#define CORTEXT_HAVE_OTEL_LOG_RECORD 1
#endif

// Exporter factories are optional depending on how opentelemetry-cpp is built.
// We use __has_include to avoid hard failures and simply no-op those signals.
#if __has_include(<opentelemetry/exporters/otlp/otlp_grpc_exporter_factory.h>)
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter_factory.h>
#define CORTEXT_HAVE_OTLP_GRPC_TRACE_EXPORTER 1
#endif

#if __has_include(<opentelemetry/exporters/otlp/otlp_http_exporter_factory.h>)
#include <opentelemetry/exporters/otlp/otlp_http_exporter_factory.h>
#define CORTEXT_HAVE_OTLP_HTTP_TRACE_EXPORTER 1
#endif

#if __has_include(<opentelemetry/exporters/otlp/otlp_grpc_metric_exporter_factory.h>)
#include <opentelemetry/exporters/otlp/otlp_grpc_metric_exporter_factory.h>
#define CORTEXT_HAVE_OTLP_GRPC_METRIC_EXPORTER 1
#endif

#if __has_include(<opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h>)
#include <opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h>
#define CORTEXT_HAVE_OTLP_HTTP_METRIC_EXPORTER 1
#endif

#if __has_include(<opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_factory.h>)
#include <opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_factory.h>
#define CORTEXT_HAVE_OTLP_GRPC_LOG_EXPORTER 1
#endif

#if __has_include(<opentelemetry/exporters/otlp/otlp_http_log_record_exporter_factory.h>)
#include <opentelemetry/exporters/otlp/otlp_http_log_record_exporter_factory.h>
#define CORTEXT_HAVE_OTLP_HTTP_LOG_EXPORTER 1
#endif

#if __has_include(<opentelemetry/exporters/otlp/otlp_file_exporter_factory.h>) \
    && __has_include(<opentelemetry/exporters/otlp/otlp_file_exporter_options.h>)
#include <opentelemetry/exporters/otlp/otlp_file_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_exporter_options.h>
#define CORTEXT_HAVE_OTLP_FILE_TRACE_EXPORTER 1
#endif

#if __has_include(<opentelemetry/exporters/otlp/otlp_file_metric_exporter_factory.h>) \
    && __has_include(<opentelemetry/exporters/otlp/otlp_file_metric_exporter_options.h>)
#include <opentelemetry/exporters/otlp/otlp_file_metric_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_metric_exporter_options.h>
#define CORTEXT_HAVE_OTLP_FILE_METRIC_EXPORTER 1
#endif

#if __has_include(<opentelemetry/exporters/otlp/otlp_file_log_record_exporter_factory.h>) \
    && __has_include(<opentelemetry/exporters/otlp/otlp_file_log_record_exporter_options.h>)
#include <opentelemetry/exporters/otlp/otlp_file_log_record_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_log_record_exporter_options.h>
#define CORTEXT_HAVE_OTLP_FILE_LOG_EXPORTER 1
#endif
#endif // defined(CORTEXT_ENABLE_OTEL)

namespace cortext::telemetry
{

#if defined(CORTEXT_ENABLE_OTEL)

namespace trace_api = opentelemetry::trace;
namespace metrics_api = opentelemetry::metrics;

namespace trace_sdk = opentelemetry::sdk::trace;
namespace metrics_sdk = opentelemetry::sdk::metrics;
namespace resource_sdk = opentelemetry::sdk::resource;

#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
namespace logs_api = opentelemetry::logs;
namespace logs_sdk = opentelemetry::sdk::logs;
#endif

namespace
{

std::string
GetEnvString (const char *name)
{
  const char *v = std::getenv (name);
  if (v == nullptr)
    {
      return {};
    }
  return std::string (v);
}

std::optional<std::uint64_t>
GetEnvU64 (const char *name)
{
  const std::string v = GetEnvString (name);
  if (v.empty ())
    {
      return std::nullopt;
    }
  try
    {
      const unsigned long long x = std::stoull (v);
      return static_cast<std::uint64_t> (x);
    }
  catch (...)
    {
      return std::nullopt;
    }
}

struct TelemetryState
{
  std::atomic<bool> is_initialized{ false };
  std::mutex mu;

  std::string file_export_path;
  std::unique_ptr<std::ofstream> file_stream;

  std::shared_ptr<trace_sdk::TracerProvider> trace_provider;
  std::shared_ptr<metrics_sdk::MeterProvider> meter_provider;
#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
  std::shared_ptr<logs_sdk::LoggerProvider> logger_provider;
#endif

  opentelemetry::nostd::shared_ptr<trace_api::Tracer> tracer;
  opentelemetry::nostd::shared_ptr<metrics_api::Meter> meter;
#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
  opentelemetry::nostd::shared_ptr<logs_api::Logger> logger;
#endif

  std::unordered_map<std::string, opentelemetry::nostd::shared_ptr<
                                   metrics_api::Counter<std::uint64_t> > >
      counters_u64;
  std::unordered_map<std::string, opentelemetry::nostd::shared_ptr<
                                   metrics_api::Histogram<double> > >
      histograms_f64;
};

TelemetryState &
GetState ()
{
  static TelemetryState state;
  return state;
}

std::unique_ptr<trace_sdk::Sampler>
MakeSamplerFromEnv ()
{
  const std::string sampler = GetEnvString ("OTEL_TRACES_SAMPLER");
  if (sampler.empty () || sampler == "parentbased_always_on")
    {
      return std::make_unique<trace_sdk::ParentBasedSampler> (
          std::make_shared<trace_sdk::AlwaysOnSampler> ());
    }
  if (sampler == "always_on")
    {
      return std::make_unique<trace_sdk::AlwaysOnSampler> ();
    }
  if (sampler == "always_off")
    {
      return std::make_unique<trace_sdk::AlwaysOffSampler> ();
    }
  if (sampler == "traceidratio")
    {
      const std::string arg = GetEnvString ("OTEL_TRACES_SAMPLER_ARG");
      double ratio = 1.0;
      if (!arg.empty ())
        {
          try
            {
              ratio = std::stod (arg);
            }
          catch (...)
            {
              ratio = 1.0;
            }
        }
      ratio = std::max (0.0, std::min (1.0, ratio));
      return std::make_unique<trace_sdk::TraceIdRatioBasedSampler> (ratio);
    }
  return std::make_unique<trace_sdk::ParentBasedSampler> (
      std::make_shared<trace_sdk::AlwaysOnSampler> ());
}

resource_sdk::Resource
MakeResourceFromEnv ()
{
  // service.name defaults to a stable value; override via OTEL_SERVICE_NAME.
  std::string service_name = GetEnvString ("OTEL_SERVICE_NAME");
  if (service_name.empty ())
    {
      service_name = "cortext";
    }

  auto attrs = resource_sdk::ResourceAttributes{
    { "service.name", service_name },
  };

  // Merge in standard OTEL_RESOURCE_ATTRIBUTES if present.
  // NOTE: We intentionally do not parse all attributes here; opentelemetry-cpp
  // also supports environment processing internally. We keep a minimal, safe
  // baseline.
  return resource_sdk::Resource::Create (attrs);
}

enum class OtlpProtocol
{
  kGrpc,
  kHttpProtobuf,
};

OtlpProtocol
GetOtlpProtocolFromEnv ()
{
  const std::string protocol = GetEnvString ("OTEL_EXPORTER_OTLP_PROTOCOL");
  if (protocol == "grpc")
    {
      return OtlpProtocol::kGrpc;
    }
  // Spec default is http/protobuf for OTLP.
  return OtlpProtocol::kHttpProtobuf;
}

bool
ExporterDisabled (const char *env_name)
{
  // Treat unset as disabled to avoid sending telemetry unless explicitly
  // configured in the environment.
  const char *v = std::getenv (env_name);
  if (v == nullptr)
    {
      return true;
    }
  const std::string_view sv (v);
  return sv.empty () || sv == "none";
}

bool
ExporterEquals (const char *env_name, const char *expected)
{
  const char *v = std::getenv (env_name);
  if (v == nullptr)
    {
      return false;
    }
  return std::string_view (v) == std::string_view (expected);
}

bool
ExporterUnset (const char *env_name)
{
  return std::getenv (env_name) == nullptr;
}

std::string
GetFileExporterPath ()
{
  return GetEnvString ("CORTEXT_OTEL_FILE_PATH");
}

bool
FileSignalEnabled (const std::string &signals_csv, const char *needle)
{
  if (signals_csv.empty ())
    {
      return true;
    }
  const std::string_view s (signals_csv);
  size_t start = 0;
  while (start < s.size ())
    {
      size_t end = s.find (',', start);
      if (end == std::string_view::npos)
        {
          end = s.size ();
        }
      std::string_view token = s.substr (start, end - start);
      while (!token.empty () && token.front () == ' ')
        token.remove_prefix (1);
      while (!token.empty () && token.back () == ' ')
        token.remove_suffix (1);
      if (token == needle)
        {
          return true;
        }
      start = end + 1;
    }
  return false;
}

bool
ShouldUseFileExporterFor (const char *otel_env, const char *signal_name)
{
  // File exporter is for local debugging. It activates when OTEL_* exporter
  // vars are unset or explicitly set to "none" (so we don't override explicit
  // OTEL configuration like "otlp").
  if (!ExporterUnset (otel_env) && !ExporterDisabled (otel_env))
    {
      return false;
    }
  const std::string path = GetFileExporterPath ();
  if (path.empty ())
    {
      return false;
    }
  const std::string signals_csv = GetEnvString ("CORTEXT_OTEL_FILE_SIGNALS");
  return FileSignalEnabled (signals_csv, signal_name);
}

} // namespace

struct ScopedSpan::Impl
{
  opentelemetry::nostd::shared_ptr<trace_api::Span> span;
  std::unique_ptr<trace_api::Scope> scope;
};

bool
InitializeFromEnv ()
{
  TelemetryState &state = GetState ();
  if (state.is_initialized.load ())
    {
      return true;
    }

  std::lock_guard<std::mutex> lock (state.mu);
  if (state.is_initialized.load ())
    {
      return true;
    }

  const resource_sdk::Resource resource = MakeResourceFromEnv ();

  // Optional local file exporter sink shared across signals.
  const std::string file_path = GetFileExporterPath ();
  if (!file_path.empty () && !state.file_stream)
    {
      state.file_export_path = file_path;
      state.file_stream = std::make_unique<std::ofstream> (
          file_path, std::ios::out | std::ios::app);
      if (!state.file_stream->is_open ())
        {
          state.file_stream.reset ();
          state.file_export_path.clear ();
        }
    }

  // Traces
  if (!ExporterDisabled ("OTEL_TRACES_EXPORTER")
      || ShouldUseFileExporterFor ("OTEL_TRACES_EXPORTER", "traces"))
    {
      auto sampler = MakeSamplerFromEnv ();
      std::unique_ptr<trace_sdk::SpanExporter> exporter;
      const OtlpProtocol protocol = GetOtlpProtocolFromEnv ();

#if defined(CORTEXT_HAVE_OSTREAM_TRACE_EXPORTER)
      // Prefer OTLP file exporter when available (writes JSONL: one line per record).
#if defined(CORTEXT_HAVE_OTLP_FILE_TRACE_EXPORTER)
      if (!exporter && state.file_stream
          && ShouldUseFileExporterFor ("OTEL_TRACES_EXPORTER", "traces"))
        {
          opentelemetry::exporter::otlp::OtlpFileExporterOptions opts;
          opts.backend_options
              = std::reference_wrapper<std::ostream> (*state.file_stream);
          exporter
              = opentelemetry::exporter::otlp::OtlpFileExporterFactory::Create (
                  opts);
        }
#endif
      if (!exporter && state.file_stream
          && ShouldUseFileExporterFor ("OTEL_TRACES_EXPORTER", "traces"))
        {
          exporter = std::make_unique<
              opentelemetry::exporter::trace::OStreamSpanExporter> (
              *state.file_stream);
        }
#endif

#if defined(CORTEXT_HAVE_OTLP_GRPC_TRACE_EXPORTER)
      if (!exporter && ExporterEquals ("OTEL_TRACES_EXPORTER", "otlp")
          && protocol == OtlpProtocol::kGrpc)
        {
          exporter
              = opentelemetry::exporter::otlp::OtlpGrpcExporterFactory::Create
                    ();
        }
#endif
#if defined(CORTEXT_HAVE_OTLP_HTTP_TRACE_EXPORTER)
      if (!exporter && ExporterEquals ("OTEL_TRACES_EXPORTER", "otlp")
          && protocol == OtlpProtocol::kHttpProtobuf)
        {
          exporter
              = opentelemetry::exporter::otlp::OtlpHttpExporterFactory::Create
                    ();
        }
#endif

      if (exporter)
        {
      trace_sdk::BatchSpanProcessorOptions processor_options;
      // Standard OTel env vars for batch span processor (milliseconds).
      if (const auto v = GetEnvU64 ("OTEL_BSP_SCHEDULE_DELAY"); v.has_value ())
        {
          processor_options.schedule_delay_millis
              = std::chrono::milliseconds (*v);
        }
      if (const auto v = GetEnvU64 ("OTEL_BSP_MAX_QUEUE_SIZE"); v.has_value ())
        {
          processor_options.max_queue_size = static_cast<size_t> (*v);
        }
      if (const auto v = GetEnvU64 ("OTEL_BSP_MAX_EXPORT_BATCH_SIZE");
          v.has_value ())
        {
          processor_options.max_export_batch_size = static_cast<size_t> (*v);
        }
          auto processor = std::make_unique<trace_sdk::BatchSpanProcessor> (
              std::move (exporter), processor_options);
          auto sdk_provider = std::make_shared<trace_sdk::TracerProvider> (
              std::move (processor), resource, std::move (sampler));
          state.trace_provider = sdk_provider;
          std::shared_ptr<trace_api::TracerProvider> api_provider = sdk_provider;
          trace_api::Provider::SetTracerProvider (
              opentelemetry::nostd::shared_ptr<trace_api::TracerProvider> (
                  api_provider));
          state.tracer = trace_api::Provider::GetTracerProvider ()->GetTracer (
              "cortext");
        }
    }

  // Metrics
  if (!ExporterDisabled ("OTEL_METRICS_EXPORTER")
      || ShouldUseFileExporterFor ("OTEL_METRICS_EXPORTER", "metrics"))
    {
      auto meter_provider = std::make_shared<metrics_sdk::MeterProvider> ();
      std::unique_ptr<metrics_sdk::PushMetricExporter> metric_exporter;
      const OtlpProtocol protocol = GetOtlpProtocolFromEnv ();

#if defined(CORTEXT_HAVE_OSTREAM_METRIC_EXPORTER)
#if defined(CORTEXT_HAVE_OTLP_FILE_METRIC_EXPORTER)
      if (!metric_exporter && state.file_stream
          && ShouldUseFileExporterFor ("OTEL_METRICS_EXPORTER", "metrics"))
        {
          opentelemetry::exporter::otlp::OtlpFileMetricExporterOptions opts;
          opts.backend_options
              = std::reference_wrapper<std::ostream> (*state.file_stream);
          metric_exporter = opentelemetry::exporter::otlp::
              OtlpFileMetricExporterFactory::Create (opts);
        }
#endif
      if (!metric_exporter && state.file_stream
          && ShouldUseFileExporterFor ("OTEL_METRICS_EXPORTER", "metrics"))
        {
          metric_exporter = std::make_unique<
              opentelemetry::exporter::metrics::OStreamMetricExporter> (
              *state.file_stream);
        }
#endif
#if defined(CORTEXT_HAVE_OTLP_GRPC_METRIC_EXPORTER)
      if (!metric_exporter && ExporterEquals ("OTEL_METRICS_EXPORTER", "otlp")
          && protocol == OtlpProtocol::kGrpc)
        {
          metric_exporter
              = opentelemetry::exporter::otlp::OtlpGrpcMetricExporterFactory::
                  Create ();
        }
#endif
#if defined(CORTEXT_HAVE_OTLP_HTTP_METRIC_EXPORTER)
      if (!metric_exporter && ExporterEquals ("OTEL_METRICS_EXPORTER", "otlp")
          && protocol == OtlpProtocol::kHttpProtobuf)
        {
          metric_exporter
              = opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::
                  Create ();
        }
#endif
      if (metric_exporter)
        {
      metrics_sdk::PeriodicExportingMetricReaderOptions reader_options;
      // Standard OTel env vars for periodic metric reader (milliseconds).
      if (const auto v = GetEnvU64 ("OTEL_METRIC_EXPORT_INTERVAL");
          v.has_value ())
        {
          reader_options.export_interval_millis
              = std::chrono::milliseconds (*v);
        }
      if (const auto v = GetEnvU64 ("OTEL_METRIC_EXPORT_TIMEOUT");
          v.has_value ())
        {
          reader_options.export_timeout_millis
              = std::chrono::milliseconds (*v);
        }
          meter_provider->AddMetricReader (
              std::make_unique<metrics_sdk::PeriodicExportingMetricReader> (
                  std::move (metric_exporter), reader_options));
        }

      state.meter_provider = meter_provider;
      std::shared_ptr<metrics_api::MeterProvider> api_meter_provider
          = meter_provider;
      metrics_api::Provider::SetMeterProvider (
          opentelemetry::nostd::shared_ptr<metrics_api::MeterProvider> (
              api_meter_provider));
      state.meter
          = metrics_api::Provider::GetMeterProvider ()->GetMeter ("cortext");
    }

#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
  if (!ExporterDisabled ("OTEL_LOGS_EXPORTER")
      || ShouldUseFileExporterFor ("OTEL_LOGS_EXPORTER", "logs"))
    {
      std::unique_ptr<logs_sdk::LogRecordExporter> log_exporter;
      const OtlpProtocol protocol = GetOtlpProtocolFromEnv ();

#if defined(CORTEXT_HAVE_OSTREAM_LOG_EXPORTER)
#if defined(CORTEXT_HAVE_OTLP_FILE_LOG_EXPORTER)
      if (!log_exporter && state.file_stream
          && ShouldUseFileExporterFor ("OTEL_LOGS_EXPORTER", "logs"))
        {
          opentelemetry::exporter::otlp::OtlpFileLogRecordExporterOptions opts;
          opts.backend_options
              = std::reference_wrapper<std::ostream> (*state.file_stream);
          log_exporter = opentelemetry::exporter::otlp::
              OtlpFileLogRecordExporterFactory::Create (opts);
        }
#endif
      if (!log_exporter && state.file_stream
          && ShouldUseFileExporterFor ("OTEL_LOGS_EXPORTER", "logs"))
        {
          log_exporter = std::make_unique<
              opentelemetry::exporter::logs::OStreamLogRecordExporter> (
              *state.file_stream);
        }
#endif

#if defined(CORTEXT_HAVE_OTLP_GRPC_LOG_EXPORTER)
      if (!log_exporter && ExporterEquals ("OTEL_LOGS_EXPORTER", "otlp")
          && protocol == OtlpProtocol::kGrpc)
        {
          log_exporter = opentelemetry::exporter::otlp::
              OtlpGrpcLogRecordExporterFactory::Create ();
        }
#endif
#if defined(CORTEXT_HAVE_OTLP_HTTP_LOG_EXPORTER)
      if (!log_exporter && ExporterEquals ("OTEL_LOGS_EXPORTER", "otlp")
          && protocol == OtlpProtocol::kHttpProtobuf)
        {
          log_exporter = opentelemetry::exporter::otlp::
              OtlpHttpLogRecordExporterFactory::Create ();
        }
#endif

      if (log_exporter)
        {
          auto processor = std::make_unique<logs_sdk::BatchLogRecordProcessor> (
              std::move (log_exporter));
          auto sdk_provider
              = std::make_shared<logs_sdk::LoggerProvider> (
                  std::move (processor), resource);
          state.logger_provider = sdk_provider;
          std::shared_ptr<logs_api::LoggerProvider> api_provider = sdk_provider;
          logs_api::Provider::SetLoggerProvider (
              opentelemetry::nostd::shared_ptr<logs_api::LoggerProvider> (
                  api_provider));
          state.logger
              = logs_api::Provider::GetLoggerProvider ()->GetLogger ("cortext");
        }
    }
#endif

  state.is_initialized.store (true);
  return true;
}

void
ForceFlush ()
{
  TelemetryState &state = GetState ();
  if (!state.is_initialized.load ())
    {
      return;
    }
  if (state.trace_provider)
    {
      state.trace_provider->ForceFlush ();
    }
  if (state.meter_provider)
    {
      state.meter_provider->ForceFlush ();
    }
#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
  if (state.logger_provider)
    {
      state.logger_provider->ForceFlush ();
    }
#endif
}

void
Shutdown ()
{
  TelemetryState &state = GetState ();
  if (!state.is_initialized.load ())
    {
      return;
    }
  ForceFlush ();
  state.trace_provider.reset ();
  state.meter_provider.reset ();
#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
  state.logger_provider.reset ();
#endif
  state.tracer = nullptr;
  state.meter = nullptr;
#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
  state.logger = nullptr;
#endif
  state.file_stream.reset ();
  state.file_export_path.clear ();
}

ScopedSpan::ScopedSpan (std::string_view name) : impl_ (nullptr)
{
  InitializeFromEnv ();
  TelemetryState &state = GetState ();
  if (!state.tracer)
    {
      return;
    }
  impl_ = std::make_unique<ScopedSpan::Impl> ();
  impl_->span = state.tracer->StartSpan (std::string (name));
  impl_->scope = std::make_unique<trace_api::Scope> (impl_->span);
}

ScopedSpan::ScopedSpan (std::string_view name,
                        std::initializer_list<Attribute> attrs)
    : ScopedSpan (name)
{
  for (const auto &a : attrs)
    {
      switch (a.type)
        {
        case Attribute::Type::kBool:
          SetAttribute (a.key, a.bool_value);
          break;
        case Attribute::Type::kInt64:
          SetAttribute (a.key, a.int64_value);
          break;
        case Attribute::Type::kDouble:
          SetAttribute (a.key, a.double_value);
          break;
        case Attribute::Type::kString:
        default:
          SetAttribute (a.key, a.string_value);
          break;
        }
    }
}

ScopedSpan::~ScopedSpan ()
{
  if (impl_ && impl_->span)
    {
      impl_->span->End ();
    }
}

ScopedSpan::ScopedSpan (ScopedSpan &&other) noexcept : impl_ (nullptr)
{
  impl_ = std::move (other.impl_);
}

ScopedSpan &
ScopedSpan::operator= (ScopedSpan &&other) noexcept
{
  if (this != &other)
    {
      impl_ = std::move (other.impl_);
    }
  return *this;
}

bool
ScopedSpan::IsRecording () const
{
  return impl_ && impl_->span && impl_->span->IsRecording ();
}

void
ScopedSpan::SetAttribute (std::string_view key, bool value)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetAttribute (std::string (key), value);
}

void
ScopedSpan::SetAttribute (std::string_view key, std::int64_t value)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetAttribute (std::string (key), value);
}

void
ScopedSpan::SetAttribute (std::string_view key, double value)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetAttribute (std::string (key), value);
}

void
ScopedSpan::SetAttribute (std::string_view key, std::string_view value)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetAttribute (std::string (key), std::string (value));
}

void
ScopedSpan::AddEvent (std::string_view name)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->AddEvent (std::string (name));
}

void
ScopedSpan::SetStatusOk ()
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetStatus (trace_api::StatusCode::kOk);
}

void
ScopedSpan::SetStatusError (std::string_view message)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetStatus (trace_api::StatusCode::kError, std::string (message));
}

void
AddCounter (std::string_view name, std::uint64_t delta)
{
  InitializeFromEnv ();
  TelemetryState &state = GetState ();
  if (!state.meter)
    {
      return;
    }

  std::lock_guard<std::mutex> lock (state.mu);
  auto it = state.counters_u64.find (std::string (name));
  if (it == state.counters_u64.end ())
    {
      auto c = state.meter->CreateUInt64Counter (std::string (name));
      it = state.counters_u64.emplace (std::string (name), std::move (c)).first;
    }
  it->second->Add (delta, opentelemetry::context::RuntimeContext::GetCurrent ());
}

void
RecordHistogram (std::string_view name, double value)
{
  InitializeFromEnv ();
  TelemetryState &state = GetState ();
  if (!state.meter)
    {
      return;
    }

  std::lock_guard<std::mutex> lock (state.mu);
  auto it = state.histograms_f64.find (std::string (name));
  if (it == state.histograms_f64.end ())
    {
      auto h = state.meter->CreateDoubleHistogram (std::string (name));
      it = state.histograms_f64.emplace (std::string (name), std::move (h))
               .first;
    }
  it->second->Record (value, opentelemetry::context::RuntimeContext::GetCurrent ());
}

namespace
{

enum class LogLevel
{
  kError,
  kWarn,
  kInfo,
};

LogLevel
GetLogLevelFromEnv ()
{
  // Reuse existing env convention in this repo.
  const std::string level = GetEnvString ("LOG_LEVEL");
  if (level == "ERROR")
    {
      return LogLevel::kError;
    }
  if (level == "WARN" || level == "WARNING")
    {
      return LogLevel::kWarn;
    }
  return LogLevel::kInfo;
}

} // namespace

void
LogError (std::string_view message)
{
  LogError (message, {});
}

void
LogWarn (std::string_view message)
{
  LogWarn (message, {});
}

void
LogInfo (std::string_view message)
{
  LogInfo (message, {});
}

namespace
{

void
ApplySpanLogAttributes (ScopedSpan &span,
                        std::initializer_list<Attribute> attrs)
{
  for (const auto &a : attrs)
    {
      switch (a.type)
        {
        case Attribute::Type::kBool:
          span.SetAttribute (a.key, a.bool_value);
          break;
        case Attribute::Type::kInt64:
          span.SetAttribute (a.key, a.int64_value);
          break;
        case Attribute::Type::kDouble:
          span.SetAttribute (a.key, a.double_value);
          break;
        case Attribute::Type::kString:
        default:
          span.SetAttribute (a.key, a.string_value);
          break;
        }
    }
}

} // namespace

void
LogError (std::string_view message, std::initializer_list<Attribute> attrs)
{
  const LogLevel lvl = GetLogLevelFromEnv ();
  if (lvl == LogLevel::kInfo || lvl == LogLevel::kWarn || lvl == LogLevel::kError)
    {
      ScopedSpan span ("cortext.log");
      span.SetAttribute ("log.severity", "error");
      span.SetAttribute ("log.message", message);
      ApplySpanLogAttributes (span, attrs);
      span.AddEvent ("cortext.log.error");
#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
      InitializeFromEnv ();
      TelemetryState &state = GetState ();
      if (state.logger)
        {
          std::vector<std::pair<opentelemetry::nostd::string_view,
                                opentelemetry::common::AttributeValue> >
              kvs;
          kvs.reserve (attrs.size ());
          for (const auto &a : attrs)
            {
              const opentelemetry::nostd::string_view k (a.key.data (),
                                                        a.key.size ());
              switch (a.type)
                {
                case Attribute::Type::kBool:
                  kvs.emplace_back (k, a.bool_value);
                  break;
                case Attribute::Type::kInt64:
                  kvs.emplace_back (k, a.int64_value);
                  break;
                case Attribute::Type::kDouble:
                  kvs.emplace_back (k, a.double_value);
                  break;
                case Attribute::Type::kString:
                default:
                  kvs.emplace_back (
                      k, opentelemetry::nostd::string_view (
                             a.string_value.data (), a.string_value.size ()));
                  break;
                }
            }
          state.logger->EmitLogRecord (
              logs_api::Severity::kError,
              opentelemetry::nostd::string_view (message.data (),
                                                 message.size ()),
              kvs);
        }
#endif
    }
}

void
LogWarn (std::string_view message, std::initializer_list<Attribute> attrs)
{
  const LogLevel lvl = GetLogLevelFromEnv ();
  if (lvl == LogLevel::kInfo || lvl == LogLevel::kWarn)
    {
      ScopedSpan span ("cortext.log");
      span.SetAttribute ("log.severity", "warn");
      span.SetAttribute ("log.message", message);
      ApplySpanLogAttributes (span, attrs);
      span.AddEvent ("cortext.log.warn");
#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
      InitializeFromEnv ();
      TelemetryState &state = GetState ();
      if (state.logger)
        {
          std::vector<std::pair<opentelemetry::nostd::string_view,
                                opentelemetry::common::AttributeValue> >
              kvs;
          kvs.reserve (attrs.size ());
          for (const auto &a : attrs)
            {
              const opentelemetry::nostd::string_view k (a.key.data (),
                                                        a.key.size ());
              switch (a.type)
                {
                case Attribute::Type::kBool:
                  kvs.emplace_back (k, a.bool_value);
                  break;
                case Attribute::Type::kInt64:
                  kvs.emplace_back (k, a.int64_value);
                  break;
                case Attribute::Type::kDouble:
                  kvs.emplace_back (k, a.double_value);
                  break;
                case Attribute::Type::kString:
                default:
                  kvs.emplace_back (
                      k, opentelemetry::nostd::string_view (
                             a.string_value.data (), a.string_value.size ()));
                  break;
                }
            }
          state.logger->EmitLogRecord (
              logs_api::Severity::kWarn,
              opentelemetry::nostd::string_view (message.data (),
                                                 message.size ()),
              kvs);
        }
#endif
    }
}

void
LogInfo (std::string_view message, std::initializer_list<Attribute> attrs)
{
  const LogLevel lvl = GetLogLevelFromEnv ();
  if (lvl == LogLevel::kInfo)
    {
      ScopedSpan span ("cortext.log");
      span.SetAttribute ("log.severity", "info");
      span.SetAttribute ("log.message", message);
      ApplySpanLogAttributes (span, attrs);
      span.AddEvent ("cortext.log.info");
#if defined(CORTEXT_HAVE_OTEL_LOGS_API)
      InitializeFromEnv ();
      TelemetryState &state = GetState ();
      if (state.logger)
        {
          std::vector<std::pair<opentelemetry::nostd::string_view,
                                opentelemetry::common::AttributeValue> >
              kvs;
          kvs.reserve (attrs.size ());
          for (const auto &a : attrs)
            {
              const opentelemetry::nostd::string_view k (a.key.data (),
                                                        a.key.size ());
              switch (a.type)
                {
                case Attribute::Type::kBool:
                  kvs.emplace_back (k, a.bool_value);
                  break;
                case Attribute::Type::kInt64:
                  kvs.emplace_back (k, a.int64_value);
                  break;
                case Attribute::Type::kDouble:
                  kvs.emplace_back (k, a.double_value);
                  break;
                case Attribute::Type::kString:
                default:
                  kvs.emplace_back (
                      k, opentelemetry::nostd::string_view (
                             a.string_value.data (), a.string_value.size ()));
                  break;
                }
            }
          state.logger->EmitLogRecord (
              logs_api::Severity::kInfo,
              opentelemetry::nostd::string_view (message.data (),
                                                 message.size ()),
              kvs);
        }
#endif
    }
}

#else  // !defined(CORTEXT_ENABLE_OTEL)

struct ScopedSpan::Impl
{
};

bool
InitializeFromEnv ()
{
  return false;
}

void
ForceFlush ()
{
}

void
Shutdown ()
{
}

ScopedSpan::ScopedSpan (std::string_view) : impl_ (nullptr)
{
}

ScopedSpan::ScopedSpan (std::string_view, std::initializer_list<Attribute>)
    : impl_ (nullptr)
{
}

ScopedSpan::~ScopedSpan () = default;

ScopedSpan::ScopedSpan (ScopedSpan &&other) noexcept : impl_ (nullptr)
{
  impl_ = std::move (other.impl_);
}

ScopedSpan &
ScopedSpan::operator= (ScopedSpan &&other) noexcept
{
  if (this != &other)
    {
      impl_ = std::move (other.impl_);
    }
  return *this;
}

bool
ScopedSpan::IsRecording () const
{
  return false;
}

void
ScopedSpan::SetAttribute (std::string_view, bool)
{
}

void
ScopedSpan::SetAttribute (std::string_view, std::int64_t)
{
}

void
ScopedSpan::SetAttribute (std::string_view, double)
{
}

void
ScopedSpan::SetAttribute (std::string_view, std::string_view)
{
}

void
ScopedSpan::AddEvent (std::string_view)
{
}

void
ScopedSpan::SetStatusOk ()
{
}

void
ScopedSpan::SetStatusError (std::string_view)
{
}

void
AddCounter (std::string_view, std::uint64_t)
{
}

void
RecordHistogram (std::string_view, double)
{
}

void
LogError (std::string_view)
{
}

void
LogWarn (std::string_view)
{
}

void
LogInfo (std::string_view)
{
}

void
LogError (std::string_view, std::initializer_list<Attribute>)
{
}

void
LogWarn (std::string_view, std::initializer_list<Attribute>)
{
}

void
LogInfo (std::string_view, std::initializer_list<Attribute>)
{
}

#endif // defined(CORTEXT_ENABLE_OTEL)

} // namespace cortext::telemetry


