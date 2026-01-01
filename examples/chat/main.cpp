#include "chat_window.hpp"
#include "context_tab.hpp"
#include "imgui_app.hpp"
#include "streaming_client.hpp"

#include <cortext/cortext.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/encoder/imagebind.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/telemetry/telemetry.hpp>

#include <openai/openai.hpp>

#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/trace/provider.h>

#include <opentelemetry/sdk/metrics/data/metric_data.h>
#include <opentelemetry/sdk/metrics/export/metric_producer.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>
#include <opentelemetry/sdk/metrics/meter_provider_factory.h>
#include <opentelemetry/sdk/metrics/push_metric_exporter.h>
#include <opentelemetry/sdk/resource/resource.h>
#include <opentelemetry/sdk/trace/recordable.h>
#include <opentelemetry/sdk/trace/simple_processor.h>
#include <opentelemetry/sdk/trace/span_data.h>
#include <opentelemetry/sdk/trace/exporter.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/sdk/trace/batch_span_processor.h>

#include <opentelemetry/exporters/otlp/otlp_grpc_exporter.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_metric_exporter.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_metric_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_options.h>

#include <opentelemetry/sdk/logs/read_write_log_record.h>

#include <opentelemetry/logs/provider.h>
#include <opentelemetry/sdk/logs/logger_provider_factory.h>
#include <opentelemetry/sdk/logs/simple_log_record_processor_factory.h>

#include <any>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <map>

namespace {

// Simple log exporter with pretty one-line format to stdout and file
class SimpleStdoutLogExporter final : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  explicit SimpleStdoutLogExporter(const std::string& log_file_path = "")
      : log_file_path_(log_file_path) {
    if (!log_file_path_.empty()) {
      log_file_.open(log_file_path_, std::ios::out | std::ios::trunc);
    }
  }

  ~SimpleStdoutLogExporter() override {
    if (log_file_.is_open()) {
      log_file_.close();
    }
  }

  std::unique_ptr<opentelemetry::sdk::logs::Recordable> MakeRecordable() noexcept override {
    return std::make_unique<opentelemetry::sdk::logs::ReadWriteLogRecord>();
  }

  opentelemetry::sdk::common::ExportResult Export(
      const opentelemetry::nostd::span<std::unique_ptr<opentelemetry::sdk::logs::Recordable>>& records) noexcept override {
    for (const auto& record : records) {
      auto* log_record = static_cast<opentelemetry::sdk::logs::ReadWriteLogRecord*>(record.get());
      if (!log_record) continue;

      // Get severity text
      auto severity = log_record->GetSeverity();
      std::uint32_t severity_index = static_cast<std::uint32_t>(severity);
      std::string severity_text = "UNKNOWN";
      if (severity_index < std::extent<decltype(opentelemetry::logs::SeverityNumToText)>::value) {
        auto sv = opentelemetry::logs::SeverityNumToText[severity_index];
        severity_text = std::string(sv.data(), sv.size());
      }

      // Get body as string
      std::string body;
      const auto& body_val = log_record->GetBody();
      if (opentelemetry::nostd::holds_alternative<std::string>(body_val)) {
        body = opentelemetry::nostd::get<std::string>(body_val);
      }

      // Get attributes - simplified output
      std::ostringstream attrs;
      bool first = true;
      for (const auto& kv : log_record->GetAttributes()) {
        if (!first) attrs << ", ";
        first = false;
        attrs << kv.first << "=";
        // Use OwnedAttributeValue visitor
        opentelemetry::nostd::visit([&attrs](auto&& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, bool>) { attrs << (v ? "true" : "false"); }
          else if constexpr (std::is_same_v<T, int32_t>) { attrs << v; }
          else if constexpr (std::is_same_v<T, int64_t>) { attrs << v; }
          else if constexpr (std::is_same_v<T, uint32_t>) { attrs << v; }
          else if constexpr (std::is_same_v<T, uint64_t>) { attrs << v; }
          else if constexpr (std::is_same_v<T, double>) { attrs << v; }
          else if constexpr (std::is_same_v<T, std::string>) { attrs << v; }
          else { attrs << "?"; }
        }, kv.second);
      }

      // Format: [SEVERITY] body {attrs}
      std::ostringstream line;
      line << "[" << severity_text << "] " << body;
      if (!first) line << " {" << attrs.str() << "}";
      line << "\n";

      // Write to stdout
      std::cout << line.str();
      std::cout.flush();

      // Write to file if open
      if (log_file_.is_open()) {
        log_file_ << line.str();
        log_file_.flush();
      }
    }
    return opentelemetry::sdk::common::ExportResult::kSuccess;
  }

  bool ForceFlush(std::chrono::microseconds) noexcept override {
    std::cout.flush();
    if (log_file_.is_open()) {
      log_file_.flush();
    }
    return true;
  }

  bool Shutdown(std::chrono::microseconds) noexcept override {
    if (log_file_.is_open()) {
      log_file_.close();
    }
    return true;
  }

private:
  std::string log_file_path_;
  std::ofstream log_file_;
};

struct RetrievalLatencyState {
  std::optional<double> encoder_latency_ms;
  std::optional<double> retrieval_latency_ms;
};

class InMemorySpanExporter final : public opentelemetry::sdk::trace::SpanExporter {
public:
  explicit InMemorySpanExporter(std::shared_ptr<RetrievalLatencyState> retrieval_state)
      : retrieval_state_(std::move(retrieval_state)) {}
  std::unique_ptr<opentelemetry::sdk::trace::Recordable> MakeRecordable() noexcept override {
    return std::unique_ptr<opentelemetry::sdk::trace::Recordable>(new opentelemetry::sdk::trace::SpanData());
  }
  opentelemetry::sdk::common::ExportResult Export(const opentelemetry::nostd::span<std::unique_ptr<opentelemetry::sdk::trace::Recordable>> &spans) noexcept override {
    for (const auto &span : spans) {
      if (!span) { continue; }
      auto *span_data = static_cast<opentelemetry::sdk::trace::SpanData *>(*span);
      if (!span_data) { continue; }
      const std::string name = std::string(span_data->GetName());
      const double duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(span_data->GetDuration()).count();
      if (name == "cortext.encode") { retrieval_active_ = true; retrieval_db_select_ms_ = 0.0; retrieval_last_encode_ms_ = duration_ms; continue; }
      if (name == "cortext.hydrate") { retrieval_last_hydrate_ms_ = duration_ms; continue; }
      if (name == "cortext.db.execute") {
        const auto &attrs = span_data->GetAttributes();
        const auto it = attrs.find("db.operation");
        if (it != attrs.end() && opentelemetry::nostd::holds_alternative<std::string>(it->second)) {
          const auto &op = opentelemetry::nostd::get<std::string>(it->second);
          if (retrieval_active_ && op == "SELECT") { retrieval_db_select_ms_ += duration_ms; }
        }
        continue;
      }
      if (name == "cortext.api.process_text") {
        if (retrieval_state_) {
          retrieval_state_->encoder_latency_ms = retrieval_last_encode_ms_;
          if (retrieval_last_hydrate_ms_.has_value()) { retrieval_state_->retrieval_latency_ms = retrieval_db_select_ms_ + *retrieval_last_hydrate_ms_; }
        }
        retrieval_active_ = false;
        retrieval_last_encode_ms_.reset();
        retrieval_last_hydrate_ms_.reset();
        continue;
      }
    }
    return opentelemetry::sdk::common::ExportResult::kSuccess;
  }
  bool ForceFlush(std::chrono::microseconds) noexcept override { return true; }
  bool Shutdown(std::chrono::microseconds) noexcept override { return true; }
private:
  std::shared_ptr<RetrievalLatencyState> retrieval_state_;
  thread_local static bool retrieval_active_;
  thread_local static double retrieval_db_select_ms_;
  thread_local static std::optional<double> retrieval_last_encode_ms_;
  thread_local static std::optional<double> retrieval_last_hydrate_ms_;
};

thread_local bool InMemorySpanExporter::retrieval_active_ = false;
thread_local double InMemorySpanExporter::retrieval_db_select_ms_ = 0.0;
thread_local std::optional<double> InMemorySpanExporter::retrieval_last_encode_ms_ = std::nullopt;
thread_local std::optional<double> InMemorySpanExporter::retrieval_last_hydrate_ms_ = std::nullopt;

class InMemoryMetricExporter final : public opentelemetry::sdk::metrics::PushMetricExporter {
public:
  explicit InMemoryMetricExporter(std::shared_ptr<chat::StatusBarState> status_state) : status_state_(std::move(status_state)) {}
  opentelemetry::sdk::common::ExportResult Export(const opentelemetry::sdk::metrics::ResourceMetrics &data) noexcept override {
    if (!status_state_) { return opentelemetry::sdk::common::ExportResult::kSuccess; }
    std::lock_guard<std::mutex> lock(status_state_->mu);
    for (const auto &scope_metrics : data.scope_metric_data_) {
      for (const auto &metric : scope_metrics.metric_data_) {
        const std::string name = metric.instrument_descriptor.name_;
        if (name == "chat.tokens_used") {
          for (const auto &point : metric.point_data_attr_) {
            if (opentelemetry::nostd::holds_alternative<opentelemetry::sdk::metrics::LastValuePointData>(point.point_data)) {
              const auto &lv = opentelemetry::nostd::get<opentelemetry::sdk::metrics::LastValuePointData>(point.point_data);
              if (!lv.is_lastvalue_valid_) { continue; }
              if (opentelemetry::nostd::holds_alternative<std::int64_t>(lv.value_)) { status_state_->tokens_used = opentelemetry::nostd::get<std::int64_t>(lv.value_); }
            }
          }
        }
        if (name == "cortext.process_duration_ms") {
          for (const auto &point : metric.point_data_attr_) {
            if (opentelemetry::nostd::holds_alternative<opentelemetry::sdk::metrics::HistogramPointData>(point.point_data)) {
              const auto &h = opentelemetry::nostd::get<opentelemetry::sdk::metrics::HistogramPointData>(point.point_data);
              if (h.count_ == 0) { continue; }
              double sum = 0.0;
              if (opentelemetry::nostd::holds_alternative<double>(h.sum_)) { sum = opentelemetry::nostd::get<double>(h.sum_); }
              else if (opentelemetry::nostd::holds_alternative<std::int64_t>(h.sum_)) { sum = static_cast<double>(opentelemetry::nostd::get<std::int64_t>(h.sum_)); }
              status_state_->processing_latency_ms = sum / static_cast<double>(h.count_);
            }
          }
        }
      }
    }
    return opentelemetry::sdk::common::ExportResult::kSuccess;
  }
  bool ForceFlush(std::chrono::microseconds) noexcept override { return true; }
  bool Shutdown(std::chrono::microseconds) noexcept override { return true; }
  opentelemetry::sdk::metrics::AggregationTemporality GetAggregationTemporality(opentelemetry::sdk::metrics::InstrumentType) const noexcept override {
    return opentelemetry::sdk::metrics::AggregationTemporality::kDelta;
  }
private:
  std::shared_ptr<chat::StatusBarState> status_state_;
};

struct StreamingState {
  std::atomic<bool> cancel_requested{false};
  std::string accumulated_tokens;
  std::vector<cortext::Cortext::Context::Memory> current_memories;
  int restart_count = 0;
  static constexpr int kMaxRestarts = 3;

  void Reset() {
    cancel_requested.store(false);
    accumulated_tokens.clear();
    restart_count = 0;
  }
};

struct IdleTracker {
  mutable std::mutex mu;
  std::chrono::steady_clock::time_point last_activity;
  bool consolidation_pending = false;

  IdleTracker() : last_activity(std::chrono::steady_clock::now()) {}

  void RecordActivity() {
    std::lock_guard<std::mutex> lock(mu);
    last_activity = std::chrono::steady_clock::now();
    consolidation_pending = true;
  }

  bool ShouldConsolidate(double stability) {
    std::lock_guard<std::mutex> lock(mu);
    if (!consolidation_pending) return false;
    int idle_required = cortext::core::IdleRequiredSeconds(stability);
    auto idle = std::chrono::steady_clock::now() - last_activity;
    return std::chrono::duration_cast<std::chrono::seconds>(idle).count() >= idle_required;
  }

  void MarkConsolidated() {
    std::lock_guard<std::mutex> lock(mu);
    consolidation_pending = false;
  }
};

/// @brief Extract text content from binary blobs (concatenated UTF-8).
/// @param blobs Vector of binary blobs (one per signal).
/// @return Concatenated text content.
std::string ExtractTextFromBlobs(const std::vector<std::vector<unsigned char>>& blobs) {
  if (blobs.empty()) return {};
  std::string result;
  for (const auto& blob : blobs) {
    if (!blob.empty()) {
      result.append(reinterpret_cast<const char*>(blob.data()), blob.size());
    }
  }
  return result;
}

chat::MemoryEvent
CreateMemoryEvent(chat::MemoryEventType type,
                  const cortext::Cortext::Context::Memory &mem)
{
  chat::MemoryEvent evt;
  evt.type = type;
  evt.memory_id = static_cast<int>(mem.id);
  evt.content = ExtractTextFromBlobs(mem.content);
  evt.source_id = mem.source_id;
  evt.timestamp = mem.timestamp;
  evt.retrieved_count = mem.retrieved_count;
  evt.used_count = mem.used_count;

  evt.relevance = mem.relevance;
  evt.mismatch = mem.mismatch;
  evt.surprise = mem.surprise;
  evt.rarity = mem.rarity;
  evt.drift = mem.drift;
  evt.contradiction = mem.contradiction;
  evt.utility = mem.utility;
  evt.periphery = mem.periphery;
  evt.coverage = mem.coverage;
  evt.salience = mem.salience;
  evt.valence = mem.valence;
  evt.arousal = mem.arousal;
  evt.composite_score = mem.composite_score;
  evt.threshold_t = mem.threshold_t;

  return evt;
}

chat::MemoryEvent
CreateStoredEvent(long long embedding_id, const std::string &content,
                  const std::string &source_id, uint64_t ts,
                  const cortext::Cortext::ProcessorOutput &output)
{
  chat::MemoryEvent evt;
  evt.type = chat::MemoryEventType::STORED;
  evt.memory_id = static_cast<int>(embedding_id);
  evt.content = content;
  evt.source_id = source_id;
  evt.timestamp = ts;
  evt.retrieved_count = 0;
  evt.used_count = 0;

  using M = cortext::operations::Metric;
  auto get = [&](M m) -> double {
    auto it = output.metrics.find(static_cast<int>(m));
    return it != output.metrics.end() ? it->second : 0.0;
  };
  evt.relevance = get(M::relevance);
  evt.mismatch = get(M::mismatch);
  evt.surprise = get(M::surprise);
  evt.rarity = get(M::rarity);
  evt.drift = get(M::drift);
  evt.contradiction = get(M::contradiction);
  evt.utility = get(M::utility);
  evt.periphery = get(M::periphery);
  evt.coverage = get(M::coverage);
  evt.salience = get(M::salience);
  evt.valence = output.valence;
  evt.arousal = output.arousal;
  evt.composite_score = output.composite_score.value_or(0.0);
  evt.threshold_t = output.threshold.value_or(0.0);

  return evt;
}

std::string GetEnv(const char* name, const std::string& fallback = {}) {
  const char* v = std::getenv(name);
  if (!v) return fallback;
  return std::string(v);
}

bool HasEnv(const char* name) {
  return std::getenv(name) != nullptr;
}

std::string EnsureTrailingSlash(std::string url) {
  if (!url.empty() && url.back() != '/') url.push_back('/');
  return url;
}

double GetEnvDouble(const char* name, double fallback) {
  const std::string s = GetEnv(name);
  if (s.empty()) return fallback;
  try {
    return std::stod(s);
  } catch (...) {
    return fallback;
  }
}

bool GetEnvBool(const char* name, bool fallback) {
  const std::string s = GetEnv(name);
  if (s.empty()) return fallback;
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") return true;
  if (lower == "0" || lower == "false" || lower == "no" || lower == "off") return false;
  return fallback;
}

uint64_t NowUnixMillis() {
  using namespace std::chrono;
  return static_cast<uint64_t>(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

std::optional<std::filesystem::path> FindRepoRootFromExe(const char* argv0) {
  namespace fs = std::filesystem;
  if (!argv0 || std::string(argv0).empty()) return std::nullopt;

  std::error_code ec;
  fs::path exe_path = fs::path(argv0);
  if (exe_path.is_relative()) {
    exe_path = fs::absolute(exe_path, ec);
  }
  if (ec) return std::nullopt;

  fs::path dir = exe_path.parent_path();
  for (int i = 0; i < 12; ++i) {
    if (dir.empty()) break;
    if (fs::exists(dir / "CMakeLists.txt", ec) && fs::exists(dir / "models", ec)) {
      return dir;
    }
    auto parent = dir.parent_path();
    if (parent == dir) break;
    dir = parent;
  }
  return std::nullopt;
}

std::string FormatTimestampRfc2822(std::uint64_t timestamp_ms) {
  if (timestamp_ms == 0) return "";
  // Convert milliseconds to seconds for time_t
  std::time_t t = static_cast<std::time_t>(timestamp_ms / 1000);
  std::tm tm_buf;
  std::tm* tm_ptr = localtime_r(&t, &tm_buf);
  if (!tm_ptr) return "";
  char buf[64];
  std::strftime(buf, sizeof(buf), "%a, %d %b %Y %H:%M:%S %z", tm_ptr);
  return std::string(buf);
}

std::string XmlEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&apos;"; break;
      default: out += c; break;
    }
  }
  return out;
}

std::string FormatMemoriesForSystemPrompt(
    const std::vector<cortext::Cortext::Context::Memory>& memories) {
  if (memories.empty()) return {};

  std::ostringstream oss;
  oss << "<memories>\n";
  for (const auto& m : memories) {
    std::string source = "unknown";
    if (m.source_id.find("user") != std::string::npos) {
      source = "user";
    } else if (m.source_id.find("assistant") != std::string::npos) {
      source = "assistant";
    } else if (!m.source_id.empty()) {
      source = m.source_id;
    }

    std::string date = FormatTimestampRfc2822(m.timestamp);
    std::string modality = m.modality.empty() ? "text" : m.modality;

    oss << "<memory source=\"" << XmlEscape(source) << "\"";
    if (!date.empty()) {
      oss << " date=\"" << XmlEscape(date) << "\"";
    }
    oss << " modality=\"" << XmlEscape(modality) << "\">";

    std::string c = ExtractTextFromBlobs(m.content);
    if (c.size() > 800) c = c.substr(0, 800) + "...";
    oss << XmlEscape(c);

    oss << "</memory>\n";
  }
  oss << "</memories>";
  return oss.str();
}

template <typename Json>
Json BuildOpenAIMessages(const std::vector<cortext::Cortext::Context::Memory>& working_memory,
                         const std::string& injected_system,
                         const std::string& latest_user_input) {
  Json messages = Json::array();
  if (!injected_system.empty()) {
    messages.push_back({{"role", "system"}, {"content", injected_system}});
  }

  // Build assistant-only history from working memory ordered by timestamp.
  std::vector<const cortext::Cortext::Context::Memory*> ordered;
  ordered.reserve(working_memory.size());
  for (const auto& m : working_memory) {
    ordered.push_back(&m);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const auto* a, const auto* b) {
              if (a->timestamp == b->timestamp) return a->id < b->id;
              return a->timestamp < b->timestamp;
            });
  for (const auto* m : ordered) {
    if (m->source_id != "chat/assistant") {
      continue;
    }
    std::string text_content = ExtractTextFromBlobs(m->content);
    messages.push_back({{"role", "assistant"}, {"content", text_content}});
  }
  if (!latest_user_input.empty()) {
    messages.push_back({{"role", "user"}, {"content", latest_user_input}});
  }
  return messages;
}

} // namespace

int main(int argc, char** argv) {
  const bool has_models_env = HasEnv("CORTEXT_MODELS_DIR");
  const bool has_db_env = HasEnv("CORTEXT_CHAT_DB");

  const std::string api_key = GetEnv("OPENAI_API_KEY");
  if (api_key.empty()) {
    std::cerr << "OPENAI_API_KEY is required\n";
    return 1;
  }

  const std::string openai_org = GetEnv("OPENAI_ORGANIZATION");
  const std::string openai_base_url = GetEnv("OPENAI_BASE_URL");

  const auto repo_root = FindRepoRootFromExe((argc > 0) ? argv[0] : nullptr);
  const std::string model = GetEnv("OPENAI_MODEL", "gpt-4o-mini");
  std::filesystem::path db_path = GetEnv("CORTEXT_CHAT_DB", "examples/chat/chat_memory.db");
  std::filesystem::path models_dir = GetEnv("CORTEXT_MODELS_DIR", "models/imagebind");

  if (!has_models_env && models_dir.is_relative() && repo_root.has_value()) {
    models_dir = *repo_root / models_dir;
  }
  if (!has_db_env && db_path.is_relative() && repo_root.has_value()) {
    db_path = *repo_root / db_path;
  }

  try {
    if (!db_path.empty()) {
      std::error_code ec;
      auto parent = db_path.parent_path();
      if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
      }
    }
  } catch (...) {
  }

  auto status_state = std::make_shared<chat::StatusBarState>();
  auto retrieval_latency_state = std::make_shared<RetrievalLatencyState>();
  auto last_tokens_used = std::make_shared<std::atomic<std::int64_t>>(0);
  auto last_context = std::make_shared<chat::LastContext>();

  // OpenTelemetry setup
  // OTLP gRPC exporters auto-read from OTEL_EXPORTER_OTLP_* environment variables:
  //   OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_HEADERS, OTEL_EXPORTER_OTLP_COMPRESSION
  {
    namespace trace_sdk = opentelemetry::sdk::trace;
    namespace metrics_sdk = opentelemetry::sdk::metrics;
    namespace resource_sdk = opentelemetry::sdk::resource;
    namespace otlp = opentelemetry::exporter::otlp;
    auto resource = resource_sdk::Resource::Create({{"service.name", "cortext_chat"}});

    auto in_memory_span_exporter = std::unique_ptr<trace_sdk::SpanExporter>(new InMemorySpanExporter(retrieval_latency_state));
    auto in_memory_processor = std::unique_ptr<trace_sdk::SpanProcessor>(new trace_sdk::SimpleSpanProcessor(std::move(in_memory_span_exporter)));
    auto* sdk_tracer_provider = new trace_sdk::TracerProvider(std::move(in_memory_processor), resource);

    // OTLP gRPC trace exporter (reads config from env vars)
    {
      otlp::OtlpGrpcExporterOptions trace_opts;
      auto otlp_exporter = std::unique_ptr<trace_sdk::SpanExporter>(new otlp::OtlpGrpcExporter(trace_opts));
      trace_sdk::BatchSpanProcessorOptions batch_opts;
      batch_opts.max_queue_size = 2048;
      batch_opts.schedule_delay_millis = std::chrono::milliseconds(5000);
      auto otlp_processor = std::unique_ptr<trace_sdk::SpanProcessor>(new trace_sdk::BatchSpanProcessor(std::move(otlp_exporter), batch_opts));
      sdk_tracer_provider->AddProcessor(std::move(otlp_processor));
    }

    auto tracer_provider = opentelemetry::nostd::shared_ptr<opentelemetry::trace::TracerProvider>(sdk_tracer_provider);
    opentelemetry::trace::Provider::SetTracerProvider(tracer_provider);

    auto sdk_meter_provider = metrics_sdk::MeterProviderFactory::Create(std::unique_ptr<metrics_sdk::ViewRegistry>(new metrics_sdk::ViewRegistry()), resource);
    auto* raw_meter_provider = sdk_meter_provider.release();
    auto meter_provider = opentelemetry::nostd::shared_ptr<opentelemetry::metrics::MeterProvider>(raw_meter_provider);

    // In-memory metric exporter for status bar
    {
      auto in_memory_metric_exporter = std::unique_ptr<metrics_sdk::PushMetricExporter>(new InMemoryMetricExporter(status_state));
      metrics_sdk::PeriodicExportingMetricReaderOptions reader_options;
      reader_options.export_interval_millis = std::chrono::milliseconds(1000);
      reader_options.export_timeout_millis = std::chrono::milliseconds(500);
      std::unique_ptr<metrics_sdk::MetricReader> reader(new metrics_sdk::PeriodicExportingMetricReader(std::move(in_memory_metric_exporter), reader_options));
      static_cast<metrics_sdk::MeterProvider*>(raw_meter_provider)->AddMetricReader(std::move(reader));
    }

    // OTLP gRPC metric exporter (reads config from env vars)
    {
      otlp::OtlpGrpcMetricExporterOptions metric_opts;
      auto otlp_metric_exporter = std::unique_ptr<metrics_sdk::PushMetricExporter>(new otlp::OtlpGrpcMetricExporter(metric_opts));
      metrics_sdk::PeriodicExportingMetricReaderOptions reader_options;
      reader_options.export_interval_millis = std::chrono::milliseconds(10000);
      reader_options.export_timeout_millis = std::chrono::milliseconds(5000);
      std::unique_ptr<metrics_sdk::MetricReader> reader(new metrics_sdk::PeriodicExportingMetricReader(std::move(otlp_metric_exporter), reader_options));
      static_cast<metrics_sdk::MeterProvider*>(raw_meter_provider)->AddMetricReader(std::move(reader));
    }

    opentelemetry::metrics::Provider::SetMeterProvider(meter_provider);
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("chat");
    static auto tokens_gauge = meter->CreateInt64ObservableGauge("chat.tokens_used", "Tokens used for last response", "1");
    auto observe_tokens = [](opentelemetry::metrics::ObserverResult result, void *state) noexcept {
      auto *tokens = static_cast<std::atomic<std::int64_t> *>(state);
      const std::int64_t v = tokens->load();
      opentelemetry::nostd::visit([&](auto &r) { if (r) { r->Observe(v); } }, result);
    };
    tokens_gauge->AddCallback(observe_tokens, last_tokens_used.get());

    // Log exporters: stdout (simple pretty format) + OTLP gRPC
    {
      namespace logs_sdk = opentelemetry::sdk::logs;

      // Create simple stdout log exporter first (also writes to logs.txt in examples/chat/)
      std::string log_file_path = "examples/chat/logs.txt";
      if (repo_root.has_value()) {
        log_file_path = (*repo_root / "examples/chat/logs.txt").string();
      }
      auto stdout_exporter = std::unique_ptr<logs_sdk::LogRecordExporter>(new SimpleStdoutLogExporter(log_file_path));
      auto stdout_processor = logs_sdk::SimpleLogRecordProcessorFactory::Create(std::move(stdout_exporter));
      auto logger_provider = logs_sdk::LoggerProviderFactory::Create(std::move(stdout_processor), resource);

      // Add OTLP gRPC log exporter as additional processor
      otlp::OtlpGrpcLogRecordExporterOptions log_opts;
      auto otlp_exporter = otlp::OtlpGrpcLogRecordExporterFactory::Create(log_opts);
      auto otlp_processor = logs_sdk::SimpleLogRecordProcessorFactory::Create(std::move(otlp_exporter));
      static_cast<logs_sdk::LoggerProvider*>(logger_provider.get())->AddProcessor(std::move(otlp_processor));

      opentelemetry::logs::Provider::SetLoggerProvider(
          opentelemetry::nostd::shared_ptr<opentelemetry::logs::LoggerProvider>(logger_provider.release()));
    }
  }
  {
    cortext::telemetry::ScopedSpan span("chat.startup");
    span.SetStatusOk();
    cortext::telemetry::AddCounter("chat.startup_total", 1);
    cortext::telemetry::LogInfo("chat.startup");
  }

  const double focus = GetEnvDouble("CORTEXT_FOCUS", 0.5);
  const double sensitivity = GetEnvDouble("CORTEXT_SENSITIVITY", 0.5);
  const double stability = GetEnvDouble("CORTEXT_STABILITY", 0.5);
  const bool stream_interrupts = GetEnvBool("CORTEXT_CHAT_STREAM_INTERRUPTS", true);

  openai::start(api_key, openai_org);
  if (!openai_base_url.empty()) {
    openai::instance().setBaseUrl(EnsureTrailingSlash(openai_base_url));
  }

  cortext::ImageBindEncoder encoder(models_dir.string());
  try {
    std::vector<float> probe;
    encoder.EncodeText("probe", probe);
    if (probe.size() != 256) {
      throw std::runtime_error("unexpected embedding dim (expected 256)");
    }
  } catch (const std::exception& e) {
    std::cerr << "ImageBind encoder unavailable: " << e.what() << "\n";
    std::cerr << "Hint: build with -DCORTEXT_ENABLE_IMAGEBIND_ORT=ON and ensure models exist under "
              << models_dir.string() << "\n";
    return 1;
  }

  cortext::Cortext::Config cfg;
  cfg.focus = focus;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
  std::unique_ptr<cortext::Cortext> cortext_ctx;
  try {
    cortext_ctx = cortext::Cortext::Create(cfg, db_path.string(), models_dir.string());
  } catch (const std::exception& e) {
    std::cerr << "Failed to create cortext instance: " << e.what() << "\n";
    std::cerr << "Hint: set CORTEXT_CHAT_DB to an absolute writable path, e.g.\n";
    std::cerr << "  export CORTEXT_CHAT_DB=\"$HOME/.cortext/chat_memory.db\"\n";
    return 1;
  }
  if (!cortext_ctx) {
    std::cerr << "Failed to create cortext instance (null)\n";
    return 1;
  }

  // UI shared state
  std::mutex mu;
  static std::mutex db_write_mu;
  std::vector<cortext::Cortext::Context::Memory> working_memory;
  std::vector<cortext::Cortext::Context::Memory> last_memories;
  std::deque<chat::MemoryEvent> memory_events;
  bool last_should_interrupt = false;
  std::optional<std::string> last_error;
  bool generating = false;
  std::string input;
  static constexpr size_t kMaxMemoryEvents = 50;
  std::deque<chat::ChatMessage> chat_history;

  StreamingState streaming_state;
  std::string partial_response;
  int generation_restarts = 0;
  IdleTracker idle_tracker;

  // Track streaming thread to ensure proper cleanup before exit
  std::thread streaming_thread;
  std::atomic<bool> streaming_thread_active{false};

  chat::StreamingChatClient streaming_client(api_key, openai_base_url.empty()
      ? "https://api.openai.com/v1/"
      : openai_base_url);

  const bool otlp_grpc_enabled = HasEnv("OTEL_EXPORTER_OTLP_ENDPOINT") || HasEnv("OTEL_EXPORTER_OTLP_HEADERS");

  // Create ImGui application
  chat::ImGuiAppConfig app_config;
  app_config.title = "Cortext Chat";
  app_config.width = 1280;
  app_config.height = 800;

  chat::ImGuiApp app(app_config);

  // Create chat window with references to shared state
  chat::ChatWindow::State window_state;
  window_state.mu = &mu;
  window_state.working_memory = &working_memory;
  window_state.chat_history = &chat_history;
  window_state.memory_events = &memory_events;
  window_state.context = last_context;
  window_state.status = status_state;
  window_state.input = &input;
  window_state.generating = &generating;
  window_state.partial_response = &partial_response;
  window_state.generation_restarts = &generation_restarts;
  window_state.last_should_interrupt = &last_should_interrupt;
  window_state.last_error = &last_error;
  window_state.otlp_enabled = otlp_grpc_enabled;

  chat::ChatWindow window(window_state);

  // Background refresh thread for idle consolidation
  std::atomic<bool> running{true};
  std::thread refresh_thread([&] {
    while (running) {
      std::this_thread::sleep_for(std::chrono::seconds(2));

      bool is_generating = false;
      {
        std::lock_guard<std::mutex> lock(mu);
        is_generating = generating;
      }

      // Consolidation disabled for debugging
      // if (!is_generating && idle_tracker.ShouldConsolidate(stability)) {
      //   try {
      //     std::lock_guard<std::mutex> lock(db_write_mu);
      //     cortext_ctx->Consolidate();
      //     idle_tracker.MarkConsolidated();
      //   } catch (const std::exception& ex) {
      //     std::lock_guard<std::mutex> lock(mu);
      //     last_error = std::string("idle consolidation: ") + ex.what();
      //   }
      // }
    }
  });

  std::cout << "Cortext Chat started. Window opened." << std::endl;

  // Main render loop
  app.Run([&] {
    window.Render();

    // Check for pending message from input
    if (window.HasPendingMessage()) {
      std::string text = window.TakePendingMessage();

      {
        std::lock_guard<std::mutex> lock(mu);
        if (!generating && !text.empty()) {
          generating = true;
          last_error.reset();
          partial_response.clear();
          generation_restarts = 0;
          streaming_state.Reset();
          chat_history.push_back({"user", text});
        } else {
          text.clear();
        }
      }

      if (!text.empty()) {
        idle_tracker.RecordActivity();

        cortext::telemetry::LogInfo("chat.user_message", {
          cortext::telemetry::Attribute::Int64("text_length", static_cast<int64_t>(text.size()))
        });

        // Wait for any previous streaming thread to complete before starting new one
        if (streaming_thread.joinable()) {
          streaming_thread.join();
        }

        // Launch background job for streaming
        streaming_thread_active.store(true);
        streaming_thread = std::thread([&, text, model_copy = model] {
          // Phase 1: Process user input through Cortext
          cortext::telemetry::LogDebug("chat.phase1.start", {
            cortext::telemetry::Attribute::String("source", "chat/user"),
            cortext::telemetry::Attribute::Int64("text_length", static_cast<int64_t>(text.size()))
          });

          cortext::Cortext::Context retrieved;
          try {
            std::lock_guard<std::mutex> lock(db_write_mu);
            retrieved = cortext_ctx->ProcessText(text, "chat/user");
          } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lock(mu);
            last_error = std::string("cortext user: ") + ex.what();
            cortext::telemetry::LogError("chat.phase1.error", {
              cortext::telemetry::Attribute::String("error", ex.what())
            });
          }

          {
            std::lock_guard<std::mutex> lock(mu);
            working_memory = retrieved.working_memory;
            // Only inject memories if the interrupt gate approved them
            if (retrieved.should_interrupt) {
              streaming_state.current_memories = retrieved.retrieved_memory;
              for (const auto& mem : retrieved.retrieved_memory) {
                memory_events.push_back(CreateMemoryEvent(chat::MemoryEventType::RETRIEVED, mem));
                if (memory_events.size() > kMaxMemoryEvents) {
                  memory_events.pop_front();
                }
              }
            } else {
              streaming_state.current_memories.clear();
            }
            if (retrieved.output.stored_embedding_id.has_value()) {
              memory_events.push_back(CreateStoredEvent(
                  *retrieved.output.stored_embedding_id, text, "chat/user", NowUnixMillis(), retrieved.output));
              if (memory_events.size() > kMaxMemoryEvents) {
                memory_events.pop_front();
              }
            }
          }

          cortext::telemetry::LogDebug("chat.phase1.complete", {
            cortext::telemetry::Attribute::Bool("should_interrupt", retrieved.should_interrupt),
            cortext::telemetry::Attribute::Int64("memories_retrieved", static_cast<int64_t>(retrieved.retrieved_memory.size())),
            cortext::telemetry::Attribute::Bool("stored", retrieved.output.stored_embedding_id.has_value()),
            cortext::telemetry::Attribute::Double("composite_score", retrieved.output.composite_score.value_or(0.0)),
            cortext::telemetry::Attribute::Double("threshold", retrieved.output.threshold.value_or(0.0)),
            cortext::telemetry::Attribute::Bool("write_decision", retrieved.output.decision.value_or(false))
          });

          // Phase 2: Streaming generation with interrupt checking
          cortext::telemetry::LogDebug("chat.phase2.start", {
            cortext::telemetry::Attribute::Int64("memories_injected", static_cast<int64_t>(streaming_state.current_memories.size())),
            cortext::telemetry::Attribute::String("model", model_copy)
          });

          std::string assistant_reply;
          int local_restarts = 0;

          while (local_restarts < StreamingState::kMaxRestarts) {
            std::string injected_system;
            {
              std::lock_guard<std::mutex> lock(mu);
              injected_system = FormatMemoriesForSystemPrompt(streaming_state.current_memories);
            }

            using Json = openai::Json;
            Json messages = BuildOpenAIMessages<Json>(retrieved.working_memory, injected_system, text);

            {
              std::lock_guard<std::mutex> lock(last_context->mu);
              last_context->system_prompt = injected_system;
              last_context->has_data = true;
            }

            {
              std::lock_guard<std::mutex> lock(mu);
              streaming_state.accumulated_tokens.clear();
              streaming_state.cancel_requested.store(false);
              partial_response.clear();
            }

            chat::StreamingRequest request;
            request.model = model_copy;
            request.messages = messages;
            request.temperature = 0.7;
            request.cancel_flag = &streaming_state.cancel_requested;

            bool interrupted = false;

            cortext::telemetry::LogDebug("chat.phase2.stream_start", {
              cortext::telemetry::Attribute::Int64("restart", static_cast<int64_t>(local_restarts)),
              cortext::telemetry::Attribute::Int64("working_memory_size", static_cast<int64_t>(retrieved.working_memory.size()))
            });

            auto result = streaming_client.Stream(request, [&](const std::string& token, bool done) {
              if (done || token.empty()) return;

              std::string accumulated;
              {
                std::lock_guard<std::mutex> lock(mu);
                streaming_state.accumulated_tokens += token;
                accumulated = streaming_state.accumulated_tokens;
                partial_response = accumulated;
              }

              cortext::telemetry::LogDebug("chat.phase2.token", {
                cortext::telemetry::Attribute::Int64("accumulated_length", static_cast<int64_t>(accumulated.size())),
                cortext::telemetry::Attribute::String("accumulated", accumulated.size() <= 100 ? accumulated : accumulated.substr(0, 100) + "...")
              });

              if (!stream_interrupts) {
                return;
              }

              try {
                cortext::Cortext::Context ctx;
                {
                  std::lock_guard<std::mutex> lock(db_write_mu);
                  ctx = cortext_ctx->ProcessText(accumulated, "chat/assistant");
                }

                const double boundary_thresh = cortext::core::BoundaryThreshold(focus, sensitivity);
                const bool boundary_score_pass
                    = ctx.boundary_score.has_value()
                      && *ctx.boundary_score >= boundary_thresh;

                if (ctx.at_boundary && boundary_score_pass && ctx.should_interrupt
                    && local_restarts < StreamingState::kMaxRestarts) {
                  cortext::telemetry::LogInfo("chat.phase2.interrupt", {
                    cortext::telemetry::Attribute::Int64("new_memories", static_cast<int64_t>(ctx.retrieved_memory.size())),
                    cortext::telemetry::Attribute::Int64("restart", static_cast<int64_t>(local_restarts)),
                    cortext::telemetry::Attribute::Int64("tokens_so_far", static_cast<int64_t>(accumulated.size()))
                  });
                  {
                    std::lock_guard<std::mutex> lock(mu);
                    std::unordered_set<long long> existing_ids;
                    for (const auto& m : streaming_state.current_memories) {
                      existing_ids.insert(m.id);
                    }
                    for (const auto& m : ctx.retrieved_memory) {
                      if (existing_ids.find(m.id) == existing_ids.end()) {
                        streaming_state.current_memories.push_back(m);
                        memory_events.push_back(CreateMemoryEvent(chat::MemoryEventType::RETRIEVED, m));
                        if (memory_events.size() > kMaxMemoryEvents) {
                          memory_events.pop_front();
                        }
                      }
                    }
                    last_should_interrupt = true;
                  }
                  streaming_state.cancel_requested.store(true);
                  interrupted = true;
                }
              } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lock(mu);
                last_error = std::string("cortext streaming: ") + ex.what();
              }
            });

            if (result.was_cancelled && interrupted && local_restarts < StreamingState::kMaxRestarts) {
              local_restarts++;
              {
                std::lock_guard<std::mutex> lock(mu);
                generation_restarts = local_restarts;
              }
              continue;
            }

            if (result.error.has_value()) {
              cortext::telemetry::LogError("chat.phase2.stream_error", {
                cortext::telemetry::Attribute::String("error", *result.error)
              });
              std::lock_guard<std::mutex> lock(mu);
              last_error = "streaming: " + *result.error;
              assistant_reply = "(Streaming error: " + *result.error + ")";
            } else {
              cortext::telemetry::LogDebug("chat.phase2.stream_complete", {
                cortext::telemetry::Attribute::Int64("reply_length", static_cast<int64_t>(result.full_content.size())),
                cortext::telemetry::Attribute::Int64("total_restarts", static_cast<int64_t>(local_restarts)),
                cortext::telemetry::Attribute::Bool("was_cancelled", result.was_cancelled)
              });
              assistant_reply = result.full_content;
            }
            break;
          }

          // Phase 3: Process final assistant reply through Cortext
          cortext::telemetry::LogDebug("chat.phase3.start", {
            cortext::telemetry::Attribute::Int64("reply_length", static_cast<int64_t>(assistant_reply.size())),
            cortext::telemetry::Attribute::String("source", "chat/assistant"),
            cortext::telemetry::Attribute::String("reply_content", assistant_reply.size() <= 200 ? assistant_reply : assistant_reply.substr(0, 200) + "...")
          });

          try {
            cortext::Cortext::Context asst_ctx;
            {
              std::lock_guard<std::mutex> lock(db_write_mu);
              asst_ctx = cortext_ctx->ProcessText(assistant_reply, "chat/assistant");
            }

            cortext::telemetry::LogDebug("chat.phase3.complete", {
              cortext::telemetry::Attribute::Bool("stored", asst_ctx.output.stored_embedding_id.has_value()),
              cortext::telemetry::Attribute::Double("composite_score", asst_ctx.output.composite_score.value_or(0.0)),
              cortext::telemetry::Attribute::Double("threshold", asst_ctx.output.threshold.value_or(0.0)),
              cortext::telemetry::Attribute::Bool("write_decision", asst_ctx.output.decision.value_or(false))
            });

            {
              std::lock_guard<std::mutex> lock(mu);
              working_memory = asst_ctx.working_memory;
              if (asst_ctx.output.stored_embedding_id.has_value()) {
                cortext::telemetry::LogInfo("chat.assistant_stored", {
                  cortext::telemetry::Attribute::Int64("embedding_id", *asst_ctx.output.stored_embedding_id),
                  cortext::telemetry::Attribute::Int64("content_length", static_cast<int64_t>(assistant_reply.size()))
                });
                memory_events.push_back(CreateStoredEvent(
                    *asst_ctx.output.stored_embedding_id, assistant_reply, "chat/assistant", NowUnixMillis(), asst_ctx.output));
                while (memory_events.size() > kMaxMemoryEvents) {
                  memory_events.pop_front();
                }
              }
            }
          } catch (const std::exception& ex) {
            cortext::telemetry::LogError("chat.phase3.error", {
              cortext::telemetry::Attribute::String("error", ex.what())
            });
            std::lock_guard<std::mutex> lock(mu);
            last_error = std::string("cortext assistant: ") + ex.what();
          }

          idle_tracker.RecordActivity();

          {
            std::lock_guard<std::mutex> lock(mu);
            last_memories = streaming_state.current_memories;
            if (!assistant_reply.empty()) {
              cortext::telemetry::LogInfo("chat.assistant_reply_added", {
                cortext::telemetry::Attribute::Int64("reply_length", static_cast<int64_t>(assistant_reply.size())),
                cortext::telemetry::Attribute::Int64("working_memory_size", static_cast<int64_t>(working_memory.size())),
                cortext::telemetry::Attribute::Int64("memories_used", static_cast<int64_t>(last_memories.size()))
              });
            }
            if (!assistant_reply.empty()) {
              chat_history.push_back({"assistant", assistant_reply});
            }
            generating = false;
            partial_response.clear();
          }
          cortext::telemetry::LogDebug("chat.generation_complete", {
            cortext::telemetry::Attribute::Int64("total_restarts", static_cast<int64_t>(local_restarts)),
            cortext::telemetry::Attribute::Bool("has_reply", !assistant_reply.empty())
          });
          streaming_thread_active.store(false);
        });
      }
    }
  });

  running = false;

  // Cancel any ongoing streaming if active
  if (streaming_thread_active.load()) {
    streaming_state.cancel_requested.store(true);
  }

  // Wait for threads to complete before cleanup
  if (streaming_thread.joinable()) {
    streaming_thread.join();
  }
  if (refresh_thread.joinable()) {
    refresh_thread.join();
  }

  try {
    cortext_ctx->Flush();
  } catch (...) {
  }

  std::cout << "Cortext Chat exited." << std::endl;
  return 0;
}
