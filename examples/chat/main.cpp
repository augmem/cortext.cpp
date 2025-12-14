#include "context_tab.hpp"

#include <cortext/cortext.hpp>
#include <cortext/encoder/imagebind.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/telemetry/telemetry.hpp>

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <openai/openai.hpp>

#include <opentelemetry/exporters/otlp/otlp_file_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_file_metric_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_metric_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_file_log_record_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_log_record_exporter_options.h>

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
#include <opentelemetry/sdk/logs/logger_provider.h>
#include <opentelemetry/sdk/logs/simple_log_record_processor.h>
#include <opentelemetry/logs/provider.h>
#include <opentelemetry/sdk/trace/recordable.h>
#include <opentelemetry/sdk/trace/simple_processor.h>
#include <opentelemetry/sdk/trace/span_data.h>
#include <opentelemetry/sdk/trace/exporter.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>

#include <any>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <fstream>

// Debug log removed
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <map>

namespace {

struct StatusBarState {
  mutable std::mutex mu;
  std::optional<double> processing_latency_ms;
  std::optional<std::int64_t> tokens_used;
};

struct RetrievalLatencyState {
  std::optional<double> encoder_latency_ms;
  std::optional<double> retrieval_latency_ms;
};

struct PersistLatencyState {
  std::optional<double> encoder_latency_ms;
  std::optional<double> storage_latency_ms;
};

class InMemorySpanExporter final : public opentelemetry::sdk::trace::SpanExporter {
public:
  explicit InMemorySpanExporter(std::shared_ptr<RetrievalLatencyState> retrieval_state, std::shared_ptr<PersistLatencyState> persist_state)
      : retrieval_state_(std::move(retrieval_state)), persist_state_(std::move(persist_state)) {}
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
          if (persist_active_ && (op == "INSERT" || op == "UPDATE" || op == "REPLACE")) { persist_db_write_ms_ += duration_ms; }
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
      if (name == "chat.encode_text") { persist_active_ = true; persist_db_write_ms_ = 0.0; persist_last_encode_ms_ = duration_ms; continue; }
      if (name == "chat.persist_text_memory") {
        if (persist_state_) {
          persist_state_->encoder_latency_ms = persist_last_encode_ms_;
          persist_state_->storage_latency_ms = persist_db_write_ms_;
        }
        persist_active_ = false;
        persist_last_encode_ms_.reset();
        continue;
      }
    }
    return opentelemetry::sdk::common::ExportResult::kSuccess;
  }
  bool ForceFlush(std::chrono::microseconds) noexcept override { return true; }
  bool Shutdown(std::chrono::microseconds) noexcept override { return true; }
private:
  std::shared_ptr<RetrievalLatencyState> retrieval_state_;
  std::shared_ptr<PersistLatencyState> persist_state_;
  thread_local static bool retrieval_active_;
  thread_local static bool persist_active_;
  thread_local static double retrieval_db_select_ms_;
  thread_local static double persist_db_write_ms_;
  thread_local static std::optional<double> retrieval_last_encode_ms_;
  thread_local static std::optional<double> retrieval_last_hydrate_ms_;
  thread_local static std::optional<double> persist_last_encode_ms_;
};

thread_local bool InMemorySpanExporter::retrieval_active_ = false;
thread_local bool InMemorySpanExporter::persist_active_ = false;
thread_local double InMemorySpanExporter::retrieval_db_select_ms_ = 0.0;
thread_local double InMemorySpanExporter::persist_db_write_ms_ = 0.0;
thread_local std::optional<double> InMemorySpanExporter::retrieval_last_encode_ms_ = std::nullopt;
thread_local std::optional<double> InMemorySpanExporter::retrieval_last_hydrate_ms_ = std::nullopt;
thread_local std::optional<double> InMemorySpanExporter::persist_last_encode_ms_ = std::nullopt;

class InMemoryMetricExporter final : public opentelemetry::sdk::metrics::PushMetricExporter {
public:
  explicit InMemoryMetricExporter(std::shared_ptr<StatusBarState> status_state) : status_state_(std::move(status_state)) {}
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
  std::shared_ptr<StatusBarState> status_state_;
};

struct ChatMessage {
  std::string role;    // "user" | "assistant" | "system"
  std::string content; // utf-8
};

enum class MemoryEventType {
  STORED,
  RETRIEVED,
};

struct MemoryEvent {
  MemoryEventType type;
  int memory_id;
  std::string content;
  std::string source_id;
  uint64_t timestamp;
  
  // Metrics (from ProcessorOutput)
  std::optional<double> composite_score;
  std::optional<double> threshold;
  std::optional<bool> decision;
  std::optional<bool> should_interrupt;
  std::optional<double> encoder_latency_ms;
  std::optional<double> retrieval_latency_ms;
  std::optional<double> storage_latency_ms;
  
  // Individual metrics (Algorithm 7 inputs)
  std::optional<double> relevance;
  std::optional<double> surprise;
  std::optional<double> mismatch;
  std::optional<double> rarity;
  std::optional<double> drift;
  std::optional<double> contradiction;
  std::optional<double> utility;
  std::optional<double> periphery;
  std::optional<double> salience;
  std::optional<double> valence;
  std::optional<double> arousal;
  
  // System-level
  std::optional<double> coherence;
  std::optional<double> focus_spread;
  std::optional<double> F_eff;
};

// Simple stderr/stdout capture buffer
struct LogEntry {
  uint64_t id;
  std::string raw;
  mutable bool expanded = false;
};

class LogBuffer {
public:
  void Append(const std::string& line) {
    std::lock_guard<std::mutex> lock(mu_);
    lines_.push_back({next_id_++, line, false});
    if (lines_.size() > kMaxLines) {
      lines_.pop_front();
    }
  }

  std::vector<LogEntry> GetEntriesSince(uint64_t last_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<LogEntry> result;
    if (lines_.empty()) return result;
    
    // valid range check
    if (last_id >= lines_.back().id) return result;
    
    for (const auto& entry : lines_) {
      if (entry.id > last_id) {
        result.push_back(entry);
      }
    }
    return result;
  }

private:
  static constexpr size_t kMaxLines = 1000;
  mutable std::mutex mu_;
  std::deque<LogEntry> lines_;
  uint64_t next_id_ = 1;
};

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

uint64_t NowUnixSeconds() {
  using namespace std::chrono;
  return static_cast<uint64_t>(duration_cast<seconds>(system_clock::now().time_since_epoch()).count());
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

std::vector<unsigned char> BlobFromAny(const std::any& value) {
  if (value.type() == typeid(std::vector<unsigned char>)) {
    return std::any_cast<const std::vector<unsigned char>&>(value);
  }
  if (value.type() == typeid(std::vector<char>)) {
    const auto& blob = std::any_cast<const std::vector<char>&>(value);
    return std::vector<unsigned char>(blob.begin(), blob.end());
  }
  return {};
}

std::optional<long long> AnyToLongLong(const std::any& v) {
  if (v.type() == typeid(long long)) return std::any_cast<long long>(v);
  if (v.type() == typeid(long)) return static_cast<long long>(std::any_cast<long>(v));
  if (v.type() == typeid(int)) return static_cast<long long>(std::any_cast<int>(v));
  return std::nullopt;
}

// EnsureChatSchema removed (handled by core migrations).

struct PersistedRow {
  long long embedding_id = 0;
  std::vector<unsigned char> blob_id;
};

PersistedRow PersistTextMemory(cortext::Store& store,
                              cortext::ImageBindEncoder& encoder,
                              const std::string& text,
                              const std::string& source_id,
                              uint64_t ts) {
  // Schema is already ensured by processor startup.
  cortext::telemetry::ScopedSpan persist_span("chat.persist_text_memory");
  persist_span.SetAttribute("chat.source_id", source_id);
  // 1) Put payload.
  const std::vector<unsigned char> payload(text.begin(), text.end());
  auto blob_rows = store.Execute("SELECT objstore_put(?1) AS id", {payload});
  if (blob_rows.empty() || blob_rows[0].count("id") == 0) {
    throw std::runtime_error("objstore_put returned no id");
  }
  const auto blob_id = BlobFromAny(blob_rows[0].at("id"));
  if (blob_id.empty()) {
    throw std::runtime_error("objstore_put returned empty blob id");
  }

  // 2) Compute embedding.
  std::vector<float> emb;
  cortext::telemetry::ScopedSpan encode_span("chat.encode_text");
  encoder.EncodeText(text, emb);
  encode_span.SetStatusOk();
  if (emb.empty()) {
    throw std::runtime_error("encoder produced empty embedding");
  }

  // 3) Insert embedding row (auto id).
  store.Execute("INSERT INTO embeddings (embedding) VALUES (?1)", {emb});
  auto id_rows = store.Execute("SELECT last_insert_rowid() AS id", {});
  if (id_rows.empty() || id_rows[0].count("id") == 0) {
    throw std::runtime_error("last_insert_rowid missing");
  }
  const auto id_opt = AnyToLongLong(id_rows[0].at("id"));
  if (!id_opt.has_value()) {
    throw std::runtime_error("last_insert_rowid had unexpected type");
  }
  const long long embedding_id = *id_opt;

  // 4) Insert metadata row.
  store.Execute(
      "INSERT OR REPLACE INTO memory_index(embedding_id, modality, mime, content_key, source_id, timestamp, blob_id) "
      "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
      {
          embedding_id,
          std::string("text"),
          std::string("text/plain"),
          cortext::Cortext::MakeContentKey(embedding_id),
          source_id,
          static_cast<long long>(ts),
          blob_id,
      });

  PersistedRow row;
  row.embedding_id = embedding_id;
  row.blob_id = blob_id;
  persist_span.SetStatusOk();
  return row;
}

std::string FormatTimestampRfc2822(std::uint64_t timestamp) {
  if (timestamp == 0) return "";
  std::time_t t = static_cast<std::time_t>(timestamp);
  std::tm tm_buf;
  std::tm* tm_ptr = localtime_r(&t, &tm_buf);
  if (!tm_ptr) return "";
  char buf[64];
  // RFC 2822 format: "Sat, 13 Dec 2025 11:45:00 -0600"
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
    // Determine source: use source_id if it contains "user" or "assistant", otherwise default
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

    // Content - truncate if too long and escape XML
    std::string c = m.content;
    if (c.size() > 800) c = c.substr(0, 800) + "...";
    oss << XmlEscape(c);

    oss << "</memory>\n";
  }
  oss << "</memories>";
  return oss.str();
}

// Build OpenAI messages from conversation history (sliding window) and an
// optional system prompt.
template <typename Json>
Json BuildOpenAIMessages(const std::vector<ChatMessage>& history,
                         int window_size,
                         const std::string& injected_system) {
  Json messages = Json::array();
  if (!injected_system.empty()) {
    messages.push_back({{"role", "system"}, {"content", injected_system}});
  }

  const int n = static_cast<int>(history.size());
  const int start = std::max(0, n - window_size);
  for (int i = start; i < n; ++i) {
    const auto& m = history[static_cast<size_t>(i)];
    messages.push_back({{"role", m.role}, {"content", m.content}});
  }
  return messages;
}

class LogListComponent : public ftxui::ComponentBase {
public:
  explicit LogListComponent(std::shared_ptr<LogBuffer> buffer) : buffer_(buffer) {
  }

  using Json = openai::Json;

  struct ParsedLog {
    std::string timestamp;
    std::string severity;
    std::string body;
    std::map<std::string, std::string> attributes;
  };


  std::optional<std::string> StringifyOtlpAnyValue(const Json& value) {
    if (value.contains("stringValue") && value["stringValue"].is_string()) {
      return value["stringValue"].get<std::string>();
    }
    if (value.contains("intValue")) {
      const auto& v = value["intValue"];
      if (v.is_string()) { return v.get<std::string>(); }
      if (v.is_number_integer()) { return std::to_string(v.get<long long>()); }
      if (v.is_number_unsigned()) { return std::to_string(v.get<unsigned long long>()); }
    }
    if (value.contains("doubleValue")) {
      const auto& v = value["doubleValue"];
      if (v.is_number_float() || v.is_number_integer() || v.is_number_unsigned()) {
        std::ostringstream oss;
        oss << std::setprecision(17) << v.get<double>();
        return oss.str();
      }
    }
    if (value.contains("boolValue") && value["boolValue"].is_boolean()) {
      return value["boolValue"].get<bool>() ? "true" : "false";
    }
    return std::nullopt;
  }

  bool IsLikelyOtlpLogLine(const std::string& line) {
    if (line.find('{') == std::string::npos) { return false; }
    if (line.find("\"resourceLogs\"") == std::string::npos) { return false; }
    if (line.find("\"logRecords\"") == std::string::npos) { return false; }
    return true;
  }

  ParsedLog ParseLogLine(const std::string& line) {
    ParsedLog result;
    size_t json_start = line.find('{');
    if (json_start == std::string::npos) {
      result.body = line;
      return result;
    }

    std::string json_str = line.substr(json_start);
    try {
      auto j = Json::parse(json_str);
      if (j.contains("resourceLogs") && !j["resourceLogs"].empty()) {
        auto& rl = j["resourceLogs"][0];
        if (rl.contains("scopeLogs") && !rl["scopeLogs"].empty()) {
          auto& sl = rl["scopeLogs"][0];
          if (sl.contains("logRecords") && !sl["logRecords"].empty()) {
            auto& lr = sl["logRecords"][0];
            
            if (lr.contains("body") && lr["body"].contains("stringValue")) {
              result.body = lr["body"]["stringValue"].get<std::string>();
            }
            if (lr.contains("severityText")) {
              result.severity = lr["severityText"].get<std::string>();
            }
            // Additional attribute parsing if needed...
            if (lr.contains("attributes") && lr["attributes"].is_array()) {
              for (const auto& attr : lr["attributes"]) {
                if (attr.contains("key") && attr.contains("value")) {
                   std::string key = attr["key"].get<std::string>();
                   const std::optional<std::string> str = StringifyOtlpAnyValue(attr["value"]);
                   if (str.has_value()) { result.attributes[key] = *str; }
                }
              }
            }
          }
        }
      }
    } catch (...) {
      result.body = line;
    }
    if (result.body.empty()) result.body = line;
    return result;
  }

  ftxui::Element Render() {
    ftxui::Elements lines;
    lines.push_back(ftxui::text("System Logs (OpenTelemetry)") | ftxui::bold);
    lines.push_back(ftxui::separator());

    if (parsed_logs_.empty()) {
      lines.push_back(ftxui::text("(no logs yet)") | ftxui::color(ftxui::Color::GrayDark));
    } else {
      // Show most recent logs first (reversed)
      for (auto it = parsed_logs_.rbegin(); it != parsed_logs_.rend(); ++it) {
        const auto& log = *it;

        // Severity + body header
        auto severity_color = ftxui::Color::White;
        if (log.severity == "ERROR") severity_color = ftxui::Color::Red;
        else if (log.severity == "WARN") severity_color = ftxui::Color::Yellow;
        else if (log.severity == "INFO") severity_color = ftxui::Color::Green;
        else if (log.severity == "DEBUG") severity_color = ftxui::Color::Cyan;

        std::string header = log.body;
        if (!log.severity.empty()) {
          header = "[" + log.severity + "] " + header;
        }
        // Truncate long headers
        if (header.size() > 100) header = header.substr(0, 100) + "...";
        lines.push_back(ftxui::text(header) | ftxui::color(severity_color));

        // Show attributes
        for (const auto& [k, v] : log.attributes) {
          std::string attr_line = "  " + k + ": " + v;
          if (attr_line.size() > 120) attr_line = attr_line.substr(0, 120) + "...";
          lines.push_back(ftxui::hbox({
              ftxui::text("  "),
              ftxui::text(k) | ftxui::color(ftxui::Color::Cyan),
              ftxui::text(": "),
              ftxui::text(v.size() > 80 ? v.substr(0, 80) + "..." : v) | ftxui::color(ftxui::Color::Yellow)
          }));
        }

        lines.push_back(ftxui::separator());
      }
    }

    return ftxui::vbox(std::move(lines))
           | ftxui::focusPositionRelative(0, scroll_position_)
           | ftxui::vscroll_indicator
           | ftxui::yframe;
  }

  void SetScrollPosition(float pos) { scroll_position_ = pos; }
  float GetScrollPosition() const { return scroll_position_; }

  void Sync() {
    auto new_entries = buffer_->GetEntriesSince(last_seen_id_);
    for (const auto& entry : new_entries) {
      last_seen_id_ = entry.id;
      if (!IsLikelyOtlpLogLine(entry.raw)) {
        continue;
      }
      ParsedLog parsed = ParseLogLine(entry.raw);
      parsed_logs_.push_back(std::move(parsed));
    }
    // Keep max 200 entries
    while (parsed_logs_.size() > 200) {
      parsed_logs_.pop_front();
    }
  }

private:
  std::shared_ptr<LogBuffer> buffer_;
  uint64_t last_seen_id_ = 0;
  float scroll_position_ = 0.0f;  // Start at top (newest first)
  std::deque<ParsedLog> parsed_logs_;
};

} // namespace

int main(int argc, char** argv) {
  // Shared log buffer for logs view.
  auto log_buffer = std::make_shared<LogBuffer>();
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

  // If defaults are used (env var not set), interpret relative paths as relative
  // to the repo root (when available). This makes VSCode/CMake launches robust
  // even when cwd is the build folder.
  if (!has_models_env && models_dir.is_relative() && repo_root.has_value()) {
    models_dir = *repo_root / models_dir;
  }
  if (!has_db_env && db_path.is_relative() && repo_root.has_value()) {
    db_path = *repo_root / db_path;
  }

  // Ensure DB parent directory exists.
  try {
    if (!db_path.empty()) {
      std::error_code ec;
      auto parent = db_path.parent_path();
      if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
      }
    }
  } catch (...) {
    // Best effort; SQLite open will fail with a useful error.
  }

  auto status_state = std::make_shared<StatusBarState>();
  auto retrieval_latency_state = std::make_shared<RetrievalLatencyState>();
  auto persist_latency_state = std::make_shared<PersistLatencyState>();
  auto last_tokens_used = std::make_shared<std::atomic<std::int64_t>>(0);
  auto last_context = std::make_shared<chat::LastContext>();
  {
    namespace trace_sdk = opentelemetry::sdk::trace;
    namespace metrics_sdk = opentelemetry::sdk::metrics;
    namespace logs_sdk = opentelemetry::sdk::logs;
    namespace resource_sdk = opentelemetry::sdk::resource;
    namespace otlp_exporter = opentelemetry::exporter::otlp;
    const std::filesystem::path jsonl_path_default = db_path.parent_path() / "otel.jsonl";
    const std::string jsonl_path = GetEnv("CORTEXT_CHAT_OTEL_JSONL", jsonl_path_default.string());
    auto MakeSiblingPath = [](const std::string &path, const std::string &suffix) -> std::string {
      const std::string ext = ".jsonl";
      if (path.size() >= ext.size() && path.compare(path.size() - ext.size(), ext.size(), ext) == 0) {
        return path.substr(0, path.size() - ext.size()) + suffix + ext;
      }
      return path + suffix;
    };
    const std::string trace_path = MakeSiblingPath(jsonl_path, ".traces");
    const std::string metric_path = MakeSiblingPath(jsonl_path, ".metrics");
    const std::string log_path = MakeSiblingPath(jsonl_path, ".logs");
    auto resource = resource_sdk::Resource::Create({{"service.name", "cortext_chat"}});
    auto in_memory_span_exporter = std::unique_ptr<trace_sdk::SpanExporter>(new InMemorySpanExporter(retrieval_latency_state, persist_latency_state));
    otlp_exporter::OtlpFileClientFileSystemOptions otlp_trace_fs;
    otlp_trace_fs.file_pattern = trace_path;
    otlp_trace_fs.alias_pattern = trace_path + ".latest";
    otlp_trace_fs.flush_interval = std::chrono::milliseconds(200);
    otlp_trace_fs.flush_count = 1;
    otlp_exporter::OtlpFileExporterOptions trace_options;
    trace_options.backend_options = otlp_trace_fs;
    auto otlp_trace_exporter = otlp_exporter::OtlpFileExporterFactory::Create(trace_options);
    std::vector<std::unique_ptr<trace_sdk::SpanProcessor>> span_processors;
    span_processors.push_back(std::unique_ptr<trace_sdk::SpanProcessor>(new trace_sdk::SimpleSpanProcessor(std::move(in_memory_span_exporter))));
    span_processors.push_back(std::unique_ptr<trace_sdk::SpanProcessor>(new trace_sdk::SimpleSpanProcessor(std::move(otlp_trace_exporter))));
    auto tracer_provider = opentelemetry::nostd::shared_ptr<opentelemetry::trace::TracerProvider>(new trace_sdk::TracerProvider(std::move(span_processors), resource));
    opentelemetry::trace::Provider::SetTracerProvider(tracer_provider);
    otlp_exporter::OtlpFileLogRecordExporterOptions log_options;
    otlp_exporter::OtlpFileClientFileSystemOptions otlp_log_fs;
    otlp_log_fs.file_pattern = log_path;
    otlp_log_fs.alias_pattern = log_path + ".latest";
    otlp_log_fs.flush_interval = std::chrono::milliseconds(200);
    otlp_log_fs.flush_count = 1;
    log_options.backend_options = otlp_log_fs;
    auto otlp_log_exporter = otlp_exporter::OtlpFileLogRecordExporterFactory::Create(log_options);
    auto log_processor = std::unique_ptr<logs_sdk::LogRecordProcessor>(new logs_sdk::SimpleLogRecordProcessor(std::move(otlp_log_exporter)));
    auto logger_provider = opentelemetry::nostd::shared_ptr<opentelemetry::logs::LoggerProvider>(new logs_sdk::LoggerProvider(std::move(log_processor), resource));
    opentelemetry::logs::Provider::SetLoggerProvider(logger_provider);
    auto sdk_meter_provider = metrics_sdk::MeterProviderFactory::Create(std::unique_ptr<metrics_sdk::ViewRegistry>(new metrics_sdk::ViewRegistry()), resource);
    auto meter_provider = opentelemetry::nostd::shared_ptr<opentelemetry::metrics::MeterProvider>(sdk_meter_provider.release());
    {
      otlp_exporter::OtlpFileMetricExporterOptions metric_options;
      otlp_exporter::OtlpFileClientFileSystemOptions otlp_metric_fs;
      otlp_metric_fs.file_pattern = metric_path;
      otlp_metric_fs.alias_pattern = metric_path + ".latest";
      otlp_metric_fs.flush_interval = std::chrono::milliseconds(200);
      otlp_metric_fs.flush_count = 1;
      metric_options.backend_options = otlp_metric_fs;
      auto otlp_metric_exporter = otlp_exporter::OtlpFileMetricExporterFactory::Create(metric_options);
      metrics_sdk::PeriodicExportingMetricReaderOptions reader_options;
      reader_options.export_interval_millis = std::chrono::milliseconds(200);
      reader_options.export_timeout_millis = std::chrono::milliseconds(1000);
      std::unique_ptr<metrics_sdk::MetricReader> reader(new metrics_sdk::PeriodicExportingMetricReader(std::move(otlp_metric_exporter), reader_options));
      static_cast<metrics_sdk::MeterProvider*>(meter_provider.get())->AddMetricReader(std::move(reader));
    }
    {
      auto in_memory_metric_exporter = std::unique_ptr<metrics_sdk::PushMetricExporter>(new InMemoryMetricExporter(status_state));
      metrics_sdk::PeriodicExportingMetricReaderOptions reader_options;
      reader_options.export_interval_millis = std::chrono::milliseconds(200);
      reader_options.export_timeout_millis = std::chrono::milliseconds(1000);
      std::unique_ptr<metrics_sdk::MetricReader> reader(new metrics_sdk::PeriodicExportingMetricReader(std::move(in_memory_metric_exporter), reader_options));
      static_cast<metrics_sdk::MeterProvider*>(meter_provider.get())->AddMetricReader(std::move(reader));
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
  }
  // Compute JSONL file locations again here for tailing in UI.
  auto MakeSiblingPathUI = [](const std::string &path, const std::string &suffix) -> std::string {
    const std::string ext = ".jsonl";
    if (path.size() >= ext.size() && path.compare(path.size() - ext.size(), ext.size(), ext) == 0) {
      return path.substr(0, path.size() - ext.size()) + suffix + ext;
    }
    return path + suffix;
  };
  const std::filesystem::path jsonl_path_default_ui = db_path.parent_path() / "otel.jsonl";
  const std::string jsonl_path_ui = GetEnv("CORTEXT_CHAT_OTEL_JSONL", jsonl_path_default_ui.string());
  const std::string log_path_ui = MakeSiblingPathUI(jsonl_path_ui, ".logs");
  const std::string log_alias_ui = log_path_ui + ".latest";
  // DebugLog("Main: Log alias UI: " + log_alias_ui);
  {
    cortext::telemetry::ScopedSpan span("chat.startup");
    span.SetStatusOk();
    cortext::telemetry::AddCounter("chat.startup_total", 1);
    cortext::telemetry::LogInfo("chat.startup");
  }

  const double focus = GetEnvDouble("CORTEXT_FOCUS", 0.5);
  const double sensitivity = GetEnvDouble("CORTEXT_SENSITIVITY", 0.5);
  const double stability = GetEnvDouble("CORTEXT_STABILITY", 0.5);

  // Initialize OpenAI client.
  openai::start(api_key, openai_org);
  if (!openai_base_url.empty()) {
    openai::instance().setBaseUrl(EnsureTrailingSlash(openai_base_url));
  }

  // Initialize encoder early so we can fail fast with actionable errors.
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

  // Initialize cortext.
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

  // UI shared state.
  std::mutex mu;
  // Serialize all SQLite write operations (both internal processor flushes
  // and our explicit persistence) to avoid SQLITE_BUSY/LOCKED conflicts.
  static std::mutex db_write_mu;
  std::vector<ChatMessage> history;
  std::vector<cortext::Cortext::Context::Memory> last_memories;
  std::deque<MemoryEvent> memory_events;
  bool last_should_interrupt = false;
  std::optional<std::string> last_error;
  bool generating = false;
  std::string input;
  int tab_selected = 0;
  static constexpr size_t kMaxMemoryEvents = 50;

  // Scroll state for each tab (0.0 = top, 1.0 = bottom)
  // Context and Logs tabs manage their own scroll state internally
  float chat_scroll = 1.0f;    // Start at bottom for chat
  float memory_scroll = 0.0f;


  auto screen = ftxui::ScreenInteractive::Fullscreen();

  auto render_chat = [&] {
    ftxui::Elements lines;
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& m : history) {
      std::string prefix;
      if (m.role == "user") prefix = "You: ";
      else if (m.role == "assistant") prefix = "Assistant: ";
      else prefix = "System: ";

      auto style = ftxui::color(ftxui::Color::White);
      if (m.role == "user") style = ftxui::color(ftxui::Color::Cyan);
      if (m.role == "assistant") style = ftxui::color(ftxui::Color::Green);
      if (m.role == "system") style = ftxui::color(ftxui::Color::GrayDark);

      lines.push_back(ftxui::text(prefix + m.content) | style);
      lines.push_back(ftxui::separator());
    }

    if (generating) {
      lines.push_back(ftxui::text("Generating...") | ftxui::color(ftxui::Color::Yellow));
    }

    return ftxui::vbox(std::move(lines))
           | ftxui::focusPositionRelative(0, chat_scroll)
           | ftxui::vscroll_indicator
           | ftxui::yframe;
  };

  auto get_metric = [](const std::unordered_map<int, double>& metrics, cortext::operations::Metric metric) -> std::optional<double> {
    auto it = metrics.find(static_cast<int>(metric));
    if (it != metrics.end()) {
      return it->second;
    }
    return std::nullopt;
  };

  auto format_metric = [](const std::optional<double>& val, const std::string& label) -> std::string {
    if (!val.has_value()) return "";
    std::ostringstream oss;
    oss << label << ": " << std::fixed << std::setprecision(2) << (*val * 100.0) << "%";
    return oss.str();
  };

  auto render_memory = [&, format_metric] {
    ftxui::Elements lines;
    std::lock_guard<std::mutex> lock(mu);

    lines.push_back(ftxui::text("🧠 MEMORY EVENTS (Real-time)") | ftxui::bold);
    lines.push_back(ftxui::separator());

    lines.push_back(ftxui::text(std::string("Interrupt: ") + (last_should_interrupt ? "⚠️ TRIGGERED" : "✓ None"))
                    | ftxui::color(last_should_interrupt ? ftxui::Color::Red : ftxui::Color::Green));

    if (last_error.has_value()) {
      lines.push_back(ftxui::text("Error: " + *last_error) | ftxui::color(ftxui::Color::Red));
    }

    lines.push_back(ftxui::separator());

    if (memory_events.empty()) {
      lines.push_back(ftxui::text("(no memory events yet)") | ftxui::color(ftxui::Color::GrayDark));
    } else {
    int shown = 0;
      for (auto it = memory_events.rbegin(); it != memory_events.rend() && shown < 10; ++it, ++shown) {
        const auto& evt = *it;
        
        // Header with type indicator
        std::string type_icon = (evt.type == MemoryEventType::STORED) ? "💾 STORED" : "🔍 RETRIEVED";
        auto type_color = (evt.type == MemoryEventType::STORED) ? ftxui::Color::Green : ftxui::Color::Blue;
        
        lines.push_back(ftxui::text(type_icon + " #" + std::to_string(evt.memory_id)) 
                        | ftxui::bold | ftxui::color(type_color));
        
        if (!evt.source_id.empty()) {
          lines.push_back(ftxui::text("  source: " + evt.source_id) | ftxui::color(ftxui::Color::GrayDark));
        }
        
        // Decision and score
        if (evt.composite_score.has_value() && evt.threshold.has_value()) {
          std::ostringstream score_line;
          score_line << "  Score: " << std::fixed << std::setprecision(3) << *evt.composite_score
                     << " | T: " << std::fixed << std::setprecision(3) << *evt.threshold;
          if (evt.decision.has_value()) {
            score_line << " | decision: " << (*evt.decision ? "STORE" : "SKIP");
          }
          lines.push_back(ftxui::text(score_line.str()) | ftxui::color(ftxui::Color::Yellow));
        }
        
        // Metrics row 1
        std::vector<std::string> row1;
        if (evt.relevance.has_value()) row1.push_back(format_metric(evt.relevance, "Rel"));
        if (evt.surprise.has_value()) row1.push_back(format_metric(evt.surprise, "Sur"));
        if (evt.mismatch.has_value()) row1.push_back(format_metric(evt.mismatch, "Mis"));
        if (evt.rarity.has_value()) row1.push_back(format_metric(evt.rarity, "Rar"));
        if (!row1.empty()) {
          std::string row1_str = "  ";
          for (size_t i = 0; i < row1.size(); ++i) {
            if (i > 0) row1_str += " | ";
            row1_str += row1[i];
          }
          lines.push_back(ftxui::text(row1_str) | ftxui::color(ftxui::Color::Cyan));
        }
        
        // Metrics row 2
        std::vector<std::string> row2;
        if (evt.drift.has_value()) row2.push_back(format_metric(evt.drift, "Drift"));
        if (evt.contradiction.has_value()) row2.push_back(format_metric(evt.contradiction, "Contra"));
        if (evt.utility.has_value()) row2.push_back(format_metric(evt.utility, "Util"));
        if (evt.periphery.has_value()) row2.push_back(format_metric(evt.periphery, "Periph"));
        if (!row2.empty()) {
          std::string row2_str = "  ";
          for (size_t i = 0; i < row2.size(); ++i) {
            if (i > 0) row2_str += " | ";
            row2_str += row2[i];
          }
          lines.push_back(ftxui::text(row2_str) | ftxui::color(ftxui::Color::Magenta));
        }
        
        // Emotion metrics
        std::vector<std::string> row3;
        if (evt.salience.has_value()) row3.push_back(format_metric(evt.salience, "Sal"));
        if (evt.valence.has_value()) row3.push_back(format_metric(evt.valence, "Val"));
        if (evt.arousal.has_value()) row3.push_back(format_metric(evt.arousal, "Aro"));
        if (!row3.empty()) {
          std::string row3_str = "  ";
          for (size_t i = 0; i < row3.size(); ++i) {
            if (i > 0) row3_str += " | ";
            row3_str += row3[i];
          }
          lines.push_back(ftxui::text(row3_str) | ftxui::color(ftxui::Color::Yellow));
        }
        
        // System-level metrics
        std::vector<std::string> row4;
        if (evt.coherence.has_value()) row4.push_back(format_metric(evt.coherence, "Coh"));
        if (evt.focus_spread.has_value()) row4.push_back(format_metric(evt.focus_spread, "FSpread"));
        if (evt.F_eff.has_value()) row4.push_back(format_metric(evt.F_eff, "F_eff"));
        if (!row4.empty()) {
          std::string row4_str = "  ";
          for (size_t i = 0; i < row4.size(); ++i) {
            if (i > 0) row4_str += " | ";
            row4_str += row4[i];
          }
          lines.push_back(ftxui::text(row4_str) | ftxui::color(ftxui::Color::GrayLight));
        }
        if (evt.encoder_latency_ms.has_value() || evt.retrieval_latency_ms.has_value() || evt.storage_latency_ms.has_value()) {
          std::ostringstream oss;
          oss << "  latency:";
          if (evt.encoder_latency_ms.has_value()) { oss << " enc=" << std::fixed << std::setprecision(1) << *evt.encoder_latency_ms << "ms"; }
          if (evt.retrieval_latency_ms.has_value()) { oss << " ret=" << std::fixed << std::setprecision(1) << *evt.retrieval_latency_ms << "ms"; }
          if (evt.storage_latency_ms.has_value()) { oss << " store=" << std::fixed << std::setprecision(1) << *evt.storage_latency_ms << "ms"; }
          lines.push_back(ftxui::text(oss.str()) | ftxui::color(ftxui::Color::GrayLight));
        }
        
        // Content preview
        std::string preview = evt.content;
        if (preview.size() > 120) preview = preview.substr(0, 120) + "...";
        lines.push_back(ftxui::text("  \"" + preview + "\"") | ftxui::color(ftxui::Color::White));
      lines.push_back(ftxui::separator());
      }
    }

    return ftxui::vbox(std::move(lines))
           | ftxui::focusPositionRelative(0, memory_scroll)
           | ftxui::vscroll_indicator
           | ftxui::yframe;
  };

  // LogListComponent is now globally defined.
  auto log_list_component = std::make_shared<LogListComponent>(log_buffer);
  ftxui::Component log_component = log_list_component;

  auto render_status_bar = [&] {
    std::optional<double> processing_ms;
    std::optional<std::int64_t> tokens_used;
    {
      std::lock_guard<std::mutex> lock(status_state->mu);
      processing_ms = status_state->processing_latency_ms;
      tokens_used = status_state->tokens_used;
    }
    std::ostringstream oss;
    if (processing_ms.has_value()) { oss << "proc=" << std::fixed << std::setprecision(1) << *processing_ms << "ms"; }
    else { oss << "proc=?"; }
    oss << " | ";
    if (tokens_used.has_value()) { oss << "tokens=" << *tokens_used; }
    else { oss << "tokens=?"; }
    return ftxui::hbox({
        ftxui::text("Tab: switch | Enter: send | Ctrl+C: quit") | ftxui::color(ftxui::Color::GrayDark),
        ftxui::filler(),
        ftxui::text(oss.str())
    }) | ftxui::border;
  };

  auto input_component = ftxui::Input(&input, "Type a message...");

  std::vector<std::string> tab_names = {"Chat", "Memory", "Context", "Logs"};
  auto tab_toggle = ftxui::Toggle(&tab_names, &tab_selected);

  auto tab_container = ftxui::Container::Tab({}, &tab_selected);
  tab_container->Add(ftxui::Renderer(render_chat));
  tab_container->Add(ftxui::Renderer(render_memory));
  tab_container->Add(chat::MakeContextTab(last_context));
  tab_container->Add(log_component);

  auto root = ftxui::Container::Vertical({
      tab_toggle,
      tab_container,
      input_component,
  });

  // Make panes look nicer.
  auto root_renderer = ftxui::Renderer(root, [&] {
    auto tab_bar = tab_toggle->Render() | ftxui::border;
    auto content = tab_container->Render() | ftxui::border | ftxui::flex;
    auto input_box = ftxui::hbox({ftxui::text(" > "), input_component->Render()}) | ftxui::border;
    auto status_bar = render_status_bar();
    return ftxui::vbox({tab_bar, content, input_box, status_bar});
  });

  // Helper to get scroll position pointer for current tab
  auto get_current_scroll = [&]() -> float* {
    switch (tab_selected) {
      case 0: return &chat_scroll;
      case 1: return &memory_scroll;
      case 2: return &last_context->scroll_position;
      default: return nullptr;  // Logs tab has its own handling
    }
  };

  // Event handling.
  auto app = ftxui::CatchEvent(root_renderer, [&, get_current_scroll] (const ftxui::Event& e) {
    if (e == ftxui::Event::Custom) {
       log_list_component->Sync();
       return false;
    }

    // Handle mouse wheel scrolling
    if (e.is_mouse()) {
      constexpr float kScrollStep = 0.05f;
      auto mouse_event = const_cast<ftxui::Event&>(e).mouse();
      if (mouse_event.button == ftxui::Mouse::WheelUp) {
        if (tab_selected == 3) {
          // Logs tab
          float pos = log_list_component->GetScrollPosition();
          log_list_component->SetScrollPosition(std::max(0.0f, pos - kScrollStep));
        } else if (float* scroll = get_current_scroll()) {
          *scroll = std::max(0.0f, *scroll - kScrollStep);
        }
        return true;
      }
      if (mouse_event.button == ftxui::Mouse::WheelDown) {
        if (tab_selected == 3) {
          // Logs tab
          float pos = log_list_component->GetScrollPosition();
          log_list_component->SetScrollPosition(std::min(1.0f, pos + kScrollStep));
        } else if (float* scroll = get_current_scroll()) {
          *scroll = std::min(1.0f, *scroll + kScrollStep);
        }
        return true;
      }
    }

    // Handle arrow keys for scrolling
    constexpr float kKeyScrollStep = 0.1f;
    if (e == ftxui::Event::ArrowUp) {
      if (tab_selected == 3) {
        float pos = log_list_component->GetScrollPosition();
        log_list_component->SetScrollPosition(std::max(0.0f, pos - kKeyScrollStep));
      } else if (float* scroll = get_current_scroll()) {
        *scroll = std::max(0.0f, *scroll - kKeyScrollStep);
      }
      return true;
    }
    if (e == ftxui::Event::ArrowDown) {
      if (tab_selected == 3) {
        float pos = log_list_component->GetScrollPosition();
        log_list_component->SetScrollPosition(std::min(1.0f, pos + kKeyScrollStep));
      } else if (float* scroll = get_current_scroll()) {
        *scroll = std::min(1.0f, *scroll + kKeyScrollStep);
      }
      return true;
    }

    if (e == ftxui::Event::Return) {
      std::string text;
      {
        std::lock_guard<std::mutex> lock(mu);
        if (generating) return true;
        text = input;
        input.clear();
        if (text.empty()) return true;
        generating = true;
        last_error.reset();
        history.push_back({"user", text});
      }

      // Launch background job: retrieval + OpenAI + persistence.
      std::thread([&, text] {
        const uint64_t ts = NowUnixSeconds();

        cortext::Cortext::Context retrieved;
        std::string injected_system;
        try {
          // Ensure cortext sees latest externally persisted rows by refreshing
          // its episode. Also serialize ProcessText() itself: it can perform
          // SQLite writes (updates/feedback) and will otherwise race with the
          // explicit persistence connection below, causing SQLITE_BUSY/LOCKED
          // and "SQLite step failed" telemetry.
          {
            std::lock_guard<std::mutex> lock(db_write_mu);
            cortext_ctx->Flush();
            retrieved = cortext_ctx->ProcessText(text, ts, "chat/user");
          }
          
          // Create memory events for retrieved memories
          {
            std::lock_guard<std::mutex> lock(mu);
            const std::optional<double> encoder_latency_ms = retrieval_latency_state->encoder_latency_ms;
            const std::optional<double> retrieval_latency_ms = retrieval_latency_state->retrieval_latency_ms;
            for (const auto& mem : retrieved.memories) {
              MemoryEvent evt;
              evt.type = MemoryEventType::RETRIEVED;
              evt.memory_id = mem.id;
              evt.content = mem.content;
              evt.source_id = mem.source_id;
              evt.timestamp = mem.timestamp;
              evt.composite_score = retrieved.output.composite_score;
              evt.threshold = retrieved.output.threshold;
              evt.should_interrupt = retrieved.should_interrupt;
              evt.encoder_latency_ms = encoder_latency_ms;
              evt.retrieval_latency_ms = retrieval_latency_ms;
              
              // Extract individual metrics using helper
              using M = cortext::operations::Metric;
              evt.relevance = get_metric(retrieved.output.metrics, M::relevance);
              evt.surprise = get_metric(retrieved.output.metrics, M::surprise);
              evt.mismatch = get_metric(retrieved.output.metrics, M::mismatch);
              evt.rarity = get_metric(retrieved.output.metrics, M::rarity);
              evt.drift = get_metric(retrieved.output.metrics, M::drift);
              evt.contradiction = get_metric(retrieved.output.metrics, M::contradiction);
              evt.utility = get_metric(retrieved.output.metrics, M::utility);
              evt.periphery = get_metric(retrieved.output.metrics, M::periphery);
              evt.salience = get_metric(retrieved.output.metrics, M::salience);
              evt.valence = retrieved.output.valence;
              evt.arousal = retrieved.output.arousal;
              evt.F_eff = retrieved.output.effective_focus;
              evt.focus_spread = get_metric(retrieved.output.metrics, M::focus_spread);
              
              memory_events.push_back(evt);
              if (memory_events.size() > kMaxMemoryEvents) {
                memory_events.pop_front();
              }
            }
          }
          
          injected_system = FormatMemoriesForSystemPrompt(retrieved.memories);
        } catch (const std::exception& ex) {
          std::lock_guard<std::mutex> lock(mu);
          last_error = std::string("cortext: ") + ex.what();
        }

        std::string assistant_reply;
        try {
          using Json = openai::Json;
          std::vector<ChatMessage> local_history;
          {
            std::lock_guard<std::mutex> lock(mu);
            local_history = history;
          }

          Json req;
          req["model"] = model;
          req["temperature"] = 0.7;
          req["messages"] = BuildOpenAIMessages<Json>(local_history, /*window*/ 10, injected_system);

          // Capture context for the Context tab
          {
            std::lock_guard<std::mutex> lock(last_context->mu);
            last_context->system_prompt = injected_system;
            last_context->messages.clear();
            for (const auto& m : req["messages"]) {
              last_context->messages.push_back({m["role"].get<std::string>(), m["content"].get<std::string>()});
            }
            last_context->raw_json = req.dump(2);
            last_context->has_data = true;
          }

          auto resp = openai::chat().create(req);
          assistant_reply = resp["choices"][0]["message"]["content"].get<std::string>();
          std::int64_t tokens_used = 0;
          if (resp.contains("usage") && resp["usage"].contains("total_tokens")) {
            tokens_used = resp["usage"]["total_tokens"].get<std::int64_t>();
          }
          last_tokens_used->store(tokens_used);
        } catch (const std::exception& ex) {
          assistant_reply = std::string("(OpenAI error: ") + ex.what() + ")";
          std::lock_guard<std::mutex> lock(mu);
          last_error = std::string("openai: ") + ex.what();
        }

        // Persist both user and assistant turns so future retrieval can find them.
        try {
          // Serialize DB writes with internal processor flushes.
          std::optional<PersistedRow> user_row_opt;
          std::optional<PersistedRow> asst_row_opt;
          std::optional<double> user_encoder_latency_ms;
          std::optional<double> user_storage_latency_ms;
          std::optional<double> asst_encoder_latency_ms;
          std::optional<double> asst_storage_latency_ms;
          {
            std::lock_guard<std::mutex> lock(db_write_mu);
            auto uniq = cortext::SQLiteStore::Create(db_path.string());
            auto store = std::shared_ptr<cortext::Store>(std::move(uniq));

            auto user_row = PersistTextMemory(*store, encoder, text, "chat/user", ts);
            user_encoder_latency_ms = persist_latency_state->encoder_latency_ms;
            user_storage_latency_ms = persist_latency_state->storage_latency_ms;
            auto asst_row = PersistTextMemory(*store, encoder, assistant_reply, "chat/assistant", ts + 1);
            asst_encoder_latency_ms = persist_latency_state->encoder_latency_ms;
            asst_storage_latency_ms = persist_latency_state->storage_latency_ms;
            user_row_opt = user_row;
            asst_row_opt = asst_row;
          }

          // Create memory events for stored memories
          {
            std::lock_guard<std::mutex> lock(mu);
            using M = cortext::operations::Metric;
            
            MemoryEvent user_evt;
            user_evt.type = MemoryEventType::STORED;
            user_evt.memory_id = static_cast<int>(user_row_opt->embedding_id);
            user_evt.content = text;
            user_evt.source_id = "chat/user";
            user_evt.timestamp = ts;
            user_evt.composite_score = retrieved.output.composite_score;
            user_evt.threshold = retrieved.output.threshold;
            user_evt.decision = retrieved.output.composite_score.has_value() && retrieved.output.threshold.has_value()
                ? std::optional<bool>((*retrieved.output.composite_score) > (*retrieved.output.threshold))
                : std::nullopt;
            user_evt.encoder_latency_ms = user_encoder_latency_ms;
            user_evt.storage_latency_ms = user_storage_latency_ms;
            user_evt.relevance = get_metric(retrieved.output.metrics, M::relevance);
            user_evt.surprise = get_metric(retrieved.output.metrics, M::surprise);
            user_evt.mismatch = get_metric(retrieved.output.metrics, M::mismatch);
            user_evt.rarity = get_metric(retrieved.output.metrics, M::rarity);
            user_evt.drift = get_metric(retrieved.output.metrics, M::drift);
            user_evt.contradiction = get_metric(retrieved.output.metrics, M::contradiction);
            user_evt.utility = get_metric(retrieved.output.metrics, M::utility);
            user_evt.periphery = get_metric(retrieved.output.metrics, M::periphery);
            user_evt.salience = get_metric(retrieved.output.metrics, M::salience);
            user_evt.valence = retrieved.output.valence;
            user_evt.arousal = retrieved.output.arousal;
            user_evt.F_eff = retrieved.output.effective_focus;
            user_evt.focus_spread = get_metric(retrieved.output.metrics, M::focus_spread);
            memory_events.push_back(user_evt);
            
            MemoryEvent asst_evt;
            asst_evt.type = MemoryEventType::STORED;
            asst_evt.memory_id = static_cast<int>(asst_row_opt->embedding_id);
            asst_evt.content = assistant_reply;
            asst_evt.source_id = "chat/assistant";
            asst_evt.timestamp = ts + 1;
            asst_evt.encoder_latency_ms = asst_encoder_latency_ms;
            asst_evt.storage_latency_ms = asst_storage_latency_ms;
            memory_events.push_back(asst_evt);
            
            while (memory_events.size() > kMaxMemoryEvents) {
              memory_events.pop_front();
            }
          }

          // Refresh cortext episode snapshot after external writes.
          {
            std::lock_guard<std::mutex> lock(db_write_mu);
            cortext_ctx->Flush();
          }
        } catch (const std::exception& ex) {
          std::lock_guard<std::mutex> lock(mu);
          last_error = std::string("persist: ") + ex.what();
        }

        // Apply UI updates.
        {
          std::lock_guard<std::mutex> lock(mu);
          last_memories = retrieved.memories;
          last_should_interrupt = retrieved.should_interrupt;
          history.push_back({"assistant", assistant_reply});
          generating = false;
        }

        screen.PostEvent(ftxui::Event::Custom);
      }).detach();

      return true;
    }



    return false;
  });

  // Periodic refresh to update logs (every 2 seconds)
  std::atomic<bool> running{true};
  std::thread refresh_thread([&] {
    while (running) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      screen.PostEvent(ftxui::Event::Custom);
    }
  });

  // Background tailer thread to stream JSONL files into the Logs tab.
  std::thread tail_thread([&, log_alias_ui] {
    std::map<std::string, std::uintmax_t> offsets; // last read byte offset per file
    auto try_tail = [&](const std::string &path) {
      namespace fs = std::filesystem;
      std::error_code ec;
      if (!fs::exists(path, ec)) {
        return; // file not yet created
      }
      const std::uintmax_t size = fs::file_size(path, ec);
      if (ec) return;
      std::uintmax_t &off = offsets[path];
      if (off > size) {
        // file rotated or truncated; reset
        off = 0;
      }
      if (size == off) {
        return; // nothing new
      }
      std::ifstream in(path, std::ios::in | std::ios::binary);
      if (!in) return;
      in.seekg(static_cast<std::streamoff>(off));
      std::string line;
      while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        log_buffer->Append(line);
      }
      off = static_cast<std::uintmax_t>(in.tellg());
    };
    while (running) {
      try_tail(log_alias_ui);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  });

  screen.Loop(app);
  running = false;
  if (refresh_thread.joinable()) {
    refresh_thread.join();
  }
  if (tail_thread.joinable()) {
    tail_thread.join();
  }

  try {
    cortext_ctx->Flush();
  } catch (...) {
  }

  return 0;
}
