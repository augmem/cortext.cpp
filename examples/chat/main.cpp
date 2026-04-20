#include "chat_window.hpp"
#include "chunk_diagnostics.hpp"
#include "context_tab.hpp"
#include "encoder/text_encoder_factory.hpp"
#include "imgui_app.hpp"
#include "metrics_state.hpp"
#include "streaming_client.hpp"
#include "streaming_text_probe.hpp"
#include "voice_session.hpp"

#include <cortext/cortext.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/internal/cancellation.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/telemetry/telemetry.hpp>

#include <openai/openai.hpp>
#include <nlohmann/json.hpp>

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

#if CORTEXT_CHAT_ENABLE_OTLP_GRPC
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_metric_exporter.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_metric_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_options.h>
#endif

#include <opentelemetry/sdk/logs/read_write_log_record.h>
#include <opentelemetry/sdk/logs/exporter.h>

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
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <map>

namespace {

template <typename Value>
std::string FormatOwnedTelemetryValue(const Value &value) {
  std::ostringstream oss;
  opentelemetry::nostd::visit([&oss](auto &&v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, bool>) { oss << (v ? "true" : "false"); }
    else if constexpr (std::is_same_v<T, int32_t>) { oss << v; }
    else if constexpr (std::is_same_v<T, int64_t>) { oss << v; }
    else if constexpr (std::is_same_v<T, uint32_t>) { oss << v; }
    else if constexpr (std::is_same_v<T, uint64_t>) { oss << v; }
    else if constexpr (std::is_same_v<T, double>) { oss << v; }
    else if constexpr (std::is_same_v<T, std::string>) { oss << v; }
    else { oss << "?"; }
  }, value);
  return oss.str();
}

// Simple log exporter with pretty one-line format to stdout and file
class SimpleStdoutLogExporter final : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  explicit SimpleStdoutLogExporter(const std::string& log_file_path = "",
                                   std::shared_ptr<chat::OTelState> otel_state = nullptr)
      : log_file_path_(log_file_path), otel_state_(std::move(otel_state)) {
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
        attrs << FormatOwnedTelemetryValue(kv.second);
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

      if (otel_state_) {
        std::lock_guard<std::mutex> lock(otel_state_->mu);
        static uint64_t next_log_id = 1;
        chat::LogEntry entry;
        entry.id = next_log_id++;
        entry.raw = line.str();
        entry.severity = severity_text;
        entry.body = body;
        for (const auto& kv : log_record->GetAttributes()) {
          entry.attributes.emplace_back(kv.first, FormatOwnedTelemetryValue(kv.second));
        }
        otel_state_->logs.push_back(std::move(entry));
        while (otel_state_->logs.size() > 400) {
          otel_state_->logs.pop_front();
        }
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
  std::shared_ptr<chat::OTelState> otel_state_;
};

struct RetrievalLatencyState {
  std::optional<double> encoder_latency_ms;
  std::optional<double> retrieval_latency_ms;
};

class InMemorySpanExporter final : public opentelemetry::sdk::trace::SpanExporter {
public:
  explicit InMemorySpanExporter(std::shared_ptr<RetrievalLatencyState> retrieval_state,
                                std::shared_ptr<chat::OTelState> otel_state)
      : retrieval_state_(std::move(retrieval_state)),
        otel_state_(std::move(otel_state)) {}
  std::unique_ptr<opentelemetry::sdk::trace::Recordable> MakeRecordable() noexcept override {
    return std::unique_ptr<opentelemetry::sdk::trace::Recordable>(new opentelemetry::sdk::trace::SpanData());
  }
  opentelemetry::sdk::common::ExportResult Export(const opentelemetry::nostd::span<std::unique_ptr<opentelemetry::sdk::trace::Recordable>> &spans) noexcept override {
    for (const auto &span : spans) {
      if (!span) { continue; }
      auto *span_data = static_cast<opentelemetry::sdk::trace::SpanData *>(span.get());
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
      }

      if (otel_state_) {
        std::lock_guard<std::mutex> lock(otel_state_->mu);
        static uint64_t next_trace_id = 1;
        chat::TraceEntry entry;
        entry.id = next_trace_id++;
        entry.name = name;
        entry.duration_ms = duration_ms;
        const auto status = span_data->GetStatus();
        switch (status) {
          case opentelemetry::trace::StatusCode::kOk:
            entry.status = "ok";
            break;
          case opentelemetry::trace::StatusCode::kError:
            entry.status = "error";
            break;
          default:
            entry.status = "unset";
            break;
        }
        for (const auto &kv : span_data->GetAttributes()) {
          entry.attributes.emplace_back(kv.first, FormatOwnedTelemetryValue(kv.second));
        }
        otel_state_->traces.push_back(std::move(entry));
        while (otel_state_->traces.size() > 300) {
          otel_state_->traces.pop_front();
        }
      }

      if (name == "cortext.api.process_text") {
        retrieval_active_ = false;
        retrieval_last_encode_ms_.reset();
        retrieval_last_hydrate_ms_.reset();
      }
    }
    return opentelemetry::sdk::common::ExportResult::kSuccess;
  }
  bool ForceFlush(std::chrono::microseconds) noexcept override { return true; }
  bool Shutdown(std::chrono::microseconds) noexcept override { return true; }
private:
  std::shared_ptr<RetrievalLatencyState> retrieval_state_;
  std::shared_ptr<chat::OTelState> otel_state_;
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
    current_memories.clear();
    restart_count = 0;
  }
};

struct PendingUserTurn {
  std::string text;
  std::string source_id = "chat/user";
  std::string speaker_id;
  bool request_reply = true;
  bool retain_input = true;
  bool voice_origin = false;
};

std::string TrimUserText(const std::string& text) {
  const auto start = text.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(start, end - start + 1);
}

bool IsSentenceBoundary(char c) {
  switch (c) {
    case '.':
    case '!':
    case '?':
    case '\n':
      return true;
    default:
      return false;
  }
}

std::vector<std::string> DrainCompletedSentences(std::string* buffer,
                                                 bool flush = false) {
  std::vector<std::string> out;
  if (buffer == nullptr || buffer->empty()) {
    return out;
  }

  std::size_t consumed = 0;
  for (std::size_t i = 0; i < buffer->size(); ++i) {
    if (!IsSentenceBoundary((*buffer)[i])) {
      continue;
    }
    const std::string sentence = TrimUserText(
        buffer->substr(consumed, i - consumed + 1));
    if (!sentence.empty()) {
      out.push_back(sentence);
    }
    consumed = i + 1;
  }

  if (consumed > 0) {
    buffer->erase(0, consumed);
  }
  if (flush && !buffer->empty()) {
    const std::string tail = TrimUserText(*buffer);
    if (!tail.empty()) {
      out.push_back(tail);
    }
    buffer->clear();
  }
  return out;
}

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

  bool ShouldConsolidate(int idle_required_seconds) {
    std::lock_guard<std::mutex> lock(mu);
    if (!consolidation_pending) return false;
    auto idle = std::chrono::steady_clock::now() - last_activity;
    return std::chrono::duration_cast<std::chrono::seconds>(idle).count()
           >= idle_required_seconds;
  }

  std::optional<int> RemainingSeconds(int idle_required_seconds) const {
    std::lock_guard<std::mutex> lock(mu);
    if (!consolidation_pending) return std::nullopt;
    const auto idle = std::chrono::steady_clock::now() - last_activity;
    const int idle_seconds
        = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(idle).count());
    return std::max(0, idle_required_seconds - idle_seconds);
  }

  void MarkConsolidated() {
    std::lock_guard<std::mutex> lock(mu);
    consolidation_pending = false;
  }
};

int IdleConsolidationSeconds(double stability, int override_seconds) {
  if (override_seconds > 0) {
    return override_seconds;
  }
  return cortext::core::IdleRequiredSeconds(stability);
}

constexpr std::size_t kStreamingProbeMinChars = 32;
constexpr std::size_t kStreamingProbeMaxChars = 128;

std::size_t TrimmedProbeLength(const std::string& text) {
  std::size_t end = text.size();
  while (end > 0
         && std::isspace(static_cast<unsigned char>(text[end - 1]))
         && text[end - 1] != '\n') {
    --end;
  }
  return end;
}

bool HasProbeBoundary(const std::string& text) {
  if (text.empty()) return false;
  const char tail = text.back();
  if (tail == '\n') return true;

  std::size_t idx = text.size();
  while (idx > 0 && std::isspace(static_cast<unsigned char>(text[idx - 1]))) {
    --idx;
  }
  if (idx == 0) return false;

  switch (text[idx - 1]) {
    case '.':
    case '!':
    case '?':
    case ':':
    case ';':
      return true;
    default:
      return false;
  }
}

bool ShouldRunStreamingProbe(const std::string& text, bool force_flush = false) {
  const std::size_t trimmed = TrimmedProbeLength(text);
  if (trimmed == 0) return false;
  if (force_flush) return true;
  if (trimmed >= kStreamingProbeMaxChars) return true;
  if (trimmed < kStreamingProbeMinChars) return false;
  return HasProbeBoundary(text);
}

bool IsCancelledConsolidationError(const std::exception& ex) {
  const std::string_view msg(ex.what());
  return msg.find("cancelled") != std::string_view::npos
      || msg.find("interrupted") != std::string_view::npos
      || msg.find("SQLITE_INTERRUPT") != std::string_view::npos;
}

uint64_t NowUnixMillis();

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

std::string TruncatePreview(const std::string& text, std::size_t max_len = 160) {
  if (text.size() <= max_len) return text;
  return text.substr(0, max_len) + "...";
}

std::string CreateVoiceSurfacedMemoryPreview(
    const cortext::Cortext::Context::Memory& mem) {
  std::string preview = TruncatePreview(ExtractTextFromBlobs(mem.content), 120);
  if (!preview.empty()) {
    return preview;
  }
  if (!mem.source_id.empty()) {
    return mem.source_id;
  }
  return "memory #" + std::to_string(mem.id);
}

void UpdateVoiceSpeakerPreview(chat::VoiceState& voice_state,
                               const std::string& speaker_id,
                               const std::string& utterance) {
  if (speaker_id.empty()) {
    return;
  }
  auto it = std::find_if(voice_state.speakers.begin(),
                         voice_state.speakers.end(),
                         [&](const chat::VoiceSpeakerPreview& item) {
                           return item.speaker_id == speaker_id;
                         });
  if (it == voice_state.speakers.end()) {
    chat::VoiceSpeakerPreview preview;
    preview.speaker_id = speaker_id;
    preview.last_utterance = TruncatePreview(utterance, 96);
    preview.utterance_count = 1;
    voice_state.speakers.insert(voice_state.speakers.begin(), std::move(preview));
    return;
  }
  it->last_utterance = TruncatePreview(utterance, 96);
  it->utterance_count += 1;
  chat::VoiceSpeakerPreview updated = *it;
  voice_state.speakers.erase(it);
  voice_state.speakers.insert(voice_state.speakers.begin(), std::move(updated));
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

chat::MemoryEvent CreateDroppedMemoryEvent(
    const cortext::Cortext::Context::Memory& mem,
    const std::string& reason) {
  auto evt = CreateMemoryEvent(chat::MemoryEventType::DROPPED, mem);
  evt.filter_reason = reason;
  return evt;
}

void PushMemoryEvent(std::deque<chat::MemoryEvent>& memory_events,
                     chat::MemoryEvent event,
                     std::size_t max_events) {
  memory_events.push_back(std::move(event));
  while (memory_events.size() > max_events) {
    memory_events.pop_front();
  }
}

chat::ChunkRetrievedMemory CreateRetrievedMemoryPreview(
    const cortext::Cortext::Context::Memory& mem) {
  chat::ChunkRetrievedMemory preview;
  preview.memory_id = mem.id;
  preview.source_id = mem.source_id;
  preview.preview = TruncatePreview(ExtractTextFromBlobs(mem.content));
  preview.relevance = mem.relevance;
  preview.composite_score = mem.composite_score;
  return preview;
}

chat::ChunkProbeEvent BeginChunkProbeEvent(
    chat::ChunkDiagnosticsState& diagnostics,
    const std::string& chunk_text,
    int restart_index,
    std::size_t tokens_so_far) {
  chat::ChunkProbeEvent event;
  {
    std::lock_guard<std::mutex> lock(diagnostics.mu);
    event.event_id = diagnostics.next_event_id++;
    diagnostics.active_probe.reset();
  }
  event.started_at_ms = NowUnixMillis();
  event.in_progress = true;
  event.chunk_text = chunk_text;
  event.chunk_char_count = chunk_text.size();
  event.restart_index = restart_index;
  event.tokens_so_far = tokens_so_far;
  return event;
}

void PublishActiveChunkProbe(chat::ChunkDiagnosticsState& diagnostics,
                             const chat::ChunkProbeEvent& event) {
  std::lock_guard<std::mutex> lock(diagnostics.mu);
  diagnostics.active_probe = event;
}

void FinalizeChunkProbeSuccess(chat::ChunkProbeEvent& event,
                               const cortext::Cortext::Context& ctx,
                               double boundary_threshold,
                               bool boundary_score_pass,
                               int new_memory_count,
                               bool interrupt_ignored_restart_cap) {
  event.in_progress = false;
  event.completed_at_ms = NowUnixMillis();
  event.encode_ms = ctx.encode_ms;
  event.process_ms = ctx.process_ms;
  event.hydrate_ms = ctx.hydrate_ms;
  event.total_ms = ctx.total_ms;
  event.at_boundary = ctx.at_boundary;
  event.boundary_score = ctx.boundary_score;
  event.boundary_threshold = boundary_threshold;
  event.boundary_score_pass = boundary_score_pass;
  event.should_interrupt = ctx.should_interrupt;
  event.new_memory_count = new_memory_count;
  event.interrupt_ignored_restart_cap = interrupt_ignored_restart_cap;
  event.raw_retrieved_count = ctx.retrieved_memory.size();
  event.interrupt_gate_has_candidates = ctx.interrupt_gate_has_candidates;
  event.interrupt_gate_blocked_no_store = ctx.interrupt_gate_blocked_no_store;
  event.interrupt_gate_rel_pass = ctx.interrupt_gate_rel_pass;
  event.interrupt_gate_novelty_pass = ctx.interrupt_gate_novelty_pass;
  event.interrupt_gate_mu_pass = ctx.interrupt_gate_mu_pass;
  event.interrupt_gate_novelty_mu_pass = ctx.interrupt_gate_novelty_mu_pass;
  event.interrupt_gate_dup_pass = ctx.interrupt_gate_dup_pass;
  event.interrupt_gate_boundary_mu_pass = ctx.interrupt_gate_boundary_mu_pass;
  event.interrupt_gate_rel_star = ctx.interrupt_gate_rel_star;
  event.interrupt_gate_retrieval_thresh = ctx.interrupt_gate_retrieval_thresh;
  event.interrupt_gate_boundary_mult_eff = ctx.interrupt_gate_boundary_mult_eff;
  event.interrupt_gate_affect_drive = ctx.interrupt_gate_affect_drive;
  for (const auto& mem : ctx.retrieved_memory) {
    event.retrieved_memories.push_back(CreateRetrievedMemoryPreview(mem));
  }
  event.reason = chat::ClassifyChunkProbeReason({
      .should_interrupt = ctx.should_interrupt,
      .new_memory_count = new_memory_count,
      .interrupt_ignored_restart_cap = interrupt_ignored_restart_cap,
      .at_boundary = ctx.at_boundary,
      .boundary_score_pass = boundary_score_pass,
      .interrupt_gate_has_candidates = ctx.interrupt_gate_has_candidates,
      .interrupt_gate_blocked_no_store = ctx.interrupt_gate_blocked_no_store,
      .interrupt_gate_rel_pass = ctx.interrupt_gate_rel_pass,
      .interrupt_gate_novelty_mu_pass = ctx.interrupt_gate_novelty_mu_pass,
      .interrupt_gate_dup_pass = ctx.interrupt_gate_dup_pass,
      .interrupt_gate_boundary_mu_pass = ctx.interrupt_gate_boundary_mu_pass,
  });
}

void FinalizeChunkProbeError(chat::ChunkProbeEvent& event,
                             const std::string& error_message) {
  event.in_progress = false;
  event.had_error = true;
  event.error_message = error_message;
  event.completed_at_ms = NowUnixMillis();
  event.reason = "error";
}

void CommitChunkProbeEvent(chat::ChunkDiagnosticsState& diagnostics,
                           chat::ChunkProbeEvent event,
                           std::size_t max_events) {
  std::lock_guard<std::mutex> lock(diagnostics.mu);
  diagnostics.active_probe.reset();
  diagnostics.recent_probes.push_back(std::move(event));
  while (diagnostics.recent_probes.size() > max_events) {
    diagnostics.recent_probes.pop_front();
  }
  diagnostics.total_probes++;
  const auto& latest = diagnostics.recent_probes.back();
  if (latest.reason == "interrupt_triggered") {
    diagnostics.interrupts++;
  } else if (latest.reason == "interrupt_suppressed_no_new_memories"
             || latest.reason == "interrupt_ignored_restart_cap") {
    diagnostics.suppressed++;
  }
  if (latest.had_error) {
    diagnostics.errors++;
  }
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

long long GetAnyInt64(const std::map<std::string, std::any>& row,
                     const std::string& key,
                     long long fallback = 0) {
  auto it = row.find(key);
  if (it == row.end() || !it->second.has_value()) return fallback;
  if (it->second.type() == typeid(long long)) {
    return std::any_cast<long long>(it->second);
  }
  if (it->second.type() == typeid(int)) {
    return static_cast<long long>(std::any_cast<int>(it->second));
  }
  if (it->second.type() == typeid(double)) {
    return static_cast<long long>(std::any_cast<double>(it->second));
  }
  return fallback;
}

double GetAnyDouble(const std::map<std::string, std::any>& row,
                    const std::string& key,
                    double fallback = 0.0) {
  auto it = row.find(key);
  if (it == row.end() || !it->second.has_value()) return fallback;
  if (it->second.type() == typeid(double)) {
    return std::any_cast<double>(it->second);
  }
  if (it->second.type() == typeid(long long)) {
    return static_cast<double>(std::any_cast<long long>(it->second));
  }
  if (it->second.type() == typeid(int)) {
    return static_cast<double>(std::any_cast<int>(it->second));
  }
  return fallback;
}

std::string GetAnyString(const std::map<std::string, std::any>& row,
                         const std::string& key,
                         const std::string& fallback = {}) {
  auto it = row.find(key);
  if (it == row.end() || !it->second.has_value()) return fallback;
  if (it->second.type() == typeid(std::string)) {
    return std::any_cast<std::string>(it->second);
  }
  if (it->second.type() == typeid(const char*)) {
    return std::string(std::any_cast<const char*>(it->second));
  }
  return fallback;
}

uint64_t NowUnixMillis();

void RefreshDatabaseExplorer(chat::DatabaseExplorerState& state,
                             const std::string& db_path) {
  auto store = cortext::SQLiteStore::Create(db_path);

  const auto mem_count = store->Execute("SELECT COUNT(*) AS cnt FROM memories", {});
  const auto sig_count = store->Execute("SELECT COUNT(*) AS cnt FROM signals", {});
  const auto assoc_count = store->Execute("SELECT COUNT(*) AS cnt FROM associations", {});
  const auto ep_count = store->Execute("SELECT COUNT(*) AS cnt FROM episodes", {});
  const auto fact_count = store->Execute("SELECT COUNT(*) AS cnt FROM fact_assertions", {});
  const auto eviction_count = store->Execute("SELECT COUNT(*) AS cnt FROM memory_evictions", {});

  const auto memory_rows = store->Execute(
      "SELECT memory_id, COALESCE(episode_id, 0) AS episode_id, "
      "       COALESCE(embedding_id, 0) AS embedding_id, kind, source_id, "
      "       COALESCE(label, '') AS label, start_ts, COALESCE(end_ts, 0) AS end_ts, "
      "       n_signals, strength, connectivity, source_reliability, "
      "       retrieved_count, used_count "
      "FROM memories "
      "ORDER BY created_at DESC, memory_id DESC "
      "LIMIT 80",
      {});

  const auto signal_rows = store->Execute(
      "SELECT signal_id, COALESCE(memory_id, 0) AS memory_id, embedding_id, "
      "       source_id, modality, COALESCE(mime, '') AS mime, timestamp, "
      "       serial_position, score, COALESCE(salience, 0.0) AS salience, "
      "       COALESCE(threshold_t, 0.0) AS threshold_t "
      "FROM signals "
      "ORDER BY timestamp DESC, signal_id DESC "
      "LIMIT 80",
      {});

  const auto association_rows = store->Execute(
      "SELECT a.source_memory_id, a.target_memory_id, a.edge_type, a.weight, "
      "       COALESCE(a.decay_rate, 0.0) AS decay_rate, "
      "       COALESCE(a.last_reinforced, 0) AS last_reinforced, "
      "       COALESCE(sm.kind, '') AS source_kind, "
      "       COALESCE(sm.label, '') AS source_label, "
      "       COALESCE(sm.source_id, '') AS source_source_id, "
      "       COALESCE(tm.kind, '') AS target_kind, "
      "       COALESCE(tm.label, '') AS target_label, "
      "       COALESCE(tm.source_id, '') AS target_source_id "
      "FROM associations a "
      "LEFT JOIN memories sm ON sm.memory_id = a.source_memory_id "
      "LEFT JOIN memories tm ON tm.memory_id = a.target_memory_id "
      "ORDER BY a.weight DESC, a.last_reinforced DESC, a.source_memory_id DESC "
      "LIMIT 120",
      {});

  const auto episode_rows = store->Execute(
      "SELECT episode_id, start_ts, COALESCE(end_ts, 0) AS end_ts, "
      "       COALESCE(boundary_type, '') AS boundary_type, created_at "
      "FROM episodes "
      "ORDER BY start_ts DESC, episode_id DESC "
      "LIMIT 40",
      {});

  const auto fact_rows = store->Execute(
      "SELECT fact_id, subject, predicate, object, "
      "       COALESCE(valid_start_ts, 0) AS valid_start_ts, "
      "       COALESCE(valid_end_ts, 0) AS valid_end_ts, "
      "       recorded_at_ts, "
      "       COALESCE(superseded_at_ts, 0) AS superseded_at_ts, "
      "       confidence, summary_memory_id, "
      "       COALESCE(lifecycle_state, '') AS lifecycle_state, "
      "       COALESCE(severity_class, '') AS severity_class "
      "FROM fact_assertions "
      "ORDER BY recorded_at_ts DESC, fact_id DESC "
      "LIMIT 80",
      {});

  const auto eviction_rows = store->Execute(
      "SELECT eviction_id, memory_id, COALESCE(embedding_id, 0) AS embedding_id, "
      "       source_id, kind, COALESCE(label, '') AS label, start_ts, "
      "       COALESCE(end_ts, 0) AS end_ts, created_at, "
      "       COALESCE(last_access, 0) AS last_access, "
      "       evicted_at, strength, use_frequency, contextual_gain, "
      "       retrieved_count, used_count, n_signals, "
      "       COALESCE(modality, '') AS modality, "
      "       COALESCE(eviction_reason, '') AS eviction_reason "
      "FROM memory_evictions "
      "ORDER BY evicted_at DESC, eviction_id DESC "
      "LIMIT 80",
      {});

  std::vector<chat::DatabaseMemoryRow> memories;
  memories.reserve(memory_rows.size());
  for (const auto& row : memory_rows) {
    chat::DatabaseMemoryRow item;
    item.memory_id = GetAnyInt64(row, "memory_id");
    item.episode_id = GetAnyInt64(row, "episode_id");
    item.embedding_id = GetAnyInt64(row, "embedding_id");
    item.kind = GetAnyString(row, "kind");
    item.source_id = GetAnyString(row, "source_id");
    item.label = GetAnyString(row, "label");
    item.start_ts = static_cast<uint64_t>(GetAnyInt64(row, "start_ts"));
    item.end_ts = static_cast<uint64_t>(GetAnyInt64(row, "end_ts"));
    item.n_signals = GetAnyInt64(row, "n_signals");
    item.strength = GetAnyDouble(row, "strength");
    item.connectivity = GetAnyDouble(row, "connectivity");
    item.source_reliability = GetAnyDouble(row, "source_reliability");
    item.retrieved_count = GetAnyInt64(row, "retrieved_count");
    item.used_count = GetAnyInt64(row, "used_count");
    memories.push_back(std::move(item));
  }

  std::vector<chat::DatabaseSignalRow> signals;
  signals.reserve(signal_rows.size());
  for (const auto& row : signal_rows) {
    chat::DatabaseSignalRow item;
    item.signal_id = GetAnyInt64(row, "signal_id");
    item.memory_id = GetAnyInt64(row, "memory_id");
    item.embedding_id = GetAnyInt64(row, "embedding_id");
    item.source_id = GetAnyString(row, "source_id");
    item.modality = GetAnyString(row, "modality");
    item.mime = GetAnyString(row, "mime");
    item.timestamp = static_cast<uint64_t>(GetAnyInt64(row, "timestamp"));
    item.serial_position = GetAnyInt64(row, "serial_position");
    item.score = GetAnyDouble(row, "score");
    item.salience = GetAnyDouble(row, "salience");
    item.threshold_t = GetAnyDouble(row, "threshold_t");
    signals.push_back(std::move(item));
  }

  std::vector<chat::DatabaseAssociationRow> associations;
  associations.reserve(association_rows.size());
  for (const auto& row : association_rows) {
    chat::DatabaseAssociationRow item;
    item.source_memory_id = GetAnyInt64(row, "source_memory_id");
    item.target_memory_id = GetAnyInt64(row, "target_memory_id");
    item.edge_type = GetAnyString(row, "edge_type");
    item.weight = GetAnyDouble(row, "weight");
    item.decay_rate = GetAnyDouble(row, "decay_rate");
    item.last_reinforced = static_cast<uint64_t>(GetAnyInt64(row, "last_reinforced"));
    item.source_kind = GetAnyString(row, "source_kind");
    item.source_label = GetAnyString(row, "source_label");
    item.source_source_id = GetAnyString(row, "source_source_id");
    item.target_kind = GetAnyString(row, "target_kind");
    item.target_label = GetAnyString(row, "target_label");
    item.target_source_id = GetAnyString(row, "target_source_id");
    associations.push_back(std::move(item));
  }

  std::vector<chat::DatabaseEpisodeRow> episodes;
  episodes.reserve(episode_rows.size());
  for (const auto& row : episode_rows) {
    chat::DatabaseEpisodeRow item;
    item.episode_id = GetAnyInt64(row, "episode_id");
    item.start_ts = static_cast<uint64_t>(GetAnyInt64(row, "start_ts"));
    item.end_ts = static_cast<uint64_t>(GetAnyInt64(row, "end_ts"));
    item.boundary_type = GetAnyString(row, "boundary_type");
    item.created_at = static_cast<uint64_t>(GetAnyInt64(row, "created_at"));
    episodes.push_back(std::move(item));
  }

  std::vector<chat::DatabaseFactRow> facts;
  facts.reserve(fact_rows.size());
  for (const auto& row : fact_rows) {
    chat::DatabaseFactRow item;
    item.fact_id = GetAnyInt64(row, "fact_id");
    item.subject = GetAnyString(row, "subject");
    item.predicate = GetAnyString(row, "predicate");
    item.object = GetAnyString(row, "object");
    item.valid_start_ts = static_cast<uint64_t>(GetAnyInt64(row, "valid_start_ts"));
    item.valid_end_ts = static_cast<uint64_t>(GetAnyInt64(row, "valid_end_ts"));
    item.recorded_at_ts = static_cast<uint64_t>(GetAnyInt64(row, "recorded_at_ts"));
    item.superseded_at_ts = static_cast<uint64_t>(GetAnyInt64(row, "superseded_at_ts"));
    item.confidence = GetAnyDouble(row, "confidence");
    item.summary_memory_id = GetAnyInt64(row, "summary_memory_id");
    item.lifecycle_state = GetAnyString(row, "lifecycle_state");
    item.severity_class = GetAnyString(row, "severity_class");
    facts.push_back(std::move(item));
  }

  std::vector<chat::DatabaseEvictionRow> evictions;
  evictions.reserve(eviction_rows.size());
  for (const auto& row : eviction_rows) {
    chat::DatabaseEvictionRow item;
    item.eviction_id = GetAnyInt64(row, "eviction_id");
    item.memory_id = GetAnyInt64(row, "memory_id");
    item.embedding_id = GetAnyInt64(row, "embedding_id");
    item.source_id = GetAnyString(row, "source_id");
    item.kind = GetAnyString(row, "kind");
    item.label = GetAnyString(row, "label");
    item.start_ts = static_cast<uint64_t>(GetAnyInt64(row, "start_ts"));
    item.end_ts = static_cast<uint64_t>(GetAnyInt64(row, "end_ts"));
    item.created_at = static_cast<uint64_t>(GetAnyInt64(row, "created_at"));
    item.last_access = static_cast<uint64_t>(GetAnyInt64(row, "last_access"));
    item.evicted_at = static_cast<uint64_t>(GetAnyInt64(row, "evicted_at"));
    item.strength = GetAnyDouble(row, "strength");
    item.use_frequency = GetAnyDouble(row, "use_frequency");
    item.contextual_gain = GetAnyDouble(row, "contextual_gain");
    item.retrieved_count = GetAnyInt64(row, "retrieved_count");
    item.used_count = GetAnyInt64(row, "used_count");
    item.n_signals = GetAnyInt64(row, "n_signals");
    item.modality = GetAnyString(row, "modality");
    item.eviction_reason = GetAnyString(row, "eviction_reason");
    evictions.push_back(std::move(item));
  }

  std::lock_guard<std::mutex> lock(state.mu);
  state.memories = std::move(memories);
  state.signals = std::move(signals);
  state.associations = std::move(associations);
  state.episodes = std::move(episodes);
  state.facts = std::move(facts);
  state.evictions = std::move(evictions);
  state.total_memories = mem_count.empty() ? 0 : GetAnyInt64(mem_count.front(), "cnt");
  state.total_signals = sig_count.empty() ? 0 : GetAnyInt64(sig_count.front(), "cnt");
  state.total_associations = assoc_count.empty() ? 0 : GetAnyInt64(assoc_count.front(), "cnt");
  state.total_episodes = ep_count.empty() ? 0 : GetAnyInt64(ep_count.front(), "cnt");
  state.total_facts = fact_count.empty() ? 0 : GetAnyInt64(fact_count.front(), "cnt");
  state.total_evictions = eviction_count.empty() ? 0 : GetAnyInt64(eviction_count.front(), "cnt");
  state.refreshed_at = NowUnixMillis();
  state.refresh_requested = false;
  state.last_refresh_status =
      "Loaded " + std::to_string(state.memories.size()) + " memories, "
      + std::to_string(state.associations.size()) + " relationships, "
      + std::to_string(state.signals.size()) + " signals, "
      + std::to_string(state.facts.size()) + " temporal facts, "
      + std::to_string(state.evictions.size()) + " evictions.";
}

void ResetChatDatabase(cortext::Cortext::Config cfg,
                       std::unique_ptr<cortext::Cortext>& cortext_ctx,
                       const std::filesystem::path& db_path,
                       const std::filesystem::path& models_dir,
                       chat::DatabaseExplorerState& db_state,
                       std::mutex& db_write_mu) {
  std::lock_guard<std::mutex> lock(db_write_mu);
  if (cortext_ctx) {
    cortext_ctx->Flush();
    cortext_ctx.reset();
  }

  if (db_path != std::filesystem::path(":memory:")) {
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path.string() + "-wal", ec);
    std::filesystem::remove(db_path.string() + "-shm", ec);
  }

  cortext_ctx = cortext::Cortext::Create(cfg, db_path.string(), models_dir.string());
  RefreshDatabaseExplorer(db_state, db_path.string());
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

struct PersistedChatSettings {
  std::optional<double> focus;
  std::optional<double> sensitivity;
  std::optional<double> stability;
  std::optional<int> idle_consolidation_seconds;
  std::optional<std::string> model;
  std::optional<std::string> memory_prompt_prefix;
  std::optional<std::string> memory_prompt_suffix;
  std::optional<std::string> voice_backend;
  std::optional<bool> voice_reply_enabled;
  std::optional<bool> voice_retrieve_without_retain;
};

std::filesystem::path DefaultChatSettingsPath(
    const std::filesystem::path& db_path,
    const std::optional<std::filesystem::path>& repo_root) {
  namespace fs = std::filesystem;
  if (!db_path.empty() && db_path != fs::path(":memory:")) {
    fs::path settings_path = db_path;
    if (settings_path.has_extension()) {
      settings_path.replace_extension(".settings.json");
    } else {
      settings_path += ".settings.json";
    }
    return settings_path;
  }
  if (repo_root.has_value()) {
    return *repo_root / "examples/chat/chat_settings.json";
  }
  return "chat_settings.json";
}

std::optional<PersistedChatSettings> TryLoadPersistedChatSettings(
    const std::filesystem::path& settings_path,
    std::string* error_message = nullptr) {
  std::ifstream in(settings_path);
  if (!in) {
    return std::nullopt;
  }

  try {
    const nlohmann::json json = nlohmann::json::parse(in);
    PersistedChatSettings settings;
    if (json.contains("focus") && json["focus"].is_number()) {
      settings.focus = json["focus"].get<double>();
    }
    if (json.contains("sensitivity") && json["sensitivity"].is_number()) {
      settings.sensitivity = json["sensitivity"].get<double>();
    }
    if (json.contains("stability") && json["stability"].is_number()) {
      settings.stability = json["stability"].get<double>();
    }
    if (json.contains("idle_consolidation_seconds")
        && json["idle_consolidation_seconds"].is_number_integer()) {
      settings.idle_consolidation_seconds
          = std::max(0, json["idle_consolidation_seconds"].get<int>());
    }
    if (json.contains("model") && json["model"].is_string()) {
      const std::string model = json["model"].get<std::string>();
      if (!model.empty()) {
        settings.model = model;
      }
    }
    if (json.contains("memory_prompt_prefix")
        && json["memory_prompt_prefix"].is_string()) {
      settings.memory_prompt_prefix
          = json["memory_prompt_prefix"].get<std::string>();
    }
    if (json.contains("memory_prompt_suffix")
        && json["memory_prompt_suffix"].is_string()) {
      settings.memory_prompt_suffix
          = json["memory_prompt_suffix"].get<std::string>();
    }
    if (json.contains("voice_backend") && json["voice_backend"].is_string()) {
      const std::string backend = json["voice_backend"].get<std::string>();
      if (backend == "sherpa" || backend == "whisper") {
        settings.voice_backend = backend;
      }
    }
    if (json.contains("voice_reply_enabled")
        && json["voice_reply_enabled"].is_boolean()) {
      settings.voice_reply_enabled
          = json["voice_reply_enabled"].get<bool>();
    }
    if (json.contains("voice_retrieve_without_retain")
        && json["voice_retrieve_without_retain"].is_boolean()) {
      settings.voice_retrieve_without_retain
          = json["voice_retrieve_without_retain"].get<bool>();
    }
    return settings;
  } catch (const std::exception& ex) {
    if (error_message) {
      *error_message = ex.what();
    }
    return std::nullopt;
  }
}

void SavePersistedChatSettings(const std::filesystem::path& settings_path,
                               double focus,
                               double sensitivity,
                               double stability,
                               int idle_consolidation_seconds,
                               const std::string& model,
                               const std::string& memory_prompt_prefix,
                               const std::string& memory_prompt_suffix,
                               const std::string& voice_backend,
                               bool voice_reply_enabled,
                               bool voice_retrieve_without_retain) {
  nlohmann::json json = {
      {"focus", focus},
      {"sensitivity", sensitivity},
      {"stability", stability},
      {"idle_consolidation_seconds", std::max(0, idle_consolidation_seconds)},
      {"model", model},
      {"memory_prompt_prefix", memory_prompt_prefix},
      {"memory_prompt_suffix", memory_prompt_suffix},
      {"voice_backend", voice_backend},
      {"voice_reply_enabled", voice_reply_enabled},
      {"voice_retrieve_without_retain", voice_retrieve_without_retain},
  };

  std::error_code ec;
  const auto parent = settings_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
  }

  std::ofstream out(settings_path, std::ios::out | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("unable to open settings file: "
                             + settings_path.string());
  }
  out << json.dump(2) << '\n';
  if (!out) {
    throw std::runtime_error("failed to write settings file: "
                             + settings_path.string());
  }
}

struct FilterInjectedMemoriesResult {
  std::vector<cortext::Cortext::Context::Memory> injected;
  std::vector<cortext::Cortext::Context::Memory> dropped_empty_text;
  std::vector<cortext::Cortext::Context::Memory> dropped_working_memory_duplicate;
  std::vector<cortext::Cortext::Context::Memory> dropped_retrieved_duplicate;
};

FilterInjectedMemoriesResult FilterInjectedMemories(
    const std::vector<cortext::Cortext::Context::Memory>& retrieved_memories,
    const std::vector<cortext::Cortext::Context::Memory>& working_memory) {
  std::unordered_set<std::string> working_memory_keys;
  working_memory_keys.reserve(working_memory.size());
  std::unordered_set<std::string> injected_keys;
  injected_keys.reserve(retrieved_memories.size());
  FilterInjectedMemoriesResult result;
  result.injected.reserve(retrieved_memories.size());

  for (const auto& mem : working_memory) {
    const std::string text = ExtractTextFromBlobs(mem.content);
    if (text.empty()) {
      continue;
    }
    working_memory_keys.insert(mem.source_id + "\n" + text);
  }

  for (const auto& mem : retrieved_memories) {
    const std::string text = ExtractTextFromBlobs(mem.content);
    if (text.empty()) {
      result.dropped_empty_text.push_back(mem);
      continue;
    }
    const std::string key = mem.source_id + "\n" + text;
    if (working_memory_keys.find(key) != working_memory_keys.end()) {
      result.dropped_working_memory_duplicate.push_back(mem);
      continue;
    }
    if (injected_keys.find(key) != injected_keys.end()) {
      result.dropped_retrieved_duplicate.push_back(mem);
      continue;
    }
    injected_keys.insert(key);
    result.injected.push_back(mem);
  }
  return result;
}

std::string FormatLocalDateTime(
    const std::chrono::system_clock::time_point& time_point) {
  const std::time_t now_time = std::chrono::system_clock::to_time_t(time_point);
  std::tm local_tm{};
#if defined(_WIN32)
  localtime_s(&local_tm, &now_time);
#else
  localtime_r(&now_time, &local_tm);
#endif

  std::ostringstream oss;
  oss << std::put_time(&local_tm, "%A, %B %d, %Y at %I:%M:%S %p %Z");
  return oss.str();
}

struct SnapshotClock {
  std::string date;
  std::string time;
  std::string timezone;
};

SnapshotClock SnapshotClockNow() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#if defined(_WIN32)
  localtime_s(&local_tm, &now_time);
#else
  localtime_r(&now_time, &local_tm);
#endif

  SnapshotClock clock;
  {
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%A, %B %d, %Y");
    clock.date = oss.str();
  }
  {
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%H:%M:%S");
    clock.time = oss.str();
  }
  {
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Z");
    clock.timezone = oss.str();
  }
  return clock;
}

std::string FormatMemoryDateTime(std::uint64_t timestamp_ms) {
  const auto time_point = std::chrono::system_clock::time_point(
      std::chrono::milliseconds(timestamp_ms));
  return FormatLocalDateTime(time_point);
}

std::string BuildInjectedSystemPrompt(
    const std::vector<cortext::Cortext::Context::Memory>& injected_memories,
    const std::string& prefix,
    const std::string& suffix) {
  auto EscapeXml = [](const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
      switch (ch) {
        case '&': escaped += "&amp;"; break;
        case '<': escaped += "&lt;"; break;
        case '>': escaped += "&gt;"; break;
        case '"': escaped += "&quot;"; break;
        case '\'': escaped += "&apos;"; break;
        default: escaped.push_back(ch); break;
      }
    }
    return escaped;
  };

  const SnapshotClock clock = SnapshotClockNow();
  std::ostringstream oss;
  oss << "<snapshot>\n";
  oss << "<clock date=\"" << EscapeXml(clock.date)
      << "\" time=\"" << EscapeXml(clock.time)
      << "\" timezone=\"" << EscapeXml(clock.timezone) << "\"/>\n";
  if (!prefix.empty()) {
    oss << prefix << "\n\n";
  }
  oss << "<memories>\n";
  for (const auto& mem : injected_memories) {
    const std::string text = ExtractTextFromBlobs(mem.content);
    if (text.empty()) {
      continue;
    }
    oss << "  <memory source_id=\"" << EscapeXml(mem.source_id)
        << "\" datetime=\"" << EscapeXml(FormatMemoryDateTime(mem.timestamp))
        << "\" memory_id=\"" << mem.id << "\">"
        << EscapeXml(text) << "</memory>\n";
  }
  oss << "</memories>";
  if (!suffix.empty()) {
    oss << "\n\n" << suffix;
  }
  oss << "\n</snapshot>";
  return oss.str();
}

std::string DefaultMemoryPromptPrefix() {
  return "The XML snapshot below represents the current chat turn. Treat the "
         "<clock> element as the current local time for temporal references "
         "like now, today, tomorrow, and deadlines. The snapshot may contain "
         "retrieved memories from earlier interaction with this same user. "
         "Treat any memories in it as facts about the current user unless a "
         "memory clearly refers to someone else or quotes someone else. Use "
         "them as supporting context when they are relevant to the user's "
         "current message. Prefer these memories over guesses, but do not "
         "mention memory IDs, the XML format, or that you were given "
         "retrieved memories.";
}

std::string DefaultMemoryPromptSuffix() {
  return "";
}

std::string RoleFromSourceId(const std::string& source_id) {
  if (source_id.rfind("chat/user", 0) == 0) return "user";
  if (source_id.rfind("chat/assistant", 0) == 0) return "assistant";
  return {};
}

template <typename Json>
Json BuildOpenAIMessages(const std::vector<cortext::Cortext::Context::Memory>& working_memory,
                         const std::string& injected_system,
                         const std::string& latest_user_input) {
  Json messages = Json::array();
  if (!injected_system.empty()) {
    messages.push_back({{"role", "system"}, {"content", injected_system}});
  }

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
    const std::string role = RoleFromSourceId(m->source_id);
    if (role.empty()) {
      continue;
    }
    std::string text_content = ExtractTextFromBlobs(m->content);
    if (text_content.empty()) {
      continue;
    }
    if (role == "user" && !latest_user_input.empty()
        && text_content == latest_user_input) {
      continue;
    }
    messages.push_back({{"role", role}, {"content", text_content}});
  }
  if (!latest_user_input.empty()) {
    messages.push_back({{"role", "user"}, {"content", latest_user_input}});
  }
  return messages;
}

template <typename Json>
Json BuildFullHistoryMessages(const std::deque<chat::ChatMessage>& chat_history) {
  Json messages = Json::array();
  for (const auto& msg : chat_history) {
    if (msg.role.empty() || msg.content.empty()) {
      continue;
    }
    messages.push_back({{"role", msg.role}, {"content", msg.content}});
  }
  return messages;
}

std::string BuildSimpleRagSystemPrompt(
    const std::vector<cortext::Cortext::Context::Memory>& retrieved_memories,
    std::size_t max_memories = 5) {
  std::ostringstream oss;
  oss << "Relevant notes from prior conversation. Use them when helpful, but answer the current user directly.\n";
  std::size_t emitted = 0;
  for (const auto& mem : retrieved_memories) {
    if (emitted >= max_memories) {
      break;
    }
    const std::string text = ExtractTextFromBlobs(mem.content);
    if (text.empty()) {
      continue;
    }
    oss << "- " << text << "\n";
    ++emitted;
  }
  return oss.str();
}

template <typename Json>
Json BuildSimpleRagMessages(
    const std::vector<cortext::Cortext::Context::Memory>& retrieved_memories,
    const std::string& latest_user_input,
    std::size_t max_memories = 5) {
  Json messages = Json::array();
  const std::string system_prompt
      = BuildSimpleRagSystemPrompt(retrieved_memories, max_memories);
  if (!system_prompt.empty()) {
    messages.push_back({{"role", "system"}, {"content", system_prompt}});
  }
  if (!latest_user_input.empty()) {
    messages.push_back({{"role", "user"}, {"content", latest_user_input}});
  }
  return messages;
}

std::size_t CountOpenAIContentChars(const openai::Json& content) {
  if (content.is_string()) {
    return content.get_ref<const std::string&>().size();
  }
  if (!content.is_array()) {
    return 0;
  }

  std::size_t chars = 0;
  for (const auto& item : content) {
    if (item.is_string()) {
      chars += item.get_ref<const std::string&>().size();
      continue;
    }
    if (!item.is_object()) {
      continue;
    }
    if (item.contains("text") && item["text"].is_string()) {
      chars += item["text"].get_ref<const std::string&>().size();
    }
  }
  return chars;
}

std::size_t CountOpenAIMessageChars(const openai::Json& messages) {
  if (!messages.is_array()) {
    return 0;
  }

  std::size_t chars = 0;
  for (const auto& message : messages) {
    if (!message.is_object()) {
      continue;
    }
    if (message.contains("role") && message["role"].is_string()) {
      chars += message["role"].get_ref<const std::string&>().size();
    }
    if (message.contains("content")) {
      chars += CountOpenAIContentChars(message["content"]);
    }
  }
  return chars;
}

std::vector<chat::ProviderMessageSnapshot>
BuildProviderMessageSnapshot(const openai::Json& messages) {
  std::vector<chat::ProviderMessageSnapshot> out;
  if (!messages.is_array()) {
    return out;
  }
  out.reserve(messages.size());
  for (const auto& message : messages) {
    if (!message.is_object()) {
      continue;
    }
    chat::ProviderMessageSnapshot snapshot;
    if (message.contains("role") && message["role"].is_string()) {
      snapshot.role = message["role"].get<std::string>();
    }
    if (snapshot.role == "system") {
      continue;
    }
    if (message.contains("content")) {
      if (message["content"].is_string()) {
        snapshot.content = message["content"].get<std::string>();
      } else if (message["content"].is_array()) {
        for (const auto& item : message["content"]) {
          if (item.is_string()) {
            snapshot.content += item.get<std::string>();
          } else if (item.is_object()
                     && item.contains("text")
                     && item["text"].is_string()) {
            if (!snapshot.content.empty()) {
              snapshot.content += "\n";
            }
            snapshot.content += item["text"].get<std::string>();
          }
        }
      }
    }
    out.push_back(std::move(snapshot));
  }
  return out;
}

struct ChunkDiagnosticsCounters {
  std::size_t total_probes = 0;
  std::size_t interrupts = 0;
  std::size_t errors = 0;
};

ChunkDiagnosticsCounters SnapshotChunkDiagnostics(
    const chat::ChunkDiagnosticsState& diagnostics) {
  std::lock_guard<std::mutex> lock(diagnostics.mu);
  return {
      .total_probes = diagnostics.total_probes,
      .interrupts = diagnostics.interrupts,
      .errors = diagnostics.errors,
  };
}

void RecordResponseMetrics(chat::MetricsState& metrics,
                           chat::ResponseMetricsSample sample,
                           std::size_t max_samples) {
  std::lock_guard<std::mutex> lock(metrics.mu);
  metrics.latest_response = sample;
  metrics.response_history.push_back(std::move(sample));
  while (metrics.response_history.size() > max_samples) {
    metrics.response_history.pop_front();
  }
  metrics.total_responses++;
}

void RecordConsolidationMetrics(chat::MetricsState& metrics,
                                chat::ConsolidationMetricsSample sample,
                                std::size_t max_samples) {
  std::lock_guard<std::mutex> lock(metrics.mu);
  metrics.latest_consolidation = sample;
  metrics.consolidation_history.push_back(std::move(sample));
  while (metrics.consolidation_history.size() > max_samples) {
    metrics.consolidation_history.pop_front();
  }
  metrics.total_consolidations++;
}

} // namespace

int main(int argc, char** argv) {
  const bool has_models_env = HasEnv("CORTEXT_MODELS_DIR");
  const bool has_db_env = HasEnv("CORTEXT_CHAT_DB");
  const bool has_settings_env = HasEnv("CORTEXT_CHAT_SETTINGS");

  const std::string api_key = GetEnv("OPENAI_API_KEY");
  if (api_key.empty()) {
    std::cerr << "OPENAI_API_KEY is required\n";
    return 1;
  }

  const std::string openai_org = GetEnv("OPENAI_ORGANIZATION");
  const std::string openai_base_url = GetEnv("OPENAI_BASE_URL");

  const auto repo_root = FindRepoRootFromExe((argc > 0) ? argv[0] : nullptr);
  const std::string model = GetEnv("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17");
  std::filesystem::path db_path = GetEnv("CORTEXT_CHAT_DB", "examples/chat/chat_memory.db");
  std::filesystem::path models_dir = GetEnv("CORTEXT_MODELS_DIR", "models");
  std::filesystem::path settings_path = GetEnv("CORTEXT_CHAT_SETTINGS");

  if (!has_models_env && models_dir.is_relative() && repo_root.has_value()) {
    models_dir = *repo_root / models_dir;
  }
  if (!has_db_env && db_path.is_relative() && repo_root.has_value()) {
    db_path = *repo_root / db_path;
  }
  if (settings_path.empty()) {
    settings_path = DefaultChatSettingsPath(db_path, repo_root);
  } else if (!has_settings_env && settings_path.is_relative()
             && repo_root.has_value()) {
    settings_path = *repo_root / settings_path;
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
  auto metrics_state = std::make_shared<chat::MetricsState>();
  auto otel_state = std::make_shared<chat::OTelState>();
  auto retrieval_latency_state = std::make_shared<RetrievalLatencyState>();
  auto last_tokens_used = std::make_shared<std::atomic<std::int64_t>>(0);
  auto last_context = std::make_shared<chat::LastContext>();

  // OpenTelemetry setup
  // Local telemetry is always enabled. OTLP gRPC exporters are optional and
  // compile-time gated so the chat example can build without the gRPC stack.
  {
    namespace trace_sdk = opentelemetry::sdk::trace;
    namespace metrics_sdk = opentelemetry::sdk::metrics;
    namespace resource_sdk = opentelemetry::sdk::resource;
#if CORTEXT_CHAT_ENABLE_OTLP_GRPC
    namespace otlp = opentelemetry::exporter::otlp;
#endif
    auto resource = resource_sdk::Resource::Create({{"service.name", "cortext_chat"}});

    auto in_memory_span_exporter = std::unique_ptr<trace_sdk::SpanExporter>(
        new InMemorySpanExporter(retrieval_latency_state, otel_state));
    auto in_memory_processor = std::unique_ptr<trace_sdk::SpanProcessor>(new trace_sdk::SimpleSpanProcessor(std::move(in_memory_span_exporter)));
    auto* sdk_tracer_provider = new trace_sdk::TracerProvider(std::move(in_memory_processor), resource);

    // OTLP gRPC trace exporter (reads config from env vars)
#if CORTEXT_CHAT_ENABLE_OTLP_GRPC
    {
      otlp::OtlpGrpcExporterOptions trace_opts;
      auto otlp_exporter = std::unique_ptr<trace_sdk::SpanExporter>(new otlp::OtlpGrpcExporter(trace_opts));
      trace_sdk::BatchSpanProcessorOptions batch_opts;
      batch_opts.max_queue_size = 2048;
      batch_opts.schedule_delay_millis = std::chrono::milliseconds(5000);
      auto otlp_processor = std::unique_ptr<trace_sdk::SpanProcessor>(new trace_sdk::BatchSpanProcessor(std::move(otlp_exporter), batch_opts));
      sdk_tracer_provider->AddProcessor(std::move(otlp_processor));
    }
#endif

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
#if CORTEXT_CHAT_ENABLE_OTLP_GRPC
    {
      otlp::OtlpGrpcMetricExporterOptions metric_opts;
      auto otlp_metric_exporter = std::unique_ptr<metrics_sdk::PushMetricExporter>(new otlp::OtlpGrpcMetricExporter(metric_opts));
      metrics_sdk::PeriodicExportingMetricReaderOptions reader_options;
      reader_options.export_interval_millis = std::chrono::milliseconds(10000);
      reader_options.export_timeout_millis = std::chrono::milliseconds(5000);
      std::unique_ptr<metrics_sdk::MetricReader> reader(new metrics_sdk::PeriodicExportingMetricReader(std::move(otlp_metric_exporter), reader_options));
      static_cast<metrics_sdk::MeterProvider*>(raw_meter_provider)->AddMetricReader(std::move(reader));
    }
#endif

    opentelemetry::metrics::Provider::SetMeterProvider(meter_provider);
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("chat");
    static auto tokens_gauge = meter->CreateInt64ObservableGauge("chat.tokens_used", "Tokens used for last response", "1");
    auto observe_tokens = [](opentelemetry::metrics::ObserverResult result, void *state) noexcept {
      auto *tokens = static_cast<std::atomic<std::int64_t> *>(state);
      const std::int64_t v = tokens->load();
      opentelemetry::nostd::visit([&](auto &r) { if (r) { r->Observe(v); } }, result);
    };
    tokens_gauge->AddCallback(observe_tokens, last_tokens_used.get());

    // Log exporters: stdout (simple pretty format) plus optional OTLP gRPC
    {
      namespace logs_sdk = opentelemetry::sdk::logs;

      // Create simple stdout log exporter first (also writes to logs.txt in examples/chat/)
      std::string log_file_path = "examples/chat/logs.txt";
      if (repo_root.has_value()) {
        log_file_path = (*repo_root / "examples/chat/logs.txt").string();
      }
      auto stdout_exporter = std::unique_ptr<logs_sdk::LogRecordExporter>(
          new SimpleStdoutLogExporter(log_file_path, otel_state));
      auto stdout_processor = logs_sdk::SimpleLogRecordProcessorFactory::Create(std::move(stdout_exporter));
      auto logger_provider = logs_sdk::LoggerProviderFactory::Create(std::move(stdout_processor), resource);

      // Add OTLP gRPC log exporter as additional processor
#if CORTEXT_CHAT_ENABLE_OTLP_GRPC
      otlp::OtlpGrpcLogRecordExporterOptions log_opts;
      auto otlp_exporter = otlp::OtlpGrpcLogRecordExporterFactory::Create(log_opts);
      auto otlp_processor = logs_sdk::SimpleLogRecordProcessorFactory::Create(std::move(otlp_exporter));
      static_cast<logs_sdk::LoggerProvider*>(logger_provider.get())->AddProcessor(std::move(otlp_processor));
#endif

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
  PersistedChatSettings persisted_settings;
  std::string persisted_settings_error;
  if (auto loaded = TryLoadPersistedChatSettings(settings_path,
                                                 &persisted_settings_error)) {
    persisted_settings = *loaded;
  } else if (!persisted_settings_error.empty()) {
    std::cerr << "Failed to load chat settings from "
              << settings_path << ": " << persisted_settings_error << "\n";
  }
  auto settings_state = std::make_shared<chat::SettingsState>();
  auto voice_state = std::make_shared<chat::VoiceState>();
  auto db_explorer_state = std::make_shared<chat::DatabaseExplorerState>();
  {
    std::lock_guard<std::mutex> lock(settings_state->mu);
    settings_state->active_focus
        = persisted_settings.focus.value_or(focus);
    settings_state->active_sensitivity
        = persisted_settings.sensitivity.value_or(sensitivity);
    settings_state->active_stability
        = persisted_settings.stability.value_or(stability);
    settings_state->default_idle_consolidation_seconds = 0;
    settings_state->active_idle_consolidation_seconds
        = persisted_settings.idle_consolidation_seconds.value_or(0);
    settings_state->draft_focus = settings_state->active_focus;
    settings_state->draft_sensitivity = settings_state->active_sensitivity;
    settings_state->draft_stability = settings_state->active_stability;
    settings_state->draft_idle_consolidation_seconds
        = settings_state->active_idle_consolidation_seconds;
    settings_state->default_model = model;
    settings_state->active_model
        = persisted_settings.model.value_or(model);
    settings_state->draft_model = settings_state->active_model;
    settings_state->default_memory_prompt_prefix = DefaultMemoryPromptPrefix();
    settings_state->default_memory_prompt_suffix = DefaultMemoryPromptSuffix();
    settings_state->active_memory_prompt_prefix =
        persisted_settings.memory_prompt_prefix.value_or(
            settings_state->default_memory_prompt_prefix);
    settings_state->draft_memory_prompt_prefix
        = settings_state->active_memory_prompt_prefix;
    settings_state->active_memory_prompt_suffix =
        persisted_settings.memory_prompt_suffix.value_or(
            settings_state->default_memory_prompt_suffix);
    settings_state->draft_memory_prompt_suffix
        = settings_state->active_memory_prompt_suffix;
    if (!persisted_settings_error.empty()) {
      settings_state->last_apply_status
          = "Settings load failed; using defaults for this run.";
    }
  }
  {
    std::lock_guard<std::mutex> lock(voice_state->mu);
    voice_state->backend
        = persisted_settings.voice_backend.value_or("sherpa");
    voice_state->reply_enabled
        = persisted_settings.voice_reply_enabled.value_or(true);
    voice_state->retrieve_without_retain
        = persisted_settings.voice_retrieve_without_retain.value_or(false);
  }

  openai::start(api_key, openai_org);
  if (!openai_base_url.empty()) {
    openai::instance().setBaseUrl(EnsureTrailingSlash(openai_base_url));
  }

  double initial_focus = focus;
  double initial_sensitivity = sensitivity;
  double initial_stability = stability;
  {
    std::lock_guard<std::mutex> lock(settings_state->mu);
    initial_focus = settings_state->active_focus;
    initial_sensitivity = settings_state->active_sensitivity;
    initial_stability = settings_state->active_stability;
  }

  cortext::Cortext::Config cfg;
  cfg.focus = initial_focus;
  cfg.sensitivity = initial_sensitivity;
  cfg.stability = initial_stability;
  std::unique_ptr<cortext::Cortext> cortext_ctx;
  try {
    cortext_ctx = cortext::Cortext::Create(cfg, db_path.string(), models_dir.string());
  } catch (const std::exception& e) {
    std::cerr << "Failed to create cortext instance: " << e.what() << "\n";
    try {
      const auto resolved = cortext::internal::CreatePreferredTextEncoder(models_dir.string());
      std::cerr << "Resolved text encoder: " << resolved.backend_name
                << " (" << resolved.resolved_path.string() << ")\n";
    } catch (...) {
    }
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
  auto chunk_diagnostics_state = std::make_shared<chat::ChunkDiagnosticsState>();
  bool last_should_interrupt = false;
  std::optional<std::string> last_error;
  bool generating = false;
  std::string input;
  static constexpr size_t kMaxMemoryEvents = 50;
  static constexpr size_t kMaxChunkProbeEvents = 100;
  static constexpr size_t kMaxResponseMetricsSamples = 100;
  static constexpr size_t kMaxConsolidationMetricsSamples = 50;
  std::deque<chat::ChatMessage> chat_history;
  std::mutex pending_turns_mu;
  std::deque<PendingUserTurn> pending_user_turns;

  auto refresh_db_explorer = [&]() {
    try {
      std::lock_guard<std::mutex> lock(db_write_mu);
      RefreshDatabaseExplorer(*db_explorer_state, db_path.string());
    } catch (const std::exception& ex) {
      std::lock_guard<std::mutex> lock(db_explorer_state->mu);
      db_explorer_state->refresh_requested = false;
      db_explorer_state->last_refresh_status = std::string("DB refresh failed: ") + ex.what();
    }
  };
  refresh_db_explorer();

  StreamingState streaming_state;
  std::string partial_response;
  int generation_restarts = 0;
  IdleTracker idle_tracker;
  std::atomic<bool> voice_barge_in_requested{false};
  std::atomic<bool> typing_interrupt_requested{false};
  std::atomic<bool> input_activity_requested{false};
  std::atomic<bool> has_input_draft{false};
  std::atomic<bool> idle_consolidating{false};
  std::mutex idle_consolidation_stop_mu;
  std::optional<cortext::StopSource> idle_consolidation_stop_source;

  // Track streaming thread to ensure proper cleanup before exit
  std::thread streaming_thread;
  std::atomic<bool> streaming_thread_active{false};

  chat::StreamingChatClient streaming_client(api_key, openai_base_url.empty()
      ? "https://api.openai.com/v1/"
      : openai_base_url);
  std::unique_ptr<chat::VoiceSession> voice_session;
  std::string active_voice_backend;
  auto build_voice_session = [&](const std::string& backend) {
    chat::VoiceSessionConfig voice_config;
    const auto sherpa_dir = models_dir / "sherpa-onnx";
    const auto asr_dir = sherpa_dir / "sherpa-onnx-whisper-tiny.en";
    const auto tts_dir = sherpa_dir / "kitten-nano-en-v0_1-fp16";
    const auto whisper_dir = models_dir / "whisper.cpp";
    voice_config.backend = backend;
    voice_config.asr_encoder = asr_dir / "tiny.en-encoder.int8.onnx";
    voice_config.asr_decoder = asr_dir / "tiny.en-decoder.int8.onnx";
    voice_config.asr_joiner.clear();
    voice_config.asr_tokens = asr_dir / "tiny.en-tokens.txt";
    voice_config.whisper_model = whisper_dir / "ggml-small.en-tdrz.bin";
    voice_config.tts_model = tts_dir / "model.fp16.onnx";
    voice_config.tts_tokens = tts_dir / "tokens.txt";
    voice_config.tts_voices = tts_dir / "voices.bin";
    voice_config.tts_data_dir = tts_dir / "espeak-ng-data";
    voice_config.speaker_embedding_model
        = sherpa_dir / "nemo_en_titanet_small.onnx";
    voice_config.speaker_segmentation_model
        = sherpa_dir / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx";
    if (const char* speaker_model = std::getenv("CORTEXT_CHAT_SPEAKER_EMBEDDING_MODEL")) {
      if (*speaker_model != '\0') {
        voice_config.speaker_embedding_model = speaker_model;
      }
    }
    if (const char* segmentation_model = std::getenv("CORTEXT_CHAT_SPEAKER_SEGMENTATION_MODEL")) {
      if (*segmentation_model != '\0') {
        voice_config.speaker_segmentation_model = segmentation_model;
      }
    }
    if (const char* whisper_model = std::getenv("CORTEXT_CHAT_WHISPER_MODEL")) {
      if (*whisper_model != '\0') {
        voice_config.whisper_model = whisper_model;
      }
    }
    voice_config.on_partial_transcript = [voice_state](const std::string& partial) {
      std::lock_guard<std::mutex> lock(voice_state->mu);
      voice_state->live_transcript = partial;
    };
    voice_config.on_final_transcript = [voice_state, &pending_turns_mu,
                                        &pending_user_turns](const chat::VoiceFinalTranscript& transcript) {
      const std::string trimmed = TrimUserText(transcript.text);
      if (trimmed.empty()) {
        return;
      }
      bool reply_enabled = true;
      bool retain_input = true;
      std::string speaker_id = transcript.speaker_id.empty()
                                   ? std::string("speaker:unknown")
                                   : transcript.speaker_id;
      std::string source_id = "chat/user/" + speaker_id;
      {
        std::lock_guard<std::mutex> lock(voice_state->mu);
        reply_enabled = voice_state->reply_enabled;
        retain_input = !voice_state->retrieve_without_retain;
        if (voice_state->self_speaker_id.has_value()
            && *voice_state->self_speaker_id == speaker_id) {
          source_id = "chat/user/self/" + speaker_id;
        }
        voice_state->live_transcript.clear();
        voice_state->last_utterance = trimmed;
        voice_state->last_speaker_id = speaker_id;
        voice_state->surfaced_memories.clear();
        if (voice_state->last_segments.size() > 12) {
          voice_state->last_segments.erase(
              voice_state->last_segments.begin(),
              voice_state->last_segments.end() - 12);
        }
        UpdateVoiceSpeakerPreview(*voice_state, speaker_id, trimmed);
      }
      std::lock_guard<std::mutex> lock(pending_turns_mu);
      pending_user_turns.push_back(
          {trimmed, source_id, speaker_id, reply_enabled, retain_input, true});
    };
    voice_config.on_segment_debug = [voice_state](float start_s,
                                                  float end_s,
                                                  int diarizer_speaker,
                                                  const std::string& speaker_id,
                                                  const std::string& text) {
      std::lock_guard<std::mutex> lock(voice_state->mu);
      chat::VoiceSegmentDebug segment;
      segment.start_s = start_s;
      segment.end_s = end_s;
      segment.diarizer_speaker = diarizer_speaker;
      segment.speaker_id = speaker_id;
      segment.text = text;
      voice_state->last_segments.push_back(std::move(segment));
      while (voice_state->last_segments.size() > 24) {
        voice_state->last_segments.erase(voice_state->last_segments.begin());
      }
    };
    voice_config.on_user_speech_start = [&voice_barge_in_requested] {
      voice_barge_in_requested.store(true);
    };
    voice_config.on_error = [voice_state](const std::string& error) {
      std::lock_guard<std::mutex> lock(voice_state->mu);
      voice_state->last_error = error;
    };
    voice_config.on_listening_changed = [voice_state](bool listening) {
      std::lock_guard<std::mutex> lock(voice_state->mu);
      voice_state->listening = listening;
      if (!listening) {
        voice_state->live_transcript.clear();
      }
    };
    voice_config.on_playback_changed = [voice_state](bool speaking) {
      std::lock_guard<std::mutex> lock(voice_state->mu);
      voice_state->assistant_speaking = speaking;
    };
    auto session = std::make_unique<chat::VoiceSession>(std::move(voice_config));
    {
      std::lock_guard<std::mutex> lock(voice_state->mu);
      voice_state->backend = backend;
      voice_state->supported = session->IsSupported();
      voice_state->available = session->IsAvailable();
      voice_state->speaker_attribution_available
          = session->HasSpeakerAttribution();
    }
    return session;
  };
  {
    std::lock_guard<std::mutex> lock(voice_state->mu);
    active_voice_backend = voice_state->backend;
  }
  voice_session = build_voice_session(active_voice_backend);

  auto persist_runtime_settings = [&]() {
    double save_focus = 0.5;
    double save_sensitivity = 0.5;
    double save_stability = 0.5;
    int save_idle = 0;
    std::string save_model;
    std::string save_prefix;
    std::string save_suffix;
    std::string save_voice_backend = "sherpa";
    bool save_voice_reply_enabled = true;
    bool save_voice_retrieve_without_retain = false;
    {
      std::lock_guard<std::mutex> lock(settings_state->mu);
      save_focus = settings_state->active_focus;
      save_sensitivity = settings_state->active_sensitivity;
      save_stability = settings_state->active_stability;
      save_idle = settings_state->active_idle_consolidation_seconds;
      save_model = settings_state->active_model;
      save_prefix = settings_state->active_memory_prompt_prefix;
      save_suffix = settings_state->active_memory_prompt_suffix;
    }
    {
      std::lock_guard<std::mutex> lock(voice_state->mu);
      save_voice_backend = voice_state->backend;
      save_voice_reply_enabled = voice_state->reply_enabled;
      save_voice_retrieve_without_retain = voice_state->retrieve_without_retain;
    }
    SavePersistedChatSettings(settings_path,
                              save_focus,
                              save_sensitivity,
                              save_stability,
                              save_idle,
                              save_model,
                              save_prefix,
                              save_suffix,
                              save_voice_backend,
                              save_voice_reply_enabled,
                              save_voice_retrieve_without_retain);
  };

#if CORTEXT_CHAT_ENABLE_OTLP_GRPC
  const bool otlp_grpc_enabled = HasEnv("OTEL_EXPORTER_OTLP_ENDPOINT") || HasEnv("OTEL_EXPORTER_OTLP_HEADERS");
#else
  const bool otlp_grpc_enabled = false;
#endif

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
  window_state.chunk_diagnostics = chunk_diagnostics_state;
  window_state.metrics = metrics_state;
  window_state.status = status_state;
  window_state.settings = settings_state;
  window_state.voice = voice_state;
  window_state.db_explorer = db_explorer_state;
  window_state.otel = otel_state;
  window_state.input = &input;
  window_state.generating = &generating;
  window_state.partial_response = &partial_response;
  window_state.generation_restarts = &generation_restarts;
  window_state.last_should_interrupt = &last_should_interrupt;
  window_state.last_error = &last_error;
  window_state.typing_interrupt_requested = &typing_interrupt_requested;
  window_state.input_activity_requested = &input_activity_requested;
  window_state.has_input_draft = &has_input_draft;
  window_state.idle_consolidating = &idle_consolidating;
  window_state.otlp_enabled = otlp_grpc_enabled;

  chat::ChatWindow window(window_state);

  // Background refresh thread for idle consolidation
  std::atomic<bool> running{true};
  std::thread refresh_thread([&] {
    while (running) {
      std::this_thread::sleep_for(std::chrono::seconds(2));

      bool is_generating = false;
      bool has_draft = has_input_draft.load();
      bool is_consolidating = idle_consolidating.load();
      int consolidation_idle_seconds = IdleConsolidationSeconds(stability, 0);
      {
        std::lock_guard<std::mutex> lock(mu);
        is_generating = generating;
      }

      {
        std::lock_guard<std::mutex> settings_lock(settings_state->mu);
        consolidation_idle_seconds = IdleConsolidationSeconds(
            settings_state->active_stability,
            settings_state->active_idle_consolidation_seconds);
      }

      if (!is_generating && !has_draft && !is_consolidating
          && idle_tracker.ShouldConsolidate(consolidation_idle_seconds)) {
        idle_consolidating.store(true);
        const auto consolidation_started_at = std::chrono::steady_clock::now();
        cortext::StopSource stop_source;
        {
          std::lock_guard<std::mutex> stop_lock(idle_consolidation_stop_mu);
          idle_consolidation_stop_source = stop_source;
        }
        const cortext::StopToken stop_token = stop_source.get_token();
        try {
          cortext::Cortext::Context cons_ctx;
          {
            std::lock_guard<std::mutex> lock(db_write_mu);
            cons_ctx = cortext_ctx->Consolidate(
                stop_token, cortext::ConsolidationMode::Both);
          }
          chat::ConsolidationMetricsSample metrics_sample;
          metrics_sample.timestamp_ms = NowUnixMillis();
          metrics_sample.duration_ms
              = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - consolidation_started_at)
                    .count();
          if (stop_token.stop_requested()) {
            metrics_sample.cancelled = true;
            cortext::telemetry::LogInfo("chat.idle_consolidation.interrupted", {
              cortext::telemetry::Attribute::String("reason", "user_typing")
            });
          } else {
            {
              std::lock_guard<std::mutex> lock(mu);
              working_memory = cons_ctx.working_memory;
            }
            idle_tracker.MarkConsolidated();
            metrics_sample.completed = true;
            metrics_sample.working_memory_size
                = static_cast<int>(cons_ctx.working_memory.size());
            cortext::telemetry::LogInfo("chat.idle_consolidation.complete", {
              cortext::telemetry::Attribute::Int64("working_memory_size", static_cast<int64_t>(cons_ctx.working_memory.size()))
            });
            refresh_db_explorer();
          }
          RecordConsolidationMetrics(
              *metrics_state, std::move(metrics_sample),
              kMaxConsolidationMetricsSamples);
        } catch (const std::exception& ex) {
          chat::ConsolidationMetricsSample metrics_sample;
          metrics_sample.timestamp_ms = NowUnixMillis();
          metrics_sample.duration_ms
              = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - consolidation_started_at)
                    .count();
          if (stop_token.stop_requested() && IsCancelledConsolidationError(ex)) {
            metrics_sample.cancelled = true;
            metrics_sample.error_message = ex.what();
            cortext::telemetry::LogInfo("chat.idle_consolidation.interrupted", {
              cortext::telemetry::Attribute::String("reason", "user_typing"),
              cortext::telemetry::Attribute::String("error", ex.what())
            });
          } else {
            metrics_sample.had_error = true;
            metrics_sample.error_message = ex.what();
            std::lock_guard<std::mutex> lock(mu);
            last_error = std::string("idle consolidation: ") + ex.what();
            cortext::telemetry::LogError("chat.idle_consolidation.error", {
              cortext::telemetry::Attribute::String("error", ex.what())
            });
          }
          RecordConsolidationMetrics(
              *metrics_state, std::move(metrics_sample),
              kMaxConsolidationMetricsSamples);
        }
        {
          std::lock_guard<std::mutex> stop_lock(idle_consolidation_stop_mu);
          idle_consolidation_stop_source.reset();
        }
        idle_consolidating.store(false);
      }
    }
  });

  std::cout << "Cortext Chat started. Window opened." << std::endl;

  // Main render loop
  app.Run([&] {
    window.Render();

    {
      int idle_required_seconds = IdleConsolidationSeconds(stability, 0);
      bool is_generating = false;
      {
        std::lock_guard<std::mutex> settings_lock(settings_state->mu);
        idle_required_seconds = IdleConsolidationSeconds(
            settings_state->active_stability,
            settings_state->active_idle_consolidation_seconds);
      }
      {
        std::lock_guard<std::mutex> lock(mu);
        is_generating = generating;
      }
      const std::optional<int> idle_remaining
          = is_generating ? std::nullopt
                          : idle_tracker.RemainingSeconds(idle_required_seconds);
      if (status_state) {
        std::lock_guard<std::mutex> lock(status_state->mu);
        status_state->idle_pending = !is_generating && idle_remaining.has_value();
        status_state->idle_seconds_remaining = idle_remaining;
      }
    }

    if (input_activity_requested.exchange(false)) {
      idle_tracker.RecordActivity();
    }

    if (voice_barge_in_requested.exchange(false)) {
      bool is_generating = false;
      {
        std::lock_guard<std::mutex> lock(mu);
        is_generating = generating;
      }
      if (voice_session) {
        voice_session->CancelAssistantReply();
      }
      if (is_generating) {
        streaming_state.cancel_requested.store(true);
      }
      input_activity_requested.store(true);
    }

    if (typing_interrupt_requested.exchange(false)) {
      if (idle_consolidating.load()) {
        bool stop_requested = false;
        {
          std::lock_guard<std::mutex> stop_lock(idle_consolidation_stop_mu);
          if (idle_consolidation_stop_source.has_value()) {
            stop_requested = idle_consolidation_stop_source->request_stop();
          }
        }
        cortext::telemetry::LogInfo("chat.user_typing_interrupt", {
          cortext::telemetry::Attribute::Bool("idle_consolidating", true),
          cortext::telemetry::Attribute::Bool("stop_requested", stop_requested)
        });
      }
    }

    bool requested_db_refresh = false;
    bool requested_db_clear = false;
    {
      std::lock_guard<std::mutex> lock(db_explorer_state->mu);
      requested_db_refresh = db_explorer_state->refresh_requested && !generating;
      requested_db_clear = db_explorer_state->clear_requested && !generating;
    }
    if (requested_db_clear) {
      try {
        ResetChatDatabase(cfg, cortext_ctx, db_path, models_dir, *db_explorer_state, db_write_mu);
        {
          std::lock_guard<std::mutex> lock(mu);
          working_memory.clear();
          last_memories.clear();
          memory_events.clear();
          chat_history.clear();
          partial_response.clear();
          generation_restarts = 0;
          last_should_interrupt = false;
          last_error.reset();
        }
        {
          std::lock_guard<std::mutex> lock(last_context->mu);
          last_context->system_prompt.clear();
          last_context->has_data = false;
          last_context->scroll_position = 0.0f;
        }
        {
          std::lock_guard<std::mutex> lock(chunk_diagnostics_state->mu);
          chunk_diagnostics_state->active_probe.reset();
          chunk_diagnostics_state->recent_probes.clear();
          chunk_diagnostics_state->next_event_id = 1;
          chunk_diagnostics_state->total_probes = 0;
          chunk_diagnostics_state->interrupts = 0;
          chunk_diagnostics_state->suppressed = 0;
          chunk_diagnostics_state->errors = 0;
        }
        {
          std::lock_guard<std::mutex> lock(metrics_state->mu);
          metrics_state->response_history.clear();
          metrics_state->consolidation_history.clear();
          metrics_state->latest_response.reset();
          metrics_state->latest_consolidation.reset();
          metrics_state->total_responses = 0;
          metrics_state->total_consolidations = 0;
        }
        last_tokens_used->store(0);
        {
          std::lock_guard<std::mutex> lock(status_state->mu);
          status_state->tokens_used = 0;
        }
        {
          std::lock_guard<std::mutex> lock(voice_state->mu);
          voice_state->live_transcript.clear();
          voice_state->last_utterance.clear();
          voice_state->last_speaker_id.clear();
          voice_state->speakers.clear();
          voice_state->last_segments.clear();
          voice_state->surfaced_memories.clear();
          voice_state->last_error.reset();
        }
        {
          std::lock_guard<std::mutex> lock(db_explorer_state->mu);
          db_explorer_state->clear_requested = false;
          db_explorer_state->last_refresh_status = "Chat database cleared.";
        }
      } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> lock(db_explorer_state->mu);
        db_explorer_state->clear_requested = false;
        db_explorer_state->last_refresh_status = std::string("Clear failed: ") + ex.what();
      }
    }
    if (requested_db_refresh) {
      refresh_db_explorer();
    }

    bool requested_apply = false;
    double new_focus = 0.5;
    double new_sensitivity = 0.5;
    double new_stability = 0.5;
    int new_idle_consolidation_seconds = 0;
    std::string new_model;
    std::string new_memory_prompt_prefix;
    std::string new_memory_prompt_suffix;
    {
      std::lock_guard<std::mutex> lock(settings_state->mu);
      if (settings_state->apply_requested && !generating) {
        requested_apply = true;
        new_focus = settings_state->draft_focus;
        new_sensitivity = settings_state->draft_sensitivity;
        new_stability = settings_state->draft_stability;
        new_idle_consolidation_seconds
            = settings_state->draft_idle_consolidation_seconds;
        new_model = settings_state->draft_model;
        new_memory_prompt_prefix = settings_state->draft_memory_prompt_prefix;
        new_memory_prompt_suffix = settings_state->draft_memory_prompt_suffix;
        settings_state->apply_requested = false;
      }
    }
    if (requested_apply) {
      if (new_model.empty()) {
        std::lock_guard<std::mutex> lock(settings_state->mu);
        settings_state->last_apply_status = "Apply failed: model cannot be empty.";
      } else {
        try {
          cortext::Cortext::Config next_cfg;
          next_cfg.focus = new_focus;
          next_cfg.sensitivity = new_sensitivity;
          next_cfg.stability = new_stability;
          next_cfg.affect_interrupt = cfg.affect_interrupt;
          next_cfg.affect_retrieval = cfg.affect_retrieval;
          next_cfg.reinforcement_enabled = cfg.reinforcement_enabled;
          next_cfg.procedural_enabled = cfg.procedural_enabled;
          next_cfg.sequential_edges_enabled = cfg.sequential_edges_enabled;
          next_cfg.label_bank_path = cfg.label_bank_path;

          std::unique_ptr<cortext::Cortext> next_ctx;
          {
            std::lock_guard<std::mutex> lock(db_write_mu);
            if (cortext_ctx) {
              cortext_ctx->Flush();
            }
            next_ctx = cortext::Cortext::Create(next_cfg, db_path.string(), models_dir.string());
            cortext_ctx = std::move(next_ctx);
          }
          cfg = next_cfg;

          std::string apply_status =
              "Applied and saved: model=" + new_model
              + " F=" + std::to_string(new_focus)
              + " S=" + std::to_string(new_sensitivity)
              + " T=" + std::to_string(new_stability)
              + " idle="
              + (new_idle_consolidation_seconds > 0
                     ? std::to_string(new_idle_consolidation_seconds) + "s"
                     : std::string("auto"));
          {
            std::lock_guard<std::mutex> lock(settings_state->mu);
            settings_state->active_focus = new_focus;
            settings_state->active_sensitivity = new_sensitivity;
            settings_state->active_stability = new_stability;
            settings_state->active_idle_consolidation_seconds
                = new_idle_consolidation_seconds;
            settings_state->active_model = new_model;
            settings_state->active_memory_prompt_prefix = new_memory_prompt_prefix;
            settings_state->active_memory_prompt_suffix = new_memory_prompt_suffix;
            settings_state->last_apply_status = apply_status;
          }
          try {
            persist_runtime_settings();
          } catch (const std::exception& ex) {
            apply_status =
                "Applied for this session, but failed to save settings: "
                + std::string(ex.what());
            std::lock_guard<std::mutex> lock(settings_state->mu);
            settings_state->last_apply_status = apply_status;
          }
        } catch (const std::exception& ex) {
          std::lock_guard<std::mutex> lock(settings_state->mu);
          settings_state->last_apply_status = std::string("Apply failed: ") + ex.what();
        }
      }
    }

    bool voice_start_requested = false;
    bool voice_stop_requested = false;
    bool voice_reply_toggle_dirty = false;
    std::string requested_voice_backend;
    bool voice_was_listening = false;
    {
      std::lock_guard<std::mutex> lock(voice_state->mu);
      voice_start_requested = voice_state->start_requested;
      voice_stop_requested = voice_state->stop_requested;
      voice_reply_toggle_dirty = voice_state->reply_toggle_dirty;
      requested_voice_backend = voice_state->backend;
      voice_was_listening = voice_state->listening;
      voice_state->start_requested = false;
      voice_state->stop_requested = false;
      voice_state->reply_toggle_dirty = false;
    }
    if (requested_voice_backend != active_voice_backend) {
      if (voice_session) {
        voice_session->Stop();
      }
      voice_session = build_voice_session(requested_voice_backend);
      active_voice_backend = requested_voice_backend;
      if (voice_was_listening && voice_session) {
        const bool started = voice_session->Start();
        std::lock_guard<std::mutex> lock(voice_state->mu);
        voice_state->available = voice_session->IsAvailable();
        voice_state->speaker_attribution_available
            = voice_session->HasSpeakerAttribution();
        if (!started && !voice_state->last_error.has_value()) {
          voice_state->last_error = "Failed to start voice capture.";
        }
      }
    }
    if (voice_start_requested && voice_session) {
      const bool started = voice_session->Start();
      std::lock_guard<std::mutex> lock(voice_state->mu);
      voice_state->available = voice_session->IsAvailable();
      voice_state->speaker_attribution_available
          = voice_session->HasSpeakerAttribution();
      if (!started && !voice_state->last_error.has_value()) {
        voice_state->last_error = "Failed to start voice capture.";
      }
    }
    if (voice_stop_requested && voice_session) {
      voice_session->Stop();
    }
    if (voice_reply_toggle_dirty) {
      try {
        persist_runtime_settings();
      } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> lock(voice_state->mu);
        voice_state->last_error
            = std::string("Failed to save voice preference: ") + ex.what();
      }
    }

    if (window.HasPendingMessage()) {
      const std::string text = TrimUserText(window.TakePendingMessage());
      if (!text.empty()) {
        std::lock_guard<std::mutex> lock(pending_turns_mu);
        pending_user_turns.push_back({text, "chat/user", {}, true, true, false});
      }
    }

    std::optional<PendingUserTurn> next_turn;
    {
      bool is_generating = false;
      {
        std::lock_guard<std::mutex> lock(mu);
        is_generating = generating;
      }
      if (!is_generating) {
        std::lock_guard<std::mutex> lock(pending_turns_mu);
        if (!pending_user_turns.empty()) {
          next_turn = std::move(pending_user_turns.front());
          pending_user_turns.pop_front();
        }
      }
    }

    if (next_turn.has_value()) {
      std::string text = next_turn->text;
      const std::string source_id = next_turn->source_id.empty()
                                        ? std::string("chat/user")
                                        : next_turn->source_id;
      const bool retain_input = next_turn->retain_input;
      if (next_turn->request_reply) {
        {
          std::lock_guard<std::mutex> lock(mu);
          generating = true;
          last_error.reset();
          partial_response.clear();
          generation_restarts = 0;
          streaming_state.Reset();
          chat_history.push_back({"user", text});
        }

        idle_tracker.RecordActivity();

        cortext::telemetry::LogInfo("chat.user_message", {
          cortext::telemetry::Attribute::Int64("text_length", static_cast<int64_t>(text.size()))
        });

        if (streaming_thread.joinable()) {
          streaming_thread.join();
        }
        if (next_turn->voice_origin && voice_session) {
          voice_session->CancelAssistantReply();
        }

        streaming_thread_active.store(true);
        streaming_thread = std::thread([&, text, source_id,
                                        speak_reply = next_turn->voice_origin && voice_session
                                                      && voice_session->IsAvailable()] {
          const auto chunk_counters_before
              = SnapshotChunkDiagnostics(*chunk_diagnostics_state);
          double focus_value = 0.5;
          double sensitivity_value = 0.5;
          std::string active_model;
          {
            std::lock_guard<std::mutex> lock(settings_state->mu);
            focus_value = settings_state->active_focus;
            sensitivity_value = settings_state->active_sensitivity;
            active_model = settings_state->active_model;
          }

          // Phase 1: Process user input through Cortext
          cortext::telemetry::LogDebug("chat.phase1.start", {
            cortext::telemetry::Attribute::String("source", source_id),
            cortext::telemetry::Attribute::Int64("text_length", static_cast<int64_t>(text.size()))
          });

          cortext::Cortext::Context retrieved;
          double phase1_total_ms = 0.0;
          double retrieval_latency_ms = 0.0;
          double processing_latency_ms = 0.0;
          try {
            std::lock_guard<std::mutex> lock(db_write_mu);
            if (retain_input) {
              retrieved = cortext_ctx->ProcessText(text, source_id);
            } else {
              cortext::internal::StreamingTextProbeSession probe_session(
                  *cortext_ctx, source_id);
              retrieved = probe_session.FinalizeText(text);
            }
            phase1_total_ms = retrieved.total_ms;
            processing_latency_ms = retrieved.process_ms;
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
            const auto filtered_memories = FilterInjectedMemories(
                retrieved.retrieved_memory, retrieved.working_memory);
            streaming_state.current_memories = filtered_memories.injected;
            for (const auto& mem : retrieved.retrieved_memory) {
              PushMemoryEvent(
                  memory_events,
                  CreateMemoryEvent(chat::MemoryEventType::RETRIEVED_RAW, mem),
                  kMaxMemoryEvents);
            }
            for (const auto& mem : streaming_state.current_memories) {
              PushMemoryEvent(
                  memory_events,
                  CreateMemoryEvent(chat::MemoryEventType::INJECTED, mem),
                  kMaxMemoryEvents);
            }
            for (const auto& mem : filtered_memories.dropped_empty_text) {
              PushMemoryEvent(memory_events,
                              CreateDroppedMemoryEvent(mem, "empty_text"),
                              kMaxMemoryEvents);
            }
            for (const auto& mem : filtered_memories.dropped_working_memory_duplicate) {
              PushMemoryEvent(memory_events,
                              CreateDroppedMemoryEvent(mem, "duplicate_of_working_memory"),
                              kMaxMemoryEvents);
            }
            for (const auto& mem : filtered_memories.dropped_retrieved_duplicate) {
              PushMemoryEvent(memory_events,
                              CreateDroppedMemoryEvent(mem, "duplicate_of_retrieved"),
                              kMaxMemoryEvents);
            }
            if (retrieved.output.stored_embedding_id.has_value()) {
              PushMemoryEvent(memory_events,
                              CreateStoredEvent(*retrieved.output.stored_embedding_id,
                                                text,
                                                source_id,
                                                NowUnixMillis(),
                                                retrieved.output),
                              kMaxMemoryEvents);
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
          if (retrieval_latency_state->retrieval_latency_ms.has_value()) {
            retrieval_latency_ms = *retrieval_latency_state->retrieval_latency_ms;
          }
          {
            std::lock_guard<std::mutex> lock(status_state->mu);
            if (status_state->processing_latency_ms.has_value()) {
              processing_latency_ms = *status_state->processing_latency_ms;
            }
          }
          refresh_db_explorer();

          // Phase 2: Streaming generation with interrupt checking
          cortext::telemetry::LogDebug("chat.phase2.start", {
            cortext::telemetry::Attribute::Int64("memories_injected", static_cast<int64_t>(streaming_state.current_memories.size())),
            cortext::telemetry::Attribute::String("model", active_model)
          });

          std::string assistant_reply;
          int local_restarts = 0;
          chat::UsageAccumulator usage_accumulator;
          bool had_stream_error = false;
          std::string stream_error_message;
          bool cancelled_by_voice = false;
          std::string tts_sentence_buffer;
          std::int64_t current_turn_cortext_prompt_tokens = 0;
          std::int64_t current_turn_rag_prompt_tokens = 0;
          std::int64_t current_turn_full_history_prompt_tokens = 0;
          std::size_t current_turn_cortext_prompt_chars = 0;
          std::size_t current_turn_rag_prompt_chars = 0;
          std::size_t current_turn_full_history_prompt_chars = 0;
          std::unique_ptr<cortext::internal::StreamingTextProbeSession> final_probe_session;
          const auto response_started_at = std::chrono::steady_clock::now();

          while (local_restarts <= StreamingState::kMaxRestarts) {
            final_probe_session = std::make_unique<cortext::internal::StreamingTextProbeSession>(
                *cortext_ctx, "chat/assistant");
            std::string probe_buffer;
            std::string memory_prompt_prefix;
            std::string memory_prompt_suffix;
            std::deque<chat::ChatMessage> chat_history_snapshot;
            {
              std::lock_guard<std::mutex> lock(settings_state->mu);
              memory_prompt_prefix = settings_state->active_memory_prompt_prefix;
              memory_prompt_suffix = settings_state->active_memory_prompt_suffix;
            }
            {
              std::lock_guard<std::mutex> lock(mu);
              chat_history_snapshot = chat_history;
            }
            const std::string injected_system
                = BuildInjectedSystemPrompt(streaming_state.current_memories,
                                            memory_prompt_prefix,
                                            memory_prompt_suffix);

            using Json = openai::Json;
            Json messages = BuildOpenAIMessages<Json>(retrieved.working_memory, injected_system, text);
            Json full_history_messages = BuildFullHistoryMessages<Json>(chat_history_snapshot);
            Json rag_messages = BuildSimpleRagMessages<Json>(streaming_state.current_memories, text);
            const std::size_t prompt_chars = CountOpenAIMessageChars(messages);
            const std::size_t full_history_prompt_chars
                = CountOpenAIMessageChars(full_history_messages);
            const std::size_t rag_prompt_chars = CountOpenAIMessageChars(rag_messages);
            current_turn_cortext_prompt_chars = prompt_chars;
            current_turn_rag_prompt_chars = rag_prompt_chars;
            current_turn_full_history_prompt_chars = full_history_prompt_chars;
            current_turn_cortext_prompt_tokens
                = chat::EstimateTokenCountFromChars(prompt_chars);
            current_turn_rag_prompt_tokens
                = chat::EstimateTokenCountFromChars(rag_prompt_chars);
            current_turn_full_history_prompt_tokens
                = chat::EstimateTokenCountFromChars(full_history_prompt_chars);

            {
              std::lock_guard<std::mutex> lock(last_context->mu);
              last_context->system_prompt = injected_system;
              last_context->provider_messages
                  = BuildProviderMessageSnapshot(messages);
              last_context->prompt_costs.cortext_prompt_chars = prompt_chars;
              last_context->prompt_costs.rag_prompt_chars = rag_prompt_chars;
              last_context->prompt_costs.full_history_prompt_chars
                  = full_history_prompt_chars;
              last_context->prompt_costs.cortext_prompt_tokens
                  = chat::EstimateTokenCountFromChars(prompt_chars);
              last_context->prompt_costs.rag_prompt_tokens
                  = chat::EstimateTokenCountFromChars(rag_prompt_chars);
              last_context->prompt_costs.full_history_prompt_tokens
                  = chat::EstimateTokenCountFromChars(full_history_prompt_chars);
              last_context->prompt_costs.rag_memory_count
                  = streaming_state.current_memories.size();
              last_context->prompt_costs.working_memory_message_count
                  = retrieved.working_memory.size();
              last_context->prompt_costs.full_history_message_count
                  = chat_history_snapshot.size();
              last_context->has_data = true;
            }

            {
              std::lock_guard<std::mutex> lock(mu);
              streaming_state.accumulated_tokens.clear();
              streaming_state.cancel_requested.store(false);
              partial_response.clear();
            }

            chat::StreamingRequest request;
            request.model = active_model;
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
              if (speak_reply && voice_session) {
                tts_sentence_buffer += token;
                for (const auto& sentence
                     : DrainCompletedSentences(&tts_sentence_buffer, false)) {
                  voice_session->QueueAssistantText(sentence);
                }
              }
              probe_buffer += token;

              cortext::telemetry::LogDebug("chat.phase2.token", {
                cortext::telemetry::Attribute::Int64("accumulated_length", static_cast<int64_t>(accumulated.size())),
                cortext::telemetry::Attribute::String("accumulated", accumulated.size() <= 100 ? accumulated : accumulated.substr(0, 100) + "...")
              });

              if (!stream_interrupts) {
                return;
              }
              if (!ShouldRunStreamingProbe(probe_buffer)) {
                return;
              }

              const std::string probe_chunk = std::move(probe_buffer);
              probe_buffer.clear();
              auto probe_event = BeginChunkProbeEvent(
                  *chunk_diagnostics_state, probe_chunk, local_restarts, accumulated.size());
              PublishActiveChunkProbe(*chunk_diagnostics_state, probe_event);

              try {
                cortext::Cortext::Context ctx
                    = final_probe_session->AppendTextChunk(probe_chunk);

                const double boundary_thresh =
                    cortext::core::BoundaryThreshold(focus_value, sensitivity_value);
                const bool boundary_score_pass
                    = ctx.boundary_score.has_value()
                      && *ctx.boundary_score >= boundary_thresh;

                int new_memory_count = 0;
                bool interrupt_ignored_restart_cap = false;
                const bool restart_budget_available
                    = local_restarts < StreamingState::kMaxRestarts;
                if (ctx.at_boundary && boundary_score_pass
                    && ctx.should_interrupt) {
                  {
                    const auto filtered_retrieved = FilterInjectedMemories(
                        ctx.retrieved_memory, working_memory);
                    if (restart_budget_available) {
                      std::lock_guard<std::mutex> lock(mu);
                      std::unordered_set<long long> existing_ids;
                      for (const auto& m : streaming_state.current_memories) {
                        existing_ids.insert(m.id);
                      }
                      for (const auto& m : ctx.retrieved_memory) {
                        PushMemoryEvent(
                            memory_events,
                            CreateMemoryEvent(chat::MemoryEventType::RETRIEVED_RAW, m),
                            kMaxMemoryEvents);
                      }
                      for (const auto& m : filtered_retrieved.dropped_empty_text) {
                        PushMemoryEvent(memory_events,
                                        CreateDroppedMemoryEvent(m, "empty_text"),
                                        kMaxMemoryEvents);
                      }
                      for (const auto& m : filtered_retrieved.dropped_working_memory_duplicate) {
                        PushMemoryEvent(memory_events,
                                        CreateDroppedMemoryEvent(m, "duplicate_of_working_memory"),
                                        kMaxMemoryEvents);
                      }
                      for (const auto& m : filtered_retrieved.dropped_retrieved_duplicate) {
                        PushMemoryEvent(memory_events,
                                        CreateDroppedMemoryEvent(m, "duplicate_of_retrieved"),
                                        kMaxMemoryEvents);
                      }
                      for (const auto& m : filtered_retrieved.injected) {
                        if (existing_ids.find(m.id) == existing_ids.end()) {
                          streaming_state.current_memories.push_back(m);
                          PushMemoryEvent(
                              memory_events,
                              CreateMemoryEvent(chat::MemoryEventType::INJECTED, m),
                              kMaxMemoryEvents);
                          ++new_memory_count;
                        }
                      }
                      last_should_interrupt = (new_memory_count > 0);
                    } else {
                      new_memory_count += filtered_retrieved.injected.size();
                      interrupt_ignored_restart_cap = (new_memory_count > 0);
                      std::lock_guard<std::mutex> lock(mu);
                      last_should_interrupt = false;
                    }
                  }
                  if (new_memory_count > 0 && restart_budget_available) {
                    cortext::telemetry::LogInfo("chat.phase2.interrupt", {
                      cortext::telemetry::Attribute::Int64("new_memories", static_cast<int64_t>(new_memory_count)),
                      cortext::telemetry::Attribute::Int64("restart", static_cast<int64_t>(local_restarts)),
                      cortext::telemetry::Attribute::Int64("tokens_so_far", static_cast<int64_t>(accumulated.size()))
                    });
                    streaming_state.cancel_requested.store(true);
                    interrupted = true;
                  } else if (interrupt_ignored_restart_cap) {
                    cortext::telemetry::LogInfo("chat.phase2.interrupt_ignored_restart_cap", {
                      cortext::telemetry::Attribute::Int64("new_memories", static_cast<int64_t>(new_memory_count)),
                      cortext::telemetry::Attribute::Int64("restart", static_cast<int64_t>(local_restarts)),
                      cortext::telemetry::Attribute::Int64("restart_cap", static_cast<int64_t>(StreamingState::kMaxRestarts)),
                      cortext::telemetry::Attribute::Int64("tokens_so_far", static_cast<int64_t>(accumulated.size()))
                    });
                  } else {
                    cortext::telemetry::LogDebug("chat.phase2.interrupt_suppressed", {
                      cortext::telemetry::Attribute::Int64("raw_retrieved", static_cast<int64_t>(ctx.retrieved_memory.size())),
                      cortext::telemetry::Attribute::Int64("restart", static_cast<int64_t>(local_restarts)),
                      cortext::telemetry::Attribute::Int64("tokens_so_far", static_cast<int64_t>(accumulated.size()))
                    });
                  }
                }

                FinalizeChunkProbeSuccess(
                    probe_event, ctx, boundary_thresh, boundary_score_pass,
                    new_memory_count, interrupt_ignored_restart_cap);
                CommitChunkProbeEvent(
                    *chunk_diagnostics_state, std::move(probe_event), kMaxChunkProbeEvents);
              } catch (const std::exception& ex) {
                FinalizeChunkProbeError(probe_event, ex.what());
                CommitChunkProbeEvent(
                    *chunk_diagnostics_state, std::move(probe_event), kMaxChunkProbeEvents);
                std::lock_guard<std::mutex> lock(mu);
                last_error = std::string("cortext streaming: ") + ex.what();
              }
            });

            chat::AccumulateUsage(
                usage_accumulator, result.usage, prompt_chars,
                result.full_content.size());

            if (!result.was_cancelled && !probe_buffer.empty()) {
              try {
                (void)final_probe_session->CacheTextChunk(probe_buffer);
                probe_buffer.clear();
              } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lock(mu);
                last_error = std::string("cortext streaming tail: ") + ex.what();
              }
            }

            if (result.was_cancelled && interrupted && local_restarts < StreamingState::kMaxRestarts) {
              if (speak_reply && voice_session) {
                voice_session->CancelAssistantReply();
                tts_sentence_buffer.clear();
              }
              local_restarts++;
              {
                std::lock_guard<std::mutex> lock(mu);
                generation_restarts = local_restarts;
              }
              continue;
            }
            if (result.was_cancelled && !interrupted) {
              cancelled_by_voice = true;
              assistant_reply.clear();
              if (speak_reply && voice_session) {
                voice_session->CancelAssistantReply();
              }
              break;
            }

            if (result.error.has_value()) {
              had_stream_error = true;
              stream_error_message = *result.error;
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
          if (speak_reply && voice_session) {
            if (cancelled_by_voice || had_stream_error) {
              voice_session->CancelAssistantReply();
            } else {
              for (const auto& sentence
                   : DrainCompletedSentences(&tts_sentence_buffer, true)) {
                voice_session->QueueAssistantText(sentence);
              }
            }
          }

          double phase3_total_ms = 0.0;
          bool phase3_failed = false;
          std::string phase3_error_message;
          if (!assistant_reply.empty() && !cancelled_by_voice) {
            cortext::telemetry::LogDebug("chat.phase3.start", {
              cortext::telemetry::Attribute::Int64("reply_length", static_cast<int64_t>(assistant_reply.size())),
              cortext::telemetry::Attribute::String("source", "chat/assistant"),
              cortext::telemetry::Attribute::String("reply_content", assistant_reply.size() <= 200 ? assistant_reply : assistant_reply.substr(0, 200) + "...")
            });
            try {
              cortext::Cortext::Context asst_ctx;
              {
                std::lock_guard<std::mutex> lock(db_write_mu);
                if (final_probe_session) {
                  asst_ctx = final_probe_session->FinalizeText(assistant_reply);
                } else {
                asst_ctx = cortext_ctx->ProcessText(assistant_reply, "chat/assistant");
              }
              phase3_total_ms = asst_ctx.total_ms;
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
            phase3_failed = true;
            phase3_error_message = ex.what();
            cortext::telemetry::LogError("chat.phase3.error", {
              cortext::telemetry::Attribute::String("error", ex.what())
            });
            std::lock_guard<std::mutex> lock(mu);
            last_error = std::string("cortext assistant: ") + ex.what();
          }
          } else if (cancelled_by_voice) {
            cortext::telemetry::LogInfo("chat.voice_barge_in_cancelled_reply");
          }

          const auto chunk_counters_after
              = SnapshotChunkDiagnostics(*chunk_diagnostics_state);
          chat::ResponseMetricsSample response_sample;
          response_sample.timestamp_ms = NowUnixMillis();
          response_sample.usage.prompt_tokens = usage_accumulator.prompt_tokens;
          response_sample.usage.completion_tokens = usage_accumulator.completion_tokens;
          response_sample.usage.total_tokens = usage_accumulator.total_tokens;
          response_sample.usage_accuracy
              = chat::GetUsageAccuracy(usage_accumulator);
          response_sample.cortext_prompt_tokens = current_turn_cortext_prompt_tokens;
          response_sample.rag_prompt_tokens = current_turn_rag_prompt_tokens;
          response_sample.full_history_prompt_tokens
              = current_turn_full_history_prompt_tokens;
          response_sample.cortext_prompt_chars = current_turn_cortext_prompt_chars;
          response_sample.rag_prompt_chars = current_turn_rag_prompt_chars;
          response_sample.full_history_prompt_chars
              = current_turn_full_history_prompt_chars;
          response_sample.response_wall_ms
              = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - response_started_at)
                    .count();
          response_sample.phase1_total_ms = phase1_total_ms;
          response_sample.phase3_total_ms = phase3_total_ms;
          response_sample.retrieval_latency_ms = retrieval_latency_ms;
          response_sample.processing_latency_ms = processing_latency_ms;
          response_sample.restart_count = local_restarts;
          response_sample.probe_count = static_cast<int>(
              chunk_counters_after.total_probes - chunk_counters_before.total_probes);
          response_sample.interrupt_count = static_cast<int>(
              chunk_counters_after.interrupts - chunk_counters_before.interrupts);
          response_sample.interrupt_rate = chat::ComputeInterruptRate(
              response_sample.probe_count, response_sample.interrupt_count);
          response_sample.had_stream_error = had_stream_error || phase3_failed;
          response_sample.error_message = stream_error_message;
          if (phase3_failed) {
            if (!response_sample.error_message.empty()) {
              response_sample.error_message += " | ";
            }
            response_sample.error_message += "phase3: " + phase3_error_message;
          }
          RecordResponseMetrics(
              *metrics_state, response_sample, kMaxResponseMetricsSamples);
          last_tokens_used->store(response_sample.usage.total_tokens);
          {
            std::lock_guard<std::mutex> lock(status_state->mu);
            status_state->tokens_used = response_sample.usage.total_tokens;
          }
          refresh_db_explorer();

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
      } else {
        {
          std::lock_guard<std::mutex> lock(mu);
          last_error.reset();
          partial_response.clear();
          generation_restarts = 0;
          chat_history.push_back({"user", text});
        }
        idle_tracker.RecordActivity();
        cortext::telemetry::LogInfo("chat.voice_ingress_only", {
          cortext::telemetry::Attribute::Int64("text_length",
                                               static_cast<int64_t>(text.size()))
        });
        try {
          cortext::Cortext::Context retrieved;
          {
            std::lock_guard<std::mutex> lock(db_write_mu);
            if (retain_input) {
              retrieved = cortext_ctx->ProcessText(text, source_id);
            } else {
              cortext::internal::StreamingTextProbeSession probe_session(
                  *cortext_ctx, source_id);
              retrieved = probe_session.FinalizeText(text);
            }
          }
          {
            std::lock_guard<std::mutex> lock(mu);
            working_memory = retrieved.working_memory;
            last_should_interrupt = retrieved.should_interrupt;
            const auto filtered_memories = FilterInjectedMemories(
                retrieved.retrieved_memory, retrieved.working_memory);
            streaming_state.current_memories = filtered_memories.injected;
            for (const auto& mem : retrieved.retrieved_memory) {
              PushMemoryEvent(
                  memory_events,
                  CreateMemoryEvent(chat::MemoryEventType::RETRIEVED_RAW, mem),
                  kMaxMemoryEvents);
            }
            for (const auto& mem : streaming_state.current_memories) {
              PushMemoryEvent(
                  memory_events,
                  CreateMemoryEvent(chat::MemoryEventType::INJECTED, mem),
                  kMaxMemoryEvents);
            }
            for (const auto& mem : filtered_memories.dropped_empty_text) {
              PushMemoryEvent(memory_events,
                              CreateDroppedMemoryEvent(mem, "empty_text"),
                              kMaxMemoryEvents);
            }
            for (const auto& mem : filtered_memories.dropped_working_memory_duplicate) {
              PushMemoryEvent(memory_events,
                              CreateDroppedMemoryEvent(mem, "duplicate_of_working_memory"),
                              kMaxMemoryEvents);
            }
            for (const auto& mem : filtered_memories.dropped_retrieved_duplicate) {
              PushMemoryEvent(memory_events,
                              CreateDroppedMemoryEvent(mem, "duplicate_of_retrieved"),
                              kMaxMemoryEvents);
            }
            if (retrieved.output.stored_embedding_id.has_value()) {
              PushMemoryEvent(memory_events,
                              CreateStoredEvent(*retrieved.output.stored_embedding_id,
                                                text,
                                                source_id,
                                                NowUnixMillis(),
                                                retrieved.output),
                              kMaxMemoryEvents);
            }
          }
          {
            std::vector<std::string> surfaced_memories;
            surfaced_memories.reserve(
                std::min<std::size_t>(streaming_state.current_memories.size(), 5));
            std::unordered_set<std::string> seen_previews;
            for (const auto& mem : streaming_state.current_memories) {
              std::string preview = CreateVoiceSurfacedMemoryPreview(mem);
              if (preview.empty()) {
                continue;
              }
              if (!seen_previews.insert(preview).second) {
                continue;
              }
              surfaced_memories.push_back(std::move(preview));
              if (surfaced_memories.size() >= 5) {
                break;
              }
            }
            std::lock_guard<std::mutex> lock(voice_state->mu);
            voice_state->surfaced_memories = std::move(surfaced_memories);
          }
          refresh_db_explorer();
        } catch (const std::exception& ex) {
          std::lock_guard<std::mutex> lock(mu);
          last_error = std::string("voice ingress: ") + ex.what();
        }
      }
    }
  });

  running = false;

  // Cancel any ongoing streaming if active
  if (streaming_thread_active.load()) {
    streaming_state.cancel_requested.store(true);
  }
  if (voice_session) {
    voice_session->CancelAssistantReply();
  }

  // Wait for threads to complete before cleanup
  if (streaming_thread.joinable()) {
    streaming_thread.join();
  }
  if (voice_session) {
    voice_session->Stop();
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
