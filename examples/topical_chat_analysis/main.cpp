#include <cortext/cortext.hpp>
#include <cortext/telemetry/telemetry.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include "encoder/text_encoder_factory.hpp"

#include <nlohmann/json.hpp>

#ifndef CORTEXT_DISABLE_OPENTELEMETRY
#  include <opentelemetry/logs/provider.h>
#  include <opentelemetry/nostd/shared_ptr.h>
#  include <opentelemetry/sdk/logs/exporter.h>
#  include <opentelemetry/sdk/logs/logger_provider.h>
#  include <opentelemetry/sdk/logs/logger_provider_factory.h>
#  include <opentelemetry/sdk/logs/read_write_log_record.h>
#  include <opentelemetry/sdk/logs/simple_log_record_processor_factory.h>
#  include <opentelemetry/sdk/resource/resource.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cortext/operations/metrics.hpp>

namespace {

struct ConsolidationTelemetry {
};

#ifndef CORTEXT_DISABLE_OPENTELEMETRY
class SimpleStdoutLogExporter final
    : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  SimpleStdoutLogExporter(const std::string &log_file_path,
                          std::vector<std::string> filters,
                          ConsolidationTelemetry *cons_metrics)
      : log_file_path_(log_file_path),
        filters_(std::move(filters)),
        cons_metrics_(cons_metrics) {
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
      const opentelemetry::nostd::span<std::unique_ptr<opentelemetry::sdk::logs::Recordable>>
          &records) noexcept override {
    for (const auto &record : records) {
      auto *log_record =
          static_cast<opentelemetry::sdk::logs::ReadWriteLogRecord *>(record.get());
      if (!log_record) {
        continue;
      }

      std::string body;
      const auto &body_val = log_record->GetBody();
      if (opentelemetry::nostd::holds_alternative<std::string>(body_val)) {
        body = opentelemetry::nostd::get<std::string>(body_val);
      }

      auto get_attr_i64 = [&](const std::string &key) -> std::optional<int64_t> {
        for (const auto &kv : log_record->GetAttributes()) {
          if (kv.first != key) {
            continue;
          }
          std::optional<int64_t> out;
          opentelemetry::nostd::visit(
              [&out](auto &&v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, int64_t>) {
                  out = static_cast<int64_t>(v);
                } else if constexpr (std::is_same_v<T, int32_t>) {
                  out = static_cast<int64_t>(v);
                } else if constexpr (std::is_same_v<T, uint64_t>) {
                  out = static_cast<int64_t>(v);
                } else if constexpr (std::is_same_v<T, uint32_t>) {
                  out = static_cast<int64_t>(v);
                } else if constexpr (std::is_same_v<T, double>) {
                  out = static_cast<int64_t>(v);
                } else if constexpr (std::is_same_v<T, float>) {
                  out = static_cast<int64_t>(v);
                }
              },
              kv.second);
          return out;
        }
        return std::nullopt;
      };

      (void)get_attr_i64;

      std::ostringstream attrs;
      bool first = true;
      for (const auto &kv : log_record->GetAttributes()) {
        if (!first) {
          attrs << ", ";
        }
        first = false;
        attrs << kv.first << "=";
        opentelemetry::nostd::visit(
            [&attrs](auto &&v) {
              using T = std::decay_t<decltype(v)>;
              if constexpr (std::is_same_v<T, bool>) {
                attrs << (v ? "true" : "false");
              } else if constexpr (std::is_same_v<T, int32_t>) {
                attrs << v;
              } else if constexpr (std::is_same_v<T, int64_t>) {
                attrs << v;
              } else if constexpr (std::is_same_v<T, uint32_t>) {
                attrs << v;
              } else if constexpr (std::is_same_v<T, uint64_t>) {
                attrs << v;
              } else if constexpr (std::is_same_v<T, double>) {
                attrs << v;
              } else if constexpr (std::is_same_v<T, std::string>) {
                attrs << v;
              } else {
                attrs << "?";
              }
            },
            kv.second);
      }

      if (!filters_.empty()) {
        bool allowed = false;
        for (const auto &prefix : filters_) {
          if (!prefix.empty() && body.rfind(prefix, 0) == 0) {
            allowed = true;
            break;
          }
        }
        if (!allowed) {
          continue;
        }
      }

      std::ostringstream line;
      line << "[LOG] " << body;
      if (!first) {
        line << " {" << attrs.str() << "}";
      }
      line << "\n";

      std::cout << line.str();
      std::cout.flush();

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
  std::vector<std::string> filters_;
  ConsolidationTelemetry *cons_metrics_ = nullptr;
};

void InstallOtelLogger(const std::string &log_path,
                       std::vector<std::string> filters,
                       ConsolidationTelemetry *cons_metrics) {
  namespace logs_sdk = opentelemetry::sdk::logs;
  namespace resource_sdk = opentelemetry::sdk::resource;
  auto resource = resource_sdk::Resource::Create({{"service.name", "cortext_topical_chat_analysis"}});
  auto exporter = std::unique_ptr<logs_sdk::LogRecordExporter>(
      new SimpleStdoutLogExporter(log_path, std::move(filters), cons_metrics));
  auto processor =
      logs_sdk::SimpleLogRecordProcessorFactory::Create(std::move(exporter));
  auto provider = opentelemetry::nostd::shared_ptr<opentelemetry::logs::LoggerProvider>(
      new logs_sdk::LoggerProvider(std::move(processor), resource));
  opentelemetry::logs::Provider::SetLoggerProvider(provider);
}
#else
void InstallOtelLogger(const std::string &log_path,
                       std::vector<std::string> filters,
                       ConsolidationTelemetry *cons_metrics) {
  (void)log_path;
  (void)filters;
  (void)cons_metrics;
}
#endif

struct AnalysisConfig {
  std::string dataset_path;
  std::string models_dir = "models";
  std::string db_path = ":memory:";
  std::string otel_log_path;
  std::vector<std::string> otel_filters = {
      "cortext.boundary",
      "cortext.write_gate",
      "cortext.memory_storage",
      "cortext.streaming_pacing",
      "cortext.graph_retrieval",
      "cortext.interrupt_gate",
      "cortext.working_memory",
  };
  int max_conversations = 3;
  int max_turns_per_conversation = 120;
  int max_total_turns = 240;
  bool reset_per_conversation = true;
  int interleave = 1;
  bool cadence_enabled = true;
  double cadence_speed = 12.0; // >1.0 speeds up; 12x keeps runs reasonable
  double cadence_wpm = 150.0;
  double cadence_jitter = 0.15;
  int cadence_min_ms = 80;
  int cadence_max_ms = 1200;
  int context_window_turns = 4;
  bool run_consolidation = false;
  int consolidation_cycles = 1;
  int consolidation_every_turns = 0;
  bool consolidate_during = false;
  bool consolidate_idle = false;
  bool semantic_overlap = false;
  unsigned int seed = 1337;
  bool deterministic = false;
  std::uint64_t synthetic_start_ms = 1700000000000ULL;
  double focus = 0.3;
  double sensitivity = 0.6;
  double stability = 0.5;
  std::string affect_mode = "all";
  bool affect_interrupt = true;
  bool affect_retrieval = true;
  bool reinforcement_enabled = true;
  bool procedural_enabled = true;
  bool sequential_edges_enabled = true;
};

struct RetrievalExample {
  double overlap = 0.0;
  std::string turn_text;
  std::string memory_text;
};

struct Turn {
  std::string agent;
  std::string message;
};

struct Conversation {
  std::string id;
  std::vector<Turn> turns;
};

struct DistSummary {
  std::size_t count = 0;
  double mean = 0.0;
  double p10 = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
};

struct Stats {
  int conversations = 0;
  int turns = 0;
  int writes = 0;
  int consolidation_runs = 0;
  int consolidation_failures = 0;
  long long consolidation_association_created = 0;
  long long consolidation_label_created = 0;
  long long consolidation_baseline_association = 0;
  long long consolidation_baseline_label = 0;
  long long consolidation_final_association = 0;
  long long consolidation_final_label = 0;
  long long reinforcement_edge_count = 0;
  double reinforcement_weight_mean = 0.0;
  long long memory_long_term_count = 0;
  double memory_strength_mean = 0.0;
  double memory_used_count_mean = 0.0;
  double memory_retrieved_count_mean = 0.0;
  int retrieval_turns = 0;
  int total_retrieved = 0;
  int retrieval_association_candidates = 0;
  int retrieval_label_candidates = 0;
  int retrieval_other_candidates = 0;
  int retrieval_turns_with_association = 0;
  int retrieval_turns_with_label = 0;
  int retrieval_summary_hit_turns = 0;
  int retrieval_summary_only_turns = 0;
  int retrieval_semantic_positive_turns = 0;
  int interrupt_semantic_true_positive = 0;
  int interrupt_semantic_false_positive = 0;
  int interrupt_semantic_false_negative = 0;
  int interrupt_gate_fail_no_candidates = 0;
  int interrupt_gate_fail_no_store = 0;
  int interrupt_gate_fail_rel = 0;
  int interrupt_gate_fail_novelty = 0;
  int interrupt_gate_fail_mu = 0;
  int interrupt_gate_fail_novelty_mu = 0;
  int interrupt_gate_fail_dup = 0;
  int interrupt_gate_fail_boundary_mu = 0;
  int boundary_at_count = 0;
  int boundary_score_pass_count = 0;
  int interrupt_turns = 0;
  int interrupt_turns_with_retrieval = 0;
  int interrupt_association_candidates = 0;
  int interrupt_label_candidates = 0;
  int interrupt_other_candidates = 0;
  int interrupt_turns_with_association = 0;
  int interrupt_turns_with_label = 0;
  int interrupt_abort_count = 0;
  int wm_samples = 0;
  int wm_total_slots = 0;
  int wm_max_slots = 0;
  int signals_since_write = 0;
  std::vector<int> signals_per_write;
  double retrieval_overlap_sum = 0.0;
  int retrieval_overlap_count = 0;
  double wm_overlap_sum = 0.0;
  int wm_overlap_count = 0;
  std::vector<double> novelty_values;
  std::vector<double> relevance_values;
  std::vector<double> surprise_values;
  std::vector<double> retrieval_overlap_values;
  std::vector<double> retrieval_context_overlap_values;
  std::vector<double> retrieval_semantic_overlap_values;
  std::vector<double> retrieval_context_semantic_overlap_values;
  std::vector<double> retrieval_summary_hit_overlap_values;
  std::vector<double> non_interrupt_semantic_overlap_values;
  std::vector<double> non_interrupt_context_semantic_overlap_values;
  std::vector<double> interrupt_overlap_values;
  std::vector<double> interrupt_context_overlap_values;
  std::vector<double> interrupt_semantic_overlap_values;
  std::vector<double> interrupt_context_semantic_overlap_values;
  std::vector<double> interrupt_novelty_values;
  std::vector<double> interrupt_relevance_values;
  std::vector<double> interrupt_surprise_values;
  std::vector<double> emotion_intensity_values;
  std::vector<double> arousal_values;
  std::vector<double> valence_values;
  std::vector<double> salience_values;
  std::vector<double> affect_drive_values;
  std::vector<double> interrupt_affect_drive_values;
  std::vector<double> non_interrupt_affect_drive_values;
  std::vector<double> interrupt_gate_affect_drive_values;
  std::vector<double> interrupt_gate_affect_drive_interrupt_values;
  std::vector<double> interrupt_gate_affect_drive_non_interrupt_values;
  std::vector<double> interrupt_gate_retrieval_thresh_values;
  std::vector<double> interrupt_gate_boundary_mult_eff_values;
  std::vector<double> boundary_score_values;
  std::vector<double> retrieval_emotion_bonus_values;
  std::vector<double> interrupt_retrieval_emotion_bonus_values;
  std::vector<double> encode_ms_values;
  std::vector<double> process_ms_values;
  std::vector<double> hydrate_ms_values;
  std::vector<double> total_ms_values;
  std::unordered_map<std::string, double> operation_time_sum;
  std::unordered_map<std::string, int> operation_time_count;
  std::vector<RetrievalExample> low_overlap_examples;
};

std::unordered_set<std::string> Tokenize(const std::string &text) {
  std::unordered_set<std::string> tokens;
  std::string current;
  for (unsigned char c : text) {
    if (std::isalnum(c)) {
      current.push_back(static_cast<char>(std::tolower(c)));
    } else if (!current.empty()) {
      tokens.insert(current);
      current.clear();
    }
  }
  if (!current.empty()) {
    tokens.insert(current);
  }
  return tokens;
}

std::unordered_set<std::string> MergeTokenWindow(
    const std::deque<std::unordered_set<std::string>> &window) {
  std::unordered_set<std::string> merged;
  for (const auto &tokens : window) {
    merged.insert(tokens.begin(), tokens.end());
  }
  return merged;
}

double Jaccard(const std::unordered_set<std::string> &a,
               const std::unordered_set<std::string> &b) {
  if (a.empty() || b.empty()) {
    return 0.0;
  }
  const auto *small = &a;
  const auto *large = &b;
  if (a.size() > b.size()) {
    small = &b;
    large = &a;
  }
  std::size_t intersect = 0;
  for (const auto &t : *small) {
    if (large->count(t)) {
      ++intersect;
    }
  }
  const std::size_t uni = a.size() + b.size() - intersect;
  if (uni == 0) {
    return 0.0;
  }
  return static_cast<double>(intersect) / static_cast<double>(uni);
}

std::optional<std::vector<float>> DecodeFloatBlobDynamic(const std::any &v) {
  const unsigned char *data = nullptr;
  size_t len = 0;
  if (!cortext::core::AnyToBytes(v, data, len) || len == 0) {
    return std::nullopt;
  }
  if (len % sizeof(float) != 0) {
    return std::nullopt;
  }
  const size_t dim = len / sizeof(float);
  std::vector<float> out(dim);
  std::memcpy(out.data(), data, len);
  return out;
}

double CosineSimilarity(const std::vector<float> &a,
                        const std::vector<float> &b) {
  if (a.empty() || b.empty() || a.size() != b.size()) {
    return 0.0;
  }
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double av = static_cast<double>(a[i]);
    const double bv = static_cast<double>(b[i]);
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  if (na <= 0.0 || nb <= 0.0) {
    return 0.0;
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

std::vector<float> MeanEmbedding(
    const std::deque<std::vector<float>> &window) {
  if (window.empty()) {
    return {};
  }
  size_t dim = window.front().size();
  if (dim == 0) {
    return {};
  }
  std::vector<float> mean(dim, 0.0f);
  size_t count = 0;
  for (const auto &emb : window) {
    if (emb.size() != dim) {
      continue;
    }
    for (size_t i = 0; i < dim; ++i) {
      mean[i] += emb[i];
    }
    count++;
  }
  if (count == 0) {
    return {};
  }
  const float inv = 1.0f / static_cast<float>(count);
  for (size_t i = 0; i < dim; ++i) {
    mean[i] *= inv;
  }
  return mean;
}

double Clamp01(double v) {
  return std::clamp(v, 0.0, 1.0);
}

double Quantile(const std::vector<double> &values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  const double pos = q * static_cast<double>(values.size() - 1);
  const std::size_t idx = static_cast<std::size_t>(std::floor(pos));
  const std::size_t idx2 = std::min(values.size() - 1, idx + 1);
  const double frac = pos - static_cast<double>(idx);
  return values[idx] * (1.0 - frac) + values[idx2] * frac;
}

DistSummary Summarize(std::vector<double> values) {
  DistSummary summary;
  if (values.empty()) {
    return summary;
  }
  summary.count = values.size();
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  summary.mean = sum / static_cast<double>(values.size());
  std::sort(values.begin(), values.end());
  summary.p10 = Quantile(values, 0.10);
  summary.p50 = Quantile(values, 0.50);
  summary.p90 = Quantile(values, 0.90);
  return summary;
}

std::optional<double> GetMetric(const cortext::Cortext::Context &ctx,
                                cortext::operations::Metric metric) {
  auto it = ctx.output.metrics.find(static_cast<int>(metric));
  if (it == ctx.output.metrics.end()) {
    return std::nullopt;
  }
  return it->second;
}

int CountWords(const std::string &text) {
  int count = 0;
  bool in_word = false;
  for (unsigned char c : text) {
    if (std::isalnum(c)) {
      if (!in_word) {
        count++;
        in_word = true;
      }
    } else {
      in_word = false;
    }
  }
  return count;
}

bool IsSentenceBoundaryToken(const std::string &token) {
  if (token.empty()) {
    return false;
  }
  const char c = token.back();
  return c == '.' || c == '!' || c == '?' || c == ':';
}

std::vector<std::string> SplitWords(const std::string &text) {
  std::vector<std::string> words;
  std::string current;
  for (unsigned char c : text) {
    if (std::isspace(c)) {
      if (!current.empty()) {
        words.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(static_cast<char>(c));
    }
  }
  if (!current.empty()) {
    words.push_back(current);
  }
  return words;
}

std::vector<std::string> ChunkMessage(const std::string &message,
                                      int target_words) {
  std::vector<std::string> chunks;
  if (target_words < 1) {
    target_words = 1;
  }
  const auto words = SplitWords(message);
  if (words.empty()) {
    return chunks;
  }
  std::vector<std::string> current;
  current.reserve(static_cast<std::size_t>(target_words));
  int count = 0;
  for (const auto &word : words) {
    current.push_back(word);
    count++;
    if (count >= target_words || IsSentenceBoundaryToken(word)) {
      std::ostringstream oss;
      for (std::size_t i = 0; i < current.size(); ++i) {
        if (i > 0) {
          oss << ' ';
        }
        oss << current[i];
      }
      chunks.push_back(oss.str());
      current.clear();
      count = 0;
    }
  }
  if (!current.empty()) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < current.size(); ++i) {
      if (i > 0) {
        oss << ' ';
      }
      oss << current[i];
    }
    chunks.push_back(oss.str());
  }
  return chunks;
}

std::string BytesToString(const std::vector<unsigned char> &bytes) {
  return std::string(bytes.begin(), bytes.end());
}

std::string MemoryToText(const cortext::Cortext::Context::Memory &mem) {
  std::ostringstream out;
  bool first = true;
  for (const auto &blob : mem.content) {
    if (!first) {
      out << "\n";
    }
    first = false;
    out << BytesToString(blob);
  }
  return out.str();
}

std::optional<std::string> GetMemoryKind(
    const std::shared_ptr<cortext::Store> &store,
    std::unordered_map<long long, std::string> &cache,
    long long memory_id) {
  if (!store || memory_id <= 0) {
    return std::nullopt;
  }
  auto it = cache.find(memory_id);
  if (it != cache.end()) {
    return it->second;
  }
  auto rows = store->Execute(
      "SELECT kind FROM memories WHERE memory_id = ?",
      {memory_id});
  if (rows.empty()) {
    return std::nullopt;
  }
  auto it_kind = rows[0].find("kind");
  if (it_kind == rows[0].end()) {
    return std::nullopt;
  }
  if (it_kind->second.type() != typeid(std::string)) {
    return std::nullopt;
  }
  std::string kind = std::any_cast<std::string>(it_kind->second);
  cache[memory_id] = kind;
  return kind;
}

struct MemoryEmotion {
  double intensity = 0.0;
  double arousal_avg = 0.0;
};

std::optional<MemoryEmotion> GetMemoryEmotion(
    const std::shared_ptr<cortext::Store> &store,
    std::unordered_map<long long, MemoryEmotion> &cache,
    long long memory_id) {
  if (!store || memory_id <= 0) {
    return std::nullopt;
  }
  auto it = cache.find(memory_id);
  if (it != cache.end()) {
    return it->second;
  }
  auto rows = store->Execute(
      "SELECT emotional_intensity, s_arousal_avg FROM memories WHERE memory_id = ?",
      {memory_id});
  if (rows.empty()) {
    return std::nullopt;
  }
  auto get_dbl = [&rows](const char *k, double def) -> double {
    auto itv = rows[0].find(k);
    if (itv == rows[0].end()) {
      return def;
    }
    return cortext::store::AnyToDouble(itv->second, def);
  };
  MemoryEmotion emotion;
  emotion.intensity = cortext::core::Clamp(get_dbl("emotional_intensity", 0.0), 0.0, 1.0);
  emotion.arousal_avg = cortext::core::Clamp(get_dbl("s_arousal_avg", 0.0), 0.0, 1.0);
  cache[memory_id] = emotion;
  return emotion;
}

long long CountMemoriesByKind(const std::shared_ptr<cortext::Store> &store,
                              const std::string &kind) {
  if (!store) {
    return 0;
  }
  auto rows = store->Execute(
      "SELECT COUNT(*) AS c FROM memories WHERE kind = ?",
      {kind});
  if (rows.empty()) {
    return 0;
  }
  auto it = rows[0].find("c");
  if (it == rows[0].end()) {
    return 0;
  }
  if (it->second.type() == typeid(long long)) {
    return std::any_cast<long long>(it->second);
  }
  if (it->second.type() == typeid(int)) {
    return static_cast<long long>(std::any_cast<int>(it->second));
  }
  if (it->second.type() == typeid(double)) {
    return static_cast<long long>(std::any_cast<double>(it->second));
  }
  return 0;
}

struct AssociationStats {
  long long count = 0;
  double weight_mean = 0.0;
};

AssociationStats GetAssociationStats(const std::shared_ptr<cortext::Store> &store,
                                     const std::string &edge_type) {
  AssociationStats stats;
  if (!store) {
    return stats;
  }
  auto rows = store->Execute(
      "SELECT COUNT(*) AS c, AVG(weight) AS avg_w FROM associations "
      "WHERE edge_type = ?",
      {edge_type});
  if (rows.empty()) {
    return stats;
  }
  auto get_dbl = [&rows](const char *k, double def) -> double {
    auto it = rows[0].find(k);
    if (it == rows[0].end()) {
      return def;
    }
    return cortext::store::AnyToDouble(it->second, def);
  };
  auto it = rows[0].find("c");
  if (it != rows[0].end()) {
    if (it->second.type() == typeid(long long)) {
      stats.count = std::any_cast<long long>(it->second);
    } else if (it->second.type() == typeid(int)) {
      stats.count = static_cast<long long>(std::any_cast<int>(it->second));
    } else if (it->second.type() == typeid(double)) {
      stats.count = static_cast<long long>(std::any_cast<double>(it->second));
    }
  }
  stats.weight_mean = get_dbl("avg_w", 0.0);
  return stats;
}

struct MemoryStats {
  long long count = 0;
  double strength_mean = 0.0;
  double used_count_mean = 0.0;
  double retrieved_count_mean = 0.0;
};

MemoryStats GetMemoryStats(const std::shared_ptr<cortext::Store> &store,
                           const std::string &kind) {
  MemoryStats stats;
  if (!store) {
    return stats;
  }
  auto rows = store->Execute(
      "SELECT COUNT(*) AS c, "
      "AVG(strength) AS strength_mean, "
      "AVG(used_count) AS used_mean, "
      "AVG(retrieved_count) AS retrieved_mean "
      "FROM memories WHERE kind = ?",
      {kind});
  if (rows.empty()) {
    return stats;
  }
  auto get_dbl = [&rows](const char *k, double def) -> double {
    auto it = rows[0].find(k);
    if (it == rows[0].end()) {
      return def;
    }
    return cortext::store::AnyToDouble(it->second, def);
  };
  auto it = rows[0].find("c");
  if (it != rows[0].end()) {
    if (it->second.type() == typeid(long long)) {
      stats.count = std::any_cast<long long>(it->second);
    } else if (it->second.type() == typeid(int)) {
      stats.count = static_cast<long long>(std::any_cast<int>(it->second));
    } else if (it->second.type() == typeid(double)) {
      stats.count = static_cast<long long>(std::any_cast<double>(it->second));
    }
  }
  stats.strength_mean = get_dbl("strength_mean", 0.0);
  stats.used_count_mean = get_dbl("used_mean", 0.0);
  stats.retrieved_count_mean = get_dbl("retrieved_mean", 0.0);
  return stats;
}

void RecordRetrievalExample(Stats &stats, double overlap, const std::string &turn_text,
                            const std::string &memory_text) {
  RetrievalExample ex{overlap, turn_text, memory_text};
  stats.low_overlap_examples.push_back(std::move(ex));
  std::sort(stats.low_overlap_examples.begin(), stats.low_overlap_examples.end(),
            [](const RetrievalExample &a, const RetrievalExample &b) {
              return a.overlap < b.overlap;
            });
  if (stats.low_overlap_examples.size() > 5) {
    stats.low_overlap_examples.resize(5);
  }
}

double ComputeAffectDrive(double sensitivity, double emotion_intensity,
                          double arousal, double salience) {
  const double s_affect = cortext::core::AffectSensitivityBias(sensitivity);
  const double w_arousal_raw = cortext::core::Lerp(0.30, 0.55, s_affect);
  const double w_emotion_raw = cortext::core::Lerp(0.30, 0.55, s_affect);
  const double w_salience_raw = cortext::core::Lerp(0.10, 0.20, s_affect);
  const double w_sum = std::max(1e-9, w_arousal_raw + w_emotion_raw + w_salience_raw);
  const double w_arousal = w_arousal_raw / w_sum;
  const double w_emotion = w_emotion_raw / w_sum;
  const double w_salience = w_salience_raw / w_sum;
  const double affect_gain = cortext::core::AffectGain(sensitivity);
  const double drive = affect_gain
                       * (w_arousal * arousal + w_emotion * emotion_intensity
                          + w_salience * salience);
  return cortext::core::Clamp(drive, 0.0, 1.0);
}

AnalysisConfig ParseArgs(int argc, char **argv) {
  AnalysisConfig cfg;
  const std::string default_path = "data/topical_chat/test_freq.jsonl";
  cfg.dataset_path = default_path;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto take = [&arg](const std::string &prefix) -> std::optional<std::string> {
      if (arg.rfind(prefix, 0) == 0) {
        return arg.substr(prefix.size());
      }
      return std::nullopt;
    };
    if (auto v = take("--data=")) {
      cfg.dataset_path = *v;
    } else if (auto v = take("--models=")) {
      cfg.models_dir = *v;
    } else if (auto v = take("--db=")) {
      cfg.db_path = *v;
    } else if (auto v = take("--max-conversations=")) {
      cfg.max_conversations = std::stoi(*v);
    } else if (auto v = take("--max-turns=")) {
      cfg.max_turns_per_conversation = std::stoi(*v);
    } else if (auto v = take("--max-total=")) {
      cfg.max_total_turns = std::stoi(*v);
    } else if (auto v = take("--otel-log=")) {
      cfg.otel_log_path = *v;
    } else if (auto v = take("--otel-filter=")) {
      cfg.otel_filters.clear();
      if (*v != "all") {
        std::string tmp = *v;
        std::string token;
        std::istringstream iss(tmp);
        while (std::getline(iss, token, ',')) {
          if (!token.empty()) {
            cfg.otel_filters.push_back(token);
          }
        }
      }
    } else if (auto v = take("--focus=")) {
      cfg.focus = std::stod(*v);
    } else if (auto v = take("--sensitivity=")) {
      cfg.sensitivity = std::stod(*v);
    } else if (auto v = take("--stability=")) {
      cfg.stability = std::stod(*v);
    } else if (auto v = take("--affect-mode=")) {
      cfg.affect_mode = *v;
      if (*v == "all") {
        cfg.affect_interrupt = true;
        cfg.affect_retrieval = true;
      } else if (*v == "interrupt") {
        cfg.affect_interrupt = true;
        cfg.affect_retrieval = false;
      } else if (*v == "retrieval") {
        cfg.affect_interrupt = false;
        cfg.affect_retrieval = true;
      } else if (*v == "off") {
        cfg.affect_interrupt = false;
        cfg.affect_retrieval = false;
      } else {
        std::cerr << "Unknown --affect-mode=" << *v
                  << "; defaulting to 'all'.\n";
        cfg.affect_mode = "all";
        cfg.affect_interrupt = true;
        cfg.affect_retrieval = true;
      }
    } else if (auto v = take("--reinforcement-mode=")) {
      if (*v == "on") {
        cfg.reinforcement_enabled = true;
      } else if (*v == "off") {
        cfg.reinforcement_enabled = false;
      } else {
        std::cerr << "Unknown --reinforcement-mode=" << *v
                  << "; defaulting to 'on'.\n";
        cfg.reinforcement_enabled = true;
      }
    } else if (auto v = take("--cadence-speed=")) {
      cfg.cadence_speed = std::stod(*v);
    } else if (auto v = take("--cadence-wpm=")) {
      cfg.cadence_wpm = std::stod(*v);
    } else if (auto v = take("--cadence-jitter=")) {
      cfg.cadence_jitter = std::stod(*v);
    } else if (auto v = take("--cadence-min-ms=")) {
      cfg.cadence_min_ms = std::stoi(*v);
    } else if (auto v = take("--cadence-max-ms=")) {
      cfg.cadence_max_ms = std::stoi(*v);
    } else if (auto v = take("--context-window=")) {
      cfg.context_window_turns = std::stoi(*v);
    } else if (auto v = take("--interleave=")) {
      cfg.interleave = std::max(1, std::stoi(*v));
    } else if (auto v = take("--consolidate-cycles=")) {
      cfg.consolidation_cycles = std::max(1, std::stoi(*v));
    } else if (auto v = take("--consolidate-every=")) {
      cfg.run_consolidation = true;
      cfg.consolidation_every_turns = std::max(1, std::stoi(*v));
    } else if (auto v = take("--seed=")) {
      cfg.seed = static_cast<unsigned int>(std::stoul(*v));
    } else if (auto v = take("--synthetic-start-ms=")) {
      cfg.synthetic_start_ms = static_cast<std::uint64_t>(std::stoull(*v));
    } else if (auto v = take("--deterministic=")) {
      const std::string val = *v;
      cfg.deterministic =
          (val == "1" || val == "true" || val == "yes" || val == "on");
    } else if (arg == "--no-cadence") {
      cfg.cadence_enabled = false;
    } else if (arg == "--deterministic") {
      cfg.deterministic = true;
    } else if (arg == "--consolidate") {
      cfg.run_consolidation = true;
    } else if (arg == "--consolidate-during") {
      cfg.run_consolidation = true;
      cfg.consolidate_during = true;
    } else if (arg == "--consolidate-idle") {
      cfg.run_consolidation = true;
      cfg.consolidate_during = true;
      cfg.consolidate_idle = true;
    } else if (arg == "--semantic") {
      cfg.semantic_overlap = true;
    } else if (arg == "--reuse") {
      cfg.reset_per_conversation = false;
    } else if (arg == "--no-procedural") {
      cfg.procedural_enabled = false;
    } else if (arg == "--no-sequential-edges") {
      cfg.sequential_edges_enabled = false;
    }
  }
  if (!std::filesystem::exists(cfg.dataset_path)) {
    const std::string fallback = "data/topical_chat/train.jsonl";
    if (std::filesystem::exists(fallback)) {
      cfg.dataset_path = fallback;
    }
  }
  return cfg;
}

} // namespace

int main(int argc, char **argv) {
  AnalysisConfig cfg = ParseArgs(argc, argv);
  if (cfg.deterministic) {
    if (std::getenv("CORTEXT_EMBED_THREADS") == nullptr) {
      setenv("CORTEXT_EMBED_THREADS", "1", 0);
    }
    if (std::getenv("CORTEXT_INFER_THREADS") == nullptr) {
      setenv("CORTEXT_INFER_THREADS", "1", 0);
    }
  }
  if (!std::filesystem::exists(cfg.dataset_path)) {
    std::cerr << "Dataset not found at: " << cfg.dataset_path << "\n";
    return 1;
  }

  ConsolidationTelemetry cons_metrics;
  InstallOtelLogger(cfg.otel_log_path, cfg.otel_filters, &cons_metrics);

  std::ifstream in(cfg.dataset_path);
  if (!in) {
    std::cerr << "Failed to open dataset: " << cfg.dataset_path << "\n";
    return 1;
  }

  std::unique_ptr<cortext::Cortext> cortext;
  std::unique_ptr<cortext::Encoder> eval_encoder;
  std::shared_ptr<cortext::Store> eval_store;
  std::shared_ptr<cortext::Store> stats_store;
  std::unordered_map<long long, std::vector<float>> embedding_cache;
  std::unordered_map<long long, std::string> memory_kind_cache;
  std::unordered_map<long long, MemoryEmotion> memory_emotion_cache;
  std::string text_encoder_backend = "unresolved";
  std::string text_encoder_path;
  bool baseline_counts_set = false;
  int consolidation_every_turns = 0;
  int last_consolidation_turn = -1;

  Stats stats;
  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<double> jitter_dist(
      1.0 - cfg.cadence_jitter, 1.0 + cfg.cadence_jitter);
  std::uint64_t synthetic_ts = cfg.synthetic_start_ms;
  bool semantic_ready = cfg.semantic_overlap;
  const double retrieval_thresh
      = cortext::core::RetrievalThresholdInterrupt(cfg.focus,
                                                   cfg.sensitivity);
  if (semantic_ready && cfg.db_path == ":memory:") {
    std::cerr << "Semantic overlap requires --db pointing to a file; disabling.\n";
    semantic_ready = false;
  }

  auto ensure_eval_encoder = [&]() -> bool {
    if (!semantic_ready) {
      return false;
    }
    if (eval_encoder) {
      return true;
    }
    try {
      auto resolved
          = cortext::internal::CreatePreferredTextEncoder(cfg.models_dir);
      text_encoder_backend = resolved.backend_name;
      text_encoder_path = resolved.resolved_path.string();
      eval_encoder = std::move(resolved.encoder);
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Semantic encoder init failed: " << e.what() << "\n";
      semantic_ready = false;
      return false;
    }
  };

  auto ensure_stats_store = [&]() -> bool {
    if (stats_store) {
      return true;
    }
    if (cfg.db_path == ":memory:") {
      return false;
    }
    auto uniq = cortext::SQLiteStore::Create(cfg.db_path.c_str());
    if (!uniq) {
      return false;
    }
    stats_store = std::shared_ptr<cortext::Store>(std::move(uniq));
    return true;
  };

  auto get_memory_embedding =
      [&](long long memory_id) -> std::optional<std::vector<float>> {
    auto it = embedding_cache.find(memory_id);
    if (it != embedding_cache.end()) {
      return it->second;
    }
    if (!eval_store) {
      return std::nullopt;
    }
    auto rows = eval_store->Execute(
        "SELECT e.embedding FROM memories m "
        "JOIN embeddings e ON m.embedding_id = e.embedding_id "
        "WHERE m.memory_id = ?",
        {memory_id});
    if (rows.empty()) {
      return std::nullopt;
    }
    auto it_emb = rows[0].find("embedding");
    if (it_emb == rows[0].end()) {
      return std::nullopt;
    }
    auto decoded = DecodeFloatBlobDynamic(it_emb->second);
    if (!decoded) {
      return std::nullopt;
    }
    embedding_cache[memory_id] = *decoded;
    return decoded;
  };

  std::vector<Conversation> conversations;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    nlohmann::json row = nlohmann::json::parse(line, nullptr, false);
    if (!row.is_array() || row.size() < 2) {
      continue;
    }
    const std::string conv_id = row[0].get<std::string>();
    const auto &conv = row[1];
    if (!conv.contains("content") || !conv["content"].is_array()) {
      continue;
    }
    Conversation c;
    c.id = conv_id;
    for (const auto &turn : conv["content"]) {
      Turn t;
      t.agent = turn.value("agent", "agent");
      t.message = turn.value("message", "");
      if (!t.message.empty()) {
        c.turns.push_back(std::move(t));
      }
    }
    if (!c.turns.empty()) {
      conversations.push_back(std::move(c));
    }
    if (static_cast<int>(conversations.size()) >= cfg.max_conversations) {
      break;
    }
  }

  if (conversations.empty()) {
    std::cerr << "No valid conversations found in dataset.\n";
    return 1;
  }

  if (semantic_ready && ensure_eval_encoder()) {
    std::cout << "text_encoder_backend=" << text_encoder_backend << "\n";
    if (!text_encoder_path.empty()) {
      std::cout << "text_encoder_path=" << text_encoder_path << "\n";
    }
  }

  const int interleave = std::max(1, cfg.interleave);
  size_t conv_index = 0;
  while (conv_index < conversations.size()) {
    const size_t batch_end =
        std::min(conv_index + static_cast<size_t>(interleave),
                 conversations.size());
    const size_t batch_size = batch_end - conv_index;

    if (!cortext || cfg.reset_per_conversation) {
      cortext::Cortext::Config c;
      c.focus = cfg.focus;
      c.sensitivity = cfg.sensitivity;
      c.stability = cfg.stability;
      c.affect_interrupt = cfg.affect_interrupt;
      c.affect_retrieval = cfg.affect_retrieval;
      c.reinforcement_enabled = cfg.reinforcement_enabled;
      c.procedural_enabled = cfg.procedural_enabled;
      c.sequential_edges_enabled = cfg.sequential_edges_enabled;
      cortext = cortext::Cortext::Create(c, cfg.db_path, cfg.models_dir);
      embedding_cache.clear();
      memory_kind_cache.clear();
      memory_emotion_cache.clear();
      eval_store.reset();
      stats_store.reset();
      if (semantic_ready) {
        auto uniq = cortext::SQLiteStore::Create(cfg.db_path.c_str());
        eval_store = std::shared_ptr<cortext::Store>(std::move(uniq));
      }
    }
    if (cfg.run_consolidation) {
      consolidation_every_turns = cfg.consolidation_every_turns;
    }
    if (!baseline_counts_set && ensure_stats_store()) {
      stats.consolidation_baseline_association =
          CountMemoriesByKind(stats_store, "ASSOCIATION");
      stats.consolidation_baseline_label =
          CountMemoriesByKind(stats_store, "LABEL");
      baseline_counts_set = true;
    }

    std::vector<std::deque<std::unordered_set<std::string>>> recent_context_tokens(
        batch_size);
    std::vector<std::deque<std::vector<float>>> recent_context_embeddings(
        batch_size);
    std::vector<size_t> turn_indices(batch_size, 0);

    for (size_t i = 0; i < batch_size; ++i) {
      std::cout << "\n=== Conversation " << conversations[conv_index + i].id
                << " ===\n";
      stats.conversations++;
    }

    bool stop_all = false;
    bool any_progress = true;
    while (any_progress && !stop_all) {
      any_progress = false;
      for (size_t i = 0; i < batch_size; ++i) {
        auto &conv = conversations[conv_index + i];
        if (turn_indices[i] >= conv.turns.size()
            || static_cast<int>(turn_indices[i])
                   >= cfg.max_turns_per_conversation) {
          continue;
        }
        any_progress = true;

        const auto &turn = conv.turns[turn_indices[i]];
        std::string agent = turn.agent.empty() ? "agent" : turn.agent;
        std::string message = turn.message;
        const double wps = std::max(1e-3, cfg.cadence_wpm / 60.0);
        const int target_words = std::max(
            1, static_cast<int>(std::round(wps * 0.7)));
        const auto chunks = ChunkMessage(message, target_words);

        const std::string source_id = "chat/" + agent;
        for (const auto &chunk : chunks) {
          if (stats.turns >= cfg.max_total_turns) {
            stop_all = true;
            break;
          }
          if (chunk.empty()) {
            continue;
          }

          int delay_ms = 0;
          const bool use_synthetic_cadence
              = cfg.cadence_enabled || cfg.deterministic;
          if (use_synthetic_cadence) {
            const int words = std::max(1, CountWords(chunk));
            double delay_s = static_cast<double>(words) / wps;
            delay_s += 0.1;
            delay_s *= jitter_dist(rng);
            if (cfg.cadence_speed > 0.0) {
              delay_s /= cfg.cadence_speed;
            }
            delay_ms = static_cast<int>(delay_s * 1000.0);
            delay_ms = std::max(cfg.cadence_min_ms, delay_ms);
            delay_ms = std::min(cfg.cadence_max_ms, delay_ms);
          }
          const int synthetic_step_ms =
              std::max(1, delay_ms);
          const std::uint64_t signal_ts =
              cfg.deterministic ? synthetic_ts : 0;

          const auto turn_tokens = Tokenize(chunk);
          const auto context_tokens = MergeTokenWindow(recent_context_tokens[i]);

          std::vector<float> turn_embedding;
          std::vector<float> context_embedding;
          if (semantic_ready && ensure_eval_encoder()) {
            try {
              eval_encoder->EncodeText(chunk, turn_embedding);
            } catch (const std::exception &e) {
              std::cerr << "Semantic encoding failed: " << e.what() << "\n";
              semantic_ready = false;
            }
          }
          if (semantic_ready && !recent_context_embeddings[i].empty()) {
            context_embedding = MeanEmbedding(recent_context_embeddings[i]);
          }

          cortext::Cortext::Context ctx;
          try {
            ctx = cfg.deterministic
                      ? cortext->ProcessTextAt(chunk, source_id, signal_ts)
                      : cortext->ProcessText(chunk, source_id);
          } catch (const std::exception &e) {
            std::cerr << "ProcessText failed: " << e.what() << "\n";
            return 1;
          }
          if (cfg.deterministic) {
            synthetic_ts += static_cast<std::uint64_t>(synthetic_step_ms);
          }

          stats.turns++;
          if (ctx.output.stored_embedding_id.has_value()) {
            stats.writes++;
            stats.signals_per_write.push_back(stats.signals_since_write + 1);
            stats.signals_since_write = 0;
          } else {
            stats.signals_since_write++;
          }

          stats.wm_samples++;
          stats.wm_total_slots += static_cast<int>(ctx.working_memory.size());
          stats.wm_max_slots = std::max(stats.wm_max_slots,
                                        static_cast<int>(ctx.working_memory.size()));
          stats.encode_ms_values.push_back(ctx.encode_ms);
          stats.process_ms_values.push_back(ctx.process_ms);
          stats.hydrate_ms_values.push_back(ctx.hydrate_ms);
          stats.total_ms_values.push_back(ctx.total_ms);
          for (const auto &kv : ctx.output.operation_ms) {
            stats.operation_time_sum[kv.first] += kv.second;
            stats.operation_time_count[kv.first] += 1;
          }

          if (!ctx.working_memory.empty()) {
            double best_overlap = 0.0;
            for (const auto &mem : ctx.working_memory) {
              const std::string mem_text = MemoryToText(mem);
              const auto mem_tokens = Tokenize(mem_text);
              best_overlap =
                  std::max(best_overlap, Jaccard(turn_tokens, mem_tokens));
            }
            stats.wm_overlap_sum += best_overlap;
            stats.wm_overlap_count++;
          }

          if (auto relevance =
                  GetMetric(ctx, cortext::operations::Metric::relevance)) {
            stats.relevance_values.push_back(*relevance);
            if (ctx.should_interrupt) {
              stats.interrupt_relevance_values.push_back(*relevance);
            }
          }
          if (auto surprise =
                  GetMetric(ctx, cortext::operations::Metric::surprise)) {
            stats.surprise_values.push_back(*surprise);
            if (ctx.should_interrupt) {
              stats.interrupt_surprise_values.push_back(*surprise);
            }
          }
          if (auto mismatch =
                  GetMetric(ctx, cortext::operations::Metric::mismatch)) {
            const double denom = (1.0 - cfg.focus) * cfg.sensitivity;
            double novelty_est = denom > 1e-6 ? (*mismatch / denom) : *mismatch;
            novelty_est = Clamp01(novelty_est);
            stats.novelty_values.push_back(novelty_est);
            if (ctx.should_interrupt) {
              stats.interrupt_novelty_values.push_back(novelty_est);
            }
          }

          const double salience_metric =
              GetMetric(ctx, cortext::operations::Metric::salience).value_or(0.0);
          stats.emotion_intensity_values.push_back(ctx.output.emotion_intensity);
          stats.arousal_values.push_back(ctx.output.arousal);
          stats.valence_values.push_back(ctx.output.valence);
          stats.salience_values.push_back(salience_metric);
          const double affect_drive = ComputeAffectDrive(
              cfg.sensitivity, ctx.output.emotion_intensity, ctx.output.arousal,
              salience_metric);
          stats.affect_drive_values.push_back(affect_drive);
          stats.interrupt_gate_affect_drive_values.push_back(
              ctx.interrupt_gate_affect_drive);
          stats.interrupt_gate_retrieval_thresh_values.push_back(
              ctx.interrupt_gate_retrieval_thresh);
          stats.interrupt_gate_boundary_mult_eff_values.push_back(
              ctx.interrupt_gate_boundary_mult_eff);
          if (ctx.boundary_score.has_value()) {
            stats.boundary_score_values.push_back(*ctx.boundary_score);
            const double boundary_thresh =
                cortext::core::BoundaryThreshold(cfg.focus, cfg.sensitivity);
            if (*ctx.boundary_score >= boundary_thresh) {
              stats.boundary_score_pass_count++;
            }
          }
          if (ctx.at_boundary) {
            stats.boundary_at_count++;
          }
          if (ctx.should_interrupt) {
            stats.interrupt_affect_drive_values.push_back(affect_drive);
            stats.interrupt_gate_affect_drive_interrupt_values.push_back(
                ctx.interrupt_gate_affect_drive);
          } else {
            stats.non_interrupt_affect_drive_values.push_back(affect_drive);
            stats.interrupt_gate_affect_drive_non_interrupt_values.push_back(
                ctx.interrupt_gate_affect_drive);
          }

          if (ctx.should_interrupt) {
            stats.interrupt_turns++;
            if (!ctx.retrieved_memory.empty()) {
              stats.interrupt_turns_with_retrieval++;
            }
          }
          if (ctx.interrupt_aborted) {
            stats.interrupt_abort_count++;
          }

          if (!ctx.retrieved_memory.empty()) {
            stats.retrieval_turns++;
            stats.total_retrieved +=
                static_cast<int>(ctx.retrieved_memory.size());
            double best_overlap = 0.0;
            double best_context_overlap = 0.0;
            double best_semantic = 0.0;
            double best_context_semantic = 0.0;
            bool has_semantic = false;
            bool has_context_semantic = false;
            double best_summary_semantic = 0.0;
            bool has_summary_semantic = false;
            int summary_candidates = 0;
            int non_summary_candidates = 0;
            std::string best_text;
            bool turn_has_association = false;
            bool turn_has_label = false;
            double best_emotion_bonus = 0.0;
            bool has_emotion_bonus = false;
            const double s_eff = cortext::core::SensitivityBias(cfg.sensitivity);
            const double w_emotion =
                cfg.affect_retrieval
                    ? cortext::core::RetrievalEmotionWeight(cfg.sensitivity)
                    : 0.0;
            const double w_mem_emotion_raw = cortext::core::Lerp(0.60, 0.80, s_eff);
            const double w_mem_arousal_raw = cortext::core::Lerp(0.20, 0.40, s_eff);
            const double w_mem_sum =
                std::max(1e-9, w_mem_emotion_raw + w_mem_arousal_raw);
            const double w_mem_emotion = w_mem_emotion_raw / w_mem_sum;
            const double w_mem_arousal = w_mem_arousal_raw / w_mem_sum;
            for (const auto &mem : ctx.retrieved_memory) {
              const std::string mem_text = MemoryToText(mem);
              const auto mem_tokens = Tokenize(mem_text);
              const double overlap = Jaccard(turn_tokens, mem_tokens);
              double mem_semantic = 0.0;
              bool has_mem_semantic = false;
              if (overlap > best_overlap) {
                best_overlap = overlap;
                best_text = mem_text;
              }
              if (!context_tokens.empty()) {
                const double ctx_overlap = Jaccard(context_tokens, mem_tokens);
                best_context_overlap =
                    std::max(best_context_overlap, ctx_overlap);
              }

              if (semantic_ready && !turn_embedding.empty()) {
                auto emb_opt = get_memory_embedding(mem.id);
                if (emb_opt) {
                  const double sim = CosineSimilarity(turn_embedding, *emb_opt);
                  if (!has_semantic || sim > best_semantic) {
                    best_semantic = sim;
                  }
                  has_semantic = true;
                  mem_semantic = sim;
                  has_mem_semantic = true;
                  if (!context_embedding.empty()) {
                    const double ctx_sim =
                        CosineSimilarity(context_embedding, *emb_opt);
                    if (!has_context_semantic || ctx_sim > best_context_semantic) {
                      best_context_semantic = ctx_sim;
                    }
                    has_context_semantic = true;
                  }
                }
              }

              if (ensure_stats_store()) {
                auto kind = GetMemoryKind(stats_store, memory_kind_cache, mem.id);
                if (kind) {
                  if (*kind == "ASSOCIATION") {
                    stats.retrieval_association_candidates++;
                    turn_has_association = true;
                    summary_candidates++;
                    if (ctx.should_interrupt) {
                      stats.interrupt_association_candidates++;
                    }
                    if (has_mem_semantic
                        && (!has_summary_semantic
                            || mem_semantic > best_summary_semantic)) {
                      best_summary_semantic = mem_semantic;
                      has_summary_semantic = true;
                    }
                  } else if (*kind == "LABEL") {
                    stats.retrieval_label_candidates++;
                    turn_has_label = true;
                    non_summary_candidates++;
                    if (ctx.should_interrupt) {
                      stats.interrupt_label_candidates++;
                    }
                  } else {
                    stats.retrieval_other_candidates++;
                    non_summary_candidates++;
                    if (ctx.should_interrupt) {
                      stats.interrupt_other_candidates++;
                    }
                  }
                } else {
                  stats.retrieval_other_candidates++;
                  non_summary_candidates++;
                  if (ctx.should_interrupt) {
                    stats.interrupt_other_candidates++;
                  }
                }
              }

              if (stats_store) {
                auto mem_emotion =
                    GetMemoryEmotion(stats_store, memory_emotion_cache, mem.id);
                if (mem_emotion) {
                  const double memory_affect =
                      w_mem_emotion * mem_emotion->intensity
                      + w_mem_arousal * mem_emotion->arousal_avg;
                  const double emotion_bonus = w_emotion * affect_drive * memory_affect;
                  if (!has_emotion_bonus || emotion_bonus > best_emotion_bonus) {
                    best_emotion_bonus = emotion_bonus;
                    has_emotion_bonus = true;
                  }
                }
              }
            }
            stats.retrieval_overlap_sum += best_overlap;
            stats.retrieval_overlap_count++;
            stats.retrieval_overlap_values.push_back(best_overlap);
            if (!context_tokens.empty()) {
              stats.retrieval_context_overlap_values.push_back(
                  best_context_overlap);
            }
            if (has_semantic) {
              stats.retrieval_semantic_overlap_values.push_back(best_semantic);
            }
            if (has_context_semantic) {
              stats.retrieval_context_semantic_overlap_values.push_back(
                  best_context_semantic);
            }
            if (has_semantic) {
              const bool semantic_positive = (best_semantic >= retrieval_thresh);
              if (semantic_positive) {
                stats.retrieval_semantic_positive_turns++;
              }
              if (ctx.should_interrupt) {
                if (semantic_positive) {
                  stats.interrupt_semantic_true_positive++;
                } else {
                  stats.interrupt_semantic_false_positive++;
                }
              } else {
                if (semantic_positive) {
                  stats.interrupt_semantic_false_negative++;
                  if (!ctx.interrupt_gate_has_candidates) {
                    stats.interrupt_gate_fail_no_candidates++;
                  }
                  if (ctx.interrupt_gate_blocked_no_store) {
                    stats.interrupt_gate_fail_no_store++;
                  }
                  if (!ctx.interrupt_gate_rel_pass) {
                    stats.interrupt_gate_fail_rel++;
                  }
                  if (!ctx.interrupt_gate_novelty_pass) {
                    stats.interrupt_gate_fail_novelty++;
                  }
                  if (!ctx.interrupt_gate_mu_pass) {
                    stats.interrupt_gate_fail_mu++;
                  }
                  if (!ctx.interrupt_gate_novelty_mu_pass) {
                    stats.interrupt_gate_fail_novelty_mu++;
                  }
                  if (!ctx.interrupt_gate_dup_pass) {
                    stats.interrupt_gate_fail_dup++;
                  }
                  if (!ctx.interrupt_gate_boundary_mu_pass) {
                    stats.interrupt_gate_fail_boundary_mu++;
                  }
                }
                stats.non_interrupt_semantic_overlap_values.push_back(best_semantic);
                if (has_context_semantic) {
                  stats.non_interrupt_context_semantic_overlap_values.push_back(
                      best_context_semantic);
                }
              }
            }
            if (summary_candidates > 0) {
              stats.retrieval_summary_hit_turns++;
              if (has_summary_semantic) {
                stats.retrieval_summary_hit_overlap_values.push_back(
                    best_summary_semantic);
              }
              if (non_summary_candidates == 0) {
                stats.retrieval_summary_only_turns++;
              }
            }
            if (has_emotion_bonus) {
              stats.retrieval_emotion_bonus_values.push_back(best_emotion_bonus);
              if (ctx.should_interrupt) {
                stats.interrupt_retrieval_emotion_bonus_values.push_back(
                    best_emotion_bonus);
              }
            }
            if (ctx.should_interrupt) {
              stats.interrupt_overlap_values.push_back(best_overlap);
              if (!context_tokens.empty()) {
                stats.interrupt_context_overlap_values.push_back(
                    best_context_overlap);
              }
              if (has_semantic) {
                stats.interrupt_semantic_overlap_values.push_back(best_semantic);
              }
              if (has_context_semantic) {
                stats.interrupt_context_semantic_overlap_values.push_back(
                    best_context_semantic);
              }
            }
            if (turn_has_association) {
              stats.retrieval_turns_with_association++;
              if (ctx.should_interrupt) {
                stats.interrupt_turns_with_association++;
              }
            }
            if (turn_has_label) {
              stats.retrieval_turns_with_label++;
              if (ctx.should_interrupt) {
                stats.interrupt_turns_with_label++;
              }
            }
            if (!best_text.empty()) {
              RecordRetrievalExample(stats, best_overlap, chunk, best_text);
            }
          }

          if (stats.turns % 20 == 0) {
            std::cout << "  signals=" << stats.turns
                      << " writes=" << stats.writes
                      << " wm_slots=" << ctx.working_memory.size()
                      << " retrieved=" << ctx.retrieved_memory.size()
                      << "\n";
          }

          if (cfg.cadence_enabled && !cfg.deterministic) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
          }

          if (cfg.context_window_turns > 0) {
            recent_context_tokens[i].push_back(turn_tokens);
            while (static_cast<int>(recent_context_tokens[i].size())
                   > cfg.context_window_turns) {
              recent_context_tokens[i].pop_front();
            }
            if (semantic_ready && !turn_embedding.empty()) {
              recent_context_embeddings[i].push_back(turn_embedding);
              while (static_cast<int>(recent_context_embeddings[i].size())
                     > cfg.context_window_turns) {
                recent_context_embeddings[i].pop_front();
              }
            }
          }

          if (cfg.run_consolidation
              && (cfg.consolidation_every_turns > 0 || cfg.consolidate_during)) {
            bool should_consolidate = false;
            if (cfg.consolidation_every_turns > 0
                && stats.turns % cfg.consolidation_every_turns == 0) {
              should_consolidate = true;
            }
            if (cfg.consolidate_during
                && (ctx.consolidation_recommended
                    || ctx.consolidation_required)) {
              if (!cfg.consolidate_idle || ctx.at_boundary) {
                should_consolidate = true;
              }
            }
            if (should_consolidate && last_consolidation_turn != stats.turns) {
              for (int cycle = 0; cycle < cfg.consolidation_cycles; ++cycle) {
                try {
                  cortext->Consolidate();
                  stats.consolidation_runs++;
                } catch (const std::exception &e) {
                  stats.consolidation_failures++;
                  std::cerr << "Consolidate failed: " << e.what() << "\n";
                  return 1;
                }
              }
              last_consolidation_turn = stats.turns;
            }
          }
        }

        turn_indices[i]++;
        if (stop_all) {
          break;
        }
      }
    }

    if (cfg.run_consolidation && cortext) {
      for (int i = 0; i < cfg.consolidation_cycles; ++i) {
        try {
          cortext->Consolidate();
          stats.consolidation_runs++;
        } catch (const std::exception &e) {
          stats.consolidation_failures++;
          std::cerr << "Consolidate failed: " << e.what() << "\n";
          return 1;
        }
      }
    }

    if (stop_all) {
      break;
    }

    if (cfg.reset_per_conversation) {
      cortext->Flush();
      cortext.reset();
    } else {
      cortext->Flush();
    }

    conv_index = batch_end;
  }

  if (cfg.run_consolidation && baseline_counts_set && ensure_stats_store()) {
    stats.consolidation_final_association =
        CountMemoriesByKind(stats_store, "ASSOCIATION");
    stats.consolidation_final_label =
        CountMemoriesByKind(stats_store, "LABEL");
    stats.consolidation_association_created =
        std::max(0LL, stats.consolidation_final_association
                           - stats.consolidation_baseline_association);
    stats.consolidation_label_created =
        std::max(0LL, stats.consolidation_final_label
                           - stats.consolidation_baseline_label);
  }
  if (ensure_stats_store()) {
    const auto assoc_stats = GetAssociationStats(stats_store, "reinforces");
    stats.reinforcement_edge_count = assoc_stats.count;
    stats.reinforcement_weight_mean = assoc_stats.weight_mean;
    const auto mem_stats = GetMemoryStats(stats_store, "LONG_TERM");
    stats.memory_long_term_count = mem_stats.count;
    stats.memory_strength_mean = mem_stats.strength_mean;
    stats.memory_used_count_mean = mem_stats.used_count_mean;
    stats.memory_retrieved_count_mean = mem_stats.retrieved_count_mean;
  }

  const double avg_signals_per_write =
      stats.signals_per_write.empty()
          ? 0.0
          : std::accumulate(stats.signals_per_write.begin(),
                            stats.signals_per_write.end(), 0.0)
                / static_cast<double>(stats.signals_per_write.size());
  const double avg_wm_slots =
      stats.wm_samples == 0
          ? 0.0
          : static_cast<double>(stats.wm_total_slots) / stats.wm_samples;
  const double retrieval_rate =
      stats.turns == 0 ? 0.0
                       : static_cast<double>(stats.retrieval_turns)
                             / static_cast<double>(stats.turns);
  const double avg_retrieval_count =
      stats.retrieval_turns == 0
          ? 0.0
          : static_cast<double>(stats.total_retrieved)
                / static_cast<double>(stats.retrieval_turns);
  const double avg_retrieval_overlap =
      stats.retrieval_overlap_count == 0
          ? 0.0
          : stats.retrieval_overlap_sum
                / static_cast<double>(stats.retrieval_overlap_count);
  const double avg_wm_overlap =
      stats.wm_overlap_count == 0
          ? 0.0
          : stats.wm_overlap_sum / static_cast<double>(stats.wm_overlap_count);
  const double interrupt_rate =
      stats.turns == 0 ? 0.0
                       : static_cast<double>(stats.interrupt_turns)
                             / static_cast<double>(stats.turns);
  const double interrupt_with_retrieval_rate =
      stats.interrupt_turns == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_turns_with_retrieval)
                / static_cast<double>(stats.interrupt_turns);
  const double retrieval_association_candidate_rate =
      stats.total_retrieved == 0
          ? 0.0
          : static_cast<double>(stats.retrieval_association_candidates)
                / static_cast<double>(stats.total_retrieved);
  const double retrieval_label_candidate_rate =
      stats.total_retrieved == 0
          ? 0.0
          : static_cast<double>(stats.retrieval_label_candidates)
                / static_cast<double>(stats.total_retrieved);
  const double retrieval_association_turn_rate =
      stats.retrieval_turns == 0
          ? 0.0
          : static_cast<double>(stats.retrieval_turns_with_association)
                / static_cast<double>(stats.retrieval_turns);
  const double retrieval_label_turn_rate =
      stats.retrieval_turns == 0
          ? 0.0
          : static_cast<double>(stats.retrieval_turns_with_label)
                / static_cast<double>(stats.retrieval_turns);
  const double retrieval_summary_hit_rate =
      stats.retrieval_turns == 0
          ? 0.0
          : static_cast<double>(stats.retrieval_summary_hit_turns)
                / static_cast<double>(stats.retrieval_turns);
  const double retrieval_summary_only_turn_rate =
      stats.retrieval_turns == 0
          ? 0.0
          : static_cast<double>(stats.retrieval_summary_only_turns)
                / static_cast<double>(stats.retrieval_turns);
  const int interrupt_candidate_total =
      stats.interrupt_association_candidates + stats.interrupt_label_candidates
      + stats.interrupt_other_candidates;
  const double interrupt_association_candidate_rate =
      interrupt_candidate_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_association_candidates)
                / static_cast<double>(interrupt_candidate_total);
  const double interrupt_label_candidate_rate =
      interrupt_candidate_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_label_candidates)
                / static_cast<double>(interrupt_candidate_total);
  const double interrupt_association_turn_rate =
      stats.interrupt_turns == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_turns_with_association)
                / static_cast<double>(stats.interrupt_turns);
  const double interrupt_label_turn_rate =
      stats.interrupt_turns == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_turns_with_label)
                / static_cast<double>(stats.interrupt_turns);
  const int interrupt_semantic_decisions =
      stats.interrupt_semantic_true_positive + stats.interrupt_semantic_false_positive;
  const double interrupt_precision =
      interrupt_semantic_decisions == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_semantic_true_positive)
                / static_cast<double>(interrupt_semantic_decisions);
  const double interrupt_recall =
      stats.retrieval_semantic_positive_turns == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_semantic_true_positive)
                / static_cast<double>(stats.retrieval_semantic_positive_turns);
  const double interrupt_false_positive_rate =
      interrupt_semantic_decisions == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_semantic_false_positive)
                / static_cast<double>(interrupt_semantic_decisions);
  const double interrupt_false_negative_rate =
      stats.retrieval_semantic_positive_turns == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_semantic_false_negative)
                / static_cast<double>(stats.retrieval_semantic_positive_turns);
  const int interrupt_false_negative_total = stats.interrupt_semantic_false_negative;
  const double interrupt_gate_fail_no_candidates_rate =
      interrupt_false_negative_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_gate_fail_no_candidates)
                / static_cast<double>(interrupt_false_negative_total);
  const double interrupt_gate_fail_no_store_rate =
      interrupt_false_negative_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_gate_fail_no_store)
                / static_cast<double>(interrupt_false_negative_total);
  const double interrupt_gate_fail_rel_rate =
      interrupt_false_negative_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_gate_fail_rel)
                / static_cast<double>(interrupt_false_negative_total);
  const double interrupt_gate_fail_novelty_rate =
      interrupt_false_negative_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_gate_fail_novelty)
                / static_cast<double>(interrupt_false_negative_total);
  const double interrupt_gate_fail_mu_rate =
      interrupt_false_negative_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_gate_fail_mu)
                / static_cast<double>(interrupt_false_negative_total);
  const double interrupt_gate_fail_novelty_mu_rate =
      interrupt_false_negative_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_gate_fail_novelty_mu)
                / static_cast<double>(interrupt_false_negative_total);
  const double interrupt_gate_fail_dup_rate =
      interrupt_false_negative_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_gate_fail_dup)
                / static_cast<double>(interrupt_false_negative_total);
  const double interrupt_gate_fail_boundary_mu_rate =
      interrupt_false_negative_total == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_gate_fail_boundary_mu)
                / static_cast<double>(interrupt_false_negative_total);

  const auto novelty_summary = Summarize(stats.novelty_values);
  const auto relevance_summary = Summarize(stats.relevance_values);
  const auto surprise_summary = Summarize(stats.surprise_values);
  const auto retrieval_overlap_summary = Summarize(stats.retrieval_overlap_values);
  const auto retrieval_context_overlap_summary =
      Summarize(stats.retrieval_context_overlap_values);
  const auto retrieval_semantic_summary =
      Summarize(stats.retrieval_semantic_overlap_values);
  const auto retrieval_context_semantic_summary =
      Summarize(stats.retrieval_context_semantic_overlap_values);
  const auto retrieval_summary_hit_overlap_summary =
      Summarize(stats.retrieval_summary_hit_overlap_values);
  const auto non_interrupt_semantic_summary =
      Summarize(stats.non_interrupt_semantic_overlap_values);
  const auto non_interrupt_context_semantic_summary =
      Summarize(stats.non_interrupt_context_semantic_overlap_values);
  const auto interrupt_overlap_summary = Summarize(stats.interrupt_overlap_values);
  const auto interrupt_context_overlap_summary =
      Summarize(stats.interrupt_context_overlap_values);
  const auto interrupt_semantic_summary =
      Summarize(stats.interrupt_semantic_overlap_values);
  const auto interrupt_context_semantic_summary =
      Summarize(stats.interrupt_context_semantic_overlap_values);
  const auto interrupt_novelty_summary = Summarize(stats.interrupt_novelty_values);
  const auto interrupt_relevance_summary = Summarize(stats.interrupt_relevance_values);
  const auto interrupt_surprise_summary = Summarize(stats.interrupt_surprise_values);
  const auto emotion_intensity_summary =
      Summarize(stats.emotion_intensity_values);
  const auto arousal_summary = Summarize(stats.arousal_values);
  const auto valence_summary = Summarize(stats.valence_values);
  const auto salience_summary = Summarize(stats.salience_values);
  const auto affect_summary = Summarize(stats.affect_drive_values);
  const auto interrupt_affect_summary =
      Summarize(stats.interrupt_affect_drive_values);
  const auto non_interrupt_affect_summary =
      Summarize(stats.non_interrupt_affect_drive_values);
  const auto interrupt_gate_affect_summary =
      Summarize(stats.interrupt_gate_affect_drive_values);
  const auto interrupt_gate_affect_interrupt_summary =
      Summarize(stats.interrupt_gate_affect_drive_interrupt_values);
  const auto interrupt_gate_affect_non_interrupt_summary =
      Summarize(stats.interrupt_gate_affect_drive_non_interrupt_values);
  const auto interrupt_gate_retrieval_thresh_summary =
      Summarize(stats.interrupt_gate_retrieval_thresh_values);
  const auto interrupt_gate_boundary_mult_summary =
      Summarize(stats.interrupt_gate_boundary_mult_eff_values);
  const auto boundary_score_summary = Summarize(stats.boundary_score_values);
  const auto retrieval_emotion_bonus_summary =
      Summarize(stats.retrieval_emotion_bonus_values);
  const auto interrupt_retrieval_emotion_bonus_summary =
      Summarize(stats.interrupt_retrieval_emotion_bonus_values);
  const auto encode_ms_summary = Summarize(stats.encode_ms_values);
  const auto process_ms_summary = Summarize(stats.process_ms_values);
  const auto hydrate_ms_summary = Summarize(stats.hydrate_ms_values);
  const auto total_ms_summary = Summarize(stats.total_ms_values);
  std::vector<std::pair<std::string, double>> op_time_means;
  op_time_means.reserve(stats.operation_time_sum.size());
  for (const auto &kv : stats.operation_time_sum) {
    const int count = stats.operation_time_count[kv.first];
    if (count > 0) {
      op_time_means.emplace_back(kv.first, kv.second / static_cast<double>(count));
    }
  }
  std::sort(op_time_means.begin(), op_time_means.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  const double interrupt_semantic_delta =
      (interrupt_semantic_summary.count > 0
       && non_interrupt_semantic_summary.count > 0)
          ? (interrupt_semantic_summary.mean - non_interrupt_semantic_summary.mean)
          : 0.0;
  const double interrupt_context_semantic_delta =
      (interrupt_context_semantic_summary.count > 0
       && non_interrupt_context_semantic_summary.count > 0)
          ? (interrupt_context_semantic_summary.mean
             - non_interrupt_context_semantic_summary.mean)
          : 0.0;
  const double boundary_at_rate =
      stats.turns == 0 ? 0.0
                       : static_cast<double>(stats.boundary_at_count)
                             / static_cast<double>(stats.turns);
  const double boundary_score_pass_rate =
      stats.turns == 0 ? 0.0
                       : static_cast<double>(stats.boundary_score_pass_count)
                             / static_cast<double>(stats.turns);
  const double interrupt_abort_rate =
      stats.interrupt_turns == 0
          ? 0.0
          : static_cast<double>(stats.interrupt_abort_count)
                / static_cast<double>(stats.interrupt_turns);

  std::cout << "\n=== Summary ===\n";
  std::cout << "conversations=" << stats.conversations << "\n";
  std::cout << "turns=" << stats.turns << "\n";
  std::cout << "writes=" << stats.writes << "\n";
  std::cout << "reinforcement_mode=" << (cfg.reinforcement_enabled ? "on" : "off")
            << "\n";
  if (cfg.run_consolidation) {
    std::cout << "consolidation_runs=" << stats.consolidation_runs << "\n";
    std::cout << "consolidation_failures=" << stats.consolidation_failures << "\n";
    std::cout << "consolidation_every_turns=" << consolidation_every_turns << "\n";
    std::cout << "consolidation_association_created="
              << stats.consolidation_association_created << "\n";
    std::cout << "consolidation_label_created="
              << stats.consolidation_label_created << "\n";
  }
  std::cout << "reinforcement_edge_count=" << stats.reinforcement_edge_count << "\n";
  std::cout << "reinforcement_weight_mean=" << stats.reinforcement_weight_mean
            << "\n";
  std::cout << "memory_long_term_count=" << stats.memory_long_term_count << "\n";
  std::cout << "memory_strength_mean=" << stats.memory_strength_mean << "\n";
  std::cout << "memory_used_count_mean=" << stats.memory_used_count_mean << "\n";
  std::cout << "memory_retrieved_count_mean=" << stats.memory_retrieved_count_mean
            << "\n";
  std::cout << "avg_signals_per_write=" << avg_signals_per_write << "\n";
  std::cout << "wm_avg_slots=" << avg_wm_slots << "\n";
  std::cout << "wm_max_slots=" << stats.wm_max_slots << "\n";
  std::cout << "perf_encode_ms_mean=" << encode_ms_summary.mean << "\n";
  std::cout << "perf_process_ms_mean=" << process_ms_summary.mean << "\n";
  std::cout << "perf_hydrate_ms_mean=" << hydrate_ms_summary.mean << "\n";
  std::cout << "perf_total_ms_mean=" << total_ms_summary.mean << "\n";
  std::cout << "retrieval_turn_rate=" << retrieval_rate << "\n";
  std::cout << "retrieval_avg_candidates=" << avg_retrieval_count << "\n";
  std::cout << "retrieval_best_overlap=" << avg_retrieval_overlap << "\n";
  std::cout << "retrieval_association_candidate_rate="
            << retrieval_association_candidate_rate << "\n";
  std::cout << "retrieval_label_candidate_rate="
            << retrieval_label_candidate_rate << "\n";
  std::cout << "retrieval_association_turn_rate="
            << retrieval_association_turn_rate << "\n";
  std::cout << "retrieval_label_turn_rate="
            << retrieval_label_turn_rate << "\n";
  std::cout << "retrieval_summary_hit_rate="
            << retrieval_summary_hit_rate << "\n";
  std::cout << "retrieval_summary_only_turn_rate="
            << retrieval_summary_only_turn_rate << "\n";
  std::cout << "wm_best_overlap=" << avg_wm_overlap << "\n";
  std::cout << "interrupt_turn_rate=" << interrupt_rate << "\n";
  std::cout << "interrupt_with_retrieval_rate=" << interrupt_with_retrieval_rate << "\n";
  std::cout << "interrupt_association_candidate_rate="
            << interrupt_association_candidate_rate << "\n";
  std::cout << "interrupt_label_candidate_rate="
            << interrupt_label_candidate_rate << "\n";
  std::cout << "interrupt_association_turn_rate="
            << interrupt_association_turn_rate << "\n";
  std::cout << "interrupt_label_turn_rate="
            << interrupt_label_turn_rate << "\n";
  std::cout << "interrupt_precision=" << interrupt_precision << "\n";
  std::cout << "interrupt_recall=" << interrupt_recall << "\n";
  std::cout << "interrupt_false_positive_rate=" << interrupt_false_positive_rate
            << "\n";
  std::cout << "interrupt_false_negative_rate=" << interrupt_false_negative_rate
            << "\n";
  std::cout << "interrupt_true_positive=" << stats.interrupt_semantic_true_positive
            << "\n";
  std::cout << "interrupt_false_positive=" << stats.interrupt_semantic_false_positive
            << "\n";
  std::cout << "interrupt_false_negative=" << stats.interrupt_semantic_false_negative
            << "\n";
  std::cout << "interrupt_abort_count=" << stats.interrupt_abort_count << "\n";
  std::cout << "interrupt_abort_rate=" << interrupt_abort_rate << "\n";
  std::cout << "boundary_at_rate=" << boundary_at_rate << "\n";
  std::cout << "boundary_score_pass_rate=" << boundary_score_pass_rate << "\n";
  std::cout << "interrupt_gate_fail_no_candidates="
            << stats.interrupt_gate_fail_no_candidates << "\n";
  std::cout << "interrupt_gate_fail_no_candidates_rate="
            << interrupt_gate_fail_no_candidates_rate << "\n";
  std::cout << "interrupt_gate_fail_no_store="
            << stats.interrupt_gate_fail_no_store << "\n";
  std::cout << "interrupt_gate_fail_no_store_rate="
            << interrupt_gate_fail_no_store_rate << "\n";
  std::cout << "interrupt_gate_fail_rel="
            << stats.interrupt_gate_fail_rel << "\n";
  std::cout << "interrupt_gate_fail_rel_rate="
            << interrupt_gate_fail_rel_rate << "\n";
  std::cout << "interrupt_gate_fail_novelty="
            << stats.interrupt_gate_fail_novelty << "\n";
  std::cout << "interrupt_gate_fail_novelty_rate="
            << interrupt_gate_fail_novelty_rate << "\n";
  std::cout << "interrupt_gate_fail_mu="
            << stats.interrupt_gate_fail_mu << "\n";
  std::cout << "interrupt_gate_fail_mu_rate="
            << interrupt_gate_fail_mu_rate << "\n";
  std::cout << "interrupt_gate_fail_novelty_mu="
            << stats.interrupt_gate_fail_novelty_mu << "\n";
  std::cout << "interrupt_gate_fail_novelty_mu_rate="
            << interrupt_gate_fail_novelty_mu_rate << "\n";
  std::cout << "interrupt_gate_fail_dup="
            << stats.interrupt_gate_fail_dup << "\n";
  std::cout << "interrupt_gate_fail_dup_rate="
            << interrupt_gate_fail_dup_rate << "\n";
  std::cout << "interrupt_gate_fail_boundary_mu="
            << stats.interrupt_gate_fail_boundary_mu << "\n";
  std::cout << "interrupt_gate_fail_boundary_mu_rate="
            << interrupt_gate_fail_boundary_mu_rate << "\n";

  if (!op_time_means.empty()) {
    std::cout << "\n=== Operation Timings (mean ms) ===\n";
    const size_t max_ops = std::min<size_t>(15, op_time_means.size());
    for (size_t i = 0; i < max_ops; ++i) {
      std::cout << "op_time_mean[" << op_time_means[i].first << "]="
                << op_time_means[i].second << "\n";
    }
  }

  std::cout << "\n=== Distributions ===\n";
  std::cout << "novelty_mean=" << novelty_summary.mean
            << " p10=" << novelty_summary.p10
            << " p50=" << novelty_summary.p50
            << " p90=" << novelty_summary.p90 << "\n";
  std::cout << "relevance_mean=" << relevance_summary.mean
            << " p10=" << relevance_summary.p10
            << " p50=" << relevance_summary.p50
            << " p90=" << relevance_summary.p90 << "\n";
  std::cout << "surprise_mean=" << surprise_summary.mean
            << " p10=" << surprise_summary.p10
            << " p50=" << surprise_summary.p50
            << " p90=" << surprise_summary.p90 << "\n";
  if (emotion_intensity_summary.count > 0) {
    std::cout << "emotion_intensity_mean=" << emotion_intensity_summary.mean
              << " p10=" << emotion_intensity_summary.p10
              << " p50=" << emotion_intensity_summary.p50
              << " p90=" << emotion_intensity_summary.p90 << "\n";
  }
  if (arousal_summary.count > 0) {
    std::cout << "arousal_mean=" << arousal_summary.mean
              << " p10=" << arousal_summary.p10
              << " p50=" << arousal_summary.p50
              << " p90=" << arousal_summary.p90 << "\n";
  }
  if (valence_summary.count > 0) {
    std::cout << "valence_mean=" << valence_summary.mean
              << " p10=" << valence_summary.p10
              << " p50=" << valence_summary.p50
              << " p90=" << valence_summary.p90 << "\n";
  }
  if (salience_summary.count > 0) {
    std::cout << "salience_mean=" << salience_summary.mean
              << " p10=" << salience_summary.p10
              << " p50=" << salience_summary.p50
              << " p90=" << salience_summary.p90 << "\n";
  }
  if (affect_summary.count > 0) {
    std::cout << "affect_drive_mean=" << affect_summary.mean
              << " p10=" << affect_summary.p10
              << " p50=" << affect_summary.p50
              << " p90=" << affect_summary.p90 << "\n";
  }
  if (interrupt_affect_summary.count > 0) {
    std::cout << "interrupt_affect_drive_mean=" << interrupt_affect_summary.mean
              << " p10=" << interrupt_affect_summary.p10
              << " p50=" << interrupt_affect_summary.p50
              << " p90=" << interrupt_affect_summary.p90 << "\n";
  }
  if (non_interrupt_affect_summary.count > 0) {
    std::cout << "non_interrupt_affect_drive_mean="
              << non_interrupt_affect_summary.mean
              << " p10=" << non_interrupt_affect_summary.p10
              << " p50=" << non_interrupt_affect_summary.p50
              << " p90=" << non_interrupt_affect_summary.p90 << "\n";
  }
  if (interrupt_gate_affect_summary.count > 0) {
    std::cout << "interrupt_gate_affect_drive_mean="
              << interrupt_gate_affect_summary.mean
              << " p10=" << interrupt_gate_affect_summary.p10
              << " p50=" << interrupt_gate_affect_summary.p50
              << " p90=" << interrupt_gate_affect_summary.p90 << "\n";
  }
  if (interrupt_gate_affect_interrupt_summary.count > 0) {
    std::cout << "interrupt_gate_affect_drive_interrupt_mean="
              << interrupt_gate_affect_interrupt_summary.mean
              << " p10=" << interrupt_gate_affect_interrupt_summary.p10
              << " p50=" << interrupt_gate_affect_interrupt_summary.p50
              << " p90=" << interrupt_gate_affect_interrupt_summary.p90 << "\n";
  }
  if (interrupt_gate_affect_non_interrupt_summary.count > 0) {
    std::cout << "interrupt_gate_affect_drive_non_interrupt_mean="
              << interrupt_gate_affect_non_interrupt_summary.mean
              << " p10=" << interrupt_gate_affect_non_interrupt_summary.p10
              << " p50=" << interrupt_gate_affect_non_interrupt_summary.p50
              << " p90=" << interrupt_gate_affect_non_interrupt_summary.p90 << "\n";
  }
  if (interrupt_gate_retrieval_thresh_summary.count > 0) {
    std::cout << "interrupt_gate_retrieval_thresh_mean="
              << interrupt_gate_retrieval_thresh_summary.mean
              << " p10=" << interrupt_gate_retrieval_thresh_summary.p10
              << " p50=" << interrupt_gate_retrieval_thresh_summary.p50
              << " p90=" << interrupt_gate_retrieval_thresh_summary.p90 << "\n";
  }
  if (interrupt_gate_boundary_mult_summary.count > 0) {
    std::cout << "interrupt_gate_boundary_mult_eff_mean="
              << interrupt_gate_boundary_mult_summary.mean
              << " p10=" << interrupt_gate_boundary_mult_summary.p10
              << " p50=" << interrupt_gate_boundary_mult_summary.p50
              << " p90=" << interrupt_gate_boundary_mult_summary.p90 << "\n";
  }
  if (boundary_score_summary.count > 0) {
    std::cout << "boundary_score_mean=" << boundary_score_summary.mean
              << " p10=" << boundary_score_summary.p10
              << " p50=" << boundary_score_summary.p50
              << " p90=" << boundary_score_summary.p90 << "\n";
  }
  std::cout << "retrieval_overlap_mean=" << retrieval_overlap_summary.mean
            << " p10=" << retrieval_overlap_summary.p10
            << " p50=" << retrieval_overlap_summary.p50
            << " p90=" << retrieval_overlap_summary.p90 << "\n";
  if (retrieval_context_overlap_summary.count > 0) {
    std::cout << "retrieval_context_overlap_mean="
              << retrieval_context_overlap_summary.mean
              << " p10=" << retrieval_context_overlap_summary.p10
              << " p50=" << retrieval_context_overlap_summary.p50
              << " p90=" << retrieval_context_overlap_summary.p90 << "\n";
  }
  if (retrieval_semantic_summary.count > 0) {
    std::cout << "retrieval_semantic_overlap_mean="
              << retrieval_semantic_summary.mean
              << " p10=" << retrieval_semantic_summary.p10
              << " p50=" << retrieval_semantic_summary.p50
              << " p90=" << retrieval_semantic_summary.p90 << "\n";
  }
  if (retrieval_context_semantic_summary.count > 0) {
    std::cout << "retrieval_context_semantic_overlap_mean="
              << retrieval_context_semantic_summary.mean
              << " p10=" << retrieval_context_semantic_summary.p10
              << " p50=" << retrieval_context_semantic_summary.p50
              << " p90=" << retrieval_context_semantic_summary.p90 << "\n";
  }
  if (retrieval_summary_hit_overlap_summary.count > 0) {
    std::cout << "summary_hit_overlap_mean="
              << retrieval_summary_hit_overlap_summary.mean
              << " p10=" << retrieval_summary_hit_overlap_summary.p10
              << " p50=" << retrieval_summary_hit_overlap_summary.p50
              << " p90=" << retrieval_summary_hit_overlap_summary.p90 << "\n";
  }
  if (retrieval_emotion_bonus_summary.count > 0) {
    std::cout << "retrieval_emotion_bonus_mean="
              << retrieval_emotion_bonus_summary.mean
              << " p10=" << retrieval_emotion_bonus_summary.p10
              << " p50=" << retrieval_emotion_bonus_summary.p50
              << " p90=" << retrieval_emotion_bonus_summary.p90 << "\n";
  }
  if (interrupt_retrieval_emotion_bonus_summary.count > 0) {
    std::cout << "interrupt_retrieval_emotion_bonus_mean="
              << interrupt_retrieval_emotion_bonus_summary.mean
              << " p10=" << interrupt_retrieval_emotion_bonus_summary.p10
              << " p50=" << interrupt_retrieval_emotion_bonus_summary.p50
              << " p90=" << interrupt_retrieval_emotion_bonus_summary.p90 << "\n";
  }

  if (interrupt_overlap_summary.count > 0) {
    std::cout << "\n=== Interrupt Quality ===\n";
    std::cout << "interrupt_overlap_mean=" << interrupt_overlap_summary.mean
              << " p10=" << interrupt_overlap_summary.p10
              << " p50=" << interrupt_overlap_summary.p50
              << " p90=" << interrupt_overlap_summary.p90 << "\n";
    if (interrupt_context_overlap_summary.count > 0) {
      std::cout << "interrupt_context_overlap_mean="
                << interrupt_context_overlap_summary.mean
                << " p10=" << interrupt_context_overlap_summary.p10
                << " p50=" << interrupt_context_overlap_summary.p50
                << " p90=" << interrupt_context_overlap_summary.p90 << "\n";
    }
    if (interrupt_semantic_summary.count > 0) {
      std::cout << "interrupt_semantic_overlap_mean="
                << interrupt_semantic_summary.mean
                << " p10=" << interrupt_semantic_summary.p10
                << " p50=" << interrupt_semantic_summary.p50
                << " p90=" << interrupt_semantic_summary.p90 << "\n";
    }
    if (interrupt_context_semantic_summary.count > 0) {
      std::cout << "interrupt_context_semantic_overlap_mean="
                << interrupt_context_semantic_summary.mean
                << " p10=" << interrupt_context_semantic_summary.p10
                << " p50=" << interrupt_context_semantic_summary.p50
                << " p90=" << interrupt_context_semantic_summary.p90 << "\n";
    }
    if (non_interrupt_semantic_summary.count > 0) {
      std::cout << "non_interrupt_semantic_overlap_mean="
                << non_interrupt_semantic_summary.mean
                << " p10=" << non_interrupt_semantic_summary.p10
                << " p50=" << non_interrupt_semantic_summary.p50
                << " p90=" << non_interrupt_semantic_summary.p90 << "\n";
    }
    if (non_interrupt_context_semantic_summary.count > 0) {
      std::cout << "non_interrupt_context_semantic_overlap_mean="
                << non_interrupt_context_semantic_summary.mean
                << " p10=" << non_interrupt_context_semantic_summary.p10
                << " p50=" << non_interrupt_context_semantic_summary.p50
                << " p90=" << non_interrupt_context_semantic_summary.p90 << "\n";
    }
    std::cout << "interrupt_semantic_delta=" << interrupt_semantic_delta << "\n";
    std::cout << "interrupt_context_semantic_delta="
              << interrupt_context_semantic_delta << "\n";
    std::cout << "interrupt_novelty_mean=" << interrupt_novelty_summary.mean
              << " p10=" << interrupt_novelty_summary.p10
              << " p50=" << interrupt_novelty_summary.p50
              << " p90=" << interrupt_novelty_summary.p90 << "\n";
    std::cout << "interrupt_relevance_mean=" << interrupt_relevance_summary.mean
              << " p10=" << interrupt_relevance_summary.p10
              << " p50=" << interrupt_relevance_summary.p50
              << " p90=" << interrupt_relevance_summary.p90 << "\n";
    std::cout << "interrupt_surprise_mean=" << interrupt_surprise_summary.mean
              << " p10=" << interrupt_surprise_summary.p10
              << " p50=" << interrupt_surprise_summary.p50
              << " p90=" << interrupt_surprise_summary.p90 << "\n";
  }

  if (!stats.low_overlap_examples.empty()) {
    std::cout << "\n=== Lowest-overlap retrieval examples ===\n";
    for (const auto &ex : stats.low_overlap_examples) {
      std::cout << "overlap=" << ex.overlap << "\n";
      std::cout << "turn: " << ex.turn_text << "\n";
      std::string mem = ex.memory_text;
      if (mem.size() > 200) {
        mem = mem.substr(0, 200) + "...";
      }
      std::cout << "retrieved: " << mem << "\n\n";
    }
  }

  return 0;
}
