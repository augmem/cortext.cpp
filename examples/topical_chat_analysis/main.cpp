#include <cortext/cortext.hpp>
#include <cortext/telemetry/telemetry.hpp>

#include <nlohmann/json.hpp>

#include <opentelemetry/logs/provider.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/sdk/logs/exporter.h>
#include <opentelemetry/sdk/logs/logger_provider.h>
#include <opentelemetry/sdk/logs/logger_provider_factory.h>
#include <opentelemetry/sdk/logs/read_write_log_record.h>
#include <opentelemetry/sdk/logs/simple_log_record_processor_factory.h>
#include <opentelemetry/sdk/resource/resource.h>

#include <algorithm>
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

class SimpleStdoutLogExporter final
    : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  SimpleStdoutLogExporter(const std::string &log_file_path,
                          std::vector<std::string> filters)
      : log_file_path_(log_file_path), filters_(std::move(filters)) {
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
};

void InstallOtelLogger(const std::string &log_path,
                       std::vector<std::string> filters) {
  namespace logs_sdk = opentelemetry::sdk::logs;
  namespace resource_sdk = opentelemetry::sdk::resource;
  auto resource = resource_sdk::Resource::Create({{"service.name", "cortext_topical_chat_analysis"}});
  auto exporter = std::unique_ptr<logs_sdk::LogRecordExporter>(
      new SimpleStdoutLogExporter(log_path, std::move(filters)));
  auto processor =
      logs_sdk::SimpleLogRecordProcessorFactory::Create(std::move(exporter));
  auto provider = opentelemetry::nostd::shared_ptr<opentelemetry::logs::LoggerProvider>(
      new logs_sdk::LoggerProvider(std::move(processor), resource));
  opentelemetry::logs::Provider::SetLoggerProvider(provider);
}

struct AnalysisConfig {
  std::string dataset_path;
  std::string models_dir = "models";
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
  bool cadence_enabled = true;
  double cadence_speed = 12.0; // >1.0 speeds up; 12x keeps runs reasonable
  double cadence_wpm = 150.0;
  double cadence_jitter = 0.15;
  int cadence_min_ms = 80;
  int cadence_max_ms = 1200;
  int context_window_turns = 4;
  unsigned int seed = 1337;
  double focus = 0.3;
  double sensitivity = 0.6;
  double stability = 0.5;
};

struct RetrievalExample {
  double overlap = 0.0;
  std::string turn_text;
  std::string memory_text;
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
  int retrieval_turns = 0;
  int total_retrieved = 0;
  int interrupt_turns = 0;
  int interrupt_turns_with_retrieval = 0;
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
  std::vector<double> interrupt_overlap_values;
  std::vector<double> interrupt_context_overlap_values;
  std::vector<double> interrupt_novelty_values;
  std::vector<double> interrupt_relevance_values;
  std::vector<double> interrupt_surprise_values;
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
    } else if (auto v = take("--seed=")) {
      cfg.seed = static_cast<unsigned int>(std::stoul(*v));
    } else if (arg == "--no-cadence") {
      cfg.cadence_enabled = false;
    } else if (arg == "--reuse") {
      cfg.reset_per_conversation = false;
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
  if (!std::filesystem::exists(cfg.dataset_path)) {
    std::cerr << "Dataset not found at: " << cfg.dataset_path << "\n";
    return 1;
  }

  InstallOtelLogger(cfg.otel_log_path, cfg.otel_filters);

  std::ifstream in(cfg.dataset_path);
  if (!in) {
    std::cerr << "Failed to open dataset: " << cfg.dataset_path << "\n";
    return 1;
  }

  std::unique_ptr<cortext::Cortext> cortext;

  Stats stats;
  std::deque<std::unordered_set<std::string>> recent_context_tokens;
  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<double> jitter_dist(
      1.0 - cfg.cadence_jitter, 1.0 + cfg.cadence_jitter);
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

    if (!cortext || cfg.reset_per_conversation) {
      cortext::Cortext::Config c;
      c.focus = cfg.focus;
      c.sensitivity = cfg.sensitivity;
      c.stability = cfg.stability;
      cortext = cortext::Cortext::Create(c, ":memory:", cfg.models_dir);
    }

    std::cout << "\n=== Conversation " << conv_id << " ===\n";
    stats.conversations++;
    recent_context_tokens.clear();
    int turn_idx = 0;
    bool stop_all = false;
    for (const auto &turn : conv["content"]) {
      if (turn_idx >= cfg.max_turns_per_conversation) {
        break;
      }

      std::string agent = turn.value("agent", "agent");
      std::string message = turn.value("message", "");
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
        const auto turn_tokens = Tokenize(chunk);
        const auto context_tokens = MergeTokenWindow(recent_context_tokens);
        cortext::Cortext::Context ctx;
        try {
          ctx = cortext->ProcessText(chunk, source_id);
        } catch (const std::exception &e) {
          std::cerr << "ProcessText failed: " << e.what() << "\n";
          return 1;
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

        if (!ctx.working_memory.empty()) {
          double best_overlap = 0.0;
          for (const auto &mem : ctx.working_memory) {
            const std::string mem_text = MemoryToText(mem);
            const auto mem_tokens = Tokenize(mem_text);
            best_overlap = std::max(best_overlap, Jaccard(turn_tokens, mem_tokens));
          }
          stats.wm_overlap_sum += best_overlap;
          stats.wm_overlap_count++;
        }

        if (auto relevance = GetMetric(ctx, cortext::operations::Metric::relevance)) {
          stats.relevance_values.push_back(*relevance);
          if (ctx.should_interrupt) {
            stats.interrupt_relevance_values.push_back(*relevance);
          }
        }
        if (auto surprise = GetMetric(ctx, cortext::operations::Metric::surprise)) {
          stats.surprise_values.push_back(*surprise);
          if (ctx.should_interrupt) {
            stats.interrupt_surprise_values.push_back(*surprise);
          }
        }
        if (auto mismatch = GetMetric(ctx, cortext::operations::Metric::mismatch)) {
          const double denom = (1.0 - cfg.focus) * cfg.sensitivity;
          double novelty_est = denom > 1e-6 ? (*mismatch / denom) : *mismatch;
          novelty_est = Clamp01(novelty_est);
          stats.novelty_values.push_back(novelty_est);
          if (ctx.should_interrupt) {
            stats.interrupt_novelty_values.push_back(novelty_est);
          }
        }

        if (ctx.should_interrupt) {
          stats.interrupt_turns++;
          if (!ctx.retrieved_memory.empty()) {
            stats.interrupt_turns_with_retrieval++;
          }
        }

        if (!ctx.retrieved_memory.empty()) {
          stats.retrieval_turns++;
          stats.total_retrieved += static_cast<int>(ctx.retrieved_memory.size());
          double best_overlap = 0.0;
          double best_context_overlap = 0.0;
          std::string best_text;
          for (const auto &mem : ctx.retrieved_memory) {
            const std::string mem_text = MemoryToText(mem);
            const auto mem_tokens = Tokenize(mem_text);
            const double overlap = Jaccard(turn_tokens, mem_tokens);
            if (overlap > best_overlap) {
              best_overlap = overlap;
              best_text = mem_text;
            }
            if (!context_tokens.empty()) {
              const double ctx_overlap = Jaccard(context_tokens, mem_tokens);
              best_context_overlap = std::max(best_context_overlap, ctx_overlap);
            }
          }
          stats.retrieval_overlap_sum += best_overlap;
          stats.retrieval_overlap_count++;
          stats.retrieval_overlap_values.push_back(best_overlap);
          if (!context_tokens.empty()) {
            stats.retrieval_context_overlap_values.push_back(best_context_overlap);
          }
          if (ctx.should_interrupt) {
            stats.interrupt_overlap_values.push_back(best_overlap);
            if (!context_tokens.empty()) {
              stats.interrupt_context_overlap_values.push_back(best_context_overlap);
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

        if (cfg.cadence_enabled) {
          const int words = std::max(1, CountWords(chunk));
          double delay_s = static_cast<double>(words) / wps;
          delay_s += 0.1;
          delay_s *= jitter_dist(rng);
          if (cfg.cadence_speed > 0.0) {
            delay_s /= cfg.cadence_speed;
          }
          int delay_ms = static_cast<int>(delay_s * 1000.0);
          delay_ms = std::max(cfg.cadence_min_ms, delay_ms);
          delay_ms = std::min(cfg.cadence_max_ms, delay_ms);
          std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }

        if (cfg.context_window_turns > 0) {
          recent_context_tokens.push_back(turn_tokens);
          while (static_cast<int>(recent_context_tokens.size()) > cfg.context_window_turns) {
            recent_context_tokens.pop_front();
          }
        }
      }
      turn_idx++;
      if (stop_all) {
        break;
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

    if (stats.conversations >= cfg.max_conversations) {
      break;
    }
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

  const auto novelty_summary = Summarize(stats.novelty_values);
  const auto relevance_summary = Summarize(stats.relevance_values);
  const auto surprise_summary = Summarize(stats.surprise_values);
  const auto retrieval_overlap_summary = Summarize(stats.retrieval_overlap_values);
  const auto retrieval_context_overlap_summary =
      Summarize(stats.retrieval_context_overlap_values);
  const auto interrupt_overlap_summary = Summarize(stats.interrupt_overlap_values);
  const auto interrupt_context_overlap_summary =
      Summarize(stats.interrupt_context_overlap_values);
  const auto interrupt_novelty_summary = Summarize(stats.interrupt_novelty_values);
  const auto interrupt_relevance_summary = Summarize(stats.interrupt_relevance_values);
  const auto interrupt_surprise_summary = Summarize(stats.interrupt_surprise_values);

  std::cout << "\n=== Summary ===\n";
  std::cout << "conversations=" << stats.conversations << "\n";
  std::cout << "turns=" << stats.turns << "\n";
  std::cout << "writes=" << stats.writes << "\n";
  std::cout << "avg_signals_per_write=" << avg_signals_per_write << "\n";
  std::cout << "wm_avg_slots=" << avg_wm_slots << "\n";
  std::cout << "wm_max_slots=" << stats.wm_max_slots << "\n";
  std::cout << "retrieval_turn_rate=" << retrieval_rate << "\n";
  std::cout << "retrieval_avg_candidates=" << avg_retrieval_count << "\n";
  std::cout << "retrieval_best_overlap=" << avg_retrieval_overlap << "\n";
  std::cout << "wm_best_overlap=" << avg_wm_overlap << "\n";
  std::cout << "interrupt_turn_rate=" << interrupt_rate << "\n";
  std::cout << "interrupt_with_retrieval_rate=" << interrupt_with_retrieval_rate << "\n";

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
