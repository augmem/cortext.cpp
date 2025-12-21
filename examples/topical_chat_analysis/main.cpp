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

struct Stats {
  int conversations = 0;
  int turns = 0;
  int writes = 0;
  int retrieval_turns = 0;
  int total_retrieved = 0;
  int wm_samples = 0;
  int wm_total_slots = 0;
  int wm_max_slots = 0;
  int signals_since_write = 0;
  std::vector<int> signals_per_write;
  double retrieval_overlap_sum = 0.0;
  int retrieval_overlap_count = 0;
  double wm_overlap_sum = 0.0;
  int wm_overlap_count = 0;
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
    int turn_idx = 0;
    for (const auto &turn : conv["content"]) {
      if (stats.turns >= cfg.max_total_turns) {
        break;
      }
      if (turn_idx >= cfg.max_turns_per_conversation) {
        break;
      }

      std::string agent = turn.value("agent", "agent");
      std::string message = turn.value("message", "");
      std::string text = agent + ": " + message;

      const std::string source_id = "chat/" + agent;
      cortext::Cortext::Context ctx;
      try {
        ctx = cortext->ProcessText(text, source_id);
      } catch (const std::exception &e) {
        std::cerr << "ProcessText failed: " << e.what() << "\n";
        return 1;
      }

      stats.turns++;
      turn_idx++;

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

      const auto turn_tokens = Tokenize(text);
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

      if (!ctx.retrieved_memory.empty()) {
        stats.retrieval_turns++;
        stats.total_retrieved += static_cast<int>(ctx.retrieved_memory.size());
        double best_overlap = 0.0;
        std::string best_text;
        for (const auto &mem : ctx.retrieved_memory) {
          const std::string mem_text = MemoryToText(mem);
          const auto mem_tokens = Tokenize(mem_text);
          const double overlap = Jaccard(turn_tokens, mem_tokens);
          if (overlap > best_overlap) {
            best_overlap = overlap;
            best_text = mem_text;
          }
        }
        stats.retrieval_overlap_sum += best_overlap;
        stats.retrieval_overlap_count++;
        if (!best_text.empty()) {
          RecordRetrievalExample(stats, best_overlap, text, best_text);
        }
      }

      if (turn_idx % 20 == 0) {
        std::cout << "  turns=" << turn_idx
                  << " writes=" << stats.writes
                  << " wm_slots=" << ctx.working_memory.size()
                  << " retrieved=" << ctx.retrieved_memory.size()
                  << "\n";
      }

      if (cfg.cadence_enabled) {
        const int words = std::max(1, CountWords(message));
        const double wps = std::max(1e-3, cfg.cadence_wpm / 60.0);
        double delay_s = static_cast<double>(words) / wps;
        delay_s += 0.35; // conversational pause
        delay_s *= jitter_dist(rng);
        if (cfg.cadence_speed > 0.0) {
          delay_s /= cfg.cadence_speed;
        }
        int delay_ms = static_cast<int>(delay_s * 1000.0);
        delay_ms = std::max(cfg.cadence_min_ms, delay_ms);
        delay_ms = std::min(cfg.cadence_max_ms, delay_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      }
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
