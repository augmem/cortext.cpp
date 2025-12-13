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
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

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
class LogBuffer {
public:
  void Append(const std::string& line) {
    std::lock_guard<std::mutex> lock(mu_);
    lines_.push_back(line);
    if (lines_.size() > kMaxLines) {
      lines_.pop_front();
    }
  }

  std::vector<std::string> GetLines() const {
    std::lock_guard<std::mutex> lock(mu_);
    return std::vector<std::string>(lines_.begin(), lines_.end());
  }

private:
  static constexpr size_t kMaxLines = 1000;
  mutable std::mutex mu_;
  std::deque<std::string> lines_;
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
  encoder.EncodeText(text, emb);
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
  return row;
}

std::string FormatMemoriesForSystemPrompt(
    const std::vector<cortext::Cortext::Context::Memory>& memories) {
  if (memories.empty()) return {};

  std::ostringstream oss;
  int i = 1;
  for (const auto& m : memories) {
    oss << "[Memory " << i++ << "] (id=" << m.id;
    if (!m.source_id.empty()) oss << ", source_id=" << m.source_id;
    if (m.timestamp != 0) oss << ", ts=" << m.timestamp;
    oss << ")\n";
    if (!m.modality.empty() || !m.mimetype.empty()) {
      oss << "meta: modality=" << (m.modality.empty() ? "?" : m.modality)
          << ", mime=" << (m.mimetype.empty() ? "?" : m.mimetype) << "\n";
    }
    // Content may include binary data. Render as a safe preview.
    std::string c = m.content;
    if (c.size() > 800) c = c.substr(0, 800) + "...";
    oss << c << "\n\n";
  }
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

} // namespace

int main(int argc, char** argv) {
  // Shared log buffer for logs view.
  auto log_buffer = std::make_shared<LogBuffer>();
  log_buffer->Append(
      "Telemetry: cortext instruments OpenTelemetry via global providers. "
      "This example does not configure SDK exporters by default.");
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
  std::vector<ChatMessage> history;
  std::vector<cortext::Cortext::Context::Memory> last_memories;
  std::deque<MemoryEvent> memory_events;
  bool last_should_interrupt = false;
  std::optional<std::string> last_error;
  bool generating = false;
  std::string input;
  int tab_selected = 0;
  static constexpr size_t kMaxMemoryEvents = 50;

  // Seed greeting.
  history.push_back({"system", "cortext_chat: Tab to switch views | Enter to send | Ctrl+C to quit"});

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

    return ftxui::vbox(std::move(lines)) | ftxui::yframe | ftxui::vscroll_indicator;
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
        
        // Content preview
        std::string preview = evt.content;
        if (preview.size() > 120) preview = preview.substr(0, 120) + "...";
        lines.push_back(ftxui::text("  \"" + preview + "\"") | ftxui::color(ftxui::Color::White));
      lines.push_back(ftxui::separator());
      }
    }

    return ftxui::vbox(std::move(lines)) | ftxui::yframe | ftxui::vscroll_indicator;
  };

  auto render_logs = [&, log_buffer] {
    ftxui::Elements lines;
    lines.push_back(ftxui::text("System Logs (OpenTelemetry)") | ftxui::bold);
    lines.push_back(ftxui::separator());

    auto log_lines = log_buffer->GetLines();
    if (log_lines.empty()) {
      lines.push_back(ftxui::text("(no logs yet)") | ftxui::color(ftxui::Color::GrayDark));
    } else {
      for (const auto& line : log_lines) {
        auto color = ftxui::Color::White;
        if (line.find("ERROR") != std::string::npos || line.find("error") != std::string::npos) {
          color = ftxui::Color::Red;
        } else if (line.find("WARN") != std::string::npos || line.find("warning") != std::string::npos) {
          color = ftxui::Color::Yellow;
        } else if (line.find("INFO") != std::string::npos) {
          color = ftxui::Color::Cyan;
        }
        lines.push_back(ftxui::text(line) | ftxui::color(color));
      }
    }

    return ftxui::vbox(std::move(lines)) | ftxui::yframe | ftxui::vscroll_indicator;
  };

  auto input_component = ftxui::Input(&input, "Type a message...");

  std::vector<std::string> tab_names = {"Chat", "Memory", "Logs"};
  auto tab_toggle = ftxui::Toggle(&tab_names, &tab_selected);

  auto tab_container = ftxui::Container::Tab(
    {
      ftxui::Renderer(render_chat),
      ftxui::Renderer(render_memory),
      ftxui::Renderer(render_logs),
    },
    &tab_selected
  );

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
    return ftxui::vbox({tab_bar, content, input_box});
  });

  // Event handling.
  auto app = ftxui::CatchEvent(root_renderer, [&] (const ftxui::Event& e) {
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
          // Ensure cortext sees latest externally persisted rows by refreshing its episode.
          cortext_ctx->Flush();
          retrieved = cortext_ctx->ProcessText(text, ts, "chat/user");
          
          // Create memory events for retrieved memories
          {
            std::lock_guard<std::mutex> lock(mu);
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
          if (!injected_system.empty()) {
            injected_system = std::string("## Relevant Context from Memory\n\n")
                              + injected_system
                              + "Use the above context to inform your response when relevant.";
          }
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

          auto resp = openai::chat().create(req);
          assistant_reply = resp["choices"][0]["message"]["content"].get<std::string>();
        } catch (const std::exception& ex) {
          assistant_reply = std::string("(OpenAI error: ") + ex.what() + ")";
          std::lock_guard<std::mutex> lock(mu);
          last_error = std::string("openai: ") + ex.what();
        }

        // Persist both user and assistant turns so future retrieval can find them.
        try {
          auto uniq = cortext::SQLiteStore::Create(db_path.string());
          auto store = std::shared_ptr<cortext::Store>(std::move(uniq));

          auto user_row = PersistTextMemory(*store, encoder, text, "chat/user", ts);
          auto asst_row = PersistTextMemory(*store, encoder, assistant_reply, "chat/assistant", ts + 1);

          // Create memory events for stored memories
          {
            std::lock_guard<std::mutex> lock(mu);
            using M = cortext::operations::Metric;
            
            MemoryEvent user_evt;
            user_evt.type = MemoryEventType::STORED;
            user_evt.memory_id = static_cast<int>(user_row.embedding_id);
            user_evt.content = text;
            user_evt.source_id = "chat/user";
            user_evt.timestamp = ts;
            user_evt.composite_score = retrieved.output.composite_score;
            user_evt.threshold = retrieved.output.threshold;
            user_evt.decision = retrieved.output.composite_score.has_value() && retrieved.output.threshold.has_value()
                ? std::optional<bool>((*retrieved.output.composite_score) > (*retrieved.output.threshold))
                : std::nullopt;
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
            asst_evt.memory_id = static_cast<int>(asst_row.embedding_id);
            asst_evt.content = assistant_reply;
            asst_evt.source_id = "chat/assistant";
            asst_evt.timestamp = ts + 1;
            memory_events.push_back(asst_evt);
            
            while (memory_events.size() > kMaxMemoryEvents) {
              memory_events.pop_front();
            }
          }

          // Refresh cortext episode snapshot after external writes.
          cortext_ctx->Flush();
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

    if (e == ftxui::Event::Custom) {
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

  screen.Loop(app);
  running = false;
  if (refresh_thread.joinable()) {
    refresh_thread.join();
  }

  try {
    cortext_ctx->Flush();
  } catch (...) {
  }

  return 0;
}
