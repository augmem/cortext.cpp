#include <cortext/cortext.hpp>
#include <cortext/encoder/imagebind.hpp>
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
#include <exception>
#include <filesystem>
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
  // Enable local OTel file export by default for this example, without
  // hardcoding any collector endpoints. Users can override via environment.
  std::string default_otel_file_path;
  if (!HasEnv("CORTEXT_OTEL_FILE_PATH") && !HasEnv("OTEL_TRACES_EXPORTER") &&
      !HasEnv("OTEL_METRICS_EXPORTER") && !HasEnv("OTEL_LOGS_EXPORTER")) {
    try {
      const auto dir = std::filesystem::temp_directory_path();
      const auto p = dir / "cortext_chat_otel.log";
      default_otel_file_path = p.string();
      setenv("CORTEXT_OTEL_FILE_PATH", p.string().c_str(), 1);
      if (!HasEnv("CORTEXT_OTEL_FILE_SIGNALS")) {
        setenv("CORTEXT_OTEL_FILE_SIGNALS", "traces,metrics,logs", 1);
      }
      if (!HasEnv("OTEL_SERVICE_NAME")) {
        setenv("OTEL_SERVICE_NAME", "cortext_chat", 1);
      }
    } catch (...) {
    }
  }
  const bool telemetry_ok = cortext::telemetry::InitializeFromEnv();
  if (!telemetry_ok) {
    std::cerr << "Telemetry disabled (build without CORTEXT_ENABLE_OTEL=ON).\n";
  } else if (!default_otel_file_path.empty()) {
    std::cerr << "Telemetry enabled: writing to " << default_otel_file_path
              << "\n";
  }
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
  bool last_should_interrupt = false;
  std::optional<std::string> last_error;
  bool generating = false;
  std::string input;

  // Seed greeting.
  history.push_back({"system", "cortext_chat: Enter to send | Ctrl+C to quit"});

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

  auto render_memory = [&] {
    ftxui::Elements lines;
    std::lock_guard<std::mutex> lock(mu);

    lines.push_back(ftxui::text("Memory / Interrupt") | ftxui::bold);
    lines.push_back(ftxui::separator());

    lines.push_back(ftxui::text(std::string("should_interrupt: ") + (last_should_interrupt ? "true" : "false"))
                    | ftxui::color(last_should_interrupt ? ftxui::Color::Red : ftxui::Color::Green));

    if (last_error.has_value()) {
      lines.push_back(ftxui::separator());
      lines.push_back(ftxui::text("Error: " + *last_error) | ftxui::color(ftxui::Color::Red));
    }

    lines.push_back(ftxui::separator());
    lines.push_back(ftxui::text("Retrieved memories:") | ftxui::bold);

    int shown = 0;
    for (const auto& m : last_memories) {
      if (shown++ >= 6) break;
      std::string preview = m.content;
      if (preview.size() > 200) preview = preview.substr(0, 200) + "...";
      std::ostringstream hdr;
      hdr << "#" << m.id;
      if (!m.source_id.empty()) hdr << " (" << m.source_id << ")";
      lines.push_back(ftxui::text(hdr.str()) | ftxui::color(ftxui::Color::Blue));
      lines.push_back(ftxui::paragraph(preview) | ftxui::color(ftxui::Color::White));
      lines.push_back(ftxui::separator());
    }

    if (last_memories.empty()) {
      lines.push_back(ftxui::text("(none)") | ftxui::color(ftxui::Color::GrayDark));
    }

    return ftxui::vbox(std::move(lines)) | ftxui::yframe | ftxui::vscroll_indicator;
  };

  auto input_component = ftxui::Input(&input, "Type a message...");

  auto root = ftxui::Container::Vertical({
      ftxui::Container::Horizontal({
          ftxui::Renderer(render_chat),
          ftxui::Renderer(render_memory),
      }),
      input_component,
  });

  // Make panes look nicer.
  auto root_renderer = ftxui::Renderer(root, [&] {
    auto left = render_chat() | ftxui::border | ftxui::flex;
    auto right = render_memory() | ftxui::border | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, 50);

    auto main = ftxui::hbox({left, right});
    auto input_box = ftxui::hbox({ftxui::text(" > "), input_component->Render()}) | ftxui::border;
    return ftxui::vbox({main | ftxui::flex, input_box});
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

          (void)PersistTextMemory(*store, encoder, text, "chat/user", ts);
          (void)PersistTextMemory(*store, encoder, assistant_reply, "chat/assistant", ts);

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

  screen.Loop(app);

  try {
    cortext_ctx->Flush();
  } catch (...) {
  }
  cortext::telemetry::ForceFlush();
  cortext::telemetry::Shutdown();

  return 0;
}
