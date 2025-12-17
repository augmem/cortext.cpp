#pragma once

#include "context_tab.hpp"

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace chat {

// Chat message (from main.cpp)
struct ChatMessage {
  std::string role;     // "user" | "assistant" | "system"
  std::string content;  // utf-8
};

// Memory event types
enum class MemoryEventType {
  STORED,
  RETRIEVED,
};

// Memory event with all metrics
struct MemoryEvent {
  MemoryEventType type;
  int memory_id;
  std::string content;
  std::string source_id;
  uint64_t timestamp;

  // Per-memory feedback
  long long retrieved_count = 0;
  long long used_count = 0;

  // Per-memory metrics
  double relevance = 0.0;
  double mismatch = 0.0;
  double surprise = 0.0;
  double rarity = 0.0;
  double drift = 0.0;
  double contradiction = 0.0;
  double utility = 0.0;
  double periphery = 0.0;
  double coverage = 0.0;
  double salience = 0.0;
  double valence = 0.5;
  double arousal = 0.0;
  double composite_score = 0.0;
  double threshold_t = 0.0;
};

// Status bar telemetry state
struct StatusBarState {
  mutable std::mutex mu;
  std::optional<double> processing_latency_ms;
  std::optional<std::int64_t> tokens_used;
};

// Log entry for the logs tab
struct LogEntry {
  uint64_t id;
  std::string raw;
  std::string severity;
  std::string body;
  std::vector<std::pair<std::string, std::string>> attributes;
};

// Main chat window that renders all tabs
class ChatWindow {
public:
  struct State {
    std::mutex* mu = nullptr;
    std::vector<ChatMessage>* history = nullptr;
    std::deque<MemoryEvent>* memory_events = nullptr;
    std::shared_ptr<LastContext> context;
    std::shared_ptr<StatusBarState> status;
    std::string* input = nullptr;
    bool* generating = nullptr;
    std::string* partial_response = nullptr;
    int* generation_restarts = nullptr;
    bool* last_should_interrupt = nullptr;
    std::optional<std::string>* last_error = nullptr;
    bool otlp_enabled = false;
  };

  explicit ChatWindow(const State& state);

  // Called each frame to render the window
  void Render();

  // Returns true if user submitted a message
  bool HasPendingMessage() const;
  std::string TakePendingMessage();

private:
  void RenderTabBar();
  void RenderChatTab();
  void RenderMemoryTab();
  void RenderContextTab();
  void RenderLogsTab();
  void RenderStatusBar();
  void RenderInputBox();

  // Color helpers
  static void TextColored(const char* role, const char* fmt, ...);

  State state_;
  int selected_tab_ = 0;
  bool scroll_chat_to_bottom_ = true;
  bool scroll_memory_to_top_ = true;
  std::string pending_message_;
  char input_buffer_[4096] = {0};

  // Log buffer for parsed logs
  std::deque<LogEntry> parsed_logs_;
  uint64_t last_log_id_ = 0;
};

}  // namespace chat
