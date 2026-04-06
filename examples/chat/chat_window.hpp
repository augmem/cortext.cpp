#pragma once

#include "chunk_diagnostics.hpp"
#include "context_tab.hpp"
#include "metrics_state.hpp"
#include "cortext/cortext.hpp"

#include <imgui.h>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace chat {

struct ChatMessage {
  std::string role;
  std::string content;
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
  std::optional<int> idle_seconds_remaining;
  bool idle_pending = false;
};

struct SettingsState {
  mutable std::mutex mu;
  double active_focus = 0.5;
  double active_sensitivity = 0.5;
  double active_stability = 0.5;
  int active_idle_consolidation_seconds = 0;
  double draft_focus = 0.5;
  double draft_sensitivity = 0.5;
  double draft_stability = 0.5;
  int draft_idle_consolidation_seconds = 0;
  int default_idle_consolidation_seconds = 0;
  std::string active_model;
  std::string draft_model;
  std::string default_model;
  std::string active_memory_prompt_prefix;
  std::string draft_memory_prompt_prefix;
  std::string default_memory_prompt_prefix;
  std::string active_memory_prompt_suffix;
  std::string draft_memory_prompt_suffix;
  std::string default_memory_prompt_suffix;
  bool apply_requested = false;
  std::optional<std::string> last_apply_status;
};

struct DatabaseMemoryRow {
  long long memory_id = 0;
  long long episode_id = 0;
  long long embedding_id = 0;
  std::string kind;
  std::string source_id;
  std::string label;
  uint64_t start_ts = 0;
  uint64_t end_ts = 0;
  long long n_signals = 0;
  double strength = 0.0;
  double connectivity = 0.0;
  double source_reliability = 0.0;
  long long retrieved_count = 0;
  long long used_count = 0;
};

struct DatabaseSignalRow {
  long long signal_id = 0;
  long long memory_id = 0;
  long long embedding_id = 0;
  std::string source_id;
  std::string modality;
  std::string mime;
  uint64_t timestamp = 0;
  long long serial_position = 0;
  double score = 0.0;
  double salience = 0.0;
  double threshold_t = 0.0;
};

struct DatabaseAssociationRow {
  long long source_memory_id = 0;
  long long target_memory_id = 0;
  std::string edge_type;
  double weight = 0.0;
  double decay_rate = 0.0;
  uint64_t last_reinforced = 0;
  std::string source_kind;
  std::string source_label;
  std::string source_source_id;
  std::string target_kind;
  std::string target_label;
  std::string target_source_id;
};

struct DatabaseEpisodeRow {
  long long episode_id = 0;
  uint64_t start_ts = 0;
  uint64_t end_ts = 0;
  std::string boundary_type;
  uint64_t created_at = 0;
};

struct DatabaseFactRow {
  long long fact_id = 0;
  std::string subject;
  std::string predicate;
  std::string object;
  uint64_t valid_start_ts = 0;
  uint64_t valid_end_ts = 0;
  uint64_t recorded_at_ts = 0;
  uint64_t superseded_at_ts = 0;
  double confidence = 0.0;
  long long summary_memory_id = 0;
  std::string lifecycle_state;
  std::string severity_class;
};

struct DatabaseEvictionRow {
  long long eviction_id = 0;
  long long memory_id = 0;
  long long embedding_id = 0;
  std::string source_id;
  std::string kind;
  std::string label;
  uint64_t start_ts = 0;
  uint64_t end_ts = 0;
  uint64_t created_at = 0;
  uint64_t last_access = 0;
  uint64_t evicted_at = 0;
  double strength = 0.0;
  double use_frequency = 0.0;
  double contextual_gain = 0.0;
  long long retrieved_count = 0;
  long long used_count = 0;
  long long n_signals = 0;
  std::string modality;
  std::string eviction_reason;
};

struct DatabaseExplorerState {
  mutable std::mutex mu;
  std::vector<DatabaseMemoryRow> memories;
  std::vector<DatabaseSignalRow> signals;
  std::vector<DatabaseAssociationRow> associations;
  std::vector<DatabaseEpisodeRow> episodes;
  std::vector<DatabaseFactRow> facts;
  std::vector<DatabaseEvictionRow> evictions;
  long long total_memories = 0;
  long long total_signals = 0;
  long long total_associations = 0;
  long long total_episodes = 0;
  long long total_facts = 0;
  long long total_evictions = 0;
  uint64_t refreshed_at = 0;
  bool refresh_requested = true;
  bool clear_requested = false;
  std::optional<std::string> last_refresh_status;
};

// Log entry for the logs tab
struct LogEntry {
  uint64_t id;
  std::string raw;
  std::string severity;
  std::string body;
  std::vector<std::pair<std::string, std::string>> attributes;
};

struct TraceEntry {
  uint64_t id = 0;
  std::string name;
  double duration_ms = 0.0;
  std::string status;
  std::vector<std::pair<std::string, std::string>> attributes;
};

struct OTelState {
  mutable std::mutex mu;
  std::deque<LogEntry> logs;
  std::deque<TraceEntry> traces;
};

// Main chat window that renders all tabs
class ChatWindow {
public:
  struct State {
    std::mutex* mu = nullptr;
    std::vector<cortext::Cortext::Context::Memory>* working_memory = nullptr;
    std::deque<ChatMessage>* chat_history = nullptr;
    std::deque<MemoryEvent>* memory_events = nullptr;
    std::shared_ptr<LastContext> context;
    std::shared_ptr<ChunkDiagnosticsState> chunk_diagnostics;
    std::shared_ptr<MetricsState> metrics;
    std::shared_ptr<StatusBarState> status;
    std::shared_ptr<SettingsState> settings;
    std::shared_ptr<DatabaseExplorerState> db_explorer;
    std::shared_ptr<OTelState> otel;
    std::string* input = nullptr;
    bool* generating = nullptr;
    std::string* partial_response = nullptr;
    int* generation_restarts = nullptr;
    bool* last_should_interrupt = nullptr;
    std::optional<std::string>* last_error = nullptr;
    std::atomic<bool>* typing_interrupt_requested = nullptr;
    std::atomic<bool>* input_activity_requested = nullptr;
    std::atomic<bool>* has_input_draft = nullptr;
    std::atomic<bool>* idle_consolidating = nullptr;
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
  void RenderChunksTab();
  void RenderMetricsTab();
  void RenderMemoryTab();
  void RenderWorkingMemoryTab();
  void RenderSettingsTab();
  void RenderDatabaseTab();
  void RenderGraphTab();
  void RenderContextTab();
  void RenderLogsTab();
  void RenderStatusBar();
  void RenderInputBox();

  // Color helpers
  static void TextColored(const char* role, const char* fmt, ...);

  State state_;
  int selected_tab_ = 0;
  bool scroll_chat_to_bottom_ = true;
  std::size_t last_chat_message_count_ = 0;
  std::size_t last_partial_response_size_ = 0;
  bool last_generating_ = false;
  bool scroll_memory_to_top_ = true;
  bool refocus_input_next_frame_ = true;
  std::string pending_message_;
  char input_buffer_[4096] = {0};
  bool show_clear_db_confirm_ = false;
  int db_memory_kind_filter_ = 0;
  ImVec2 graph_pan_offset_ = ImVec2(80.0f, 80.0f);
  std::unordered_map<long long, ImVec2> graph_node_positions_;
  std::size_t graph_layout_signature_ = 0;
  bool graph_layout_dirty_ = true;
  long long graph_dragging_node_id_ = 0;
};

}  // namespace chat
