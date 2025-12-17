#include "chat_window.hpp"

#include <imgui.h>

#include <cstdarg>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace chat {

namespace {

ImVec4 RoleColor(const std::string& role) {
  if (role == "user") return ImVec4(0.0f, 1.0f, 1.0f, 1.0f);       // Cyan
  if (role == "assistant") return ImVec4(0.4f, 1.0f, 0.4f, 1.0f);  // Green
  if (role == "system") return ImVec4(0.6f, 0.6f, 0.6f, 1.0f);     // Gray
  return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);                           // White
}

ImVec4 SeverityColor(const std::string& severity) {
  if (severity == "ERROR") return ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red
  if (severity == "WARN") return ImVec4(1.0f, 1.0f, 0.3f, 1.0f);   // Yellow
  if (severity == "INFO") return ImVec4(0.3f, 1.0f, 0.3f, 1.0f);   // Green
  if (severity == "DEBUG") return ImVec4(0.3f, 1.0f, 1.0f, 1.0f);  // Cyan
  return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);                           // White
}

std::string RolePrefix(const std::string& role) {
  if (role == "user") return "You";
  if (role == "assistant") return "Assistant";
  if (role == "system") return "System";
  return role;
}

std::string FormatDouble(double v) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2) << v;
  return ss.str();
}

std::string Truncate(const std::string& s, size_t max_len) {
  if (s.size() <= max_len) return s;
  return s.substr(0, max_len) + "...";
}

}  // namespace

ChatWindow::ChatWindow(const State& state) : state_(state) {}

void ChatWindow::Render() {
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                           ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

  ImGui::Begin("Cortext Chat", nullptr, flags);

  // Calculate layout heights
  float status_height = 30.0f;
  float input_height = 40.0f;
  float tab_bar_height = 30.0f;
  float content_height = ImGui::GetContentRegionAvail().y - status_height - input_height - tab_bar_height - 20.0f;

  RenderTabBar();

  // Tab content area
  ImGui::BeginChild("TabContent", ImVec2(0, content_height), true);
  switch (selected_tab_) {
    case 0:
      RenderChatTab();
      break;
    case 1:
      RenderMemoryTab();
      break;
    case 2:
      RenderContextTab();
      break;
    case 3:
      RenderLogsTab();
      break;
  }
  ImGui::EndChild();

  RenderInputBox();
  RenderStatusBar();

  ImGui::End();
}

void ChatWindow::RenderTabBar() {
  if (ImGui::BeginTabBar("MainTabs")) {
    if (ImGui::BeginTabItem("Chat")) {
      selected_tab_ = 0;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Memory")) {
      selected_tab_ = 1;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Context")) {
      selected_tab_ = 2;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Logs")) {
      selected_tab_ = 3;
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void ChatWindow::RenderChatTab() {
  if (!state_.mu || !state_.history) return;

  std::lock_guard<std::mutex> lock(*state_.mu);

  for (const auto& msg : *state_.history) {
    ImGui::TextColored(RoleColor(msg.role), "%s:", RolePrefix(msg.role).c_str());
    ImGui::SameLine();
    ImGui::TextWrapped("%s", msg.content.c_str());
    ImGui::Separator();
  }

  // Show partial streaming response
  if (state_.generating && *state_.generating && state_.partial_response && !state_.partial_response->empty()) {
    ImGui::TextColored(RoleColor("assistant"), "Assistant:");
    ImGui::SameLine();
    ImGui::TextWrapped("%s", state_.partial_response->c_str());
    ImGui::Separator();
  }

  // Show streaming status
  if (state_.generating && *state_.generating) {
    std::string status = "Streaming...";
    if (state_.generation_restarts && *state_.generation_restarts > 0) {
      status += " (restarted " + std::to_string(*state_.generation_restarts) + "x with new context)";
    }
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "%s", status.c_str());
  }

  // Auto-scroll to bottom when new messages arrive
  if (scroll_chat_to_bottom_) {
    ImGui::SetScrollHereY(1.0f);
  }
}

void ChatWindow::RenderMemoryTab() {
  if (!state_.mu || !state_.memory_events) return;

  std::lock_guard<std::mutex> lock(*state_.mu);

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "MEMORY EVENTS (Real-time)");
  ImGui::Separator();

  // Interrupt status
  if (state_.last_should_interrupt && *state_.last_should_interrupt) {
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Interrupt: TRIGGERED");
  } else {
    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Interrupt: None");
  }

  // Error status
  if (state_.last_error && state_.last_error->has_value()) {
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", state_.last_error->value().c_str());
  }

  ImGui::Separator();

  if (state_.memory_events->empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no memory events yet)");
  } else {
    int shown = 0;
    for (auto it = state_.memory_events->rbegin(); it != state_.memory_events->rend() && shown < 10; ++it, ++shown) {
      const auto& evt = *it;

      // Header with type indicator
      const char* type_icon = (evt.type == MemoryEventType::STORED) ? "STORED" : "RETRIEVED";
      ImVec4 type_color = (evt.type == MemoryEventType::STORED) ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                                                                 : ImVec4(0.3f, 0.6f, 1.0f, 1.0f);

      ImGui::TextColored(type_color, "%s #%d", type_icon, evt.memory_id);

      if (!evt.source_id.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "  source: %s", evt.source_id.c_str());
      }

      // Feedback counts
      ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "  retrieved: %lld | used: %lld",
                         evt.retrieved_count, evt.used_count);

      // Score and threshold
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "  score=%s thresh=%s",
                         FormatDouble(evt.composite_score).c_str(),
                         FormatDouble(evt.threshold_t).c_str());

      // Metrics row 1
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f), "  rel=%s mis=%s sur=%s rar=%s",
                         FormatDouble(evt.relevance).c_str(),
                         FormatDouble(evt.mismatch).c_str(),
                         FormatDouble(evt.surprise).c_str(),
                         FormatDouble(evt.rarity).c_str());

      // Metrics row 2
      ImGui::TextColored(ImVec4(1.0f, 0.5f, 1.0f, 1.0f), "  drf=%s con=%s utl=%s per=%s",
                         FormatDouble(evt.drift).c_str(),
                         FormatDouble(evt.contradiction).c_str(),
                         FormatDouble(evt.utility).c_str(),
                         FormatDouble(evt.periphery).c_str());

      // Metrics row 3
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "  cov=%s sal=%s val=%s aro=%s",
                         FormatDouble(evt.coverage).c_str(),
                         FormatDouble(evt.salience).c_str(),
                         FormatDouble(evt.valence).c_str(),
                         FormatDouble(evt.arousal).c_str());

      // Content preview
      std::string preview = Truncate(evt.content, 120);
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "  \"%s\"", preview.c_str());

      ImGui::Separator();
    }
  }
}

void ChatWindow::RenderContextTab() {
  if (!state_.context) return;

  std::lock_guard<std::mutex> lock(state_.context->mu);

  if (!state_.context->has_data) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no context data yet - send a message first)");
    return;
  }

  // System Prompt Section
  if (ImGui::CollapsingHeader("System Prompt (Injected Memories)", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BeginChild("SystemPrompt", ImVec2(0, 200), true);
    if (state_.context->system_prompt.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no memories injected)");
    } else {
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "%s", state_.context->system_prompt.c_str());
    }
    ImGui::EndChild();
  }

  // Messages Section
  if (ImGui::CollapsingHeader("Messages Array", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BeginChild("Messages", ImVec2(0, 300), true);
    for (size_t i = 0; i < state_.context->messages.size(); ++i) {
      const auto& msg = state_.context->messages[i];
      ImGui::TextColored(RoleColor(msg.role), "[%zu] %s:", i, RolePrefix(msg.role).c_str());
      std::string content = Truncate(msg.content, 500);
      ImGui::TextWrapped("%s", content.c_str());
      ImGui::Separator();
    }
    ImGui::EndChild();
  }

  // Raw JSON Section
  if (ImGui::CollapsingHeader("Raw JSON Request")) {
    ImGui::BeginChild("RawJSON", ImVec2(0, 300), true);
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 1.0f, 1.0f), "%s", state_.context->raw_json.c_str());
    ImGui::EndChild();
  }
}

void ChatWindow::RenderLogsTab() {
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "System Logs (OpenTelemetry)");
  ImGui::Separator();

  if (parsed_logs_.empty()) {
    if (state_.otlp_enabled) {
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "OTLP gRPC enabled");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Traces and metrics sent to Uptrace");
      ImGui::Text("");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Telemetry active for:");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Uptrace (traces + metrics)");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Status bar metrics");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Retrieval latency tracking");
    } else {
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "OTLP gRPC not configured");
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Set UPTRACE_DSN to enable Uptrace export");
      ImGui::Text("");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "In-memory telemetry active for:");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Status bar metrics");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Retrieval latency tracking");
    }
  } else {
    // Show most recent logs first
    for (auto it = parsed_logs_.rbegin(); it != parsed_logs_.rend(); ++it) {
      const auto& log = *it;
      std::string header = log.body;
      if (!log.severity.empty()) {
        header = "[" + log.severity + "] " + header;
      }
      header = Truncate(header, 100);
      ImGui::TextColored(SeverityColor(log.severity), "%s", header.c_str());

      // Show attributes
      for (const auto& [k, v] : log.attributes) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f), "  %s:", k.c_str());
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "%s", Truncate(v, 80).c_str());
      }

      ImGui::Separator();
    }
  }
}

void ChatWindow::RenderStatusBar() {
  ImGui::Separator();

  std::optional<double> processing_ms;
  std::optional<std::int64_t> tokens_used;

  if (state_.status) {
    std::lock_guard<std::mutex> lock(state_.status->mu);
    processing_ms = state_.status->processing_latency_ms;
    tokens_used = state_.status->tokens_used;
  }

  std::ostringstream oss;
  if (processing_ms.has_value()) {
    oss << "proc=" << std::fixed << std::setprecision(1) << *processing_ms << "ms";
  } else {
    oss << "proc=?";
  }
  oss << " | ";
  if (tokens_used.has_value()) {
    oss << "tokens=" << *tokens_used;
  } else {
    oss << "tokens=?";
  }

  ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Tab: switch | Enter: send | Esc: quit");
  ImGui::SameLine(ImGui::GetWindowWidth() - 250);
  ImGui::Text("%s", oss.str().c_str());
}

void ChatWindow::RenderInputBox() {
  ImGui::Separator();

  ImGui::Text(">");
  ImGui::SameLine();

  // Check if we should focus the input
  bool generating = state_.generating && *state_.generating;

  ImGui::PushItemWidth(-1);
  ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue;
  if (generating) {
    flags |= ImGuiInputTextFlags_ReadOnly;
  }

  bool submitted = ImGui::InputText("##input", input_buffer_, sizeof(input_buffer_), flags);

  if (submitted && !generating && input_buffer_[0] != '\0') {
    pending_message_ = input_buffer_;
    input_buffer_[0] = '\0';
    scroll_chat_to_bottom_ = true;
  }

  ImGui::PopItemWidth();

  // Focus input on first frame
  if (ImGui::IsWindowAppearing()) {
    ImGui::SetKeyboardFocusHere(-1);
  }
}

bool ChatWindow::HasPendingMessage() const {
  return !pending_message_.empty();
}

std::string ChatWindow::TakePendingMessage() {
  std::string msg = std::move(pending_message_);
  pending_message_.clear();
  return msg;
}

}  // namespace chat
