#include "chat_window.hpp"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
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

std::string FormatPercent(double v) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1) << (v * 100.0) << "%";
  return ss.str();
}

bool NearlyEqual(double a, double b) {
  return std::abs(a - b) < 1e-6;
}

std::string FormatTimestamp(uint64_t timestamp_ms) {
  if (timestamp_ms == 0) return "0";
  std::ostringstream ss;
  ss << timestamp_ms;
  return ss.str();
}

std::string NullOrValue(uint64_t value) {
  if (value == 0) return "-";
  return FormatTimestamp(value);
}

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

std::string RoleFromSourceId(const std::string& source_id) {
  if (source_id == "chat/user") return "user";
  if (source_id == "chat/assistant") return "assistant";
  return {};
}

struct UiMessage {
  std::string role;
  std::string content;
};

std::vector<UiMessage> BuildMessagesFromWorkingMemory(
    const std::vector<cortext::Cortext::Context::Memory>& working_memory) {
  std::vector<const cortext::Cortext::Context::Memory*> ordered;
  ordered.reserve(working_memory.size());
  for (const auto& mem : working_memory) {
    ordered.push_back(&mem);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const auto* a, const auto* b) {
              if (a->timestamp == b->timestamp) return a->id < b->id;
              return a->timestamp < b->timestamp;
            });

  std::vector<UiMessage> messages;
  messages.reserve(ordered.size());
  for (const auto* mem : ordered) {
    const std::string role = RoleFromSourceId(mem->source_id);
    if (role.empty()) {
      continue;
    }
    std::string content = ExtractTextFromBlobs(mem->content);
    if (content.empty()) {
      content = "(empty)";
    }
    messages.push_back({role, std::move(content)});
  }
  return messages;
}

std::string FormatMemoryTitle(const DatabaseMemoryRow& row) {
  std::string title = "#" + std::to_string(row.memory_id) + " " + row.kind;
  if (!row.label.empty()) {
    title += " | " + Truncate(row.label, 72);
  } else {
    title += " | " + row.source_id;
  }
  return title;
}

std::string FormatAssociationNode(const std::string& kind,
                                  const std::string& label,
                                  const std::string& source_id,
                                  long long memory_id) {
  if (!label.empty()) {
    return label + " (#" + std::to_string(memory_id) + ")";
  }
  if (!kind.empty()) {
    return kind + " " + source_id + " (#" + std::to_string(memory_id) + ")";
  }
  return "#" + std::to_string(memory_id);
}

template <typename Getter>
std::vector<double> BuildPlotValues(
    const std::deque<ResponseMetricsSample>& samples, Getter getter) {
  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto& sample : samples) {
    values.push_back(static_cast<double>(getter(sample)));
  }
  return values;
}

template <typename Getter>
std::vector<double> BuildPlotValues(
    const std::deque<ConsolidationMetricsSample>& samples, Getter getter) {
  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto& sample : samples) {
    values.push_back(static_cast<double>(getter(sample)));
  }
  return values;
}

std::vector<double> BuildSampleIndices(std::size_t count) {
  std::vector<double> indices;
  indices.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    indices.push_back(static_cast<double>(i));
  }
  return indices;
}

void SetupHistoryAxes(const char* y_label) {
  ImPlot::SetupAxes("sample", y_label,
                    ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_AutoFit,
                    ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_AutoFit);
}

void PlotSeriesIfAvailable(const char* label,
                           const std::vector<double>& xs,
                           const std::vector<double>& ys) {
  if (xs.empty() || ys.empty()) {
    return;
  }
  ImPlot::PlotLine(label, xs.data(), ys.data(), static_cast<int>(ys.size()));
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
      RenderChunksTab();
      break;
    case 2:
      RenderMetricsTab();
      break;
    case 3:
      RenderMemoryTab();
      break;
    case 4:
      RenderWorkingMemoryTab();
      break;
    case 5:
      RenderSettingsTab();
      break;
    case 6:
      RenderDatabaseTab();
      break;
    case 7:
      RenderContextTab();
      break;
    case 8:
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
    if (ImGui::BeginTabItem("Chunks")) {
      selected_tab_ = 1;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Metrics")) {
      selected_tab_ = 2;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Memory")) {
      selected_tab_ = 3;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Working Memory")) {
      selected_tab_ = 4;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Settings")) {
      selected_tab_ = 5;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("DB")) {
      selected_tab_ = 6;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Context")) {
      selected_tab_ = 7;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Logs")) {
      selected_tab_ = 8;
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void ChatWindow::RenderChatTab() {
  if (!state_.mu) return;

  std::lock_guard<std::mutex> lock(*state_.mu);

  if (state_.chat_history) {
    for (const auto& msg : *state_.chat_history) {
      ImGui::TextColored(RoleColor(msg.role), "%s:", RolePrefix(msg.role).c_str());
      ImGui::SameLine();
      ImGui::TextWrapped("%s", msg.content.c_str());
      ImGui::Separator();
    }
  }

  if (state_.generating && *state_.generating) {
    int dots = (static_cast<int>(ImGui::GetTime() / 0.4) % 3) + 1;
    std::string status = "generating" + std::string(dots, '.');
    if (state_.generation_restarts && *state_.generation_restarts > 0) {
      status += " (restarted " + std::to_string(*state_.generation_restarts) +
                "x with new context)";
    }
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "%s", status.c_str());
  }

  // Auto-scroll to bottom when new messages arrive
  if (scroll_chat_to_bottom_) {
    ImGui::SetScrollHereY(1.0f);
  }
}

void ChatWindow::RenderChunksTab() {
  if (!state_.chunk_diagnostics) return;

  std::optional<ChunkProbeEvent> active_probe;
  std::deque<ChunkProbeEvent> recent_probes;
  std::size_t total_probes = 0;
  std::size_t interrupts = 0;
  std::size_t suppressed = 0;
  std::size_t errors = 0;
  {
    std::lock_guard<std::mutex> lock(state_.chunk_diagnostics->mu);
    active_probe = state_.chunk_diagnostics->active_probe;
    recent_probes = state_.chunk_diagnostics->recent_probes;
    total_probes = state_.chunk_diagnostics->total_probes;
    interrupts = state_.chunk_diagnostics->interrupts;
    suppressed = state_.chunk_diagnostics->suppressed;
    errors = state_.chunk_diagnostics->errors;
  }

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "STREAMING CHUNK PROBES");
  ImGui::Separator();

  if (active_probe.has_value() && active_probe->in_progress) {
    int dots = (static_cast<int>(ImGui::GetTime() / 0.4) % 3) + 1;
    std::string status = "processing chunk" + std::string(dots, '.');
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "%s", status.c_str());
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                       "restart=%d chars=%zu tokens_so_far=%zu",
                       active_probe->restart_index,
                       active_probe->chunk_char_count,
                       active_probe->tokens_so_far);
    if (!active_probe->chunk_text.empty()) {
      ImGui::TextWrapped("%s", active_probe->chunk_text.c_str());
    }
  } else {
    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "idle");
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                     "total=%zu interrupts=%zu suppressed=%zu errors=%zu",
                     total_probes,
                     interrupts,
                     suppressed,
                     errors);
  ImGui::Separator();

  if (recent_probes.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "(no streaming chunk probes yet)");
    return;
  }

  for (auto it = recent_probes.rbegin(); it != recent_probes.rend(); ++it) {
    const auto& probe = *it;
    const bool triggered = probe.reason == "interrupt_triggered";
    const bool suppressed_probe = probe.reason == "interrupt_suppressed_no_new_memories";
    const std::string boundary_score_text
        = probe.boundary_score.has_value() ? FormatDouble(*probe.boundary_score) : "-";
    const ImVec4 header_color = probe.had_error
                                    ? ImVec4(1.0f, 0.3f, 0.3f, 1.0f)
                                    : (triggered ? ImVec4(1.0f, 0.6f, 0.3f, 1.0f)
                                                 : (suppressed_probe ? ImVec4(1.0f, 1.0f, 0.3f, 1.0f)
                                                                     : ImVec4(0.3f, 1.0f, 1.0f, 1.0f)));
    std::string header = "#" + std::to_string(probe.event_id) + " " + probe.reason
                         + " | restart=" + std::to_string(probe.restart_index)
                         + " | " + Truncate(probe.chunk_text, 72);
    if (header.empty()) {
      header = "#" + std::to_string(probe.event_id);
    }

    if (ImGui::CollapsingHeader(header.c_str())) {
      ImGui::TextColored(header_color, "reason=%s", probe.reason.c_str());
      ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                         "chars=%zu tokens_so_far=%zu started=%s completed=%s",
                         probe.chunk_char_count,
                         probe.tokens_so_far,
                         FormatTimestamp(probe.started_at_ms).c_str(),
                         FormatTimestamp(probe.completed_at_ms).c_str());
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                         "encode=%sms process=%sms hydrate=%sms total=%sms",
                         FormatDouble(probe.encode_ms).c_str(),
                         FormatDouble(probe.process_ms).c_str(),
                         FormatDouble(probe.hydrate_ms).c_str(),
                         FormatDouble(probe.total_ms).c_str());
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f),
                         "boundary=%s score=%s threshold=%s pass=%s",
                         probe.at_boundary ? "true" : "false",
                         boundary_score_text.c_str(),
                         FormatDouble(probe.boundary_threshold).c_str(),
                         probe.boundary_score_pass ? "true" : "false");
      ImGui::TextColored(ImVec4(1.0f, 0.5f, 1.0f, 1.0f),
                         "interrupt=%s new_memories=%d retrieved=%zu",
                         probe.should_interrupt ? "true" : "false",
                         probe.new_memory_count,
                         probe.raw_retrieved_count);
      ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f),
                         "has_candidates=%s no_store=%s rel=%s novelty=%s mu=%s novelty_mu=%s dup=%s boundary_mu=%s",
                         probe.interrupt_gate_has_candidates ? "true" : "false",
                         probe.interrupt_gate_blocked_no_store ? "true" : "false",
                         probe.interrupt_gate_rel_pass ? "true" : "false",
                         probe.interrupt_gate_novelty_pass ? "true" : "false",
                         probe.interrupt_gate_mu_pass ? "true" : "false",
                         probe.interrupt_gate_novelty_mu_pass ? "true" : "false",
                         probe.interrupt_gate_dup_pass ? "true" : "false",
                         probe.interrupt_gate_boundary_mu_pass ? "true" : "false");
      ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.9f, 1.0f),
                         "rel*=%s thresh=%s boundary_mult=%s affect=%s",
                         FormatDouble(probe.interrupt_gate_rel_star).c_str(),
                         FormatDouble(probe.interrupt_gate_retrieval_thresh).c_str(),
                         FormatDouble(probe.interrupt_gate_boundary_mult_eff).c_str(),
                         FormatDouble(probe.interrupt_gate_affect_drive).c_str());

      if (probe.had_error && !probe.error_message.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "error: %s",
                           probe.error_message.c_str());
      }

      ImGui::BeginChild(("chunk-text-" + std::to_string(probe.event_id)).c_str(),
                        ImVec2(0, 90), true);
      ImGui::TextWrapped("%s", probe.chunk_text.empty() ? "(empty)" : probe.chunk_text.c_str());
      ImGui::EndChild();

      if (!probe.retrieved_memories.empty()) {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Retrieved memories");
        for (const auto& mem : probe.retrieved_memories) {
          ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
                             "#%lld %s", mem.memory_id, mem.source_id.c_str());
          ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                             "rel=%s score=%s",
                             FormatDouble(mem.relevance).c_str(),
                             FormatDouble(mem.composite_score).c_str());
          ImGui::TextWrapped("%s", mem.preview.c_str());
          ImGui::Separator();
        }
      }
    }
  }
}

void ChatWindow::RenderMetricsTab() {
  if (!state_.metrics) return;

  std::deque<ResponseMetricsSample> response_history;
  std::deque<ConsolidationMetricsSample> consolidation_history;
  std::optional<ResponseMetricsSample> latest_response;
  std::optional<ConsolidationMetricsSample> latest_consolidation;
  std::size_t total_responses = 0;
  std::size_t total_consolidations = 0;
  {
    std::lock_guard<std::mutex> lock(state_.metrics->mu);
    response_history = state_.metrics->response_history;
    consolidation_history = state_.metrics->consolidation_history;
    latest_response = state_.metrics->latest_response;
    latest_consolidation = state_.metrics->latest_consolidation;
    total_responses = state_.metrics->total_responses;
    total_consolidations = state_.metrics->total_consolidations;
  }

  long long total_memories = 0;
  long long total_signals = 0;
  long long total_associations = 0;
  long long total_episodes = 0;
  if (state_.db_explorer) {
    std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
    total_memories = state_.db_explorer->total_memories;
    total_signals = state_.db_explorer->total_signals;
    total_associations = state_.db_explorer->total_associations;
    total_episodes = state_.db_explorer->total_episodes;
  }

  const bool consolidating
      = state_.idle_consolidating && state_.idle_consolidating->load();

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "SESSION METRICS");
  ImGui::Separator();

  if (latest_response.has_value()) {
    const auto& sample = *latest_response;
    ImGui::TextColored(
        ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
        "latest tokens=%lld prompt=%lld completion=%lld (%s)",
        sample.usage.total_tokens,
        sample.usage.prompt_tokens,
        sample.usage.completion_tokens,
        UsageAccuracyLabel(sample.usage_accuracy));
    ImGui::TextColored(
        ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
        "wall=%sms phase1=%sms phase3=%sms retrieval=%sms proc=%sms",
        FormatDouble(sample.response_wall_ms).c_str(),
        FormatDouble(sample.phase1_total_ms).c_str(),
        FormatDouble(sample.phase3_total_ms).c_str(),
        FormatDouble(sample.retrieval_latency_ms).c_str(),
        FormatDouble(sample.processing_latency_ms).c_str());
    ImGui::TextColored(
        ImVec4(0.3f, 1.0f, 1.0f, 1.0f),
        "restarts=%d probes=%d interrupts=%d rate=%s stream_error=%s",
        sample.restart_count,
        sample.probe_count,
        sample.interrupt_count,
        FormatPercent(sample.interrupt_rate).c_str(),
        sample.had_stream_error ? "true" : "false");
    if (sample.had_stream_error && !sample.error_message.empty()) {
      ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s",
                         sample.error_message.c_str());
    }
  } else {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "(no response metrics yet)");
  }

  if (latest_consolidation.has_value()) {
    const auto& sample = *latest_consolidation;
    const char* status = sample.completed
                             ? "completed"
                             : (sample.cancelled
                                    ? "cancelled"
                                    : (sample.had_error ? "error" : "recorded"));
    ImGui::TextColored(
        consolidating ? ImVec4(1.0f, 1.0f, 0.3f, 1.0f)
                      : ImVec4(0.6f, 0.8f, 1.0f, 1.0f),
        "consolidation=%s latest=%s duration=%sms wm=%d",
        consolidating ? "running" : "idle",
        status,
        FormatDouble(sample.duration_ms).c_str(),
        sample.working_memory_size);
  } else {
    ImGui::TextColored(
        consolidating ? ImVec4(1.0f, 1.0f, 0.3f, 1.0f)
                      : ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
        "consolidation=%s total=%zu",
        consolidating ? "running" : "idle",
        total_consolidations);
  }

  ImGui::TextColored(
      ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
      "responses=%zu consolidations=%zu db: memories=%lld signals=%lld associations=%lld episodes=%lld",
      total_responses,
      total_consolidations,
      total_memories,
      total_signals,
      total_associations,
      total_episodes);
  ImGui::Separator();

  if (response_history.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "Charts will appear after the first assistant response.");
  } else {
    const auto response_xs = BuildSampleIndices(response_history.size());
    const auto total_tokens = BuildPlotValues(response_history, [](const auto& s) {
      return s.usage.total_tokens;
    });
    const auto prompt_tokens = BuildPlotValues(response_history, [](const auto& s) {
      return s.usage.prompt_tokens;
    });
    const auto completion_tokens = BuildPlotValues(response_history, [](const auto& s) {
      return s.usage.completion_tokens;
    });
    const auto response_wall_ms = BuildPlotValues(response_history, [](const auto& s) {
      return s.response_wall_ms;
    });
    const auto phase1_ms = BuildPlotValues(response_history, [](const auto& s) {
      return s.phase1_total_ms;
    });
    const auto phase3_ms = BuildPlotValues(response_history, [](const auto& s) {
      return s.phase3_total_ms;
    });
    const auto retrieval_ms = BuildPlotValues(response_history, [](const auto& s) {
      return s.retrieval_latency_ms;
    });
    const auto processing_ms = BuildPlotValues(response_history, [](const auto& s) {
      return s.processing_latency_ms;
    });
    const auto probe_count = BuildPlotValues(response_history, [](const auto& s) {
      return s.probe_count;
    });
    const auto interrupt_count = BuildPlotValues(response_history, [](const auto& s) {
      return s.interrupt_count;
    });
    const auto restart_count = BuildPlotValues(response_history, [](const auto& s) {
      return s.restart_count;
    });
    const auto interrupt_rate = BuildPlotValues(response_history, [](const auto& s) {
      return s.interrupt_rate * 100.0;
    });

    if (ImPlot::BeginPlot("Token usage", ImVec2(-1, 220))) {
      SetupHistoryAxes("tokens");
      ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_None);
      PlotSeriesIfAvailable("total", response_xs, total_tokens);
      PlotSeriesIfAvailable("prompt", response_xs, prompt_tokens);
      PlotSeriesIfAvailable("completion", response_xs, completion_tokens);
      ImPlot::EndPlot();
    }

    if (ImPlot::BeginPlot("Response latency", ImVec2(-1, 240))) {
      SetupHistoryAxes("ms");
      ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_None);
      PlotSeriesIfAvailable("wall", response_xs, response_wall_ms);
      PlotSeriesIfAvailable("phase1", response_xs, phase1_ms);
      PlotSeriesIfAvailable("phase3", response_xs, phase3_ms);
      PlotSeriesIfAvailable("retrieval", response_xs, retrieval_ms);
      PlotSeriesIfAvailable("processing", response_xs, processing_ms);
      ImPlot::EndPlot();
    }

    if (ImPlot::BeginPlot("Interrupt activity", ImVec2(-1, 220))) {
      SetupHistoryAxes("count");
      ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_None);
      PlotSeriesIfAvailable("probes", response_xs, probe_count);
      PlotSeriesIfAvailable("interrupts", response_xs, interrupt_count);
      PlotSeriesIfAvailable("restarts", response_xs, restart_count);
      ImPlot::EndPlot();
    }

    if (ImPlot::BeginPlot("Interrupt rate", ImVec2(-1, 180))) {
      SetupHistoryAxes("percent");
      ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_None);
      PlotSeriesIfAvailable("interrupt %", response_xs, interrupt_rate);
      ImPlot::EndPlot();
    }
  }

  if (consolidation_history.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "Consolidation chart will appear after the first idle run.");
  } else {
    const auto consolidation_xs = BuildSampleIndices(consolidation_history.size());
    const auto consolidation_ms = BuildPlotValues(consolidation_history, [](const auto& s) {
      return s.duration_ms;
    });

    if (ImPlot::BeginPlot("Consolidation duration", ImVec2(-1, 180))) {
      SetupHistoryAxes("ms");
      ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_None);
      PlotSeriesIfAvailable("duration", consolidation_xs, consolidation_ms);
      ImPlot::EndPlot();
    }
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Recent responses");
  if (response_history.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no responses yet)");
  } else if (ImGui::BeginTable("MetricsResponses", 8,
                               ImGuiTableFlags_Borders
                                   | ImGuiTableFlags_RowBg
                                   | ImGuiTableFlags_ScrollY,
                               ImVec2(0, 220))) {
    ImGui::TableSetupColumn("ts");
    ImGui::TableSetupColumn("usage");
    ImGui::TableSetupColumn("total");
    ImGui::TableSetupColumn("wall");
    ImGui::TableSetupColumn("phase1");
    ImGui::TableSetupColumn("phase3");
    ImGui::TableSetupColumn("interrupts");
    ImGui::TableSetupColumn("notes");
    ImGui::TableHeadersRow();

    int shown = 0;
    for (auto it = response_history.rbegin();
         it != response_history.rend() && shown < 20; ++it, ++shown) {
      const auto& sample = *it;
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(FormatTimestamp(sample.timestamp_ms).c_str());
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(UsageAccuracyLabel(sample.usage_accuracy));
      ImGui::TableNextColumn();
      ImGui::Text("%lld", sample.usage.total_tokens);
      ImGui::TableNextColumn();
      ImGui::Text("%s", FormatDouble(sample.response_wall_ms).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", FormatDouble(sample.phase1_total_ms).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", FormatDouble(sample.phase3_total_ms).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d/%d (%s)", sample.interrupt_count, sample.probe_count,
                  FormatPercent(sample.interrupt_rate).c_str());
      ImGui::TableNextColumn();
      std::string note = "restarts=" + std::to_string(sample.restart_count);
      if (sample.had_stream_error) {
        note += " stream_error";
      }
      ImGui::TextUnformatted(note.c_str());
    }
    ImGui::EndTable();
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Recent consolidations");
  if (consolidation_history.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "(no consolidations yet)");
    return;
  }

  if (ImGui::BeginTable("MetricsConsolidations", 5,
                         ImGuiTableFlags_Borders
                             | ImGuiTableFlags_RowBg
                             | ImGuiTableFlags_ScrollY,
                         ImVec2(0, 180))) {
    ImGui::TableSetupColumn("ts");
    ImGui::TableSetupColumn("status");
    ImGui::TableSetupColumn("duration");
    ImGui::TableSetupColumn("wm");
    ImGui::TableSetupColumn("notes");
    ImGui::TableHeadersRow();

    int shown = 0;
    for (auto it = consolidation_history.rbegin();
         it != consolidation_history.rend() && shown < 12; ++it, ++shown) {
      const auto& sample = *it;
      const char* status = sample.completed
                               ? "completed"
                               : (sample.cancelled
                                      ? "cancelled"
                                      : (sample.had_error ? "error" : "recorded"));
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(FormatTimestamp(sample.timestamp_ms).c_str());
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(status);
      ImGui::TableNextColumn();
      ImGui::Text("%s", FormatDouble(sample.duration_ms).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", sample.working_memory_size);
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(
          sample.error_message.empty() ? "-" : sample.error_message.c_str());
    }
    ImGui::EndTable();
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

void ChatWindow::RenderWorkingMemoryTab() {
  if (!state_.mu || !state_.working_memory) return;

  std::lock_guard<std::mutex> lock(*state_.mu);

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "WORKING MEMORY SLOTS");
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(%zu active)", state_.working_memory->size());
  ImGui::Separator();

  if (state_.working_memory->empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no active working memory slots)");
    return;
  }

  std::vector<const cortext::Cortext::Context::Memory*> ordered;
  ordered.reserve(state_.working_memory->size());
  for (const auto& mem : *state_.working_memory) {
    ordered.push_back(&mem);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const auto* a, const auto* b) {
              if (a->timestamp == b->timestamp) return a->id < b->id;
              return a->timestamp < b->timestamp;
            });

  int slot_index = 0;
  for (const auto* mem : ordered) {
    const std::string header
        = "Slot " + std::to_string(slot_index++) + "##wm-" + std::to_string(mem->id);
    if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f), "id=%lld", mem->id);
      ImGui::SameLine();
      ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "source=%s", mem->source_id.c_str());
      ImGui::SameLine();
      ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "ts=%s",
                         FormatTimestamp(mem->timestamp).c_str());

      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                         "score=%s salience=%s retrieved=%lld used=%lld",
                         FormatDouble(mem->composite_score).c_str(),
                         FormatDouble(mem->salience).c_str(),
                         mem->retrieved_count,
                         mem->used_count);

      ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f),
                         "rel=%s mis=%s sur=%s rar=%s",
                         FormatDouble(mem->relevance).c_str(),
                         FormatDouble(mem->mismatch).c_str(),
                         FormatDouble(mem->surprise).c_str(),
                         FormatDouble(mem->rarity).c_str());

      ImGui::TextColored(ImVec4(1.0f, 0.5f, 1.0f, 1.0f),
                         "drf=%s con=%s utl=%s per=%s",
                         FormatDouble(mem->drift).c_str(),
                         FormatDouble(mem->contradiction).c_str(),
                         FormatDouble(mem->utility).c_str(),
                         FormatDouble(mem->periphery).c_str());

      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                         "cov=%s val=%s aro=%s thresh=%s",
                         FormatDouble(mem->coverage).c_str(),
                         FormatDouble(mem->valence).c_str(),
                         FormatDouble(mem->arousal).c_str(),
                         FormatDouble(mem->threshold_t).c_str());

      const std::string text = ExtractTextFromBlobs(mem->content);
      ImGui::BeginChild(("wm-content-" + std::to_string(mem->id)).c_str(),
                        ImVec2(0, 110), true);
      if (text.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(empty)");
      } else {
        ImGui::TextWrapped("%s", text.c_str());
      }
      ImGui::EndChild();
    }
  }
}

void ChatWindow::RenderSettingsTab() {
  if (!state_.settings) return;

  double active_focus = 0.5;
  double active_sensitivity = 0.5;
  double active_stability = 0.5;
  double draft_focus = 0.5;
  double draft_sensitivity = 0.5;
  double draft_stability = 0.5;
  bool apply_requested = false;
  std::optional<std::string> last_status;
  {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    active_focus = state_.settings->active_focus;
    active_sensitivity = state_.settings->active_sensitivity;
    active_stability = state_.settings->active_stability;
    draft_focus = state_.settings->draft_focus;
    draft_sensitivity = state_.settings->draft_sensitivity;
    draft_stability = state_.settings->draft_stability;
    apply_requested = state_.settings->apply_requested;
    last_status = state_.settings->last_apply_status;
  }

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "SETTINGS");
  ImGui::Separator();

  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                     "Active: F=%s  S=%s  T=%s",
                     FormatDouble(active_focus).c_str(),
                     FormatDouble(active_sensitivity).c_str(),
                     FormatDouble(active_stability).c_str());

  const bool dirty = !NearlyEqual(active_focus, draft_focus)
                     || !NearlyEqual(active_sensitivity, draft_sensitivity)
                     || !NearlyEqual(active_stability, draft_stability);
  ImGui::TextColored(dirty ? ImVec4(1.0f, 1.0f, 0.3f, 1.0f)
                           : ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
                     dirty ? "Draft differs from active config"
                           : "Draft matches active config");
  if (apply_requested) {
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "Apply pending");
  }
  if (last_status.has_value()) {
    ImGui::TextWrapped("%s", last_status->c_str());
  }

  ImGui::Separator();

  float draft_focus_f = static_cast<float>(draft_focus);
  float draft_sensitivity_f = static_cast<float>(draft_sensitivity);
  float draft_stability_f = static_cast<float>(draft_stability);
  bool changed = false;
  changed |= ImGui::SliderFloat("Focus", &draft_focus_f, 0.0f, 1.0f, "%.2f");
  changed |= ImGui::SliderFloat("Sensitivity", &draft_sensitivity_f, 0.0f, 1.0f, "%.2f");
  changed |= ImGui::SliderFloat("Stability", &draft_stability_f, 0.0f, 1.0f, "%.2f");
  draft_focus = static_cast<double>(draft_focus_f);
  draft_sensitivity = static_cast<double>(draft_sensitivity_f);
  draft_stability = static_cast<double>(draft_stability_f);

  if (changed) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_focus = draft_focus;
    state_.settings->draft_sensitivity = draft_sensitivity;
    state_.settings->draft_stability = draft_stability;
  }

  const bool generating = state_.generating && *state_.generating;
  if (generating) {
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                       "Wait for the current generation to finish before applying.");
  }

  const bool can_apply = dirty && !apply_requested && !generating;
  if (!can_apply) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Apply")) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->apply_requested = true;
    state_.settings->last_apply_status = "Applying updated knobs...";
  }
  if (!can_apply) {
    ImGui::EndDisabled();
  }

  ImGui::SameLine();
  if (ImGui::Button("Reset")) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_focus = state_.settings->active_focus;
    state_.settings->draft_sensitivity = state_.settings->active_sensitivity;
    state_.settings->draft_stability = state_.settings->active_stability;
  }

  ImGui::Separator();
  ImGui::TextWrapped(
      "Changing these knobs rebuilds the Cortext instance against the same chat DB. "
      "New generations use the updated config; active working-memory contents may take "
      "a turn to fully reflect the new settings.");
}

void ChatWindow::RenderDatabaseTab() {
  if (!state_.db_explorer) return;

  std::vector<DatabaseMemoryRow> memories;
  std::vector<DatabaseSignalRow> signals;
  std::vector<DatabaseAssociationRow> associations;
  std::vector<DatabaseEpisodeRow> episodes;
  long long total_memories = 0;
  long long total_signals = 0;
  long long total_associations = 0;
  long long total_episodes = 0;
  uint64_t refreshed_at = 0;
  std::optional<std::string> last_status;
  {
    std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
    memories = state_.db_explorer->memories;
    signals = state_.db_explorer->signals;
    associations = state_.db_explorer->associations;
    episodes = state_.db_explorer->episodes;
    total_memories = state_.db_explorer->total_memories;
    total_signals = state_.db_explorer->total_signals;
    total_associations = state_.db_explorer->total_associations;
    total_episodes = state_.db_explorer->total_episodes;
    refreshed_at = state_.db_explorer->refreshed_at;
    last_status = state_.db_explorer->last_refresh_status;
  }

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "DATABASE EXPLORER");
  ImGui::SameLine();
  if (ImGui::Button("Refresh")) {
    std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
    state_.db_explorer->refresh_requested = true;
    state_.db_explorer->last_refresh_status = "Refresh requested...";
  }
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.18f, 0.18f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.68f, 0.24f, 0.24f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.46f, 0.12f, 0.12f, 1.0f));
  if (ImGui::Button("Clear DB")) {
    show_clear_db_confirm_ = true;
    ImGui::OpenPopup("ConfirmClearDb");
  }
  ImGui::PopStyleColor(3);
  ImGui::Separator();

  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                     "Totals: memories=%lld signals=%lld associations=%lld episodes=%lld",
                     total_memories,
                     total_signals,
                     total_associations,
                     total_episodes);
  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Last refresh: %s",
                     NullOrValue(refreshed_at).c_str());
  if (last_status.has_value()) {
    ImGui::TextWrapped("%s", last_status->c_str());
  }

  if (show_clear_db_confirm_) {
    ImGui::SetNextWindowSize(ImVec2(460.0f, 0.0f), ImGuiCond_Appearing);
  }
  if (ImGui::BeginPopupModal("ConfirmClearDb", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Clear the chat database?");
    ImGui::Separator();
    ImGui::TextWrapped(
        "This deletes all stored memories, signals, relationships, episodes, and working-memory state "
        "for the current chat database. This action cannot be undone.");
    ImGui::Spacing();
    ImGui::TextWrapped("Only continue if you want to reset this chat memory from scratch.");
    ImGui::Spacing();
    if (ImGui::Button("Cancel", ImVec2(120.0f, 0.0f))) {
      show_clear_db_confirm_ = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.18f, 0.18f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.68f, 0.24f, 0.24f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.46f, 0.12f, 0.12f, 1.0f));
    if (ImGui::Button("Yes, clear it", ImVec2(140.0f, 0.0f))) {
      std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
      state_.db_explorer->clear_requested = true;
      state_.db_explorer->last_refresh_status = "Clearing chat database...";
      show_clear_db_confirm_ = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor(3);
    ImGui::EndPopup();
  }

  ImGui::Separator();

  if (ImGui::CollapsingHeader("Relationships", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BeginChild("DbAssociations", ImVec2(0, 220), true);
    if (associations.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no associations)");
    } else {
      for (const auto& row : associations) {
        const std::string source = FormatAssociationNode(
            row.source_kind, row.source_label, row.source_source_id, row.source_memory_id);
        const std::string target = FormatAssociationNode(
            row.target_kind, row.target_label, row.target_source_id, row.target_memory_id);
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f), "%s", row.edge_type.c_str());
        ImGui::SameLine();
        ImGui::TextWrapped("%s -> %s", source.c_str(), target.c_str());
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                           "weight=%s decay=%s reinforced=%s",
                           FormatDouble(row.weight).c_str(),
                           FormatDouble(row.decay_rate).c_str(),
                           NullOrValue(row.last_reinforced).c_str());
        ImGui::Separator();
      }
    }
    ImGui::EndChild();
  }

  if (ImGui::CollapsingHeader("Memories", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BeginChild("DbMemories", ImVec2(0, 220), true);
    if (memories.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no memories)");
    } else {
      for (const auto& row : memories) {
        if (ImGui::CollapsingHeader(FormatMemoryTitle(row).c_str())) {
          ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f),
                             "episode=%lld embedding=%lld source=%s",
                             row.episode_id,
                             row.embedding_id,
                             row.source_id.c_str());
          ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                             "signals=%lld strength=%s connectivity=%s source_conf=%s",
                             row.n_signals,
                             FormatDouble(row.strength).c_str(),
                             FormatDouble(row.connectivity).c_str(),
                             FormatDouble(row.source_reliability).c_str());
          ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                             "retrieved=%lld used=%lld start=%s end=%s",
                             row.retrieved_count,
                             row.used_count,
                             NullOrValue(row.start_ts).c_str(),
                             NullOrValue(row.end_ts).c_str());
          if (!row.label.empty()) {
            ImGui::TextWrapped("%s", row.label.c_str());
          }
        }
      }
    }
    ImGui::EndChild();
  }

  if (ImGui::CollapsingHeader("Signals")) {
    ImGui::BeginChild("DbSignals", ImVec2(0, 180), true);
    if (signals.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no signals)");
    } else {
      for (const auto& row : signals) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f),
                           "#%lld mem=%lld %s/%s",
                           row.signal_id,
                           row.memory_id,
                           row.source_id.c_str(),
                           row.modality.c_str());
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                           "embedding=%lld pos=%lld score=%s salience=%s thresh=%s",
                           row.embedding_id,
                           row.serial_position,
                           FormatDouble(row.score).c_str(),
                           FormatDouble(row.salience).c_str(),
                           FormatDouble(row.threshold_t).c_str());
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "ts=%s mime=%s",
                           NullOrValue(row.timestamp).c_str(),
                           row.mime.empty() ? "-" : row.mime.c_str());
        ImGui::Separator();
      }
    }
    ImGui::EndChild();
  }

  if (ImGui::CollapsingHeader("Episodes")) {
    ImGui::BeginChild("DbEpisodes", ImVec2(0, 140), true);
    if (episodes.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no episodes)");
    } else {
      for (const auto& row : episodes) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f),
                           "#%lld %s", row.episode_id,
                           row.boundary_type.empty() ? "(no boundary_type)" : row.boundary_type.c_str());
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                           "start=%s end=%s created=%s",
                           NullOrValue(row.start_ts).c_str(),
                           NullOrValue(row.end_ts).c_str(),
                           NullOrValue(row.created_at).c_str());
        ImGui::Separator();
      }
    }
    ImGui::EndChild();
  }
}

void ChatWindow::RenderContextTab() {
  if (!state_.context || !state_.working_memory) return;

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

  // Messages Section (derived from working memory; no local history)
  if (ImGui::CollapsingHeader("Messages (Working Memory)", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BeginChild("Messages", ImVec2(0, 300), true);
    const auto messages = BuildMessagesFromWorkingMemory(*state_.working_memory);
    for (size_t i = 0; i < messages.size(); ++i) {
      const auto& msg = messages[i];
      ImGui::TextColored(RoleColor(msg.role), "[%zu] %s:", i, RolePrefix(msg.role).c_str());
      std::string content = Truncate(msg.content, 500);
      ImGui::TextWrapped("%s", content.c_str());
      ImGui::Separator();
    }
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
  std::optional<int> idle_seconds_remaining;
  bool idle_pending = false;
  const bool consolidating
      = state_.idle_consolidating && state_.idle_consolidating->load();

  if (state_.status) {
    std::lock_guard<std::mutex> lock(state_.status->mu);
    processing_ms = state_.status->processing_latency_ms;
    tokens_used = state_.status->tokens_used;
    idle_seconds_remaining = state_.status->idle_seconds_remaining;
    idle_pending = state_.status->idle_pending;
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
  if (consolidating) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "| consolidating...");
  } else if (idle_pending && idle_seconds_remaining.has_value()) {
    ImGui::SameLine();
    ImGui::TextColored(
        ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "| idle=%ds", *idle_seconds_remaining);
  }
  ImGui::SameLine(ImGui::GetWindowWidth() - 250);
  ImGui::Text("%s", oss.str().c_str());
}

void ChatWindow::RenderInputBox() {
  ImGui::Separator();

  ImGui::Text(">");
  ImGui::SameLine();

  // Check if we should focus the input
  bool generating = state_.generating && *state_.generating;
  const std::string before_input = input_buffer_;

  ImGui::PushItemWidth(-1);
  ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue;

  bool submitted = ImGui::InputText("##input", input_buffer_, sizeof(input_buffer_), flags);
  const std::string after_input = input_buffer_;
  const bool input_changed = (before_input != after_input);

  if (input_changed && state_.input_activity_requested) {
    state_.input_activity_requested->store(true);
  }

  if (input_changed && input_buffer_[0] != '\0'
      && state_.typing_interrupt_requested) {
    state_.typing_interrupt_requested->store(true);
  }

  if (submitted && !generating && input_buffer_[0] != '\0') {
    pending_message_ = input_buffer_;
    input_buffer_[0] = '\0';
    scroll_chat_to_bottom_ = true;
  }

  if (state_.has_input_draft) {
    state_.has_input_draft->store(input_buffer_[0] != '\0' || !pending_message_.empty());
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
