#include "chat_window.hpp"

#include <imgui.h>
#include <implot.h>

#if defined(__APPLE__)
#ifndef GL_SILENCE_DEPRECATION
#define GL_SILENCE_DEPRECATION
#endif
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <unordered_set>

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

std::string VoiceSpeakerDisplayName(
    const std::string& speaker_id,
    const std::optional<std::string>& self_speaker_id) {
  if (self_speaker_id.has_value() && *self_speaker_id == speaker_id) {
    return "Me";
  }
  if (!speaker_id.empty()) {
    return speaker_id;
  }
  return "speaker:unknown";
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

bool HasImagePreview(const MemoryMediaPreview& mem) {
  return mem.modality == "image" && !mem.media_bytes.empty()
      && mem.image_width > 0 && mem.image_height > 0
      && (mem.image_channels == 3 || mem.image_channels == 4);
}

bool HasAudioPreview(const MemoryMediaPreview& mem) {
  return mem.modality == "audio" && !mem.audio_samples.empty()
      && mem.audio_sample_rate > 0;
}

void RequestMediaLoad(const std::shared_ptr<DatabaseExplorerState>& state,
                      const std::string& kind,
                      long long memory_id,
                      long long signal_id) {
  if (!state) {
    return;
  }
  std::lock_guard<std::mutex> lock(state->mu);
  state->media_load_request = MediaLoadRequest{kind, memory_id, signal_id};
  state->last_refresh_status = "Loading media...";
}

std::string FormatPercent(double v) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1) << (v * 100.0) << "%";
  return ss.str();
}

std::string FormatPromptRatio(std::int64_t baseline, std::int64_t current) {
  if (baseline <= 0 || current <= 0) {
    return "-";
  }
  std::ostringstream ss;
  if (current < baseline) {
    ss << std::fixed << std::setprecision(1)
       << (static_cast<double>(baseline) / static_cast<double>(current))
       << "x smaller";
  } else if (current > baseline) {
    ss << std::fixed << std::setprecision(1)
       << (static_cast<double>(current) / static_cast<double>(baseline))
       << "x larger";
  } else {
    ss << "same";
  }
  return ss.str();
}

bool NearlyEqual(double a, double b) {
  return std::abs(a - b) < 1e-6;
}

bool EditMultilineString(const char* label,
                         std::string& value,
                         const ImVec2& size = ImVec2(0, 0)) {
  const std::size_t buffer_size = std::max<std::size_t>(8192, value.size() + 1024);
  std::vector<char> buffer(buffer_size, '\0');
  std::snprintf(buffer.data(), buffer.size(), "%s", value.c_str());
  if (!ImGui::InputTextMultiline(label, buffer.data(), buffer.size(), size)) {
    return false;
  }
  value = buffer.data();
  return true;
}

bool EditSingleLineString(const char* label, std::string& value) {
  const std::size_t buffer_size = std::max<std::size_t>(512, value.size() + 128);
  std::vector<char> buffer(buffer_size, '\0');
  std::snprintf(buffer.data(), buffer.size(), "%s", value.c_str());
  if (!ImGui::InputText(label, buffer.data(), buffer.size())) {
    return false;
  }
  value = buffer.data();
  return true;
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
  if (source_id.rfind("chat/user", 0) == 0) return "user";
  if (source_id.rfind("chat/assistant", 0) == 0) return "assistant";
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

ImU32 NodeFillColor(const std::string& kind) {
  if (kind == "WORKING") return IM_COL32(40, 130, 180, 220);
  if (kind == "LONG_TERM") return IM_COL32(60, 160, 90, 220);
  if (kind == "LABEL") return IM_COL32(160, 120, 40, 220);
  if (kind == "ASSOCIATION") return IM_COL32(130, 80, 160, 220);
  return IM_COL32(90, 90, 90, 220);
}

ImU32 EdgeColor(const std::string& edge_type) {
  if (edge_type == "derived_from") return IM_COL32(255, 180, 80, 220);
  if (edge_type == "has_label") return IM_COL32(140, 210, 100, 220);
  if (edge_type == "references") return IM_COL32(90, 180, 255, 220);
  return IM_COL32(190, 190, 190, 180);
}

std::string GraphNodeTitle(const std::string& kind,
                           const std::string& label,
                           const std::string& source_id,
                           long long memory_id) {
  if (!label.empty()) {
    return Truncate(label, 40);
  }
  if (!source_id.empty()) {
    return source_id;
  }
  if (!kind.empty()) {
    return kind + " #" + std::to_string(memory_id);
  }
  return "#" + std::to_string(memory_id);
}

enum class DatabaseExplorerFilter {
  All = 0,
  Associations,
  Labels,
  LongTerm,
  Working,
  Temporal,
};

const char* const kDatabaseExplorerFilterLabels[] = {
    "All",
    "Associations",
    "Labels",
    "Long-Term",
    "Working",
    "Temporal",
};

bool MatchesDatabaseExplorerFilter(const DatabaseMemoryRow& row,
                                   DatabaseExplorerFilter filter) {
  switch (filter) {
    case DatabaseExplorerFilter::All:
      return true;
    case DatabaseExplorerFilter::Labels:
      return row.kind == "LABEL";
    case DatabaseExplorerFilter::LongTerm:
      return row.kind == "LONG_TERM";
    case DatabaseExplorerFilter::Working:
      return row.kind == "WORKING";
    case DatabaseExplorerFilter::Associations:
    case DatabaseExplorerFilter::Temporal:
      return false;
  }
  return true;
}

std::string FormatEvictionTitle(const DatabaseEvictionRow& row) {
  std::string title = "#" + std::to_string(row.memory_id) + " ";
  if (!row.label.empty()) {
    title += Truncate(row.label, 56);
  } else if (!row.source_id.empty()) {
    title += row.source_id;
  } else {
    title += row.kind;
  }
  return title;
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

ChatWindow::~ChatWindow() {
  for (const auto& item : webcam_image_textures_) {
    if (item.second.texture_id != 0) {
      const GLuint texture_id = item.second.texture_id;
      glDeleteTextures(1, &texture_id);
    }
  }
  if (signal_filter_image_texture_.texture_id != 0) {
    const GLuint texture_id = signal_filter_image_texture_.texture_id;
    glDeleteTextures(1, &texture_id);
  }
}

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
      RenderPlaygroundTab();
      break;
    case 2:
      RenderVoiceTab();
      break;
    case 3:
      RenderWebcamTab();
      break;
    case 4:
      RenderEventsTab();
      break;
    case 5:
      RenderMetricsTab();
      break;
    case 6:
      RenderSettingsTab();
      break;
    case 7:
      RenderDatabaseTab();
      break;
    case 8:
      RenderGraphTab();
      break;
    case 9:
      RenderContextTab();
      break;
    case 10:
      RenderLogsTab();
      break;
  }
  ImGui::EndChild();

  RenderMediaModal();
  if (selected_tab_ == 0) {
    RenderInputBox();
  }
  RenderStatusBar();

  ImGui::End();
}

void ChatWindow::RenderTabBar() {
  if (ImGui::BeginTabBar("MainTabs")) {
    if (ImGui::BeginTabItem("Chat")) {
      selected_tab_ = 0;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Playground")) {
      selected_tab_ = 1;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Voice")) {
      selected_tab_ = 2;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Webcam")) {
      selected_tab_ = 3;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Events")) {
      selected_tab_ = 4;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Metrics")) {
      selected_tab_ = 5;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Settings")) {
      selected_tab_ = 6;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("DB")) {
      selected_tab_ = 7;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Graph")) {
      selected_tab_ = 8;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Context")) {
      selected_tab_ = 9;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Telemetry")) {
      selected_tab_ = 10;
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void ChatWindow::RenderEventsTab() {
  RenderChunksTab();
  ImGui::Separator();
  ImGui::Spacing();
  RenderMemoryTab();
}

void ChatWindow::RenderPlaygroundTab() {
  if (!state_.playground) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                       "Playground state is unavailable.");
    return;
  }

  bool processing = false;
  std::string last_input_kind;
  std::string last_input_summary;
  std::vector<MemoryMediaPreview> retrieved_memories;
  double last_total_ms = 0.0;
  double last_process_ms = 0.0;
  std::optional<std::string> last_status;
  std::optional<std::string> last_error;
  {
    std::lock_guard<std::mutex> lock(state_.playground->mu);
    processing = state_.playground->processing;
    last_input_kind = state_.playground->last_input_kind;
    last_input_summary = state_.playground->last_input_summary;
    retrieved_memories = state_.playground->retrieved_memories;
    last_total_ms = state_.playground->last_total_ms;
    last_process_ms = state_.playground->last_process_ms;
    last_status = state_.playground->last_status;
    last_error = state_.playground->last_error;
  }

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
                     "CORTEXT PLAYGROUND");
  ImGui::TextWrapped(
      "Run text, visual, or audio inputs through Cortext retrieval without "
      "calling the chat model. Retrieved image and audio memories load only "
      "when you open them.");

  ImGui::Separator();
  ImGui::BeginDisabled(processing);
  ImGui::SetNextItemWidth(-120.0f);
  const bool enter_pressed = ImGui::InputTextWithHint(
      "##playground_text",
      "Type text and press Enter",
      playground_text_buffer_,
      sizeof(playground_text_buffer_),
      ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  const bool run_clicked = ImGui::Button("Run text");
  if (enter_pressed || run_clicked) {
    std::string text(playground_text_buffer_);
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start != std::string::npos) {
      const auto end = text.find_last_not_of(" \t\r\n");
      text = text.substr(start, end - start + 1);
      {
        std::lock_guard<std::mutex> lock(state_.playground->mu);
        state_.playground->pending_text = std::move(text);
        state_.playground->process_text_requested = true;
        state_.playground->last_status = "Queued text playground run.";
        state_.playground->last_error.reset();
      }
      playground_text_buffer_[0] = '\0';
    }
  }
  ImGui::EndDisabled();

  ImGui::Separator();
  bool webcam_supported = false;
  bool webcam_capturing = false;
  std::string webcam_capture_mode;
  std::optional<MemoryMediaPreview> signal_filter_image;
  if (state_.webcam) {
    std::lock_guard<std::mutex> lock(state_.webcam->mu);
    webcam_supported = state_.webcam->supported;
    webcam_capturing = state_.webcam->capturing;
    webcam_capture_mode = state_.webcam->capture_mode;
    signal_filter_image = state_.webcam->signal_filter_image;
  }

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
                     "Live media playground inputs");
  if (!webcam_supported) {
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                       "Direct camera and microphone input is unavailable in this build.");
  } else if (state_.webcam) {
    ImGui::BeginDisabled(webcam_capturing);
    if (ImGui::Button("Record audio##playground")) {
      std::lock_guard<std::mutex> lock(state_.webcam->mu);
      state_.webcam->capture_mode = "audio";
      state_.webcam->image_capture_pending = false;
      state_.webcam->start_requested = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Capture video##playground")) {
      std::lock_guard<std::mutex> lock(state_.webcam->mu);
      state_.webcam->capture_mode = "video";
      state_.webcam->image_capture_pending = false;
      state_.webcam->start_requested = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Capture image##playground")) {
      std::lock_guard<std::mutex> lock(state_.webcam->mu);
      state_.webcam->capture_mode = "image";
      state_.webcam->image_capture_pending = true;
      state_.webcam->start_requested = true;
    }
    ImGui::EndDisabled();
    if (webcam_capturing) {
      ImGui::SameLine();
      if (ImGui::Button("Stop capture##playground")) {
        std::lock_guard<std::mutex> lock(state_.webcam->mu);
        state_.webcam->stop_requested = true;
        state_.webcam->image_capture_pending = false;
      }
    }
    ImGui::SameLine();
    ImGui::TextColored(webcam_capturing ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                                        : ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                       webcam_capturing ? "capture: live" : "capture: stopped");
    if (!webcam_capture_mode.empty()) {
      ImGui::SameLine();
      ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                         "mode=%s",
                         webcam_capture_mode.c_str());
    }

    if (signal_filter_image.has_value()
        && HasImagePreview(*signal_filter_image)) {
      ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                         "latest accepted visual signal is displayed below as soon as it is processed");
    }
  }

  ImGui::Separator();
  if (processing) {
    ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f),
                       "Processing playground input...");
  }
  if (last_status.has_value()) {
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f),
                       "%s",
                       last_status->c_str());
  }
  if (last_error.has_value()) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                       "%s",
                       last_error->c_str());
  }
  if (!last_input_kind.empty()) {
    ImGui::Text("Last input: %s / %s",
                last_input_kind.c_str(),
                last_input_summary.c_str());
    if (last_total_ms > 0.0 || last_process_ms > 0.0) {
      ImGui::Text("Latency: total %.2f ms, process %.2f ms",
                  last_total_ms,
                  last_process_ms);
    }
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
                     "Retrieved memories");
  std::size_t text_count = 0;
  std::size_t image_count = 0;
  std::size_t audio_count = 0;
  std::size_t other_count = 0;
  for (const auto& mem : retrieved_memories) {
    if (mem.modality == "text") {
      ++text_count;
    } else if (mem.modality == "image") {
      ++image_count;
    } else if (mem.modality == "audio") {
      ++audio_count;
    } else {
      ++other_count;
    }
  }
  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                     "results=%zu text=%zu image=%zu audio=%zu other=%zu",
                     retrieved_memories.size(),
                     text_count,
                     image_count,
                     audio_count,
                     other_count);
  if (retrieved_memories.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "(no playground retrieval results yet)");
    return;
  }

  const ImGuiTableFlags flags = ImGuiTableFlags_Borders
                                | ImGuiTableFlags_RowBg
                                | ImGuiTableFlags_Resizable
                                | ImGuiTableFlags_SizingStretchProp;
  ImGui::BeginChild("PlaygroundRetrievedMemories", ImVec2(0, 0), true);
  if (ImGui::BeginTable("PlaygroundRetrievedMemoriesTable", 6, flags)) {
    ImGui::TableSetupColumn("id", ImGuiTableColumnFlags_WidthFixed, 54.0f);
    ImGui::TableSetupColumn("type", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("source", ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableSetupColumn("score", ImGuiTableColumnFlags_WidthFixed, 84.0f);
    ImGui::TableSetupColumn("use", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("memory");
    ImGui::TableHeadersRow();

    for (const auto& mem : retrieved_memories) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%lld", mem.memory_id);
      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(mem.modality == "image"
                             ? ImVec4(0.5f, 0.8f, 1.0f, 1.0f)
                             : (mem.modality == "audio"
                                    ? ImVec4(0.8f, 0.7f, 1.0f, 1.0f)
                                    : ImVec4(0.8f, 1.0f, 0.7f, 1.0f)),
                         "%s",
                         mem.modality.c_str());
      ImGui::TableSetColumnIndex(2);
      ImGui::TextWrapped("%s", mem.source_id.c_str());
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%.3f", mem.composite_score);
      if (mem.relevance != 0.0) {
        ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                           "rel %.3f",
                           mem.relevance);
      }
      ImGui::TableSetColumnIndex(4);
      ImGui::Text("%lld/%lld", mem.used_count, mem.retrieved_count);
      ImGui::TableSetColumnIndex(5);
      ImGui::PushTextWrapPos();
      ImGui::TextWrapped("%s", mem.preview.c_str());
      if (!mem.mimetype.empty()) {
        ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                           "%s, %zu bytes",
                           mem.mimetype.c_str(),
                           mem.byte_count);
      }
      if (mem.modality == "image") {
        if (ImGui::SmallButton(("Open image##playground_image_"
                                + std::to_string(mem.memory_id)).c_str())) {
          RequestMediaLoad(state_.db_explorer, "memory_image",
                           mem.memory_id, 0);
        }
      } else if (mem.modality == "audio") {
        if (ImGui::SmallButton(("Open audio##playground_audio_"
                                + std::to_string(mem.memory_id)).c_str())) {
          RequestMediaLoad(state_.db_explorer, "memory_audio",
                           mem.memory_id, 0);
        }
      }
      ImGui::PopTextWrapPos();
    }

    ImGui::EndTable();
  }
  ImGui::EndChild();
}

void ChatWindow::RenderWebcamTab() {
  if (!state_.webcam) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                       "Webcam state is unavailable.");
    return;
  }

  bool supported = false;
  bool available = false;
  bool capturing = false;
  int video_width = 0;
  int video_height = 0;
  std::uint64_t video_frames_captured = 0;
  std::uint64_t video_frames_ingested = 0;
  std::uint64_t video_frames_filtered = 0;
  std::uint64_t audio_chunks_captured = 0;
  std::uint64_t audio_chunks_ingested = 0;
  std::uint64_t audio_chunks_filtered = 0;
  double last_video_ingest_ms = 0.0;
  double last_audio_ingest_ms = 0.0;
  double mean_video_ingest_ms = 0.0;
  double mean_audio_ingest_ms = 0.0;
  std::optional<MemoryMediaPreview> signal_filter_image;
  std::uint64_t signal_filter_image_version = 0;
  std::string last_signal_filter_modality;
  std::string last_signal_filter_reason;
  double last_signal_filter_score = 0.0;
  double last_signal_filter_threshold = 0.0;
  std::uint64_t retrieval_update_count = 0;
  std::vector<MemoryMediaPreview> retrieved_memories;
  std::deque<MemoryMediaPreview> live_memory_feed;
  bool audio_playing = false;
  long long audio_playing_memory_id = 0;
  std::optional<std::string> last_error;
  {
    std::lock_guard<std::mutex> lock(state_.webcam->mu);
    supported = state_.webcam->supported;
    available = state_.webcam->available;
    capturing = state_.webcam->capturing;
    video_width = state_.webcam->video_width;
    video_height = state_.webcam->video_height;
    video_frames_captured = state_.webcam->video_frames_captured;
    video_frames_ingested = state_.webcam->video_frames_ingested;
    video_frames_filtered = state_.webcam->video_frames_filtered;
    audio_chunks_captured = state_.webcam->audio_chunks_captured;
    audio_chunks_ingested = state_.webcam->audio_chunks_ingested;
    audio_chunks_filtered = state_.webcam->audio_chunks_filtered;
    last_video_ingest_ms = state_.webcam->last_video_ingest_ms;
    last_audio_ingest_ms = state_.webcam->last_audio_ingest_ms;
    mean_video_ingest_ms = state_.webcam->mean_video_ingest_ms;
    mean_audio_ingest_ms = state_.webcam->mean_audio_ingest_ms;
    signal_filter_image = state_.webcam->signal_filter_image;
    signal_filter_image_version = state_.webcam->signal_filter_image_version;
    last_signal_filter_modality = state_.webcam->last_signal_filter_modality;
    last_signal_filter_reason = state_.webcam->last_signal_filter_reason;
    last_signal_filter_score = state_.webcam->last_signal_filter_score;
    last_signal_filter_threshold = state_.webcam->last_signal_filter_threshold;
    retrieval_update_count = state_.webcam->retrieval_update_count;
    retrieved_memories = state_.webcam->retrieved_memories;
    live_memory_feed = state_.webcam->live_memory_feed;
    audio_playing = state_.webcam->audio_playing;
    audio_playing_memory_id = state_.webcam->audio_playing_memory_id;
    last_error = state_.webcam->last_error;
  }

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
                     "DIRECT CAMERA + MICROPHONE INGEST");
  ImGui::TextWrapped(
      "This path feeds raw webcam frames and microphone PCM directly into Cortext. "
      "It does not run speech-to-text, diarization, speaker labeling, or reply generation.");

  if (!supported) {
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                       "Direct webcam capture is only available on macOS builds.");
    return;
  }

  if (capturing) {
    if (ImGui::Button("Stop Webcam Ingest")) {
      std::lock_guard<std::mutex> lock(state_.webcam->mu);
      state_.webcam->stop_requested = true;
      state_.webcam->image_capture_pending = false;
    }
  } else {
    if (ImGui::Button("Start Webcam Ingest")) {
      std::lock_guard<std::mutex> lock(state_.webcam->mu);
      state_.webcam->capture_mode = "all";
      state_.webcam->image_capture_pending = false;
      state_.webcam->start_requested = true;
    }
  }

  ImGui::SameLine();
  ImGui::TextColored(capturing ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                               : ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                     capturing ? "Capture: live" : "Capture: stopped");

  if (!available) {
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                       "Camera or microphone capture is unavailable.");
  }
  if (last_error.has_value()) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                       "%s",
                       last_error->c_str());
  }

  ImGui::Separator();
  ImGui::Text("Video: %dx%d", video_width, video_height);
  ImGui::Text("Frames captured: %llu",
              static_cast<unsigned long long>(video_frames_captured));
  ImGui::Text("Frames ingested: %llu",
              static_cast<unsigned long long>(video_frames_ingested));
  ImGui::Text("Frames skipped by SignalFilter: %llu",
              static_cast<unsigned long long>(video_frames_filtered));
  ImGui::Text("Last / mean video ingest: %.2f ms / %.2f ms",
              last_video_ingest_ms,
              mean_video_ingest_ms);

  ImGui::Separator();
  ImGui::Text("Audio chunks captured: %llu",
              static_cast<unsigned long long>(audio_chunks_captured));
  ImGui::Text("Audio chunks ingested: %llu",
              static_cast<unsigned long long>(audio_chunks_ingested));
  ImGui::Text("Audio chunks skipped by SignalFilter: %llu",
              static_cast<unsigned long long>(audio_chunks_filtered));
  ImGui::Text("Last / mean audio ingest: %.2f ms / %.2f ms",
              last_audio_ingest_ms,
              mean_audio_ingest_ms);

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
                     "Last visual signal accepted by SignalFilter");
  if (!last_signal_filter_modality.empty()) {
    ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f),
                       "last filter: %s / %s score %.3f threshold %.3f",
                       last_signal_filter_modality.c_str(),
                       last_signal_filter_reason.empty()
                           ? "-"
                           : last_signal_filter_reason.c_str(),
                       last_signal_filter_score,
                       last_signal_filter_threshold);
  }
  if (signal_filter_image.has_value() && HasImagePreview(*signal_filter_image)) {
    const auto& preview = *signal_filter_image;
    ImageTexture& cached = signal_filter_image_texture_;
    if (cached.texture_id == 0
        || cached.width != preview.image_width
        || cached.height != preview.image_height
        || cached.version != signal_filter_image_version) {
      if (cached.texture_id != 0) {
        const GLuint old_texture_id = cached.texture_id;
        glDeleteTextures(1, &old_texture_id);
      }
      GLuint texture_id = 0;
      glGenTextures(1, &texture_id);
      glBindTexture(GL_TEXTURE_2D, texture_id);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexImage2D(GL_TEXTURE_2D,
                   0,
                   preview.image_channels == 4 ? GL_RGBA : GL_RGB,
                   preview.image_width,
                   preview.image_height,
                   0,
                   preview.image_channels == 4 ? GL_RGBA : GL_RGB,
                   GL_UNSIGNED_BYTE,
                   preview.media_bytes.data());
      glBindTexture(GL_TEXTURE_2D, 0);
      cached.texture_id = texture_id;
      cached.width = preview.image_width;
      cached.height = preview.image_height;
      cached.version = signal_filter_image_version;
    }
    const float max_side = 260.0f;
    const float scale = std::min(
        max_side / static_cast<float>(std::max(1, preview.image_width)),
        max_side / static_cast<float>(std::max(1, preview.image_height)));
    ImGui::Image(static_cast<ImTextureID>(cached.texture_id),
                 ImVec2(static_cast<float>(preview.image_width) * scale,
                        static_cast<float>(preview.image_height) * scale));
  } else {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "(no visual frame has passed the SignalFilter yet)");
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
                     "Retrieved memories from latest media ingest");
  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                     "updates=%llu results=%zu",
                     static_cast<unsigned long long>(retrieval_update_count),
                     retrieved_memories.size());
  if (retrieved_memories.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "(no memories retrieved for the latest webcam or mic signal)");
  }

  const ImGuiTableFlags flags = ImGuiTableFlags_Borders
                                | ImGuiTableFlags_RowBg
                                | ImGuiTableFlags_Resizable
                                | ImGuiTableFlags_SizingStretchProp;
  if (!retrieved_memories.empty()
      && ImGui::BeginTable("WebcamRetrievedMemories", 6, flags)) {
    ImGui::TableSetupColumn("id", ImGuiTableColumnFlags_WidthFixed, 54.0f);
    ImGui::TableSetupColumn("type", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("source", ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableSetupColumn("score", ImGuiTableColumnFlags_WidthFixed, 84.0f);
    ImGui::TableSetupColumn("use", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("memory");
    ImGui::TableHeadersRow();

    for (const auto& mem : retrieved_memories) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%lld", mem.memory_id);

      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(mem.modality == "image"
                             ? ImVec4(0.5f, 0.8f, 1.0f, 1.0f)
                             : (mem.modality == "audio"
                                    ? ImVec4(0.8f, 0.7f, 1.0f, 1.0f)
                                    : ImVec4(0.8f, 1.0f, 0.7f, 1.0f)),
                         "%s",
                         mem.modality.c_str());

      ImGui::TableSetColumnIndex(2);
      ImGui::TextWrapped("%s", mem.source_id.c_str());

      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%.3f", mem.composite_score);
      if (mem.relevance != 0.0) {
        ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                           "rel %.3f",
                           mem.relevance);
      }

      ImGui::TableSetColumnIndex(4);
      ImGui::Text("%lld/%lld", mem.used_count, mem.retrieved_count);

      ImGui::TableSetColumnIndex(5);
      ImGui::PushTextWrapPos();
      ImGui::TextWrapped("%s", mem.preview.c_str());
      if (!mem.mimetype.empty()) {
        ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                           "%s, %zu bytes",
                           mem.mimetype.c_str(),
                           mem.byte_count);
      }
      if (HasImagePreview(mem)) {
        if (ImGui::SmallButton(("Open image##webcam_image_"
                                + std::to_string(mem.memory_id)).c_str())) {
          RequestMediaLoad(state_.db_explorer, "memory_image",
                           mem.memory_id, 0);
        }
      } else if (mem.modality == "image") {
        if (ImGui::SmallButton(("Open image##webcam_image_lazy_"
                                + std::to_string(mem.memory_id)).c_str())) {
          RequestMediaLoad(state_.db_explorer, "memory_image",
                           mem.memory_id, 0);
        }
      }

      if (mem.modality == "audio") {
        const bool playing_this = audio_playing
                                  && audio_playing_memory_id == mem.preview_id;
        if (playing_this) {
          if (ImGui::SmallButton(("Stop##webcam_audio_"
                                  + std::to_string(mem.memory_id)).c_str())) {
            std::lock_guard<std::mutex> lock(state_.webcam->mu);
            state_.webcam->audio_stop_requested = true;
          }
        } else if (ImGui::SmallButton(("Open audio##webcam_audio_"
                                       + std::to_string(mem.memory_id)).c_str())) {
          RequestMediaLoad(state_.db_explorer, "memory_audio",
                           mem.memory_id, 0);
        }
      }
      ImGui::PopTextWrapPos();
    }

    ImGui::EndTable();
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
                     "Live surfaced memory feed");
  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                     "rolling=%zu; this is fed directly from ingestion results, not DB refresh",
                     live_memory_feed.size());
  if (live_memory_feed.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "(no surfaced memories yet)");
    return;
  }

  ImGui::BeginChild("WebcamLiveMemoryFeed", ImVec2(0, 280), true);
  if (ImGui::BeginTable("WebcamLiveMemoryFeedTable", 5, flags)) {
    ImGui::TableSetupColumn("id", ImGuiTableColumnFlags_WidthFixed, 54.0f);
    ImGui::TableSetupColumn("type", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("score", ImGuiTableColumnFlags_WidthFixed, 84.0f);
    ImGui::TableSetupColumn("use", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("memory");
    ImGui::TableHeadersRow();

    int feed_row = 0;
    for (const auto& mem : live_memory_feed) {
      ImGui::PushID(feed_row++);
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%lld", mem.memory_id);

      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(mem.modality == "image"
                             ? ImVec4(0.5f, 0.8f, 1.0f, 1.0f)
                             : (mem.modality == "audio"
                                    ? ImVec4(0.8f, 0.7f, 1.0f, 1.0f)
                                    : ImVec4(0.8f, 1.0f, 0.7f, 1.0f)),
                         "%s",
                         mem.modality.c_str());

      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%.3f", mem.composite_score);
      if (mem.relevance != 0.0) {
        ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                           "rel %.3f",
                           mem.relevance);
      }

      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%lld/%lld", mem.used_count, mem.retrieved_count);

      ImGui::TableSetColumnIndex(4);
      ImGui::PushTextWrapPos();
      ImGui::TextWrapped("%s", mem.preview.c_str());
      if (mem.modality == "image") {
        if (ImGui::SmallButton("Open image")) {
          RequestMediaLoad(state_.db_explorer, "memory_image",
                           mem.memory_id, 0);
        }
      } else if (mem.modality == "audio") {
        if (ImGui::SmallButton("Open audio")) {
          RequestMediaLoad(state_.db_explorer, "memory_audio",
                           mem.memory_id, 0);
        }
      }
      ImGui::PopTextWrapPos();
      ImGui::PopID();
    }

    ImGui::EndTable();
  }
  ImGui::EndChild();
}

void ChatWindow::RenderChatTab() {
  if (!state_.mu) return;

  ImGui::BeginChild("ChatScroll", ImVec2(0, 0), false);

  constexpr float kScrollBottomSlack = 24.0f;
  const bool was_near_bottom
      = (ImGui::GetScrollMaxY() - ImGui::GetScrollY()) <= kScrollBottomSlack;

  std::size_t chat_message_count = 0;
  std::size_t partial_response_size = 0;
  bool generating = false;
  int generation_restarts = 0;

  {
    std::lock_guard<std::mutex> lock(*state_.mu);

    if (state_.chat_history) {
      chat_message_count = state_.chat_history->size();
      for (const auto& msg : *state_.chat_history) {
        ImGui::TextColored(RoleColor(msg.role), "%s:", RolePrefix(msg.role).c_str());
        ImGui::SameLine();
        ImGui::TextWrapped("%s", msg.content.c_str());
        ImGui::Separator();
      }
    }

    if (state_.partial_response) {
      partial_response_size = state_.partial_response->size();
    }

    generating = state_.generating && *state_.generating;
    if (generating) {
      int dots = (static_cast<int>(ImGui::GetTime() / 0.4) % 3) + 1;
      std::string status = "generating" + std::string(dots, '.');
      if (state_.generation_restarts && *state_.generation_restarts > 0) {
        generation_restarts = *state_.generation_restarts;
        status += " (restarted " + std::to_string(generation_restarts) +
                  "x with new context)";
      }
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "%s", status.c_str());
    }
  }

  const bool content_changed
      = chat_message_count != last_chat_message_count_
        || partial_response_size != last_partial_response_size_
        || generating != last_generating_;
  const bool should_auto_scroll
      = content_changed && (scroll_chat_to_bottom_ || was_near_bottom);
  if (should_auto_scroll) {
    ImGui::SetScrollHereY(1.0f);
  }

  last_chat_message_count_ = chat_message_count;
  last_partial_response_size_ = partial_response_size;
  last_generating_ = generating;
  scroll_chat_to_bottom_ = false;

  ImGui::EndChild();
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
                         "interrupt=%s cap_ignored=%s new_memories=%d retrieved=%zu",
                         probe.should_interrupt ? "true" : "false",
                         probe.interrupt_ignored_restart_cap ? "true" : "false",
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

void ChatWindow::RenderVoiceTab() {
  if (!state_.voice) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Voice state is unavailable.");
    return;
  }

  bool supported = false;
  bool available = false;
  bool speaker_attribution_available = false;
  std::string backend = "sherpa";
  bool listening = false;
  bool reply_enabled = true;
  bool retrieve_without_retain = false;
  bool assistant_speaking = false;
  std::string live_transcript;
  std::string last_utterance;
  std::string last_speaker_id;
  std::optional<std::string> self_speaker_id;
  std::vector<VoiceSpeakerPreview> speakers;
  std::vector<VoiceSegmentDebug> last_segments;
  std::vector<std::string> surfaced_memories;
  std::optional<std::string> last_error;
  {
    std::lock_guard<std::mutex> lock(state_.voice->mu);
    supported = state_.voice->supported;
    available = state_.voice->available;
    speaker_attribution_available = state_.voice->speaker_attribution_available;
    backend = state_.voice->backend;
    listening = state_.voice->listening;
    reply_enabled = state_.voice->reply_enabled;
    retrieve_without_retain = state_.voice->retrieve_without_retain;
    assistant_speaking = state_.voice->assistant_speaking;
    live_transcript = state_.voice->live_transcript;
    last_utterance = state_.voice->last_utterance;
    last_speaker_id = state_.voice->last_speaker_id;
    self_speaker_id = state_.voice->self_speaker_id;
    speakers = state_.voice->speakers;
    last_segments = state_.voice->last_segments;
    surfaced_memories = state_.voice->surfaced_memories;
    last_error = state_.voice->last_error;
  }

  std::deque<ChatMessage> chat_history;
  if (state_.mu && state_.chat_history) {
    std::lock_guard<std::mutex> lock(*state_.mu);
    chat_history = *state_.chat_history;
  }

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "VOICE");
  if (!supported) {
    ImGui::TextWrapped("Voice chat is only available on macOS builds.");
    return;
  }

  if (listening) {
    if (ImGui::Button("Stop Listening")) {
      std::lock_guard<std::mutex> lock(state_.voice->mu);
      state_.voice->stop_requested = true;
    }
  } else {
    if (ImGui::Button("Start Listening")) {
      std::lock_guard<std::mutex> lock(state_.voice->mu);
      state_.voice->start_requested = true;
    }
  }

  ImGui::SameLine();
  int backend_index = backend == "whisper" ? 1 : 0;
  if (ImGui::BeginCombo("Backend", backend_index == 1
                                       ? "Whisper.cpp tinydiarize"
                                       : "Sherpa diarization")) {
    if (ImGui::Selectable("Sherpa diarization", backend_index == 0)) {
      std::lock_guard<std::mutex> lock(state_.voice->mu);
      state_.voice->backend = "sherpa";
      state_.voice->reply_toggle_dirty = true;
    }
    if (ImGui::Selectable("Whisper.cpp tinydiarize", backend_index == 1)) {
      std::lock_guard<std::mutex> lock(state_.voice->mu);
      state_.voice->backend = "whisper";
      state_.voice->reply_toggle_dirty = true;
    }
    ImGui::EndCombo();
  }

  bool next_reply_enabled = reply_enabled;
  if (ImGui::Checkbox("Reply enabled", &next_reply_enabled)) {
    std::lock_guard<std::mutex> lock(state_.voice->mu);
    state_.voice->reply_enabled = next_reply_enabled;
    state_.voice->reply_toggle_dirty = true;
  }

  bool next_retrieve_without_retain = retrieve_without_retain;
  if (ImGui::Checkbox("Retrieve without retain", &next_retrieve_without_retain)) {
    std::lock_guard<std::mutex> lock(state_.voice->mu);
    state_.voice->retrieve_without_retain = next_retrieve_without_retain;
    state_.voice->reply_toggle_dirty = true;
  }

  ImGui::TextColored(listening ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                               : ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                     listening ? "Mic: live" : "Mic: stopped");
  ImGui::SameLine();
  ImGui::TextColored(assistant_speaking ? ImVec4(0.3f, 1.0f, 1.0f, 1.0f)
                                        : ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                     assistant_speaking ? "Assistant audio: speaking"
                                        : "Assistant audio: idle");
  if (!available) {
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                       "Voice backend is unavailable. Check the active backend models and microphone permissions.");
  }
  const bool whisper_backend = backend == "whisper";
  ImGui::TextColored(speaker_attribution_available ? ImVec4(0.4f, 1.0f, 0.8f, 1.0f)
                                                   : ImVec4(0.8f, 0.6f, 0.2f, 1.0f),
                     whisper_backend
                         ? (speaker_attribution_available
                                ? "Speaker turns: tinydiarize active (turn boundaries only)"
                                : "Speaker turns: unavailable (check the whisper tinydiarize model)")
                         : (speaker_attribution_available
                                ? "Speaker streams: diarization active"
                                : "Speaker streams: unavailable (check the sherpa diarization models)"));
  if (last_error.has_value()) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", last_error->c_str());
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Live transcript");
  if (live_transcript.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       listening ? "(waiting for speech...)" : "(voice input stopped)");
  } else {
    ImGui::TextWrapped("%s", live_transcript.c_str());
  }
  if (!last_utterance.empty()) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Last finalized utterance");
    if (!last_speaker_id.empty()) {
      ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f),
                         "Speaker: %s",
                         VoiceSpeakerDisplayName(last_speaker_id, self_speaker_id).c_str());
    }
    ImGui::TextWrapped("%s", last_utterance.c_str());
  }

  if (speaker_attribution_available || !speakers.empty()) {
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Speakers");
    if (speakers.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                         "(no speaker-attributed utterances yet)");
    } else {
      for (const auto& speaker : speakers) {
        const std::string display_name
            = VoiceSpeakerDisplayName(speaker.speaker_id, self_speaker_id);
        ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f),
                           "%s",
                           display_name.c_str());
        ImGui::SameLine();
        if (!self_speaker_id.has_value() || *self_speaker_id != speaker.speaker_id) {
          if (ImGui::SmallButton(("This is me##" + speaker.speaker_id).c_str())) {
            std::lock_guard<std::mutex> lock(state_.voice->mu);
            state_.voice->self_speaker_id = speaker.speaker_id;
          }
        } else if (ImGui::SmallButton(("Clear me tag##" + speaker.speaker_id).c_str())) {
          std::lock_guard<std::mutex> lock(state_.voice->mu);
          state_.voice->self_speaker_id.reset();
        }
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                           "utterances: %zu",
                           speaker.utterance_count);
        if (!speaker.last_utterance.empty()) {
          ImGui::PushTextWrapPos();
          ImGui::BulletText("%s", speaker.last_utterance.c_str());
          ImGui::PopTextWrapPos();
        }
        ImGui::Spacing();
      }
    }
  }

  if (!last_segments.empty()) {
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Diarization segments");
    for (const auto& segment : last_segments) {
      ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f),
                         "[%.2fs - %.2fs] diarizer=%d mapped=%s",
                         segment.start_s,
                         segment.end_s,
                         segment.diarizer_speaker,
                         segment.speaker_id.c_str());
      if (!segment.text.empty()) {
        ImGui::PushTextWrapPos();
        ImGui::BulletText("%s", segment.text.c_str());
        ImGui::PopTextWrapPos();
      }
      ImGui::Spacing();
    }
  }

  if (!reply_enabled || !surfaced_memories.empty()) {
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Surfaced memories");
    if (surfaced_memories.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                         reply_enabled
                             ? "(memory cues only appear in reply-disabled voice mode)"
                             : "(no memory cues surfaced for the latest utterance)");
    } else {
      for (const auto& memory : surfaced_memories) {
        ImGui::PushTextWrapPos();
        ImGui::BulletText("%s", memory.c_str());
        ImGui::PopTextWrapPos();
      }
    }
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Conversation");
  ImGui::BeginChild("VoiceConversation", ImVec2(0, 0), false);
  if (chat_history.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "(no conversation yet)");
  } else {
    const std::size_t start = chat_history.size() > 12
                                  ? chat_history.size() - 12
                                  : 0;
    for (std::size_t i = start; i < chat_history.size(); ++i) {
      const auto& msg = chat_history[i];
      ImGui::PushTextWrapPos();
      ImGui::TextColored(RoleColor(msg.role), "%s:", RolePrefix(msg.role).c_str());
      ImGui::SameLine();
      ImGui::TextWrapped("%s", msg.content.c_str());
      ImGui::PopTextWrapPos();
      ImGui::Spacing();
    }
  }
  ImGui::EndChild();
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
    ImGui::TextColored(
        ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
        "prompt compare: full-history=%lld simple-rag=%lld cortext=%lld",
        static_cast<long long>(sample.full_history_prompt_tokens),
        static_cast<long long>(sample.rag_prompt_tokens),
        static_cast<long long>(sample.cortext_prompt_tokens));
    ImGui::TextColored(
        ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
        "full-history vs cortext: %s | simple-rag vs cortext: %s",
        FormatPromptRatio(sample.full_history_prompt_tokens,
                          sample.cortext_prompt_tokens).c_str(),
        FormatPromptRatio(sample.rag_prompt_tokens,
                          sample.cortext_prompt_tokens).c_str());
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
    const auto cortext_prompt_compare = BuildPlotValues(response_history, [](const auto& s) {
      return s.cortext_prompt_tokens;
    });
    const auto rag_prompt_compare = BuildPlotValues(response_history, [](const auto& s) {
      return s.rag_prompt_tokens;
    });
    const auto full_history_prompt_compare = BuildPlotValues(response_history, [](const auto& s) {
      return s.full_history_prompt_tokens;
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

    if (ImPlot::BeginPlot("Prompt strategy comparison", ImVec2(-1, 220))) {
      SetupHistoryAxes("est. prompt tokens");
      ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_None);
      PlotSeriesIfAvailable("full history", response_xs, full_history_prompt_compare);
      PlotSeriesIfAvailable("simple rag", response_xs, rag_prompt_compare);
      PlotSeriesIfAvailable("cortext", response_xs, cortext_prompt_compare);
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
  } else if (ImGui::BeginTable("MetricsResponses", 9,
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
    ImGui::TableSetupColumn("compare");
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
      ImGui::Text("fh=%lld rag=%lld ctx=%lld",
                  static_cast<long long>(sample.full_history_prompt_tokens),
                  static_cast<long long>(sample.rag_prompt_tokens),
                  static_cast<long long>(sample.cortext_prompt_tokens));
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
  ImGui::TextColored(
      ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
      "Raw retrieved vs injected into prompt vs dropped by chat-side filtering.");
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
      const char* type_icon = "UNKNOWN";
      ImVec4 type_color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
      switch (evt.type) {
        case MemoryEventType::STORED:
          type_icon = "STORED";
          type_color = ImVec4(0.3f, 1.0f, 0.3f, 1.0f);
          break;
        case MemoryEventType::RETRIEVED_RAW:
          type_icon = "RETRIEVED";
          type_color = ImVec4(0.3f, 0.6f, 1.0f, 1.0f);
          break;
        case MemoryEventType::INJECTED:
          type_icon = "INJECTED";
          type_color = ImVec4(0.9f, 0.8f, 0.3f, 1.0f);
          break;
        case MemoryEventType::DROPPED:
          type_icon = "DROPPED";
          type_color = ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
          break;
      }

      ImGui::TextColored(type_color, "%s #%d", type_icon, evt.memory_id);

      if (!evt.source_id.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "  source: %s", evt.source_id.c_str());
      }

      // Feedback counts
      ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "  retrieved: %lld | used: %lld",
                         evt.retrieved_count, evt.used_count);

      if (!evt.filter_reason.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.4f, 1.0f), "  filter: %s",
                           evt.filter_reason.c_str());
      }

      // Score and threshold
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "  score=%s thresh=%s",
                         FormatDouble(evt.composite_score).c_str(),
                         FormatDouble(evt.threshold_t).c_str());

      if (!evt.soft_anchors.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "  SoftAnchor");
        for (const auto& anchor : evt.soft_anchors) {
          ImGui::TextColored(
              ImVec4(0.7f, 0.9f, 1.0f, 1.0f),
              "    %s strength=%s likelihood=%s label=%s tier=%s",
              anchor.id.empty() ? "(unknown)" : anchor.id.c_str(),
              FormatDouble(anchor.strength).c_str(),
              FormatDouble(anchor.likelihood).c_str(),
              anchor.label.empty() ? "none" : anchor.label.c_str(),
              anchor.tier.empty() ? "unknown" : anchor.tier.c_str());
        }
      }

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

      if (evt.soft_anchor_enabled) {
        ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f),
                           "  soft anchor: mem=%lld states=%d links=%d create=%d update=%d none=%d last=%sus mean=%sus",
                           evt.stored_memory_id,
                           evt.soft_anchor_state_count,
                           evt.soft_anchor_link_count,
                           evt.soft_anchor_create_count,
                           evt.soft_anchor_update_count,
                           evt.soft_anchor_none_count,
                           FormatDouble(evt.soft_anchor_last_update_us).c_str(),
                           FormatDouble(evt.soft_anchor_mean_update_us).c_str());
      }

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
                         "score=%s salience=%s signals=%zu retrieved=%lld used=%lld",
                         FormatDouble(mem->composite_score).c_str(),
                         FormatDouble(mem->salience).c_str(),
                         mem->content.size(),
                         mem->retrieved_count,
                         mem->used_count);

      ImGui::TextColored(
          ImVec4(0.6f, 0.9f, 0.9f, 1.0f),
          "slot-level metrics only: avg score, max salience, avg arousal, usage counts");
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "aro_avg=%s",
                         FormatDouble(mem->arousal).c_str());

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
  int active_idle_consolidation_seconds = 0;
  double active_webcam_video_fps = 1.0;
  double draft_focus = 0.5;
  double draft_sensitivity = 0.5;
  double draft_stability = 0.5;
  int draft_idle_consolidation_seconds = 0;
  double draft_webcam_video_fps = 1.0;
  int default_idle_consolidation_seconds = 0;
  double default_webcam_video_fps = 1.0;
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
  std::optional<std::string> last_status;
  {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    active_focus = state_.settings->active_focus;
    active_sensitivity = state_.settings->active_sensitivity;
    active_stability = state_.settings->active_stability;
    active_idle_consolidation_seconds
        = state_.settings->active_idle_consolidation_seconds;
    active_webcam_video_fps = state_.settings->active_webcam_video_fps;
    draft_focus = state_.settings->draft_focus;
    draft_sensitivity = state_.settings->draft_sensitivity;
    draft_stability = state_.settings->draft_stability;
    draft_idle_consolidation_seconds
        = state_.settings->draft_idle_consolidation_seconds;
    draft_webcam_video_fps = state_.settings->draft_webcam_video_fps;
    default_idle_consolidation_seconds
        = state_.settings->default_idle_consolidation_seconds;
    default_webcam_video_fps = state_.settings->default_webcam_video_fps;
    active_model = state_.settings->active_model;
    draft_model = state_.settings->draft_model;
    default_model = state_.settings->default_model;
    active_memory_prompt_prefix = state_.settings->active_memory_prompt_prefix;
    draft_memory_prompt_prefix = state_.settings->draft_memory_prompt_prefix;
    default_memory_prompt_prefix = state_.settings->default_memory_prompt_prefix;
    active_memory_prompt_suffix = state_.settings->active_memory_prompt_suffix;
    draft_memory_prompt_suffix = state_.settings->draft_memory_prompt_suffix;
    default_memory_prompt_suffix = state_.settings->default_memory_prompt_suffix;
    apply_requested = state_.settings->apply_requested;
    last_status = state_.settings->last_apply_status;
  }

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "SETTINGS");
  ImGui::Separator();
  ImGui::TextWrapped(
      "Applied settings are saved and restored on the next app launch.");

  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                     "Active: F=%s  S=%s  T=%s",
                     FormatDouble(active_focus).c_str(),
                     FormatDouble(active_sensitivity).c_str(),
                     FormatDouble(active_stability).c_str());
  if (!active_model.empty()) {
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                       "Model: %s", active_model.c_str());
  }
  const std::string idle_label
      = active_idle_consolidation_seconds > 0
            ? std::to_string(active_idle_consolidation_seconds) + "s"
            : "auto";
  ImGui::TextColored(
      ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
      "Idle consolidation: %s", idle_label.c_str());

  const bool dirty = !NearlyEqual(active_focus, draft_focus)
                     || !NearlyEqual(active_sensitivity, draft_sensitivity)
                     || !NearlyEqual(active_stability, draft_stability)
                     || active_idle_consolidation_seconds
                            != draft_idle_consolidation_seconds
                     || !NearlyEqual(active_webcam_video_fps,
                                     draft_webcam_video_fps)
                     || active_model != draft_model
                     || active_memory_prompt_prefix != draft_memory_prompt_prefix
                     || active_memory_prompt_suffix != draft_memory_prompt_suffix;
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

  int idle_seconds_input = draft_idle_consolidation_seconds;
  if (ImGui::InputInt("Idle consolidation seconds", &idle_seconds_input)) {
    idle_seconds_input = std::max(0, idle_seconds_input);
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_idle_consolidation_seconds = idle_seconds_input;
    draft_idle_consolidation_seconds = idle_seconds_input;
  }
  ImGui::TextWrapped(
      "Use 0 to keep the stability-derived idle timer. Any positive value overrides it.");
  if (ImGui::Button("Reset Idle Timer")) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_idle_consolidation_seconds
        = state_.settings->default_idle_consolidation_seconds;
    draft_idle_consolidation_seconds
        = state_.settings->default_idle_consolidation_seconds;
  }
  if (default_idle_consolidation_seconds == 0) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "Default uses stability.");
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Webcam");
  ImGui::TextWrapped(
      "Direct webcam ingest sends camera frames to Cortext as images. Lower FPS reduces multimodal ingest cost.");
  float webcam_fps_f = static_cast<float>(draft_webcam_video_fps);
  if (ImGui::SliderFloat("Webcam FPS", &webcam_fps_f, 0.1f, 30.0f, "%.1f")) {
    webcam_fps_f = std::clamp(webcam_fps_f, 0.1f, 30.0f);
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_webcam_video_fps
        = static_cast<double>(webcam_fps_f);
    draft_webcam_video_fps = state_.settings->draft_webcam_video_fps;
  }
  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                     "Active webcam FPS: %.1f",
                     active_webcam_video_fps);
  if (ImGui::Button("Reset Webcam FPS")) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_webcam_video_fps
        = state_.settings->default_webcam_video_fps;
    draft_webcam_video_fps = state_.settings->default_webcam_video_fps;
  }
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                     "Default: %.1f FPS",
                     default_webcam_video_fps);

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Model");
  ImGui::TextWrapped(
      "This model name is sent directly to the OpenAI-compatible API for new generations.");
  if (EditSingleLineString("##model", draft_model)) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_model = draft_model;
  }
  if (ImGui::Button("Reset Model")) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_model = state_.settings->default_model;
  }
  if (!default_model.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "Startup default: %s", default_model.c_str());
  }
  if (draft_model.empty()) {
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                       "Model cannot be empty.");
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Memory Prompt Prefix");
  ImGui::TextWrapped(
      "This text is inserted inside the <snapshot> block immediately above "
      "the <memories> XML block. Edit the draft here, then use Apply to make "
      "it active.");

  if (EditMultilineString("##memory_prompt_prefix",
                          draft_memory_prompt_prefix,
                          ImVec2(-1.0f, 140.0f))) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_memory_prompt_prefix = draft_memory_prompt_prefix;
  }

  if (ImGui::Button("Reset Memory Prompt")) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_memory_prompt_prefix
        = state_.settings->default_memory_prompt_prefix;
  }
  if (!default_memory_prompt_prefix.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "Restores the default instruction text.");
  }

  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Memory Prompt Suffix");
  ImGui::TextWrapped(
      "This text is inserted inside the <snapshot> block immediately below "
      "the </memories> XML block. Use it if you want the main instruction "
      "block to appear after the memories.");

  if (EditMultilineString("##memory_prompt_suffix",
                          draft_memory_prompt_suffix,
                          ImVec2(-1.0f, 120.0f))) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_memory_prompt_suffix = draft_memory_prompt_suffix;
  }

  if (ImGui::Button("Reset Memory Suffix")) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->draft_memory_prompt_suffix
        = state_.settings->default_memory_prompt_suffix;
  }
  if (default_memory_prompt_suffix.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "Default is empty.");
  }

  const bool generating = state_.generating && *state_.generating;
  if (generating) {
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                       "Wait for the current generation to finish before applying.");
  }

  const bool can_apply = dirty && !apply_requested && !generating
                         && !draft_model.empty();
  if (!can_apply) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Apply")) {
    std::lock_guard<std::mutex> lock(state_.settings->mu);
    state_.settings->apply_requested = true;
    state_.settings->last_apply_status = "Applying updated settings...";
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
    state_.settings->draft_idle_consolidation_seconds
        = state_.settings->active_idle_consolidation_seconds;
    state_.settings->draft_webcam_video_fps
        = state_.settings->active_webcam_video_fps;
    state_.settings->draft_model = state_.settings->active_model;
    state_.settings->draft_memory_prompt_prefix
        = state_.settings->active_memory_prompt_prefix;
    state_.settings->draft_memory_prompt_suffix
        = state_.settings->active_memory_prompt_suffix;
  }

  ImGui::Separator();
  ImGui::TextWrapped(
      "Changing these knobs rebuilds the Cortext instance against the same chat DB. "
      "New generations use the updated config; active working-memory contents may take "
      "a turn to fully reflect the new settings.");
}

void ChatWindow::RenderMediaModal() {
  if (!state_.db_explorer) {
    return;
  }

  std::optional<MemoryMediaPreview> media;
  bool open_requested = false;
  bool db_audio_playing = false;
  long long db_audio_playing_id = 0;
  {
    std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
    media = state_.db_explorer->loaded_media_modal;
    open_requested = state_.db_explorer->media_modal_open_requested;
    state_.db_explorer->media_modal_open_requested = false;
    db_audio_playing = state_.db_explorer->audio_playing;
    db_audio_playing_id = state_.db_explorer->audio_playing_memory_id;
  }

  if (open_requested) {
    ImGui::OpenPopup("Media preview");
  }

  ImGui::SetNextWindowSize(ImVec2(720.0f, 620.0f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Media preview", nullptr,
                              ImGuiWindowFlags_NoSavedSettings)) {
    return;
  }

  if (!media.has_value()) {
    ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.4f, 1.0f),
                       "No media loaded.");
    if (ImGui::Button("Close")) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
    return;
  }

  const auto& preview = *media;
  ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f),
                     "memory=%lld preview=%lld modality=%s",
                     preview.memory_id,
                     preview.preview_id,
                     preview.modality.c_str());
  ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                     "%s, %zu bytes",
                     preview.mimetype.empty() ? "-" : preview.mimetype.c_str(),
                     preview.byte_count);
  ImGui::Separator();

  if (HasImagePreview(preview)) {
    ImageTexture& cached = webcam_image_textures_[preview.preview_id];
    if (cached.texture_id == 0
        || cached.width != preview.image_width
        || cached.height != preview.image_height) {
      if (cached.texture_id != 0) {
        const GLuint old_texture_id = cached.texture_id;
        glDeleteTextures(1, &old_texture_id);
      }
      GLuint texture_id = 0;
      glGenTextures(1, &texture_id);
      glBindTexture(GL_TEXTURE_2D, texture_id);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexImage2D(GL_TEXTURE_2D,
                   0,
                   preview.image_channels == 4 ? GL_RGBA : GL_RGB,
                   preview.image_width,
                   preview.image_height,
                   0,
                   preview.image_channels == 4 ? GL_RGBA : GL_RGB,
                   GL_UNSIGNED_BYTE,
                   preview.media_bytes.data());
      glBindTexture(GL_TEXTURE_2D, 0);
      cached.texture_id = texture_id;
      cached.width = preview.image_width;
      cached.height = preview.image_height;
    }
    const float max_side = 560.0f;
    const float scale = std::min(
        max_side / static_cast<float>(std::max(1, preview.image_width)),
        max_side / static_cast<float>(std::max(1, preview.image_height)));
    ImGui::Image(static_cast<ImTextureID>(cached.texture_id),
                 ImVec2(static_cast<float>(preview.image_width) * scale,
                        static_cast<float>(preview.image_height) * scale));
  } else if (preview.modality == "audio" && HasAudioPreview(preview)) {
    const bool playing_this = db_audio_playing
                              && db_audio_playing_id == preview.preview_id;
    if (playing_this) {
      if (ImGui::Button("Stop audio")) {
        std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
        state_.db_explorer->audio_stop_requested = true;
      }
    } else if (ImGui::Button("Play audio")) {
      std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
      state_.db_explorer->audio_play_request = preview;
    }
    const double duration_s = static_cast<double>(preview.audio_samples.size())
                              / static_cast<double>(preview.audio_sample_rate);
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                       "%.2fs @ %d Hz",
                       duration_s,
                       preview.audio_sample_rate);
  } else {
    ImGui::TextWrapped("%s", preview.preview.c_str());
  }

  ImGui::Separator();
  if (ImGui::Button("Close")) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void ChatWindow::RenderDatabaseTab() {
  if (!state_.db_explorer) return;

  std::vector<DatabaseMemoryRow> memories;
  std::vector<DatabaseSignalRow> signals;
  std::vector<DatabaseAssociationRow> associations;
  std::vector<DatabaseEpisodeRow> episodes;
  std::vector<DatabaseEvictionRow> evictions;
  long long total_memories = 0;
  long long total_signals = 0;
  long long total_associations = 0;
  long long total_episodes = 0;
  long long total_evictions = 0;
  uint64_t refreshed_at = 0;
  bool db_audio_playing = false;
  long long db_audio_playing_id = 0;
  bool is_consolidating = state_.idle_consolidating
                          && state_.idle_consolidating->load();
  std::optional<std::string> last_status;
  {
    std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
    memories = state_.db_explorer->memories;
    signals = state_.db_explorer->signals;
    associations = state_.db_explorer->associations;
    episodes = state_.db_explorer->episodes;
    evictions = state_.db_explorer->evictions;
    total_memories = state_.db_explorer->total_memories;
    total_signals = state_.db_explorer->total_signals;
    total_associations = state_.db_explorer->total_associations;
    total_episodes = state_.db_explorer->total_episodes;
    total_evictions = state_.db_explorer->total_evictions;
    refreshed_at = state_.db_explorer->refreshed_at;
    db_audio_playing = state_.db_explorer->audio_playing;
    db_audio_playing_id = state_.db_explorer->audio_playing_memory_id;
    last_status = state_.db_explorer->last_refresh_status;
  }

  const auto filter = static_cast<DatabaseExplorerFilter>(db_memory_kind_filter_);
  std::vector<DatabaseMemoryRow> filtered_memories;
  filtered_memories.reserve(memories.size());
  for (const auto& row : memories) {
    if (MatchesDatabaseExplorerFilter(row, filter)) {
      filtered_memories.push_back(row);
    }
  }
  std::unordered_map<long long, std::vector<DatabaseSignalRow>> signals_by_memory;
  signals_by_memory.reserve(signals.size());
  for (const auto& signal : signals) {
    signals_by_memory[signal.memory_id].push_back(signal);
  }
  for (auto& kv : signals_by_memory) {
    std::sort(kv.second.begin(), kv.second.end(), [](const auto& a, const auto& b) {
      if (a.serial_position != b.serial_position) {
        return a.serial_position < b.serial_position;
      }
      return a.signal_id < b.signal_id;
    });
  }

  const bool show_associations = filter == DatabaseExplorerFilter::All
                                 || filter == DatabaseExplorerFilter::Associations;
  const bool show_memories = filter == DatabaseExplorerFilter::All
                             || filter == DatabaseExplorerFilter::Labels
                             || filter == DatabaseExplorerFilter::LongTerm
                             || filter == DatabaseExplorerFilter::Working;
  const bool show_temporal = filter == DatabaseExplorerFilter::All
                             || filter == DatabaseExplorerFilter::Temporal;
  const bool show_supporting_sections = filter == DatabaseExplorerFilter::All;

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "DATABASE EXPLORER");
  ImGui::SameLine();
  if (ImGui::Button("Refresh")) {
    std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
    state_.db_explorer->refresh_requested = true;
    state_.db_explorer->last_refresh_status = "Refresh requested...";
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(is_consolidating);
  if (ImGui::Button("Consolidate")) {
    std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
    state_.db_explorer->consolidate_requested = true;
    state_.db_explorer->last_refresh_status = "Consolidation requested...";
  }
  ImGui::EndDisabled();
  if (is_consolidating) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f),
                       "consolidating");
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
                     "Totals: memories=%lld signals=%lld associations=%lld episodes=%lld evictions=%lld",
                     total_memories,
                     total_signals,
                     total_associations,
                     total_episodes,
                     total_evictions);
  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Last refresh: %s",
                     NullOrValue(refreshed_at).c_str());
  if (last_status.has_value()) {
    ImGui::TextWrapped("%s", last_status->c_str());
  }
  ImGui::SetNextItemWidth(220.0f);
  ImGui::Combo("View", &db_memory_kind_filter_, kDatabaseExplorerFilterLabels,
               IM_ARRAYSIZE(kDatabaseExplorerFilterLabels));

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

  if (show_associations
      && ImGui::CollapsingHeader("Relationships", ImGuiTreeNodeFlags_DefaultOpen)) {
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

  if (show_memories
      && ImGui::CollapsingHeader("Memories", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BeginChild("DbMemories", ImVec2(0, 420), true);
    if (filtered_memories.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no memories)");
    } else {
      for (const auto& row : filtered_memories) {
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
          const auto signal_it = signals_by_memory.find(row.memory_id);
          if (signal_it != signals_by_memory.end()) {
            ImGui::Spacing();
            int audio_count = 0;
            int image_count = 0;
            int text_count = 0;
            for (const auto& signal : signal_it->second) {
              if (signal.modality == "audio") {
                ++audio_count;
              } else if (signal.modality == "image") {
                ++image_count;
              } else if (signal.modality == "text") {
                ++text_count;
              }
            }

            ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
                               "Memory media");
            ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                               "audio=%d image=%d text=%d",
                               audio_count,
                               image_count,
                               text_count);
            if (audio_count > 0) {
              const bool playing_this = db_audio_playing
                                        && db_audio_playing_id
                                               == -row.memory_id;
              if (playing_this) {
                if (ImGui::SmallButton(
                        ("Stop memory audio##db_memory_audio_"
                         + std::to_string(row.memory_id))
                            .c_str())) {
                  std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
                  state_.db_explorer->audio_stop_requested = true;
                }
              } else if (ImGui::SmallButton(
                             ("Open memory audio##db_memory_audio_"
                              + std::to_string(row.memory_id))
                                 .c_str())) {
                RequestMediaLoad(state_.db_explorer, "memory_audio",
                                 row.memory_id, 0);
              }
              ImGui::SameLine();
              ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                                 "%d audio signal(s), loaded on demand",
                                 audio_count);
            }
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f),
                               "Signal payloads");
            for (const auto& signal : signal_it->second) {
              const auto& preview = signal.preview;
              ImGui::PushID(static_cast<int>(signal.signal_id));
              ImGui::Separator();
              ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f),
                                 "signal #%lld %s",
                                 signal.signal_id,
                                 signal.modality.c_str());
              ImGui::SameLine();
              ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                                 "%s, %zu bytes",
                                 signal.mime.empty() ? "-" : signal.mime.c_str(),
                                 preview.byte_count);

              if (signal.modality == "text") {
                ImGui::PushTextWrapPos();
                ImGui::TextWrapped("%s", preview.preview.c_str());
                ImGui::PopTextWrapPos();
              } else if (signal.modality == "image") {
                const std::string show_label
                    = "Open image##db_image_" + std::to_string(signal.signal_id);
                if (ImGui::SmallButton(show_label.c_str())) {
                  RequestMediaLoad(state_.db_explorer, "signal_image",
                                   signal.memory_id, signal.signal_id);
                }
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                                   "loaded on demand");
              } else if (signal.modality == "audio") {
                const bool playing_this = db_audio_playing
                                          && db_audio_playing_id == preview.preview_id;
                if (playing_this) {
                  if (ImGui::SmallButton("Stop")) {
                    std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
                      state_.db_explorer->audio_stop_requested = true;
                    }
                  } else if (ImGui::SmallButton("Open audio")) {
                    RequestMediaLoad(state_.db_explorer, "signal_audio",
                                     signal.memory_id, signal.signal_id);
                  }
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                                   "loaded on demand");
              } else {
                ImGui::TextWrapped("%s", preview.preview.c_str());
              }
              ImGui::PopID();
            }
          }
        }
      }
    }
    ImGui::EndChild();
  }

  if (show_supporting_sections && ImGui::CollapsingHeader("Signals")) {
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

  if (show_supporting_sections && ImGui::CollapsingHeader("Episodes")) {
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

  if (ImGui::CollapsingHeader("Evictions")) {
    ImGui::BeginChild("DbEvictions", ImVec2(0, 180), true);
    if (evictions.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no eviction records)");
    } else {
      for (const auto& row : evictions) {
        if (ImGui::CollapsingHeader(FormatEvictionTitle(row).c_str())) {
          ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                             "reason=%s evicted_at=%s",
                             row.eviction_reason.c_str(),
                             NullOrValue(row.evicted_at).c_str());
          ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f),
                             "kind=%s source=%s embedding=%lld modality=%s",
                             row.kind.c_str(),
                             row.source_id.c_str(),
                             row.embedding_id,
                             row.modality.empty() ? "-" : row.modality.c_str());
          ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f),
                             "strength=%s use_freq=%s contextual_gain=%s signals=%lld",
                             FormatDouble(row.strength).c_str(),
                             FormatDouble(row.use_frequency).c_str(),
                             FormatDouble(row.contextual_gain).c_str(),
                             row.n_signals);
          ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                             "retrieved=%lld used=%lld start=%s end=%s last_access=%s created=%s",
                             row.retrieved_count,
                             row.used_count,
                             NullOrValue(row.start_ts).c_str(),
                             NullOrValue(row.end_ts).c_str(),
                             NullOrValue(row.last_access).c_str(),
                             NullOrValue(row.created_at).c_str());
          if (!row.label.empty()) {
            ImGui::TextWrapped("%s", row.label.c_str());
          }
        }
      }
    }
    ImGui::EndChild();
  }
}

void ChatWindow::RenderGraphTab() {
  if (!state_.db_explorer) return;

  std::vector<DatabaseAssociationRow> associations;
  {
    std::lock_guard<std::mutex> lock(state_.db_explorer->mu);
    associations = state_.db_explorer->associations;
  }

  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "RELATIONSHIP GRAPH");
  ImGui::SameLine();
  if (ImGui::Button("Reset Layout")) {
    graph_pan_offset_ = ImVec2(80.0f, 80.0f);
    graph_node_positions_.clear();
    graph_layout_dirty_ = true;
    graph_dragging_node_id_ = 0;
  }
  ImGui::TextWrapped(
      "Drag nodes with the left mouse button. Drag empty space with the right mouse button to pan.");
  ImGui::Separator();

  if (associations.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no relationships to graph)");
    return;
  }

  struct GraphNode {
    long long memory_id = 0;
    std::string kind;
    std::string label;
    std::string source_id;
  };

  std::vector<GraphNode> nodes;
  nodes.reserve(associations.size() * 2);
  for (const auto& row : associations) {
    const auto append_node = [&nodes](long long memory_id,
                                      const std::string& kind,
                                      const std::string& label,
                                      const std::string& source_id) {
      auto it = std::find_if(nodes.begin(), nodes.end(),
                             [memory_id](const GraphNode& node) {
                               return node.memory_id == memory_id;
                             });
      if (it == nodes.end()) {
        nodes.push_back(GraphNode{memory_id, kind, label, source_id});
      }
    };
    append_node(row.source_memory_id, row.source_kind, row.source_label, row.source_source_id);
    append_node(row.target_memory_id, row.target_kind, row.target_label, row.target_source_id);
  }

  std::sort(nodes.begin(), nodes.end(), [](const GraphNode& a, const GraphNode& b) {
    return a.memory_id < b.memory_id;
  });

  std::size_t layout_signature = nodes.size();
  for (const auto& node : nodes) {
    layout_signature ^= std::hash<long long>{}(node.memory_id)
                        + 0x9e3779b97f4a7c15ULL
                        + (layout_signature << 6)
                        + (layout_signature >> 2);
  }
  for (const auto& row : associations) {
    std::size_t edge_hash = std::hash<long long>{}(row.source_memory_id);
    edge_hash ^= std::hash<long long>{}(row.target_memory_id)
                 + 0x9e3779b97f4a7c15ULL
                 + (edge_hash << 6)
                 + (edge_hash >> 2);
    edge_hash ^= std::hash<std::string>{}(row.edge_type)
                 + 0x9e3779b97f4a7c15ULL
                 + (edge_hash << 6)
                 + (edge_hash >> 2);
    layout_signature ^= edge_hash + 0x9e3779b97f4a7c15ULL
                        + (layout_signature << 6)
                        + (layout_signature >> 2);
  }
  if (layout_signature != graph_layout_signature_) {
    graph_layout_signature_ = layout_signature;
    graph_layout_dirty_ = true;
  }

  const ImVec2 canvas_size = ImGui::GetContentRegionAvail();
  const ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
  const float canvas_width = std::max(canvas_size.x, 300.0f);
  const float canvas_height = std::max(canvas_size.y, 260.0f);
  const ImVec2 actual_canvas_size(canvas_width, canvas_height);

  ImGui::InvisibleButton("GraphCanvas", actual_canvas_size,
                         ImGuiButtonFlags_MouseButtonRight
                             | ImGuiButtonFlags_MouseButtonMiddle);
  const bool canvas_hovered = ImGui::IsItemHovered();
  if (canvas_hovered && (ImGui::IsMouseDragging(ImGuiMouseButton_Right)
                         || ImGui::IsMouseDragging(ImGuiMouseButton_Middle))) {
    graph_pan_offset_.x += ImGui::GetIO().MouseDelta.x;
    graph_pan_offset_.y += ImGui::GetIO().MouseDelta.y;
  }

  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  const ImVec2 canvas_max(canvas_pos.x + actual_canvas_size.x,
                          canvas_pos.y + actual_canvas_size.y);
  draw_list->AddRectFilled(canvas_pos, canvas_max, IM_COL32(18, 18, 22, 255), 6.0f);
  draw_list->AddRect(canvas_pos, canvas_max, IM_COL32(70, 70, 78, 255), 6.0f);
  draw_list->PushClipRect(canvas_pos, canvas_max, true);

  const float node_width = 180.0f;
  const float node_height = 56.0f;
  const float horizontal_spacing = 56.0f;
  const float vertical_spacing = 34.0f;
  const float layout_width = std::max(actual_canvas_size.x - node_width - 40.0f,
                                      3.0f * (node_width + horizontal_spacing));
  const ImVec2 center(layout_width * 0.5f, 0.0f);

  std::unordered_set<long long> live_ids;
  live_ids.reserve(nodes.size());
  for (const auto& node : nodes) {
    live_ids.insert(node.memory_id);
  }
  for (auto it = graph_node_positions_.begin(); it != graph_node_positions_.end();) {
    if (!live_ids.count(it->first)) {
      it = graph_node_positions_.erase(it);
    } else {
      ++it;
    }
  }

  for (std::size_t i = 0; i < nodes.size(); ++i) {
    auto pos_it = graph_node_positions_.find(nodes[i].memory_id);
    if (pos_it == graph_node_positions_.end()) {
      graph_node_positions_[nodes[i].memory_id] = ImVec2(0.0f, 0.0f);
    }
  }

  if (graph_layout_dirty_) {
    std::unordered_map<long long, int> degrees;
    degrees.reserve(nodes.size());
    for (const auto& row : associations) {
      degrees[row.source_memory_id] += 1;
      degrees[row.target_memory_id] += 1;
    }

    std::vector<GraphNode> layout_nodes = nodes;
    std::stable_sort(layout_nodes.begin(), layout_nodes.end(),
                     [&degrees](const GraphNode& a, const GraphNode& b) {
                       const int degree_a = degrees[a.memory_id];
                       const int degree_b = degrees[b.memory_id];
                       if (degree_a != degree_b) return degree_a > degree_b;
                       if (a.kind != b.kind) return a.kind < b.kind;
                       return a.memory_id < b.memory_id;
                     });

    const int columns = std::max(
        3,
        static_cast<int>(std::ceil(std::sqrt(static_cast<float>(layout_nodes.size())))));
    const float x_step = node_width + horizontal_spacing;
    const float y_step = node_height + vertical_spacing;
    const float total_width = (static_cast<float>(columns - 1) * x_step);
    const float x_origin = std::max(20.0f, center.x - total_width * 0.5f);

    for (std::size_t i = 0; i < layout_nodes.size(); ++i) {
      const int row = static_cast<int>(i / static_cast<std::size_t>(columns));
      const int col = static_cast<int>(i % static_cast<std::size_t>(columns));
      const float stagger = (row % 2 == 0) ? 0.0f : x_step * 0.25f;
      graph_node_positions_[layout_nodes[i].memory_id]
          = ImVec2(x_origin + static_cast<float>(col) * x_step + stagger,
                   24.0f + static_cast<float>(row) * y_step);
    }
    graph_layout_dirty_ = false;
  }

  for (const auto& row : associations) {
    const ImVec2 source_pos = graph_node_positions_[row.source_memory_id];
    const ImVec2 target_pos = graph_node_positions_[row.target_memory_id];
    const ImVec2 p1(canvas_pos.x + graph_pan_offset_.x + source_pos.x + node_width * 0.5f,
                    canvas_pos.y + graph_pan_offset_.y + source_pos.y + node_height * 0.5f);
    const ImVec2 p2(canvas_pos.x + graph_pan_offset_.x + target_pos.x + node_width * 0.5f,
                    canvas_pos.y + graph_pan_offset_.y + target_pos.y + node_height * 0.5f);
    const ImVec2 cp1(p1.x + 50.0f, p1.y);
    const ImVec2 cp2(p2.x - 50.0f, p2.y);
    draw_list->AddBezierCubic(p1, cp1, cp2, p2, EdgeColor(row.edge_type), 2.0f);

    const ImVec2 mid((p1.x + p2.x) * 0.5f, (p1.y + p2.y) * 0.5f);
    draw_list->AddText(ImVec2(mid.x + 6.0f, mid.y - 10.0f),
                       IM_COL32(220, 220, 220, 180),
                       row.edge_type.c_str());
  }

  for (const auto& node : nodes) {
    ImVec2& local_pos = graph_node_positions_[node.memory_id];
    const ImVec2 node_min(canvas_pos.x + graph_pan_offset_.x + local_pos.x,
                          canvas_pos.y + graph_pan_offset_.y + local_pos.y);
    const ImVec2 node_max(node_min.x + node_width, node_min.y + node_height);

    ImGui::SetCursorScreenPos(node_min);
    ImGui::InvisibleButton(("graph-node-" + std::to_string(node.memory_id)).c_str(),
                           ImVec2(node_width, node_height));
    const bool hovered = ImGui::IsItemHovered();
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      graph_dragging_node_id_ = node.memory_id;
    }
    if (graph_dragging_node_id_ == node.memory_id && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      local_pos.x += ImGui::GetIO().MouseDelta.x;
      local_pos.y += ImGui::GetIO().MouseDelta.y;
    }
    if (graph_dragging_node_id_ == node.memory_id && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      graph_dragging_node_id_ = 0;
    }

    const ImU32 fill = NodeFillColor(node.kind);
    const bool dragging = graph_dragging_node_id_ == node.memory_id;
    const ImU32 border = (hovered || dragging) ? IM_COL32(255, 255, 255, 220)
                                               : IM_COL32(40, 40, 40, 255);
    draw_list->AddRectFilled(node_min, node_max, fill, 8.0f);
    draw_list->AddRect(node_min, node_max, border, 8.0f, 0, hovered ? 2.0f : 1.0f);
    draw_list->AddText(ImVec2(node_min.x + 10.0f, node_min.y + 8.0f),
                       IM_COL32(255, 255, 255, 255),
                       GraphNodeTitle(node.kind, node.label, node.source_id, node.memory_id).c_str());
    draw_list->AddText(ImVec2(node_min.x + 10.0f, node_min.y + 30.0f),
                       IM_COL32(230, 230, 230, 200),
                       (node.kind + " #" + std::to_string(node.memory_id)).c_str());

    if (hovered) {
      ImGui::BeginTooltip();
      ImGui::Text("#%lld", node.memory_id);
      if (!node.kind.empty()) {
        ImGui::Text("kind: %s", node.kind.c_str());
      }
      if (!node.source_id.empty()) {
        ImGui::Text("source: %s", node.source_id.c_str());
      }
      if (!node.label.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", node.label.c_str());
      }
      ImGui::EndTooltip();
    }
  }

  draw_list->PopClipRect();
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
  if (ImGui::CollapsingHeader("Provider Messages", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BeginChild("Messages", ImVec2(0, 300), true);
    const auto& messages = state_.context->provider_messages;
    for (size_t i = 0; i < messages.size(); ++i) {
      const auto& msg = messages[i];
      ImGui::TextColored(RoleColor(msg.role), "[%zu] %s:", i, RolePrefix(msg.role).c_str());
      std::string content = Truncate(msg.content, 500);
      ImGui::TextWrapped("%s", content.c_str());
      ImGui::Separator();
    }
    if (messages.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no provider messages captured)");
    }
    ImGui::EndChild();
  }
}

void ChatWindow::RenderLogsTab() {
  ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "OpenTelemetry");
  ImGui::Separator();

  std::vector<LogEntry> logs;
  std::vector<TraceEntry> traces;
  if (state_.otel) {
    std::lock_guard<std::mutex> lock(state_.otel->mu);
    logs.assign(state_.otel->logs.begin(), state_.otel->logs.end());
    traces.assign(state_.otel->traces.begin(), state_.otel->traces.end());
  }

  if (logs.empty() && traces.empty()) {
    if (state_.otlp_enabled) {
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "OTLP gRPC enabled");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Traces, logs, and metrics are exported");
      ImGui::Text("");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Telemetry active for:");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Uptrace (traces + logs + metrics)");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Status bar metrics");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - In-memory trace/log viewer");
    } else {
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "OTLP gRPC not configured");
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Set UPTRACE_DSN to enable Uptrace export");
      ImGui::Text("");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "In-memory telemetry active for:");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Recent traces");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Recent logs");
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  - Status bar metrics");
    }
  } else {
    ImGui::Text("traces=%zu logs=%zu", traces.size(), logs.size());
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Traces", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::BeginChild("OTelTraces", ImVec2(0, 240), true);
      for (auto it = traces.rbegin(); it != traces.rend(); ++it) {
        const auto& trace = *it;
        std::ostringstream header;
        header << trace.name << " | " << std::fixed << std::setprecision(2)
               << trace.duration_ms << "ms";
        if (!trace.status.empty()) {
          header << " | " << trace.status;
        }
        ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.3f, 1.0f), "%s",
                           header.str().c_str());
        for (const auto& [k, v] : trace.attributes) {
          ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f), "  %s:", k.c_str());
          ImGui::SameLine();
          ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "%s",
                             Truncate(v, 96).c_str());
        }
        ImGui::Separator();
      }
      ImGui::EndChild();
    }

    if (ImGui::CollapsingHeader("Logs", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::BeginChild("OTelLogs", ImVec2(0, 0), true);
      for (auto it = logs.rbegin(); it != logs.rend(); ++it) {
        const auto& log = *it;
        std::string header = log.body;
        if (!log.severity.empty()) {
          header = "[" + log.severity + "] " + header;
        }
        header = Truncate(header, 100);
        ImGui::TextColored(SeverityColor(log.severity), "%s", header.c_str());

        for (const auto& [k, v] : log.attributes) {
          ImGui::TextColored(ImVec4(0.3f, 1.0f, 1.0f, 1.0f), "  %s:", k.c_str());
          ImGui::SameLine();
          ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "%s",
                             Truncate(v, 80).c_str());
        }
        ImGui::Separator();
      }
      ImGui::EndChild();
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

  bool generating = state_.generating && *state_.generating;
  const std::string before_input = input_buffer_;

  ImGui::PushItemWidth(-1);
  ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue;

  if (refocus_input_next_frame_ || ImGui::IsWindowAppearing()) {
    ImGui::SetKeyboardFocusHere();
    refocus_input_next_frame_ = false;
  }

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
    refocus_input_next_frame_ = true;
  }

  if (state_.has_input_draft) {
    state_.has_input_draft->store(input_buffer_[0] != '\0' || !pending_message_.empty());
  }

  ImGui::PopItemWidth();
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
