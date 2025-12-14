#include "context_tab.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <sstream>

namespace chat {

namespace {

ftxui::Element RenderSystemPrompt(const std::string& prompt) {
  ftxui::Elements lines;
  lines.push_back(ftxui::text("SYSTEM PROMPT") | ftxui::bold |
                  ftxui::color(ftxui::Color::Yellow));
  lines.push_back(ftxui::separator());

  if (prompt.empty()) {
    lines.push_back(ftxui::text("(none)") | ftxui::color(ftxui::Color::GrayDark));
  } else {
    // Split prompt into lines for better display
    std::istringstream iss(prompt);
    std::string line;
    while (std::getline(iss, line)) {
      lines.push_back(ftxui::text(line) | ftxui::color(ftxui::Color::White));
    }
  }

  return ftxui::vbox(std::move(lines));
}

ftxui::Element RenderMessages(const std::vector<ContextMessage>& messages) {
  ftxui::Elements lines;
  lines.push_back(ftxui::text("MESSAGES (" + std::to_string(messages.size()) + ")") |
                  ftxui::bold | ftxui::color(ftxui::Color::Cyan));
  lines.push_back(ftxui::separator());

  if (messages.empty()) {
    lines.push_back(ftxui::text("(no messages)") | ftxui::color(ftxui::Color::GrayDark));
  } else {
    int idx = 0;
    for (const auto& msg : messages) {
      // Role header with color coding
      auto role_color = ftxui::Color::White;
      if (msg.role == "system") {
        role_color = ftxui::Color::Yellow;
      } else if (msg.role == "user") {
        role_color = ftxui::Color::Cyan;
      } else if (msg.role == "assistant") {
        role_color = ftxui::Color::Green;
      }

      lines.push_back(ftxui::hbox({
          ftxui::text("[" + std::to_string(idx++) + "] ") |
              ftxui::color(ftxui::Color::GrayDark),
          ftxui::text(msg.role) | ftxui::bold | ftxui::color(role_color),
      }));

      // Content (truncate very long messages in display, show first 500 chars)
      std::string content = msg.content;
      bool truncated = false;
      if (content.size() > 500) {
        content = content.substr(0, 500);
        truncated = true;
      }

      // Split content into lines
      std::istringstream iss(content);
      std::string line;
      while (std::getline(iss, line)) {
        lines.push_back(ftxui::text("  " + line) | ftxui::color(ftxui::Color::White));
      }
      if (truncated) {
        lines.push_back(ftxui::text("  ... (truncated)") |
                        ftxui::color(ftxui::Color::GrayDark));
      }

      lines.push_back(ftxui::text(""));
    }
  }

  return ftxui::vbox(std::move(lines));
}

ftxui::Element RenderRawJson(const std::string& json) {
  ftxui::Elements lines;
  lines.push_back(ftxui::text("RAW JSON REQUEST") | ftxui::bold |
                  ftxui::color(ftxui::Color::Magenta));
  lines.push_back(ftxui::separator());

  if (json.empty()) {
    lines.push_back(ftxui::text("(none)") | ftxui::color(ftxui::Color::GrayDark));
  } else {
    std::istringstream iss(json);
    std::string line;
    while (std::getline(iss, line)) {
      lines.push_back(ftxui::text(line) | ftxui::color(ftxui::Color::White));
    }
  }

  return ftxui::vbox(std::move(lines));
}

} // namespace

ftxui::Component MakeContextTab(std::shared_ptr<LastContext> context) {
  return ftxui::Renderer([context] {
    std::string system_prompt;
    std::vector<ContextMessage> messages;
    std::string raw_json;
    bool has_data = false;
    float scroll_pos = 0.0f;

    {
      std::lock_guard<std::mutex> lock(context->mu);
      system_prompt = context->system_prompt;
      messages = context->messages;
      raw_json = context->raw_json;
      has_data = context->has_data;
      scroll_pos = context->scroll_position;
    }

    ftxui::Elements sections;
    sections.push_back(ftxui::text("CONTEXT SENT TO MODEL") | ftxui::bold |
                       ftxui::color(ftxui::Color::White));
    sections.push_back(ftxui::separator());

    if (!has_data) {
      sections.push_back(ftxui::text("(send a message to see context)") |
                         ftxui::color(ftxui::Color::GrayDark));
    } else {
      sections.push_back(RenderSystemPrompt(system_prompt));
      sections.push_back(ftxui::text(""));
      sections.push_back(ftxui::separator());
      sections.push_back(RenderMessages(messages));
      sections.push_back(ftxui::text(""));
      sections.push_back(ftxui::separator());
      sections.push_back(RenderRawJson(raw_json));
    }

    return ftxui::vbox(std::move(sections))
           | ftxui::focusPositionRelative(0, scroll_pos)
           | ftxui::vscroll_indicator
           | ftxui::yframe;
  });
}

} // namespace chat
