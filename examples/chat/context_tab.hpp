#pragma once

#include <mutex>
#include <string>
#include <vector>

namespace chat {

struct ContextMessage {
  std::string role;
  std::string content;
};

struct LastContext {
  mutable std::mutex mu;
  std::string system_prompt;             // injected_system
  std::vector<ContextMessage> messages;  // full messages array
  std::string raw_json;                  // serialized request JSON
  bool has_data = false;
  float scroll_position = 0.0f;  // scroll state (0.0 = top, 1.0 = bottom)
};

}  // namespace chat
