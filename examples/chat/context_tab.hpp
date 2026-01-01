#pragma once

#include <mutex>
#include <string>
namespace chat {

struct LastContext {
  mutable std::mutex mu;
  std::string system_prompt;             // injected_system
  bool has_data = false;
  float scroll_position = 0.0f;  // scroll state (0.0 = top, 1.0 = bottom)
};

}  // namespace chat
