#pragma once

#include <cstdint>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>
namespace chat {

struct PromptCostSnapshot {
  std::int64_t cortext_prompt_tokens = 0;
  std::int64_t rag_prompt_tokens = 0;
  std::int64_t full_history_prompt_tokens = 0;
  std::size_t cortext_prompt_chars = 0;
  std::size_t rag_prompt_chars = 0;
  std::size_t full_history_prompt_chars = 0;
  std::size_t rag_memory_count = 0;
  std::size_t working_memory_message_count = 0;
  std::size_t full_history_message_count = 0;
};

struct ProviderMessageSnapshot {
  std::string role;
  std::string content;
};

struct LastContext {
  mutable std::mutex mu;
  std::string system_prompt;             // injected_system
  PromptCostSnapshot prompt_costs;
  std::vector<ProviderMessageSnapshot> provider_messages;
  bool has_data = false;
  float scroll_position = 0.0f;  // scroll state (0.0 = top, 1.0 = bottom)
};

}  // namespace chat
