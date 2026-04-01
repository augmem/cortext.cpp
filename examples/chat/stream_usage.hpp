#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace chat {

struct StreamingUsage {
  std::int64_t prompt_tokens = 0;
  std::int64_t completion_tokens = 0;
  std::int64_t total_tokens = 0;
};

inline std::optional<StreamingUsage> ParseStreamingUsageJson(
    const nlohmann::json& json) {
  if (!json.contains("usage") || json["usage"].is_null()) {
    return std::nullopt;
  }

  const auto& usage_json = json["usage"];
  StreamingUsage usage;
  if (usage_json.contains("prompt_tokens") && usage_json["prompt_tokens"].is_number_integer()) {
    usage.prompt_tokens = usage_json["prompt_tokens"].get<std::int64_t>();
  }
  if (usage_json.contains("completion_tokens")
      && usage_json["completion_tokens"].is_number_integer()) {
    usage.completion_tokens = usage_json["completion_tokens"].get<std::int64_t>();
  }
  if (usage_json.contains("total_tokens") && usage_json["total_tokens"].is_number_integer()) {
    usage.total_tokens = usage_json["total_tokens"].get<std::int64_t>();
  } else {
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
  }
  return usage;
}

inline std::string ExtractStreamingDeltaText(const nlohmann::json& json) {
  if (!json.contains("choices") || !json["choices"].is_array() || json["choices"].empty()) {
    return {};
  }
  const auto& choice = json["choices"][0];
  if (!choice.contains("delta") || !choice["delta"].is_object()
      || !choice["delta"].contains("content")
      || !choice["delta"]["content"].is_string()) {
    return {};
  }
  return choice["delta"]["content"].get<std::string>();
}

inline bool HasStreamingFinishReason(const nlohmann::json& json) {
  if (!json.contains("choices") || !json["choices"].is_array() || json["choices"].empty()) {
    return false;
  }
  const auto& choice = json["choices"][0];
  return choice.contains("finish_reason") && !choice["finish_reason"].is_null();
}

}  // namespace chat
