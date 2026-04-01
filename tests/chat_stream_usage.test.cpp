#include <catch2/catch_test_macros.hpp>

#include "../examples/chat/stream_usage.hpp"

#include <nlohmann/json.hpp>

TEST_CASE("Streaming usage parser reads the final usage event",
          "[chat][metrics]") {
  const auto json = nlohmann::json::parse(R"json(
    {
      "choices": [
        {
          "delta": {},
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 58,
        "completion_tokens": 17,
        "total_tokens": 75
      }
    }
  )json");

  const auto usage = chat::ParseStreamingUsageJson(json);
  REQUIRE(usage.has_value());
  REQUIRE(usage->prompt_tokens == 58);
  REQUIRE(usage->completion_tokens == 17);
  REQUIRE(usage->total_tokens == 75);
  REQUIRE(chat::HasStreamingFinishReason(json));
  REQUIRE(chat::ExtractStreamingDeltaText(json).empty());
}

TEST_CASE("Streaming delta chunks expose text before usage arrives",
          "[chat][metrics]") {
  const auto json = nlohmann::json::parse(R"json(
    {
      "choices": [
        {
          "delta": {
            "content": "Hello there"
          },
          "finish_reason": null
        }
      ]
    }
  )json");

  REQUIRE_FALSE(chat::ParseStreamingUsageJson(json).has_value());
  REQUIRE(chat::ExtractStreamingDeltaText(json) == "Hello there");
  REQUIRE_FALSE(chat::HasStreamingFinishReason(json));
}
