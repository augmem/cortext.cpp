#include "../src/signal_filter.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

TEST_CASE ("SignalFilter audio adapts to quiet ambient and accepts onset",
           "[signal_filter][audio]")
{
  cortext::internal::SignalFilterConfig cfg;
  cfg.audio_enabled = true;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cortext::internal::SignalFilter filter (cfg);

  std::vector<float> quiet (4000, 0.0f); // 250 ms at 16 kHz.
  auto first = filter.EvaluateAudio (quiet.data (), quiet.size (), 16000);
  REQUIRE (first.evaluated);
  REQUIRE (first.accepted);
  REQUIRE (first.reason == "first_item");

  auto steady = filter.EvaluateAudio (quiet.data (), quiet.size (), 16000);
  REQUIRE (steady.evaluated);
  REQUIRE_FALSE (steady.accepted);
  REQUIRE (steady.reason == "reject");

  std::vector<float> onset (4000, 0.35f);
  auto change = filter.EvaluateAudio (onset.data (), onset.size (), 16000);
  REQUIRE (change.evaluated);
  REQUIRE (change.accepted);
  REQUIRE (change.reason == "adaptive_delta");
}

TEST_CASE ("SignalFilter can disable or pass through unregistered modalities",
           "[signal_filter]")
{
  cortext::internal::SignalFilterConfig cfg;
  cfg.audio_enabled = false;
  cfg.text_enabled = true;
  cortext::internal::SignalFilter filter (cfg);

  std::vector<float> pcm (4000, 0.0f);
  auto audio = filter.EvaluateAudio (pcm.data (), pcm.size (), 16000);
  REQUIRE_FALSE (audio.evaluated);
  REQUIRE (audio.accepted);
  REQUIRE (audio.reason == "disabled");

  auto text = filter.EvaluateText ("hello");
  REQUIRE (text.evaluated);
  REQUIRE (text.accepted);
  REQUIRE (text.reason == "no_registered_filter");
}

TEST_CASE ("SignalFilter image adapts to repeated frames and accepts change",
           "[signal_filter][image]")
{
  cortext::internal::SignalFilterConfig cfg;
  cfg.image_enabled = true;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cortext::internal::SignalFilter filter (cfg);

  std::vector<std::uint8_t> dark (64 * 64 * 3, 0);
  auto first = filter.EvaluateImage (dark.data (), 64, 64, 3);
  REQUIRE (first.evaluated);
  REQUIRE (first.accepted);
  REQUIRE (first.reason == "first_item");

  auto steady = filter.EvaluateImage (dark.data (), 64, 64, 3);
  REQUIRE (steady.evaluated);
  REQUIRE_FALSE (steady.accepted);

  std::vector<std::uint8_t> bright (64 * 64 * 3, 255);
  auto change = filter.EvaluateImage (bright.data (), 64, 64, 3);
  REQUIRE (change.evaluated);
  REQUIRE (change.accepted);
  REQUIRE (change.reason == "adaptive_delta");
}
