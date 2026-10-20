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
  auto first = filter.EvaluateAudio ("mic", quiet.data (), quiet.size (), 16000);
  REQUIRE (first.evaluated);
  REQUIRE (first.accepted);
  REQUIRE (first.reason == "first_item");

  auto steady = filter.EvaluateAudio ("mic", quiet.data (), quiet.size (), 16000);
  REQUIRE (steady.evaluated);
  REQUIRE_FALSE (steady.accepted);
  REQUIRE (steady.reason == "reject");

  std::vector<float> onset (4000, 0.35f);
  auto change = filter.EvaluateAudio ("mic", onset.data (), onset.size (), 16000);
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
  auto audio = filter.EvaluateAudio ("mic", pcm.data (), pcm.size (), 16000);
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
  auto first = filter.EvaluateImage ("cam", dark.data (), 64, 64, 3);
  REQUIRE (first.evaluated);
  REQUIRE (first.accepted);
  REQUIRE (first.reason == "first_item");

  auto steady = filter.EvaluateImage ("cam", dark.data (), 64, 64, 3);
  REQUIRE (steady.evaluated);
  REQUIRE_FALSE (steady.accepted);

  std::vector<std::uint8_t> bright (64 * 64 * 3, 255);
  auto change = filter.EvaluateImage ("cam", bright.data (), 64, 64, 3);
  REQUIRE (change.evaluated);
  REQUIRE (change.accepted);
  REQUIRE (change.reason == "adaptive_delta");
}

TEST_CASE ("SignalFilter compares items only within one source stream",
           "[signal_filter][image]")
{
  cortext::internal::SignalFilterConfig cfg;
  cfg.image_enabled = true;
  cortext::internal::SignalFilter filter (cfg);

  // A camera stream keeps adapting: a repeated frame is rejected.
  std::vector<std::uint8_t> dark (64 * 64 * 3, 0);
  REQUIRE (filter.EvaluateImage ("cam", dark.data (), 64, 64, 3).accepted);
  REQUIRE_FALSE (
      filter.EvaluateImage ("cam", dark.data (), 64, 64, 3).accepted);

  // Still photos attached under their own source ids are separate streams,
  // even when interleaved with the camera and identical in size.
  for (int i = 0; i < 40; ++i)
    {
      std::vector<std::uint8_t> photo (64 * 64 * 3,
                                       static_cast<std::uint8_t> (i * 5));
      const auto decision = filter.EvaluateImage (
          "chat/turn" + std::to_string (i) + "/img0", photo.data (), 64, 64,
          3);
      REQUIRE (decision.accepted);
      REQUIRE (decision.reason == "first_item");
    }

  // 40 photo streams evicted the camera (bounded state): it restarts.
  const auto restarted = filter.EvaluateImage ("cam", dark.data (), 64, 64, 3);
  REQUIRE (restarted.accepted);
  REQUIRE (restarted.reason == "first_item");

  // Audio is keyed the same way: a clip under its own source id starts a
  // new stream even right after a steady microphone feed.
  std::vector<float> hum (4000, 0.01f);
  cortext::internal::SignalFilterConfig audio_cfg;
  cortext::internal::SignalFilter audio_filter (audio_cfg);
  REQUIRE (audio_filter.EvaluateAudio ("mic", hum.data (), hum.size (), 16000)
               .accepted);
  REQUIRE_FALSE (
      audio_filter.EvaluateAudio ("mic", hum.data (), hum.size (), 16000)
          .accepted);
  const auto clip = audio_filter.EvaluateAudio ("chat/turn1/clip0", hum.data (),
                                                hum.size (), 16000);
  REQUIRE (clip.accepted);
  REQUIRE (clip.reason == "first_item");
}
