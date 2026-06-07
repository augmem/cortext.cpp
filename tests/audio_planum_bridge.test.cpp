#include <cstdint>
#include <string>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

#include "cortext/capi.h"
#include "cortext/cortext.hpp"
#include "planum/contract/perception_event.hpp"

#include "../src/audio/planum_bridge.hpp"

namespace
{

class RecordingBridgeTarget final : public cortext::audio::PlanumBridgeTarget
{
public:
  cortext::Cortext::Context
  ProcessTextAt (const std::string &text, const std::string &source_id,
                 std::uint64_t timestamp) override
  {
    ++process_text_at_calls;
    last_text = text;
    last_source_id = source_id;
    last_timestamp = timestamp;
    return {};
  }

  cortext::Cortext::Context
  ProcessAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
                const std::string &source_id) override
  {
    ++process_audio_calls;
    last_audio_source_id = source_id;
    return {};
  }

  int process_text_at_calls = 0;
  int process_audio_calls = 0;
  std::string last_text;
  std::string last_source_id;
  std::string last_audio_source_id;
  std::uint64_t last_timestamp = 0;
};

planum::contract::PerceptionEvent
MakeEvent (planum::contract::EventKind kind)
{
  planum::contract::PerceptionEvent event;
  event.kind = kind;
  event.emitted_at_ms = 1400;
  event.segment_started_at_ms = 1100;
  event.segment_ended_at_ms = 1300;
  event.stream_id = planum::contract::StreamId{ "session-a" };
  event.segment_id = planum::contract::SegmentId{ "segment-7" };
  event.turn_id = planum::contract::TurnId{ "turn-2" };
  event.transcript.text = "hello from the bridge";
  event.transcript.confidence = 0.98f;
  event.speaker.id = planum::contract::SpeakerId{ "speaker-3" };
  event.speaker.confidence = 0.72f;
  return event;
}

template <typename T>
concept HasProcessPlanumEvent
    = requires (T &value, const planum::contract::PerceptionEvent &event)
{
  value.ProcessPlanumEvent (event);
};

} // namespace

TEST_CASE ("planum bridge routes finalized transcripts through ProcessTextAt",
           "[audio][planum_bridge]")
{
  RecordingBridgeTarget target;
  cortext::audio::PlanumBridge bridge (target);

  const auto result = bridge.Accept (
      MakeEvent (planum::contract::EventKind::final_transcript));

  REQUIRE (result.has_value ());
  CHECK (result->routed_to_text);
  CHECK (result->source_id == "planum/session-a/speaker-3");
  CHECK (result->timestamp == 1300);
  CHECK (target.process_text_at_calls == 1);
  CHECK (target.process_audio_calls == 0);
  CHECK (target.last_text == "hello from the bridge");
  CHECK (target.last_source_id == "planum/session-a/speaker-3");
  CHECK (target.last_timestamp == 1300);
}

TEST_CASE ("planum bridge keeps non-final events in explicit no-write mode",
           "[audio][planum_bridge]")
{
  RecordingBridgeTarget target;
  cortext::audio::PlanumBridge bridge (target);

  const auto partial_result = bridge.Accept (
      MakeEvent (planum::contract::EventKind::partial_transcript));
  const auto endpoint_result = bridge.Accept (
      MakeEvent (planum::contract::EventKind::endpoint_reached));

  auto degraded_event = MakeEvent (planum::contract::EventKind::degraded);
  degraded_event.degraded.active = true;
  degraded_event.degraded.code = 9;
  const auto degraded_result = bridge.Accept (degraded_event);

  auto error_event = MakeEvent (planum::contract::EventKind::error);
  error_event.error.active = true;
  error_event.error.code = 12;
  const auto error_result = bridge.Accept (error_event);

  CHECK_FALSE (partial_result.has_value ());
  CHECK_FALSE (endpoint_result.has_value ());
  CHECK_FALSE (degraded_result.has_value ());
  CHECK_FALSE (error_result.has_value ());
  CHECK (target.process_text_at_calls == 0);
  CHECK (target.process_audio_calls == 0);
}

TEST_CASE ("planum bridge stays private to src and does not widen public APIs",
           "[audio][planum_bridge]")
{
  STATIC_REQUIRE (std::is_class_v<cortext::audio::PlanumBridge>);
  STATIC_REQUIRE (!HasProcessPlanumEvent<cortext::Cortext>);
  STATIC_REQUIRE (sizeof (cortext_config) > 0);
}
