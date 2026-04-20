#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "audio/planum_bridge.hpp"
#include "planum/perception_event.hpp"

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

planum::PerceptionEvent
MakeEvent (planum::EventKind kind)
{
  planum::PerceptionEvent event;
  event.kind = kind;
  event.emitted_at_ms = 2400;
  event.segment_started_at_ms = 2200;
  event.segment_ended_at_ms = 2300;
  event.stream_id = planum::StreamId{ "smoke-stream" };
  event.segment_id = planum::SegmentId{ "segment-1" };
  event.turn_id = planum::TurnId{ "turn-1" };
  event.transcript.text = "smoke final transcript";
  event.transcript.confidence = 0.99f;
  event.speaker.id = planum::SpeakerId{ "speaker-a" };
  event.speaker.confidence = 0.81f;
  return event;
}

int
Fail (const char *message)
{
  std::cerr << "planum_bridge_smoke: " << message << '\n';
  return EXIT_FAILURE;
}

} // namespace

int
main ()
{
  RecordingBridgeTarget target;
  cortext::audio::PlanumBridge bridge (target);

  const auto final_result = bridge.Accept (
      MakeEvent (planum::EventKind::final_transcript));
  const auto partial_result = bridge.Accept (
      MakeEvent (planum::EventKind::partial_transcript));

  if (!final_result.has_value ())
    {
      return Fail ("expected finalized transcript to route");
    }
  if (partial_result.has_value ())
    {
      return Fail ("expected partial transcript to stay no-write");
    }
  if (final_result->source_id != "planum/smoke-stream/speaker-a")
    {
      return Fail ("unexpected source-id derivation");
    }
  if (final_result->timestamp != 2300)
    {
      return Fail ("unexpected timestamp derivation");
    }
  if (target.process_text_at_calls != 1)
    {
      return Fail ("expected one ProcessTextAt call");
    }
  if (target.process_audio_calls != 0)
    {
      return Fail ("expected zero ProcessAudio calls");
    }
  if (target.last_text != "smoke final transcript")
    {
      return Fail ("unexpected transcript payload");
    }

  std::cout << "planum_bridge_smoke: ok\n";
  return EXIT_SUCCESS;
}
