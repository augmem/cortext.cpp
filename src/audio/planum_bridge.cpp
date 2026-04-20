#include "planum_bridge.hpp"

#include <string_view>
#include <utility>

namespace cortext::audio
{

namespace
{

constexpr std::string_view kSourcePrefix = "planum";
constexpr std::string_view kUnknownStream = "unknown-stream";
constexpr std::string_view kUnknownSpeaker = "unknown-speaker";

std::string_view
SourceComponentOr (std::string_view value, std::string_view fallback) noexcept
{
  return value.empty () ? fallback : value;
}

} // namespace

CortextPlanumBridgeTarget::CortextPlanumBridgeTarget (Cortext &cortext) noexcept
    : cortext_ (&cortext)
{
}

Cortext::Context
CortextPlanumBridgeTarget::ProcessTextAt (const std::string &text,
                                          const std::string &source_id,
                                          std::uint64_t timestamp)
{
  return cortext_->ProcessTextAt (text, source_id, timestamp);
}

Cortext::Context
CortextPlanumBridgeTarget::ProcessAudio (const float *pcm,
                                         std::size_t num_samples,
                                         const std::string &source_id)
{
  return cortext_->ProcessAudio (pcm, num_samples, source_id);
}

PlanumBridge::PlanumBridge (PlanumBridgeTarget &target) noexcept : target_ (&target)
{
}

std::optional<PlanumBridgeResult>
PlanumBridge::Accept (const planum::PerceptionEvent &event) const
{
  if (!ShouldRouteToText (event))
    {
      return std::nullopt;
    }

  PlanumBridgeResult result;
  result.source_id = DeriveSourceId (event);
  result.timestamp = DeriveTimestamp (event);
  result.routed_to_text = true;

  (void)target_->ProcessTextAt (std::string (event.transcript.text),
                                result.source_id, result.timestamp);
  return result;
}

bool
PlanumBridge::ShouldRouteToText (
    const planum::PerceptionEvent &event) noexcept
{
  return event.kind == planum::EventKind::final_transcript
         && !event.transcript.text.empty ();
}

std::uint64_t
PlanumBridge::DeriveTimestamp (
    const planum::PerceptionEvent &event) noexcept
{
  return event.segment_ended_at_ms != 0 ? event.segment_ended_at_ms
                                        : event.emitted_at_ms;
}

std::string
PlanumBridge::DeriveSourceId (const planum::PerceptionEvent &event)
{
  const std::string_view stream_component
      = SourceComponentOr (event.stream_id.value, kUnknownStream);
  const std::string_view speaker_component
      = SourceComponentOr (event.speaker.id.value, kUnknownSpeaker);

  std::string source_id;
  source_id.reserve (kSourcePrefix.size () + 1 + stream_component.size () + 1
                     + speaker_component.size ());
  source_id.append (kSourcePrefix);
  source_id.push_back ('/');
  source_id.append (stream_component);
  source_id.push_back ('/');
  source_id.append (speaker_component);
  return source_id;
}

} // namespace cortext::audio
