#include "cortext/operations/streaming_pacing.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
CheckStreamingPacing::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &signal = context.GetSignal ();

  // Compute thresholds from knobs
  const double pacing_thresh = core::StreamingPacingThreshold (cfg.sensitivity);
  const double max_wait = core::MaxWaitDrift (cfg.focus);

  // Determine if we should trigger a retrieval check
  const bool exceeds_threshold = p_ctx.drift_accum > pacing_thresh;
  const bool force_check = p_ctx.drift_accum > max_wait;
  const bool should_check = exceeds_threshold || force_check;

  context.SetShouldCheckRetrieval (should_check);

  // If triggered, reset drift accumulator and update reference embedding
  if (should_check && signal.embedding.size () > 0)
    {
      p_ctx.last_pacing_check_embedding = signal.embedding;
      p_ctx.drift_accum = 0.0;
    }

  // Debug logging
  telemetry::LogDebug ("cortext.streaming_pacing", {
    telemetry::Attribute::Double ("drift_accum", p_ctx.drift_accum),
    telemetry::Attribute::Double ("pacing_threshold", pacing_thresh),
    telemetry::Attribute::Bool ("should_check_retrieval", should_check)
  });
}

} // namespace cortext::operations
