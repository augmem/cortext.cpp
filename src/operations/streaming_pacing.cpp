#include "cortext/operations/streaming_pacing.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"

namespace cortext::operations
{

void
CheckStreamingPacing::Execute (OperationContext &context, Transaction &tx) const
{
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
}

} // namespace cortext::operations
