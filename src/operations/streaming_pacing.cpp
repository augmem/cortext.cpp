#include "cortext/operations/streaming_pacing.hpp"

#include "cortext/core/algorithms.hpp"
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

  auto it = p_ctx.accumulator_states.find (signal.source_id);
  if (it == p_ctx.accumulator_states.end ())
    {
      p_ctx.accumulator_states[signal.source_id] = AccumulatorState{};
      it = p_ctx.accumulator_states.find (signal.source_id);
    }
  auto &acc = it->second;

  const bool at_boundary = context.GetFlushRequired ();

  if (signal.embedding.size () > 0)
    {
      if (acc.x_last_check.size () == 0
          || acc.x_last_check.size () != signal.embedding.size ())
        {
          acc.x_last_check = signal.embedding;
          acc.drift_acc_pacing = 0.0;
        }
      else
        {
          const double sim
              = core::CosineSimilarity (signal.embedding, acc.x_last_check);
          const double dist = 1.0 - sim;
          acc.drift_acc_pacing += dist;
        }
    }

  // Determine if we should trigger a retrieval check
  const bool exceeds_threshold = acc.drift_acc_pacing > pacing_thresh;
  const bool force_check = acc.drift_acc_pacing > max_wait;
  const bool should_check = at_boundary || exceeds_threshold || force_check;

  context.SetShouldCheckRetrieval (should_check);

  // If triggered, reset drift accumulator and update reference embedding
  if (should_check && signal.embedding.size () > 0)
    {
      acc.x_last_check = signal.embedding;
      acc.drift_acc_pacing = 0.0;
    }

  // Debug logging
  telemetry::LogDebug ("cortext.streaming_pacing", {
    telemetry::Attribute::Double ("drift_acc_pacing", acc.drift_acc_pacing),
    telemetry::Attribute::Double ("pacing_threshold", pacing_thresh),
    telemetry::Attribute::Bool ("should_check_retrieval", should_check)
  });
}

} // namespace cortext::operations
