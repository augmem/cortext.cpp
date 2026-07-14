#include "cortext/operations/streaming_pacing.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <limits>

namespace cortext::operations
{

void
CheckStreamingPacing::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &signal = context.GetSignal ();
  const bool ephemeral_query = signal.retention == Retention::Ephemeral;
  if (ephemeral_query)
    {
      // A probe always retrieves, but must not advance or reset the pacing
      // state that belongs to the open same-source Natural stream.
      context.SetShouldCheckRetrieval (true);
      return;
    }

  // Compute thresholds from knobs
  const double pacing_thresh = core::StreamingPacingThreshold (cfg.sensitivity);
  const double max_wait = core::MaxWaitDrift (cfg.focus);
  const int adjacent_window = core::AdjacentWindow (cfg.focus);

  auto it = p_ctx.accumulator_states.find (signal.source_id);
  if (it == p_ctx.accumulator_states.end ())
    {
      p_ctx.accumulator_states[signal.source_id] = AccumulatorState{};
      it = p_ctx.accumulator_states.find (signal.source_id);
    }
  auto &acc = it->second;

  const bool at_boundary = context.GetFlushRequired ();
  const Eigen::VectorXf *x_ptr = &signal.embedding;
  if (acc.mu_acc.size () > 0)
    {
      x_ptr = &acc.mu_acc;
    }
  if (x_ptr->size () > 0)
    {
      if (acc.x_last_check.size () == 0
          || acc.x_last_check.size () != x_ptr->size ())
        {
          acc.x_last_check = *x_ptr;
          acc.drift_acc_pacing = 0.0;
        }
      else
        {
          const double sim
              = core::CosineSimilarity (*x_ptr, acc.x_last_check);
          const double dist = 1.0 - sim;
          acc.drift_acc_pacing += dist;
        }
    }

  // Determine if we should trigger a retrieval check
  const bool exceeds_threshold = acc.drift_acc_pacing > pacing_thresh;
  const bool force_check = acc.drift_acc_pacing > max_wait;
  const uint64_t now_ts = signal.timestamp;
  double since_last_s = std::numeric_limits<double>::infinity ();
  if (p_ctx.last_retrieval_ts > 0)
    {
      since_last_s = now_ts > p_ctx.last_retrieval_ts
                         ? static_cast<double> (
                               now_ts - p_ctx.last_retrieval_ts)
                               / 1000.0
                         : 0.0;
    }
  const double min_gap_s
      = static_cast<double> (std::max (0, adjacent_window))
        * std::max (p_ctx.dt_ema, 0.0);
  const bool adjacent_ok = since_last_s >= min_gap_s;

  const double retrieval_bias = core::Clamp (p_ctx.retrieval_bias, 0.0, 1.0);
  const double retrieval_bias_threshold
      = core::StreamingRetrievalBiasThreshold (cfg.focus, cfg.sensitivity,
                                               cfg.stability);
  const bool bias_ok = ephemeral_query
                       || (retrieval_bias >= retrieval_bias_threshold)
                       || at_boundary || force_check;
  const bool should_check
      = ephemeral_query
        || ((at_boundary || exceeds_threshold || force_check)
            && (adjacent_ok || at_boundary || force_check) && bias_ok);

  context.SetShouldCheckRetrieval (should_check);

  // If triggered, reset drift accumulator and update reference embedding
  if (should_check && x_ptr->size () > 0)
    {
      acc.x_last_check = *x_ptr;
      acc.drift_acc_pacing = 0.0;
      p_ctx.last_retrieval_ts = now_ts;
    }

  // Debug logging
  telemetry::LogDebug ("cortext.streaming_pacing", {
    telemetry::Attribute::Double ("drift_acc_pacing", acc.drift_acc_pacing),
    telemetry::Attribute::Double ("pacing_threshold", pacing_thresh),
    telemetry::Attribute::Bool ("should_check_retrieval", should_check),
    telemetry::Attribute::Int64 ("adjacent_window", adjacent_window),
    telemetry::Attribute::Double ("since_last_retrieval_s", since_last_s),
    telemetry::Attribute::Double ("retrieval_bias_threshold",
                                  retrieval_bias_threshold),
    telemetry::Attribute::Double ("retrieval_bias", retrieval_bias),
    telemetry::Attribute::Bool ("ephemeral_query", ephemeral_query)
  });
}

} // namespace cortext::operations
