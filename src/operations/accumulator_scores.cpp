#include "cortext/operations/accumulator_scores.hpp"

#include "cortext/processor/accumulator_state.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <algorithm>

namespace cortext::operations
{

void
UpdateAccumulatorScores::Execute (OperationContext &context,
                                  Transaction & /*tx*/) const
{
  const auto &signal = context.GetSignal ();
  if (signal.retention == Retention::Ephemeral)
    {
      return;
    }
  auto &p_ctx = context.GetProcessorContext ();

  auto it = p_ctx.accumulator_states.find (signal.source_id);
  if (it == p_ctx.accumulator_states.end () || it->second.n_signals == 0)
    {
      return;
    }
  auto &acc = it->second;

  const double score = context.GetCompositeScore ().value_or (0.0);

  acc.s_sum += score;
  if (score > acc.s_max)
    {
      acc.s_max = score;
      if (acc.mu_acc.size () > 0)
        {
          acc.e_peak = acc.mu_acc;
        }
      else if (signal.embedding.size () > 0)
        {
          acc.e_peak = signal.embedding;
        }
    }

  if (!acc.signals.empty ())
    {
      acc.signals.back ().score = score;
    }

  telemetry::RecordHistogram ("cortext.accumulator.s_sum", acc.s_sum);
  telemetry::RecordHistogram ("cortext.accumulator.s_max", acc.s_max);

  telemetry::LogDebug ("cortext.accumulator_scores", {
    telemetry::Attribute::Int64 ("signal_count", acc.n_signals),
    telemetry::Attribute::Double ("score", score),
    telemetry::Attribute::Double ("s_sum", acc.s_sum),
    telemetry::Attribute::Double ("s_max", acc.s_max)
  });
}

} // namespace cortext::operations
