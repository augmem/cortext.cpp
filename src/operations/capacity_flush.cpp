#include "cortext/operations/capacity_flush.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <algorithm>
#include <string>

namespace cortext::operations
{

void
CheckCapacityFlush::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const auto &config = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();

  if (signal.retention == Retention::Ephemeral)
    {
      // Ephemeral retrieval probes must not turn an open Natural unit into a
      // capacity boundary or finalize its episode.
      return;
    }

  int n_signals = 0;
  auto it = p_ctx.accumulator_states.find (signal.source_id);
  if (it != p_ctx.accumulator_states.end ())
    {
      n_signals = std::max (0, it->second.n_signals);
    }

  const double boundary_score = context.GetBoundaryScore ().value_or (0.0);
  const double boundary_floor = core::BoundaryFallbackFloor (
      config.focus, config.sensitivity, config.stability);
  const int max_signals = core::MaxSignalsPerMemory (
      config.focus, config.sensitivity, config.stability);
  const bool capacity_flush = (max_signals > 0) && (n_signals >= max_signals)
                              && (boundary_score >= boundary_floor);

  if (capacity_flush)
    {
      context.SetFlushRequired (true);
      context.SetAtBoundary (true);
      context.RequestFinalizeEpisode ();
      if (!context.GetBoundaryType ().has_value ())
        {
          context.SetBoundaryType (std::string ("capacity"));
        }
      telemetry::AddCounter ("cortext.accumulator.max_signal_flush_total", 1);
    }

  telemetry::LogDebug ("cortext.capacity_flush", {
    telemetry::Attribute::Int64 ("n_signals", n_signals),
    telemetry::Attribute::Int64 ("max_signals", max_signals),
    telemetry::Attribute::Double ("boundary_score", boundary_score),
    telemetry::Attribute::Double ("boundary_floor", boundary_floor),
    telemetry::Attribute::Bool ("capacity_flush", capacity_flush)
  });
}

} // namespace cortext::operations
