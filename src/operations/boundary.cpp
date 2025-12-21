#include "cortext/operations/boundary.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

namespace
{
constexpr double kEpsilon = 1e-9;
}  // namespace

void
DetectBoundary::Execute (OperationContext &context,
                         Transaction & /*tx*/) const
{
  const auto &signal = context.GetSignal ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();
  const std::string &source_id = signal.source_id;

  // Get accumulator state
  auto it = p_ctx.accumulator_states.find (source_id);
  if (it == p_ctx.accumulator_states.end () || it->second.n_signals == 0)
    {
      // No accumulator state - no boundary
      context.SetFlushRequired (false);
      context.SetBoundaryScore (0.0);
      return;
    }

  auto &acc = it->second;

  // Get coherence and drift step from previous operation
  const double coherence = context.GetAccumulatorCoherence ();
  const double d_step = context.GetAccumulatorDriftStep ();

  // Compute drift spike (Section 4.4.3)
  const double eta_safe = std::max (acc.eta_acc, kEpsilon);
  const double drift_spike = (d_step - acc.eta_acc) / eta_safe;

  // Compute coherence drop
  const double coh_drop = std::max (0.0, acc.coherence_prev - coherence);

  // Update coherence_prev for next signal
  acc.coherence_prev = coherence;

  // Boundary score (weighted combination)
  const double w_drift = core::BoundaryWeightDrift (config.stability);
  const double w_coh = 1.0 - w_drift;
  const double boundary_score
      = w_drift * core::Sigmoid (drift_spike) + w_coh * coh_drop;

  context.SetBoundaryScore (boundary_score);

  // Check flush conditions
  bool flush = false;
  bool drift_trigger = false;
  bool timeout_trigger = false;
  bool gap_trigger = false;

  // 1. Boundary score exceeds threshold
  const double b_thresh
      = core::BoundaryThreshold (config.focus, config.sensitivity);
  if (boundary_score > b_thresh)
    {
      flush = true;
      drift_trigger = true;
      telemetry::AddCounter ("cortext.accumulator.flush_boundary_score", 1);
    }

  // 2. Memory elapsed time exceeds max
  const double max_time = core::MaxMemoryTime (config.stability);
  const double memory_elapsed
      = static_cast<double> (signal.timestamp - acc.t_start) / 1000.0;
  if (memory_elapsed > max_time)
    {
      flush = true;
      timeout_trigger = true;
      telemetry::AddCounter ("cortext.accumulator.flush_max_time", 1);
    }

  // 3. Accumulated drift exceeds max
  const double max_drift = core::MaxMemoryDrift (config.sensitivity);
  if (acc.drift_acc > max_drift)
    {
      flush = true;
      drift_trigger = true;
      telemetry::AddCounter ("cortext.accumulator.flush_max_drift", 1);
    }

  // 4. Signal gap exceeds threshold (natural pause detection)
  const double gap_thresh = core::GapThreshold (config.stability);
  const double signal_gap
      = static_cast<double> (signal.timestamp - acc.last_signal_ts) / 1000.0;
  if (acc.last_signal_ts > 0 && signal_gap > gap_thresh)
    {
      flush = true;
      gap_trigger = true;
      telemetry::AddCounter ("cortext.accumulator.flush_gap", 1);
    }

  context.SetFlushRequired (flush);
  context.SetAtBoundary (flush);

  if (flush)
    {
      // Select boundary type based on dominant trigger.
      if (gap_trigger)
        {
          context.SetBoundaryType (std::string ("explicit"));
        }
      else if (timeout_trigger)
        {
          context.SetBoundaryType (std::string ("timeout"));
        }
      else if (drift_trigger)
        {
          context.SetBoundaryType (std::string ("drift"));
        }
      else
        {
          context.SetBoundaryType (std::string ("explicit"));
        }

      if (acc.mu_acc.size () > 0)
        {
          context.SetBoundaryCentroid (acc.mu_acc);
        }

      context.RequestFinalizeEpisode ();
    }
  else
    {
      context.SetBoundaryType (std::nullopt);
      context.SetBoundaryCentroid (std::nullopt);
    }

  telemetry::RecordHistogram ("cortext.accumulator.boundary_score", boundary_score);
  telemetry::RecordHistogram ("cortext.accumulator.memory_elapsed",
                              memory_elapsed);
  telemetry::RecordHistogram ("cortext.accumulator.signal_gap", signal_gap);

  telemetry::LogDebug ("cortext.boundary", {
    telemetry::Attribute::Double ("boundary_score", boundary_score),
    telemetry::Attribute::Double ("elapsed_time_ms", memory_elapsed),
    telemetry::Attribute::Double ("drift_accum", acc.drift_acc),
    telemetry::Attribute::Bool ("should_flush", flush)
  });
}

} // namespace cortext::operations
