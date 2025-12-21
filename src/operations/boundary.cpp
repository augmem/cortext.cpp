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
constexpr double kEtaColdStart = 0.01;
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

  const double eta_prev
      = context.GetAccumulatorEtaPrev ().value_or (acc.eta_acc);

  // Compute drift spike (Section 4.4.3) with cold-start guard
  double drift_spike = 0.0;
  if (eta_prev >= kEtaColdStart)
    {
      const double eta_safe = std::max (eta_prev, kEpsilon);
      drift_spike = (d_step - eta_prev) / eta_safe;
    }

  // Compute coherence drop (normalized to [0,1])
  const double coh_drop = core::Clamp ((acc.coherence_prev - coherence) * 0.5,
                                       0.0, 1.0);

  // Update coherence_prev for next signal
  acc.coherence_prev = coherence;

  // Boundary score (weighted combination)
  const double w_drift = core::BoundaryWeightDrift (config.stability);
  const double w_coh = 1.0 - w_drift;
  double boundary_score
      = w_drift * core::Sigmoid (drift_spike) + w_coh * coh_drop;
  boundary_score = core::Clamp (boundary_score, 0.0, 1.0);

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

  // Update last_signal_ts after gap computation (spec: compute gap before update)
  acc.last_signal_ts = signal.timestamp;

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
