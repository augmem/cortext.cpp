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

  const double eta_prev
      = context.GetAccumulatorEtaPrev ().value_or (acc.eta_acc);

  // Compute drift spike (Section 4.4.3) with cold-start guard
  double drift_spike = 0.0;
  if (eta_prev >= core::EtaColdStart (config.stability))
    {
      const double eta_safe = std::max (eta_prev, kEpsilon);
      drift_spike = (d_step - eta_prev) / eta_safe;
    }

  // Compute coherence drop (normalized to [0,1])
  const double coherence_prev = acc.coherence_prev;
  const double coh_drop = core::Clamp ((coherence_prev - coherence) * 0.5,
                                       0.0, 1.0);

  // Update coherence_prev for next signal
  acc.coherence_prev = coherence;

  // Adaptive gap signal (soft influence only)
  const double signal_gap
      = static_cast<double> (signal.timestamp - acc.last_signal_ts) / 1000.0;
  const double dt_ref = std::max (p_ctx.dt_ema, core::DtFloor (config.stability));
  const double gap_ref_s = core::GapScale (config.stability) * dt_ref;
  const double gap_z = (signal_gap - gap_ref_s) / std::max (gap_ref_s, kEpsilon);
  const double gap_score = core::Sigmoid (gap_z);

  // Bayesian change-point inference for boundary probability
  const double h_base = core::Lerp (0.02, 0.12, config.sensitivity)
                        * core::Lerp (1.2, 0.8, config.stability);
  const double h_t
      = core::Clamp (h_base * (0.8 + 0.2 * (1.0 - config.focus)), 0.0, 0.5);

  const double surprisal
      = context.GetMetric (operations::Metric::embedding_surprisal)
            .value_or (0.0);
  const double w_s = 0.45 + 0.25 * config.sensitivity;
  const double w_d = 0.25 + 0.10 * (1.0 - config.stability);
  const double w_c = 0.20 + 0.10 * config.focus;
  const double w_g = 0.10;
  const double w_sum = std::max (kEpsilon, w_s + w_d + w_c + w_g);
  const double z_t
      = (w_s / w_sum) * (surprisal - 0.5)
        + (w_d / w_sum) * (core::Sigmoid (drift_spike) - 0.5)
        + (w_c / w_sum) * (coh_drop - 0.5)
        + (w_g / w_sum) * (gap_score - 0.5);
  const double k_cp
      = core::Lerp (3.0, 9.0, config.sensitivity)
        * core::Lerp (1.1, 0.9, config.stability);
  const double likelihood = core::Sigmoid (k_cp * z_t);
  double boundary_score
      = (h_t * likelihood)
        / std::max (kEpsilon, h_t * likelihood + (1.0 - h_t) * (1.0 - likelihood));
  boundary_score = core::Clamp (boundary_score, 0.0, 1.0);

  context.SetBoundaryScore (boundary_score);

  // Check flush conditions
  bool flush = false;
  bool drift_trigger = false;
  bool timeout_trigger = false;
  bool pressure_trigger = false;

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

  // 3. Pressure vs capacity (dynamic flush probability)
  const double base_capacity = core::MaxMemoryDrift (config.sensitivity);
  const double capacity_scale = (1.0 + config.stability)
                                * (1.0 + config.stability);
  const double capacity = base_capacity * capacity_scale;
  const double pressure = acc.drift_acc * (1.0 + config.sensitivity);
  const double saturation_ratio = pressure / std::max (capacity, kEpsilon);
  const double k_flush
      = core::SurpriseGain (config.sensitivity, config.stability);
  const double pressure_score
      = core::Sigmoid ((saturation_ratio - 1.0) * k_flush);
  if (pressure_score > b_thresh)
    {
      flush = true;
      pressure_trigger = true;
      drift_trigger = true;
      telemetry::AddCounter ("cortext.accumulator.flush_pressure", 1);
    }

  context.SetFlushRequired (flush);
  context.SetAtBoundary (flush);

  if (flush)
    {
      // Select boundary type based on dominant trigger.
      if (timeout_trigger)
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
  telemetry::RecordHistogram ("cortext.accumulator.gap_ref_s", gap_ref_s);
  telemetry::RecordHistogram ("cortext.accumulator.gap_score", gap_score);

  telemetry::LogDebug ("cortext.boundary", {
    telemetry::Attribute::Double ("F", config.focus),
    telemetry::Attribute::Double ("S", config.sensitivity),
    telemetry::Attribute::Double ("T", config.stability),
    telemetry::Attribute::Double ("coherence_prev", coherence_prev),
    telemetry::Attribute::Double ("coherence_curr", coherence),
    telemetry::Attribute::Double ("coh_drop", coh_drop),
    telemetry::Attribute::Double ("d_step", d_step),
    telemetry::Attribute::Double ("eta_prev", eta_prev),
    telemetry::Attribute::Double ("drift_spike", drift_spike),
    telemetry::Attribute::Double ("hazard", h_t),
    telemetry::Attribute::Double ("w_s", w_s),
    telemetry::Attribute::Double ("w_d", w_d),
    telemetry::Attribute::Double ("w_c", w_c),
    telemetry::Attribute::Double ("w_g", w_g),
    telemetry::Attribute::Double ("likelihood", likelihood),
    telemetry::Attribute::Double ("boundary_score", boundary_score),
    telemetry::Attribute::Double ("boundary_threshold", b_thresh),
    telemetry::Attribute::Double ("elapsed_time_ms", memory_elapsed),
    telemetry::Attribute::Double ("max_time_s", max_time),
    telemetry::Attribute::Double ("drift_accum", acc.drift_acc),
    telemetry::Attribute::Double ("base_capacity", base_capacity),
    telemetry::Attribute::Double ("capacity", capacity),
    telemetry::Attribute::Double ("pressure", pressure),
    telemetry::Attribute::Double ("saturation_ratio", saturation_ratio),
    telemetry::Attribute::Double ("pressure_score", pressure_score),
    telemetry::Attribute::Double ("k_flush", k_flush),
    telemetry::Attribute::Double ("signal_gap_s", signal_gap),
    telemetry::Attribute::Double ("dt_ref_s", dt_ref),
    telemetry::Attribute::Double ("gap_ref_s", gap_ref_s),
    telemetry::Attribute::Double ("gap_z", gap_z),
    telemetry::Attribute::Double ("gap_score", gap_score),
    telemetry::Attribute::Bool ("trigger_drift", drift_trigger),
    telemetry::Attribute::Bool ("trigger_timeout", timeout_trigger),
    telemetry::Attribute::Bool ("trigger_pressure", pressure_trigger),
    telemetry::Attribute::Bool ("should_flush", flush)
  });
}

} // namespace cortext::operations
