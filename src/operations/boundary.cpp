#include "cortext/operations/boundary.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "../experimental_env.hpp"

#include <cmath>
#include <string>

namespace cortext::operations
{

namespace
{
constexpr double kEpsilon = 1e-9;

double
NormalizeLocal (double raw, double alpha, double min_var, double gain,
                double &mean, double &var, bool use_norm)
{
  if (!std::isfinite (raw))
    {
      raw = 0.0;
    }
  const double mean_old = (var <= 0.0 && mean == 0.0) ? raw : mean;
  const double var_old = (var <= 0.0) ? min_var : var;
  const double denom = std::sqrt (var_old + kEpsilon);
  const double z = (raw - mean_old) / denom;
  const double norm = use_norm ? core::Sigmoid (z * gain) : raw;
  const double delta = raw - mean_old;
  mean = mean_old + alpha * delta;
  var = (1.0 - alpha) * var_old + alpha * delta * delta;
  if (var < min_var)
    {
      var = min_var;
    }
  return norm;
}
}  // namespace

void
DetectBoundary::Execute (OperationContext &context,
                         Transaction & /*tx*/) const
{
  const bool disable_pressure = internal::experimental_env::Flag (
      "CORTEXT_BOUNDARY_DISABLE_PRESSURE");
  const bool disable_surprisal = internal::experimental_env::Flag (
      "CORTEXT_BOUNDARY_DISABLE_SURPRISAL");
  const bool disable_natural = internal::experimental_env::Flag (
      "CORTEXT_BOUNDARY_DISABLE_NATURAL");

  const auto &signal = context.GetSignal ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();
  const std::string &source_id = signal.source_id;

  if (signal.force_boundary)
    {
      context.SetFlushRequired (true);
      context.SetAtBoundary (true);
      context.SetBoundaryScore (1.0);
      context.SetBoundaryType (std::string ("explicit_turn"));
      BoundaryDiagnostics diagnostics;
      diagnostics.boundary_score = 1.0;
      diagnostics.should_flush = true;
      context.SetBoundaryDiagnostics (diagnostics);
      return;
    }

  // Get accumulator state
  auto it = p_ctx.accumulator_states.find (source_id);
  if (it == p_ctx.accumulator_states.end () || it->second.n_signals == 0)
    {
      // No accumulator state - no boundary
      context.SetFlushRequired (false);
      context.SetBoundaryScore (0.0);
      BoundaryDiagnostics diagnostics;
      diagnostics.boundary_score = 0.0;
      diagnostics.boundary_rate_ema = p_ctx.boundary_rate_ema;
      context.SetBoundaryDiagnostics (diagnostics);
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
  const double dt_ref = (p_ctx.dt_ema > 0.0) ? p_ctx.dt_ema : signal_gap;
  const double gap_ref_s = core::GapScale (config.stability)
                           * std::max (dt_ref, kEpsilon);
  const double gap_z = (signal_gap - gap_ref_s) / std::max (gap_ref_s, kEpsilon);
  const double gap_score = core::Sigmoid (gap_z);
  const bool gap_has_history = p_ctx.dt_ema > 0.0;
  const double gap_ratio
      = gap_has_history ? (signal_gap / std::max (p_ctx.dt_ema, kEpsilon))
                        : 1.0;
  const double gap_z_inact
      = std::log1p (std::max (0.0, gap_ratio - 1.0));
  const auto inactivity_policy = core::BoundaryInactivityPolicyForKnobs (
      config.sensitivity, config.stability);
  const double gap_score_inact
      = core::Sigmoid (inactivity_policy.gap_score_scale * gap_z_inact);
  const double gap_force
      = core::Sigmoid (
          inactivity_policy.gap_force_k
          * (gap_z_inact - inactivity_policy.gap_force_center));

  // Bayesian change-point inference for boundary probability
  const double h_t_base = core::BoundaryHazardPrior (
      config.focus, config.sensitivity, config.stability);

  double surprisal_raw
      = context.GetMetric (operations::Metric::embedding_surprisal)
            .value_or (0.0);
  if (disable_surprisal)
    {
      surprisal_raw = 0.0;
    }
  const double drift_raw = core::Sigmoid (drift_spike);
  const double coh_raw = coh_drop;
  double topic_raw = 0.0;
  if (signal.embedding.size () > 0)
    {
      const Eigen::VectorXf *anchor = nullptr;
      if (acc.c_t.size () == signal.embedding.size () && acc.c_t.size () > 0)
        {
          anchor = &acc.c_t;
        }
      else if (acc.e_peak.size () == signal.embedding.size ()
               && acc.e_peak.size () > 0)
        {
          anchor = &acc.e_peak;
        }
      else if (acc.mu_acc.size () == signal.embedding.size ()
               && acc.mu_acc.size () > 0)
        {
          anchor = &acc.mu_acc;
        }
      if (anchor)
        {
          const double cos = core::CosineSimilarity (signal.embedding, *anchor);
          topic_raw = 1.0 - core::Map01 (cos);
        }
    }

  const auto norm_policy = core::BoundaryLocalNormPolicyForKnobs (
      config.sensitivity, config.stability);
  const bool use_local = acc.n_signals >= norm_policy.warmup_signals;
  const double alpha_local = norm_policy.alpha;
  const double norm_gain = norm_policy.gain;
  const double surprisal_norm
      = NormalizeLocal (surprisal_raw, alpha_local, norm_policy.variance_floor,
                        norm_gain,
                        acc.boundary_surprisal_mean,
                        acc.boundary_surprisal_var, use_local);
  const double drift_norm
      = NormalizeLocal (drift_raw, alpha_local, norm_policy.variance_floor,
                        norm_gain,
                        acc.boundary_drift_spike_mean,
                        acc.boundary_drift_spike_var, use_local);
  const double coh_norm
      = NormalizeLocal (coh_raw, alpha_local, norm_policy.variance_floor,
                        norm_gain,
                        acc.boundary_coh_drop_mean,
                        acc.boundary_coh_drop_var, use_local);
  const double topic_norm
      = NormalizeLocal (topic_raw, alpha_local, norm_policy.variance_floor,
                        norm_gain,
                        acc.boundary_topic_shift_mean,
                        acc.boundary_topic_shift_var, use_local);

  const auto boundary_weights = core::BoundaryEvidenceScoringWeights (
      config.focus, config.sensitivity, config.stability);
  const double w_s = boundary_weights.surprisal;
  const double w_d = boundary_weights.drift;
  const double w_c = boundary_weights.coherence;
  const double w_t = boundary_weights.topic;
  const double w_g = boundary_weights.gap;

  // Natural-indicator gate: emphasize coherence/topic shifts over isolated spikes.
  const auto support_policy = core::BoundarySupportPolicyForKnobs (
      config.focus, config.sensitivity, config.stability, coh_norm,
      topic_norm);
  const double support = support_policy.support;
  const double support_gate = support_policy.support_gate;
  const double gap_gate = support_policy.gap_gate;
  const double support_boost = support_policy.support_boost;
  const double gap_z_inact_pos = std::max (0.0, gap_z_inact);
  const double support_relax
      = std::exp (-inactivity_policy.support_relax_rate * gap_z_inact_pos);
  const double inactivity
      = core::Clamp (gap_score_inact * (1.0 - support * support_relax), 0.0, 1.0);
  const double inactivity_scale
      = std::exp (inactivity_policy.inactivity_exp_rate * gap_z_inact_pos);

  const double w_s_eff = w_s * support_gate;
  const double w_d_eff = w_d * support_gate;
  const double w_c_eff = w_c * support_boost;
  const double w_t_eff = w_t * support_boost;
  const double w_g_eff = w_g * gap_gate;
  const double w_sum
      = std::max (kEpsilon,
                  w_s_eff + w_d_eff + w_c_eff + w_t_eff + w_g_eff);
  const auto change_point_policy = core::BoundaryChangePointPolicyForKnobs (
      config.sensitivity, config.stability, support);
  const double z_center = change_point_policy.z_center;
  const double z_center_eff = change_point_policy.z_center_eff;
  const double z_t
      = (w_s_eff / w_sum) * (surprisal_norm - z_center_eff)
        + (w_d_eff / w_sum) * (drift_norm - z_center_eff)
        + (w_c_eff / w_sum) * (coh_norm - z_center_eff)
        + (w_t_eff / w_sum) * (topic_norm - z_center_eff)
        + (w_g_eff / w_sum) * (gap_score - z_center_eff);
  const double k_cp_eff = change_point_policy.k;
  const double likelihood = core::Sigmoid (k_cp_eff * z_t);
  const double target_rate
      = core::BoundaryTargetRate (config.focus, config.sensitivity,
                                  config.stability);
  const double rate_gain
      = core::BoundaryRateGain (config.sensitivity, config.stability);
  const double rate_mult
      = core::Clamp (1.0 + rate_gain * (target_rate - p_ctx.boundary_rate_ema),
                     0.5, 1.5);
  const double h_t
      = core::Clamp (h_t_base * rate_mult * core::Lerp (0.8, 1.3, support),
                     0.0, 0.5);
  double boundary_score
      = (h_t * likelihood)
        / std::max (kEpsilon, h_t * likelihood + (1.0 - h_t) * (1.0 - likelihood));
  boundary_score = core::Clamp (boundary_score, 0.0, 1.0);
  if (disable_natural)
    {
      boundary_score = 0.0;
    }
  else if (inactivity > 0.0)
    {
      boundary_score
          = core::Clamp (boundary_score
                             + inactivity_policy.inactivity_weight * inactivity
                                   * inactivity_scale,
                         0.0, 1.0);
    }
  boundary_score = std::max (boundary_score, gap_force);

  context.SetBoundaryScore (boundary_score);

  // Check flush conditions
  bool flush = false;
  bool drift_trigger = false;
  bool timeout_trigger = false;
  bool pressure_trigger = false;

  // 1. Boundary score exceeds threshold
  const double b_thresh
      = core::BoundaryThreshold (config.focus, config.sensitivity);
  const double boundary_floor
      = core::BoundaryFallbackFloor (config.focus, config.sensitivity,
                                     config.stability);
  const bool boundary_hit = boundary_score > b_thresh;
  if (boundary_hit)
    {
      flush = true;
      drift_trigger = true;
      telemetry::AddCounter ("cortext.accumulator.flush_boundary_score", 1);
    }
  const double rate_alpha
      = core::BoundaryRateAlpha (config.sensitivity, config.stability);
  p_ctx.boundary_rate_ema
      = core::Ewma (p_ctx.boundary_rate_ema,
                    boundary_hit ? 1.0 : 0.0, rate_alpha);

  // 2. Memory elapsed time exceeds adaptive cadence
  const double max_time
      = core::AdaptiveMaxMemoryTime (config.stability, gap_ref_s);
  const double memory_elapsed
      = static_cast<double> (signal.timestamp - acc.t_start) / 1000.0;
  // 2. No hard timeouts; inactivity only boosts boundary score.

  // 3. Pressure vs capacity (dynamic flush probability)
  const double base_capacity = core::MaxMemoryDrift (config.sensitivity);
  const auto pressure_policy = core::BoundaryPressurePolicyForKnobs (
      config.sensitivity, config.stability, acc.drift_acc);
  const double capacity = pressure_policy.capacity;
  const double pressure = pressure_policy.pressure;
  const double saturation_ratio = pressure / std::max (capacity, kEpsilon);
  const double k_flush
      = core::SurpriseGain (config.sensitivity, config.stability);
  const double pressure_score
      = core::Sigmoid ((saturation_ratio - 1.0) * k_flush);
  if (!disable_pressure && pressure_score > b_thresh
      && boundary_score >= boundary_floor)
    {
      flush = true;
      pressure_trigger = true;
      drift_trigger = true;
      telemetry::AddCounter ("cortext.accumulator.flush_pressure", 1);
    }

  context.SetFlushRequired (flush);
  context.SetAtBoundary (flush);

  BoundaryDiagnostics diagnostics;
  diagnostics.coherence_prev = coherence_prev;
  diagnostics.coherence_curr = coherence;
  diagnostics.coh_drop = coh_drop;
  diagnostics.coh_drop_norm = coh_norm;
  diagnostics.d_step = d_step;
  diagnostics.eta_prev = eta_prev;
  diagnostics.drift_spike = drift_spike;
  diagnostics.drift_norm = drift_norm;
  diagnostics.surprisal_raw = surprisal_raw;
  diagnostics.surprisal_norm = surprisal_norm;
  diagnostics.topic_shift = topic_raw;
  diagnostics.topic_norm = topic_norm;
  diagnostics.boundary_rate_ema = p_ctx.boundary_rate_ema;
  diagnostics.boundary_rate_mult = rate_mult;
  diagnostics.boundary_threshold = b_thresh;
  diagnostics.boundary_floor = boundary_floor;
  diagnostics.boundary_score = boundary_score;
  diagnostics.should_flush = flush;
  context.SetBoundaryDiagnostics (diagnostics);

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
    telemetry::Attribute::Double ("surprisal_raw", surprisal_raw),
    telemetry::Attribute::Double ("surprisal_norm", surprisal_norm),
    telemetry::Attribute::Double ("coh_drop_norm", coh_norm),
    telemetry::Attribute::Double ("drift_norm", drift_norm),
    telemetry::Attribute::Double ("topic_shift", topic_raw),
    telemetry::Attribute::Double ("topic_norm", topic_norm),
    telemetry::Attribute::Double ("norm_gain", norm_gain),
    telemetry::Attribute::Double ("support", support),
    telemetry::Attribute::Double ("support_gate", support_gate),
    telemetry::Attribute::Double ("gap_gate", gap_gate),
    telemetry::Attribute::Double ("hazard", h_t),
    telemetry::Attribute::Double ("hazard_base", h_t_base),
    telemetry::Attribute::Double ("boundary_rate_target", target_rate),
    telemetry::Attribute::Double ("boundary_rate_ema", p_ctx.boundary_rate_ema),
    telemetry::Attribute::Double ("boundary_rate_mult", rate_mult),
    telemetry::Attribute::Double ("w_s", w_s),
    telemetry::Attribute::Double ("w_d", w_d),
    telemetry::Attribute::Double ("w_c", w_c),
    telemetry::Attribute::Double ("w_g", w_g),
    telemetry::Attribute::Double ("w_s_eff", w_s_eff),
    telemetry::Attribute::Double ("w_d_eff", w_d_eff),
    telemetry::Attribute::Double ("w_c_eff", w_c_eff),
    telemetry::Attribute::Double ("w_t_eff", w_t_eff),
    telemetry::Attribute::Double ("w_g_eff", w_g_eff),
    telemetry::Attribute::Double ("z_center", z_center),
    telemetry::Attribute::Double ("z_center_eff", z_center_eff),
    telemetry::Attribute::Double ("k_cp_eff", k_cp_eff),
    telemetry::Attribute::Double ("likelihood", likelihood),
    telemetry::Attribute::Double ("boundary_score", boundary_score),
    telemetry::Attribute::Double ("boundary_floor", boundary_floor),
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
    telemetry::Attribute::Double ("inactivity", inactivity),
    telemetry::Attribute::Double ("inactivity_gain",
                                  inactivity_policy.inactivity_weight),
    telemetry::Attribute::Double ("inactivity_scale", inactivity_scale),
    telemetry::Attribute::Bool ("trigger_drift", drift_trigger),
    telemetry::Attribute::Bool ("trigger_timeout", timeout_trigger),
    telemetry::Attribute::Bool ("trigger_pressure", pressure_trigger),
    telemetry::Attribute::Bool ("should_flush", flush)
  });
}

} // namespace cortext::operations
