#include "cortext/operations/threshold.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <optional>
#include <vector>
namespace cortext::operations
{
namespace
{
constexpr double kMilli = 1e-3;
constexpr double kTiny = 1e-6;
constexpr double kEssCap = 100.0;
constexpr double kThree = 3.0;
constexpr double kSecondsPerMinute = 60.0;
inline double
PercentileP90 (const std::deque<double> &values, int window)
{
  if (values.empty () || window <= 0)
    {
      return 0.0;
    }
  const int n = static_cast<int> (values.size ());
  const int start = std::max (0, n - window);
  std::vector<double> tail;
  tail.reserve (n - start);
  for (int i = start; i < n; ++i)
    {
      tail.push_back (values[static_cast<size_t> (i)]);
    }
  if (tail.empty ())
    {
      return 0.0;
    }
  std::sort (tail.begin (), tail.end ());
  const int idx = std::max (
      0, std::min (static_cast<int> (tail.size ()) - 1,
                   static_cast<int> (std::floor (
                       0.9 * (static_cast<int> (tail.size ()) - 1)))));
  return tail[static_cast<size_t> (idx)];
}

} // namespace

void
UpdateThreshold::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // 1) Determine observed score for this signal.
  // Prefer a composite score set by earlier operations; fallback to
  // relevance weight (bounded [0,1]).
  double observed_score
      = context.GetCompositeScore ().value_or (p_ctx.weight_relevance);
  observed_score = core::Clamp (observed_score, constants::kNormalizedMin,
                                constants::kNormalizedMax);

  // 2) Update rolling score history (cap size to a reasonable maximum).
  p_ctx.recent_scores.push_back (observed_score);
  const int w = core::WScore (cfg.stability);
  const size_t kMaxScores = 1024;
  if (p_ctx.recent_scores.size () > kMaxScores)
    {
      p_ctx.recent_scores.pop_front ();
    }

  // 3) Compute observed p90 over the last w samples.
  const double observed_p90 = PercentileP90 (p_ctx.recent_scores, w);

  // 4) Bayesian blend target using prior and experiential mass.
  const double rho_prior
      = static_cast<double> (core::PriorMass (cfg.stability));
  const int count_scores = static_cast<int> (p_ctx.recent_scores.size ());
  const double rho_obs = core::Clamp (p_ctx.u_t, constants::kNormalizedMin,
                                      constants::kNormalizedMax)
                         * static_cast<double> (std::min (w, count_scores));
  const double T_prior
      = core::TPrior (cfg.focus, cfg.sensitivity, cfg.stability);
  const double denom
      = std::max (constants::kNormEpsilon, rho_prior + rho_obs);
  const double T_target
      = (rho_prior * T_prior + rho_obs * observed_p90) / denom;

  // 5) EWMA toward target with α_T schedule.
  const double alpha_T = core::AlphaT (cfg.stability, p_ctx.u_t);
  p_ctx.T_dynamic = core::Ewma (p_ctx.T_dynamic, T_target,
                                core::Clamp (alpha_T, constants::kNormalizedMin,
                                             constants::kNormalizedMax));

  // 6) Threshold deltas (Algorithm 8 steps 8-13)
  const double kSensitivityGain = constants::kGainMedium;
  const double delta_sens = context.GetDeltaThresholdSensitivity ().value_or (
      (cfg.sensitivity - 0.5) * kSensitivityGain);
  const double delta_prec
      = context.GetDeltaThresholdPrecision ().value_or (constants::kNormalizedMin);
  const double delta_emo
      = context.GetDeltaThresholdEmotion ().value_or (constants::kNormalizedMin);
  const double delta_mood
      = context.GetDeltaThresholdMood ().value_or (constants::kNormalizedMin);
  double delta_total = delta_sens + delta_prec + delta_emo + delta_mood;

  // 7) Continuous-time rate control (EMA + ESS) → ΔT_homeo
  // Use signal timestamps to estimate Δt; fallback to 1.0s if unavailable.
  const uint64_t now_ts = context.GetSignal ().timestamp;
  double delta_t = 1.0;
  if (p_ctx.last_rate_timestamp != 0 && now_ts > p_ctx.last_rate_timestamp)
    {
      delta_t = static_cast<double> (now_ts - p_ctx.last_rate_timestamp);
      if (delta_t <= constants::kNormEpsilon)
        {
          delta_t = kMilli;
        }
    }
  p_ctx.last_rate_timestamp = now_ts == 0 ? p_ctx.last_rate_timestamp : now_ts;

  // α_dt smoothing for dt_ema
  const double denom_dt = std::max (delta_t, kMilli);
  const double alpha_dt = 1.0 - std::exp (-delta_t / denom_dt);
  p_ctx.dt_ema = (1.0 - alpha_dt) * p_ctx.dt_ema + alpha_dt * delta_t;
  const double dt_base = std::max (p_ctx.dt_ema, 1.0);

  // τ_rate and α for rate EWMA
  const double tau_rate = std::max (std::pow (constants::kTwo, kThree * cfg.stability)
                                        * dt_base,
                                    1.0);
  const double alpha_rate = 1.0 - std::exp (-delta_t / tau_rate);

  // Observed instantaneous rate (writes/min); Δwrites=1 per signal
  const double rho_inst = (delta_t > constants::kNormEpsilon)
                              ? (kSecondsPerMinute / delta_t)
                              : constants::kNormalizedMin;
  p_ctx.m_rate = (1.0 - alpha_rate) * p_ctx.m_rate + alpha_rate * rho_inst;
  p_ctx.rate_ticks = p_ctx.rate_ticks + 1;

  // Debiased rate estimate ρ_hat
  const double denom_bias
      = std::max (1.0
                      - std::pow (1.0 - alpha_rate,
                                  static_cast<double> (p_ctx.rate_ticks + 1)),
                  kTiny);
  const double rho_hat = p_ctx.m_rate / denom_bias;

  // Reliability via ESS
  const double beta
      = std::max (constants::kNormalizedMin, 1.0 - alpha_rate);
  const double ess
      = std::min ((1.0 + beta) / std::max (1.0 - beta, kTiny), kEssCap);
  const double reliability = 1.0 - std::exp (-ess * (1.0 - cfg.stability));

  // Rate error vs target write rate
  const double rate_target
      = (context.GetProcessorContext ().rate_target > 0.0)
            ? context.GetProcessorContext ().rate_target
            : context.GetProcessorContext ().rate_target_prior;
  const double rate_err = std::tanh ((rho_hat - std::max (rate_target, constants::kNormalizedMin))
                                     / std::max (rate_target, kTiny));

  // ΔT_homeo with cap from hysteresis (use the current stability-derived band)
  const double hysteresis_val
      = core::Lerp (constants::kHysteresisBandMin, constants::kHysteresisBandMax,
                    cfg.stability);
  const double cap = constants::kQuarter * hysteresis_val;
  const double maturity
      = core::ComputeMaturity (p_ctx.signals_processed, cfg.stability);
  const double kappa_r = constants::kGainMedium;
  const double delta_homeo
      = core::Clamp (reliability * kappa_r * (1.0 - cfg.stability)
                         * (1.0 - maturity) * rate_err,
                     -cap, +cap);
  delta_total += delta_homeo;

  // 8) Apply time-scaled annealed rails and clamp total delta.
  // Per algorithms.md Section 4.3: cap_total = max_ΔT_per_min × (Δt / 60.0)
  const double max_delta_per_min
      = core::MaxDeltaTPerMin (p_ctx.signals_processed, cfg.stability);
  // Scale by elapsed time; use minimum floor of 0.1s to avoid zeroing deltas
  const double cap_total
      = max_delta_per_min * std::max (delta_t, 0.1) / kSecondsPerMinute;
  delta_total = core::Clamp (delta_total, -cap_total, +cap_total);
  const double Tmin = core::TMin (p_ctx.signals_processed, cfg.stability);
  const double Tmax = core::TMax (p_ctx.signals_processed, cfg.stability);
  p_ctx.T_dynamic = core::Clamp (p_ctx.T_dynamic + delta_total, Tmin, Tmax);

  // 9) Hysteresis band derived from Stability knob.
  p_ctx.hysteresis = hysteresis_val;

  // Expose as intermediate results for downstream operations/telemetry.
  context.SetThresholdTDynamic (p_ctx.T_dynamic);
  context.SetThresholdHysteresis (p_ctx.hysteresis);
}

} // namespace cortext::operations
