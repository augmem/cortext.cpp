#include "cortext/operations/signal_history.hpp"
#include "consolidation_throughput_state_internal.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cmath>
namespace cortext::operations
{
namespace
{
constexpr double kMillisToSeconds = 1e-3;
constexpr double kSecondsPerMinute = 60.0;

// Seconds since the last rate update, floored by Δt_min(T); 1 s before any.
double
SecondsSinceLastRate (const ProcessorContext &p_ctx, std::uint64_t now_ts,
                      double stability)
{
  double delta_t = 1.0;
  if (p_ctx.last_rate_timestamp != 0 && now_ts > p_ctx.last_rate_timestamp)
    {
      delta_t = static_cast<double> (now_ts - p_ctx.last_rate_timestamp)
                * kMillisToSeconds;
    }
  return std::max (delta_t, core::DeltaTMin (stability));
}

} // namespace

void
UpdateSignalHistory::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // Rolling score history: the composite score, or the relevance weight
  // when none was set, bounded to [0,1].
  const double observed_score = core::Clamp (
      context.GetCompositeScore ().value_or (p_ctx.weight_relevance),
      constants::kNormalizedMin, constants::kNormalizedMax);
  p_ctx.recent_scores.push_back (observed_score);
  const size_t max_scores = static_cast<size_t> (
      std::max (core::WScore (cfg.stability),
                core::RecentScoreHistoryLimit (cfg.stability)));
  if (p_ctx.recent_scores.size () > max_scores)
    {
      p_ctx.recent_scores.pop_front ();
    }

  // Inter-signal time EWMA with α_dt = 1 − exp(−Δt / τ_dt(T)).
  const double delta_t = SecondsSinceLastRate (
      p_ctx, context.GetSignal ().timestamp, cfg.stability);
  const double alpha_dt
      = 1.0 - std::exp (-delta_t / core::TauDt (cfg.stability));
  const double dt_ema_prev = p_ctx.dt_ema;
  p_ctx.dt_ema = (1.0 - alpha_dt) * p_ctx.dt_ema + alpha_dt * delta_t;

  telemetry::LogDebug ("cortext.signal_history", {
    telemetry::Attribute::Double ("observed_score", observed_score),
    telemetry::Attribute::Int64 (
        "recent_scores_count",
        static_cast<int64_t> (p_ctx.recent_scores.size ())),
    telemetry::Attribute::Double ("delta_t", delta_t),
    telemetry::Attribute::Double ("alpha_dt", alpha_dt),
    telemetry::Attribute::Double ("dt_ema_prev", dt_ema_prev),
    telemetry::Attribute::Double ("dt_ema", p_ctx.dt_ema)
  });
}

void
UpdateRateState::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  const uint64_t now_ts = context.GetSignal ().timestamp;
  const double delta_t = SecondsSinceLastRate (p_ctx, now_ts, cfg.stability);

  const double dt_base = std::max (p_ctx.dt_ema, core::DtFloor (cfg.stability));
  const double tau_rate
      = std::max (std::pow (
                      constants::kTwo,
                      core::RateTauStabilityExponentScale (cfg.stability))
                      * dt_base,
                  core::DtFloor (cfg.stability));
  const double alpha_rate = 1.0 - std::exp (-delta_t / tau_rate);

  const double writes = context.GetWriteDecision () ? 1.0 : 0.0;
  const double rho_inst = (delta_t > constants::kNormEpsilon)
                              ? (writes * kSecondsPerMinute / delta_t)
                              : constants::kNormalizedMin;

  p_ctx.m_rate = (1.0 - alpha_rate) * p_ctx.m_rate + alpha_rate * rho_inst;
  consolidation_throughput_state_internal::Observe (
      p_ctx, p_ctx.m_rate, cfg.focus, cfg.sensitivity, cfg.stability);
  if (now_ts != 0)
    {
      p_ctx.last_rate_timestamp = now_ts;
    }
}

} // namespace cortext::operations
