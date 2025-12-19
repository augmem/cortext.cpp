#include "cortext/operations/stability.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
InitializeStabilityPriors::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double T = cfg.stability;

  if (p_ctx.stability_priors_initialized)
    {
      return;
    }

  p_ctx.hysteresis_band_prior = core::BaseBandPrior (T);
  p_ctx.half_life_prior = core::BaseHalfLifePrior (T);
  p_ctx.rate_decay_prior = core::Lerp (0.60, 0.98, T);
  p_ctx.periphery_half_life_prior
      = core::ClampHalfLife (constants::kOneHalf * p_ctx.half_life_prior);
  p_ctx.salience_half_life_prior
      = core::ClampHalfLife (constants::kOneHalf * p_ctx.half_life_prior);
  p_ctx.drift_weight_prior = constants::kOneHalf * (1.0 - T);

  // Seed dynamic state from priors (Alg 6 bootstrap)
  p_ctx.half_life = p_ctx.half_life_prior;
  p_ctx.rate_decay = core::Lerp (0.60, 0.98, T);
  p_ctx.periphery_half_life
      = core::ClampHalfLife (constants::kOneHalf * p_ctx.half_life);
  p_ctx.salience_half_life
      = core::ClampHalfLife (constants::kOneHalf * p_ctx.half_life);
  p_ctx.stability_priors_initialized = true;

  telemetry::LogDebug("cortext.initialize_stability_priors", {
    telemetry::Attribute::Double("T", T),
    telemetry::Attribute::Double("half_life_prior", p_ctx.half_life_prior),
    telemetry::Attribute::Double("hysteresis_band_prior", p_ctx.hysteresis_band_prior),
    telemetry::Attribute::Double("rate_decay_prior", p_ctx.rate_decay_prior)
  });
}

void
UpdateStability::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double T = cfg.stability;

  // Append observed retention if provided by upstream (SignalProcessor)
  if (auto obs = context.GetObservedRetentionSeconds (); obs.has_value ())
    {
      p_ctx.observed_retention_history.push_back (*obs);
    }

  // Observed retention = last value in history if present
  double zscore = 0.0;
  const int window = core::WRet (T);
  const int n = static_cast<int> (p_ctx.observed_retention_history.size ());
  if (n >= 1 && window > 0)
    {
      const int start = std::max (0, n - window);
      const int m = n - start;
      // Compute μ and σ over tail
      double sum = 0.0;
      for (int i = start; i < n; ++i)
        {
          sum += p_ctx.observed_retention_history[static_cast<size_t> (i)];
        }
      const double mu = (m > 0) ? (sum / static_cast<double> (m)) : 0.0;
      double var_accum = 0.0;
      for (int i = start; i < n; ++i)
        {
          const double v
              = p_ctx.observed_retention_history[static_cast<size_t> (i)];
          const double d = v - mu;
          var_accum += d * d;
        }
      const double var = (m > 0) ? (var_accum / static_cast<double> (m)) : 0.0;
      const double sigma = std::max (1.0, std::sqrt (std::max (0.0, var)));
      const double observed
          = p_ctx.observed_retention_history[static_cast<size_t> (n - 1)];
      zscore = core::Clamp ((observed - mu) / sigma, -3.0, 3.0);
    }

  // target_half_life = clamp(base_half_life_prior(T) * (1 + 0.25*z + adj),
  // ...)
  const double base_hl = core::BaseHalfLifePrior (T);
  const double adj = context.GetDeltaHalfLifeAdjustment ().value_or (0.0);
  const double target_hl
      = core::ClampHalfLife (base_hl
                             * (1.0 + constants::kQuarter * zscore + adj));

  // half_life_t ← EWMA(half_life_{t−1}, target, α = α_T(T, u_t))
  const double alpha_T = core::AlphaT (T, p_ctx.u_t);
  p_ctx.half_life
      = core::ClampHalfLife (core::Ewma (p_ctx.half_life, target_hl, alpha_T));

  // Keep rate_decay within [0,1] but do not change it otherwise
  p_ctx.rate_decay = core::Clamp (p_ctx.rate_decay,
                                  constants::kNormalizedMin,
                                  constants::kNormalizedMax);

  // Keep derived half-lives in sync
  p_ctx.periphery_half_life
      = core::ClampHalfLife (constants::kOneHalf * p_ctx.half_life);
  p_ctx.salience_half_life
      = core::ClampHalfLife (constants::kOneHalf * p_ctx.half_life);

  telemetry::LogDebug("cortext.update_stability", {
    telemetry::Attribute::Double("zscore", zscore),
    telemetry::Attribute::Double("target_hl", target_hl),
    telemetry::Attribute::Double("half_life", p_ctx.half_life),
    telemetry::Attribute::Double("alpha_T", alpha_T)
  });
}

} // namespace cortext::operations
