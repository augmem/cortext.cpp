#include "cortext/operations/stability.hpp"
#include "meta_learning_internal.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <any>
#include <chrono>
#include <map>
#include <string>

namespace cortext::operations
{

namespace
{
using SteadyClock = std::chrono::steady_clock;
constexpr int kRetentionScanIntervalSignals = 128;
constexpr int kRetentionScanSampleLimit = 4096;

double
ElapsedMillis (SteadyClock::time_point start)
{
  return std::chrono::duration<double, std::milli> (SteadyClock::now () - start)
      .count ();
}
} // namespace

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

  p_ctx.hysteresis_band_prior
      = meta_learning::ResolveHysteresisBandPrior (tx, cfg);
  const auto stability_priors = core::StabilityStatePriorsForKnobs (T);
  p_ctx.half_life_prior = core::BaseHalfLifePrior (T);
  p_ctx.rate_decay_prior = stability_priors.rate_decay;
  p_ctx.periphery_half_life_prior
      = core::ClampHalfLife (stability_priors.secondary_half_life_scale
                             * p_ctx.half_life_prior);
  p_ctx.salience_half_life_prior
      = core::ClampHalfLife (stability_priors.secondary_half_life_scale
                             * p_ctx.half_life_prior);
  p_ctx.drift_weight_prior = stability_priors.drift_weight;

  // Seed dynamic state from priors (Alg 6 bootstrap)
  p_ctx.half_life = p_ctx.half_life_prior;
  p_ctx.rate_decay = stability_priors.rate_decay;
  p_ctx.periphery_half_life
      = core::ClampHalfLife (stability_priors.secondary_half_life_scale
                             * p_ctx.half_life);
  p_ctx.salience_half_life
      = core::ClampHalfLife (stability_priors.secondary_half_life_scale
                             * p_ctx.half_life);
  p_ctx.drift_weight = p_ctx.drift_weight_prior;
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
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double T = cfg.stability;

  // Compute observed retention from active memories (strength >= cutoff).
  std::optional<double> observed_retention;
  const double cutoff = core::PeripheryCutoff (T);
  const double now_s
      = static_cast<double> (context.GetSignal ().timestamp) / 1000.0;
  if (auto obs = context.GetObservedRetentionSeconds (); obs.has_value ())
    {
      observed_retention = *obs;
    }
  else if (context.GetStore ())
    {
      const bool first_scan = p_ctx.last_retention_scan_signal < 0;
      const bool consolidation_advanced
          = p_ctx.last_retention_scan_consolidation_ts
            != p_ctx.last_consolidation_ts;
      const bool periodic_scan
          = (p_ctx.signals_processed - p_ctx.last_retention_scan_signal)
            >= kRetentionScanIntervalSignals;
      if (first_scan || consolidation_advanced || periodic_scan)
        {
          try
            {
              auto get_int64 = [] (const std::map<std::string, std::any> &row,
                                   const std::string &key,
                                   long long def) -> long long {
                auto it = row.find (key);
                if (it == row.end () || !it->second.has_value ())
                  return def;
                if (it->second.type () == typeid (long long))
                  return std::any_cast<long long> (it->second);
                if (it->second.type () == typeid (int))
                  return static_cast<long long> (std::any_cast<int> (it->second));
                if (it->second.type () == typeid (double))
                  return static_cast<long long> (
                      std::any_cast<double> (it->second));
                return def;
              };
              auto get_double = [] (const std::map<std::string, std::any> &row,
                                    const std::string &key,
                                    double def) -> double {
                auto it = row.find (key);
                if (it == row.end () || !it->second.has_value ())
                  return def;
                if (it->second.type () == typeid (double))
                  return std::any_cast<double> (it->second);
                if (it->second.type () == typeid (float))
                  return static_cast<double> (std::any_cast<float> (it->second));
                if (it->second.type () == typeid (long long))
                  return static_cast<double> (std::any_cast<long long> (it->second));
                if (it->second.type () == typeid (int))
                  return static_cast<double> (std::any_cast<int> (it->second));
                return def;
              };
              const auto retention_start = SteadyClock::now ();
              auto rows = tx.Execute (
                  "WITH active_sample AS ("
                  "  SELECT created_at, start_ts "
                  "  FROM memories INDEXED BY idx_memories_ltm_created_strength "
                  "  WHERE kind = 'LONG_TERM' "
                  "    AND strength >= ?2 "
                  "    AND (created_at > 0 OR start_ts > 0) "
                  "  ORDER BY created_at DESC, start_ts DESC "
                  "  LIMIT ?3"
                  ") "
                  "SELECT COUNT(*) AS count, "
                  "       AVG(MAX(0.0, ?1 - "
                  "           (CASE WHEN created_at > 0 "
                  "                 THEN created_at ELSE start_ts END) / 1000.0)) "
                  "           AS avg_age "
                  "FROM active_sample",
                  { now_s, cutoff,
                    static_cast<long long> (kRetentionScanSampleLimit) });
              context.AddOperationTiming ("Stability.retention_scan_sql",
                                          ElapsedMillis (retention_start));
              p_ctx.last_retention_scan_signal = p_ctx.signals_processed;
              p_ctx.last_retention_scan_consolidation_ts
                  = p_ctx.last_consolidation_ts;
              const auto &row = rows.empty ()
                                    ? std::map<std::string, std::any> {}
                                    : rows.front ();
              const long long count = get_int64 (row, "count", 0);
              observed_retention
                  = (count > 0) ? get_double (row, "avg_age", 0.0) : 0.0;
            }
          catch (...)
            {
              observed_retention = std::nullopt;
            }
        }
      else
        {
          context.AddOperationTiming ("Stability.retention_scan_deferred",
                                      0.0);
        }
    }

  if (observed_retention.has_value ())
    {
      const double alpha_T = core::AlphaT (T, p_ctx.u_t);
      p_ctx.retention_ema = core::Ewma (p_ctx.retention_ema,
                                        *observed_retention, alpha_T);
      p_ctx.observed_retention_history.push_back (*observed_retention);

      const int max_window = core::WRet (T);
      while (static_cast<int> (p_ctx.observed_retention_history.size ())
             > max_window)
        {
          p_ctx.observed_retention_history.pop_front ();
        }
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
