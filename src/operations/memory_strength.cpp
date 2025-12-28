#include "cortext/operations/memory_strength.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <cmath>
#include <string>
#include <typeinfo>
#include <vector>

namespace cortext::operations
{

void
UpdateMemoryStrength::Execute (OperationContext &context, Transaction &tx) const
{
  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();

  const double F_raw = cfg.focus;
  const double S_raw = cfg.sensitivity;
  const double T = cfg.stability;
  const double F_eff = core::FocusBias (F_raw);
  const double S_eff = core::SensitivityBias (S_raw);
  const double alpha = core::AlphaS (S_raw, p_ctx.u_t);
  const double half_life
      = (p_ctx.half_life > constants::kNormEpsilon)
            ? p_ctx.half_life
            : core::BaseHalfLifePrior (T);
  const double cutoff = core::PeripheryCutoff (T);
  const double L_cg = std::round (core::Lerp (8.0, 32.0, T));
  const double alpha_cg
      = (L_cg > 0.0) ? (constants::kTwo / (L_cg + 1.0)) : 1.0;
  const double serial_mult
      = std::max (0.0,
                  context.GetSerialPositionMultiplier ().value_or (1.0));

  int64_t update_count = 0;

  const auto &events = context.GetMemoryUsageEvents ();
  for (const auto &e : events)
    {
      const double used_flag = e.used ? 1.0 : 0.0;
      const double cg_event = core::Clamp (
          e.contextual_gain.value_or (constants::kNormalizedMin), -1.0, 1.0);
      const long long id = static_cast<long long> (e.embedding_id);
      const long long ts
          = static_cast<long long> (context.GetSignal ().timestamp);

      auto rows = tx.Execute (
          "SELECT strength, use_frequency, contextual_gain, retrieved_count, "
          "       used_count, last_access, created_at, flashbulb, "
          "       half_life_bonus, trace_fast, trace_med, trace_slow, trace_ultra "
          "FROM memories WHERE embedding_id = ?",
          { id });
      if (rows.empty ())
        {
          continue;
        }

      const double strength_prev
          = std::any_cast<double> (rows[0].at ("strength"));
      const double use_freq_prev
          = std::any_cast<double> (rows[0].at ("use_frequency"));
      const double contextual_gain_prev
          = std::any_cast<double> (rows[0].at ("contextual_gain"));
      auto get_double = [] (const std::any &v, double def) -> double {
        if (v.type () == typeid (double))
          return std::any_cast<double> (v);
        if (v.type () == typeid (float))
          return static_cast<double> (std::any_cast<float> (v));
        if (v.type () == typeid (int))
          return static_cast<double> (std::any_cast<int> (v));
        if (v.type () == typeid (long long))
          return static_cast<double> (std::any_cast<long long> (v));
        return def;
      };
      auto get_int = [] (const std::any &v, long long def) -> long long {
        if (v.type () == typeid (long long))
          return std::any_cast<long long> (v);
        if (v.type () == typeid (int))
          return static_cast<long long> (std::any_cast<int> (v));
        if (v.type () == typeid (double))
          return static_cast<long long> (std::any_cast<double> (v));
        return def;
      };
      const long long retrieved_prev
          = std::any_cast<long long> (rows[0].at ("retrieved_count"));
      const long long used_prev
          = std::any_cast<long long> (rows[0].at ("used_count"));
      const auto &last_access_any = rows[0].at ("last_access");
      const bool has_last_access
          = (last_access_any.type () != typeid (std::nullptr_t));
      const long long last_access_prev
          = has_last_access ? std::any_cast<long long> (last_access_any) : 0LL;
      const long long created_at
          = std::any_cast<long long> (rows[0].at ("created_at"));
      const int flashbulb
          = get_int (rows[0].at ("flashbulb"), 0) != 0 ? 1 : 0;
      const double half_life_bonus_raw
          = get_double (rows[0].at ("half_life_bonus"), 0.0);
      const double trace_fast_prev
          = get_double (rows[0].at ("trace_fast"), strength_prev);
      const double trace_med_prev
          = get_double (rows[0].at ("trace_med"), strength_prev * 0.5);
      const double trace_slow_prev
          = get_double (rows[0].at ("trace_slow"), strength_prev * 0.2);
      const double trace_ultra_prev
          = get_double (rows[0].at ("trace_ultra"), strength_prev * 0.05);

      const long long access_base
          = has_last_access ? last_access_prev : created_at;
      const double delta_seconds = std::max (
          0.0, static_cast<double> (ts - access_base) / 1000.0);
      const double half_life_bonus
          = (flashbulb != 0)
                ? std::max (1.0, half_life_bonus_raw)
                : 1.0;
      const double memory_half_life
          = std::max (half_life * half_life_bonus, constants::kNormEpsilon);
      const double use_frequency
          = core::Clamp (core::Ewma (use_freq_prev, used_flag, alpha),
                         constants::kNormalizedMin,
                         constants::kNormalizedMax);
      const double cg_sample = e.used ? cg_event : 0.0;
      const double contextual_gain
          = core::Clamp (core::Ewma (contextual_gain_prev, cg_sample, alpha_cg),
                         -1.0, 1.0);

      const long long retrieved_new = retrieved_prev + 1;
      const long long used_new = used_prev + static_cast<long long> (used_flag);
      const double influence_factor
          = (retrieved_new > 0)
                ? (static_cast<double> (used_new)
                   / static_cast<double> (std::max (retrieved_new, 1LL)))
                      * contextual_gain
                : 0.0;

      const int n_traces = 2 + static_cast<int> (std::round (2.0 * T));
      const double tau_fast = 0.10 * memory_half_life;
      const double tau_med = 0.50 * memory_half_life;
      const double tau_slow = 2.00 * memory_half_life;
      const double tau_ultra = 8.00 * memory_half_life;
      const double taus[4] = { tau_fast, tau_med, tau_slow, tau_ultra };
      double traces[4] = { trace_fast_prev, trace_med_prev,
                           trace_slow_prev, trace_ultra_prev };

      for (int i = 0; i < 4; ++i)
        {
          const double tau = std::max (taus[i], constants::kNormEpsilon);
          const double decay_i
              = std::exp (-std::log (constants::kTwo) / tau
                          * std::max (0.0, delta_seconds));
          const double increment = (alpha * used_flag) / std::max (1, n_traces);
          traces[i] = core::Clamp (traces[i] * decay_i + increment,
                                   constants::kNormalizedMin,
                                   constants::kNormalizedMax);
        }

      const double coupling = 0.05 + 0.10 * T;
      traces[1] = core::Clamp (traces[1] + coupling * traces[0],
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);
      traces[2] = core::Clamp (traces[2] + coupling * traces[1],
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);
      traces[3] = core::Clamp (traces[3] + coupling * traces[2],
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);

      const double boost
          = core::Clamp ((S_eff * used_flag + F_eff * influence_factor) * serial_mult,
                         constants::kNormalizedMin,
                         constants::kNormalizedMax);
      traces[0] = core::Clamp (traces[0] + 0.6 * boost,
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);
      traces[1] = core::Clamp (traces[1] + 0.3 * boost,
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);
      traces[2] = core::Clamp (traces[2] + 0.1 * boost,
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);

      const double w_raw[4] = { 0.55 - 0.20 * T, 0.25,
                                0.15 + 0.10 * T, 0.05 + 0.10 * T };
      double w_sum = 0.0;
      for (int i = 0; i < 4; ++i)
        {
          if (i < n_traces)
            w_sum += w_raw[i];
        }
      if (w_sum <= constants::kNormEpsilon)
        w_sum = 1.0;
      double strength = 0.0;
      for (int i = 0; i < 4; ++i)
        {
          if (i < n_traces)
            {
              strength += (w_raw[i] / w_sum) * traces[i];
            }
        }
      strength = core::Clamp (strength, constants::kNormalizedMin,
                              constants::kNormalizedMax);

      tx.Execute (
          "UPDATE memories "
          "SET retrieved_count = ?, "
          "    used_count = ?, "
          "    last_used = CASE WHEN ? > 0 THEN ? ELSE last_used END, "
          "    contextual_gain = ?, "
          "    influence_factor = ?, "
          "    use_frequency = ?, "
          "    strength = ?, "
          "    trace_fast = ?, "
          "    trace_med = ?, "
          "    trace_slow = ?, "
          "    trace_ultra = ?, "
          "    last_access = ? "
          "WHERE embedding_id = ?",
          { retrieved_new, used_new, used_flag, ts, contextual_gain,
            influence_factor, use_frequency, strength,
            traces[0], traces[1], traces[2], traces[3], ts, id });

      ++update_count;
    }

  // Evict weak memories below periphery cutoff (v2: memories table)
  // Also delete corresponding embeddings.
  tx.Execute (
      "DELETE FROM embeddings WHERE embedding_id IN "
      "(SELECT embedding_id FROM memories WHERE strength < ?)",
      { cutoff });
  auto eviction_result
      = tx.Execute ("DELETE FROM memories WHERE strength < ?", { cutoff });
  const int64_t eviction_count = eviction_result.size ();

  telemetry::LogDebug ("cortext.memory_strength",
                       { telemetry::Attribute::Int64 ("update_count",
                                                      update_count),
                         telemetry::Attribute::Int64 ("eviction_count",
                                                      eviction_count) });
}

} // namespace cortext::operations
