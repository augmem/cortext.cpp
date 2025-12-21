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

  const double F = cfg.focus;
  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const double alpha = core::AlphaS (S, p_ctx.u_t);
  const double half_life
      = (p_ctx.half_life > constants::kNormEpsilon)
            ? p_ctx.half_life
            : core::BaseHalfLifePrior (T);
  const double lambda
      = std::log (constants::kTwo) / std::max (half_life, constants::kNormEpsilon);
  const double cutoff = core::PeripheryCutoff (T);
  const double L_cg = std::round (core::Lerp (8.0, 32.0, T));
  const double alpha_cg
      = (L_cg > 0.0) ? (constants::kTwo / (L_cg + 1.0)) : 1.0;

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
          "       used_count, last_access, created_at "
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

      const long long access_base
          = has_last_access ? last_access_prev : created_at;
      const double delta_seconds = std::max (
          0.0, static_cast<double> (ts - access_base) / 1000.0);
      const double decay
          = std::exp (-lambda * std::max (0.0, delta_seconds));

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

      const double strength
          = core::Clamp (strength_prev * decay + S * use_frequency
                             + F * influence_factor,
                         constants::kNormalizedMin,
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
          "    last_access = ? "
          "WHERE embedding_id = ?",
          { retrieved_new, used_new, used_flag, ts, contextual_gain,
            influence_factor, use_frequency, strength, ts, id });

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
