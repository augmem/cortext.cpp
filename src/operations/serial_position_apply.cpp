#include "cortext/operations/serial_position_apply.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace cortext::operations
{

namespace
{
inline double
Clamp01 (double v)
{
  return core::Clamp (v, 0.0, 1.0);
}
} // namespace

void
ApplySerialPositionMultiplier::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const auto &config = context.GetConfig ();
  const int primacy_window = std::max (0, context.GetSerialPrimacyWindow ());
  const int recency_window = std::max (0, context.GetSerialRecencyWindow ());
  const double primacy_bonus
      = Clamp01 (context.GetSerialPrimacyBonus ()); // for primacy/recency zones

  // Algorithm 26 zone modulation parameters
  const double von_restorff = context.GetSerialVonRestorffMultiplier ();
  const double middle_suppression = context.GetSerialMiddleSuppression ();
  const double middle_floor = core::SerialMiddleMultiplierFloor (
      config.focus, config.sensitivity);
  // Note: distinctiveness_threshold is used internally by SerialDetermineZone

  // Use signal-level rarity as proxy for item distinctiveness
  // (items retrieved for this signal share context with the signal)
  const double signal_rarity
      = context.GetMetric (operations::Metric::rarity).value_or (0.0);

  // Build ordered list of used memory events for this signal.
  std::vector<OperationContext::MemoryUsageEvent> used;
  used.reserve (context.GetMemoryUsageEvents ().size ());
  for (const auto &e : context.GetMemoryUsageEvents ())
    {
      if (e.used)
        {
          used.push_back (e);
        }
    }
  const int K = static_cast<int> (used.size ());
  if (K < 2)
    {
      context.SetSerialPositionMultiplier (1.0);
      telemetry::LogDebug ("cortext.serial_position_apply",
                           { telemetry::Attribute::Double ("position_multiplier",
                                                           1.0) });
      return;
    }

  // Compute per-used multiplier using zone classification per Algorithm 26.
  // Zone determination (algorithms.md lines 977-981):
  // - if position ≤ primacy_window → "primacy"
  // - elif position ≥ N − recency_window + 1 → "recency"
  // - elif rarity > distinctiveness_threshold → "distinctive"
  // - else → "middle"
  double sum_mult = 0.0;
  for (int rank = 0; rank < K; ++rank)
    {
      // Use 1-indexed position for zone determination (rank 0 = position 1)
      const int position = rank + 1;

      // Determine zone using knobs.hpp helper
      const core::SerialZone zone = core::SerialDetermineZone (
          config.focus, position, K, signal_rarity);

      double mult = 1.0;
      switch (zone)
        {
        case core::SerialZone::Primacy:
          {
            // Primacy boost: higher for earlier items
            const double denom
                = std::max (1.0, static_cast<double> (primacy_window - 1));
            const double norm_pos
                = 1.0 - static_cast<double> (rank) / denom;
            mult = 1.0 + primacy_bonus * Clamp01 (norm_pos);
          }
          break;

        case core::SerialZone::Recency:
          {
            // Recency boost: higher for later items
            const int idx_in_zone
                = rank - (K - std::max (1, recency_window));
            const double denom
                = std::max (1.0, static_cast<double> (recency_window - 1));
            const double norm_pos
                = static_cast<double> (std::max (0, idx_in_zone)) / denom;
            mult = 1.0 + primacy_bonus * Clamp01 (norm_pos);
          }
          break;

        case core::SerialZone::Distinctive:
          // von Restorff effect: distinctive items get strength boost
          // von_restorff_multiplier = lerp(1.5, 3.0, S)
          mult = von_restorff;
          break;

        case core::SerialZone::Middle:
          // Middle suppression: reduce salience of interference zone items
          // middle_suppression = lerp(0.8, 0.5, S) × (1 − F)
          // Apply as (1 - suppression) to reduce the multiplier
          mult = std::max (middle_floor, 1.0 - middle_suppression);
          break;
        }

      sum_mult += mult;
    }

  const double avg_mult = sum_mult / static_cast<double> (K);
  context.SetSerialPositionMultiplier (avg_mult);

  telemetry::LogDebug ("cortext.serial_position_apply",
                       { telemetry::Attribute::Double ("position_multiplier",
                                                       avg_mult) });
}

} // namespace cortext::operations
