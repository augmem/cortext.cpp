#include "cortext/operations/serial_position_apply.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/processor/operation_context.hpp"
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
ApplySerialPositionMultiplier::Execute (OperationContext &context) const
{
  const int primacy_window = std::max (0, context.GetSerialPrimacyWindow ());
  const int recency_window = std::max (0, context.GetSerialRecencyWindow ());
  const double primacy_bonus
      = Clamp01 (context.GetSerialPrimacyBonus ()); // reuse for both ends

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
  if (K < 2 || (primacy_window <= 0 && recency_window <= 0))
    {
      context.SetSerialPositionMultiplier (1.0);
      return;
    }

  // Compute per-used multiplier, then average for a per-signal multiplier.
  double sum_mult = 0.0;
  for (int rank = 0; rank < K; ++rank)
    {
      double mult = 1.0;
      // Primacy zone
      if (primacy_window > 0 && rank < primacy_window)
        {
          const double denom = std::max (1, primacy_window - 1);
          const double norm_pos = 1.0 - static_cast<double> (rank) / denom;
          mult = std::max (mult, 1.0 + primacy_bonus * Clamp01 (norm_pos));
        }
      // Recency zone
      if (recency_window > 0 && rank >= K - recency_window)
        {
          const int idx_in_zone = rank - (K - recency_window);
          const double denom = std::max (1, recency_window - 1);
          const double norm_pos = static_cast<double> (idx_in_zone) / denom;
          mult = std::max (mult, 1.0 + primacy_bonus * Clamp01 (norm_pos));
        }
      sum_mult += mult;
    }
  const double avg_mult = sum_mult / static_cast<double> (K);
  context.SetSerialPositionMultiplier (avg_mult);
}

} // namespace cortext::operations
