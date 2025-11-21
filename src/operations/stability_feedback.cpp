#include "cortext/operations/stability_feedback.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include <numeric>
#include <vector>

namespace cortext::operations
{

void
ApplyStabilityFeedback::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  const double T = core::Clamp (cfg.stability, constants::kNormalizedMin,
                                constants::kNormalizedMax);
  const double gamma_T = constants::kGammaTBase * T;

  const auto &events = context.GetMemoryUsageEvents ();
  std::vector<double> stability_factors;
  stability_factors.reserve (events.size ());

  for (const auto &e : events)
    {
      if (!e.used || !e.contextual_gain.has_value ())
        {
          continue;
        }
      const double cg = *e.contextual_gain; // may be negative
      if (cg > 0.0)
        {
          stability_factors.push_back (1.0 + gamma_T);
        }
      else
        {
          stability_factors.push_back (1.0 - gamma_T);
        }
    }

  if (stability_factors.empty ())
    {
      return;
    }

  const double sum = std::accumulate (stability_factors.begin (),
                                      stability_factors.end (), 0.0);
  const double mean = sum / static_cast<double> (stability_factors.size ());

  // adj ← clamp(mean(stability(m_used)) − 1, −0.25, +0.25)
  const double adj
      = core::Clamp (mean - constants::kNormalizedMax,
                     constants::kHalfLifeAdjClampMin,
                     constants::kHalfLifeAdjClampMax);
  // Emit ΔHalfLife_adj_t for Algorithm 6 to consume
  context.SetDeltaHalfLifeAdjustment (adj);
}

} // namespace cortext::operations
