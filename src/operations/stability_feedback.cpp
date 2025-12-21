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
ApplyStabilityFeedback::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const double gamma_T = constants::kGammaTBase;

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
      auto rows = tx.Execute (
          "SELECT stability FROM memories WHERE embedding_id = ?",
          { static_cast<long long> (e.embedding_id) });
      if (rows.empty ())
        {
          continue;
        }

      const double stability_prev
          = std::any_cast<double> (rows[0].at ("stability"));
      const double stability_new = (cg > 0.0)
                                       ? (stability_prev + gamma_T)
                                       : (stability_prev * (1.0 - gamma_T));
      const double stability_clamped
          = core::Clamp (stability_new, 0.0, 2.0);

      tx.Execute ("UPDATE memories SET stability = ? WHERE embedding_id = ?",
                  { stability_clamped,
                    static_cast<long long> (e.embedding_id) });
      stability_factors.push_back (stability_clamped);
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
