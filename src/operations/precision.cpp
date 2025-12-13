#include "cortext/operations/precision.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"

namespace cortext::operations
{

void
UpdatePrecisionDelta::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  const double F = core::Clamp (cfg.focus, constants::kNormalizedMin,
                                constants::kNormalizedMax);
  const double T = core::Clamp (cfg.stability, constants::kNormalizedMin,
                                constants::kNormalizedMax);

  const auto &events = context.GetMemoryUsageEvents ();
  const int retrieved = static_cast<int> (events.size ());
  int used = 0;
  for (const auto &e : events)
    {
      if (e.used)
        {
          used += 1;
        }
    }

  const double retrieval_precision
      = (retrieved > 0) ? (static_cast<double> (used)
                           / static_cast<double> (retrieved))
                        : constants::kNormalizedMax;
  // Keep the computation local to avoid relying on optional helpers in
  // compilation units that may not be indexed by all tooling.
  const double target_precision = core::Clamp (
      core::CertaintyRequirement (T)
          * (constants::kTargetPrecisionBaseScale
             + constants::kTargetPrecisionBaseScale * F),
      constants::kNormalizedMin, constants::kNormalizedMax);

  // κ = 0.10 * (1 - T) (algorithms.md §5.5), using shared medium gain constant.
  const double kappa = constants::kGainMedium * (constants::kNormalizedMax - T);

  double delta = constants::kNormalizedMin;
  if (retrieval_precision < target_precision)
    {
      delta = -kappa * (target_precision - retrieval_precision);
    }
  context.SetDeltaThresholdPrecision (delta);
}

} // namespace cortext::operations
