#include "cortext/operations/effective_focus.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"

namespace cortext::operations
{

void
ComputeEffectiveFocus::Execute (OperationContext &context, Transaction &tx) const
{
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  // If coherence not yet computed, default to 1.0 (no reduction)
  const double coherence
      = context.GetCoherence () > constants::kNormalizedMin
            ? context.GetCoherence ()
            : constants::kNormalizedMax;
  const double scale
      = constants::kOneHalf
        + constants::kOneHalf
              * core::Clamp (coherence, constants::kNormalizedMin,
                             constants::kNormalizedMax);
  // Algorithm 11 modifier: F_eff ← F_eff × (1 − focus_spread)
  double spread = constants::kNormalizedMin;
  if (auto v = context.GetMetric (operations::Metric::focus_spread))
    {
      spread
          = core::Clamp (*v, constants::kNormalizedMin, constants::kNormalizedMax);
    }
  const double f_eff
      = core::Clamp (F * scale * (1.0 - spread), constants::kNormalizedMin,
                     constants::kNormalizedMax);
  context.SetEffectiveFocus (f_eff);
}

} // namespace cortext::operations
