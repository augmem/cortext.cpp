#include "cortext/operations/effective_focus.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
ComputeEffectiveFocus::Execute (OperationContext &context, Transaction &tx) const
{
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  // Spec (§3.1.1, line 387): F_eff = F × (0.5 + 0.5 × coherence_struct_t)
  // Use structural coherence for effective focus modulation
  const double coherence_struct = context.GetStructuralCoherence ();
  const double scale
      = constants::kOneHalf
        + constants::kOneHalf
              * core::Clamp (coherence_struct, constants::kNormalizedMin,
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

  telemetry::LogDebug("cortext.compute_effective_focus", {
    telemetry::Attribute::Double("F", F),
    telemetry::Attribute::Double("coherence_struct", coherence_struct),
    telemetry::Attribute::Double("scale", scale),
    telemetry::Attribute::Double("focus_spread", spread),
    telemetry::Attribute::Double("F_eff", f_eff)
  });
}

} // namespace cortext::operations
