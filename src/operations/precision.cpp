#include "cortext/operations/precision.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

namespace
{
// Spec (§4.2.5, line 633): coherence offset
constexpr double kCoherenceOffset = 0.5;
}  // namespace

void
UpdatePrecisionDelta::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();

  const double F = core::Clamp (cfg.focus, constants::kNormalizedMin,
                                constants::kNormalizedMax);

  // Spec (§4.2.5, lines 627-637): Precision-Based Threshold Adjustment
  // Focus-driven precision tightens threshold when structural coherence is high:
  // κ_prec = 0.06
  // cap_prec ← 0.15 × hysteresis_t
  // Δθ_prec ← clamp(κ_prec × F × (coherence_struct_t − 0.5), −cap_prec, +cap_prec)

  const double coherence_struct = context.GetStructuralCoherence ();
  const double cap_prec
      = core::CapPrec (cfg.focus, cfg.sensitivity, cfg.stability,
                       p_ctx.hysteresis);
  const double kappa_prec
      = core::KappaPrec (cfg.focus, cfg.sensitivity, cfg.stability);
  const double raw_delta
      = kappa_prec * F * (coherence_struct - kCoherenceOffset);
  const double delta = core::Clamp (raw_delta, -cap_prec, cap_prec);

  context.SetDeltaThresholdPrecision (delta);

  telemetry::LogDebug("cortext.precision_delta", {
    telemetry::Attribute::Double("F", F),
    telemetry::Attribute::Double("coherence_struct", coherence_struct),
    telemetry::Attribute::Double("hysteresis", p_ctx.hysteresis),
    telemetry::Attribute::Double("cap_prec", cap_prec),
    telemetry::Attribute::Double("raw_delta", raw_delta),
    telemetry::Attribute::Double("delta_prec", delta)
  });
}

} // namespace cortext::operations
