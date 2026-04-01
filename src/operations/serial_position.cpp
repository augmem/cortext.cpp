#include "cortext/operations/serial_position.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
ApplySerialPositionEffects::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  const double S = cfg.sensitivity;

  // Compute derived parameters (Algorithm 26).
  const int primacy_window = core::SerialPrimacyWindow (F);
  const int recency_window = core::SerialRecencyWindow (F);
  const double primacy_bonus = core::SerialPrimacyBonus (S);
  const double rehearsal_curve_depth = core::SerialRehearsalCurveDepth (S);
  const double distinctiveness_threshold
      = core::SerialDistinctivenessThreshold (F);
  const double von_restorff_multiplier = core::SerialVonRestorffMultiplier (S);
  const double middle_suppression = core::SerialMiddleSuppression (S, F);

  // Expose via OperationContext for downstream consumers/telemetry.
  context.SetSerialPrimacyWindow (primacy_window);
  context.SetSerialRecencyWindow (recency_window);
  context.SetSerialPrimacyBonus (primacy_bonus);
  context.SetSerialRehearsalCurveDepth (rehearsal_curve_depth);
  context.SetSerialDistinctivenessThreshold (distinctiveness_threshold);
  context.SetSerialVonRestorffMultiplier (von_restorff_multiplier);
  context.SetSerialMiddleSuppression (middle_suppression);

  telemetry::LogDebug ("cortext.serial_position",
                       { telemetry::Attribute::Int64 ("primacy_window",
                                                      primacy_window),
                         telemetry::Attribute::Int64 ("recency_window",
                                                      recency_window) });
}

} // namespace cortext::operations
