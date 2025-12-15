#include "cortext/operations/write_gate.hpp"

#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
ComputeWriteGate::Execute (OperationContext &context) const
{
  // Get composite score (defaults to 0.0 if not computed)
  const auto composite_opt = context.GetCompositeScore ();
  const double composite_score = composite_opt.value_or (0.0);

  // Get threshold and hysteresis from Algorithm 8
  const double T_dynamic = context.GetThresholdTDynamic ();
  const double hysteresis = context.GetThresholdHysteresis ();

  // Write gate decision: composite_score > (T_dynamic - hysteresis)
  const double effective_threshold = T_dynamic - hysteresis;
  const bool write_decision = composite_score > effective_threshold;

  context.SetWriteDecision (write_decision);

  // Telemetry for observability
  telemetry::RecordHistogram ("cortext.write_gate.effective_threshold",
                              effective_threshold);
  telemetry::RecordHistogram ("cortext.write_gate.composite_score",
                              composite_score);
  if (write_decision)
    {
      telemetry::AddCounter ("cortext.write_gate.accept_total", 1);
    }
  else
    {
      telemetry::AddCounter ("cortext.write_gate.reject_total", 1);
    }
}

} // namespace cortext::operations
