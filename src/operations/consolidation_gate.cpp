#include "cortext/internal/cancellation.hpp"
#include "cortext/operations/consolidation.hpp"
#include "cortext/operations/consolidation_gate.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
ConsolidationGate::Execute (OperationContext &context, Transaction &tx) const
{
  const bool gate_open = context.GetConsolidationShouldStart();
  if (!gate_open)
    {
      telemetry::LogDebug("cortext.consolidation_gate", {
        telemetry::Attribute::Bool("gate_open", false)
      });
      return;
    }
  internal::ThrowIfStopRequested ();
  // Run embedding-only consolidation scoring when start signal is present.
  ScoreConsolidation scorer;
  scorer.Execute (context, tx);

  telemetry::LogDebug("cortext.consolidation_gate", {
    telemetry::Attribute::Bool("gate_open", true)
  });
}

} // namespace cortext::operations
