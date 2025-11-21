#include "cortext/operations/consolidation_gate.hpp"
#include "cortext/operations/consolidation.hpp"
#include "cortext/processor/operation_context.hpp"

namespace cortext::operations
{

void
ConsolidationGate::Execute (OperationContext &context) const
{
  if (!context.GetConsolidationShouldStart ())
    {
      return;
    }
  // Run scoring and enqueue extraction jobs when start signal is present.
  ScoreConsolidation scorer;
  scorer.Execute (context);

  EnqueueExtractionJobs jobs;
  jobs.Execute (context);
}

} // namespace cortext::operations
