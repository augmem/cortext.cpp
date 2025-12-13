#include "cortext/operations/consolidation_gate.hpp"
#include "cortext/operations/consolidation.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"

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

void
ConsolidationGate::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  // Forward schema collection to dynamically instantiated operations.
  // This ensures their table migrations (extraction_jobs, consolidation_candidates)
  // are applied during SignalProcessor initialization.
  ScoreConsolidation scorer;
  scorer.CollectSchema (registry);

  EnqueueExtractionJobs jobs;
  jobs.CollectSchema (registry);
}

} // namespace cortext::operations
