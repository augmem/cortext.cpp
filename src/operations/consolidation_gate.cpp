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
  // No-op: all tables now in core schema.cpp migration 0.
  // consolidation_candidates is in core schema.
  // extraction_jobs and consolidation_events have been removed (undocumented).
  (void)registry;
}

} // namespace cortext::operations
