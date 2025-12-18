#include "cortext/operations/consolidation_gate.hpp"
#include "cortext/operations/consolidation.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"
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
  // Run scoring and enqueue extraction jobs when start signal is present.
  ScoreConsolidation scorer;
  scorer.Execute (context, tx);

  EnqueueExtractionJobs jobs;
  jobs.Execute (context, tx);

  telemetry::LogDebug("cortext.consolidation_gate", {
    telemetry::Attribute::Bool("gate_open", true)
  });
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
