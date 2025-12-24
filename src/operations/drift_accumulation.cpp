#include "cortext/operations/drift_accumulation.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
UpdateDriftAccumulation::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();

  auto it = p_ctx.accumulator_states.find (signal.source_id);
  if (it == p_ctx.accumulator_states.end ())
    {
      p_ctx.accumulator_states[signal.source_id] = AccumulatorState{};
      it = p_ctx.accumulator_states.find (signal.source_id);
    }
  auto &acc = it->second;

  const Eigen::VectorXf *x_ptr = &signal.embedding;
  if (acc.mu_acc.size () > 0)
    {
      x_ptr = &acc.mu_acc;
    }

  // If no embedding or empty, skip
  if (x_ptr->size () == 0)
    {
      return;
    }

  // First step: initialize prev_x, no drift to add
  if (acc.prev_x.size () == 0)
    {
      acc.prev_x = *x_ptr;
      acc.drift_accum = 0.0;
      context.SetDriftAccumSnapshot (0.0);
      return;
    }

  // Verify dimension match
  if (acc.prev_x.size () != x_ptr->size ())
    {
      // Dimension mismatch: reset tracking
      acc.prev_x = *x_ptr;
      acc.drift_accum = 0.0;
      acc.drift_at_last_interrupt = 0.0;
      context.SetDriftAccumSnapshot (0.0);
      return;
    }

  // Compute cosine distance between current and previous accumulator centroids
  const double sim = core::CosineSimilarity (*x_ptr, acc.prev_x);
  const double dist = 1.0 - sim;
  acc.drift_accum += dist;
  acc.prev_x = *x_ptr;
  context.SetDriftAccumSnapshot (acc.drift_accum);

  telemetry::LogDebug("cortext.drift_accumulation", {
    telemetry::Attribute::Double("cos_sim", sim),
    telemetry::Attribute::Double("drift_increment", dist),
    telemetry::Attribute::Double("drift_accum", acc.drift_accum)
  });
}

} // namespace cortext::operations
