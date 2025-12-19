#include "cortext/operations/drift_accumulation.hpp"

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

  // If no embedding or empty, skip
  if (signal.embedding.size () == 0)
    {
      return;
    }

  // First signal: initialize last_pacing_check_embedding, no drift to add
  if (!p_ctx.last_pacing_check_embedding.has_value ())
    {
      p_ctx.last_pacing_check_embedding = signal.embedding;
      p_ctx.drift_accum = 0.0;
      context.SetDriftAccumSnapshot (0.0);
      return;
    }

  // Verify dimension match
  const Eigen::VectorXf &last = *p_ctx.last_pacing_check_embedding;
  if (last.size () != signal.embedding.size ())
    {
      // Dimension mismatch: reset tracking
      p_ctx.last_pacing_check_embedding = signal.embedding;
      p_ctx.drift_accum = 0.0;
      context.SetDriftAccumSnapshot (0.0);
      return;
    }

  // Compute Euclidean distance (L2 norm) between current and last check
  const double dist = (signal.embedding - last).norm ();
  p_ctx.drift_accum += dist;
  context.SetDriftAccumSnapshot (p_ctx.drift_accum);

  telemetry::LogDebug("cortext.drift_accumulation", {
    telemetry::Attribute::Double("drift_increment", dist),
    telemetry::Attribute::Double("drift_accum", p_ctx.drift_accum)
  });
}

} // namespace cortext::operations
