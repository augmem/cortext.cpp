#include "cortext/operations/goal_alignment_fallback.hpp"

#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"

namespace cortext::operations
{

void
ComputeGoalAlignmentFallback::Execute (OperationContext &context, Transaction &tx) const
{
  if (context.GetMetric (operations::Metric::goal_alignment).has_value ())
    {
      return;
    }
  auto &p_ctx = context.GetProcessorContext ();
  if (!p_ctx.centroids.has_value ())
    {
      return;
    }
  const auto &x = context.GetSignal ().embedding;
  if (x.size () != 256)
    {
      return;
    }
  const float ga01 = p_ctx.centroids->affect.ComputeGoalAlignment (x);
  context.SetMetric (operations::Metric::goal_alignment,
                     static_cast<double> (ga01));
}

} // namespace cortext::operations


