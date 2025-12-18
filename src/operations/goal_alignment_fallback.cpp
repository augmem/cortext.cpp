#include "cortext/operations/goal_alignment_fallback.hpp"

#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
ComputeGoalAlignmentFallback::Execute (OperationContext &context, Transaction &tx) const
{
  const bool fallback_used = !context.GetMetric (operations::Metric::goal_alignment).has_value();
  if (!fallback_used)
    {
      telemetry::LogDebug("cortext.goal_alignment_fallback", {
        telemetry::Attribute::Bool("fallback_used", false),
        telemetry::Attribute::Double("alignment_score", 0.0)
      });
      return;
    }
  auto &p_ctx = context.GetProcessorContext ();
  if (!p_ctx.centroids.has_value ())
    {
      telemetry::LogDebug("cortext.goal_alignment_fallback", {
        telemetry::Attribute::Bool("fallback_used", false),
        telemetry::Attribute::Double("alignment_score", 0.0)
      });
      return;
    }
  const auto &x = context.GetSignal ().embedding;
  if (x.size () != 256)
    {
      telemetry::LogDebug("cortext.goal_alignment_fallback", {
        telemetry::Attribute::Bool("fallback_used", false),
        telemetry::Attribute::Double("alignment_score", 0.0)
      });
      return;
    }
  const float ga01 = p_ctx.centroids->affect.ComputeGoalAlignment (x);
  context.SetMetric (operations::Metric::goal_alignment,
                     static_cast<double> (ga01));

  telemetry::LogDebug("cortext.goal_alignment_fallback", {
    telemetry::Attribute::Bool("fallback_used", true),
    telemetry::Attribute::Double("alignment_score", static_cast<double>(ga01))
  });
}

} // namespace cortext::operations


