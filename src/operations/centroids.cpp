#include "cortext/operations/centroids.hpp"

#include "cortext/data/centroids.hpp"
#include "cortext/processor/operation_context.hpp"

namespace cortext::operations
{

void
InitializeEmbeddedCentroids::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  if (p_ctx.centroids.has_value ())
    {
      return;
    }
  p_ctx.centroids = cortext::data::GetEmbeddedCentroids ();
}

} // namespace cortext::operations


