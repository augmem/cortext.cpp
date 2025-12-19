#include "cortext/operations/centroids.hpp"

#include "cortext/data/centroids.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
InitializeEmbeddedCentroids::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  if (p_ctx.centroids.has_value ())
    {
      telemetry::LogDebug("cortext.initialize_embedded_centroids", {
        telemetry::Attribute::Bool("has_centroids", true),
        telemetry::Attribute::Int64("centroid_count", static_cast<int64_t>(p_ctx.centroids->emotion_centroids.size()))
      });
      return;
    }
  p_ctx.centroids = cortext::data::GetEmbeddedCentroids ();

  telemetry::LogDebug("cortext.initialize_embedded_centroids", {
    telemetry::Attribute::Bool("has_centroids", p_ctx.centroids.has_value()),
    telemetry::Attribute::Int64("centroid_count", p_ctx.centroids.has_value() ? static_cast<int64_t>(p_ctx.centroids->emotion_centroids.size()) : 0)
  });
}

} // namespace cortext::operations


