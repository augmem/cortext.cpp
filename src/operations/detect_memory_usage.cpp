#include "cortext/operations/detect_memory_usage.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
DetectMemoryUsage::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();
  const auto &config = context.GetConfig ();

  // Skip if no cached retrievals
  if (p_ctx.recent_retrievals_cache.empty ())
    {
      telemetry::AddCounter ("cortext.detect_memory_usage.no_cache_total", 1);
      return;
    }

  // Skip if no embedding in current signal
  if (signal.embedding.size () == 0)
    {
      telemetry::AddCounter ("cortext.detect_memory_usage.no_embedding_total",
                             1);
      return;
    }

  // Get knob-derived parameters
  const double usage_threshold = core::MemoryUsageThreshold (config.focus);
  const double cache_duration = core::MemoryUsageCacheDuration (config.stability);
  const uint64_t now_ts = signal.timestamp;

  std::vector<OperationContext::MemoryUsageEvent> events;
  int used_count = 0;
  int total_checked = 0;

  for (const auto &cached : p_ctx.recent_retrievals_cache)
    {
      // Skip stale entries based on stability-derived cache duration
      const double age_seconds
          = static_cast<double> (now_ts - cached.retrieved_at);
      if (age_seconds > cache_duration)
        {
          continue;
        }

      // Skip if embedding dimensions don't match
      if (cached.embedding.size () != signal.embedding.size ())
        {
          continue;
        }

      ++total_checked;

      // Compute semantic similarity
      const double sim
          = core::CosineSimilarity (signal.embedding, cached.embedding);

      // Memory is considered "used" if similarity exceeds threshold
      const bool used = (sim >= usage_threshold);

      events.push_back ({ cached.embedding_id, used,
                          used ? std::make_optional (sim) : std::nullopt });

      if (used)
        {
          ++used_count;
        }
    }

  // Set events in context for downstream feedback operations
  context.SetMemoryUsageEvents (std::move (events));

  // Telemetry for observability
  telemetry::AddCounter ("cortext.detect_memory_usage.signals_processed_total",
                         1);
  telemetry::RecordHistogram ("cortext.detect_memory_usage.checked_count",
                              static_cast<double> (total_checked));
  telemetry::RecordHistogram ("cortext.detect_memory_usage.used_count",
                              static_cast<double> (used_count));
  telemetry::RecordHistogram ("cortext.detect_memory_usage.usage_threshold",
                              usage_threshold);
  if (total_checked > 0)
    {
      const double usage_rate
          = static_cast<double> (used_count) / static_cast<double> (total_checked);
      telemetry::RecordHistogram ("cortext.detect_memory_usage.usage_rate",
                                  usage_rate);
    }
}

} // namespace cortext::operations
