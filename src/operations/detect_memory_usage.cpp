#include "cortext/operations/detect_memory_usage.hpp"

#include "neuromodulator_internal.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/core/sparse.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"

#include <algorithm>
#include <cmath>

namespace cortext::operations
{

namespace
{
long long
ResolveMemoryIdForEmbedding (ProcessorContext &p_ctx, Store *store,
                             long long embedding_id)
{
  auto cache_it = p_ctx.retrieval_surface_embedding_index.find (embedding_id);
  if (cache_it != p_ctx.retrieval_surface_embedding_index.end ())
    {
      const long long memory_id
          = p_ctx.retrieval_surface_cache[cache_it->second].memory_id;
      if (memory_id > 0)
        {
          return memory_id;
        }
    }
  if (!store)
    {
      return 0;
    }

  auto rows = store->Execute (
      "SELECT memory_id FROM memories WHERE embedding_id = ? LIMIT 1",
      { embedding_id });
  if (!rows.empty () && rows[0].count ("memory_id") == 1)
    {
      return store::AnyToLongLong (rows[0].at ("memory_id")).value_or (0);
    }

  auto sig_rows = store->Execute (
      "SELECT memory_id FROM signals WHERE embedding_id = ? LIMIT 1",
      { embedding_id });
  if (sig_rows.empty () || sig_rows[0].count ("memory_id") != 1)
    {
      return 0;
    }

  const long long memory_id
      = store::AnyToLongLong (sig_rows[0].at ("memory_id")).value_or (0);
  if (memory_id <= 0)
    {
      return 0;
    }

  auto mem_exists = store->Execute (
      "SELECT 1 AS present FROM memories WHERE memory_id = ? LIMIT 1",
      { memory_id });
  return mem_exists.empty () ? 0 : memory_id;
}
} // namespace

void
DetectMemoryUsage::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const auto &signal = context.GetSignal ();
  if (signal.retention == Retention::Ephemeral)
    {
      context.SetMemoryUsageEvents ({});
      return;
    }
  auto &p_ctx = context.GetProcessorContext ();
  const Eigen::VectorXf *x_ptr = &signal.embedding;
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it != p_ctx.accumulator_states.end ()
      && acc_it->second.mu_acc.size () > 0)
    {
      x_ptr = &acc_it->second.mu_acc;
    }

  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  const auto &retrieved_records = context.GetRetrievedMemoryCandidates ();
  const auto &cfg = context.GetConfig ();
  Store *store = context.GetStore ();
  const bool interrupt_allowed = context.GetInterruptAllowed ();
  const auto selected_id = context.GetSelectedCandidateId ();

  // Clear events if nothing was retrieved.
  if (retrieved.empty () && retrieved_records.empty ())
    {
      context.SetMemoryUsageEvents ({});
      telemetry::AddCounter ("cortext.detect_memory_usage.no_candidates_total", 1);
      return;
    }

  std::vector<OperationContext::MemoryUsageEvent> events;
  events.reserve (retrieved_records.empty () ? retrieved.size ()
                                             : retrieved_records.size ());

  int used_count = 0;
  int total_checked = 0;

  if (!retrieved_records.empty ())
    {
      for (const auto &candidate : retrieved_records)
        {
          const long long embedding_id = candidate.embedding_id;
          const long long memory_id = candidate.memory_id;
          const Eigen::VectorXf &emb = candidate.embedding;

          ++total_checked;

          const bool used
              = interrupt_allowed && selected_id.has_value ()
                && memory_id > 0 && memory_id == *selected_id;
          std::optional<double> contextual_gain = std::nullopt;
          if (x_ptr->size () > 0 && emb.size () == x_ptr->size ())
            {
              contextual_gain = core::CosineSimilarity (*x_ptr, emb);
            }

          events.push_back (
              { embedding_id, used, contextual_gain, memory_id });

          if (used)
            {
              ++used_count;
            }
        }
    }
  else
    {
      for (const auto &kv : retrieved)
        {
          const long long embedding_id = kv.first;
          const Eigen::VectorXf &emb = kv.second;
          const long long memory_id
              = ResolveMemoryIdForEmbedding (p_ctx, store, embedding_id);

          ++total_checked;

          const bool used
              = interrupt_allowed && selected_id.has_value ()
                    && (embedding_id == *selected_id);
          std::optional<double> contextual_gain = std::nullopt;
          if (x_ptr->size () > 0 && emb.size () == x_ptr->size ())
            {
              contextual_gain = core::CosineSimilarity (*x_ptr, emb);
            }

          events.push_back (
              { embedding_id, used, contextual_gain, memory_id });

          if (used)
            {
              ++used_count;
            }
        }
    }

  // Set events in context for downstream feedback operations
  context.SetMemoryUsageEvents (std::move (events));

  const double usage_rate
      = (total_checked > 0)
            ? static_cast<double> (used_count) / static_cast<double> (total_checked)
            : 0.0;
  p_ctx.last_used_rate = usage_rate;
  p_ctx.last_used_flag = used_count > 0 ? 1.0 : 0.0;

  // Update procedural store for successful usage
  if (cfg.procedural_enabled && used_count > 0 && x_ptr->size () > 0)
    {
      const int k_key = core::SparseKeySize (
          context.GetConfig ().focus, context.GetConfig ().sensitivity,
          context.GetConfig ().stability);
      const std::string key = core::SparseKey (*x_ptr, k_key);
      if (!key.empty ())
        {
          for (const auto &event : context.GetMemoryUsageEvents ())
            {
              if (!event.used || event.memory_id <= 0)
                continue;

              const double gain
                  = neuromodulation::ValueUpdateGain (p_ctx.neuromod_da);
              double &q = p_ctx.procedural_store[key][event.memory_id];
              q = core::Clamp (q + gain * std::max (0.0, p_ctx.delta_reward),
                               0.0, 1.0);
            }
        }
    }

  // Telemetry for observability
  telemetry::AddCounter ("cortext.detect_memory_usage.signals_processed_total",
                         1);
  telemetry::RecordHistogram ("cortext.detect_memory_usage.checked_count",
                              static_cast<double> (total_checked));
  telemetry::RecordHistogram ("cortext.detect_memory_usage.used_count",
                              static_cast<double> (used_count));
  if (total_checked > 0)
    {
      telemetry::RecordHistogram ("cortext.detect_memory_usage.usage_rate",
                                  usage_rate);
    }

  // Debug logging
  telemetry::LogDebug ("cortext.detect_memory_usage", {
    telemetry::Attribute::Int64 ("candidate_count", static_cast<int64_t> (retrieved.size ())),
    telemetry::Attribute::Bool ("interrupt_allowed", interrupt_allowed),
    telemetry::Attribute::Int64 ("usage_count", static_cast<int64_t> (used_count))
  });
}

} // namespace cortext::operations
