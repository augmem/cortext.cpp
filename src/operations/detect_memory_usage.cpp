#include "cortext/operations/detect_memory_usage.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/core/sparse.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "cortext/store/store.hpp"

namespace cortext::operations
{

void
DetectMemoryUsage::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const auto &signal = context.GetSignal ();
  auto &p_ctx = context.GetProcessorContext ();
  const Eigen::VectorXf *x_ptr = &signal.embedding;
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it != p_ctx.accumulator_states.end ()
      && acc_it->second.mu_acc.size () > 0)
    {
      x_ptr = &acc_it->second.mu_acc;
    }

  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  const bool interrupt_allowed = context.GetInterruptAllowed ();
  const auto selected_id = context.GetSelectedCandidateId ();

  // Clear events if nothing was retrieved.
  if (retrieved.empty ())
    {
      context.SetMemoryUsageEvents ({});
      telemetry::AddCounter ("cortext.detect_memory_usage.no_candidates_total", 1);
      return;
    }

  std::vector<OperationContext::MemoryUsageEvent> events;
  events.reserve (retrieved.size ());

  int used_count = 0;
  int total_checked = 0;

  for (const auto &kv : retrieved)
    {
      const long long embedding_id = kv.first;
      const Eigen::VectorXf &emb = kv.second;

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
          { embedding_id, used, contextual_gain });

      if (used)
        {
          ++used_count;
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
  if (used_count > 0 && x_ptr->size () > 0)
    {
      const int k_key = core::SparseKeySize (context.GetConfig ().focus);
      const std::string key = core::SparseKey (*x_ptr, k_key);
      if (!key.empty ())
        {
          for (const auto &kv : retrieved)
            {
              const long long embedding_id = kv.first;
              bool used = false;
              if (selected_id.has_value () && embedding_id == *selected_id)
                used = true;
              if (!used)
                continue;
              // Map embedding_id -> memory_id for procedural store
              Store *store = context.GetStore ();
              long long memory_id = 0;
              if (store)
                {
                  auto rows = store->Execute (
                      "SELECT memory_id FROM memories WHERE embedding_id = ?",
                      { embedding_id });
                  if (!rows.empty () && rows[0].count ("memory_id") == 1
                      && rows[0].at ("memory_id").type () == typeid (long long))
                    {
                      memory_id = std::any_cast<long long> (rows[0].at ("memory_id"));
                    }
                }
              if (memory_id > 0)
                {
                  const double gain = 0.5 + 0.5 * p_ctx.neuromod_da;
                  double &q = p_ctx.procedural_store[key][memory_id];
                  q = core::Clamp (q + gain * std::max (0.0, p_ctx.delta_reward),
                                   0.0, 1.0);
                }
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
