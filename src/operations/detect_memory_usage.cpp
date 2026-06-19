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
#include <unordered_map>
#include <unordered_set>

namespace cortext::operations
{

namespace
{
struct ReinforcementCandidate
{
  long long memory_id = 0;
  long long embedding_id = 0;
  double contextual_support = 0.0;
  bool used = false;
};

/// @brief Creates reinforcement edges for co-retrieved memories.
double
CreateReinforcementEdges (Transaction &tx,
                          const std::vector<ReinforcementCandidate> &candidates,
                          long long now_ts,
                          const SignalProcessor::Config &cfg,
                          ProcessorContext::AssociationFanoutCache &cache)
{
  if (candidates.size () < 2)
    {
      return 0.0;
    }

  const double base_step = core::ReinforcementCoRetrievalStep (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double unselected_scale = core::ReinforcementUnselectedScale (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double fanout_scale = core::ReinforcementFanoutDamping (
      cfg.focus, cfg.sensitivity, cfg.stability,
      static_cast<int> (candidates.size ()));

  double step_sum = 0.0;
  int step_count = 0;
  std::unordered_map<long long, long long> embedding_by_memory_id;
  embedding_by_memory_id.reserve (candidates.size ());
  for (const auto &candidate : candidates)
    {
      if (candidate.memory_id > 0 && candidate.embedding_id > 0)
        {
          embedding_by_memory_id[candidate.memory_id]
              = candidate.embedding_id;
        }
    }
  std::unordered_set<long long> touched_sources;
  std::unordered_set<long long> touched_targets;
  auto upsert_cache_edge =
      [&] (long long source_id, long long target_id, double step) {
    if (!cache.valid || source_id <= 0 || target_id <= 0 || step <= 0.0)
      {
        return;
      }
    const long long source_embedding_id
        = embedding_by_memory_id[source_id];
    const long long target_embedding_id
        = embedding_by_memory_id[target_id];
    auto update_edges =
        [&] (auto &edges, long long neighbor_id,
             long long neighbor_embedding_id) {
      auto edge_it = std::find_if (
          edges.begin (), edges.end (), [&] (const auto &edge) {
        return edge.memory_id == neighbor_id
               && edge.edge_type == "reinforces";
      });
      if (edge_it == edges.end ())
        {
          edges.push_back ({ neighbor_id, neighbor_embedding_id,
                            "reinforces", core::Clamp (step, 0.0, 1.0),
                            now_ts });
          return;
        }
      edge_it->weight = std::min (edge_it->weight + step, 1.0);
      edge_it->last_reinforced = now_ts;
      edge_it->embedding_id = neighbor_embedding_id;
    };
    update_edges (cache.out_by_source[source_id], target_id,
                  target_embedding_id);
    update_edges (cache.in_by_target[target_id], source_id,
                  source_embedding_id);
    touched_sources.insert (source_id);
    touched_targets.insert (target_id);
  };
  for (size_t i = 0; i < candidates.size (); ++i)
    {
      for (size_t j = i + 1; j < candidates.size (); ++j)
        {
          long long id1 = std::min (candidates[i].memory_id,
                                    candidates[j].memory_id);
          long long id2 = std::max (candidates[i].memory_id,
                                    candidates[j].memory_id);
          const double support = std::sqrt (
              core::Clamp (candidates[i].contextual_support, 0.0, 1.0)
              * core::Clamp (candidates[j].contextual_support, 0.0, 1.0));
          const bool anchored = candidates[i].used || candidates[j].used;
          const double pair_scale = anchored ? 1.0 : unselected_scale;
          const double pair_step = core::Clamp (
              base_step * fanout_scale * support * pair_scale, 0.0, 1.0);
          if (pair_step <= 0.0)
            {
              continue;
            }

          tx.Execute (
              "INSERT INTO associations "
              "(source_memory_id, target_memory_id, edge_type, weight, last_reinforced) "
              "VALUES (?1, ?2, 'reinforces', ?3, ?4) "
              "ON CONFLICT (source_memory_id, target_memory_id, edge_type) DO UPDATE "
              "SET weight = MIN(weight + excluded.weight, 1.0), "
              "    last_reinforced = excluded.last_reinforced",
              { id1, id2, pair_step, now_ts });
          upsert_cache_edge (id1, id2, pair_step);
          step_sum += pair_step;
          ++step_count;
        }
    }
  auto sort_edges = [] (auto &edges) {
    std::sort (edges.begin (), edges.end (), [] (const auto &a,
                                                 const auto &b) {
      if (a.weight != b.weight)
        {
          return a.weight > b.weight;
        }
      if (a.last_reinforced != b.last_reinforced)
        {
          return a.last_reinforced > b.last_reinforced;
        }
      return a.memory_id < b.memory_id;
    });
  };
  for (const long long source_id : touched_sources)
    {
      sort_edges (cache.out_by_source[source_id]);
    }
  for (const long long target_id : touched_targets)
    {
      sort_edges (cache.in_by_target[target_id]);
    }
  return step_count > 0 ? step_sum / static_cast<double> (step_count) : 0.0;
}

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
  const auto &cfg = context.GetConfig ();
  Store *store = context.GetStore ();
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

  // Set events in context for downstream feedback operations
  context.SetMemoryUsageEvents (std::move (events));

  int64_t reinforcement_candidate_count = 0;
  double reinforcement_mean_step = 0.0;
  if (cfg.reinforcement_enabled && store)
    {
      std::vector<ReinforcementCandidate> reinforcement_candidates;
      reinforcement_candidates.reserve (retrieved.size ());
      std::unordered_set<long long> seen_memory_ids;
      seen_memory_ids.reserve (retrieved.size ());
      for (const auto &event : context.GetMemoryUsageEvents ())
        {
          const long long embedding_id = event.embedding_id;
          const long long memory_id = event.memory_id;
          if (memory_id > 0 && seen_memory_ids.insert (memory_id).second)
            {
              double contextual_support
                  = core::ReinforcementFallbackContextualSupport (
                      cfg.focus, cfg.sensitivity, cfg.stability);
              if (event.contextual_gain.has_value ())
                {
                  contextual_support = core::Clamp (*event.contextual_gain,
                                                    0.0, 1.0);
                }
              reinforcement_candidates.push_back (
                  { memory_id,
                    embedding_id,
                    contextual_support,
                    event.used });
            }
        }
      reinforcement_candidate_count
          = static_cast<int64_t> (reinforcement_candidates.size ());
      reinforcement_mean_step = CreateReinforcementEdges (
          tx, reinforcement_candidates, static_cast<long long> (signal.timestamp),
          cfg, p_ctx.association_fanout_cache);
    }

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
    telemetry::Attribute::Int64 ("usage_count", static_cast<int64_t> (used_count)),
    telemetry::Attribute::Int64 ("reinforcement_candidate_count",
                                 reinforcement_candidate_count),
    telemetry::Attribute::Double ("reinforcement_mean_step",
                                  reinforcement_mean_step)
  });
}

} // namespace cortext::operations
