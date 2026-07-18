#include "cortext/operations/emotion_cascade.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "association_fanout_cache_internal.hpp"
#include "emotional_metadata_cache_internal.hpp"
#include "../experimental_env.hpp"
#include <algorithm>
#include <any>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cortext::operations
{

EmotionCascadeParams
EmotionCascadeParams::FromKnobs (double /*F*/, double S, double /*T*/)
{
  EmotionCascadeParams p;
  p.cascade_radius = core::CascadeRadius (S);
  p.cascade_decay = core::CascadeDecay (S);
  return p;
}

namespace
{

/// @brief Adds a write instruction to the transaction.
void
Add (Transaction &tx, const std::string &q,
     const std::vector<std::any> &p = {})
{
  tx.Execute (q, p);
}

/// @brief Structure for emotional source memories.
struct EmotionalSource
{
  long long memory_id;
  long long embedding_id;
  double intensity;
  double arousal;
  double valence;
  double half_life_bonus;
  int cascade_radius;
  double cascade_decay;
};

/// @brief Load recently tagged high-intensity emotional memories.
/// v2: Uses memories table (emotional fields merged from emotional_tags)
std::vector<EmotionalSource>
LoadEmotionalSources (Store *store, long long recent_window_ts,
                      double theta_intensity, double theta_arousal,
                      const EmotionCascadeParams &fallback,
                      std::size_t *accepted_count)
{
  std::vector<EmotionalSource> sources;

  // v2: Query memories for high-intensity flashbulb memories
  // Only process recently tagged ones (within consolidation window)
  auto rows = store->Execute (
      "SELECT source.embedding_id, source.emotional_intensity, "
      "       source.half_life_bonus, source.cascade_radius, "
      "       source.cascade_decay, "
      "       (SELECT resolved.memory_id FROM memories resolved "
      "        WHERE resolved.embedding_id = source.embedding_id "
      "        ORDER BY resolved.memory_id ASC LIMIT 1) "
      "         AS traversal_memory_id "
      "FROM memories source "
      "WHERE flashbulb = 1 AND emotional_intensity >= ?1 "
      "AND s_arousal_avg >= ?2 "
      "AND created_at >= ?3 "
      "ORDER BY emotional_intensity DESC, source.memory_id ASC",
      { theta_intensity, theta_arousal, recent_window_ts });

  for (const auto &row : rows)
    {
      auto it_id = row.find ("embedding_id");
      auto it_intensity = row.find ("emotional_intensity");

      if (it_id == row.end () || it_intensity == row.end ())
        {
          continue;
        }

      if (it_id->second.type () != typeid (long long))
        {
          continue;
        }

      EmotionalSource s;
      const auto it_memory_id = row.find ("traversal_memory_id");
      if (it_memory_id == row.end ()
          || it_memory_id->second.type () != typeid (long long))
        {
          continue;
        }
      s.memory_id = std::any_cast<long long> (it_memory_id->second);
      s.embedding_id = std::any_cast<long long> (it_id->second);

      if (it_intensity->second.type () == typeid (double))
        {
          s.intensity = std::any_cast<double> (it_intensity->second);
        }
      else
        {
          continue;
        }

      // v2: arousal/valence not stored per-memory in v2, use defaults
      s.arousal = 0.5;
      s.valence = 0.5;

      auto it_bonus = row.find ("half_life_bonus");
      if (it_bonus != row.end () && it_bonus->second.type () == typeid (double))
        {
          s.half_life_bonus = std::any_cast<double> (it_bonus->second);
        }
      else
        {
          s.half_life_bonus = 1.0;
        }

      auto it_radius = row.find ("cascade_radius");
      if (it_radius != row.end () && it_radius->second.type () == typeid (long long))
        {
          s.cascade_radius
              = static_cast<int> (std::any_cast<long long> (it_radius->second));
        }
      else
        {
          s.cascade_radius = fallback.cascade_radius;
        }

      auto it_decay = row.find ("cascade_decay");
      if (it_decay != row.end () && it_decay->second.type () == typeid (double))
        {
          s.cascade_decay = std::any_cast<double> (it_decay->second);
        }
      else
        {
          s.cascade_decay = fallback.cascade_decay;
        }

      sources.push_back (s);
      if (accepted_count)
        ++*accepted_count;
    }

  return sources;
}

std::vector<EmotionalSource>
LoadEmotionalSourcesFromCache (const ProcessorContext &p_ctx,
                               long long recent_window_ts,
                               double theta_intensity,
                               double theta_arousal,
                               const EmotionCascadeParams &fallback,
                               std::size_t *accepted_count)
{
  const auto state
      = emotional_metadata_cache_internal::FindState (p_ctx);
  std::vector<EmotionalSource> sources;
  if (!state || !state->emotional_metadata.valid)
    return sources;
  const auto &cache = state->emotional_metadata;
  for (const long long memory_id : cache.source_query_order)
    {
      const auto row_it = cache.rows_by_memory.find (memory_id);
      if (row_it == cache.rows_by_memory.end ())
        continue;
      const auto &row = row_it->second;
      if (!row.flashbulb || row.intensity < theta_intensity
          || row.arousal < theta_arousal
          || row.created_at < recent_window_ts)
        continue;
      const auto members = cache.memory_ids_by_embedding.find (row.embedding_id);
      if (members == cache.memory_ids_by_embedding.end ()
          || members->second.empty ())
        continue;
      sources.push_back (
          { members->second.front (), row.embedding_id, row.intensity, 0.5,
            0.5, row.half_life_bonus,
            row.cascade_radius > 0 ? row.cascade_radius
                                   : fallback.cascade_radius,
            row.cascade_decay > 0.0 ? row.cascade_decay
                                    : fallback.cascade_decay });
      if (accepted_count)
        ++*accepted_count;
    }
  std::stable_sort (sources.begin (), sources.end (),
                    [] (const auto &left, const auto &right) {
                      return left.intensity > right.intensity;
                    });
  return sources;
}

/// @brief Structure for cascade neighbor.
struct CascadeNeighbor
{
  long long embedding_id;
  int depth;
};

using CurrentEmotionalValues
    = execution_cache_sidecar_internal::EmotionalEmbeddingValues;

using AssociationFanoutCache = ProcessorContext::AssociationFanoutCache;

bool
MatchesCascadeFixedPointInputs (const ProcessorContext &p_ctx,
                               long long recent_window_ts,
                               double theta_intensity, double theta_arousal,
                               double intensity_floor,
                               const EmotionCascadeParams &params)
{
  const auto state
      = emotional_metadata_cache_internal::FindState (p_ctx);
  if (!state)
    return false;
  const auto &fixed = state->emotional_fixed_point;
  const auto &metadata = state->emotional_metadata;
  return fixed.valid && metadata.valid
         && fixed.emotional_input_generation
                == metadata.cascade_input_generation
         && recent_window_ts >= fixed.recent_window_ts
         && theta_intensity == fixed.theta_intensity
         && theta_arousal == fixed.theta_arousal
         && intensity_floor == fixed.intensity_floor
         && params.cascade_radius == fixed.cascade_radius
         && params.cascade_decay == fixed.cascade_decay;
}

bool
MatchesCascadeFixedPoint (const ProcessorContext &p_ctx,
                          const AssociationFanoutCache &fanout,
                          long long recent_window_ts,
                          double theta_intensity, double theta_arousal,
                          double intensity_floor,
                          const EmotionCascadeParams &params)
{
  const auto state
      = emotional_metadata_cache_internal::FindState (p_ctx);
  if (!state
      || !MatchesCascadeFixedPointInputs (
          p_ctx, recent_window_ts, theta_intensity, theta_arousal,
          intensity_floor, params))
    return false;
  const auto &fixed = state->emotional_fixed_point;
  return fixed.association_edge_count == fanout.edge_count
         && fixed.association_source_sum == fanout.source_sum
         && fixed.association_target_sum == fanout.target_sum
         && fixed.association_source_max == fanout.source_max
         && fixed.association_target_max == fanout.target_max
         && fixed.association_weight_sum_micros == fanout.weight_sum_micros
         && fixed.association_last_reinforced_sum
                == fanout.last_reinforced_sum;
}

bool
AssociationChangesCanReachCascade (
    const execution_cache_sidecar_internal::State &state)
{
  const auto &changes = state.association_topology_changes;
  const auto &footprint = state.emotional_cascade_topology_footprint;
  if (changes.reset || !footprint.valid)
    return true;
  for (const auto &[source_memory_id, target_memory_id] :
       changes.inserted_edges)
    for (const auto &expandable : footprint.expandable_by_source)
      if (expandable.contains (source_memory_id)
          || expandable.contains (target_memory_id))
        return true;
  return false;
}

void
ConsumeAssociationTopologyChanges (
    execution_cache_sidecar_internal::State &state)
{
  state.association_topology_changes.reset = false;
  state.association_topology_changes.inserted_edges.clear ();
}

void
RecordCascadeFixedPoint (ProcessorContext &p_ctx,
                         const AssociationFanoutCache &fanout,
                         std::uint64_t emotional_input_generation,
                         long long recent_window_ts,
                         double theta_intensity, double theta_arousal,
                         double intensity_floor,
                         const EmotionCascadeParams &params)
{
  const auto state
      = emotional_metadata_cache_internal::EnsureState (p_ctx);
  auto &fixed = state->emotional_fixed_point;
  fixed.valid = true;
  fixed.emotional_input_generation = emotional_input_generation;
  fixed.association_edge_count = fanout.edge_count;
  fixed.association_source_sum = fanout.source_sum;
  fixed.association_target_sum = fanout.target_sum;
  fixed.association_source_max = fanout.source_max;
  fixed.association_target_max = fanout.target_max;
  fixed.association_weight_sum_micros = fanout.weight_sum_micros;
  fixed.association_last_reinforced_sum = fanout.last_reinforced_sum;
  fixed.recent_window_ts = recent_window_ts;
  fixed.theta_intensity = theta_intensity;
  fixed.theta_arousal = theta_arousal;
  fixed.intensity_floor = intensity_floor;
  fixed.cascade_radius = params.cascade_radius;
  fixed.cascade_decay = params.cascade_decay;
}

/// @brief Find each source's shortest positive-depth neighbors while sharing
/// graph traversal across source-order batches.
std::vector<std::vector<CascadeNeighbor>>
FindCascadeNeighbors (const AssociationFanoutCache &cache,
                      const std::vector<EmotionalSource> &sources,
                      const EmotionCascadeParams &fallback,
                      std::vector<std::unordered_set<long long>>
                          *expandable_by_source)
{
  std::vector<std::vector<CascadeNeighbor>> neighbors (sources.size ());
  if (expandable_by_source)
    expandable_by_source->assign (sources.size (), {});
  constexpr std::size_t kSourcesPerBatch = 64;
  for (std::size_t batch_begin = 0; batch_begin < sources.size ();
       batch_begin += kSourcesPerBatch)
    {
      const std::size_t batch_size = std::min (
          kSourcesPerBatch, sources.size () - batch_begin);
      std::vector<std::uint64_t> active_by_depth (1, 0);
      int batch_max_depth = 0;
      std::unordered_map<long long, std::uint64_t> visited;
      std::unordered_map<long long, std::uint64_t> frontier;
      for (std::size_t local = 0; local < batch_size; ++local)
        {
          const auto &source = sources[batch_begin + local];
          const int radius = source.cascade_radius > 0
                                 ? source.cascade_radius
                                 : fallback.cascade_radius;
          if (source.memory_id <= 0 || radius < 1)
            continue;
          batch_max_depth = std::max (batch_max_depth, radius);
          active_by_depth.resize (
              static_cast<std::size_t> (batch_max_depth + 1), 0);
          const std::uint64_t bit = std::uint64_t { 1 } << local;
          for (int depth = 1; depth <= radius; ++depth)
            active_by_depth[static_cast<std::size_t> (depth)] |= bit;
          visited[source.memory_id] |= bit;
          frontier[source.memory_id] |= bit;
          if (expandable_by_source)
            (*expandable_by_source)[batch_begin + local].insert (
                source.memory_id);
        }

      std::unordered_map<long long, std::uint64_t> seen_embeddings;
      for (int depth = 1;
           depth <= batch_max_depth && !frontier.empty (); ++depth)
        {
          std::unordered_map<long long, std::uint64_t> next_frontier;
          std::unordered_map<long long, std::uint64_t> reached_embeddings;
          const std::uint64_t active
              = active_by_depth[static_cast<std::size_t> (depth)];
          auto visit_edges = [&] (const auto &fanout, long long memory_id,
                                  std::uint64_t source_mask) {
            const auto edge_it = fanout.find (memory_id);
            if (edge_it == fanout.end ())
              return;
            for (const auto &edge : edge_it->second)
              {
                if (edge.memory_id <= 0)
                  continue;
                if (edge.embedding_id > 0)
                  reached_embeddings[edge.embedding_id] |= source_mask;
                const std::uint64_t newly_visited
                    = source_mask & ~visited[edge.memory_id];
                if (newly_visited != 0)
                  {
                    visited[edge.memory_id] |= newly_visited;
                    next_frontier[edge.memory_id] |= newly_visited;
                    if (expandable_by_source
                        && depth < batch_max_depth)
                      {
                        std::uint64_t expandable
                            = newly_visited
                              & active_by_depth[static_cast<std::size_t> (
                                  depth + 1)];
                        while (expandable != 0)
                          {
                            const unsigned local
                                = std::countr_zero (expandable);
                            (*expandable_by_source)[batch_begin + local]
                                .insert (edge.memory_id);
                            expandable &= expandable - 1;
                          }
                      }
                  }
              }
          };
          for (const auto &[memory_id, frontier_mask] : frontier)
            {
              const std::uint64_t source_mask = frontier_mask & active;
              if (source_mask == 0)
                continue;
              visit_edges (cache.out_by_source, memory_id, source_mask);
              visit_edges (cache.in_by_target, memory_id, source_mask);
            }
          for (const auto &[embedding_id, reached] : reached_embeddings)
            {
              std::uint64_t unseen = reached & ~seen_embeddings[embedding_id];
              seen_embeddings[embedding_id] |= unseen;
              while (unseen != 0)
                {
                  const unsigned local = std::countr_zero (unseen);
                  neighbors[batch_begin + local].push_back (
                      { embedding_id, depth });
                  unseen &= unseen - 1;
                }
            }
          frontier = std::move (next_frontier);
        }
    }
  return neighbors;
}

std::unordered_map<long long, CurrentEmotionalValues>
LoadCurrentEmotionalValues (Store *store)
{
  std::unordered_map<long long, CurrentEmotionalValues> current;
  auto rows = store->Execute (
      "SELECT embedding_id, MIN(emotional_intensity) AS current_intensity, "
      "       MIN(half_life_bonus) AS current_bonus "
      "FROM memories WHERE embedding_id IS NOT NULL GROUP BY embedding_id");
  current.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const auto id = store::AnyToLongLong (row.at ("embedding_id"));
      if (!id)
        continue;
      current.emplace (
          *id,
          CurrentEmotionalValues {
              store::AnyToDouble (row.at ("current_intensity"), 0.0),
              store::AnyToDouble (row.at ("current_bonus"), 0.0) });
    }
  return current;
}

} // namespace

void
PropagateEmotionalCascade::Execute (OperationContext &context, Transaction &tx) const
{
  const bool profile_cascade = internal::experimental_env::Flag (
      "CORTEXT_PROFILE_EMOTIONAL_CASCADE");
  const auto total_start = std::chrono::steady_clock::now ();
  auto record_timing = [&context, profile_cascade] (
                           std::string_view name,
                           std::chrono::steady_clock::time_point start) {
    if (profile_cascade)
      {
        context.AddOperationTiming (
            name,
            std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - start)
                .count ());
      }
  };
  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto params
      = EmotionCascadeParams::FromKnobs (cfg.focus, cfg.sensitivity, cfg.stability);

  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Define the stability-derived lookback window for emotional sources.
  const int consolidation_interval
      = core::EmotionCascadeWindowSeconds (cfg.stability);
  const long long recent_window_ms
      = static_cast<long long> (consolidation_interval) * 1000LL;
  const long long recent_window_ts = std::max (0LL, now_ts - recent_window_ms);
  const double theta_intensity = core::ThetaIntensity (cfg.sensitivity);
  const double theta_arousal = core::ThetaArousal (cfg.sensitivity);
  const double intensity_floor = core::CascadeIntensityFloor (
      cfg.focus, cfg.sensitivity, cfg.stability);

  auto &p_ctx = context.GetProcessorContext ();
  const auto emotional_state
      = emotional_metadata_cache_internal::FindState (p_ctx);
  const bool use_metadata_cache
      = emotional_state && emotional_state->emotional_metadata.valid
        && !internal::experimental_env::Flag (
            "CORTEXT_PROFILE_EMOTIONAL_METADATA_SQL");
  const std::uint64_t emotional_input_generation
      = use_metadata_cache
            ? emotional_state->emotional_metadata.cascade_input_generation
            : 0;

  const auto cache_ensure_start = std::chrono::steady_clock::now ();
  auto *fanout_cache = association_fanout_cache::Ensure (store, p_ctx);
  const double cache_ensure_ms
      = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - cache_ensure_start)
            .count ();

  if (use_metadata_cache && fanout_cache && fanout_cache->valid
      && !internal::experimental_env::Flag (
          "CORTEXT_DISABLE_EMOTIONAL_CASCADE_FIXED_POINT")
      && (MatchesCascadeFixedPoint (
              p_ctx, *fanout_cache, recent_window_ts, theta_intensity,
              theta_arousal, intensity_floor, params)
          || (MatchesCascadeFixedPointInputs (
                  p_ctx, recent_window_ts, theta_intensity, theta_arousal,
                  intensity_floor, params)
              && emotional_state
              && !AssociationChangesCanReachCascade (*emotional_state))))
    {
      RecordCascadeFixedPoint (
          p_ctx, *fanout_cache, emotional_input_generation,
          recent_window_ts, theta_intensity, theta_arousal,
          intensity_floor, params);
      ConsumeAssociationTopologyChanges (*emotional_state);
      const auto telemetry_start = std::chrono::steady_clock::now ();
      telemetry::LogDebug("cortext.emotion_cascade", {
        telemetry::Attribute::Int64("cascade_sources", 0),
        telemetry::Attribute::Int64("max_hops", 0)
      });
      record_timing ("EmotionalCascade.telemetry", telemetry_start);
      if (profile_cascade)
        {
          context.AddOperationTiming ("EmotionalCascade.source_query", 0.0);
          context.AddOperationTiming ("EmotionalCascade.source_count", 0.0);
          context.AddOperationTiming (
              "EmotionalCascade.source_execution_count", 0.0);
          context.AddOperationTiming ("EmotionalCascade.source_activity", 0.0);
          context.AddOperationTiming ("EmotionalCascade.neighbor_lookup",
                                      cache_ensure_ms);
          context.AddOperationTiming ("EmotionalCascade.cache_ensure",
                                      cache_ensure_ms);
          context.AddOperationTiming ("EmotionalCascade.cache_bfs", 0.0);
          context.AddOperationTiming ("EmotionalCascade.current_values", 0.0);
          context.AddOperationTiming ("EmotionalCascade.neighbor_count", 0.0);
          context.AddOperationTiming (
              "EmotionalCascade.neighbor_execution_count", 0.0);
          context.AddOperationTiming ("EmotionalCascade.neighbor_activity", 0.0);
          context.AddOperationTiming ("EmotionalCascade.update_enqueue", 0.0);
          context.AddOperationTiming ("EmotionalCascade.update_count", 0.0);
          context.AddOperationTiming (
              "EmotionalCascade.update_execution_count", 0.0);
          context.AddOperationTiming ("EmotionalCascade.update_activity", 0.0);
          record_timing ("EmotionalCascade.total", total_start);
        }
      return;
    }

  // Load high-intensity emotional sources only when the fixed input changed.
  const auto source_query_start = std::chrono::steady_clock::now ();
  std::size_t source_execution_count = 0;
  auto sources = use_metadata_cache
                     ? LoadEmotionalSourcesFromCache (
                           p_ctx, recent_window_ts,
                           theta_intensity, theta_arousal, params,
                           &source_execution_count)
                     : LoadEmotionalSources (store, recent_window_ts,
                                             theta_intensity, theta_arousal,
                                             params, &source_execution_count);
  record_timing ("EmotionalCascade.source_query", source_query_start);
  if (profile_cascade)
    {
      context.AddOperationTiming (
          "EmotionalCascade.source_count",
          static_cast<double> (sources.size ())); // Experimental count, not ms.
      context.AddOperationTiming ("EmotionalCascade.source_activity",
                                  sources.empty () ? 0.0 : 1.0);
      context.AddOperationTiming (
          "EmotionalCascade.source_execution_count",
          static_cast<double> (source_execution_count));
    }

  if (sources.empty ())
    {
      if (use_metadata_cache && fanout_cache && fanout_cache->valid)
        {
          RecordCascadeFixedPoint (
              p_ctx, *fanout_cache, emotional_input_generation,
              recent_window_ts, theta_intensity, theta_arousal,
              intensity_floor, params);
          emotional_state->emotional_cascade_topology_footprint
              = { true, {} };
          ConsumeAssociationTopologyChanges (*emotional_state);
        }
      const auto telemetry_start = std::chrono::steady_clock::now ();
      telemetry::LogDebug("cortext.emotion_cascade", {
        telemetry::Attribute::Int64("cascade_sources", 0),
        telemetry::Attribute::Int64("max_hops", 0)
      });
      record_timing ("EmotionalCascade.telemetry", telemetry_start);
      if (profile_cascade)
        {
          context.AddOperationTiming ("EmotionalCascade.neighbor_lookup", 0.0);
          context.AddOperationTiming ("EmotionalCascade.cache_ensure",
                                      cache_ensure_ms);
          context.AddOperationTiming ("EmotionalCascade.cache_bfs", 0.0);
          context.AddOperationTiming ("EmotionalCascade.current_values", 0.0);
          context.AddOperationTiming ("EmotionalCascade.neighbor_count", 0.0);
          context.AddOperationTiming (
              "EmotionalCascade.neighbor_execution_count", 0.0);
          context.AddOperationTiming ("EmotionalCascade.neighbor_activity", 0.0);
          context.AddOperationTiming ("EmotionalCascade.update_enqueue", 0.0);
          context.AddOperationTiming ("EmotionalCascade.update_count", 0.0);
          context.AddOperationTiming (
              "EmotionalCascade.update_execution_count", 0.0);
          context.AddOperationTiming ("EmotionalCascade.update_activity", 0.0);
          record_timing ("EmotionalCascade.total", total_start);
        }
      return;
    }

  // Track which embeddings we've already processed to avoid duplicates
  std::unordered_set<long long> processed;
  int max_hops = 0;
  std::size_t neighbor_count = 0;
  std::size_t neighbor_execution_count = 0;
  std::size_t update_count = 0;
  std::size_t update_execution_count = 0;
  double neighbor_lookup_ms = 0.0;
  double cache_bfs_ms = 0.0;
  double current_values_ms = 0.0;
  double update_enqueue_ms = 0.0;

  const auto neighbor_start = std::chrono::steady_clock::now ();
  const auto cache_bfs_start = std::chrono::steady_clock::now ();
  std::vector<std::vector<CascadeNeighbor>> source_neighbors;
  if (fanout_cache && fanout_cache->valid)
    {
      std::vector<std::unordered_set<long long>> expandable_by_source;
      source_neighbors = FindCascadeNeighbors (*fanout_cache, sources,
                                               params,
                                               &expandable_by_source);
      if (use_metadata_cache && emotional_state)
        {
          emotional_state->emotional_cascade_topology_footprint
              = { true, std::move (expandable_by_source) };
          ConsumeAssociationTopologyChanges (*emotional_state);
        }
      for (const auto &neighbors : source_neighbors)
        neighbor_count += neighbors.size ();
    }
  else
    source_neighbors.resize (sources.size ());
  cache_bfs_ms = std::chrono::duration<double, std::milli> (
      std::chrono::steady_clock::now () - cache_bfs_start).count ();
  const auto current_values_start = std::chrono::steady_clock::now ();
  const auto fallback_current_values
      = use_metadata_cache ? std::unordered_map<long long,
                                                CurrentEmotionalValues>{}
                           : LoadCurrentEmotionalValues (store);
  const auto &current_values
      = use_metadata_cache
            ? emotional_state->emotional_metadata.values_by_embedding
            : fallback_current_values;
  current_values_ms = std::chrono::duration<double, std::milli> (
      std::chrono::steady_clock::now () - current_values_start).count ();
  neighbor_lookup_ms = std::chrono::duration<double, std::milli> (
      std::chrono::steady_clock::now () - neighbor_start).count ();

  for (std::size_t source_index = 0; source_index < sources.size ();
       ++source_index)
    {
      const auto &src = sources[source_index];
      const int radius = (src.cascade_radius > 0) ? src.cascade_radius
                                                  : params.cascade_radius;
      const double decay = (src.cascade_decay > 0.0) ? src.cascade_decay
                                                     : params.cascade_decay;
      if (radius > max_hops)
        max_hops = radius;

      for (const auto &neighbor : source_neighbors[source_index])
        {
          ++neighbor_execution_count;

          // Preserve source ordering and the existing first-source-wins rule.
          if (processed.count (neighbor.embedding_id) > 0)
            {
              continue;
            }
          processed.insert (neighbor.embedding_id);

          const double decayed_intensity
              = src.intensity * std::pow (decay, neighbor.depth);
          const double decayed_bonus
              = src.half_life_bonus * std::pow (decay, neighbor.depth);

          if (decayed_intensity < intensity_floor)
            {
              continue;
            }
          const auto current_it
              = current_values.find (neighbor.embedding_id);
          if (current_it != current_values.end ()
              && decayed_intensity <= current_it->second.intensity
              && decayed_bonus <= current_it->second.half_life_bonus)
            {
              continue;
            }

          const auto update_start = std::chrono::steady_clock::now ();
          Add (tx,
               "UPDATE memories "
               "SET emotional_intensity = MAX(emotional_intensity, ?1), "
               "    half_life_bonus = MAX(half_life_bonus, ?2) "
               "WHERE embedding_id = ?3",
               { decayed_intensity, decayed_bonus, neighbor.embedding_id });
          emotional_metadata_cache_internal::MaxEmbedding (
              context.GetProcessorContext (), neighbor.embedding_id,
              decayed_intensity, decayed_bonus);
          update_enqueue_ms += std::chrono::duration<double, std::milli> (
              std::chrono::steady_clock::now () - update_start).count ();
          ++update_count;
          ++update_execution_count;
        }
    }

  if (use_metadata_cache && fanout_cache && fanout_cache->valid)
    RecordCascadeFixedPoint (
        p_ctx, *fanout_cache, emotional_input_generation, recent_window_ts,
        theta_intensity, theta_arousal, intensity_floor, params);

  const auto telemetry_start = std::chrono::steady_clock::now ();
  telemetry::LogDebug("cortext.emotion_cascade", {
    telemetry::Attribute::Int64("cascade_sources", static_cast<long long>(sources.size())),
    telemetry::Attribute::Int64("max_hops", max_hops)
  });
  record_timing ("EmotionalCascade.telemetry", telemetry_start);
  if (profile_cascade)
    {
      context.AddOperationTiming (
          "EmotionalCascade.neighbor_lookup", neighbor_lookup_ms);
      context.AddOperationTiming (
          "EmotionalCascade.cache_ensure", cache_ensure_ms);
      context.AddOperationTiming ("EmotionalCascade.cache_bfs", cache_bfs_ms);
      context.AddOperationTiming (
          "EmotionalCascade.current_values", current_values_ms);
      context.AddOperationTiming (
          "EmotionalCascade.neighbor_count",
          static_cast<double> (neighbor_count)); // Experimental count, not ms.
      context.AddOperationTiming ("EmotionalCascade.neighbor_activity",
                                  neighbor_count > 0 ? 1.0 : 0.0);
      context.AddOperationTiming (
          "EmotionalCascade.neighbor_execution_count",
          static_cast<double> (neighbor_execution_count));
      context.AddOperationTiming (
          "EmotionalCascade.update_enqueue", update_enqueue_ms);
      context.AddOperationTiming (
          "EmotionalCascade.update_count",
          static_cast<double> (update_count)); // Experimental count, not ms.
      context.AddOperationTiming ("EmotionalCascade.update_activity",
                                  update_count > 0 ? 1.0 : 0.0);
      context.AddOperationTiming (
          "EmotionalCascade.update_execution_count",
          static_cast<double> (update_execution_count));
      record_timing ("EmotionalCascade.total", total_start);
    }
}

} // namespace cortext::operations
