#pragma once

#include "cortext/processor/processor_context.hpp"
#include "cortext/store/store.hpp"
#include "execution_cache_sidecar_internal.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace cortext::operations::emotional_metadata_cache_internal
{

using Row = execution_cache_sidecar_internal::EmotionalMemoryMetadata;
using Cache = execution_cache_sidecar_internal::EmotionalMetadataCache;
using State = execution_cache_sidecar_internal::State;

inline std::shared_ptr<State>
EnsureState (const ProcessorContext &p_ctx)
{
  return execution_cache_sidecar_internal::Ensure (p_ctx);
}

inline std::shared_ptr<State>
FindState (const ProcessorContext &p_ctx)
{
  return execution_cache_sidecar_internal::Find (p_ctx);
}

inline Cache &
Ensure (const ProcessorContext &p_ctx)
{
  return EnsureState (p_ctx)->emotional_metadata;
}

inline void
BumpCascadeInputGeneration (
    Cache &cache)
{
  ++cache.cascade_input_generation;
}

inline void
RecomputeEmbedding (Cache &cache,
                    long long embedding_id)
{
  const auto members = cache.memory_ids_by_embedding.find (embedding_id);
  if (members == cache.memory_ids_by_embedding.end ()
      || members->second.empty ())
    {
      cache.values_by_embedding.erase (embedding_id);
      return;
    }
  double intensity = std::numeric_limits<double>::infinity ();
  double bonus = std::numeric_limits<double>::infinity ();
  for (const long long memory_id : members->second)
    {
      const auto row = cache.rows_by_memory.find (memory_id);
      if (row == cache.rows_by_memory.end ())
        {
          cache.valid = false;
          return;
        }
      intensity = std::min (intensity, row->second.intensity);
      bonus = std::min (bonus, row->second.half_life_bonus);
    }
  cache.values_by_embedding[embedding_id] = { intensity, bonus };
}

inline void
Remove (ProcessorContext &p_ctx, long long memory_id)
{
  const auto state = FindState (p_ctx);
  if (!state)
    return;
  auto &cache = state->emotional_metadata;
  if (!cache.valid)
    return;
  const auto row = cache.rows_by_memory.find (memory_id);
  if (row == cache.rows_by_memory.end ())
    return;
  const long long embedding_id = row->second.embedding_id;
  if (row->second.flashbulb
      && cache.source_query_capacity
             != std::numeric_limits<std::size_t>::max ())
    cache.source_query_order_dirty = true;
  cache.rows_by_memory.erase (row);
  auto members = cache.memory_ids_by_embedding.find (embedding_id);
  if (members != cache.memory_ids_by_embedding.end ())
    {
      std::erase (members->second, memory_id);
      if (members->second.empty ())
        cache.memory_ids_by_embedding.erase (members);
    }
  std::erase (cache.source_query_order, memory_id);
  RecomputeEmbedding (cache, embedding_id);
  BumpCascadeInputGeneration (cache);
}

inline bool
BeforeSourceQuery (const Row &left, const Row &right)
{
  return left.memory_id < right.memory_id;
}

inline bool
BeforeBoundedSourceQuery (const Row &left, const Row &right)
{
  if (left.intensity != right.intensity)
    return left.intensity > right.intensity;
  return left.memory_id < right.memory_id;
}

inline void
Upsert (ProcessorContext &p_ctx, Row row)
{
  const auto state = FindState (p_ctx);
  if (!state)
    return;
  auto &cache = state->emotional_metadata;
  if (!cache.valid || row.memory_id <= 0 || row.embedding_id <= 0)
    return;
  const bool inserts_unshared_non_source
      = !row.flashbulb
        && cache.rows_by_memory.find (row.memory_id)
               == cache.rows_by_memory.end ()
        && cache.memory_ids_by_embedding.find (row.embedding_id)
               == cache.memory_ids_by_embedding.end ();
  Remove (p_ctx, row.memory_id);
  auto &members = cache.memory_ids_by_embedding[row.embedding_id];
  members.insert (std::lower_bound (members.begin (), members.end (),
                                    row.memory_id),
                  row.memory_id);
  const long long memory_id = row.memory_id;
  const long long embedding_id = row.embedding_id;
  cache.rows_by_memory.emplace (memory_id, std::move (row));
  const Row &stored = cache.rows_by_memory.at (memory_id);
  if (stored.flashbulb)
    {
      const auto position = std::lower_bound (
          cache.source_query_order.begin (), cache.source_query_order.end (),
          stored,
          [&cache] (long long existing_memory_id, const Row &candidate) {
            const auto &existing
                = cache.rows_by_memory.at (existing_memory_id);
            return cache.source_query_capacity
                           == std::numeric_limits<std::size_t>::max ()
                       ? BeforeSourceQuery (existing, candidate)
                       : BeforeBoundedSourceQuery (existing, candidate);
          });
      cache.source_query_order.insert (position, memory_id);
      if (cache.source_query_order.size () > cache.source_query_capacity)
        cache.source_query_order.pop_back ();
      if (cache.source_query_capacity
          != std::numeric_limits<std::size_t>::max ())
        cache.source_query_order_dirty = true;
    }
  RecomputeEmbedding (cache, embedding_id);
  // A newly persisted, unconnected non-source cannot change a completed
  // cascade. If an association is subsequently added, the fanout fingerprint
  // independently invalidates the fixed point. Shared embeddings remain
  // conservative because a new member can lower the aggregate target value.
  if (!inserts_unshared_non_source)
    BumpCascadeInputGeneration (cache);
}

inline void
Reset (ProcessorContext &p_ctx, std::vector<Row> rows,
       std::size_t source_query_capacity
           = std::numeric_limits<std::size_t>::max ())
{
  const std::uint64_t next_generation
      = Ensure (p_ctx).cascade_input_generation + 1;
  auto &cache = Ensure (p_ctx);
  cache = {};
  cache.valid = true;
  cache.source_query_capacity = source_query_capacity;
  cache.cascade_input_generation = next_generation;
  cache.rows_by_memory.reserve (rows.size ());
  cache.source_query_order.reserve (rows.size ());
  for (auto &row : rows)
    Upsert (p_ctx, std::move (row));
  cache.source_query_order_dirty = false;
}

inline bool
RefreshBoundedSourceOrder (ProcessorContext &p_ctx, Transaction &tx)
{
  const auto state = FindState (p_ctx);
  if (!state)
    return false;
  auto &cache = state->emotional_metadata;
  if (!cache.valid)
    return false;
  if (cache.source_query_capacity
          == std::numeric_limits<std::size_t>::max ()
      || !cache.source_query_order_dirty)
    return true;

  auto rows = tx.Execute (
      "SELECT memory_id FROM memories "
      "WHERE flashbulb = 1 AND embedding_id IS NOT NULL "
      "AND kind != 'WORKING' "
      "ORDER BY emotional_intensity DESC, memory_id ASC LIMIT ?",
      { static_cast<long long> (cache.source_query_capacity) });
  std::vector<long long> order;
  order.reserve (rows.size ());
  for (const auto &result : rows)
    {
      const auto id = result.find ("memory_id");
      if (id == result.end () || id->second.type () != typeid (long long))
        {
          cache.valid = false;
          return false;
        }
      const long long memory_id = std::any_cast<long long> (id->second);
      const auto cached = cache.rows_by_memory.find (memory_id);
      if (cached == cache.rows_by_memory.end () || !cached->second.flashbulb)
        {
          cache.valid = false;
          return false;
        }
      order.push_back (memory_id);
    }
  cache.source_query_order = std::move (order);
  cache.source_query_order_dirty = false;
  return true;
}

inline void
OverwriteEmbedding (ProcessorContext &p_ctx, long long embedding_id,
                    bool flashbulb, double intensity, double bonus, int radius,
                    double decay)
{
  const auto state = FindState (p_ctx);
  if (!state)
    return;
  auto &cache = state->emotional_metadata;
  if (!cache.valid)
    return;
  const auto members = cache.memory_ids_by_embedding.find (embedding_id);
  if (members == cache.memory_ids_by_embedding.end ())
    return;
  const auto previous_values = cache.values_by_embedding.find (embedding_id);
  const double previous_intensity
      = previous_values == cache.values_by_embedding.end ()
            ? std::numeric_limits<double>::infinity ()
            : previous_values->second.intensity;
  const double previous_bonus
      = previous_values == cache.values_by_embedding.end ()
            ? std::numeric_limits<double>::infinity ()
            : previous_values->second.half_life_bonus;
  bool source_input_changed = false;
  for (const long long memory_id : members->second)
    {
      auto &row = cache.rows_by_memory.at (memory_id);
      const bool became_source = !row.flashbulb && flashbulb;
      const bool source_rank_changed
          = (row.flashbulb || flashbulb) && row.intensity != intensity;
      source_input_changed
          = source_input_changed || became_source
            || (row.flashbulb
                && (row.intensity != intensity
                    || row.half_life_bonus != bonus
                    || row.cascade_radius != radius
                    || row.cascade_decay != decay));
      row.flashbulb = row.flashbulb || flashbulb;
      row.intensity = intensity;
      row.half_life_bonus = bonus;
      row.cascade_radius = radius;
      row.cascade_decay = decay;
      if (became_source)
        {
          const auto position = std::lower_bound (
              cache.source_query_order.begin (),
              cache.source_query_order.end (), row,
              [&cache] (long long existing_memory_id, const Row &candidate) {
                const auto &existing
                    = cache.rows_by_memory.at (existing_memory_id);
                return cache.source_query_capacity
                               == std::numeric_limits<std::size_t>::max ()
                           ? BeforeSourceQuery (existing, candidate)
                           : BeforeBoundedSourceQuery (existing, candidate);
              });
          cache.source_query_order.insert (position, memory_id);
          if (cache.source_query_order.size () > cache.source_query_capacity)
            cache.source_query_order.pop_back ();
        }
      if (cache.source_query_capacity
              != std::numeric_limits<std::size_t>::max ()
          && (became_source || source_rank_changed))
        cache.source_query_order_dirty = true;
    }
  RecomputeEmbedding (cache, embedding_id);
  const auto current_values = cache.values_by_embedding.find (embedding_id);
  const bool target_value_decreased
      = current_values != cache.values_by_embedding.end ()
        && (current_values->second.intensity < previous_intensity
            || current_values->second.half_life_bonus < previous_bonus);
  if (source_input_changed || target_value_decreased)
    BumpCascadeInputGeneration (cache);
}

inline void
MaxEmbedding (ProcessorContext &p_ctx, long long embedding_id,
              double intensity, double bonus)
{
  const auto state = FindState (p_ctx);
  if (!state)
    return;
  auto &cache = state->emotional_metadata;
  if (!cache.valid)
    return;
  const auto members = cache.memory_ids_by_embedding.find (embedding_id);
  if (members == cache.memory_ids_by_embedding.end ())
    return;
  bool source_input_changed = false;
  for (const long long memory_id : members->second)
    {
      auto &row = cache.rows_by_memory.at (memory_id);
      source_input_changed
          = source_input_changed
            || (row.flashbulb
                && (intensity > row.intensity
                    || bonus > row.half_life_bonus));
      if (cache.source_query_capacity
              != std::numeric_limits<std::size_t>::max ()
          && row.flashbulb && intensity > row.intensity)
        cache.source_query_order_dirty = true;
      row.intensity = std::max (row.intensity, intensity);
      row.half_life_bonus = std::max (row.half_life_bonus, bonus);
    }
  RecomputeEmbedding (cache, embedding_id);
  if (source_input_changed)
    BumpCascadeInputGeneration (cache);
}

} // namespace cortext::operations::emotional_metadata_cache_internal
