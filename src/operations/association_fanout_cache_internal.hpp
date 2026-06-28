#pragma once

#include "cortext/processor/processor_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <optional>

namespace cortext::operations::association_fanout_cache
{

struct Fingerprint
{
  long long edge_count = 0;
  long long source_sum = 0;
  long long target_sum = 0;
  long long source_max = 0;
  long long target_max = 0;
  long long weight_sum_micros = 0;
  long long last_reinforced_sum = 0;
};

inline Fingerprint
LoadFingerprint (Store &store)
{
  Fingerprint out;
  auto rows = store.Execute (
      "SELECT COUNT(*) AS edge_count, "
      "       COALESCE(SUM(source_memory_id), 0) AS source_sum, "
      "       COALESCE(SUM(target_memory_id), 0) AS target_sum, "
      "       COALESCE(MAX(source_memory_id), 0) AS source_max, "
      "       COALESCE(MAX(target_memory_id), 0) AS target_max, "
      "       COALESCE(SUM(CAST(ROUND(weight * 1000000.0) AS INTEGER)), 0) "
      "         AS weight_sum_micros, "
      "       COALESCE(SUM(COALESCE(last_reinforced, 0)), 0) "
      "         AS last_reinforced_sum "
      "FROM associations");
  if (rows.empty ())
    {
      return out;
    }
  out.edge_count = store::AnyToLongLong (rows[0].at ("edge_count"))
                       .value_or (0);
  out.source_sum = store::AnyToLongLong (rows[0].at ("source_sum"))
                       .value_or (0);
  out.target_sum = store::AnyToLongLong (rows[0].at ("target_sum"))
                       .value_or (0);
  out.source_max = store::AnyToLongLong (rows[0].at ("source_max"))
                       .value_or (0);
  out.target_max = store::AnyToLongLong (rows[0].at ("target_max"))
                       .value_or (0);
  out.weight_sum_micros = store::AnyToLongLong (
                              rows[0].at ("weight_sum_micros"))
                              .value_or (0);
  out.last_reinforced_sum = store::AnyToLongLong (
                                rows[0].at ("last_reinforced_sum"))
                                .value_or (0);
  return out;
}

inline bool
Matches (const ProcessorContext::AssociationFanoutCache &cache,
         const Fingerprint &fingerprint)
{
  return cache.valid && cache.edge_count == fingerprint.edge_count
         && cache.source_sum == fingerprint.source_sum
         && cache.target_sum == fingerprint.target_sum
         && cache.source_max == fingerprint.source_max
         && cache.target_max == fingerprint.target_max
         && cache.weight_sum_micros == fingerprint.weight_sum_micros
         && cache.last_reinforced_sum == fingerprint.last_reinforced_sum;
}

inline void
SortFanoutEdges (
    std::vector<ProcessorContext::AssociationFanoutEdge> &edges)
{
  std::sort (
      edges.begin (), edges.end (),
      [] (const auto &a, const auto &b) {
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
}

inline void
SortFanoutMap (
    std::unordered_map<long long,
                       std::vector<ProcessorContext::AssociationFanoutEdge>>
        &edges_by_memory)
{
  for (auto &[memory_id, edges] : edges_by_memory)
    {
      (void)memory_id;
      SortFanoutEdges (edges);
    }
}

inline bool
UpsertFanoutEdge (
    std::vector<ProcessorContext::AssociationFanoutEdge> &edges,
    long long memory_id, long long embedding_id, const std::string &edge_type,
    double weight, long long last_reinforced, double *old_weight,
    long long *old_last_reinforced)
{
  for (auto &edge : edges)
    {
      if (edge.memory_id == memory_id && edge.edge_type == edge_type)
        {
          if (old_weight)
            {
              *old_weight = edge.weight;
            }
          if (old_last_reinforced)
            {
              *old_last_reinforced = edge.last_reinforced;
            }
          edge.embedding_id = embedding_id;
          edge.weight = weight;
          edge.last_reinforced = last_reinforced;
          SortFanoutEdges (edges);
          return false;
        }
    }
  edges.push_back (
      { memory_id, embedding_id, edge_type, weight, last_reinforced });
  SortFanoutEdges (edges);
  return true;
}

inline void
UpsertAssociation (ProcessorContext::AssociationFanoutCache &cache,
                   long long source_memory_id, long long target_memory_id,
                   long long source_embedding_id, long long target_embedding_id,
                   const std::string &edge_type, double weight,
                   long long last_reinforced)
{
  if (!cache.valid || source_memory_id <= 0 || target_memory_id <= 0
      || source_embedding_id <= 0 || target_embedding_id <= 0)
    {
      return;
    }
  double old_weight = 0.0;
  long long old_last_reinforced = 0;
  auto &out_edges = cache.out_by_source[source_memory_id];
  const bool inserted = UpsertFanoutEdge (
      out_edges, target_memory_id, target_embedding_id, edge_type, weight,
      last_reinforced, &old_weight, &old_last_reinforced);

  auto &in_edges = cache.in_by_target[target_memory_id];
  (void)UpsertFanoutEdge (in_edges, source_memory_id, source_embedding_id,
                          edge_type, weight, last_reinforced, nullptr,
                          nullptr);

  if (inserted)
    {
      cache.edge_count += 1;
      cache.source_sum += source_memory_id;
      cache.target_sum += target_memory_id;
      cache.source_max = std::max (cache.source_max, source_memory_id);
      cache.target_max = std::max (cache.target_max, target_memory_id);
      cache.weight_sum_micros += static_cast<long long> (
          std::llround (weight * 1000000.0));
      cache.last_reinforced_sum += last_reinforced;
    }
  else
    {
      cache.weight_sum_micros += static_cast<long long> (
          std::llround (weight * 1000000.0)
          - std::llround (old_weight * 1000000.0));
      cache.last_reinforced_sum += last_reinforced - old_last_reinforced;
    }
}

inline ProcessorContext::AssociationFanoutCache *
Ensure (Store *store, ProcessorContext &p_ctx)
{
  if (!store)
    {
      return nullptr;
    }
  auto &cache = p_ctx.association_fanout_cache;
  if (cache.valid
      && cache.last_fingerprint_check_signal == p_ctx.signals_processed)
    {
      return &cache;
    }
  if (cache.valid && cache.consolidation_count == p_ctx.consolidation_count)
    {
      return &cache;
    }

  const Fingerprint fingerprint = LoadFingerprint (*store);
  if (Matches (cache, fingerprint))
    {
      cache.last_fingerprint_check_signal = p_ctx.signals_processed;
      cache.consolidation_count = p_ctx.consolidation_count;
      return &cache;
    }

  cache.valid = false;
  cache.last_fingerprint_check_signal = p_ctx.signals_processed;
  cache.consolidation_count = p_ctx.consolidation_count;
  cache.out_by_source.clear ();
  cache.in_by_target.clear ();

  auto rows = store->Execute (
      "SELECT a.source_memory_id, a.target_memory_id, a.edge_type, "
      "       a.weight, COALESCE(a.last_reinforced, 0) AS last_reinforced, "
      "       COALESCE(src.embedding_id, 0) AS source_embedding_id, "
      "       COALESCE(dst.embedding_id, 0) AS target_embedding_id "
      "FROM associations a "
      "LEFT JOIN memories src ON src.memory_id = a.source_memory_id "
      "LEFT JOIN memories dst ON dst.memory_id = a.target_memory_id",
      {});
  cache.out_by_source.reserve (rows.size ());
  cache.in_by_target.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const long long source_id = store::AnyToLongLong (
                                      row.at ("source_memory_id"))
                                      .value_or (0);
      const long long target_id = store::AnyToLongLong (
                                      row.at ("target_memory_id"))
                                      .value_or (0);
      if (source_id <= 0 || target_id <= 0)
        {
          continue;
        }
      const std::string edge_type
          = row.at ("edge_type").type () == typeid (std::string)
                ? std::any_cast<std::string> (row.at ("edge_type"))
                : std::string ();
      const double weight = cortext::store::AnyToDouble (row.at ("weight"),
                                                        1.0);
      const long long last_reinforced
          = store::AnyToLongLong (row.at ("last_reinforced")).value_or (0);
      const long long source_embedding_id
          = store::AnyToLongLong (row.at ("source_embedding_id"))
                .value_or (0);
      const long long target_embedding_id
          = store::AnyToLongLong (row.at ("target_embedding_id"))
                .value_or (0);
      cache.out_by_source[source_id].push_back (
          { target_id, target_embedding_id, edge_type, weight,
            last_reinforced });
      cache.in_by_target[target_id].push_back (
          { source_id, source_embedding_id, edge_type, weight,
            last_reinforced });
    }
  SortFanoutMap (cache.out_by_source);
  SortFanoutMap (cache.in_by_target);
  cache.edge_count = fingerprint.edge_count;
  cache.source_sum = fingerprint.source_sum;
  cache.target_sum = fingerprint.target_sum;
  cache.source_max = fingerprint.source_max;
  cache.target_max = fingerprint.target_max;
  cache.weight_sum_micros = fingerprint.weight_sum_micros;
  cache.last_reinforced_sum = fingerprint.last_reinforced_sum;
  cache.valid = true;
  return &cache;
}

} // namespace cortext::operations::association_fanout_cache
