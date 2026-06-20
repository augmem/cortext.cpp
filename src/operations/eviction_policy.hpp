#pragma once

#include "cortext/core/knobs.hpp"
#include "cortext/store/store.hpp"
#include "eviction_policy_override.hpp"
#include "storage_pressure.hpp"

#include <any>
#include <string>
#include <unordered_set>
#include <vector>

namespace cortext::operations::eviction_policy
{

struct EvictionFrontier
{
  bool storage_gate_active = false;
  bool storage_allows_eviction = true;
  long long storage_used_bytes = 0;
  long long storage_threshold_bytes = 0;
  double cutoff = 0.0;
  bool consolidation_gate_active = true;
  long long consolidation_ts = 0;
};

inline std::string
EvictionWhereClause (const EvictionFrontier &frontier)
{
  std::string where = "WHERE m.strength < ? AND m.kind = 'LONG_TERM'";
  if (frontier.consolidation_gate_active)
    {
      where += " AND m.created_at < ?";
    }
  return where;
}

inline std::vector<std::any>
EvictionWhereParams (const EvictionFrontier &frontier)
{
  std::vector<std::any> params = { frontier.cutoff };
  if (frontier.consolidation_gate_active)
    {
      params.push_back (frontier.consolidation_ts);
    }
  return params;
}

inline std::vector<std::any>
EvictionInsertParams (long long evicted_at, const EvictionFrontier &frontier)
{
  std::vector<std::any> params = { evicted_at, frontier.cutoff };
  if (frontier.consolidation_gate_active)
    {
      params.push_back (frontier.consolidation_ts);
    }
  return params;
}

inline EvictionFrontier
ResolveEvictionFrontier (Transaction &tx, double T,
                         long long last_consolidation_ts,
                         const eviction::EvictionPolicyOverride &override)
{
  EvictionFrontier frontier;
  frontier.cutoff
      = override.periphery_cutoff.value_or (core::PeripheryCutoff (T));
  frontier.consolidation_gate_active
      = override.consolidation_gate_enabled.value_or (true);
  frontier.consolidation_ts = last_consolidation_ts;

  const pressure::StoragePressureState storage_gate
      = pressure::ComputeStoragePressureState (tx, override);
  frontier.storage_gate_active = storage_gate.active;
  frontier.storage_used_bytes = storage_gate.used_bytes;
  frontier.storage_threshold_bytes = storage_gate.threshold_bytes;
  frontier.storage_allows_eviction
      = !(storage_gate.active
          && storage_gate.used_bytes < storage_gate.threshold_bytes);
  return frontier;
}

inline EvictionFrontier
ResolveEvictionFrontier (ProcessorContext &ctx, Transaction &tx, double T,
                         long long last_consolidation_ts,
                         const eviction::EvictionPolicyOverride &override)
{
  EvictionFrontier frontier;
  frontier.cutoff
      = override.periphery_cutoff.value_or (core::PeripheryCutoff (T));
  frontier.consolidation_gate_active
      = override.consolidation_gate_enabled.value_or (true);
  frontier.consolidation_ts = last_consolidation_ts;

  const pressure::StoragePressureState storage_gate
      = pressure::ComputeCachedStoragePressureState (ctx, tx, override);
  frontier.storage_gate_active = storage_gate.active;
  frontier.storage_used_bytes = storage_gate.used_bytes;
  frontier.storage_threshold_bytes = storage_gate.threshold_bytes;
  frontier.storage_allows_eviction
      = !(storage_gate.active
          && storage_gate.used_bytes < storage_gate.threshold_bytes);
  return frontier;
}

inline std::string
MakePlaceholders (std::size_t count)
{
  std::string out;
  for (std::size_t i = 0; i < count; ++i)
    {
      if (i > 0)
        {
          out += ",";
        }
      out += "?";
    }
  return out;
}

inline std::unordered_set<long long>
LoadEvictableMemoryIds (Transaction &tx, const EvictionFrontier &frontier,
                        const std::vector<long long> &memory_ids)
{
  std::unordered_set<long long> evictable;
  if (!frontier.storage_allows_eviction || memory_ids.empty ())
    {
      return evictable;
    }

  std::vector<std::any> params;
  params.reserve (memory_ids.size () + 3);
  for (long long id : memory_ids)
    {
      params.push_back (id);
    }
  params.push_back (frontier.cutoff);
  if (frontier.consolidation_gate_active)
    {
      params.push_back (frontier.consolidation_ts);
    }

  std::string where = "WHERE m.memory_id IN ("
                      + MakePlaceholders (memory_ids.size ())
                      + ") AND m.strength < ? AND m.kind = 'LONG_TERM'";
  if (frontier.consolidation_gate_active)
    {
      where += " AND m.created_at < ?";
    }

  auto rows = tx.Execute (
      "SELECT m.memory_id FROM memories m " + where,
      params);
  for (const auto &row : rows)
    {
      auto it = row.find ("memory_id");
      if (it == row.end ())
        {
          continue;
        }
      if (it->second.type () == typeid (long long))
        {
          evictable.insert (std::any_cast<long long> (it->second));
        }
      else if (it->second.type () == typeid (int))
        {
          evictable.insert (static_cast<long long> (
              std::any_cast<int> (it->second)));
        }
    }
  return evictable;
}

} // namespace cortext::operations::eviction_policy
