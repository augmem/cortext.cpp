#pragma once

#include "cortext/processor/processor_context.hpp"
#include "rif_active_epoch_cache_internal.hpp"

#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations::execution_cache_sidecar_internal
{

struct EmotionalMemoryMetadata
{
  long long memory_id = 0;
  long long embedding_id = 0;
  long long created_at = 0;
  bool flashbulb = false;
  double intensity = 0.0;
  double arousal = 0.0;
  double half_life_bonus = 0.0;
  int cascade_radius = 0;
  double cascade_decay = 0.0;
};

struct EmotionalEmbeddingValues
{
  double intensity = 0.0;
  double half_life_bonus = 0.0;
};

struct EmotionalMetadataCache
{
  bool valid = false;
  std::uint64_t cascade_input_generation = 0;
  std::unordered_map<long long, EmotionalMemoryMetadata> rows_by_memory;
  std::unordered_map<long long, std::vector<long long>>
      memory_ids_by_embedding;
  std::unordered_map<long long, EmotionalEmbeddingValues>
      values_by_embedding;
  std::size_t source_query_capacity
      = std::numeric_limits<std::size_t>::max ();
  std::vector<long long> source_query_order;
};

struct EmotionalCascadeFixedPoint
{
  bool valid = false;
  std::uint64_t emotional_input_generation = 0;
  long long association_edge_count = 0;
  long long association_source_sum = 0;
  long long association_target_sum = 0;
  long long association_source_max = 0;
  long long association_target_max = 0;
  long long association_weight_sum_micros = 0;
  long long association_last_reinforced_sum = 0;
  long long recent_window_ts = 0;
  double theta_intensity = 0.0;
  double theta_arousal = 0.0;
  double intensity_floor = 0.0;
  int cascade_radius = 0;
  double cascade_decay = 0.0;
};

struct AssociationTopologyChanges
{
  bool reset = true;
  std::vector<std::pair<long long, long long>> inserted_edges;
};

struct EmotionalCascadeTopologyFootprint
{
  bool valid = false;
  std::vector<std::unordered_set<long long>> expandable_by_source;
};

struct SupersessionEligibility
{
  bool valid = false;
  std::unordered_map<long long, long long> activation_ts_by_target;
};

struct State
{
  EmotionalMetadataCache emotional_metadata;
  EmotionalCascadeFixedPoint emotional_fixed_point;
  AssociationTopologyChanges association_topology_changes;
  EmotionalCascadeTopologyFootprint emotional_cascade_topology_footprint;
  SupersessionEligibility supersession_eligibility;
  rif_active_epoch_cache_internal::State rif_active_epoch;
};

struct RegistryState
{
  std::mutex mutex;
  std::unordered_map<const ProcessorContext *, std::shared_ptr<State>> states;
};

inline RegistryState &
Registry ()
{
  static auto *registry = new RegistryState ();
  return *registry;
}

inline std::shared_ptr<State>
Ensure (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  auto &state = registry.states[&ctx];
  if (!state)
    state = std::make_shared<State> ();
  return state;
}

inline std::shared_ptr<State>
Find (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  const auto it = registry.states.find (&ctx);
  return it == registry.states.end () ? nullptr : it->second;
}

inline std::shared_ptr<State>
Detach (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  const auto it = registry.states.find (&ctx);
  if (it == registry.states.end ())
    return nullptr;
  auto state = std::move (it->second);
  registry.states.erase (it);
  return state;
}

inline void
Restore (const ProcessorContext &ctx, std::shared_ptr<State> state)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  if (state)
    registry.states[&ctx] = std::move (state);
  else
    registry.states.erase (&ctx);
}

inline void
Erase (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  registry.states.erase (&ctx);
}

#if defined(CORTEXT_TESTING)
inline std::size_t
RegistrySizeForTest ()
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  return registry.states.size ();
}
#endif

} // namespace cortext::operations::execution_cache_sidecar_internal
