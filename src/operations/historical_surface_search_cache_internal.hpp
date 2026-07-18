#pragma once

#include "cortext/processor/processor_context.hpp"
#include "family_embedding_features_internal.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations::historical_surface_search_cache_internal
{

struct Entry
{
  struct MemoryReference
  {
    long long memory_id = 0;
    long long start_ts = 0;
    std::string kind;
    std::string source_id;
  };

  long long embedding_id = 0;
  long long base_embedding_id = 0;
  long long memory_id = 0;
  long long start_ts = 0;
  std::string kind;
  std::string source_id;
  Eigen::VectorXf embedding;
  std::vector<MemoryReference> memory_references;
  mutable std::optional<family_embedding_features_internal::Features>
      family_features;

  Entry () = default;

  Entry (long long embedding_id_value, long long memory_id_value,
         long long start_ts_value, std::string kind_value,
         std::string source_id_value, Eigen::VectorXf embedding_value,
         long long base_embedding_id_value = 0)
      : embedding_id (embedding_id_value),
        base_embedding_id (base_embedding_id_value),
        memory_id (memory_id_value),
        start_ts (start_ts_value), kind (std::move (kind_value)),
        source_id (std::move (source_id_value)),
        embedding (std::move (embedding_value))
  {
  }
};

inline const family_embedding_features_internal::Features &
FamilyFeatures (const Entry &entry)
{
  if (!entry.family_features)
    entry.family_features
        = family_embedding_features_internal::Build (entry.embedding);
  return *entry.family_features;
}

struct RankedEntry
{
  std::size_t index = 0;
  float distance = 0.0f;
};

inline bool
HasLongTermReference (const Entry &entry)
{
  return std::any_of (entry.memory_references.begin (),
                      entry.memory_references.end (), [] (const auto &reference) {
                        return reference.memory_id > 0
                               && reference.kind == "LONG_TERM";
                      });
}

inline bool
IsSupersessionCandidateEntry (const Entry &entry)
{
  return std::any_of (
      entry.memory_references.begin (), entry.memory_references.end (),
      [] (const auto &reference) {
        return reference.memory_id > 0
               && (reference.kind == "LONG_TERM"
                   || reference.kind == "ASSOCIATION");
      });
}

inline std::size_t
SupersessionReferenceCount (const Entry &entry)
{
  return static_cast<std::size_t> (std::count_if (
      entry.memory_references.begin (), entry.memory_references.end (),
      [] (const auto &reference) {
        return reference.memory_id > 0
               && (reference.kind == "LONG_TERM"
                   || reference.kind == "ASSOCIATION");
      }));
}

inline const Entry::MemoryReference *
FindSupersessionMemoryReference (const Entry &entry, long long memory_id)
{
  const auto reference = std::find_if (
      entry.memory_references.begin (), entry.memory_references.end (),
      [memory_id] (const auto &candidate) {
        return candidate.memory_id == memory_id
               && (candidate.kind == "LONG_TERM"
                   || candidate.kind == "ASSOCIATION");
      });
  return reference == entry.memory_references.end () ? nullptr
                                                      : &*reference;
}

struct State
{
  bool recovery_failed = false;
  bool current_surface_database_current = false;
  bool current_surface_search_current = false;
  bool processor_surface_complete = false;
  int embedding_dim = 0;
  std::vector<Entry> entries;
  std::vector<float> search;
  std::unordered_map<long long, std::size_t> embedding_index;
  std::vector<std::size_t> long_term_entry_indices;
  std::unordered_map<std::size_t, std::size_t> long_term_index_positions;
  std::vector<std::size_t> supersession_entry_indices;
  std::unordered_map<std::size_t, std::size_t>
      supersession_index_positions;
  std::unordered_map<long long, std::size_t>
      supersession_entry_by_memory;
  std::unordered_set<long long> supersession_population_mismatches;
  bool supersession_population_ambiguous = false;
  bool supersession_tie_order_equivalent = true;
  bool supersession_embedding_fanout = false;
  long long supersession_max_embedding_id = 0;
  long long supersession_max_memory_id = 0;
  std::vector<Entry> current_entries;
  std::vector<float> current_search;
  std::unordered_map<long long, std::size_t> current_memory_index;
  mutable std::vector<float> distance_scratch;
  mutable std::vector<RankedEntry> ranked_scratch;
};

inline bool
CurrentMatchesSupersessionEntry (const State &state, long long memory_id)
{
  const auto historical = state.supersession_entry_by_memory.find (memory_id);
  const auto current = state.current_memory_index.find (memory_id);
  if (historical == state.supersession_entry_by_memory.end ()
      || current == state.current_memory_index.end ()
      || historical->second >= state.entries.size ()
      || current->second >= state.current_entries.size ())
    return false;
  const auto &base = state.entries[historical->second];
  const auto &surface = state.current_entries[current->second];
  return surface.base_embedding_id == base.embedding_id
         && surface.embedding.size () == base.embedding.size ()
         && surface.embedding.isApprox (base.embedding, 0.0f);
}

inline void
RefreshSupersessionPopulationMismatch (State &state, long long memory_id)
{
  const bool has_historical
      = state.supersession_entry_by_memory.count (memory_id) != 0;
  const bool has_current = state.current_memory_index.count (memory_id) != 0;
  if (!has_historical && !has_current)
    {
      state.supersession_population_mismatches.erase (memory_id);
      return;
    }
  if (CurrentMatchesSupersessionEntry (state, memory_id))
    state.supersession_population_mismatches.erase (memory_id);
  else
    state.supersession_population_mismatches.insert (memory_id);
}

inline bool
CurrentPopulationCoversHistorical (const State &state,
                                   long long excluded_memory_id)
{
  if (state.supersession_population_ambiguous
      || state.supersession_embedding_fanout
      || !state.supersession_tie_order_equivalent)
    return false;
  if (state.supersession_population_mismatches.empty ())
    return true;
  return state.supersession_population_mismatches.size () == 1
         && state.supersession_population_mismatches.count (
                excluded_memory_id)
                == 1;
}

struct SupersessionCandidate
{
  long long memory_id = 0;
  const Eigen::VectorXf *borrowed_embedding = nullptr;
  std::optional<Eigen::VectorXf> owned_embedding;

  const Eigen::VectorXf *
  Embedding () const
  {
    return borrowed_embedding != nullptr
               ? borrowed_embedding
               : (owned_embedding ? &*owned_embedding : nullptr);
  }
};

struct SupersessionCandidateRows
{
  std::shared_ptr<const State> cache_owner;
  std::vector<SupersessionCandidate> rows;
};

inline double
SupersessionCosineSimilarity (const Eigen::VectorXf &query,
                              double query_norm,
                              const Eigen::VectorXf &target)
{
  if (query.size () == 0 || query.size () != target.size ())
    {
      return 0.0;
    }
  const double target_norm = target.norm ();
  if (query_norm == 0.0 || target_norm == 0.0)
    {
      return 0.0;
    }
  return query.dot (target) / (query_norm * target_norm);
}

struct RegistryState
{
  std::mutex mutex;
  std::unordered_map<const ProcessorContext *, std::shared_ptr<State>> states;
};

inline RegistryState &
Registry ()
{
  // Avoid static-destruction ordering hazards for process-lifetime processors.
  // Entries are still erased by SignalProcessor teardown; only the empty
  // registry owner itself intentionally survives until process exit.
  static RegistryState *registry = new RegistryState ();
  return *registry;
}

inline std::shared_ptr<State>
FindMutable (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  const auto it = registry.states.find (&ctx);
  return it == registry.states.end () ? nullptr : it->second;
}

inline std::shared_ptr<const State>
Find (const ProcessorContext &ctx)
{
  return FindMutable (ctx);
}

inline bool
RecoveryFailed (const ProcessorContext &ctx)
{
  const auto state = Find (ctx);
  return state && state->recovery_failed;
}

inline bool
CurrentSurfaceDatabaseCurrent (const ProcessorContext &ctx)
{
  const auto state = Find (ctx);
  return state && state->current_surface_database_current;
}

inline bool
CurrentSurfaceSearchCurrent (const ProcessorContext &ctx)
{
  const auto state = Find (ctx);
  return state && state->current_surface_search_current;
}

inline void
SetCurrentSurfaceDatabaseCurrent (const ProcessorContext &ctx,
                                  bool database_current)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  auto &state = registry.states[&ctx];
  if (!state)
    {
      state = std::make_shared<State> ();
    }
  state->current_surface_database_current = database_current;
}

inline void
SetProcessorSurfaceComplete (const ProcessorContext &ctx, bool complete)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  auto &state = registry.states[&ctx];
  if (!state)
    {
      state = std::make_shared<State> ();
    }
  state->processor_surface_complete = complete;
}

inline void
MarkRecoveryFailed (const ProcessorContext &ctx,
                    bool preserve_surface_state = false)
{
  auto state = std::make_shared<State> ();
  state->recovery_failed = true;
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  const auto existing = registry.states.find (&ctx);
  if (preserve_surface_state && existing != registry.states.end ()
      && existing->second)
    {
      state->current_surface_database_current
          = existing->second->current_surface_database_current;
      state->processor_surface_complete
          = existing->second->processor_surface_complete;
      // Preserve only the current-to-base lineage needed to reconstruct the
      // search surface.  The searchable vector buffer is intentionally not
      // copied, and recovery_failed keeps every cache reader fail-closed.
      state->current_entries = existing->second->current_entries;
      state->current_memory_index = existing->second->current_memory_index;
    }
  registry.states[&ctx] = std::move (state);
}

inline void
Erase (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  registry.states.erase (&ctx);
}

inline bool
Reset (const ProcessorContext &ctx, std::vector<Entry> entries,
       std::vector<Entry> current_entries = {})
{
  auto state = std::make_shared<State> ();
  std::unordered_map<long long, std::size_t> unique_embedding_index;
  state->entries.reserve (entries.size ());
  for (auto &entry : entries)
    {
      if (entry.memory_id > 0)
        {
          entry.memory_references.push_back (
              { entry.memory_id, entry.start_ts, entry.kind,
                entry.source_id });
        }
      const auto existing = unique_embedding_index.find (entry.embedding_id);
      if (existing == unique_embedding_index.end ())
        {
          unique_embedding_index.emplace (entry.embedding_id,
                                          state->entries.size ());
          state->entries.push_back (std::move (entry));
          continue;
        }
      auto &representative = state->entries[existing->second];
      if (entry.embedding.size () != representative.embedding.size ()
          || !entry.embedding.isApprox (representative.embedding, 0.0f))
        {
          Erase (ctx);
          return false;
        }
      representative.memory_references.insert (
          representative.memory_references.end (),
          std::make_move_iterator (entry.memory_references.begin ()),
          std::make_move_iterator (entry.memory_references.end ()));
    }
  for (auto &entry : state->entries)
    {
      std::sort (entry.memory_references.begin (),
                 entry.memory_references.end (), [] (const auto &a,
                                                      const auto &b) {
                   if (a.start_ts != b.start_ts)
                     {
                       return a.start_ts < b.start_ts;
                     }
                   return a.memory_id < b.memory_id;
                 });
      entry.memory_references.erase (
          std::unique (entry.memory_references.begin (),
                       entry.memory_references.end (), [] (const auto &a,
                                                           const auto &b) {
                         return a.memory_id == b.memory_id;
                       }),
          entry.memory_references.end ());
    }
  if (state->entries.empty () && current_entries.empty ())
    {
      state->current_surface_search_current = true;
      auto &registry = Registry ();
      std::lock_guard<std::mutex> lock (registry.mutex);
      registry.states[&ctx] = std::move (state);
      return true;
    }
  state->embedding_dim = static_cast<int> (
      state->entries.empty () ? current_entries.front ().embedding.size ()
                              : state->entries.front ().embedding.size ());
  if (state->embedding_dim <= 0)
    {
      Erase (ctx);
      return false;
    }
  state->search.reserve (state->entries.size ()
                         * static_cast<std::size_t> (state->embedding_dim));
  state->embedding_index.reserve (state->entries.size ());
  for (std::size_t index = 0; index < state->entries.size (); ++index)
    {
      const auto &entry = state->entries[index];
      if (entry.embedding_id <= 0
          || entry.embedding.size () != state->embedding_dim
          || !state->embedding_index.emplace (entry.embedding_id, index).second)
        {
          Erase (ctx);
          return false;
        }
      state->search.insert (state->search.end (), entry.embedding.data (),
                            entry.embedding.data () + entry.embedding.size ());
      if (HasLongTermReference (entry))
        {
          state->long_term_index_positions.emplace (
              index, state->long_term_entry_indices.size ());
          state->long_term_entry_indices.push_back (index);
        }
      if (IsSupersessionCandidateEntry (entry))
        {
          state->supersession_embedding_fanout
              = state->supersession_embedding_fanout
                || SupersessionReferenceCount (entry) > 1;
          state->supersession_index_positions.emplace (
              index, state->supersession_entry_indices.size ());
          state->supersession_entry_indices.push_back (index);
          for (const auto &reference : entry.memory_references)
            {
              if (reference.memory_id <= 0
                  || (reference.kind != "LONG_TERM"
                      && reference.kind != "ASSOCIATION"))
                continue;
              if (!state->supersession_entry_by_memory.emplace (
                      reference.memory_id, index)
                       .second)
                state->supersession_population_ambiguous = true;
            }
        }
    }
  std::vector<std::pair<long long, long long>> supersession_order;
  supersession_order.reserve (state->supersession_entry_by_memory.size ());
  for (const auto &[memory_id, index] :
       state->supersession_entry_by_memory)
    {
      if (index >= state->entries.size ())
        {
          Erase (ctx);
          return false;
        }
      supersession_order.emplace_back (state->entries[index].embedding_id,
                                       memory_id);
    }
  std::sort (supersession_order.begin (), supersession_order.end ());
  for (std::size_t index = 0; index < supersession_order.size (); ++index)
    {
      const auto &[embedding_id, memory_id] = supersession_order[index];
      if (index > 0
          && memory_id <= supersession_order[index - 1].second)
        state->supersession_tie_order_equivalent = false;
      state->supersession_max_embedding_id
          = std::max (state->supersession_max_embedding_id, embedding_id);
      state->supersession_max_memory_id
          = std::max (state->supersession_max_memory_id, memory_id);
    }
  state->current_entries = std::move (current_entries);
  state->current_memory_index.reserve (state->current_entries.size ());
  state->current_search.reserve (
      state->current_entries.size ()
      * static_cast<std::size_t> (state->embedding_dim));
  for (std::size_t index = 0; index < state->current_entries.size (); ++index)
    {
      auto &entry = state->current_entries[index];
      if (entry.base_embedding_id <= 0)
        entry.base_embedding_id = entry.embedding_id;
      if (entry.memory_id <= 0 || entry.embedding_id <= 0
          || entry.embedding.size () != state->embedding_dim
          || !state->current_memory_index.emplace (entry.memory_id, index)
                  .second)
        {
          Erase (ctx);
          return false;
        }
      state->current_search.insert (state->current_search.end (),
                                    entry.embedding.data (),
                                    entry.embedding.data ()
                                        + entry.embedding.size ());
    }
  for (const auto &[memory_id, index] :
       state->supersession_entry_by_memory)
    {
      (void)index;
      state->supersession_population_mismatches.insert (memory_id);
    }
  for (const auto &entry : state->current_entries)
    RefreshSupersessionPopulationMismatch (*state, entry.memory_id);
  auto &registry = Registry ();
  state->current_surface_search_current = true;
  std::lock_guard<std::mutex> lock (registry.mutex);
  registry.states[&ctx] = std::move (state);
  return true;
}

inline void
UpsertCurrent (const ProcessorContext &ctx, Entry entry)
{
  auto state_owner = FindMutable (ctx);
  if (!state_owner)
    {
      return;
    }
  auto &state = *state_owner;
  if (state.recovery_failed)
    {
      return;
    }
  if (entry.memory_id <= 0 || entry.embedding_id <= 0
      || entry.embedding.size () != state.embedding_dim)
    {
      Erase (ctx);
      return;
    }
  const auto existing = state.current_memory_index.find (entry.memory_id);
  if (existing != state.current_memory_index.end ())
    {
      const std::size_t index = existing->second;
      if (entry.base_embedding_id <= 0)
        entry.base_embedding_id
            = state.current_entries[index].base_embedding_id;
      state.current_entries[index] = std::move (entry);
      const std::size_t offset
          = index * static_cast<std::size_t> (state.embedding_dim);
      std::copy (state.current_entries[index].embedding.data (),
                 state.current_entries[index].embedding.data ()
                     + state.embedding_dim,
                 state.current_search.begin ()
                     + static_cast<std::ptrdiff_t> (offset));
      RefreshSupersessionPopulationMismatch (
          state, state.current_entries[index].memory_id);
      return;
    }
  if (entry.base_embedding_id <= 0)
    entry.base_embedding_id = entry.embedding_id;
  state.current_memory_index.emplace (entry.memory_id,
                                      state.current_entries.size ());
  state.current_search.insert (state.current_search.end (),
                               entry.embedding.data (),
                               entry.embedding.data () + entry.embedding.size ());
  state.current_entries.push_back (std::move (entry));
  RefreshSupersessionPopulationMismatch (
      state, state.current_entries.back ().memory_id);
}

inline long long
BaseEmbeddingIdForMemory (const ProcessorContext &ctx, long long memory_id,
                          long long fallback_embedding_id)
{
  const auto state = Find (ctx);
  if (!state)
    return fallback_embedding_id;
  const auto index = state->current_memory_index.find (memory_id);
  if (index == state->current_memory_index.end ()
      || index->second >= state->current_entries.size ())
    return fallback_embedding_id;
  const long long base_embedding_id
      = state->current_entries[index->second].base_embedding_id;
  return base_embedding_id > 0 ? base_embedding_id : fallback_embedding_id;
}

inline void
RemoveCurrent (const ProcessorContext &ctx, long long memory_id)
{
  auto state_owner = FindMutable (ctx);
  if (!state_owner)
    {
      return;
    }
  auto &state = *state_owner;
  if (state.recovery_failed)
    {
      return;
    }
  const auto index_it = state.current_memory_index.find (memory_id);
  if (index_it == state.current_memory_index.end ())
    {
      return;
    }
  const std::size_t index = index_it->second;
  const std::size_t last = state.current_entries.size () - 1;
  if (index != last)
    {
      state.current_entries[index] = std::move (state.current_entries[last]);
      state.current_memory_index[state.current_entries[index].memory_id] = index;
      const std::size_t dst
          = index * static_cast<std::size_t> (state.embedding_dim);
      const std::size_t src
          = last * static_cast<std::size_t> (state.embedding_dim);
      std::copy_n (
          state.current_search.begin () + static_cast<std::ptrdiff_t> (src),
          state.embedding_dim,
          state.current_search.begin () + static_cast<std::ptrdiff_t> (dst));
    }
  state.current_memory_index.erase (memory_id);
  state.current_entries.pop_back ();
  state.current_search.resize (
      state.current_entries.size ()
      * static_cast<std::size_t> (state.embedding_dim));
  RefreshSupersessionPopulationMismatch (state, memory_id);
}

inline void
Append (const ProcessorContext &ctx, Entry entry)
{
  auto state_owner = FindMutable (ctx);
  if (!state_owner)
    {
      return;
    }
  auto &state = *state_owner;
  if (state.recovery_failed)
    {
      return;
    }
  if (entry.embedding_id <= 0 || entry.embedding.size () <= 0
      || (state.embedding_dim != 0
          && entry.embedding.size () != state.embedding_dim)
      || state.embedding_index.count (entry.embedding_id) != 0)
    {
      Erase (ctx);
      return;
    }
  if (state.embedding_dim == 0)
    {
      state.embedding_dim = static_cast<int> (entry.embedding.size ());
    }
  if (entry.memory_id > 0 && entry.memory_references.empty ())
    {
      entry.memory_references.push_back (
          { entry.memory_id, entry.start_ts, entry.kind, entry.source_id });
    }
  state.embedding_index.emplace (entry.embedding_id, state.entries.size ());
  const std::size_t index = state.entries.size ();
  state.search.insert (state.search.end (), entry.embedding.data (),
                       entry.embedding.data () + entry.embedding.size ());
  state.entries.push_back (std::move (entry));
  if (HasLongTermReference (state.entries.back ()))
    {
      state.long_term_index_positions.emplace (
          index, state.long_term_entry_indices.size ());
      state.long_term_entry_indices.push_back (index);
    }
  if (IsSupersessionCandidateEntry (state.entries.back ()))
    {
      const auto &candidate = state.entries.back ();
      state.supersession_embedding_fanout
          = state.supersession_embedding_fanout
            || SupersessionReferenceCount (candidate) > 1;
      state.supersession_index_positions.emplace (
          index, state.supersession_entry_indices.size ());
      state.supersession_entry_indices.push_back (index);
      long long candidate_max_memory_id = 0;
      for (const auto &reference : candidate.memory_references)
        {
          if (reference.memory_id <= 0
              || (reference.kind != "LONG_TERM"
                  && reference.kind != "ASSOCIATION"))
            continue;
          if (!state.supersession_entry_by_memory.emplace (
                  reference.memory_id, index)
                   .second)
            state.supersession_population_ambiguous = true;
          if (reference.memory_id <= state.supersession_max_memory_id)
            state.supersession_tie_order_equivalent = false;
          candidate_max_memory_id
              = std::max (candidate_max_memory_id, reference.memory_id);
          RefreshSupersessionPopulationMismatch (state,
                                                 reference.memory_id);
        }
      if (candidate.embedding_id <= state.supersession_max_embedding_id)
        state.supersession_tie_order_equivalent = false;
      state.supersession_max_embedding_id = std::max (
          state.supersession_max_embedding_id, candidate.embedding_id);
      state.supersession_max_memory_id = std::max (
          state.supersession_max_memory_id, candidate_max_memory_id);
    }
}

inline void
RemoveEmbedding (const ProcessorContext &ctx, long long embedding_id)
{
  auto state_owner = FindMutable (ctx);
  if (!state_owner)
    {
      return;
    }
  auto &state = *state_owner;
  if (state.recovery_failed)
    {
      return;
    }
  const auto index_it = state.embedding_index.find (embedding_id);
  if (index_it == state.embedding_index.end ())
    {
      return;
    }
  const std::size_t index = index_it->second;
  if (index >= state.entries.size () || state.embedding_dim <= 0)
    {
      Erase (ctx);
      return;
    }
  const std::size_t last = state.entries.size () - 1;
  std::vector<long long> removed_supersession_memory_ids;
  if (IsSupersessionCandidateEntry (state.entries[index]))
    for (const auto &reference : state.entries[index].memory_references)
      if (reference.memory_id > 0
          && (reference.kind == "LONG_TERM"
              || reference.kind == "ASSOCIATION"))
        removed_supersession_memory_ids.push_back (reference.memory_id);
  const auto long_term_position = state.long_term_index_positions.find (index);
  if (long_term_position != state.long_term_index_positions.end ())
    {
      const std::size_t position = long_term_position->second;
      const std::size_t last_position
          = state.long_term_entry_indices.size () - 1;
      if (position != last_position)
        {
          const std::size_t moved_index
              = state.long_term_entry_indices[last_position];
          state.long_term_entry_indices[position] = moved_index;
          state.long_term_index_positions[moved_index] = position;
        }
      state.long_term_entry_indices.pop_back ();
      state.long_term_index_positions.erase (long_term_position);
    }
  const auto supersession_position
      = state.supersession_index_positions.find (index);
  if (supersession_position != state.supersession_index_positions.end ())
    {
      const std::size_t position = supersession_position->second;
      const std::size_t last_position
          = state.supersession_entry_indices.size () - 1;
      if (position != last_position)
        {
          const std::size_t moved_index
              = state.supersession_entry_indices[last_position];
          state.supersession_entry_indices[position] = moved_index;
          state.supersession_index_positions[moved_index] = position;
        }
      state.supersession_entry_indices.pop_back ();
      state.supersession_index_positions.erase (supersession_position);
      for (const long long memory_id : removed_supersession_memory_ids)
        state.supersession_entry_by_memory.erase (memory_id);
    }
  if (index != last)
    {
      state.entries[index] = std::move (state.entries[last]);
      state.embedding_index[state.entries[index].embedding_id] = index;
      const auto moved_long_term_position
          = state.long_term_index_positions.find (last);
      if (moved_long_term_position != state.long_term_index_positions.end ())
        {
          state.long_term_entry_indices[moved_long_term_position->second]
              = index;
          state.long_term_index_positions.emplace (
              index, moved_long_term_position->second);
          state.long_term_index_positions.erase (moved_long_term_position);
        }
      const auto moved_supersession_position
          = state.supersession_index_positions.find (last);
      if (moved_supersession_position
          != state.supersession_index_positions.end ())
        {
          state.supersession_entry_indices[
              moved_supersession_position->second] = index;
          state.supersession_index_positions.emplace (
              index, moved_supersession_position->second);
          state.supersession_index_positions.erase (
              moved_supersession_position);
          if (IsSupersessionCandidateEntry (state.entries[index]))
            for (const auto &reference :
                 state.entries[index].memory_references)
              if (reference.memory_id > 0
                  && (reference.kind == "LONG_TERM"
                      || reference.kind == "ASSOCIATION"))
                state.supersession_entry_by_memory[reference.memory_id]
                    = index;
        }
      const std::size_t dst
          = index * static_cast<std::size_t> (state.embedding_dim);
      const std::size_t src
          = last * static_cast<std::size_t> (state.embedding_dim);
      std::copy_n (state.search.begin () + static_cast<std::ptrdiff_t> (src),
                   state.embedding_dim,
                   state.search.begin () + static_cast<std::ptrdiff_t> (dst));
    }
  state.embedding_index.erase (embedding_id);
  state.entries.pop_back ();
  state.search.resize (state.entries.size ()
      * static_cast<std::size_t> (state.embedding_dim));
  state.supersession_embedding_fanout = std::any_of (
      state.entries.begin (), state.entries.end (), [] (const auto &entry) {
        return SupersessionReferenceCount (entry) > 1;
      });
  for (const long long memory_id : removed_supersession_memory_ids)
    RefreshSupersessionPopulationMismatch (
        state, memory_id);
}

#ifdef CORTEXT_TESTING
inline std::size_t
RegistrySizeForTest ()
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  return registry.states.size ();
}
#endif

} // namespace cortext::operations::historical_surface_search_cache_internal
