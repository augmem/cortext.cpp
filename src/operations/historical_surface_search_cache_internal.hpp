#pragma once

#include "cortext/processor/processor_context.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cortext::operations::historical_surface_search_cache_internal
{

struct Entry
{
  long long embedding_id = 0;
  long long memory_id = 0;
  long long start_ts = 0;
  std::string kind;
  std::string source_id;
  Eigen::VectorXf embedding;
};

struct RankedEntry
{
  std::size_t index = 0;
  float distance = 0.0f;
};

struct State
{
  int embedding_dim = 0;
  std::vector<Entry> entries;
  std::vector<float> search;
  std::unordered_map<long long, std::size_t> embedding_index;
  std::vector<Entry> current_entries;
  std::vector<float> current_search;
  std::unordered_map<long long, std::size_t> current_memory_index;
  mutable std::vector<float> distance_scratch;
  mutable std::vector<RankedEntry> ranked_scratch;
};

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
  state->entries = std::move (entries);
  if (state->entries.empty ())
    {
      auto &registry = Registry ();
      std::lock_guard<std::mutex> lock (registry.mutex);
      registry.states[&ctx] = std::move (state);
      return true;
    }
  state->embedding_dim = static_cast<int> (
      state->entries.front ().embedding.size ());
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
    }
  state->current_entries = std::move (current_entries);
  state->current_memory_index.reserve (state->current_entries.size ());
  state->current_search.reserve (
      state->current_entries.size ()
      * static_cast<std::size_t> (state->embedding_dim));
  for (std::size_t index = 0; index < state->current_entries.size (); ++index)
    {
      const auto &entry = state->current_entries[index];
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
  auto &registry = Registry ();
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
      state.current_entries[index] = std::move (entry);
      const std::size_t offset
          = index * static_cast<std::size_t> (state.embedding_dim);
      std::copy (state.current_entries[index].embedding.data (),
                 state.current_entries[index].embedding.data ()
                     + state.embedding_dim,
                 state.current_search.begin ()
                     + static_cast<std::ptrdiff_t> (offset));
      return;
    }
  state.current_memory_index.emplace (entry.memory_id,
                                      state.current_entries.size ());
  state.current_search.insert (state.current_search.end (),
                               entry.embedding.data (),
                               entry.embedding.data () + entry.embedding.size ());
  state.current_entries.push_back (std::move (entry));
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
  state.embedding_index.emplace (entry.embedding_id, state.entries.size ());
  state.search.insert (state.search.end (), entry.embedding.data (),
                       entry.embedding.data () + entry.embedding.size ());
  state.entries.push_back (std::move (entry));
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
  if (index != last)
    {
      state.entries[index] = std::move (state.entries[last]);
      state.embedding_index[state.entries[index].embedding_id] = index;
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
