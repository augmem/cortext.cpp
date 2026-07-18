#pragma once

#include "cortext/processor/processor_context.hpp"
#include "../store/mutation_audit_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cortext::operations::consolidation_epoch_profile_internal
{

struct Counters
{
  std::uint64_t accumulator_signal_count = 0;
  std::uint64_t working_memory_pending_signal_count = 0;
  std::uint64_t consolidation_dirty_memory_count = 0;
  std::uint64_t consolidation_dirty_association_count = 0;
  std::uint64_t consolidation_dirty_index_count = 0;
};

struct EventSnapshot
{
  std::uint64_t epoch_id = 0;
  std::uint64_t events_since_epoch_start = 0;
  Counters counters;
};

struct SealSnapshot
{
  std::uint64_t closed_epoch_id = 0;
  std::uint64_t sealed_epoch_event_count = 0;
  std::uint64_t sealed_epoch_mutation_count = 0;
  std::uint64_t sealed_mutation_identity_count = 0;
  bool sealed_mutation_identity_verified = false;
  Counters pre;
  Counters post;
};

struct TypedIdentitySets
{
  std::unordered_set<std::string> dirty_memories;
  std::unordered_set<std::string> dirty_associations;
  std::unordered_set<std::string> dirty_indexes;
};

struct State
{
  std::uint64_t epoch_id = 0;
  std::uint64_t events_since_epoch_start = 0;
  TypedIdentitySets trigger_identities;
  TypedIdentitySets hook_identities;
  bool trigger_journal_ready = true;
  std::unordered_map<long long, std::size_t> working_memory_baseline;
};

struct RegistryState
{
  std::mutex mutex;
  std::unordered_map<const ProcessorContext *, State> states;
};

inline RegistryState &
Registry ()
{
  static auto *registry = new RegistryState ();
  return *registry;
}

inline bool
Enabled ()
{
  const char *value = std::getenv ("CORTEXT_PROFILE_CONSOLIDATION_EPOCH");
  return value != nullptr && value[0] != '\0' && std::string_view (value) != "0"
         && std::string_view (value) != "false";
}

inline std::string
IdentityKey (const internal::SQLiteMutationIdentity &identity)
{
  return std::to_string (static_cast<int> (identity.kind)) + ":"
         + identity.logical_identity;
}

inline void
AddMutations (TypedIdentitySets &sets,
              const std::vector<internal::SQLiteMutationIdentity> &mutations)
{
  for (const auto &mutation : mutations)
    {
      const std::string key = IdentityKey (mutation);
      if (mutation.kind == internal::SQLiteMutationKind::Memory)
        sets.dirty_memories.insert (key);
      else if (mutation.kind == internal::SQLiteMutationKind::Association)
        sets.dirty_associations.insert (key);
      else if (mutation.kind == internal::SQLiteMutationKind::Index)
        sets.dirty_indexes.insert (key);
    }
}

inline bool
EqualIdentitySets (const TypedIdentitySets &lhs,
                   const TypedIdentitySets &rhs)
{
  return lhs.dirty_memories == rhs.dirty_memories
         && lhs.dirty_associations == rhs.dirty_associations
         && lhs.dirty_indexes == rhs.dirty_indexes;
}

inline std::uint64_t
IdentitySetCount (const TypedIdentitySets &sets)
{
  return static_cast<std::uint64_t> (sets.dirty_memories.size ())
         + static_cast<std::uint64_t> (sets.dirty_associations.size ())
         + static_cast<std::uint64_t> (sets.dirty_indexes.size ());
}

inline std::uint64_t
DistinctTypedMutationCount (
    const std::vector<internal::SQLiteMutationIdentity> &mutations)
{
  std::unordered_set<std::string> identities;
  identities.reserve (mutations.size ());
  for (const auto &mutation : mutations)
    identities.insert (IdentityKey (mutation));
  return static_cast<std::uint64_t> (identities.size ());
}

inline std::uint64_t
AccumulatorSignalCount (const ProcessorContext &ctx)
{
  std::uint64_t count = 0;
  for (const auto &[source_id, accumulator] : ctx.accumulator_states)
    {
      (void)source_id;
      count += static_cast<std::uint64_t> (accumulator.signals.size ());
    }
  return count;
}

inline long long
WorkingMemoryKey (const ProcessorContext::WMSlot &slot, std::size_t index)
{
  if (slot.memory_id > 0)
    return slot.memory_id;
  return std::numeric_limits<long long>::min ()
         + static_cast<long long> (index);
}

inline std::uint64_t
WorkingMemoryPendingSignalCount (const ProcessorContext &ctx,
                                 const State &state)
{
  std::uint64_t count = 0;
  for (std::size_t index = 0; index < ctx.wm_slots.size (); ++index)
    {
      const auto &slot = ctx.wm_slots[index];
      const auto baseline = state.working_memory_baseline.find (
          WorkingMemoryKey (slot, index));
      const std::size_t baseline_count
          = baseline == state.working_memory_baseline.end ()
                ? 0
                : baseline->second;
      if (slot.signal_records.size () > baseline_count)
        count += static_cast<std::uint64_t> (
            slot.signal_records.size () - baseline_count);
    }
  return count;
}

inline void
CaptureWorkingMemoryBaseline (const ProcessorContext &ctx, State &state)
{
  state.working_memory_baseline.clear ();
  state.working_memory_baseline.reserve (ctx.wm_slots.size ());
  for (std::size_t index = 0; index < ctx.wm_slots.size (); ++index)
    {
      const auto &slot = ctx.wm_slots[index];
      state.working_memory_baseline.emplace (
          WorkingMemoryKey (slot, index), slot.signal_records.size ());
    }
}

inline Counters
CurrentCounters (const ProcessorContext &ctx, const State &state)
{
  return {
    AccumulatorSignalCount (ctx),
    WorkingMemoryPendingSignalCount (ctx, state),
    static_cast<std::uint64_t> (
        state.trigger_identities.dirty_memories.size ()),
    static_cast<std::uint64_t> (
        state.trigger_identities.dirty_associations.size ()),
    static_cast<std::uint64_t> (
        state.trigger_identities.dirty_indexes.size ()),
  };
}

inline void
Reset (const ProcessorContext &ctx)
{
  if (!Enabled ())
    return;
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  State state;
  CaptureWorkingMemoryBaseline (ctx, state);
  registry.states[&ctx] = std::move (state);
}

inline EventSnapshot
ObserveEvent (const ProcessorContext &ctx,
              const internal::SQLiteMutationAuditBatch &mutations)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  auto &state = registry.states[&ctx];
  AddMutations (state.trigger_identities,
                mutations.committed_trigger_identities);
  AddMutations (state.hook_identities, mutations.committed_hook_identities);
  state.trigger_journal_ready
      = state.trigger_journal_ready && mutations.trigger_journal_ready;
  ++state.events_since_epoch_start;
  return { state.epoch_id, state.events_since_epoch_start,
           CurrentCounters (ctx, state) };
}

inline EventSnapshot
Current (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  const auto it = registry.states.find (&ctx);
  if (it == registry.states.end ())
    return {};
  return { it->second.epoch_id, it->second.events_since_epoch_start,
           CurrentCounters (ctx, it->second) };
}

inline SealSnapshot
Seal (const ProcessorContext &ctx,
      const internal::SQLiteMutationAuditBatch &mutations)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  auto &state = registry.states[&ctx];
  const Counters pre = CurrentCounters (ctx, state);
  State sealed = state;
  AddMutations (sealed.trigger_identities,
                mutations.committed_trigger_identities);
  AddMutations (sealed.hook_identities, mutations.committed_hook_identities);
  sealed.trigger_journal_ready
      = sealed.trigger_journal_ready && mutations.trigger_journal_ready;
  const std::uint64_t mutation_count
      = IdentitySetCount (sealed.trigger_identities);
  const std::uint64_t identity_count
      = IdentitySetCount (sealed.hook_identities);
  const bool identities_verified
      = sealed.trigger_journal_ready
        && EqualIdentitySets (sealed.trigger_identities,
                              sealed.hook_identities);
  const std::uint64_t closed_epoch_id = state.epoch_id;
  const std::uint64_t event_count = state.events_since_epoch_start;
  state.trigger_identities = {};
  state.hook_identities = {};
  state.trigger_journal_ready = true;
  CaptureWorkingMemoryBaseline (ctx, state);
  ++state.epoch_id;
  state.events_since_epoch_start = 0;
  return { closed_epoch_id, event_count, mutation_count, identity_count,
           identities_verified, pre, CurrentCounters (ctx, state) };
}

inline void
Erase (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  registry.states.erase (&ctx);
}

} // namespace cortext::operations::consolidation_epoch_profile_internal
