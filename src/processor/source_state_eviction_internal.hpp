#pragma once

// Bounded per-source processor state.
//
// Every source id gets an AccumulatorState (accumulation, drift, pacing,
// refractory). Hosts that mint many ids (one per turn, per call, per device
// session) leave a closed ~4.5 KB state behind for each, which grew live heap
// linearly with ingest and made every full-context snapshot copy all of them.
// A closed state is what a source restarts from after an engine restart (it
// is never persisted), so evicting the least recently active closed sources
// beyond a fixed budget changes nothing for recently active sources and never
// drops an open unit.

#include "cortext/processor/accumulator_state.hpp"
#include "cortext/processor/processor_context.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace cortext::internal::source_state
{

/// Closed sources kept beyond the open ones (~1.1 MB at ~4.5 KB each).
constexpr std::size_t kIdleSourceCapacity = 256;

/// @brief True when the source holds no open unit: nothing accumulated, no
/// pending signal payloads, no pending interrupt.
inline bool
IsIdle (const AccumulatorState &state)
{
  return state.n_signals == 0 && state.signals.empty ()
         && state.blob_ids.empty () && !state.pending_interrupt_abort;
}

/// @brief Stamp @p source_id as the most recently active source, then evict
/// the least recently active idle sources (never @p source_id) until at most
/// @p idle_capacity idle sources remain. Returns the number evicted.
inline std::size_t
TouchAndEvictIdle (ProcessorContext &ctx, const std::string &source_id,
                   std::size_t idle_capacity = kIdleSourceCapacity)
{
  auto &states = ctx.accumulator_states;
  const uint64_t seq = ++ctx.accumulator_activity_seq;
  if (const auto it = states.find (source_id); it != states.end ())
    {
      it->second.last_active_seq = seq;
    }
  std::size_t idle = 0;
  for (const auto &[id, state] : states)
    {
      if (IsIdle (state))
        {
          ++idle;
        }
    }
  std::size_t evicted = 0;
  while (idle > idle_capacity)
    {
      auto oldest = states.end ();
      for (auto it = states.begin (); it != states.end (); ++it)
        {
          if (it->first == source_id || !IsIdle (it->second))
            {
              continue;
            }
          if (oldest == states.end ()
              || it->second.last_active_seq < oldest->second.last_active_seq)
            {
              oldest = it;
            }
        }
      if (oldest == states.end ())
        {
          break;
        }
      states.erase (oldest);
      --idle;
      ++evicted;
    }
  return evicted;
}

} // namespace cortext::internal::source_state
