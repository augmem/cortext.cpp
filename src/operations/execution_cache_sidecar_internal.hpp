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

struct State
{
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
