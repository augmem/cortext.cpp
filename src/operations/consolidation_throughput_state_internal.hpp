#pragma once

#include "cortext/consolidation_state.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/processor_context.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>

namespace cortext::operations::consolidation_throughput_state_internal
{

struct State
{
  double floor = 0.0;
  double peak = 0.0;
  bool initialized = false;
  bool armed = true;
};

struct RegistryState
{
  std::mutex mutex;
  std::unordered_map<const ProcessorContext *, State> states;
};

inline RegistryState &
Registry ()
{
  static RegistryState *registry = new RegistryState ();
  return *registry;
}

inline State
Find (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  const auto it = registry.states.find (&ctx);
  return it == registry.states.end () ? State {} : it->second;
}

inline void
Reset (const ProcessorContext &ctx, State state = {})
{
  if (!state.initialized)
    {
      state.floor = 0.0;
      state.peak = 0.0;
    }
  if (!std::isfinite (state.floor) || state.floor < 0.0)
    {
      state.floor = 0.0;
    }
  if (!std::isfinite (state.peak) || state.peak < state.floor)
    {
      state.peak = state.floor;
    }
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  registry.states[&ctx] = state;
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

inline double TriggerFraction (double focus, double sensitivity,
                               double stability);

inline double
RearmFraction (double focus, double sensitivity, double stability)
{
  const double recommended = TriggerFraction (focus, sensitivity, stability);
  return recommended
         + (1.0 - recommended) * core::Clamp (stability, 0.0, 1.0);
}

inline void
Observe (const ProcessorContext &ctx, double rate, double focus,
         double sensitivity, double stability)
{
  if (!std::isfinite (rate) || rate < 0.0)
    {
      return;
    }
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  auto &state = registry.states[&ctx];
  if (!state.initialized)
    {
      state.floor = rate;
      state.peak = rate;
      state.initialized = true;
      return;
    }

  state.peak = std::max (state.peak, rate);
  if (rate < state.floor)
    {
      state.floor = rate;
    }
  else
    {
      const double alpha = 1.0 / static_cast<double> (
          std::max (1, core::WRateSeconds (stability)));
      state.floor += alpha * (rate - state.floor);
    }
  state.floor = std::min (state.floor, state.peak);

  constexpr double kRangeEpsilon = 1e-9;
  const double range = state.peak - state.floor;
  if (!state.armed && range > kRangeEpsilon)
    {
      const double position = (rate - state.floor) / range;
      if (position >= RearmFraction (focus, sensitivity, stability))
        {
          state.armed = true;
        }
    }
}

inline void
Acknowledge (const ProcessorContext &ctx, double rate)
{
  if (!std::isfinite (rate) || rate < 0.0)
    {
      return;
    }
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  auto &state = registry.states[&ctx];
  // A successful maintenance transaction closes the observed excursion.
  // Start the next event-derived range at the acknowledged rate so a
  // historical peak cannot permanently suppress rearming after a regime
  // shift. No count or wall-clock horizon participates in this reset.
  state.floor = rate;
  state.peak = rate;
  state.initialized = true;
  state.armed = false;
}

inline double
TriggerFraction (double focus, double sensitivity, double stability)
{
  const double focus_factor
      = core::Lerp (1.10, 0.75, core::FocusBias (focus));
  const double sensitivity_factor
      = core::Lerp (0.80, 1.20, core::SensitivityBias (sensitivity));
  const double stability_factor
      = core::Lerp (1.10, 0.80, core::Clamp (stability, 0.0, 1.0));
  return core::Clamp (0.5 * focus_factor * sensitivity_factor
                          * stability_factor,
                      0.20, 0.70);
}

inline double
RequiredTriggerFraction (double focus, double sensitivity, double stability)
{
  return TriggerFraction (focus, sensitivity, stability)
         / core::ConsolidationEscalationMultiplier (stability);
}

inline ConsolidationState
Classify (const State &state, double current_rate, long long backlog,
          double focus, double sensitivity, double stability)
{
  constexpr double kRangeEpsilon = 1e-9;
  if (!state.armed || backlog <= 0 || !std::isfinite (current_rate))
    {
      return ConsolidationState::None;
    }
  const double range = state.peak - state.floor;
  if (!std::isfinite (range) || range <= kRangeEpsilon)
    {
      return ConsolidationState::None;
    }
  const double position = (current_rate - state.floor) / range;
  const double recommended
      = TriggerFraction (focus, sensitivity, stability);
  const double required
      = RequiredTriggerFraction (focus, sensitivity, stability);
  if (position <= required)
    {
      return ConsolidationState::Required;
    }
  if (position <= recommended)
    {
      return ConsolidationState::Recommended;
    }
  return ConsolidationState::None;
}

} // namespace cortext::operations::consolidation_throughput_state_internal
