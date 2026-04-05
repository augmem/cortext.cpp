#pragma once

#include <optional>

namespace cortext::operations::eviction
{

enum class ReinforcementStrength
{
  Off,
  Weak,
  Strong
};

enum class WeightDistribution
{
  Default,
  Equal
};

struct EvictionAblationOverride
{
  std::optional<int> trace_count;
  std::optional<bool> coupling_enabled;
  std::optional<double> coupling_strength;
  std::optional<ReinforcementStrength> reinforcement;
  std::optional<double> periphery_cutoff;
  std::optional<double> half_life;
  std::optional<bool> flashbulb_enabled;
  std::optional<WeightDistribution> weights;
  std::optional<bool> fact_floor_enabled;
  std::optional<bool> consolidation_gate_enabled;
};

class ScopedEvictionAblationOverride
{
public:
  explicit ScopedEvictionAblationOverride (
      const EvictionAblationOverride &override);
  ~ScopedEvictionAblationOverride ();

  ScopedEvictionAblationOverride (
      const ScopedEvictionAblationOverride &) = delete;
  ScopedEvictionAblationOverride &
  operator= (const ScopedEvictionAblationOverride &) = delete;

private:
  EvictionAblationOverride previous_;
};

EvictionAblationOverride GetEvictionAblationOverride ();
void SetEvictionAblationOverride (const EvictionAblationOverride &override);
void ClearEvictionAblationOverride ();

} // namespace cortext::operations::eviction
