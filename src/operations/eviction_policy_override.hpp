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

struct EvictionPolicyOverride
{
  std::optional<int> trace_count;
  std::optional<bool> coupling_enabled;
  std::optional<double> coupling_strength;
  std::optional<ReinforcementStrength> reinforcement;
  std::optional<double> periphery_cutoff;
  std::optional<double> half_life;
  std::optional<bool> flashbulb_enabled;
  std::optional<WeightDistribution> weights;
  std::optional<bool> consolidation_gate_enabled;
  std::optional<bool> storage_gate_enabled;
  std::optional<long long> min_storage_bytes;
  std::optional<double> min_storage_fraction_of_available;
  std::optional<long long> used_storage_bytes;
};

class ScopedEvictionPolicyOverride
{
public:
  explicit ScopedEvictionPolicyOverride (
      const EvictionPolicyOverride &override);
  ~ScopedEvictionPolicyOverride ();

  ScopedEvictionPolicyOverride (
      const ScopedEvictionPolicyOverride &) = delete;
  ScopedEvictionPolicyOverride &
  operator= (const ScopedEvictionPolicyOverride &) = delete;

private:
  EvictionPolicyOverride previous_;
};

EvictionPolicyOverride GetEvictionPolicyOverride ();
void SetEvictionPolicyOverride (const EvictionPolicyOverride &override);
void ClearEvictionPolicyOverride ();

} // namespace cortext::operations::eviction
