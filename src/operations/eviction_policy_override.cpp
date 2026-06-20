#include "eviction_policy_override.hpp"

namespace cortext::operations::eviction
{

namespace
{

thread_local EvictionPolicyOverride g_override;

} // namespace

ScopedEvictionPolicyOverride::ScopedEvictionPolicyOverride (
    const EvictionPolicyOverride &override)
    : previous_ (GetEvictionPolicyOverride ())
{
  SetEvictionPolicyOverride (override);
}

ScopedEvictionPolicyOverride::~ScopedEvictionPolicyOverride ()
{
  g_override = previous_;
}

EvictionPolicyOverride
GetEvictionPolicyOverride ()
{
  return g_override;
}

void
SetEvictionPolicyOverride (const EvictionPolicyOverride &override)
{
  g_override = override;
}

void
ClearEvictionPolicyOverride ()
{
  g_override = {};
}

} // namespace cortext::operations::eviction
