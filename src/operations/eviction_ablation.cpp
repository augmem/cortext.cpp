#include "eviction_ablation.hpp"

namespace cortext::operations::eviction
{

namespace
{

thread_local EvictionAblationOverride g_override;

} // namespace

ScopedEvictionAblationOverride::ScopedEvictionAblationOverride (
    const EvictionAblationOverride &override)
    : previous_ (GetEvictionAblationOverride ())
{
  SetEvictionAblationOverride (override);
}

ScopedEvictionAblationOverride::~ScopedEvictionAblationOverride ()
{
  g_override = previous_;
}

EvictionAblationOverride
GetEvictionAblationOverride ()
{
  return g_override;
}

void
SetEvictionAblationOverride (const EvictionAblationOverride &override)
{
  g_override = override;
}

void
ClearEvictionAblationOverride ()
{
  g_override = {};
}

} // namespace cortext::operations::eviction
