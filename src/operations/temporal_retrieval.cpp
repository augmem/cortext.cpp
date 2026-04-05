#include "temporal_retrieval.hpp"

namespace cortext::operations::temporal
{

namespace
{

thread_local RetrievalOverride g_override;
thread_local RetrievalAblationOverride g_ablation_override;

} // namespace

ScopedRetrievalOverride::ScopedRetrievalOverride (
    RetrievalMode mode, std::optional<std::uint64_t> timestamp)
    : previous_ (GetRetrievalOverride ())
{
  SetRetrievalOverride (mode, timestamp);
}

ScopedRetrievalOverride::~ScopedRetrievalOverride ()
{
  g_override = previous_;
}

ScopedRetrievalAblationOverride::ScopedRetrievalAblationOverride (
    const RetrievalAblationOverride &override)
    : previous_ (GetRetrievalAblationOverride ())
{
  SetRetrievalAblationOverride (override);
}

ScopedRetrievalAblationOverride::~ScopedRetrievalAblationOverride ()
{
  g_ablation_override = previous_;
}

RetrievalOverride
GetRetrievalOverride ()
{
  return g_override;
}

void
SetRetrievalOverride (RetrievalMode mode,
                      std::optional<std::uint64_t> timestamp)
{
  g_override.mode = mode;
  g_override.timestamp = timestamp;
}

void
ClearRetrievalOverride ()
{
  g_override = {};
}

RetrievalAblationOverride
GetRetrievalAblationOverride ()
{
  return g_ablation_override;
}

void
SetRetrievalAblationOverride (const RetrievalAblationOverride &override)
{
  g_ablation_override = override;
}

void
ClearRetrievalAblationOverride ()
{
  g_ablation_override = {};
}

RetrievalMode
ResolveRetrievalMode (RetrievalMode requested)
{
  const auto override = GetRetrievalAblationOverride ();
  if (override.history_enabled.has_value () && !*override.history_enabled)
    {
      return RetrievalMode::Current;
    }
  return requested;
}

std::uint64_t
ResolveRetrievalTimestamp (RetrievalMode requested,
                           std::optional<std::uint64_t> requested_timestamp,
                           std::uint64_t fallback_timestamp)
{
  const auto override = GetRetrievalAblationOverride ();
  if (override.history_enabled.has_value () && !*override.history_enabled
      && requested != RetrievalMode::Current)
    {
      return fallback_timestamp;
    }
  return requested_timestamp.value_or (fallback_timestamp);
}

store::FactQueryMode
ResolveFactQueryMode (store::FactQueryMode requested)
{
  const auto override = GetRetrievalAblationOverride ();
  if (override.history_enabled.has_value () && !*override.history_enabled)
    {
      return store::FactQueryMode::Current;
    }
  return requested;
}

std::uint64_t
ResolveFactQueryTimestamp (store::FactQueryMode requested,
                           std::uint64_t requested_timestamp,
                           std::uint64_t fallback_timestamp)
{
  const auto override = GetRetrievalAblationOverride ();
  if (override.history_enabled.has_value () && !*override.history_enabled
      && requested != store::FactQueryMode::Current)
    {
      return fallback_timestamp;
    }
  return requested_timestamp == 0 ? fallback_timestamp : requested_timestamp;
}

bool
IsFactLayerEnabled ()
{
  const auto override = GetRetrievalAblationOverride ();
  return override.fact_layer_enabled.value_or (true);
}

store::FactQueryMode
ToFactQueryMode (RetrievalMode mode)
{
  switch (mode)
    {
    case RetrievalMode::Current:
      return store::FactQueryMode::Current;
    case RetrievalMode::ValidAt:
      return store::FactQueryMode::ValidAt;
    case RetrievalMode::KnownAt:
      return store::FactQueryMode::KnownAt;
    }
  return store::FactQueryMode::Current;
}

const char *
ToString (RetrievalMode mode)
{
  switch (mode)
    {
    case RetrievalMode::Current:
      return "current";
    case RetrievalMode::ValidAt:
      return "valid_at";
    case RetrievalMode::KnownAt:
      return "known_at";
    }
  return "current";
}

const char *
ToString (FactBoostStrength strength)
{
  switch (strength)
    {
    case FactBoostStrength::Off:
      return "off";
    case FactBoostStrength::Weak:
      return "weak";
    case FactBoostStrength::Strong:
      return "strong";
    }
  return "strong";
}

const char *
ToString (StalePenaltyStrength strength)
{
  switch (strength)
    {
    case StalePenaltyStrength::Off:
      return "off";
    case StalePenaltyStrength::Moderate:
      return "moderate";
    case StalePenaltyStrength::Strong:
      return "strong";
    }
  return "moderate";
}

const char *
ToString (ProvenanceMode mode)
{
  switch (mode)
    {
    case ProvenanceMode::DirectLinkOnly:
      return "direct_link_only";
    case ProvenanceMode::AnyFactMatch:
      return "any_fact_match";
    }
  return "direct_link_only";
}

const char *
ToString (RoutineRecencyMode mode)
{
  switch (mode)
    {
    case RoutineRecencyMode::Balanced:
      return "balanced";
    case RoutineRecencyMode::RoutineBiased:
      return "routine_biased";
    case RoutineRecencyMode::RecencyBiased:
      return "recency_biased";
    }
  return "balanced";
}

} // namespace cortext::operations::temporal
