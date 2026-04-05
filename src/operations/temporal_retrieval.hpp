#pragma once

#include "../store/facts.hpp"

#include <cstdint>
#include <optional>

namespace cortext::operations::temporal
{

enum class RetrievalMode
{
  Current,
  ValidAt,
  KnownAt
};

enum class FactBoostStrength
{
  Off,
  Weak,
  Strong
};

enum class StalePenaltyStrength
{
  Off,
  Moderate,
  Strong
};

enum class ProvenanceMode
{
  DirectLinkOnly,
  AnyFactMatch
};

enum class RoutineRecencyMode
{
  Balanced,
  RoutineBiased,
  RecencyBiased
};

struct RetrievalOverride
{
  std::optional<RetrievalMode> mode;
  std::optional<std::uint64_t> timestamp;
};

struct RetrievalAblationOverride
{
  std::optional<bool> fact_layer_enabled;
  std::optional<bool> history_enabled;
  std::optional<FactBoostStrength> fact_boost_strength;
  std::optional<StalePenaltyStrength> stale_penalty_strength;
  std::optional<ProvenanceMode> provenance_mode;
  std::optional<RoutineRecencyMode> routine_recency_mode;
};

class ScopedRetrievalOverride
{
public:
  ScopedRetrievalOverride (RetrievalMode mode,
                           std::optional<std::uint64_t> timestamp);
  ~ScopedRetrievalOverride ();

  ScopedRetrievalOverride (const ScopedRetrievalOverride &) = delete;
  ScopedRetrievalOverride &
  operator= (const ScopedRetrievalOverride &) = delete;

private:
  RetrievalOverride previous_;
};

class ScopedRetrievalAblationOverride
{
public:
  explicit ScopedRetrievalAblationOverride (
      const RetrievalAblationOverride &override);
  ~ScopedRetrievalAblationOverride ();

  ScopedRetrievalAblationOverride (
      const ScopedRetrievalAblationOverride &) = delete;
  ScopedRetrievalAblationOverride &
  operator= (const ScopedRetrievalAblationOverride &) = delete;

private:
  RetrievalAblationOverride previous_;
};

RetrievalOverride GetRetrievalOverride ();
void SetRetrievalOverride (RetrievalMode mode,
                           std::optional<std::uint64_t> timestamp);
void ClearRetrievalOverride ();
RetrievalAblationOverride GetRetrievalAblationOverride ();
void SetRetrievalAblationOverride (const RetrievalAblationOverride &override);
void ClearRetrievalAblationOverride ();
RetrievalMode ResolveRetrievalMode (RetrievalMode requested);
std::uint64_t ResolveRetrievalTimestamp (
    RetrievalMode requested, std::optional<std::uint64_t> requested_timestamp,
    std::uint64_t fallback_timestamp);
store::FactQueryMode ResolveFactQueryMode (store::FactQueryMode requested);
std::uint64_t ResolveFactQueryTimestamp (store::FactQueryMode requested,
                                         std::uint64_t requested_timestamp,
                                         std::uint64_t fallback_timestamp);
bool IsFactLayerEnabled ();
store::FactQueryMode ToFactQueryMode (RetrievalMode mode);
const char *ToString (RetrievalMode mode);
const char *ToString (FactBoostStrength strength);
const char *ToString (StalePenaltyStrength strength);
const char *ToString (ProvenanceMode mode);
const char *ToString (RoutineRecencyMode mode);

} // namespace cortext::operations::temporal
