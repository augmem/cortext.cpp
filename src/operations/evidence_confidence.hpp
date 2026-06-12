#pragma once

#include "cortext/core/algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cortext::operations::evidence
{

struct EvidenceStamp
{
  long long source_memory_id = 0;
  std::string evidence_type;

  bool
  operator< (const EvidenceStamp &other) const
  {
    if (source_memory_id != other.source_memory_id)
      {
        return source_memory_id < other.source_memory_id;
      }
    return evidence_type < other.evidence_type;
  }
};

struct EvidenceTruth
{
  double frequency = 1.0;
  double confidence = 0.5;
  double evidence_weight = 1.0;
  std::set<EvidenceStamp> stamps;
  double contradiction_mass = 0.0;
  long long projected_at_ts = 0;
};

enum class RevisionDecision
{
  Revised,
  Duplicate,
  ChoiceExisting,
  ChoiceIncoming,
  DampenedConflict
};

struct RevisionResult
{
  RevisionDecision decision = RevisionDecision::ChoiceExisting;
  EvidenceTruth truth;
  int independent_stamp_count = 0;
  int overlapping_stamp_count = 0;
  double old_confidence = 0.0;
  double incoming_confidence = 0.0;
  double revised_confidence = 0.0;
};

struct RevisionOptions
{
  double horizon = 1.0;
  double contradiction_saturation = 1.0;
  double conflict_dampening = 0.5;
  double duplicate_choice_margin = 0.02;
};

inline RevisionOptions
OptionsForKnobs (double focus, double sensitivity, double stability)
{
  (void)focus;
  RevisionOptions options;
  options.horizon = core::Lerp (0.75, 2.0, core::Clamp (stability, 0.0, 1.0));
  options.contradiction_saturation
      = core::Lerp (1.5, 0.75, core::Clamp (sensitivity, 0.0, 1.0));
  options.conflict_dampening
      = core::Lerp (0.35, 0.75, core::Clamp (sensitivity, 0.0, 1.0));
  options.duplicate_choice_margin
      = core::Lerp (0.05, 0.01, core::Clamp (sensitivity, 0.0, 1.0));
  return options;
}

inline const char *
ToString (RevisionDecision decision)
{
  switch (decision)
    {
    case RevisionDecision::Revised:
      return "revised";
    case RevisionDecision::Duplicate:
      return "duplicate";
    case RevisionDecision::ChoiceExisting:
      return "choice_existing";
    case RevisionDecision::ChoiceIncoming:
      return "choice_incoming";
    case RevisionDecision::DampenedConflict:
      return "dampened_conflict";
    }
  return "choice_existing";
}

inline double
ConfidenceToWeight (double confidence, double horizon = 1.0)
{
  const double h = std::max (horizon, 1e-9);
  const double c = core::Clamp (confidence, 0.0, 0.999999);
  return h * c / std::max (1e-9, 1.0 - c);
}

inline double
WeightToConfidence (double weight, double horizon = 1.0)
{
  const double h = std::max (horizon, 1e-9);
  const double w = std::max (0.0, weight);
  return core::Clamp (w / (w + h), 0.0, 0.999999);
}

inline EvidenceTruth
MakeTruth (double frequency, double confidence,
           const std::vector<EvidenceStamp> &stamps,
           double contradiction_mass = 0.0, long long projected_at_ts = 0,
           double horizon = 1.0)
{
  EvidenceTruth truth;
  truth.frequency = core::Clamp (frequency, 0.0, 1.0);
  truth.confidence = core::Clamp (confidence, 0.0, 0.999999);
  truth.evidence_weight = ConfidenceToWeight (truth.confidence, horizon);
  truth.contradiction_mass = std::max (0.0, contradiction_mass);
  truth.projected_at_ts = projected_at_ts;
  for (const auto &stamp : stamps)
    {
      if (stamp.source_memory_id > 0)
        {
          truth.stamps.insert (stamp);
        }
    }
  return truth;
}

inline EvidenceTruth
ProjectConfidence (const EvidenceTruth &truth, long long from_ts, long long to_ts,
                   double half_life_ms, double horizon = 1.0)
{
  EvidenceTruth projected = truth;
  if (to_ts <= from_ts || half_life_ms <= 0.0)
    {
      projected.projected_at_ts = to_ts;
      return projected;
    }

  const double age = static_cast<double> (to_ts - from_ts);
  const double decay = std::exp (-std::log (2.0) * age / half_life_ms);
  projected.evidence_weight = std::max (0.0, truth.evidence_weight * decay);
  projected.confidence = WeightToConfidence (projected.evidence_weight, horizon);
  projected.projected_at_ts = to_ts;
  return projected;
}

inline double
DampenConfidenceForContradiction (double confidence, double contradiction_mass,
                                  const RevisionOptions &options = {})
{
  const double contradiction_norm = core::Clamp (
      contradiction_mass / std::max (options.contradiction_saturation, 1e-9),
      0.0, 1.0);
  const double multiplier
      = core::Clamp (1.0 - options.conflict_dampening * contradiction_norm,
                     0.0, 1.0);
  return WeightToConfidence (
      ConfidenceToWeight (confidence, options.horizon) * multiplier,
      options.horizon);
}

inline RevisionResult
Revise (const EvidenceTruth &existing, const EvidenceTruth &incoming,
        bool conflict = false, const RevisionOptions &options = {})
{
  RevisionResult result;
  result.truth = existing;
  result.old_confidence = existing.confidence;
  result.incoming_confidence = incoming.confidence;

  std::set<EvidenceStamp> union_stamps = existing.stamps;
  int overlap_count = 0;
  int independent_count = 0;
  for (const auto &stamp : incoming.stamps)
    {
      if (existing.stamps.find (stamp) != existing.stamps.end ())
        {
          ++overlap_count;
        }
      else
        {
          ++independent_count;
          union_stamps.insert (stamp);
        }
    }
  result.independent_stamp_count = independent_count;
  result.overlapping_stamp_count = overlap_count;

  const double existing_weight = std::max (
      existing.evidence_weight,
      ConfidenceToWeight (existing.confidence, options.horizon));
  const double incoming_weight = std::max (
      incoming.evidence_weight,
      ConfidenceToWeight (incoming.confidence, options.horizon));
  const double contradiction_mass
      = std::max (0.0, existing.contradiction_mass)
        + std::max (0.0, incoming.contradiction_mass)
        + (conflict ? incoming_weight : 0.0);

  if (conflict)
    {
      result.decision = RevisionDecision::DampenedConflict;
      const double combined_weight = existing_weight + incoming_weight;
      const double chosen_weight = std::max (existing_weight, incoming_weight);
      const double raw_frequency
          = combined_weight > 1e-9
                ? (existing.frequency * existing_weight
                   + incoming.frequency * incoming_weight)
                      / combined_weight
                : existing.frequency;
      result.truth.frequency = core::Clamp (raw_frequency, 0.0, 1.0);
      result.truth.evidence_weight
          = chosen_weight
            * core::Clamp (
                1.0 - options.conflict_dampening
                          * core::Clamp (
                              contradiction_mass
                                  / std::max (options.contradiction_saturation,
                                              1e-9),
                              0.0, 1.0),
                0.0, 1.0);
      result.truth.confidence
          = WeightToConfidence (result.truth.evidence_weight, options.horizon);
      result.truth.stamps = std::move (union_stamps);
      result.truth.contradiction_mass = contradiction_mass;
      result.truth.projected_at_ts = std::max (existing.projected_at_ts,
                                               incoming.projected_at_ts);
      result.revised_confidence = result.truth.confidence;
      return result;
    }

  if (independent_count > 0)
    {
      result.decision = RevisionDecision::Revised;
      const double combined_weight = existing_weight + incoming_weight;
      result.truth.frequency
          = combined_weight > 1e-9
                ? core::Clamp (
                      (existing.frequency * existing_weight
                       + incoming.frequency * incoming_weight)
                          / combined_weight,
                      0.0, 1.0)
                : existing.frequency;
      result.truth.evidence_weight = combined_weight;
      result.truth.confidence
          = WeightToConfidence (combined_weight, options.horizon);
      result.truth.stamps = std::move (union_stamps);
      result.truth.contradiction_mass = contradiction_mass;
      result.truth.projected_at_ts = std::max (existing.projected_at_ts,
                                               incoming.projected_at_ts);
      result.revised_confidence = result.truth.confidence;
      return result;
    }

  if (incoming.confidence > existing.confidence + options.duplicate_choice_margin)
    {
      result.decision = RevisionDecision::ChoiceIncoming;
      result.truth = incoming;
      result.truth.stamps = std::move (union_stamps);
      result.truth.contradiction_mass = contradiction_mass;
    }
  else
    {
      result.decision = overlap_count > 0 ? RevisionDecision::Duplicate
                                          : RevisionDecision::ChoiceExisting;
      result.truth = existing;
      result.truth.stamps = std::move (union_stamps);
      result.truth.contradiction_mass = contradiction_mass;
    }
  result.revised_confidence = result.truth.confidence;
  return result;
}

} // namespace cortext::operations::evidence
