#pragma once

#include "cortext/core/algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace cortext::operations::cognitive
{

inline constexpr const char *kAttentionLedgerFlag
    = "CORTEXT_ENABLE_COG_ATTENTION_LEDGER";
inline constexpr const char *kPacketCompetitionFlag
    = "CORTEXT_ENABLE_COG_PACKET_COMPETITION";
inline constexpr const char *kCueRarityFlag = "CORTEXT_ENABLE_COG_CUE_RARITY";
inline constexpr const char *kUsefulnessFlag = "CORTEXT_ENABLE_COG_USEFULNESS";
inline constexpr const char *kEvidenceLanesFlag
    = "CORTEXT_ENABLE_COG_EVIDENCE_LANES";
inline constexpr const char *kFactorFusionFlag
    = "CORTEXT_ENABLE_COG_FACTOR_FUSION";

inline bool
EnvFlagEnabled (const char *name)
{
  const char *value = std::getenv (name);
  if (value == nullptr)
    {
      return false;
    }
  const std::string text = value;
  return !(text.empty () || text == "0" || text == "false"
           || text == "FALSE");
}

struct AttentionLedgerInput
{
  double semantic_relevance = 0.0;
  double context_relevance = 0.0;
  double graph_support = 0.0;
  double fact_support = 0.0;
  double source_confidence = 0.0;
  double evidence_confidence = 0.0;
  double retrieved_count = 0.0;
  double used_count = 0.0;
  double persistence_intent = 0.0;
  double negative_evidence = 0.0;
};

struct BoundedAttentionLedger
{
  double transient_activation = 0.0;
  double durable_importance = 0.0;
  double persistence_intent = 0.0;
  double evidence_confidence = 0.0;
  double total = 0.0;
};

inline BoundedAttentionLedger
BuildBoundedAttentionLedger (const AttentionLedgerInput &input)
{
  BoundedAttentionLedger ledger;
  ledger.transient_activation = core::Clamp (
      0.65 * core::Clamp (input.semantic_relevance, 0.0, 1.0)
          + 0.25 * core::Clamp (input.context_relevance, 0.0, 1.0)
          + 0.10 * core::Clamp (input.graph_support, 0.0, 1.0),
      0.0, 1.0);
  const double use_mass
      = std::log1p (std::max (0.0, input.retrieved_count)
                    + 2.0 * std::max (0.0, input.used_count))
        / std::log1p (24.0);
  ledger.durable_importance = core::Clamp (
      0.45 * core::Clamp (input.fact_support, 0.0, 1.0)
          + 0.25 * core::Clamp (input.source_confidence, 0.0, 1.0)
          + 0.30 * core::Clamp (use_mass, 0.0, 1.0),
      0.0, 1.0);
  ledger.persistence_intent
      = core::Clamp (0.65 * core::Clamp (input.persistence_intent, 0.0, 1.0)
                         + 0.35 * ledger.durable_importance,
                     0.0, 1.0);
  ledger.evidence_confidence
      = core::Clamp (input.evidence_confidence, 0.0, 1.0);
  ledger.total = core::Clamp (
      0.38 * ledger.transient_activation + 0.30 * ledger.durable_importance
          + 0.17 * ledger.persistence_intent
          + 0.15 * ledger.evidence_confidence
          - 0.25 * core::Clamp (input.negative_evidence, 0.0, 1.0),
      0.0, 1.0);
  return ledger;
}

struct PacketProposal
{
  std::string id;
  std::string refractory_key;
  double activation = 0.0;
  double evidence = 0.0;
  double novelty = 0.0;
  double source_diversity = 0.0;
  int member_count = 0;
};

struct PacketProposalSelection
{
  int index = -1;
  double score = 0.0;
  double refractory_penalty = 0.0;
};

inline double
PacketProposalScore (const PacketProposal &proposal)
{
  const double diversity = core::Clamp (proposal.source_diversity / 4.0, 0.0,
                                        1.0);
  const double size_penalty = proposal.member_count > 0
                                  ? 0.02 * std::max (0, proposal.member_count - 3)
                                  : 0.0;
  return core::Clamp (0.42 * core::Clamp (proposal.activation, 0.0, 1.0)
                          + 0.34 * core::Clamp (proposal.evidence, 0.0, 1.0)
                          + 0.18 * core::Clamp (proposal.novelty, 0.0, 1.0)
                          + 0.06 * diversity - size_penalty,
                      0.0, 1.0);
}

inline PacketProposalSelection
SelectPacketProposal (const std::vector<PacketProposal> &proposals,
                      const std::set<std::string> &recent_refractory_keys,
                      double refractory_penalty = 0.35)
{
  PacketProposalSelection selected;
  for (int i = 0; i < static_cast<int> (proposals.size ()); ++i)
    {
      const auto &proposal = proposals[static_cast<std::size_t> (i)];
      const double penalty
          = (!proposal.refractory_key.empty ()
             && recent_refractory_keys.count (proposal.refractory_key) > 0)
                ? refractory_penalty
                : 0.0;
      const double score = PacketProposalScore (proposal) - penalty;
      if (selected.index < 0 || score > selected.score)
        {
          selected.index = i;
          selected.score = score;
          selected.refractory_penalty = penalty;
        }
    }
  return selected;
}

struct CueEvidence
{
  double support = 0.0;
  double document_frequency = 1.0;
  bool negative = false;
};

inline double
CueRarityWeight (double document_count, double document_frequency)
{
  const double n = std::max (1.0, document_count);
  const double df = core::Clamp (document_frequency, 1.0, n);
  return core::Clamp (std::log ((n + 1.0) / (df + 1.0)) / std::log (n + 1.0),
                      0.0, 1.0);
}

inline double
CueRarityScore (double base_score, const std::vector<CueEvidence> &cues,
                double document_count, double negative_scale = 0.45)
{
  double positive = 0.0;
  double negative = 0.0;
  for (const auto &cue : cues)
    {
      const double weighted
          = core::Clamp (cue.support, 0.0, 1.0)
            * CueRarityWeight (document_count, cue.document_frequency);
      if (cue.negative)
        {
          negative += weighted;
        }
      else
        {
          positive += weighted;
        }
    }
  return core::Clamp (base_score + 0.35 * positive - negative_scale * negative,
                      0.0, 1.0);
}

inline double
UsefulnessScore (double retrieved_count, double selected_count,
                 double feedback_count, double age_ms, double half_life_ms)
{
  const double evidence_count = std::max (0.0, retrieved_count)
                                + 2.0 * std::max (0.0, selected_count)
                                + 3.0 * std::max (0.0, feedback_count);
  const double support
      = core::Clamp (std::log1p (evidence_count) / std::log1p (32.0), 0.0,
                     1.0);
  const double decay
      = half_life_ms > 0.0
            ? std::exp (-std::log (2.0) * std::max (0.0, age_ms)
                        / half_life_ms)
            : 1.0;
  return core::Clamp (support * decay, 0.0, 1.0);
}

inline double
UsefulnessRankScore (double base_score, double usefulness, double weight = 0.28)
{
  return core::Clamp (base_score + weight * core::Clamp (usefulness, 0.0, 1.0),
                      0.0, 1.0);
}

inline double
FuseExplicitImplicitEvidence (double explicit_fact_confidence,
                              double implicit_semantic_score,
                              double procedural_score,
                              double source_confidence)
{
  const double explicit_lane
      = core::Clamp (explicit_fact_confidence, 0.0, 1.0)
        * core::Clamp (source_confidence, 0.0, 1.0);
  const double implicit_lane = core::Clamp (implicit_semantic_score, 0.0, 1.0);
  const double procedural_lane = core::Clamp (procedural_score, 0.0, 1.0);
  return core::Clamp (0.54 * explicit_lane + 0.34 * implicit_lane
                          + 0.12 * procedural_lane,
                      0.0, 1.0);
}

struct FusionFactor
{
  double value = 0.5;
  double weight = 1.0;
};

inline double
ProductOfExpertsFusion (const std::vector<FusionFactor> &factors)
{
  if (factors.empty ())
    {
      return 0.0;
    }
  double log_sum = 0.0;
  double weight_sum = 0.0;
  for (const auto &factor : factors)
    {
      const double weight = std::max (0.0, factor.weight);
      if (weight <= 0.0)
        {
          continue;
        }
      log_sum += weight * std::log (core::Clamp (factor.value, 1e-6, 1.0));
      weight_sum += weight;
    }
  if (weight_sum <= 1e-9)
    {
      return 0.0;
    }
  return core::Clamp (std::exp (log_sum / weight_sum), 0.0, 1.0);
}

} // namespace cortext::operations::cognitive
