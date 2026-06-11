#include "../src/operations/cognitive_mechanisms.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <set>

namespace cognitive = cortext::operations::cognitive;

TEST_CASE ("Bounded attention ledger separates transient and durable value",
           "[operations][cognitive-mechanisms]")
{
  cognitive::AttentionLedgerInput transient;
  transient.semantic_relevance = 0.92;
  transient.context_relevance = 0.80;
  transient.source_confidence = 0.30;
  const auto transient_ledger
      = cognitive::BuildBoundedAttentionLedger (transient);

  cognitive::AttentionLedgerInput durable;
  durable.semantic_relevance = 0.50;
  durable.fact_support = 0.95;
  durable.source_confidence = 0.95;
  durable.evidence_confidence = 0.90;
  durable.used_count = 12.0;
  durable.persistence_intent = 0.80;
  const auto durable_ledger = cognitive::BuildBoundedAttentionLedger (durable);

  REQUIRE (transient_ledger.transient_activation
           > durable_ledger.transient_activation);
  REQUIRE (durable_ledger.durable_importance
           > transient_ledger.durable_importance);
  REQUIRE (durable_ledger.total > transient_ledger.total);
}

TEST_CASE ("Packet proposal competition applies refractory suppression",
           "[operations][cognitive-mechanisms]")
{
  std::vector<cognitive::PacketProposal> proposals{
    { "semantic-repeat", "same-source", 0.99, 0.72, 0.20, 2.0, 2 },
    { "fact-novel", "new-source", 0.72, 0.70, 0.40, 2.0, 2 },
  };

  const auto without_refractory
      = cognitive::SelectPacketProposal (proposals, {});
  REQUIRE (without_refractory.index == 0);

  const auto with_refractory
      = cognitive::SelectPacketProposal (proposals, { "same-source" });
  REQUIRE (with_refractory.index == 1);
  REQUIRE (cognitive::PacketProposalScore (proposals[0]) - 0.35
           < cognitive::PacketProposalScore (proposals[1]));
}

TEST_CASE ("Cue rarity boosts rare support and negative cues penalize",
           "[operations][cognitive-mechanisms]")
{
  const double common_score = cognitive::CueRarityScore (
      0.55, { { 0.80, 80.0, false } }, 100.0);
  const double rare_score = cognitive::CueRarityScore (
      0.55, { { 0.80, 1.0, false } }, 100.0);
  const double contradicted_score = cognitive::CueRarityScore (
      0.55, { { 0.80, 1.0, false }, { 0.80, 1.0, true } }, 100.0);

  REQUIRE (rare_score > common_score);
  REQUIRE (contradicted_score < rare_score);
}

TEST_CASE ("Usefulness score rewards selected feedback and decays with age",
           "[operations][cognitive-mechanisms]")
{
  const double unused = cognitive::UsefulnessScore (2.0, 0.0, 0.0, 0.0,
                                                    60'000.0);
  const double useful = cognitive::UsefulnessScore (2.0, 3.0, 2.0, 0.0,
                                                    60'000.0);
  const double stale = cognitive::UsefulnessScore (2.0, 3.0, 2.0, 240'000.0,
                                                   60'000.0);

  REQUIRE (useful > unused);
  REQUIRE (stale < useful);
}

TEST_CASE ("Explicit and implicit evidence lanes remain separable before fusion",
           "[operations][cognitive-mechanisms]")
{
  const double implicit_only
      = cognitive::FuseExplicitImplicitEvidence (0.0, 0.90, 0.0, 1.0);
  const double explicit_supported
      = cognitive::FuseExplicitImplicitEvidence (0.90, 0.45, 0.0, 0.95);
  const double explicit_unsupported
      = cognitive::FuseExplicitImplicitEvidence (0.90, 0.45, 0.0, 0.20);

  REQUIRE (explicit_supported > implicit_only);
  REQUIRE (explicit_unsupported < explicit_supported);
}

TEST_CASE ("Product-of-experts fusion rewards balanced support",
           "[operations][cognitive-mechanisms]")
{
  const double weighted_sum_spiky = (0.98 + 0.98 + 0.20) / 3.0;
  const double weighted_sum_balanced = (0.72 + 0.72 + 0.72) / 3.0;
  REQUIRE (weighted_sum_spiky == Catch::Approx (weighted_sum_balanced));

  const double spiky = cognitive::ProductOfExpertsFusion (
      { { 0.98, 1.0 }, { 0.98, 1.0 }, { 0.20, 1.0 } });
  const double balanced = cognitive::ProductOfExpertsFusion (
      { { 0.72, 1.0 }, { 0.72, 1.0 }, { 0.72, 1.0 } });

  REQUIRE (balanced > spiky);
}
