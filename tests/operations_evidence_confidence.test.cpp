#include "../src/operations/evidence_confidence.hpp"

#include <catch2/catch_test_macros.hpp>

namespace evidence = cortext::operations::evidence;

TEST_CASE ("Evidence confidence increases with independent supporting stamps",
           "[operations][evidence-confidence]")
{
  const auto options = evidence::OptionsForKnobs (0.5, 0.5, 0.5);
  const auto first = evidence::MakeTruth (
      1.0, 0.60, { { 10LL, "summary" } }, 0.0, 1000LL,
      options.horizon);
  const auto second = evidence::MakeTruth (
      1.0, 0.60, { { 20LL, "summary" } }, 0.0, 1000LL,
      options.horizon);

  const auto revised = evidence::Revise (first, second, false, options);

  REQUIRE (revised.decision == evidence::RevisionDecision::Revised);
  REQUIRE (revised.independent_stamp_count == 1);
  REQUIRE (revised.overlapping_stamp_count == 0);
  REQUIRE (revised.truth.confidence > first.confidence);
  REQUIRE (revised.truth.frequency == 1.0);
  REQUIRE (revised.truth.stamps.size () == 2);
}

TEST_CASE ("Duplicate evidence stamps do not inflate confidence",
           "[operations][evidence-confidence]")
{
  const auto options = evidence::OptionsForKnobs (0.5, 0.5, 0.5);
  const auto first = evidence::MakeTruth (
      1.0, 0.60, { { 10LL, "summary" } }, 0.0, 1000LL,
      options.horizon);
  const auto duplicate = evidence::MakeTruth (
      1.0, 0.60, { { 10LL, "summary" } }, 0.0, 2000LL,
      options.horizon);

  const auto revised = evidence::Revise (first, duplicate, false, options);

  REQUIRE (revised.decision == evidence::RevisionDecision::Duplicate);
  REQUIRE (revised.independent_stamp_count == 0);
  REQUIRE (revised.overlapping_stamp_count == 1);
  REQUIRE (revised.truth.confidence == first.confidence);
  REQUIRE (revised.truth.stamps.size () == 1);
}

TEST_CASE ("Contradictory evidence dampens confidence",
           "[operations][evidence-confidence]")
{
  const auto options = evidence::OptionsForKnobs (0.5, 0.8, 0.5);
  const auto first = evidence::MakeTruth (
      1.0, 0.80, { { 10LL, "summary" } }, 0.0, 1000LL,
      options.horizon);
  const auto contradiction = evidence::MakeTruth (
      0.0, 0.80, { { 20LL, "summary" } }, 0.0, 1000LL,
      options.horizon);

  const auto revised = evidence::Revise (first, contradiction, true, options);

  REQUIRE (revised.decision == evidence::RevisionDecision::DampenedConflict);
  REQUIRE (revised.truth.confidence < first.confidence);
  REQUIRE (revised.truth.contradiction_mass > 0.0);
  REQUIRE (revised.truth.frequency < first.frequency);
}

TEST_CASE ("Projected confidence decays as evidence goes stale",
           "[operations][evidence-confidence]")
{
  const auto options = evidence::OptionsForKnobs (0.5, 0.5, 0.5);
  const auto fresh = evidence::MakeTruth (
      1.0, 0.80, { { 10LL, "summary" } }, 0.0, 1000LL,
      options.horizon);

  const auto stale = evidence::ProjectConfidence (
      fresh, 1000LL, 1000LL + 2'000LL, 1'000.0, options.horizon);

  REQUIRE (stale.confidence < fresh.confidence);
  REQUIRE (stale.evidence_weight < fresh.evidence_weight);
  REQUIRE (stale.projected_at_ts == 3000LL);
}
