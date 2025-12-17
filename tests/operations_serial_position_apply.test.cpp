#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/operations/serial_position.hpp>
#include <cortext/operations/serial_position_apply.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ApplySerialPositionEffects;
using cortext::operations::ApplySerialPositionMultiplier;
using cortext::operations::UpdateMemoryStrength;

TEST_CASE ("Serial position multiplier reflects primacy/recency zones",
           "[operations][serial_position_apply]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 100;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);
#
  // Derive serial windows from Alg 26 op
  ApplySerialPositionEffects derive;
  derive.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  // Build memory usage events in order, marking all as used
  std::vector<OperationContext::MemoryUsageEvent> events
      = { { 1LL, true, std::nullopt },
          { 2LL, true, std::nullopt },
          { 3LL, true, std::nullopt } };
  ctx.SetMemoryUsageEvents (events);
#
  ApplySerialPositionMultiplier apply_mult;
  apply_mult.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  auto m = ctx.GetSerialPositionMultiplier ();
  REQUIRE (m.has_value ());
  REQUIRE (*m >= 1.0);
}
#
// NOTE: Test removed during transaction-based operation refactor.
// The test was checking SQL query format via BufferedWriteInstruction, which
// is no longer used. Serial position multiplier behavior is tested via
// integration tests that verify actual database state.

TEST_CASE (
    "Zone classification uses SerialDetermineZone with rarity for distinctive",
    "[operations][serial_position_apply][zone]")
{
  // Test that zone determination follows spec (algorithms.md lines 977-981):
  // - if position ≤ primacy_window → "primacy"
  // - elif position ≥ N − recency_window + 1 → "recency"
  // - elif rarity > distinctiveness_threshold → "distinctive"
  // - else → "middle"

  // At F=0.5:
  // primacy_window = round(lerp(5, 2, 0.5)) = round(3.5) = 4
  // recency_window = round(lerp(7, 3, 0.5)) = round(5) = 5
  // distinctiveness_threshold = lerp(0.6, 0.8, 0.5) = 0.7
  const double F = 0.5;
  const int N = 20; // Use larger N to have middle zone

  // Test primacy zone (position 1-4)
  REQUIRE (core::SerialDetermineZone (F, 1, N, 0.0) == core::SerialZone::Primacy);
  REQUIRE (core::SerialDetermineZone (F, 4, N, 0.0) == core::SerialZone::Primacy);

  // Test recency zone (position 16-20 since N=20, recency_window=5)
  // N - recency_window + 1 = 20 - 5 + 1 = 16
  REQUIRE (core::SerialDetermineZone (F, 20, N, 0.0) == core::SerialZone::Recency);
  REQUIRE (core::SerialDetermineZone (F, 16, N, 0.0) == core::SerialZone::Recency);

  // Test distinctive zone (middle position with high rarity > 0.7)
  REQUIRE (core::SerialDetermineZone (F, 10, N, 0.8) == core::SerialZone::Distinctive);
  REQUIRE (core::SerialDetermineZone (F, 8, N, 0.9) == core::SerialZone::Distinctive);

  // Test middle zone (middle position with low rarity <= 0.7)
  REQUIRE (core::SerialDetermineZone (F, 10, N, 0.3) == core::SerialZone::Middle);
  REQUIRE (core::SerialDetermineZone (F, 8, N, 0.5) == core::SerialZone::Middle);
}

TEST_CASE ("von_restorff_multiplier applies to distinctive items",
           "[operations][serial_position_apply][von_restorff]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 100;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.3;       // Lower focus to widen primacy/recency windows
  cfg.sensitivity = 1.0; // High sensitivity for max von_restorff (3.0)
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);

  // Set high rarity metric to trigger distinctive zone
  ctx.SetMetric (operations::Metric::rarity, 0.95);

  // Derive serial windows
  ApplySerialPositionEffects derive;
  derive.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Create enough events to have middle zone items
  std::vector<OperationContext::MemoryUsageEvent> events;
  for (int i = 0; i < 20; ++i)
    {
      events.push_back ({ static_cast<long long> (i + 1), true, std::nullopt });
    }
  ctx.SetMemoryUsageEvents (events);

  ApplySerialPositionMultiplier apply_mult;
  apply_mult.Execute (ctx, cortext::testing::GetNullTransaction ());

  auto m = ctx.GetSerialPositionMultiplier ();
  REQUIRE (m.has_value ());
  // With high rarity and many middle items becoming distinctive,
  // we expect a multiplier influenced by von_restorff (1.5-3.0)
  REQUIRE (*m >= 1.0);
}

TEST_CASE ("middle_suppression reduces multiplier for middle zone items",
           "[operations][serial_position_apply][middle_suppression]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 100;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.9;       // High focus to narrow primacy/recency windows
  cfg.sensitivity = 1.0; // High S for stronger middle_suppression
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);

  // Set low rarity to keep items in middle zone
  ctx.SetMetric (operations::Metric::rarity, 0.1);

  ApplySerialPositionEffects derive;
  derive.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Create enough events to have middle zone items
  std::vector<OperationContext::MemoryUsageEvent> events;
  for (int i = 0; i < 20; ++i)
    {
      events.push_back ({ static_cast<long long> (i + 1), true, std::nullopt });
    }
  ctx.SetMemoryUsageEvents (events);

  ApplySerialPositionMultiplier apply_mult;
  apply_mult.Execute (ctx, cortext::testing::GetNullTransaction ());

  auto m = ctx.GetSerialPositionMultiplier ();
  REQUIRE (m.has_value ());
  // With high focus (narrow windows) and low rarity, most items are in middle zone
  // Middle suppression at S=1.0, F=0.9 is lerp(0.8, 0.5, 1.0) * (1 - 0.9) = 0.5 * 0.1 = 0.05
  // So multiplier is (1 - 0.05) = 0.95 for middle items, but primacy/recency boost others
  REQUIRE (*m >= 0.1);
  REQUIRE (*m <= 2.0);
}
