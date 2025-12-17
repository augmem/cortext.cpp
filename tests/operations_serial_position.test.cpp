#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/serial_position.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::ApplySerialPositionEffects;

TEST_CASE ("Alg26 exposes parameter derivations per algorithms.md",
           "[operations][serial_position][params]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 10;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.3;       // F
  cfg.sensitivity = 0.7; // S
  cfg.stability = 0.5;


  ApplySerialPositionEffects op;
  OperationContext ctx (s, pctx, cfg);
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Expectations via core helpers
  REQUIRE (ctx.GetSerialPrimacyWindow ()
           == core::SerialPrimacyWindow (cfg.focus));
  REQUIRE (ctx.GetSerialRecencyWindow ()
           == core::SerialRecencyWindow (cfg.focus));
  REQUIRE (ctx.GetSerialPrimacyBonus ()
           == Catch::Approx (core::SerialPrimacyBonus (cfg.sensitivity))
                  .epsilon (1e-9));
  REQUIRE (ctx.GetSerialRehearsalCurveDepth ()
           == Catch::Approx (core::SerialRehearsalCurveDepth (cfg.sensitivity))
                  .epsilon (1e-9));
  REQUIRE (ctx.GetSerialDistinctivenessThreshold ()
           == Catch::Approx (core::SerialDistinctivenessThreshold (cfg.focus))
                  .epsilon (1e-9));
  REQUIRE (
      ctx.GetSerialVonRestorffMultiplier ()
      == Catch::Approx (core::SerialVonRestorffMultiplier (cfg.sensitivity))
             .epsilon (1e-9));
  REQUIRE (ctx.GetSerialMiddleSuppression ()
           == Catch::Approx (
                  core::SerialMiddleSuppression (cfg.sensitivity, cfg.focus))
                  .epsilon (1e-9));
}

TEST_CASE ("Alg26 does not create any tables",
           "[operations][serial_position][no_db]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<ApplySerialPositionEffects> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = 123;
  s.source_id = "test";
  processor.Process (s);
  processor.Flush ();

  auto rows = store->Execute ("SELECT COUNT(*) AS c FROM sqlite_master WHERE "
                              "type='table' AND name = ?",
                              { std::string ("serial_position") });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 0LL);
}

TEST_CASE ("Alg26 edge cases and rounding",
           "[operations][serial_position][edges]")
{
  SECTION ("F=0, S=0")
  {
    REQUIRE (core::SerialPrimacyWindow (0.0) == 5);
    REQUIRE (core::SerialRecencyWindow (0.0) == 7);
    REQUIRE (core::SerialPrimacyBonus (0.0)
             == Catch::Approx (1.2).epsilon (1e-12));
    REQUIRE (core::SerialRehearsalCurveDepth (0.0)
             == Catch::Approx (0.2).epsilon (1e-12));
    REQUIRE (core::SerialDistinctivenessThreshold (0.0)
             == Catch::Approx (0.6).epsilon (1e-12));
    REQUIRE (core::SerialVonRestorffMultiplier (0.0)
             == Catch::Approx (1.5).epsilon (1e-12));
    REQUIRE (core::SerialMiddleSuppression (0.0, 0.0)
             == Catch::Approx (0.8).epsilon (1e-12));
  }

  SECTION ("F=1, S=1")
  {
    REQUIRE (core::SerialPrimacyWindow (1.0) == 2);
    REQUIRE (core::SerialRecencyWindow (1.0) == 3);
    REQUIRE (core::SerialPrimacyBonus (1.0)
             == Catch::Approx (2.0).epsilon (1e-12));
    REQUIRE (core::SerialRehearsalCurveDepth (1.0)
             == Catch::Approx (0.6).epsilon (1e-12));
    REQUIRE (core::SerialDistinctivenessThreshold (1.0)
             == Catch::Approx (0.8).epsilon (1e-12));
    REQUIRE (core::SerialVonRestorffMultiplier (1.0)
             == Catch::Approx (3.0).epsilon (1e-12));
    REQUIRE (core::SerialMiddleSuppression (1.0, 1.0)
             == Catch::Approx (0.0).epsilon (1e-12));
  }

  SECTION ("Rounding midpoints")
  {
    // F=0.5 → primacy round(3.5)=4, recency round(5)=5
    REQUIRE (core::SerialPrimacyWindow (0.5) == 4);
    REQUIRE (core::SerialRecencyWindow (0.5) == 5);
  }
}

TEST_CASE ("Alg26 SerialDetermineZone behavior",
           "[operations][serial_position][zones]")
{
  // F=0, N=20: primacy=5, recency=7 → recency positions start at 14
  {
    const double F = 0.0;
    const int N = 20;
    REQUIRE (core::SerialDetermineZone (F, 1, N, 0.0)
             == core::SerialZone::Primacy);
    REQUIRE (core::SerialDetermineZone (F, 14, N, 0.0)
             == core::SerialZone::Recency);
    // Distinctive in the middle when rarity > threshold (0.6 at F=0)
    REQUIRE (core::SerialDetermineZone (F, 10, N, 0.7)
             == core::SerialZone::Distinctive);
    // Middle when not distinctive and not in primacy/recency
    REQUIRE (core::SerialDetermineZone (F, 10, N, 0.1)
             == core::SerialZone::Middle);
  }

  // Overlap check at small N: F=1 → primacy=2, recency=3, N=5 → recency starts
  // at 3
  {
    const double F = 1.0;
    const int N = 5;
    REQUIRE (core::SerialDetermineZone (F, 2, N, 1.0)
             == core::SerialZone::Primacy); // primacy check wins first
    REQUIRE (core::SerialDetermineZone (F, 3, N, 0.0)
             == core::SerialZone::Recency);
    REQUIRE (core::SerialDetermineZone (F, 5, N, 0.0)
             == core::SerialZone::Recency);
  }
}
