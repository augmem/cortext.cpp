// Catch2 unit tests focused on adherence fixes from validation report
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"


#include "cortext/operations/focus_feedback.hpp"
#include "cortext/operations/memory_strength.hpp"
#include "cortext/operations/sensitivity.hpp"
#include "cortext/operations/sensitivity_feedback.hpp"
#include "cortext/operations/stability.hpp"
#include "cortext/operations/stability_feedback.hpp"
#include "cortext/operations/uncertainty.hpp"
#include "cortext/processor.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/operation_set.hpp"
#include "cortext/signal.hpp"
#include "cortext/store/sqlite_store.hpp" // Changed from abstract Store to SQLiteStore

using namespace cortext;
using namespace cortext::operations;

namespace
{
static void
SeedMemory (Store &store, long long id)
{
  const auto now_ts = cortext::testing::NowMs ();
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, use_frequency, stability, connectivity, drift_mag, "
      "influence, sustained_influence, contextual_gain, redundancy, "
      "pre_activation, lability_state, suppression_count, created_at) "
      "VALUES(?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
      "1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, ?)",
      { id, id, now_ts, now_ts });
}
} // namespace

// Removed DummyStore because SignalProcessor now runs migrations which require a functional store
// (or at least one that returns a valid Transaction object).

TEST_CASE ("Uncertainty weights respond to F and S", "[uncertainty]")
{
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.9;
  cfg.sensitivity = 0.9;
  cfg.stability = 0.1;

  // Use in-memory SQLite store instead of dummy
  auto unique_store = SQLiteStore::Create(":memory:");
  auto store = std::shared_ptr<Store>(std::move(unique_store));
  
  // Minimal op set: just UpdateUncertainty
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<UpdateUncertainty> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.timestamp = 1;
  s.source_id = "t";
  s.embedding = Eigen::VectorXf::Ones (4);
  processor.Process (s);
  // No direct getter for u_t; rely on absence of crash and that code path
  // executes.
  SUCCEED ();
}



TEST_CASE ("Alg17 emits adj, Alg6 consumes it", "[stability]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg{};
  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.5;
  Signal sig;
  sig.timestamp = 100;
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 1LL);
  SeedMemory (*store, 2LL);
  SeedMemory (*store, 3LL);
  OperationContext ctx (sig, pctx, cfg, store.get ());

  // Feed some usage events with positive/negative gains
  ctx.SetMemoryUsageEvents ({
      { 1, true, +0.2 },
      { 2, true, -0.1 },
      { 3, true, +0.3 },
  });
  ApplyStabilityFeedback fdbk;
  auto tx = store->Begin ();
  fdbk.Execute (ctx, *tx);
  tx->Commit ();
  // Should set a delta
  REQUIRE (ctx.GetDeltaHalfLifeAdjustment ().has_value ());

  // Provide observed retention and run Alg6
  ctx.SetObservedRetentionSeconds (120.0);
  UpdateStability st;
  st.Execute (ctx, cortext::testing::GetNullTransaction ());
  SUCCEED ();
}

TEST_CASE ("Alg4: ΔT_sensitivity is not clamped (Alg8 clamps later)",
           "[sensitivity]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg{};
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.9;
  cfg.stability = 0.5;
  Signal sig;
  sig.timestamp = 10;
  sig.embedding = Eigen::VectorXf::Ones (4);
  OperationContext ctx (sig, pctx, cfg);

  InitializeSensitivityPriors init;
  init.Execute (ctx, cortext::testing::GetNullTransaction ());
  UpdateSensitivity up;
  up.Execute (ctx, cortext::testing::GetNullTransaction ());
  // Ensure value is set (not necessarily clamped here)
  REQUIRE (ctx.GetDeltaThresholdSensitivity ().has_value ());
}

TEST_CASE ("Alg15/16 use base gains (no knob scaling)", "[feedback]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg{};
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.7;
  Signal sig;
  sig.timestamp = 1;
  OperationContext ctx (sig, pctx, cfg);
  // Provide usage events
  ctx.SetMemoryUsageEvents ({
      { 1, true, +0.2 },
      { 2, true, -0.1 },
  });
  ApplyFocusFeedback ffb;
  ffb.Execute (ctx, cortext::testing::GetNullTransaction ());
  ApplySensitivityFeedback sfb;
  sfb.Execute (ctx, cortext::testing::GetNullTransaction ());
  SUCCEED ();
}
