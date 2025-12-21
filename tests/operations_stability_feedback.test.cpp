#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/stability.hpp>
#include <cortext/operations/stability_feedback.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::ApplyStabilityFeedback;
using cortext::operations::InitializeStabilityPriors;

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

TEST_CASE ("Alg17 positive contextual gain increases half_life",
           "[operations][stability_feedback]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.5;

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 1LL);

  // Initialize stability priors and seed dynamic state
  {
    OperationContext ctx (s, pctx, cfg);
    InitializeStabilityPriors init;
    init.Execute (ctx, cortext::testing::GetNullTransaction ());
  }
  const double prior_hl = pctx.half_life;

  OperationContext ctx (s, pctx, cfg);
  // Attach a positive contextual gain event
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 1LL;
  ev.used = true;
  ev.contextual_gain = 0.6; // positive
  ctx.SetMemoryUsageEvents ({ ev });

  ApplyStabilityFeedback op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
  // Alg17 now emits ΔHalfLife_adj_t for Alg6 to consume; apply Alg6
  REQUIRE (ctx.GetDeltaHalfLifeAdjustment ().has_value ());
  // Provide a retention observation and run Alg6
  ctx.SetObservedRetentionSeconds (120.0);
  cortext::operations::UpdateStability st;
  st.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.half_life > prior_hl);
  REQUIRE (
      pctx.periphery_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
  REQUIRE (
      pctx.salience_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
}

TEST_CASE ("Alg17 non-positive contextual gain decreases half_life",
           "[operations][stability_feedback]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.5;

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 2LL);

  // Initialize stability priors and seed dynamic state
  {
    OperationContext ctx (s, pctx, cfg);
    InitializeStabilityPriors init;
    init.Execute (ctx, cortext::testing::GetNullTransaction ());
  }
  const double prior_hl = pctx.half_life;

  OperationContext ctx (s, pctx, cfg);
  // Attach a negative contextual gain event
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 2LL;
  ev.used = true;
  ev.contextual_gain = -0.4; // negative
  ctx.SetMemoryUsageEvents ({ ev });

  ApplyStabilityFeedback op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
  // Alg17 now emits ΔHalfLife_adj_t for Alg6 to consume; apply Alg6
  REQUIRE (ctx.GetDeltaHalfLifeAdjustment ().has_value ());
  // Provide a retention observation and run Alg6
  ctx.SetObservedRetentionSeconds (120.0);
  cortext::operations::UpdateStability st;
  st.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.half_life < prior_hl);
  REQUIRE (
      pctx.periphery_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
  REQUIRE (
      pctx.salience_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
}
