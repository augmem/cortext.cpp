// Catch2 unit tests focused on adherence fixes from validation report
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"


#include "cortext/operations/sensitivity.hpp"
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
      "s_max, s_avg, strength, drift_mag, "
      "lability_state, suppression_count, created_at) "
      "VALUES(?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
      "1.0, 0.0, 0.0, 0, ?)",
      { id, id, now_ts, now_ts });
}
} // namespace

// Removed DummyStore because SignalProcessor now runs migrations which require a functional store
// (or at least one that returns a valid Transaction object).

TEST_CASE ("Uncertainty weights respond to F and S", "[uncertainty]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.9;
  cfg.sensitivity = 0.9;
  cfg.stability = 0.1;

  Signal s;
  s.timestamp = 1;
  s.source_id = "t";
  s.embedding = Eigen::VectorXf::Ones (4);
  s.embedding.normalize ();

  pctx.recent_scores = { 0.0, 1.0, 0.0, 1.0 };
  pctx.recent_context_embeddings.push_back (-s.embedding);
  OperationContext ctx (s, pctx, cfg);
  ctx.SetCoherence (0.2);
  ctx.SetMetric (operations::Metric::embedding_surprisal, 0.8);

  UpdateUncertainty op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.u_t > 0.0);
  REQUIRE (pctx.u_t <= 1.0);
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
