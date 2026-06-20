#include "../src/operations/meta_learning_internal.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/core/constants.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/operations/stability.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <memory>

using namespace cortext;

namespace
{
Signal
MakeSignal (uint64_t timestamp = 1000)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (256);
  s.embedding[0] = 1.0f;
  s.timestamp = timestamp;
  s.source_id = "meta";
  return s;
}

void
InsertConstantCoeff (Store &store, const char *family, double value)
{
  store.Execute (
      "INSERT OR REPLACE INTO meta_learning_coeffs("
      "family, alpha_f, alpha_s, alpha_t, beta, a, b, update_count, updated_at) "
      "VALUES(?, 0.0, 0.0, 0.0, 0.0, ?, ?, 0, 0)",
      { std::string (family), value, value });
}

} // namespace

TEST_CASE ("Initialize priors load learned meta-learning values",
           "[operations][meta_learning]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  InsertConstantCoeff (*store, "focus_attention_width", 1.11);
  InsertConstantCoeff (*store, "sensitivity_rate_target", 3.25);
  InsertConstantCoeff (*store, "stability_hysteresis_band", 0.13);

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto tx = store->Begin ();
  const Signal signal = MakeSignal ();
  OperationContext ctx (signal, pctx, cfg, store.get ());

  operations::InitializeFocusPriors focus;
  operations::InitializeSensitivityPriors sensitivity;
  operations::InitializeStabilityPriors stability;
  focus.Execute (ctx, *tx);
  sensitivity.Execute (ctx, *tx);
  stability.Execute (ctx, *tx);

  REQUIRE (pctx.attention_width_prior == Catch::Approx (1.11));
  REQUIRE (pctx.base_rate_prior == Catch::Approx (3.25));
  REQUIRE (pctx.rate_target_prior == Catch::Approx (3.25));
  REQUIRE (pctx.rate_target == Catch::Approx (3.25));
  REQUIRE (pctx.hysteresis_band_prior == Catch::Approx (0.13));
}

TEST_CASE ("ApplyMetaLearning consolidates successful dynamic state into priors",
           "[operations][meta_learning]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto tx = store->Begin ();
  OperationContext init_ctx (MakeSignal (), pctx, cfg, store.get ());
  operations::InitializeFocusPriors focus;
  operations::InitializeSensitivityPriors sensitivity;
  operations::InitializeStabilityPriors stability;
  focus.Execute (init_ctx, *tx);
  sensitivity.Execute (init_ctx, *tx);
  stability.Execute (init_ctx, *tx);

  const double width_before = pctx.attention_width_prior;
  const double rate_before = pctx.rate_target_prior;
  const double hysteresis_before = pctx.hysteresis_band_prior;

  operations::ApplyMetaLearning meta;
  for (int i = 0; i < 8; ++i)
    {
      Signal s = MakeSignal (2000 + static_cast<uint64_t> (i) * 1000);
      OperationContext ctx (s, pctx, cfg, store.get ());
      pctx.attention_width = static_cast<double> (core::kAttentionWidthMin);
      pctx.rate_target = 4.8;
      pctx.rho_hat_prev = 4.8;
      pctx.hysteresis = 0.22;
      pctx.u_t = 0.1;
      pctx.last_used_flag = 1.0;
      pctx.delta_reward = 0.6;
      ctx.SetWriteDecision (true);
      meta.Execute (ctx, *tx);
    }

  REQUIRE (pctx.attention_width_prior < width_before);
  REQUIRE (pctx.rate_target_prior > rate_before);
  REQUIRE (pctx.hysteresis_band_prior > hysteresis_before);

  auto rows = tx->Execute (
      "SELECT family, update_count FROM meta_learning_coeffs ORDER BY family");
  REQUIRE (rows.size () == 3);
  for (const auto &row : rows)
    {
      REQUIRE (cortext::testing::GetInt64 (row, "update_count") == 8);
    }
}

TEST_CASE ("ApplyMetaLearning disable flag freezes learned coefficients",
           "[operations][meta_learning]")
{
  cortext::testing::ScopedEnvVar disable ("CORTEXT_DISABLE_META_LEARNING", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto tx = store->Begin ();
  const Signal signal = MakeSignal ();
  OperationContext ctx (signal, pctx, cfg, store.get ());
  operations::InitializeFocusPriors focus;
  operations::InitializeSensitivityPriors sensitivity;
  operations::InitializeStabilityPriors stability;
  focus.Execute (ctx, *tx);
  sensitivity.Execute (ctx, *tx);
  stability.Execute (ctx, *tx);

  const double width_before = pctx.attention_width_prior;
  const double rate_before = pctx.rate_target_prior;
  const double hysteresis_before = pctx.hysteresis_band_prior;

  pctx.attention_width = static_cast<double> (core::kAttentionWidthMin);
  pctx.rate_target = 4.5;
  pctx.rho_hat_prev = 4.5;
  pctx.hysteresis = 0.22;
  pctx.u_t = 0.1;
  pctx.last_used_flag = 1.0;
  pctx.delta_reward = 0.8;
  ctx.SetWriteDecision (true);

  operations::ApplyMetaLearning meta;
  meta.Execute (ctx, *tx);

  REQUIRE (pctx.attention_width_prior == Catch::Approx (width_before));
  REQUIRE (pctx.rate_target_prior == Catch::Approx (rate_before));
  REQUIRE (pctx.hysteresis_band_prior == Catch::Approx (hysteresis_before));

  auto rows = tx->Execute ("SELECT COUNT(*) AS c FROM meta_learning_coeffs");
  REQUIRE (cortext::testing::GetInt64 (rows[0], "c") == 0);
}
