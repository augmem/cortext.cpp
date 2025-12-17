#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/stability.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#
using namespace cortext;
using cortext::operations::InitializeStabilityPriors;
using cortext::operations::UpdateStability;
#
TEST_CASE ("InitializeStabilityPriors sets priors at T=0",
           "[operations][stability]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.stability = 0.0;

  OperationContext ctx (s, pctx, cfg);
#
  InitializeStabilityPriors op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  REQUIRE (pctx.hysteresis_band_prior
           == Catch::Approx (cortext::core::BaseBandPrior (0.0)));
  REQUIRE (pctx.half_life_prior
           == Catch::Approx (cortext::core::BaseHalfLifePrior (0.0)));
  REQUIRE (pctx.rate_decay_prior == Catch::Approx (0.60));
  REQUIRE (pctx.periphery_half_life_prior
           == Catch::Approx (
               cortext::core::ClampHalfLife (0.5 * pctx.half_life_prior)));
  REQUIRE (pctx.salience_half_life_prior
           == Catch::Approx (
               cortext::core::ClampHalfLife (0.5 * pctx.half_life_prior)));
  REQUIRE (pctx.drift_weight_prior == Catch::Approx (0.5));
}
#
TEST_CASE ("InitializeStabilityPriors sets priors at T=1",
           "[operations][stability]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.stability = 1.0;

  OperationContext ctx (s, pctx, cfg);
#
  InitializeStabilityPriors op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  REQUIRE (pctx.hysteresis_band_prior
           == Catch::Approx (cortext::core::BaseBandPrior (1.0)));
  REQUIRE (pctx.half_life_prior
           == Catch::Approx (cortext::core::BaseHalfLifePrior (1.0)));
  REQUIRE (pctx.rate_decay_prior == Catch::Approx (0.98));
  REQUIRE (pctx.periphery_half_life_prior
           == Catch::Approx (
               cortext::core::ClampHalfLife (0.5 * pctx.half_life_prior)));
  REQUIRE (pctx.salience_half_life_prior
           == Catch::Approx (
               cortext::core::ClampHalfLife (0.5 * pctx.half_life_prior)));
  REQUIRE (pctx.drift_weight_prior == Catch::Approx (0.0));
}
#
TEST_CASE ("InitializeStabilityPriors mid T", "[operations][stability]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);
#
  InitializeStabilityPriors op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  REQUIRE (pctx.hysteresis_band_prior
           == Catch::Approx (cortext::core::BaseBandPrior (0.5)));
  REQUIRE (pctx.half_life_prior
           == Catch::Approx (cortext::core::BaseHalfLifePrior (0.5)));
  REQUIRE (pctx.rate_decay_prior
           == Catch::Approx (cortext::core::Lerp (0.60, 0.98, 0.5)));
  REQUIRE (pctx.periphery_half_life_prior
           == Catch::Approx (
               cortext::core::ClampHalfLife (0.5 * pctx.half_life_prior)));
  REQUIRE (pctx.salience_half_life_prior
           == Catch::Approx (
               cortext::core::ClampHalfLife (0.5 * pctx.half_life_prior)));
  REQUIRE (pctx.drift_weight_prior == Catch::Approx (0.25));
}

TEST_CASE ("UpdateStability no history keeps half-life at prior",
           "[operations][stability][alg6]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.stability = 0.5;

  OperationContext ctx_init (s, pctx, cfg);
  InitializeStabilityPriors init_op;
  init_op.Execute (ctx_init, cortext::testing::GetNullTransaction ());
  const double prior_hl = pctx.half_life_prior;
  const double hysteresis_before = pctx.hysteresis;
  //
  UpdateStability op;
  OperationContext ctx (s, pctx, cfg);
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
  //
  REQUIRE (pctx.half_life == Catch::Approx (prior_hl));
  REQUIRE (
      pctx.periphery_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
  REQUIRE (
      pctx.salience_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
  // Alg 6 should not modify hysteresis; Alg 8 is authoritative for the band.
  REQUIRE (pctx.hysteresis == Catch::Approx (hysteresis_before));
}

TEST_CASE ("UpdateStability increases half-life when retention above mean",
           "[operations][stability][alg6]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.stability = 0.5;

  InitializeStabilityPriors init_op;
  {
    OperationContext ctx (s, pctx, cfg);
    init_op.Execute (ctx, cortext::testing::GetNullTransaction ());
  }
  const double prior_hl = pctx.half_life;
  // Fill history with tail near 100, last value much larger (e.g., 500)
  const int w = cortext::core::WRet (cfg.stability);
  for (int i = 0; i < w - 1; ++i)
    {
      pctx.observed_retention_history.push_back (100.0);
    }
  pctx.observed_retention_history.push_back (500.0);
  //
  UpdateStability op;
  OperationContext ctx2 (s, pctx, cfg);
  op.Execute (ctx2, cortext::testing::GetNullTransaction ());
  //
  REQUIRE (pctx.half_life > prior_hl);
  REQUIRE (
      pctx.periphery_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
  REQUIRE (
      pctx.salience_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
}

TEST_CASE ("UpdateStability decreases half-life when retention below mean",
           "[operations][stability][alg6]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.stability = 0.5;

  InitializeStabilityPriors init_op;
  {
    OperationContext ctx (s, pctx, cfg);
    init_op.Execute (ctx, cortext::testing::GetNullTransaction ());
  }
  const double prior_hl = pctx.half_life;
  // Tail near 100, last value much smaller (e.g., 10)
  const int w = cortext::core::WRet (cfg.stability);
  for (int i = 0; i < w - 1; ++i)
    {
      pctx.observed_retention_history.push_back (100.0);
    }
  pctx.observed_retention_history.push_back (10.0);
  //
  UpdateStability op;
  OperationContext ctx2 (s, pctx, cfg);
  op.Execute (ctx2, cortext::testing::GetNullTransaction ());
  //
  REQUIRE (pctx.half_life < prior_hl);
}

TEST_CASE ("UpdateStability uses tail window w_ret(T)",
           "[operations][stability][alg6]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  // Use a small but non-zero T so half-life prior > hl_min and can decrease.
  cfg.stability = 0.1; // w_ret ≈ 30

  InitializeStabilityPriors init_op;
  {
    OperationContext ctx (s, pctx, cfg);
    init_op.Execute (ctx, cortext::testing::GetNullTransaction ());
  }
  const double prior_hl = pctx.half_life;
  const int w = cortext::core::WRet (cfg.stability); // expect ~30
  // First 10 very low (10), next w-1 very high (100), last observed 80.
  for (int i = 0; i < 10; ++i)
    {
      pctx.observed_retention_history.push_back (10.0);
    }
  for (int i = 0; i < w - 1; ++i)
    {
      pctx.observed_retention_history.push_back (100.0);
    }
  pctx.observed_retention_history.push_back (80.0);
  //
  UpdateStability op;
  OperationContext ctx2 (s, pctx, cfg);
  op.Execute (ctx2, cortext::testing::GetNullTransaction ());
  //
  // With tail-only stats, observed 80 is below tail mean (~high) → half-life
  // decreases.
  REQUIRE (pctx.half_life < prior_hl);
}
