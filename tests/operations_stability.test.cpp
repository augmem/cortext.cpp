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
#
TEST_CASE ("InitializeStabilityPriors sets the hysteresis prior at T=0",
           "[operations][stability]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.0;

  OperationContext ctx (s, pctx, cfg);
  InitializeStabilityPriors op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.hysteresis_band_prior
           == Catch::Approx (cortext::core::BaseBandPrior (0.0)));
  REQUIRE (pctx.stability_priors_initialized);
}

TEST_CASE ("InitializeStabilityPriors sets the hysteresis prior at T=1",
           "[operations][stability]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 1.0;

  OperationContext ctx (s, pctx, cfg);
  InitializeStabilityPriors op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.hysteresis_band_prior
           == Catch::Approx (cortext::core::BaseBandPrior (1.0)));
  REQUIRE (pctx.stability_priors_initialized);
}

TEST_CASE ("InitializeStabilityPriors sets the hysteresis prior at mid T",
           "[operations][stability]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);
  InitializeStabilityPriors op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.hysteresis_band_prior
           == Catch::Approx (cortext::core::BaseBandPrior (0.5)));
  REQUIRE (pctx.stability_priors_initialized);
}
