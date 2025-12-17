#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#
using namespace cortext;
using cortext::operations::InitializeSensitivityPriors;
#
TEST_CASE ("InitializeSensitivityPriors computes priors (S=0.5)",
           "[operations][sensitivity]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;

  OperationContext ctx (s, pctx, cfg);
#
  InitializeSensitivityPriors op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  // Expected values per algorithms.md
  const double S = 0.5;
  const double base_rate_prior = cortext::core::BaseRatePrior (S);
  REQUIRE (pctx.base_rate_prior == Catch::Approx (base_rate_prior));
  REQUIRE (pctx.weight_novelty_prior == Catch::Approx (0.3 + 0.7 * S));
  REQUIRE (pctx.weight_surprise_prior == Catch::Approx (0.2 + 0.8 * S));
  REQUIRE (pctx.weight_valence_prior == Catch::Approx (0.4 + 0.6 * S));
  REQUIRE (pctx.weight_arousal_prior == Catch::Approx (S));
  REQUIRE (pctx.weight_emotion_prior == Catch::Approx (0.2 + 0.8 * S));
  REQUIRE (pctx.emotion_gain_prior == Catch::Approx (std::exp (1.5 * S)));
  REQUIRE (pctx.score_gain_prior == Catch::Approx (std::exp (2.0 * S)));
  REQUIRE (pctx.rate_target_prior
           == Catch::Approx (base_rate_prior * (0.5 + 1.5 * S)));
}
#
TEST_CASE ("InitializeSensitivityPriors edge cases (S=0 and S=1)",
           "[operations][sensitivity]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);

#
  // S = 0
  {
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cfg.sensitivity = 0.0;
    OperationContext ctx (s, pctx, cfg);
    InitializeSensitivityPriors op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
    REQUIRE (pctx.base_rate_prior
             == Catch::Approx (cortext::core::BaseRatePrior (0.0)));
    REQUIRE (pctx.weight_novelty_prior == Catch::Approx (0.3));
    REQUIRE (pctx.weight_surprise_prior == Catch::Approx (0.2));
    REQUIRE (pctx.weight_valence_prior == Catch::Approx (0.4));
    REQUIRE (pctx.weight_arousal_prior == Catch::Approx (0.0));
    REQUIRE (pctx.weight_emotion_prior == Catch::Approx (0.2));
    REQUIRE (pctx.emotion_gain_prior == Catch::Approx (std::exp (0.0)));
    REQUIRE (pctx.score_gain_prior == Catch::Approx (std::exp (0.0)));
    REQUIRE (pctx.rate_target_prior
             == Catch::Approx (pctx.base_rate_prior * (0.5 + 0.0)));
  }
#
  // S = 1
  {
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cfg.sensitivity = 1.0;
    OperationContext ctx (s, pctx, cfg);
    InitializeSensitivityPriors op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
    REQUIRE (pctx.base_rate_prior
             == Catch::Approx (cortext::core::BaseRatePrior (1.0)));
    REQUIRE (pctx.weight_novelty_prior == Catch::Approx (1.0));
    REQUIRE (pctx.weight_surprise_prior == Catch::Approx (1.0));
    REQUIRE (pctx.weight_valence_prior == Catch::Approx (1.0));
    REQUIRE (pctx.weight_arousal_prior == Catch::Approx (1.0));
    REQUIRE (pctx.weight_emotion_prior == Catch::Approx (1.0));
    REQUIRE (pctx.emotion_gain_prior == Catch::Approx (std::exp (1.5)));
    REQUIRE (pctx.score_gain_prior == Catch::Approx (std::exp (2.0)));
    REQUIRE (pctx.rate_target_prior
             == Catch::Approx (pctx.base_rate_prior * (0.5 + 1.5)));
  }
}
