#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/stability.hpp>
#include <cortext/operations/stability_feedback.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ApplyStabilityFeedback;
using cortext::operations::InitializeStabilityPriors;

TEST_CASE ("Alg17 positive contextual gain increases half_life",
           "[operations][stability_feedback]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  // Initialize stability priors and seed dynamic state
  {
    OperationContext ctx (s, pctx, cfg, buf);
    InitializeStabilityPriors init;
    init.Execute (ctx);
  }
  const double prior_hl = pctx.half_life;

  OperationContext ctx (s, pctx, cfg, buf);
  // Attach a positive contextual gain event
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 1LL;
  ev.used = true;
  ev.contextual_gain = 0.6; // positive
  ctx.SetMemoryUsageEvents ({ ev });

  ApplyStabilityFeedback op;
  op.Execute (ctx);
  // Alg17 now emits ΔHalfLife_adj_t for Alg6 to consume; apply Alg6
  REQUIRE (ctx.GetDeltaHalfLifeAdjustment ().has_value ());
  // Provide a retention observation and run Alg6
  ctx.SetObservedRetentionSeconds (120.0);
  cortext::operations::UpdateStability st;
  st.Execute (ctx);
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
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  // Initialize stability priors and seed dynamic state
  {
    OperationContext ctx (s, pctx, cfg, buf);
    InitializeStabilityPriors init;
    init.Execute (ctx);
  }
  const double prior_hl = pctx.half_life;

  OperationContext ctx (s, pctx, cfg, buf);
  // Attach a negative contextual gain event
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 2LL;
  ev.used = true;
  ev.contextual_gain = -0.4; // negative
  ctx.SetMemoryUsageEvents ({ ev });

  ApplyStabilityFeedback op;
  op.Execute (ctx);
  // Alg17 now emits ΔHalfLife_adj_t for Alg6 to consume; apply Alg6
  REQUIRE (ctx.GetDeltaHalfLifeAdjustment ().has_value ());
  // Provide a retention observation and run Alg6
  ctx.SetObservedRetentionSeconds (120.0);
  cortext::operations::UpdateStability st;
  st.Execute (ctx);
  REQUIRE (pctx.half_life < prior_hl);
  REQUIRE (
      pctx.periphery_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
  REQUIRE (
      pctx.salience_half_life
      == Catch::Approx (cortext::core::ClampHalfLife (0.5 * pctx.half_life)));
}
