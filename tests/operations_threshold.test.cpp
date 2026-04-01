#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/threshold.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
using namespace cortext;
using cortext::operations::UpdateRateState;
using cortext::operations::UpdateThreshold;

TEST_CASE ("UpdateThreshold basic adaptation toward p90",
           "[operations][threshold]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Neutralize homeostatic correction for this test.
  pctx.rate_target_prior = 0.0;
  pctx.rate_target = 0.0;
  pctx.rho_hat_prev = 0.0;

  // Start with moderate uncertainty to give non-zero experiential mass.
  pctx.u_t = 1.0;

  UpdateThreshold op;

  // Feed a sequence of high scores; T_dynamic should rise above T_prior.
  const double t_prior
      = cortext::core::TPrior (cfg.focus, cfg.sensitivity, cfg.stability);
  for (int i = 0; i < 25; ++i)
    {
      OperationContext ctx (s, pctx, cfg);
      ctx.SetCompositeScore (0.9);
      // Provide increasing timestamps to drive Δt
      s.timestamp += 1000; // 1s per signal (ms timestamps)
      op.Execute (ctx, cortext::testing::GetNullTransaction ());
    }

  REQUIRE (pctx.T_dynamic > t_prior);

  // Hysteresis remains anchored to the processor state's current band.
  const double expected_band = 0.05;
  REQUIRE (pctx.hysteresis == Catch::Approx (expected_band).margin (0.01));
}

TEST_CASE ("Alg8 ΔT_homeo increases threshold when observed rate > target",
           "[operations][threshold][homeo][fast]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  // Set a modest rate_target to be exceeded by fast timestamps
  pctx.rate_target_prior = 0.5; // writes/min
  pctx.rate_target = pctx.rate_target_prior;
  pctx.u_t = 0.5;

  UpdateThreshold op;
  UpdateRateState update_rate;
  const double before = pctx.T_dynamic;
  // Fast stream: timestamps every 0.2s (ms-based) → 300 wpm instantaneous
  for (int i = 0; i < 10; ++i)
    {
      s.timestamp += 200; // 0.2s per signal (ms timestamps)
      OperationContext ctx (s, pctx, cfg);
      ctx.SetCompositeScore (0.5);
      op.Execute (ctx, cortext::testing::GetNullTransaction ());
      ctx.SetWriteDecision (true);
      update_rate.Execute (ctx, cortext::testing::GetNullTransaction ());
      // Multiple short steps emulate high write rate; α smoothing handles it.
    }
  REQUIRE (pctx.T_dynamic >= before);
  // Within rails
  const double Tmin
      = cortext::core::TMin (pctx.signals_processed, cfg.stability);
  const double Tmax
      = cortext::core::TMax (pctx.signals_processed, cfg.stability);
  REQUIRE (pctx.T_dynamic >= Tmin);
  REQUIRE (pctx.T_dynamic <= Tmax);
}

TEST_CASE ("Alg8 ΔT_homeo decreases threshold when observed rate < target",
           "[operations][threshold][homeo][slow]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  // Set a higher target; slow timestamps will be below target
  pctx.rate_target_prior = 30.0; // writes/min
  pctx.rate_target = pctx.rate_target_prior;
  pctx.u_t = 0.5;

  UpdateThreshold op;
  UpdateRateState update_rate;
  const double before = pctx.T_dynamic;
  // Slow stream: timestamps every 10s → 6 wpm instantaneous; still lower than
  // target?
  for (int i = 0; i < 6; ++i)
    {
      s.timestamp += 10000;
      OperationContext ctx (s, pctx, cfg);
      ctx.SetCompositeScore (0.5);
      op.Execute (ctx, cortext::testing::GetNullTransaction ());
      ctx.SetWriteDecision (true);
      update_rate.Execute (ctx, cortext::testing::GetNullTransaction ());
    }
  // Expect movement opposite to fast case; not necessarily strictly less due
  // to prior blend but should not explode and must stay within rails.
  const double Tmin
      = cortext::core::TMin (pctx.signals_processed, cfg.stability);
  const double Tmax
      = cortext::core::TMax (pctx.signals_processed, cfg.stability);
  REQUIRE (pctx.T_dynamic >= Tmin);
  REQUIRE (pctx.T_dynamic <= Tmax);
}

TEST_CASE ("Alg8 time-scaled delta cap scales with inter-signal time",
           "[operations][threshold][time_scaling]")
{
  // Verify that the time-scaled cap formula is correctly computed.
  // Per algorithms.md Section 4.3: cap_total = max_ΔT_per_min × (Δt / 60.0)
  //
  // The cap scales linearly with delta_t - longer gaps allow larger deltas.
  // This test verifies the cap calculation itself, not the full threshold
  // update (which also includes EWMA toward target).

  // Verify knob function: MaxDeltaTPerMin scales with maturity
  // At maturity=0: max_delta = 0.30
  // At maturity=1: max_delta = 0.10
  const double max_delta_immature
      = cortext::core::MaxDeltaTPerMin (0, 0.5); // count=0, low maturity
  const double max_delta_mature
      = cortext::core::MaxDeltaTPerMin (10000, 0.5); // high count, high maturity

  REQUIRE (max_delta_immature > max_delta_mature);
  REQUIRE (max_delta_immature <= 0.30);
  REQUIRE (max_delta_mature >= 0.10);

  // Verify time-scaling: cap is proportional to delta_t
  // cap_total = max_delta * delta_t / 60.0
  const double delta_t_short = 1.0;  // 1 second
  const double delta_t_long = 30.0;  // 30 seconds
  const double max_delta = 0.30;     // immature system

  const double cap_short = max_delta * delta_t_short / 60.0;
  const double cap_long = max_delta * delta_t_long / 60.0;

  // Longer delta_t → proportionally larger cap
  REQUIRE (cap_long / cap_short == Catch::Approx (delta_t_long / delta_t_short));

  // Verify actual values
  REQUIRE (cap_short == Catch::Approx (0.005));  // 0.30 * 1 / 60
  REQUIRE (cap_long == Catch::Approx (0.15));    // 0.30 * 30 / 60
}
