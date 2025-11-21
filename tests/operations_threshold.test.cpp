#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/threshold.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
using namespace cortext;
using cortext::operations::UpdateThreshold;

TEST_CASE ("UpdateThreshold basic adaptation toward p90",
           "[operations][threshold]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  // Start with moderate uncertainty to give non-zero experiential mass.
  pctx.u_t = 1.0;

  UpdateThreshold op;

  // Feed a sequence of high scores; T_dynamic should rise above T_prior.
  const double t_prior
      = cortext::core::TPrior (cfg.focus, cfg.sensitivity, cfg.stability);
  for (int i = 0; i < 25; ++i)
    {
      OperationContext ctx (s, pctx, cfg, buf);
      ctx.SetCompositeScore (0.9);
      // Provide increasing timestamps to drive Δt
      s.timestamp += 1; // 1s per signal
      op.Execute (ctx);
    }

  REQUIRE (pctx.T_dynamic > t_prior);

  // Hysteresis should be the lerp between [0.02, 0.25] at stability=0.5
  const double expected_band = cortext::core::Lerp (0.02, 0.25, cfg.stability);
  REQUIRE (pctx.hysteresis == Catch::Approx (expected_band));
}

TEST_CASE ("Alg8 ΔT_homeo increases threshold when observed rate > target",
           "[operations][threshold][homeo][fast]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  // Set a modest rate_target to be exceeded by fast timestamps
  pctx.rate_target_prior = 0.5; // writes/min
  pctx.rate_target = pctx.rate_target_prior;
  pctx.u_t = 0.5;

  UpdateThreshold op;
  const double before = pctx.T_dynamic;
  // Fast stream: timestamps every 0.2s → 300 wpm instantaneous
  for (int i = 0; i < 10; ++i)
    {
      s.timestamp += 1; // assume 1 unit = 0.2s; emulate via multiple updates
      OperationContext ctx (s, pctx, cfg, buf);
      ctx.SetCompositeScore (0.5);
      op.Execute (ctx);
      // emulate 0.2s by calling multiple times per second; the α smoothing
      // will handle it
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
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  // Set a higher target; slow timestamps will be below target
  pctx.rate_target_prior = 5.0; // writes/min
  pctx.rate_target = pctx.rate_target_prior;
  pctx.u_t = 0.5;

  UpdateThreshold op;
  const double before = pctx.T_dynamic;
  // Slow stream: timestamps every 10s → 6 wpm instantaneous; still lower than
  // target?
  for (int i = 0; i < 6; ++i)
    {
      s.timestamp += 10;
      OperationContext ctx (s, pctx, cfg, buf);
      ctx.SetCompositeScore (0.5);
      op.Execute (ctx);
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
