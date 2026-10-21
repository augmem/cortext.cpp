#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/core/knobs.hpp>
#include <cortext/operations/signal_history.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

#include <cmath>

using namespace cortext;
using cortext::operations::UpdateRateState;
using cortext::operations::UpdateSignalHistory;

TEST_CASE ("UpdateSignalHistory records clamped composite scores",
           "[operations][signal_history]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  pctx.weight_relevance = 0.4;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  UpdateSignalHistory op;

  {
    OperationContext ctx (s, pctx, cfg);
    ctx.SetCompositeScore (1.7);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
  }
  {
    // No composite score: the relevance weight stands in.
    OperationContext ctx (s, pctx, cfg);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
  }
  REQUIRE (pctx.recent_scores.size () == 2);
  REQUIRE (pctx.recent_scores[0] == Catch::Approx (1.0));
  REQUIRE (pctx.recent_scores[1] == Catch::Approx (0.4));

  const auto cap = static_cast<std::size_t> (
      std::max (core::WScore (cfg.stability),
                core::RecentScoreHistoryLimit (cfg.stability)));
  for (std::size_t i = 0; i < cap + 5; ++i)
    {
      OperationContext ctx (s, pctx, cfg);
      ctx.SetCompositeScore (0.5);
      op.Execute (ctx, cortext::testing::GetNullTransaction ());
    }
  REQUIRE (pctx.recent_scores.size () == cap);
}

TEST_CASE ("UpdateSignalHistory smooths the inter-signal time",
           "[operations][signal_history]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  pctx.last_rate_timestamp = 10'000;
  pctx.dt_ema = 2.0;
  s.timestamp = 14'000; // 4 s after the last rate update

  OperationContext ctx (s, pctx, cfg);
  ctx.SetCompositeScore (0.5);
  UpdateSignalHistory op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  const double delta_t = std::max (4.0, core::DeltaTMin (cfg.stability));
  const double alpha = 1.0 - std::exp (-delta_t / core::TauDt (cfg.stability));
  REQUIRE (pctx.dt_ema == Catch::Approx ((1.0 - alpha) * 2.0 + alpha * delta_t));
  // The rate timestamp advances only in UpdateRateState.
  REQUIRE (pctx.last_rate_timestamp == 10'000);
}

TEST_CASE ("UpdateRateState tracks the write rate",
           "[operations][signal_history][rate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  UpdateRateState op;
  // A steady stream, one write every 2 s, from a known last update.
  pctx.last_rate_timestamp = 1'000;
  s.timestamp = 1'000;

  double previous = pctx.m_rate;
  for (int i = 0; i < 5; ++i)
    {
      s.timestamp += 2'000;
      OperationContext ctx (s, pctx, cfg);
      ctx.SetWriteDecision (true);
      op.Execute (ctx, cortext::testing::GetNullTransaction ());
      REQUIRE (pctx.m_rate > previous);
      REQUIRE (pctx.last_rate_timestamp == s.timestamp);
      previous = pctx.m_rate;
    }

  // No writes: the rate decays.
  s.timestamp += 2'000;
  OperationContext ctx (s, pctx, cfg);
  ctx.SetWriteDecision (false);
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.m_rate < previous);
}
