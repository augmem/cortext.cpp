#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/focus_feedback.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ApplyFocusFeedback;
using cortext::operations::InitializeFocusPriors;

namespace
{

static OperationContext
MakeContext (double focus, double stability)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = 1;
  s.source_id = "test";

  static ProcessorContext pctx;
  pctx = ProcessorContext (); // reset

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = focus;
  cfg.stability = stability;

  OperationContext ctx (s, pctx, cfg);
  InitializeFocusPriors init;
  init.Execute (ctx, cortext::testing::GetNullTransaction ());
  return ctx;
}

} // namespace

TEST_CASE ("Alg15 positive contextual gain boosts relevance and narrows width",
           "[operations][focus_feedback]")
{
  auto ctx = MakeContext (0.8, 0.5);
  auto &pctx = ctx.GetProcessorContext ();

  const double prev_wr = pctx.weight_relevance;
  const double prev_aw = pctx.attention_width;

  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 1LL;
  ev.used = true;
  ev.contextual_gain = 0.5; // positive
  ctx.SetMemoryUsageEvents ({ ev });

  ApplyFocusFeedback op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.weight_relevance > prev_wr);
  REQUIRE (pctx.attention_width < prev_aw);
}

TEST_CASE (
    "Alg15 non-positive gain widens attention, leaves relevance unchanged",
    "[operations][focus_feedback]")
{
  auto ctx = MakeContext (0.7, 0.5);
  auto &pctx = ctx.GetProcessorContext ();

  const double prev_wr = pctx.weight_relevance;
  const double prev_aw = pctx.attention_width;

  // Use zero gain first
  {
    OperationContext::MemoryUsageEvent ev0{};
    ev0.embedding_id = 2LL;
    ev0.used = true;
    ev0.contextual_gain = 0.0;
    ctx.SetMemoryUsageEvents ({ ev0 });
    ApplyFocusFeedback op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
  }

  REQUIRE (pctx.weight_relevance == Catch::Approx (prev_wr));
  REQUIRE (pctx.attention_width > prev_aw);

  // Negative gain further widens (monotonic non-decrease)
  const double aw_after_zero = pctx.attention_width;
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 3LL;
  ev.used = true;
  ev.contextual_gain = -0.3;
  ctx.SetMemoryUsageEvents ({ ev });
  ApplyFocusFeedback op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.attention_width >= aw_after_zero);
}
