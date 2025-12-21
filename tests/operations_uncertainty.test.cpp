#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/operations/uncertainty.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::UpdateUncertainty;

TEST_CASE ("UpdateUncertainty uses structural coherence when weights degenerate",
           "[operations][uncertainty]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.0; // zero weights
  cfg.stability = 1.0; // 1 - T = 0 -> degenerate weights

  OperationContext ctx (s, pctx, cfg);
  ctx.SetStructuralCoherence (0.5); // fallback coherence for short context

  UpdateUncertainty op;

  // Initial u_t = 0; first update should move toward coherence complement (0.5)
  pctx.signals_processed = 0;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.u_t > 0.0);

  // With only structural coherence available, u_raw stays constant
  double prev_u = pctx.u_t;
  pctx.signals_processed = 1000; // very mature
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.u_t >= prev_u);
}
