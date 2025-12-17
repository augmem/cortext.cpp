#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/operations/uncertainty.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::UpdateUncertainty;

TEST_CASE ("UpdateUncertainty fallback maturity path",
           "[operations][uncertainty]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.stability = 0.5; // mid stability

  OperationContext ctx (s, pctx, cfg);

  UpdateUncertainty op;

  // Initial u_t = 0; first update should move toward u_raw=1 - maturity(0)
  pctx.signals_processed = 0;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.u_t > 0.0);

  // As signals increase, maturity increases -> u_raw decreases -> u_t should
  // decrease with smoothing
  double prev_u = pctx.u_t;
  pctx.signals_processed = 1000; // very mature
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.u_t < prev_u);
}
