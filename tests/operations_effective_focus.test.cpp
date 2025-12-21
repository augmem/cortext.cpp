#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/effective_focus.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ComputeEffectiveFocus;

TEST_CASE ("ComputeEffectiveFocus uses coherence to scale F",
           "[operations][effective_focus]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.6; // base F


  ComputeEffectiveFocus op;

  // coherence=1.0 → F_eff = F × (0.5 + 0.5*1) = F
  {
    OperationContext ctx (s, pctx, cfg);
    ctx.SetCoherence (1.0);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
    REQUIRE (ctx.GetEffectiveFocus () == Catch::Approx (0.6));
  }

  // coherence=0.0 → F_eff = F × (0.5 + 0.5*0) = 0.5*F
  {
    OperationContext ctx (s, pctx, cfg);
    ctx.SetCoherence (0.0);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
    REQUIRE (ctx.GetEffectiveFocus () == Catch::Approx (0.3));
  }
}
