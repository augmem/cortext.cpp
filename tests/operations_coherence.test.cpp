#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/operations/coherence.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#
using namespace cortext;
using cortext::operations::ComputeCoherence;
#
TEST_CASE ("ComputeCoherence handles small windows", "[operations][coherence]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;

#
  ComputeCoherence op;
#
  // Case 1: Empty context → variance 0 → coherence = 1
  {
    OperationContext ctx (s, pctx, cfg);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
    REQUIRE (ctx.GetCoherence () == Catch::Approx (1.0));
  }
#
  // Case 2: Single element context → variance 0 → coherence = 1
  {
    pctx.recent_context_embeddings.clear ();
    pctx.recent_context_embeddings.push_back (Eigen::VectorXf::Ones (4));
    OperationContext ctx (s, pctx, cfg);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
    REQUIRE (ctx.GetCoherence () == Catch::Approx (1.0));
  }
}
#
TEST_CASE ("ComputeCoherence lowers with mixed directions",
           "[operations][coherence]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (4);
  s.embedding[0] = 1.0f; // unit vector along axis 0
  ProcessorContext pctx;
  SignalProcessor::Config cfg;

#
  // Fill window with diverse directions to increase variance of cosines.
  pctx.recent_context_embeddings.clear ();
  Eigen::VectorXf e1 = Eigen::VectorXf::Zero (4);
  e1[0] = 1.0f; // same as s → cos = 1
  Eigen::VectorXf e2 = Eigen::VectorXf::Zero (4);
  e2[1] = 1.0f; // orthogonal → cos = 0
  Eigen::VectorXf e3 = Eigen::VectorXf::Zero (4);
  e3[0] = -1.0f; // opposite → cos = -1
  pctx.recent_context_embeddings.push_back (e1);
  pctx.recent_context_embeddings.push_back (e2);
  pctx.recent_context_embeddings.push_back (e3);
#
  ComputeCoherence op;
  OperationContext ctx (s, pctx, cfg);
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  const double coh = ctx.GetCoherence ();
  REQUIRE (coh >= 0.0);
  REQUIRE (coh <= 1.0);
  REQUIRE (coh < 1.0); // non-trivial variance yields coherence below 1
}
#
TEST_CASE ("ComputeCoherence clamping", "[operations][coherence]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (4);
  s.embedding[0] = 1.0f;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;

#
  // Extreme mix of cosines should still clamp in [0,1]
  pctx.recent_context_embeddings.clear ();
  for (int i = 0; i < 16; ++i)
    {
      Eigen::VectorXf e = Eigen::VectorXf::Zero (4);
      e[(i % 4)] = (i % 2 == 0) ? 1.0f : -1.0f;
      pctx.recent_context_embeddings.push_back (e);
    }
#
  ComputeCoherence op;
  OperationContext ctx (s, pctx, cfg);
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  const double coh = ctx.GetCoherence ();
  REQUIRE (coh >= 0.0);
  REQUIRE (coh <= 1.0);
}
