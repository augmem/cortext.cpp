#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/boundary.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::CheckEpisodeBoundary;

TEST_CASE ("Boundary check requests finalize when drift exceeds threshold",
           "[operations][boundary][alg12]")
{
  const int dim = 4;
  Eigen::VectorXf a = Eigen::VectorXf::Ones (dim);
  a.normalize ();
  Eigen::VectorXf b = -a;

  Signal s;
  s.embedding = a;

  ProcessorContext pctx;
  // Build two windows with opposite directions to maximize drift
  for (int i = 0; i < 6; ++i)
    {
      pctx.recent_context_embeddings.push_back (b);
    }
  for (int i = 0; i < 6; ++i)
    {
      pctx.recent_context_embeddings.push_back (a);
    }

  SignalProcessor::Config cfg;
  cfg.stability = 0.5; // mid threshold
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);

  CheckEpisodeBoundary op;
  op.Execute (ctx);

  REQUIRE (ctx.ShouldFinalizeEpisode ());
  auto drift = ctx.GetMetric (operations::Metric::drift_mag);
  REQUIRE (drift.has_value ());
  REQUIRE (*drift >= 0.5); // should be large
}

TEST_CASE ("Boundary check does not request finalize for low drift",
           "[operations][boundary][alg12]")
{
  const int dim = 4;
  Eigen::VectorXf a = Eigen::VectorXf::Ones (dim);
  a.normalize ();

  Signal s;
  s.embedding = a;

  ProcessorContext pctx;
  // Both windows aligned → low drift
  for (int i = 0; i < 12; ++i)
    {
      pctx.recent_context_embeddings.push_back (a);
    }

  SignalProcessor::Config cfg;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);

  CheckEpisodeBoundary op;
  op.Execute (ctx);

  REQUIRE_FALSE (ctx.ShouldFinalizeEpisode ());
  auto drift = ctx.GetMetric (operations::Metric::drift_mag);
  REQUIRE (drift.has_value ());
  REQUIRE (*drift <= 0.1);
}
