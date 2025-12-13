#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/operations/centroids.hpp>
#include <cortext/operations/goal_alignment_fallback.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ComputeGoalAlignmentFallback;
using cortext::operations::ComputeMetrics;
using cortext::operations::InitializeEmbeddedCentroids;

TEST_CASE ("InitializeEmbeddedCentroids populates ProcessorContext centroids",
           "[operations][centroids]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (256);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);
  InitializeEmbeddedCentroids op;
  op.Execute (ctx);
  REQUIRE (pctx.centroids.has_value ());
  REQUIRE (pctx.centroids->emotion_centroids.size () == 6);
  REQUIRE (pctx.centroids->affect.valence_positive.size () == 256);
  REQUIRE (pctx.centroids->affect.valence_negative.size () == 256);
}

TEST_CASE ("ComputeMetrics uses affect centroids and sets violation telemetry",
           "[operations][centroids][metrics]")
{
  Signal s;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;
  {
    OperationContext init_ctx (s, pctx, cfg, buf);
    InitializeEmbeddedCentroids init;
    init.Execute (init_ctx);
  }
  REQUIRE (pctx.centroids.has_value ());
  s.embedding = pctx.centroids->affect.valence_positive;
  OperationContext ctx (s, pctx, cfg, buf);
  ComputeMetrics metrics;
  metrics.Execute (ctx);
  const auto v = ctx.GetMetric (operations::Metric::valence);
  REQUIRE (v.has_value ());
  REQUIRE (*v > 0.5);
  REQUIRE (ctx.GetViolation ().has_value ());
}

TEST_CASE ("ComputeGoalAlignmentFallback sets metric only when missing",
           "[operations][centroids][goal_alignment]")
{
  Signal s;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  std::vector<BufferedWriteInstruction> buf;
  {
    OperationContext init_ctx (s, pctx, cfg, buf);
    InitializeEmbeddedCentroids init;
    init.Execute (init_ctx);
  }
  REQUIRE (pctx.centroids.has_value ());
  s.embedding = pctx.centroids->affect.goal_aligned;
  OperationContext ctx (s, pctx, cfg, buf);
  ComputeGoalAlignmentFallback fb;
  fb.Execute (ctx);
  auto ga = ctx.GetMetric (operations::Metric::goal_alignment);
  REQUIRE (ga.has_value ());
  REQUIRE (*ga > 0.5);
  ctx.SetMetric (operations::Metric::goal_alignment, 0.1);
  fb.Execute (ctx);
  ga = ctx.GetMetric (operations::Metric::goal_alignment);
  REQUIRE (ga.has_value ());
  REQUIRE (*ga == Catch::Approx (0.1));
}


