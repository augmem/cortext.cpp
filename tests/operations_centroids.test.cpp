#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>

#include <cortext/operations/centroids.hpp>

#include <cortext/operations/metrics.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;

using cortext::operations::ComputeMetrics;
using cortext::operations::InitializeEmbeddedCentroids;

TEST_CASE ("InitializeEmbeddedCentroids populates ProcessorContext centroids",
           "[operations][centroids]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (256);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;

  OperationContext ctx (s, pctx, cfg);
  InitializeEmbeddedCentroids op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
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

  {
    OperationContext init_ctx (s, pctx, cfg);
    InitializeEmbeddedCentroids init;
    init.Execute (init_ctx, cortext::testing::GetNullTransaction ());
  }
  REQUIRE (pctx.centroids.has_value ());
  s.embedding = pctx.centroids->affect.valence_positive;
  OperationContext ctx (s, pctx, cfg);
  ComputeMetrics metrics;
  metrics.Execute (ctx, cortext::testing::GetNullTransaction ());
  const auto v = ctx.GetMetric (operations::Metric::valence);
  REQUIRE (v.has_value ());
  REQUIRE (*v > 0.5);
  REQUIRE (ctx.GetViolation ().has_value ());
}



