#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/blend.hpp>
#include <cortext/operations/threshold.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ComputeCompositeScore;
using cortext::operations::FitMetricWeightsRLS;
using cortext::operations::UpdateThreshold;

namespace
{

struct SetMetricsOp : IOperation
{
  // Provide a fixed metrics map (normalized values in [-1,1] or [0,1]).
  std::unordered_map<operations::Metric, double> metrics;
  explicit SetMetricsOp (std::unordered_map<operations::Metric, double> m)
      : metrics (std::move (m))
  {
  }
  void
  Execute (OperationContext &ctx) const override
  {
    for (const auto &kv : metrics)
      {
        ctx.SetMetric (kv.first, kv.second);
      }
  }
};

} // namespace

TEST_CASE ("Alg7 composite produces clamped normalized score",
           "[operations][blend]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.6;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.4;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);

  // Set a variety of metrics; values are normalized.
  SetMetricsOp set_metrics ({
      { operations::Metric::relevance, 0.80 },
      { operations::Metric::mismatch, 0.05 },
      { operations::Metric::surprise, 0.20 },
      { operations::Metric::rarity, 0.40 },
      { operations::Metric::drift, 0.10 },
      { operations::Metric::contradiction, 0.0 },
      { operations::Metric::utility, 0.35 },
      { operations::Metric::periphery, 0.15 },
      { operations::Metric::coverage, 0.50 },
      { operations::Metric::salience, 0.30 },
      { operations::Metric::valence, 0.60 },
      { operations::Metric::arousal, 0.25 },
  });
  set_metrics.Execute (ctx);

  ComputeCompositeScore compute;
  compute.Execute (ctx);

  const auto score = ctx.GetCompositeScore ();
  REQUIRE (score.has_value ());
  REQUIRE (*score >= 0.0);
  REQUIRE (*score <= 1.0);
  REQUIRE (ctx.GetLastWeightSum () > 0.0);
  REQUIRE (ctx.GetLastEffectiveMetricCount () >= 1);
}

TEST_CASE ("Alg7 RLS increases relevance weight with consistent evidence",
           "[operations][blend]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.4; // modest initial relevance prior
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);

  // Initial weights via bootstrap
  ComputeCompositeScore bootstrap_compute;
  bootstrap_compute.Execute (ctx); // initializes blender state
  const double w0 = pctx.blender_state[operations::Metric::relevance];

  // Provide strong evidence that relevance should be high: x has only
  // relevance=1.0
  SetMetricsOp set_rel_only ({ { operations::Metric::relevance, 1.0 } });
  FitMetricWeightsRLS fit;
  for (int i = 0; i < 50; ++i)
    {
      set_rel_only.Execute (ctx);
      fit.Execute (ctx);
    }
  const double w1 = pctx.blender_state[operations::Metric::relevance];
  REQUIRE (w1 >= w0);
  REQUIRE (w1 <= 1.0);
}

TEST_CASE ("Alg7→Alg8 integration adjusts threshold with composite",
           "[integration][blend][threshold]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (4);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);

  FitMetricWeightsRLS fit;
  ComputeCompositeScore compute;
  UpdateThreshold update_T;

  // Low composite case
  SetMetricsOp low ({ { operations::Metric::relevance, 0.0 } });
  low.Execute (ctx);
  compute.Execute (ctx);
  update_T.Execute (ctx);
  const double T1 = ctx.GetThresholdTDynamic ();

  // High composite case
  SetMetricsOp high ({ { operations::Metric::relevance, 1.0 } });
  high.Execute (ctx);
  // Run a small loop to populate recent scores and allow a visible change.
  for (int i = 0; i < 5; ++i)
    {
      fit.Execute (ctx);
      compute.Execute (ctx);
      update_T.Execute (ctx);
    }
  const double T2 = ctx.GetThresholdTDynamic ();
  REQUIRE (T1 != T2);
}

TEST_CASE ("Alg7 RLS resets covariance on ill-conditioning",
           "[operations][blend]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.4;
  cfg.sensitivity = 0.4;
  cfg.stability = 0.4;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);

  // Initialize state and then corrupt P with NaN to trigger reset.
  ComputeCompositeScore bootstrap_compute;
  bootstrap_compute.Execute (ctx);
  // corrupt
  if (!pctx.blender_P.empty ())
    {
      pctx.blender_P[0][0] = std::numeric_limits<double>::quiet_NaN ();
    }
  // Provide a simple metric vector
  SetMetricsOp set_rel_only ({ { operations::Metric::relevance, 1.0 } });
  FitMetricWeightsRLS fit;
  // Loop enough times to trigger throttled update
  for (int i = 0; i < 10; ++i)
    {
      set_rel_only.Execute (ctx);
      fit.Execute (ctx);
    }
  // After reset and update, diagonal should be positive and finite (not NaN)
  REQUIRE (!pctx.blender_P.empty ());
  REQUIRE (std::isfinite (pctx.blender_P[0][0]));
  REQUIRE (pctx.blender_P[0][0] > 0.0);
}
