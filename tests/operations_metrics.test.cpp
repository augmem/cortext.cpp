#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/operations/effective_focus.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ComputeEffectiveFocus;
using cortext::operations::ComputeMetrics;

TEST_CASE ("ComputeMetrics sets all 12 metrics in normalized ranges",
           "[operations][metrics]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  ProcessorContext pctx;
  // Seed some recent context so relevance etc. are meaningful
  pctx.recent_context_embeddings.push_back (Eigen::VectorXf::Ones (4));
  pctx.recent_context_embeddings.push_back (2.0f * Eigen::VectorXf::Ones (4));
  pctx.recent_scores.push_back (0.2);
  pctx.recent_scores.push_back (0.4);
  pctx.recent_scores.push_back (0.6);

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.6;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);

  ComputeMetrics op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  static const operations::Metric kMetrics[] = {
    operations::Metric::relevance, operations::Metric::mismatch,
    operations::Metric::surprise,  operations::Metric::rarity,
    operations::Metric::drift,     operations::Metric::contradiction,
    operations::Metric::utility,   operations::Metric::periphery,
    operations::Metric::coverage,  operations::Metric::salience,
    operations::Metric::valence,   operations::Metric::arousal,
  };
  for (const auto metric : kMetrics)
    {
      auto v = ctx.GetMetric (metric);
      REQUIRE (v.has_value ());
      // All metrics should be normalized: within [0,1]
      REQUIRE (*v >= 0.0);
      REQUIRE (*v <= 1.0);
    }
}

TEST_CASE (
    "ComputeMetrics uses embedding_surprisal for surprise metric",
    "[operations][metrics][surprise]")
{
  // Reference: algorithms.md Section 3.2 Table 1 row "Surprise"
  // surprise = embedding_surprisal × S × (1 − 0.5T)

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  ProcessorContext pctx;
  pctx.recent_context_embeddings.push_back (Eigen::VectorXf::Ones (4));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.8;  // S
  cfg.stability = 0.2;     // T

  OperationContext ctx (s, pctx, cfg);

  // Pre-set embedding_surprisal as if computed by EmbeddingPredictionError
  const double known_surprisal = 0.6;
  ctx.SetMetric (operations::Metric::embedding_surprisal, known_surprisal);

  ComputeMetrics op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Verify surprise uses the formula: surprisal × S × (1 − 0.5T)
  // = 0.6 × 0.8 × (1 − 0.1) = 0.6 × 0.8 × 0.9 = 0.432
  auto surprise = ctx.GetMetric (operations::Metric::surprise);
  REQUIRE (surprise.has_value ());
  REQUIRE (*surprise == Catch::Approx (0.432).epsilon (0.001));
}

TEST_CASE (
    "Metrics use F_eff to reduce F-dependent metrics when coherence is low",
    "[operations][metrics][feff]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  ProcessorContext pctx;
  // Seed recent context so relevance computations are meaningful
  pctx.recent_context_embeddings.push_back (Eigen::VectorXf::Ones (4));
  pctx.recent_context_embeddings.push_back (Eigen::VectorXf::Ones (4));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.6; // base F
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  ComputeEffectiveFocus feff;
  ComputeMetrics metrics;

  // High coherence path
  OperationContext ctx_hi (s, pctx, cfg);
  ctx_hi.SetCoherence (1.0);
  feff.Execute (ctx_hi, cortext::testing::GetNullTransaction ());
  metrics.Execute (ctx_hi, cortext::testing::GetNullTransaction ());
  auto cov_hi = ctx_hi.GetMetric (operations::Metric::coverage).value_or (0.0);
  auto sal_hi = ctx_hi.GetMetric (operations::Metric::salience).value_or (0.0);

  // Low coherence path
  OperationContext ctx_lo (s, pctx, cfg);
  ctx_lo.SetCoherence (0.0);
  feff.Execute (ctx_lo, cortext::testing::GetNullTransaction ());
  metrics.Execute (ctx_lo, cortext::testing::GetNullTransaction ());
  auto cov_lo = ctx_lo.GetMetric (operations::Metric::coverage).value_or (0.0);
  auto sal_lo = ctx_lo.GetMetric (operations::Metric::salience).value_or (0.0);

  REQUIRE (cov_lo <= cov_hi);
  REQUIRE (sal_lo <= sal_hi);
}

TEST_CASE ("ComputeMetrics uses ΔSSE for utility when available",
           "[operations][metrics][utility]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  ProcessorContext pctx;
  pctx.recent_context_embeddings.push_back (Eigen::VectorXf::Ones (4));

  // Provide SSE history to enable ΔSSE
  pctx.prediction_error_sse_prev = 2.0;
  pctx.prediction_error_sse = 1.0;

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);
  ComputeMetrics op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  auto utility = ctx.GetMetric (operations::Metric::utility);
  REQUIRE (utility.has_value ());
  // ΔSSE = (2 - 1) / 2 = 0.5
  // utility = ΔSSE × (0.5 + 0.5F) × (1 - 0.3S)
  // = 0.5 × 0.75 × 1.0 = 0.375
  REQUIRE (*utility == Catch::Approx (0.375).epsilon (1e-6));
}

TEST_CASE ("ComputeMetrics relevance fallback uses 0.5 baseline with empty context",
           "[operations][metrics][relevance]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  ProcessorContext pctx;

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.6;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);
  ComputeMetrics op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  auto relevance = ctx.GetMetric (operations::Metric::relevance);
  REQUIRE (relevance.has_value ());
  const double expected = 0.5 * (0.5 + 0.5 * cfg.focus);
  REQUIRE (*relevance == Catch::Approx (expected).epsilon (1e-6));
}
