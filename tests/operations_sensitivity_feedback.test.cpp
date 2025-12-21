#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/operations/sensitivity_feedback.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ApplySensitivityFeedback;
using cortext::operations::ComputeMetrics;
using cortext::operations::InitializeSensitivityPriors;

namespace
{

static Eigen::VectorXf
unit (int idx, int dim)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (dim);
  v[idx] = 1.0f;
  return v;
}

} // namespace

TEST_CASE ("Alg16 positive gain with novelty increases weight_novelty",
           "[operations][sensitivity_feedback]")
{
  const int dim = 4;
  Signal s;
  s.embedding = unit (0, dim); // x along axis 0
  s.timestamp = 1;

  ProcessorContext pctx;
  // Provide a recent context orthogonal to x to get novelty ~ 0.5
  pctx.recent_context_embeddings.push_back (unit (1, dim));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.8;
  cfg.stability = 0.5;


  OperationContext ctx (s, pctx, cfg);

  // Initialize sensitivity priors and dynamic novelty weight
  InitializeSensitivityPriors init;
  init.Execute (ctx, cortext::testing::GetNullTransaction ());
  auto &pc = ctx.GetProcessorContext ();
  const double before = pc.weight_novelty;

  // Compute metrics so 'relevance' is available
  ComputeMetrics metrics;
  metrics.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Attach a positive contextual gain event
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 42LL;
  ev.used = true;
  ev.contextual_gain = 0.5; // positive gain
  ctx.SetMemoryUsageEvents ({ ev });

  ApplySensitivityFeedback op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pc.weight_novelty > before);
  REQUIRE (pc.weight_novelty <= 1.0);
}

TEST_CASE (
    "Alg16 negative gain with novelty decreases or holds novelty weight",
    "[operations][sensitivity_feedback]")
{
  const int dim = 4;
  Signal s;
  s.embedding = unit (0, dim); // x along axis 0
  s.timestamp = 2;

  ProcessorContext pctx;
  // Provide a recent context orthogonal to x to get novelty ~ 0.5
  pctx.recent_context_embeddings.push_back (unit (1, dim));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.6;
  cfg.sensitivity = 0.7;
  cfg.stability = 0.5;


  OperationContext ctx (s, pctx, cfg);

  InitializeSensitivityPriors init;
  init.Execute (ctx, cortext::testing::GetNullTransaction ());
  auto &pc = ctx.GetProcessorContext ();
  const double before = pc.weight_novelty;

  ComputeMetrics metrics;
  metrics.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Attach a negative contextual gain event
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 7LL;
  ev.used = true;
  ev.contextual_gain = -0.4; // negative
  ctx.SetMemoryUsageEvents ({ ev });

  ApplySensitivityFeedback op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pc.weight_novelty <= before);
  REQUIRE (pc.weight_novelty >= 0.0);
}
