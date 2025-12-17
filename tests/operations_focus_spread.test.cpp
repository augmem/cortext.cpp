#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/effective_focus.hpp>
#include <cortext/operations/focus_spread.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ComputeEffectiveFocus;
using cortext::operations::ComputeFocusSpread;
using cortext::operations::ComputeMetrics;

TEST_CASE ("ComputeFocusSpread produces high entropy for uniform similarities",
           "[operations][focus_spread][alg11]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (8);
  s.embedding.normalize ();
  ProcessorContext pctx;
  // Seed recent context with multiple identical embeddings to produce uniform
  // sims.
  for (int i = 0; i < 6; ++i)
    {
      pctx.recent_context_embeddings.push_back (s.embedding);
    }

  SignalProcessor::Config cfg;
  cfg.focus = 0.6;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);

  ComputeFocusSpread op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  auto v = ctx.GetMetric (operations::Metric::focus_spread);
  REQUIRE (v.has_value ());
  REQUIRE (*v <= 1.0);
  REQUIRE (*v >= 0.0);
  // Uniform similarities → softmax uniform → normalized entropy ≈ 1
  REQUIRE (*v == Catch::Approx (1.0).margin (1e-6));
}

TEST_CASE ("Effective focus is reduced when focus_spread is high",
           "[operations][focus_spread][feff]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Random (16);
  if (s.embedding.norm () == 0.0f)
    {
      s.embedding = Eigen::VectorXf::Ones (16);
    }
  s.embedding.normalize ();

  SignalProcessor::Config cfg;
  cfg.focus = 0.7;
  cfg.sensitivity = 0.4;
  cfg.stability = 0.5;


  // High-spread context: many identical (uniform sims)
  ProcessorContext p_high;
  for (int i = 0; i < 6; ++i)
    {
      p_high.recent_context_embeddings.push_back (s.embedding);
    }
  OperationContext ctx_high (s, p_high, cfg);
  ctx_high.SetCoherence (1.0); // isolate spread effect
  ComputeFocusSpread spread_op;
  spread_op.Execute (ctx_high, cortext::testing::GetNullTransaction ());
  ComputeEffectiveFocus feff;
  feff.Execute (ctx_high, cortext::testing::GetNullTransaction ());
  const double f_eff_high = ctx_high.GetEffectiveFocus ();

  // Low-spread context: one very similar, others dissimilar
  ProcessorContext p_low;
  p_low.recent_context_embeddings.push_back (s.embedding); // high sim
  for (int i = 0; i < 5; ++i)
    {
      Eigen::VectorXf v = -s.embedding; // low/negative similarity
      p_low.recent_context_embeddings.push_back (v);
    }
  OperationContext ctx_low (s, p_low, cfg);
  ctx_low.SetCoherence (1.0); // isolate spread effect
  spread_op.Execute (ctx_low, cortext::testing::GetNullTransaction ());
  feff.Execute (ctx_low, cortext::testing::GetNullTransaction ());
  const double f_eff_low = ctx_low.GetEffectiveFocus ();

  REQUIRE (f_eff_high <= f_eff_low);
  REQUIRE (f_eff_low <= cfg.focus); // coherence=1, no spread cap beyond base F
}
