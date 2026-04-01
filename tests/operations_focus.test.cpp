#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/recent_context.hpp>
#include <cortext/operations/uncertainty.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::InitializeFocusPriors;
using cortext::operations::UpdateFocus;
using cortext::operations::UpdateRecentContext;
using cortext::operations::UpdateUncertainty;

TEST_CASE ("InitializeFocusPriors computes from config", "[operations][focus]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.8;

  OperationContext ctx (s, pctx, cfg);

  InitializeFocusPriors init;
  init.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.weight_relevance_prior
           == Catch::Approx (cortext::core::Sigmoid (2 * cfg.focus - 1)));
  REQUIRE (pctx.attention_width_prior > 0.0);
  REQUIRE (pctx.weight_relevance
           == Catch::Approx (pctx.weight_relevance_prior));
}

TEST_CASE ("UpdateFocus EWMA toward observed cosine and clamping",
           "[operations][focus]")
{
  // Build a context with one prior embedding
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;     // mid
  cfg.stability = 0.5; // mid

  OperationContext ctx (s, pctx, cfg);

  // Priors must be initialized
  InitializeFocusPriors init;
  init.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Seed context with a mean slightly different than current signal to get
  // cosine<1
  pctx.recent_context_embeddings.push_back (0.5f * s.embedding);

  UpdateRecentContext ctx_op;
  ctx_op.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.recent_context_embeddings.size () == 2);

  // Set uncertainty to a mid value to get non-trivial alpha_f
  UpdateUncertainty uop;
  pctx.signals_processed = 10;
  uop.Execute (ctx, cortext::testing::GetNullTransaction ()); // updates pctx.u_t

  double prev = pctx.weight_relevance;
  UpdateFocus op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // After one EWMA step, weight_relevance should move toward observed cosine
  REQUIRE (pctx.weight_relevance > prev);
  REQUIRE (pctx.attention_width >= 0.1 * 3.14159);
  REQUIRE (pctx.attention_width <= 3.14159);

  // Window management: UpdateFocus should not mutate the buffer
  REQUIRE (pctx.recent_context_embeddings.size () == 2);
}
