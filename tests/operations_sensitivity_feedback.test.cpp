#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/operations/sensitivity_feedback.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <unordered_map>

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
  // Provide a recent context opposite of x to get redundancy ~0 (novelty ~1)
  pctx.recent_context_embeddings.push_back (-unit (0, dim));

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

  // Attach a positive contextual gain event with embedding available
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 42LL;
  ev.used = true;
  ev.contextual_gain = 0.5; // positive gain
  ctx.SetMemoryUsageEvents ({ ev });
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 42LL, s.embedding } });

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
  // Provide a recent context opposite of x to get redundancy ~0 (novelty ~1)
  pctx.recent_context_embeddings.push_back (-unit (0, dim));

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

  // Attach a negative contextual gain event with embedding available
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 7LL;
  ev.used = true;
  ev.contextual_gain = -0.4; // negative
  ctx.SetMemoryUsageEvents ({ ev });
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 7LL, s.embedding } });

  ApplySensitivityFeedback op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pc.weight_novelty <= before);
  REQUIRE (pc.weight_novelty >= 0.0);
}

TEST_CASE ("Alg16 updates redundancy by memory id when embeddings are shared",
           "[operations][sensitivity_feedback]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf embedding = unit (0, 256);
  const Eigen::VectorXf reconstructed = unit (1, 256);
  cortext::testing::SeedEmbeddingV2 (*store, 420LL, embedding, 1);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 420LL, "used",
                                  "LONG_TERM", 1.0, 1);
  cortext::testing::SeedMemoryV2 (*store, 101LL, 420LL, "sibling",
                                  "LONG_TERM", 1.0, 1);

  Signal s;
  s.embedding = embedding;
  s.timestamp = 3;
  ProcessorContext pctx;
  pctx.recent_context_embeddings.push_back (embedding);
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  OperationContext ctx (s, pctx, cfg, store.get ());
  OperationContext::MemoryUsageEvent ev{};
  ev.memory_id = 100LL;
  ev.embedding_id = 420LL;
  ev.used = true;
  ev.contextual_gain = 0.5;
  ctx.SetMemoryUsageEvents ({ ev });
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 420LL, embedding } });
  ctx.SetRetrievedMemoryCandidates (
      std::vector<OperationContext::RetrievedMemoryCandidate>{
        { 100LL, 420LL, reconstructed, 1.0 } });

  ApplySensitivityFeedback op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT memory_id, redundancy FROM memories "
      "WHERE memory_id IN (100, 101) ORDER BY memory_id",
      {});
  REQUIRE (rows.size () == 2);
  REQUIRE (std::any_cast<double> (rows[0].at ("redundancy")) < 0.75);
  REQUIRE (std::any_cast<double> (rows[1].at ("redundancy")) == 0.0);
}
