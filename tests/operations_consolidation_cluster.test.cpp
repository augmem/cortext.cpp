// tests/operations_consolidation_cluster.test.cpp
#include <Eigen/Dense>
#include <any>
#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/consolidation_cluster.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <string>
#include <vector>

using namespace cortext;
using cortext::operations::ConsolidationCluster;

namespace
{

static Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (256);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

/// @brief Op to seed consolidation_candidates and embeddings.
struct SeedCandidatesOp : IOperation
{
  struct Candidate
  {
    long long embedding_id;
    double score;
    std::vector<float> embedding;
  };
  std::vector<Candidate> candidates_;

  explicit SeedCandidatesOp (std::vector<Candidate> candidates)
      : candidates_ (std::move (candidates))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    std::vector<ConsolidationCandidate> consolidation_candidates;
    consolidation_candidates.reserve (candidates_.size ());

    for (const auto &c : candidates_)
      {
        ConsolidationCandidate cc;
        cc.embedding_id = c.embedding_id;
        cc.score = c.score;

        // Convert std::vector<float> to Eigen::VectorXf
        cc.embedding.resize (c.embedding.size ());
        for (size_t i = 0; i < c.embedding.size (); ++i)
          {
            cc.embedding[static_cast<int> (i)] = c.embedding[i];
          }

        consolidation_candidates.push_back (std::move (cc));
      }

    ctx.SetConsolidationCandidates (std::move (consolidation_candidates));
  }
};

/// @brief Op to enable consolidation.
struct EnableConsolidationOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetConsolidationShouldStart (true);
  }
};

/// @brief Op to verify cluster output.
struct AssertClustersOp : IOperation
{
  int expected_cluster_count_;

  explicit AssertClustersOp (int expected) : expected_cluster_count_ (expected)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    const auto &clusters = ctx.GetConsolidationClusters ();
    REQUIRE (static_cast<int> (clusters.size ()) == expected_cluster_count_);
  }
};

} // namespace

/// @brief Create a 256-dim embedding with a value at a specific index.
std::vector<float>
Make256Embedding (int primary_index, float value = 1.0f)
{
  std::vector<float> emb (256, 0.0f);
  emb[static_cast<size_t> (primary_index)] = value;
  return emb;
}

/// @brief Create a 256-dim embedding similar to another.
std::vector<float>
Make256EmbeddingSimilar (int primary_index, float noise = 0.1f)
{
  std::vector<float> emb (256, 0.0f);
  emb[static_cast<size_t> (primary_index)] = 1.0f - noise;
  emb[static_cast<size_t> ((primary_index + 1) % 256)] = noise;
  return emb;
}

TEST_CASE ("ConsolidationCluster groups similar embeddings",
           "[operations][consolidation_cluster]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0; // Low focus = lower merge threshold, larger clusters
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Create 4 embeddings: 3 similar (should cluster), 1 different
  std::vector<SeedCandidatesOp::Candidate> candidates = {
    { 1LL, 0.1, Make256Embedding (0) }, // Group A
    { 2LL, 0.1, Make256EmbeddingSimilar (0, 0.05f) }, // Group A
    { 3LL, 0.1, Make256EmbeddingSimilar (0, 0.08f) }, // Group A
    { 4LL, 0.1, Make256Embedding (128) }, // Group B (different)
  };

  auto seed = std::make_unique<SeedCandidatesOp> (candidates);
  auto enable = std::make_unique<EnableConsolidationOp> ();
  auto cluster = std::make_unique<ConsolidationCluster> ();
  auto assert_op = std::make_unique<AssertClustersOp> (1);
  // At F=0, min_cluster_size=3, so Group A with 3 members passes

  auto ops = std::make_unique<DynamicOperationSet> (
      std::move (seed), std::move (enable), std::move (cluster),
      std::move (assert_op));

  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (1000));
  processor.Flush ();
}

TEST_CASE ("ConsolidationCluster filters small clusters",
           "[operations][consolidation_cluster]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0; // High focus = min_cluster_size=10
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Only 2 similar embeddings - won't meet min_cluster_size
  std::vector<SeedCandidatesOp::Candidate> candidates = {
    { 1LL, 0.1, Make256Embedding (0) },
    { 2LL, 0.1, Make256EmbeddingSimilar (0, 0.05f) },
  };

  auto seed = std::make_unique<SeedCandidatesOp> (candidates);
  auto enable = std::make_unique<EnableConsolidationOp> ();
  auto cluster = std::make_unique<ConsolidationCluster> ();
  auto assert_op = std::make_unique<AssertClustersOp> (0); // No clusters

  auto ops = std::make_unique<DynamicOperationSet> (
      std::move (seed), std::move (enable), std::move (cluster),
      std::move (assert_op));

  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (1000));
  processor.Flush ();
}

TEST_CASE ("ConsolidationCluster returns empty on no candidates",
           "[operations][consolidation_cluster]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto enable = std::make_unique<EnableConsolidationOp> ();
  auto cluster = std::make_unique<ConsolidationCluster> ();
  auto assert_op = std::make_unique<AssertClustersOp> (0);

  auto ops = std::make_unique<DynamicOperationSet> (
      std::move (enable), std::move (cluster), std::move (assert_op));

  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (1000));
  processor.Flush ();
}

TEST_CASE ("ConsolidationCluster skips when consolidation not started",
           "[operations][consolidation_cluster]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  std::vector<SeedCandidatesOp::Candidate> candidates = {
    { 1LL, 0.1, Make256Embedding (0) },
    { 2LL, 0.1, Make256EmbeddingSimilar (0, 0.05f) },
    { 3LL, 0.1, Make256EmbeddingSimilar (0, 0.08f) },
  };

  auto seed = std::make_unique<SeedCandidatesOp> (candidates);
  // No EnableConsolidationOp - consolidation not started
  auto cluster = std::make_unique<ConsolidationCluster> ();
  auto assert_op = std::make_unique<AssertClustersOp> (0);

  auto ops = std::make_unique<DynamicOperationSet> (
      std::move (seed), std::move (cluster), std::move (assert_op));

  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (1000));
  processor.Flush ();
}
