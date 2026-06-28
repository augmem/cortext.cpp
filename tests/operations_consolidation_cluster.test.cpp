// tests/operations_consolidation_cluster.test.cpp
#include <Eigen/Dense>
#include <any>
#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/consolidation_cluster.hpp>
#include <cortext/operations/consolidation_shallow.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include <string>
#include <vector>

using namespace cortext;
using cortext::operations::ConsolidationCluster;
using cortext::operations::ConsolidationShallow;

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
        cc.memory_id = c.embedding_id;
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

struct AssertSingleClusterOp : IOperation
{
  std::vector<long long> expected_ids_;
  int expected_centroid_size_;
  int centroid_index_;
  double expected_centroid_value_;

  AssertSingleClusterOp (std::vector<long long> expected_ids,
                         int expected_centroid_size, int centroid_index,
                         double expected_centroid_value)
      : expected_ids_ (std::move (expected_ids)),
        expected_centroid_size_ (expected_centroid_size),
        centroid_index_ (centroid_index),
        expected_centroid_value_ (expected_centroid_value)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    const auto &clusters = ctx.GetConsolidationClusters ();
    REQUIRE (clusters.size () == 1);
    REQUIRE (clusters[0].embedding_ids == expected_ids_);
    REQUIRE (static_cast<int> (clusters[0].centroid.size ())
             == expected_centroid_size_);
    REQUIRE (clusters[0].centroid[static_cast<size_t> (centroid_index_)]
             == Catch::Approx (expected_centroid_value_));
  }
};

struct SetClustersOp : IOperation
{
  std::vector<ClusterInfo> clusters_;

  explicit SetClustersOp (std::vector<ClusterInfo> clusters)
      : clusters_ (std::move (clusters))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetConsolidationClusters (clusters_);
  }
};

struct AssertLabelEdgesOp : IOperation
{
  explicit AssertLabelEdgesOp (long long expected) : expected_ (expected)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    (void)ctx;
    auto rows = tx.Execute (
        "SELECT COUNT(*) AS cnt FROM associations "
        "WHERE edge_type = 'has_label'",
        {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == expected_);
  }

  long long expected_;
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

TEST_CASE ("ConsolidationCluster excludes mixed-dimension candidates from "
           "membership",
           "[operations][consolidation_cluster][robustness]")
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
    { 4LL, 0.1, std::vector<float>{ 1.0f, 0.0f } },
  };

  auto seed = std::make_unique<SeedCandidatesOp> (candidates);
  auto enable = std::make_unique<EnableConsolidationOp> ();
  auto cluster = std::make_unique<ConsolidationCluster> ();
  auto assert_op = std::make_unique<AssertSingleClusterOp> (
      std::vector<long long>{ 1LL, 2LL, 3LL }, 256, 0,
      (1.0 + 0.95 + 0.92) / 3.0);

  auto ops = std::make_unique<DynamicOperationSet> (
      std::move (seed), std::move (enable), std::move (cluster),
      std::move (assert_op));

  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (1000));
  processor.Flush ();
}

TEST_CASE ("ConsolidationCluster forced fallback skips empty lowest-score "
           "anchor",
           "[operations][consolidation_cluster][robustness]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  std::vector<SeedCandidatesOp::Candidate> candidates = {
    { 1LL, 0.0, {} },
    { 2LL, 0.2, Make256Embedding (0) },
    { 3LL, 0.3, Make256Embedding (64) },
    { 4LL, 0.4, Make256Embedding (128) },
  };

  auto seed = std::make_unique<SeedCandidatesOp> (candidates);
  auto enable = std::make_unique<EnableConsolidationOp> ();
  auto cluster = std::make_unique<ConsolidationCluster> ();
  auto assert_op = std::make_unique<AssertSingleClusterOp> (
      std::vector<long long>{ 2LL, 3LL, 4LL }, 256, 0, 1.0 / 3.0);

  auto ops = std::make_unique<DynamicOperationSet> (
      std::move (seed), std::move (enable), std::move (cluster),
      std::move (assert_op));

  Signal signal = MakeSignal (1000);
  signal.force_consolidation = true;
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (signal);
  processor.Flush ();
}

TEST_CASE ("ConsolidationShallow uses first non-empty cluster centroid dimension",
           "[operations][consolidation_shallow][robustness]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> centroid = Make256Embedding (0);
  cortext::testing::SeedEmbeddingV2 (*store, 100LL, centroid, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "test", "LONG_TERM",
                                  1.0, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, centroid, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL, "label", "LABEL",
                                  1.0, 1000);

  ClusterInfo empty_first;
  empty_first.cluster_id = 1;
  empty_first.avg_score = 0.0;

  ClusterInfo real_cluster;
  real_cluster.cluster_id = 2;
  real_cluster.embedding_ids = { 100LL };
  real_cluster.centroid = centroid;
  real_cluster.avg_score = 0.1;

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto enable = std::make_unique<EnableConsolidationOp> ();
  auto set_clusters = std::make_unique<SetClustersOp> (
      std::vector<ClusterInfo>{ empty_first, real_cluster });
  auto shallow = std::make_unique<ConsolidationShallow> ();
  auto assert_edges = std::make_unique<AssertLabelEdgesOp> (1LL);

  auto ops = std::make_unique<DynamicOperationSet> (
      std::move (enable), std::move (set_clusters), std::move (shallow),
      std::move (assert_edges));

  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (5000));
  processor.Flush ();
}

TEST_CASE ("ConsolidationShallow advances completion state only after persistence",
           "[operations][consolidation_shallow]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> centroid = Make256Embedding (0);
  cortext::testing::SeedEmbeddingV2 (*store, 100LL, centroid, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "test", "LONG_TERM",
                                  1.0, 1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pctx;
  pctx.last_consolidation_ts = 1234ULL;
  pctx.consolidation_count = 2;
  pctx.memories_since_consolidation = 5;
  pctx.association_fanout_cache.valid = true;
  auto signal = MakeSignal (5000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetConsolidationShouldStart (true);

  ClusterInfo cluster;
  cluster.cluster_id = 9;
  cluster.memory_ids = { 100LL };
  cluster.embedding_ids = { 100LL };
  cluster.centroid = centroid;
  cluster.avg_score = 0.1;
  ctx.SetConsolidationClusters ({ cluster });

  ConsolidationShallow shallow;
  auto tx = store->Begin ();
  shallow.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (ctx.GetConsolidationPersisted ());
  REQUIRE (pctx.last_consolidation_ts == 5000ULL);
  REQUIRE (pctx.consolidation_count == 3);
  REQUIRE (pctx.memories_since_consolidation == 0);
  REQUIRE_FALSE (pctx.association_fanout_cache.valid);

  auto assoc_rows = store->Execute (
      "SELECT memory_id, embedding_id, source_id FROM memories "
      "WHERE kind = 'ASSOCIATION' "
      "ORDER BY memory_id DESC LIMIT 1",
      {});
  REQUIRE (assoc_rows.size () == 1);
  const long long assoc_memory_id = cortext::store::AnyToLongLong (
      assoc_rows[0].at ("memory_id")).value_or (0);
  const long long assoc_embedding_id = cortext::store::AnyToLongLong (
      assoc_rows[0].at ("embedding_id")).value_or (0);
  REQUIRE (assoc_memory_id > 0);
  REQUIRE (assoc_embedding_id > 0);

  auto current_rows = store->Execute (
      "SELECT embedding_id FROM current_memory_embeddings "
      "WHERE memory_id = ?",
      { assoc_memory_id });
  REQUIRE (current_rows.size () == 1);
  REQUIRE (cortext::store::AnyToLongLong (
               current_rows[0].at ("embedding_id")).value_or (0)
           == assoc_embedding_id);

  auto surface_it = pctx.retrieval_surface_index.find (assoc_memory_id);
  REQUIRE (surface_it != pctx.retrieval_surface_index.end ());
  const auto &surface = pctx.retrieval_surface_cache[surface_it->second];
  REQUIRE (surface.embedding_id == assoc_embedding_id);
  REQUIRE (surface.kind == "ASSOCIATION");
  REQUIRE (surface.is_association);
  REQUIRE (surface.embedding.size () == 256);
  REQUIRE (surface.embedding[0] == Catch::Approx (1.0f));
  REQUIRE (pctx.retrieval_surface_embedding_index.at (assoc_embedding_id)
           == surface_it->second);

  auto assoc_cache_it = pctx.association_cache_index.find (assoc_memory_id);
  REQUIRE (assoc_cache_it != pctx.association_cache_index.end ());
  const auto &assoc_cache = pctx.association_cache[assoc_cache_it->second];
  REQUIRE (assoc_cache.embedding_id == assoc_embedding_id);
  REQUIRE (assoc_cache.is_association);
}
