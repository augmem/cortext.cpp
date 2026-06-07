#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/short_term_memory_shadow.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <algorithm>

using namespace cortext;

namespace
{

constexpr int kEmbeddingDim = 256;

Eigen::VectorXf
Vec (float x, float y)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = x;
  v[1] = y;
  const float n = v.norm ();
  if (n > 1e-9f)
    {
      v /= n;
    }
  return v;
}

Signal
MakeSignal (const Eigen::VectorXf &embedding, uint64_t ts)
{
  Signal s;
  s.source_id = "julie-smoke";
  s.timestamp = ts;
  s.embedding = embedding;
  return s;
}

void
SeedLabel (Store &store, ProcessorContext &p_ctx, long long memory_id,
           long long embedding_id, const std::string &label,
           const Eigen::VectorXf &embedding)
{
  cortext::testing::SeedEmbeddingV2 (store, embedding_id, embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (store, memory_id, embedding_id, label,
                                  "LABEL", 1.0, 1000LL);
  store.Execute ("UPDATE memories SET label = ? WHERE memory_id = ?",
                 { label, memory_id });
  p_ctx.UpsertSummaryCache (memory_id, embedding_id, embedding, false, true);
}

} // namespace

TEST_CASE ("STM shadow reuses cached label clusters and selects top labels",
           "[operations][stm][labels]")
{
  cortext::testing::ScopedEnvVar enable ("CORTEXT_STM_SHADOW_ENABLE", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  ProcessorContext p_ctx;
  SeedLabel (*store, p_ctx, 101LL, 201LL, "kitchen table", Vec (1.0f, 0.0f));
  SeedLabel (*store, p_ctx, 102LL, 202LL, "coffee mug", Vec (0.95f, 0.05f));
  SeedLabel (*store, p_ctx, 103LL, 203LL, "train station", Vec (0.0f, 1.0f));
  SeedLabel (*store, p_ctx, 104LL, 204LL, "airport gate", Vec (0.05f, 0.95f));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const auto expected_cluster_count = static_cast<size_t> (std::min (
      4, cortext::core::RetrievalClusterLabelK (cfg.focus, cfg.stability)));
  operations::UpdateShortTermMemoryShadow op;

  {
    auto tx = store->Begin ();
    OperationContext ctx (MakeSignal (Vec (1.0f, 0.0f), 2000ULL), p_ctx, cfg,
                          store.get ());
    op.Execute (ctx, *tx);
    tx->Commit ();
  }

  REQUIRE (p_ctx.label_cluster_cache.valid);
  REQUIRE (p_ctx.label_cluster_cache.centroids.size ()
           == expected_cluster_count);
  REQUIRE_FALSE (p_ctx.short_term_graphs["julie-smoke"].label_edges.empty ());
  const auto first_centroid_0 = p_ctx.label_cluster_cache.centroids[0];
  const auto first_centroid_1 = p_ctx.label_cluster_cache.centroids[1];

  {
    auto tx = store->Begin ();
    OperationContext ctx (MakeSignal (Vec (1.0f, 0.0f), 3000ULL), p_ctx, cfg,
                          store.get ());
    op.Execute (ctx, *tx);
    tx->Commit ();
  }

  REQUIRE (p_ctx.label_cluster_cache.valid);
  REQUIRE (p_ctx.label_cluster_cache.centroids.size ()
           == expected_cluster_count);
  REQUIRE (p_ctx.label_cluster_cache.centroids[0].isApprox (first_centroid_0));
  REQUIRE (p_ctx.label_cluster_cache.centroids[1].isApprox (first_centroid_1));
}

TEST_CASE ("STM label cluster cache invalidates when labels change",
           "[operations][stm][labels]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  ProcessorContext p_ctx;
  SeedLabel (*store, p_ctx, 101LL, 201LL, "kitchen table", Vec (1.0f, 0.0f));
  p_ctx.label_cluster_cache.valid = true;
  p_ctx.label_cluster_cache.requested_clusters = 1;
  p_ctx.label_cluster_cache.embedding_dim = kEmbeddingDim;
  p_ctx.label_cluster_cache.summary_cache_size = p_ctx.summary_cache.size ();

  SeedLabel (*store, p_ctx, 102LL, 202LL, "train station", Vec (0.0f, 1.0f));

  REQUIRE_FALSE (p_ctx.label_cluster_cache.valid);
}

TEST_CASE ("STM label selection expands top labels from flat-routed clusters",
           "[operations][stm][labels]")
{
  cortext::testing::ScopedEnvVar enable ("CORTEXT_STM_SHADOW_ENABLE", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  ProcessorContext p_ctx;
  SeedLabel (*store, p_ctx, 101LL, 201LL, "alice guitar", Vec (1.0f, 0.0f));
  SeedLabel (*store, p_ctx, 102LL, 202LL, "bailey garden", Vec (0.92f, 0.08f));
  SeedLabel (*store, p_ctx, 103LL, 203LL, "train station", Vec (0.0f, 1.0f));
  SeedLabel (*store, p_ctx, 104LL, 204LL, "airport gate", Vec (0.08f, 0.92f));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  operations::UpdateShortTermMemoryShadow op;

  auto tx = store->Begin ();
  OperationContext ctx (MakeSignal (Vec (1.0f, 0.0f), 2000ULL), p_ctx, cfg,
                        store.get ());
  op.Execute (ctx, *tx);
  tx->Commit ();

  const auto &edges = p_ctx.short_term_graphs["julie-smoke"].label_edges;
  REQUIRE (edges.size () >= 2);
  const auto has_alice = std::any_of (
      edges.begin (), edges.end (), [] (const auto &edge) {
        return edge.label == "alice guitar";
      });
  const auto has_bailey = std::any_of (
      edges.begin (), edges.end (), [] (const auto &edge) {
        return edge.label == "bailey garden";
      });
  REQUIRE (has_alice);
  REQUIRE (has_bailey);
}
