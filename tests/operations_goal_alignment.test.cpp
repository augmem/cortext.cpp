// tests/operations_goal_alignment.test.cpp
#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/operations/goal_alignment.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::ComputeGoalAlignment;
using cortext::operations::Metric;

namespace
{
constexpr int kEmbeddingDim = 256;

/// @brief Creates a 256D unit vector with first two dimensions set.
static Eigen::VectorXf
Unit256 (float x, float y)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = x;
  v[1] = y;
  const float n = v.norm ();
  if (n > 1e-9f)
    v /= n;
  return v;
}

static Signal
MakeSignal (const Eigen::VectorXf &emb)
{
  Signal s;
  s.embedding = emb;
  s.timestamp = 1;
  s.source_id = "test";
  return s;
}
} // namespace

TEST_CASE ("Alg33 computes goal alignment from goal neighborhood",
           "[operations][graph][alg33]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create 256D embedding vectors
  Eigen::VectorXf emb1 = Unit256 (1.0f, 0.0f);
  Eigen::VectorXf emb2 = Unit256 (0.0f, 1.0f);
  std::vector<float> emb1_vec (emb1.data (), emb1.data () + emb1.size ());
  std::vector<float> emb2_vec (emb2.data (), emb2.data () + emb2.size ());

  // Two embeddings in the neighborhood: goal (id=1) and neighbor (id=2).
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, type, strength, "
                  "use_frequency, stability, connectivity, drift_mag, influence, "
                  "sustained_influence, contextual_gain, redundancy, pre_activation, "
                  "lability_state, suppression_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  { 1LL, emb1_vec, std::string ("goal"), 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0LL });
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, type, strength, "
                  "use_frequency, stability, connectivity, drift_mag, influence, "
                  "sustained_influence, contextual_gain, redundancy, pre_activation, "
                  "lability_state, suppression_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  { 2LL, emb2_vec, std::string ("memory"), 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0LL });

  store->Execute ("INSERT INTO graph_nodes(node_id, type, embedding_id) "
                  "VALUES (?,?,?)",
                  { std::string ("goal:g"), std::string ("goal"), 1LL });
  store->Execute ("INSERT INTO graph_nodes(node_id, type, embedding_id) "
                  "VALUES (?,?,?)",
                  { std::string ("emb:2"), std::string ("memory"), 2LL });
  store->Execute ("INSERT INTO goal_nodes(node_id) VALUES (?)",
                  { std::string ("goal:g") });
  store->Execute (
      "INSERT INTO graph_edges(source_id, target_id, edge_type, weight) "
      "VALUES (?,?,?,?)",
      { std::string ("goal:g"), std::string ("emb:2"), std::string ("links"), 1.0 });

  SignalProcessor::Config cfg;
  auto ops = std::make_unique<OperationSet> (std::make_unique<ComputeGoalAlignment> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  auto out = processor.Process (MakeSignal (Unit256 (1.0f, 0.0f)));

  // Metric should be present and fairly high (mean of [1,0] and [0,1] has cos 0.707..).
  auto it = out.metrics.find (Metric::goal_alignment);
  REQUIRE (it != out.metrics.end ());
  const double ga = it->second;
  REQUIRE (ga > 0.8);
  REQUIRE (ga < 0.9);
}

