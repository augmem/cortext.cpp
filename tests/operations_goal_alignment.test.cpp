// tests/operations_goal_alignment.test.cpp
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
static Eigen::VectorXf
Unit2 (float x, float y)
{
  Eigen::VectorXf v (2);
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

  store->Execute ("CREATE TABLE IF NOT EXISTS embeddings ("
                  "embedding_id INTEGER PRIMARY KEY,"
                  "embedding BLOB"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS graph_nodes ("
                  "node_id TEXT PRIMARY KEY,"
                  "type TEXT NOT NULL,"
                  "label TEXT,"
                  "embedding_id INTEGER,"
                  "metadata TEXT,"
                  "created_at INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS graph_edges ("
                  "source_id TEXT NOT NULL,"
                  "target_id TEXT NOT NULL,"
                  "edge_type TEXT NOT NULL,"
                  "weight REAL NOT NULL DEFAULT 1.0,"
                  "decay_rate REAL,"
                  "last_reinforced INTEGER,"
                  "PRIMARY KEY (source_id, target_id, edge_type)"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS goal_nodes (node_id TEXT PRIMARY KEY);");

  // Two embeddings in the neighborhood: goal (id=1) and neighbor (id=2).
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding) VALUES (?,?)",
                  { 1LL, std::vector<float>{ 1.0f, 0.0f } });
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding) VALUES (?,?)",
                  { 2LL, std::vector<float>{ 0.0f, 1.0f } });

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

  auto out = processor.Process (MakeSignal (Unit2 (1.0f, 0.0f)));

  // Metric should be present and fairly high (mean of [1,0] and [0,1] has cos 0.707..).
  auto it = out.metrics.find (Metric::goal_alignment);
  REQUIRE (it != out.metrics.end ());
  const double ga = it->second;
  REQUIRE (ga > 0.8);
  REQUIRE (ga < 0.9);
}

