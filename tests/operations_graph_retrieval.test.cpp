// tests/operations_graph_retrieval.test.cpp
#include <any>
#include <catch2/catch_test_macros.hpp>

#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::GraphAugmentedRetrieveCandidates;

namespace
{
constexpr int kEmbeddingDim = 256;

/// @brief Creates a 256-dim unit vector with value at index 0.
static Eigen::VectorXf
UnitVec256 (float first_val)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = first_val;
  const float n = v.norm ();
  if (n > 1e-9f)
    v /= n;
  return v;
}

/// @brief Creates a 256-dim unit vector with value at index 1.
static Eigen::VectorXf
UnitVec256Second (float second_val)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[1] = second_val;
  const float n = v.norm ();
  if (n > 1e-9f)
    v /= n;
  return v;
}

/// @brief Converts Eigen vector to std::vector<float> for DB storage.
static std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

static Signal
MakeSignal (const Eigen::VectorXf &emb, uint64_t ts)
{
  Signal s;
  s.embedding = emb;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}
} // namespace

TEST_CASE ("Alg31 expands vector seeds via graph_edges and returns expanded ids",
           "[operations][graph][alg31]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create 256-dim embeddings for vec0 compatibility.
  Eigen::VectorXf emb1 = UnitVec256 (1.0f);   // First dimension = 1
  Eigen::VectorXf emb2 = UnitVec256Second (1.0f); // Second dimension = 1

  // Unified embeddings table using vec0 with auxiliary columns.
  store->Execute ("CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0("
                  "embedding_id INTEGER PRIMARY KEY,"
                  "embedding float[256],"
                  "+type text,"
                  "+strength float,"
                  "+use_frequency float,"
                  "+stability float,"
                  "+connectivity float,"
                  "+drift_mag float,"
                  "+influence float,"
                  "+sustained_influence float,"
                  "+contextual_gain float,"
                  "+redundancy float,"
                  "+pre_activation float,"
                  "+lability_state float,"
                  "+suppression_count integer,"
                  "+cluster_id integer,"
                  "+last_access integer,"
                  "+created_at integer"
                  ");");

  // id=1 aligns with query, id=2 does not.
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, type, strength, "
                  "use_frequency, stability, connectivity, drift_mag, influence, "
                  "sustained_influence, contextual_gain, redundancy, pre_activation, "
                  "lability_state, suppression_count) VALUES (?,?,'memory',1.0,0.0,"
                  "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0)",
                  { 1LL, ToFloatVec (emb1) });
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, type, strength, "
                  "use_frequency, stability, connectivity, drift_mag, influence, "
                  "sustained_influence, contextual_gain, redundancy, pre_activation, "
                  "lability_state, suppression_count) VALUES (?,?,'memory',1.0,0.0,"
                  "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0)",
                  { 2LL, ToFloatVec (emb2) });

  // Graph tables connecting emb:1 -> entity:Alice -> emb:2
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

  store->Execute ("INSERT INTO graph_nodes(node_id, type, embedding_id) "
                  "VALUES (?,?,?)",
                  { std::string ("emb:1"), std::string ("memory"), 1LL });
  store->Execute ("INSERT INTO graph_nodes(node_id, type, embedding_id) "
                  "VALUES (?,?,?)",
                  { std::string ("emb:2"), std::string ("memory"), 2LL });
  store->Execute ("INSERT INTO graph_nodes(node_id, type, label) VALUES (?,?,?)",
                  { std::string ("entity:Alice"), std::string ("entity"),
                    std::string ("Alice") });

  store->Execute ("INSERT INTO graph_edges(source_id, target_id, edge_type, weight) "
                  "VALUES (?,?,?,?)",
                  { std::string ("emb:1"), std::string ("entity:Alice"),
                    std::string ("mentions"), 1.0 });
  store->Execute ("INSERT INTO graph_edges(source_id, target_id, edge_type, weight) "
                  "VALUES (?,?,?,?)",
                  { std::string ("entity:Alice"), std::string ("emb:2"),
                    std::string ("mentions"), 1.0 });

  // Low focus => depth=2 (GraphDepth(F)), ensuring expansion reaches emb:2.
  SignalProcessor::Config cfg;
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto ops = std::make_unique<OperationSet> (
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  auto out = processor.Process (MakeSignal (UnitVec256 (1.0f), 10));

  // Should include both seed (1) and expanded (2).
  bool has1 = false, has2 = false;
  for (const auto id : out.candidate_memory_ids)
    {
      if (id == 1LL)
        has1 = true;
      if (id == 2LL)
        has2 = true;
    }
  REQUIRE (has1);
  REQUIRE (has2);
}

