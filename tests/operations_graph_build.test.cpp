// tests/operations_graph_build.test.cpp
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include <cortext/core/knobs.hpp>
#include <cortext/operations/graph_build.hpp>
#include <cortext/operations/graph_schema.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::BuildGraphFromConsolidation;
using cortext::operations::EnsureGraphSchema;

namespace
{
static Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}
} // namespace

TEST_CASE ("Alg30 builds nodes and edges from extraction outputs",
           "[operations][graph][alg30]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Seed input tables and rows (external worker outputs).
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_summaries ("
                  "  summary_id TEXT PRIMARY KEY,"
                  "  summary_text TEXT,"
                  "  cluster_size INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_sources ("
                  "  summary_id TEXT,"
                  "  source_text TEXT,"
                  "  source_embedding_id INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS extraction_entities ("
                  "  summary_id TEXT NOT NULL,"
                  "  name TEXT NOT NULL,"
                  "  type TEXT NOT NULL,"
                  "  salience REAL,"
                  "  embedding_id INTEGER,"
                  "  PRIMARY KEY (summary_id, name, type)"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS extraction_relations ("
                  "  summary_id TEXT NOT NULL,"
                  "  subject TEXT NOT NULL,"
                  "  predicate TEXT NOT NULL,"
                  "  object TEXT NOT NULL,"
                  "  confidence REAL"
                  ");");

  store->Execute ("INSERT INTO consolidation_summaries(summary_id, summary_text, "
                  "cluster_size) VALUES (?,?,?)",
                  { std::string ("s1"), std::string ("summary one"), 10LL });
  store->Execute ("INSERT INTO consolidation_sources(summary_id, source_text, "
                  "source_embedding_id) VALUES (?,?,?)",
                  { std::string ("s1"), std::string ("source"), 42LL });
  store->Execute ("INSERT INTO extraction_entities(summary_id, name, type, "
                  "salience, embedding_id) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("Alice"),
                    std::string ("Person"), 0.7, 100LL });
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("Alice"),
                    std::string ("KNOWS"), std::string ("Bob"), 0.8 });

  SignalProcessor::Config cfg;
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EnsureGraphSchema> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // Nodes
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS c FROM graph_nodes WHERE node_id IN "
        "('summary:s1','entity:Alice','entity:Bob','emb:42')",
        {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 4LL);
  }

  // mentions edge
  {
    auto rows = store->Execute (
        "SELECT weight FROM graph_edges "
        "WHERE source_id='summary:s1' AND target_id='entity:Alice' "
        "AND edge_type='mentions'",
        {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<double> (rows[0].at ("weight"))
             == Catch::Approx (0.7).margin (1e-6));
  }

  // relation edge
  {
    auto rows = store->Execute (
        "SELECT weight FROM graph_edges "
        "WHERE source_id='entity:Alice' AND target_id='entity:Bob' "
        "AND edge_type='KNOWS'",
        {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<double> (rows[0].at ("weight"))
             == Catch::Approx (0.8).margin (1e-6));
  }

  // derived_from edge
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS c FROM graph_edges "
        "WHERE source_id='summary:s1' AND target_id='emb:42' "
        "AND edge_type='derived_from'",
        {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 1LL);
  }

  // entity_index mapping
  {
    auto rows = store->Execute (
        "SELECT node_id FROM entity_index WHERE name='Alice'", {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<std::string> (rows[0].at ("node_id"))
             == "entity:Alice");
  }
}

// Helper to encode float vector as blob for vec_embeddings
namespace
{
constexpr int kTestEmbeddingDim = 256;

std::vector<char>
EncodeFloatBlob (const std::vector<float> &vec)
{
  std::vector<char> blob (vec.size () * sizeof (float));
  std::memcpy (blob.data (), vec.data (), blob.size ());
  return blob;
}

// Create a 256D embedding with first few dimensions set, rest zeros
std::vector<float>
Make256DEmb (std::initializer_list<float> first_dims)
{
  std::vector<float> emb (kTestEmbeddingDim, 0.0f);
  size_t i = 0;
  for (float v : first_dims)
    {
      if (i < kTestEmbeddingDim)
        emb[i++] = v;
    }
  return emb;
}
} // namespace

TEST_CASE ("Phase4: co-occurrence edges are created for similar embeddings",
           "[operations][graph][phase4]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create required tables
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_summaries ("
                  "  summary_id TEXT PRIMARY KEY,"
                  "  summary_text TEXT,"
                  "  cluster_size INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_sources ("
                  "  summary_id TEXT,"
                  "  source_embedding_id INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS memories ("
                  "  embedding_id INTEGER PRIMARY KEY,"
                  "  timestamp INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS embeddings_meta ("
                  "  embedding_id INTEGER PRIMARY KEY,"
                  "  created_at INTEGER"
                  ");");

  // Create vec_embeddings table (simplified for testing)
  store->Execute ("CREATE TABLE IF NOT EXISTS vec_embeddings ("
                  "  embedding_id INTEGER PRIMARY KEY,"
                  "  embedding BLOB"
                  ");");

  // Create two similar 256D embeddings (high cosine similarity > 0.85)
  auto emb1 = Make256DEmb ({ 1.0f, 0.0f, 0.0f, 0.0f }); // Normalized direction
  auto emb2 = Make256DEmb ({ 0.95f, 0.05f, 0.0f, 0.0f }); // Similar direction
  auto blob1 = EncodeFloatBlob (emb1);
  auto blob2 = EncodeFloatBlob (emb2);

  store->Execute ("INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
                  { 1LL, blob1 });
  store->Execute ("INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
                  { 2LL, blob2 });

  // Create consolidation data
  store->Execute ("INSERT INTO consolidation_summaries (summary_id, summary_text, "
                  "cluster_size) VALUES (?, ?, ?)",
                  { std::string ("s1"), std::string ("test"), 2LL });
  store->Execute ("INSERT INTO consolidation_sources (summary_id, "
                  "source_embedding_id) VALUES (?, ?)",
                  { std::string ("s1"), 1LL });
  store->Execute ("INSERT INTO consolidation_sources (summary_id, "
                  "source_embedding_id) VALUES (?, ?)",
                  { std::string ("s1"), 2LL });

  SignalProcessor::Config cfg;
  cfg.focus = 0.0; // Low focus = threshold 0.85
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EnsureGraphSchema> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // Verify co_occurs_with edge was created
  auto rows = store->Execute (
      "SELECT weight FROM graph_edges "
      "WHERE edge_type = 'co_occurs_with' "
      "AND ((source_id = 'emb:1' AND target_id = 'emb:2') "
      "  OR (source_id = 'emb:2' AND target_id = 'emb:1'))",
      {});

  // Edge should exist with similarity as weight
  REQUIRE (rows.size () == 1);
  double weight = std::any_cast<double> (rows[0].at ("weight"));
  REQUIRE (weight > 0.85); // Should be high similarity
}

TEST_CASE ("Phase4: causal edges are created for temporal drift",
           "[operations][graph][phase4]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create required tables
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_summaries ("
                  "  summary_id TEXT PRIMARY KEY,"
                  "  summary_text TEXT,"
                  "  cluster_size INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_sources ("
                  "  summary_id TEXT,"
                  "  source_embedding_id INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS memories ("
                  "  embedding_id INTEGER PRIMARY KEY,"
                  "  timestamp INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS embeddings_meta ("
                  "  embedding_id INTEGER PRIMARY KEY,"
                  "  created_at INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS vec_embeddings ("
                  "  embedding_id INTEGER PRIMARY KEY,"
                  "  embedding BLOB"
                  ");");

  // Create two 256D embeddings with significant drift (L2 distance > 0.35)
  auto emb1 = Make256DEmb ({ 1.0f, 0.0f, 0.0f, 0.0f });
  auto emb2 = Make256DEmb ({ 0.5f, 0.5f, 0.5f, 0.0f }); // Different direction
  auto blob1 = EncodeFloatBlob (emb1);
  auto blob2 = EncodeFloatBlob (emb2);

  store->Execute ("INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
                  { 1LL, blob1 });
  store->Execute ("INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
                  { 2LL, blob2 });

  // Add timestamps for temporal ordering
  store->Execute ("INSERT INTO memories (embedding_id, timestamp) VALUES (?, ?)",
                  { 1LL, 1000LL });
  store->Execute ("INSERT INTO memories (embedding_id, timestamp) VALUES (?, ?)",
                  { 2LL, 2000LL }); // Later timestamp

  // Create consolidation data
  store->Execute ("INSERT INTO consolidation_summaries (summary_id, summary_text, "
                  "cluster_size) VALUES (?, ?, ?)",
                  { std::string ("s1"), std::string ("test"), 2LL });
  store->Execute ("INSERT INTO consolidation_sources (summary_id, "
                  "source_embedding_id) VALUES (?, ?)",
                  { std::string ("s1"), 1LL });
  store->Execute ("INSERT INTO consolidation_sources (summary_id, "
                  "source_embedding_id) VALUES (?, ?)",
                  { std::string ("s1"), 2LL });

  SignalProcessor::Config cfg;
  cfg.stability = 0.0; // Low stability = threshold 0.15
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EnsureGraphSchema> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (3000ULL));
  processor.Flush ();

  // Verify causes edge was created (directional: earlier -> later)
  auto rows = store->Execute (
      "SELECT weight FROM graph_edges "
      "WHERE edge_type = 'causes' "
      "AND source_id = 'emb:1' AND target_id = 'emb:2'",
      {});

  // Edge should exist with drift magnitude as weight
  REQUIRE (rows.size () == 1);
  double weight = std::any_cast<double> (rows[0].at ("weight"));
  REQUIRE (weight > 0.15); // Should exceed threshold
}

TEST_CASE ("Phase4: knob functions return expected values",
           "[operations][graph][phase4][knobs]")
{
  // Test CausalDriftThreshold
  REQUIRE (core::CausalDriftThreshold (0.0) == Catch::Approx (0.15).margin (1e-6));
  REQUIRE (core::CausalDriftThreshold (1.0) == Catch::Approx (0.35).margin (1e-6));
  REQUIRE (core::CausalDriftThreshold (0.5) == Catch::Approx (0.25).margin (1e-6));

  // Test ReinforcementDecay
  REQUIRE (core::ReinforcementDecay (0.0) == Catch::Approx (0.9).margin (1e-6));
  REQUIRE (core::ReinforcementDecay (1.0) == Catch::Approx (0.99).margin (1e-6));

  // Test MinEpisodesForConcept
  REQUIRE (core::MinEpisodesForConcept (0.0) == 2);
  REQUIRE (core::MinEpisodesForConcept (1.0) == 5);

  // Test CoOccurrenceThreshold (same as MergeThreshold)
  REQUIRE (core::CoOccurrenceThreshold (0.0) == Catch::Approx (0.85).margin (1e-6));
  REQUIRE (core::CoOccurrenceThreshold (1.0) == Catch::Approx (0.95).margin (1e-6));

  // Test ContradictionThreshold
  REQUIRE (core::ContradictionThreshold () == Catch::Approx (-0.5).margin (1e-6));
}

