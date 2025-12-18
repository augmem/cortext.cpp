// tests/operations_graph_build.test.cpp
#include "test_helpers.hpp"
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
constexpr int kEmbeddingDim = 256;

static Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
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

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  store->Execute ("INSERT INTO consolidation_summaries(summary_id, summary_text, "
                  "cluster_size) VALUES (?,?,?)",
                  { std::string ("s1"), std::string ("summary one"), 10LL });
  store->Execute ("INSERT INTO consolidation_sources(summary_id, "
                  "source_embedding_id) VALUES (?,?)",
                  { std::string ("s1"), 42LL });
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

// Helper to encode float vector as blob for embeddings
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

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create two similar 256D embeddings (high cosine similarity > 0.85)
  auto emb1 = Make256DEmb ({ 1.0f, 0.0f, 0.0f, 0.0f }); // Normalized direction
  auto emb2 = Make256DEmb ({ 0.95f, 0.05f, 0.0f, 0.0f }); // Similar direction

  // v2: Insert into embeddings (minimal vec0 table)
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 1LL, emb1, 0LL });
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 2LL, emb2, 0LL });
  // v2: Insert into memories (comprehensive metadata)
  store->Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
                  "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, 1.0, 0)",
                  { 1LL, 1LL });
  store->Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
                  "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, 1.0, 0)",
                  { 2LL, 2LL });

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

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create two 256D embeddings with significant drift (L2 distance > 0.35)
  auto emb1 = Make256DEmb ({ 1.0f, 0.0f, 0.0f, 0.0f });
  auto emb2 = Make256DEmb ({ 0.5f, 0.5f, 0.5f, 0.0f }); // Different direction

  // v2: Insert into embeddings (minimal vec0 table)
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 1LL, emb1, 0LL });
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 2LL, emb2, 0LL });

  // v2: Add memories with temporal ordering (end_ts used for ordering per schema)
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "end_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, ?, ?, ?, ?, ?, 1.0, 0)",
      { 1LL, 1LL, 900LL, 1000LL, 1LL, std::string ("text"), 0.5, 0.5 });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "end_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, ?, ?, ?, ?, ?, 1.0, 0)",
      { 2LL, 2LL, 1900LL, 2000LL, 1LL, std::string ("text"), 0.5, 0.5 }); // Later timestamp

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

TEST_CASE ("Phase2: implies edges created from implication predicates",
           "[operations][graph][phase2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  store->Execute ("INSERT INTO consolidation_summaries(summary_id, summary_text, "
                  "cluster_size) VALUES (?,?,?)",
                  { std::string ("s1"), std::string ("test"), 1LL });

  // Insert relations with various implication predicates
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("A"),
                    std::string ("implies"), std::string ("B"), 0.9 });
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("C"),
                    std::string ("suggests"), std::string ("D"), 0.85 });
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("E"),
                    std::string ("indicates"), std::string ("F"), 0.8 });
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("G"),
                    std::string ("means"), std::string ("H"), 0.75 });
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("I"),
                    std::string ("entails"), std::string ("J"), 0.7 });

  SignalProcessor::Config cfg;
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EnsureGraphSchema> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // All implication predicates should be normalized to 'implies' edge type
  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM graph_edges WHERE edge_type = 'implies'", {});
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 5LL);

  // Verify specific edge with correct weight
  auto edge_rows = store->Execute (
      "SELECT weight FROM graph_edges "
      "WHERE source_id = 'entity:A' AND target_id = 'entity:B' "
      "AND edge_type = 'implies'",
      {});
  REQUIRE (edge_rows.size () == 1);
  REQUIRE (std::any_cast<double> (edge_rows[0].at ("weight"))
           == Catch::Approx (0.9).margin (1e-6));
}

TEST_CASE ("Phase2: implies edge directionality preserved",
           "[operations][graph][phase2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  store->Execute ("INSERT INTO consolidation_summaries(summary_id, summary_text, "
                  "cluster_size) VALUES (?,?,?)",
                  { std::string ("s1"), std::string ("test"), 1LL });

  // Insert: "Rain implies Wet" - directional relationship
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("Rain"),
                    std::string ("implies"), std::string ("Wet"), 0.95 });

  SignalProcessor::Config cfg;
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EnsureGraphSchema> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // Edge should exist: Rain -> Wet (not Wet -> Rain)
  auto forward = store->Execute (
      "SELECT COUNT(*) AS c FROM graph_edges "
      "WHERE source_id = 'entity:Rain' AND target_id = 'entity:Wet' "
      "AND edge_type = 'implies'",
      {});
  REQUIRE (std::any_cast<long long> (forward[0].at ("c")) == 1LL);

  // Reverse edge should NOT exist
  auto reverse = store->Execute (
      "SELECT COUNT(*) AS c FROM graph_edges "
      "WHERE source_id = 'entity:Wet' AND target_id = 'entity:Rain' "
      "AND edge_type = 'implies'",
      {});
  REQUIRE (std::any_cast<long long> (reverse[0].at ("c")) == 0LL);
}

TEST_CASE ("Phase2: non-implication predicates unchanged",
           "[operations][graph][phase2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  store->Execute ("INSERT INTO consolidation_summaries(summary_id, summary_text, "
                  "cluster_size) VALUES (?,?,?)",
                  { std::string ("s1"), std::string ("test"), 1LL });

  // Insert non-implication relations
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("Alice"),
                    std::string ("KNOWS"), std::string ("Bob"), 0.8 });
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("Alice"),
                    std::string ("works_at"), std::string ("Acme"), 0.9 });
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("Fire"),
                    std::string ("causes"), std::string ("Smoke"), 0.85 });

  SignalProcessor::Config cfg;
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EnsureGraphSchema> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // Non-implication predicates should retain their original edge_type
  auto knows = store->Execute (
      "SELECT COUNT(*) AS c FROM graph_edges "
      "WHERE source_id = 'entity:Alice' AND target_id = 'entity:Bob' "
      "AND edge_type = 'KNOWS'",
      {});
  REQUIRE (std::any_cast<long long> (knows[0].at ("c")) == 1LL);

  auto works = store->Execute (
      "SELECT COUNT(*) AS c FROM graph_edges "
      "WHERE source_id = 'entity:Alice' AND target_id = 'entity:Acme' "
      "AND edge_type = 'works_at'",
      {});
  REQUIRE (std::any_cast<long long> (works[0].at ("c")) == 1LL);

  auto causes = store->Execute (
      "SELECT COUNT(*) AS c FROM graph_edges "
      "WHERE source_id = 'entity:Fire' AND target_id = 'entity:Smoke' "
      "AND edge_type = 'causes'",
      {});
  REQUIRE (std::any_cast<long long> (causes[0].at ("c")) == 1LL);

  // No 'implies' edges should exist
  auto implies = store->Execute (
      "SELECT COUNT(*) AS c FROM graph_edges WHERE edge_type = 'implies'", {});
  REQUIRE (std::any_cast<long long> (implies[0].at ("c")) == 0LL);
}

