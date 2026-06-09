// tests/operations_graph_build.test.cpp
#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include <cortext/consolidation_mode.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/operations/graph_build.hpp>

#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::EvaluateConsolidation;
using cortext::operations::BuildGraphFromConsolidation;


namespace
{
constexpr int kEmbeddingDim = 256;

static Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = ts;
  s.source_id = "test/consolidation";
  s.consolidation_mode = ConsolidationMode::Both;
  return s;
}

// Create a 256D embedding with first few dimensions set, rest zeros
std::vector<float>
Make256DEmb (std::initializer_list<float> first_dims)
{
  std::vector<float> emb (kEmbeddingDim, 0.0f);
  size_t i = 0;
  for (float v : first_dims)
    {
      if (i < static_cast<size_t> (kEmbeddingDim))
        emb[i++] = v;
    }
  return emb;
}
} // namespace

TEST_CASE ("V2: GraphBuild creates co-occurrence edges for similar memories in same cluster",
           "[operations][graph][v2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create two similar 256D embeddings (high cosine similarity > 0.85)
  auto emb1 = Make256DEmb ({ 1.0f, 0.0f, 0.0f, 0.0f }); // Normalized direction
  auto emb2 = Make256DEmb ({ 0.95f, 0.05f, 0.0f, 0.0f }); // Similar direction

  // Insert embeddings
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 1LL, emb1, 0LL });
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 2LL, emb2, 0LL });

  // Insert memories with same cluster_id (as set by ConsolidationSummarize)
  store->Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
                  "start_ts, cluster_id, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, ?, 0)",
                  { 1LL, 1LL, 100 }); // cluster_id = 100
  store->Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
                  "start_ts, cluster_id, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, ?, 0)",
                  { 2LL, 2LL, 100 }); // Same cluster

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0; // Low focus = threshold 0.85
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EvaluateConsolidation> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // Verify co_occurs edge was created in associations
  auto rows = store->Execute (
      "SELECT weight FROM associations "
      "WHERE edge_type = 'co_occurs' "
      "AND ((source_memory_id = 1 AND target_memory_id = 2) "
      "  OR (source_memory_id = 2 AND target_memory_id = 1))",
      {});

  // Edge should exist with normalized weight in [0,1]
  REQUIRE (rows.size () == 1);
  double weight = std::any_cast<double> (rows[0].at ("weight"));
  REQUIRE (weight > 0.9); // Map01(sim) should still be high
}

TEST_CASE ("V2: GraphBuild creates causal edges for temporal drift within cluster",
           "[operations][graph][v2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create two 256D embeddings with significant drift (L2 distance > 0.35)
  auto emb1 = Make256DEmb ({ 1.0f, 0.0f, 0.0f, 0.0f });
  auto emb2 = Make256DEmb ({ 0.5f, 0.5f, 0.5f, 0.0f }); // Different direction

  // Insert embeddings
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 1LL, emb1, 0LL });
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 2LL, emb2, 0LL });

  // Insert memories with temporal ordering and same cluster_id
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "end_ts, cluster_id, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, ?, ?, 0)",
      { 1LL, 1LL, 900LL, 1000LL, 100 }); // Earlier, cluster_id = 100
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "end_ts, cluster_id, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, ?, ?, 0)",
      { 2LL, 2LL, 1900LL, 2000LL, 100 }); // Later, same cluster

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.0; // Low stability = threshold 0.15
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EvaluateConsolidation> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (3000ULL));
  processor.Flush ();

  // Verify causes edge was created (directional: earlier -> later)
  auto rows = store->Execute (
      "SELECT weight FROM associations "
      "WHERE edge_type = 'causes' "
      "AND source_memory_id = 1 AND target_memory_id = 2",
      {});

  // Edge should exist with normalized drift weight
  REQUIRE (rows.size () == 1);
  double weight = std::any_cast<double> (rows[0].at ("weight"));
  REQUIRE (weight > 0.2); // drift_mag/2 should exceed threshold
}

TEST_CASE ("V2: GraphBuild creates contradiction edges for opposing semantics",
           "[operations][graph][v2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create two embeddings with strong negative similarity (< -0.5)
  auto emb1 = Make256DEmb ({ 1.0f, 0.0f, 0.0f, 0.0f });
  auto emb2 = Make256DEmb ({ -1.0f, 0.0f, 0.0f, 0.0f }); // Opposite direction

  // Insert embeddings
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 1LL, emb1, 0LL });
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 2LL, emb2, 0LL });

  // Insert memories with same cluster_id
  store->Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
                  "start_ts, cluster_id, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, ?, 0)",
                  { 1LL, 1LL, 100 });
  store->Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
                  "start_ts, cluster_id, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, ?, 0)",
                  { 2LL, 2LL, 100 });

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EvaluateConsolidation> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // Verify contradicts edge was created
  auto rows = store->Execute (
      "SELECT weight FROM associations "
      "WHERE edge_type = 'contradicts' "
      "AND ((source_memory_id = 1 AND target_memory_id = 2) "
      "  OR (source_memory_id = 2 AND target_memory_id = 1))",
      {});

  // Edge should exist with normalized weight in [0,1]
  REQUIRE (rows.size () == 1);
  double weight = std::any_cast<double> (rows[0].at ("weight"));
  REQUIRE (weight < 0.3); // map01(negative sim) should be low
}

TEST_CASE ("V2: GraphBuild does not create edges across different clusters",
           "[operations][graph][v2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create two similar embeddings
  auto emb1 = Make256DEmb ({ 1.0f, 0.0f, 0.0f, 0.0f });
  auto emb2 = Make256DEmb ({ 0.95f, 0.05f, 0.0f, 0.0f });

  // Insert embeddings
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 1LL, emb1, 0LL });
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 2LL, emb2, 0LL });

  // Insert memories with DIFFERENT cluster_ids
  store->Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
                  "start_ts, cluster_id, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, ?, 0)",
                  { 1LL, 1LL, 100 }); // cluster_id = 100
  store->Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
                  "start_ts, cluster_id, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, ?, 0)",
                  { 2LL, 2LL, 200 }); // cluster_id = 200 (different)

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EvaluateConsolidation> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // No edges should be created between memories in different clusters
  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations "
      "WHERE (source_memory_id = 1 AND target_memory_id = 2) "
      "   OR (source_memory_id = 2 AND target_memory_id = 1)",
      {});

  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 0LL);
}

TEST_CASE ("Knob functions return expected values",
           "[operations][graph][knobs]")
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

TEST_CASE ("V2: GraphBuild decays reinforcement edges",
           "[operations][graph][v2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Pre-insert a reinforcement edge
  store->Execute (
      "INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'reinforces', ?)",
      { 1LL, 2LL, 1.0 });

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.0; // decay = 0.9
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EvaluateConsolidation> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // Check that weight was decayed
  auto rows = store->Execute (
      "SELECT weight FROM associations "
      "WHERE source_memory_id = 1 AND target_memory_id = 2 "
      "AND edge_type = 'reinforces'",
      {});

  REQUIRE (rows.size () == 1);
  double weight = std::any_cast<double> (rows[0].at ("weight"));
  REQUIRE (weight == Catch::Approx (0.9).margin (1e-6)); // 1.0 * 0.9
}

TEST_CASE ("V2: GraphBuild removes weak reinforcement edges",
           "[operations][graph][v2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Pre-insert a weak reinforcement edge (below 0.1 threshold after decay)
  store->Execute (
      "INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'reinforces', ?)",
      { 1LL, 2LL, 0.05 }); // Will be < 0.1 after decay

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.0; // decay = 0.9
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EvaluateConsolidation> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // Edge should be deleted
  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations "
      "WHERE source_memory_id = 1 AND target_memory_id = 2 "
      "AND edge_type = 'reinforces'",
      {});

  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 0LL);
}
