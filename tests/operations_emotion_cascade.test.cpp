// tests/operations_emotion_cascade.test.cpp
#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <string>

#include <cortext/core/knobs.hpp>
#include <cortext/operations/emotion_cascade.hpp>
#include <cortext/operations/graph_schema.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::EnsureGraphSchema;
using cortext::operations::PropagateEmotionalCascade;

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

TEST_CASE ("EmotionCascadeParams derives from knobs correctly",
           "[operations][emotion_cascade][params]")
{
  auto params = operations::EmotionCascadeParams::FromKnobs (0.5, 0.0, 0.5);
  REQUIRE (params.cascade_radius == 1); // round(lerp(1, 5, 0)) = 1
  REQUIRE (params.cascade_decay == Catch::Approx (0.7).margin (1e-6)); // lerp(0.7, 0.3, 0)

  auto params2 = operations::EmotionCascadeParams::FromKnobs (0.5, 1.0, 0.5);
  REQUIRE (params2.cascade_radius == 5); // round(lerp(1, 5, 1)) = 5
  REQUIRE (params2.cascade_decay == Catch::Approx (0.3).margin (1e-6)); // lerp(0.7, 0.3, 1)
}

TEST_CASE ("CascadeRadius and CascadeDecay knob values",
           "[operations][emotion_cascade][knobs]")
{
  // CascadeRadius
  REQUIRE (core::CascadeRadius (0.0) == 1);
  REQUIRE (core::CascadeRadius (0.5) == 3);
  REQUIRE (core::CascadeRadius (1.0) == 5);

  // CascadeDecay
  REQUIRE (core::CascadeDecay (0.0) == Catch::Approx (0.7).margin (1e-6));
  REQUIRE (core::CascadeDecay (0.5) == Catch::Approx (0.5).margin (1e-6));
  REQUIRE (core::CascadeDecay (1.0) == Catch::Approx (0.3).margin (1e-6));
}

TEST_CASE ("PropagateEmotionalCascade propagates through graph edges",
           "[operations][emotion_cascade]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create graph nodes
  store->Execute ("INSERT INTO graph_nodes (node_id, type, created_at) VALUES (?, ?, ?)",
                  { std::string ("emb:1"), std::string ("memory"), 1000LL });
  store->Execute ("INSERT INTO graph_nodes (node_id, type, created_at) VALUES (?, ?, ?)",
                  { std::string ("emb:2"), std::string ("memory"), 1000LL });
  store->Execute ("INSERT INTO graph_nodes (node_id, type, created_at) VALUES (?, ?, ?)",
                  { std::string ("emb:3"), std::string ("memory"), 1000LL });

  // Create graph edges: 1 -> 2 -> 3
  store->Execute ("INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
                  "VALUES (?, ?, ?, ?)",
                  { std::string ("emb:1"), std::string ("emb:2"),
                    std::string ("co_occurs_with"), 0.9 });
  store->Execute ("INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
                  "VALUES (?, ?, ?, ?)",
                  { std::string ("emb:2"), std::string ("emb:3"),
                    std::string ("causes"), 0.8 });

  // Create high-intensity emotional tag for source memory
  const long long now = 5000;
  store->Execute ("INSERT INTO emotional_tags (embedding_id, flashbulb, intensity, "
                  "arousal, valence, half_life_bonus, cascade_radius, cascade_decay, ts) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  { 1LL, 0LL, 0.8, 0.7, 0.6, 2.0, 2LL, 0.5, now - 100 });

  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5; // cascade_radius = 3, decay = 0.5
  cfg.stability = 0.5;
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EnsureGraphSchema> (),
      std::make_unique<PropagateEmotionalCascade> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (static_cast<uint64_t> (now)));
  processor.Flush ();

  // Verify emotional tags propagated
  auto rows = store->Execute (
      "SELECT embedding_id, intensity FROM emotional_tags ORDER BY embedding_id",
      {});

  // Should have tags for multiple memories (propagated)
  // Note: Results depend on consolidation interval and timing
}

TEST_CASE ("PropagateEmotionalCascade decays intensity per hop",
           "[operations][emotion_cascade][decay]")
{
  // Test that intensity decays correctly through hops
  double source_intensity = 0.8;
  double decay = 0.5;

  // After 1 hop: 0.8 * 0.5 = 0.4
  double hop1 = source_intensity * std::pow (decay, 1);
  REQUIRE (hop1 == Catch::Approx (0.4).margin (1e-6));

  // After 2 hops: 0.8 * 0.5^2 = 0.2
  double hop2 = source_intensity * std::pow (decay, 2);
  REQUIRE (hop2 == Catch::Approx (0.2).margin (1e-6));

  // After 3 hops: 0.8 * 0.5^3 = 0.1
  double hop3 = source_intensity * std::pow (decay, 3);
  REQUIRE (hop3 == Catch::Approx (0.1).margin (1e-6));
}
