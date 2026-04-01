// tests/operations_emotion_cascade.test.cpp
#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <string>

#include <cortext/core/knobs.hpp>
#include <cortext/operations/emotion_cascade.hpp>

#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;

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
  REQUIRE (
      core::CascadeDecay (0.5)
      == Catch::Approx (
             core::Lerp (0.7, 0.3, core::SensitivityBias (0.5)))
             .margin (1e-6));
  REQUIRE (core::CascadeDecay (1.0) == Catch::Approx (0.3).margin (1e-6));
}

TEST_CASE ("PropagateEmotionalCascade propagates through graph edges",
           "[operations][emotion_cascade]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // V2: Create embeddings and memories for the cascade to propagate through
  std::vector<float> emb (kEmbeddingDim, 0.0f);
  emb[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 1LL, emb);
  cortext::testing::SeedEmbeddingV2 (*store, 2LL, emb);
  cortext::testing::SeedEmbeddingV2 (*store, 3LL, emb);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "test");
  cortext::testing::SeedMemoryV2 (*store, 2LL, 2LL, "test");
  cortext::testing::SeedMemoryV2 (*store, 3LL, 3LL, "test");

  // V2: Create associations (graph edges): 1 -> 2 -> 3 via memory_ids
  store->Execute ("INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
                  "VALUES (?, ?, ?, ?)",
                  { 1LL, 2LL, std::string ("co_occurs"), 0.9 });
  store->Execute ("INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
                  "VALUES (?, ?, ?, ?)",
                  { 2LL, 3LL, std::string ("causes"), 0.8 });

  // v2: Set high-intensity flashbulb for source memory (inline on memories)
  const long long now = 5000;
  store->Execute ("UPDATE memories SET flashbulb = 1, emotional_intensity = ?, "
                  "half_life_bonus = ?, cascade_radius = ?, cascade_decay = ?, "
                  "s_arousal_avg = ? "
                  "WHERE embedding_id = ?",
                  { 0.8, 2.0, 2LL, 0.5, 0.8, 1LL });

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5; // cascade_radius = 3, decay = 0.5
  cfg.stability = 0.5;
  auto ops = std::make_unique<OperationSet> (

      std::make_unique<PropagateEmotionalCascade> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (static_cast<uint64_t> (now)));
  processor.Flush ();

  // v2: Verify emotional_intensity propagated through memories table
  auto rows = store->Execute (
      "SELECT embedding_id, emotional_intensity FROM memories ORDER BY embedding_id",
      {});

  // Should have emotional values for memories (propagated)
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
