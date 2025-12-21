// tests/operations_emotion.test.cpp
#include "test_helpers.hpp"
#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/emotion.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace cortext;

using cortext::operations::ApplyEmotionalConsolidation;

namespace
{

constexpr int kEmbeddingDim = 256;

class SetupEmotionInputsOp : public IOperation
{
public:
  explicit SetupEmotionInputsOp (std::optional<long long> stored_id)
      : stored_id_ (stored_id)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetStoredEmbeddingId (stored_id_);
  }

private:
  std::optional<long long> stored_id_;
};

static Signal
MakeSignal (uint64_t ts = 1)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

} // namespace

TEST_CASE ("Alg23 triggers and persists emotional tags for stored memory",
           "[operations][emotion]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // v2: Seed embedding and memory rows for the emotion operation to update
  std::vector<float> emb (kEmbeddingDim, 0.0f);
  emb[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 101LL, emb);
  cortext::testing::SeedEmbeddingV2 (*store, 102LL, emb);
  cortext::testing::SeedMemoryV2 (*store, 101LL, 101LL, "test");
  cortext::testing::SeedMemoryV2 (*store, 102LL, 102LL, "test");

  // Seed emotional metadata on memories (used by consolidation)
  store->Execute (
      "UPDATE memories SET s_emotion_max = ?, s_arousal_avg = ? "
      "WHERE embedding_id = ?",
      { 0.9, 0.8, 101LL });
  store->Execute (
      "UPDATE memories SET s_emotion_max = ?, s_arousal_avg = ? "
      "WHERE embedding_id = ?",
      { 0.2, 0.1, 102LL });

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.4;       // F
  cfg.sensitivity = 0.8; // S
  cfg.stability = 0.5;

  auto setup = std::make_unique<SetupEmotionInputsOp> (101LL);
  auto apply = std::make_unique<ApplyEmotionalConsolidation> ();
  auto ops
      = std::make_unique<OperationSet> (std::move (setup), std::move (apply));

  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (/*ts=*/12345));
  processor.Flush ();

  // v2: Row for used id exists in memories table (emotional_tags merged into memories)
  {
    auto rows = store->Execute (
        "SELECT embedding_id, flashbulb, emotional_intensity, half_life_bonus, "
        "detail_suppression, gist_components, cascade_radius, cascade_decay "
        "FROM memories WHERE embedding_id = ?", { 101LL });
    REQUIRE (rows.size () == 1);
    const auto &row = rows[0];
    REQUIRE (std::any_cast<long long> (row.at ("embedding_id")) == 101LL);
    REQUIRE (std::any_cast<long long> (row.at ("flashbulb")) == 1LL);
    REQUIRE (std::any_cast<double> (row.at ("emotional_intensity"))
             == Catch::Approx (0.9));

    const double half_life_bonus
        = std::any_cast<double> (row.at ("half_life_bonus"));
    REQUIRE (half_life_bonus > 1.0);

    const double detail
        = std::any_cast<double> (row.at ("detail_suppression"));
    REQUIRE (
        detail
        == Catch::Approx (core::DetailSuppression (cfg.sensitivity, cfg.focus))
               .margin (1e-6));

    const long long gist
        = std::any_cast<long long> (row.at ("gist_components"));
    REQUIRE (gist >= 2);
    REQUIRE (gist <= 5);

    const long long cr = std::any_cast<long long> (row.at ("cascade_radius"));
    REQUIRE (cr == core::CascadeRadius (cfg.sensitivity));

    const double cd = std::any_cast<double> (row.at ("cascade_decay"));
    REQUIRE (
        cd
        == Catch::Approx (core::CascadeDecay (cfg.sensitivity)).margin (1e-6));
  }

  // v2: Unused id should have default flashbulb=0 (row should exist from SeedMemory)
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS c FROM memories WHERE embedding_id = ? AND flashbulb = 1",
        { 102LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 0LL);
  }
}

TEST_CASE ("Alg23 below thresholds performs no-op", "[operations][emotion]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // v2: Seed embedding and memory rows
  std::vector<float> emb (kEmbeddingDim, 0.0f);
  emb[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 201LL, emb);
  cortext::testing::SeedMemoryV2 (*store, 201LL, 201LL, "test");

  store->Execute (
      "UPDATE memories SET s_emotion_max = ?, s_arousal_avg = ? "
      "WHERE embedding_id = ?",
      { 0.6, 0.25, 201LL });

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5; // θ_intensity=0.7, θ_arousal=0.3
  cfg.stability = 0.5;

  auto setup = std::make_unique<SetupEmotionInputsOp> (201LL);
  auto apply = std::make_unique<ApplyEmotionalConsolidation> ();
  auto ops
      = std::make_unique<OperationSet> (std::move (setup), std::move (apply));

  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (/*ts=*/99));
  processor.Flush ();

  {
    // v2: Check that flashbulb was not triggered (remains 0) in memories table
    auto rows = store->Execute ("SELECT COUNT(*) AS c FROM memories WHERE flashbulb = 1");
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 0LL);
  }
}
