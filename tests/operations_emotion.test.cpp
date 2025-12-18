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
  SetupEmotionInputsOp (double intensity, double arousal, double valence,
                        std::vector<OperationContext::MemoryUsageEvent> events)
      : intensity_ (intensity), arousal_ (arousal), valence_ (valence),
        events_ (std::move (events))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetEmotionIntensity (intensity_);
    ctx.SetArousal (arousal_);
    ctx.SetValence (valence_);
    ctx.SetMemoryUsageEvents (events_);
  }

private:
  double intensity_;
  double arousal_;
  double valence_;
  std::vector<OperationContext::MemoryUsageEvent> events_;
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

TEST_CASE ("Alg23 triggers and persists emotional tags for used memories",
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

  SignalProcessor::Config cfg;
  cfg.focus = 0.4;       // F
  cfg.sensitivity = 0.8; // S
  cfg.stability = 0.5;

  // High emotion above thresholds for S=0.8: θ_intensity≈0.64, θ_arousal≈0.24
  const double intensity = 0.9;
  const double arousal = 0.8;
  const double valence = 0.7;

  std::vector<OperationContext::MemoryUsageEvent> events{
    { 101LL, true, std::nullopt }, { 102LL, false, std::nullopt }
  };

  auto setup = std::make_unique<SetupEmotionInputsOp> (intensity, arousal,
                                                       valence, events);
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
             == Catch::Approx (intensity));

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

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5; // θ_intensity=0.7, θ_arousal=0.3
  cfg.stability = 0.5;

  const double intensity = 0.6; // below
  const double arousal = 0.25;  // below
  const double valence = 0.4;

  std::vector<OperationContext::MemoryUsageEvent> events{ { 201LL, true,
                                                            std::nullopt } };

  auto setup = std::make_unique<SetupEmotionInputsOp> (intensity, arousal,
                                                       valence, events);
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
