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
  explicit SetupEmotionInputsOp (std::optional<long long> stored_id,
                                 std::vector<double> emotion_history = {},
                                 double flashbulb_rate_ewma = 0.0)
      : stored_id_ (stored_id), emotion_history_ (std::move (emotion_history)),
        flashbulb_rate_ewma_ (flashbulb_rate_ewma)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetStoredEmbeddingId (stored_id_);
    auto &pctx = ctx.GetProcessorContext ();
    pctx.recent_emotion_intensities.assign (emotion_history_.begin (),
                                            emotion_history_.end ());
    pctx.flashbulb_rate_ewma = flashbulb_rate_ewma_;
  }

private:
  std::optional<long long> stored_id_;
  std::vector<double> emotion_history_;
  double flashbulb_rate_ewma_ = 0.0;
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

static long long
ReadFlashbulb (Store &store, long long embedding_id)
{
  auto rows = store.Execute ("SELECT flashbulb FROM memories WHERE embedding_id = ?",
                             { embedding_id });
  REQUIRE (rows.size () == 1);
  return std::any_cast<long long> (rows[0].at ("flashbulb"));
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
      = std::make_unique<DynamicOperationSet> (std::move (setup), std::move (apply));

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
      = std::make_unique<DynamicOperationSet> (std::move (setup), std::move (apply));

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

TEST_CASE ("Flashbulb rate no longer weakens percentile gate below target",
           "[operations][emotion]")
{
  auto make_store = [] () {
    auto unique_store = SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<Store> (std::move (unique_store));
    cortext::testing::InitializeCoreSchema (*store);
    std::vector<float> emb (kEmbeddingDim, 0.0f);
    emb[0] = 1.0f;
    cortext::testing::SeedEmbeddingV2 (*store, 301LL, emb);
    cortext::testing::SeedMemoryV2 (*store, 301LL, 301LL, "test");
    store->Execute (
        "UPDATE memories SET s_emotion_max = ?, s_arousal_avg = ? "
        "WHERE embedding_id = ?",
        { 0.90, 0.90, 301LL });
    return store;
  };

  auto run_case = [] (const std::shared_ptr<Store> &store, bool disable_percentile) {
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.4;
    cfg.sensitivity = 0.8;
    cfg.stability = 0.5;

    std::unique_ptr<cortext::testing::ScopedEnvVar> percentile_guard;
    if (disable_percentile)
      {
        percentile_guard = std::make_unique<cortext::testing::ScopedEnvVar> (
            "CORTEXT_FLASHBULB_DISABLE_PERCENTILE", "1");
      }

    auto setup = std::make_unique<SetupEmotionInputsOp> (
        301LL, std::vector<double> (16, 0.98), 0.0);
    auto apply = std::make_unique<ApplyEmotionalConsolidation> ();
    auto ops = std::make_unique<DynamicOperationSet> (std::move (setup),
                                               std::move (apply));
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (123));
    processor.Flush ();
    return ReadFlashbulb (*store, 301LL);
  };

  REQUIRE (run_case (make_store (), false) == 0LL);
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  REQUIRE (run_case (make_store (), true) == 1LL);
#else
  REQUIRE (run_case (make_store (), true) == 0LL);
#endif
}

TEST_CASE ("Flashbulb rate is neutral below target when percentile is absent",
           "[operations][emotion]")
{
  auto make_store = [] () {
    auto unique_store = SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<Store> (std::move (unique_store));
    cortext::testing::InitializeCoreSchema (*store);
    std::vector<float> emb (kEmbeddingDim, 0.0f);
    emb[0] = 1.0f;
    cortext::testing::SeedEmbeddingV2 (*store, 401LL, emb);
    cortext::testing::SeedMemoryV2 (*store, 401LL, 401LL, "test");
    store->Execute (
        "UPDATE memories SET s_emotion_max = ?, s_arousal_avg = ? "
        "WHERE embedding_id = ?",
        { 0.76, 0.88, 401LL });
    return store;
  };

  auto run_case = [] (const std::shared_ptr<Store> &store, bool disable_rate) {
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.4;
    cfg.sensitivity = 0.8;
    cfg.stability = 0.5;

    std::unique_ptr<cortext::testing::ScopedEnvVar> rate_guard;
    if (disable_rate)
      {
        rate_guard = std::make_unique<cortext::testing::ScopedEnvVar> (
            "CORTEXT_FLASHBULB_DISABLE_RATE", "1");
      }

    auto setup = std::make_unique<SetupEmotionInputsOp> (401LL,
                                                         std::vector<double> (),
                                                         0.0);
    auto apply = std::make_unique<ApplyEmotionalConsolidation> ();
    auto ops = std::make_unique<DynamicOperationSet> (std::move (setup),
                                               std::move (apply));
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (456));
    processor.Flush ();
    return ReadFlashbulb (*store, 401LL);
  };

  REQUIRE (run_case (make_store (), false) == 1LL);
  REQUIRE (run_case (make_store (), true) == 1LL);
}
