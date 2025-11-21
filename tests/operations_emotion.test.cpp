// tests/operations_emotion.test.cpp
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
  Execute (OperationContext &ctx) const override
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
  s.embedding = Eigen::VectorXf::Ones (4);
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

  // Row for used id exists
  {
    auto rows = store->Execute (
        "SELECT * FROM emotional_tags WHERE embedding_id = ?", { 101LL });
    REQUIRE (rows.size () == 1);
    const auto &row = rows[0];
    REQUIRE (std::any_cast<long long> (row.at ("embedding_id")) == 101LL);
    REQUIRE (std::any_cast<long long> (row.at ("flashbulb")) == 1LL);
    REQUIRE (std::any_cast<double> (row.at ("intensity"))
             == Catch::Approx (intensity));
    REQUIRE (std::any_cast<double> (row.at ("arousal"))
             == Catch::Approx (arousal));
    REQUIRE (std::any_cast<double> (row.at ("valence"))
             == Catch::Approx (valence));

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

    const double fb_eff
        = std::any_cast<double> (row.at ("flashbulb_threshold_eff"));
    const double fb_exp
        = core::FlashbulbThresholdEff (cfg.sensitivity, intensity, arousal);
    REQUIRE (fb_eff == Catch::Approx (fb_exp).margin (1e-6));

    const long long ts = std::any_cast<long long> (row.at ("ts"));
    REQUIRE (ts == 12345LL);
  }

  // No row for unused id
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS c FROM emotional_tags WHERE embedding_id = ?",
        { 102LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 0LL);
  }
}

TEST_CASE ("Alg23 below thresholds performs no-op", "[operations][emotion]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

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
    // Ensure table exists to allow counting rows even if operation was a
    // no-op.
    store->Execute (
        "CREATE TABLE IF NOT EXISTS emotional_tags ("
        "embedding_id INTEGER PRIMARY KEY,"
        "flashbulb INTEGER NOT NULL DEFAULT 0,"
        "intensity REAL, arousal REAL, valence REAL,"
        "half_life_bonus REAL, detail_suppression REAL,"
        "gist_components INTEGER, cascade_radius INTEGER,"
        "cascade_decay REAL, flashbulb_threshold_eff REAL, ts INTEGER);");
    auto rows = store->Execute ("SELECT COUNT(*) AS c FROM emotional_tags");
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 0LL);
  }
}
