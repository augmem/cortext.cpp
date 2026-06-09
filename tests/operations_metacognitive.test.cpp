#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/metacognitive.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

#include <optional>

#include "../src/operations/eviction_ablation.hpp"
#include "../src/operations/temporal_retrieval.hpp"

using namespace cortext;
using cortext::operations::MetacognitiveMonitoring;
namespace temporal = cortext::operations::temporal;
namespace eviction = cortext::operations::eviction;

TEST_CASE ("Alg25 detects TOT when FOK high and retrieval low",
           "[operations][metacognitive][tot]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 1;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.8; // higher focus
  cfg.sensitivity = 0.2;
  cfg.stability = 0.7;


  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.90); // high FOK
  ctx.SetCompositeScore (0.20);   // low retrieval
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  const double tot_fok_cut = cortext::core::TOTFokCutoff (cfg.focus);
  const double tot_ret_cut = cortext::core::TOTRetrievalCutoff (cfg.focus);
  REQUIRE (0.90 > tot_fok_cut);
  REQUIRE (0.20 < tot_ret_cut);
  REQUIRE (ctx.GetMetacogTOTDetected () == true);
}

TEST_CASE ("Alg25 detects unknown when retrieval below threshold",
           "[operations][metacognitive][unknown]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 2;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg);
  const double unk = cortext::core::UnknownThreshold (cfg.focus);
  ctx.SetFeelingOfKnowing (0.10);
  ctx.SetCompositeScore (unk - 0.05);
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (ctx.GetMetacogUnknownDetected () == true);
}

TEST_CASE ("Alg25 exposes parameter derivations per algorithms.md",
           "[operations][metacognitive][params]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 3;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.6;
  cfg.sensitivity = 0.3;
  cfg.stability = 0.8;


  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.4);
  ctx.SetCompositeScore (0.7);
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  const double expected_fok_th = cortext::core::FOKThreshold (cfg.focus);
  REQUIRE (ctx.GetMetacogFOKThreshold ()
           == Catch::Approx (expected_fok_th).epsilon (1e-9));

  const double expected_decay
      = cortext::core::ConfidenceDecayRate (cfg.stability);
  REQUIRE (ctx.GetMetacogConfidenceDecayRate ()
           == Catch::Approx (expected_decay).epsilon (1e-9));

  const int expected_latency
      = cortext::core::StrategySwitchLatencyMs (cfg.sensitivity);
  REQUIRE (ctx.GetMetacogStrategySwitchLatencyMs () == expected_latency);

  const double expected_cert
      = cortext::core::CertaintyRequirement (1.0);
  REQUIRE (ctx.GetMetacogCertaintyRequirement ()
           == Catch::Approx (expected_cert).epsilon (1e-9));

  const double expected_meta_sens
      = cortext::core::MetacognitiveSensitivity (cfg.focus, cfg.sensitivity);
  REQUIRE (ctx.GetMetacogSensitivity ()
           == Catch::Approx (expected_meta_sens).epsilon (1e-9));
}

TEST_CASE ("Metacognitive TOT arms next-turn recovery mode",
           "[operations][metacognitive][state]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 1000;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.4;
  cfg.stability = 0.6;

  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg);
  ctx.SetFeelingOfKnowing (1.0);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetMetacogTOTDetected ());
  REQUIRE (pctx.metacognitive_mode
           == ProcessorContext::MetacognitiveMode::TotRecovery);
  REQUIRE (pctx.metacognitive_mode_expires_at
           == s.timestamp
                  + static_cast<std::uint64_t> (
                      cortext::core::StrategySwitchLatencyMs (
                          cfg.sensitivity)));
  REQUIRE (pctx.metacognitive_tot_trigger_count == 1);
}

TEST_CASE ("Metacognitive confidence decays across delayed signals",
           "[operations][metacognitive][confidence]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 11000;
  ProcessorContext pctx;
  pctx.metacognitive_confidence = 1.0;
  pctx.last_signal_timestamp = 1000;

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0;

  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.2);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetMetacogConfidenceDecayRate ()
           == Catch::Approx (cortext::core::ConfidenceDecayRate (cfg.stability)));
  REQUIRE (pctx.metacognitive_confidence < 1.0);
  REQUIRE (pctx.metacognitive_confidence > 0.2);
}

TEST_CASE ("Pressure-weighted idle decay preserves more confidence at low storage pressure",
           "[operations][metacognitive][confidence][ablation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 121000;

  ProcessorContext time_only_ctx;
  time_only_ctx.metacognitive_confidence = 1.0;
  time_only_ctx.last_signal_timestamp = 1000;

  ProcessorContext pressure_ctx = time_only_ctx;

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  MetacognitiveMonitoring op;

  auto run = [&] (ProcessorContext &pctx,
                  std::optional<temporal::ResurfacingDecayMode> decay_mode) {
    temporal::RetrievalAblationOverride retrieval_override;
    retrieval_override.resurfacing_decay_mode = decay_mode;
    temporal::ScopedRetrievalAblationOverride retrieval_guard (
        retrieval_override);

    eviction::EvictionAblationOverride eviction_override;
    eviction_override.storage_gate_enabled = true;
    eviction_override.min_storage_bytes = 1000;
    eviction_override.used_storage_bytes = 100;
    eviction::ScopedEvictionAblationOverride eviction_guard (
        eviction_override);

    OperationContext ctx (s, pctx, cfg);
    ctx.SetFeelingOfKnowing (0.2);
    ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
    auto tx = store->Begin ();
    op.Execute (ctx, *tx);
    tx->Commit ();
    return pctx.metacognitive_confidence;
  };

  ProcessorContext default_ctx = time_only_ctx;
  const double time_only_confidence
      = run (time_only_ctx, temporal::ResurfacingDecayMode::TimeOnly);
  const double default_confidence = run (default_ctx, std::nullopt);
  const double pressure_weighted_confidence
      = run (pressure_ctx, temporal::ResurfacingDecayMode::PressureRamp);

  REQUIRE (default_confidence
           == Catch::Approx (pressure_weighted_confidence).margin (1e-6));
  REQUIRE (pressure_weighted_confidence > time_only_confidence);
  REQUIRE (pressure_weighted_confidence > 0.2);
}

TEST_CASE ("Pressure-weighted certainty requirement follows stability",
           "[operations][metacognitive][certainty][ablation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 1000;

  auto run = [&] (double stability) {
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.5;
    cfg.sensitivity = 0.5;
    cfg.stability = stability;

    temporal::RetrievalAblationOverride retrieval_override;
    retrieval_override.resurfacing_decay_mode
        = temporal::ResurfacingDecayMode::PressureRamp;
    temporal::ScopedRetrievalAblationOverride retrieval_guard (
        retrieval_override);

    eviction::EvictionAblationOverride eviction_override;
    eviction_override.storage_gate_enabled = true;
    eviction_override.min_storage_bytes = 1000;
    eviction_override.used_storage_bytes = 100;
    eviction::ScopedEvictionAblationOverride eviction_guard (
        eviction_override);

    MetacognitiveMonitoring op;
    OperationContext ctx (s, pctx, cfg);
    ctx.SetFeelingOfKnowing (0.4);
    ctx.SetCompositeScore (0.7);
    auto tx = store->Begin ();
    op.Execute (ctx, *tx);
    tx->Commit ();
    return ctx.GetMetacogCertaintyRequirement ();
  };

  const double low_t = run (0.2);
  const double high_t = run (0.9);

  REQUIRE (high_t < low_t);
}

TEST_CASE ("Pressure-weighted idle decay converges to time-only under high storage pressure",
           "[operations][metacognitive][confidence][ablation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 121000;

  ProcessorContext time_only_ctx;
  time_only_ctx.metacognitive_confidence = 1.0;
  time_only_ctx.last_signal_timestamp = 1000;

  ProcessorContext pressure_ctx = time_only_ctx;

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  MetacognitiveMonitoring op;

  auto run = [&] (ProcessorContext &pctx,
                  std::optional<temporal::ResurfacingDecayMode> decay_mode) {
    temporal::RetrievalAblationOverride retrieval_override;
    retrieval_override.resurfacing_decay_mode = decay_mode;
    temporal::ScopedRetrievalAblationOverride retrieval_guard (
        retrieval_override);

    eviction::EvictionAblationOverride eviction_override;
    eviction_override.storage_gate_enabled = true;
    eviction_override.min_storage_bytes = 1000;
    eviction_override.used_storage_bytes = 1000;
    eviction::ScopedEvictionAblationOverride eviction_guard (
        eviction_override);

    OperationContext ctx (s, pctx, cfg);
    ctx.SetFeelingOfKnowing (0.2);
    ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
    auto tx = store->Begin ();
    op.Execute (ctx, *tx);
    tx->Commit ();
    return pctx.metacognitive_confidence;
  };

  ProcessorContext default_ctx = time_only_ctx;
  const double time_only_confidence
      = run (time_only_ctx, temporal::ResurfacingDecayMode::TimeOnly);
  const double default_confidence = run (default_ctx, std::nullopt);
  const double pressure_weighted_confidence
      = run (pressure_ctx, temporal::ResurfacingDecayMode::PressureRamp);

  REQUIRE (default_confidence
           == Catch::Approx (pressure_weighted_confidence).margin (1e-6));
  REQUIRE (pressure_weighted_confidence
           == Catch::Approx (time_only_confidence).margin (1e-6));
}

TEST_CASE ("Disabling metacognitive confidence decay preserves prior confidence",
           "[operations][metacognitive][confidence]")
{
  cortext::testing::ScopedEnvVar disable (
      "CORTEXT_DISABLE_METACOG_CONFIDENCE_DECAY", "1");

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 11000;
  ProcessorContext pctx;
  pctx.metacognitive_confidence = 1.0;
  pctx.last_signal_timestamp = 1000;

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0;

  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.2);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  const double expected_alpha = cortext::core::MetacognitiveConfidenceAlpha (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double expected_confidence
      = std::max (0.2, cortext::core::Ewma (1.0, 0.2, expected_alpha));
  REQUIRE (pctx.metacognitive_confidence
           == Catch::Approx (expected_confidence).margin (1e-6));
}

TEST_CASE ("Metacognitive unknown mode clears after certainty is satisfied",
           "[operations][metacognitive][state]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 2000;
  ProcessorContext pctx;
  pctx.metacognitive_mode
      = ProcessorContext::MetacognitiveMode::UnknownCaution;
  pctx.metacognitive_mode_expires_at = 3000;
  pctx.metacognitive_certainty_satisfied = true;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.7;

  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.2);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, 1.0 } });
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE_FALSE (ctx.GetMetacogUnknownDetected ());
  REQUIRE (pctx.metacognitive_mode
           == ProcessorContext::MetacognitiveMode::Normal);
  REQUIRE (pctx.metacognitive_mode_expires_at == 0);
  REQUIRE_FALSE (pctx.metacognitive_certainty_satisfied);
}

TEST_CASE ("Expired metacognitive mode is cleared before new decisions",
           "[operations][metacognitive][state]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 5000;
  ProcessorContext pctx;
  pctx.metacognitive_mode
      = ProcessorContext::MetacognitiveMode::TotRecovery;
  pctx.metacognitive_mode_expires_at = 4000;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.1);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, 1.0 } });
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE_FALSE (ctx.GetMetacogTOTDetected ());
  REQUIRE_FALSE (ctx.GetMetacogUnknownDetected ());
  REQUIRE (pctx.metacognitive_mode
           == ProcessorContext::MetacognitiveMode::Normal);
  REQUIRE (pctx.metacognitive_mode_expires_at == 0);
}
