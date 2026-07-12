#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/drift_accumulation.hpp>
#include <cortext/operations/streaming_pacing.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <Eigen/Dense>
#include <algorithm>

using namespace cortext;
using cortext::operations::UpdateDriftAccumulation;
using cortext::operations::CheckStreamingPacing;

namespace
{

Signal
MakeSignal (Eigen::VectorXf emb)
{
  Signal s;
  s.embedding = std::move (emb);
  s.timestamp = 1;
  s.source_id = "test";
  return s;
}

} // namespace

TEST_CASE ("UpdateDriftAccumulation initializes on first signal",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);


  Eigen::VectorXf emb = Eigen::VectorXf::Ones (4);
  Signal s = MakeSignal (emb);
  OperationContext ctx (s, pctx, cfg);

  REQUIRE (pctx.accumulator_states.empty ());

  UpdateDriftAccumulation op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.accumulator_states.count ("test") == 1);
  const auto &acc = pctx.accumulator_states.at ("test");
  REQUIRE (acc.prev_x.size () == emb.size ());
  REQUIRE (acc.drift_accum == 0.0);
  REQUIRE (ctx.GetDriftAccumSnapshot () == 0.0);
}

TEST_CASE ("UpdateDriftAccumulation accumulates drift",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);


  // First signal: initialize
  Eigen::VectorXf emb1 = Eigen::VectorXf::Zero (4);
  Signal s1 = MakeSignal (emb1);
  OperationContext ctx1 (s1, pctx, cfg);

  UpdateDriftAccumulation op;
  op.Execute (ctx1, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.accumulator_states.at ("test").drift_accum == 0.0);

  // Second signal: distance = ||[1,0,0,0] - [0,0,0,0]|| = 1.0
  Eigen::VectorXf emb2 = Eigen::VectorXf::Zero (4);
  emb2[0] = 1.0f;
  Signal s2 = MakeSignal (emb2);
  OperationContext ctx2 (s2, pctx, cfg);

  op.Execute (ctx2, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.accumulator_states.at ("test").drift_accum == Catch::Approx (1.0));
  REQUIRE (ctx2.GetDriftAccumSnapshot () == Catch::Approx (1.0));
}

TEST_CASE ("UpdateDriftAccumulation handles empty embedding",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);


  // Empty embedding should be skipped
  Eigen::VectorXf emb;
  Signal s = MakeSignal (emb);
  OperationContext ctx (s, pctx, cfg);

  UpdateDriftAccumulation op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.accumulator_states.count ("test") == 1);
  REQUIRE (pctx.accumulator_states.at ("test").prev_x.size () == 0);
  REQUIRE (pctx.accumulator_states.at ("test").drift_accum == 0.0);
}

TEST_CASE ("UpdateDriftAccumulation handles dimension mismatch",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);


  // First signal: 4D
  Eigen::VectorXf emb1 = Eigen::VectorXf::Ones (4);
  Signal s1 = MakeSignal (emb1);
  OperationContext ctx1 (s1, pctx, cfg);

  UpdateDriftAccumulation op;
  op.Execute (ctx1, cortext::testing::GetNullTransaction ());

  pctx.accumulator_states["test"].drift_accum = 1.0; // Simulate accumulated drift

  // Second signal: 8D (dimension mismatch)
  Eigen::VectorXf emb2 = Eigen::VectorXf::Ones (8);
  Signal s2 = MakeSignal (emb2);
  OperationContext ctx2 (s2, pctx, cfg);

  op.Execute (ctx2, cortext::testing::GetNullTransaction ());

  // Should reset tracking
  REQUIRE (pctx.accumulator_states.at ("test").drift_accum == 0.0);
  REQUIRE (pctx.accumulator_states.at ("test").prev_x.size () == 8);
}

TEST_CASE ("CheckStreamingPacing gates retrieval below threshold",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5;  // threshold from StreamingPacingThreshold(...)
  cfg.focus = 0.5;        // max_wait from MaxWaitDrift(...)


  auto &acc = pctx.accumulator_states["test"];
  acc.drift_acc_pacing = 0.1;  // Below helper-derived threshold.
  acc.x_last_check = Eigen::VectorXf::Ones (4);

  Signal s = MakeSignal (Eigen::VectorXf::Ones (4));
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetShouldCheckRetrieval () == false);
  // Drift accum should NOT be reset
  REQUIRE (pctx.accumulator_states.at ("test").drift_acc_pacing
           == Catch::Approx (0.1));
}

TEST_CASE ("CheckStreamingPacing triggers retrieval above threshold",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5;  // threshold from StreamingPacingThreshold(...)
  cfg.focus = 0.5;        // max_wait from MaxWaitDrift(...)


  auto &acc = pctx.accumulator_states["test"];
  acc.drift_acc_pacing = 0.5;  // Above helper-derived threshold.
  acc.x_last_check = Eigen::VectorXf::Zero (4);

  Eigen::VectorXf emb = Eigen::VectorXf::Ones (4);
  Signal s = MakeSignal (emb);
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetShouldCheckRetrieval () == true);
  // Should reset after trigger
  REQUIRE (pctx.accumulator_states.at ("test").drift_acc_pacing == 0.0);
  // Should update x_last_check
  REQUIRE (pctx.accumulator_states.at ("test").x_last_check.size () > 0);
}

TEST_CASE ("CheckStreamingPacing derives retrieval bias gate from knobs",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const double pacing_threshold
      = cortext::core::StreamingPacingThreshold (cfg.sensitivity);
  const double bias_threshold
      = cortext::core::StreamingRetrievalBiasThreshold (
          cfg.focus, cfg.sensitivity, cfg.stability);

  auto &acc = pctx.accumulator_states["test"];
  acc.drift_acc_pacing = pacing_threshold + 0.01;
  acc.x_last_check = Eigen::VectorXf::Ones (4);

  Signal s = MakeSignal (Eigen::VectorXf::Ones (4));
  pctx.retrieval_bias = std::max (0.0, bias_threshold - 0.01);
  OperationContext blocked_ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (blocked_ctx, cortext::testing::GetNullTransaction ());

  REQUIRE_FALSE (blocked_ctx.GetShouldCheckRetrieval ());
  REQUIRE (pctx.accumulator_states.at ("test").drift_acc_pacing
           == Catch::Approx (pacing_threshold + 0.01));

  pctx.retrieval_bias = std::min (1.0, bias_threshold + 0.01);
  OperationContext allowed_ctx (s, pctx, cfg);
  op.Execute (allowed_ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (allowed_ctx.GetShouldCheckRetrieval ());
  REQUIRE (pctx.accumulator_states.at ("test").drift_acc_pacing == 0.0);
}

TEST_CASE ("CheckStreamingPacing always retrieves for ephemeral queries",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const double bias_threshold
      = cortext::core::StreamingRetrievalBiasThreshold (
          cfg.focus, cfg.sensitivity, cfg.stability);

  auto &acc = pctx.accumulator_states["test"];
  acc.drift_acc_pacing = 0.0;
  acc.x_last_check = Eigen::VectorXf::Ones (4);
  pctx.retrieval_bias = std::max (0.0, bias_threshold - 0.01);

  Signal s = MakeSignal (Eigen::VectorXf::Ones (4));
  s.retention = Retention::Ephemeral;
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetShouldCheckRetrieval ());
  REQUIRE (pctx.accumulator_states.at ("test").drift_acc_pacing == 0.0);
  REQUIRE (pctx.last_retrieval_ts == 0);
}

TEST_CASE ("CheckStreamingPacing force check on max_wait_drift",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.0;  // threshold from StreamingPacingThreshold(...)
  cfg.focus = 1.0;        // max_wait from MaxWaitDrift(...)


  auto &acc = pctx.accumulator_states["test"];
  acc.drift_acc_pacing = 0.6;  // Above helper-derived max_wait.
  acc.x_last_check = Eigen::VectorXf::Zero (4);

  Signal s = MakeSignal (Eigen::VectorXf::Ones (4));
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Should trigger because drift_acc_pacing (0.6) > max_wait (0.5)
  REQUIRE (ctx.GetShouldCheckRetrieval () == true);
  REQUIRE (pctx.accumulator_states.at ("test").drift_acc_pacing == 0.0);  // Reset after trigger
}

TEST_CASE ("CheckStreamingPacing default is true for backward compatibility",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);


  Signal s = MakeSignal (Eigen::VectorXf::Ones (4));
  OperationContext ctx (s, pctx, cfg);

  // Before any operation runs, default should be true
  REQUIRE (ctx.GetShouldCheckRetrieval () == true);
}
