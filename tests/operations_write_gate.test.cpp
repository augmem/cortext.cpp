#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/operations/write_gate.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/accumulator_state.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ComputeWriteGate;

namespace
{
constexpr int kEmbeddingDim = 256;

/// @brief Helper to set up accumulator state for write gate tests
void
SetupAccumulatorState (ProcessorContext &pctx, const Signal &s, double s_max,
                       double s_sum, int n_signals)
{
  AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.e_peak = s.embedding;
  acc.s_max = s_max;
  acc.s_sum = s_sum;
  acc.n_signals = n_signals;
  acc.t_start = s.timestamp - 5000;
  pctx.accumulator_states[s.source_id] = std::move (acc);
}
} // namespace

TEST_CASE ("ComputeWriteGate accepts when S_window above threshold",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "test";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Set up accumulator with high scores
  SetupAccumulatorState (pctx, s, 0.8, 2.0, 3);

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.3); // Low threshold

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // S_window should exceed threshold → write accepted
  REQUIRE (ctx.GetAccumulatorWriteDecision () == true);
  // Representative embedding should be set
  REQUIRE (ctx.GetRepresentativeEmbedding ().has_value ());
}

TEST_CASE ("ComputeWriteGate stores a closed unit whatever its window score",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "test";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Low scores against a high dynamic threshold: the unit still closed.
  SetupAccumulatorState (pctx, s, 0.1, 0.2, 2);

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.9);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetAccumulatorWriteDecision () == true);
  REQUIRE (ctx.GetWriteDecision () == true);
  REQUIRE (ctx.GetRepresentativeEmbedding ().has_value ());
  REQUIRE (ctx.GetWindowScore ().has_value ());
  REQUIRE (*ctx.GetWindowScore () < 0.9);
}

TEST_CASE ("ComputeWriteGate rejects when no flush required",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "test";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Set up accumulator with high scores
  SetupAccumulatorState (pctx, s, 0.9, 2.5, 4);

  OperationContext ctx (s, pctx, cfg);
  // No flush required - signal not at boundary
  ctx.SetFlushRequired (false);
  ctx.SetThresholdTDynamic (0.2);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // No flush → no write decision regardless of scores
  REQUIRE (ctx.GetAccumulatorWriteDecision () == false);
}

TEST_CASE ("ComputeWriteGate accepts with spike_bypass",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "test";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Set up accumulator with high scores
  SetupAccumulatorState (pctx, s, 0.8, 2.0, 3);

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (false); // No boundary
  ctx.SetSpikeBypass (true);    // But spike bypass triggers
  ctx.SetThresholdTDynamic (0.3);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Spike bypass → can still write
  REQUIRE (ctx.GetAccumulatorWriteDecision () == true);
}

TEST_CASE ("ComputeWriteGate rejects ephemeral signals even with spike_bypass",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "test";
  s.retention = Retention::Ephemeral;

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  SetupAccumulatorState (pctx, s, 0.9, 2.5, 4);

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetSpikeBypass (true);
  ctx.SetThresholdTDynamic (0.1);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetAccumulatorWriteDecision () == false);
  REQUIRE (ctx.GetWriteDecision () == false);
  REQUIRE_FALSE (ctx.GetRepresentativeEmbedding ().has_value ());
}

TEST_CASE ("ComputeWriteGate rejects when no accumulator state",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "test";

  ProcessorContext pctx;
  // No accumulator state set up

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.0);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // No accumulator state → rejected
  REQUIRE (ctx.GetAccumulatorWriteDecision () == false);
}

TEST_CASE ("ComputeWriteGate populates recent_memory_centroids on accept",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "test";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Set up accumulator with high scores
  SetupAccumulatorState (pctx, s, 0.8, 2.0, 3);

  const size_t before_size = pctx.recent_memory_centroids.size ();

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.3);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetAccumulatorWriteDecision () == true);
  // Section 8.2: recent_memory_centroids should be populated
  REQUIRE (pctx.recent_memory_centroids.size () == before_size + 1);
}

TEST_CASE ("ComputeWriteGate profiles every decision input without changing it",
           "[operations][write_gate][profile]")
{
  cortext::testing::ScopedEnvVar profile (
      "CORTEXT_PROFILE_WORK_COUNTERS", "1");

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "opaque/profile/source";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  SetupAccumulatorState (pctx, s, 0.8, 2.0, 3);

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.3);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  const auto &profile_values = ctx.GetOperationTimings ();
  REQUIRE (ctx.GetAccumulatorWriteDecision ());
  REQUIRE (profile_values.at ("WriteGate.flush_trigger") == 1.0);
  REQUIRE (profile_values.at ("WriteGate.spike_bypass") == 0.0);
  REQUIRE (profile_values.at ("WriteGate.accumulator_available") == 1.0);
  REQUIRE (profile_values.at ("WriteGate.n_signals") == 3.0);
  REQUIRE (profile_values.at ("WriteGate.write_accumulator") == 1.0);
  REQUIRE (profile_values.at ("WriteGate.reason_code") == 5.0);
  REQUIRE (profile_values.at ("WriteGate.window_score") > 0.0);
  REQUIRE_FALSE (profile_values.contains ("WriteGate.effective_threshold"));
}

TEST_CASE ("ComputeWriteGate profiles non-boundary rejection",
           "[operations][write_gate][profile]")
{
  cortext::testing::ScopedEnvVar profile (
      "CORTEXT_PROFILE_WORK_COUNTERS", "1");

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "opaque/profile/source";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  SetupAccumulatorState (pctx, s, 0.8, 2.0, 3);

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (false);
  ctx.SetThresholdTDynamic (0.3);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  const auto &profile_values = ctx.GetOperationTimings ();
  REQUIRE_FALSE (ctx.GetAccumulatorWriteDecision ());
  REQUIRE (profile_values.at ("WriteGate.flush_trigger") == 0.0);
  REQUIRE (profile_values.at ("WriteGate.write_accumulator") == 0.0);
  REQUIRE (profile_values.at ("WriteGate.reason_code") == 2.0);
}

TEST_CASE ("ComputeWriteGate profiles every remaining exit class",
           "[operations][write_gate][profile]")
{
  cortext::testing::ScopedEnvVar profile (
      "CORTEXT_PROFILE_WORK_COUNTERS", "1");

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.embedding.normalize ();
  s.timestamp = 100000;
  s.source_id = "opaque/profile/source";

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ComputeWriteGate op;

  SECTION ("ephemeral")
  {
    s.retention = Retention::Ephemeral;
    ProcessorContext pctx;
    SetupAccumulatorState (pctx, s, 0.8, 2.0, 3);
    OperationContext ctx (s, pctx, cfg);
    ctx.SetFlushRequired (true);
    ctx.SetSpikeBypass (true);
    ctx.SetThresholdTDynamic (0.3);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    const auto &values = ctx.GetOperationTimings ();
    REQUIRE_FALSE (ctx.GetAccumulatorWriteDecision ());
    REQUIRE (values.at ("WriteGate.accumulator_available") == 0.0);
    REQUIRE (values.at ("WriteGate.write_accumulator") == 0.0);
    REQUIRE (values.at ("WriteGate.reason_code") == 1.0);
  }

  SECTION ("no accumulator")
  {
    ProcessorContext pctx;
    OperationContext ctx (s, pctx, cfg);
    ctx.SetFlushRequired (true);
    ctx.SetThresholdTDynamic (0.3);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    const auto &values = ctx.GetOperationTimings ();
    REQUIRE_FALSE (ctx.GetAccumulatorWriteDecision ());
    REQUIRE (values.at ("WriteGate.accumulator_available") == 0.0);
    REQUIRE (values.at ("WriteGate.write_accumulator") == 0.0);
    REQUIRE (values.at ("WriteGate.reason_code") == 3.0);
  }

  SECTION ("forced")
  {
    ProcessorContext pctx;
    SetupAccumulatorState (pctx, s, 0.1, 0.2, 2);
    OperationContext ctx (s, pctx, cfg);
    ctx.SetFlushRequired (false);
    ctx.SetSpikeBypass (true);
    ctx.SetThresholdTDynamic (0.9);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    const auto &values = ctx.GetOperationTimings ();
    REQUIRE (ctx.GetAccumulatorWriteDecision ());
    REQUIRE (values.at ("WriteGate.force_write") == 1.0);
    REQUIRE (values.at ("WriteGate.write_accumulator") == 1.0);
    REQUIRE (values.at ("WriteGate.reason_code") == 4.0);
  }

  SECTION ("closed with a low window score")
  {
    ProcessorContext pctx;
    SetupAccumulatorState (pctx, s, 0.1, 0.2, 2);
    OperationContext ctx (s, pctx, cfg);
    ctx.SetFlushRequired (true);
    ctx.SetThresholdTDynamic (0.9);
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    const auto &values = ctx.GetOperationTimings ();
    REQUIRE (ctx.GetAccumulatorWriteDecision ());
    REQUIRE (values.at ("WriteGate.force_write") == 0.0);
    REQUIRE (values.at ("WriteGate.write_accumulator") == 1.0);
    REQUIRE (values.at ("WriteGate.reason_code") == 5.0);
  }
}
