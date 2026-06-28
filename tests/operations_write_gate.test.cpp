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
  acc.last_write_ts = 0; // Never written before
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

TEST_CASE ("ComputeWriteGate rejects when S_window below threshold",
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

  // Set up accumulator with low scores
  SetupAccumulatorState (pctx, s, 0.1, 0.2, 2);

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.9); // High threshold

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // S_window below threshold → write rejected
  REQUIRE (ctx.GetAccumulatorWriteDecision () == false);
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
  s.force_write = true;

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

TEST_CASE ("ComputeWriteGate treats backward refractory timestamps as zero delta",
           "[operations][write_gate][robustness]")
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

  SetupAccumulatorState (pctx, s, 0.5, 1.0, 2);
  pctx.accumulator_states[s.source_id].last_write_ts = s.timestamp + 10000;

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.45);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetAccumulatorWriteDecision () == false);
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

TEST_CASE ("ComputeWriteGate lowers threshold under high NE",
           "[operations][write_gate][neuromod]")
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

  SetupAccumulatorState (pctx, s, 0.5, 1.0, 2);
  pctx.neuromod_ne = 1.0;

  OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.6);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
  REQUIRE (ctx.GetAccumulatorWriteDecision () == true);

  {
    cortext::testing::ScopedEnvVar disable (
        "CORTEXT_DISABLE_NEUROMOD_WRITE_SCALE", "1");
    ProcessorContext baseline_pctx;
    SetupAccumulatorState (baseline_pctx, s, 0.5, 1.0, 2);
    baseline_pctx.neuromod_ne = 1.0;
    OperationContext baseline_ctx (s, baseline_pctx, cfg);
    baseline_ctx.SetFlushRequired (true);
    baseline_ctx.SetThresholdTDynamic (0.6);
    op.Execute (baseline_ctx, cortext::testing::GetNullTransaction ());
    REQUIRE (baseline_ctx.GetAccumulatorWriteDecision () == false);
  }
}
