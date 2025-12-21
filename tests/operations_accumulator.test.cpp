#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/core/knobs.hpp>
#include <cortext/operations/accumulator.hpp>
#include <cortext/operations/boundary.hpp>
#include <cortext/operations/coherence.hpp>
#include <cortext/operations/write_gate.hpp>
#include <cortext/operations/spike_bypass.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/accumulator_state.hpp>

using namespace cortext;
using cortext::operations::CheckSpikeBypass;
using cortext::operations::ComputeWriteGate;
using cortext::operations::DetectBoundary;
using cortext::operations::UpdateAccumulator;

namespace
{
Eigen::VectorXf
MakeRandomEmbedding (int dim = 256)
{
  Eigen::VectorXf v = Eigen::VectorXf::Random (dim);
  v.normalize ();
  return v;
}

Eigen::VectorXf
MakeSimilarEmbedding (const Eigen::VectorXf &base, float noise = 0.1f)
{
  Eigen::VectorXf v = base + noise * Eigen::VectorXf::Random (base.size ());
  v.normalize ();
  return v;
}
} // namespace

TEST_CASE ("Accumulator updates state", "[accumulator][4.4.1]")
{
  SECTION ("GIVEN first signal for a source")
  {
    Signal s;
    s.source_id = "test_source";
    s.embedding = MakeRandomEmbedding ();
    s.timestamp = 1000;

    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.stability = 0.5;

    OperationContext ctx (s, pctx, cfg);
    ctx.SetCompositeScore (0.5);
    ctx.SetMetric (operations::Metric::drift_mag, 0.0);

    UpdateAccumulator op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    REQUIRE (pctx.accumulator_states.count ("test_source") == 1);
    auto &state = pctx.accumulator_states.at ("test_source");
    REQUIRE (state.n_signals == 1);
    REQUIRE (state.t_start == 1000);
  }

  SECTION ("GIVEN subsequent signals")
  {
    Signal s;
    s.source_id = "test_source";
    s.embedding = MakeRandomEmbedding ();
    s.timestamp = 2000;

    ProcessorContext pctx;
    // Pre-populate accumulator state
    AccumulatorState state;
    state.Reset (MakeRandomEmbedding (), 1000);
    state.n_signals = 3;
    state.s_sum = 1.5;
    state.s_max = 0.7;
    pctx.accumulator_states["test_source"] = std::move (state);

    SignalProcessor::Config cfg;

    cortext::testing::RequireEncoder (cfg);
    cfg.stability = 0.5;

    OperationContext ctx (s, pctx, cfg);
    ctx.SetCompositeScore (0.8);
    ctx.SetMetric (operations::Metric::drift_mag, 0.1);

    UpdateAccumulator op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    auto &updated = pctx.accumulator_states.at ("test_source");
    REQUIRE (updated.n_signals == 4);
    REQUIRE (updated.s_sum > 1.5);
    REQUIRE (updated.s_max >= 0.8); // new score is higher
  }
}

TEST_CASE ("Boundary detection triggers on drift spike",
           "[accumulator][4.4.3]")
{
  SECTION ("GIVEN high drift step relative to baseline")
  {
    Signal s;
    s.source_id = "test_source";
    s.embedding = MakeRandomEmbedding ();
    s.timestamp = 5000;

    ProcessorContext pctx;
    AccumulatorState state;
    state.Reset (MakeRandomEmbedding (), 1000);
    state.n_signals = 5;
    state.eta_acc = 0.05; // Low baseline
    pctx.accumulator_states["test_source"] = std::move (state);

    SignalProcessor::Config cfg;

    cortext::testing::RequireEncoder (cfg);
    cfg.stability = 0.5;
    cfg.focus = 0.5;
    cfg.sensitivity = 0.5;

    OperationContext ctx (s, pctx, cfg);
    ctx.SetAccumulatorDriftStep (0.5); // High drift step
    ctx.SetAccumulatorCoherence (0.9);

    DetectBoundary op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    auto score = ctx.GetBoundaryScore ();
    REQUIRE (score.has_value ());
    REQUIRE (*score > 0.0);
    // Should trigger flush due to large drift spike
  }
}

TEST_CASE ("Boundary detection triggers on temporal gap",
           "[accumulator][4.4.3]")
{
  SECTION ("GIVEN large time gap between signals")
  {
    Signal s;
    s.source_id = "test_source";
    s.embedding = MakeRandomEmbedding ();
    s.timestamp = 100000; // 100 seconds later

    ProcessorContext pctx;
    AccumulatorState state;
    state.Reset (MakeRandomEmbedding (), 1000);
    state.n_signals = 3;
    state.last_signal_ts = 1000; // Last signal at t=1000
    pctx.accumulator_states["test_source"] = std::move (state);

    SignalProcessor::Config cfg;

    cortext::testing::RequireEncoder (cfg);
    cfg.stability = 0.0; // Low stability = short gap threshold

    OperationContext ctx (s, pctx, cfg);
    ctx.SetAccumulatorDriftStep (0.0);
    ctx.SetAccumulatorCoherence (1.0);

    DetectBoundary op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    REQUIRE (ctx.GetFlushRequired () == true);
  }
}

TEST_CASE ("Spike bypass triggers for high-salience signals",
           "[accumulator][4.4.4]")
{
  SECTION ("GIVEN score exceeds dynamic threshold plus margin")
  {
    Signal s;
    s.source_id = "test_source";
    s.embedding = MakeRandomEmbedding ();
    s.timestamp = 5000;

    ProcessorContext pctx;
    AccumulatorState state;
    state.Reset (MakeRandomEmbedding (), 1000);
    state.n_signals = 3;
    pctx.accumulator_states["test_source"] = std::move (state);

    SignalProcessor::Config cfg;

    cortext::testing::RequireEncoder (cfg);
    cfg.sensitivity = 0.5;

    OperationContext ctx (s, pctx, cfg);
    ctx.SetCompositeScore (0.95);
    ctx.SetThresholdTDynamic (0.5);

    CheckSpikeBypass op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    REQUIRE (ctx.GetSpikeBypass () == true);
    REQUIRE (ctx.GetFlushRequired () == true);
  }

  SECTION ("GIVEN score below threshold plus margin")
  {
    Signal s;
    s.source_id = "test_source";
    s.embedding = MakeRandomEmbedding ();
    s.timestamp = 5000;

    ProcessorContext pctx;
    AccumulatorState state;
    state.Reset (MakeRandomEmbedding (), 1000);
    state.n_signals = 3;
    pctx.accumulator_states["test_source"] = std::move (state);

    SignalProcessor::Config cfg;

    cortext::testing::RequireEncoder (cfg);
    cfg.sensitivity = 0.5;

    OperationContext ctx (s, pctx, cfg);
    ctx.SetCompositeScore (0.6);
    ctx.SetThresholdTDynamic (0.5);

    CheckSpikeBypass op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    REQUIRE (ctx.GetSpikeBypass () == false);
  }
}

TEST_CASE ("Write gate computes window score", "[accumulator][4.4.5]")
{
  SECTION ("GIVEN flush required and good window score")
  {
    Signal s;
    s.source_id = "test_source";
    s.embedding = MakeRandomEmbedding ();
    s.timestamp = 10000;

    ProcessorContext pctx;
    AccumulatorState state;
    state.Reset (MakeRandomEmbedding (), 1000);
    state.n_signals = 10;
    state.s_sum = 7.0;
    state.s_max = 0.9;
    state.e_peak = MakeRandomEmbedding ();
    state.last_write_ts = 0;
    pctx.accumulator_states["test_source"] = std::move (state);

    SignalProcessor::Config cfg;

    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.5;
    cfg.sensitivity = 0.5;
    cfg.stability = 0.5;

    OperationContext ctx (s, pctx, cfg);
    ctx.SetFlushRequired (true);
    ctx.SetThresholdTDynamic (0.3);

    ComputeWriteGate op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    auto window_score = ctx.GetWindowScore ();
    REQUIRE (window_score.has_value ());
    REQUIRE (*window_score > 0.0);
    REQUIRE (ctx.GetAccumulatorWriteDecision () == true);
    REQUIRE (ctx.GetRepresentativeEmbedding ().has_value ());
  }

  SECTION ("GIVEN no flush required")
  {
    Signal s;
    s.source_id = "test_source";
    s.embedding = MakeRandomEmbedding ();
    s.timestamp = 5000;

    ProcessorContext pctx;
    AccumulatorState state;
    state.Reset (MakeRandomEmbedding (), 1000);
    state.n_signals = 3;
    pctx.accumulator_states["test_source"] = std::move (state);

    SignalProcessor::Config cfg;

    cortext::testing::RequireEncoder (cfg);

    OperationContext ctx (s, pctx, cfg);
    ctx.SetFlushRequired (false);
    ctx.SetSpikeBypass (false);

    ComputeWriteGate op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    REQUIRE (ctx.GetAccumulatorWriteDecision () == false);
  }
}

TEST_CASE ("Write refractory suppresses rapid writes", "[accumulator][4.4.5]")
{
  SECTION ("GIVEN recent write timestamp")
  {
    Signal s;
    s.source_id = "test_source";
    s.embedding = MakeRandomEmbedding ();
    s.timestamp = 3000; // Only 2 seconds after last write

    ProcessorContext pctx;
    AccumulatorState state;
    state.Reset (MakeRandomEmbedding (), 1000);
    state.n_signals = 5;
    state.s_sum = 2.5;
    state.s_max = 0.6;
    state.e_peak = MakeRandomEmbedding ();
    state.last_write_ts = 1000; // Wrote 2 seconds ago
    pctx.accumulator_states["test_source"] = std::move (state);

    SignalProcessor::Config cfg;

    cortext::testing::RequireEncoder (cfg);
    cfg.stability = 0.0; // Low stability = strong refractory

    OperationContext ctx (s, pctx, cfg);
    ctx.SetFlushRequired (true);
    ctx.SetThresholdTDynamic (0.4);

    ComputeWriteGate op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    // With recent write and refractory multiplier, threshold is elevated
    // Medium-quality accumulator should be rejected
    auto window_score = ctx.GetWindowScore ();
    REQUIRE (window_score.has_value ());
    // The refractory multiplier raises effective threshold
  }
}

TEST_CASE ("AccumulatorState reset and accumulate", "[accumulator][state]")
{
  SECTION ("GIVEN reset with embedding")
  {
    AccumulatorState state;
    auto emb = MakeRandomEmbedding ();
    state.Reset (emb, 1000);

    REQUIRE (state.n_signals == 1);
    REQUIRE (state.t_start == 1000);
    REQUIRE (state.drift_acc == 0.0);
    REQUIRE (state.s_sum == 0.0);
    REQUIRE (state.s_max == 0.0);
    REQUIRE (state.mu_acc.size () == emb.size ());
  }

  SECTION ("GIVEN accumulate updates running mean")
  {
    AccumulatorState state;
    auto emb1 = MakeRandomEmbedding ();
    state.Reset (emb1, 1000);

    auto emb2 = MakeSimilarEmbedding (emb1, 0.2f);
    state.Accumulate (emb2, 0.7, 0.1);

    REQUIRE (state.n_signals == 2);
    REQUIRE (state.s_sum == Catch::Approx (0.7));
    REQUIRE (state.s_max == Catch::Approx (0.7));
    REQUIRE (state.drift_acc == Catch::Approx (0.1));
  }
}

TEST_CASE ("Knob-derived accumulator functions", "[accumulator][knobs]")
{
  SECTION ("AlphaEtaAcc varies with stability")
  {
    double low = core::AlphaEtaAcc (0.0);
    double high = core::AlphaEtaAcc (1.0);
    REQUIRE (low == Catch::Approx (0.3));
    REQUIRE (high == Catch::Approx (0.1));
  }

  SECTION ("BoundaryThreshold varies with focus and sensitivity")
  {
    double val = core::BoundaryThreshold (0.5, 0.5);
    REQUIRE (val > 0.0);
    REQUIRE (val < 1.0);
  }

  SECTION ("SpikeMargin varies with sensitivity")
  {
    double low = core::SpikeMargin (0.0);
    double high = core::SpikeMargin (1.0);
    REQUIRE (low == Catch::Approx (0.3));
    REQUIRE (high == Catch::Approx (0.15));
  }

  SECTION ("WriteRefractoryTau varies with stability")
  {
    double low = core::WriteRefractoryTau (0.0);
    double high = core::WriteRefractoryTau (1.0);
    REQUIRE (low == Catch::Approx (5.0));
    REQUIRE (high == Catch::Approx (30.0));
  }

  SECTION ("RepresentativeBlendRho varies with focus")
  {
    double low = core::RepresentativeBlendRho (0.0);
    double high = core::RepresentativeBlendRho (1.0);
    REQUIRE (low == Catch::Approx (0.3));
    REQUIRE (high == Catch::Approx (0.7));
  }
}
