// tests/operations_precision_modulation.test.cpp
// Tests for §4.2.5 Precision-Based Threshold Adjustment
// Spec: Δθ_prec = clamp(κ_prec × F × (coherence_struct_t − 0.5), −cap_prec, +cap_prec)
// where κ_prec = 0.06, cap_prec = 0.15 × hysteresis_t

#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>

#include <cortext/core/knobs.hpp>
#include <cortext/operations/precision.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/signal.hpp>

using namespace cortext;
using cortext::operations::UpdatePrecisionDelta;

namespace
{
static Signal
MakeSignal ()
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = 0;
  s.source_id = "test";
  return s;
}

constexpr double kKappaPrec = 0.06;
constexpr double kCapPrecCoeff = 0.15;
} // namespace

TEST_CASE ("§4.2.5 sets positive Δ when structural coherence above 0.5",
           "[operations][precision][threshold]")
{
  ProcessorContext pc;
  pc.hysteresis = 0.10;  // cap_prec = 0.15 * 0.10 = 0.015
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto sig = MakeSignal ();
  OperationContext ctx (sig, pc, cfg);

  // Set high structural coherence (default is 1.0, but be explicit)
  ctx.SetStructuralCoherence (0.8);

  // Expected: Δθ_prec = clamp(0.06 × 1.0 × (0.8 − 0.5), −0.015, +0.015)
  //                   = clamp(0.06 × 0.3, −0.015, +0.015)
  //                   = clamp(0.018, −0.015, +0.015) = 0.015 (capped)
  const double expected = 0.015;

  UpdatePrecisionDelta op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetDeltaThresholdPrecision ().has_value ());
  const double delta = *ctx.GetDeltaThresholdPrecision ();
  REQUIRE (delta > 0.0);
  REQUIRE (delta == Catch::Approx (expected).epsilon (0.001));
}

TEST_CASE ("§4.2.5 sets negative Δ when structural coherence below 0.5",
           "[operations][precision][threshold]")
{
  ProcessorContext pc;
  pc.hysteresis = 0.10;  // cap_prec = 0.15 * 0.10 = 0.015
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto sig = MakeSignal ();
  OperationContext ctx (sig, pc, cfg);

  // Set low structural coherence
  ctx.SetStructuralCoherence (0.2);

  // Expected: Δθ_prec = clamp(0.06 × 1.0 × (0.2 − 0.5), −0.015, +0.015)
  //                   = clamp(0.06 × (−0.3), −0.015, +0.015)
  //                   = clamp(−0.018, −0.015, +0.015) = −0.015 (capped)
  const double expected = -0.015;

  UpdatePrecisionDelta op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetDeltaThresholdPrecision ().has_value ());
  const double delta = *ctx.GetDeltaThresholdPrecision ();
  REQUIRE (delta < 0.0);
  REQUIRE (delta == Catch::Approx (expected).epsilon (0.001));
}

TEST_CASE ("§4.2.5 sets zero Δ when structural coherence equals 0.5",
           "[operations][precision][threshold]")
{
  ProcessorContext pc;
  pc.hysteresis = 0.10;
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto sig = MakeSignal ();
  OperationContext ctx (sig, pc, cfg);

  // Set structural coherence exactly at midpoint
  ctx.SetStructuralCoherence (0.5);

  // Expected: Δθ_prec = 0.06 × F × (0.5 − 0.5) = 0
  UpdatePrecisionDelta op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetDeltaThresholdPrecision ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdPrecision () == Catch::Approx (0.0));
}

TEST_CASE ("§4.2.5 Δ scales with Focus knob",
           "[operations][precision][threshold]")
{
  auto sig = MakeSignal ();

  // High Focus should give larger delta magnitude
  // coherence = 0.6 means (0.6 - 0.5) = 0.1, so delta = 0.06 * F * 0.1 = 0.006 * F
  // For F=1.0: delta = 0.006, for F=0.2: delta = 0.0012, well below cap of 0.15

  // Create separate ProcessorContext and config for each case
  ProcessorContext pc_high;
  pc_high.hysteresis = 1.0;  // Very large hysteresis to avoid capping (cap = 0.15)
  SignalProcessor::Config cfg_high;
  cfg_high.focus = 1.0;
  cfg_high.sensitivity = 0.5;
  cfg_high.stability = 0.5;
  OperationContext ctx_high (sig, pc_high, cfg_high);
  ctx_high.SetStructuralCoherence (0.6);

  ProcessorContext pc_low;
  pc_low.hysteresis = 1.0;
  SignalProcessor::Config cfg_low;
  cfg_low.focus = 0.2;
  cfg_low.sensitivity = 0.5;
  cfg_low.stability = 0.5;
  OperationContext ctx_low (sig, pc_low, cfg_low);
  ctx_low.SetStructuralCoherence (0.6);

  UpdatePrecisionDelta op;
  op.Execute (ctx_high, cortext::testing::GetNullTransaction ());
  op.Execute (ctx_low, cortext::testing::GetNullTransaction ());

  const double delta_high = *ctx_high.GetDeltaThresholdPrecision ();
  const double delta_low = *ctx_low.GetDeltaThresholdPrecision ();

  // Both should be positive (coherence > 0.5)
  REQUIRE (delta_high > 0.0);
  REQUIRE (delta_low > 0.0);
  // High focus should give ~5x larger delta
  REQUIRE (delta_high > delta_low);
  REQUIRE (delta_high / delta_low == Catch::Approx (5.0).epsilon (0.01));
}

