// tests/operations_precision_modulation.test.cpp
// Tests for §4.2.5 Precision-Based Threshold Adjustment
// Spec: Δθ_prec = clamp(κ_prec × F × (coherence_struct_t − 0.5), −cap_prec, +cap_prec)
// where κ_prec and cap_prec are computed by KappaPrec(...) and CapPrec(...).

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

} // namespace

TEST_CASE ("§4.2.5 sets positive Δ when structural coherence above 0.5",
           "[operations][precision][threshold]")
{
  ProcessorContext pc;
  pc.hysteresis = 0.10;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto sig = MakeSignal ();
  OperationContext ctx (sig, pc, cfg);

  // Set high structural coherence (default is 1.0, but be explicit)
  ctx.SetStructuralCoherence (0.8);

  const double cap = cortext::core::CapPrec (cfg.focus, cfg.sensitivity,
                                             cfg.stability, pc.hysteresis);
  const double expected = cortext::core::Clamp (
      cortext::core::KappaPrec (cfg.focus, cfg.sensitivity, cfg.stability)
          * cortext::core::FocusBias (cfg.focus) * (0.8 - 0.5),
      -cap, cap);

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
  pc.hysteresis = 0.10;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto sig = MakeSignal ();
  OperationContext ctx (sig, pc, cfg);

  // Set low structural coherence
  ctx.SetStructuralCoherence (0.2);

  const double cap = cortext::core::CapPrec (cfg.focus, cfg.sensitivity,
                                             cfg.stability, pc.hysteresis);
  const double expected = cortext::core::Clamp (
      cortext::core::KappaPrec (cfg.focus, cfg.sensitivity, cfg.stability)
          * cortext::core::FocusBias (cfg.focus) * (0.2 - 0.5),
      -cap, cap);

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
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto sig = MakeSignal ();
  OperationContext ctx (sig, pc, cfg);

  // Set structural coherence exactly at midpoint
  ctx.SetStructuralCoherence (0.5);

  // Expected: Δθ_prec is zero at the structural-coherence midpoint.
  UpdatePrecisionDelta op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetDeltaThresholdPrecision ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdPrecision () == Catch::Approx (0.0));
}

TEST_CASE ("§4.2.5 Δ scales with Focus knob",
           "[operations][precision][threshold]")
{
  auto sig = MakeSignal ();

  // High Focus should give larger delta magnitude through KappaPrec(...)
  // and FocusBias(...), with hysteresis set high enough to avoid capping.

  // Create separate ProcessorContext and config for each case
  ProcessorContext pc_high;
  pc_high.hysteresis = 1.0;  // Large enough to avoid capping.
  SignalProcessor::Config cfg_high;
  cortext::testing::RequireEncoder (cfg_high);
  cfg_high.focus = 1.0;
  cfg_high.sensitivity = 0.5;
  cfg_high.stability = 0.5;
  OperationContext ctx_high (sig, pc_high, cfg_high);
  ctx_high.SetStructuralCoherence (0.6);

  ProcessorContext pc_low;
  pc_low.hysteresis = 1.0;
  SignalProcessor::Config cfg_low;
  cortext::testing::RequireEncoder (cfg_low);
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
  const double expected_ratio
      = (cortext::core::KappaPrec (cfg_high.focus, cfg_high.sensitivity,
                                   cfg_high.stability)
         * cortext::core::FocusBias (cfg_high.focus))
        / (cortext::core::KappaPrec (cfg_low.focus, cfg_low.sensitivity,
                                     cfg_low.stability)
           * cortext::core::FocusBias (cfg_low.focus));
  REQUIRE (delta_high > delta_low);
  REQUIRE (delta_high / delta_low == Catch::Approx (expected_ratio).epsilon (0.01));
}
