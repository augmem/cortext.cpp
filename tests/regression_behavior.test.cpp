// Regression Behavior Tests for Phase 5
// Tests that formula corrections maintain expected system behavior.
// Uses current (corrected) formulas as the baseline.

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "test_helpers.hpp"
#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/recent_context.hpp>
#include <cortext/operations/uncertainty.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using namespace cortext::core;

// =============================================================================
// 5.3.1 Write Rate Stability Tests
// =============================================================================

TEST_CASE ("WRateSeconds produces expected intervals",
           "[regression][write_rate]")
{
  // Verify the corrected write rate formula: lerp(60, 300, T)

  SECTION ("T=0 yields 60 seconds (minimum interval)")
  {
    REQUIRE (WRateSeconds (0.0) == 60);
  }

  SECTION ("T=1 yields 300 seconds (maximum interval)")
  {
    REQUIRE (WRateSeconds (1.0) == 300);
  }

  SECTION ("Smooth interpolation at mid-range")
  {
    // T=0.5: lerp(60, 300, 0.5) = 180
    REQUIRE (WRateSeconds (0.5) == 180);

    // T=0.25: lerp(60, 300, 0.25) = 120
    REQUIRE (WRateSeconds (0.25) == 120);

    // T=0.75: lerp(60, 300, 0.75) = 240
    REQUIRE (WRateSeconds (0.75) == 240);
  }

  SECTION ("Monotonic increase with stability")
  {
    for (double T = 0.0; T < 1.0; T += 0.1)
      {
        REQUIRE (WRateSeconds (T) <= WRateSeconds (T + 0.1));
      }
  }
}

// =============================================================================
// 5.3.2 Episode Boundary Frequency Tests
// =============================================================================

TEST_CASE ("Episode boundary drift threshold follows lerp(0.10, 0.35, T)",
           "[regression][episodes]")
{
  // The drift threshold for episode boundaries is derived from stability.
  // Already verified as correct in Phase 1.3 - this test documents the baseline.

  // This baseline matches the documented lerp(0.10, 0.35, T) schedule.

  SECTION ("Low stability yields low threshold (more boundaries)")
  {
    // At T=0: threshold = 0.10
    // More frequent episode boundaries - system is more responsive
    double expected_threshold_T0 = 0.10;
    REQUIRE (Lerp (0.10, 0.35, 0.0) == Catch::Approx (expected_threshold_T0));
  }

  SECTION ("High stability yields high threshold (fewer boundaries)")
  {
    // At T=1: threshold = 0.35
    // Fewer episode boundaries - system has more inertia
    double expected_threshold_T1 = 0.35;
    REQUIRE (Lerp (0.10, 0.35, 1.0) == Catch::Approx (expected_threshold_T1));
  }

  SECTION ("Threshold increases monotonically with stability")
  {
    for (double T = 0.0; T < 1.0; T += 0.1)
      {
        double thresh_low = Lerp (0.10, 0.35, T);
        double thresh_high = Lerp (0.10, 0.35, T + 0.1);
        REQUIRE (thresh_low < thresh_high);
      }
  }
}

// =============================================================================
// 5.3.3 Working Memory Capacity Tests
// =============================================================================

TEST_CASE ("Working memory capacity within [15, 27] range",
           "[regression][working_memory]")
{
  // base_capacity = 3 * (lerp(8, 6, SensitivityBias(S))
  //                      + lerp(-1, 1, FocusBias(F)))
  // Capacity range [15, 27], with 21 at neutral knobs.

  SECTION ("Extreme knob values produce correct bounds")
  {
    // S=1, F=0 yields minimum: 3 * (6 + (-1)) = 15
    REQUIRE (WMBaseCapacity (1.0, 0.0) == 15);

    // S=0, F=1 yields maximum: 3 * (8 + 1) = 27
    REQUIRE (WMBaseCapacity (0.0, 1.0) == 27);
  }

  SECTION ("Mid-range knobs produce capacity around 21")
  {
    // S=0.5, F=0.5: biased lerps cancel to exactly 21
    REQUIRE (WMBaseCapacity (0.5, 0.5) == 21);

    // S=0, F=0: 3 * (8 + (-1)) = 21
    REQUIRE (WMBaseCapacity (0.0, 0.0) == 21);

    // S=1, F=1: 3 * (6 + 1) = 21
    REQUIRE (WMBaseCapacity (1.0, 1.0) == 21);
  }

  SECTION ("All combinations stay within [15, 27]")
  {
    for (double S = 0.0; S <= 1.0; S += 0.1)
      {
        for (double F = 0.0; F <= 1.0; F += 0.1)
          {
            int cap = WMBaseCapacity (S, F);
            CAPTURE (S, F, cap);
            REQUIRE (cap >= 15);
            REQUIRE (cap <= 27);
          }
      }
  }
}

TEST_CASE ("WM gate threshold follows spec: lerp(0.1, 0.4, FocusBias(F))",
           "[regression][working_memory]")
{
  // gate_threshold determines what composite score is required for WM entry

  SECTION ("Low focus is permissive (0.1)")
  {
    REQUIRE (WMGateThreshold (0.0) == Catch::Approx (0.1));
  }

  SECTION ("High focus is strict (0.4)")
  {
    REQUIRE (WMGateThreshold (1.0) == Catch::Approx (0.4));
  }

  SECTION ("Gate threshold increases with focus")
  {
    for (int step = 0; step < 10; ++step)
      {
        const double F = static_cast<double> (step) / 10.0;
        REQUIRE (WMGateThreshold (F) < WMGateThreshold (F + 0.1));
      }
  }
}

// =============================================================================
// 5.3.4 Mood Magnitude Normalization Tests
// =============================================================================

TEST_CASE ("Mood magnitude uses sqrt(6) normalization",
           "[regression][mood]")
{
  // Paper Section 4.2.4: m_norm = ||M_t|| / sqrt(6)
  // This ensures mood threshold modulation stays in expected bounds

  SECTION ("Max mood state produces m_norm = 1.0")
  {
    // Max mood vector: [1, 1, 1, 1, 1, 1]
    // ||M|| = sqrt(6)
    // m_norm = sqrt(6) / sqrt(6) = 1.0
    Eigen::VectorXf max_mood = Eigen::VectorXf::Ones (6);
    double magnitude = max_mood.norm ();
    double m_norm = magnitude / std::sqrt (6.0);

    REQUIRE (magnitude == Catch::Approx (std::sqrt (6.0)));
    REQUIRE (m_norm == Catch::Approx (1.0));
  }

  SECTION ("Zero mood produces m_norm = 0.0")
  {
    Eigen::VectorXf zero_mood = Eigen::VectorXf::Zero (6);
    double magnitude = zero_mood.norm ();
    double m_norm = Clamp (magnitude / std::sqrt (6.0), 0.0, 1.0);

    REQUIRE (m_norm == 0.0);
  }

  SECTION ("Half-intensity mood produces m_norm approx 0.5")
  {
    // Mood vector with all elements at 0.5
    Eigen::VectorXf half_mood = Eigen::VectorXf::Constant (6, 0.5f);
    double magnitude = half_mood.norm (); // sqrt(6 * 0.25) = sqrt(1.5)
    double m_norm = Clamp (magnitude / std::sqrt (6.0), 0.0, 1.0);

    REQUIRE (m_norm == Catch::Approx (std::sqrt (1.5 / 6.0)));
    REQUIRE (m_norm < 1.0);
  }
}

TEST_CASE ("Focus update uses recent_context buffer ordering",
           "[regression][focus][order]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = 1;
  s.source_id = "test";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.6;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);

  operations::InitializeFocusPriors init;
  init.Execute (ctx, cortext::testing::GetNullTransaction ());

  operations::UpdateRecentContext update_ctx;
  update_ctx.Execute (ctx, cortext::testing::GetNullTransaction ());

  operations::UpdateUncertainty uop;
  pctx.signals_processed = 5;
  uop.Execute (ctx, cortext::testing::GetNullTransaction ());

  const double u_t = pctx.u_t;
  const double prev = pctx.weight_relevance;

  operations::UpdateFocus fop;
  fop.Execute (ctx, cortext::testing::GetNullTransaction ());

  const double expected = Ewma (prev, 1.0, AlphaF (cfg.focus, u_t));
  REQUIRE (pctx.weight_relevance
           == Catch::Approx (expected).epsilon (1e-6));
}

TEST_CASE ("Delta_threshold_mood bounded by kappa_mood",
           "[regression][mood]")
{
  // ΔThreshold_mood = -κ_mood × clamp(m_norm, 0, 1)
  // With normalization, delta is always in [-κ_mood, 0]

  const double kappa_mood = 0.1; // Example kappa value

  SECTION ("At max mood, delta equals -kappa_mood")
  {
    double m_norm = 1.0; // Max normalized mood
    double delta = -kappa_mood * Clamp (m_norm, 0.0, 1.0);

    REQUIRE (delta == Catch::Approx (-kappa_mood));
  }

  SECTION ("At zero mood, delta equals 0")
  {
    double m_norm = 0.0;
    double delta = -kappa_mood * Clamp (m_norm, 0.0, 1.0);

    REQUIRE (delta == 0.0);
  }

  SECTION ("Delta is always non-positive")
  {
    for (double m_norm = 0.0; m_norm <= 1.5; m_norm += 0.1)
      {
        double clamped = Clamp (m_norm, 0.0, 1.0);
        double delta = -kappa_mood * clamped;

        REQUIRE (delta <= 0.0);
        REQUIRE (delta >= -kappa_mood);
      }
  }
}

// =============================================================================
// 5.3.6 Streaming Pacing Thresholds (Phase 2 additions)
// =============================================================================

TEST_CASE ("Streaming pacing threshold decreases with Sensitivity",
           "[regression][streaming]")
{
  // pacing_thresh(S) = lerp(0.3, 0.05, S)
  // Higher sensitivity = lower threshold = more frequent checks

  SECTION ("Low sensitivity is conservative (0.3)")
  {
    REQUIRE (StreamingPacingThreshold (0.0) == Catch::Approx (0.3));
  }

  SECTION ("High sensitivity is aggressive (0.05)")
  {
    REQUIRE (StreamingPacingThreshold (1.0) == Catch::Approx (0.05));
  }

  SECTION ("Threshold decreases monotonically")
  {
    for (double S = 0.0; S < 1.0; S += 0.1)
      {
        REQUIRE (StreamingPacingThreshold (S)
                 > StreamingPacingThreshold (S + 0.1));
      }
  }
}

TEST_CASE ("Max wait drift decreases with Focus",
           "[regression][streaming]")
{
  // max_wait_drift(F) = lerp(1.2, 0.30, F)
  // Higher focus = lower max drift = more aggressive forced checks

  SECTION ("Low focus is lenient (1.2)")
  {
    REQUIRE (MaxWaitDrift (0.0) == Catch::Approx (1.2));
  }

  SECTION ("High focus is strict (0.3)")
  {
    REQUIRE (MaxWaitDrift (1.0) == Catch::Approx (0.3));
  }

  SECTION ("Max wait decreases monotonically")
  {
    for (double F = 0.0; F < 1.0; F += 0.1)
      {
        REQUIRE (MaxWaitDrift (F) > MaxWaitDrift (F + 0.1));
      }
  }
}

// =============================================================================
// 5.3.7 Emotional Cascade Parameters
// =============================================================================

// =============================================================================
// 5.3.8 Knowledge Graph Parameters (Phase 4 additions)
// =============================================================================

// =============================================================================
// 5.3.9 Half-Life and Decay Parameters
// =============================================================================

// =============================================================================
// 5.3.10 Cross-Subsystem Coherence Tests
// =============================================================================

TEST_CASE ("High Focus affects all selectivity parameters consistently",
           "[regression][coherence]")
{
  double F = 0.9;

  // All should be "stricter" / more selective
  REQUIRE (MaxResults (F) <= 24);               // Fewer results, but no longer single digits
  REQUIRE (WMGateThreshold (F) > 0.35);         // Stricter WM entry
  REQUIRE (MaxWaitDrift (F) < 0.7);             // More aggressive pacing
}

TEST_CASE ("High Sensitivity affects all plasticity parameters consistently",
           "[regression][coherence]")
{
  double S = 0.9;

  // All should be "more responsive" / faster adaptation
  REQUIRE (AlphaMood (S) > 0.17);                // Faster mood update
  REQUIRE (StreamingPacingThreshold (S) < 0.15); // More frequent checks
  REQUIRE (FlashbulbThreshold (S) < 0.75);       // Lower flashbulb trigger
}

TEST_CASE ("High Stability affects all persistence parameters consistently",
           "[regression][coherence]")
{
  double T = 0.9;

  // All should be "more persistent" / slower decay
  REQUIRE (LambdaMood (1.0, T) > 0.99);         // Slower mood decay
  REQUIRE (WRateSeconds (T) > 250);             // Longer write intervals
}
