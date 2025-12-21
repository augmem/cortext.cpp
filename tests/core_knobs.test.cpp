#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>

using namespace cortext::core;

TEST_CASE ("NCtx and TauM ranges", "[core][knobs]")
{
  REQUIRE (NCtx (0.0) == Catch::Approx (32.0));
  REQUIRE (NCtx (1.0) == Catch::Approx (256.0));

  REQUIRE (TauM (0.0) == Catch::Approx (10.0));
  REQUIRE (TauM (1.0) == Catch::Approx (200.0));
}

TEST_CASE ("AlphaU decreases with T", "[core][knobs]")
{
  double a0 = AlphaU (0.0);
  double a1 = AlphaU (1.0);
  REQUIRE (a0 > a1);
  REQUIRE (a0 <= Catch::Approx (0.70));
  REQUIRE (a0 >= Catch::Approx (0.10));
}

TEST_CASE ("AlphaF monotonicity and bounds", "[core][knobs]")
{
  double low = AlphaF (0.0, 0.0);
  double highF = AlphaF (1.0, 0.0);
  double highU = AlphaF (0.5, 1.0);

  REQUIRE (highF >= low);
  REQUIRE (highU >= low);

  REQUIRE (low >= Catch::Approx (0.05));
  REQUIRE (highU <= Catch::Approx (0.50));
}

TEST_CASE ("WMGateThreshold follows spec: lerp(0.1, 0.4, F)", "[core][knobs]")
{
  // gate_threshold = lerp(0.1, 0.4, F) per Algorithm 24
  // At F=0 (wide attention): permissive (0.1)
  // At F=1 (narrow attention): strict (0.4)
  REQUIRE (WMGateThreshold (0.0) == Catch::Approx (0.1));
  REQUIRE (WMGateThreshold (1.0) == Catch::Approx (0.4));
  REQUIRE (WMGateThreshold (0.5) == Catch::Approx (0.25));

  // Monotonic: higher F means higher (stricter) threshold
  REQUIRE (WMGateThreshold (0.3) < WMGateThreshold (0.7));
}

TEST_CASE ("WMBaseCapacity follows paper spec: range [2, 6]", "[core][knobs]")
{
  // base_capacity = round(lerp(5, 3, S) + lerp(-1, 1, F))
  // Per paper Section 8.1: capacity range [2, 6]

  // At S=0, F=0: lerp(5, 3, 0) + lerp(-1, 1, 0) = 5 + (-1) = 4
  REQUIRE (WMBaseCapacity (0.0, 0.0) == 4);

  // At S=1, F=1: lerp(5, 3, 1) + lerp(-1, 1, 1) = 3 + 1 = 4
  REQUIRE (WMBaseCapacity (1.0, 1.0) == 4);

  // At S=0, F=1: lerp(5, 3, 0) + lerp(-1, 1, 1) = 5 + 1 = 6 (max)
  REQUIRE (WMBaseCapacity (0.0, 1.0) == 6);

  // At S=1, F=0: lerp(5, 3, 1) + lerp(-1, 1, 0) = 3 + (-1) = 2 (min)
  REQUIRE (WMBaseCapacity (1.0, 0.0) == 2);

  // Verify range bounds across all extreme values
  REQUIRE (WMBaseCapacity (0.0, 1.0) == 6); // max
  REQUIRE (WMBaseCapacity (1.0, 0.0) == 2); // min

  // Mid-point: S=0.5, F=0.5: lerp(5, 3, 0.5) + lerp(-1, 1, 0.5) = 4 + 0 = 4
  REQUIRE (WMBaseCapacity (0.5, 0.5) == 4);
}

TEST_CASE ("StreamingPacingThreshold knob function", "[core][knobs]")
{
  // pacing_thresh(S) = lerp(0.5, 0.1, S)
  // Higher sensitivity = lower threshold = more frequent pacing checks

  // S=0: conservative (0.5)
  REQUIRE (StreamingPacingThreshold (0.0) == Catch::Approx (0.5));
  // S=1: aggressive (0.1)
  REQUIRE (StreamingPacingThreshold (1.0) == Catch::Approx (0.1));
  // S=0.5: midpoint (0.3)
  REQUIRE (StreamingPacingThreshold (0.5) == Catch::Approx (0.3));

  // Monotonic: higher S means lower threshold
  REQUIRE (StreamingPacingThreshold (0.3) > StreamingPacingThreshold (0.7));
}

TEST_CASE ("MaxWaitDrift knob function", "[core][knobs]")
{
  // max_wait_drift(F) = lerp(2.0, 0.5, F)
  // Higher focus = lower max drift = more aggressive forced checks

  // F=0: lenient (2.0)
  REQUIRE (MaxWaitDrift (0.0) == Catch::Approx (2.0));
  // F=1: strict (0.5)
  REQUIRE (MaxWaitDrift (1.0) == Catch::Approx (0.5));
  // F=0.5: midpoint (1.25)
  REQUIRE (MaxWaitDrift (0.5) == Catch::Approx (1.25));

  // Monotonic: higher F means lower max wait
  REQUIRE (MaxWaitDrift (0.3) > MaxWaitDrift (0.7));
}

TEST_CASE ("TauNovelty follows spec: lerp(0.10, 0.35, F) * (1 - 0.15S) * (1 + "
           "0.3T)",
           "[core][knobs]")
{
  // tau_novelty = lerp(0.10, 0.35, F) * (1 - 0.15S) * (1 + 0.3T)
  // Reference: algorithms.md Section 8.1

  // At F=0, S=0, T=0: 0.10 * 1.0 * 1.0 = 0.10
  REQUIRE (TauNovelty (0.0, 0.0, 0.0) == Catch::Approx (0.10));

  // At F=1, S=0, T=0: 0.35 * 1.0 * 1.0 = 0.35
  REQUIRE (TauNovelty (1.0, 0.0, 0.0) == Catch::Approx (0.35));

  // At F=0, S=1, T=0: 0.10 * 0.85 * 1.0 = 0.085
  REQUIRE (TauNovelty (0.0, 1.0, 0.0) == Catch::Approx (0.085));

  // At F=0, S=0, T=1: 0.10 * 1.0 * 1.3 = 0.13
  REQUIRE (TauNovelty (0.0, 0.0, 1.0) == Catch::Approx (0.13));

  // At F=1, S=1, T=1: 0.35 * 0.85 * 1.3 = 0.38675
  REQUIRE (TauNovelty (1.0, 1.0, 1.0) == Catch::Approx (0.38675));

  // Mid-point: F=0.5, S=0.5, T=0.5
  // lerp(0.10, 0.35, 0.5) = 0.225
  // (1 - 0.15*0.5) = 0.925
  // (1 + 0.3*0.5) = 1.15
  // 0.225 * 0.925 * 1.15 = 0.23934375
  REQUIRE (TauNovelty (0.5, 0.5, 0.5) == Catch::Approx (0.23934375));
}

TEST_CASE ("RetrievalThreshold follows spec: lerp(0.25, 0.60, F)",
           "[core][knobs]")
{
  // retrieval_thresh(F) = lerp(0.25, 0.60, F)
  // Reference: algorithms.md Section 8.1

  // F=0: permissive (0.25)
  REQUIRE (RetrievalThreshold (0.0) == Catch::Approx (0.25));

  // F=1: strict (0.60)
  REQUIRE (RetrievalThreshold (1.0) == Catch::Approx (0.60));

  // F=0.5: midpoint (0.425)
  REQUIRE (RetrievalThreshold (0.5) == Catch::Approx (0.425));

  // Monotonic: higher F means higher (stricter) threshold
  REQUIRE (RetrievalThreshold (0.3) < RetrievalThreshold (0.7));
}

TEST_CASE ("InterruptCandidateCount follows spec: round(lerp(10, 6, F))",
           "[core][knobs]")
{
  // K = round(lerp(10, 6, F))
  // Reference: algorithms.md Section 8.3

  // F=0: max candidates (10)
  REQUIRE (InterruptCandidateCount (0.0) == 10);

  // F=1: min candidates (6)
  REQUIRE (InterruptCandidateCount (1.0) == 6);

  // F=0.5: midpoint (8)
  REQUIRE (InterruptCandidateCount (0.5) == 8);

  // Monotonic: higher F means fewer candidates
  REQUIRE (InterruptCandidateCount (0.3) >= InterruptCandidateCount (0.7));
}

TEST_CASE (
    "ConsolidationRate follows spec: (60/interval) * (0.3+0.7T) * (1-0.5S)",
    "[core][knobs]")
{
  // rate_consolidate = (60 / max(interval, 1)) × (0.3 + 0.7T) × (1 − 0.5S)
  // Reference: algorithms.md Section 7.1

  // At T=0, S=0: interval=300, rate = (60/300) * 0.3 * 1.0 = 0.06
  REQUIRE (ConsolidationRate (0.0, 0.0) == Catch::Approx (0.06));

  // At T=1, S=0: interval=3600, rate = (60/3600) * 1.0 * 1.0 ≈ 0.0166667
  REQUIRE (ConsolidationRate (1.0, 0.0) == Catch::Approx (60.0 / 3600.0));

  // At T=0, S=1: interval=300, rate = (60/300) * 0.3 * 0.5 = 0.03
  REQUIRE (ConsolidationRate (0.0, 1.0) == Catch::Approx (0.03));

  // Higher S decreases rate factor (1-0.5S)
  REQUIRE (ConsolidationRate (0.0, 0.5) < ConsolidationRate (0.0, 0.0));

  // Higher T: interval increases faster than rate factor, so overall rate decreases
  // (interval grows 300→3600 while factor grows 0.3→1.0, net effect is lower rate)
  REQUIRE (ConsolidationRate (0.5, 0.0) < ConsolidationRate (0.0, 0.0));
}
