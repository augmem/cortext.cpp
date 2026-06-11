// Formula Validation Tests for Phase 5
// Tests all knob-derived functions against paper specifications
// Reference: docs/algorithms.md Section 0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cortext/core/knobs.hpp>

using namespace cortext::core;

namespace
{

double
FocusExpected (double lo, double hi, double F)
{
  return Lerp (lo, hi, FocusBias (F));
}

double
SensitivityExpected (double lo, double hi, double S)
{
  return Lerp (lo, hi, SensitivityBias (S));
}

int
FocusExpectedRound (double lo, double hi, double F)
{
  return static_cast<int> (std::round (FocusExpected (lo, hi, F)));
}

int
SensitivityExpectedRound (double lo, double hi, double S)
{
  return static_cast<int> (std::round (SensitivityExpected (lo, hi, S)));
}

} // namespace

// =============================================================================
// 5.1.1 Context Windows & Temporal Scales (6 functions)
// =============================================================================

TEST_CASE ("WScore follows spec: lerp(20, 120, T)", "[formula][knobs][temporal]")
{
  // WScore = round(lerp(20, 120, T))
  REQUIRE (WScore (0.0) == 20);
  REQUIRE (WScore (1.0) == 120);
  REQUIRE (WScore (0.5) == 70);

  // Monotonic: higher T means higher score
  REQUIRE (WScore (0.3) < WScore (0.7));
}

TEST_CASE ("KNeighbors follows spec: lerp(8, 32, T)",
           "[formula][knobs][temporal]")
{
  // KNeighbors = round(lerp(8, 32, T))
  REQUIRE (KNeighbors (0.0) == 8);
  REQUIRE (KNeighbors (1.0) == 32);
  REQUIRE (KNeighbors (0.5) == 20);

  // Monotonic
  REQUIRE (KNeighbors (0.3) < KNeighbors (0.7));
}

TEST_CASE ("KCtx follows spec: lerp(3, 10, T)",
           "[formula][knobs][temporal]")
{
  // KCtx = round(lerp(3, 10, T))
  REQUIRE (KCtx (0.0) == 3);
  REQUIRE (KCtx (1.0) == 10);
  REQUIRE (KCtx (0.5) == 7);

  // Monotonic
  REQUIRE (KCtx (0.3) < KCtx (0.7));
}

TEST_CASE ("RLSWindowN follows spec: lerp(64, 512, T)",
           "[formula][knobs][temporal]")
{
  // RLSWindowN = round(lerp(64, 512, T))
  REQUIRE (RLSWindowN (0.0) == 64);
  REQUIRE (RLSWindowN (1.0) == 512);
  REQUIRE (RLSWindowN (0.5) == 288);

  // Monotonic
  REQUIRE (RLSWindowN (0.3) < RLSWindowN (0.7));
}

TEST_CASE ("MaxResults follows spec: lerp(96, 8, FocusBias(F))",
           "[formula][knobs][temporal]")
{
  // MaxResults = round(lerp(96, 8, FocusBias(F)))
  // Higher focus = fewer results (more selective)
  REQUIRE (MaxResults (0.0) == 96);
  REQUIRE (MaxResults (1.0) == 8);
  REQUIRE (MaxResults (0.5) == FocusExpectedRound (96.0, 8.0, 0.5));

  // Monotonic: higher F means fewer results
  REQUIRE (MaxResults (0.3) > MaxResults (0.7));
}

TEST_CASE ("AdjacentWindow follows spec: lerp(6, 1, FocusBias(F))",
           "[formula][knobs][temporal]")
{
  // AdjacentWindow = round(lerp(6, 1, FocusBias(F)))
  REQUIRE (AdjacentWindow (0.0) == 6);
  REQUIRE (AdjacentWindow (1.0) == 1);
  REQUIRE (AdjacentWindow (0.5) == FocusExpectedRound (6.0, 1.0, 0.5));

  // Monotonic: higher F means smaller window
  REQUIRE (AdjacentWindow (0.3) > AdjacentWindow (0.7));
}

TEST_CASE ("MaxWaitTokens follows spec: lerp(128, 16, F)",
           "[formula][knobs][temporal]")
{
  // MaxWaitTokens = round(lerp(128, 16, F))
  REQUIRE (MaxWaitTokens (0.0) == 128);
  REQUIRE (MaxWaitTokens (1.0) == 16);
  REQUIRE (MaxWaitTokens (0.5) == FocusExpectedRound (128.0, 16.0, 0.5));

  // Monotonic: higher F means fewer wait tokens
  REQUIRE (MaxWaitTokens (0.3) > MaxWaitTokens (0.7));
}

// =============================================================================
// 5.1.2 Rate & Interval Parameters (6 functions)
// =============================================================================

TEST_CASE ("CheckIntervalTokens follows spec: lerp(64, 8, S)",
           "[formula][knobs][rate]")
{
  // CheckIntervalTokens = round(lerp(64, 8, S))
  REQUIRE (CheckIntervalTokens (0.0) == 64);
  REQUIRE (CheckIntervalTokens (1.0) == 8);
  REQUIRE (CheckIntervalTokens (0.5)
           == SensitivityExpectedRound (64.0, 8.0, 0.5));

  // Monotonic: higher S means more frequent checks (fewer tokens)
  REQUIRE (CheckIntervalTokens (0.3) > CheckIntervalTokens (0.7));
}

TEST_CASE ("WRateSeconds follows spec: lerp(60, 300, T)",
           "[formula][knobs][rate]")
{
  // WRateSeconds = round(lerp(60, 300, T))
  REQUIRE (WRateSeconds (0.0) == 60);
  REQUIRE (WRateSeconds (1.0) == 300);
  REQUIRE (WRateSeconds (0.5) == 180);

  // Monotonic
  REQUIRE (WRateSeconds (0.3) < WRateSeconds (0.7));
}

TEST_CASE ("ConsolidationIntervalSeconds follows spec: lerp(300, 3600, T)",
           "[formula][knobs][rate]")
{
  // ConsolidationIntervalSeconds = round(lerp(300, 3600, T))
  REQUIRE (ConsolidationIntervalSeconds (0.0) == 300);
  REQUIRE (ConsolidationIntervalSeconds (1.0) == 3600);
  REQUIRE (ConsolidationIntervalSeconds (0.5) == 1950);

  // Monotonic
  REQUIRE (ConsolidationIntervalSeconds (0.3)
           < ConsolidationIntervalSeconds (0.7));
}

TEST_CASE ("IdleRequiredSeconds follows spec: 0.25 * WRateSeconds(T)",
           "[formula][knobs][rate]")
{
  // IdleRequiredSeconds = round(0.25 * WRateSeconds(T))
  REQUIRE (IdleRequiredSeconds (0.0)
           == static_cast<int> (std::round (0.25 * 60)));
  REQUIRE (IdleRequiredSeconds (1.0)
           == static_cast<int> (std::round (0.25 * 300)));

  // Derived from WRateSeconds
  for (double T : { 0.0, 0.25, 0.5, 0.75, 1.0 })
    {
      int expected
          = static_cast<int> (std::round (0.25 * WRateSeconds (T)));
      REQUIRE (IdleRequiredSeconds (T) == expected);
    }
}

TEST_CASE ("ExtractionBatchSize follows spec: lerp(8, 32, T)",
           "[formula][knobs][rate]")
{
  // ExtractionBatchSize = round(lerp(8, 32, T))
  REQUIRE (ExtractionBatchSize (0.0) == 8);
  REQUIRE (ExtractionBatchSize (1.0) == 32);
  REQUIRE (ExtractionBatchSize (0.5) == 20);

  // Monotonic
  REQUIRE (ExtractionBatchSize (0.3) < ExtractionBatchSize (0.7));
}

TEST_CASE ("MaxExtractionsPerCycle follows spec: lerp(20, 5, T)",
           "[formula][knobs][rate]")
{
  // MaxExtractionsPerCycle = round(lerp(20, 5, T))
  // Higher stability = fewer extractions per cycle
  REQUIRE (MaxExtractionsPerCycle (0.0) == 20);
  REQUIRE (MaxExtractionsPerCycle (1.0) == 5);
  REQUIRE (MaxExtractionsPerCycle (0.5) == 13); // round(12.5) = 13 (rounds up)

  // Monotonic: higher T means fewer extractions
  REQUIRE (MaxExtractionsPerCycle (0.3) > MaxExtractionsPerCycle (0.7));
}

// =============================================================================
// 5.1.3 Learning Rate Functions (3 functions)
// =============================================================================

TEST_CASE ("AlphaT follows spec: α_min_T + (1−T)×α_span_T×u", "[formula][knobs][alpha]")
{
  // AlphaT = 0.02 + (1-T)*0.18*u_t

  // At u_t=0, reduces to α_min_T
  REQUIRE (AlphaT (0.0, 0.0) == Catch::Approx (0.02));
  REQUIRE (AlphaT (1.0, 0.0) == Catch::Approx (0.02));

  // At u_t=1, T=0: 0.02 + 1*0.18*1 = 0.20
  REQUIRE (AlphaT (0.0, 1.0) == Catch::Approx (0.20));

  // At u_t=1, T=1: 0.02
  REQUIRE (AlphaT (1.0, 1.0) == Catch::Approx (0.02));

  // Verify higher T (more stable) means lower alpha at high uncertainty
  REQUIRE (AlphaT (0.2, 0.8) > AlphaT (0.8, 0.8));
}

TEST_CASE ("AlphaS follows spec: α_min_S + S×α_span_S×u", "[formula][knobs][alpha]")
{
  // AlphaS = 0.05 + S*0.35*u_t

  // At u_t=0, reduces to α_min_S
  REQUIRE (AlphaS (0.0, 0.0) == Catch::Approx (0.05));
  REQUIRE (AlphaS (1.0, 0.0) == Catch::Approx (0.05));

  // At u_t=1, S=1: 0.40
  REQUIRE (AlphaS (1.0, 1.0) == Catch::Approx (0.40));

  // At u_t=1, S=0: 0.05
  REQUIRE (AlphaS (0.0, 1.0) == Catch::Approx (0.05));

  // Higher S means higher alpha (faster adaptation)
  REQUIRE (AlphaS (0.8, 0.8) > AlphaS (0.2, 0.8));
}

TEST_CASE ("AlphaMood follows spec: lerp(0.01, 0.20, S)",
           "[formula][knobs][alpha]")
{
  // AlphaMood = lerp(0.01, 0.20, S)
  REQUIRE (AlphaMood (0.0) == Catch::Approx (0.01));
  REQUIRE (AlphaMood (1.0) == Catch::Approx (0.20));
  REQUIRE (AlphaMood (0.5)
           == Catch::Approx (SensitivityExpected (0.01, 0.20, 0.5)));

  // Monotonic: higher S means faster mood reactivity
  REQUIRE (AlphaMood (0.3) < AlphaMood (0.7));
}

// =============================================================================
// 5.1.4 Maturity & Threshold Functions (5 functions)
// =============================================================================

TEST_CASE ("ComputeMaturity follows spec: 1 - exp(-count/TauM(T))",
           "[formula][knobs][maturity]")
{
  // ComputeMaturity = 1 - exp(-count / TauM(T))

  // At count=0, maturity should be 0
  REQUIRE (ComputeMaturity (0, 0.0) == Catch::Approx (0.0));
  REQUIRE (ComputeMaturity (0, 1.0) == Catch::Approx (0.0));

  // At high count, maturity approaches 1
  REQUIRE (ComputeMaturity (10000, 0.0) == Catch::Approx (1.0).epsilon (0.01));
  REQUIRE (ComputeMaturity (10000, 1.0) == Catch::Approx (1.0).epsilon (0.01));

  // With T=0 (TauM=10), count=10: 1 - exp(-1) ≈ 0.632
  REQUIRE (ComputeMaturity (10, 0.0) == Catch::Approx (0.632).epsilon (0.01));

  // With T=1 (TauM=200), count=200: 1 - exp(-1) ≈ 0.632
  REQUIRE (ComputeMaturity (200, 1.0) == Catch::Approx (0.632).epsilon (0.01));
}

TEST_CASE ("TMin follows spec: lerp(0.01, 0.05, maturity)",
           "[formula][knobs][maturity]")
{
  // TMin = lerp(0.01, 0.05, maturity)

  // At low maturity, TMin is near 0.01
  REQUIRE (TMin (0, 0.5) == Catch::Approx (0.01));

  // At high maturity (count >> tau_m), TMin approaches 0.05
  REQUIRE (TMin (10000, 0.5) == Catch::Approx (0.05).epsilon (0.01));

  // Verify explicit formula
  for (int count : { 1, 10, 100, 1000 })
    {
      double T = 0.5;
      double maturity = ComputeMaturity (count, T);
      double expected = 0.01 + (0.05 - 0.01) * maturity;
      REQUIRE (TMin (count, T) == Catch::Approx (expected));
    }
}

TEST_CASE ("TMax follows spec: lerp(0.99, 0.95, maturity)",
           "[formula][knobs][maturity]")
{
  // TMax = lerp(0.99, 0.95, maturity)

  // At low maturity, TMax is near 0.99
  REQUIRE (TMax (0, 0.5) == Catch::Approx (0.99));

  // At high maturity, TMax approaches 0.95
  REQUIRE (TMax (10000, 0.5) == Catch::Approx (0.95).epsilon (0.01));

  // TMin < TMax always
  for (int count : { 0, 1, 10, 100, 1000, 10000 })
    {
      for (double T : { 0.0, 0.5, 1.0 })
        {
          REQUIRE (TMin (count, T) < TMax (count, T));
        }
    }
}

TEST_CASE ("MaxDeltaTPerMin follows spec: lerp(0.30, 0.10, maturity)",
           "[formula][knobs][maturity]")
{
  // MaxDeltaTPerMin = lerp(0.30, 0.10, maturity)

  // At low maturity, higher delta allowed (0.30)
  REQUIRE (MaxDeltaTPerMin (0, 0.5) == Catch::Approx (0.30));

  // At high maturity, lower delta allowed (0.10)
  REQUIRE (MaxDeltaTPerMin (10000, 0.5) == Catch::Approx (0.10).epsilon (0.01));

  // Monotonic in count: more mature = less delta
  REQUIRE (MaxDeltaTPerMin (10, 0.5) > MaxDeltaTPerMin (1000, 0.5));
}

TEST_CASE ("ConsolidationThresholdCount follows spec: NCtx(T) * WScore(T)",
           "[formula][knobs][maturity]")
{
  // ConsolidationThresholdCount = NCtx(T) * WScore(T)

  // At T=0: NCtx(0)=32, WScore(0)=20 => 640
  REQUIRE (ConsolidationThresholdCount (0.0) == 32LL * 20LL);

  // At T=1: NCtx(1)=256, WScore(1)=120 => 30720
  REQUIRE (ConsolidationThresholdCount (1.0) == 256LL * 120LL);

  // Verify formula for intermediate values
  for (double T : { 0.0, 0.25, 0.5, 0.75, 1.0 })
    {
      long long n = static_cast<long long> (NCtx (T));
      long long w = static_cast<long long> (WScore (T));
      REQUIRE (ConsolidationThresholdCount (T) == n * w);
    }
}

// =============================================================================
// 5.1.5 Half-Life & Decay Functions (9 functions)
// =============================================================================

TEST_CASE ("BaseHalfLifePrior follows spec: exp(ln(120) + T*ln(360))",
           "[formula][knobs][decay]")
{
  // BaseHalfLifePrior = exp(ln(120) + T * ln(43200/120))
  // At T=0: exp(ln(120)) = 120
  REQUIRE (BaseHalfLifePrior (0.0) == Catch::Approx (120.0));

  // At T=1: exp(ln(120) + ln(360)) = exp(ln(43200)) = 43200
  REQUIRE (BaseHalfLifePrior (1.0) == Catch::Approx (43200.0));

  // Monotonic
  REQUIRE (BaseHalfLifePrior (0.3) < BaseHalfLifePrior (0.7));

  // Range check
  REQUIRE (BaseHalfLifePrior (0.5) >= 120.0);
  REQUIRE (BaseHalfLifePrior (0.5) <= 43200.0);
}

TEST_CASE ("ClampHalfLife follows spec: clamp(value, 120, 43200)",
           "[formula][knobs][decay]")
{
  // ClampHalfLife = clamp(value, 120, 43200)
  REQUIRE (ClampHalfLife (100.0) == 120.0);
  REQUIRE (ClampHalfLife (120.0) == 120.0);
  REQUIRE (ClampHalfLife (1000.0) == 1000.0);
  REQUIRE (ClampHalfLife (43200.0) == 43200.0);
  REQUIRE (ClampHalfLife (50000.0) == 43200.0);
}

TEST_CASE ("TauLabile follows spec: lerp(30, 300, T)",
           "[formula][knobs][decay]")
{
  // TauLabile = lerp(30, 300, T)
  REQUIRE (TauLabile (0.0) == Catch::Approx (30.0));
  REQUIRE (TauLabile (1.0) == Catch::Approx (300.0));
  REQUIRE (TauLabile (0.5) == Catch::Approx (165.0));

  // Monotonic
  REQUIRE (TauLabile (0.3) < TauLabile (0.7));
}

TEST_CASE ("ReconsolidationGain follows spec: lerp(0.2, 0.02, T)",
           "[formula][knobs][decay]")
{
  // ReconsolidationGain = lerp(0.2, 0.02, T)
  REQUIRE (ReconsolidationGain (0.0) == Catch::Approx (0.2));
  REQUIRE (ReconsolidationGain (1.0) == Catch::Approx (0.02));
  REQUIRE (ReconsolidationGain (0.5) == Catch::Approx (0.11));

  // Monotonic: higher T means lower gain
  REQUIRE (ReconsolidationGain (0.3) > ReconsolidationGain (0.7));
}

TEST_CASE ("RippleDecay follows spec: lerp(0.5, 0.1, T)",
           "[formula][knobs][decay]")
{
  // RippleDecay = lerp(0.5, 0.1, T)
  REQUIRE (RippleDecay (0.0) == Catch::Approx (0.5));
  REQUIRE (RippleDecay (1.0) == Catch::Approx (0.1));
  REQUIRE (RippleDecay (0.5) == Catch::Approx (0.3));

  // Monotonic
  REQUIRE (RippleDecay (0.3) > RippleDecay (0.7));
}

TEST_CASE ("ReinforcementDecay follows spec: lerp(0.9, 0.99, T)",
           "[formula][knobs][decay]")
{
  // ReinforcementDecay = lerp(0.9, 0.99, T)
  REQUIRE (ReinforcementDecay (0.0) == Catch::Approx (0.9));
  REQUIRE (ReinforcementDecay (1.0) == Catch::Approx (0.99));
  REQUIRE (ReinforcementDecay (0.5) == Catch::Approx (0.945));

  // Monotonic
  REQUIRE (ReinforcementDecay (0.3) < ReinforcementDecay (0.7));
}

TEST_CASE ("Reinforcement co-retrieval update is knob-derived",
           "[formula][knobs][reinforcement]")
{
  const double low = ReinforcementCoRetrievalStep (1.0, 0.0, 0.0);
  const double mid = ReinforcementCoRetrievalStep (0.5, 0.5, 0.5);
  const double high = ReinforcementCoRetrievalStep (0.0, 1.0, 1.0);

  REQUIRE (low > 0.0);
  REQUIRE (mid > low);
  REQUIRE (high > mid);
  REQUIRE (high <= 0.08);

  REQUIRE (ReinforcementUnselectedScale (0.5, 0.5, 0.5) < 0.40);
  REQUIRE (ReinforcementUnselectedScale (0.0, 1.0, 1.0)
           > ReinforcementUnselectedScale (1.0, 0.0, 0.0));

  REQUIRE (ReinforcementFanoutDamping (0.5, 0.5, 0.5, 2)
           == Catch::Approx (1.0));
  REQUIRE (ReinforcementFanoutDamping (0.5, 0.5, 0.5, 20) < 1.0);
  REQUIRE (ReinforcementFanoutDamping (1.0, 0.5, 0.5, 20)
           < ReinforcementFanoutDamping (0.0, 0.5, 0.5, 20));

  const double prune_mid = ReinforcementPruneThreshold (0.5, 0.5, 0.5);
  REQUIRE (prune_mid > 0.08);
  REQUIRE (prune_mid < 0.13);
  REQUIRE (ReinforcementPruneThreshold (1.0, 0.5, 0.5)
           > ReinforcementPruneThreshold (0.0, 0.5, 0.5));
  REQUIRE (ReinforcementPruneThreshold (0.5, 1.0, 0.5)
           > ReinforcementPruneThreshold (0.5, 0.0, 0.5));
  REQUIRE (ReinforcementPruneThreshold (0.5, 0.5, 1.0)
           < ReinforcementPruneThreshold (0.5, 0.5, 0.0));
}

TEST_CASE ("LambdaMood follows spec: exp(-ln2 * Δt / half_life)",
           "[formula][knobs][decay]")
{
  // half_life_mood(T) = lerp(30, 600, T)
  // λ_mood(Δt, T) = exp(-ln(2) * Δt / half_life_mood(T))
  REQUIRE (LambdaMood (0.0, 0.0) == Catch::Approx (1.0));
  REQUIRE (LambdaMood (30.0, 0.0) == Catch::Approx (0.5));
  REQUIRE (LambdaMood (600.0, 1.0) == Catch::Approx (0.5));

  // Monotonic: higher T => longer half-life => higher lambda for same Δt
  REQUIRE (LambdaMood (30.0, 0.3) < LambdaMood (30.0, 0.7));
}

TEST_CASE ("ConfidenceDecayRate follows spec: lerp(0.01, 0.1, 1-T)",
           "[formula][knobs][decay]")
{
  // ConfidenceDecayRate = lerp(0.01, 0.1, 1-T)
  // At T=0: lerp(0.01, 0.1, 1) = 0.1
  REQUIRE (ConfidenceDecayRate (0.0) == Catch::Approx (0.1));
  // At T=1: lerp(0.01, 0.1, 0) = 0.01
  REQUIRE (ConfidenceDecayRate (1.0) == Catch::Approx (0.01));
  // At T=0.5: lerp(0.01, 0.1, 0.5) = 0.055
  REQUIRE (ConfidenceDecayRate (0.5) == Catch::Approx (0.055));

  // Monotonic: higher T (more stable) means slower decay
  REQUIRE (ConfidenceDecayRate (0.3) > ConfidenceDecayRate (0.7));
}

TEST_CASE ("EmotionalHalfLifeBonus follows spec",
           "[formula][knobs][decay]")
{
  // EmotionalHalfLifeBonus = exp(lerp(0, ln(3), S)) * (1 + emotion_intensity)

  // At S=0, intensity=0: exp(0) * 1 = 1
  REQUIRE (EmotionalHalfLifeBonus (0.0, 0.0) == Catch::Approx (1.0));

  // At S=1, intensity=0: exp(ln(3)) * 1 = 3
  REQUIRE (EmotionalHalfLifeBonus (1.0, 0.0) == Catch::Approx (3.0));

  // At S=0, intensity=1: exp(0) * 2 = 2
  REQUIRE (EmotionalHalfLifeBonus (0.0, 1.0) == Catch::Approx (2.0));

  // At S=1, intensity=1: exp(ln(3)) * 2 = 6
  REQUIRE (EmotionalHalfLifeBonus (1.0, 1.0) == Catch::Approx (6.0));

  // Higher S means higher bonus
  REQUIRE (EmotionalHalfLifeBonus (0.8, 0.5)
           > EmotionalHalfLifeBonus (0.2, 0.5));
}

// =============================================================================
// 5.1.6 Working Memory Functions
// =============================================================================

TEST_CASE ("WMMaintenanceCostPerSlot keeps the full-WM budget "
           "capacity-invariant",
           "[formula][knobs][working_memory]")
{
  // Per-slot maintenance shares a fixed total budget: cost-per-slot x
  // capacity = lerp(0.05, 0.15, S) x 7 (the neutral capacity), so capacity
  // changes never change gate strictness.
  for (const double S : { 0.0, 0.3, 0.5, 0.7, 1.0 })
    {
      for (const double F : { 0.0, 0.5, 1.0 })
        {
          const double total = WMMaintenanceCostPerSlot (S, F)
                               * static_cast<double> (WMBaseCapacity (S, F));
          REQUIRE (total
                   == Catch::Approx (SensitivityExpected (0.05, 0.15, S)
                                     * 7.0));
        }
    }

  // At neutral knobs (capacity exactly 7) the per-slot cost matches the
  // original spec value lerp(0.05, 0.15, S).
  REQUIRE (WMBaseCapacity (0.5, 0.5) == 7);
  REQUIRE (WMMaintenanceCostPerSlot (0.5, 0.5)
           == Catch::Approx (SensitivityExpected (0.05, 0.15, 0.5)));

  // Monotonic in S at fixed capacity.
  REQUIRE (WMMaintenanceCostPerSlot (0.3, 0.5)
           < WMMaintenanceCostPerSlot (0.7, 0.5));
}

TEST_CASE ("WMStrengthFloor derives from F/S/T",
           "[formula][knobs][working_memory]")
{
  REQUIRE (WMStrengthFloor (0.5, 0.5, 0.5) > 0.0);
  REQUIRE (WMStrengthFloor (0.5, 0.5, 1.0)
           > WMStrengthFloor (0.5, 0.5, 0.0));
  REQUIRE (WMStrengthFloor (1.0, 0.5, 0.5)
           < WMStrengthFloor (0.0, 0.5, 0.5));
  REQUIRE (WMStrengthFloor (0.5, 1.0, 0.5)
           < WMStrengthFloor (0.5, 0.0, 0.5));
}

TEST_CASE ("WMChunkingThreshold follows spec: lerp(0.7, 0.9, F)",
           "[formula][knobs][working_memory]")
{
  // WMChunkingThreshold = lerp(0.7, 0.9, F)
  REQUIRE (WMChunkingThreshold (0.0) == Catch::Approx (0.7));
  REQUIRE (WMChunkingThreshold (1.0) == Catch::Approx (0.9));
  REQUIRE (WMChunkingThreshold (0.5)
           == Catch::Approx (FocusExpected (0.7, 0.9, 0.5)));

  // Monotonic
  REQUIRE (WMChunkingThreshold (0.3) < WMChunkingThreshold (0.7));
}

TEST_CASE ("WMComplexityScale follows spec: lerp(0.5, 1.5, S)",
           "[formula][knobs][working_memory]")
{
  // WMComplexityScale = lerp(0.5, 1.5, S)
  REQUIRE (WMComplexityScale (0.0) == Catch::Approx (0.5));
  REQUIRE (WMComplexityScale (1.0) == Catch::Approx (1.5));
  REQUIRE (WMComplexityScale (0.5)
           == Catch::Approx (SensitivityExpected (0.5, 1.5, 0.5)));

  // Monotonic
  REQUIRE (WMComplexityScale (0.3) < WMComplexityScale (0.7));
}

TEST_CASE ("WM rehearsal and eviction pressure derive from F/S/T",
           "[formula][knobs][working_memory]")
{
  REQUIRE (WMRehearsalBaseDelta (0.0, 0.5, 0.5)
           > WMRehearsalBaseDelta (1.0, 0.5, 0.5));
  REQUIRE (WMRehearsalBaseDelta (0.5, 1.0, 0.5)
           > WMRehearsalBaseDelta (0.5, 0.0, 0.5));
  REQUIRE (WMRehearsalBaseDelta (0.5, 0.5, 1.0)
           > WMRehearsalBaseDelta (0.5, 0.5, 0.0));

  REQUIRE (WMRecencyTauSeconds (0.5, 0.5, 1.0)
           > WMRecencyTauSeconds (0.5, 0.5, 0.0));
  REQUIRE (WMRecencyTauSeconds (1.0, 0.5, 0.5)
           < WMRecencyTauSeconds (0.0, 0.5, 0.5));
  REQUIRE (WMRecencyTauSeconds (0.5, 1.0, 0.5)
           < WMRecencyTauSeconds (0.5, 0.0, 0.5));

  REQUIRE (WMCapacityPressureExponent (1.0, 0.5, 0.5)
           > WMCapacityPressureExponent (0.0, 0.5, 0.5));
  REQUIRE (WMCapacityPressureExponent (0.5, 0.5, 1.0)
           > WMCapacityPressureExponent (0.5, 0.5, 0.0));
}

TEST_CASE ("StrategySwitchLatencyMs follows spec: lerp(500, 100, S)",
           "[formula][knobs][working_memory]")
{
  // StrategySwitchLatencyMs = round(lerp(500, 100, S))
  REQUIRE (StrategySwitchLatencyMs (0.0) == 500);
  REQUIRE (StrategySwitchLatencyMs (1.0) == 100);
  REQUIRE (StrategySwitchLatencyMs (0.5)
           == SensitivityExpectedRound (500.0, 100.0, 0.5));

  // Monotonic: higher S means faster switching (lower latency)
  REQUIRE (StrategySwitchLatencyMs (0.3) > StrategySwitchLatencyMs (0.7));
}

// =============================================================================
// 5.1.7 Clustering & Consolidation Functions (10 functions)
// =============================================================================

TEST_CASE ("BaseRatePrior follows spec: lerp(0.2, 5.0, S)",
           "[formula][knobs][consolidation]")
{
  // BaseRatePrior = lerp(0.2, 5.0, S)
  REQUIRE (BaseRatePrior (0.0) == Catch::Approx (0.2));
  REQUIRE (BaseRatePrior (1.0) == Catch::Approx (5.0));
  REQUIRE (BaseRatePrior (0.5)
           == Catch::Approx (SensitivityExpected (0.2, 5.0, 0.5)));

  // Monotonic
  REQUIRE (BaseRatePrior (0.3) < BaseRatePrior (0.7));
}

TEST_CASE ("BaseBandPrior follows spec: lerp(0.02, 0.25, T)",
           "[formula][knobs][consolidation]")
{
  // BaseBandPrior = lerp(0.02, 0.25, T)
  REQUIRE (BaseBandPrior (0.0) == Catch::Approx (0.02));
  REQUIRE (BaseBandPrior (1.0) == Catch::Approx (0.25));
  REQUIRE (BaseBandPrior (0.5) == Catch::Approx (0.135));

  // Monotonic
  REQUIRE (BaseBandPrior (0.3) < BaseBandPrior (0.7));
}

TEST_CASE ("MergeThreshold follows spec: lerp(0.85, 0.95, F)",
           "[formula][knobs][consolidation]")
{
  // MergeThreshold = lerp(0.85, 0.95, F)
  REQUIRE (MergeThreshold (0.0) == Catch::Approx (0.85));
  REQUIRE (MergeThreshold (1.0) == Catch::Approx (0.95));
  REQUIRE (MergeThreshold (0.5)
           == Catch::Approx (FocusExpected (0.85, 0.95, 0.5)));

  // Monotonic: higher focus = stricter merging
  REQUIRE (MergeThreshold (0.3) < MergeThreshold (0.7));
}

TEST_CASE ("MinClusterSize follows spec: lerp(3, 10, F)",
           "[formula][knobs][consolidation]")
{
  // MinClusterSize = round(lerp(3, 10, F))
  REQUIRE (MinClusterSize (0.0) == 3);
  REQUIRE (MinClusterSize (1.0) == 10);
  REQUIRE (MinClusterSize (0.5) == FocusExpectedRound (3.0, 10.0, 0.5));

  // Monotonic
  REQUIRE (MinClusterSize (0.3) < MinClusterSize (0.7));
}

TEST_CASE ("MinClusterSizeForExtraction follows spec: lerp(3, 10, F)",
           "[formula][knobs][consolidation]")
{
  // Same formula as MinClusterSize
  REQUIRE (MinClusterSizeForExtraction (0.0) == 3);
  REQUIRE (MinClusterSizeForExtraction (1.0) == 10);

  // Should equal MinClusterSize
  for (double F : { 0.0, 0.25, 0.5, 0.75, 1.0 })
    {
      REQUIRE (MinClusterSizeForExtraction (F) == MinClusterSize (F));
    }
}

TEST_CASE ("LabelFrequencyThreshold follows spec: lerp(5, 15, T)",
           "[formula][knobs][consolidation]")
{
  // LabelFrequencyThreshold = round(lerp(5, 15, T))
  REQUIRE (LabelFrequencyThreshold (0.0) == 5);
  REQUIRE (LabelFrequencyThreshold (1.0) == 15);
  REQUIRE (LabelFrequencyThreshold (0.5) == 10);

  // Monotonic
  REQUIRE (LabelFrequencyThreshold (0.3) < LabelFrequencyThreshold (0.7));
}

TEST_CASE ("ExtractionIntervalSeconds follows spec: lerp(300, 3600, T)",
           "[formula][knobs][consolidation]")
{
  // ExtractionIntervalSeconds = round(lerp(300, 3600, T))
  REQUIRE (ExtractionIntervalSeconds (0.0) == 300);
  REQUIRE (ExtractionIntervalSeconds (1.0) == 3600);
  REQUIRE (ExtractionIntervalSeconds (0.5) == 1950);

  // Monotonic
  REQUIRE (ExtractionIntervalSeconds (0.3) < ExtractionIntervalSeconds (0.7));
}

TEST_CASE ("WRet follows spec: lerp(10, 50, T)",
           "[formula][knobs][consolidation]")
{
  // WRet = round(lerp(10, 50, T))
  // Spec (§2.3.2, line 339): w_ret(T) = round(lerp(10, 50, T))
  REQUIRE (WRet (0.0) == 10);
  REQUIRE (WRet (1.0) == 50);
  REQUIRE (WRet (0.5) == 30);

  // Monotonic
  REQUIRE (WRet (0.3) < WRet (0.7));
}

TEST_CASE ("PeripheryCutoff follows spec: lerp(0.03, 0.20, T)",
           "[formula][knobs][consolidation]")
{
  // PeripheryCutoff = lerp(0.03, 0.20, T)
  REQUIRE (PeripheryCutoff (0.0) == Catch::Approx (0.03));
  REQUIRE (PeripheryCutoff (1.0) == Catch::Approx (0.20));
  REQUIRE (PeripheryCutoff (0.5) == Catch::Approx (0.115));

  // Monotonic
  REQUIRE (PeripheryCutoff (0.3) < PeripheryCutoff (0.7));
}

TEST_CASE ("CoOccurrenceThreshold equals MergeThreshold",
           "[formula][knobs][consolidation]")
{
  // CoOccurrenceThreshold should equal MergeThreshold
  for (double F : { 0.0, 0.25, 0.5, 0.75, 1.0 })
    {
      REQUIRE (CoOccurrenceThreshold (F)
               == Catch::Approx (MergeThreshold (F)));
    }
}

// =============================================================================
// 5.1.8 Metacognitive Functions (6 functions)
// =============================================================================

TEST_CASE ("FOKThreshold follows spec: lerp(0.2, 0.5, F)",
           "[formula][knobs][metacognitive]")
{
  // FOKThreshold = lerp(0.2, 0.5, F)
  REQUIRE (FOKThreshold (0.0) == Catch::Approx (0.2));
  REQUIRE (FOKThreshold (1.0) == Catch::Approx (0.5));
  REQUIRE (FOKThreshold (0.5)
           == Catch::Approx (FocusExpected (0.2, 0.5, 0.5)));

  // Monotonic
  REQUIRE (FOKThreshold (0.3) < FOKThreshold (0.7));
}

TEST_CASE ("TOTFokCutoff follows spec: lerp(0.5, 0.8, F)",
           "[formula][knobs][metacognitive]")
{
  // TOTFokCutoff = lerp(0.5, 0.8, F)
  REQUIRE (TOTFokCutoff (0.0) == Catch::Approx (0.5));
  REQUIRE (TOTFokCutoff (1.0) == Catch::Approx (0.8));
  REQUIRE (TOTFokCutoff (0.5)
           == Catch::Approx (FocusExpected (0.5, 0.8, 0.5)));

  // Monotonic
  REQUIRE (TOTFokCutoff (0.3) < TOTFokCutoff (0.7));
}

TEST_CASE ("TOTRetrievalCutoff follows spec: lerp(0.4, 0.2, F)",
           "[formula][knobs][metacognitive]")
{
  // TOTRetrievalCutoff = lerp(0.4, 0.2, F)
  REQUIRE (TOTRetrievalCutoff (0.0) == Catch::Approx (0.4));
  REQUIRE (TOTRetrievalCutoff (1.0) == Catch::Approx (0.2));
  REQUIRE (TOTRetrievalCutoff (0.5)
           == Catch::Approx (FocusExpected (0.4, 0.2, 0.5)));

  // Monotonic: higher F means lower cutoff
  REQUIRE (TOTRetrievalCutoff (0.3) > TOTRetrievalCutoff (0.7));
}

TEST_CASE ("UnknownThreshold follows spec: lerp(0.3, 0.1, F)",
           "[formula][knobs][metacognitive]")
{
  // UnknownThreshold = lerp(0.3, 0.1, F)
  REQUIRE (UnknownThreshold (0.0) == Catch::Approx (0.3));
  REQUIRE (UnknownThreshold (1.0) == Catch::Approx (0.1));
  REQUIRE (UnknownThreshold (0.5)
           == Catch::Approx (FocusExpected (0.3, 0.1, 0.5)));

  // Monotonic: higher F means lower threshold
  REQUIRE (UnknownThreshold (0.3) > UnknownThreshold (0.7));
}

TEST_CASE ("CertaintyRequirement follows spec: lerp(0.6, 0.9, T)",
           "[formula][knobs][metacognitive]")
{
  // CertaintyRequirement = lerp(0.6, 0.9, T)
  REQUIRE (CertaintyRequirement (0.0) == Catch::Approx (0.6));
  REQUIRE (CertaintyRequirement (1.0) == Catch::Approx (0.9));
  REQUIRE (CertaintyRequirement (0.5) == Catch::Approx (0.75));

  // Monotonic
  REQUIRE (CertaintyRequirement (0.3) < CertaintyRequirement (0.7));
}

TEST_CASE ("MetacognitiveSensitivity follows spec: F * (1 + 0.5 * S)",
           "[formula][knobs][metacognitive]")
{
  // MetacognitiveSensitivity = F * (1 + 0.5 * S)

  // At F=0, always 0 regardless of S
  REQUIRE (MetacognitiveSensitivity (0.0, 0.0) == Catch::Approx (0.0));
  REQUIRE (MetacognitiveSensitivity (0.0, 1.0) == Catch::Approx (0.0));

  // At F=1, S=0: 1 * 1 = 1
  REQUIRE (MetacognitiveSensitivity (1.0, 0.0) == Catch::Approx (1.0));

  // At F=1, S=1: 1 * 1.5 = 1.5
  REQUIRE (MetacognitiveSensitivity (1.0, 1.0) == Catch::Approx (1.5));

  REQUIRE (MetacognitiveSensitivity (0.5, 0.5)
           == Catch::Approx (FocusBias (0.5)
                             * (1.0 + 0.5 * SensitivityBias (0.5))));
}

// =============================================================================
// 5.1.9 Serial Position Effects (8 functions)
// =============================================================================

TEST_CASE ("SerialPrimacyWindow follows spec: lerp(5, 2, F)",
           "[formula][knobs][serial]")
{
  // SerialPrimacyWindow = round(lerp(5, 2, F))
  REQUIRE (SerialPrimacyWindow (0.0) == 5);
  REQUIRE (SerialPrimacyWindow (1.0) == 2);
  REQUIRE (SerialPrimacyWindow (0.5) == 4); // round(3.5)

  // Monotonic: higher F means smaller primacy window
  REQUIRE (SerialPrimacyWindow (0.3) > SerialPrimacyWindow (0.7));
}

TEST_CASE ("SerialRecencyWindow follows spec: lerp(7, 3, F)",
           "[formula][knobs][serial]")
{
  // SerialRecencyWindow = round(lerp(7, 3, F))
  REQUIRE (SerialRecencyWindow (0.0) == 7);
  REQUIRE (SerialRecencyWindow (1.0) == 3);
  REQUIRE (SerialRecencyWindow (0.5) == 5);

  // Monotonic: higher F means smaller recency window
  REQUIRE (SerialRecencyWindow (0.3) > SerialRecencyWindow (0.7));
}

TEST_CASE ("SerialPrimacyBonus follows spec: lerp(1.2, 2.0, S)",
           "[formula][knobs][serial]")
{
  // SerialPrimacyBonus = lerp(1.2, 2.0, S)
  REQUIRE (SerialPrimacyBonus (0.0) == Catch::Approx (1.2));
  REQUIRE (SerialPrimacyBonus (1.0) == Catch::Approx (2.0));
  REQUIRE (SerialPrimacyBonus (0.5)
           == Catch::Approx (SensitivityExpected (1.2, 2.0, 0.5)));

  // Monotonic
  REQUIRE (SerialPrimacyBonus (0.3) < SerialPrimacyBonus (0.7));
}

TEST_CASE ("SerialRehearsalCurveDepth follows spec: lerp(0.2, 0.6, S)",
           "[formula][knobs][serial]")
{
  // SerialRehearsalCurveDepth = lerp(0.2, 0.6, S)
  REQUIRE (SerialRehearsalCurveDepth (0.0) == Catch::Approx (0.2));
  REQUIRE (SerialRehearsalCurveDepth (1.0) == Catch::Approx (0.6));
  REQUIRE (SerialRehearsalCurveDepth (0.5)
           == Catch::Approx (SensitivityExpected (0.2, 0.6, 0.5)));

  // Monotonic
  REQUIRE (SerialRehearsalCurveDepth (0.3) < SerialRehearsalCurveDepth (0.7));
}

TEST_CASE ("SerialDistinctivenessThreshold follows spec: lerp(0.6, 0.8, F)",
           "[formula][knobs][serial]")
{
  // SerialDistinctivenessThreshold = lerp(0.6, 0.8, F)
  REQUIRE (SerialDistinctivenessThreshold (0.0) == Catch::Approx (0.6));
  REQUIRE (SerialDistinctivenessThreshold (1.0) == Catch::Approx (0.8));
  REQUIRE (SerialDistinctivenessThreshold (0.5)
           == Catch::Approx (FocusExpected (0.6, 0.8, 0.5)));

  // Monotonic
  REQUIRE (SerialDistinctivenessThreshold (0.3)
           < SerialDistinctivenessThreshold (0.7));
}

TEST_CASE ("SerialVonRestorffMultiplier follows spec: lerp(1.5, 3.0, S)",
           "[formula][knobs][serial]")
{
  // SerialVonRestorffMultiplier = lerp(1.5, 3.0, S)
  REQUIRE (SerialVonRestorffMultiplier (0.0) == Catch::Approx (1.5));
  REQUIRE (SerialVonRestorffMultiplier (1.0) == Catch::Approx (3.0));
  REQUIRE (SerialVonRestorffMultiplier (0.5)
           == Catch::Approx (SensitivityExpected (1.5, 3.0, 0.5)));

  // Monotonic
  REQUIRE (SerialVonRestorffMultiplier (0.3)
           < SerialVonRestorffMultiplier (0.7));
}

TEST_CASE ("SerialMiddleSuppression follows spec: lerp(0.8, 0.5, S) * (1 - F)",
           "[formula][knobs][serial]")
{
  // SerialMiddleSuppression = lerp(0.8, 0.5, S) * (1 - F)

  // At S=0, F=0: 0.8 * 1 = 0.8
  REQUIRE (SerialMiddleSuppression (0.0, 0.0) == Catch::Approx (0.8));

  // At S=1, F=0: 0.5 * 1 = 0.5
  REQUIRE (SerialMiddleSuppression (1.0, 0.0) == Catch::Approx (0.5));

  // At S=0, F=1: 0.8 * 0 = 0
  REQUIRE (SerialMiddleSuppression (0.0, 1.0) == Catch::Approx (0.0));

  // At S=1, F=1: 0.5 * 0 = 0
  REQUIRE (SerialMiddleSuppression (1.0, 1.0) == Catch::Approx (0.0));

  REQUIRE (SerialMiddleSuppression (0.5, 0.5)
           == Catch::Approx (SensitivityExpected (0.8, 0.5, 0.5)
                             * (1.0 - FocusBias (0.5))));
}

TEST_CASE ("GistComponents follows spec: lerp(5, 2, F)",
           "[formula][knobs][serial]")
{
  // GistComponents = round(lerp(5, 2, F))
  REQUIRE (GistComponents (0.0) == 5);
  REQUIRE (GistComponents (1.0) == 2);
  REQUIRE (GistComponents (0.5) == 4); // round(3.5)

  // Monotonic: higher F means fewer gist components
  REQUIRE (GistComponents (0.3) > GistComponents (0.7));
}

// =============================================================================
// 5.1.10 Emotional Functions (10 functions)
// =============================================================================

TEST_CASE ("ThetaIntensity follows spec: lerp(0.6, 0.8, 1 - S)",
           "[formula][knobs][emotional]")
{
  // ThetaIntensity = lerp(0.6, 0.8, 1 - S)
  // At S=0: lerp(0.6, 0.8, 1) = 0.8
  REQUIRE (ThetaIntensity (0.0) == Catch::Approx (0.8));
  // At S=1: lerp(0.6, 0.8, 0) = 0.6
  REQUIRE (ThetaIntensity (1.0) == Catch::Approx (0.6));
  // At S=0.5: lerp(0.6, 0.8, 0.5) = 0.7
  REQUIRE (ThetaIntensity (0.5)
           == Catch::Approx (Lerp (0.6, 0.8, 1.0 - SensitivityBias (0.5))));

  // Monotonic: higher S means lower threshold
  REQUIRE (ThetaIntensity (0.3) > ThetaIntensity (0.7));
}

TEST_CASE ("ThetaArousal follows spec: lerp(0.4, 0.2, S)",
           "[formula][knobs][emotional]")
{
  // ThetaArousal = lerp(0.4, 0.2, S)
  REQUIRE (ThetaArousal (0.0) == Catch::Approx (0.4));
  REQUIRE (ThetaArousal (1.0) == Catch::Approx (0.2));
  REQUIRE (ThetaArousal (0.5)
           == Catch::Approx (SensitivityExpected (0.4, 0.2, 0.5)));

  // Monotonic: higher S means lower threshold
  REQUIRE (ThetaArousal (0.3) > ThetaArousal (0.7));
}

TEST_CASE ("FlashbulbThreshold follows spec: lerp(0.97, 0.65, SensitivityBias(S))",
           "[formula][knobs][emotional]")
{
  REQUIRE (FlashbulbThreshold (0.0) == Catch::Approx (0.97));
  REQUIRE (FlashbulbThreshold (1.0) == Catch::Approx (0.65));
  REQUIRE (FlashbulbThreshold (0.5)
           == Catch::Approx (SensitivityExpected (0.97, 0.65, 0.5)));

  // Monotonic: higher S means lower threshold
  REQUIRE (FlashbulbThreshold (0.3) > FlashbulbThreshold (0.7));
}

TEST_CASE ("FlashbulbThresholdEff follows spec",
           "[formula][knobs][emotional]")
{
  const double base = FlashbulbThreshold (0.5);
  const double p = Lerp (1.25, 0.85, SensitivityBias (0.5));
  REQUIRE (FlashbulbThresholdEff (0.5, 0.0, 0.0)
           == Catch::Approx (base));
  REQUIRE (FlashbulbThresholdEff (0.5, 1.0, 0.0)
           == Catch::Approx (base * 0.5));
  REQUIRE (FlashbulbThresholdEff (0.5, 0.0, 1.0)
           == Catch::Approx (base * std::pow (0.7, p)));
  REQUIRE (FlashbulbThresholdEff (0.5, 1.0, 1.0)
           == Catch::Approx (base * 0.5 * std::pow (0.7, p)));
}

TEST_CASE ("CascadeRadius follows spec: lerp(1, 5, S)",
           "[formula][knobs][emotional]")
{
  // CascadeRadius = round(lerp(1, 5, S))
  REQUIRE (CascadeRadius (0.0) == 1);
  REQUIRE (CascadeRadius (1.0) == 5);
  REQUIRE (CascadeRadius (0.5) == 3);

  // Monotonic
  REQUIRE (CascadeRadius (0.3) < CascadeRadius (0.7));
}

TEST_CASE ("CascadeDecay follows spec: lerp(0.7, 0.3, S)",
           "[formula][knobs][emotional]")
{
  // CascadeDecay = lerp(0.7, 0.3, S)
  REQUIRE (CascadeDecay (0.0) == Catch::Approx (0.7));
  REQUIRE (CascadeDecay (1.0) == Catch::Approx (0.3));
  REQUIRE (CascadeDecay (0.5)
           == Catch::Approx (SensitivityExpected (0.7, 0.3, 0.5)));

  // Monotonic: higher S means faster decay
  REQUIRE (CascadeDecay (0.3) > CascadeDecay (0.7));
}

TEST_CASE ("DetailSuppression follows spec: S * (1 - F) * 0.5",
           "[formula][knobs][emotional]")
{
  // DetailSuppression = S * (1 - F) * 0.5

  // At S=0, always 0
  REQUIRE (DetailSuppression (0.0, 0.0) == Catch::Approx (0.0));
  REQUIRE (DetailSuppression (0.0, 1.0) == Catch::Approx (0.0));

  // At F=1, always 0 (no detail suppression at high focus)
  REQUIRE (DetailSuppression (1.0, 1.0) == Catch::Approx (0.0));

  // At S=1, F=0: 1 * 1 * 0.5 = 0.5 (max suppression)
  REQUIRE (DetailSuppression (1.0, 0.0) == Catch::Approx (0.5));

  REQUIRE (DetailSuppression (0.5, 0.5)
           == Catch::Approx (SensitivityBias (0.5)
                             * (1.0 - FocusBias (0.5)) * 0.5));
}

TEST_CASE ("LabilitySusceptibility follows spec: (1 - T) * (0.5 + 0.5 * S)",
           "[formula][knobs][emotional]")
{
  // LabilitySusceptibility = (1 - T) * (0.5 + 0.5 * S)

  // At T=1, always 0 (high stability = no lability)
  REQUIRE (LabilitySusceptibility (0.0, 1.0) == Catch::Approx (0.0));
  REQUIRE (LabilitySusceptibility (1.0, 1.0) == Catch::Approx (0.0));

  // At T=0, S=0: 1 * 0.5 = 0.5
  REQUIRE (LabilitySusceptibility (0.0, 0.0) == Catch::Approx (0.5));

  // At T=0, S=1: 1 * 1.0 = 1.0
  REQUIRE (LabilitySusceptibility (1.0, 0.0) == Catch::Approx (1.0));

  REQUIRE (LabilitySusceptibility (0.5, 0.5)
           == Catch::Approx ((1.0 - 0.5)
                             * (0.5 + 0.5 * SensitivityBias (0.5))));
}

// =============================================================================
// 5.1.11 Knowledge Graph Functions (3 functions)
// =============================================================================

TEST_CASE ("CausalDriftThreshold follows spec: lerp(0.15, 0.35, T)",
           "[formula][knobs][graph]")
{
  // CausalDriftThreshold = lerp(0.15, 0.35, T)
  REQUIRE (CausalDriftThreshold (0.0) == Catch::Approx (0.15));
  REQUIRE (CausalDriftThreshold (1.0) == Catch::Approx (0.35));
  REQUIRE (CausalDriftThreshold (0.5) == Catch::Approx (0.25));

  // Monotonic
  REQUIRE (CausalDriftThreshold (0.3) < CausalDriftThreshold (0.7));
}

TEST_CASE ("MinEpisodesForConcept follows spec: lerp(2, 5, T)",
           "[formula][knobs][graph]")
{
  // MinEpisodesForConcept = round(lerp(2, 5, T))
  REQUIRE (MinEpisodesForConcept (0.0) == 2);
  REQUIRE (MinEpisodesForConcept (1.0) == 5);
  REQUIRE (MinEpisodesForConcept (0.5) == 4); // round(3.5)

  // Monotonic
  REQUIRE (MinEpisodesForConcept (0.3) < MinEpisodesForConcept (0.7));
}

TEST_CASE ("ContradictionThreshold midpoint wrapper returns -0.5",
           "[formula][knobs][graph]")
{
  // Legacy zero-arg wrapper returns the midpoint F/S/T threshold.
  REQUIRE (ContradictionThreshold () == Catch::Approx (-0.5));
}

// =============================================================================
// 5.1.12 Multi-Parameter Functions (5 functions)
// =============================================================================

TEST_CASE ("TPrior follows spec: lerp(0.10, 0.30, T) * (1 - 0.3 * S)",
           "[formula][knobs][multi]")
{
  // TPrior = lerp(0.10, 0.30, T) * (1 - 0.3 * S)

  // At T=0, S=0: 0.10 * 1 = 0.10
  REQUIRE (TPrior (0.5, 0.0, 0.0) == Catch::Approx (0.10));

  // At T=1, S=0: 0.30 * 1 = 0.30
  REQUIRE (TPrior (0.5, 0.0, 1.0) == Catch::Approx (0.30));

  // At T=0, S=1: 0.10 * 0.7 = 0.07
  REQUIRE (TPrior (0.5, 1.0, 0.0) == Catch::Approx (0.07));

  // At T=1, S=1: 0.30 * 0.7 = 0.21
  REQUIRE (TPrior (0.5, 1.0, 1.0) == Catch::Approx (0.21));

  REQUIRE (TPrior (0.5, 0.5, 0.5)
           == Catch::Approx (Lerp (0.10, 0.30, 0.5)
                             * (1.0 - 0.3 * SensitivityBias (0.5))));
}

TEST_CASE ("TargetPrecision follows spec: CertaintyRequirement(T) * (0.5 + "
           "0.5*F)",
           "[formula][knobs][multi]")
{
  // TargetPrecision = CertaintyRequirement(T) * (0.5 + 0.5*F)

  // At T=0, F=0: 0.6 * 0.5 = 0.3
  REQUIRE (TargetPrecision (0.0, 0.0) == Catch::Approx (0.3));

  // At T=1, F=1: 0.9 * 1.0 = 0.9
  REQUIRE (TargetPrecision (1.0, 1.0) == Catch::Approx (0.9));

  REQUIRE (TargetPrecision (0.5, 0.5)
           == Catch::Approx (CertaintyRequirement (0.5)
                             * (0.5 + 0.5 * FocusBias (0.5))));

  // Verify it uses CertaintyRequirement
  for (double T : { 0.0, 0.5, 1.0 })
    {
      for (double F : { 0.0, 0.5, 1.0 })
        {
          double expected
              = CertaintyRequirement (T) * (0.5 + 0.5 * FocusBias (F));
          REQUIRE (TargetPrecision (F, T) == Catch::Approx (expected));
        }
    }
}

TEST_CASE ("PriorMass follows spec: lerp(2, 32, T)",
           "[formula][knobs][multi]")
{
  // PriorMass = round(lerp(2, 32, T))
  REQUIRE (PriorMass (0.0) == 2);
  REQUIRE (PriorMass (1.0) == 32);
  REQUIRE (PriorMass (0.5) == 17);

  // Monotonic
  REQUIRE (PriorMass (0.3) < PriorMass (0.7));
}

TEST_CASE ("GraphDepth follows spec: lerp(3.0, 2.0, T)",
           "[formula][knobs][multi]")
{
  // GraphDepth = round(lerp(3.0, 2.0, T))
  REQUIRE (GraphDepth (0.0) == 3);
  REQUIRE (GraphDepth (1.0) == 2);
  REQUIRE (GraphDepth (0.5) == 3);

  // Range check
  for (double F : { 0.0, 0.25, 0.5, 0.75, 1.0 })
    {
      int depth = GraphDepth (F);
      REQUIRE (depth >= 2);
      REQUIRE (depth <= 3);
    }
}

TEST_CASE ("MemoryUsageThreshold follows spec: lerp(0.5, 0.8, F)",
           "[formula][knobs][multi]")
{
  // MemoryUsageThreshold = lerp(0.5, 0.8, F)
  REQUIRE (MemoryUsageThreshold (0.0) == Catch::Approx (0.5));
  REQUIRE (MemoryUsageThreshold (1.0) == Catch::Approx (0.8));
  REQUIRE (MemoryUsageThreshold (0.5)
           == Catch::Approx (FocusExpected (0.5, 0.8, 0.5)));

  // Monotonic
  REQUIRE (MemoryUsageThreshold (0.3) < MemoryUsageThreshold (0.7));
}

TEST_CASE ("MemoryUsageCacheDuration follows spec: lerp(30, 300, T)",
           "[formula][knobs][multi]")
{
  // MemoryUsageCacheDuration = lerp(30, 300, T)
  REQUIRE (MemoryUsageCacheDuration (0.0) == Catch::Approx (30.0));
  REQUIRE (MemoryUsageCacheDuration (1.0) == Catch::Approx (300.0));
  REQUIRE (MemoryUsageCacheDuration (0.5) == Catch::Approx (165.0));

  // Monotonic
  REQUIRE (MemoryUsageCacheDuration (0.3) < MemoryUsageCacheDuration (0.7));
}

// =============================================================================
// 5.1.13 SerialDetermineZone Helper (comprehensive test)
// =============================================================================

TEST_CASE ("SerialDetermineZone correctly classifies positions",
           "[formula][knobs][serial]")
{
  // Test zone determination with F=0.5
  // primacy_window = 4, recency_window = 5
  // distinctiveness_threshold = 0.7

  int N = 20; // Total items
  double F = 0.5;

  // Position 1 should be Primacy
  REQUIRE (SerialDetermineZone (F, 1, N, 0.0) == SerialZone::Primacy);

  // Position 4 should be Primacy (at window edge)
  REQUIRE (SerialDetermineZone (F, 4, N, 0.0) == SerialZone::Primacy);

  // Position 16 should be Recency (N - recency + 1 = 20 - 5 + 1 = 16)
  REQUIRE (SerialDetermineZone (F, 16, N, 0.0) == SerialZone::Recency);

  // Position 20 should be Recency
  REQUIRE (SerialDetermineZone (F, 20, N, 0.0) == SerialZone::Recency);

  // Position 10 with rarity > threshold should be Distinctive
  REQUIRE (SerialDetermineZone (F, 10, N, 0.8) == SerialZone::Distinctive);

  // Position 10 with rarity <= threshold should be Middle
  REQUIRE (SerialDetermineZone (F, 10, N, 0.5) == SerialZone::Middle);
}

// =============================================================================
// Composite Tests: Verify Knob Interactions
// =============================================================================

TEST_CASE ("High Focus + Low Sensitivity interactions",
           "[formula][knobs][composite]")
{
  double F = 0.9;
  double S = 0.1;
  double T = 0.5;

  // High F: fewer results, stricter thresholds
  REQUIRE (MaxResults (F) <= 24);
  REQUIRE (MergeThreshold (F) >= 0.93);
  REQUIRE (WMGateThreshold (F) >= 0.35);

  // Low S: slower mood, lower learning rates
  REQUIRE (AlphaMood (S) < 0.03);
  REQUIRE (StreamingPacingThreshold (S) > 0.25);

  // Multi-parameter: high F mitigates S
  REQUIRE (MetacognitiveSensitivity (F, S) > 0.8);
  REQUIRE (DetailSuppression (S, F) < 0.01);
}

TEST_CASE ("Low Focus + High Sensitivity interactions",
           "[formula][knobs][composite]")
{
  double F = 0.1;
  double S = 0.9;
  double T = 0.5;

  // Low F: more results, permissive thresholds
  REQUIRE (MaxResults (F) >= 55);
  REQUIRE (MergeThreshold (F) <= 0.86);
  REQUIRE (WMGateThreshold (F) <= 0.15);

  // High S: faster mood, higher learning rates
  REQUIRE (AlphaMood (S) > 0.17);
  REQUIRE (StreamingPacingThreshold (S) < 0.15);

  // Multi-parameter: high S amplified with low F
  REQUIRE (DetailSuppression (S, F) > 0.4);
  REQUIRE (CascadeRadius (S) >= 4);
}

TEST_CASE ("Extreme stability values", "[formula][knobs][composite]")
{
  // High stability: long half-lives, slow decay, stable persistence
  double T_high = 1.0;
  REQUIRE (BaseHalfLifePrior (T_high) == Catch::Approx (43200.0));
  REQUIRE (LambdaMood (30.0, T_high) > 0.95);
  REQUIRE (ReinforcementDecay (T_high) >= 0.98);
  REQUIRE (ConsolidationIntervalSeconds (T_high) == 3600);

  // Low stability: short half-lives, fast decay, responsive
  double T_low = 0.0;
  REQUIRE (BaseHalfLifePrior (T_low) == Catch::Approx (120.0));
  REQUIRE (LambdaMood (30.0, T_low) == Catch::Approx (0.5));
  REQUIRE (ReinforcementDecay (T_low) == Catch::Approx (0.9));
  REQUIRE (ConsolidationIntervalSeconds (T_low) == 300);
}
