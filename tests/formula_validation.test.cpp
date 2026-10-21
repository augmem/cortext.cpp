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

// =============================================================================
// 5.1.5 Half-Life & Decay Functions (9 functions)
// =============================================================================

TEST_CASE ("Reinforcement edge-maintenance calibrations remain bounded",
           "[formula][knobs][reinforcement]")
{
  // The step still calibrates the production pruning floor. The unselected
  // and fanout helpers remain covered for public source compatibility only;
  // production retrieval no longer creates packet-pair edges.
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
}

TEST_CASE ("WMMaintenanceCostPerSlot keeps the full-WM budget "
           "capacity-invariant",
           "[formula][knobs][working_memory]")
{
  // Per-slot maintenance shares a fixed total budget: cost-per-slot x
  // capacity = lerp(0.05, 0.15, S) x 7 (the legacy neutral-capacity budget),
  // so capacity changes never change gate strictness.
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

  // At neutral knobs, the production default is capacity 21 and the per-slot
  // cost is one third of the original seven-slot spec value.
  REQUIRE (WMBaseCapacity (0.5, 0.5) == 21);
  REQUIRE (WMMaintenanceCostPerSlot (0.5, 0.5)
           == Catch::Approx (SensitivityExpected (0.05, 0.15, 0.5) / 3.0));

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

TEST_CASE ("WM eviction pressure derives from F/S/T",
           "[formula][knobs][working_memory]")
{
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

// =============================================================================
// 5.1.8 Serial Position Effects (8 functions)
// =============================================================================

// =============================================================================
// 5.1.10 Emotional Functions (10 functions)
// =============================================================================

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

TEST_CASE ("TargetPrecision follows spec: lerp(0.6, 0.9, T) * (0.5 + "
           "0.5*F)",
           "[formula][knobs][multi]")
{
  // TargetPrecision = lerp(0.6, 0.9, T) * (0.5 + 0.5*F)

  // At T=0, F=0: 0.6 * 0.5 = 0.3
  REQUIRE (TargetPrecision (0.0, 0.0) == Catch::Approx (0.3));

  // At T=1, F=1: 0.9 * 1.0 = 0.9
  REQUIRE (TargetPrecision (1.0, 1.0) == Catch::Approx (0.9));

  REQUIRE (TargetPrecision (0.5, 0.5)
           == Catch::Approx (Lerp (0.6, 0.9, 0.5)
                             * (0.5 + 0.5 * FocusBias (0.5))));

  // Verify it uses the retained stability/focus formula.
  for (double T : { 0.0, 0.5, 1.0 })
    {
      for (double F : { 0.0, 0.5, 1.0 })
        {
          double expected
              = Lerp (0.6, 0.9, T) * (0.5 + 0.5 * FocusBias (F));
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
  REQUIRE (WMGateThreshold (F) >= 0.35);

  // Low S: lower learning rates
  REQUIRE (StreamingPacingThreshold (S) > 0.25);

  // Multi-parameter: high F mitigates S
}

TEST_CASE ("Low Focus + High Sensitivity interactions",
           "[formula][knobs][composite]")
{
  double F = 0.1;
  double S = 0.9;
  double T = 0.5;

  // Low F: more results, permissive thresholds
  REQUIRE (MaxResults (F) >= 55);
  REQUIRE (WMGateThreshold (F) <= 0.15);

  // High S: higher learning rates
  REQUIRE (StreamingPacingThreshold (S) < 0.15);

  // Multi-parameter: high S amplified with low F
}
