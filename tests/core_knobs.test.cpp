#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include "test_helpers.hpp"
#include <cmath>

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

TEST_CASE ("WMGateThreshold follows spec: lerp(0.1, 0.4, F_eff)", "[core][knobs]")
{
  // gate_threshold = lerp(0.1, 0.4, F_eff) per Algorithm 24
  // At F=0 (wide attention): permissive (0.1)
  // At F=1 (narrow attention): strict (0.4)
  REQUIRE (WMGateThreshold (0.0) == Catch::Approx (0.1));
  REQUIRE (WMGateThreshold (1.0) == Catch::Approx (0.4));
  REQUIRE (WMGateThreshold (0.5) == Catch::Approx (0.22));

  // Monotonic: higher F means higher (stricter) threshold
  REQUIRE (WMGateThreshold (0.3) < WMGateThreshold (0.7));
}

TEST_CASE ("Working memory benefit weights derive from all knobs",
           "[core][knobs]")
{
  const auto mid = WMBenefitScoringWeights (0.5, 0.5, 0.5);
  REQUIRE (mid.window + mid.relevance + mid.novelty == Catch::Approx (1.0));

  REQUIRE (WMBenefitScoringWeights (0.5, 0.5, 1.0).window
           > WMBenefitScoringWeights (0.5, 0.5, 0.0).window);
  REQUIRE (WMBenefitScoringWeights (0.5, 1.0, 0.5).novelty
           > WMBenefitScoringWeights (0.5, 0.0, 0.5).novelty);
  REQUIRE (WMBenefitScoringWeights (1.0, 0.5, 0.5).relevance
           > WMBenefitScoringWeights (0.0, 0.5, 0.5).relevance);
}

TEST_CASE ("SourceReliabilityPrior is knob-derived with historical midpoint",
           "[core][knobs]")
{
  REQUIRE (SourceReliabilityPrior (0.5, 0.5, 0.5)
           == Catch::Approx (0.70));

  const double low_trust = SourceReliabilityPrior (0.0, 1.0, 0.0);
  const double high_trust = SourceReliabilityPrior (1.0, 0.0, 1.0);
  REQUIRE (low_trust < SourceReliabilityPrior (0.5, 0.5, 0.5));
  REQUIRE (high_trust > SourceReliabilityPrior (0.5, 0.5, 0.5));
  REQUIRE (low_trust >= 0.50);
  REQUIRE (high_trust <= 0.90);
}

TEST_CASE ("Sensitivity state priors are centralized knob helpers",
           "[core][knobs]")
{
  const auto priors = SensitivityStatePriorsForKnobs (0.5);
  const double s = SensitivityBias (0.5);

  REQUIRE (priors.novelty_weight == Catch::Approx (0.3 + 0.7 * s));
  REQUIRE (priors.surprise_weight == Catch::Approx (0.2 + 0.8 * s));
  REQUIRE (priors.valence_weight == Catch::Approx (0.4 + 0.6 * s));
  REQUIRE (priors.arousal_weight == Catch::Approx (s));
  REQUIRE (priors.emotion_weight == Catch::Approx (0.2 + 0.8 * s));
  REQUIRE (priors.emotion_gain == Catch::Approx (std::exp (1.5 * s)));
  REQUIRE (priors.score_gain == Catch::Approx (std::exp (2.0 * s)));
}

TEST_CASE ("Emotion projection policy derives from Sensitivity",
           "[core][knobs]")
{
  const auto low = EmotionProjectionPolicyForKnobs (0.0);
  const auto mid = EmotionProjectionPolicyForKnobs (0.5);
  const auto high = EmotionProjectionPolicyForKnobs (1.0);

  REQUIRE (mid.softmax_beta
           == Catch::Approx (8.0 + 24.0 * SensitivityBias (0.5)));
  REQUIRE (mid.intensity_gamma
           == Catch::Approx (Lerp (0.5, 0.25, SensitivityBias (0.5))));
  REQUIRE (mid.intensity_gain
           == Catch::Approx (Lerp (1.0, 2.2, SensitivityBias (0.5))));
  REQUIRE (mid.ewma_alpha
           == Catch::Approx (Lerp (0.05, 0.30, SensitivityBias (0.5))));

  REQUIRE (high.softmax_beta > low.softmax_beta);
  REQUIRE (high.intensity_gamma < low.intensity_gamma);
  REQUIRE (high.intensity_gain > low.intensity_gain);
  REQUIRE (high.ewma_alpha > low.ewma_alpha);
}

TEST_CASE ("WMBaseCapacity defaults to the capacity-21 operating point",
           "[core][knobs]")
{
  // base_capacity = 3 * (lerp(8, 6, SensitivityBias(S))
  //                      + lerp(-1, 1, FocusBias(F)))
  // Capacity range [15, 27], with 21 at neutral knobs.

  // At S=0, F=0: 3 * (8 + (-1)) = 21
  REQUIRE (WMBaseCapacity (0.0, 0.0) == 21);

  // At S=1, F=1: 3 * (6 + 1) = 21
  REQUIRE (WMBaseCapacity (1.0, 1.0) == 21);

  // At S=0, F=1: 3 * (8 + 1) = 27 (max)
  REQUIRE (WMBaseCapacity (0.0, 1.0) == 27);

  // At S=1, F=0: 3 * (6 + (-1)) = 15 (min)
  REQUIRE (WMBaseCapacity (1.0, 0.0) == 15);

  // Verify range bounds across all extreme values
  REQUIRE (WMBaseCapacity (0.0, 1.0) == 27); // max
  REQUIRE (WMBaseCapacity (1.0, 0.0) == 15); // min

  // Mid-point: biased lerps cancel to exactly 21 at neutral knobs
  REQUIRE (WMBaseCapacity (0.5, 0.5) == 21);
  REQUIRE (WMStrengthBase (0.5) == Catch::Approx (0.60));
  REQUIRE (WMStrengthBase (1.0) > WMStrengthBase (0.0));
  REQUIRE (WMStrengthMax (0.5, 0.5, 0.5) == Catch::Approx (10.0));
  REQUIRE (WMStrengthMax (0.5, 0.5, 1.0)
           > WMStrengthMax (0.5, 0.5, 0.0));
  REQUIRE (WMInitialSlotStrength (0.5, 0.5, 0.5, 0.5)
           == Catch::Approx (0.45));
  REQUIRE (WMInitialSlotStrength (0.5, 0.5, 0.5, 1.0)
           > WMInitialSlotStrength (0.5, 0.5, 0.5, 0.0));
  REQUIRE (WMCostSaturator (2.0) == Catch::Approx (2.0 / 3.0));
}

TEST_CASE ("WM capacity override is ignored without experiment hooks",
           "[core][knobs][experiment_hooks]")
{
  cortext::testing::ScopedEnvVar override ("CORTEXT_WM_CAPACITY_OVERRIDE",
                                           "42");
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  SUCCEED (
      "CORTEXT_WM_CAPACITY_OVERRIDE is process-cached and covered by "
      "ablation hook integration tests when experiment hooks are enabled.");
#else
  REQUIRE (WMBaseCapacity (0.5, 0.5) == 21);
#endif
}

TEST_CASE ("Bootstrap state priors derive from knobs", "[core][knobs]")
{
  const auto focus_mid = FocusStatePriorsForKnobs (0.5);
  REQUIRE (focus_mid.relevance_weight == Catch::Approx (0.5));
  REQUIRE (focus_mid.coverage_gain_floor == Catch::Approx (0.65));
  REQUIRE (focus_mid.mismatch_weight == Catch::Approx (0.5));
  REQUIRE (FocusStatePriorsForKnobs (1.0).relevance_weight
           > FocusStatePriorsForKnobs (0.0).relevance_weight);

  const auto stability_mid = StabilityStatePriorsForKnobs (0.5);
  REQUIRE (stability_mid.rate_decay == Catch::Approx (0.79));
  REQUIRE (stability_mid.secondary_half_life_scale == Catch::Approx (0.50));
  REQUIRE (stability_mid.drift_weight == Catch::Approx (0.25));
  REQUIRE (StabilityStatePriorsForKnobs (1.0).rate_decay
           > StabilityStatePriorsForKnobs (0.0).rate_decay);
  REQUIRE (StabilityStatePriorsForKnobs (1.0).drift_weight
           < StabilityStatePriorsForKnobs (0.0).drift_weight);
}

TEST_CASE ("StreamingPacingThreshold knob function", "[core][knobs]")
{
  // pacing_thresh(S_eff) = lerp(0.3, 0.05, S_eff)
  // Higher sensitivity = lower threshold = more frequent pacing checks

  // S=0: conservative (0.3)
  REQUIRE (StreamingPacingThreshold (0.0) == Catch::Approx (0.3));
  // S=1: aggressive (0.05)
  REQUIRE (StreamingPacingThreshold (1.0) == Catch::Approx (0.05));
  // S=0.5: midpoint after bias (0.2)
  REQUIRE (StreamingPacingThreshold (0.5) == Catch::Approx (0.2));

  // Monotonic: higher S means lower threshold
  REQUIRE (StreamingPacingThreshold (0.3) > StreamingPacingThreshold (0.7));
}

TEST_CASE ("StreamingRetrievalBiasThreshold derives from all knobs",
           "[core][knobs]")
{
  const double mid = StreamingRetrievalBiasThreshold (0.5, 0.5, 0.5);
  REQUIRE (mid > 0.12);
  REQUIRE (mid < 0.18);
  REQUIRE (StreamingRetrievalBiasThreshold (1.0, 0.5, 0.5)
           > StreamingRetrievalBiasThreshold (0.0, 0.5, 0.5));
  REQUIRE (StreamingRetrievalBiasThreshold (0.5, 1.0, 0.5)
           < StreamingRetrievalBiasThreshold (0.5, 0.0, 0.5));
  REQUIRE (StreamingRetrievalBiasThreshold (0.5, 0.5, 1.0)
           > StreamingRetrievalBiasThreshold (0.5, 0.5, 0.0));
}

TEST_CASE ("Signal filter adaptive policy derives from knobs",
           "[core][knobs]")
{
  const auto mid = SignalFilterAdaptivePolicyForKnobs (0.5, 0.5, 0.5);
  REQUIRE (mid.alpha == Catch::Approx (0.14));
  REQUIRE (mid.base_threshold == Catch::Approx (0.012));
  REQUIRE (mid.heartbeat_seconds == Catch::Approx (2.0));
  REQUIRE (mid.focus_pressure == Catch::Approx (1.10));
  REQUIRE (mid.mean_score_weight + mid.max_score_weight
           == Catch::Approx (1.0));
  REQUIRE (SignalFilterAudioMinAcceptSeconds (0.5, 0.5, 0.5)
           == Catch::Approx (0.10));
  REQUIRE (SignalFilterImageMinAcceptSeconds (0.5, 0.5, 0.5)
           == Catch::Approx (1.0 / 60.0));

  REQUIRE (SignalFilterAdaptivePolicyForKnobs (0.5, 0.5, 1.0).alpha
           < SignalFilterAdaptivePolicyForKnobs (0.5, 0.5, 0.0).alpha);
  REQUIRE (SignalFilterAdaptivePolicyForKnobs (0.5, 1.0, 0.5)
               .sensitivity_release
           < SignalFilterAdaptivePolicyForKnobs (0.5, 0.0, 0.5)
                 .sensitivity_release);
  REQUIRE (SignalFilterAdaptivePolicyForKnobs (1.0, 0.5, 0.5)
               .focus_pressure
           > SignalFilterAdaptivePolicyForKnobs (0.0, 0.5, 0.5)
                 .focus_pressure);
}

TEST_CASE ("Soft anchor hard boundary retention derives from knobs",
           "[core][knobs]")
{
  REQUIRE (SoftAnchorHardBoundaryRetainSteps (1.0, 0.5, 0.5)
           <= SoftAnchorHardBoundaryRetainSteps (0.0, 0.5, 0.5));
  REQUIRE (SoftAnchorHardBoundaryRetainSteps (0.5, 0.5, 1.0)
           >= SoftAnchorHardBoundaryRetainSteps (0.5, 0.5, 0.0));
}

TEST_CASE ("Boundary evidence weights derive from all knobs", "[core][knobs]")
{
  const auto mid = BoundaryEvidenceScoringWeights (0.5, 0.5, 0.5);
  REQUIRE (mid.gap == Catch::Approx (0.01));
  REQUIRE (BoundaryEvidenceScoringWeights (1.0, 0.5, 0.5).coherence
           > BoundaryEvidenceScoringWeights (0.0, 0.5, 0.5).coherence);
  REQUIRE (BoundaryEvidenceScoringWeights (0.5, 1.0, 0.5).surprisal
           > BoundaryEvidenceScoringWeights (0.5, 0.0, 0.5).surprisal);
  REQUIRE (BoundaryEvidenceScoringWeights (0.5, 0.5, 1.0).drift
           < BoundaryEvidenceScoringWeights (0.5, 0.5, 0.0).drift);
  REQUIRE (BoundaryEvidenceScoringWeights (0.5, 1.0, 1.0).gap
           > BoundaryEvidenceScoringWeights (0.5, 0.0, 0.0).gap);
}

TEST_CASE ("Boundary and accumulator micro-policies derive from knobs",
           "[core][knobs]")
{
  const auto norm_mid = BoundaryLocalNormPolicyForKnobs (0.5, 0.5);
  REQUIRE (norm_mid.alpha == Catch::Approx (0.25));
  REQUIRE (norm_mid.variance_floor == Catch::Approx (0.0025));
  REQUIRE (norm_mid.gain == Catch::Approx (2.3));
  REQUIRE (norm_mid.warmup_signals == 3);

  const auto inactivity_mid = BoundaryInactivityPolicyForKnobs (0.5, 0.5);
  REQUIRE (inactivity_mid.gap_score_scale == Catch::Approx (1.35));
  REQUIRE (inactivity_mid.gap_force_center == Catch::Approx (0.65));
  REQUIRE (inactivity_mid.gap_force_k == Catch::Approx (3.5));
  REQUIRE (inactivity_mid.inactivity_weight == Catch::Approx (0.15));

  REQUIRE (BoundaryHazardPrior (0.5, 0.5, 0.5)
           == Catch::Approx (0.0945));

  const auto support_mid
      = BoundarySupportPolicyForKnobs (0.5, 0.5, 0.5, 0.4, 0.8);
  REQUIRE (support_mid.support == Catch::Approx (0.6));
  REQUIRE (support_mid.support_gate == Catch::Approx (0.78));
  REQUIRE (support_mid.gap_gate == Catch::Approx (0.34));
  REQUIRE (support_mid.support_boost == Catch::Approx (1.15));

  const auto cp_mid = BoundaryChangePointPolicyForKnobs (0.5, 0.5, 0.6);
  REQUIRE (cp_mid.z_center == Catch::Approx (0.37));
  REQUIRE (cp_mid.z_center_eff == Catch::Approx (0.3145));
  REQUIRE (cp_mid.k == Catch::Approx (17.55));

  const auto pressure_mid = BoundaryPressurePolicyForKnobs (0.5, 0.5, 2.0);
  REQUIRE (pressure_mid.capacity == Catch::Approx (2.88));
  REQUIRE (pressure_mid.pressure == Catch::Approx (3.0));

  REQUIRE (AccumulatorDriftGain (0.5, 0.5) == Catch::Approx (0.5));
  REQUIRE (InterruptAbortAcceptanceMargin (0.5, 0.5, 0.5)
           == Catch::Approx (0.025));
  REQUIRE (AccumulatorTemporalContextAlpha (0.5, 0.5)
           == Catch::Approx (0.035));
  REQUIRE (RecentScoreHistoryLimit (0.5) == 1024);
  REQUIRE (RecentScoreHistoryLimit (1.0) > RecentScoreHistoryLimit (0.0));
  REQUIRE (RateTauStabilityExponentScale (0.5) == Catch::Approx (1.5));
  REQUIRE (AccumulatorDriftGain (1.0, 0.5)
           > AccumulatorDriftGain (0.0, 0.5));
  REQUIRE (AccumulatorTemporalContextAlpha (0.5, 1.0)
           < AccumulatorTemporalContextAlpha (0.5, 0.0));
}

TEST_CASE ("Interrupt gate policy helpers preserve midpoint and use knobs",
           "[core][knobs]")
{
  const auto weights = InterruptGateCandidateScoringWeights (0.5, 0.5, 0.5);
  REQUIRE (weights.coverage_raw == Catch::Approx (0.48));
  REQUIRE (weights.relevance_raw == Catch::Approx (0.31));
  REQUIRE (weights.redundancy_raw == Catch::Approx (0.19));
  REQUIRE (weights.coherence_raw == Catch::Approx (0.19));
  REQUIRE (weights.surprise_raw == Catch::Approx (0.32));
  REQUIRE (weights.coverage + weights.relevance + weights.redundancy
               + weights.coherence + weights.surprise
           == Catch::Approx (1.0));

  REQUIRE (InterruptBoundaryMultiplier (0.5, 0.5, 0.5)
           == Catch::Approx (1.518));
  REQUIRE (InterruptMuThreshold (0.5, 0.5, 0.5)
           == Catch::Approx (0.1248));
  REQUIRE (InterruptRefractoryTau (0.5, 0.5) == Catch::Approx (72.0));
  REQUIRE (InterruptRefractoryGain (0.5, 0.5) == Catch::Approx (0.125));
  REQUIRE (InterruptMaturityScale (0.0, 0.5) == Catch::Approx (1.7));

  REQUIRE (InterruptGateCandidateScoringWeights (1.0, 0.5, 0.5).coverage_raw
           > InterruptGateCandidateScoringWeights (0.0, 0.5, 0.5)
                 .coverage_raw);
  REQUIRE (InterruptGateCandidateScoringWeights (0.5, 1.0, 0.5).surprise_raw
           > InterruptGateCandidateScoringWeights (0.5, 0.0, 0.5)
                 .surprise_raw);
  REQUIRE (InterruptRefractoryTau (0.5, 1.0)
           > InterruptRefractoryTau (0.5, 0.0));
}

TEST_CASE ("Soft Anchor continuity gates derive from knobs", "[core][knobs]")
{
  REQUIRE (SoftAnchorActiveTTLSteps (1.0) > SoftAnchorActiveTTLSteps (0.0));
  REQUIRE (SoftAnchorRecentMemoryLimit (0.5, 0.5, 0.5) >= 24);
  REQUIRE (SoftAnchorRecentMemoryLimit (0.5, 0.5, 0.5) <= 40);

  REQUIRE (SoftAnchorSupportRawThreshold (1.0, 0.5, 0.5)
           > SoftAnchorSupportRawThreshold (0.0, 0.5, 0.5));
  REQUIRE (SoftAnchorSupportRawThreshold (0.5, 1.0, 0.5)
           < SoftAnchorSupportRawThreshold (0.5, 0.0, 0.5));
  REQUIRE (SoftAnchorSupportMarginThreshold (1.0, 0.5, 0.5)
           > SoftAnchorSupportMarginThreshold (0.0, 0.5, 0.5));
  REQUIRE (SoftAnchorSupportEntropyMax (1.0, 0.5, 0.5)
           < SoftAnchorSupportEntropyMax (0.0, 0.5, 0.5));

  REQUIRE (SoftAnchorContradictionSemanticThreshold (1.0, 0.5, 0.5)
           > SoftAnchorContradictionSemanticThreshold (0.0, 0.5, 0.5));
  REQUIRE (SoftAnchorContradictionEntityMax (1.0, 0.5, 0.5)
           < SoftAnchorContradictionEntityMax (0.0, 0.5, 0.5));

  REQUIRE (SoftAnchorDurableMarginThreshold (1.0, 0.5, 0.5)
           > SoftAnchorDurableMarginThreshold (0.0, 0.5, 0.5));
  REQUIRE (SoftAnchorDurableEntropyMax (1.0, 0.5, 0.5)
           < SoftAnchorDurableEntropyMax (0.0, 0.5, 0.5));

  const auto view_weights = SoftAnchorViewScoringWeights (
      0.5, 0.5, 0.5, 1.0, 0.5, 1.0);
  REQUIRE (view_weights.semantic + view_weights.entity + view_weights.full
           == Catch::Approx (1.0));
  const auto candidate_weights = SoftAnchorCandidateScoringWeights (
      0.5, 0.5, 0.5);
  REQUIRE (candidate_weights.view > 0.0);
  REQUIRE (candidate_weights.source > 0.0);
  REQUIRE (candidate_weights.recency > 0.0);
  REQUIRE (candidate_weights.support > 0.0);
  REQUIRE (candidate_weights.boundary > 0.0);
  REQUIRE (candidate_weights.contradiction > 0.0);
  REQUIRE (SoftAnchorSourceAffinity (0.5, 0.5, 0.5, true)
           == Catch::Approx (1.0));
  REQUIRE (SoftAnchorSourceAffinity (0.5, 1.0, 0.5, false)
           > SoftAnchorSourceAffinity (0.5, 0.0, 0.5, false));
  REQUIRE (SoftAnchorSingleViewEntityQuality (0.5, 0.5, 0.5)
           == Catch::Approx (0.5));
  REQUIRE (SoftAnchorActiveDecayMultiplier (0.5, 0.5, 0.5)
           == Catch::Approx (2.0));
  REQUIRE (SoftAnchorActiveDecayMultiplier (0.5, 0.5, 1.0)
           > SoftAnchorActiveDecayMultiplier (0.5, 0.5, 0.0));
  REQUIRE (SoftAnchorGenericScore (0.5, 0.5, 0.5, 0.5, 0.2, 0.8)
           == Catch::Approx (0.32));
  REQUIRE (SoftAnchorNoneScore (0.5, 0.5, 0.5, 0.3, 0.2, 0.4, 0.1)
           == Catch::Approx (0.5));
  REQUIRE (SoftAnchorNewScore (0.5, 0.5, 0.5, 0.8, 0.4, 0.6, 0.2)
           == Catch::Approx (0.62));

  const auto decision_mid = SoftAnchorDecisionPolicyForKnobs (0.5, 0.5, 0.5);
  REQUIRE (decision_mid.keep_count >= 2);
  REQUIRE (decision_mid.durable_support_required >= 2);
  REQUIRE (SoftAnchorDecisionPolicyForKnobs (0.5, 1.0, 0.5)
               .durable_support_required
           <= SoftAnchorDecisionPolicyForKnobs (0.5, 0.0, 0.5)
                  .durable_support_required);
}

TEST_CASE ("Graph build thresholds derive from knobs", "[core][knobs]")
{
  const double same_event_mid
      = GraphBuildSameEventBoundaryThreshold (0.5, 0.5, 0.5);
  REQUIRE (same_event_mid > 0.25);
  REQUIRE (same_event_mid < 0.35);
  REQUIRE (GraphBuildSameEventBoundaryThreshold (0.5, 1.0, 0.5)
           < GraphBuildSameEventBoundaryThreshold (0.5, 0.0, 0.5));
  REQUIRE (GraphBuildSameEventBoundaryThreshold (0.5, 0.5, 1.0)
           > GraphBuildSameEventBoundaryThreshold (0.5, 0.5, 0.0));

  REQUIRE (CausalDriftWeightScale (1.0) > CausalDriftWeightScale (0.0));

  REQUIRE (ContradictionThreshold () == Catch::Approx (-0.5));
  REQUIRE (ContradictionThreshold (0.5, 1.0, 0.5)
           > ContradictionThreshold (0.5, 0.0, 0.5));
  REQUIRE (ContradictionThreshold (1.0, 0.5, 0.5)
           < ContradictionThreshold (0.0, 0.5, 0.5));

  REQUIRE (SupersessionSimilarityThreshold (1.0, 0.5, 0.5)
           > SupersessionSimilarityThreshold (0.0, 0.5, 0.5));
  REQUIRE (SupersessionSimilarityThreshold (0.5, 1.0, 0.5)
           < SupersessionSimilarityThreshold (0.5, 0.0, 0.5));
  REQUIRE (SupersessionDuplicateThreshold (0.5, 0.5, 0.5)
           > SupersessionSimilarityThreshold (0.5, 0.5, 0.5));
  REQUIRE (SupersessionEdgeWeight (
               SupersessionDuplicateThreshold (0.5, 0.5, 0.5),
               0.5, 0.5, 0.5)
           == Catch::Approx (1.0));
  REQUIRE (SupersessionMaxEdges (0.5, 1.0, 0.5)
           >= SupersessionMaxEdges (0.5, 0.0, 0.5));
  REQUIRE (SupersessionCandidateLimit (0.5, 0.5, 0.5) > 0);
  REQUIRE (RetrievalSupersededMemoryPenalty (0.5, 1.0, 0.5)
           > RetrievalSupersededMemoryPenalty (0.5, 0.0, 0.5));

  REQUIRE (LabelCooccurrenceEdgeWeight (0.5, 1.0, 0.5)
           > LabelCooccurrenceEdgeWeight (0.5, 0.0, 0.5));
  REQUIRE (RetrievalProceduralSeedMinScore (1.0, 0.5, 0.5)
           > RetrievalProceduralSeedMinScore (0.0, 0.5, 0.5));
  REQUIRE (RetrievalProceduralSeedMinScore (0.5, 1.0, 0.5)
           < RetrievalProceduralSeedMinScore (0.5, 0.0, 0.5));
}

TEST_CASE ("Neutral retrieval knobs map to judged mixed-media optimum",
           "[core][knobs][retrieval]")
{
  REQUIRE (RetrievalFocusBias (0.5) == Catch::Approx (FocusBias (0.75)));
  REQUIRE (RetrievalSensitivityBias (0.5)
           == Catch::Approx (SensitivityBias (0.25)));
  REQUIRE (RetrievalStabilityBias (0.5) == Catch::Approx (0.75));
  REQUIRE (RetrievalMaxResults (0.5) == MaxResults (0.75));
  const int previous_winner_items = static_cast<int> (std::round (
      Lerp (20.0, 8.0, FocusBias (0.75)) * Lerp (1.08, 0.92, 0.75)));
  REQUIRE (RetrievalGraphExpandedRagMaxItems (0.5, 0.5)
           == previous_winner_items);
}

TEST_CASE ("Retrieval source pressure and scoring helpers derive from knobs",
           "[core][knobs][retrieval]")
{
  REQUIRE (RetrievalPressureRampLowScale (1.0)
           < RetrievalPressureRampLowScale (0.0));
  REQUIRE (RetrievalPressureGateLowScale (1.0)
           < RetrievalPressureGateLowScale (0.0));

  REQUIRE (RetrievalSourceFreshnessTauSeconds (0.5, 0.5, 1.0)
           > RetrievalSourceFreshnessTauSeconds (0.5, 0.5, 0.0));
  REQUIRE (RetrievalSourceFreshnessTauSeconds (1.0, 0.5, 0.5)
           < RetrievalSourceFreshnessTauSeconds (0.0, 0.5, 0.5));
  REQUIRE (RetrievalSourceFreshnessTauSeconds (0.5, 1.0, 0.5)
           < RetrievalSourceFreshnessTauSeconds (0.5, 0.0, 0.5));
  REQUIRE (RetrievalSourceFreshnessWeight (0.5, 0.5, 1.0)
           < RetrievalSourceFreshnessWeight (0.5, 0.5, 0.0));
  REQUIRE (RetrievalSourceFreshnessWeight (1.0, 0.5, 0.5)
           > RetrievalSourceFreshnessWeight (0.0, 0.5, 0.5));
  REQUIRE (RetrievalSourceFreshnessWeight (0.5, 1.0, 0.5)
           > RetrievalSourceFreshnessWeight (0.5, 0.0, 0.5));
  REQUIRE (RetrievalSourceConfidenceThreshold (1.0, 0.5, 0.5, 1.0)
           > RetrievalSourceConfidenceThreshold (0.0, 0.5, 0.5, 1.0));

  REQUIRE (RetrievalGraphExpandedRagTemporalRankScore (0.5, 0.5, 0.5, 0)
           > RetrievalGraphExpandedRagTemporalRankScore (0.5, 0.5, 0.5, 1));
  REQUIRE (RetrievalContextReinstatementAlpha (0.5, 0.5, 1.0)
           < RetrievalContextReinstatementAlpha (0.5, 0.5, 0.0));
  REQUIRE (RetrievalGraphExpansionFanout (1.0, 0.5, 0.5)
           < RetrievalGraphExpansionFanout (0.0, 0.5, 0.5));
  REQUIRE (RetrievalGraphExpansionFanout (0.5, 1.0, 0.5)
           > RetrievalGraphExpansionFanout (0.5, 0.0, 0.5));
  REQUIRE (RetrievalProceduralSeedReserveCount (0.0, 1.0, 0.0, 2)
           >= RetrievalProceduralSeedReserveCount (1.0, 0.0, 1.0, 2));
  REQUIRE (RetrievalProceduralSeedFanout (0.5, 1.0, 0.5)
           > RetrievalProceduralSeedFanout (0.5, 0.0, 0.5));
  REQUIRE (RetrievalProceduralSeedFanout (1.0, 0.5, 0.5)
           <= RetrievalProceduralSeedFanout (0.0, 0.5, 0.5));
  REQUIRE (RetrievalSourceConfidenceThreshold (0.5, 1.0, 0.5, 1.0)
           < RetrievalSourceConfidenceThreshold (0.5, 0.0, 0.5, 1.0));
  REQUIRE (RetrievalSourceConfidenceThreshold (0.5, 0.5, 1.0, 1.0)
           < RetrievalSourceConfidenceThreshold (0.5, 0.5, 0.0, 1.0));
  REQUIRE (RetrievalSeedFallbackSourceConfidence (0.5, 0.5, 0.5) < 1.0);
  REQUIRE (RetrievalSeedFallbackSourceConfidence (0.5, 0.5, 0.5)
           > 0.50);
  REQUIRE (RetrievalSourceBackedBoostFloor (0.5, 0.5, 0.5) > 0.01);
  REQUIRE (RetrievalSourceBackedBoostFloor (1.0, 0.5, 0.5)
           > RetrievalSourceBackedBoostFloor (0.0, 0.5, 0.5));
  REQUIRE (RetrievalContextMix (0.5, 0.5, 0.5) > 0.18);
  REQUIRE (RetrievalContextMix (0.5, 1.0, 0.5)
           > RetrievalContextMix (0.5, 0.0, 0.5));
  REQUIRE (RetrievalContextMix (0.5, 0.5, 1.0)
           > RetrievalContextMix (0.5, 0.5, 0.0));
  REQUIRE (RetrievalContextMix (1.0, 0.5, 0.5)
           < RetrievalContextMix (0.0, 0.5, 0.5));

  REQUIRE (RetrievalStalePenaltyStrongMultiplier (0.5, 0.5, 0.5)
           == Catch::Approx (1.75));
  REQUIRE (RetrievalStalePenaltyStrongMultiplier (0.5, 0.5, 1.0)
           > RetrievalStalePenaltyStrongMultiplier (0.5, 0.5, 0.0));

  REQUIRE (RetrievalVectorDistanceScore (0.0, 0.5, 0.5, 0.5)
           == Catch::Approx (1.0));
  REQUIRE (RetrievalVectorDistanceScore (1.0, 0.5, 0.5, 0.5)
           == Catch::Approx (0.5));
  REQUIRE (RetrievalVectorDistanceScore (1.0, 1.0, 0.5, 0.5)
           < RetrievalVectorDistanceScore (1.0, 0.0, 0.5, 0.5));
  REQUIRE (RetrievalSparseKeySize (0.5, 0.5, 0.5) == 40);
  REQUIRE (RetrievalSparseKeySize (1.0, 0.5, 0.5)
           > RetrievalSparseKeySize (0.0, 0.5, 0.5));
  REQUIRE (RetrievalSparseKeySize (0.5, 1.0, 0.5)
           > RetrievalSparseKeySize (0.5, 0.0, 0.5));
  REQUIRE (RetrievalSparseKeySize (0.5, 0.5, 1.0)
           < RetrievalSparseKeySize (0.5, 0.5, 0.0));
  REQUIRE (RetrievalTokenOverlapQueryWeight (1.0, 0.5, 0.5)
           > RetrievalTokenOverlapQueryWeight (0.0, 0.5, 0.5));
  REQUIRE (RetrievalRouteTokenMinChars (0.5, 0.5, 0.5) == 3);
  REQUIRE (RetrievalRouteTokenMinChars (1.0, 0.0, 1.0)
           > RetrievalRouteTokenMinChars (0.0, 1.0, 0.0));
  REQUIRE (RetrievalDurableSourceSupportSaturationCount (0.0, 0.5)
           > RetrievalDurableSourceSupportSaturationCount (1.0, 0.5));

  const auto low_t_reconstruct
      = RetrievalConstructiveReconstructionPolicy (0.5, 0.5, 0.0);
  const auto high_t_reconstruct
      = RetrievalConstructiveReconstructionPolicy (0.5, 0.5, 1.0);
  REQUIRE (high_t_reconstruct.max_blend < low_t_reconstruct.max_blend);
  REQUIRE (high_t_reconstruct.query_weight < low_t_reconstruct.query_weight);
  REQUIRE (RetrievalReconstructionUpdateCount (0.5, 0.5, 0.5) == 1);
  REQUIRE (RetrievalReconstructionUpdateCount (0.0, 1.0, 0.0)
           > RetrievalReconstructionUpdateCount (1.0, 0.0, 1.0));
}

TEST_CASE ("Retrieval hydration and expansion limits derive from knobs",
           "[core][knobs][retrieval]")
{
  REQUIRE (HydratedSoftAnchorLimit (0.5, 0.5, 0.5) == 3);
  REQUIRE (RetrievalLinkedHydrationCandidateLimit (0.5, 0.5, 0.5)
           == 12);

  const auto linked_weights
      = RetrievalLinkedHydrationOrderingWeights (0.5, 0.5, 0.5);
  REQUIRE (linked_weights.first == Catch::Approx (0.82));
  REQUIRE (linked_weights.first + linked_weights.second
           == Catch::Approx (1.0));

  REQUIRE (RetrievalDurableSourceLinkFanout (0.5, 1.0, 0.5, 4)
           > RetrievalDurableSourceLinkFanout (0.5, 0.0, 0.5, 4));
  const auto evidence_mid
      = RetrievalGraphExpansionEvidenceCounts (0.5, 0.5, 0.5);
  REQUIRE (evidence_mid.min_association_count == 2);
  REQUIRE (evidence_mid.min_label_count == 0);

  REQUIRE (ObservedRetentionHistoryLimit (0.5, 0.5, 1.0)
           > ObservedRetentionHistoryLimit (0.5, 0.5, 0.0));
  REQUIRE (RecentRetrievedIdWindow (0.5, 0.5, 1.0)
           > RecentRetrievedIdWindow (0.5, 0.5, 0.0));
}

TEST_CASE ("Retrieval candidate blend scoring policies derive from knobs",
           "[core][knobs][retrieval]")
{
  const auto blend_low
      = RetrievalCandidateBlendScoringWeights (0.0, 0.0, 0.5);
  const auto blend_high
      = RetrievalCandidateBlendScoringWeights (1.0, 1.0, 0.5);
  REQUIRE (blend_high.context > blend_low.context);
  REQUIRE (blend_high.procedural > blend_low.procedural);
  REQUIRE (blend_high.label_boost > blend_low.label_boost);

  REQUIRE (RetrievalPredictiveBonusWeight (1.0, 0.5, 0.5)
           > RetrievalPredictiveBonusWeight (0.0, 0.5, 0.5));
  REQUIRE (RetrievalPredictiveBonusWeight (0.5, 1.0, 0.5)
           > RetrievalPredictiveBonusWeight (0.5, 0.0, 0.5));

  REQUIRE (RetrievalCompetitionSuppressionBase (0.5, 0.5, 0.5)
           == Catch::Approx (0.055));
  REQUIRE (RetrievalCompetitionSuppressionBase (1.0, 0.5, 0.5)
           < RetrievalCompetitionSuppressionBase (0.0, 0.5, 0.5));
  REQUIRE (RetrievalCompetitionSuppressionBase (0.5, 1.0, 0.5)
           > RetrievalCompetitionSuppressionBase (0.5, 0.0, 0.5));
  REQUIRE (RetrievalCompetitionSuppressionBase (0.5, 0.5, 1.0)
           < RetrievalCompetitionSuppressionBase (0.5, 0.5, 0.0));
  REQUIRE (RetrievalCompetitionWinnerCount (0.5) == 5);
  REQUIRE (RetrievalCompetitionWinnerCount (0.0)
           > RetrievalCompetitionWinnerCount (1.0));
  REQUIRE (RetrievalCompetitionInhibitionRadius (0.5)
           == Catch::Approx (0.64));
  REQUIRE (RetrievalCompetitionInhibitionRadius (1.0)
           > RetrievalCompetitionInhibitionRadius (0.0));
  REQUIRE (RetrievalCompetitionRecoverySeconds (0.5)
           == Catch::Approx (1050.0));
  REQUIRE (RetrievalCompetitionRecoverySeconds (1.0)
           > RetrievalCompetitionRecoverySeconds (0.0));
  const auto influence_mid = InfluenceFeedbackPolicyForKnobs (0.5, 0.5, 0.5);
  REQUIRE (influence_mid.sustain_window == Catch::Approx (4.0));
  REQUIRE (influence_mid.sustain_alpha == Catch::Approx (0.4));
  REQUIRE (influence_mid.contextual_gain_weight == Catch::Approx (0.50));
  REQUIRE (influence_mid.predictive_similarity_weight == Catch::Approx (0.40));
  REQUIRE (influence_mid.drift_weight == Catch::Approx (0.30));
  REQUIRE (InfluenceFeedbackPolicyForKnobs (0.5, 0.5, 1.0).sustain_window
           > InfluenceFeedbackPolicyForKnobs (0.5, 0.5, 0.0).sustain_window);

  const auto affect_mid = AffectDriveWeights (0.5);
  REQUIRE (affect_mid.arousal_weight
           == Catch::Approx (affect_mid.emotion_weight));
  REQUIRE (affect_mid.arousal_weight + affect_mid.emotion_weight
               + affect_mid.salience_weight
           == Catch::Approx (1.0));
  REQUIRE (AffectDriveWeights (1.0).salience_weight
           > AffectDriveWeights (0.0).salience_weight);

  const auto memory_affect_mid = RetrievalMemoryAffectScoringWeights (0.5);
  REQUIRE (memory_affect_mid.emotion + memory_affect_mid.arousal
           == Catch::Approx (1.0));
  REQUIRE (RetrievalMemoryAffectScoringWeights (1.0).arousal
           > RetrievalMemoryAffectScoringWeights (0.0).arousal);

  const auto routine_recency_mid
      = RetrievalRoutineRecencyAdjustmentPolicy (0.5, 0.5, 0.5);
  REQUIRE (routine_recency_mid.routine_boost_weight == Catch::Approx (0.40));
  REQUIRE (routine_recency_mid.routine_recency_penalty_weight
           == Catch::Approx (0.16));
  REQUIRE (routine_recency_mid.recency_boost_weight == Catch::Approx (0.46));
  REQUIRE (routine_recency_mid.recency_routine_penalty_weight
           == Catch::Approx (0.18));
  REQUIRE (routine_recency_mid.stale_routine_penalty_weight
           == Catch::Approx (0.65));
  REQUIRE (routine_recency_mid.stale_recency_boost_weight
           == Catch::Approx (0.45));
  REQUIRE (RetrievalRoutineRecencyAdjustmentPolicy (0.5, 0.0, 1.0)
               .routine_boost_weight
           > RetrievalRoutineRecencyAdjustmentPolicy (0.5, 1.0, 0.0)
                 .routine_boost_weight);
  REQUIRE (RetrievalRoutineRecencyAdjustmentPolicy (0.0, 1.0, 0.0)
               .recency_boost_weight
           > RetrievalRoutineRecencyAdjustmentPolicy (1.0, 0.0, 1.0)
                 .recency_boost_weight);
}

TEST_CASE ("Neuromodulator and serial floor policies derive from knobs",
           "[core][knobs]")
{
  const auto mid = NeuromodulatorPolicyForKnobs (0.5, 0.5, 0.5);
  REQUIRE (mid.ach_base >= 0.0);
  REQUIRE (mid.ach_base <= 1.0);
  REQUIRE (mid.ne_base >= 0.0);
  REQUIRE (mid.ne_base <= 1.0);
  REQUIRE (mid.da_base >= 0.0);
  REQUIRE (mid.da_base <= 1.0);
  REQUIRE (mid.retrieval_pressure_depth > 0.0);
  REQUIRE (mid.oscillation_rate > 0.0);

  REQUIRE (NeuromodulatorPolicyForKnobs (0.5, 1.0, 0.5).ach_base
           > NeuromodulatorPolicyForKnobs (0.5, 0.0, 0.5).ach_base);
  REQUIRE (NeuromodulatorPolicyForKnobs (0.5, 1.0, 0.5).ne_base
           > NeuromodulatorPolicyForKnobs (0.5, 0.0, 0.5).ne_base);
  REQUIRE (NeuromodulatorPolicyForKnobs (1.0, 0.5, 0.5).da_base
           > NeuromodulatorPolicyForKnobs (0.0, 0.5, 0.5).da_base);
  REQUIRE (NeuromodulatorPolicyForKnobs (0.5, 0.5, 1.0).oscillation_rate
           < NeuromodulatorPolicyForKnobs (0.5, 0.5, 0.0).oscillation_rate);

  REQUIRE (SerialMiddleMultiplierFloor (0.5, 0.5) == Catch::Approx (0.10));
  REQUIRE (SerialMiddleMultiplierFloor (1.0, 0.5)
           > SerialMiddleMultiplierFloor (0.0, 0.5));
  REQUIRE (SerialMiddleMultiplierFloor (0.5, 1.0)
           < SerialMiddleMultiplierFloor (0.5, 0.0));
}

TEST_CASE ("Consolidation cluster policy derives from knobs",
           "[core][knobs]")
{
  REQUIRE (ConsolidationMaxClusters (0.5, 0.5, 0.5) == 100);
  REQUIRE (ConsolidationMaxClusters (0.0, 0.5, 0.5)
           > ConsolidationMaxClusters (1.0, 0.5, 0.5));
  REQUIRE (ConsolidationMaxClusters (0.5, 0.5, 1.0)
           > ConsolidationMaxClusters (0.5, 0.5, 0.0));
}

TEST_CASE ("Memory trace policy derives from knobs with neutral midpoint",
           "[core][knobs]")
{
  const auto tau = MemoryTraceTauPolicy (0.5, 0.5, 0.5);
  REQUIRE (tau.fast == Catch::Approx (0.10));
  REQUIRE (tau.medium == Catch::Approx (0.50));
  REQUIRE (tau.slow == Catch::Approx (2.00));
  REQUIRE (tau.ultra == Catch::Approx (8.00));
  REQUIRE (MemoryTraceTauPolicy (0.5, 0.5, 1.0).slow > tau.slow);
  REQUIRE (MemoryTraceTauPolicy (0.5, 1.0, 0.5).fast > tau.fast);

  REQUIRE (MemoryTraceCoupling (0.5, 0.5, 0.5)
           == Catch::Approx (0.10));
  REQUIRE (MemoryTraceCoupling (0.5, 0.5, 1.0)
           > MemoryTraceCoupling (0.5, 0.5, 0.0));

  const auto weights = MemoryTraceWeightPolicy (0.5, 0.5, 0.5);
  REQUIRE (weights.fast == Catch::Approx (0.275));
  REQUIRE (weights.medium == Catch::Approx (0.25));
  REQUIRE (weights.slow == Catch::Approx (0.275));
  REQUIRE (weights.ultra == Catch::Approx (0.20));
  REQUIRE (MemoryTraceWeightPolicy (0.5, 0.5, 1.0).slow
           > MemoryTraceWeightPolicy (0.5, 0.5, 0.0).slow);

  const auto initial = MemoryInitialTracePolicy (0.5, 0.5, 0.5);
  REQUIRE (initial.fast == Catch::Approx (1.0));
  REQUIRE (initial.medium == Catch::Approx (0.0));
  REQUIRE (initial.slow == Catch::Approx (0.0));
  REQUIRE (initial.ultra == Catch::Approx (0.0));
  REQUIRE (MemoryInitialTracePolicy (0.5, 0.5, 1.0).slow
           > initial.slow);
  REQUIRE (MemoryInitialStrengthPolicy (0.5, 0.5, 0.5)
           == Catch::Approx (1.0));
  REQUIRE (MemoryInitialStabilityPolicy (0.5, 0.5, 0.5)
           == Catch::Approx (1.0));
  REQUIRE (MemoryInitialStrengthPolicy (0.0, 1.0, 0.0)
           < MemoryInitialStrengthPolicy (0.5, 0.5, 0.5));
  REQUIRE (MemoryInitialStabilityPolicy (0.0, 1.0, 0.0)
           < MemoryInitialStabilityPolicy (0.5, 0.5, 0.5));
}

TEST_CASE ("MaxWaitDrift knob function", "[core][knobs]")
{
  // max_wait_drift(F_eff) = lerp(1.2, 0.30, F_eff)
  // Higher focus = lower max drift = more aggressive forced checks

  // F=0: lenient (1.2)
  REQUIRE (MaxWaitDrift (0.0) == Catch::Approx (1.2));
  // F=1: strict (0.30)
  REQUIRE (MaxWaitDrift (1.0) == Catch::Approx (0.30));
  // F=0.5: midpoint after bias (0.84)
  REQUIRE (MaxWaitDrift (0.5) == Catch::Approx (0.84));

  // Monotonic: higher F means lower max wait
  REQUIRE (MaxWaitDrift (0.3) > MaxWaitDrift (0.7));
}

TEST_CASE ("TauNovelty follows spec: lerp(0.12, 0.32, F_eff) * (1 - 0.12S_eff) * (1 + "
           "0.25T)",
           "[core][knobs]")
{
  // tau_novelty = lerp(0.12, 0.32, F_eff) * (1 - 0.12S_eff) * (1 + 0.25T)
  // Reference: algorithms.md Section 8.1

  // At F=0, S=0, T=0: 0.12 * 1.0 * 1.0 = 0.12
  REQUIRE (TauNovelty (0.0, 0.0, 0.0) == Catch::Approx (0.12));

  // At F=1, S=0, T=0: 0.32 * 1.0 * 1.0 = 0.32
  REQUIRE (TauNovelty (1.0, 0.0, 0.0) == Catch::Approx (0.32));

  // At F=0, S=1, T=0: 0.12 * 0.88 * 1.0 = 0.1056
  REQUIRE (TauNovelty (0.0, 1.0, 0.0) == Catch::Approx (0.1056));

  // At F=0, S=0, T=1: 0.12 * 1.0 * 1.25 = 0.15
  REQUIRE (TauNovelty (0.0, 0.0, 1.0) == Catch::Approx (0.15));

  // At F=1, S=1, T=1: 0.32 * 0.88 * 1.25 = 0.352
  REQUIRE (TauNovelty (1.0, 1.0, 1.0) == Catch::Approx (0.352));

  // Mid-point: F=0.5, S=0.5, T=0.5
  // F_eff=0.4, S_eff=0.4
  // lerp(0.12, 0.32, 0.4) = 0.20
  // (1 - 0.12*0.4) = 0.952
  // (1 + 0.25*0.5) = 1.125
  // 0.20 * 0.952 * 1.125 = 0.2142
  REQUIRE (TauNovelty (0.5, 0.5, 0.5) == Catch::Approx (0.2142));
}

TEST_CASE ("RetrievalThreshold follows spec: lerp(0.12, 0.45, F_eff)",
           "[core][knobs]")
{
  // retrieval_thresh(F_eff) = lerp(0.12, 0.45, F_eff)
  // Reference: algorithms.md Section 8.1

  // F=0: permissive (0.12)
  REQUIRE (RetrievalThreshold (0.0) == Catch::Approx (0.12));

  // F=1: strict (0.45)
  REQUIRE (RetrievalThreshold (1.0) == Catch::Approx (0.45));

  // F=0.5: midpoint after bias (0.252)
  REQUIRE (RetrievalThreshold (0.5) == Catch::Approx (0.252));

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

TEST_CASE ("Consolidation label budgets derive from knobs",
           "[core][knobs]")
{
  REQUIRE (ShallowConsolidationMaxLabels (0.5, 0.5, 0.5) == 3);
  REQUIRE (ShallowConsolidationLabelMinSimilarity (0.5, 0.5, 0.5)
           == Catch::Approx (0.441));

  REQUIRE (ShallowConsolidationMaxLabels (0.0, 1.0, 0.5)
           > ShallowConsolidationMaxLabels (1.0, 0.0, 0.5));
  REQUIRE (ShallowConsolidationLabelMinSimilarity (1.0, 0.5, 0.5)
           > ShallowConsolidationLabelMinSimilarity (0.0, 0.5, 0.5));
  REQUIRE (ShallowConsolidationLabelMinSimilarity (0.5, 1.0, 0.5)
           < ShallowConsolidationLabelMinSimilarity (0.5, 0.0, 0.5));

  REQUIRE (ConsolidationCandidateTagWeight (0.5, 0.5, 0.5)
           == Catch::Approx (0.16));
  REQUIRE (ConsolidationCandidateTagWeight (0.5, 1.0, 0.5)
           > ConsolidationCandidateTagWeight (0.5, 0.0, 0.5));

}

TEST_CASE ("Runtime adaptation constants derive from knobs", "[core][knobs]")
{
  REQUIRE (BlenderRLSObservationTau (0.5) == Catch::Approx (50.0));
  REQUIRE (BlenderRLSForgettingFactor (0.5) == Catch::Approx (0.945));
  REQUIRE (BlenderUpdateInterval (0.5) == 5);
  REQUIRE (BlenderRLSObservationTau (1.0) > BlenderRLSObservationTau (0.0));
  REQUIRE (BlenderRLSForgettingFactor (1.0)
           > BlenderRLSForgettingFactor (0.0));

  const auto outcome = BlenderOutcomeScoringWeights (0.5, 0.5, 0.5);
  REQUIRE (outcome.used == Catch::Approx (0.48));
  REQUIRE (outcome.predictive == Catch::Approx (0.38));
  REQUIRE (outcome.uncertainty == Catch::Approx (0.30));
  REQUIRE (outcome.user == Catch::Approx (0.10));
  REQUIRE (BlenderOutcomeScoringWeights (0.5, 1.0, 0.5).user
           > BlenderOutcomeScoringWeights (0.5, 0.0, 0.5).user);
  REQUIRE (BlenderOutcomeAlpha (0.0) > BlenderOutcomeAlpha (1.0));

  REQUIRE (PredictionHorizon (0.5) == 4);
  REQUIRE (PredictionConfidenceThreshold (0.5) == Catch::Approx (0.46));
  REQUIRE (PredictivePreActivationDecay (0.5) == Catch::Approx (0.50));
  REQUIRE (PredictiveBaseDelta (0.5, 0.5, 0.5)
           == Catch::Approx (0.016));
  REQUIRE (PredictiveSurpriseSensitivity (0.5, 0.5)
           == Catch::Approx (0.50));
  REQUIRE (PredictiveSurpriseUpdateRate (0.5, 0.5)
           == Catch::Approx (0.044));
  REQUIRE (PredictiveDeltaCap (0.5, 0.5, 0.5)
           == Catch::Approx (0.20));
  REQUIRE (PredictionConfidenceThreshold (1.0)
           > PredictionConfidenceThreshold (0.0));
  REQUIRE (PredictiveSurpriseUpdateRate (1.0, 0.5)
           > PredictiveSurpriseUpdateRate (0.0, 0.5));
}

TEST_CASE ("Graph and memory policy constants derive from knobs",
           "[core][knobs]")
{
  REQUIRE (MemoryContextualGainWindow (0.5, 0.5, 0.5)
           == Catch::Approx (20.0));
  REQUIRE (MemoryTraceCount (0.5, 0.5, 0.5) == 3);
  const auto trace_fallback = MemoryStoredTraceFallbackPolicy (0.5, 0.5, 0.5);
  REQUIRE (trace_fallback.fast == Catch::Approx (1.0));
  REQUIRE (trace_fallback.medium == Catch::Approx (0.50));
  REQUIRE (trace_fallback.slow == Catch::Approx (0.20));
  REQUIRE (trace_fallback.ultra == Catch::Approx (0.05));
  REQUIRE (MemoryTraceCount (0.5, 0.5, 1.0)
           > MemoryTraceCount (0.5, 0.5, 0.0));

  REQUIRE (GraphBuildSequentialTauSeconds (0.5, 0.5, 0.5)
           == Catch::Approx (35.0));
  REQUIRE (GraphBuildSequentialWeight (0.5, 0.5, 0.5, 0.0, 0.0)
           == Catch::Approx (1.0));
  REQUIRE (GraphBuildSequentialWeight (0.5, 0.5, 0.5, 0.0, 0.5)
           == Catch::Approx (0.5));
  REQUIRE (GraphBuildSequentialTauSeconds (0.5, 0.5, 1.0)
           > GraphBuildSequentialTauSeconds (0.5, 0.5, 0.0));
  REQUIRE (DerivedSourceEdgeWeight (0.5, 0.5, 0.5)
           == Catch::Approx (1.0));
  REQUIRE (DerivedSourceEdgeWeight (0.5, 0.5, 1.0)
           >= DerivedSourceEdgeWeight (0.5, 0.5, 0.0));
  REQUIRE (DerivedSourceFallbackEdgeWeight (0.5, 0.5, 0.0)
           == Catch::Approx (DerivedSourceEdgeWeight (0.5, 0.5, 0.0)));

  // Compatibility calibration retained without a production packet-pair
  // writer so existing public-header consumers do not break.
  REQUIRE (ReinforcementFallbackContextualSupport (0.5, 0.5, 0.5)
           == Catch::Approx (0.50));
  REQUIRE (ReinforcementFallbackContextualSupport (0.5, 0.0, 0.5)
           > ReinforcementFallbackContextualSupport (0.5, 1.0, 0.5));
  REQUIRE (MetricRarityStabilityScale (0.5) == Catch::Approx (0.90));
  REQUIRE (MetricRarityStabilityScale (1.0)
           < MetricRarityStabilityScale (0.0));
  REQUIRE (MetricUtilitySensitivityScale (0.5) == Catch::Approx (0.88));
  REQUIRE (MetricUtilitySensitivityScale (1.0)
           < MetricUtilitySensitivityScale (0.0));
}

TEST_CASE ("Reconsolidation and cascade gates derive from knobs",
           "[core][knobs]")
{
  REQUIRE (ReconsolidationDriftClamp (0.5, 0.5, 0.5)
           == Catch::Approx (0.30));
  REQUIRE (ReconsolidationDriftSkipEpsilon (0.5, 0.5, 0.5)
           == Catch::Approx (0.001));
  REQUIRE (RippleStrengthMin (0.5, 0.5, 0.5)
           == Catch::Approx (0.01));
  REQUIRE (RippleDriftCapFactor (0.5, 0.5, 0.5)
           == Catch::Approx (0.50));
  REQUIRE (ReconsolidationRippleReconstructionLimit (0.5, 0.5, 0.5)
           == 0);
  REQUIRE (ReconsolidationPrimaryReconstructionLimit (0.5, 0.5, 0.5)
           == 1);
  REQUIRE (ReconsolidationUncertaintyRelevanceWeight (0.5, 0.5, 0.5)
           == Catch::Approx (0.50));
  REQUIRE (ReconsolidationPrimaryDriftMagnitude (0.5, 0.5, 0.5,
                                                 1.0, 1.0, 1.0)
           == Catch::Approx (0.0275));
  REQUIRE (ReconsolidationPrimaryDriftMagnitude (0.5, 0.5, 0.0,
                                                 1.0, 1.0, 1.0)
           > ReconsolidationPrimaryDriftMagnitude (0.5, 0.5, 1.0,
                                                   1.0, 1.0, 1.0));
  REQUIRE (ReconsolidationPrimaryDriftMagnitude (0.5, 1.0, 0.5,
                                                 1.0, 1.0, 1.0)
           > ReconsolidationPrimaryDriftMagnitude (0.5, 0.0, 0.5,
                                                   1.0, 1.0, 1.0));
  REQUIRE (ReconsolidationDriftClamp (0.5, 0.5, 0.0)
           > ReconsolidationDriftClamp (0.5, 0.5, 1.0));
  REQUIRE (RippleStrengthMin (0.5, 0.0, 0.5)
           > RippleStrengthMin (0.5, 1.0, 0.5));
  REQUIRE (ReconsolidationRippleReconstructionLimit (0.0, 1.0, 0.0)
           > ReconsolidationRippleReconstructionLimit (1.0, 0.0, 1.0));
  REQUIRE (ReconsolidationPrimaryReconstructionLimit (0.0, 1.0, 0.0)
           > ReconsolidationPrimaryReconstructionLimit (1.0, 0.0, 1.0));

  REQUIRE (CascadeIntensityFloor (0.5, 0.5, 0.5)
           == Catch::Approx (0.10));
  REQUIRE (CascadeIntensityFloor (0.5, 0.0, 0.5)
           > CascadeIntensityFloor (0.5, 1.0, 0.5));
  REQUIRE (AffectThresholdGain (0.5) == Catch::Approx (0.04));
  REQUIRE (AffectThresholdGain (1.0) > AffectThresholdGain (0.0));
  REQUIRE (ThresholdObservedQuantile (0.5, 0.5, 0.5)
           == Catch::Approx (0.90));
  REQUIRE (ThresholdObservedQuantile (1.0, 0.5, 0.5)
           > ThresholdObservedQuantile (0.0, 0.5, 0.5));
  REQUIRE (ThresholdObservedQuantile (0.5, 1.0, 0.5)
           < ThresholdObservedQuantile (0.5, 0.0, 0.5));
}
