#pragma once

namespace cortext::operations::constants
{
// Shared gains and base coefficients (see validation.md references in Alg §0.5)
constexpr double kAlphaFBase = 0.10;  // Focus feedback base gain
constexpr double kBetaFBase = 0.05;   // Focus feedback narrowing/widening gain
constexpr double kEtaBase = 0.10;     // Sensitivity feedback base gain
constexpr double kGammaTBase = 0.05;  // Stability feedback base scaled by T

// Generic small/medium gains reused across operations
constexpr double kGainSmall = 0.05;
constexpr double kGainMedium = 0.10;

// Hysteresis band
constexpr double kHysteresisBandMin = 0.02;
constexpr double kHysteresisBandMax = 0.25;

// Stability half-life adjustment clamp bounds
constexpr double kHalfLifeAdjClampMin = -0.25;
constexpr double kHalfLifeAdjClampMax = 0.25;
constexpr double kQuarter = 0.25;

// Influence weighting coefficients
constexpr double kLambda1 = 0.5;
constexpr double kLambda2 = 0.4;
constexpr double kLambda3 = 0.3;

// Sustain window bounds for EWMA horizon
constexpr double kSustainWindowMin = 3.0;
constexpr double kSustainWindowMax = 5.0;

// Common numeric constants
constexpr double kNormEpsilon = 1e-9;
constexpr double kNormalizedMin = 0.0;
constexpr double kNormalizedMax = 1.0;
constexpr double kOneHalf = 0.5;
constexpr double kTwo = 2.0;

// Additional weights used in gating/threshold blending
constexpr double kWeightHigh = 0.35;
constexpr double kWeightMid = 0.25;
constexpr double kWeightLow = 0.15;
constexpr double kWeight20 = 0.20;

// Competition (Alg 20: Retrieval-Induced Forgetting)
constexpr double kWinnersKMax = 7.0;         // Max winners count (low F)
constexpr double kWinnersKMin = 3.0;         // Min winners count (high F)
constexpr double kInhibitionRadiusMin = 0.5; // Min inhibition radius (low F)
constexpr double kInhibitionRadiusMax = 0.85; // Max inhibition radius (high F)
constexpr double kSuppressionBaseMax = 0.1;  // Max suppression per retrieval (low T)
constexpr double kSuppressionBaseMin = 0.01; // Min suppression per retrieval (high T)
constexpr double kLateralInhibitionSMult = 0.5; // S multiplier for lateral inhibition
constexpr double kCompetitionIterMin = 3.0;  // Min competition iterations (low F)
constexpr double kCompetitionIterMax = 10.0; // Max competition iterations (high F)
// Spec (§6.4, line 1087): recovery_time_RIF = lerp(300, 1800, T) seconds
constexpr double kRecoveryTimeMinSeconds = 300.0;  // Min recovery time (low T)
constexpr double kRecoveryTimeMaxSeconds = 1800.0; // Max recovery time (high T)

// Working Memory
constexpr double kTiny = 1e-6;           // Small epsilon for WM strength
constexpr double kStrengthMax = 10.0;    // Max WM slot strength
constexpr double kWMBaseMin = 0.3;       // Min WM strength base (low T)
constexpr double kWMBaseMax = 0.9;       // Max WM strength base (high T)
constexpr double kWMRehearsalBaseDelta = 0.1;  // Base rehearsal strength boost
constexpr double kWMRecencyTauSeconds = 60.0; // Recency decay time constant

// Precision Modulation (Section 5.5)
constexpr double kTargetPrecisionBaseScale = 0.5; // Base scale for target precision

// Sensitivity Priors (Alg 3)

// Focus Priors (Alg 1)
constexpr double kCoverageGainFloorBase = 0.3; // Base coverage gain floor
constexpr double kCoverageGainScale = 0.7;     // F-scaled coverage gain

// Metrics (Alg 16)
constexpr double kRarityTCoeff = 0.2;  // T coefficient for rarity
constexpr double kUtilitySCoeff = 0.3; // S coefficient for utility

// Goal Alignment (Alg 33)
constexpr double kCosineToAlignmentScale = 0.5;  // cos -> alignment scale
constexpr double kCosineToAlignmentOffset = 1.0; // cos -> alignment offset

// Consolidation (Alg 29)
constexpr double kExtractionStabilityThreshold = 0.2; // Min stability to enable extraction

// Embedding defaults (for INSERT into sqlite-vec embeddings table)
// Note: sqlite-vec doesn't support DEFAULT values, so we must provide these explicitly
constexpr double kEmbeddingStrengthDefault = 1.0;
constexpr double kEmbeddingUseFrequencyDefault = 0.0;
constexpr double kEmbeddingStabilityDefault = 1.0;
constexpr double kEmbeddingConnectivityDefault = 0.0;
constexpr double kEmbeddingDriftMagDefault = 0.0;
constexpr double kEmbeddingInfluenceDefault = 0.0;
constexpr double kEmbeddingSustainedInfluenceDefault = 0.0;
constexpr double kEmbeddingContextualGainDefault = 0.0;
constexpr double kEmbeddingRedundancyDefault = 0.0;
constexpr double kEmbeddingPreActivationDefault = 0.0;
constexpr double kEmbeddingLabilityStateDefault = 0.0;
constexpr int kEmbeddingSuppressionCountDefault = 0;

} // namespace cortext::operations::constants

