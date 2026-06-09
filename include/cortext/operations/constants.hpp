#pragma once

namespace cortext::operations::constants
{
// Hysteresis band
constexpr double kHysteresisBandMin = 0.02;
constexpr double kHysteresisBandMax = 0.25;

// Stability half-life adjustment clamp bounds
constexpr double kHalfLifeAdjClampMin = -0.25;
constexpr double kHalfLifeAdjClampMax = 0.25;
constexpr double kQuarter = 0.25;

// Common numeric constants
constexpr double kNormEpsilon = 1e-9;
constexpr double kNormalizedMin = 0.0;
constexpr double kNormalizedMax = 1.0;
constexpr double kOneHalf = 0.5;
constexpr double kTwo = 2.0;

// Working Memory
constexpr double kTiny = 1e-6;           // Small epsilon for WM strength

// Precision Modulation (Section 5.5)
constexpr double kTargetPrecisionBaseScale = 0.5; // Base scale for target precision

// Sensitivity Priors (Alg 3)

// Goal Alignment (Alg 33)
constexpr double kCosineToAlignmentScale = 0.5;  // cos -> alignment scale
constexpr double kCosineToAlignmentOffset = 1.0; // cos -> alignment offset

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
