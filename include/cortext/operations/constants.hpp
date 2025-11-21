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
} // namespace cortext::operations::constants


