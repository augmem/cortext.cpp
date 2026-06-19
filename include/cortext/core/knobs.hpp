#pragma once

#include "cortext/core/algorithms.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace cortext::core
{

// This file implements the knob-derived functions from `algorithms.md` Section
// 0.

// --- 0.2 Knob-Derived Functions ---

// Midpoint bias for "all‑arounder" tuning: endpoints fixed, midpoint nudged.
// x' = x − b * 4x(1−x), where b controls the midpoint shift (x=0.5 → 0.5−b).
// This keeps the UI knobs unchanged while centering the neutral sweet spot at 0.5.
constexpr double kFocusMidBias = 0.10;
constexpr double kSensitivityMidBias = 0.10;
constexpr double kAffectMidBias = -0.06;
constexpr double kRetrievalFocusMidBias = -0.175;
constexpr double kRetrievalSensitivityMidBias = 0.325;
constexpr double kRetrievalStabilityMidBias = -0.25;

inline double
BiasMid (double x, double bias)
{
  const double x01 = Clamp (x, 0.0, 1.0);
  const double hump = 4.0 * x01 * (1.0 - x01);
  return Clamp (x01 - bias * hump, 0.0, 1.0);
}

inline double
FocusBias (double F)
{
  return BiasMid (F, kFocusMidBias);
}

inline double
SensitivityBias (double S)
{
  return BiasMid (S, kSensitivityMidBias);
}

inline double
AffectSensitivityBias (double S)
{
  // Affective gain uses a lighter (slightly lifted) midpoint bias to ensure
  // mid-range Sensitivity still yields meaningful affect modulation.
  return BiasMid (S, kAffectMidBias);
}

inline double
RetrievalFocusBias (double F)
{
  // The mixed-media judged matrix found the compact retrieval sweet spot at
  // external F=0.75 under the global bias. Shift only retrieval/STM graph
  // routing so neutral F=0.5 lands on that effective operating point.
  return BiasMid (F, kRetrievalFocusMidBias);
}

inline double
RetrievalSensitivityBias (double S)
{
  // The same matrix favored external S=0.25. This keeps the UI midpoint
  // neutral while making the default retrieval policy less eager to admit
  // marginal graph/context evidence.
  return BiasMid (S, kRetrievalSensitivityMidBias);
}

inline double
RetrievalStabilityBias (double T)
{
  // The post-remap judged matrix found the best compact retrieval point at
  // external T=0.75. Shift only retrieval/STM graph policy so neutral T=0.5
  // receives the same continuity/temporal support.
  return BiasMid (T, kRetrievalStabilityMidBias);
}

inline double
BaseRatePrior (double S)
{
  // base_rate_prior(S) = lerp(r_min, r_max, S), r_min=0.2, r_max=5.0
  return Lerp (0.2, 5.0, SensitivityBias (S));
}

struct FocusStatePriors
{
  double relevance_weight;
  double coverage_gain_floor;
  double mismatch_weight;
};

inline FocusStatePriors
FocusStatePriorsForKnobs (double F)
{
  const double f = Clamp (F, 0.0, 1.0);
  return { Sigmoid (2.0 * f - 1.0), 0.3 + 0.7 * f, 1.0 - f };
}

struct StabilityStatePriors
{
  double rate_decay;
  double secondary_half_life_scale;
  double drift_weight;
};

inline StabilityStatePriors
StabilityStatePriorsForKnobs (double T)
{
  const double t = Clamp (T, 0.0, 1.0);
  return { Lerp (0.60, 0.98, t), 0.50, 0.50 * (1.0 - t) };
}

inline double
MetricRarityStabilityScale (double T)
{
  return Clamp (1.0 - 0.20 * Clamp (T, 0.0, 1.0), 0.0, 1.0);
}

inline double
MetricUtilitySensitivityScale (double S)
{
  return Clamp (1.0 - 0.30 * SensitivityBias (S), 0.0, 1.0);
}

struct SensitivityStatePriors
{
  double novelty_weight;
  double surprise_weight;
  double valence_weight;
  double arousal_weight;
  double emotion_weight;
  double emotion_gain;
  double score_gain;
};

inline SensitivityStatePriors
SensitivityStatePriorsForKnobs (double S)
{
  const double s = SensitivityBias (S);
  return { 0.3 + 0.7 * s,
           0.2 + 0.8 * s,
           0.4 + 0.6 * s,
           s,
           0.2 + 0.8 * s,
           std::exp (1.5 * s),
           std::exp (2.0 * s) };
}

struct EmotionProjectionPolicy
{
  double softmax_beta;
  double intensity_gamma;
  double intensity_gain;
  double ewma_alpha;
};

inline EmotionProjectionPolicy
EmotionProjectionPolicyForKnobs (double S)
{
  const double s = SensitivityBias (S);
  return { 8.0 + 24.0 * s,
           Lerp (0.5, 0.25, s),
           Lerp (1.0, 2.2, s),
           Lerp (0.05, 0.30, s) };
}

inline double
AffectThresholdGain (double S)
{
  return 0.10 * SensitivityBias (S);
}

inline double
BlendBootstrapFallback (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.50 + 0.03 * (f - f0) + 0.02 * (s - s0)
                    + 0.02 * (t - 0.5),
                0.40, 0.60);
}

inline double
NCtx (double T)
{
  return std::round (Lerp (32.0, 256.0, T));
}

inline int
WinMemCtx (double T)
{
  return static_cast<int> (std::round (Lerp (4.0, 32.0, Clamp (T, 0.0, 1.0))));
}

inline int
WScore (double T)
{
  return static_cast<int> (std::round (Lerp (20.0, 120.0, T)));
}

inline int
KNeighbors (double T)
{
  return static_cast<int> (std::round (Lerp (8.0, 32.0, T)));
}

// Coherence window size (Section 4.4)
inline int
WinCoh (double T)
{
  return static_cast<int> (std::round (Lerp (8.0, 32.0, Clamp (T, 0.0, 1.0))));
}

inline int
KCtx (double T)
{
  return static_cast<int> (std::round (Lerp (3.0, 10.0, Clamp (T, 0.0, 1.0))));
}

// RLS observation window size (Algorithm 7)
inline int
RLSWindowN (double T)
{
  // N = round(lerp(64, 512, T))
  // Higher stability = larger window = more historical context
  return static_cast<int> (std::round (Lerp (64.0, 512.0, Clamp (T, 0.0, 1.0))));
}

// --- Section 0.7: Operational Retrieval & Streaming Parameters ---
inline int
MaxResults (double F)
{
  // max_results(F) = round(lerp(R_max, R_min, F)), R_min=8, R_max=96
  return static_cast<int> (std::round (Lerp (96.0, 8.0, FocusBias (F))));
}

inline int
RetrievalMaxResults (double F)
{
  return static_cast<int> (
      std::round (Lerp (96.0, 8.0, RetrievalFocusBias (F))));
}

inline double
RetrievalContextMix (double F, double S, double T)
{
  // Recent context should be a load-bearing STM cue, not a fixed weak nudge.
  // Lower Focus broadens the query, higher Sensitivity admits more immediate
  // context, and higher Stability preserves temporal continuity.
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.34, 0.12, f) * Lerp (0.92, 1.28, s)
                    * Lerp (0.95, 1.12, t),
                0.08, 0.38);
}

struct SignalFilterAdaptivePolicy
{
  double alpha;
  double base_threshold;
  double heartbeat_seconds;
  double focus_pressure;
  double sensitivity_release;
  double stability_pressure;
  double settling_velocity_scale;
  double quiet_release_weight;
  double settling_release;
  double ema_abs_dev_weight;
  double entropy_weight;
  double min_threshold;
  double max_threshold;
  double mean_score_weight;
  double max_score_weight;
};

inline SignalFilterAdaptivePolicy
SignalFilterAdaptivePolicyForKnobs (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double base_scale = Clamp (1.0 + 0.12 * (f - f0)
                                      - 0.18 * (s - s0)
                                      + 0.12 * (t - 0.5),
                                  0.50, 2.0);
  const double heartbeat_scale = Clamp (1.0 + 0.10 * (f - f0)
                                            - 0.25 * (s - s0)
                                            + 0.25 * (t - 0.5),
                                        0.375, 2.0);
  const double quiet_scale = Clamp (1.0 + 0.18 * (s - s0)
                                       + 0.10 * (t - 0.5),
                                   0.60, 1.45);
  const double settling_release
      = Clamp (0.525 + 0.35 * (s - s0) - 0.08 * (t - 0.5), 0.20, 0.85);
  const double mean_score_weight
      = Clamp (0.60 - 0.10 * (f - f0) + 0.04 * (t - 0.5), 0.45, 0.75);
  return {
    Clamp (0.08 + 0.12 * (1.0 - t), 0.04, 0.24),
    Clamp (0.012 * base_scale, 0.0025, 0.050),
    Clamp (2.0 * heartbeat_scale, 0.50, 4.0),
    Clamp (1.10 + 0.70 * (f - f0), 0.65, 1.60),
    Clamp (0.875 - 0.45 * (s - s0), 0.45, 1.20),
    Clamp (1.05 + 0.40 * (t - 0.5), 0.70, 1.30),
    Clamp (0.15 * (1.0 + 0.10 * (f - f0) - 0.10 * (s - s0)),
           0.08, 0.24),
    Clamp (0.35 * quiet_scale, 0.18, 0.55),
    settling_release,
    Clamp (0.95 + 0.40 * (f - f0), 0.60, 1.25),
    Clamp (0.40 * (1.0 + 0.15 * (s - s0) - 0.08 * (t - 0.5)),
           0.22, 0.58),
    0.0025,
    Clamp (0.50 * (1.0 + 0.10 * (f - f0) - 0.10 * (s - s0)),
           0.30, 0.65),
    mean_score_weight,
    1.0 - mean_score_weight
  };
}

inline double
SignalFilterAudioMinAcceptSeconds (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double scale = Clamp (1.0 - 0.10 * (f - f0) - 0.20 * (s - s0)
                                  + 0.20 * (Clamp (T, 0.0, 1.0) - 0.5),
                              0.50, 2.0);
  return Clamp (0.10 * scale, 0.05, 0.20);
}

inline double
SignalFilterImageMinAcceptSeconds (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double scale = Clamp (1.0 - 0.08 * (f - f0) - 0.16 * (s - s0)
                                  + 0.12 * (Clamp (T, 0.0, 1.0) - 0.5),
                              0.50, 2.0);
  return Clamp ((1.0 / 60.0) * scale, 1.0 / 120.0, 1.0 / 24.0);
}

inline double
RetrievalTokenOverlapQueryWeight (double F, double S, double T)
{
  // Text route overlap is a blend of "does the source cover the query" and
  // "does the query meaningfully touch the source". Focus favors the former;
  // Sensitivity and Stability keep broad source coverage available.
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.54, 0.76, f) * Lerp (1.06, 0.94, s)
                    * Lerp (1.02, 0.98, t),
                0.50, 0.90);
}

inline int
RetrievalRouteTokenMinChars (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double raw = 3.0 + 1.00 * (f - f0) - 0.80 * (s - s0)
                     + 0.60 * (t - t0);
  return static_cast<int> (std::round (Clamp (raw, 2.0, 4.0)));
}

inline double
RetrievalPressureGateLowScale (double T)
{
  // Under low storage pressure, high Stability should preserve old resurfacing
  // confidence more aggressively; low Stability lets time decay dominate.
  return Lerp (0.30, 0.12, RetrievalStabilityBias (T));
}

inline double
RetrievalPressureRampLowScale (double T)
{
  return Lerp (0.16, 0.08, RetrievalStabilityBias (T));
}

inline int
HydratedSoftAnchorLimit (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Clamp (Lerp (4.0, 2.0, f) * Lerp (1.05, 0.95, s)
                             * Lerp (1.00, 1.10, t),
                         1.0, 6.0)));
}

inline int
RetrievalTextQueryWMSlots (double F, double T)
{
  // Lower focus permits broader conversational addressing; higher stability
  // trusts more short-term context while keeping the query bounded.
  const double breadth
      = (1.0 - RetrievalFocusBias (F))
        * Lerp (0.75, 1.25, RetrievalStabilityBias (T));
  return static_cast<int> (
      std::round (Lerp (1.0, 4.0, Clamp (breadth, 0.0, 1.0))));
}

inline int
RetrievalTextQueryWMChars (double F, double T)
{
  // Character budget follows the same addressing breadth as slot count.
  const double breadth
      = (1.0 - RetrievalFocusBias (F))
        * Lerp (0.75, 1.25, RetrievalStabilityBias (T));
  return static_cast<int> (
      std::round (Lerp (512.0, 1800.0, Clamp (breadth, 0.0, 1.0))));
}

inline int
RetrievalSeedSearchK (double F, double S, double T, int seed_k)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double multiplier = Lerp (14.0, 8.0, f) * Lerp (1.05, 0.90, t);
  const int floor_extra = static_cast<int> (
      std::round (Lerp (48.0, 24.0, f) * Lerp (1.08, 0.92, s)));
  const int cap = static_cast<int> (
      std::round (Lerp (768.0, 384.0, f) * Lerp (1.05, 0.90, t)));
  const int expanded = static_cast<int> (
      std::round (static_cast<double> (std::max (1, seed_k)) * multiplier));
  const int floor = std::max (seed_k, seed_k + floor_extra);
  return std::max (seed_k, std::min (cap, std::max (expanded, floor)));
}

inline int
RetrievalFactVectorSearchK (double F, double S, double T, int fact_k)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double multiplier = Lerp (14.0, 8.0, f) * Lerp (1.08, 0.92, s)
                            * Lerp (1.05, 0.90, t);
  const int cap = static_cast<int> (
      std::round (Lerp (320.0, 128.0, f) * Lerp (1.05, 0.90, t)));
  const int floor = static_cast<int> (
      std::round (Lerp (48.0, 24.0, f) * Lerp (1.05, 0.95, t)));
  const int expanded = static_cast<int> (
      std::round (static_cast<double> (std::max (1, fact_k)) * multiplier));
  return std::max (fact_k, std::min (cap, std::max (floor, expanded)));
}

inline int
RetrievalFactTextSearchLimit (double F, double S, double T, int seed_count)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double multiplier = Lerp (24.0, 10.0, f) * Lerp (1.10, 0.90, s)
                            * Lerp (1.05, 0.90, t);
  const int cap = static_cast<int> (
      std::round (Lerp (900.0, 240.0, f) * Lerp (1.05, 0.90, t)));
  const int floor = std::max (seed_count * 4, 64);
  const int expanded = static_cast<int> (
      std::round (static_cast<double> (std::max (1, seed_count)) * multiplier));
  return std::max (seed_count, std::min (cap, std::max (floor, expanded)));
}

inline int
RetrievalLabelTextRouteLimit (double F, double S, double T, int top_labels)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const int base = static_cast<int> (
      std::round (Lerp (512.0, 128.0, f) * Lerp (1.08, 0.92, s)
                  * Lerp (1.05, 0.90, t)));
  return std::max (top_labels, base);
}

inline bool
RetrievalLabelTokenTextRouteEnabled (double F, double S, double T)
{
  return RetrievalLabelTextRouteLimit (F, S, T, 1) > 0
         && RetrievalTokenOverlapQueryWeight (F, S, T) > 0.0;
}

inline int
RetrievalLabelBankStaticSearchK (double F, double S, double T, int top_labels)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const int multiplier = static_cast<int> (
      std::round (Lerp (4.0, 2.0, f) * Lerp (1.05, 0.95, s)
                  * Lerp (1.05, 0.90, t)));
  return std::max (top_labels, std::max (8, top_labels * multiplier));
}

inline int
RetrievalDurableSourceTextSearchLimit (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Clamp (Lerp (1200.0, 320.0, f) * Lerp (1.10, 0.90, s)
                             * Lerp (1.05, 0.90, t),
                         192.0, 1600.0)));
}

inline int
RetrievalDurableSourceTextRefreshBatch (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Clamp (Lerp (9.0, 3.0, f) * Lerp (1.10, 0.90, s)
                             * Lerp (1.05, 0.90, t),
                         2.0, 16.0)));
}

inline int
RetrievalDurableSourceTextRefreshInterval (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Clamp (Lerp (32.0, 96.0, f) * Lerp (1.25, 0.75, s)
                             * Lerp (0.85, 1.15, t),
                         16.0, 192.0)));
}

inline int
RetrievalDurableSourceTextMaxBytes (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Clamp (Lerp (4096.0, 1536.0, f) * Lerp (0.95, 1.10, s)
                             * Lerp (1.05, 0.90, t),
                         1024.0, 8192.0)));
}

inline int
RetrievalGraphExpansionRowLimit (double F, double S, double T, int result_k)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double multiplier = Lerp (18.0, 8.0, f) * Lerp (1.08, 0.92, s)
                            * Lerp (1.05, 0.90, t);
  const int cap = static_cast<int> (
      std::round (Lerp (768.0, 256.0, f) * Lerp (1.05, 0.90, t)));
  return std::max (result_k, std::min (
                                 cap, static_cast<int> (std::round (
                                          std::max (1, result_k) * multiplier))));
}

inline int
RetrievalGraphExpansionFanout (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double raw = Lerp (48.0, 16.0, f) * Lerp (0.90, 1.20, s)
                     * Lerp (1.05, 0.85, t);
  return static_cast<int> (std::round (Clamp (raw, 8.0, 64.0)));
}

inline int
RetrievalLabelGraphFanout (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double raw = Lerp (96.0, 24.0, f) * Lerp (0.90, 1.15, s)
                     * Lerp (1.05, 0.85, t);
  return static_cast<int> (std::round (Clamp (raw, 12.0, 128.0)));
}

inline int
RetrievalFactEvidenceFanout (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double raw = Lerp (16.0, 6.0, f) * Lerp (0.95, 1.15, s)
                     * Lerp (1.05, 0.90, t);
  return static_cast<int> (std::round (Clamp (raw, 4.0, 24.0)));
}

inline int
RetrievalFactStaleExpansionLimit (double F, double S, double T,
                                  int matched_fact_count)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const int per_fact = static_cast<int> (
      std::round (Clamp (Lerp (8.0, 3.0, f) * Lerp (0.95, 1.15, s)
                             * Lerp (1.05, 0.90, t),
                         2.0, 12.0)));
  return std::max (std::max (1, matched_fact_count),
                   std::max (1, matched_fact_count) * per_fact);
}

inline int
RetrievalDurableSourceLinkFanout (double F, double S, double T,
                                  int candidate_count)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const int per_candidate = static_cast<int> (
      std::round (Clamp (Lerp (12.0, 4.0, f) * Lerp (0.95, 1.12, s)
                             * Lerp (1.05, 0.90, t),
                         2.0, 16.0)));
  return std::max (std::max (1, candidate_count),
                   std::max (1, candidate_count) * per_candidate);
}

inline int
RetrievalSummaryLabelSeedCount (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return std::max (1, static_cast<int> (
                          std::round (Lerp (2.0, 6.0, s)
                                      * Lerp (1.0, 0.75, f)
                                      * Lerp (1.0, 0.85, t))));
}

inline int
RetrievalFactVectorSeedCount (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t = Clamp (T, 0.0, 1.0);
  return std::max (2, static_cast<int> (
                          std::round (Lerp (12.0, 3.0, f)
                                      * Lerp (1.0, 0.85, t)
                                      * Clamp (1.0 + 0.18 * (s - s0),
                                               0.85, 1.18))));
}

struct RetrievalGraphExpansionEvidencePolicy
{
  int min_association_count;
  int min_label_count;
};

inline RetrievalGraphExpansionEvidencePolicy
RetrievalGraphExpansionEvidenceCounts (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return { std::max (0, static_cast<int> (
                             std::round (Lerp (3.0, 1.0, f)
                                         * Lerp (1.0, 0.85, t)))),
           std::max (0, static_cast<int> (
                             std::round (Lerp (0.0, 3.0, s)
                                         * Lerp (1.0, 0.85, f)))) };
}

inline int
RetrievalFactTextSeedCount (double F, double S, double T)
{
  // Lower focus and higher sensitivity allow a wider fact-text rescue set;
  // higher stability tightens it slightly because durable fact state should be
  // less volatile.
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t = RetrievalStabilityBias (T);
  return std::max (
      1, static_cast<int> (
             std::round (Lerp (24.0, 6.0, f) * Lerp (1.05, 0.90, t)
                         * Clamp (1.0 + 0.20 * (s - s0), 0.85, 1.20))));
}

inline double
RetrievalFactTextSeedMinScore (double F, double S, double T)
{
  // Fact text scores are query-token coverage over concise fact text. When
  // working memory is folded into the query, useful overlaps are smaller than
  // exact-label matches, so Sensitivity lowers the admission floor.
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.18, 0.32, f) * Lerp (1.05, 0.82, s)
                    * Lerp (0.95, 1.05, t),
                0.14, 0.40);
}

inline double
RetrievalPreconsolidatedLabelGraphWeight (double F, double S, double T)
{
  // Label-graph boosts are a soft candidate-ranking prior. Higher Sensitivity
  // trusts label evidence more; higher Focus keeps the boost conservative.
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.14, 0.26, s) * Lerp (1.10, 0.82, f)
                    * Lerp (1.05, 0.92, t),
                0.08, 0.32);
}

inline int
RetrievalPreconsolidatedLabelGraphTopLabels (double F, double T)
{
  // Lower Focus explores more provisional labels; higher Stability narrows the
  // set because durable state should already carry persistent anchors.
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (16.0, 6.0, f) * Lerp (1.08, 0.92, t)));
}

inline double
RetrievalPreconsolidatedLabelGraphMinQueryScore (double F, double S, double T)
{
  // Keep this low: the label graph is a rescue route, not a hard exact-match
  // gate. Sensitivity lowers the floor for weak but useful query overlap.
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.0, 0.04, f) * Lerp (1.05, 0.70, s)
                    * Lerp (0.90, 1.10, t),
                0.0, 0.06);
}

inline double
RetrievalPreconsolidatedLabelRelationWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.24, 0.46, s) * Lerp (1.05, 0.90, f)
                    * Lerp (0.95, 1.05, t),
                0.18, 0.55);
}

inline double
RetrievalPreconsolidatedLabelGraphDegreeDamping (double F, double S, double T)
{
  // Generic-hub suppression rises with Sensitivity and lower Focus.
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.03, 0.12, s) * Lerp (1.20, 0.75, f)
                    * Lerp (1.00, 0.85, t),
                0.0, 0.15);
}

inline int
RetrievalPreconsolidatedLabelGraphSeedSources (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (32.0, 12.0, f) * Lerp (1.05, 0.90, t)));
}

inline int
RetrievalDurableSourceTextSeedSources (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (24.0, 8.0, f) * Lerp (1.05, 0.92, t)));
}

inline double
RetrievalDurableSourceTextMinScore (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.12, 0.24, f) * Lerp (1.05, 0.82, s)
                    * Lerp (0.95, 1.05, t),
                0.08, 0.30);
}

inline int
RetrievalLinkedHydrationCandidateLimit (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Clamp (Lerp (20.0, 8.0, f) * Lerp (1.05, 0.95, s)
                             * Lerp (1.05, 0.92, t),
                         6.0, 24.0)));
}

inline std::pair<double, double>
RetrievalLinkedHydrationOrderingWeights (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double query_raw = 0.82 + 0.14 * (f - f0) - 0.04 * (s - s0)
                           + 0.04 * (t - t0);
  const double query_weight = Clamp (query_raw, 0.60, 0.92);
  return { query_weight, 1.0 - query_weight };
}

inline double
RetrievalDurableSourceSupportSaturationCount (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (6.0, 4.0, f) * Lerp (1.05, 0.95, t), 2.0, 8.0);
}

inline double
RetrievalDurableSourceTextBaseWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.72, 0.86, f) * Lerp (1.00, 0.94, s)
                    * Lerp (0.98, 1.02, t),
                0.60, 0.92);
}

struct RetrievalFactCandidateScoringPolicy
{
  double boost_weight;
  double stale_penalty_weight;
};

inline RetrievalFactCandidateScoringPolicy
RetrievalFactCandidateScoringWeights (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return { Lerp (0.08, 0.28, f) * Lerp (0.90, 1.10, s)
               * Lerp (1.05, 0.95, t),
           Lerp (0.05, 0.20, f) * Lerp (0.90, 1.05, s)
               * Lerp (0.98, 1.02, t) };
}

inline double
RetrievalFactMissingConfidence (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  return Clamp (0.50 + 0.04 * (s - s0) - 0.03 * (f - f0)
                    + 0.02 * (t - t0),
                0.40, 0.60);
}

inline double
RetrievalFactMissingEvidenceSupportWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  return Clamp (0.72 - 0.06 * (f - f0) + 0.08 * (s - s0)
                    + 0.04 * (t - t0),
                0.55, 0.90);
}

inline double
RetrievalFactBoostWeakMultiplier (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double scale = 1.0 - 0.12 * (f - f0) + 0.18 * (s - s0)
                       + 0.08 * (t - t0);
  return Clamp (0.55 * scale, 0.40, 0.75);
}

inline double
RetrievalFactBoostStrongMultiplier (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double scale = 1.0 - 0.10 * (f - f0) + 0.16 * (s - s0)
                       + 0.10 * (t - t0);
  return Clamp (1.45 * scale, 1.10, 1.85);
}

inline double
RetrievalStalePenaltyStrongMultiplier (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double scale = 1.0 + 0.08 * (f - f0) + 0.10 * (s - s0)
                       + 0.14 * (t - t0);
  return Clamp (1.75 * scale, 1.30, 2.20);
}

inline double
RetrievalVectorDistanceScore (double distance, double F, double S, double T)
{
  if (!std::isfinite (distance) || distance < 0.0)
    {
      return 0.0;
    }
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double sharpness
      = Clamp (1.0 + 0.18 * (f - f0) - 0.10 * (s - s0)
                   + 0.08 * (t - t0),
               0.70, 1.35);
  return Clamp (1.0 / (1.0 + sharpness * distance), 0.0, 1.0);
}

inline int
RetrievalSparseKeySize (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = 0.5;
  const double s0 = SensitivityBias (0.5);
  const double raw = (16.0 + (64.0 - 16.0) * f)
                     * Clamp (1.0 + 0.10 * (s - s0)
                                  - 0.06 * (t - 0.5)
                                  - 0.04 * (f - f0),
                              0.85, 1.15);
  return std::max (8, static_cast<int> (std::round (Clamp (raw, 8.0, 72.0))));
}

inline double
RetrievalPredictiveBonusWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t = Clamp (T, 0.0, 1.0);
  return Lerp (0.05, 0.20, f) * Lerp (1.0, 0.85, t)
         * Clamp (1.0 + 0.12 * (s - s0), 0.90, 1.12);
}

inline double
RetrievalCompetitionSuppressionBase (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double base = Lerp (0.10, 0.01, t);
  return Clamp (base * (1.0 - 0.10 * (f - f0) + 0.10 * (s - s0)),
                0.005, 0.12);
}

inline int
RetrievalCompetitionWinnerCount (double F)
{
  return std::max (1, static_cast<int> (
                          std::round (Lerp (7.0, 3.0, FocusBias (F)))));
}

inline double
RetrievalCompetitionInhibitionRadius (double F)
{
  return Lerp (0.50, 0.85, FocusBias (F));
}

inline double
RetrievalCompetitionRecoverySeconds (double T)
{
  return Lerp (300.0, 1800.0, Clamp (T, 0.0, 1.0));
}

inline double
RetrievalRecentInhibitionWeight (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.04, 0.16, s) * Lerp (1.20, 0.75, f)
                    * Lerp (1.25, 0.65, t),
                0.0, 0.20);
}

inline double
RetrievalRecentInhibitionTauSeconds (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (90.0, 900.0, t) * Lerp (0.80, 1.20, f)
                    * Lerp (1.10, 0.85, s),
                30.0, 1800.0);
}

inline double
RetrievalBaseLevelAvailabilityWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.04, 0.14, s) * Lerp (1.20, 0.75, f)
                    * Lerp (0.85, 1.20, t),
                0.0, 0.18);
}

inline double
RetrievalBaseLevelAvailabilityTauSeconds (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (1800.0, 21600.0, t) * Lerp (0.85, 1.15, f)
                    * Lerp (1.10, 0.85, s),
                600.0, 43200.0);
}

inline double
RetrievalBaseLevelAvailabilityCountSaturation (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (std::round (Lerp (8.0, 32.0, t) * Lerp (0.80, 1.20, f)
                            * Lerp (0.90, 1.10, s)),
                4.0, 64.0);
}

inline double
RetrievalPartialMatchPenaltyWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.06, 0.18, s) * Lerp (0.85, 1.15, f)
                    * Lerp (1.10, 0.85, t),
                0.0, 0.22);
}

inline double
RetrievalPartialMatchContradictionSaturation (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (std::round (Lerp (6.0, 3.0, f) * Lerp (1.15, 0.85, s)
                            * Lerp (0.90, 1.20, t)),
                2.0, 8.0);
}

inline double
RetrievalPartialMatchSourceMismatchWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.20, 0.42, f) * Lerp (0.90, 1.10, s)
                    * Lerp (1.08, 0.92, t),
                0.10, 0.55);
}

inline double
RetrievalPartialMatchModalityMismatchWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.12, 0.32, f) * Lerp (0.90, 1.15, s)
                    * Lerp (1.05, 0.95, t),
                0.06, 0.45);
}

inline double
RetrievalEvidenceBlendTieMargin (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.030, 0.070, s) * Lerp (0.78, 1.12, f)
                    * Lerp (0.92, 1.08, t),
                0.015, 0.10);
}

inline double
RetrievalEvidenceBlendTemperature (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.018, 0.055, s) * Lerp (0.82, 1.10, f)
                    * Lerp (0.95, 1.08, t),
                0.010, 0.075);
}

inline int
RetrievalEvidenceBlendMaxMembers (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return std::max (
      2, static_cast<int> (std::round (Lerp (2.0, 4.0, s)
                                      * Lerp (1.10, 0.90, f)
                                      * Lerp (0.95, 1.10, t))));
}

struct InfluenceFeedbackPolicy
{
  double sustain_window;
  double sustain_alpha;
  double contextual_gain_weight;
  double generative_similarity_weight;
  double drift_weight;
};

inline InfluenceFeedbackPolicy
InfluenceFeedbackPolicyForKnobs (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double sustain_window = std::round (Lerp (3.0, 5.0, t));
  const double sustain_alpha
      = (sustain_window > 0.0) ? (2.0 / (sustain_window + 1.0)) : 1.0;
  return { sustain_window,
           sustain_alpha,
           Clamp (0.50 * (1.0 + 0.08 * (f - f0) + 0.04 * (t - 0.5)),
                  0.35, 0.65),
           Clamp (0.40 * (1.0 - 0.05 * (s - s0)), 0.25, 0.55),
           Clamp (0.30 * (1.0 + 0.08 * (s - s0) - 0.10 * (t - 0.5)),
                  0.15, 0.45) };
}

struct RetrievalCandidateBlendWeights
{
  double context;
  double procedural;
  double label_boost;
};

inline RetrievalCandidateBlendWeights
RetrievalCandidateBlendScoringWeights (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return { Lerp (0.15, 0.35, f) * Lerp (1.0, 0.85, s)
               * Lerp (0.98, 1.02, t),
           Lerp (0.10, 0.25, s) * Lerp (1.02, 0.98, f)
               * Lerp (0.98, 1.02, t),
           Lerp (0.05, 0.18, s) * Lerp (1.04, 0.96, f)
	               * Lerp (0.98, 1.02, t) };
}

struct RetrievalMemoryAffectWeights
{
  double emotion;
  double arousal;
};

inline RetrievalMemoryAffectWeights
RetrievalMemoryAffectScoringWeights (double S)
{
  const double s = AffectSensitivityBias (S);
  const double emotion_raw = Lerp (0.60, 0.80, s);
  const double arousal_raw = Lerp (0.20, 0.40, s);
  const double sum = std::max (1e-12, emotion_raw + arousal_raw);
  return { emotion_raw / sum, arousal_raw / sum };
}

inline double
RetrievalSummaryDuplicateThresholdScale (double F, double S, double T)
{
  (void)S;
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (1.35, 1.10, f) * Lerp (1.02, 0.98, t), 1.0,
                1.45);
}

inline double
RetrievalSummaryDuplicateThresholdCap (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double scale = 1.0 - 0.02 * (f - f0) + 0.02 * (s - s0)
                       + 0.01 * (t - t0);
  return Clamp (0.99 * scale, 0.94, 0.995);
}

inline bool
RetrievalBypassSummaryOverlap (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double threshold = Clamp (0.75 * Lerp (1.05, 0.95, s)
                                      * Lerp (0.98, 1.02, t),
                                  0.62, 0.88);
  return f < threshold;
}

inline double
RetrievalDurableSourceSetWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.08, 0.18, s) * Lerp (1.08, 0.88, f)
                    * Lerp (1.05, 0.92, t),
                0.04, 0.24);
}

inline double
RetrievalDurableSourceSetBaseWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.60, 0.78, f) * Lerp (1.02, 0.96, s)
                    * Lerp (0.98, 1.02, t),
                0.50, 0.88);
}

inline double
RetrievalDurableSourceSetMinScore (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.0, 0.04, f) * Lerp (1.0, 0.70, s)
                    * Lerp (0.90, 1.05, t),
                0.0, 0.06);
}

inline double
RetrievalSourceBackedBoostFloor (double F, double S, double T)
{
  // A nonzero graph/source boost is not enough to prove a durable source hit.
  // The floor stays low, but rises with Focus/Stability so generic graph dust
  // cannot force promotion ahead of stronger recent context.
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.012, 0.045, f) * Lerp (1.05, 0.80, s)
                    * Lerp (0.90, 1.05, t),
                0.010, 0.060);
}

struct RetrievalFactStalePolicy
{
  double similarity_weight;
  double confidence_weight;
  double lifecycle_floor;
};

inline RetrievalFactStalePolicy
RetrievalFactStaleScoringPolicy (double F, double S, double T)
{
  (void)S;
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  const double similarity_weight
      = Clamp (Lerp (0.45, 0.67, f) * Lerp (0.98, 1.02, t), 0.35,
               0.80);
  return { similarity_weight,
           1.0 - similarity_weight,
           Clamp (Lerp (0.40, 0.60, t) * Lerp (0.98, 1.02, f), 0.35,
                  0.70) };
}

struct RetrievalFactBoostPolicy
{
  double similarity_weight;
  double confidence_weight;
  double support_weight;
  double temporal_weight;
};

inline RetrievalFactBoostPolicy
RetrievalFactBoostScoringPolicy (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double similarity_raw = Lerp (0.35, 0.50, f);
  const double confidence_raw = Lerp (0.40, 0.30, f) * Lerp (1.05, 0.95, s);
  const double support_raw = Lerp (0.18, 0.28, s) * Lerp (1.05, 0.95, f);
  const double sum = std::max (1e-12,
                               similarity_raw + confidence_raw + support_raw);
  return { similarity_raw / sum,
           confidence_raw / sum,
           support_raw / sum,
           Clamp (Lerp (0.55, 0.42, t) * Lerp (0.98, 1.02, s), 0.30,
                  0.65) };
}

struct RetrievalRoutineRecencyAdjustmentWeights
{
  double routine_boost_weight;
  double routine_recency_penalty_weight;
  double routine_min_multiplier;
  double routine_max_multiplier;
  double recency_boost_weight;
  double recency_routine_penalty_weight;
  double recency_min_multiplier;
  double recency_max_multiplier;
  double stale_routine_penalty_weight;
  double stale_routine_min_multiplier;
  double stale_routine_max_multiplier;
  double stale_recency_boost_weight;
  double stale_recency_floor;
  double stale_recency_routine_weight;
  double stale_recency_min_multiplier;
  double stale_recency_max_multiplier;
};

inline RetrievalRoutineRecencyAdjustmentWeights
RetrievalRoutineRecencyAdjustmentPolicy (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double routine_scale = 1.0 + 0.10 * (f - f0) - 0.10 * (s - s0)
                               + 0.12 * (t - t0);
  const double recency_scale = 1.0 + 0.10 * ((1.0 - f) - (1.0 - f0))
                               + 0.16 * (s - s0) - 0.08 * (t - t0);
  const double stale_routine_scale
      = 1.0 + 0.06 * (f - f0) - 0.08 * (s - s0) + 0.14 * (t - t0);
  const double stale_recency_scale
      = 1.0 + 0.08 * ((1.0 - f) - (1.0 - f0)) + 0.12 * (s - s0)
        - 0.06 * (t - t0);
  return {
    Clamp (0.40 * routine_scale, 0.24, 0.56),
    Clamp (0.16 * routine_scale, 0.08, 0.28),
    Clamp (0.70 * routine_scale, 0.55, 0.82),
    Clamp (1.45 * routine_scale, 1.15, 1.75),
    Clamp (0.46 * recency_scale, 0.28, 0.64),
    Clamp (0.18 * recency_scale, 0.08, 0.30),
    Clamp (0.70 * recency_scale, 0.55, 0.82),
    Clamp (1.55 * recency_scale, 1.20, 1.90),
    Clamp (0.65 * stale_routine_scale, 0.42, 0.85),
    Clamp (0.35 * stale_routine_scale, 0.22, 0.45),
    Clamp (1.05 * stale_routine_scale, 0.92, 1.20),
    Clamp (0.45 * stale_recency_scale, 0.28, 0.62),
    Clamp (0.35 * stale_recency_scale, 0.20, 0.45),
    Clamp (0.24 * stale_recency_scale, 0.12, 0.36),
    Clamp (0.95 * stale_recency_scale, 0.82, 1.05),
    Clamp (1.75 * stale_recency_scale, 1.40, 2.05)
  };
}

inline double
RetrievalSourceFreshnessTauSeconds (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (1500.0, 4800.0, t) * Lerp (1.12, 0.86, f)
                    * Lerp (1.05, 0.88, s),
                900.0, 7200.0);
}

inline double
RetrievalSourceContradictionPenalty (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.12, 0.24, s) * Lerp (0.95, 1.05, f)
                    * Lerp (1.05, 0.90, t),
                0.04, 0.30);
}

inline double
RetrievalSourceFreshnessWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.44, 0.20, t) * Lerp (0.90, 1.18, f)
                    * Lerp (0.92, 1.16, s),
                0.12, 0.56);
}

inline double
RetrievalTemporalRankTauSeconds (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (900.0, 5400.0, t) * Lerp (1.15, 0.85, f)
                    * Lerp (1.05, 0.90, s),
                600.0, 7200.0);
}

inline double
RetrievalTemporalRankWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.08, 0.18, 1.0 - f) * Lerp (0.95, 1.10, s)
                    * Lerp (0.90, 1.08, t),
                0.04, 0.24);
}

inline double
RetrievalSourceConfidenceThreshold (double F, double S, double T,
                                    double resurfacing_decay_scale)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double low = Clamp (Lerp (0.12, 0.22, f) * Lerp (1.08, 0.85, s)
                                * Lerp (1.02, 0.86, t),
                            0.06, 0.30);
  const double high = Clamp (Lerp (0.38, 0.52, f) * Lerp (1.06, 0.90, s)
                                 * Lerp (1.02, 0.92, t),
                             low + 0.05, 0.65);
  return Lerp (low, high, Clamp (resurfacing_decay_scale, 0.0, 1.0));
}

inline double
SourceReliabilityPrior (double F, double S, double T)
{
  // Runtime memory writes should not depend on the SQLite DEFAULT literal. Keep
  // the neutral midpoint at the historical 0.70 while allowing the three knobs
  // to express trust in newly written source-backed memories.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.70 + 0.06 * (f - FocusBias (0.5))
                    - 0.05 * (s - SensitivityBias (0.5))
                    + 0.08 * (t - 0.5),
                0.50, 0.90);
}

inline double
RetrievalSeedFallbackSourceConfidence (double F, double S, double T)
{
  // Seed-only rows have vector support but no loaded source-reliability row.
  // Treat them as plausible, not perfectly trusted.
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (SourceReliabilityPrior (F, S, T) * Lerp (0.90, 0.78, s)
                    * Lerp (0.98, 1.04, t),
                0.42, 0.82);
}

enum class FactSeverity
{
  Low,
  Medium,
  High
};

enum class FactRoutineClass
{
  Generic,
  StableRoutine,
  Preference,
  MutableState
};

enum class FactCriticalityClass
{
  Generic,
  High,
  Medium,
  Preference
};

struct FactRetrievalPolicy
{
  double evidence_mass_weight;
  double evidence_count_weight;
  double evidence_count_saturation;
  double routine_duration_window_ms;
  double routine_duration_weight;
  double routine_evidence_weight;
  double routine_confirmation_saturation;
  double recency_window_ms;
  double active_lifecycle_multiplier;
  double weak_current_lifecycle_multiplier;
  double weak_history_lifecycle_multiplier;
  double archived_current_lifecycle_multiplier;
  double archived_history_lifecycle_multiplier;
  double temporal_match_weight;
  double confidence_weight;
  double evidence_support_weight;
  double current_boost_weight;
  double source_diversity_weight;
  double source_diversity_saturation;
  double supersession_penalty_weight;
  double routine_bias_weight;
  double routine_recency_penalty_weight;
  double recency_bias_weight;
  double recency_routine_penalty_weight;
};

inline FactRetrievalPolicy
FactRetrievalScoringPolicy (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double evidence_mass_weight = Clamp (
      Lerp (0.64, 0.76, s) * Lerp (0.98, 1.02, t), 0.55, 0.85);
  const double routine_duration_weight = Clamp (
      Lerp (0.50, 0.60, t) * Lerp (1.04, 0.96, s), 0.40, 0.70);
  const double routine_evidence_weight = Clamp (
      Lerp (0.50, 0.60, t) * Lerp (0.96, 1.04, s), 0.40, 0.70);
  return { evidence_mass_weight,
           1.0 - evidence_mass_weight,
           Clamp (Lerp (4.0, 6.0, t) * Lerp (1.06, 0.94, s), 3.0, 8.0),
           Clamp (Lerp (6000.0, 10000.0, t) * Lerp (1.05, 0.95, s),
                  4000.0, 14000.0),
           routine_duration_weight,
           routine_evidence_weight,
           Clamp (Lerp (5.0, 7.0, t) * Lerp (1.06, 0.94, s), 3.0, 10.0),
           Clamp (Lerp (6000.0, 10000.0, t) * Lerp (1.08, 0.92, s),
                  4000.0, 14000.0),
           1.0,
           Clamp (Lerp (0.56, 0.68, t) * Lerp (1.02, 0.98, s), 0.45,
                  0.78),
           Clamp (Lerp (0.78, 0.90, t) * Lerp (1.02, 0.98, s), 0.68,
                  0.96),
           0.0,
           Clamp (Lerp (0.80, 0.90, t) * Lerp (1.02, 0.98, s), 0.70,
                  0.96),
           Clamp (Lerp (0.34, 0.46, f) * Lerp (1.03, 0.97, s), 0.25,
                  0.55),
           Clamp (Lerp (0.18, 0.26, f) * Lerp (1.03, 0.97, s), 0.12,
                  0.32),
           Clamp (Lerp (0.18, 0.30, s) * Lerp (1.02, 0.98, f), 0.12,
                  0.36),
           Clamp (Lerp (0.08, 0.12, t) * Lerp (0.98, 1.02, f), 0.04,
                  0.16),
           Clamp (Lerp (0.03, 0.05, s) * Lerp (0.98, 1.02, t), 0.01,
                  0.08),
           Clamp (Lerp (2.0, 4.0, t) * Lerp (1.05, 0.95, s), 1.0, 6.0),
           Clamp (Lerp (0.14, 0.22, s) * Lerp (1.03, 0.97, t), 0.08,
                  0.30),
           Clamp (Lerp (0.24, 0.36, t) * Lerp (0.94, 1.06, s), 0.15,
                  0.45),
           Clamp (Lerp (0.08, 0.16, s) * Lerp (1.04, 0.96, t), 0.04,
                  0.24),
           Clamp (Lerp (0.28, 0.40, s) * Lerp (0.96, 1.04, 1.0 - t), 0.18,
                  0.50),
           Clamp (Lerp (0.12, 0.20, s) * Lerp (1.04, 0.96, t), 0.06,
                  0.28) };
}

inline double
FactRetrievalEvidenceTypeWeight (double F, double S, double T,
                                 const char *evidence_type)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const bool episodic = evidence_type != nullptr && evidence_type[0] == 'e'
                        && evidence_type[1] == 'p';
  const bool summary = evidence_type != nullptr && evidence_type[0] == 's'
                       && evidence_type[1] == 'u';
  if (episodic)
    {
      return 1.0;
    }
  if (summary)
    {
      return Clamp (Lerp (0.82, 0.94, t) * Lerp (1.02, 0.98, s), 0.74,
                    0.98);
    }
  return Clamp (Lerp (0.64, 0.76, s) * Lerp (0.98, 1.02, f)
                    * Lerp (1.02, 0.98, t),
                0.52, 0.86);
}

inline double
FactRetrievalEvidenceSupportFloor (double F, double S, double T)
{
  (void)F;
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.24, 0.16, s) * Lerp (0.96, 1.04, t), 0.10,
                0.32);
}

inline double
FactRoutineClassAffinity (double F, double S, double T,
                          FactRoutineClass routine_class)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double s0 = SensitivityBias (0.5);
  switch (routine_class)
    {
    case FactRoutineClass::StableRoutine:
      return Clamp (1.0 * (1.0 + 0.04 * (f - 0.5) - 0.06 * (s - s0)
                           + 0.08 * (t - 0.5)),
                    0.0, 1.0);
    case FactRoutineClass::Preference:
      return Clamp (0.80 * (1.0 + 0.02 * (f - 0.5) + 0.04 * (s - s0)
                            + 0.04 * (t - 0.5)),
                    0.0, 1.0);
    case FactRoutineClass::MutableState:
      return Clamp (0.35 * (1.0 - 0.04 * (f - 0.5) + 0.08 * (s - s0)
                            - 0.08 * (t - 0.5)),
                    0.0, 1.0);
    case FactRoutineClass::Generic:
      break;
    }
  return Clamp (0.15 * (1.0 - 0.06 * (f - 0.5) + 0.08 * (s - s0)
                        - 0.04 * (t - 0.5)),
                0.0, 1.0);
}

inline double
FactRoutineAffinity (double F, double S, double T, double class_affinity)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double s0 = SensitivityBias (0.5);
  const double scale = 1.0 - 0.08 * (f - 0.5) + 0.12 * (s - s0)
                       + 0.08 * (t - 0.5);
  return Clamp (class_affinity * scale,
                0.0, 1.0);
}

inline double
FactCriticalityClassPrior (double F, double S, double T,
                           FactCriticalityClass criticality_class)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double s0 = SensitivityBias (0.5);
  switch (criticality_class)
    {
    case FactCriticalityClass::High:
      return Clamp (1.0 * (1.0 + 0.03 * (f - 0.5) + 0.05 * (s - s0)
                           + 0.05 * (t - 0.5)),
                    0.0, 1.0);
    case FactCriticalityClass::Medium:
      return Clamp (0.75 * (1.0 + 0.03 * (f - 0.5) + 0.03 * (s - s0)
                            + 0.04 * (t - 0.5)),
                    0.0, 1.0);
    case FactCriticalityClass::Preference:
      return Clamp (0.45 * (1.0 - 0.02 * (f - 0.5) + 0.05 * (s - s0)
                            - 0.02 * (t - 0.5)),
                    0.0, 1.0);
    case FactCriticalityClass::Generic:
      break;
    }
  return Clamp (0.60 * (1.0 + 0.02 * (f - 0.5) + 0.03 * (s - s0)
                        + 0.02 * (t - 0.5)),
                0.0, 1.0);
}

inline double
FactCriticality (double F, double S, double T, double class_criticality)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double s0 = SensitivityBias (0.5);
  const double scale = 1.0 + 0.06 * (f - 0.5) + 0.10 * (s - s0)
                       + 0.08 * (t - 0.5);
  return Clamp (class_criticality * scale, 0.0, 1.0);
}

inline double
FactSupersessionConfidenceMargin (double S, double T)
{
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Lerp (0.18, 0.03, s) * Lerp (1.10, 0.90, t);
}

inline double
FactDerivedEventConfidence (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double s0 = SensitivityBias (0.5);
  const double scale = 1.0 - 0.04 * (f - 0.5) + 0.08 * (s - s0)
                       + 0.05 * (t - 0.5);
  return Clamp (0.62 * scale,
                0.50, 0.75);
}

inline double
FactEvidenceWriteSupportWeight (double F, double S, double T,
                                const char *evidence_type)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const bool episodic = evidence_type != nullptr && evidence_type[0] == 'e'
                        && evidence_type[1] == 'p';
  if (!episodic)
    {
      return 1.0;
    }
  const double s0 = SensitivityBias (0.5);
  const double scale = 1.0 - 0.08 * (f - 0.5) + 0.08 * (s - s0)
                       + 0.06 * (t - 0.5);
  return Clamp (0.75 * scale,
                0.55, 0.95);
}

inline double
FactLifecycleDecayWindowMillis (double F, double S, double T,
                                FactSeverity severity)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double base = severity == FactSeverity::High
                          ? 18000.0
                          : (severity == FactSeverity::Medium ? 12000.0
                                                               : 7000.0);
  return Clamp (base * Lerp (0.92, 1.22, t) * Lerp (1.08, 0.92, s)
                    * Lerp (0.97, 1.03, f),
                base * 0.60, base * 1.60);
}

inline double
FactLifecycleDeletionGraceWindowMillis (double F, double S, double T,
                                        FactSeverity severity)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double base = severity == FactSeverity::High
                          ? 16000.0
                          : (severity == FactSeverity::Medium ? 9000.0
                                                               : 4500.0);
  return Clamp (base * Lerp (0.90, 1.24, t) * Lerp (1.10, 0.90, s)
                    * Lerp (0.96, 1.04, f),
                base * 0.60, base * 1.70);
}

inline double
FactLifecycleEvidenceNormDenominator (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (1.9, 2.5, t) * Lerp (1.08, 0.92, s)
                    * Lerp (0.98, 1.02, f),
                1.4, 3.2);
}

inline double
FactLifecycleDiversitySaturation (double F, double S, double T)
{
  (void)F;
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (2.5, 3.5, t) * Lerp (1.06, 0.94, s), 2.0, 5.0);
}

inline double
FactLifecycleRepeatedCompressedSaturation (double F, double S, double T)
{
  (void)F;
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (16.0, 24.0, t) * Lerp (1.08, 0.92, s), 10.0,
                32.0);
}

inline double
FactLifecycleRepeatedRawSaturation (double F, double S, double T)
{
  (void)F;
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (5.0, 7.0, t) * Lerp (1.06, 0.94, s), 3.0, 10.0);
}

inline double
FactLifecycleHighSeverityRecencyFloor (double F, double S, double T)
{
  (void)F;
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.24, 0.36, t) * Lerp (1.04, 0.96, s), 0.16,
                0.45);
}

inline double
FactLifecycleContradictionNormDenominator (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (1.35, 1.85, t) * Lerp (0.94, 1.06, s)
                    * Lerp (0.98, 1.02, f),
                1.0, 2.4);
}

struct FactLifecycleSupportPolicy
{
  double confidence_weight;
  double evidence_weight;
  double diversity_weight;
  double repeated_weight;
  double recency_base_weight;
  double recency_dynamic_weight;
  double contradiction_weight;
};

inline FactLifecycleSupportPolicy
FactLifecycleSupportScoringPolicy (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double confidence_raw = Lerp (0.34, 0.42, f) * Lerp (1.02, 0.98, s);
  const double evidence_raw = Lerp (0.24, 0.36, s) * Lerp (1.02, 0.98, f);
  const double diversity_raw = Lerp (0.13, 0.21, t) * Lerp (1.04, 0.96, f);
  const double repeated_raw = Lerp (0.12, 0.18, t) * Lerp (1.06, 0.94, s);
  const double sum = std::max (1e-12, confidence_raw + evidence_raw
                                           + diversity_raw + repeated_raw);
  const double recency_dynamic = Clamp (
      Lerp (0.48, 0.62, t) * Lerp (1.04, 0.96, s), 0.35, 0.72);
  return { confidence_raw / sum,
           evidence_raw / sum,
           diversity_raw / sum,
           repeated_raw / sum,
           1.0 - recency_dynamic,
           recency_dynamic,
           Clamp (Lerp (0.14, 0.22, s) * Lerp (1.05, 0.95, t), 0.08,
                  0.30) };
}

inline double
FactLifecycleSupportScoreFloor (double F, double S, double T,
                                FactSeverity severity)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double base = severity == FactSeverity::High ? 0.22 : 0.0;
  return Clamp (base * Lerp (0.90, 1.16, t) * Lerp (1.04, 0.96, s)
                    * Lerp (0.98, 1.02, f),
                0.0, 0.35);
}

inline double
FactLifecycleActiveThreshold (double F, double S, double T,
                              FactSeverity severity)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double base = severity == FactSeverity::High ? 0.45 : 0.52;
  return Clamp (base * Lerp (0.96, 1.06, f) * Lerp (1.06, 0.92, s)
                    * Lerp (0.98, 1.04, t),
                0.30, 0.70);
}

inline double
FactLifecycleWeakThreshold (double F, double S, double T,
                            FactSeverity severity)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double base = severity == FactSeverity::Low ? 0.22 : 0.18;
  return Clamp (base * Lerp (1.02, 0.96, f) * Lerp (1.06, 0.92, s)
                    * Lerp (0.98, 1.04, t),
                0.08, 0.34);
}

inline double
FactLifecycleDeleteThreshold (double F, double S, double T,
                              FactSeverity severity)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double base = severity == FactSeverity::Medium ? 0.08 : 0.18;
  return Clamp (base * Lerp (1.04, 0.96, f) * Lerp (1.08, 0.90, s)
                    * Lerp (0.96, 1.08, t),
                0.03, 0.30);
}

inline int
FactLifecycleMaintenanceSweepLimit (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = Lerp (360.0, 100.0, f) * Lerp (0.92, 1.12, s)
                     * Lerp (0.95, 1.05, t);
  return static_cast<int> (std::round (Clamp (raw, 96.0, 512.0)));
}

inline int
RetrievalDurableSourceMinTopK (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (2.0, 1.0, f) * Lerp (1.05, 0.95, t)));
}

inline int
RetrievalGraphExpandedRagMaxItems (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (20.0, 8.0, f) * Lerp (1.08, 0.92, t)));
}

inline int
RetrievalGraphExpandedRagTemporalWindow (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (4.0, 1.0, f) * Lerp (1.05, 0.90, t)));
}

inline int
RetrievalGraphExpandedRagCompactItemLimit (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (4.0, 1.0, f) * Lerp (1.05, 0.90, t)));
}

inline double
RetrievalGraphExpandedRagSeedWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.72, 0.90, f) * Lerp (1.02, 0.96, s)
                    * Lerp (1.00, 0.96, t),
                0.60, 0.95);
}

inline double
RetrievalGraphExpandedRagGraphWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.16, 0.34, s) * Lerp (1.14, 0.78, f)
                    * Lerp (1.04, 0.92, t),
                0.08, 0.40);
}

inline double
RetrievalGraphExpandedRagRelationWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.24, 0.48, s) * Lerp (1.10, 0.86, f)
                    * Lerp (0.96, 1.04, t),
                0.16, 0.55);
}

inline double
RetrievalGraphExpandedRagTemporalWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.12, 0.26, 1.0 - f) * Lerp (0.94, 1.08, s)
                    * Lerp (0.92, 1.04, t),
                0.06, 0.30);
}

inline double
RetrievalGraphExpandedRagFactWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.12, 0.30, s) * Lerp (1.05, 0.88, f)
	                    * Lerp (0.96, 1.06, t),
	                0.06, 0.34);
}

inline double
RetrievalGraphExpandedRagTemporalRankScore (double F, double S, double T,
                                            int rank)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double offset = Clamp (2.0
                                   * (1.0 + 0.08 * (f - f0)
                                      - 0.06 * (s - s0)
                                      + 0.06 * (t - t0)),
                               1.5, 2.6);
  return RetrievalGraphExpandedRagTemporalWeight (F, S, T)
         / (std::max (0, rank) + offset);
}

inline double
RetrievalAnyFactMatchSupportFloor (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  return Clamp (0.25
                    * (1.0 + 0.06 * (f - f0) - 0.10 * (s - s0)
                       + 0.04 * (t - t0)),
                0.15, 0.36);
}

inline int
RetrievalAnyFactMatchLinkCap (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Clamp (Lerp (3.0, 1.0, f) * Lerp (1.10, 0.90, s)
                             * Lerp (0.95, 1.05, t),
                         1.0, 4.0)));
}

inline double
RetrievalUnknownCautionCutoffMultiplier (double F, double S, double T,
                                         double unknown_caution_scale)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double max_lift = Clamp (0.15
                                    * (1.0 + 0.08 * (f - f0)
                                       - 0.06 * (s - s0)
                                       + 0.06 * (t - t0)),
                                0.08, 0.22);
  return 1.0 + max_lift * Clamp (unknown_caution_scale, 0.0, 1.0);
}

inline double
RetrievalContextReinstatementAlpha (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double base = Lerp (0.20, 0.05, Clamp (T, 0.0, 1.0));
  return Clamp (base * (1.0 - 0.06 * (f - f0) + 0.08 * (s - s0)
                        - 0.04 * (t - RetrievalStabilityBias (0.5))),
                0.02, 0.24);
}

inline int
RetrievalProceduralSeedReserveCount (double F, double S, double T,
                                     int remaining_slots)
{
  if (remaining_slots <= 0)
    {
      return 0;
    }
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double raw = Lerp (1.4, 0.6, f) * Lerp (0.85, 1.20, s)
                     * Lerp (1.05, 0.90, t);
  const int reserve = static_cast<int> (std::round (Clamp (raw, 0.0, 2.0)));
  return std::min (reserve, remaining_slots);
}

inline int
RetrievalProceduralSeedFanout (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double f0 = RetrievalFocusBias (0.5);
  const double s = RetrievalSensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double focus_scale = Clamp (1.0 - 0.20 * (f - f0), 0.85, 1.15);
  return std::max (
      1, static_cast<int> (std::round (Lerp (1.0, 4.0, s)
                                       * Lerp (1.0, 0.75, t)
                                       * focus_scale)));
}

inline double
RetrievalTotExpansionFactor (double F, double S, double T,
                             double confidence_scale)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double base = Lerp (0.22, 0.12, f) * Lerp (1.02, 0.96, t);
  const double span = Lerp (0.30, 0.42, s) * Lerp (1.06, 0.92, f)
                      * Lerp (1.02, 0.96, t);
  return 1.0 + base + span * Clamp (confidence_scale, 0.0, 1.0);
}

inline double
RetrievalTotDepthConfidenceThreshold (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.26, 0.12, s) * Lerp (0.94, 1.06, f)
                    * Lerp (1.04, 0.96, t),
                0.08, 0.40);
}

inline int
RetrievalTotMaxDepth (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Clamp (Lerp (4.0, 3.0, f) * Lerp (1.05, 0.95, t),
                         2.0, 4.0)));
}

struct RetrievalReconstructionPolicy
{
  double source_confidence_weight;
  double candidate_score_weight;
  double context_score_weight;
  double min_blend;
  double max_blend;
  double query_weight;
};

inline RetrievalReconstructionPolicy
RetrievalConstructiveReconstructionPolicy (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double source_raw = Lerp (0.38, 0.52, f);
  const double candidate_raw = Lerp (0.38, 0.30, f) * Lerp (1.00, 0.95, s);
  const double context_raw = Lerp (0.18, 0.26, s) * Lerp (1.04, 0.94, f);
  const double sum = std::max (1e-12,
                               source_raw + candidate_raw + context_raw);
  return {
    source_raw / sum,
    candidate_raw / sum,
    context_raw / sum,
    Clamp (Lerp (0.04, 0.07, s) * Lerp (1.04, 0.94, t), 0.02, 0.10),
    Clamp (Lerp (0.18, 0.30, s) * Lerp (1.04, 0.90, t), 0.12, 0.35),
    Clamp (Lerp (0.80, 0.72, t) * Lerp (1.02, 0.98, f), 0.60, 0.90)
  };
}

inline int
ReconstructionHistoryLimit (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double raw = Lerp (6.0, 14.0, t) * Lerp (1.08, 0.92, f)
                     * Lerp (0.94, 1.08, s);
  return static_cast<int> (std::round (Clamp (raw, 4.0, 24.0)));
}

inline int
ReconstructionPruneBatchLimit (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double raw = Lerp (64.0, 192.0, t) * Lerp (0.90, 1.05, f)
                     * Lerp (0.95, 1.10, s);
  return static_cast<int> (std::round (Clamp (raw, 32.0, 256.0)));
}

inline long long
ReconstructionMinUpdateIntervalMs (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double seconds = Lerp (300.0, 1800.0, t) * Lerp (0.90, 1.18, f)
                         * Lerp (1.10, 0.80, s);
  return static_cast<long long> (
      std::llround (Clamp (seconds, 120.0, 3600.0) * 1000.0));
}

inline int
RetrievalReconstructionUpdateCount (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double raw = Lerp (3.0, 1.0, f) * Lerp (0.85, 1.25, s)
                     * Lerp (1.20, 0.75, t);
  return static_cast<int> (std::round (Clamp (raw, 1.0, 4.0)));
}

inline int
RetrievalClusterLabelK (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (80.0, 24.0, f) * Lerp (1.10, 0.90, t)));
}

inline int
RetrievalClusterLabelM (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (5.0, 2.0, f) * Lerp (1.05, 0.95, t)));
}

inline int
RetrievalClusterLabelN (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (12.0, 6.0, f) * Lerp (0.95, 1.08, s)
                  * Lerp (1.05, 0.92, t)));
}

inline bool
RetrievalClusterLabelEnabled (double F, double S, double T)
{
  return RetrievalClusterLabelK (F, T) > 0 && RetrievalClusterLabelM (F, T) > 0
         && RetrievalClusterLabelN (F, S, T) > 0;
}

inline int
STMLabelClusterIterations (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Clamp (Lerp (30.0, 16.0, f) * Lerp (0.95, 1.08, s)
                             * Lerp (1.05, 0.92, t),
                         8.0, 40.0)));
}

inline int
STMShadowTTLSteps (double T)
{
  return static_cast<int> (
      std::round (Lerp (8.0, 32.0, RetrievalStabilityBias (T))));
}

inline int
STMShadowCapacity (double T)
{
  return static_cast<int> (
      std::round (Lerp (16.0, 64.0, RetrievalStabilityBias (T))));
}

// Section 4.4.3 - Boundary threshold
inline double
BoundaryThreshold (double F, double S);

inline double
STMShadowHardBoundaryThreshold (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double margin = Lerp (0.06, 0.14, f) * Lerp (1.05, 0.85, s)
                        * Lerp (0.95, 1.10, t);
  return Clamp (BoundaryThreshold (F, S) + margin, 0.55, 0.85);
}

inline int
STMShadowHardBoundaryRetainSteps (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return static_cast<int> (std::round (
      Clamp (Lerp (5.0, 2.0, f) * Lerp (1.10, 0.85, s)
                 * Lerp (0.85, 1.25, t),
             1.0, 8.0)));
}

inline int
STMLabelRouterTopK (double F, double T)
{
  // Reuse the retrieval cluster breadth as the flat-router budget.
  return RetrievalClusterLabelK (F, T);
}

inline double
STMLabelMinScore (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.0, 0.05, f) * Lerp (1.0, 0.65, s)
                    * Lerp (0.95, 1.05, t),
                0.0, 0.08);
}

inline int
STMLabelEdgeCapacity (double F, double S, double T)
{
  const int stm_capacity = STMShadowCapacity (T);
  const int labels_per_cluster = RetrievalClusterLabelN (F, S, T);
  return std::max (stm_capacity, stm_capacity * labels_per_cluster);
}

inline double
STMLabelConsolidationMinSimilarity (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return Clamp (Lerp (0.82, 0.93, f) * Lerp (0.98, 1.02, t)
                    * Lerp (0.98, 0.94, s),
                0.75, 0.95);
}

inline int
STMLabelConsolidationMaxLabels (double F, double T)
{
  const double f = RetrievalFocusBias (F);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (18.0, 8.0, f) * Lerp (1.05, 0.90, t)));
}

inline int
STMLabelConsolidationMaxLabels (double F, double S, double T)
{
  const double s = RetrievalSensitivityBias (S);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double scale = Clamp (1.0 + 0.10 * (s - s0), 0.92, 1.12);
  return std::max (
      1, static_cast<int> (
             std::round (scale
                         * static_cast<double> (
                             STMLabelConsolidationMaxLabels (F, T)))));
}

inline int
STMLabelConsolidationMaxUngrounded (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  return static_cast<int> (
      std::round (Lerp (5.0, 1.0, f) * Lerp (0.85, 1.10, s)
                  * Lerp (1.05, 0.85, t)));
}

inline int
STMLTMSourceSpanCandidateLimit (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double f0 = FocusBias (0.5);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = Lerp (21.0, -1.5, s) * Lerp (1.05, 0.95, t)
                     * Clamp (1.0 - 0.20 * (f - f0), 0.88, 1.12);
  return std::max (0, static_cast<int> (std::round (raw)));
}

inline int
STMLTMDurableMinLabels (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = Lerp (4.0, 1.0, f) * Lerp (0.95, 1.05, s)
                     * Lerp (1.05, 0.90, t);
  return std::max (1, static_cast<int> (std::round (raw)));
}

inline int
STMLTMDurableMaxLabels (double F, double S, double T)
{
  const int min_labels = STMLTMDurableMinLabels (F, S, T);
  const double f = FocusBias (F);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw_extra = Lerp (5.0, -1.0, f) * Lerp (1.05, 0.90, t);
  const int extra = std::max (
      0, static_cast<int> (std::round (raw_extra)));
  return std::max (
      min_labels, min_labels + extra);
}

inline int
STMLTMLabelPromptMinWords (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = Lerp (2.0, 3.0, f) * (1.0 - 0.06 * (s - SensitivityBias (0.5)))
                     * (1.0 + 0.04 * (t - 0.5));
  return static_cast<int> (std::round (Clamp (raw, 2.0, 3.0)));
}

inline int
STMLTMLabelPromptMaxWords (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = Lerp (6.0, 4.0, f) * (1.0 + 0.08 * (s - SensitivityBias (0.5)))
                     * (1.0 - 0.06 * (t - 0.5));
  return static_cast<int> (std::round (Clamp (raw, 3.0, 7.0)));
}

inline int
STMLTMLabelCooccurrenceMaxLabels (double F, double S, double T)
{
  return STMLTMDurableMaxLabels (F, S, T);
}

struct STMLTMSourceSpanPolicy
{
  int contextual_min_width;
  int contextual_max_width;
  int contextual_min_content_tokens;
  int action_object_max_tokens;
  int subject_search_max_gap;
  int proper_noun_max_parts;
  int phrase_min_width;
  int phrase_max_width;
  int singleton_min_chars;
};

inline STMLTMSourceSpanPolicy
STMLTMSourceSpanCandidatePolicy (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double breadth_scale = 1.0 + 0.10 * (s - SensitivityBias (0.5))
                               + 0.04 * (t - 0.5);
  const double precision_scale = 1.0 - 0.08 * (s - SensitivityBias (0.5))
                                 + 0.04 * (t - 0.5);

  STMLTMSourceSpanPolicy policy;
  policy.contextual_min_width = static_cast<int> (
      std::round (Clamp (Lerp (3.0, 5.0, f) * precision_scale, 3.0, 6.0)));
  policy.contextual_max_width = static_cast<int> (
      std::round (Clamp (Lerp (6.0, 4.0, f) * breadth_scale, 3.0, 7.0)));
  if (policy.contextual_max_width < policy.contextual_min_width)
    {
      policy.contextual_max_width = policy.contextual_min_width;
    }
  policy.contextual_min_content_tokens = static_cast<int> (
      std::round (Clamp (Lerp (2.0, 4.0, f) * precision_scale, 2.0, 5.0)));
  policy.action_object_max_tokens = static_cast<int> (
      std::round (Clamp (Lerp (4.0, 2.0, f) * breadth_scale, 1.0, 5.0)));
  policy.subject_search_max_gap = static_cast<int> (
      std::round (Clamp (Lerp (4.0, 2.0, f) * breadth_scale, 1.0, 5.0)));
  policy.proper_noun_max_parts = static_cast<int> (
      std::round (Clamp (Lerp (4.0, 2.0, f) * breadth_scale, 1.0, 5.0)));
  policy.phrase_min_width = static_cast<int> (
      std::round (Clamp (Lerp (2.0, 3.0, f) * precision_scale, 2.0, 4.0)));
  policy.phrase_max_width = static_cast<int> (
      std::round (Clamp (Lerp (4.0, 2.0, f) * breadth_scale, 2.0, 5.0)));
  if (policy.phrase_max_width < policy.phrase_min_width)
    {
      policy.phrase_max_width = policy.phrase_min_width;
    }
  policy.singleton_min_chars = static_cast<int> (
      std::round (Clamp (Lerp (4.0, 6.0, f) * precision_scale, 4.0, 7.0)));
  return policy;
}

inline int
STMLTMRelationEndpointAliasMinSharedTokens (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = Lerp (1.6, 2.6, f) * (1.0 - 0.10 * (s - SensitivityBias (0.5)))
                     * (1.0 + 0.06 * (t - 0.5));
  return static_cast<int> (std::round (Clamp (raw, 1.0, 3.0)));
}

inline double
STMLTMRelationEndpointMinConfidence (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.42, 0.58, f) * Lerp (1.08, 0.92, s)
                    * Lerp (0.98, 1.02, t),
                0.35, 0.65);
}

// --- Soft Anchor continuity policy ---

inline int
SoftAnchorActiveTTLSteps (double T)
{
  return static_cast<int> (
      std::round (Lerp (16.0, 56.0, Clamp (T, 0.0, 1.0))));
}

inline int
SoftAnchorRecentMemoryLimit (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = Lerp (20.0, 44.0, t) * Lerp (1.10, 0.90, f)
                     * Lerp (0.95, 1.05, s);
  return static_cast<int> (std::round (Clamp (raw, 8.0, 64.0)));
}

inline double
SoftAnchorSupportRawThreshold (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.70 + 0.10 * f - 0.05 * s + 0.04 * (t - 0.5),
                0.58, 0.84);
}

inline double
SoftAnchorSupportMarginThreshold (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.10 + 0.08 * f - 0.03 * s + 0.02 * (t - 0.5),
                0.03, 0.20);
}

inline double
SoftAnchorSupportEntropyMax (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.51 - 0.18 * f + 0.08 * s - 0.04 * t, 0.25,
                0.75);
}

inline double
SoftAnchorContradictionSemanticThreshold (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.78 + 0.08 * f - 0.04 * s + 0.02 * t, 0.68,
                0.90);
}

inline double
SoftAnchorContradictionEntityMax (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.505 - 0.08 * f + 0.04 * s - 0.02 * t, 0.35,
                0.65);
}

inline double
SoftAnchorDurableMarginThreshold (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.184 + 0.12 * f - 0.03 * s + 0.04 * t, 0.08,
                0.35);
}

inline double
SoftAnchorDurableEntropyMax (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.352 - 0.14 * f + 0.06 * s - 0.04 * t, 0.15,
                0.55);
}

struct SoftAnchorViewWeights
{
  double semantic;
  double entity;
  double full;
};

inline SoftAnchorViewWeights
SoftAnchorViewScoringWeights (double F, double S, double T,
                              double semantic_quality,
                              double entity_quality,
                              double full_quality)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  const double semantic_raw = (0.45 - 0.15 * f + 0.10 * s)
                              * Clamp (semantic_quality, 0.0, 1.0);
  const double entity_raw = (0.25 + 0.35 * f)
                            * Clamp (entity_quality, 0.0, 1.0);
  const double full_raw = (0.20 + 0.25 * t + 0.05 * f)
                          * Clamp (full_quality, 0.0, 1.0);
  const double sum = std::max (1e-9, semantic_raw + entity_raw + full_raw);
  return { semantic_raw / sum, entity_raw / sum, full_raw / sum };
}

struct SoftAnchorCandidateWeights
{
  double view;
  double source;
  double recency;
  double support;
  double boundary;
  double contradiction;
};

inline SoftAnchorCandidateWeights
SoftAnchorCandidateScoringWeights (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  const double sum = 1.55 + 1.05 * f + 0.50 * s + 0.40 * t;
  return { (0.55 + 0.20 * f) / sum,
           (0.20 + 0.20 * f + 0.10 * t) / sum,
           (0.20 + 0.20 * s + 0.10 * t) / sum,
           (0.15 + 0.45 * t) / sum,
           (0.25 + 0.35 * f + 0.15 * t) / sum,
           (0.20 + 0.30 * s + 0.20 * f) / sum };
}

inline double
SoftAnchorSourceAffinity (double F, double S, double T, bool same_source)
{
  if (same_source)
    {
      return 1.0;
    }
  const double f = Clamp (F, 0.0, 1.0);
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  const double scale = 1.0 - 0.20 * (f - 0.5) + 0.20 * (s - 0.5)
                       - 0.10 * (t - 0.5);
  return Clamp (0.25 * scale, 0.12, 0.42);
}

inline double
SoftAnchorRecencyTauSteps (double T)
{
  return std::max (1.0, Lerp (4.0, 32.0, Clamp (T, 0.0, 1.0)));
}

inline double
SoftAnchorSupportSaturation (double T)
{
  return Lerp (1.5, 4.0, Clamp (T, 0.0, 1.0));
}

inline double
SoftAnchorContradictionSaturation (double S)
{
  return Lerp (2.5, 1.0, Clamp (S, 0.0, 1.0));
}

inline double
SoftAnchorBoundaryShiftScale (double F, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.40 + 0.40 * f + 0.20 * t, 0.35, 1.10);
}

inline double
SoftAnchorSoftmaxTemperature (double F)
{
  return Lerp (4.0, 14.0, Clamp (F, 0.0, 1.0));
}

inline double
SoftAnchorSingleViewEntityQuality (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double scale = 1.0 - 0.10 * (f - f0) + 0.12 * (s - s0)
                       - 0.06 * (Clamp (T, 0.0, 1.0) - 0.5);
  return Clamp (0.50 * scale, 0.35, 0.65);
}

inline double
SoftAnchorActiveDecayMultiplier (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double scale = 1.0 + 0.08 * (f - f0) - 0.12 * (s - s0)
                       + 0.18 * (Clamp (T, 0.0, 1.0) - 0.5);
  return Clamp (2.0 * scale, 1.25, 3.0);
}

inline double
SoftAnchorGenericScore (double F, double S, double T, double entity_quality,
                        double real_entropy, double specificity)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double entity_weight = 0.40 * Clamp (1.0 + 0.08 * (f - f0)
                                                 - 0.06 * (s - s0),
                                             0.80, 1.20);
  const double entropy_weight = 0.35 * Clamp (1.0 - 0.05 * (f - f0)
                                                  + 0.08 * (s - s0)
                                                  - 0.04 * (t - 0.5),
                                              0.80, 1.20);
  const double specificity_weight = 0.25 * Clamp (1.0 + 0.10 * (f - f0)
                                                      - 0.04 * (s - s0)
                                                      + 0.04 * (t - 0.5),
                                                  0.80, 1.20);
  const double sum = std::max (1e-12,
                               entity_weight + entropy_weight
                                   + specificity_weight);
  return Clamp ((entity_weight * (1.0 - Clamp (entity_quality, 0.0, 1.0))
                 + entropy_weight * Clamp (real_entropy, 0.0, 1.0)
                 + specificity_weight
                       * (1.0 - Clamp (specificity, 0.0, 1.0)))
                    / sum,
                0.0, 1.0);
}

inline double
SoftAnchorNoneScore (double F, double S, double T, double generic,
                     double real_entropy, double no_real_penalty,
                     double top_contradiction)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double base = 0.20 * Clamp (1.0 + 0.08 * (f - f0)
                                        - 0.08 * (s - s0)
                                        + 0.06 * (t - 0.5),
                                    0.75, 1.25);
  const double generic_weight = 0.55 * Clamp (1.0 + 0.10 * (f - f0)
                                                  - 0.06 * (s - s0),
                                              0.80, 1.20);
  const double entropy_weight = 0.20 * Clamp (1.0 - 0.04 * (f - f0)
                                                  + 0.08 * (s - s0),
                                              0.80, 1.20);
  const double no_real_weight = 0.20 * Clamp (1.0 + 0.06 * (f - f0)
                                                 - 0.04 * (s - s0)
                                                 + 0.06 * (t - 0.5),
                                             0.80, 1.20);
  const double contradiction_weight = 0.15 * Clamp (1.0 + 0.08 * (s - s0),
                                                    0.80, 1.20);
  return Clamp (base + generic_weight * Clamp (generic, 0.0, 1.0)
                    + entropy_weight * Clamp (real_entropy, 0.0, 1.0)
                    + no_real_weight * Clamp (no_real_penalty, 0.0, 1.0)
                    + contradiction_weight
                          * Clamp (top_contradiction, 0.0, 1.0),
                0.0, 1.0);
}

inline double
SoftAnchorNewScore (double F, double S, double T, double info,
                    double boundary_score, double best_real, double generic)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double base = 0.15 * Clamp (1.0 - 0.04 * (f - f0)
                                        + 0.08 * (s - s0)
                                        - 0.08 * (t - 0.5),
                                    0.75, 1.25);
  const double info_weight = 0.45 * Clamp (1.0 + 0.04 * (f - f0)
                                               + 0.06 * (s - s0)
                                               - 0.06 * (t - 0.5),
                                           0.80, 1.20);
  const double boundary_weight = 0.25 * Clamp (1.0 + 0.10 * (s - s0)
                                                   - 0.04 * (t - 0.5),
                                               0.80, 1.20);
  const double novelty_weight = 0.20 * Clamp (1.0 + 0.06 * (s - s0)
                                                  - 0.08 * (t - 0.5),
                                              0.80, 1.20);
  const double generic_penalty = 0.35 * Clamp (1.0 + 0.08 * (f - f0)
                                                   - 0.04 * (s - s0),
                                               0.80, 1.20);
  return Clamp (base + info_weight * Clamp (info, 0.0, 1.0)
                    + boundary_weight * Clamp (boundary_score, 0.0, 1.0)
                    + novelty_weight * (1.0 - Clamp (best_real, 0.0, 1.0))
                    - generic_penalty * Clamp (generic, 0.0, 1.0),
                0.0, 1.0);
}

struct SoftAnchorDecisionPolicy
{
  double new_threshold;
  double keep_threshold;
  double tentative_threshold;
  double margin_min;
  double ambiguous_entropy;
  double generic_hard_threshold;
  int keep_count;
  double centroid_alpha;
  int durable_support_required;
};

inline SoftAnchorDecisionPolicy
SoftAnchorDecisionPolicyForKnobs (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  return {
    Clamp (Lerp (0.44, 0.62, f) * Lerp (1.05, 0.90, s), 0.0, 1.0),
    Clamp (Lerp (0.035, 0.080, f) * Lerp (1.15, 0.75, s), 0.0, 1.0),
    Clamp (Lerp (0.38, 0.58, f) * Lerp (1.10, 0.85, s), 0.0, 1.0),
    Clamp (Lerp (0.03, 0.15, f) * Lerp (1.05, 0.75, s), 0.0, 1.0),
    Clamp (Lerp (0.85, 0.55, f) * Lerp (1.05, 1.15, s), 0.0, 1.0),
    Clamp (Lerp (0.82, 0.65, f) * Lerp (1.05, 0.95, s), 0.0, 1.0),
    std::max (2, std::min (10, static_cast<int> (
                                  std::round (Lerp (8.0, 3.0, f)
                                              * Lerp (1.10, 0.85, t)
                                              + 2.0 * s)))),
    Clamp (0.35 - 0.15 * t + 0.10 * s - 0.04 * (f - 0.5), 0.05, 0.60),
    std::max (2, std::min (6, static_cast<int> (
                                  std::ceil (2.0 + 1.5 * f + 2.0 * t
                                             - 0.5 * (s - 0.5)))))
  };
}

inline int
AdjacentWindow (double F)
{
  // adjacent_window(F) = round(lerp(adjacent_max, adjacent_min, F)),
  // adjacent_min=1, adjacent_max=6
  return static_cast<int> (std::round (Lerp (6.0, 1.0, FocusBias (F))));
}

inline int
MaxWaitTokens (double F)
{
  // max_wait_tokens(F) = round(lerp(wait_max, wait_min, F)),
  // wait_min=16, wait_max=128
  return static_cast<int> (std::round (Lerp (128.0, 16.0, FocusBias (F))));
}

inline int
CheckIntervalTokens (double S)
{
  // check_interval(S) = round(lerp(check_max_tokens, check_min_tokens, S)),
  // check_min_tokens=8, check_max_tokens=64
  return static_cast<int> (
      std::round (Lerp (64.0, 8.0, SensitivityBias (S))));
}

// --- Section 10.4: Streaming Pacing Parameters ---

inline double
StreamingPacingThreshold (double S)
{
  // pacing_thresh(S) = lerp(0.3, 0.05, S)
  // Higher sensitivity = lower threshold = more frequent pacing checks
  return Lerp (0.3, 0.05, SensitivityBias (S));
}

inline double
MaxWaitDrift (double F)
{
  // max_wait_drift(F) = lerp(1.2, 0.30, F)
  // Higher focus = lower max drift = more aggressive forced checks
  return Lerp (1.2, 0.30, FocusBias (F));
}

inline double
StreamingRetrievalBiasThreshold (double F, double S, double T)
{
  // Retrieval bias gates routine streaming checks; boundary and force-check
  // paths still bypass this gate. Higher Focus asks for cleaner retrieval
  // pressure, higher Sensitivity lets weak pressure through, and higher
  // Stability slightly resists reactive lookup.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.08, 0.24, f) * Lerp (1.10, 0.75, s)
                    * Lerp (0.95, 1.10, t),
                0.04, 0.30);
}

inline int
GraphDepth (double T)
{
  // graph_depth(T): small traversal depth, in [2,3].
  return static_cast<int> (std::round (Lerp (3.0, 2.0, Clamp (T, 0.0, 1.0))));
}

inline int
RetrievalGraphDepth (double T)
{
  return static_cast<int> (
      std::round (Lerp (3.0, 2.0, RetrievalStabilityBias (T))));
}

inline std::pair<double, double>
RetrievalDiversificationWeights (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);

  const double w_rel_raw = Lerp (0.65, 0.94, f) * Lerp (1.0, 0.85, s)
                           * Lerp (1.0, 0.90, t);
  const double w_div_raw = Lerp (0.35, 0.06, f) * Lerp (0.75, 1.15, s)
                           * Lerp (0.85, 0.65, t);
  const double sum = w_rel_raw + w_div_raw;
  if (sum <= 0.0)
    {
      return { 1.0, 0.0 };
    }
  return { w_rel_raw / sum, w_div_raw / sum };
}

inline double
RetrievalTotDiversificationScale (double F, double S, double T,
                                  double confidence_scale)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  const double base = Lerp (0.95, 0.75, Clamp (confidence_scale, 0.0, 1.0));
  const double knob_scale = 1.0 - 0.05 * (f - f0) + 0.04 * (s - s0)
                            - 0.04 * (t - t0);
  return Clamp (base * knob_scale, 0.65, 1.0);
}

inline double
RetrievalProceduralSeedMinScore (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.44, 0.72, f) * Lerp (1.08, 0.92, s)
                    * Lerp (0.96, 1.04, t),
                0.30, 0.80);
}

// Small boost for consolidated association memories during retrieval ranking.
inline double
AssociationBoost (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  // 0.015..0.06 scaled by Sensitivity; higher Focus reduces extra breadth.
  return Lerp (0.015, 0.06, s) * Lerp (1.0, 0.7, f) * Lerp (1.0, 0.9, t);
}

// Baseline salience for seeded label-bank entries.
inline double
LabelBankSalience (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double mix = 0.5 * f + 0.5 * s;
  return Clamp (Lerp (0.35, 0.65, mix) * Lerp (1.0, 0.9, t), 0.0, 1.0);
}

inline double
LabelSalienceFallback (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.50 + 0.04 * (s - s0) - 0.03 * (f - f0)
                    + 0.03 * (t - 0.5),
                0.40, 0.60);
}

inline int
PriorMass (double T)
{
  return static_cast<int> (std::round (Lerp (2.0, 32.0, T)));
}

// Rate estimation smoothing time constant (seconds)
inline double
TauDt (double T)
{
  return Lerp (0.5, 2.0, Clamp (T, 0.0, 1.0));
}

// Minimum delta time (seconds)
inline double
DeltaTMin (double T)
{
  return std::pow (10.0, -4.0 + 2.0 * Clamp (T, 0.0, 1.0));
}

// Minimum cadence floor for dt_ema (seconds)
inline double
DtFloor (double T)
{
  return Lerp (0.1, 0.5, Clamp (T, 0.0, 1.0));
}

// ESS cap (stability-derived)
inline double
EssCap (double T)
{
  return Lerp (30.0, 120.0, Clamp (T, 0.0, 1.0));
}

inline double
ThresholdObservedQuantile (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.90 + 0.03 * (f - f0) - 0.02 * (s - s0)
                    + 0.02 * (t - 0.5),
                0.75, 0.97);
}

inline int
RecentScoreHistoryLimit (double T)
{
  const double t = Clamp (T, 0.0, 1.0);
  return static_cast<int> (std::round (Lerp (512.0, 1536.0, t)));
}

inline double
RateTauStabilityExponentScale (double T)
{
  return 3.0 * Clamp (T, 0.0, 1.0);
}

inline double
TPrior (double /*F*/, double S, double T)
{
  // T_prior(F,S,T) = lerp(0.10, 0.30, T) × (1 − 0.3×S)
  return Lerp (0.10, 0.30, T) * (1.0 - 0.3 * SensitivityBias (S));
}

inline double
TauM (double T)
{
  return Lerp (10.0, 200.0, T);
}

inline double
ComputeMaturity (int count, double T)
{
  // maturity(t) = 1 − exp(−count/τ_m(T))
  const double tau_m = TauM (T);
  return 1.0 - std::exp (-static_cast<double> (count) / tau_m);
}

inline double
TMin (int count, double T)
{
  // Tmin(t) = lerp(0.01, 0.05, maturity(t))
  return Lerp (0.01, 0.05, ComputeMaturity (count, T));
}

inline double
TMax (int count, double T)
{
  // Tmax(t) = lerp(0.99, 0.95, maturity(t))
  return Lerp (0.99, 0.95, ComputeMaturity (count, T));
}

inline double
MaxDeltaTPerMin (int count, double T)
{
  // max_ΔT_per_min(t) = lerp(0.30, 0.10, maturity(t))
  return Lerp (0.30, 0.10, ComputeMaturity (count, T));
}

// Homeostatic gain for threshold adjustment
inline double
KappaR (double /*F*/, double S, double T)
{
  return Lerp (0.06, 0.14, SensitivityBias (S))
         * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

inline double
CapHomeo (double /*F*/, double S, double T, double hysteresis)
{
  return Lerp (0.35, 0.15, Clamp (T, 0.0, 1.0))
         * Lerp (0.8, 1.2, SensitivityBias (S))
         * hysteresis;
}

// Sensitivity-based threshold adjustment
inline double
KappaSens (double /*F*/, double S, double T)
{
  return Lerp (0.04, 0.12, SensitivityBias (S))
         * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

inline double
CapSens (double /*F*/, double S, double T, double hysteresis)
{
  return Lerp (0.30, 0.10, Clamp (T, 0.0, 1.0))
         * Lerp (0.9, 1.1, SensitivityBias (S))
         * hysteresis;
}

inline double
SigmaRef (double S, double T)
{
  return Lerp (0.08, 0.14, SensitivityBias (S))
         * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

// Precision-based threshold adjustment
inline double
KappaPrec (double F, double /*S*/, double T)
{
  return Lerp (0.04, 0.10, FocusBias (F))
         * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

inline double
CapPrec (double F, double /*S*/, double T, double hysteresis)
{
  return Lerp (0.25, 0.08, Clamp (T, 0.0, 1.0))
         * Lerp (0.9, 1.1, FocusBias (F))
         * hysteresis;
}

// --- 0.3 Global EWMA & Uncertainty Schedules ---

inline double
AlphaU (double T)
{
  const double kAlphaUMin = 0.10;
  const double kAlphaUSpan = 0.60;
  return kAlphaUMin + (1.0 - T) * kAlphaUSpan;
}

inline double
AlphaT (double T, double u_t)
{
  // α_T(t) = α_min_T + (1 − T) × α_span_T × u(t)
  const double kAlphaMinT = 0.02;
  const double kAlphaSpanT = 0.18;
  return kAlphaMinT + (1.0 - T) * kAlphaSpanT * u_t;
}

inline double
AlphaF (double F, double u_t)
{
  // α_F(t) = α_min_F + F × α_span_F × u(t)
  const double kAlphaMinF = 0.05;
  const double kAlphaSpanF = 0.45;
  return kAlphaMinF + FocusBias (F) * kAlphaSpanF * u_t;
}

inline double
AlphaS (double S, double u_t)
{
  const double kAlphaMinS = 0.05;
  const double kAlphaSpanS = 0.35;
  return kAlphaMinS + SensitivityBias (S) * kAlphaSpanS * u_t;
}

// Uncertainty variance normalization ceiling
inline double
VarScoreMax (double S)
{
  return Lerp (0.15, 0.35, SensitivityBias (S));
}

// Neutral structural coherence fallback
inline double
CoherenceNeutral (double T)
{
  return Lerp (0.45, 0.55, Clamp (T, 0.0, 1.0));
}

// EMA prediction update rate
inline double
PredEmaBeta (double T)
{
  return Lerp (0.25, 0.02, Clamp (T, 0.0, 1.0));
}

// Surprise reference floor for EMA prediction error
inline double
SurpriseErrRef (double S)
{
  return Lerp (0.25, 0.05, SensitivityBias (S));
}

// Surprise sigmoid gain
inline double
SurpriseGain (double S, double T)
{
  return Lerp (6.0, 14.0, SensitivityBias (S))
       * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

// RLS initialization scale
inline double
BlenderPInit (double T)
{
  return Lerp (500.0, 2000.0, 1.0 - Clamp (T, 0.0, 1.0));
}

inline double
BlenderRLSObservationTau (double T)
{
  return Lerp (20.0, 80.0, Clamp (T, 0.0, 1.0));
}

inline double
BlenderRLSForgettingFactor (double T)
{
  return Clamp (0.90 + 0.09 * Clamp (T, 0.0, 1.0), 0.90, 0.99);
}

inline int
BlenderUpdateInterval (double T)
{
  return std::max (
      1, static_cast<int> (std::round (Lerp (1.0, 8.0, Clamp (T, 0.0, 1.0)))));
}

struct BlenderOutcomeWeights
{
  double used;
  double predictive;
  double uncertainty;
  double user;
};

inline BlenderOutcomeWeights
BlenderOutcomeScoringWeights (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return { 0.4 + 0.2 * f,
           0.3 + 0.2 * s,
           0.2 + 0.2 * t,
           Clamp (0.10
                      * (1.0 - 0.10 * (f - f0) + 0.12 * (s - s0)
                         - 0.06 * (t - 0.5)),
                  0.06, 0.14) };
}

inline double
BlenderOutcomeAlpha (double T)
{
  return Lerp (0.12, 0.03, Clamp (T, 0.0, 1.0));
}

inline int
PredictionHorizon (double F)
{
  return std::max (
      1, static_cast<int> (
             std::round (Lerp (2.0, 8.0, FocusBias (F)))));
}

inline double
PredictionConfidenceThreshold (double F)
{
  return Lerp (0.3, 0.7, FocusBias (F));
}

inline double
PredictivePreActivationDecay (double T)
{
  return Lerp (0.7, 0.3, Clamp (T, 0.0, 1.0));
}

inline double
PredictiveSurpriseSensitivity (double S, double T)
{
  return SensitivityBias (S) * Lerp (2.0, 0.5, Clamp (T, 0.0, 1.0));
}

inline double
PredictiveBaseDelta (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double pad = PredictivePreActivationDecay (T);
  return Clamp (0.02 * (1.0 - (pad - 0.3))
                    * (1.0 - 0.06 * (f - f0) + 0.08 * (s - s0)),
                0.006, 0.030);
}

inline double
PredictiveSurpriseUpdateRate (double S, double T)
{
  return Lerp (0.2, 0.02, Clamp (T, 0.0, 1.0)) * SensitivityBias (S);
}

inline double
PredictiveDeltaCap (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.20
                    * (1.0 - 0.06 * (f - f0) + 0.10 * (s - s0)
                       - 0.08 * (t - 0.5)),
                0.10, 0.28);
}

// Rate observation window (seconds)
inline int
WRateSeconds (double T)
{
  return static_cast<int> (std::round (Lerp (60.0, 300.0, T)));
}

// Consolidation interval (seconds) — Algorithm 28
inline int
ConsolidationIntervalSeconds (double T)
{
  // consolidation interval = lerp(300, 3600, T)  # 5 min → 1 hour
  return static_cast<int> (std::round (Lerp (300.0, 3600.0, T)));
}

// Escalation multiplier for consolidation urgency — Algorithm 28b
inline double
ConsolidationEscalationMultiplier (double T)
{
  // escalates required thresholds with stability (1.5 → 2.5)
  return Lerp (1.5, 2.5, Clamp (T, 0.0, 1.0));
}

// Required consolidation interval (seconds) — Algorithm 28b
inline int
ConsolidationRequiredIntervalSeconds (double T)
{
  return static_cast<int> (
      std::round (ConsolidationIntervalSeconds (T)
                  * ConsolidationEscalationMultiplier (T)));
}

// Consolidation rate (writes/min) — Section 7.1
inline double
ConsolidationRate (double T, double S)
{
  // rate_consolidate = (1 / max(interval, 1)) × (0.3 + 0.7T) × (1 − 0.5S)
  // Reference: algorithms.md Section 7.1, lines 1015-1017
  const double interval
      = static_cast<double> (ConsolidationIntervalSeconds (T));
  // Convert to writes/min to match m_rate units.
  return (60.0 / std::max (interval, 1.0)) * (0.3 + 0.7 * T)
         * (1.0 - 0.5 * SensitivityBias (S));
}

// Idle required seconds — Algorithm 28b
inline int
IdleRequiredSeconds (double T)
{
  // idle_required_seconds(T) = round(0.25 × w_rate_seconds(T))
  return static_cast<int> (
      std::round (0.25 * static_cast<double> (WRateSeconds (T))));
}

// Extraction batch size — Algorithm 29c
inline int
ExtractionBatchSize (double T)
{
  // extraction_batch_size = round(lerp(8, 32, T))
  return static_cast<int> (std::round (Lerp (8.0, 32.0, T)));
}

// Max extraction jobs per consolidation cycle — Algorithm 32
inline int
MaxExtractionsPerCycle (double T)
{
  // max_extractions_per_cycle = round(lerp(20, 5, T))
  return static_cast<int> (std::round (Lerp (20.0, 5.0, Clamp (T, 0.0, 1.0))));
}

// Capacity trigger threshold — Algorithm 28 (capacity trigger).
inline long long
ConsolidationThresholdCount (double T)
{
  // Derive a threshold from stability-scaled windows so it grows with system
  // maturity and persistence. This stays knob-only and avoids extra tunables.
  const long long n = static_cast<long long> (NCtx (Clamp (T, 0.0, 1.0)));
  const long long w = static_cast<long long> (WScore (Clamp (T, 0.0, 1.0)));
  const long long thr = n * w;
  return (thr < 1) ? 1 : thr;
}

// Required consolidation threshold — Algorithm 28b
inline long long
ConsolidationRequiredCount (double T)
{
  const double base
      = static_cast<double> (ConsolidationThresholdCount (T));
  const double scaled = base * ConsolidationEscalationMultiplier (T);
  return static_cast<long long> (std::max (1.0, std::round (scaled)));
}

// Minimum cluster size for extraction — Algorithm 29c
inline int
MinClusterSizeForExtraction (double F)
{
  // min_cluster_size_for_extraction = round(lerp(3, 10, F))
  return static_cast<int> (std::round (Lerp (3.0, 10.0, FocusBias (F))));
}

inline int
ShallowConsolidationMaxLabels (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = Lerp (2.0, 6.0, s) * Lerp (1.0, 0.70, f)
                     * Lerp (1.04, 0.96, t);
  return std::max (1, static_cast<int> (std::round (raw)));
}

inline double
ShallowConsolidationLabelMinSimilarity (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double s0 = SensitivityBias (0.5);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.30, 0.60, f) * Lerp (1.0, 1.1, t)
                    * Clamp (1.0 - 0.06 * (s - s0), 0.94, 1.04),
                0.15, 0.95);
}

struct ConsolidationSummaryEvidenceBudget
{
  int max_source_texts;
  int max_total_chars;
  int max_text_chars;
};

inline ConsolidationSummaryEvidenceBudget
ConsolidationSummaryEvidenceBudgetForKnobs (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double s0 = SensitivityBias (0.5);
  const double breadth_scale = Clamp (1.0 + 0.12 * (s - s0), 0.88, 1.14);
  return {
    std::max (
        2, static_cast<int> (
               std::round (Lerp (3.0, 8.0, f) * breadth_scale))),
    std::max (
        256, static_cast<int> (
                 std::round (Lerp (1200.0, 3600.0, t) * breadth_scale))),
    std::max (
        128, static_cast<int> (
                 std::round (Lerp (300.0, 900.0, f) * breadth_scale)))
  };
}

// --- Consolidation Clustering (Section 7.3-7.4) ---

// Merge threshold for clustering — Section 7.5.1
inline double
MergeThreshold (double F)
{
  // merge_threshold(F) = lerp(0.85, 0.95, F)
  // Higher focus = stricter merging (higher similarity required)
  // Also used for co-occurrence edges in Section 7.5.1
  return Lerp (0.85, 0.95, FocusBias (F));
}

// Minimum cluster size — Section 7.4
inline int
MinClusterSize (double F)
{
  // min_cluster_size(F) = round(lerp(3, 10, F))
  // Higher focus = larger minimum clusters required
  return static_cast<int> (std::round (Lerp (3.0, 10.0, FocusBias (F))));
}

inline int
ConsolidationMaxClusters (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = Lerp (140.0, 40.0, f)
                     * (1.0 + 0.12 * (s - SensitivityBias (0.5))
                        + 0.10 * (t - 0.5));
  return static_cast<int> (std::round (Clamp (raw, 32.0, 256.0)));
}

// Label frequency threshold — Section 7.4
inline int
LabelFrequencyThreshold (double T)
{
  // label_frequency_threshold(T) = round(lerp(5, 15, T))
  // Higher stability = higher frequency required for label to be notable
  return static_cast<int> (std::round (Lerp (5.0, 15.0, Clamp (T, 0.0, 1.0))));
}

// Extraction interval — Section 7.4
inline int
ExtractionIntervalSeconds (double T)
{
  // extraction_interval(T) = lerp(300, 3600, T)
  // Section 7.4: 5 minutes to 1 hour based on stability
  return static_cast<int> (std::round (Lerp (300.0, 3600.0, Clamp (T, 0.0, 1.0))));
}

// Retention window size (w_ret) — Algorithm 0.2
// Spec (§2.3.2, line 339): w_ret(T) = round(lerp(10, 50, T))
inline int
WRet (double T)
{
  return static_cast<int> (std::round (Lerp (10.0, 50.0, T)));
}

inline int
ObservedRetentionHistoryLimit (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return static_cast<int> (
      std::round (Clamp (Lerp (192.0, 384.0, t) * Lerp (1.08, 0.94, f)
                             * Lerp (0.95, 1.05, s),
                         96.0, 512.0)));
}

inline int
RecentRetrievedIdWindow (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return static_cast<int> (
      std::round (Clamp (Lerp (768.0, 1536.0, t) * Lerp (1.10, 0.90, f)
                             * Lerp (0.95, 1.05, s),
	                         256.0, 2048.0)));
}

struct MemoryTraceTauMultipliers
{
  double fast;
  double medium;
  double slow;
  double ultra;
};

inline MemoryTraceTauMultipliers
MemoryTraceTauPolicy (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double reactive_scale = 1.0 + 0.14 * (s - s0) - 0.08 * (f - f0)
                                + 0.06 * (t - 0.5);
  const double stable_scale = 1.0 + 0.24 * (t - 0.5) - 0.06 * (s - s0);
  return {
    Clamp (0.10 * reactive_scale, 0.05, 0.20),
    Clamp (0.50 * (1.0 + 0.12 * (t - 0.5) - 0.06 * (f - f0)), 0.30, 0.80),
    Clamp (2.00 * stable_scale, 1.20, 3.20),
    Clamp (8.00 * stable_scale * (1.0 + 0.10 * (f - f0)), 4.00,
           12.00)
  };
}

inline double
MemoryTraceCoupling (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp ((0.05 + 0.10 * t)
                    * (1.0 - 0.10 * (f - f0) + 0.12 * (s - s0)),
                0.02, 0.20);
}

struct MemoryTraceWeights
{
  double fast;
  double medium;
  double slow;
  double ultra;
};

inline MemoryTraceWeights
MemoryTraceWeightPolicy (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return {
    Clamp ((0.40 - 0.25 * t)
               * (1.0 - 0.08 * (f - f0) - 0.10 * (s - s0)),
           0.05, 0.50),
    Clamp (0.25 * (1.0 - 0.04 * (f - f0) - 0.04 * (t - 0.5)),
           0.12, 0.35),
    Clamp ((0.20 + 0.15 * t)
               * (1.0 + 0.08 * (f - f0) + 0.06 * (s - s0)),
           0.12, 0.45),
    Clamp ((0.15 + 0.10 * t)
               * (1.0 + 0.10 * (f - f0) + 0.06 * (s - s0)),
           0.08, 0.35)
  };
}

inline MemoryTraceWeights
MemoryInitialTracePolicy (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double stable_seed = Clamp ((t - 0.5) * 2.0, 0.0, 1.0);
  const double slow_seed = Clamp (0.04 * stable_seed
                                      * (1.0 - 0.10 * (f - f0)
                                         + 0.10 * (s - s0)),
                                  0.0, 0.08);
  const double ultra_seed = Clamp (0.02 * stable_seed
                                       * (1.0 - 0.06 * (f - f0)
                                          + 0.08 * (s - s0)),
                                   0.0, 0.04);
  const double medium_seed = Clamp (0.08 * stable_seed
                                        * (1.0 - 0.12 * (f - f0)
                                           + 0.08 * (s - s0)),
                                    0.0, 0.14);
  return {
    Clamp (1.0 - 0.5 * medium_seed - slow_seed - ultra_seed, 0.80, 1.0),
    medium_seed,
    slow_seed,
    ultra_seed
  };
}

inline double
MemoryInitialStrengthPolicy (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double instability = std::max (0.0, 0.5 - t);
  const double sensitivity_pressure = std::max (0.0, s - s0);
  const double diffuse_pressure = std::max (0.0, f0 - f);
  return Clamp (1.0
                    - 0.05 * instability
                          * (1.0 + 0.50 * sensitivity_pressure
                             + 0.25 * diffuse_pressure),
                0.90, 1.0);
}

inline double
MemoryInitialStabilityPolicy (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double instability = std::max (0.0, 0.5 - t);
  const double sensitivity_pressure = std::max (0.0, s - s0);
  const double diffuse_pressure = std::max (0.0, f0 - f);
  return Clamp (1.0
                    - 0.12 * instability
                          * (1.0 + 0.25 * sensitivity_pressure
                             + 0.15 * diffuse_pressure),
                0.82, 1.0);
}

inline MemoryTraceWeights
MemoryStoredTraceFallbackPolicy (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return {
    1.0,
    Clamp (0.50 * (1.0 - 0.08 * (f - f0) + 0.08 * (s - s0)
                   + 0.10 * (t - 0.5)),
           0.30, 0.70),
    Clamp (0.20 * (1.0 + 0.08 * (f - f0) + 0.06 * (s - s0)
                   + 0.18 * (t - 0.5)),
           0.10, 0.35),
    Clamp (0.05 * (1.0 + 0.10 * (f - f0) + 0.06 * (s - s0)
                   + 0.25 * (t - 0.5)),
           0.02, 0.12)
  };
}

inline double
MemoryContextualGainWindow (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (Lerp (8.0, 32.0, Clamp (T, 0.0, 1.0))
                    * (1.0 - 0.06 * (f - f0) + 0.08 * (s - s0)),
                4.0, 48.0);
}

inline int
MemoryTraceCount (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double raw = (2.0 + 2.0 * Clamp (T, 0.0, 1.0))
                     * (1.0 + 0.08 * (f - f0) + 0.06 * (s - s0));
  return std::max (1, std::min (4, static_cast<int> (std::round (raw))));
}

// Periphery cutoff — Algorithm 0.2
inline double
PeripheryCutoff (double T)
{
  return Lerp (0.03, 0.20, T);
}

// Fact-evidence eviction floor — memories supporting active facts are protected
// proportionally to Stability. At T=0 no protection; at T=1 floor reaches
// the periphery cutoff itself (near-immune).
inline double
FactEvictionFloor (double T)
{
  return Lerp (0.0, PeripheryCutoff (T), T);
}

// --- Stability prior helpers (Algorithm 5) ---
inline double
BaseBandPrior (double T)
{
  // band_min = 0.02; band_max = 0.25
  return Lerp (0.02, 0.25, T);
}

inline double
BaseHalfLifePrior (double T)
{
  // hl_min = 120.0; hl_max = 43200.0 (2 min → 12 h)
  const double hl_min = 120.0;
  const double hl_max = 43200.0;
  return std::exp (std::log (hl_min) + T * std::log (hl_max / hl_min));
}

inline double
ClampHalfLife (double value)
{
  const double hl_min = 120.0;
  const double hl_max = 43200.0;
  return Clamp (value, hl_min, hl_max);
}

// --- Algorithm 20 (Reconsolidation) Helpers ---
inline double
TauLabile (double T)
{
  // τ_labile = lerp(30, 300, T) seconds
  return Lerp (30.0, 300.0, T);
}

inline double
ReconsolidationGain (double T)
{
  // reconsolidation_gain = lerp(0.2, 0.02, T)
  return Lerp (0.2, 0.02, T);
}

inline double
ReconsolidationDriftClamp (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.30
                    * (1.0 - 0.06 * (f - f0) + 0.08 * (s - s0)
                       - 0.18 * (t - 0.5)),
                0.12, 0.40);
}

inline double
ReconsolidationDriftSkipEpsilon (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.001
                    * (1.0 + 0.08 * (f - f0) - 0.10 * (s - s0)
                       + 0.12 * (t - 0.5)),
                0.0004, 0.0020);
}

inline double
ReconsolidationPrimaryDriftMagnitude (double F, double S, double T,
                                      double current_lability,
                                      double contextual_relevance,
                                      double recon_mod_scale)
{
  const double s = SensitivityBias (S);
  const double s0 = SensitivityBias (0.5);
  const double sensitivity_term = Clamp (
      Clamp (S, 0.0, 1.0) * (1.0 + 0.20 * (s - s0)), 0.0, 1.0);
  const double raw = (1.0 - Clamp (T, 0.0, 1.0)) * sensitivity_term
                     * std::max (0.0, current_lability)
                     * Clamp (contextual_relevance, 0.0, 1.0)
                     * ReconsolidationGain (T)
                     * std::max (0.0, recon_mod_scale);
  return Clamp (raw, 0.0, ReconsolidationDriftClamp (F, S, T));
}

inline double
RippleStrengthMin (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.01
                    * (1.0 + 0.08 * (f - f0) - 0.20 * (s - s0)
                       + 0.30 * (t - 0.5)),
                0.003, 0.020);
}

inline double
RippleDriftCapFactor (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.50
                    * (1.0 - 0.06 * (f - f0) + 0.10 * (s - s0)
                       - 0.12 * (t - 0.5)),
                0.30, 0.70);
}

inline double
ReconsolidationUncertaintyRelevanceWeight (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.50
                    * (1.0 + 0.06 * (f - f0) - 0.06 * (s - s0)
                       + 0.08 * (t - 0.5)),
                0.30, 0.70);
}

inline double
RippleDecay (double T)
{
  // ripple_decay = lerp(0.5, 0.1, T)
  return Lerp (0.5, 0.1, T);
}

inline int
RippleDepth (double T)
{
  // ripple_depth = round(lerp(2, 1, T)) — higher stability = shallower ripple
  return static_cast<int> (std::round (Lerp (2.0, 1.0, T)));
}

inline int
ReconsolidationRippleReconstructionLimit (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = (1.0 - t) * s * Lerp (1.0, 0.4, f) * 5.0 - 1.0;
  return static_cast<int> (std::round (Clamp (raw, 0.0, 6.0)));
}

inline int
ReconsolidationPrimaryReconstructionLimit (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double raw = (1.0 - t) * s * Lerp (1.1, 0.75, f) * 3.0;
  return static_cast<int> (std::round (Clamp (raw, 0.0, 3.0)));
}

inline double
LabilitySusceptibility (double S, double T)
{
  // lability_susceptibility = (1 − T) × (0.5 + 0.5 × S)
  return (1.0 - T) * (0.5 + 0.5 * SensitivityBias (S));
}

// --- Algorithm 23 (Emotional Consolidation Tags) Helpers ---
inline double
ThetaIntensity (double S)
{
  // θ_intensity = lerp(0.6, 0.8, 1 − S)
  return Lerp (0.6, 0.8, 1.0 - SensitivityBias (S));
}

inline double
ThetaArousal (double S)
{
  // θ_arousal = lerp(0.4, 0.2, S)
  return Lerp (0.4, 0.2, SensitivityBias (S));
}

inline double
FlashbulbThreshold (double S)
{
  // flashbulb_threshold = lerp(0.97, 0.65, S)
  return Lerp (0.97, 0.65, SensitivityBias (S));
}

inline int
CascadeRadius (double S)
{
  // cascade_radius = round(lerp(1, 5, S))
  return static_cast<int> (std::round (Lerp (1.0, 5.0, SensitivityBias (S))));
}

inline double
CascadeDecay (double S)
{
  // cascade_decay = lerp(0.7, 0.3, S)
  return Lerp (0.7, 0.3, SensitivityBias (S));
}

inline double
CascadeIntensityFloor (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.10
                    * (1.0 + 0.08 * (f - f0) - 0.18 * (s - s0)
                       + 0.14 * (t - 0.5)),
                0.04, 0.18);
}

inline double
EmotionalHalfLifeBonus (double S, double emotion_intensity)
{
  // emotional_half_life_bonus = exp(lerp(0, ln(3), S)) × (1 +
  // emotion_intensity)
  const double ln3 = std::log (3.0);
  return std::exp (Lerp (0.0, ln3, SensitivityBias (S)))
         * (1.0 + emotion_intensity);
}

inline double
DetailSuppression (double S, double F)
{
  // detail_suppression = S × (1 − F) × 0.5
  return SensitivityBias (S) * (1.0 - FocusBias (F)) * 0.5;
}

inline int
GistComponents (double F)
{
  // gist_components = round(lerp(5, 2, F))
  return static_cast<int> (std::round (Lerp (5.0, 2.0, FocusBias (F))));
}

inline double
FlashbulbThresholdEff (double S, double emotion_intensity, double arousal)
{
  // flashbulb_threshold_eff = flashbulb_threshold × (1 − 0.5 × intensity)
  //                           × (1 − 0.3 × arousal)^{p(S)}
  const double base = FlashbulbThreshold (S);
  const double p = Lerp (1.25, 0.85, SensitivityBias (S));
  return base * (1.0 - 0.5 * emotion_intensity)
         * std::pow (1.0 - 0.3 * arousal, p);
}

inline int
EmotionHistoryWindow (double S, double T)
{
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double base = Lerp (24.0, 96.0, s) * Lerp (0.8, 1.2, t);
  return static_cast<int> (std::round (Clamp (base, 16.0, 160.0)));
}

inline double
FlashbulbPercentile (double S)
{
  const double s = SensitivityBias (S);
  return Lerp (0.95, 0.80, s);
}

inline double
FlashbulbGain (double S)
{
  const double s = SensitivityBias (S);
  return Lerp (1.0, 1.1, s);
}

inline double
ConsolidationCandidateTagWeight (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double raw = Lerp (0.10, 0.25, s) * (1.0 - 0.05 * (f - f0))
                     * (1.0 + 0.04 * (t - 0.5));
  return Clamp (raw, 0.05, 0.35);
}

// --- Algorithm 4b (Mood Integrator) Helpers ---
inline double
AlphaMood (double S)
{
  // α_mood(S) = lerp(0.01, 0.20, S)
  // Higher sensitivity = faster mood reactivity to emotion events
  return Lerp (0.01, 0.20, SensitivityBias (S));
}

inline double
LambdaMood (double delta_seconds, double T)
{
  // λ_mood(Δt, T) = exp(−ln(2) × Δt / max(half_life_mood(T), ε))
  const double half_life = std::max (Lerp (30.0, 600.0, Clamp (T, 0.0, 1.0)),
                                     1e-6);
  const double decay = -std::log (2.0) * delta_seconds / half_life;
  return std::exp (decay);
}

// --- Algorithm 24 (Working Memory) Helpers ---

inline int
WMBaseCapacity (double S, double F)
{
  // Ablation-arm override (wm_capacity_* arms in
  // chat_replay_release_protocol_spec.json); cached once per process. The
  // production default is the capacity-21 operating point from the live-judge
  // sweep, while preserving the original F/S shape around that point.
  static const int kCapacityOverride = [] {
    const char *value = std::getenv ("CORTEXT_WM_CAPACITY_OVERRIDE");
    const int parsed = value != nullptr ? std::atoi (value) : 0;
    return parsed > 0 ? parsed : 0;
  }();
  if (kCapacityOverride > 0)
    {
      return kCapacityOverride;
    }
  // base_capacity = 3 * round-ish Miller window:
  //   3 * (lerp(8, 6, S) + lerp(-1, 1, F))
  // Capacity range [15, 27], with 21 at neutral knobs.
  const double cap = 3.0 * (Lerp (8.0, 6.0, SensitivityBias (S))
                            + Lerp (-1.0, 1.0, FocusBias (F)));
  return static_cast<int> (std::round (cap));
}

inline double
WMMaintenanceCostPerSlot (double S, double F)
{
  // Per-slot maintenance shares a fixed total budget: a FULL working
  // memory always costs lerp(0.05, 0.15, S) x 7 (the neutral capacity),
  // regardless of actual capacity. Capacity changes - including the
  // wm_capacity_* ablation override - therefore do not change gate
  // strictness; the size arms measure window value, not gate side effects.
  constexpr double kNeutralCapacity = 7.0;
  const double per_slot_at_neutral = Lerp (0.05, 0.15, SensitivityBias (S));
  const int capacity = std::max (1, WMBaseCapacity (S, F));
  return per_slot_at_neutral * kNeutralCapacity
         / static_cast<double> (capacity);
}

inline double
WMStrengthBase (double T)
{
  return Lerp (0.30, 0.90, Clamp (T, 0.0, 1.0));
}

inline double
WMStrengthMax (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (10.0
                    * (1.0 - 0.06 * (f - f0) + 0.08 * (s - s0)
                       + 0.18 * (t - 0.5)),
                6.0, 14.0);
}

inline double
WMInitialSlotStrength (double F, double S, double T, double benefit)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double floor = Clamp (0.50 - 0.04 * (f - f0) + 0.04 * (s - s0)
                                  + 0.06 * (t - 0.5),
                              0.35, 0.65);
  return std::min (WMStrengthMax (F, S, T),
                   std::max (0.0, WMStrengthBase (T)
                                       * Lerp (floor, 1.0,
                                               Clamp (benefit, 0.0, 1.0))));
}

inline double
WMCostSaturator (double raw_cost)
{
  const double cost = std::max (0.0, raw_cost);
  return cost / (1.0 + cost);
}

inline double
WMStrengthFloor (double F, double S, double T)
{
  // Passive maintenance should not erase active WM outright; eviction handles
  // replacement. The floor rises with Stability and falls slightly with Focus
  // and Sensitivity so reactive settings can turn over weak slots sooner.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.01 * Lerp (1.05, 0.95, f) * Lerp (1.10, 0.90, s)
                    * Lerp (0.75, 1.25, t),
                0.003, 0.025);
}

inline double
WMChunkingThreshold (double F)
{
  // chunking_threshold = lerp(0.7, 0.9, F)
  return Lerp (0.7, 0.9, FocusBias (F));
}

inline double
WMGateThreshold (double F)
{
  // gate_threshold = lerp(0.1, 0.4, F)
  // At F=0 (wide attention): permissive (0.1)
  // At F=1 (narrow attention): strict (0.4)
  return Lerp (0.1, 0.4, FocusBias (F));
}

inline double
WMTaskRelevanceFallback (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.50 + 0.04 * (f - f0) - 0.03 * (s - s0)
                    + 0.03 * (t - 0.5),
                0.40, 0.60);
}

struct WMBenefitWeights
{
  double window;
  double relevance;
  double novelty;
};

inline WMBenefitWeights
WMBenefitScoringWeights (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double window_raw = Lerp (0.55, 0.70, f) * (1.0 + 0.10 * (t - 0.5));
  const double relevance_raw = Lerp (0.20, 0.35, f) * (1.0 - 0.08 * (t - 0.5));
  const double novelty_raw = Lerp (0.10, 0.30, s) * (1.0 - 0.12 * (t - 0.5));
  const double sum = std::max (1e-12, window_raw + relevance_raw + novelty_raw);
  return { window_raw / sum, relevance_raw / sum, novelty_raw / sum };
}

inline double
WMComplexityScale (double S)
{
  // complexity penalty scale ~ sensitivity
  return Lerp (0.5, 1.5, SensitivityBias (S));
}

inline double
WMRehearsalRate (double S)
{
  // rehearsal_rate = lerp(0.5, 2.0, S)
  // Higher Sensitivity = faster rehearsal boost
  return Lerp (0.5, 2.0, SensitivityBias (S));
}

inline double
WMRehearsalBaseDelta (double F, double S, double T)
{
  // Base rehearsal increment before WMRehearsalRate(S). Focus suppresses
  // broad rehearsal, Sensitivity raises weak-match responsiveness, and
  // Stability preserves practiced slots more strongly.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.10 * Lerp (1.15, 0.85, f) * Lerp (0.92, 1.08, s)
                    * Lerp (0.90, 1.10, t),
                0.04, 0.18);
}

inline double
WMRehearsalThreshold (double F)
{
  // rehearsal_threshold = lerp(0.5, 0.7, F)
  // Below chunking_threshold(F) = lerp(0.7, 0.9, F)
  // Slots in [rehearsal_threshold, chunking_threshold) get rehearsal boost
  return Lerp (0.5, 0.7, FocusBias (F));
}

inline double
WMSlotDedicationStrength (double T)
{
  // slot_dedication_strength = lerp(0.3, 0.9, T)
  // Higher Stability = slots become more dedicated (resistant to eviction)
  return Lerp (0.3, 0.9, T);
}

inline double
WMRecencyTauSeconds (double F, double S, double T)
{
  // Recency half-life for eviction pressure. Lower Focus/Sensitivity keeps a
  // wider, longer-lived conversational context; higher Stability extends it.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (30.0, 90.0, t) * Lerp (1.10, 0.92, f)
                    * Lerp (1.08, 0.90, s),
                15.0, 120.0);
}

inline double
WMCapacityPressureExponent (double F, double S, double T)
{
  // Capacity pressure steepens under focused/stable policy and relaxes under
  // broad, sensitive policy where exploration is allowed to run hotter.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (2.2, 4.2, f) * Lerp (0.90, 1.08, s)
	                    * Lerp (0.95, 1.05, t),
	                1.5, 5.0);
}

inline double
DerivedSourceEdgeWeight (double F, double S, double T)
{
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double f0 = RetrievalFocusBias (0.5);
  const double s0 = RetrievalSensitivityBias (0.5);
  const double t0 = RetrievalStabilityBias (0.5);
  return Clamp (1.0 + 0.03 * (f - f0) - 0.04 * (s - s0)
                    + 0.08 * (t - t0),
                0.80, 1.0);
}

inline double
DerivedSourceFallbackEdgeWeight (double F, double S, double T)
{
  return DerivedSourceEdgeWeight (F, S, T);
}

// --- Algorithm 25 (Metacognitive Monitoring) Helpers ---
inline double
FOKThreshold (double F)
{
  // FOK_threshold = lerp(0.2, 0.5, F)
  return Lerp (0.2, 0.5, FocusBias (F));
}

inline double
TOTFokCutoff (double F)
{
  // TOT FOK cutoff = lerp(0.5, 0.8, F)
  return Lerp (0.5, 0.8, FocusBias (F));
}

inline double
TOTRetrievalCutoff (double F)
{
  // TOT retrieval cutoff = lerp(0.4, 0.2, F)
  return Lerp (0.4, 0.2, FocusBias (F));
}

inline double
ConfidenceDecayRate (double T)
{
  // confidence_decay_rate = lerp(0.01, 0.1, 1 − T)
  return Lerp (0.01, 0.1, 1.0 - T);
}

inline double
UnknownThreshold (double F)
{
  // unknown_threshold = lerp(0.3, 0.1, F)
  return Lerp (0.3, 0.1, FocusBias (F));
}

inline int
StrategySwitchLatencyMs (double S)
{
  // strategy_switch_latency = lerp(500, 100, S) ms
  return static_cast<int> (
      std::round (Lerp (500.0, 100.0, SensitivityBias (S))));
}

inline double
CertaintyRequirement (double T)
{
  // certainty_requirement = lerp(0.6, 0.9, T)
  return Lerp (0.6, 0.9, T);
}

inline double
RetrievalUnknownCautionCertaintyRequirement (double F, double S, double T,
                                             double resurfacing_decay_scale)
{
  // Unknown-caution retrieval should primarily follow the memory knobs. Storage
  // pressure can relax it slightly, but cannot become the certainty setting.
  const double f = RetrievalFocusBias (F);
  const double s = RetrievalSensitivityBias (S);
  const double t = RetrievalStabilityBias (T);
  const double pressure_scale = Lerp (
      0.92, 1.0, Clamp (resurfacing_decay_scale, 0.0, 1.0));
  return Clamp (Lerp (0.52, 0.74, t) * Lerp (0.98, 1.04, f)
                    * Lerp (1.04, 0.96, s) * pressure_scale,
                0.45, 0.82);
}

// --- Section 5.5 (Adaptive Threshold Precision Modulation) ---
inline double
TargetPrecision (double F, double T)
{
  // algorithms.md defines ΔThreshold_precision_t in terms of a
  // target_precision. The target itself is derived (not configurable).
  //
  // We anchor it to the metacognitive certainty requirement (Alg 25) and
  // scale it by focus to reflect that higher focus demands higher precision.
  //
  // target_precision = certainty_requirement(T) * (0.5 + 0.5*F)
  const double f01 = FocusBias (F);
  const double t01 = Clamp (T, 0.0, 1.0);
  const double certainty = CertaintyRequirement (t01);
  return Clamp (certainty * (0.5 + 0.5 * f01), 0.0, 1.0);
}

inline double
MetacognitiveSensitivity (double F, double S)
{
  // metacognitive_sensitivity = F × (1 + 0.5 × S)
  const double f01 = FocusBias (F);
  const double s01 = SensitivityBias (S);
  return f01 * (1.0 + 0.5 * s01);
}

inline double
MetacognitiveConfidenceAlpha (double F, double S, double T)
{
  const double meta_sens = MetacognitiveSensitivity (F, S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp ((0.15 + 0.20 * meta_sens) * Lerp (1.08, 0.92, t),
                0.12, 0.55);
}

struct NeuromodulatorPolicy
{
  double ach_base;
  double ne_base;
  double da_base;
  double retrieval_pressure_depth;
  double ach_novelty_gain;
  double ach_retrieval_pressure_gain;
  double ne_surprisal_gain;
  double ne_arousal_gain;
  double encode_sensitivity_floor;
  double encode_sensitivity_gain;
  double oscillation_rate;
  double encode_oscillation_floor;
  double encode_oscillation_gain;
};

inline NeuromodulatorPolicy
NeuromodulatorPolicyForKnobs (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return {
    Clamp (0.15 + 0.55 * s + 0.25 * (1.0 - t) - 0.15 * f, 0.0, 1.0),
    Clamp (0.10 + 0.60 * s + 0.20 * (1.0 - t), 0.0, 1.0),
    Clamp (0.10 + 0.40 * f + 0.30 * t, 0.0, 1.0),
    Clamp (10.0 * (1.0 - 0.10 * (s - s0) + 0.15 * (t - 0.5)),
           6.0, 14.0),
    Clamp (0.35 * (1.0 - 0.06 * (f - f0) + 0.10 * (s - s0)),
           0.25, 0.45),
    Clamp (0.20 * (1.0 + 0.12 * (f - f0) - 0.08 * (t - 0.5)),
           0.12, 0.28),
    Clamp (0.50 * (1.0 + 0.12 * (s - s0) - 0.08 * (t - 0.5)),
           0.35, 0.65),
    Clamp (0.30 * (1.0 - 0.06 * (f - f0) + 0.10 * (s - s0)),
           0.20, 0.40),
    Clamp (0.70 - 0.06 * (s - s0) + 0.04 * (t - 0.5), 0.58,
           0.80),
    Clamp (0.30 + 0.06 * (s - s0) - 0.04 * (t - 0.5), 0.20,
           0.42),
    Lerp (0.03, 0.12, s) * Lerp (1.2, 0.8, t),
    Clamp (0.60 - 0.03 * (s - s0) + 0.04 * (t - 0.5), 0.50,
           0.70),
    Clamp (0.40 + 0.03 * (s - s0) - 0.04 * (t - 0.5), 0.30,
           0.50)
  };
}

struct SynapticTaggingPolicy
{
  double surprisal_threshold;
  double arousal_threshold;
  int tag_window;
  int tag_decay_seconds;
};

inline SynapticTaggingPolicy
SynapticTaggingPolicyForKnobs (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  return {
    Clamp (Lerp (0.6, 0.4, s) * (1.0 + 0.04 * (f - f0)), 0.34, 0.66),
    Clamp (Lerp (0.7, 0.5, s) * (1.0 + 0.03 * (f - f0)), 0.44, 0.76),
    std::max (
        1, static_cast<int> (std::round (
               Lerp (2.0, 8.0, s) * Lerp (0.92, 1.08, 1.0 - f)))),
    static_cast<int> (
        std::round (Clamp (Lerp (300.0, 3600.0, t)
                               * Lerp (1.05, 0.95, s),
                           180.0, 5400.0)))
  };
}

// --- Algorithm 26 (Serial Position Effects) Helpers ---
inline int
SerialPrimacyWindow (double F)
{
  // primacy_window = round(lerp(5, 2, F))
  return static_cast<int> (std::round (Lerp (5.0, 2.0, FocusBias (F))));
}

inline int
SerialRecencyWindow (double F)
{
  // recency_window = round(lerp(7, 3, F))
  return static_cast<int> (std::round (Lerp (7.0, 3.0, FocusBias (F))));
}

inline double
SerialPrimacyBonus (double S)
{
  // primacy_bonus = lerp(1.2, 2.0, S)
  return Lerp (1.2, 2.0, SensitivityBias (S));
}

inline double
SerialRehearsalCurveDepth (double S)
{
  // rehearsal_curve_depth = lerp(0.2, 0.6, S)
  return Lerp (0.2, 0.6, SensitivityBias (S));
}

inline double
SerialDistinctivenessThreshold (double F)
{
  // distinctiveness_threshold = lerp(0.6, 0.8, F)
  return Lerp (0.6, 0.8, FocusBias (F));
}

inline double
SerialVonRestorffMultiplier (double S)
{
  // von_restorff_multiplier = lerp(1.5, 3.0, S)
  return Lerp (1.5, 3.0, SensitivityBias (S));
}

inline double
SerialMiddleSuppression (double S, double F)
{
  // middle_suppression = lerp(0.8, 0.5, S) × (1 − F)
  return Lerp (0.8, 0.5, SensitivityBias (S)) * (1.0 - FocusBias (F));
}

inline double
SerialMiddleMultiplierFloor (double F, double S)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.10
                    * (1.0 + 0.08 * (f - f0) - 0.12 * (s - s0)),
                0.05, 0.16);
}

// Zone determination helper for observability (Algorithm 26)
enum class SerialZone
{
  Primacy,
  Recency,
  Distinctive,
  Middle
};

inline SerialZone
SerialDetermineZone (double F, int position, int N, double rarity)
{
  const int primacy = SerialPrimacyWindow (F);
  const int recency = SerialRecencyWindow (F);
  const double thr = SerialDistinctivenessThreshold (F);
  if (position <= primacy)
    {
      return SerialZone::Primacy;
    }
  if (position >= N - recency + 1)
    {
      return SerialZone::Recency;
    }
  if (rarity > thr)
    {
      return SerialZone::Distinctive;
    }
  return SerialZone::Middle;
}

// --- Implicit Feedback (Memory Usage Detection) Helpers ---

inline double
MemoryUsageThreshold (double F)
{
  // Higher focus = stricter threshold for what counts as "used"
  // Range [0.5, 0.8] - at F=0, similarity of 0.5 counts as used;
  // at F=1, requires 0.8 similarity
  return Lerp (0.5, 0.8, FocusBias (F));
}

inline double
MemoryUsageCacheDuration (double T)
{
  // Higher stability = longer memory of what was retrieved
  // Range [30, 300] seconds (30s to 5 minutes)
  return Lerp (30.0, 300.0, Clamp (T, 0.0, 1.0));
}

inline double
ReinforcementFallbackContextualSupport (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (0.50
                    * (1.0 + 0.06 * (f - f0) - 0.08 * (s - s0)
                       + 0.06 * (t - 0.5)),
                0.35, 0.65);
}

inline double
FocusFeedbackWeightGain (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.10 * Lerp (0.85, 1.20, f) * Lerp (0.90, 1.12, s)
                    * Lerp (1.08, 0.92, t),
                0.04, 0.18);
}

inline double
FocusFeedbackWidthGain (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.05 * Lerp (0.90, 1.15, f) * Lerp (0.92, 1.08, s)
                    * Lerp (1.06, 0.94, t),
                0.02, 0.10);
}

inline double
SensitivityFeedbackNoveltyGain (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.10 * Lerp (1.12, 0.90, f) * Lerp (0.85, 1.25, s)
                    * Lerp (1.08, 0.90, t),
                0.04, 0.20);
}

inline double
StabilityFeedbackUsageGain (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (0.05 * Lerp (0.95, 1.05, f) * Lerp (1.12, 0.88, s)
                    * Lerp (0.75, 1.25, t),
                0.02, 0.10);
}

// --- Knowledge Graph Enhancement (Section 9.5) ---

inline double
CausalDriftThreshold (double T)
{
  // causal_drift_threshold(T) = lerp(0.15, 0.35, T)
  // Higher stability = higher drift required for causal edge
  return Lerp (0.15, 0.35, Clamp (T, 0.0, 1.0));
}

inline double
ReinforcementDecay (double T)
{
  // reinforcement_decay(T) = lerp(0.9, 0.99, T)
  // Higher stability = slower decay of reinforcement edges
  return Lerp (0.9, 0.99, Clamp (T, 0.0, 1.0));
}

inline double
ReinforcementCoRetrievalStep (double F, double S, double T)
{
  // Co-retrieval reinforcement is a small online edge update. Sensitivity
  // controls learning speed, Focus suppresses noisy broad packets, and
  // Stability lets repeated durable co-activation accumulate.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.015, 0.055, s) * Lerp (1.20, 0.70, f)
                    * Lerp (0.75, 1.15, t),
                0.005, 0.08);
}

inline double
ReinforcementUnselectedScale (double F, double S, double T)
{
  // Pairs that merely co-occurred in a retrieved packet should learn much more
  // slowly than pairs anchored by an actually used memory.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.10, 0.35, s) * Lerp (0.85, 0.65, f)
                    * Lerp (0.80, 1.05, t),
                0.05, 0.40);
}

inline double
ReinforcementFanoutDamping (double F, double S, double T, int candidate_count)
{
  if (candidate_count <= 2)
    {
      return 1.0;
    }
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double reference_fanout = Lerp (8.0, 3.0, f) * Lerp (0.90, 1.10, s)
                                  * Lerp (0.90, 1.10, t);
  return Clamp (
      std::sqrt (std::max (reference_fanout, 1.0)
                 / static_cast<double> (std::max (candidate_count, 1))),
      0.25, 1.0);
}

inline double
ReinforcementPruneThreshold (double F, double S, double T)
{
  // Prune only reinforcement edges below a knob-derived evidence floor. The
  // floor is expressed as "how many anchored co-retrievals would it take to
  // survive": higher Focus requires cleaner evidence, higher Sensitivity
  // learns faster and can prune weak edges more aggressively, and higher
  // Stability preserves weaker long-term associations.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double anchored_step = ReinforcementCoRetrievalStep (F, S, T);
  const double required_retrievals = Lerp (2.0, 6.0, f)
                                     * Lerp (1.25, 0.85, s)
                                     * Lerp (1.15, 0.65, t);
  return Clamp (anchored_step * required_retrievals, 0.03, 0.18);
}

inline double
GraphBuildSameEventBoundaryThreshold (double F, double S, double T)
{
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (BoundaryThreshold (F, S) * Lerp (0.62, 0.44, s)
                    * Lerp (0.95, 1.10, t),
                0.18, 0.45);
}

inline double
GraphBuildSequentialTauSeconds (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  return Clamp (Lerp (10.0, 60.0, t)
                    * (1.0 - 0.08 * (f - f0) + 0.06 * (s - s0)),
                5.0, 90.0);
}

inline double
GraphBuildSequentialWeight (double F, double S, double T, double gap_s,
                            double boundary_score)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double tau = GraphBuildSequentialTauSeconds (F, S, T);
  const double temporal = std::exp (-std::max (0.0, gap_s) / std::max (tau, 1e-6));
  const double boundary_power = Clamp (1.0 + 0.08 * (f - f0)
                                           - 0.08 * (s - s0)
                                           + 0.10 * (t - 0.5),
                                      0.75, 1.30);
  const double boundary = std::pow (
      1.0 - Clamp (boundary_score, 0.0, 1.0), boundary_power);
  return Clamp (temporal * boundary, 0.0, 1.0);
}

inline double
CausalDriftWeightScale (double T)
{
  return Lerp (1.6, 2.4, Clamp (T, 0.0, 1.0));
}

inline double
LabelCooccurrenceEdgeWeight (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (Lerp (0.25, 0.50, s) * Lerp (1.10, 0.90, f)
                    * Lerp (0.90, 1.20, t),
                0.10, 0.65);
}

inline int
MinEpisodesForConcept (double T)
{
  // min_episodes_for_concept(T) = round(lerp(2, 5, T))
  // Higher stability = more episodes required for concept detection
  return static_cast<int> (std::round (Lerp (2.0, 5.0, Clamp (T, 0.0, 1.0))));
}

inline double
CoOccurrenceThreshold (double F)
{
  // Same as MergeThreshold - co-occurrence uses same similarity threshold
  // co_occurrence_threshold(F) = lerp(0.85, 0.95, F)
  return MergeThreshold (F);
}

inline double
SimilarToThreshold (double F)
{
  // similar_to threshold: high cosine similarity (soft equivalence)
  return Lerp (0.90, 0.98, FocusBias (F));
}

inline double
MinEdgeWeight (double F)
{
  // min_edge_weight(F) for graph traversal
  return Lerp (0.08, 0.40, FocusBias (F));
}

inline double
ImpliesDriftThreshold (double T)
{
  // implies threshold (directional drift correlation)
  return Lerp (0.10, 0.25, Clamp (T, 0.0, 1.0));
}

inline double
ContradictionThreshold (double F, double S, double T)
{
  // Contradiction detection remains centered at the historical -0.5 default.
  // Higher Sensitivity admits weaker contradiction cues; higher Focus and
  // Stability require clearer opposition before writing contradiction edges.
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  return Clamp (-0.50 + 0.08 * (s - SensitivityBias (0.5))
                    - 0.05 * (f - FocusBias (0.5))
                    - 0.04 * (t - 0.5),
                -0.70, -0.30);
}

inline double
ContradictionThreshold ()
{
  return ContradictionThreshold (0.5, 0.5, 0.5);
}

// --- Section 8: Interrupt Gate Parameters ---

inline double
TauNovelty (double F, double S, double T)
{
  // tau_novelty = lerp(0.12, 0.32, F) * (1 - 0.12S) * (1 + 0.25T)
  // Reference: algorithms.md Section 8.1
  return Lerp (0.12, 0.32, FocusBias (F))
       * (1.0 - 0.12 * SensitivityBias (S))
       * (1.0 + 0.25 * Clamp (T, 0.0, 1.0));
}

inline double
RetrievalThreshold (double F)
{
  // retrieval_thresh(F) = lerp(0.12, 0.45, F)
  // Reference: algorithms.md Section 8.1
  return Lerp (0.12, 0.45, FocusBias (F));
}

inline double
RetrievalThresholdInterrupt (double F, double S)
{
  // retrieval_thresh_interrupt(F,S) = retrieval_thresh(F) * (1 - 0.12 * S̃)
  // Sensitivity relaxes the interrupt gate to improve recall while preserving
  // the Focus-defined baseline.
  const double base = RetrievalThreshold (F);
  const double relax = 1.0 - 0.12 * SensitivityBias (S);
  return base * relax;
}

inline double
InterruptBoundaryMultiplier (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  const double base = Lerp (1.3, 2.0, f) * Lerp (1.1, 0.9, s)
                      * Lerp (1.4, 0.6, t);
  return base * (1.0 - 0.20 * SensitivityBias (S));
}

inline double
InterruptAffectRelaxCoeff (double S)
{
  // affect_relax_coeff(S) = lerp(0.08, 0.28, S̃_affect)
  // Keep affect as a modulation term for interrupt gating rather than a
  // dominant relaxer on mixed-content streams.
  return Lerp (0.08, 0.28, AffectSensitivityBias (S));
}

inline double
RetrievalEmotionWeight (double S)
{
  // w_emotion = lerp(0.0, 0.12, S̃_affect)
  return Lerp (0.0, 0.12, AffectSensitivityBias (S));
}

inline double
AffectGain (double S)
{
  // affect_gain(S) = lerp(1.0, 2.4, S̃_affect)
  return Lerp (1.0, 2.4, AffectSensitivityBias (S));
}

struct AffectDrivePolicy
{
  double arousal_weight;
  double emotion_weight;
  double salience_weight;
};

inline AffectDrivePolicy
AffectDriveWeights (double S)
{
  const double s_affect = AffectSensitivityBias (S);
  const double w_arousal_raw = Lerp (0.30, 0.55, s_affect);
  const double w_emotion_raw = Lerp (0.30, 0.55, s_affect);
  const double w_salience_raw = Lerp (0.10, 0.20, s_affect);
  const double sum
      = std::max (1e-12, w_arousal_raw + w_emotion_raw + w_salience_raw);
  return { w_arousal_raw / sum, w_emotion_raw / sum,
           w_salience_raw / sum };
}

struct InterruptGateScoringWeights
{
  double coverage_raw;
  double relevance_raw;
  double redundancy_raw;
  double coherence_raw;
  double surprise_raw;
  double coverage;
  double relevance;
  double redundancy;
  double coherence;
  double surprise;
};

inline InterruptGateScoringWeights
InterruptGateCandidateScoringWeights (double F, double S, double T)
{
  (void)T;
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double w_cov_raw = Lerp (0.40, 0.60, f);
  const double w_rel_raw = Lerp (0.35, 0.25, f);
  const double w_red_raw = Lerp (0.15, 0.25, s);
  const double w_coh_raw = Lerp (0.15, 0.25, s);
  const double w_surp_raw = Lerp (0.20, 0.50, s);
  const double total = std::max (
      1e-12, w_cov_raw + w_rel_raw + w_red_raw + w_coh_raw + w_surp_raw);
  return { w_cov_raw,
           w_rel_raw,
           w_red_raw,
           w_coh_raw,
           w_surp_raw,
           w_cov_raw / total,
           w_rel_raw / total,
           w_red_raw / total,
           w_coh_raw / total,
           w_surp_raw / total };
}

inline double
InterruptMuThreshold (double F, double S, double T)
{
  return Lerp (0.08, 0.18, Clamp (F, 0.0, 1.0))
       * (1.0 - 0.4 * Clamp (S, 0.0, 1.0))
       * (1.0 + 0.4 * Clamp (T, 0.0, 1.0));
}

inline double
InterruptRefractoryTau (double S, double T)
{
  return Lerp (24.0, 96.0, Clamp (T, 0.0, 1.0))
       * Lerp (1.4, 1.0, Clamp (S, 0.0, 1.0));
}

inline double
InterruptRefractoryGain (double F, double T)
{
  return Lerp (0.20, 0.05, Clamp (T, 0.0, 1.0))
       * Lerp (0.8, 1.2, Clamp (F, 0.0, 1.0));
}

inline double
InterruptMaturityScale (double maturity, double T)
{
  return 1.0 + (1.0 - Clamp (maturity, 0.0, 1.0))
                   * Lerp (0.4, 1.0, Clamp (T, 0.0, 1.0));
}

inline int
InterruptCandidateCount (double F)
{
  // K = round(lerp(10, 6, F))
  // Reference: algorithms.md Section 8.3
  return static_cast<int> (std::round (Lerp (10.0, 6.0, FocusBias (F))));
}

inline double
DupThresh (double F, double T)
{
  // dup_thresh = lerp(0.985, 0.95, F) × (0.98 + 0.02T)
  return Lerp (0.985, 0.95, FocusBias (F))
       * (0.98 + 0.02 * Clamp (T, 0.0, 1.0));
}

// Section 4.4: Write Pacing and Memory Accumulation ---

// Section 4.4.2 - Drift EWMA alpha
inline double
AlphaEtaAcc (double T)
{
  // α = lerp(0.3, 0.1, T)
  // Higher stability = slower EWMA update
  return Lerp (0.3, 0.1, Clamp (T, 0.0, 1.0));
}

// Section 4.4.2 - Drift noise floor
inline double
DriftNoiseFloor (double T)
{
  // ε_noise(T) = lerp(0.01, 0.05, 1 − T)
  return Lerp (0.01, 0.05, 1.0 - Clamp (T, 0.0, 1.0));
}

// Section 4.4.3 - Cold-start guard for eta
inline double
EtaColdStart (double T)
{
  // ε0(T) = lerp(0.005, 0.02, 1 − T)
  return Lerp (0.005, 0.02, 1.0 - Clamp (T, 0.0, 1.0));
}

// Section 4.4.3 - Boundary detection weight on drift
struct BoundaryEvidenceWeights
{
  double surprisal;
  double drift;
  double coherence;
  double topic;
  double gap;
};

inline BoundaryEvidenceWeights
BoundaryEvidenceScoringWeights (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  const double gap = Clamp (0.01 * Lerp (1.10, 0.90, f)
                                * Lerp (0.90, 1.10, s)
                                * Lerp (0.90, 1.10, t),
                            0.004, 0.020);
  return { 0.34 + 0.14 * s,
           0.18 + 0.05 * (1.0 - t),
           0.36 + 0.16 * f,
           0.30 + 0.22 * s,
           gap };
}

inline double
BoundaryVarianceFloor (double S, double T)
{
  const double s = SensitivityBias (S);
  const double s0 = SensitivityBias (0.5);
  const double scale = Clamp (1.0 - 0.20 * (s - s0)
                                  + 0.20 * (Clamp (T, 0.0, 1.0) - 0.5),
                              0.50, 2.0);
  return Clamp (0.0025 * scale, 0.001, 0.006);
}

inline int
BoundaryWarmupSignals (double S, double T)
{
  const double s = SensitivityBias (S);
  const double s0 = SensitivityBias (0.5);
  const double scale = Clamp (1.0 - 0.20 * (s - s0)
                                  + 0.30 * (Clamp (T, 0.0, 1.0) - 0.5),
                              0.60, 1.80);
  return std::max (1, static_cast<int> (std::round (3.0 * scale)));
}

struct BoundaryInactivityPolicy
{
  double gap_score_scale;
  double gap_force_center;
  double gap_force_k;
  double support_relax_rate;
  double inactivity_weight;
  double inactivity_exp_rate;
};

inline BoundaryInactivityPolicy
BoundaryInactivityPolicyForKnobs (double S, double T)
{
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  return { Lerp (0.9, 1.8, s) * Lerp (1.1, 0.9, t),
           Lerp (0.9, 0.4, s) * Lerp (1.1, 0.9, t),
           Lerp (2.0, 5.0, s) * Lerp (1.1, 0.9, t),
           Lerp (0.6, 1.6, s) * Lerp (1.15, 0.85, t),
           Lerp (0.05, 0.25, s) * Lerp (1.1, 0.9, t),
           Lerp (0.6, 1.6, s) * Lerp (1.1, 0.9, t) };
}

inline double
BoundaryHazardPrior (double F, double S, double T)
{
  const double f = Clamp (F, 0.0, 1.0);
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  const double h_base = Lerp (0.03, 0.18, s) * Lerp (1.2, 0.8, t);
  return Clamp (h_base * (0.8 + 0.2 * (1.0 - f)), 0.0, 0.5);
}

struct BoundaryLocalNormPolicy
{
  double alpha;
  double variance_floor;
  double gain;
  int warmup_signals;
};

inline BoundaryLocalNormPolicy
BoundaryLocalNormPolicyForKnobs (double S, double T)
{
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  return { Lerp (0.35, 0.15, t),
           BoundaryVarianceFloor (S, T),
           Clamp (Lerp (1.6, 3.0, s) * Lerp (1.05, 0.95, t), 1.0,
                  3.2),
           BoundaryWarmupSignals (S, T) };
}

struct BoundarySupportPolicy
{
  double support;
  double support_gate;
  double gap_gate;
  double support_boost;
};

inline BoundarySupportPolicy
BoundarySupportPolicyForKnobs (double F, double S, double T,
                               double coherence_norm, double topic_norm)
{
  const double f = FocusBias (F);
  const double s = SensitivityBias (S);
  const double t = Clamp (T, 0.0, 1.0);
  const double f0 = FocusBias (0.5);
  const double s0 = SensitivityBias (0.5);
  const double coherence_raw = Clamp (0.50 + 0.10 * (f - f0)
                                          - 0.06 * (s - s0),
                                      0.35, 0.65);
  const double topic_raw = Clamp (1.0 - coherence_raw, 0.35, 0.65);
  const double support = Clamp (
      coherence_raw * Clamp (coherence_norm, 0.0, 1.0)
          + topic_raw * Clamp (topic_norm, 0.0, 1.0),
      0.0, 1.0);
  const double gate_scale = Clamp (1.0 + 0.06 * (s - s0)
                                       - 0.04 * (t - 0.5),
                                   0.85, 1.15);
  return { support,
           Clamp (Lerp (0.45, 1.0, support) * gate_scale, 0.30, 1.0),
           Clamp (Lerp (0.10, 0.50, support) * gate_scale, 0.05, 0.70),
           Clamp (Lerp (1.0, 1.25, support) * gate_scale, 0.85, 1.40) };
}

struct BoundaryChangePointPolicy
{
  double z_center;
  double z_center_eff;
  double k;
};

inline BoundaryChangePointPolicy
BoundaryChangePointPolicyForKnobs (double S, double T, double support)
{
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  const double support01 = Clamp (support, 0.0, 1.0);
  const double z_center
      = Clamp (Lerp (0.44, 0.30, s) * Lerp (1.05, 0.95, t), 0.26,
               0.58);
  const double z_center_eff
      = Clamp (z_center * Lerp (1.0, 0.75, support01), 0.24, 0.58);
  const double k_cp = Lerp (8.0, 22.0, s) * Lerp (1.1, 0.9, t);
  return { z_center, z_center_eff, k_cp * Lerp (0.9, 1.35, support01) };
}

struct BoundaryPressurePolicy
{
  double capacity;
  double pressure;
};

inline BoundaryPressurePolicy
BoundaryPressurePolicyForKnobs (double S, double T, double drift_acc)
{
  const double s = Clamp (S, 0.0, 1.0);
  const double t = Clamp (T, 0.0, 1.0);
  const double base_capacity = Lerp (0.8, 2.0, SensitivityBias (S));
  const double capacity_scale = (1.0 + t) * (1.0 + t);
  return { base_capacity * capacity_scale, drift_acc * (1.0 + s) };
}

inline double
AccumulatorDriftGain (double S, double T)
{
  const double s = SensitivityBias (S);
  const double s0 = SensitivityBias (0.5);
  const double scale = Clamp (1.0 + 0.12 * (s - s0)
                                  - 0.12 * (Clamp (T, 0.0, 1.0) - 0.5),
                              0.75, 1.25);
  return Clamp (0.5 * scale, 0.35, 0.65);
}

inline double
InterruptAbortAcceptanceMargin (double F, double S, double T)
{
  const double f = FocusBias (F);
  const double f0 = FocusBias (0.5);
  return Lerp (0.01, 0.04, Clamp (S, 0.0, 1.0))
         * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0))
         * Clamp (1.0 + 0.08 * (f - f0), 0.94, 1.08);
}

inline double
AccumulatorTemporalContextAlpha (double S, double T)
{
  const double s = SensitivityBias (S);
  const double s0 = SensitivityBias (0.5);
  return Clamp (Lerp (0.06, 0.01, Clamp (T, 0.0, 1.0))
                    * Clamp (1.0 + 0.10 * (s - s0), 0.90, 1.10),
                0.005, 0.08);
}

inline double
BoundaryWeightDrift (double T)
{
  // w_drift = lerp(0.6, 0.4, T)
  // Higher stability = less weight on drift
  return Lerp (0.6, 0.4, Clamp (T, 0.0, 1.0));
}

// Section 4.4.3 - Boundary detection weight on gap (soft influence)
inline double
BoundaryWeightGap (double T)
{
  // w_gap = lerp(0.05, 0.20, T)
  // Higher stability = slightly more weight on gaps (still low)
  return Lerp (0.05, 0.20, Clamp (T, 0.0, 1.0));
}

// Section 4.4.3 - Boundary threshold
inline double
BoundaryThreshold (double F, double S)
{
  // b_thresh(F, S) = lerp(0.48, 0.66, F) × lerp(1.1, 0.9, S)
  // Higher focus = stricter boundary; higher sensitivity = looser boundary
  return Lerp (0.48, 0.66, FocusBias (F))
       * Lerp (1.1, 0.9, SensitivityBias (S));
}

// Section 4.4.3 - Fallback boundary floor (timeout/capacity/pressure)
inline double
BoundaryFallbackFloor (double F, double S, double T)
{
  const double base = Lerp (0.05, 0.15, SensitivityBias (S));
  const double f_scale = Lerp (1.2, 0.8, FocusBias (F));
  const double t_scale = Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
  return Clamp (base * f_scale * t_scale, 0.02, 0.25);
}

// Section 4.4.3 - Target boundary rate (calibration)
inline double
BoundaryTargetRate (double F, double S, double T)
{
  const double base = Lerp (0.12, 0.35, SensitivityBias (S));
  const double f_scale = Lerp (1.10, 0.70, FocusBias (F));
  const double t_scale = Lerp (1.00, 0.70, Clamp (T, 0.0, 1.0));
  return Clamp (base * f_scale * t_scale, 0.05, 0.45);
}

// Section 4.4.3 - Boundary rate calibration gain
inline double
BoundaryRateGain (double S, double T)
{
  const double s_scale = Lerp (0.4, 1.0, SensitivityBias (S));
  const double t_scale = Lerp (1.1, 0.8, Clamp (T, 0.0, 1.0));
  return Clamp (s_scale * t_scale, 0.2, 1.2);
}

// Section 4.4.3 - Boundary rate EMA alpha
inline double
BoundaryRateAlpha (double S, double T)
{
  const double s_scale = Lerp (0.06, 0.16, SensitivityBias (S));
  const double t_scale = Lerp (1.1, 0.8, Clamp (T, 0.0, 1.0));
  return Clamp (s_scale * t_scale, 0.04, 0.2);
}

// Section 4.4.3 - Adaptive memory time (seconds) from recent gap cadence
inline double
AdaptiveMaxMemoryTime (double T, double gap_ref_s)
{
  const double t = Clamp (T, 0.0, 1.0);
  const double scale = Lerp (2.5, 5.5, t);
  const double min_time = Lerp (6.0, 20.0, t);
  const double max_time = Lerp (60.0, 180.0, t);
  const double cadence = std::max (gap_ref_s, 0.001);
  return Clamp (scale * cadence, min_time, max_time);
}

// Section 4.4.3 - Maximum cumulative drift before flush
inline double
MaxMemoryDrift (double S)
{
  // max_mem_drift(S) = lerp(0.8, 2.0, S)
  // Higher sensitivity = more drift allowed before forced flush
  return Lerp (0.8, 2.0, SensitivityBias (S));
}

// Section 4.4.3 - Max signals per memory (fallback boundary)
inline int
MaxSignalsPerMemory (double F, double S, double T)
{
  const double base = Lerp (4.0, 12.0, Clamp (T, 0.0, 1.0));
  const double f_scale = Lerp (1.00, 0.35, FocusBias (F));
  const double s_scale = Lerp (1.00, 0.55, SensitivityBias (S));
  const int capped = static_cast<int> (std::round (base * f_scale * s_scale));
  const int floor = static_cast<int> (std::round (Lerp (2.0, 3.0, Clamp (T, 0.0, 1.0))));
  return std::max (floor, capped);
}

// Section 4.4.3 - Adaptive gap scale (seconds, multiplier applied to dt_ema)
inline double
GapScale (double T)
{
  // gap_scale(T) = lerp(3.0, 8.0, T) seconds
  // Higher stability = longer pause expected before affecting boundary score
  return Lerp (3.0, 8.0, Clamp (T, 0.0, 1.0));
}

// Section 4.4.4 - Spike bypass margin above θ_dynamic
inline double
SpikeMargin (double S, double T)
{
  // spike_margin(S,T) = lerp(0.2, 0.5, T) × lerp(1.2, 0.8, S)
  return Lerp (0.2, 0.5, Clamp (T, 0.0, 1.0))
       * Lerp (1.2, 0.8, SensitivityBias (S));
}

// Section 4.4.5 - Window score peak vs average weight
inline double
WindowScoreAlpha (double F)
{
  // α(F) = lerp(0.3, 0.7, F)
  // Higher focus = more weight on peak score
  return Lerp (0.3, 0.7, FocusBias (F));
}

// Section 4.4.5 - Coverage bonus weight
inline double
WindowScoreCoverageBeta (double S)
{
  // β(S) = lerp(0.05, 0.15, S)
  // Higher sensitivity = more bonus for complete memorys
  return Lerp (0.05, 0.15, SensitivityBias (S));
}

// Section 4.4.5 - Write refractory time constant (seconds)
inline double
WriteRefractoryTau (double T)
{
  // τ_write_refrac(T) = lerp(5, 30, T) seconds
  // Higher stability = longer refractory period
  return Lerp (5.0, 30.0, Clamp (T, 0.0, 1.0));
}

// Section 4.4.5 - Write refractory gain
inline double
WriteRefractoryK (double T)
{
  // k_write_refrac = lerp(0.3, 0.1, T)
  // Higher stability = weaker refractory suppression
  return Lerp (0.3, 0.1, Clamp (T, 0.0, 1.0));
}

// Section 4.4.5 - Representative embedding blend factor
inline double
RepresentativeBlendRho (double F)
{
  // ρ(F) = lerp(0.2, 0.6, F)
  // Higher focus = more weight on memory mean vs peak
  return Lerp (0.2, 0.6, FocusBias (F));
}

} // namespace cortext::core
