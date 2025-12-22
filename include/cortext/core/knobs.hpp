#pragma once

#include "cortext/core/algorithms.hpp"
#include <cmath>

namespace cortext::core
{

// This file implements the knob-derived functions from `algorithms.md` Section
// 0.

// --- 0.2 Knob-Derived Functions ---

inline double
BaseRatePrior (double S)
{
  // base_rate_prior(S) = lerp(r_min, r_max, S), r_min=0.2, r_max=5.0
  return Lerp (0.2, 5.0, S);
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
  // max_results(F) = round(lerp(R_max, R_min, F)), R_min=4, R_max=64
  return static_cast<int> (std::round (Lerp (64.0, 4.0, Clamp (F, 0.0, 1.0))));
}

inline int
AdjacentWindow (double F)
{
  // adjacent_window(F) = round(lerp(adjacent_max, adjacent_min, F)),
  // adjacent_min=1, adjacent_max=8
  return static_cast<int> (std::round (Lerp (8.0, 1.0, Clamp (F, 0.0, 1.0))));
}

inline int
MaxWaitTokens (double F)
{
  // max_wait_tokens(F) = round(lerp(wait_max, wait_min, F)),
  // wait_min=16, wait_max=128
  return static_cast<int> (std::round (Lerp (128.0, 16.0, Clamp (F, 0.0, 1.0))));
}

inline int
CheckIntervalTokens (double S)
{
  // check_interval(S) = round(lerp(check_max_tokens, check_min_tokens, S)),
  // check_min_tokens=8, check_max_tokens=64
  return static_cast<int> (std::round (Lerp (64.0, 8.0, Clamp (S, 0.0, 1.0))));
}

// --- Section 10.4: Streaming Pacing Parameters ---

inline double
StreamingPacingThreshold (double S)
{
  // pacing_thresh(S) = lerp(0.5, 0.1, S)
  // Higher sensitivity = lower threshold = more frequent pacing checks
  return Lerp (0.5, 0.1, Clamp (S, 0.0, 1.0));
}

inline double
MaxWaitDrift (double F)
{
  // max_wait_drift(F) = lerp(2.0, 0.5, F)
  // Higher focus = lower max drift = more aggressive forced checks
  return Lerp (2.0, 0.5, Clamp (F, 0.0, 1.0));
}

inline int
GraphDepth (double T)
{
  // graph_depth(T): small traversal depth, in [1,2].
  return static_cast<int> (std::round (Lerp (2.0, 1.0, Clamp (T, 0.0, 1.0))));
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
TPrior (double /*F*/, double S, double T)
{
  // T_prior(F,S,T) = lerp(0.10, 0.30, T) × (1 − 0.3×S)
  return Lerp (0.10, 0.30, T) * (1.0 - 0.3 * S);
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
  return Lerp (0.06, 0.14, Clamp (S, 0.0, 1.0))
         * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

inline double
CapHomeo (double /*F*/, double S, double T, double hysteresis)
{
  return Lerp (0.35, 0.15, Clamp (T, 0.0, 1.0))
         * Lerp (0.8, 1.2, Clamp (S, 0.0, 1.0))
         * hysteresis;
}

// Sensitivity-based threshold adjustment
inline double
KappaSens (double /*F*/, double S, double T)
{
  return Lerp (0.04, 0.12, Clamp (S, 0.0, 1.0))
         * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

inline double
CapSens (double /*F*/, double S, double T, double hysteresis)
{
  return Lerp (0.30, 0.10, Clamp (T, 0.0, 1.0))
         * Lerp (0.9, 1.1, Clamp (S, 0.0, 1.0))
         * hysteresis;
}

inline double
SigmaRef (double S, double T)
{
  return Lerp (0.08, 0.14, Clamp (S, 0.0, 1.0))
         * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

// Precision-based threshold adjustment
inline double
KappaPrec (double F, double /*S*/, double T)
{
  return Lerp (0.04, 0.10, Clamp (F, 0.0, 1.0))
         * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

inline double
CapPrec (double F, double /*S*/, double T, double hysteresis)
{
  return Lerp (0.25, 0.08, Clamp (T, 0.0, 1.0))
         * Lerp (0.9, 1.1, Clamp (F, 0.0, 1.0))
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
  return kAlphaMinF + F * kAlphaSpanF * u_t;
}

inline double
AlphaS (double S, double u_t)
{
  const double kAlphaMinS = 0.05;
  const double kAlphaSpanS = 0.35;
  return kAlphaMinS + S * kAlphaSpanS * u_t;
}

// Uncertainty variance normalization ceiling
inline double
VarScoreMax (double S)
{
  return Lerp (0.15, 0.35, Clamp (S, 0.0, 1.0));
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
  return Lerp (0.25, 0.05, Clamp (S, 0.0, 1.0));
}

// Surprise sigmoid gain
inline double
SurpriseGain (double S, double T)
{
  return Lerp (6.0, 14.0, Clamp (S, 0.0, 1.0))
       * Lerp (1.1, 0.9, Clamp (T, 0.0, 1.0));
}

// RLS initialization scale
inline double
BlenderPInit (double T)
{
  return Lerp (500.0, 2000.0, 1.0 - Clamp (T, 0.0, 1.0));
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
         * (1.0 - 0.5 * S);
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

// Minimum cluster size for extraction — Algorithm 29c
inline int
MinClusterSizeForExtraction (double F)
{
  // min_cluster_size_for_extraction = round(lerp(3, 10, F))
  return static_cast<int> (std::round (Lerp (3.0, 10.0, F)));
}

// --- Consolidation Clustering (Section 7.3-7.4) ---

// Merge threshold for clustering — Section 7.5.1
inline double
MergeThreshold (double F)
{
  // merge_threshold(F) = lerp(0.85, 0.95, F)
  // Higher focus = stricter merging (higher similarity required)
  // Also used for co-occurrence edges in Section 7.5.1
  return Lerp (0.85, 0.95, Clamp (F, 0.0, 1.0));
}

// Minimum cluster size — Section 7.4
inline int
MinClusterSize (double F)
{
  // min_cluster_size(F) = round(lerp(3, 10, F))
  // Higher focus = larger minimum clusters required
  return static_cast<int> (std::round (Lerp (3.0, 10.0, Clamp (F, 0.0, 1.0))));
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

// Periphery cutoff — Algorithm 0.2
inline double
PeripheryCutoff (double T)
{
  return Lerp (0.05, 0.25, T);
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

inline double
LabilitySusceptibility (double S, double T)
{
  // lability_susceptibility = (1 − T) × (0.5 + 0.5 × S)
  return (1.0 - T) * (0.5 + 0.5 * S);
}

// --- Algorithm 23 (Emotional Consolidation Tags) Helpers ---
inline double
ThetaIntensity (double S)
{
  // θ_intensity = lerp(0.6, 0.8, 1 − S)
  return Lerp (0.6, 0.8, 1.0 - S);
}

inline double
ThetaArousal (double S)
{
  // θ_arousal = lerp(0.4, 0.2, S)
  return Lerp (0.4, 0.2, S);
}

inline double
FlashbulbThreshold (double S)
{
  // flashbulb_threshold = lerp(0.9, 0.4, S)
  return Lerp (0.9, 0.4, S);
}

inline int
CascadeRadius (double S)
{
  // cascade_radius = round(lerp(1, 5, S))
  return static_cast<int> (std::round (Lerp (1.0, 5.0, S)));
}

inline double
CascadeDecay (double S)
{
  // cascade_decay = lerp(0.7, 0.3, S)
  return Lerp (0.7, 0.3, S);
}

inline double
EmotionalHalfLifeBonus (double S, double emotion_intensity)
{
  // emotional_half_life_bonus = exp(lerp(0, ln(3), S)) × (1 +
  // emotion_intensity)
  const double ln3 = std::log (3.0);
  return std::exp (Lerp (0.0, ln3, S)) * (1.0 + emotion_intensity);
}

inline double
DetailSuppression (double S, double F)
{
  // detail_suppression = S × (1 − F) × 0.5
  return S * (1.0 - F) * 0.5;
}

inline int
GistComponents (double F)
{
  // gist_components = round(lerp(5, 2, F))
  return static_cast<int> (std::round (Lerp (5.0, 2.0, F)));
}

inline double
FlashbulbThresholdEff (double S, double emotion_intensity, double arousal)
{
  // flashbulb_threshold_eff = flashbulb_threshold × (1 − 0.5 × intensity) × (1
  // − 0.3 × arousal)
  const double base = FlashbulbThreshold (S);
  return base * (1.0 - 0.5 * emotion_intensity) * (1.0 - 0.3 * arousal);
}

// --- Algorithm 4b (Mood Integrator) Helpers ---
inline double
AlphaMood (double S)
{
  // α_mood(S) = lerp(0.01, 0.20, S)
  // Higher sensitivity = faster mood reactivity to emotion events
  return Lerp (0.01, 0.20, Clamp (S, 0.0, 1.0));
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
  // base_capacity = round(lerp(5, 3, S) + lerp(-1, 1, F))
  // Paper Section 8.1: capacity range [2, 6]
  const double cap = Lerp (5.0, 3.0, S) + Lerp (-1.0, 1.0, F);
  return static_cast<int> (std::round (cap));
}

inline double
WMMaintenanceCostPerSlot (double S)
{
  // maintenance_cost_per_slot = lerp(0.05, 0.15, S)
  return Lerp (0.05, 0.15, S);
}

inline double
WMChunkingThreshold (double F)
{
  // chunking_threshold = lerp(0.7, 0.9, F)
  return Lerp (0.7, 0.9, F);
}

inline double
WMGateThreshold (double F)
{
  // gate_threshold = lerp(0.1, 0.4, F)
  // At F=0 (wide attention): permissive (0.1)
  // At F=1 (narrow attention): strict (0.4)
  return Lerp (0.1, 0.4, F);
}

inline double
WMComplexityScale (double S)
{
  // complexity penalty scale ~ sensitivity
  return Lerp (0.5, 1.5, S);
}

inline double
WMRehearsalRate (double S)
{
  // rehearsal_rate = lerp(0.5, 2.0, S)
  // Higher Sensitivity = faster rehearsal boost
  return Lerp (0.5, 2.0, S);
}

inline double
WMRehearsalThreshold (double F)
{
  // rehearsal_threshold = lerp(0.5, 0.7, F)
  // Below chunking_threshold(F) = lerp(0.7, 0.9, F)
  // Slots in [rehearsal_threshold, chunking_threshold) get rehearsal boost
  return Lerp (0.5, 0.7, F);
}

inline double
WMSlotDedicationStrength (double T)
{
  // slot_dedication_strength = lerp(0.3, 0.9, T)
  // Higher Stability = slots become more dedicated (resistant to eviction)
  return Lerp (0.3, 0.9, T);
}

// --- Algorithm 25 (Metacognitive Monitoring) Helpers ---
inline double
FOKThreshold (double F)
{
  // FOK_threshold = lerp(0.2, 0.5, F)
  return Lerp (0.2, 0.5, F);
}

inline double
TOTFokCutoff (double F)
{
  // TOT FOK cutoff = lerp(0.5, 0.8, F)
  return Lerp (0.5, 0.8, F);
}

inline double
TOTRetrievalCutoff (double F)
{
  // TOT retrieval cutoff = lerp(0.4, 0.2, F)
  return Lerp (0.4, 0.2, F);
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
  return Lerp (0.3, 0.1, F);
}

inline int
StrategySwitchLatencyMs (double S)
{
  // strategy_switch_latency = lerp(500, 100, S) ms
  return static_cast<int> (std::round (Lerp (500.0, 100.0, S)));
}

inline double
CertaintyRequirement (double T)
{
  // certainty_requirement = lerp(0.6, 0.9, T)
  return Lerp (0.6, 0.9, T);
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
  const double f01 = Clamp (F, 0.0, 1.0);
  const double t01 = Clamp (T, 0.0, 1.0);
  const double certainty = CertaintyRequirement (t01);
  return Clamp (certainty * (0.5 + 0.5 * f01), 0.0, 1.0);
}

inline double
MetacognitiveSensitivity (double F, double S)
{
  // metacognitive_sensitivity = F × (1 + 0.5 × S)
  return F * (1.0 + 0.5 * S);
}

// --- Algorithm 26 (Serial Position Effects) Helpers ---
inline int
SerialPrimacyWindow (double F)
{
  // primacy_window = round(lerp(5, 2, F))
  return static_cast<int> (std::round (Lerp (5.0, 2.0, F)));
}

inline int
SerialRecencyWindow (double F)
{
  // recency_window = round(lerp(7, 3, F))
  return static_cast<int> (std::round (Lerp (7.0, 3.0, F)));
}

inline double
SerialPrimacyBonus (double S)
{
  // primacy_bonus = lerp(1.2, 2.0, S)
  return Lerp (1.2, 2.0, S);
}

inline double
SerialRehearsalCurveDepth (double S)
{
  // rehearsal_curve_depth = lerp(0.2, 0.6, S)
  return Lerp (0.2, 0.6, S);
}

inline double
SerialDistinctivenessThreshold (double F)
{
  // distinctiveness_threshold = lerp(0.6, 0.8, F)
  return Lerp (0.6, 0.8, F);
}

inline double
SerialVonRestorffMultiplier (double S)
{
  // von_restorff_multiplier = lerp(1.5, 3.0, S)
  return Lerp (1.5, 3.0, S);
}

inline double
SerialMiddleSuppression (double S, double F)
{
  // middle_suppression = lerp(0.8, 0.5, S) × (1 − F)
  return Lerp (0.8, 0.5, S) * (1.0 - F);
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
  return Lerp (0.5, 0.8, Clamp (F, 0.0, 1.0));
}

inline double
MemoryUsageCacheDuration (double T)
{
  // Higher stability = longer memory of what was retrieved
  // Range [30, 300] seconds (30s to 5 minutes)
  return Lerp (30.0, 300.0, Clamp (T, 0.0, 1.0));
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
  return Lerp (0.90, 0.98, Clamp (F, 0.0, 1.0));
}

inline double
MinEdgeWeight (double F)
{
  // min_edge_weight(F) for graph traversal
  return Lerp (0.20, 0.60, Clamp (F, 0.0, 1.0));
}

inline double
ImpliesDriftThreshold (double T)
{
  // implies threshold (directional drift correlation)
  return Lerp (0.10, 0.25, Clamp (T, 0.0, 1.0));
}

inline double
ContradictionThreshold ()
{
  // Fixed threshold for contradiction detection
  // cos_sim < -0.5 indicates contradiction
  return -0.5;
}

// --- Section 8: Interrupt Gate Parameters ---

inline double
TauNovelty (double F, double S, double T)
{
  // tau_novelty = lerp(0.10, 0.35, F) * (1 - 0.15S) * (1 + 0.3T)
  // Reference: algorithms.md Section 8.1
  return Lerp (0.10, 0.35, Clamp (F, 0.0, 1.0))
       * (1.0 - 0.15 * Clamp (S, 0.0, 1.0))
       * (1.0 + 0.3 * Clamp (T, 0.0, 1.0));
}

inline double
RetrievalThreshold (double F)
{
  // retrieval_thresh(F) = lerp(0.25, 0.60, F)
  // Reference: algorithms.md Section 8.1
  return Lerp (0.25, 0.60, Clamp (F, 0.0, 1.0));
}

inline int
InterruptCandidateCount (double F)
{
  // K = round(lerp(10, 6, F))
  // Reference: algorithms.md Section 8.3
  return static_cast<int> (std::round (Lerp (10.0, 6.0, Clamp (F, 0.0, 1.0))));
}

inline double
DupThresh (double F, double T)
{
  // dup_thresh = lerp(0.96, 0.88, F) × (0.98 + 0.02T)
  return Lerp (0.96, 0.88, Clamp (F, 0.0, 1.0))
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
  return Lerp (0.48, 0.66, Clamp (F, 0.0, 1.0))
       * Lerp (1.1, 0.9, Clamp (S, 0.0, 1.0));
}

// Section 4.4.3 - Maximum memory time (seconds)
inline double
MaxMemoryTime (double T)
{
  // max_mem_time(T) = lerp(30, 120, T) seconds
  // Higher stability = longer memories allowed
  return Lerp (30.0, 120.0, Clamp (T, 0.0, 1.0));
}

// Section 4.4.3 - Maximum cumulative drift before flush
inline double
MaxMemoryDrift (double S)
{
  // max_mem_drift(S) = lerp(0.8, 2.0, S)
  // Higher sensitivity = more drift allowed before forced flush
  return Lerp (0.8, 2.0, Clamp (S, 0.0, 1.0));
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
       * Lerp (1.2, 0.8, Clamp (S, 0.0, 1.0));
}

// Section 4.4.5 - Window score peak vs average weight
inline double
WindowScoreAlpha (double F)
{
  // α(F) = lerp(0.3, 0.7, F)
  // Higher focus = more weight on peak score
  return Lerp (0.3, 0.7, Clamp (F, 0.0, 1.0));
}

// Section 4.4.5 - Coverage bonus weight
inline double
WindowScoreCoverageBeta (double S)
{
  // β(S) = lerp(0.05, 0.15, S)
  // Higher sensitivity = more bonus for complete memorys
  return Lerp (0.05, 0.15, Clamp (S, 0.0, 1.0));
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
  // ρ(F) = lerp(0.3, 0.7, F)
  // Higher focus = more weight on memory mean vs peak
  return Lerp (0.3, 0.7, Clamp (F, 0.0, 1.0));
}

} // namespace cortext::core
