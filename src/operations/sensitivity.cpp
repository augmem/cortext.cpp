#include "cortext/operations/sensitivity.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace cortext::operations
{

void
InitializeSensitivityPriors::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();

  if (p_ctx.sensitivity_priors_initialized)
    {
      return;
    }

  const double S = config.sensitivity;

  // Priors per algorithms.md Algorithm 3
  p_ctx.base_rate_prior = core::BaseRatePrior (S);
  p_ctx.weight_novelty_prior = 0.3 + 0.7 * S;
  p_ctx.weight_surprise_prior = 0.2 + 0.8 * S;
  p_ctx.weight_valence_prior = 0.4 + 0.6 * S;
  p_ctx.weight_arousal_prior = S;
  p_ctx.weight_emotion_prior = 0.2 + 0.8 * S;
  p_ctx.emotion_gain_prior = std::exp (1.5 * S);
  p_ctx.score_gain_prior = std::exp (2.0 * S);
  p_ctx.rate_target_prior = p_ctx.base_rate_prior
                           * (constants::kRateTargetBase
                              + constants::kRateTargetMultiplier * S);
  // Initialize dynamic values from priors
  p_ctx.weight_novelty = p_ctx.weight_novelty_prior;
  p_ctx.sensitivity_priors_initialized = true;
}

namespace
{
constexpr int kNumEmotions = 6; // anger, fear, joy, love, sadness, surprise
constexpr double kTiny = 1e-6;
constexpr double kPerMinute = 60.0;
constexpr double kClampRateMax = 1000.0;
// Valence map (−1..+1 approx scaled to ~[−0.9,+0.9])
static const double kVMap[kNumEmotions]
    = { -0.9, -0.8, +0.9, +0.8, -0.9, 0.0 };
// Arousal map ([0..1])
static const double kAMap[kNumEmotions]
    = { +0.9, +0.9, +0.6, +0.5, +0.3, +0.8 };

inline double
SoftmaxNormalize (std::vector<double> &logits, double beta)
{
  // logits will be transformed in place to probabilities
  double max_logit = *std::max_element (logits.begin (), logits.end ());
  double denom = 0.0;
  for (double &z : logits)
    {
      z = std::exp (beta * (z - max_logit));
      denom += z;
    }
  if (denom <= 0.0)
    {
      return 0.0;
    }
  for (double &z : logits)
    {
      z /= denom;
    }
  return denom;
}

inline double
Entropy (const std::vector<double> &p)
{
  double h = 0.0;
  for (double v : p)
    {
      if (v > 0.0)
        {
          h -= v * std::log (v);
        }
    }
  return h;
}
} // namespace

void
UpdateSensitivity::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double S = cfg.sensitivity;

  // --- Emotion projections if centroids provided ---
  double emotion_intensity = 0.0;
  double valence = constants::kOneHalf;
  double arousal = 0.0;
  const auto &x = context.GetSignal ().embedding;
  if (p_ctx.centroids.has_value ()
      && p_ctx.centroids->emotion_centroids.size () == kNumEmotions)
    {
      const auto &centroids = p_ctx.centroids->emotion_centroids;
      std::vector<double> raw_cos;
      raw_cos.reserve (kNumEmotions);
      bool any_pos = false;
      for (int i = 0; i < kNumEmotions; ++i)
        {
          const double c = core::CosineSimilarity (
              x, centroids[static_cast<size_t> (i)]);
          raw_cos.push_back (c);
          if (c > 0.0)
            any_pos = true;
        }
      if (any_pos)
        {
          // logits = max(0, raw_cos)
          for (double &v : raw_cos)
            {
              v = std::max (0.0, v);
            }
          // softmax with β(S) = 4 + 8S
          const double beta = 4.0 + 8.0 * S;
          SoftmaxNormalize (raw_cos, beta);

          // Expose emotion probabilities for Algorithm 4b (UpdateMood)
          std::array<double, 6> emotion_probs;
          for (int i = 0; i < kNumEmotions; ++i)
            {
              emotion_probs[static_cast<size_t> (i)]
                  = raw_cos[static_cast<size_t> (i)];
            }
          context.SetEmotionProbabilities (emotion_probs);

          const double peak
              = *std::max_element (raw_cos.begin (), raw_cos.end ());
          const double conf
              = 1.0
                - Entropy (raw_cos)
                      / std::log (static_cast<double> (kNumEmotions));
          emotion_intensity = std::sqrt (std::max (0.0, peak * conf));
          // valence, arousal
          double v_sum = 0.0;
          double a_sum = 0.0;
          for (int i = 0; i < kNumEmotions; ++i)
            {
              v_sum += raw_cos[static_cast<size_t> (i)] * kVMap[i];
              a_sum += raw_cos[static_cast<size_t> (i)] * kAMap[i];
            }
          // Normalize valence to [0,1] with (valence_raw + 0.9)/1.8
          valence = (v_sum + 0.9) / 1.8;
          arousal = core::Clamp (a_sum, 0.0, 1.0);
        }
      else
        {
          emotion_intensity = 0.0;
          valence = constants::kOneHalf;
          arousal = 0.0;
        }
    }
  // else: fallback path → keep defaults above

  // Apply EWMA smoothing to emotion state for persistence
  // Higher Sensitivity = faster adaptation (α ∈ [0.05, 0.30])
  const double alpha_emotion = core::Lerp (0.05, 0.30, S);
  p_ctx.emotion_intensity_ewma
      = core::Ewma (p_ctx.emotion_intensity_ewma, emotion_intensity, alpha_emotion);
  p_ctx.valence_ewma = core::Ewma (p_ctx.valence_ewma, valence, alpha_emotion);
  p_ctx.arousal_ewma = core::Ewma (p_ctx.arousal_ewma, arousal, alpha_emotion);

  // Set raw values for output (use smoothed values for threshold modulation)
  context.SetEmotionIntensity (emotion_intensity);
  context.SetValence (valence);
  context.SetArousal (arousal);

  // --- ΔThreshold components ---
  // ΔT_emo = − κ_emo × emotion_intensity × (0.5 + 0.5 × arousal)
  const double kappa_base = constants::kGainMedium;
  const double kappa_emo = kappa_base * S;
  const double delta_T_emo
      = -kappa_emo * emotion_intensity
        * (constants::kOneHalf + constants::kOneHalf * arousal);
  context.SetDeltaThresholdEmotion (delta_T_emo);

  // gain_t = exp(2S) × (1 + 0.1 × var(scores_{t−w:t}))
  const int w = core::WScore (cfg.stability);
  const int n = static_cast<int> (p_ctx.recent_scores.size ());
  const int start = std::max (0, n - w);
  int m = n - start;
  double var_scores = 0.0;
  if (m >= 2)
    {
      double sum = 0.0;
      for (int i = start; i < n; ++i)
        {
          sum += core::Clamp (p_ctx.recent_scores[static_cast<size_t> (i)],
                              constants::kNormalizedMin,
                              constants::kNormalizedMax);
        }
      const double mean = sum / static_cast<double> (m);
      double accum = 0.0;
      for (int i = start; i < n; ++i)
        {
          const double v
              = core::Clamp (p_ctx.recent_scores[static_cast<size_t> (i)],
                             constants::kNormalizedMin,
                             constants::kNormalizedMax);
          const double d = v - mean;
          accum += d * d;
        }
      var_scores
          = core::Clamp (accum / static_cast<double> (m),
                         constants::kNormalizedMin, constants::kNormalizedMax);
    }
  const double gain_t = std::exp (2.0 * S) * (1.0 + 0.1 * var_scores);
  double delta_T_sens = -gain_t * (S - constants::kOneHalf) + delta_T_emo;
  context.SetDeltaThresholdSensitivity (delta_T_sens);

  // --- rate_target_t update (EWMA with observed write rate) ---
  // Prefer observed write rate from window (writes/min); fallback to Δt proxy.
  const uint64_t ts = context.GetSignal ().timestamp;
  if (p_ctx.rate_target == 0.0)
    {
      p_ctx.rate_target = p_ctx.rate_target_prior;
    }
  // EWMA alpha derived from Sensitivity: α ∈ [0.2, 0.5]
  const double alpha_rate = core::Lerp (0.2, 0.5, S);
  double rate_obs = p_ctx.write_rate_window_.RatePerMinute (alpha_rate);
  if (rate_obs <= 0.0 && p_ctx.last_signal_timestamp != 0
      && ts > p_ctx.last_signal_timestamp)
    {
      const double dt_seconds
          = static_cast<double> (ts - p_ctx.last_signal_timestamp);
      rate_obs = (dt_seconds > kTiny) ? (kPerMinute / dt_seconds) : 0.0;
    }
  const double alpha_s = core::AlphaS (S, p_ctx.u_t);
  p_ctx.rate_target = core::Ewma (
      p_ctx.rate_target,
      core::Clamp (rate_obs, constants::kNormalizedMin, kClampRateMax),
      alpha_s);
  p_ctx.last_signal_timestamp = ts;
}

void
UpdateMood::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double S = cfg.sensitivity;
  const double T = cfg.stability;

  // Get instantaneous emotion probabilities from Algorithm 4
  const auto &e_t = context.GetEmotionProbabilities ();

  // Get mood dynamics parameters (Algorithm 4b)
  const double alpha_mood = core::AlphaMood (S);
  const double lambda_mood = core::LambdaMood (T);

  // Update mood vector: M_t = λ_mood(T) × M_{t-1} + α_mood(S) × e_t
  // Then clamp each dimension to [-1, 1]
  auto &M = p_ctx.mood_vector;
  for (size_t i = 0; i < 6; ++i)
    {
      M[i] = lambda_mood * M[i] + alpha_mood * e_t[i];
      M[i] = core::Clamp (M[i], -1.0, 1.0);
    }

  // Compute mood magnitude ||M_t||
  double magnitude_sq = 0.0;
  for (size_t i = 0; i < 6; ++i)
    {
      magnitude_sq += M[i] * M[i];
    }
  const double magnitude = std::sqrt (magnitude_sq);

  // Normalize by √6 (max possible magnitude for 6-dim unit-bounded vector)
  // and clamp to [0, 1] per paper Section 4.2.4
  const double m_norm = core::Clamp (magnitude / std::sqrt (6.0), 0.0, 1.0);

  // Compute threshold delta: ΔT_mood = −κ_mood × clamp(m_norm, 0, 1)
  const double kappa_mood = constants::kGainMedium * S;
  const double delta_T_mood = -kappa_mood * m_norm;
  context.SetDeltaThresholdMood (delta_T_mood);
}

} // namespace cortext::operations
