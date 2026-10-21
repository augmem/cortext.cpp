#include "cortext/operations/sensitivity.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace cortext::operations
{

void
InitializeSensitivityPriors::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();

  if (p_ctx.sensitivity_priors_initialized)
    {
      return;
    }

  const double S_raw = config.sensitivity;
  const double S_eff = core::SensitivityBias (S_raw);
  const auto sensitivity_priors = core::SensitivityStatePriorsForKnobs (S_raw);

  // Priors per algorithms.md Algorithm 3
  p_ctx.weight_novelty_prior = sensitivity_priors.novelty_weight;
  p_ctx.weight_surprise_prior = sensitivity_priors.surprise_weight;
  p_ctx.weight_valence_prior = sensitivity_priors.valence_weight;
  p_ctx.weight_arousal_prior = sensitivity_priors.arousal_weight;
  p_ctx.weight_emotion_prior = sensitivity_priors.emotion_weight;
  p_ctx.emotion_gain_prior = sensitivity_priors.emotion_gain;
  p_ctx.score_gain_prior = sensitivity_priors.score_gain;
  // Initialize dynamic values from priors
  p_ctx.weight_novelty = p_ctx.weight_novelty_prior;
  p_ctx.weight_surprise = p_ctx.weight_surprise_prior;
  p_ctx.weight_valence = p_ctx.weight_valence_prior;
  p_ctx.weight_arousal = p_ctx.weight_arousal_prior;
  p_ctx.emotion_gain = p_ctx.emotion_gain_prior;
  p_ctx.score_gain = p_ctx.score_gain_prior;
  p_ctx.sensitivity_priors_initialized = true;

  telemetry::LogDebug("cortext.initialize_sensitivity_priors", {
    telemetry::Attribute::Double("S", S_raw),
    telemetry::Attribute::Double("S_eff", S_eff),
    telemetry::Attribute::Double("weight_novelty_prior", p_ctx.weight_novelty_prior),
    telemetry::Attribute::Double("weight_surprise_prior", p_ctx.weight_surprise_prior)
  });
}

namespace
{
constexpr int kNumEmotions = 6; // anger, fear, joy, love, sadness, surprise
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
UpdateSensitivity::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double S_raw = cfg.sensitivity;
  const auto emotion_policy = core::EmotionProjectionPolicyForKnobs (S_raw);

  // --- Emotion projections if centroids provided ---
  double emotion_intensity = 0.0;
  double valence = constants::kOneHalf;
  double arousal = 0.0;
  std::array<double, 6> emotion_probs;
  const double uniform_prob = 1.0 / static_cast<double> (kNumEmotions);
  emotion_probs.fill (uniform_prob);
  const auto &signal = context.GetSignal ();
  const Eigen::VectorXf *x_ptr = &signal.embedding;
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it != p_ctx.accumulator_states.end ()
      && acc_it->second.mu_acc.size () > 0)
    {
      x_ptr = &acc_it->second.mu_acc;
    }
  const auto &x = *x_ptr;
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
          SoftmaxNormalize (raw_cos, emotion_policy.softmax_beta);

          // Expose emotion probabilities (stored with the memory)
          for (int i = 0; i < kNumEmotions; ++i)
            {
              emotion_probs[static_cast<size_t> (i)]
                  = raw_cos[static_cast<size_t> (i)];
            }

          const double peak
              = *std::max_element (raw_cos.begin (), raw_cos.end ());
          const double conf
              = 1.0
                - Entropy (raw_cos)
                      / std::log (static_cast<double> (kNumEmotions));
          const double intensity_base = std::max (0.0, peak * conf);
          emotion_intensity = core::Clamp (
              std::pow (intensity_base, emotion_policy.intensity_gamma)
                  * emotion_policy.intensity_gain,
              0.0, 1.0);
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
  context.SetEmotionProbabilities (emotion_probs);

  // Apply EWMA smoothing to emotion state for persistence.
  const double alpha_emotion = emotion_policy.ewma_alpha;
  p_ctx.emotion_intensity_ewma
      = core::Ewma (p_ctx.emotion_intensity_ewma, emotion_intensity, alpha_emotion);
  p_ctx.valence_ewma = core::Ewma (p_ctx.valence_ewma, valence, alpha_emotion);
  p_ctx.arousal_ewma = core::Ewma (p_ctx.arousal_ewma, arousal, alpha_emotion);

  // Set raw values for output
  context.SetEmotionIntensity (emotion_intensity);
  context.SetValence (valence);
  context.SetArousal (arousal);

  // Track recent emotion intensity for percentile-based flashbulb gating.
  const int emo_window = core::EmotionHistoryWindow (S_raw, cfg.stability);
  auto &emo_hist = p_ctx.recent_emotion_intensities;
  emo_hist.push_back (emotion_intensity);
  while (static_cast<int> (emo_hist.size ()) > emo_window)
    {
      emo_hist.pop_front ();
    }

  telemetry::LogDebug("cortext.update_sensitivity", {
    telemetry::Attribute::Double("emotion_intensity", emotion_intensity),
    telemetry::Attribute::Double("valence", valence),
    telemetry::Attribute::Double("arousal", arousal)
  });
}

} // namespace cortext::operations
