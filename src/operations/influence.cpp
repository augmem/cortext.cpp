#include "cortext/operations/influence.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <vector>

namespace cortext::operations
{

inline Eigen::VectorXf
Unit (const Eigen::VectorXf &v)
{
  const double n = v.norm ();
  if (n <= constants::kNormEpsilon)
    {
      return v;
    }
  return v / static_cast<float> (n);
}

void
ApplyInfluenceFeedback::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // Use recent_context_embeddings as source of current/previous generation
  // embeddings to avoid relying on transient signal lifetime.
  if (p_ctx.recent_context_embeddings.empty ())
    {
      return;
    }
  const Eigen::VectorXf &x_t
      = p_ctx.recent_context_embeddings.back (); // current
  const Eigen::VectorXf u_cur = Unit (x_t);

  // Previous embedding from history if available.
  Eigen::VectorXf u_prev = u_cur;
  if (p_ctx.recent_context_embeddings.size () >= 2)
    {
      const Eigen::VectorXf &x_prev
          = p_ctx.recent_context_embeddings
                [p_ctx.recent_context_embeddings.size () - 2];
      u_prev = Unit (x_prev);
    }

  const Eigen::VectorXf delta = u_cur - u_prev;
  const double n_delta = delta.norm ();
  const Eigen::VectorXf delta_hat
      = (n_delta > constants::kNormEpsilon)
            ? (delta / static_cast<float> (n_delta))
            : u_cur;

  const auto &events = context.GetMemoryUsageEvents ();
  const auto &emb_map = context.GetRetrievedMemoryEmbeddings ();

  std::vector<double> influences;
  influences.reserve (events.size ());

  for (const auto &e : events)
    {
      if (!e.used)
        {
          continue;
        }
      auto it = emb_map.find (e.embedding_id);
      if (it == emb_map.end ())
        {
          continue;
        }
      const Eigen::VectorXf u_m = Unit (it->second);
      const double contextual_gain = e.contextual_gain.value_or (0.0);
      double sim_gen = core::CosineSimilarity (u_cur, u_m);
      if (sim_gen < constants::kNormalizedMin)
        sim_gen = constants::kNormalizedMin;
      if (sim_gen > constants::kNormalizedMax)
        sim_gen = constants::kNormalizedMax;
      const double cos_md = core::CosineSimilarity (u_m, delta_hat);
      const double drift_contrib
          = n_delta * std::max (constants::kNormalizedMin, cos_md);
      const double influence
          = constants::kLambda1 * contextual_gain
            + constants::kLambda2 * sim_gen
            - constants::kLambda3 * drift_contrib;
      influences.push_back (influence);
    }

  if (influences.empty ())
    {
      return;
    }

  const double mean_influence
      = std::accumulate (influences.begin (), influences.end (), 0.0)
        / static_cast<double> (influences.size ());

  // Apply to derived parameters (not knobs)
  // attention_width_t ← clamp(attention_width_t × (1 − 0.05 × mean_influence))
  const double prev_attention_width = p_ctx.attention_width;
  p_ctx.attention_width
      = core::Clamp (p_ctx.attention_width
                         * (1.0 - constants::kGainSmall * mean_influence),
                     static_cast<double> (core::kAttentionWidthMin),
                     static_cast<double> (core::kAttentionWidthMax));
  const double attention_delta = p_ctx.attention_width - prev_attention_width;

  // rate_target_t ← EWMA(rate_target_t, rate_target_t × (1 + 0.10 ×
  // mean_influence), α = α_S(rate))
  if (p_ctx.rate_target == 0.0)
    {
      p_ctx.rate_target = p_ctx.rate_target_prior;
    }
  const double alpha_s = core::AlphaS (cfg.sensitivity, p_ctx.u_t);
  const double rate_target_new
      = p_ctx.rate_target * (1.0 + constants::kGainMedium * mean_influence);
  const double prev_rate_target = p_ctx.rate_target;
  p_ctx.rate_target = core::Ewma (p_ctx.rate_target, rate_target_new, alpha_s);
  const double rate_delta = p_ctx.rate_target - prev_rate_target;

  // sustained_influence ← EWMA(sustained_influence, mean_influence, α =
  // 2/(L_sustain(T)+1))
  const double L_sustain
      = std::round (core::Lerp (constants::kSustainWindowMin,
                                constants::kSustainWindowMax, cfg.stability));
  const double alpha_sustain
      = (L_sustain > 0.0) ? (constants::kTwo / (L_sustain + 1.0)) : 1.0;
  p_ctx.sustained_influence
      = core::Ewma (p_ctx.sustained_influence, mean_influence, alpha_sustain);

  // hysteresis_t ← clamp(hysteresis_t × (1 + 0.05 × sustained_influence),
  // band_min, band_max)
  const double prev_hysteresis = p_ctx.hysteresis;
  p_ctx.hysteresis = core::Clamp (
      p_ctx.hysteresis * (1.0 + constants::kGainSmall * p_ctx.sustained_influence),
      constants::kHysteresisBandMin, constants::kHysteresisBandMax);
  const double hysteresis_delta = p_ctx.hysteresis - prev_hysteresis;

  telemetry::LogDebug ("cortext.influence",
                       { telemetry::Attribute::Double ("mean_influence",
                                                       mean_influence),
                         telemetry::Attribute::Double ("attention_delta",
                                                       attention_delta),
                         telemetry::Attribute::Double ("rate_delta",
                                                       rate_delta),
                         telemetry::Attribute::Double ("hysteresis_delta",
                                                       hysteresis_delta) });
}

} // namespace cortext::operations
