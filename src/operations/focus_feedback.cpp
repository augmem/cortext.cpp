#include "cortext/operations/focus_feedback.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/constants.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
ApplyFocusFeedback::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  // Avoid relying on context.GetConfig() here because some tests create
  // OperationContext with a short-lived Config. Derive an effective focus
  // scale from available, stable state. Prefer explicitly provided effective
  // focus, otherwise fall back to the prior-derived relevance weight.
  const double alpha_f = constants::kAlphaFBase;
  const double beta_f = constants::kBetaFBase;
  const double aw_min = static_cast<double> (core::kAttentionWidthMin);
  const double aw_max = static_cast<double> (core::kAttentionWidthMax);
  double F = 1.0;
  if (aw_max > aw_min)
    {
      const double prior_ratio
          = (p_ctx.attention_width_prior - aw_min) / (aw_max - aw_min);
      F = core::Clamp (1.0 - prior_ratio, 0.0, 1.0);
    }
  const double S = core::Clamp ((p_ctx.weight_surprise_prior - 0.2) / 0.8,
                                0.0, 1.0);
  const double T
      = core::Clamp (1.0 - 2.0 * p_ctx.drift_weight_prior, 0.0, 1.0);

  double contextual_gain_mean = 0.0;
  double weight_adjustment = 0.0;
  double attention_width_delta = 0.0;
  int event_count = 0;

  const auto &events = context.GetMemoryUsageEvents ();
  for (const auto &e : events)
    {
      if (!e.used || !e.contextual_gain.has_value ())
        {
          continue;
        }
      const double cg = *e.contextual_gain;
      contextual_gain_mean += cg;
      ++event_count;

      const double prev_weight = p_ctx.weight_relevance;
      const double prev_width = p_ctx.attention_width;

      if (cg > 0.0)
        {
          // Positive gain: boost relevance, narrow attention
          p_ctx.weight_relevance = core::Clamp (
              p_ctx.weight_relevance + alpha_f * cg, constants::kNormalizedMin,
              constants::kNormalizedMax);
          p_ctx.attention_width
              = core::Clamp (p_ctx.attention_width * (1.0 - beta_f),
                             static_cast<double> (core::kAttentionWidthMin),
                             static_cast<double> (core::kAttentionWidthMax));
        }
      else
        {
          // Non-positive: widen attention
          p_ctx.attention_width
              = core::Clamp (p_ctx.attention_width * (1.0 + beta_f),
                             static_cast<double> (core::kAttentionWidthMin),
                             static_cast<double> (core::kAttentionWidthMax));
        }

      weight_adjustment += (p_ctx.weight_relevance - prev_weight);
      attention_width_delta += (p_ctx.attention_width - prev_width);
    }

  if (event_count > 0)
    {
      contextual_gain_mean /= event_count;
    }

  const double d_step = context.GetAccumulatorDriftStep ();
  const double dilation_force = S * (1.0 - T) * d_step;
  const double restoring_force
      = F * (p_ctx.attention_width_prior - p_ctx.attention_width);
  const double prev_width = p_ctx.attention_width;
  p_ctx.attention_width = core::Clamp (
      p_ctx.attention_width + dilation_force + restoring_force,
      aw_min, aw_max);
  attention_width_delta += (p_ctx.attention_width - prev_width);

  telemetry::LogDebug ("cortext.focus_feedback",
                       { telemetry::Attribute::Double ("contextual_gain_mean",
                                                       contextual_gain_mean),
                         telemetry::Attribute::Double ("weight_adjustment",
                                                       weight_adjustment),
                         telemetry::Attribute::Double ("attention_width_delta",
                                                       attention_width_delta),
                         telemetry::Attribute::Double ("F", F),
                         telemetry::Attribute::Double ("S", S),
                         telemetry::Attribute::Double ("T", T),
                         telemetry::Attribute::Double ("drift_step", d_step),
                         telemetry::Attribute::Double ("dilation_force", dilation_force),
                         telemetry::Attribute::Double ("restoring_force", restoring_force) });
}

} // namespace cortext::operations
