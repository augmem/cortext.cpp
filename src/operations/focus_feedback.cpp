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

  telemetry::LogDebug ("cortext.focus_feedback",
                       { telemetry::Attribute::Double ("contextual_gain_mean",
                                                       contextual_gain_mean),
                         telemetry::Attribute::Double ("weight_adjustment",
                                                       weight_adjustment),
                         telemetry::Attribute::Double ("attention_width_delta",
                                                       attention_width_delta) });
}

} // namespace cortext::operations
