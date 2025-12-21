#include "cortext/operations/focus.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>

namespace cortext::operations
{

void
InitializeFocusPriors::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const auto &config = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();

  if (p_ctx.focus_priors_initialized)
    {
      return;
    }

  p_ctx.weight_relevance_prior
      = core::Sigmoid (operations::constants::kTwo * config.focus
                       - operations::constants::kNormalizedMax);
  p_ctx.coverage_gain_floor_prior
      = operations::constants::kCoverageGainFloorBase
        + operations::constants::kCoverageGainScale * config.focus;
  p_ctx.mismatch_weight_prior
      = operations::constants::kNormalizedMax - config.focus;

  // Values from algorithms.md, section 0.2
  p_ctx.attention_width_prior = core::Lerp (
      static_cast<double> (core::kAttentionWidthMin),
      static_cast<double> (core::kAttentionWidthMax), 1.0 - config.focus);

  // Also initialize the dynamic values from the priors.
  p_ctx.weight_relevance = p_ctx.weight_relevance_prior;
  p_ctx.attention_width = p_ctx.attention_width_prior;
  p_ctx.coverage_gain_floor = p_ctx.coverage_gain_floor_prior;
  p_ctx.mismatch_weight = p_ctx.mismatch_weight_prior;
  p_ctx.focus_priors_initialized = true;

  telemetry::LogDebug ("cortext.focus.init",
                       { telemetry::Attribute::Double ("F", config.focus),
                         telemetry::Attribute::Double (
                             "weight_relevance_prior", p_ctx.weight_relevance_prior),
                         telemetry::Attribute::Double (
                             "coverage_gain_floor_prior", p_ctx.coverage_gain_floor_prior),
                         telemetry::Attribute::Double (
                             "mismatch_weight_prior", p_ctx.mismatch_weight_prior),
                         telemetry::Attribute::Double (
                             "attention_width_prior", p_ctx.attention_width_prior) });
}

void
UpdateFocus::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();
  const auto &signal = context.GetSignal ();

  // Calculate the mean of the recent context embeddings.
  // Appendix D: recent_context already includes x_t via UpdateRecentContext.
  double observed_cosine = 0.0;
  const size_t n_ctx = p_ctx.recent_context_embeddings.size ();
  size_t ctx_window_size = 0;
  size_t denom = 0;
  size_t start = 0;
  if (n_ctx > 0 && signal.embedding.size () > 0)
    {
      Eigen::VectorXf mean_context
          = Eigen::VectorXf::Zero (signal.embedding.size ());
      ctx_window_size = static_cast<size_t> (core::NCtx (config.stability));
      start = (n_ctx > ctx_window_size) ? (n_ctx - ctx_window_size) : 0;
      for (size_t i = start; i < n_ctx; ++i)
        {
          mean_context += p_ctx.recent_context_embeddings[i];
        }
      denom = (n_ctx > ctx_window_size) ? ctx_window_size : n_ctx;
      if (denom > 0)
        {
          mean_context /= static_cast<float> (denom);
        }

      // Alg 2: observed_cosine ← cos(x_t, mean(recent_context))
      observed_cosine
          = core::CosineSimilarity (signal.embedding, mean_context);
    }

  // Calculate the learning rate using the schedule from Section 0.3
  const double alpha_f = core::AlphaF (config.focus, p_ctx.u_t);

  // Transform cosine [-1, 1] to [0, 1] per algorithms.md Section 2.1.2
  const double mapped_cosine = core::Map01 (observed_cosine);

  // Alg 2: weight_relevance_t ← EWMA(...)
  p_ctx.weight_relevance
      = core::Ewma (p_ctx.weight_relevance, mapped_cosine, alpha_f);

  // Alg 2: attention_width_t ← clamp(...) - the doc doesn't specify a change,
  // just that it should be clamped. The actual change comes from feedback in
  // later algorithms (e.g., Alg 15). So we just ensure it stays in bounds.
  p_ctx.attention_width = core::Clamp (
      p_ctx.attention_width, static_cast<double> (core::kAttentionWidthMin),
      static_cast<double> (core::kAttentionWidthMax));

  telemetry::LogDebug (
      "cortext.focus.update",
      { telemetry::Attribute::Double ("F", config.focus),
        telemetry::Attribute::Double ("u_t", p_ctx.u_t),
        telemetry::Attribute::Int64 ("n_ctx", static_cast<int64_t> (n_ctx)),
        telemetry::Attribute::Int64 ("ctx_window_size", static_cast<int64_t> (ctx_window_size)),
        telemetry::Attribute::Int64 ("ctx_start", static_cast<int64_t> (start)),
        telemetry::Attribute::Int64 ("ctx_denom", static_cast<int64_t> (denom)),
        telemetry::Attribute::Double ("observed_cosine", observed_cosine),
        telemetry::Attribute::Double ("mapped_cosine", mapped_cosine),
        telemetry::Attribute::Double ("alpha_f", alpha_f),
        telemetry::Attribute::Double ("weight_relevance", p_ctx.weight_relevance),
        telemetry::Attribute::Double ("attention_width", p_ctx.attention_width),
        telemetry::Attribute::Int64 (
            "context_window_size",
            static_cast<int64_t> (p_ctx.recent_context_embeddings.size ())) });
}

} // namespace cortext::operations
