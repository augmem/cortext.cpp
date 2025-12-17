#include "cortext/operations/focus.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include <Eigen/Dense>

namespace cortext::operations
{

void
InitializeFocusPriors::Execute (OperationContext &context, Transaction &tx) const
{
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
  p_ctx.focus_priors_initialized = true;
}

void
UpdateFocus::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();
  const auto &signal = context.GetSignal ();

  // On the first signal, just add it to the context and return.
  if (p_ctx.recent_context_embeddings.empty ())
    {
      p_ctx.recent_context_embeddings.push_back (signal.embedding);
      return;
    }

  // Calculate the mean of the recent context embeddings
  Eigen::VectorXf mean_context
      = Eigen::VectorXf::Zero (signal.embedding.size ());
  for (const auto &emb : p_ctx.recent_context_embeddings)
    {
      mean_context += emb;
    }
  mean_context /= p_ctx.recent_context_embeddings.size ();

  // Alg 2: observed_cosine ← cos(x_t, mean(recent_context))
  double observed_cosine
      = core::CosineSimilarity (signal.embedding, mean_context);

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

  // --- Post-update ---
  // Manage the recent context window.
  const size_t context_window_size = core::NCtx (config.stability);
  p_ctx.recent_context_embeddings.push_back (signal.embedding);
  if (p_ctx.recent_context_embeddings.size () > context_window_size)
    {
      p_ctx.recent_context_embeddings.pop_front ();
    }
}

} // namespace cortext::operations
