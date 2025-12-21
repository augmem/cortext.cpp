#include "cortext/operations/recent_context.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"

namespace cortext::operations
{

void
UpdateRecentContext::Execute (OperationContext &context,
                              Transaction & /*tx*/) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &signal = context.GetSignal ();

  p_ctx.recent_context_embeddings.push_back (signal.embedding);

  const size_t context_window_size
      = static_cast<size_t> (core::NCtx (cfg.stability));
  const size_t drift_lag = static_cast<size_t> (core::KCtx (cfg.stability));
  const size_t max_window = context_window_size + drift_lag;
  if (p_ctx.recent_context_embeddings.size () > max_window)
    {
      p_ctx.recent_context_embeddings.pop_front ();
    }
}

} // namespace cortext::operations
