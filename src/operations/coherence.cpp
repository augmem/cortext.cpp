#include "cortext/operations/coherence.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include <cmath>
#include <vector>

namespace cortext::operations
{

namespace
{
inline double
Variance (const std::vector<double> &values)
{
  if (values.size () < 2)
    {
      return 0.0;
    }
  double sum = 0.0;
  for (double v : values)
    {
      sum += v;
    }
  const double mean = sum / static_cast<double> (values.size ());
  double accum = 0.0;
  for (double v : values)
    {
      const double d = v - mean;
      accum += d * d;
    }
  return accum / static_cast<double> (values.size ());
}
} // namespace

void
ComputeCoherence::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();

  // Use last NCtx(T) embeddings (or fewer if not available).
  const int window = static_cast<int> (core::NCtx (cfg.stability));
  const int n = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  const int start = std::max (0, n - window);

  std::vector<double> cos_values;
  cos_values.reserve (static_cast<size_t> (n - start));

  for (int i = start; i < n; ++i)
    {
      const auto &emb
          = p_ctx.recent_context_embeddings[static_cast<size_t> (i)];
      const double c = core::CosineSimilarity (signal.embedding, emb);
      cos_values.push_back (c);
    }

  // If there are fewer than 2 values, variance is 0 => coherence = 1.
  const double var = Variance (cos_values);
  const double coherence = 1.0 - core::Clamp (var, 0.0, 1.0);
  context.SetCoherence (coherence);
}

} // namespace cortext::operations
