#include "cortext/operations/focus_spread.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace cortext::operations
{

void
ComputeFocusSpread::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &x = context.GetSignal ().embedding;

  const int n = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  if (n < 2)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  const int k = std::min (core::KNeighbors (cfg.stability), n);
  if (k < 2)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  const int start = n - k;
  std::vector<double> sims;
  sims.reserve (static_cast<size_t> (k));
  for (int i = start; i < n; ++i)
    {
      const auto &emb
          = p_ctx.recent_context_embeddings[static_cast<size_t> (i)];
      sims.push_back (core::CosineSimilarity (x, emb));
    }

  // Softmax with numerical stability: subtract max similarity.
  const double max_s = *std::max_element (sims.begin (), sims.end ());
  double denom = 0.0;
  for (double &s : sims)
    {
      s = std::exp (s - max_s);
      denom += s;
    }
  if (denom <= 0.0)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  double entropy = 0.0;
  for (double s : sims)
    {
      const double p = s / denom;
      entropy -= (p > 0.0) ? (p * std::log (p)) : 0.0;
    }

  const double norm = std::log (static_cast<double> (k));
  if (norm <= 0.0)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  const double focus_spread = core::Clamp (entropy / norm, 0.0, 1.0);
  context.SetMetric (operations::Metric::focus_spread, focus_spread);
}

} // namespace cortext::operations
