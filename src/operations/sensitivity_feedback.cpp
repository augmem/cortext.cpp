#include "cortext/operations/sensitivity_feedback.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <vector>

namespace cortext::operations
{
inline Eigen::VectorXf
ComputeMean (const std::deque<Eigen::VectorXf> &embs)
{
  if (embs.empty ())
    {
      return Eigen::VectorXf ();
    }
  const int dim = static_cast<int> (embs.front ().size ());
  Eigen::VectorXf mean = Eigen::VectorXf::Zero (dim);
  for (const auto &v : embs)
    {
      mean += v;
    }
  mean /= static_cast<float> (embs.size ());
  return mean;
}

inline double
CosTo01 (double c /* [-1,1] */)
{
  return std::min (
      constants::kNormalizedMax,
      std::max (constants::kNormalizedMin,
                constants::kOneHalf * (c + constants::kNormalizedMax)));
}

void
ApplySensitivityFeedback::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();

  const double eta = constants::kEtaBase;

  // Attempt to obtain novelty from previously computed metrics.
  // Fallback computes novelty from embeddings if necessary.
  auto relevance_metric = context.GetMetric (operations::Metric::relevance);
  double novelty_from_metric = constants::kNormalizedMin;
  if (relevance_metric.has_value ())
    {
      novelty_from_metric
          = core::Clamp (constants::kNormalizedMax - *relevance_metric,
                         constants::kNormalizedMin, constants::kNormalizedMax);
    }
  else
    {
      const auto &x = context.GetSignal ().embedding;
      const Eigen::VectorXf mean_ctx
          = ComputeMean (p_ctx.recent_context_embeddings);
      if (mean_ctx.size () == x.size () && x.size () > 0)
        {
          const double c = core::CosineSimilarity (x, mean_ctx);
          novelty_from_metric = core::Clamp (
              constants::kNormalizedMax - CosTo01 (c),
              constants::kNormalizedMin, constants::kNormalizedMax);
        }
      else
        {
          novelty_from_metric = constants::kNormalizedMin;
        }
    }

  // Redundancy fallback (no kNN DB here) → 0.0 per plan choice.
  const double redundancy = constants::kNormalizedMin;

  const auto &events = context.GetMemoryUsageEvents ();
  for (const auto &e : events)
    {
      if (!e.used || !e.contextual_gain.has_value ())
        {
          continue;
        }
      const double cg = *e.contextual_gain; // may be negative
      const double adjustment = eta * (novelty_from_metric * cg - redundancy);
      p_ctx.weight_novelty
          = core::Clamp (p_ctx.weight_novelty + adjustment,
                         constants::kNormalizedMin, constants::kNormalizedMax);
    }
}

} // namespace cortext::operations
