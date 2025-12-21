#include "cortext/operations/sensitivity_feedback.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <deque>
#include <vector>

namespace cortext::operations
{
inline double
ComputeRedundancyToContext (const std::deque<Eigen::VectorXf> &embs,
                            const Eigen::VectorXf &x, int window)
{
  if (x.size () == 0)
    {
      return constants::kNormalizedMin;
    }
  const int n = static_cast<int> (embs.size ());
  if (n == 0)
    {
      return constants::kNormalizedMin;
    }
  const int start = std::max (0, n - window);
  double max_cos = -1.0;
  int count = 0;
  for (int i = start; i < n; ++i)
    {
      const auto &emb = embs[static_cast<size_t> (i)];
      if (emb.size () != x.size ())
        {
          continue;
        }
      const double c = core::CosineSimilarity (x, emb);
      max_cos = std::max (max_cos, c);
      ++count;
    }
  if (count == 0)
    {
      return constants::kNormalizedMin;
    }
  const double redundancy = core::Map01 (max_cos);
  return core::Clamp (redundancy, constants::kNormalizedMin,
                      constants::kNormalizedMax);
}

void
ApplySensitivityFeedback::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();

  const double eta = constants::kEtaBase;

  const int window = static_cast<int> (core::NCtx (context.GetConfig ().stability));
  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();

  double redundancy_mean = 0.0;
  double weight_novelty_delta = 0.0;
  int event_count = 0;

  const auto &events = context.GetMemoryUsageEvents ();
  for (const auto &e : events)
    {
      if (!e.used || !e.contextual_gain.has_value ())
        {
          continue;
        }
      auto it = retrieved.find (e.embedding_id);
      if (it == retrieved.end ())
        {
          continue;
        }
      const double cg = *e.contextual_gain; // may be negative
      const double redundancy = ComputeRedundancyToContext (
          p_ctx.recent_context_embeddings, it->second, window);
      const double novelty_reward
          = core::Clamp (constants::kNormalizedMax - redundancy,
                         constants::kNormalizedMin,
                         constants::kNormalizedMax);
      redundancy_mean += redundancy;
      ++event_count;

      const double prev_weight = p_ctx.weight_novelty;
      const double adjustment = eta * (novelty_reward * cg - redundancy);
      p_ctx.weight_novelty
          = core::Clamp (p_ctx.weight_novelty + adjustment,
                         constants::kNormalizedMin, constants::kNormalizedMax);
      weight_novelty_delta += (p_ctx.weight_novelty - prev_weight);

      // v2: Store computed redundancy in memories for consolidation scoring
      // (Section 9.2: score = T*strength - F*redundancy + S*connectivity + T*stability)
      tx.Execute ("UPDATE memories SET redundancy = ? WHERE embedding_id = ?",
                  { redundancy, e.embedding_id });
    }

  if (event_count > 0)
    {
      redundancy_mean /= event_count;
    }

  telemetry::LogDebug ("cortext.sensitivity_feedback",
                       { telemetry::Attribute::Double ("novelty_reward",
                                                       (event_count > 0)
                                                           ? (1.0 - redundancy_mean)
                                                           : 0.0),
                         telemetry::Attribute::Double ("redundancy",
                                                       redundancy_mean),
                         telemetry::Attribute::Double ("weight_novelty_delta",
                                                       weight_novelty_delta) });
}

} // namespace cortext::operations
