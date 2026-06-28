#include "cortext/operations/sensitivity_feedback.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <deque>
#include <unordered_map>
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
  const auto &cfg = context.GetConfig ();

  const double eta = core::SensitivityFeedbackNoveltyGain (
      cfg.focus, cfg.sensitivity, cfg.stability);

  const int window = static_cast<int> (core::NCtx (cfg.stability));
  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  const auto &retrieved_records = context.GetRetrievedMemoryCandidates ();
  std::unordered_map<long long, const Eigen::VectorXf *> retrieved_by_memory;
  if (!retrieved_records.empty ())
    {
      retrieved_by_memory.reserve (retrieved_records.size ());
      for (const auto &candidate : retrieved_records)
        {
          if (candidate.memory_id > 0 && candidate.embedding.size () > 0)
            {
              retrieved_by_memory[candidate.memory_id] = &candidate.embedding;
            }
        }
    }

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
      const Eigen::VectorXf *retrieved_embedding = nullptr;
      if (!retrieved_by_memory.empty () && e.memory_id > 0)
        {
          auto it = retrieved_by_memory.find (e.memory_id);
          if (it != retrieved_by_memory.end ())
            {
              retrieved_embedding = it->second;
            }
        }
      else
        {
          auto it = retrieved.find (e.embedding_id);
          if (it != retrieved.end ())
            {
              retrieved_embedding = &it->second;
            }
        }
      if (!retrieved_embedding)
        {
          continue;
        }
      const double cg = *e.contextual_gain; // may be negative
      const double redundancy = ComputeRedundancyToContext (
          p_ctx.recent_context_embeddings, *retrieved_embedding, window);
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
      if (e.memory_id > 0)
        {
          tx.Execute ("UPDATE memories SET redundancy = ? WHERE memory_id = ?",
                      { redundancy, e.memory_id });
        }
      else
        {
          tx.Execute ("UPDATE memories SET redundancy = ? WHERE embedding_id = ?",
                      { redundancy, e.embedding_id });
        }
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
