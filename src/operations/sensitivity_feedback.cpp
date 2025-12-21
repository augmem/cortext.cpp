#include "cortext/operations/sensitivity_feedback.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <deque>
#include <vector>

namespace cortext::operations
{
inline double
ComputeNovelty (const std::deque<Eigen::VectorXf> &embs,
                const Eigen::VectorXf &x, int window)
{
  if (x.size () == 0)
    {
      return constants::kNormalizedMax;
    }
  const int n = static_cast<int> (embs.size ());
  if (n == 0)
    {
      return constants::kNormalizedMax;
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
      return constants::kNormalizedMax;
    }
  return core::Clamp ((constants::kNormalizedMax - max_cos)
                          * constants::kOneHalf,
                      constants::kNormalizedMin, constants::kNormalizedMax);
}

/// @brief Compute redundancy for a memory via kNN similarity lookup.
/// @param store The store to query (may be null).
/// @param embedding_id The memory's embedding row ID.
/// @param k Number of neighbors to consider.
/// @return Redundancy in [0,1]: 1 - mean similarity to kNN neighbors.
inline double
ComputeRedundancy (Store *store, int64_t embedding_id, int k)
{
  if (!store || embedding_id <= 0 || k <= 0)
    {
      return constants::kNormalizedMin;
    }

  // Query kNN neighbors for this embedding (k+1 to exclude self at distance 0)
  auto rows = store->Execute (
      "SELECT distance FROM embeddings "
      "WHERE embedding MATCH (SELECT embedding FROM embeddings WHERE embedding_id = ?) "
      "ORDER BY distance LIMIT ?",
      { embedding_id, static_cast<long long> (k + 1) });

  if (rows.size () <= 1)
    {
      // No neighbors or only self returned
      return constants::kNormalizedMin;
    }

  double sum_sim = 0.0;
  int count = 0;
  for (size_t i = 1; i < rows.size (); ++i)
    { // Skip self (i=0)
      auto it = rows[i].find ("distance");
      if (it != rows[i].end () && it->second.type () == typeid (double))
        {
          double dist = std::any_cast<double> (it->second);
          sum_sim += 1.0 / (1.0 + dist); // L2 distance → similarity
          ++count;
        }
    }

  if (count == 0)
    {
      return constants::kNormalizedMin;
    }
  const double mean_sim = sum_sim / count;
  return 1.0 - mean_sim; // redundancy = 1 - mean similarity
}

void
ApplySensitivityFeedback::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();

  const double eta = constants::kEtaBase;

  const auto &x = context.GetSignal ().embedding;
  const int window = static_cast<int> (core::NCtx (context.GetConfig ().stability));
  const double novelty_from_metric
      = ComputeNovelty (p_ctx.recent_context_embeddings, x, window);

  // Algorithm 16: Compute per-memory redundancy from kNN neighbors.
  const auto &cfg = context.GetConfig ();
  Store *store = context.GetStore ();
  const int k = core::KNeighbors (cfg.stability);

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
      const double cg = *e.contextual_gain; // may be negative
      const double redundancy
          = ComputeRedundancy (store, e.embedding_id, k);
      redundancy_mean += redundancy;
      ++event_count;

      const double prev_weight = p_ctx.weight_novelty;
      const double adjustment = eta * (novelty_from_metric * cg - redundancy);
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
                                                       novelty_from_metric),
                         telemetry::Attribute::Double ("redundancy",
                                                       redundancy_mean),
                         telemetry::Attribute::Double ("weight_novelty_delta",
                                                       weight_novelty_delta) });
}

} // namespace cortext::operations
