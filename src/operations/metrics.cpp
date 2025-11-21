#include "cortext/operations/metrics.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace cortext::operations
{
namespace
{

inline Eigen::VectorXf
ComputeMean (const std::deque<Eigen::VectorXf> &embs, int start, int end)
{
  if (start >= end)
    {
      return Eigen::VectorXf ();
    }
  const int n = end - start;
  const int dim = static_cast<int> (embs[static_cast<size_t> (start)].size ());
  Eigen::VectorXf mean = Eigen::VectorXf::Zero (dim);
  for (int i = start; i < end; ++i)
    {
      mean += embs[static_cast<size_t> (i)];
    }
  mean /= static_cast<float> (n);
  return mean;
}

inline double
CosSim01 (const Eigen::VectorXf &a, const Eigen::VectorXf &b)
{
  const double c = core::CosineSimilarity (a, b); // [-1,1]
  return core::Clamp (constants::kOneHalf * (c + constants::kNormalizedMax),
                      constants::kNormalizedMin, constants::kNormalizedMax);
}

inline double
VarRecentScores01 (const std::deque<double> &scores, int window)
{
  const int n = static_cast<int> (scores.size ());
  if (n < 2)
    {
      return 0.0;
    }
  const int start = std::max (0, n - window);
  const int m = n - start;
  if (m < 2)
    {
      return 0.0;
    }
  double sum = 0.0;
  for (int i = start; i < n; ++i)
    {
      sum += core::Clamp (scores[static_cast<size_t> (i)],
                          constants::kNormalizedMin, constants::kNormalizedMax);
    }
  const double mean = sum / static_cast<double> (m);
  double accum = 0.0;
  for (int i = start; i < n; ++i)
    {
      const double v = core::Clamp (scores[static_cast<size_t> (i)],
                                    constants::kNormalizedMin,
                                    constants::kNormalizedMax);
      const double d = v - mean;
      accum += d * d;
    }
  return core::Clamp (accum / static_cast<double> (m), constants::kNormalizedMin,
                      constants::kNormalizedMax);
}

} // namespace

void
ComputeMetrics::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &x = context.GetSignal ().embedding;
  const double F = cfg.focus;
  const double F_eff_in = context.GetEffectiveFocus ();
  const double F_eff = (F_eff_in > constants::kNormalizedMin) ? F_eff_in : F;
  const double S = cfg.sensitivity;
  const double T = cfg.stability;

  // Relevance: cos(x, mean(ctx)) mapped to [0,1], scaled by (0.5 + F)
  double relevance = constants::kNormalizedMin;
  if (!p_ctx.recent_context_embeddings.empty ())
    {
      const Eigen::VectorXf mean_ctx = ComputeMean (
          p_ctx.recent_context_embeddings, 0,
          static_cast<int> (p_ctx.recent_context_embeddings.size ()));
      const double cos01 = CosSim01 (x, mean_ctx);
      relevance = core::Clamp (cos01 * (constants::kOneHalf + F_eff),
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);
    }
  context.SetMetric (operations::Metric::relevance, relevance);

  // Novelty approximate: 1 - relevance
  const double novelty
      = core::Clamp (constants::kNormalizedMax - relevance,
                     constants::kNormalizedMin, constants::kNormalizedMax);

  // Mismatch: (1 − F) × S × novelty
  const double mismatch = core::Clamp (
      (constants::kNormalizedMax - F_eff) * S * novelty,
      constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::mismatch, mismatch);

  // Surprise: var recent scores × S × (1 − T)
  const int w = core::WScore (T);
  const double var_scores = VarRecentScores01 (p_ctx.recent_scores, w);
  const double surprise = core::Clamp (
      var_scores * S * (constants::kNormalizedMax - T),
      constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::surprise, surprise);

  // Rarity: (1 − mean_sim) × (0.5 + 0.5F) × (1 − 0.2T)
  const double rarity
      = core::Clamp (
          (constants::kNormalizedMax - relevance)
              * (constants::kOneHalf + constants::kOneHalf * F_eff)
              * (1.0 - 0.2 * T),
          constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::rarity, rarity);

  // Drift: ||mean(ctx_recent) − mean(ctx_prev)|| mapped to [0,1], × (1 − T)
  double drift = 0.0;
  const int n_ctx = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  if (n_ctx >= 4)
    {
      const int k = std::min (core::KNeighbors (T), n_ctx / 2);
      const int split = n_ctx - k;
      if (k > 0 && split >= k)
        {
          const Eigen::VectorXf mean_recent = ComputeMean (
              p_ctx.recent_context_embeddings, n_ctx - k, n_ctx);
          const Eigen::VectorXf mean_prev = ComputeMean (
              p_ctx.recent_context_embeddings, split - k, split);
          if (mean_recent.size () == mean_prev.size ()
              && mean_prev.size () > 0)
            {
              const Eigen::VectorXf nr
                  = (mean_recent.norm () > 0.0f)
                        ? (mean_recent / mean_recent.norm ())
                        : mean_recent;
              const Eigen::VectorXf np = (mean_prev.norm () > 0.0f)
                                             ? (mean_prev / mean_prev.norm ())
                                             : mean_prev;
              const double d = (nr - np).norm (); // ∈ [0,2]
              const double d01 = core::Clamp (constants::kOneHalf * d,
                                              constants::kNormalizedMin,
                                              constants::kNormalizedMax);
              drift = core::Clamp (d01 * (constants::kNormalizedMax - T),
                                   constants::kNormalizedMin,
                                   constants::kNormalizedMax);
            }
        }
    }
  context.SetMetric (operations::Metric::drift, drift);

  // Contradiction: max(0, S − F)
  const double contradiction
      = std::max (constants::kNormalizedMin, S - F_eff);
  context.SetMetric (operations::Metric::contradiction, contradiction);

  // Utility (Δ proxy): positive deviation from mean recent score,
  // scaled per table: × (0.5 + 0.5F) × (1 − 0.3S)
  double mean_recent_score = constants::kNormalizedMin;
  if (!p_ctx.recent_scores.empty ())
    {
      mean_recent_score = std::accumulate (p_ctx.recent_scores.begin (),
                                           p_ctx.recent_scores.end (), 0.0)
                          / static_cast<double> (p_ctx.recent_scores.size ());
      mean_recent_score = core::Clamp (mean_recent_score,
                                       constants::kNormalizedMin,
                                       constants::kNormalizedMax);
    }
  const double delta_score = std::max (constants::kNormalizedMin,
                                       relevance - mean_recent_score);
  const double utility
      = core::Clamp (delta_score
                         * (constants::kOneHalf + constants::kOneHalf * F_eff)
                         * (1.0 - 0.3 * S),
                     constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::utility, utility);

  // Periphery: ↑T; approximate as (1 − relevance) × T
  const double periphery = core::Clamp (
      (constants::kNormalizedMax - relevance) * T,
      constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::periphery, periphery);

  // Coverage: ↑F; approximate as F × max(0, relevance − baseline).
  // No DB baseline available → baseline 0.0.
  const double coverage = core::Clamp (F_eff * relevance,
                                       constants::kNormalizedMin,
                                       constants::kNormalizedMax);
  context.SetMetric (operations::Metric::coverage, coverage);

  // Salience: (rarity + novelty_recent)/2 × (F + S)/2
  const double salience
      = core::Clamp (constants::kOneHalf * (rarity + novelty)
                         * constants::kOneHalf * (F_eff + S),
                     constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::salience, salience);

  // Valence/Arousal from sensitivity op outputs (already in [0,1])
  const double valence
      = core::Clamp (context.GetValence (), constants::kNormalizedMin,
                     constants::kNormalizedMax);
  const double arousal
      = core::Clamp (context.GetArousal (), constants::kNormalizedMin,
                     constants::kNormalizedMax);
  context.SetMetric (operations::Metric::valence, valence);
  context.SetMetric (operations::Metric::arousal, arousal);
}

} // namespace cortext::operations
