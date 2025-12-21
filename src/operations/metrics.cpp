#include "cortext/operations/metrics.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cmath>
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

} // namespace

void
ComputeMetrics::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &x = context.GetSignal ().embedding;
  const double F = cfg.focus;
  const double S = cfg.sensitivity;
  const double T = cfg.stability;

  const int n_ctx_total
      = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  const int ctx_window = static_cast<int> (core::NCtx (T));
  const int ctx_start = std::max (0, n_ctx_total - ctx_window);

  Eigen::VectorXf mean_ctx = Eigen::VectorXf::Zero (x.size ());
  double max_cos = -1.0;
  double sum_cos = 0.0;
  int count = 0;
  for (int i = ctx_start; i < n_ctx_total; ++i)
    {
      const auto &emb
          = p_ctx.recent_context_embeddings[static_cast<size_t> (i)];
      if (emb.size () != x.size () || x.size () == 0)
        {
          continue;
        }
      mean_ctx += emb;
      const double c = core::CosineSimilarity (x, emb);
      max_cos = std::max (max_cos, c);
      sum_cos += c;
      ++count;
    }

  double cos01 = 0.5;
  if (count > 0)
    {
      mean_ctx /= static_cast<float> (count);
      const double cos_mean = core::CosineSimilarity (x, mean_ctx);
      cos01 = core::Map01 (cos_mean);
    }
  const double relevance = core::Clamp (
      cos01 * (constants::kOneHalf + constants::kOneHalf * F),
      constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::relevance, relevance);

  const double novelty = (count > 0)
                             ? core::Clamp (
                                   (constants::kNormalizedMax - max_cos)
                                       * constants::kOneHalf,
                                   constants::kNormalizedMin,
                                   constants::kNormalizedMax)
                             : constants::kNormalizedMax;
  const double mean_cos = (count > 0) ? (sum_cos / static_cast<double> (count))
                                      : 0.0;
  const double mu_sim = (count > 0)
                            ? core::Clamp (
                                  constants::kOneHalf
                                      * (mean_cos + constants::kNormalizedMax),
                                  constants::kNormalizedMin,
                                  constants::kNormalizedMax)
                            : constants::kOneHalf;
  const double rarity_base
      = core::Clamp (constants::kNormalizedMax - mu_sim,
                     constants::kNormalizedMin, constants::kNormalizedMax);

  // Mismatch: (1 − F) × S × novelty
  const double mismatch = core::Clamp (
      (constants::kNormalizedMax - F) * S * novelty,
      constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::mismatch, mismatch);

  // Prediction Error: embedding_surprisal × S × (1 − T)
  // Per algorithms.md Section 3.2 Table 1 row "Prediction Error"
  // Use embedding_surprisal from EmbeddingPredictionError operation
  const double embedding_surprisal
      = context.GetMetric (operations::Metric::embedding_surprisal)
            .value_or (constants::kNormalizedMin);
  const double surprise = core::Clamp (
      embedding_surprisal * S * (constants::kNormalizedMax - T),
      constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::surprise, surprise);

  // Rarity: rarity_t × (0.5 + 0.5F) × (1 − 0.2T)
  const double rarity
      = core::Clamp (
          rarity_base
              * (constants::kOneHalf + constants::kOneHalf * F)
              * (constants::kNormalizedMax - constants::kRarityTCoeff * T),
          constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::rarity, rarity);

  // Drift: lagged centroid drift with k_ctx(T) step lag
  const int k_ctx = core::KCtx (T);
  double drift_mag = 0.0;
  if (n_ctx_total >= ctx_window + k_ctx && x.size () > 0)
    {
      const Eigen::VectorXf mean_recent = ComputeMean (
          p_ctx.recent_context_embeddings, n_ctx_total - ctx_window,
          n_ctx_total);
      const Eigen::VectorXf mean_prev
          = ComputeMean (p_ctx.recent_context_embeddings,
                         n_ctx_total - k_ctx - ctx_window,
                         n_ctx_total - k_ctx);
      if (mean_recent.size () == mean_prev.size ()
          && mean_recent.size () > 0)
        {
          const Eigen::VectorXf nr
              = (mean_recent.norm () > 0.0f)
                    ? (mean_recent / mean_recent.norm ())
                    : mean_recent;
          const Eigen::VectorXf np = (mean_prev.norm () > 0.0f)
                                         ? (mean_prev / mean_prev.norm ())
                                         : mean_prev;
          drift_mag = (nr - np).norm ();
        }
    }
  context.SetMetric (operations::Metric::drift_mag, drift_mag);
  const double drift = core::Clamp (
      (drift_mag * constants::kOneHalf)
          * (constants::kNormalizedMax - T),
      constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::drift, drift);

  // Contradiction: max(0, S − F)
  const double contradiction
      = std::max (constants::kNormalizedMin, S - F);
  context.SetMetric (operations::Metric::contradiction, contradiction);

  // Utility (ΔSSE): normalized improvement in prediction error
  double delta_sse = 0.0;
  if (p_ctx.prediction_error_sse.has_value ()
      && p_ctx.prediction_error_sse_prev.has_value ())
    {
      const double sse_prev = *p_ctx.prediction_error_sse_prev;
      const double sse_curr = *p_ctx.prediction_error_sse;
      const double denom = std::max (sse_prev, 1e-9);
      if (sse_prev > 0.0)
        {
          delta_sse = core::Clamp ((sse_prev - sse_curr) / denom,
                                   constants::kNormalizedMin,
                                   constants::kNormalizedMax);
        }
    }
  const double utility
      = core::Clamp (delta_sse
                         * (constants::kOneHalf + constants::kOneHalf * F)
                         * (constants::kNormalizedMax - constants::kUtilitySCoeff * S),
                     constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::utility, utility);

  // Periphery: ↑T; approximate as (1 − relevance) × T
  const double periphery = core::Clamp (
      (constants::kNormalizedMax - relevance) * T,
      constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::periphery, periphery);

  // Coverage: ↑F; approximate as F × max(0, relevance − baseline).
  // No DB baseline available → baseline 0.0.
  const double coverage = core::Clamp (F * relevance,
                                       constants::kNormalizedMin,
                                       constants::kNormalizedMax);
  context.SetMetric (operations::Metric::coverage, coverage);

  // Salience: (rarity_t + novelty_t)/2 × (F + S)/2
  const double salience
      = core::Clamp (constants::kOneHalf * (rarity_base + novelty)
                         * constants::kOneHalf * (F + S),
                     constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::salience, salience);

  // Valence/Arousal (prefer affect centroids when available)
  double valence = core::Clamp (context.GetValence (), constants::kNormalizedMin,
                                constants::kNormalizedMax);
  double arousal = core::Clamp (context.GetArousal (), constants::kNormalizedMin,
                                constants::kNormalizedMax);
  if (p_ctx.centroids.has_value () && x.size () == 256)
    {
      const float v_signed = p_ctx.centroids->affect.ComputeValence (x); // [-1,1]
      valence = core::Clamp (constants::kOneHalf
                                 * (static_cast<double> (v_signed)
                                    + constants::kNormalizedMax),
                             constants::kNormalizedMin,
                             constants::kNormalizedMax);
      const float a01 = p_ctx.centroids->affect.ComputeArousal (x); // [0,1]
      arousal = core::Clamp (static_cast<double> (a01),
                             constants::kNormalizedMin,
                             constants::kNormalizedMax);
      const float viol01 = p_ctx.centroids->affect.ComputeViolation (x); // [0,1]
      context.SetViolation (
          core::Clamp (static_cast<double> (viol01), constants::kNormalizedMin,
                       constants::kNormalizedMax));
    }
  context.SetMetric (operations::Metric::valence, valence);
  context.SetMetric (operations::Metric::arousal, arousal);

  telemetry::LogDebug("cortext.metrics", {
    telemetry::Attribute::Double("relevance", relevance),
    telemetry::Attribute::Double("mismatch", mismatch),
    telemetry::Attribute::Double("surprise", surprise),
    telemetry::Attribute::Double("rarity", rarity),
    telemetry::Attribute::Double("drift", drift),
    telemetry::Attribute::Double("contradiction", contradiction),
    telemetry::Attribute::Double("utility", utility),
    telemetry::Attribute::Double("periphery", periphery),
    telemetry::Attribute::Double("coverage", coverage),
    telemetry::Attribute::Double("salience", salience),
    telemetry::Attribute::Double("valence", valence),
    telemetry::Attribute::Double("arousal", arousal)
  });
}

} // namespace cortext::operations
