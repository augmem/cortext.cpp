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
void
ComputeMetrics::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &signal = context.GetSignal ();
  const Eigen::VectorXf *x_ptr = &signal.embedding;
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it != p_ctx.accumulator_states.end ()
      && acc_it->second.mu_acc.size () > 0)
    {
      x_ptr = &acc_it->second.mu_acc;
    }
  const auto &x = *x_ptr;
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
  double cos_mean = 0.0;
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
      cos_mean = core::CosineSimilarity (x, mean_ctx);
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

  // Prediction Error: embedding_surprisal × S × (1 − 0.5T)
  // Per algorithms.md Section 3.2 Table 1 row "Prediction Error"
  // Use embedding_surprisal from EmbeddingPredictionError operation
  const double embedding_surprisal
      = context.GetMetric (operations::Metric::embedding_surprisal)
            .value_or (constants::kNormalizedMin);
  const double surprise = core::Clamp (
      embedding_surprisal * S
          * (constants::kNormalizedMax - constants::kOneHalf * T),
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

  // Drift: instantaneous step based on consecutive signals (Section 3.1.2)
  const double drift_mag = context.GetAccumulatorDriftStep ();
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
  double sse_prev = -1.0;
  double sse_curr = -1.0;
  if (p_ctx.prediction_error_sse.has_value ()
      && p_ctx.prediction_error_sse_prev.has_value ())
    {
      sse_prev = *p_ctx.prediction_error_sse_prev;
      sse_curr = *p_ctx.prediction_error_sse;
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
  double v_signed = -1.0;
  double a01 = -1.0;
  double viol01 = -1.0;
  const bool has_centroids = p_ctx.centroids.has_value ();
  if (p_ctx.centroids.has_value () && x.size () == 256)
    {
      v_signed = static_cast<double> (p_ctx.centroids->affect.ComputeValence (x)); // [-1,1]
      valence = core::Clamp (constants::kOneHalf
                                 * (v_signed
                                    + constants::kNormalizedMax),
                             constants::kNormalizedMin,
                             constants::kNormalizedMax);
      a01 = static_cast<double> (p_ctx.centroids->affect.ComputeArousal (x)); // [0,1]
      arousal = core::Clamp (a01,
                             constants::kNormalizedMin,
                             constants::kNormalizedMax);
      viol01 = static_cast<double> (p_ctx.centroids->affect.ComputeViolation (x)); // [0,1]
      context.SetViolation (
          core::Clamp (viol01, constants::kNormalizedMin,
                       constants::kNormalizedMax));
    }
  context.SetMetric (operations::Metric::valence, valence);
  context.SetMetric (operations::Metric::arousal, arousal);

  telemetry::LogDebug("cortext.metrics", {
    telemetry::Attribute::Double("F", F),
    telemetry::Attribute::Double("S", S),
    telemetry::Attribute::Double("T", T),
    telemetry::Attribute::Int64("ctx_window", static_cast<int64_t> (ctx_window)),
    telemetry::Attribute::Int64("ctx_start", static_cast<int64_t> (ctx_start)),
    telemetry::Attribute::Int64("n_ctx_total", static_cast<int64_t> (n_ctx_total)),
    telemetry::Attribute::Int64("ctx_count", static_cast<int64_t> (count)),
    telemetry::Attribute::Double("max_cos", max_cos),
    telemetry::Attribute::Double("mean_cos", mean_cos),
    telemetry::Attribute::Double("cos01", cos01),
    telemetry::Attribute::Double("novelty", novelty),
    telemetry::Attribute::Double("rarity_base", rarity_base),
    telemetry::Attribute::Double("mu_sim", mu_sim),
    telemetry::Attribute::Double("embedding_surprisal", embedding_surprisal),
    telemetry::Attribute::Double("drift_mag", drift_mag),
    telemetry::Attribute::Double("delta_sse", delta_sse),
    telemetry::Attribute::Double("sse_prev", sse_prev),
    telemetry::Attribute::Double("sse_curr", sse_curr),
    telemetry::Attribute::Bool("has_centroids", has_centroids),
    telemetry::Attribute::Double("v_signed", v_signed),
    telemetry::Attribute::Double("a01", a01),
    telemetry::Attribute::Double("viol01", viol01),
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
