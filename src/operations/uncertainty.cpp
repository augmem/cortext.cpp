#include "cortext/operations/uncertainty.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace cortext::operations
{

void
UpdateUncertainty::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();

  // --- Primary structural estimator (algorithms.md §0.4) ---
  // u_raw = normalize_weighted_blend(
  //   [var_recent_scores, focus_spread_entropy, coherence_complement,
  //   novelty_surprise_spikes], weights = normalize([F, S, 1 − T, S])
  // )

  // Helper: variance of last w scores in [0,1]
  auto compute_scores_variance = [&] () -> std::optional<double> {
    const int w = core::WScore (config.stability);
    const int n = static_cast<int> (p_ctx.recent_scores.size ());
    if (n < 2)
      {
        return std::nullopt;
      }
    const int start = std::max (0, n - w);
    const int m = n - start;
    if (m < 2)
      {
        return std::nullopt;
      }
    double sum = 0.0;
    for (int i = start; i < n; ++i)
      {
        sum += core::Clamp (p_ctx.recent_scores[static_cast<size_t> (i)],
                            constants::kNormalizedMin, constants::kNormalizedMax);
      }
    const double mean = sum / static_cast<double> (m);
    double accum = 0.0;
    for (int i = start; i < n; ++i)
      {
        const double v = core::Clamp (
            p_ctx.recent_scores[static_cast<size_t> (i)],
            constants::kNormalizedMin, constants::kNormalizedMax);
        const double d = v - mean;
        accum += d * d;
      }
    const double var = accum / static_cast<double> (m);
    return core::Clamp (var, constants::kNormalizedMin,
                        constants::kNormalizedMax);
  };

  // Helper: focus_spread_entropy via softmax over similarities to recent
  // context
  auto compute_focus_spread_entropy = [&] () -> std::optional<double> {
    const auto &x = context.GetSignal ().embedding;
    const int n = static_cast<int> (p_ctx.recent_context_embeddings.size ());
    if (n < 2)
      {
        return std::nullopt;
      }
    const int k = std::min (core::KNeighbors (config.stability), n);
    const int start = n - k;
    std::vector<double> sims;
    sims.reserve (static_cast<size_t> (k));
    for (int i = start; i < n; ++i)
      {
        const auto &emb
            = p_ctx.recent_context_embeddings[static_cast<size_t> (i)];
        sims.push_back (core::CosineSimilarity (x, emb));
      }
    // Softmax with stability: subtract max for numerical stability
    const double max_s = *std::max_element (sims.begin (), sims.end ());
    double denom = 0.0;
    for (double &s : sims)
      {
        s = std::exp (s - max_s);
        denom += s;
      }
    if (denom <= 0.0)
      {
        return std::nullopt;
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
        return std::nullopt;
      }
    return core::Clamp (entropy / norm, 0.0, 1.0);
  };

  // Helper: quick coherence (Algorithm 10) to avoid ordering dependency
  auto compute_coherence_complement = [&] () -> std::optional<double> {
    const auto &x = context.GetSignal ().embedding;
    const int window = static_cast<int> (core::NCtx (config.stability));
    const int n = static_cast<int> (p_ctx.recent_context_embeddings.size ());
    if (n < 2)
      {
        return std::nullopt;
      }
    const int start = std::max (0, n - window);
    std::vector<double> vals;
    vals.reserve (static_cast<size_t> (n - start));
    for (int i = start; i < n; ++i)
      {
        const auto &emb
            = p_ctx.recent_context_embeddings[static_cast<size_t> (i)];
        vals.push_back (core::CosineSimilarity (x, emb));
      }
    if (vals.size () < 2)
      {
        return std::nullopt;
      }
    double sum = std::accumulate (vals.begin (), vals.end (), 0.0);
    const double mean = sum / static_cast<double> (vals.size ());
    double accum = 0.0;
    for (double v : vals)
      {
        const double d = v - mean;
        accum += d * d;
      }
    const double var = accum / static_cast<double> (vals.size ());
    const double coherence
        = constants::kNormalizedMax
          - core::Clamp (var, constants::kNormalizedMin,
                         constants::kNormalizedMax);
    const double cc = constants::kNormalizedMax - coherence;
    return core::Clamp (cc, constants::kNormalizedMin,
                        constants::kNormalizedMax);
  };

  // Helper: novelty measure as distance from mean of recent context embeddings
  // (contributes to Algorithm 4 novelty component)
  auto compute_novelty_measure = [&] () -> std::optional<double> {
    const auto &x = context.GetSignal ().embedding;
    if (x.size () == 0)
      {
        return std::nullopt;
      }
    const int n = static_cast<int> (p_ctx.recent_context_embeddings.size ());
    if (n < 2)
      {
        return std::nullopt;
      }

    // Compute mean of recent context embeddings
    Eigen::VectorXf mean_ctx = Eigen::VectorXf::Zero (x.size ());
    int valid_count = 0;
    for (const auto &emb : p_ctx.recent_context_embeddings)
      {
        if (emb.size () == x.size ())
          {
            mean_ctx += emb;
            ++valid_count;
          }
      }
    if (valid_count < 2)
      {
        return std::nullopt;
      }
    mean_ctx /= static_cast<float> (valid_count);

    // Novelty = 1 - similarity to mean context
    // High novelty when current signal is dissimilar to recent context
    const double sim = core::CosineSimilarity (x, mean_ctx);
    // Map cosine similarity from [-1, 1] to [0, 1]
    const double sim_01
        = core::Clamp ((sim + 1.0) / 2.0, constants::kNormalizedMin,
                       constants::kNormalizedMax);
    // Novelty is inverse of similarity
    return constants::kNormalizedMax - sim_01;
  };

  const auto var_scores = compute_scores_variance ();
  const auto focus_entropy = compute_focus_spread_entropy ();
  const auto coh_complement = compute_coherence_complement ();
  const auto novelty_measure = compute_novelty_measure ();

  // Fuse novelty (Alg 4) + surprise (Alg 13) into novelty_surprise_spikes
  // per algorithms.md §0.4: fusion of Algorithm 4 novelty + Algorithm 13
  // surprisal
  const double surprise_val
      = context.GetMetric (operations::Metric::surprise).value_or (0.0);
  double novelty_surprise_spikes = 0.0;
  bool has_novelty_surprise = false;
  if (novelty_measure.has_value () && surprise_val > 0.0)
    {
      // Average of novelty and surprise when both available
      novelty_surprise_spikes
          = core::Clamp ((novelty_measure.value () + surprise_val) / 2.0,
                         constants::kNormalizedMin, constants::kNormalizedMax);
      has_novelty_surprise = true;
    }
  else if (novelty_measure.has_value ())
    {
      novelty_surprise_spikes = novelty_measure.value ();
      has_novelty_surprise = true;
    }
  else if (surprise_val > 0.0)
    {
      novelty_surprise_spikes = surprise_val;
      has_novelty_surprise = true;
    }

  // Collect available metrics and corresponding weights.
  std::vector<double> metrics;
  std::vector<double> weights;
  if (var_scores.has_value ())
    {
      metrics.push_back (*var_scores);
      // Weight by Focus (F) per §0.4
      weights.push_back (config.focus);
    }
  if (focus_entropy.has_value ())
    {
      metrics.push_back (*focus_entropy);
      // Weight by Sensitivity (S) per §0.4
      weights.push_back (config.sensitivity);
    }
  if (coh_complement.has_value ())
    {
      metrics.push_back (*coh_complement);
      // Tie complement coherence to Stability inverse.
      weights.push_back (1.0 - config.stability);
    }
  // Include novelty_surprise_spikes (fusion of Alg 4 + Alg 13) when available.
  // Weight with Sensitivity (S) per §0.4.
  if (has_novelty_surprise)
    {
      metrics.push_back (novelty_surprise_spikes);
      weights.push_back (config.sensitivity);
    }

  double u_raw = 0.0;
  bool used_primary = false;
  if (!metrics.empty ())
    {
      // Normalize weights and compute weighted average
      double wsum = 0.0;
      for (double w : weights)
        {
          wsum += std::max (w, 0.0);
        }
      if (wsum > 0.0)
        {
          for (double &w : weights)
            {
              w = std::max (w, 0.0) / wsum;
            }
          for (size_t i = 0; i < metrics.size (); ++i)
            {
              u_raw += weights[i] * core::Clamp (metrics[i], 0.0, 1.0);
            }
          used_primary = true;
        }
    }

  if (!used_primary)
    {
      // --- Fallback Uncertainty (from Section 0.4) ---
      // u_raw(t) = 1 − maturity(t)
      const double count = p_ctx.signals_processed;
      const double tau_m = core::TauM (config.stability);
      const double maturity = 1.0 - std::exp (-count / tau_m);
      u_raw = 1.0 - maturity;
    }

  // --- Use Fallback Uncertainty (from Section 0.4) ---
  // --- Calculate Smoothed Uncertainty (from Section 0.3) ---
  // u(t) = EWMA(u(t−1), u_raw(t), α = α_u(T))
  const double alpha_u = core::AlphaU (config.stability);
  p_ctx.u_t = core::Ewma (p_ctx.u_t, u_raw, alpha_u);
}

} // namespace cortext::operations
