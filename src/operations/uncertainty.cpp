#include "cortext/operations/uncertainty.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

namespace cortext::operations
{

namespace
{
// Spec (§0.4, line 139): var_score_max = 0.25
constexpr double kVarScoreMax = 0.25;
constexpr double kWeightEpsilon = 1e-9;
}  // namespace

void
UpdateUncertainty::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();

  // --- Primary structural estimator (algorithms.md §0.4) ---
  // u_raw = normalize_weighted_blend(
  //   [var_recent_norm, focus_spread, coherence_complement, novelty_surprise],
  //   weights = normalize([S, F, 1 − T, S × (1 − T)])
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
    // Spec (§0.4, line 141): var_recent_norm = clamp(var / var_score_max, 0, 1)
    return core::Clamp (var / kVarScoreMax, constants::kNormalizedMin,
                        constants::kNormalizedMax);
  };

  // Helper: focus_spread_entropy via softmax over similarities to memory_stream
  auto compute_focus_spread_entropy = [&] () -> std::optional<double> {
    if (auto v = context.GetMetric (operations::Metric::focus_spread))
      {
        return *v;
      }
    const auto &x = context.GetSignal ().embedding;
    const int n = static_cast<int> (p_ctx.memory_stream.size ());
    if (n < 2)
      {
        return std::nullopt;
      }
    const int k = std::min (core::KNeighbors (config.stability), n);
    std::vector<double> sims;
    sims.reserve (static_cast<size_t> (n));
    for (int i = 0; i < n; ++i)
      {
        const auto &emb
            = p_ctx.memory_stream[static_cast<size_t> (i)];
        if (emb.size () == x.size () && x.size () > 0)
          {
            sims.push_back (core::CosineSimilarity (x, emb));
          }
      }
    if (static_cast<int> (sims.size ()) < 2)
      {
        return std::nullopt;
      }
    const int k_eff = std::min (k, static_cast<int> (sims.size ()));
    if (k_eff < 2)
      {
        return std::nullopt;
      }
    if (static_cast<int> (sims.size ()) > k_eff)
      {
        std::nth_element (sims.begin (), sims.begin () + k_eff, sims.end (),
                          std::greater<double> ());
        sims.resize (static_cast<size_t> (k_eff));
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
    const double norm = std::log (static_cast<double> (k_eff));
    if (norm <= 0.0)
      {
        return std::nullopt;
      }
    return core::Clamp (entropy / norm, 0.0, 1.0);
  };

  // Helper: novelty measure as dissimilarity to recent context (Appendix B)
  auto compute_novelty_measure = [&] () -> std::optional<double> {
    const auto &x = context.GetSignal ().embedding;
    if (x.size () == 0)
      {
        return std::nullopt;
      }
    const int n = static_cast<int> (p_ctx.recent_context_embeddings.size ());
    if (n == 0)
      {
        return constants::kNormalizedMax;
      }
    const int window = static_cast<int> (core::NCtx (config.stability));
    const int start = std::max (0, n - window);
    double max_cos = -1.0;
    int count = 0;
    for (int i = start; i < n; ++i)
      {
        const auto &emb
            = p_ctx.recent_context_embeddings[static_cast<size_t> (i)];
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
  };

  const auto var_scores = compute_scores_variance ();
  const auto focus_entropy = compute_focus_spread_entropy ();
  const double coherence_struct = core::Clamp (context.GetStructuralCoherence (),
                                               constants::kNormalizedMin,
                                               constants::kNormalizedMax);
  const double coh_complement = core::Clamp (1.0 - coherence_struct,
                                             constants::kNormalizedMin,
                                             constants::kNormalizedMax);
  const auto novelty_measure = compute_novelty_measure ();

  // Fuse novelty (Alg 4) + embedding surprisal (Section 3.1.4) into
  // novelty_surprise_spikes per algorithms.md §0.4
  const double embedding_surprisal_val
      = context.GetMetric (operations::Metric::embedding_surprisal)
            .value_or (0.0);
  double novelty_surprise_spikes = 0.0;
  bool has_novelty_surprise = false;
  if (novelty_measure.has_value () && embedding_surprisal_val > 0.0)
    {
      // Weighted blend with weights = normalize([S, 1 − T])
      double w_novelty = config.sensitivity;
      double w_surprise = 1.0 - config.stability;
      const double wsum = std::max (w_novelty, 0.0) + std::max (w_surprise, 0.0);
      if (wsum <= kWeightEpsilon)
        {
          w_novelty = 0.5;
          w_surprise = 0.5;
        }
      else
        {
          w_novelty = std::max (w_novelty, 0.0) / wsum;
          w_surprise = std::max (w_surprise, 0.0) / wsum;
        }
      novelty_surprise_spikes
          = core::Clamp (w_novelty * novelty_measure.value ()
                             + w_surprise * embedding_surprisal_val,
                         constants::kNormalizedMin, constants::kNormalizedMax);
      has_novelty_surprise = true;
    }
  else if (novelty_measure.has_value ())
    {
      novelty_surprise_spikes = novelty_measure.value ();
      has_novelty_surprise = true;
    }
  else if (embedding_surprisal_val > 0.0)
    {
      novelty_surprise_spikes = embedding_surprisal_val;
      has_novelty_surprise = true;
    }

  // Collect available metrics and corresponding weights.
  // Spec (§0.4, line 153): weights_u = normalize([S, F, 1 - T, S × (1 - T)])
  // Metrics order: var_recent_norm, focus_spread, coherence_complement, novelty_surprise
  std::vector<double> metrics;
  std::vector<double> weights;
  if (var_scores.has_value ())
    {
      metrics.push_back (*var_scores);
      // Weight by Sensitivity (S) per §0.4 line 153
      weights.push_back (config.sensitivity);
    }
  if (focus_entropy.has_value ())
    {
      metrics.push_back (*focus_entropy);
      // Weight by Focus (F) per §0.4 line 153
      weights.push_back (config.focus);
    }
  metrics.push_back (coh_complement);
  // Weight by (1 - T) per §0.4 line 153
  weights.push_back (1.0 - config.stability);
  // Include novelty_surprise_spikes (fusion of Alg 4 + Alg 13) when available.
  // Weight with S × (1 - T) per §0.4 line 153.
  if (has_novelty_surprise)
    {
      metrics.push_back (novelty_surprise_spikes);
      weights.push_back (config.sensitivity * (1.0 - config.stability));
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
      if (wsum <= kWeightEpsilon)
        {
          const double uniform = 1.0 / static_cast<double> (weights.size ());
          for (double &w : weights)
            {
              w = uniform;
            }
        }
      else
        {
          for (double &w : weights)
            {
              w = std::max (w, 0.0) / wsum;
            }
        }
      for (size_t i = 0; i < metrics.size (); ++i)
        {
          u_raw += weights[i] * core::Clamp (metrics[i], 0.0, 1.0);
        }
      used_primary = true;
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

  // Compute maturity for telemetry
  const double count = p_ctx.signals_processed;
  const double tau_m = core::TauM (config.stability);
  const double maturity = 1.0 - std::exp (-count / tau_m);

  telemetry::LogDebug("cortext.uncertainty", {
    telemetry::Attribute::Double("maturity", maturity),
    telemetry::Attribute::Double("u_raw", u_raw),
    telemetry::Attribute::Double("u_t", p_ctx.u_t),
    telemetry::Attribute::Double("var_scores", var_scores.value_or(-1.0)),
    telemetry::Attribute::Double("focus_entropy", focus_entropy.value_or(-1.0)),
    telemetry::Attribute::Double("coh_complement", coh_complement),
    telemetry::Attribute::Double("novelty_surprise", has_novelty_surprise ? novelty_surprise_spikes : -1.0),
    telemetry::Attribute::Double("weight_S", config.sensitivity),
    telemetry::Attribute::Double("weight_F", config.focus),
    telemetry::Attribute::Double("weight_1minusT", 1.0 - config.stability),
    telemetry::Attribute::Double("weight_Sx1minusT", config.sensitivity * (1.0 - config.stability)),
    telemetry::Attribute::Bool("used_primary", used_primary)
  });
}

} // namespace cortext::operations
