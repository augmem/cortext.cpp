#include "cortext/operations/blend.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cortext::operations
{
namespace
{
constexpr double kTiny = 1e-6;
constexpr double kLarge = 1000.0;
constexpr double kLamMin = 0.90;
constexpr double kLamMax = 0.99;
constexpr double kLamSlope = 0.09;
constexpr double kTauRlsMin = 20.0;
constexpr double kTauRlsMax = 80.0;

inline const std::vector<operations::Metric> &
SupportedMetrics ()
{
  static const std::vector<operations::Metric> kNames = {
    operations::Metric::relevance, operations::Metric::mismatch,
    operations::Metric::surprise,  operations::Metric::rarity,
    operations::Metric::drift,     operations::Metric::contradiction,
    operations::Metric::utility,   operations::Metric::periphery,
    operations::Metric::coverage,  operations::Metric::salience,
    operations::Metric::valence,   operations::Metric::arousal,
  };
  return kNames;
}

inline double
NormTo01 (double v_norm)
{
  if (std::isnan (v_norm) || std::isinf (v_norm))
    {
      return constants::kNormalizedMin;
    }
  if (v_norm < constants::kNormalizedMin)
    {
      const double mapped
          = constants::kOneHalf * (v_norm + constants::kNormalizedMax);
      return std::min (constants::kNormalizedMax,
                       std::max (constants::kNormalizedMin, mapped));
    }
  return std::min (constants::kNormalizedMax,
                   std::max (constants::kNormalizedMin, v_norm));
}

inline double
BootstrapWeight (operations::Metric name, double F, double S, double T)
{
  double a = 0.5;
  if (name == operations::Metric::relevance)
    {
      a = 0.7 * F + 0.2 * T;
    }
  else if (name == operations::Metric::mismatch)
    {
      a = (1.0 - F) * S;
    }
  else if (name == operations::Metric::surprise)
    {
      a = S * (1.0 - 0.5 * T);
    }
  else if (name == operations::Metric::rarity)
    {
      a = (0.5 + 0.5 * F) * (1.0 - 0.2 * T);
    }
  else if (name == operations::Metric::drift)
    {
      a = 0.5 * (1.0 - T);
    }
  else if (name == operations::Metric::contradiction)
    {
      a = std::max (0.0, S - F);
    }
  else if (name == operations::Metric::utility)
    {
      a = (0.5 + 0.5 * F) * (1.0 - 0.3 * S);
    }
  else if (name == operations::Metric::periphery)
    {
      a = 0.5 * T;
    }
  else if (name == operations::Metric::coverage)
    {
      a = F;
    }
  else if (name == operations::Metric::salience)
    {
      a = 0.5 * (F + S);
    }
  else if (name == operations::Metric::valence)
    {
      a = 0.4 + 0.6 * S * (1.0 - 0.3 * T);
    }
  else if (name == operations::Metric::arousal)
    {
      a = S * (1.0 - 0.2 * T);
    }
  else
    {
      a = 0.5 * (F + S);
    }
  const double z = constants::kTwo * a - 1.0;
  const double w = core::Sigmoid (z);
  return core::Clamp (w, constants::kNormalizedMin, constants::kNormalizedMax);
}

inline void
EnsureStateInitialized (cortext::ProcessorContext &p_ctx,
                        const cortext::SignalProcessor::Config &cfg)
{
  const auto &names = SupportedMetrics ();
  if (p_ctx.blender_order.empty ())
    {
      p_ctx.blender_order = names;
    }
  if (p_ctx.blender_state.empty ())
    {
      for (const auto &n : names)
        {
          p_ctx.blender_state[n]
              = BootstrapWeight (n, cfg.focus, cfg.sensitivity, cfg.stability);
        }
    }
  const std::size_t n = p_ctx.blender_order.size ();
  if (p_ctx.blender_P.size () != n)
    {
      p_ctx.blender_P.assign (n, std::vector<double> (n, constants::kNormalizedMin));
      for (std::size_t i = 0; i < n; ++i)
        {
          p_ctx.blender_P[i][i] = kLarge;
        }
    }
}

std::vector<double>
BuildMetricVector (const std::vector<operations::Metric> &names,
                   const std::unordered_map<operations::Metric, double> &metrics)
{
  std::vector<double> x (names.size (), 0.0);
  for (std::size_t i = 0; i < names.size (); ++i)
    {
      auto it = metrics.find (names[i]);
      const double v_norm = (it == metrics.end ()) ? 0.0 : it->second;
      x[i] = NormTo01 (v_norm);
    }
  return x;
}

std::vector<double>
BuildWeightVector (const std::vector<operations::Metric> &names,
                   const std::unordered_map<operations::Metric, double> &state)
{
  std::vector<double> w (names.size (), 0.0);
  for (std::size_t i = 0; i < names.size (); ++i)
    {
      w[i] = state.at (names[i]);
    }
  return w;
}

void
ComputeRLSGain (const std::vector<double> &x,
                const std::vector<std::vector<double> > &P,
                double lam, std::vector<double> &Px, double &denom)
{
  Px.assign (x.size (), 0.0);
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      double acc = 0.0;
      for (std::size_t j = 0; j < x.size (); ++j)
        {
          acc += P[i][j] * x[j];
        }
      Px[i] = acc;
    }
  denom = lam;
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      denom += x[i] * Px[i];
    }
}

void
UpdateRLSCovariance (const std::vector<double> &K,
                     const std::vector<double> &x,
                     const std::vector<std::vector<double> > &P,
                     double lam,
                     std::vector<std::vector<double> > &P_new)
{
  const std::size_t n = x.size ();
  P_new.assign (n, std::vector<double> (n, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          double I_minus_KxT = (i == j ? 1.0 : 0.0) - K[i] * x[j];
          double acc = 0.0;
          for (std::size_t k = 0; k < n; ++k)
            {
              acc += I_minus_KxT * P[k][j];
            }
          P_new[i][j] = acc / lam;
        }
    }
}

double
ComputeBlendConfidence (int signals_processed, double stability)
{
  const double tau_rls = core::Lerp (kTauRlsMin, kTauRlsMax, stability);
  const double count = static_cast<double> (std::max (0, signals_processed));
  return (count >= constants::kTwo)
            ? (1.0 - std::exp (-count / std::max (constants::kNormEpsilon, tau_rls)))
            : constants::kNormalizedMin;
}

std::vector<double>
BlendWeights (const std::vector<double> &w_boot,
              const std::vector<double> &w_rls,
              double confidence)
{
  std::vector<double> w_raw (w_boot.size (), 0.0);
  for (std::size_t i = 0; i < w_boot.size (); ++i)
    {
      const double wb = w_boot[i];
      const double wr = w_rls[i];
      const double w = (1.0 - confidence) * wb + confidence * wr;
      w_raw[i]
          = (w < constants::kNormalizedMin) ? constants::kNormalizedMin : w;
    }
  return w_raw;
}

double
ComputeWeightedScore (const std::vector<double> &x,
                      const std::vector<double> &w_raw)
{
  double weight_sum = 0.0;
  for (double w : w_raw)
    {
      weight_sum += w;
    }
  if (weight_sum <= std::numeric_limits<double>::epsilon ())
    {
      return 0.0;
    }
  double y = 0.0;
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      const double w_norm = w_raw[i] / weight_sum;
      const double xi = core::Clamp (x[i], constants::kNormalizedMin,
                                     constants::kNormalizedMax);
      y += w_norm * xi;
    }
  return core::Clamp (y, constants::kNormalizedMin, constants::kNormalizedMax);
}

} // namespace

void
FitMetricWeightsRLS::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  EnsureStateInitialized (p_ctx, cfg);
  const int k = std::max (
      1, static_cast<int> (std::round (core::Lerp (1.0, 8.0, cfg.stability))));
  p_ctx.blender_update_count += 1;
  if ((p_ctx.blender_update_count % k) != 0)
    {
      return;
    }
  const auto &names = SupportedMetrics ();
  const auto &metrics = context.GetAllMetrics ();
  std::vector<double> x = BuildMetricVector (names, metrics);
  auto it = metrics.find (operations::Metric::relevance);
  const double v = (it == metrics.end ()) ? 0.0 : it->second;
  double y = NormTo01 (v);
  std::vector<double> w = BuildWeightVector (names, p_ctx.blender_state);
  double y_hat = 0.0;
  for (std::size_t i = 0; i < names.size (); ++i)
    {
      y_hat += w[i] * x[i];
    }
  y_hat = core::Clamp (y_hat, 0.0, 1.0);
  const double e = y - y_hat;
  const double lam
      = std::min (kLamMax, std::max (kLamMin, kLamMin + kLamSlope * cfg.stability));
  auto &P = p_ctx.blender_P;
  if (P.empty ())
    {
      EnsureStateInitialized (p_ctx, cfg);
    }
  std::vector<double> Px;
  double denom = 0.0;
  ComputeRLSGain (x, P, lam, Px, denom);
  if (!std::isfinite (denom) || denom < kTiny)
    {
      const std::size_t n = P.size ();
      for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = 0; j < n; ++j)
            {
              P[i][j] = (i == j) ? kLarge : constants::kNormalizedMin;
            }
        }
      ComputeRLSGain (x, P, lam, Px, denom);
      if (!std::isfinite (denom) || denom < kTiny)
        {
          denom = lam + 1.0;
        }
    }
  std::vector<double> K (x.size (), 0.0);
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      K[i] = Px[i] / denom;
    }
  for (std::size_t i = 0; i < w.size (); ++i)
    {
      const double w_new = w[i] + K[i] * e;
      p_ctx.blender_state[names[i]]
          = core::Clamp (w_new, constants::kNormalizedMin,
                         constants::kNormalizedMax);
    }
  std::vector<std::vector<double> > P_new;
  UpdateRLSCovariance (K, x, P, lam, P_new);
  p_ctx.blender_P = std::move (P_new);
  p_ctx.blender_ready = true;
}

void
ComputeCompositeScore::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  EnsureStateInitialized (p_ctx, cfg);
  const auto &names = SupportedMetrics ();
  const auto &metrics = context.GetAllMetrics ();
  std::vector<double> x = BuildMetricVector (names, metrics);
  std::vector<double> w_boot (names.size (), 0.0);
  std::vector<double> w_rls (names.size (), 0.0);
  for (std::size_t i = 0; i < names.size (); ++i)
    {
      const auto &n = names[i];
      w_boot[i]
          = BootstrapWeight (n, cfg.focus, cfg.sensitivity, cfg.stability);
      auto it = p_ctx.blender_state.find (n);
      w_rls[i] = (it == p_ctx.blender_state.end ()) ? 0.0 : it->second;
    }
  const double confidence = ComputeBlendConfidence (p_ctx.signals_processed,
                                                    cfg.stability);
  std::vector<double> w_raw = BlendWeights (w_boot, w_rls, confidence);
  double weight_sum = 0.0;
  int effective = 0;
  for (double w : w_raw)
    {
      weight_sum += w;
      if (w > 0.0)
        {
          ++effective;
        }
    }
  context.SetLastWeightSum (weight_sum);
  context.SetLastEffectiveMetricCount (effective);
  if (weight_sum <= std::numeric_limits<double>::epsilon ())
    {
      context.SetCompositeScore (0.0);
      return;
    }
  const double y = ComputeWeightedScore (x, w_raw);
  context.SetCompositeScore (y);
}

} // namespace cortext::operations
