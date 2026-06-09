#include "cortext/operations/blend.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace cortext::operations
{
namespace
{
constexpr double kTiny = 1e-6;

// Number of metrics supported
constexpr size_t kNumMetrics = 12;

// Bootstrap coefficient matrices derived from current heuristics
// w_bootstrap[i] = sigmoid(c_F[i]*F + c_S[i]*S + c_T[i]*T + d[i])
// Order: relevance, mismatch, surprise, rarity, drift, utility,
//        salience, valence, arousal, contradiction, periphery, coverage
constexpr double kBootstrapCF[kNumMetrics] = {
    1.4,   // relevance: 0.7*F + 0.2*T → z=1.4F+0.4T-1
    -1.0,  // mismatch: (1-F)*S → linearized
    0.0,   // surprise: S*(1-0.5T)
    0.9,   // rarity: (0.5+0.5F)*(1-0.2T)
    0.0,   // drift: 0.5*(1-T)
    0.85,  // utility: (0.5+0.5F)*(1-0.3S)
    1.0,   // salience: 0.5*(F+S)
    0.0,   // valence: (0.4+0.6S)*(1-0.3T)
    0.0,   // arousal: S*(1-0.2T)
    -2.0,  // contradiction: max(0,S-F) → linearized
    0.0,   // periphery: 0.5*T
    2.0    // coverage: F
};
constexpr double kBootstrapCS[kNumMetrics] = {
    0.0,   // relevance
    1.0,   // mismatch
    1.5,   // surprise
    0.0,   // rarity
    0.0,   // drift
    -0.45, // utility
    1.0,   // salience
    1.02,  // valence
    1.8,   // arousal
    2.0,   // contradiction
    0.0,   // periphery
    0.0    // coverage
};
constexpr double kBootstrapCT[kNumMetrics] = {
    0.4,   // relevance
    0.0,   // mismatch
    -0.5,  // surprise
    -0.3,  // rarity
    -1.0,  // drift
    0.0,   // utility
    0.0,   // salience
    -0.42, // valence
    -0.2,  // arousal
    0.0,   // contradiction
    1.0,   // periphery
    0.0    // coverage
};
constexpr double kBootstrapD[kNumMetrics] = {
    -1.0,   // relevance
    -0.5,   // mismatch
    -0.75,  // surprise
    0.05,   // rarity
    0.0,    // drift
    0.075,  // utility
    -1.0,   // salience
    -0.11,  // valence
    -0.9,   // arousal
    -1.0,   // contradiction
    -1.0,   // periphery
    -1.0    // coverage
};

inline const std::vector<operations::Metric> &
SupportedMetrics ()
{
  static const std::vector<operations::Metric> kNames = {
    operations::Metric::relevance, operations::Metric::mismatch,
    operations::Metric::surprise,  operations::Metric::rarity,
    operations::Metric::drift,     operations::Metric::utility,
    operations::Metric::salience,  operations::Metric::valence,
    operations::Metric::arousal,   operations::Metric::contradiction,
    operations::Metric::periphery, operations::Metric::coverage,
  };
  return kNames;
}

inline const char *
MetricLabel (operations::Metric m)
{
  switch (m)
    {
    case operations::Metric::relevance:
      return "relevance";
    case operations::Metric::mismatch:
      return "mismatch";
    case operations::Metric::surprise:
      return "surprise";
    case operations::Metric::rarity:
      return "rarity";
    case operations::Metric::drift:
      return "drift";
    case operations::Metric::utility:
      return "utility";
    case operations::Metric::salience:
      return "salience";
    case operations::Metric::valence:
      return "valence";
    case operations::Metric::arousal:
      return "arousal";
    case operations::Metric::contradiction:
      return "contradiction";
    case operations::Metric::periphery:
      return "periphery";
    case operations::Metric::coverage:
      return "coverage";
    case operations::Metric::focus_spread:
      return "focus_spread";
    case operations::Metric::drift_mag:
      return "drift_mag";
    case operations::Metric::embedding_surprisal:
      return "embedding_surprisal";
    case operations::Metric::aw_prev:
      return "aw_prev";
    case operations::Metric::rate_prev:
      return "rate_prev";
    case operations::Metric::hys_prev:
      return "hys_prev";
    }
  return "unknown";
}

inline std::string
BuildVectorString (const std::vector<operations::Metric> &names,
                   const std::vector<double> &values)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < names.size () && i < values.size (); ++i)
    {
      if (i > 0)
        {
          out << ", ";
        }
      out << MetricLabel (names[i]) << "=" << values[i];
    }
  return out.str ();
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

// Computes bootstrap weight using formal coefficient matrices
// w_bootstrap[i] = sigmoid(c_F[i]*F + c_S[i]*S + c_T[i]*T + d[i])
inline double
BootstrapWeightByIndex (size_t metric_index, double F, double S, double T)
{
  if (metric_index >= kNumMetrics)
    {
      return core::BlendBootstrapFallback (F, S, T);
    }
  const double f_eff = core::FocusBias (F);
  const double s_eff = core::SensitivityBias (S);
  const double z = kBootstrapCF[metric_index] * f_eff
                 + kBootstrapCS[metric_index] * s_eff
                 + kBootstrapCT[metric_index] * T
                 + kBootstrapD[metric_index];
  return core::Clamp (core::Sigmoid (z),
                      constants::kNormalizedMin,
                      constants::kNormalizedMax);
}

// Map metric enum to bootstrap index
inline double
BootstrapWeight (operations::Metric name, double F, double S, double T)
{
  const auto &names = SupportedMetrics ();
  for (size_t i = 0; i < names.size (); ++i)
    {
      if (names[i] == name)
        {
          return BootstrapWeightByIndex (i, F, S, T);
        }
    }
  return core::BlendBootstrapFallback (F, S, T);
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
      const double init
          = core::BlenderPInit (cfg.stability);
      p_ctx.blender_P.assign (n, std::vector<double> (n, constants::kNormalizedMin));
      for (std::size_t i = 0; i < n; ++i)
        {
          p_ctx.blender_P[i][i] = init;
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
                     const std::vector<double> &Px,
                     const std::vector<std::vector<double> > &P,
                     double lam,
                     std::vector<std::vector<double> > &P_new)
{
  const std::size_t n = K.size ();
  P_new.assign (n, std::vector<double> (n, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          P_new[i][j] = (P[i][j] - K[i] * Px[j]) / lam;
        }
    }
}

double
ComputeBlendConfidence (int signals_processed, double stability)
{
  const double tau_rls = core::BlenderRLSObservationTau (stability);
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
FitMetricWeightsRLS::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  EnsureStateInitialized (p_ctx, cfg);
  const int k = core::BlenderUpdateInterval (cfg.stability);
  p_ctx.blender_update_count += 1;
  if ((p_ctx.blender_update_count % k) != 0)
    {
      return;
    }
  const auto &names = SupportedMetrics ();
  const auto &metrics = context.GetAllMetrics ();
  std::vector<double> x = BuildMetricVector (names, metrics);
  const double o_use = core::Clamp (p_ctx.last_used_flag, 0.0, 1.0);
  const double o_pred
      = NormTo01 (
          context.GetMetric (operations::Metric::utility).value_or (0.0));
  const double o_unc = core::Clamp (1.0 - p_ctx.u_t, 0.0, 1.0);
  const double o_user = 0.0;
  const auto outcome_weights = core::BlenderOutcomeScoringWeights (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double w_use = outcome_weights.used;
  const double w_pred = outcome_weights.predictive;
  const double w_unc = outcome_weights.uncertainty;
  const double w_user = outcome_weights.user;
  const double w_sum = std::max (kTiny, w_use + w_pred + w_unc + w_user);
  const double outcome_t
      = core::Clamp ((w_use * o_use + w_pred * o_pred + w_unc * o_unc
                      + w_user * o_user)
                         / w_sum,
                     0.0, 1.0);
  const double alpha_out = core::BlenderOutcomeAlpha (cfg.stability);
  p_ctx.outcome_pred = core::Ewma (p_ctx.outcome_pred, outcome_t, alpha_out);
  p_ctx.delta_reward = outcome_t - p_ctx.outcome_pred;
  double y = outcome_t;
  std::vector<double> w = BuildWeightVector (names, p_ctx.blender_state);
  double y_hat = 0.0;
  for (std::size_t i = 0; i < names.size (); ++i)
    {
      y_hat += w[i] * x[i];
    }
  y_hat = core::Clamp (y_hat, 0.0, 1.0);
  const double e = y - y_hat;
  const double lam = core::BlenderRLSForgettingFactor (cfg.stability);
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
              P[i][j] = (i == j)
                            ? core::BlenderPInit (cfg.stability)
                            : constants::kNormalizedMin;
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
  UpdateRLSCovariance (K, Px, P, lam, P_new);
  p_ctx.blender_P = std::move (P_new);
  p_ctx.blender_ready = true;

  // Compute RLS confidence for logging
  const double rls_confidence
      = ComputeBlendConfidence (p_ctx.signals_processed, cfg.stability);

  telemetry::LogDebug("cortext.blend.fit_rls", {
    telemetry::Attribute::Double("forgetting_factor", lam),
    telemetry::Attribute::Double("rls_confidence", rls_confidence)
  });
}

void
ComputeCompositeScore::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  EnsureStateInitialized (p_ctx, cfg);
  const auto &names = SupportedMetrics ();
  const auto &metrics = context.GetAllMetrics ();
  std::vector<double> x = BuildMetricVector (names, metrics);
  std::vector<double> w_boot (names.size (), 0.0);
  std::vector<double> w_rls (names.size (), 0.0);
  std::vector<double> mults (names.size (), 1.0);

  // Compute bootstrap weights using formal coefficient matrices
  for (std::size_t i = 0; i < names.size (); ++i)
    {
      w_boot[i] = BootstrapWeightByIndex (i, cfg.focus, cfg.sensitivity,
                                          cfg.stability);
    }

  w_rls = BuildWeightVector (names, p_ctx.blender_state);
  for (double &w : w_rls)
    {
      w = core::Clamp (w, constants::kNormalizedMin,
                       constants::kNormalizedMax);
    }

  const double confidence
      = ComputeBlendConfidence (p_ctx.signals_processed, cfg.stability);
  std::vector<double> w_raw = BlendWeights (w_boot, w_rls, confidence);

  // Apply control-weight multipliers before normalization.
  for (std::size_t i = 0; i < w_raw.size (); ++i)
    {
      const auto metric = names[i];
      double mult = 1.0;
      switch (metric)
        {
        case operations::Metric::relevance:
          mult = p_ctx.weight_relevance;
          break;
        case operations::Metric::mismatch:
          mult = p_ctx.mismatch_weight;
          break;
        case operations::Metric::surprise:
          mult = p_ctx.weight_surprise;
          break;
        case operations::Metric::coverage:
          mult = p_ctx.coverage_gain_floor;
          break;
        case operations::Metric::valence:
          mult = p_ctx.weight_valence * p_ctx.emotion_gain;
          break;
        case operations::Metric::arousal:
          mult = p_ctx.weight_arousal * p_ctx.emotion_gain;
          break;
        default:
          break;
        }
      w_raw[i] *= core::Clamp (mult, constants::kNormalizedMin,
                               constants::kNormalizedMax);
      mults[i] = mult;
    }
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
  const double y_raw = ComputeWeightedScore (x, w_raw);
  const double y = core::Clamp (y_raw * p_ctx.score_gain,
                                constants::kNormalizedMin,
                                constants::kNormalizedMax);
  context.SetCompositeScore (y);

  // Compute tau_rls for logging
  const double tau_rls = core::BlenderRLSObservationTau (cfg.stability);
  std::vector<double> w_norm (w_raw.size (), 0.0);
  if (weight_sum > 0.0)
    {
      for (std::size_t i = 0; i < w_raw.size (); ++i)
        {
          w_norm[i] = w_raw[i] / weight_sum;
        }
    }

  telemetry::LogDebug("cortext.blend.composite_score", {
    telemetry::Attribute::Double("confidence", confidence),
    telemetry::Attribute::Double("weight_sum", weight_sum),
    telemetry::Attribute::Int64("effective_metrics", static_cast<int64_t> (effective)),
    telemetry::Attribute::Double("score_gain", p_ctx.score_gain),
    telemetry::Attribute::String("metrics", BuildVectorString(names, x)),
    telemetry::Attribute::String("weights_boot", BuildVectorString(names, w_boot)),
    telemetry::Attribute::String("weights_rls", BuildVectorString(names, w_rls)),
    telemetry::Attribute::String("weights_raw", BuildVectorString(names, w_raw)),
    telemetry::Attribute::String("weights_norm", BuildVectorString(names, w_norm)),
    telemetry::Attribute::String("multipliers", BuildVectorString(names, mults)),
    telemetry::Attribute::Double("tau_rls", tau_rls),
    telemetry::Attribute::Double("composite_score", y)
  });
}

} // namespace cortext::operations
