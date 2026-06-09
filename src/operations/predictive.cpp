#include "cortext/operations/predictive.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <vector>

namespace cortext::operations
{

namespace
{
inline Eigen::VectorXf
Unit (const Eigen::VectorXf &v)
{
  const double n = v.norm ();
  if (n <= constants::kNormEpsilon)
    {
      return v;
    }
  return v / static_cast<float> (n);
}

inline double
Clamp01 (double v)
{
  if (v < constants::kNormalizedMin)
    return constants::kNormalizedMin;
  if (v > constants::kNormalizedMax)
    return constants::kNormalizedMax;
  return v;
}

} // namespace

void
ApplyPredictivePreActivation::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double pad = core::PredictivePreActivationDecay (cfg.stability);

  // Keep predictive state transient by decaying prior pre-activation every turn.
  tx.Execute ("UPDATE memories "
              "SET pre_activation = CASE "
              "  WHEN COALESCE(pre_activation, 0.0) <= 1e-6 THEN 0.0 "
              "  ELSE COALESCE(pre_activation, 0.0) * ? "
              "END "
              "WHERE COALESCE(pre_activation, 0.0) > 0.0;",
              { pad });

  // Need some recent context to estimate a trajectory.
  if (p_ctx.recent_context_embeddings.empty ())
    {
      return;
    }
  const int available
      = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  const int want = core::PredictionHorizon (cfg.focus);
  const int take = std::max (1, std::min (available, want));

  // Predicted direction: mean of last `take` embeddings, then normalized.
  Eigen::VectorXf pred = Eigen::VectorXf::Zero (
      p_ctx.recent_context_embeddings.back ().size ());
  for (int i = available - take; i < available; ++i)
    {
      pred += p_ctx.recent_context_embeddings[static_cast<size_t> (i)];
    }
  if (pred.size () == 0)
    {
      return;
    }
  pred = Unit (pred);
  // Fallback to last context if degenerate.
  if (pred.norm () <= 1e-9f)
    {
      pred = Unit (p_ctx.recent_context_embeddings.back ());
    }

  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  if (retrieved.empty ())
    {
      return;
    }

  const double conf_thresh = core::PredictionConfidenceThreshold (cfg.focus);
  const double base_delta = core::PredictiveBaseDelta (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double update_rate_on_surprise
      = core::PredictiveSurpriseUpdateRate (cfg.sensitivity, cfg.stability);
  const double surp_sens
      = core::PredictiveSurpriseSensitivity (cfg.sensitivity, cfg.stability);

  // Optional surprise modulation from metrics (map to [0,1]).
  double surprise_01 = constants::kNormalizedMin;
  if (auto m = context.GetMetric (operations::Metric::surprise))
    {
      double v = *m;
      if (std::isnan (v) || std::isinf (v))
        {
          v = constants::kNormalizedMin;
        }
      if (v < constants::kNormalizedMin)
        {
          surprise_01 = Clamp01 (
              (v + constants::kNormalizedMax) / constants::kTwo);
        }
      else
        {
          surprise_01 = Clamp01 (v);
        }
    }

  int boost_count = 0;
  for (const auto &kv : retrieved)
    {
      const long long id = kv.first;
      const Eigen::VectorXf &vec = kv.second;
      if (vec.size () == 0 || vec.size () != pred.size ())
        {
          continue;
        }
      double sim_pred = core::CosineSimilarity (pred, vec);
      sim_pred = Clamp01 (sim_pred);
      if (sim_pred < conf_thresh)
        {
          continue;
        }

      // Compute modulated delta; keep safe bounds.
      double delta = base_delta * (1.0 + constants::kQuarter * surp_sens
                                              * surprise_01);
      delta += update_rate_on_surprise * surprise_01;
      delta = core::Clamp (
          delta, 0.0,
          core::PredictiveDeltaCap (cfg.focus, cfg.sensitivity, cfg.stability));

      // v2: Update memories pre_activation (row exists from storage)
      tx.Execute ("UPDATE memories "
                  "SET pre_activation = MIN(?, COALESCE(pre_activation, 0.0) * ? + ?) "
                  "WHERE embedding_id = ?;",
                  { 1.0, pad, delta, id });
      ++boost_count;
    }

  // Debug logging
  telemetry::LogDebug ("cortext.predictive", {
    telemetry::Attribute::Double ("conf_thresh", conf_thresh),
    telemetry::Attribute::Double ("pad", pad),
    telemetry::Attribute::Double ("base_delta", base_delta),
    telemetry::Attribute::Double ("update_rate_on_surprise", update_rate_on_surprise),
    telemetry::Attribute::Double ("surp_sens", surp_sens),
    telemetry::Attribute::Double ("surprise_01", surprise_01),
    telemetry::Attribute::Double ("prediction_norm", static_cast<double> (pred.norm ())),
    telemetry::Attribute::Int64 ("boost_count", static_cast<int64_t> (boost_count))
  });
}

} // namespace cortext::operations
