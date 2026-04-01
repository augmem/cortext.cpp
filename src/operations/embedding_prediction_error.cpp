#include "cortext/operations/embedding_prediction_error.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <cmath>

namespace cortext::operations
{

namespace
{
Eigen::VectorXf
Normalize (const Eigen::VectorXf &v)
{
  if (v.size () == 0)
    return v;
  const float n = v.norm ();
  if (n > 0.0f)
    return v / n;
  return v;
}

inline double
Clamp01 (double v)
{
  return cortext::core::Clamp (v, constants::kNormalizedMin,
                               constants::kNormalizedMax);
}
} // namespace

void
UpdateEmbeddingPredictionError::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &signal = context.GetSignal ();
  const Eigen::VectorXf *x_ptr = &signal.embedding;
  auto it = p_ctx.accumulator_states.find (signal.source_id);
  if (it != p_ctx.accumulator_states.end ()
      && it->second.mu_acc.size () > 0)
    {
      x_ptr = &it->second.mu_acc;
    }
  const auto &x_t = *x_ptr;

  if (x_t.size () == 0)
    {
      return; // No embedding available
    }

  // Initialize predictor state on first signal or dimension change
  if (!p_ctx.x_pred_ema.has_value ()
      || p_ctx.x_pred_ema->size () != x_t.size ())
    {
      p_ctx.x_pred_ema = Normalize (x_t);
      p_ctx.last_embedding = x_t;
      p_ctx.prediction_error_sse.reset ();
      p_ctx.prediction_error_sse_prev.reset ();
      context.SetMetric (operations::Metric::embedding_surprisal,
                         constants::kNormalizedMin);
      return;
    }

  const Eigen::VectorXf &x_pred = *p_ctx.x_pred_ema;
  double cos_sim = core::CosineSimilarity (x_pred, x_t);
  if (!std::isfinite (cos_sim))
    cos_sim = 0.0;
  const double prediction_error = 1.0 - std::max (0.0, cos_sim);
  const double sse_curr = (x_t - x_pred).squaredNorm ();
  const double sse_prev = p_ctx.prediction_error_sse.value_or (-1.0);

  const double err_ref = core::SurpriseErrRef (cfg.sensitivity);
  const double k_surprise = core::SurpriseGain (cfg.sensitivity, cfg.stability);
  const double surprisal = Clamp01 (
      core::Sigmoid ((prediction_error - err_ref) * k_surprise));

  context.SetMetric (operations::Metric::embedding_surprisal, surprisal);

  p_ctx.prediction_error_sse_prev = p_ctx.prediction_error_sse;
  p_ctx.prediction_error_sse = sse_curr;

  const float beta = static_cast<float> (core::PredEmaBeta (cfg.stability));
  Eigen::VectorXf updated = (1.0f - beta) * x_pred + beta * x_t;
  p_ctx.x_pred_ema = Normalize (updated);
  p_ctx.last_embedding = x_t;

  telemetry::LogDebug("cortext.embedding_prediction_error", {
    telemetry::Attribute::Double("cos_sim", cos_sim),
    telemetry::Attribute::Double("prediction_error", prediction_error),
    telemetry::Attribute::Double("embedding_surprisal", surprisal),
    telemetry::Attribute::Double("prediction_error_sse", sse_curr),
    telemetry::Attribute::Double("prediction_error_sse_prev", sse_prev),
    telemetry::Attribute::Double("err_ref", err_ref),
    telemetry::Attribute::Double("surprise_gain", k_surprise),
    telemetry::Attribute::Double("ema_beta", static_cast<double> (beta))
  });
}

} // namespace cortext::operations
