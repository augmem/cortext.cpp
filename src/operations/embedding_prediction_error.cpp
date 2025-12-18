#include "cortext/operations/embedding_prediction_error.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <cmath>

namespace cortext::operations
{

namespace
{
constexpr double kTrendAlpha = 0.1; // EWMA smoothing for trend
constexpr double kErrMax = 0.5;     // Normalization ceiling
} // namespace

void
UpdateEmbeddingPredictionError::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &x_t = context.GetSignal ().embedding;

  if (x_t.size () == 0)
    {
      return; // No embedding available
    }

  // First signal: initialize state, no prediction possible yet
  if (!p_ctx.last_embedding.has_value ())
    {
      p_ctx.last_embedding = x_t;
      return;
    }

  const auto &x_prev = *p_ctx.last_embedding;

  // Dimension mismatch: reset state
  if (x_prev.size () != x_t.size ())
    {
      p_ctx.last_embedding = x_t;
      p_ctx.delta_x_trend.reset ();
      return;
    }

  // Δx_t = x_t − x_{t−1}
  Eigen::VectorXf delta_x = x_t - x_prev;

  // Store previous trend for prediction before updating
  Eigen::VectorXf prev_trend = p_ctx.delta_x_trend.value_or (
      Eigen::VectorXf::Zero (x_t.size ()));

  // Δx_trend_t = EWMA(Δx_trend_{t−1}, Δx_t, α=0.1)
  if (!p_ctx.delta_x_trend.has_value ())
    {
      p_ctx.delta_x_trend = delta_x;
    }
  else
    {
      const float alpha = static_cast<float> (kTrendAlpha);
      p_ctx.delta_x_trend
          = (1.0f - alpha) * (*p_ctx.delta_x_trend) + alpha * delta_x;
    }

  // x_pred_t = x_{t−1} + Δx_trend_{t−1} (use PREVIOUS trend)
  Eigen::VectorXf x_pred = x_prev + prev_trend;

  // prediction_error_t = 1 − cos(x_pred_t, x_t)
  const double cos_sim = core::CosineSimilarity (x_pred, x_t);
  const double prediction_error = 1.0 - cos_sim;

  // surprisal_t = clamp(prediction_error_t / err_max, 0, 1)
  const double surprisal
      = core::Clamp (prediction_error / kErrMax, constants::kNormalizedMin,
                     constants::kNormalizedMax);

  context.SetMetric (operations::Metric::embedding_surprisal, surprisal);

  // Update state for next iteration
  p_ctx.last_embedding = x_t;

  telemetry::LogDebug("cortext.embedding_prediction_error", {
    telemetry::Attribute::Double("prediction_error", prediction_error),
    telemetry::Attribute::Double("embedding_surprisal", surprisal)
  });
}

} // namespace cortext::operations
