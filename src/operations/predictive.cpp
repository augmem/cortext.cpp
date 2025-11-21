#include "cortext/operations/predictive.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace cortext::operations
{

namespace
{
constexpr double kTrajSamplesMax = 5.0;
constexpr double kTrajSamplesMin = 1.0;
constexpr double kConfMin = 0.3;
constexpr double kConfMax = 0.7;
constexpr double kDecayMax = 0.7;
constexpr double kDecayMin = 0.3;
constexpr double kSurpriseMax = 2.0;
constexpr double kSurpriseMin = 0.5;
constexpr double kBaseDeltaScale = 0.02;
constexpr double kPadRef = 0.3;
static const double kStrengthMax = 1.2;
inline int
TrajectorySamples (double F)
{
  // trajectory_samples = round(lerp(5, 1, F))
  return std::max (1, static_cast<int> (
                          std::round (core::Lerp (kTrajSamplesMax,
                                                   kTrajSamplesMin, F))));
}

inline double
PredictionConfidenceThreshold (double F)
{
  // prediction_conf_threshold = lerp(0.3, 0.7, F)
  return core::Lerp (kConfMin, kConfMax, F);
}

inline double
PreActivationDecay (double T)
{
  // pre_activation_decay = lerp(0.7, 0.3, T)
  return core::Lerp (kDecayMax, kDecayMin, T);
}

inline double
SurpriseSensitivity (double S, double T)
{
  // surprise_sensitivity = S × lerp(2.0, 0.5, T)
  return S * core::Lerp (kSurpriseMax, kSurpriseMin, T);
}

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
ApplyPredictivePreActivation::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // Need some recent context to estimate a trajectory.
  if (p_ctx.recent_context_embeddings.empty ())
    {
      return;
    }
  const int available
      = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  const int want = TrajectorySamples (cfg.focus);
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

  // Ensure persistence table exists (idempotent).
  {
    BufferedWriteInstruction op;
    op.query = "CREATE TABLE IF NOT EXISTS memory_feedback ("
               "  embedding_id INTEGER PRIMARY KEY,"
               "  retrieved_count INTEGER NOT NULL DEFAULT 0,"
               "  used_count INTEGER NOT NULL DEFAULT 0,"
               "  contextual_gain REAL NOT NULL DEFAULT 0.0,"
               "  use_frequency REAL NOT NULL DEFAULT 0.0,"
               "  strength REAL NOT NULL DEFAULT 1.0"
               ");";
    context.AddWriteInstruction (std::move (op));
  }

  const double conf_thresh = PredictionConfidenceThreshold (cfg.focus);
  const double pad = PreActivationDecay (cfg.stability);
  // Base delta in ~[0.012, 0.02]
  const double base_delta = kBaseDeltaScale * (1.0 - (pad - kPadRef));
  const double surp_sens
      = SurpriseSensitivity (cfg.sensitivity, cfg.stability);

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
      if (delta < constants::kNormalizedMin)
        delta = constants::kNormalizedMin;
      if (delta > constants::kGainSmall)
        delta = constants::kGainSmall;

      // Ensure row, then apply boost with clamp to a small ceiling > 1.0.
      {
        BufferedWriteInstruction op;
        op.query = "INSERT OR IGNORE INTO memory_feedback (embedding_id) "
                   "VALUES (?);";
        op.params = { id };
        context.AddWriteInstruction (std::move (op));
      }
      {
        BufferedWriteInstruction op;
        op.query = "UPDATE memory_feedback "
                   "SET strength = MIN(?, strength + ?) "
                   "WHERE embedding_id = ?;";
        op.params = { kStrengthMax, delta, id };
        context.AddWriteInstruction (std::move (op));
      }
    }
}

} // namespace cortext::operations
