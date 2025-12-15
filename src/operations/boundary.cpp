#include "cortext/operations/boundary.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cmath>
#include <string>
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
CheckEpisodeBoundary::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();

  const int n = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  if (n < 4)
    {
      context.SetMetric (operations::Metric::drift_mag, 0.0);
      return;
    }
  // Use two adjacent windows of size k
  const int k = std::min (core::KNeighbors (cfg.stability), n / 2);
  if (k < 1)
    {
      context.SetMetric (operations::Metric::drift_mag, 0.0);
      return;
    }
  const int split = n - k;
  if (split < k)
    {
      context.SetMetric (operations::Metric::drift_mag, 0.0);
      return;
    }

  const Eigen::VectorXf mean_recent
      = ComputeMean (p_ctx.recent_context_embeddings, n - k, n);
  const Eigen::VectorXf mean_prev
      = ComputeMean (p_ctx.recent_context_embeddings, split - k, split);

  if (mean_recent.size () == 0 || mean_prev.size () == 0
      || mean_recent.size () != mean_prev.size ())
    {
      context.SetMetric (operations::Metric::drift_mag, 0.0);
      return;
    }

  const Eigen::VectorXf nr = (mean_recent.norm () > 0.0f)
                                 ? (mean_recent / mean_recent.norm ())
                                 : mean_recent;
  const Eigen::VectorXf np = (mean_prev.norm () > 0.0f)
                                 ? (mean_prev / mean_prev.norm ())
                                 : mean_prev;
  const double d = (nr - np).norm (); // ∈ [0,2]
  const double drift01
      = core::Clamp (constants::kOneHalf * d, constants::kNormalizedMin,
                     constants::kNormalizedMax);
  context.SetMetric (operations::Metric::drift_mag, drift01);

  const double threshold
      = core::Lerp (constants::kGainMedium, constants::kWeightHigh,
                    cfg.stability);
  if (drift01 > threshold)
    {
      // Compute episode centroid from recent context embeddings
      const Eigen::VectorXf centroid
          = ComputeMean (p_ctx.recent_context_embeddings, 0, n);

      // Convert to std::vector<float> for sqlite-vec compatible BLOB storage
      std::vector<float> centroid_blob (centroid.data (),
                                        centroid.data () + centroid.size ());

      // Buffer episode INSERT
      BufferedWriteInstruction op;
      op.query = "INSERT INTO episodes (start_ts, end_ts, boundary_type, "
                 "centroid) VALUES (?, ?, ?, ?)";
      op.params = { static_cast<long long> (p_ctx.episode_start_ts),
                    static_cast<long long> (context.GetSignal ().timestamp),
                    std::string ("drift"), centroid_blob };
      context.AddWriteInstruction (std::move (op));

      // Update episode_start_ts for next episode
      p_ctx.episode_start_ts = context.GetSignal ().timestamp;

      context.RequestFinalizeEpisode ();
    }
}

} // namespace cortext::operations
