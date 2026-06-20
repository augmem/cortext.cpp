#include "cortext/operations/influence.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <vector>

namespace cortext::operations
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

void
ApplyInfluenceFeedback::Execute (OperationContext &context, Transaction &tx) const
{
  const auto &cfg = context.GetConfig ();

  const auto &events = context.GetMemoryUsageEvents ();
  const auto &emb_map = context.GetRetrievedMemoryEmbeddings ();

  const auto influence_policy = core::InfluenceFeedbackPolicyForKnobs (
      cfg.focus, cfg.sensitivity, cfg.stability);

  std::vector<double> influences;
  influences.reserve (events.size ());

  for (const auto &e : events)
    {
      if (!e.used)
        {
          continue;
        }
      auto it = emb_map.find (e.embedding_id);
      if (it == emb_map.end ())
        {
          continue;
        }
      const Eigen::VectorXf u_m = Unit (it->second);
      const double contextual_gain
          = core::Clamp (e.contextual_gain.value_or (0.0), -1.0, 1.0);
      const double predictive_similarity = 0.0;
      const double drift_contrib = 0.0;
      const double influence
          = influence_policy.contextual_gain_weight * contextual_gain
            + influence_policy.predictive_similarity_weight
                  * predictive_similarity
            - influence_policy.drift_weight * drift_contrib;
      influences.push_back (influence);

      auto rows = tx.Execute (
          "SELECT sustained_influence, mean_influence, used_count "
          "FROM memories WHERE embedding_id = ?",
          { static_cast<long long> (e.embedding_id) });
      if (rows.empty ())
        {
          continue;
        }
      const double sustained_prev
          = std::any_cast<double> (rows[0].at ("sustained_influence"));
      const double mean_prev
          = std::any_cast<double> (rows[0].at ("mean_influence"));
      const long long used_prev
          = std::any_cast<long long> (rows[0].at ("used_count"));
      const double sustained_new
          = core::Ewma (sustained_prev, influence,
                        influence_policy.sustain_alpha);
      const double denom = static_cast<double> (std::max (used_prev, 0LL) + 1);
      const double mean_new = (denom > 0.0)
                                  ? ((mean_prev * used_prev + influence) / denom)
                                  : influence;
      tx.Execute ("UPDATE memories "
                  "SET influence = ?, sustained_influence = ?, mean_influence = ? "
                  "WHERE embedding_id = ?",
                  { influence, sustained_new, mean_new,
                    static_cast<long long> (e.embedding_id) });
    }

  if (influences.empty ())
    {
      return;
    }

  const double mean_influence
      = std::accumulate (influences.begin (), influences.end (), 0.0)
        / static_cast<double> (influences.size ());

  telemetry::LogDebug ("cortext.influence",
                       { telemetry::Attribute::Double ("mean_influence",
                                                       mean_influence) });
}

} // namespace cortext::operations
