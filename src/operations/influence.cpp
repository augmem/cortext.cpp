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

  const double L_sustain
      = std::round (core::Lerp (constants::kSustainWindowMin,
                                constants::kSustainWindowMax, cfg.stability));
  const double alpha_sustain
      = (L_sustain > 0.0) ? (constants::kTwo / (L_sustain + 1.0)) : 1.0;

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
      const double sim_gen = 0.0;
      const double drift_contrib = 0.0;
      const double influence
          = constants::kLambda1 * contextual_gain
            + constants::kLambda2 * sim_gen
            - constants::kLambda3 * drift_contrib;
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
          = core::Ewma (sustained_prev, influence, alpha_sustain);
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
