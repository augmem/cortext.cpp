#include "cortext/operations/emotion.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace cortext::operations
{

void
ApplyEmotionalConsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  const double S = cfg.sensitivity;
  const double theta_intensity = core::ThetaIntensity (S);
  const double theta_arousal = core::ThetaArousal (S);

  // Derived parameters.
  const double detail_suppression = core::DetailSuppression (S, F);
  const int gist_components = core::GistComponents (F);
  const int cascade_radius = core::CascadeRadius (S);
  const double cascade_decay = core::CascadeDecay (S);

  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);
  (void)now_ts;

  for (const auto &evt : context.GetMemoryUsageEvents ())
    {
      if (!evt.used)
        {
          continue;
        }
      const long long id = static_cast<long long> (evt.embedding_id);

      auto rows = tx.Execute (
          "SELECT s_emotion_max, s_arousal_avg FROM memories "
          "WHERE embedding_id = ?",
          { id });
      if (rows.empty ())
        {
          continue;
        }

      auto get_double = [] (const std::any &v, double def) -> double {
        if (v.type () == typeid (double))
          return std::any_cast<double> (v);
        if (v.type () == typeid (float))
          return static_cast<double> (std::any_cast<float> (v));
        if (v.type () == typeid (int))
          return static_cast<double> (std::any_cast<int> (v));
        if (v.type () == typeid (long long))
          return static_cast<double> (std::any_cast<long long> (v));
        return def;
      };

      const auto &row = rows[0];
      const double mem_emotion
          = row.count ("s_emotion_max") ? get_double (row.at ("s_emotion_max"), 0.0)
                                        : 0.0;
      const double mem_arousal
          = row.count ("s_arousal_avg") ? get_double (row.at ("s_arousal_avg"), 0.0)
                                        : 0.0;

      const bool triggered = (mem_emotion >= theta_intensity)
                             && (mem_arousal >= theta_arousal);
      if (!triggered)
        {
          continue;
        }

      const double half_life_bonus
          = core::EmotionalHalfLifeBonus (S, mem_emotion);

      // v2: Update flashbulb columns in memories table (merged from emotional_tags)
      tx.Execute ("UPDATE memories "
                  "SET flashbulb = 1, "
                  "    emotional_intensity = ?, "
                  "    half_life_bonus = ?, "
                  "    detail_suppression = ?, "
                  "    gist_components = ?, "
                  "    cascade_radius = ?, "
                  "    cascade_decay = ? "
                  "WHERE embedding_id = ?;",
                  { mem_emotion, half_life_bonus, detail_suppression,
                    gist_components, cascade_radius, cascade_decay, id });
    }
}

} // namespace cortext::operations
