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

  // Read emotions produced by Algorithm 4 (or external instrumentation).
  double intensity = core::Clamp (context.GetEmotionIntensity (), 0.0, 1.0);
  double arousal = core::Clamp (context.GetArousal (), 0.0, 1.0);
  double valence = core::Clamp (context.GetValence (), 0.0, 1.0);
  (void)valence;

  // Trigger condition.
  const double theta_intensity = core::ThetaIntensity (S);
  const double theta_arousal = core::ThetaArousal (S);
  const bool triggered
      = (intensity >= theta_intensity) && (arousal >= theta_arousal);
  if (!triggered)
    {
      return;
    }

  // Derived parameters.
  const double half_life_bonus
      = core::EmotionalHalfLifeBonus (S, intensity); // >1
  const double detail_suppression = core::DetailSuppression (S, F);
  const int gist_components = core::GistComponents (F);
  const int cascade_radius = core::CascadeRadius (S);
  const double cascade_decay = core::CascadeDecay (S);

  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);
  (void)now_ts;

  // v2: Update flashbulb columns in memories table (merged from emotional_tags)
  for (const auto &evt : context.GetMemoryUsageEvents ())
    {
      if (!evt.used)
        {
          continue;
        }
      const long long id = static_cast<long long> (evt.embedding_id);

      tx.Execute ("UPDATE memories "
                  "SET flashbulb = 1, "
                  "    emotional_intensity = ?, "
                  "    half_life_bonus = ?, "
                  "    detail_suppression = ?, "
                  "    gist_components = ?, "
                  "    cascade_radius = ?, "
                  "    cascade_decay = ? "
                  "WHERE embedding_id = ?;",
                  { intensity, half_life_bonus, detail_suppression,
                    gist_components, cascade_radius, cascade_decay, id });
    }
}

} // namespace cortext::operations
