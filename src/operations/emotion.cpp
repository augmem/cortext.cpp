#include "cortext/operations/emotion.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace cortext::operations
{

void
ApplyEmotionalConsolidation::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  const double S = cfg.sensitivity;

  // Read emotions produced by Algorithm 4 (or external instrumentation).
  double intensity = core::Clamp (context.GetEmotionIntensity (), 0.0, 1.0);
  double arousal = core::Clamp (context.GetArousal (), 0.0, 1.0);
  double valence = core::Clamp (context.GetValence (), 0.0, 1.0);

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
  const double fb_threshold_eff
      = core::FlashbulbThresholdEff (S, intensity, arousal);

  // Ensure emotional_tags table exists.
  {
    BufferedWriteInstruction op;
    op.query = "CREATE TABLE IF NOT EXISTS emotional_tags ("
               "embedding_id INTEGER PRIMARY KEY,"
               "flashbulb INTEGER NOT NULL DEFAULT 0,"
               "intensity REAL,"
               "arousal REAL,"
               "valence REAL,"
               "half_life_bonus REAL,"
               "detail_suppression REAL,"
               "gist_components INTEGER,"
               "cascade_radius INTEGER,"
               "cascade_decay REAL,"
               "flashbulb_threshold_eff REAL,"
               "ts INTEGER"
               ");";
    context.AddWriteInstruction (std::move (op));
  }

  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Persist tags for memories used in this signal.
  for (const auto &evt : context.GetMemoryUsageEvents ())
    {
      if (!evt.used)
        {
          continue;
        }
      const long long id = static_cast<long long> (evt.embedding_id);

      BufferedWriteInstruction op;
      op.query = "INSERT OR REPLACE INTO emotional_tags ("
                 "embedding_id, flashbulb, intensity, arousal, valence, "
                 "half_life_bonus, detail_suppression, gist_components, "
                 "cascade_radius, cascade_decay, flashbulb_threshold_eff, ts"
                 ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
      op.params = { id,
                    1,
                    intensity,
                    arousal,
                    valence,
                    half_life_bonus,
                    detail_suppression,
                    gist_components,
                    cascade_radius,
                    cascade_decay,
                    fb_threshold_eff,
                    now_ts };
      context.AddWriteInstruction (std::move (op));
    }
}

} // namespace cortext::operations
