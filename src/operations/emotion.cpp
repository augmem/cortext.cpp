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
  const double S = cfg.sensitivity;
  const double theta_intensity = core::ThetaIntensity (S);
  const double theta_arousal = core::ThetaArousal (S);
  const double flashbulb_threshold = core::FlashbulbThreshold (S);

  // Derived parameters.
  const int cascade_radius = core::CascadeRadius (S);
  const double cascade_decay = core::CascadeDecay (S);

  const auto stored_id = context.GetStoredEmbeddingId ();
  if (!stored_id.has_value ())
    {
      return;
    }

  const long long id = static_cast<long long> (*stored_id);
  auto rows = tx.Execute (
      "SELECT s_emotion_max, s_arousal_avg, flashbulb FROM memories "
      "WHERE embedding_id = ?",
      { id });
  if (rows.empty ())
    {
      return;
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
      return;
    }

  const int flashbulb = (mem_emotion >= flashbulb_threshold) ? 1 : 0;
  const double half_life_bonus
      = flashbulb ? core::EmotionalHalfLifeBonus (S, mem_emotion) : 0.0;

  // v2: Update flashbulb columns in memories table (merged from emotional_tags)
  tx.Execute ("UPDATE memories "
              "SET flashbulb = MAX(flashbulb, ?), "
              "    emotional_intensity = ?, "
              "    half_life_bonus = ?, "
              "    cascade_radius = ?, "
              "    cascade_decay = ? "
              "WHERE embedding_id = ?;",
              { flashbulb, mem_emotion, half_life_bonus, cascade_radius,
                cascade_decay, id });
}

} // namespace cortext::operations
