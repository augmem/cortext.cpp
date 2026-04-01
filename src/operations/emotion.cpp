#include "cortext/operations/emotion.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace cortext::operations
{
namespace
{
bool
EnvFlag (const char *name)
{
  const char *value = std::getenv (name);
  if (!value)
    {
      return false;
    }
  std::string s (value);
  std::transform (s.begin (), s.end (), s.begin (),
                  [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
  return s == "1" || s == "true" || s == "yes" || s == "on";
}
} // namespace

void
ApplyEmotionalConsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  static const bool disable_percentile
      = EnvFlag ("CORTEXT_FLASHBULB_DISABLE_PERCENTILE");
  static const bool disable_rate = EnvFlag ("CORTEXT_FLASHBULB_DISABLE_RATE");
  static const bool disable_arousal
      = EnvFlag ("CORTEXT_FLASHBULB_DISABLE_AROUSAL");

  const auto &cfg = context.GetConfig ();
  const double S = cfg.sensitivity;
  const double S_eff = core::SensitivityBias (S);
  const double theta_intensity = core::ThetaIntensity (S);
  const double theta_arousal = core::ThetaArousal (S);
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
  const double mem_emotion_adj
      = core::Clamp (mem_emotion * core::FlashbulbGain (S), 0.0, 1.0);
  const double flashbulb_threshold
      = core::FlashbulbThresholdEff (S, mem_emotion_adj, mem_arousal);
  auto &p_ctx = context.GetProcessorContext ();
  const double flashbulb_target = core::Lerp (0.02, 0.06, S_eff);
  const double flashbulb_gain = core::Lerp (1.0, 0.6, S_eff);
  const double rate_adjust = disable_rate
                                 ? 1.0
                                 : core::Clamp (
                                       1.0 + flashbulb_gain
                                                  * (p_ctx.flashbulb_rate_ewma
                                                     - flashbulb_target),
                                       0.8, 1.2);
  double flashbulb_threshold_adj = flashbulb_threshold * rate_adjust;
  const double percentile = core::FlashbulbPercentile (S);
  const auto &emo_hist = p_ctx.recent_emotion_intensities;
  if (!disable_percentile && emo_hist.size () >= 12)
    {
      std::vector<double> sorted (emo_hist.begin (), emo_hist.end ());
      const size_t idx = static_cast<size_t> (
          std::round (percentile * (sorted.size () - 1)));
      std::nth_element (sorted.begin (), sorted.begin () + idx, sorted.end ());
      const double percentile_threshold = sorted[idx] * rate_adjust;
      flashbulb_threshold_adj
          = std::max (flashbulb_threshold_adj, percentile_threshold);
    }

  const bool triggered = (mem_emotion >= theta_intensity)
                         && (mem_arousal >= theta_arousal);
  if (!triggered)
    {
      return;
    }

  const double flashbulb_arousal = core::Lerp (0.7, 0.5, S_eff);
  const bool flashbulb_gate
      = disable_arousal ? true : (mem_arousal >= flashbulb_arousal);
  const int flashbulb
      = (flashbulb_gate && mem_emotion_adj >= flashbulb_threshold_adj) ? 1 : 0;
  const double flashbulb_alpha = core::Lerp (0.02, 0.12, S_eff);
  if (!disable_rate)
    {
      p_ctx.flashbulb_rate_ewma
          = core::Ewma (p_ctx.flashbulb_rate_ewma,
                       static_cast<double> (flashbulb), flashbulb_alpha);
    }
  const double half_life_bonus
      = flashbulb ? core::EmotionalHalfLifeBonus (S, mem_emotion) : 0.0;
  const double detail_suppression
      = flashbulb ? core::DetailSuppression (S, cfg.focus) : 0.0;
  const int gist_components = flashbulb ? core::GistComponents (cfg.focus) : 0;

  // v2: Update flashbulb columns in memories table (merged from emotional_tags)
  tx.Execute ("UPDATE memories "
              "SET flashbulb = MAX(flashbulb, ?), "
              "    emotional_intensity = ?, "
              "    half_life_bonus = ?, "
              "    detail_suppression = ?, "
              "    gist_components = ?, "
              "    cascade_radius = ?, "
              "    cascade_decay = ? "
              "WHERE embedding_id = ?;",
              { flashbulb, mem_emotion, half_life_bonus, detail_suppression,
                gist_components, cascade_radius, cascade_decay, id });
}

} // namespace cortext::operations
