#include "cortext/operations/synaptic_tagging.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <cmath>

namespace cortext::operations
{

void
ApplySynapticTagging::Execute (OperationContext &context, Transaction &tx) const
{
  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  const uint64_t now_ts = context.GetSignal ().timestamp;
  const double surprisal
      = context.GetMetric (operations::Metric::embedding_surprisal)
            .value_or (0.0);
  const double arousal = context.GetArousal ();

  const double s_eff = core::SensitivityBias (cfg.sensitivity);
  const double surprisal_thresh = core::Lerp (0.6, 0.4, s_eff);
  const double arousal_thresh = core::Lerp (0.7, 0.5, s_eff);
  const bool tag_trigger
      = (surprisal > surprisal_thresh) || (arousal > arousal_thresh);

  if (!tag_trigger)
    {
      return;
    }

  const int tag_window
      = std::max (1, static_cast<int> (std::round (core::Lerp (
                             2.0, 8.0, s_eff))));
  const int tag_decay_s
      = static_cast<int> (std::round (core::Lerp (300.0, 3600.0, cfg.stability)));
  const long long tag_expires_at
      = static_cast<long long> (now_ts + static_cast<uint64_t> (tag_decay_s) * 1000ULL);

  tx.Execute (
      "UPDATE memories SET "
      "tag_strength = MAX(tag_strength, 1.0), "
      "tag_expires_at = MAX(tag_expires_at, ?1) "
      "WHERE memory_id IN ("
      "  SELECT memory_id FROM memories "
      "  ORDER BY created_at DESC LIMIT ?2"
      ")",
      { tag_expires_at, static_cast<long long> (tag_window) });

  telemetry::LogDebug ("cortext.synaptic_tagging", {
    telemetry::Attribute::Double ("surprisal", surprisal),
    telemetry::Attribute::Double ("arousal", arousal),
    telemetry::Attribute::Int64 ("tag_window", tag_window),
    telemetry::Attribute::Int64 ("tag_expires_at", tag_expires_at)
  });
}

} // namespace cortext::operations
