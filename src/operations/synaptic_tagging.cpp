#include "cortext/operations/synaptic_tagging.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "../experimental_env.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cortext::operations
{

void
ApplySynapticTagging::Execute (OperationContext &context, Transaction &tx) const
{
  if (internal::experimental_env::Flag ("CORTEXT_DISABLE_SYNAPTIC_TAGGING"))
    {
      return;
    }

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

  const auto tag_policy = core::SynapticTaggingPolicyForKnobs (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const bool tag_trigger
      = (surprisal > tag_policy.surprisal_threshold)
        || (arousal > tag_policy.arousal_threshold);

  if (!tag_trigger)
    {
      return;
    }

  const bool tag_ttl_disabled = internal::experimental_env::Flag (
      "CORTEXT_DISABLE_SYNAPTIC_TAG_TTL");
  const long long tag_expires_at
      = tag_ttl_disabled
            ? std::numeric_limits<long long>::max ()
            : static_cast<long long> (
                now_ts
                + static_cast<uint64_t> (tag_policy.tag_decay_seconds)
                      * 1000ULL);
  const long long stored_memory_id
      = context.GetStoredMemoryId ().value_or (0);
  const long long neighbor_limit
      = std::max (0, tag_policy.tag_window - (stored_memory_id > 0 ? 1 : 0));
  const long long signal_ts = static_cast<long long> (now_ts);
  const std::string &source_id = context.GetSignal ().source_id;

  tx.Execute (
      "WITH stored(memory_id) AS ("
      "  SELECT ?2 WHERE ?2 > 0"
      "), "
      "source_neighbors(memory_id) AS ("
      "  SELECT memory_id FROM ("
      "    SELECT m.memory_id, "
      "           MIN(ABS(COALESCE(s.timestamp, m.start_ts) - ?4)) AS spike_distance, "
      "           MAX(COALESCE(s.timestamp, m.start_ts)) AS event_ts "
      "    FROM memories m "
      "    LEFT JOIN signals s ON s.memory_id = m.memory_id "
      "                       AND COALESCE(s.source_id, '') = ?3 "
      "    WHERE COALESCE(m.source_id, '') = ?3 "
      "      AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
      "      AND COALESCE(m.start_ts, 0) > 0 "
      "      AND (?2 <= 0 OR m.memory_id <> ?2) "
      "    GROUP BY m.memory_id "
      "    ORDER BY spike_distance ASC, event_ts DESC, m.memory_id DESC "
      "    LIMIT ?5"
      "  )"
      "), "
      "target_memories(memory_id) AS ("
      "  SELECT memory_id FROM stored "
      "  UNION "
      "  SELECT memory_id FROM source_neighbors"
      ") "
      "UPDATE memories SET "
      "tag_strength = MAX(tag_strength, 1.0), "
      "tag_expires_at = MAX(tag_expires_at, ?1) "
      "WHERE memory_id IN (SELECT memory_id FROM target_memories)",
      { tag_expires_at, stored_memory_id, source_id, signal_ts,
        neighbor_limit });

  telemetry::LogDebug ("cortext.synaptic_tagging", {
    telemetry::Attribute::Double ("surprisal", surprisal),
    telemetry::Attribute::Double ("arousal", arousal),
    telemetry::Attribute::Double ("surprisal_threshold",
                                  tag_policy.surprisal_threshold),
    telemetry::Attribute::Double ("arousal_threshold",
                                  tag_policy.arousal_threshold),
    telemetry::Attribute::Int64 ("tag_window", tag_policy.tag_window),
    telemetry::Attribute::Int64 ("stored_memory_id", stored_memory_id),
    telemetry::Attribute::Int64 ("source_neighbor_limit", neighbor_limit),
    telemetry::Attribute::Int64 ("tag_expires_at", tag_expires_at)
  });
}

} // namespace cortext::operations
