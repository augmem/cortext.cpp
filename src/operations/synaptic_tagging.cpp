#include "cortext/operations/synaptic_tagging.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "../experimental_env.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace cortext::operations
{

namespace
{

void
AppendMemoryIds (const std::vector<std::map<std::string, std::any>> &rows,
                 std::unordered_set<long long> &memory_ids)
{
  for (const auto &row : rows)
    {
      auto it = row.find ("memory_id");
      if (it == row.end ())
        {
          continue;
        }
      const auto memory_id = store::AnyToLongLong (it->second);
      if (memory_id && *memory_id > 0)
        {
          memory_ids.insert (*memory_id);
        }
    }
}

void
AppendNearestSignalMemories (
    Transaction &tx, const std::string &source_id, long long signal_ts,
    long long stored_memory_id, long long limit, bool before,
    std::unordered_set<long long> &memory_ids)
{
  std::vector<long long> side_memory_ids;
  side_memory_ids.reserve (static_cast<std::size_t> (limit));
  std::optional<long long> boundary_ts;
  for (long long i = 0; i < limit; ++i)
    {
      std::string exclusions;
      std::vector<std::any> params{
        source_id, signal_ts, source_id, stored_memory_id
      };
      for (const long long memory_id : side_memory_ids)
        {
          if (!exclusions.empty ())
            {
              exclusions += ",";
            }
          exclusions += "?";
          params.push_back (memory_id);
        }
      std::string sql
          = "SELECT s.memory_id, s.timestamp FROM signals s "
            "INDEXED BY idx_signals_source_ts_serial "
            "JOIN memories m ON m.memory_id = s.memory_id "
            "WHERE s.source_id = ? AND s.timestamp ";
      sql += before ? "<= ? " : ">= ? ";
      sql += "AND s.memory_id IS NOT NULL "
             "AND COALESCE(m.source_id, '') = ? "
             "AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
             "AND COALESCE(m.start_ts, 0) > 0 "
             "AND (? <= 0 OR m.memory_id <> ?) ";
      params.insert (params.begin () + 4, stored_memory_id);
      if (!exclusions.empty ())
        {
          sql += "AND s.memory_id NOT IN (" + exclusions + ") ";
        }
      sql += before
                 ? "ORDER BY s.timestamp DESC, s.serial_position DESC, "
                   "s.signal_id DESC LIMIT 1"
                 : "ORDER BY s.timestamp ASC, s.serial_position ASC, "
                   "s.signal_id ASC LIMIT 1";
      const auto rows = tx.Execute (sql, params);
      if (rows.empty ())
        {
          break;
        }
      const auto memory_id
          = store::AnyToLongLong (rows[0].at ("memory_id"));
      const auto timestamp = store::AnyToLongLong (rows[0].at ("timestamp"));
      if (!memory_id || *memory_id <= 0 || !timestamp)
        {
          break;
        }
      side_memory_ids.push_back (*memory_id);
      memory_ids.insert (*memory_id);
      boundary_ts = *timestamp;
    }

  if (boundary_ts)
    {
      const auto tie_rows = tx.Execute (
          "SELECT DISTINCT s.memory_id FROM signals s "
          "INDEXED BY idx_signals_source_ts_serial "
          "JOIN memories m ON m.memory_id = s.memory_id "
          "WHERE s.source_id = ? AND s.timestamp = ? "
          "  AND s.memory_id IS NOT NULL "
          "  AND COALESCE(m.source_id, '') = ? "
          "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
          "  AND COALESCE(m.start_ts, 0) > 0 "
          "  AND (? <= 0 OR m.memory_id <> ?)",
          { source_id, *boundary_ts, source_id, stored_memory_id,
            stored_memory_id });
      AppendMemoryIds (tie_rows, memory_ids);
    }
}

void
AppendNearestMemoryStarts (
    Transaction &tx, const std::string &source_id, long long signal_ts,
    long long stored_memory_id, long long limit, bool before,
    std::unordered_set<long long> &memory_ids)
{
  std::string sql
      = "SELECT memory_id, start_ts FROM memories "
        "INDEXED BY idx_memories_source_start "
        "WHERE source_id = ? AND start_ts ";
  sql += before ? "<= ? " : ">= ? ";
  sql += "AND kind IN ('LONG_TERM', 'ASSOCIATION') "
         "AND COALESCE(start_ts, 0) > 0 "
         "AND (? <= 0 OR memory_id <> ?) ";
  sql += before ? "ORDER BY start_ts DESC, memory_id DESC LIMIT ?"
                : "ORDER BY start_ts ASC, memory_id DESC LIMIT ?";
  const auto rows = tx.Execute (
      sql, { source_id, signal_ts, stored_memory_id, stored_memory_id, limit });
  AppendMemoryIds (rows, memory_ids);
  if (rows.empty ())
    {
      return;
    }
  const auto boundary_ts = store::AnyToLongLong (rows.back ().at ("start_ts"));
  if (!boundary_ts)
    {
      return;
    }
  const auto tie_rows = tx.Execute (
      "SELECT memory_id FROM memories INDEXED BY idx_memories_source_start "
      "WHERE source_id = ? AND start_ts = ? "
      "  AND kind IN ('LONG_TERM', 'ASSOCIATION') "
      "  AND (? <= 0 OR memory_id <> ?)",
      { source_id, *boundary_ts, stored_memory_id, stored_memory_id });
  AppendMemoryIds (tie_rows, memory_ids);
}

std::vector<long long>
LoadTemporalNeighborIds (Transaction &tx, const std::string &source_id,
                         long long signal_ts, long long stored_memory_id,
                         long long limit)
{
  std::unordered_set<long long> candidate_ids;
  candidate_ids.reserve (static_cast<std::size_t> (limit * 6));
  AppendNearestSignalMemories (tx, source_id, signal_ts, stored_memory_id,
                               limit, true, candidate_ids);
  AppendNearestSignalMemories (tx, source_id, signal_ts, stored_memory_id,
                               limit, false, candidate_ids);
  AppendNearestMemoryStarts (tx, source_id, signal_ts, stored_memory_id,
                             limit, true, candidate_ids);
  AppendNearestMemoryStarts (tx, source_id, signal_ts, stored_memory_id,
                             limit, false, candidate_ids);
  if (candidate_ids.empty ())
    {
      return {};
    }

  std::string values;
  std::vector<std::any> params;
  params.reserve (candidate_ids.size () + 3);
  for (const long long memory_id : candidate_ids)
    {
      if (!values.empty ())
        {
          values += ",";
        }
      values += "(?)";
      params.push_back (memory_id);
    }
  params.push_back (signal_ts);
  params.push_back (source_id);
  params.push_back (limit);
  const auto rows = tx.Execute (
      "WITH candidate(memory_id) AS (VALUES " + values + ") "
      "SELECT m.memory_id, "
      "       MIN(ABS(COALESCE(s.timestamp, m.start_ts) - ?)) "
      "         AS spike_distance, "
      "       MAX(COALESCE(s.timestamp, m.start_ts)) AS event_ts "
      "FROM candidate c "
      "JOIN memories m ON m.memory_id = c.memory_id "
      "LEFT JOIN signals s ON s.memory_id = m.memory_id "
      "                   AND COALESCE(s.source_id, '') = ? "
      "GROUP BY m.memory_id "
      "ORDER BY spike_distance ASC, event_ts DESC, m.memory_id DESC "
      "LIMIT ?", params);

  std::vector<long long> out;
  out.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const auto memory_id = store::AnyToLongLong (row.at ("memory_id"));
      if (memory_id && *memory_id > 0)
        {
          out.push_back (*memory_id);
        }
    }
  return out;
}

} // namespace

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

  auto target_memory_ids = LoadTemporalNeighborIds (
      tx, source_id, signal_ts, stored_memory_id, neighbor_limit);
  if (stored_memory_id > 0)
    {
      target_memory_ids.push_back (stored_memory_id);
    }
  if (!target_memory_ids.empty ())
    {
      std::string placeholders;
      std::vector<std::any> params{ tag_expires_at };
      params.reserve (target_memory_ids.size () + 1);
      for (const long long memory_id : target_memory_ids)
        {
          if (!placeholders.empty ())
            {
              placeholders += ",";
            }
          placeholders += "?";
          params.push_back (memory_id);
        }
      tx.Execute (
          "UPDATE memories SET "
          "tag_strength = MAX(tag_strength, 1.0), "
          "tag_expires_at = MAX(tag_expires_at, ?) "
          "WHERE memory_id IN (" + placeholders + ")",
          params);
    }

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
