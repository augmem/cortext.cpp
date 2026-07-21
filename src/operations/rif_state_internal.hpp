#pragma once

#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cortext::operations::rif_state_internal
{

constexpr double kActiveSuppressionFloor = 1e-9;

struct Clock
{
  long long generation = 1;
  double log_factor = 0.0;
  long long last_ts = 0;
};

struct RecoveryResult
{
  std::size_t expired_rows = 0;
  std::size_t retired_rows = 0;
  std::size_t maximum_statement_rows = 0;
  bool generation_reset = false;
  std::vector<long long> changed_memory_ids;
  Clock clock;
};

inline std::string
Placeholders (std::size_t count)
{
  std::string out;
  out.reserve (count * 2);
  for (std::size_t index = 0; index < count; ++index)
    {
      if (index > 0)
        out += ",";
      out += "?";
    }
  return out;
}

inline long long
GetInt64 (const std::map<std::string, std::any> &row,
          const std::string &key, long long fallback = 0)
{
  const auto it = row.find (key);
  if (it == row.end ())
    return fallback;
  return store::AnyToLongLong (it->second).value_or (fallback);
}

inline double
GetDouble (const std::map<std::string, std::any> &row,
           const std::string &key, double fallback = 0.0)
{
  const auto it = row.find (key);
  if (it == row.end ())
    return fallback;
  return store::AnyToDouble (it->second, fallback);
}

template <typename Executor>
Clock
LoadClock (Executor &executor)
{
  const auto rows = executor.Execute (
      "SELECT generation, log_factor, last_ts "
      "FROM rif_recovery_clock WHERE singleton = 1");
  if (rows.size () != 1)
    throw StoreError ("lazy RIF recovery clock is missing");
  return {
    GetInt64 (rows[0], "generation", 1),
    GetDouble (rows[0], "log_factor", 0.0),
    GetInt64 (rows[0], "last_ts", 0),
  };
}

template <typename Executor>
std::size_t
CountRows (Executor &tx, const std::string &query,
           const std::vector<std::any> &params = {})
{
  const auto rows = tx.Execute (query, params);
  if (rows.empty ())
    return 0;
  const auto count = GetInt64 (rows[0], "row_count", 0);
  return count > 0 ? static_cast<std::size_t> (count) : 0;
}

inline void
MaterializeWhere (Transaction &tx, const std::string &where_sql,
                  const std::vector<std::any> &params)
{
  tx.Execute (
      "UPDATE memories "
      "SET strength = COALESCE(("
      "      SELECT e.strength FROM rif_effective_memories e "
      "      WHERE e.memory_id = memories.memory_id), strength), "
      "    suppression = COALESCE(("
      "      SELECT e.suppression FROM rif_effective_memories e "
      "      WHERE e.memory_id = memories.memory_id), suppression), "
      "    suppression_ts = COALESCE(("
      "      SELECT e.suppression_ts FROM rif_effective_memories e "
      "      WHERE e.memory_id = memories.memory_id), suppression_ts) "
      "WHERE memory_id IN (SELECT memory_id FROM rif_active_state WHERE "
          + where_sql + ")",
      params);
}

template <typename Executor>
void
PruneRetiredGenerationResets (Executor &executor)
{
  executor.Execute (
      "DELETE FROM rif_generation_resets "
      "WHERE generation < ("
      "  SELECT generation FROM rif_recovery_clock WHERE singleton = 1) "
      "  AND NOT EXISTS(SELECT 1 FROM rif_active_state a "
      "                 WHERE a.generation = rif_generation_resets.generation)");
}

inline std::vector<long long>
MaterializeRetiredBatch (Transaction &tx, long long current_generation,
                         std::size_t row_batch_size)
{
  if (row_batch_size == 0)
    throw std::invalid_argument ("RIF row batch is zero");
  const auto rows = tx.Execute (
      "SELECT memory_id FROM rif_active_state "
      "WHERE generation < ? ORDER BY generation, memory_id LIMIT ?",
      { current_generation, static_cast<long long> (row_batch_size) });
  std::vector<long long> memory_ids;
  memory_ids.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const long long memory_id = GetInt64 (row, "memory_id", 0);
      if (memory_id > 0)
        memory_ids.push_back (memory_id);
    }
  if (memory_ids.empty ())
    {
      PruneRetiredGenerationResets (tx);
      return {};
    }
  std::vector<std::any> ids;
  ids.reserve (memory_ids.size ());
  for (const long long memory_id : memory_ids)
    ids.push_back (memory_id);
  const std::string placeholders = Placeholders (memory_ids.size ());
  tx.Execute (
      "UPDATE memories "
      "SET strength = (SELECT e.strength FROM rif_effective_memories e "
      "                WHERE e.memory_id = memories.memory_id), "
      "    suppression = (SELECT e.suppression "
      "                   FROM rif_effective_memories e "
      "                   WHERE e.memory_id = memories.memory_id), "
      "    suppression_ts = (SELECT e.suppression_ts "
      "                      FROM rif_effective_memories e "
      "                      WHERE e.memory_id = memories.memory_id) "
      "WHERE memory_id IN (" + placeholders + ")",
      ids);
  tx.Execute ("DELETE FROM rif_active_state WHERE memory_id IN ("
                  + placeholders + ")",
              ids);
  PruneRetiredGenerationResets (tx);
  return memory_ids;
}

inline std::vector<long long>
LoadMemoryIdsWhere (Transaction &tx, const std::string &where_sql,
                    const std::vector<std::any> &params)
{
  const auto rows = tx.Execute (
      "SELECT memory_id FROM rif_active_state WHERE " + where_sql
          + " ORDER BY memory_id",
      params);
  std::vector<long long> memory_ids;
  memory_ids.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const long long memory_id = GetInt64 (row, "memory_id", 0);
      if (memory_id > 0)
        memory_ids.push_back (memory_id);
    }
  return memory_ids;
}

inline std::vector<long long>
RecoverMemoryIds (Transaction &tx, const std::vector<long long> &memory_ids,
                  long long now_ts, double recovery_time,
                  std::size_t row_batch_size)
{
  if (row_batch_size == 0)
    throw std::invalid_argument ("RIF row batch is zero");
  for (std::size_t begin = 0; begin < memory_ids.size ();
       begin += row_batch_size)
    {
      const std::size_t end
          = std::min (memory_ids.size (), begin + row_batch_size);
      std::vector<std::any> params;
      params.reserve (11 + end - begin);
      params.insert (params.end (), {
        now_ts, now_ts, recovery_time, now_ts, recovery_time,
        now_ts, now_ts, recovery_time, now_ts, recovery_time, now_ts
      });
      for (std::size_t index = begin; index < end; ++index)
        params.push_back (memory_ids[index]);
      tx.Execute (
          "UPDATE memories SET "
          "strength = MAX(0.0, strength + suppression * (CASE "
          "  WHEN (? - COALESCE(suppression_ts, 0)) <= 0 THEN 0.0 "
          "  WHEN (? - COALESCE(suppression_ts, 0)) >= ? THEN 1.0 "
          "  ELSE ((? - COALESCE(suppression_ts, 0)) * 1.0 / ?) END)), "
          "suppression = MAX(0.0, suppression - suppression * (CASE "
          "  WHEN (? - COALESCE(suppression_ts, 0)) <= 0 THEN 0.0 "
          "  WHEN (? - COALESCE(suppression_ts, 0)) >= ? THEN 1.0 "
          "  ELSE ((? - COALESCE(suppression_ts, 0)) * 1.0 / ?) END)), "
          "suppression_ts = ? WHERE memory_id IN ("
              + Placeholders (end - begin) + ")",
          params);
    }
  return memory_ids;
}

inline void
UpsertActiveMemoryIdsAtClock (Transaction &tx,
                              const std::vector<long long> &memory_ids,
                              std::size_t row_batch_size)
{
  if (memory_ids.empty ())
    return;
  if (row_batch_size == 0)
    throw std::invalid_argument ("RIF row batch is zero");
  for (std::size_t begin = 0; begin < memory_ids.size ();
       begin += row_batch_size)
    {
      const std::size_t end
          = std::min (memory_ids.size (), begin + row_batch_size);
      std::vector<std::any> ids;
      ids.reserve (end - begin);
      for (std::size_t index = begin; index < end; ++index)
        ids.push_back (memory_ids[index]);
      tx.Execute (
          "INSERT OR REPLACE INTO rif_active_state("
          "memory_id, generation, anchor_suppression, recovery_total, "
          "anchor_log_factor, expires_log_factor) "
          "SELECT m.memory_id, c.generation, m.suppression, "
          "       m.strength + m.suppression, c.log_factor, "
          "       c.log_factor + ln(1e-9 / m.suppression) "
          "FROM memories m CROSS JOIN rif_recovery_clock c "
          "WHERE c.singleton = 1 AND m.memory_id IN ("
              + Placeholders (end - begin) + ") "
          "  AND m.suppression > 1e-9",
          ids);
    }
}

inline std::vector<long long>
CalibrateMemoryIdsAtClock (Transaction &tx,
                           const std::vector<long long> &memory_ids,
                           double recovery_time,
                           std::size_t row_batch_size)
{
  if (memory_ids.empty ())
    return {};
  if (row_batch_size == 0)
    throw std::invalid_argument ("RIF row batch is zero");
  const Clock clock = LoadClock (tx);
  for (std::size_t begin = 0; begin < memory_ids.size ();
       begin += row_batch_size)
    {
      const std::size_t end
          = std::min (memory_ids.size (), begin + row_batch_size);
      std::vector<std::any> ids;
      ids.reserve (end - begin);
      for (std::size_t index = begin; index < end; ++index)
        ids.push_back (memory_ids[index]);
      tx.Execute ("DELETE FROM rif_active_state WHERE memory_id IN ("
                      + Placeholders (end - begin) + ")",
                  ids);
    }
  RecoverMemoryIds (
      tx, memory_ids, clock.last_ts, recovery_time, row_batch_size);
  UpsertActiveMemoryIdsAtClock (tx, memory_ids, row_batch_size);
  return memory_ids;
}

inline RecoveryResult
AdvanceRecovery (Transaction &tx, long long now_ts, double recovery_time,
                 std::size_t row_batch_size)
{
  if (row_batch_size == 0)
    throw std::invalid_argument ("RIF row batch is zero");
  const Clock before = LoadClock (tx);
  const double elapsed
      = static_cast<double> (now_ts - before.last_ts);
  const bool has_current_active = CountRows (
      tx,
      "SELECT COUNT(*) AS row_count FROM ("
      "SELECT 1 FROM rif_active_state WHERE generation = ? LIMIT 1)",
      { before.generation })
                                  > 0;
  const bool full_recovery = has_current_active
                             && (recovery_time <= 0.0
                                 || elapsed >= recovery_time);

  if (!has_current_active)
    {
      tx.Execute (
          "UPDATE rif_recovery_clock SET log_factor = 0.0, last_ts = ? "
          "WHERE singleton = 1",
          { now_ts });
      auto retired = MaterializeRetiredBatch (
          tx, before.generation, row_batch_size);
      return { 0, retired.size (), retired.size (), false,
               std::move (retired),
               { before.generation, 0.0, now_ts } };
    }

  if (full_recovery)
    {
      tx.Execute (
          "INSERT INTO rif_generation_resets(generation, reset_ts) "
          "VALUES(?, ?) "
          "ON CONFLICT(generation) DO UPDATE SET reset_ts = excluded.reset_ts",
          { before.generation, now_ts });
      tx.Execute (
          "UPDATE rif_recovery_clock "
          "SET generation = generation + 1, log_factor = 0.0, last_ts = ? "
          "WHERE singleton = 1",
          { now_ts });
      auto retired = MaterializeRetiredBatch (
          tx, before.generation + 1, row_batch_size);
      return { 0, retired.size (), retired.size (), true,
               std::move (retired),
               { before.generation + 1, 0.0, now_ts } };
    }

  double next_log_factor = before.log_factor;
  if (elapsed > 0.0)
    {
      const double fraction = std::clamp (
          elapsed / recovery_time, 0.0,
          std::nextafter (1.0, 0.0));
      next_log_factor += std::log1p (-fraction);
    }
  tx.Execute (
      "UPDATE rif_recovery_clock SET log_factor = ?, last_ts = ? "
      "WHERE singleton = 1",
      { next_log_factor, now_ts });

  const std::vector<std::any> due_params {
    before.generation, next_log_factor,
    static_cast<long long> (row_batch_size)
  };
  const auto expired_rows = tx.Execute (
      "SELECT memory_id FROM rif_active_state "
      "WHERE generation = ? AND expires_log_factor >= ? "
      "ORDER BY expires_log_factor DESC, memory_id LIMIT ?",
      due_params);
  std::vector<long long> expired;
  expired.reserve (expired_rows.size ());
  for (const auto &row : expired_rows)
    {
      const long long memory_id = GetInt64 (row, "memory_id", 0);
      if (memory_id > 0)
        expired.push_back (memory_id);
    }
  if (!expired.empty ())
    {
      std::vector<std::any> expired_params;
      expired_params.reserve (expired.size ());
      for (const long long memory_id : expired)
        expired_params.push_back (memory_id);
      const std::string placeholders = Placeholders (expired.size ());
      MaterializeWhere (
          tx, "memory_id IN (" + placeholders + ")", expired_params);
      tx.Execute ("DELETE FROM rif_active_state WHERE memory_id IN ("
                      + placeholders + ")",
                  expired_params);
    }
  auto retired = MaterializeRetiredBatch (
      tx, before.generation, row_batch_size);
  const std::size_t retired_count = retired.size ();
  std::vector<long long> changed = expired;
  changed.insert (changed.end (), retired.begin (), retired.end ());
  return { expired.size (), retired_count,
           std::max (expired.size (), retired_count), false,
           std::move (changed),
           { before.generation, next_log_factor, now_ts } };
}

inline RecoveryResult
AdvanceRecovery (Transaction &tx, long long now_ts, double recovery_time,
                 const std::vector<long long> &calibration_memory_ids,
                 std::size_t row_batch_size)
{
  if (calibration_memory_ids.empty ())
    return AdvanceRecovery (tx, now_ts, recovery_time, row_batch_size);
  if (row_batch_size == 0)
    throw std::invalid_argument ("RIF row batch is zero");
  for (std::size_t begin = 0; begin < calibration_memory_ids.size ();
       begin += row_batch_size)
    {
      const std::size_t end = std::min (
          calibration_memory_ids.size (), begin + row_batch_size);
      std::vector<std::any> calibration_params;
      calibration_params.reserve (end - begin);
      for (std::size_t index = begin; index < end; ++index)
        calibration_params.push_back (calibration_memory_ids[index]);
      tx.Execute ("DELETE FROM rif_active_state WHERE memory_id IN ("
                      + Placeholders (end - begin) + ")",
                  calibration_params);
    }
  auto result = AdvanceRecovery (
      tx, now_ts, recovery_time, row_batch_size);
  const auto calibrated = RecoverMemoryIds (
      tx, calibration_memory_ids, now_ts, recovery_time, row_batch_size);
  UpsertActiveMemoryIdsAtClock (tx, calibrated, row_batch_size);
  result.maximum_statement_rows = std::max (
      result.maximum_statement_rows,
      std::min (calibration_memory_ids.size (), row_batch_size));
  result.changed_memory_ids.insert (result.changed_memory_ids.end (),
                                    calibrated.begin (), calibrated.end ());
  return result;
}

inline std::vector<long long>
ResolveMemoryIds (Transaction &tx, long long memory_id, long long embedding_id)
{
  if (memory_id > 0)
    return { memory_id };
  if (embedding_id <= 0)
    return {};
  const auto rows = tx.Execute (
      "SELECT memory_id FROM memories WHERE embedding_id = ? "
      "ORDER BY memory_id",
      { embedding_id });
  std::vector<long long> resolved;
  resolved.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const long long resolved_id = GetInt64 (row, "memory_id", 0);
      if (resolved_id > 0)
        resolved.push_back (resolved_id);
    }
  return resolved;
}

inline bool
SuppressMemory (Transaction &tx, long long memory_id, double amount,
                long long now_ts)
{
  if (memory_id <= 0 || amount <= std::numeric_limits<double>::epsilon ())
    return false;
  const bool already_active = !tx.Execute (
      "SELECT 1 FROM rif_active_state a "
      "JOIN rif_recovery_clock c ON c.generation = a.generation "
      "WHERE c.singleton = 1 AND a.memory_id = ? LIMIT 1",
      { memory_id }).empty ();
  tx.Execute (
      "UPDATE memories "
      "SET strength = MAX(0.0, COALESCE(("
      "      SELECT e.strength FROM rif_effective_memories e "
      "      WHERE e.memory_id = memories.memory_id), strength) - ?), "
      "    suppression = COALESCE(("
      "      SELECT e.suppression FROM rif_effective_memories e "
      "      WHERE e.memory_id = memories.memory_id), suppression) + ?, "
      "    suppression_ts = ?, "
      "    suppression_count = suppression_count + 1 "
      "WHERE memory_id = ?",
      { amount, amount, now_ts, memory_id });
  tx.Execute (
      "INSERT OR REPLACE INTO rif_active_state("
      "memory_id, generation, anchor_suppression, recovery_total, "
      "anchor_log_factor, expires_log_factor) "
      "SELECT m.memory_id, c.generation, m.suppression, "
      "       m.strength + m.suppression, c.log_factor, "
      "       c.log_factor + ln(1e-9 / m.suppression) "
      "FROM memories m CROSS JOIN rif_recovery_clock c "
      "WHERE c.singleton = 1 AND m.memory_id = ? "
      "  AND m.suppression > 1e-9",
      { memory_id });
  return !already_active;
}

inline void
RefreshActiveStrengthWhere (Transaction &tx, const std::string &where_sql,
                            const std::vector<std::any> &params)
{
  tx.Execute (
      "UPDATE rif_active_state "
      "SET recovery_total = ("
      "  SELECT m.strength + rif_active_state.anchor_suppression * exp("
      "           c.log_factor - rif_active_state.anchor_log_factor) "
      "  FROM memories m CROSS JOIN rif_recovery_clock c "
      "  WHERE c.singleton = 1 "
      "    AND m.memory_id = rif_active_state.memory_id) "
      "WHERE memory_id IN (SELECT memory_id FROM memories WHERE "
          + where_sql + ")",
      params);
}

template <typename Executor>
void
MaterializeAllAndClear (Executor &executor)
{
  executor.Execute (
      "UPDATE memories "
      "SET strength = COALESCE(("
      "      SELECT e.strength FROM rif_effective_memories e "
      "      WHERE e.memory_id = memories.memory_id), strength), "
      "    suppression = COALESCE(("
      "      SELECT e.suppression FROM rif_effective_memories e "
      "      WHERE e.memory_id = memories.memory_id), suppression), "
      "    suppression_ts = COALESCE(("
      "      SELECT e.suppression_ts FROM rif_effective_memories e "
      "      WHERE e.memory_id = memories.memory_id), suppression_ts) "
      "WHERE memory_id IN (SELECT memory_id FROM rif_active_state)");
  executor.Execute ("DELETE FROM rif_active_state");
  PruneRetiredGenerationResets (executor);
}

template <typename Executor>
void
RebuildFromMaterialized (Executor &executor)
{
  executor.Execute ("DELETE FROM rif_active_state");
  executor.Execute (
      "UPDATE rif_recovery_clock SET log_factor = 0.0, "
      "last_ts = COALESCE((SELECT MAX(suppression_ts) FROM memories "
      "                    WHERE suppression > 1e-9), last_ts) "
      "WHERE singleton = 1");
  executor.Execute (
      "INSERT OR REPLACE INTO rif_active_state("
      "memory_id, generation, anchor_suppression, recovery_total, "
      "anchor_log_factor, expires_log_factor) "
      "SELECT m.memory_id, c.generation, m.suppression, "
      "       m.strength + m.suppression, c.log_factor, "
      "       c.log_factor + ln(1e-9 / m.suppression) "
      "FROM memories m CROSS JOIN rif_recovery_clock c "
      "WHERE m.suppression > 1e-9");
  PruneRetiredGenerationResets (executor);
}

} // namespace cortext::operations::rif_state_internal
