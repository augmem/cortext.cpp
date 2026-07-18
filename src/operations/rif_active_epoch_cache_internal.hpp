#pragma once

#include "cortext/store/sqlite_store.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"

#include <algorithm>
#include <any>
#include <atomic>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations::rif_active_epoch_cache_internal
{

inline constexpr std::size_t kEventLimit = 512;
inline constexpr std::size_t kMutationLimit = 32768;
inline constexpr std::size_t kAllocatedByteLimit = 64ULL * 1024ULL * 1024ULL;

struct State
{
  // The database is intentionally shared when processor rollback state is
  // snapshotted. It is read-only until the authoritative transaction commits;
  // pending publication metadata below is copied by value.
  std::shared_ptr<SQLiteStore> database;
  bool valid = false;
  long long generation = 0;
  double log_factor = 0.0;
  long long last_ts = 0;
  std::size_t active_rows = 0;
  std::size_t event_count = 0;
  std::size_t mutation_count = 0;
  std::size_t allocated_bytes = 0;
  std::vector<long long> calibration_memory_ids;
  std::vector<long long> pending_memory_ids;
  bool pending_clock = false;
  long long pending_generation = 0;
  double pending_log_factor = 0.0;
  long long pending_last_ts = 0;
  bool pending_rebuild = false;
  bool pending_rebuild_resets_counters = false;
  bool invalid_rebuild_resets_counters = true;
};

struct PublishResult
{
  bool published = false;
  bool rebuilt = false;
  bool recovered_from_failure = false;
  std::size_t changed_rows = 0;
};

inline bool
RequiresConsolidation (const State &state)
{
  return state.event_count >= kEventLimit
         || state.mutation_count >= kMutationLimit
         || state.allocated_bytes >= kAllocatedByteLimit;
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

inline std::size_t
DatabaseBytes (SQLiteStore &database)
{
  const auto page_count = database.Execute ("PRAGMA page_count");
  const auto page_size = database.Execute ("PRAGMA page_size");
  if (page_count.empty () || page_size.empty ())
    return 0;
  const long long count = GetInt64 (page_count.front (), "page_count", 0);
  const long long size = GetInt64 (page_size.front (), "page_size", 0);
  if (count <= 0 || size <= 0)
    return 0;
  return static_cast<std::size_t> (count)
         * static_cast<std::size_t> (size);
}

inline std::shared_ptr<SQLiteStore>
CreateDatabase ()
{
  auto owned = SQLiteStore::Create (":memory:");
  auto database = std::shared_ptr<SQLiteStore> (owned.release ());
  database->Execute (
      "CREATE TABLE epoch_clock("
      "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
      "generation INTEGER NOT NULL, log_factor REAL NOT NULL, "
      "last_ts INTEGER NOT NULL)");
  database->Execute (
      "CREATE TABLE active_state("
      "memory_id INTEGER PRIMARY KEY, generation INTEGER NOT NULL, "
      "anchor_suppression REAL NOT NULL, recovery_total REAL NOT NULL, "
      "anchor_log_factor REAL NOT NULL, expires_log_factor REAL NOT NULL)");
  database->Execute (
      "CREATE INDEX active_state_due ON active_state("
      "generation, expires_log_factor, memory_id)");
  return database;
}

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

inline std::vector<std::map<std::string, std::any>>
LoadCurrentRows (Store &persistent, long long generation,
                 const std::vector<long long> *memory_ids = nullptr)
{
  if (memory_ids && memory_ids->empty ())
    return {};
  std::vector<std::any> params { generation };
  std::string predicate;
  if (memory_ids && !memory_ids->empty ())
    {
      predicate = " AND memory_id IN (" + Placeholders (memory_ids->size ())
                  + ")";
      params.reserve (memory_ids->size () + 1);
      for (const long long memory_id : *memory_ids)
        params.push_back (memory_id);
    }
  return persistent.Execute (
      "SELECT memory_id, generation, anchor_suppression, recovery_total, "
      "anchor_log_factor, expires_log_factor FROM rif_active_state "
      "WHERE generation = ?" + predicate + " ORDER BY memory_id",
      params);
}

inline std::map<std::string, std::any>
LoadClock (Store &persistent)
{
  const auto rows = persistent.Execute (
      "SELECT generation, log_factor, last_ts FROM rif_recovery_clock "
      "WHERE singleton = 1");
  if (rows.size () != 1)
    throw StoreError ("lazy RIF recovery clock is missing");
  return rows.front ();
}

inline void
InsertRow (Transaction &transaction,
           const std::map<std::string, std::any> &row)
{
  transaction.Execute (
      "INSERT OR REPLACE INTO active_state("
      "memory_id, generation, anchor_suppression, recovery_total, "
      "anchor_log_factor, expires_log_factor) VALUES(?, ?, ?, ?, ?, ?)",
      { row.at ("memory_id"), row.at ("generation"),
        row.at ("anchor_suppression"), row.at ("recovery_total"),
        row.at ("anchor_log_factor"), row.at ("expires_log_factor") });
}

inline void
Rebuild (State &state, Store &persistent, bool reset_counters = true)
{
  const std::size_t prior_event_count = state.event_count;
  const std::size_t prior_mutation_count = state.mutation_count;
  const auto clock = LoadClock (persistent);
  const long long generation = GetInt64 (clock, "generation", 1);
  const auto rows = LoadCurrentRows (persistent, generation);
  const auto calibration_rows = persistent.Execute (
      "SELECT a.memory_id FROM rif_active_state a "
      "JOIN rif_recovery_clock c ON c.singleton = 1 "
      "JOIN memories m ON m.memory_id = a.memory_id "
      "WHERE a.generation = c.generation "
      "  AND a.anchor_log_factor = c.log_factor "
      "  AND COALESCE(m.suppression_ts, 0) <> c.last_ts "
      "ORDER BY a.memory_id");
  auto replacement = CreateDatabase ();
  auto transaction = replacement->Begin ();
  transaction->Execute (
      "INSERT INTO epoch_clock(singleton, generation, log_factor, last_ts) "
      "VALUES(1, ?, ?, ?)",
      { generation, clock.at ("log_factor"), clock.at ("last_ts") });
  for (const auto &row : rows)
    InsertRow (*transaction, row);
  transaction->Commit ();

  state.database = std::move (replacement);
  state.valid = true;
  state.generation = generation;
  state.log_factor = store::AnyToDouble (clock.at ("log_factor"), 0.0);
  state.last_ts = GetInt64 (clock, "last_ts", 0);
  state.active_rows = rows.size ();
  state.event_count = reset_counters ? 0 : prior_event_count;
  state.mutation_count = reset_counters ? 0 : prior_mutation_count;
  state.allocated_bytes = DatabaseBytes (*state.database);
  state.calibration_memory_ids.clear ();
  state.calibration_memory_ids.reserve (calibration_rows.size ());
  for (const auto &row : calibration_rows)
    {
      const long long memory_id = GetInt64 (row, "memory_id", 0);
      if (memory_id > 0)
        state.calibration_memory_ids.push_back (memory_id);
    }
  state.pending_memory_ids.clear ();
  state.pending_clock = false;
  state.pending_rebuild = false;
  state.pending_rebuild_resets_counters = false;
  state.invalid_rebuild_resets_counters = true;
}

inline void
Ensure (State &state, Store &persistent)
{
  if (!state.valid || !state.database)
    Rebuild (state, persistent, state.invalid_rebuild_resets_counters);
}

inline void
StageMemory (State &state, long long memory_id)
{
  if (memory_id > 0)
    state.pending_memory_ids.push_back (memory_id);
}

inline void
StageMemories (State &state, const std::vector<long long> &memory_ids)
{
  for (const long long memory_id : memory_ids)
    StageMemory (state, memory_id);
}

inline void
StageRebuild (State &state, bool reset_counters = false)
{
  state.pending_rebuild = true;
  state.pending_rebuild_resets_counters
      = state.pending_rebuild_resets_counters || reset_counters;
  state.pending_clock = true;
}

inline void
StageClock (State &state, long long generation, double log_factor,
            long long last_ts)
{
  state.pending_clock = true;
  state.pending_generation = generation;
  state.pending_log_factor = log_factor;
  state.pending_last_ts = last_ts;
}

#if defined(CORTEXT_TESTING)
inline std::atomic<unsigned int> g_publish_failure_mask { 0 };

inline void
SetPublishFailureStageForTest (int stage)
{
  const unsigned int mask
      = stage > 0 ? (1U << static_cast<unsigned int> (stage)) : 0U;
  g_publish_failure_mask.store (mask, std::memory_order_relaxed);
}

inline void
SetPublishFailureMaskForTest (unsigned int mask)
{
  g_publish_failure_mask.store (mask, std::memory_order_relaxed);
}

inline void
MaybeFailPublicationForTest (int stage)
{
  const unsigned int bit = 1U << static_cast<unsigned int> (stage);
  unsigned int observed
      = g_publish_failure_mask.load (std::memory_order_relaxed);
  while ((observed & bit) != 0U)
    {
      if (g_publish_failure_mask.compare_exchange_weak (
              observed, observed & ~bit, std::memory_order_relaxed))
        {
          throw StoreError (
              "injected RIF active-epoch publication failure");
        }
    }
}
#else
inline void
MaybeFailPublicationForTest (int)
{
}
#endif

inline PublishResult
PublishOnce (State &state, Store &persistent)
{
  if (!state.pending_clock && state.pending_memory_ids.empty ()
      && !state.pending_rebuild)
    {
      ++state.event_count;
      return { true, false, false, 0 };
    }
  if (state.pending_rebuild || !state.valid || !state.database)
    {
      const bool reset_counters
          = state.pending_rebuild
                ? state.pending_rebuild_resets_counters
                : state.invalid_rebuild_resets_counters;
      MaybeFailPublicationForTest (3);
      Rebuild (state, persistent, reset_counters);
      if (!reset_counters)
        {
          ++state.event_count;
          state.mutation_count += state.active_rows;
        }
      return { true, true, false, state.active_rows };
    }

  const long long generation
      = state.pending_clock ? state.pending_generation : state.generation;
  const double log_factor
      = state.pending_clock ? state.pending_log_factor : state.log_factor;
  const long long last_ts
      = state.pending_clock ? state.pending_last_ts : state.last_ts;
  if (generation != state.generation)
    {
      Rebuild (state, persistent, false);
      ++state.event_count;
      state.mutation_count += state.active_rows;
      return { true, true, false, state.active_rows };
    }

  std::sort (state.pending_memory_ids.begin (),
             state.pending_memory_ids.end ());
  state.pending_memory_ids.erase (
      std::unique (state.pending_memory_ids.begin (),
                   state.pending_memory_ids.end ()),
      state.pending_memory_ids.end ());
  const auto persistent_rows
      = LoadCurrentRows (persistent, generation, &state.pending_memory_ids);
  std::unordered_set<long long> present;
  present.reserve (persistent_rows.size ());
  for (const auto &row : persistent_rows)
    present.insert (GetInt64 (row, "memory_id", 0));

  std::unordered_set<long long> cached;
  cached.reserve (state.pending_memory_ids.size ());
  if (!state.pending_memory_ids.empty ())
    {
      std::vector<std::any> params;
      params.reserve (state.pending_memory_ids.size ());
      for (const long long memory_id : state.pending_memory_ids)
        params.push_back (memory_id);
      const auto cached_rows = state.database->Execute (
          "SELECT memory_id FROM active_state WHERE memory_id IN ("
              + Placeholders (state.pending_memory_ids.size ()) + ")",
          params);
      for (const auto &row : cached_rows)
        cached.insert (GetInt64 (row, "memory_id", 0));
    }

  MaybeFailPublicationForTest (1);
  auto transaction = state.database->Begin ();
  transaction->Execute (
      "UPDATE epoch_clock SET log_factor = ?, last_ts = ? "
      "WHERE singleton = 1",
      { log_factor, last_ts });
  for (const auto &row : persistent_rows)
    InsertRow (*transaction, row);
  for (const long long memory_id : state.pending_memory_ids)
    if (!present.contains (memory_id))
      transaction->Execute ("DELETE FROM active_state WHERE memory_id = ?",
                            { memory_id });
  MaybeFailPublicationForTest (2);
  transaction->Commit ();

  for (const long long memory_id : state.pending_memory_ids)
    {
      const bool was_present = cached.contains (memory_id);
      const bool is_present = present.contains (memory_id);
      if (!was_present && is_present)
        ++state.active_rows;
      else if (was_present && !is_present && state.active_rows > 0)
        --state.active_rows;
    }
  if (!state.calibration_memory_ids.empty ()
      && !state.pending_memory_ids.empty ())
    {
      const std::unordered_set<long long> changed (
          state.pending_memory_ids.begin (), state.pending_memory_ids.end ());
      state.calibration_memory_ids.erase (
          std::remove_if (state.calibration_memory_ids.begin (),
                          state.calibration_memory_ids.end (),
                          [&changed] (long long memory_id) {
                            return changed.contains (memory_id);
                          }),
          state.calibration_memory_ids.end ());
    }
  const std::size_t changed_rows = state.pending_memory_ids.size ();
  state.log_factor = log_factor;
  state.last_ts = last_ts;
  ++state.event_count;
  state.mutation_count += changed_rows + (state.pending_clock ? 1 : 0);
  if (changed_rows > 0)
    state.allocated_bytes = DatabaseBytes (*state.database);
  state.pending_memory_ids.clear ();
  state.pending_clock = false;
  state.pending_rebuild = false;
  state.pending_rebuild_resets_counters = false;
  return { true, false, false, changed_rows };
}

inline PublishResult
PublishAfterPersistentCommit (State &state, Store &persistent)
{
  try
    {
      return PublishOnce (state, persistent);
    }
  catch (...)
    {
      // Persistent SQLite has already committed. Never replay it: discard the
      // disposable epoch and rebuild only its bounded current-generation state.
      const bool reset_counters = state.pending_rebuild_resets_counters;
      const std::size_t committed_rebuild_rows = state.active_rows;
      state.database.reset ();
      state.valid = false;
      state.pending_memory_ids.clear ();
      state.calibration_memory_ids.clear ();
      state.pending_clock = false;
      state.pending_rebuild = false;
      state.pending_rebuild_resets_counters = false;
      state.invalid_rebuild_resets_counters = reset_counters;
      try
        {
          MaybeFailPublicationForTest (4);
          Rebuild (state, persistent, reset_counters);
          if (!reset_counters)
            {
              ++state.event_count;
              state.mutation_count += state.active_rows;
            }
          return { true, true, true, state.active_rows };
        }
      catch (...)
        {
          state.database.reset ();
          state.valid = false;
          state.invalid_rebuild_resets_counters = reset_counters;
          if (reset_counters)
            {
              state.event_count = 0;
              state.mutation_count = 0;
            }
          else
            {
              ++state.event_count;
              state.mutation_count += committed_rebuild_rows;
            }
          return { false, false, true, 0 };
        }
    }
}

} // namespace cortext::operations::rif_active_epoch_cache_internal
