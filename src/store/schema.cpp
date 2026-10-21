#include "cortext/store/schema.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "schema_internal.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace cortext::store
{

namespace
{

using schema_internal::Migration;

/// @brief Reads the set of already-applied migration IDs from the tracking table.
///
/// Migration IDs are monotonically increasing integers (e.g., YYYYMMDDnn format)
/// that must be unique across all registered migrations to prevent collisions.
std::set<int64_t>
GetAppliedMigrations (Store &store)
{
  std::set<int64_t> applied_ids;
  auto rows = store.Execute ("SELECT id FROM cortext_schema_migrations");
  for (const auto &row : rows)
    {
      if (row.count ("id"))
        {
          auto val = row.at ("id");
          if (val.type () == typeid (int))
            applied_ids.insert (std::any_cast<int> (val));
          else if (val.type () == typeid (long long))
            applied_ids.insert (std::any_cast<long long> (val));
        }
    }
  return applied_ids;
}

void
ApplySingleMigrationInTransaction (Store &store, const Migration &m)
{
  try
    {
      for (const auto &sql : m.up_statements)
        {
          try
            {
              store.Execute (sql);
            }
          catch (const std::exception &e)
            {
              telemetry::LogError (
                  "Migration statement failed",
                  { telemetry::Attribute::String ("component", "store.schema"),
                    telemetry::Attribute::Int64 ("migration_id", m.id),
                    telemetry::Attribute::String ("migration_description", m.description),
                    telemetry::Attribute::String ("sql", sql),
                    telemetry::Attribute::String ("error", e.what ()) });
              throw;
            }
        }
      store.Execute (
          "INSERT INTO cortext_schema_migrations (id, description, applied_at) "
          "VALUES (?, ?, strftime('%s', 'now'))",
          { m.id, m.description });
    }
  catch (const std::exception &e)
    {
      telemetry::LogError (
          "Schema migration failed",
          { telemetry::Attribute::String ("component", "store.schema"),
            telemetry::Attribute::Int64 ("migration_id", m.id),
            telemetry::Attribute::String ("migration_description", m.description),
            telemetry::Attribute::String ("error", e.what ()) });
      throw;
    }
}

void
ApplyMigrationsThrough (Store &store, int64_t maximum_id)
{
  // Migrations must own the write lock before reading the applied set. Use
  // direct SQL so this does not depend on the configurable default transaction
  // mode used by normal processing transactions.
  store.Execute ("BEGIN IMMEDIATE");
  bool committed = false;

  try
    {
      store.Execute (
          "CREATE TABLE IF NOT EXISTS cortext_schema_migrations ("
          "  id INTEGER PRIMARY KEY,"
          "  description TEXT,"
          "  applied_at INTEGER"
          ")");

      std::set<int64_t> applied_ids = GetAppliedMigrations (store);
      auto migrations = schema_internal::LegacyMigrations ();

      std::sort (migrations.begin (), migrations.end (),
                 [] (const Migration &a, const Migration &b) {
                   return a.id < b.id;
                 });

      for (const auto &m : migrations)
        {
          if (m.id > maximum_id || applied_ids.count (m.id))
            {
              continue;
            }
          ApplySingleMigrationInTransaction (store, m);
        }

      const auto memory_text_fts = store.Execute (
          "SELECT 1 FROM sqlite_master "
          "WHERE type = 'table' AND name = 'memory_text_fts' LIMIT 1");
      if (!memory_text_fts.empty ())
        {
          // Connection-local helpers for keyword queries: a scratch FTS5
          // table with the index's tokenizer maps query words to index terms,
          // and vocabularies give those terms' document frequencies
          // (memory_text_index_internal::Search).
          store.Execute (
              "CREATE VIRTUAL TABLE IF NOT EXISTS temp.memory_query_terms "
              "USING fts5(text, content='',"
              "  tokenize='porter unicode61 remove_diacritics 2')");
          store.Execute (
              "CREATE VIRTUAL TABLE IF NOT EXISTS temp.memory_query_term_stems "
              "USING fts5vocab(temp, memory_query_terms, 'instance')");
          store.Execute (
              "CREATE VIRTUAL TABLE IF NOT EXISTS temp.memory_text_vocab "
              "USING fts5vocab(main, memory_text_fts, 'row')");
        }

      store.Execute ("COMMIT");
      committed = true;
    }
  catch (...)
    {
      if (!committed)
        {
          try
            {
              store.Execute ("ROLLBACK");
            }
          catch (...)
            {
            }
        }
      throw;
    }
}

} // namespace

#if defined(CORTEXT_TESTING)
std::set<int64_t>
DebugGetAppliedMigrationIdsForTest (Store &store)
{
  return GetAppliedMigrations (store);
}

void
DebugApplyCoreMigrationsThroughForTest (Store &store, int64_t maximum_id)
{
  ApplyMigrationsThrough (store, maximum_id);
}
#endif

bool
TableExists (Store &store, const std::string &table_name)
{
  auto rows = store.Execute (
      "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
      { table_name });
  return !rows.empty ();
}

bool
ColumnExists (Store &store, const std::string &table_name,
              const std::string &column_name)
{
  // Quote the table identifier to prevent injection
  std::string quoted_table = "\"";
  for (char c : table_name)
    {
      if (c == '"')
        quoted_table += "\"\"";
      else
        quoted_table += c;
    }
  quoted_table += "\"";

  // PRAGMA table_info returns one row per column.
  // Columns: cid, name, type, notnull, dflt_value, pk
  auto rows
      = store.Execute ("PRAGMA table_info(" + quoted_table + ")", {});
  for (const auto &row : rows)
    {
      auto it = row.find ("name");
      if (it != row.end ())
        {
          // Check for string or text type in the variant/any
          if (it->second.type () == typeid (std::string))
            {
              if (std::any_cast<std::string> (it->second) == column_name)
                return true;
            }
        }
    }
  return false;
}

void
ApplyMigrations (Store &store)
{
  ApplyMigrationsThrough (store, std::numeric_limits<int64_t>::max ());
}

} // namespace cortext::store
