#include "cortext/store/schema.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace cortext::store
{

namespace
{

/// @brief Reads the set of already-applied migration IDs from the tracking table.
///
/// Migration IDs are monotonically increasing integers (e.g., YYYYMMDDnn format)
/// that must be unique across all registered migrations to prevent collisions.
std::set<int>
GetAppliedMigrations (Store &store)
{
  std::set<int> applied_ids;
  auto rows = store.Execute ("SELECT id FROM cortext_schema_migrations");
  for (const auto &row : rows)
    {
      if (row.count ("id"))
        {
          auto val = row.at ("id");
          if (val.type () == typeid (int))
            applied_ids.insert (std::any_cast<int> (val));
          else if (val.type () == typeid (long long))
            applied_ids.insert (static_cast<int> (std::any_cast<long long> (val)));
        }
    }
  return applied_ids;
}

void
ApplySingleMigration (Store &store, const Migration &m)
{
  auto tx = store.Begin ();
  try
    {
      for (const auto &sql : m.up_statements)
        {
          try
            {
              tx->Execute (sql);
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
      tx->Execute (
          "INSERT INTO cortext_schema_migrations (id, description, applied_at) "
          "VALUES (?, ?, strftime('%s', 'now'))",
          { m.id, m.description });
      tx->Commit ();
    }
  catch (const std::exception &e)
    {
      tx->Rollback ();
      telemetry::LogError (
          "Schema migration failed",
          { telemetry::Attribute::String ("component", "store.schema"),
            telemetry::Attribute::Int64 ("migration_id", m.id),
            telemetry::Attribute::String ("migration_description", m.description),
            telemetry::Attribute::String ("error", e.what ()) });
      throw;
    }
}

} // namespace

void
SchemaRegistry::Register (Migration migration)
{
  migrations_.push_back (std::move (migration));
}

std::vector<Migration>
SchemaRegistry::GetSortedMigrations () const
{
  std::vector<Migration> sorted = migrations_;
  std::sort (sorted.begin (), sorted.end (),
             [] (const Migration &a, const Migration &b) {
               return a.id < b.id;
             });
  return sorted;
}

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
           if (it->second.type() == typeid(std::string)) {
               if (std::any_cast<std::string>(it->second) == column_name) return true;
           }
        }
    }
  return false;
}

void
RegisterCoreSchema (SchemaRegistry &registry)
{
  registry.Register ({
      1,
      "Core tables (memory_index, memory_feedback, embeddings)",
      {
          // Embeddings table (vector store)
          "CREATE TABLE IF NOT EXISTS embeddings ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  embedding BLOB"
          ")",

          // Memory index (metadata + payload pointer)
          "CREATE TABLE IF NOT EXISTS memory_index ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  modality TEXT,"
          "  mime TEXT,"
          "  content_key TEXT,"
          "  source_id TEXT,"
          "  timestamp INTEGER,"
          "  width INTEGER,"
          "  height INTEGER,"
          "  channels INTEGER,"
          "  sample_rate INTEGER,"
          "  num_samples INTEGER,"
          "  blob_id BLOB"
          ")",

          // Memory feedback (metrics/learning)
          "CREATE TABLE IF NOT EXISTS memory_feedback ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  retrieved_count INTEGER NOT NULL DEFAULT 0,"
          "  used_count INTEGER NOT NULL DEFAULT 0,"
          "  contextual_gain REAL NOT NULL DEFAULT 0.0,"
          "  use_frequency REAL NOT NULL DEFAULT 0.0,"
          "  last_used INTEGER NOT NULL DEFAULT 0,"
          "  strength REAL NOT NULL DEFAULT 1.0,"
          "  original_embedding BLOB,"
          "  lability_state REAL NOT NULL DEFAULT 0.0,"
          "  lability_ts INTEGER"
          ")",
      },
  });

  registry.Register ({
      2,
      "Ensure objstore virtual table",
      {
          // Only works if extension is loaded; harmless if fails usually, but
          // better to wrap in try-catch or check module existence if possible.
          // For now, we assume if extension is missing, this will fail safely or
          // be caught by the migration runner loop.
          // Note: CREATE VIRTUAL TABLE IF NOT EXISTS is supported in newer SQLite.
          "CREATE VIRTUAL TABLE IF NOT EXISTS objstore USING objstore()",
      },
  });
}

void
ApplyMigrations (Store &store, const SchemaRegistry &registry)
{
  store.Execute (
      "CREATE TABLE IF NOT EXISTS cortext_schema_migrations ("
      "  id INTEGER PRIMARY KEY,"
      "  description TEXT,"
      "  applied_at INTEGER"
      ")");
  std::set<int> applied_ids = GetAppliedMigrations (store);
  SchemaRegistry full_registry = registry;
  RegisterCoreSchema(full_registry);
  auto migrations = full_registry.GetSortedMigrations ();
  for (const auto &m : migrations)
    {
      if (applied_ids.count (m.id))
        {
          continue;
        }
      ApplySingleMigration (store, m);
    }
}

} // namespace cortext::store
