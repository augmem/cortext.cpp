#include "cortext/store/schema.hpp"
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace cortext::store
{

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
  // 1. Ensure migrations table exists.
  store.Execute (
      "CREATE TABLE IF NOT EXISTS cortext_schema_migrations ("
      "  id INTEGER PRIMARY KEY,"
      "  description TEXT,"
      "  applied_at INTEGER"
      ")");

  // 2. Get already applied migrations.
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

  // 3. Register core schema (always present).
  // We make a copy of registry to append core migrations, or modify the function sig.
  // Better: ApplyMigrations takes a registry, and the caller (SignalProcessor)
  // is responsible for populating it. But we need to ensure core schema is there.
  // Let's modify ApplyMigrations to automatically include core schema.
  
  SchemaRegistry full_registry = registry;
  RegisterCoreSchema(full_registry);

  // 4. Apply pending migrations in order.
  auto migrations = full_registry.GetSortedMigrations ();
  for (const auto &m : migrations)
    {
      if (applied_ids.count (m.id))
        {
          continue;
        }

      // Run migration in a transaction.
      auto tx = store.Begin ();
      try
        {
          for (const auto &sql : m.up_statements)
            {
              // Handle conditional logic for virtual tables or columns here if needed,
              // or just execute raw SQL.
              // For robustness, we catch errors on individual statements if they are "soft" fails,
              // but typically a migration failure should halt.
              try 
              {
                  tx->Execute (sql);
              }
              catch (const std::exception& e)
              {
                  // Special case: objstore vtab might fail if extension not loaded.
                  // We log/ignore? For now, we let it bubble up unless it is critically optional.
                  // If "objstore" migration fails, it usually means extension missing.
                  // We'll rethrow for now to be safe.
                  throw;
              }
            }

          // Record as applied.
          tx->Execute (
              "INSERT INTO cortext_schema_migrations (id, description, applied_at) "
              "VALUES (?, ?, strftime('%s', 'now'))",
              { m.id, m.description });

          tx->Commit ();
        }
      catch (const std::exception &e)
        {
          tx->Rollback ();
          // Log failure
          std::cerr << "Migration " << m.id << " (" << m.description << ") failed: " << e.what() << std::endl;
          throw; 
        }
    }
}

} // namespace cortext::store
