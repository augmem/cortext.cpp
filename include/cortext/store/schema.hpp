#pragma once

#include "cortext/store/store.hpp"
#include <functional>
#include <string>
#include <vector>

namespace cortext::store
{

/// @brief Represents a single migration step.
struct Migration
{
  int id;                     // Unique integer ID (e.g., 2024010101)
  std::string description;    // Human-readable description
  std::vector<std::string> up_statements; // SQL statements to apply
};

/// @brief Registry for collecting migrations from various components.
class SchemaRegistry
{
public:
  /// @brief Register a migration.
  /// @param migration The migration to register.
  void Register (Migration migration);

  /// @brief Get all registered migrations sorted by ID.
  std::vector<Migration> GetSortedMigrations () const;

private:
  std::vector<Migration> migrations_;
};

/// @brief Applies pending migrations to the store.
/// @param store The store to migrate.
/// @param registry The registry containing all known migrations.
void ApplyMigrations (Store &store, const SchemaRegistry &registry);

// --- Helpers for idempotent DDL (useful inside migrations) ---

/// @brief Checks if a table exists.
bool TableExists (Store &store, const std::string &table_name);

/// @brief Checks if a column exists in a table.
bool ColumnExists (Store &store, const std::string &table_name,
                   const std::string &column_name);

/// @brief Returns a CREATE TABLE statement if table doesn't exist.
/// Note: prefer using full migrations, but this helper is useful for
/// strictly additive logic or virtual tables.
std::string EnsureTableSql (const std::string &table_name,
                            const std::string &schema_sql);

/// @brief Generates DDL to add a column if it doesn't exist.
/// This is a "soft" migration helper often used when a formal migration step
/// might be too heavy-handed for optional columns.
/// @return SQL statement or empty string if already exists (requires runtime check).
/// Note: In this system, prefer explicit ALTER TABLE in a versioned Migration.
/// These helpers are exposed for the migration implementation itself.

} // namespace cortext::store

