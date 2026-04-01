#pragma once

#include "cortext/store/store.hpp"
#include <string>

namespace cortext::store
{

/// @brief Applies pending schema migrations to the store.
///
/// This function ensures the database schema is up to date with the core
/// schema defined in schema.cpp. It creates the migration tracking table
/// if needed and applies any pending migrations.
///
/// @param store The store to migrate.
void ApplyMigrations (Store &store);

// --- Helpers for idempotent DDL (useful inside migrations) ---

/// @brief Checks if a table exists.
bool TableExists (Store &store, const std::string &table_name);

/// @brief Checks if a column exists in a table.
bool ColumnExists (Store &store, const std::string &table_name,
                   const std::string &column_name);

} // namespace cortext::store
