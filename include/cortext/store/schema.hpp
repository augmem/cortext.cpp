#pragma once

#include "cortext/store/store.hpp"
#include <string>

namespace cortext::store
{

/// @brief Brings the store's schema up to date.
///
/// An empty store gets the baseline schema (src/store/schema.cpp) directly.
/// A store created earlier runs the legacy migrations it has not applied
/// (src/store/schema_legacy_migrations.cpp). Both end at the same schema,
/// then run any newer migrations. The pass holds the write lock throughout.
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
