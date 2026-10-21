#pragma once

// Schema migration internals shared by the runner (schema.cpp) and the
// migration list (schema_legacy_migrations.cpp).

#include <cstdint>
#include <string>
#include <vector>

namespace cortext::store::schema_internal
{

/// @brief One schema migration step.
struct Migration
{
  int64_t id;                             // Unique, increasing migration id
  std::string description;                // Recorded in the ledger
  std::vector<std::string> up_statements; // SQL statements to apply
};

/// @brief Migrations 0..58, applied in id order to every store.
std::vector<Migration> LegacyMigrations ();

} // namespace cortext::store::schema_internal
