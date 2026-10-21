#pragma once

// Schema versioning internals shared by the baseline (schema.cpp) and the
// legacy upgrade chain (schema_legacy_migrations.cpp).
//
// A store's version is its set of applied migration ids in
// cortext_schema_migrations. An empty store gets the baseline schema
// directly: the schema the legacy chain 0..kBaselineVersion produces, with
// every legacy id recorded as applied. A store created before the baseline
// still runs the legacy migrations it has not applied, in id order. Either
// way, migrations with ids above kBaselineVersion then run on top.

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

/// @brief The newest migration id the baseline schema stands for.
inline constexpr int64_t kBaselineVersion = 58;

/// @brief Migrations 0..kBaselineVersion, kept so stores created before the
/// baseline upgrade in place. Fresh stores never run them.
std::vector<Migration> LegacyMigrations ();

} // namespace cortext::store::schema_internal
