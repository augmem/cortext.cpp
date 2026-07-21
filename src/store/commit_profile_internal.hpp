#pragma once

#include "cortext/store/store.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cortext::internal
{

/// @brief Profiling-only decomposition of the most recent root SQLite commit.
struct SQLiteCommitProfile
{
  bool available = false;
  double sqlite_commit_ms = 0.0;
  double mutation_audit_finalize_ms = 0.0;
  std::int64_t cache_write_count = 0;
  std::int64_t cache_spill_count = 0;
  std::vector<std::pair<std::string, std::int64_t>> table_row_counts;
};

/// @brief Consume the last root-commit profile for this store.
SQLiteCommitProfile ConsumeSQLiteCommitProfile (Store *store);

} // namespace cortext::internal
