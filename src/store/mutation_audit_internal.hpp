#pragma once

#include "cortext/store/store.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cortext::internal
{

enum class SQLiteMutationKind
{
  Memory,
  Association,
  Index,
};

/// @brief One connection-local logical mutation observed after a successful
/// root transaction commit. This private profiling surface is enabled only by
/// CORTEXT_PROFILE_CONSOLIDATION_EPOCH.
struct SQLiteMutationIdentity
{
  SQLiteMutationKind kind = SQLiteMutationKind::Memory;
  std::string table;
  std::string logical_identity;
};

struct SQLiteMutationAuditBatch
{
  std::vector<SQLiteMutationIdentity> committed_hook_identities;
  std::vector<SQLiteMutationIdentity> committed_trigger_identities;
  bool trigger_journal_ready = false;
#if defined(CORTEXT_TESTING)
  std::size_t commit_vec_lookup_count = 0;
#endif
};

/// @brief Drop schema/setup mutations and begin a fresh profiling stream.
void ResetSQLiteMutationAudit (Store *store);

/// @brief Consume mutations committed since the previous call.
SQLiteMutationAuditBatch
ConsumeSQLiteCommittedMutations (Store *store);

} // namespace cortext::internal
