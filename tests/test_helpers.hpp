#pragma once

#include "cortext/store/schema.hpp"
#include "cortext/store/store.hpp"
#include <any>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cortext::testing
{

/// @brief Initialize the core schema on a store using ApplyMigrations.
/// This ensures tests use the same schema as production code.
inline void
InitializeCoreSchema (Store &store)
{
  cortext::store::SchemaRegistry registry;
  cortext::store::ApplyMigrations (store, registry);
}

/// @brief A no-op transaction for tests that don't need actual database writes.
class NullTransaction : public Transaction
{
public:
  std::unique_ptr<Transaction>
  Begin () override
  {
    return std::make_unique<NullTransaction> ();
  }

  std::vector<std::map<std::string, std::any>>
  Execute (const std::string & /*query*/,
           const std::vector<std::any> & /*params*/ = {}) override
  {
    return {};
  }

  void
  Commit () override
  {
  }

  void
  Rollback () override
  {
  }
};

/// @brief Get a global NullTransaction instance for tests.
inline Transaction &
GetNullTransaction ()
{
  static NullTransaction null_tx;
  return null_tx;
}

} // namespace cortext::testing
