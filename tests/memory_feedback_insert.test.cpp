// tests/memory_feedback_insert.test.cpp
// Tests for memory_feedback INSERT patterns and NOT NULL constraint handling
#include <catch2/catch_test_macros.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;

namespace
{

void
CreateMemoryFeedbackTable (Store &store)
{
  store.Execute ("CREATE TABLE IF NOT EXISTS memory_feedback ("
                 "embedding_id INTEGER PRIMARY KEY,"
                 "retrieved_count INTEGER NOT NULL DEFAULT 0,"
                 "used_count INTEGER NOT NULL DEFAULT 0,"
                 "contextual_gain REAL NOT NULL DEFAULT 0.0,"
                 "use_frequency REAL NOT NULL DEFAULT 0.0,"
                 "last_used INTEGER NOT NULL DEFAULT 0,"
                 "strength REAL NOT NULL DEFAULT 1.0,"
                 "original_embedding BLOB,"
                 "lability_state REAL NOT NULL DEFAULT 0.0,"
                 "lability_ts INTEGER"
                 ")",
                 {});
}

} // namespace

TEST_CASE ("INSERT OR IGNORE sets strength to default value",
           "[memory_feedback]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  CreateMemoryFeedbackTable (*store);

  // Use the INSERT OR IGNORE pattern that's in the buggy code
  store->Execute ("INSERT OR IGNORE INTO memory_feedback (embedding_id) "
                  "VALUES (?)",
                  { 1LL });

  // Query and verify strength is set to default 1.0, not NULL
  auto rows = store->Execute (
      "SELECT strength FROM memory_feedback WHERE embedding_id = ?", { 1LL });
  REQUIRE (rows.size () == 1);
  const double strength = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (strength == 1.0);
}

TEST_CASE ("BUG REPRO: INSERT OR IGNORE silently fails without DEFAULT",
           "[memory_feedback][bug]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create schema with NOT NULL but WITHOUT DEFAULT - simulates misconfigured
  // schema or schema created differently
  store->Execute ("CREATE TABLE memory_feedback ("
                  "embedding_id INTEGER PRIMARY KEY,"
                  "strength REAL NOT NULL"  // No DEFAULT clause!
                  ")",
                  {});

  // INSERT OR IGNORE silently fails when constraint is violated
  // It doesn't throw - it just doesn't insert the row!
  REQUIRE_NOTHROW (store->Execute (
      "INSERT OR IGNORE INTO memory_feedback (embedding_id) VALUES (?)",
      { 1LL }));

  // Verify NO row was created - this is the bug!
  auto rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memory_feedback WHERE embedding_id = ?",
      { 1LL });
  REQUIRE (rows.size () == 1);
  // BUG: Row was not created because INSERT OR IGNORE silently failed
  REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 0LL);

  // Now subsequent UPDATE won't find any row to update
  // This doesn't fail, but it also doesn't do anything useful
  REQUIRE_NOTHROW (store->Execute (
      "UPDATE memory_feedback SET strength = 0.5 WHERE embedding_id = ?",
      { 1LL }));

  // Still no row
  rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memory_feedback WHERE embedding_id = ?",
      { 1LL });
  REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 0LL);
}

TEST_CASE ("BUG REPRO: Explicit INSERT pattern works without DEFAULT clause",
           "[memory_feedback][bug]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create schema with NOT NULL but WITHOUT DEFAULT
  store->Execute ("CREATE TABLE memory_feedback ("
                  "embedding_id INTEGER PRIMARY KEY,"
                  "strength REAL NOT NULL"
                  ")",
                  {});

  // The explicit INSERT pattern from memory_strength.cpp works because it
  // provides the value for strength explicitly
  REQUIRE_NOTHROW (store->Execute (
      "INSERT INTO memory_feedback (embedding_id, strength) "
      "SELECT ?, 1.0 "
      "WHERE NOT EXISTS (SELECT 1 FROM memory_feedback WHERE embedding_id = ?)",
      { 1LL, 1LL }));

  // Verify row was created
  auto rows = store->Execute (
      "SELECT strength FROM memory_feedback WHERE embedding_id = ?", { 1LL });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength")) == 1.0);
}

TEST_CASE ("UPDATE strength succeeds after INSERT OR IGNORE",
           "[memory_feedback]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  CreateMemoryFeedbackTable (*store);

  // Insert with INSERT OR IGNORE
  store->Execute ("INSERT OR IGNORE INTO memory_feedback (embedding_id) "
                  "VALUES (?)",
                  { 2LL });

  // UPDATE that references existing strength - should not throw
  REQUIRE_NOTHROW (store->Execute (
      "UPDATE memory_feedback SET strength = MAX(0.0, strength - ?) "
      "WHERE embedding_id = ?",
      { 0.1, 2LL }));

  // Verify strength was updated
  auto rows = store->Execute (
      "SELECT strength FROM memory_feedback WHERE embedding_id = ?", { 2LL });
  REQUIRE (rows.size () == 1);
  const double strength = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (strength == 0.9);
}

TEST_CASE ("UPDATE with NULL strength demonstrates the problem",
           "[memory_feedback][regression]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create a modified schema without NOT NULL on strength to simulate legacy
  // rows
  store->Execute ("CREATE TABLE memory_feedback ("
                  "  embedding_id INTEGER PRIMARY KEY,"
                  "  retrieved_count INTEGER NOT NULL DEFAULT 0,"
                  "  used_count INTEGER NOT NULL DEFAULT 0,"
                  "  contextual_gain REAL NOT NULL DEFAULT 0.0,"
                  "  use_frequency REAL NOT NULL DEFAULT 0.0,"
                  "  last_used INTEGER NOT NULL DEFAULT 0,"
                  "  strength REAL,"  // No NOT NULL, no DEFAULT
                  "  original_embedding BLOB,"
                  "  lability_state REAL NOT NULL DEFAULT 0.0,"
                  "  lability_ts INTEGER"
                  ")",
                  {});

  // Insert row with explicit NULL strength (simulating legacy/corrupted data)
  store->Execute ("INSERT INTO memory_feedback (embedding_id, strength) "
                  "VALUES (?, NULL)",
                  { 3LL });

  // Verify NULL was inserted
  auto rows = store->Execute (
      "SELECT strength FROM memory_feedback WHERE embedding_id = ?", { 3LL });
  REQUIRE (rows.size () == 1);
  // strength should be NULL (nullptr in std::any)
  REQUIRE (rows[0].at ("strength").type () == typeid (std::nullptr_t));

  // UPDATE that references strength when it's NULL
  // NULL - 0.1 = NULL, and MAX(0.0, NULL) = 0.0 in SQLite
  store->Execute (
      "UPDATE memory_feedback SET strength = MAX(0.0, strength - ?) "
      "WHERE embedding_id = ?",
      { 0.1, 3LL });

  // Verify strength was updated (result depends on SQLite behavior)
  rows = store->Execute (
      "SELECT strength FROM memory_feedback WHERE embedding_id = ?", { 3LL });
  REQUIRE (rows.size () == 1);
  // After MAX(0.0, NULL) the result should be 0.0
  // If it's still NULL, that's the type, otherwise it's a double
  const auto &strength_any = rows[0].at ("strength");
  if (strength_any.type () == typeid (double))
    {
      REQUIRE (std::any_cast<double> (strength_any) == 0.0);
    }
  else
    {
      // This would indicate the computation kept NULL
      // Either way the test documents the behavior
      REQUIRE (strength_any.type () == typeid (std::nullptr_t));
    }
}

TEST_CASE ("UPDATE fails with NOT NULL constraint when strength is NULL",
           "[memory_feedback][regression]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create schema with NOT NULL on strength but allow inserting NULL via
  // trigger bypass
  store->Execute ("CREATE TABLE memory_feedback ("
                  "  embedding_id INTEGER PRIMARY KEY,"
                  "  strength REAL NOT NULL DEFAULT 1.0"
                  ")",
                  {});

  // We cannot directly insert NULL into NOT NULL column, but we can
  // demonstrate that if someone tried to UPDATE with a NULL result, it would
  // fail This simulates what happens when computation produces NULL

  store->Execute ("INSERT INTO memory_feedback (embedding_id) VALUES (?)",
                  { 4LL });

  // Try to set strength to NULL explicitly - should throw
  REQUIRE_THROWS (store->Execute (
      "UPDATE memory_feedback SET strength = NULL WHERE embedding_id = ?",
      { 4LL }));
}

TEST_CASE ("Explicit INSERT with all columns sets strength correctly",
           "[memory_feedback]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  CreateMemoryFeedbackTable (*store);

  // Use the explicit INSERT pattern from memory_strength.cpp
  store->Execute (
      "INSERT INTO memory_feedback "
      "(embedding_id, strength, retrieved_count, used_count, "
      "contextual_gain, use_frequency, last_used, lability_state) "
      "SELECT ?, 1.0, 0, 0, 0.0, 0.0, 0, 0.0 "
      "WHERE NOT EXISTS (SELECT 1 FROM memory_feedback WHERE embedding_id = ?)",
      { 5LL, 5LL });

  // Verify row was created with correct values
  auto rows = store->Execute (
      "SELECT strength, retrieved_count, used_count FROM memory_feedback "
      "WHERE embedding_id = ?",
      { 5LL });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength")) == 1.0);
  REQUIRE (std::any_cast<long long> (rows[0].at ("retrieved_count")) == 0LL);
  REQUIRE (std::any_cast<long long> (rows[0].at ("used_count")) == 0LL);
}

TEST_CASE ("Explicit INSERT pattern is idempotent", "[memory_feedback]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  CreateMemoryFeedbackTable (*store);

  // Insert first time
  store->Execute (
      "INSERT INTO memory_feedback "
      "(embedding_id, strength, retrieved_count, used_count, "
      "contextual_gain, use_frequency, last_used, lability_state) "
      "SELECT ?, 1.0, 0, 0, 0.0, 0.0, 0, 0.0 "
      "WHERE NOT EXISTS (SELECT 1 FROM memory_feedback WHERE embedding_id = ?)",
      { 6LL, 6LL });

  // Modify strength
  store->Execute ("UPDATE memory_feedback SET strength = 0.5 "
                  "WHERE embedding_id = ?",
                  { 6LL });

  // Try to insert again - should not overwrite
  store->Execute (
      "INSERT INTO memory_feedback "
      "(embedding_id, strength, retrieved_count, used_count, "
      "contextual_gain, use_frequency, last_used, lability_state) "
      "SELECT ?, 1.0, 0, 0, 0.0, 0.0, 0, 0.0 "
      "WHERE NOT EXISTS (SELECT 1 FROM memory_feedback WHERE embedding_id = ?)",
      { 6LL, 6LL });

  // Strength should still be 0.5, not reset to 1.0
  auto rows = store->Execute (
      "SELECT strength FROM memory_feedback WHERE embedding_id = ?", { 6LL });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength")) == 0.5);
}
