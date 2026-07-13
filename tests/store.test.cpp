#include "test_helpers.hpp"

// tests/store.test.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include <algorithm>
#include <any>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

class UnsupportedBindParam
{
};

// Helper function to create a temporary database file
std::string
create_temp_db ()
{
  return cortext::testing::UniqueTempPath ("test_store_", ".db").string ();
}

TEST_CASE ("SQLiteStore rejects bind parameter count mismatches",
           "[store][binding][regression]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  REQUIRE_THROWS_WITH (
      store->Execute ("SELECT ? AS value", { 1LL, 2LL }),
      Catch::Matchers::ContainsSubstring ("parameter count mismatch"));
  REQUIRE_THROWS_WITH (
      store->Execute ("SELECT ? AS value", {}),
      Catch::Matchers::ContainsSubstring ("parameter count mismatch"));
}

// Helper function to clean up temporary database file
void
cleanup_temp_db (const std::string &db_path)
{
  try
    {
      std::filesystem::remove (db_path);
      std::filesystem::remove (db_path + "-wal");
      std::filesystem::remove (db_path + "-shm");
    }
  catch (const std::filesystem::filesystem_error &)
    {
      // Ignore cleanup errors in tests
    }
}

// RAII wrapper for temporary database
class TempDatabase
{
public:
  TempDatabase () : db_path_ (create_temp_db ()) {}
  ~TempDatabase () { cleanup_temp_db (db_path_); }

  const std::string &
  path () const
  {
    return db_path_;
  }

private:
  std::string db_path_;
};

} // namespace

namespace cortext::internal
{

class SQLiteStoreStatementCacheInspector
{
public:
  static std::size_t
  CacheSize (const SQLiteStore &store)
  {
    return store.statement_cache_.size ();
  }

  static std::size_t
  FifoSize (const SQLiteStore &store)
  {
    return store.statement_cache_fifo_.size ();
  }

  static std::size_t
  Capacity ()
  {
    return SQLiteStore::kStatementCacheCapacity;
  }

  static bool
  CacheContains (const SQLiteStore &store, const std::string &query)
  {
    return store.statement_cache_.find (query)
           != store.statement_cache_.end ();
  }

  static bool
  FifoContains (const SQLiteStore &store, const std::string &query)
  {
    return std::find (store.statement_cache_fifo_.begin (),
                      store.statement_cache_fifo_.end (), query)
           != store.statement_cache_fifo_.end ();
  }

  static int
  SetLengthLimit (SQLiteStore &store, int value)
  {
    return sqlite3_limit (store.connection_->GetConnection (),
                          SQLITE_LIMIT_LENGTH, value);
  }
};

} // namespace cortext::internal

TEST_CASE ("SQLiteStore propagates SQLite bind failures",
           "[store][binding][regression]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  const int old_limit
      = cortext::internal::SQLiteStoreStatementCacheInspector::SetLengthLimit (
          *store, 32);
  REQUIRE_THROWS_WITH (
      store->Execute ("SELECT ? AS value", { std::string (128, 'x') }),
      Catch::Matchers::ContainsSubstring ("Failed to bind SQLite parameter"));
  cortext::internal::SQLiteStoreStatementCacheInspector::SetLengthLimit (
      *store, old_limit);
}

TEST_CASE ("Store any numeric double conversion", "[store][utils]")
{
  REQUIRE (cortext::store::AnyToDouble (std::any (1.5)).value () == 1.5);
  REQUIRE (cortext::store::AnyToDouble (std::any (1.5f)).value () == 1.5);
  REQUIRE (cortext::store::AnyToDouble (std::any (7)).value () == 7.0);
  REQUIRE (cortext::store::AnyToDouble (std::any (7LL)).value () == 7.0);

  REQUIRE_FALSE (
      cortext::store::AnyToDouble (std::any (std::string ("7"))).has_value ());
  REQUIRE (cortext::store::AnyToDouble (std::any (), 9.0) == 9.0);
}

TEST_CASE ("SQLiteStore creation", "[store]")
{
  SECTION ("Create in-memory database")
  {
    auto store = cortext::SQLiteStore::Create (":memory:");
    REQUIRE (store != nullptr);
  }

  SECTION ("Create temporary database file")
  {
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());
    REQUIRE (store != nullptr);

    // Test that file was created
    REQUIRE (std::filesystem::exists (temp_db.path ()));
  }
}

TEST_CASE ("Store basic operations", "[store]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");

  SECTION ("Execute SELECT query")
  {
    // Create a test table
    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)",
                    {});

    // Insert data
    store->Execute ("INSERT INTO test (name) VALUES (?)", { "Alice" });
    store->Execute ("INSERT INTO test (name) VALUES (?)", { "Bob" });

    // Query data
    auto results
        = store->Execute ("SELECT id, name FROM test ORDER BY id", {});
    REQUIRE (results.size () == 2);

    REQUIRE (std::any_cast<long long> (results[0].at ("id")) == 1LL);
    REQUIRE (std::any_cast<std::string> (results[0].at ("name")) == "Alice");
    REQUIRE (std::any_cast<long long> (results[1].at ("id")) == 2LL);
    REQUIRE (std::any_cast<std::string> (results[1].at ("name")) == "Bob");
  }

  SECTION ("Execute INSERT/UPDATE/DELETE")
  {
    // Create table
    store->Execute ("CREATE TABLE counter (value INTEGER)", {});

    // Insert
    auto insert_result
        = store->Execute ("INSERT INTO counter (value) VALUES (?)", { 42 });
    REQUIRE (insert_result.empty ()); // INSERT returns no rows

    // Update
    auto update_result
        = store->Execute ("UPDATE counter SET value = ?", { 100 });
    REQUIRE (update_result.empty ()); // UPDATE returns no rows

    // Verify update
    auto select_result = store->Execute ("SELECT value FROM counter", {});
    REQUIRE (select_result.size () == 1);
    REQUIRE (std::any_cast<long long> (select_result[0].at ("value"))
             == 100LL);

    // Delete
    auto delete_result = store->Execute ("DELETE FROM counter", {});
    REQUIRE (delete_result.empty ()); // DELETE returns no rows

    // Verify deletion
    auto empty_result
        = store->Execute ("SELECT COUNT(*) as count FROM counter", {});
    REQUIRE (std::any_cast<long long> (empty_result[0].at ("count")) == 0LL);
  }

  SECTION ("Execute with multiple parameters")
  {
    store->Execute ("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)",
                    {});

    // Insert with multiple parameters
    store->Execute ("INSERT INTO users (id, name, age) VALUES (?, ?, ?)",
                    { 1LL, "Alice", 25LL });

    auto result = store->Execute ("SELECT * FROM users WHERE id = ?", { 1LL });
    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<std::string> (result[0].at ("name")) == "Alice");
    REQUIRE (std::any_cast<long long> (result[0].at ("age")) == 25LL);
  }
}

TEST_CASE ("Transaction management", "[store]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");

  SECTION ("Basic transaction commit")
  {
    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
                    {});

    // Start transaction
    auto transaction = store->Begin ();
    REQUIRE (transaction != nullptr);

    // Execute within transaction
    transaction->Execute ("INSERT INTO test (value) VALUES (?)",
                          { "test_value" });

    // Commit transaction
    transaction->Commit ();

    // Verify data was committed
    auto results = store->Execute ("SELECT COUNT(*) as count FROM test", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("count")) == 1LL);
  }

  SECTION ("Transaction rollback")
  {
    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
                    {});

    // Start transaction
    auto transaction = store->Begin ();
    transaction->Execute ("INSERT INTO test (value) VALUES (?)",
                          { "should_be_rolled_back" });

    // Verify data exists within transaction
    auto mid_transaction
        = transaction->Execute ("SELECT COUNT(*) as count FROM test", {});
    REQUIRE (std::any_cast<long long> (mid_transaction[0].at ("count"))
             == 1LL);

    // Rollback transaction
    transaction->Rollback ();

    // Verify data was rolled back
    auto results = store->Execute ("SELECT COUNT(*) as count FROM test", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("count")) == 0LL);
  }

  SECTION ("Nested transactions with savepoints")
  {
    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
                    {});

    // Outer transaction
    auto outer_tx = store->Begin ();
    outer_tx->Execute ("INSERT INTO test (value) VALUES (?)", { "outer" });

    // Inner transaction
    auto inner_tx = outer_tx->Begin ();
    inner_tx->Execute ("INSERT INTO test (value) VALUES (?)", { "inner" });

    // Verify both insertions visible
    auto results
        = inner_tx->Execute ("SELECT COUNT(*) as count FROM test", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("count")) == 2LL);

    // Rollback inner transaction
    inner_tx->Rollback ();

    // Verify only outer insertion remains
    results = outer_tx->Execute ("SELECT COUNT(*) as count FROM test", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("count")) == 1LL);

    // Commit outer transaction
    outer_tx->Commit ();

    // Verify final state
    results = store->Execute ("SELECT value FROM test", {});
    REQUIRE (results.size () == 1);
    REQUIRE (std::any_cast<std::string> (results[0].at ("value")) == "outer");
  }

  SECTION ("Multiple sequential transactions")
  {
    store->Execute ("CREATE TABLE counter (value INTEGER)", {});
    store->Execute ("INSERT INTO counter VALUES (0)", {});

    // First transaction
    auto tx1 = store->Begin ();
    tx1->Execute ("UPDATE counter SET value = value + 1", {});
    tx1->Commit ();

    // Second transaction
    auto tx2 = store->Begin ();
    tx2->Execute ("UPDATE counter SET value = value + 1", {});
    tx2->Commit ();

    // Verify final value
    auto result = store->Execute ("SELECT value FROM counter", {});
    REQUIRE (std::any_cast<long long> (result[0].at ("value")) == 2LL);
  }
}

TEST_CASE ("Error handling", "[store]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");

  SECTION ("Invalid SQL syntax")
  {
    REQUIRE_THROWS_AS (store->Execute ("INVALID SQL QUERY", {}),
                       cortext::StoreError);
  }

  SECTION ("Transaction already finished error")
  {
    auto transaction = store->Begin ();
    transaction->Commit ();

    REQUIRE_THROWS_AS (transaction->Commit (),
                       cortext::TransactionAlreadyFinishedError);

    REQUIRE_THROWS_AS (transaction->Rollback (),
                       cortext::TransactionAlreadyFinishedError);

    REQUIRE_THROWS_AS (transaction->Execute ("SELECT 1", {}),
                       cortext::TransactionAlreadyFinishedError);

    REQUIRE_THROWS_AS (transaction->Begin (),
                       cortext::TransactionAlreadyFinishedError);
  }

  SECTION ("Store-level commit invalidates transaction handle")
  {
    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY)", {});
    auto transaction = store->Begin ();
    transaction->Execute ("INSERT INTO test (id) VALUES (1)", {});

    store->Commit ();

    REQUIRE_THROWS_AS (transaction->Execute ("SELECT 1", {}),
                       cortext::TransactionAlreadyFinishedError);
    REQUIRE_THROWS_AS (transaction->Commit (),
                       cortext::TransactionAlreadyFinishedError);
    REQUIRE_THROWS_AS (transaction->Rollback (),
                       cortext::TransactionAlreadyFinishedError);
    REQUIRE_THROWS_AS (transaction->Begin (),
                       cortext::TransactionAlreadyFinishedError);
  }

  SECTION ("Store-level rollback invalidates transaction handle")
  {
    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY)", {});
    auto transaction = store->Begin ();
    transaction->Execute ("INSERT INTO test (id) VALUES (1)", {});

    store->Rollback ();

    REQUIRE_THROWS_AS (transaction->Execute ("SELECT 1", {}),
                       cortext::TransactionAlreadyFinishedError);
    REQUIRE_THROWS_AS (transaction->Commit (),
                       cortext::TransactionAlreadyFinishedError);
    REQUIRE_THROWS_AS (transaction->Rollback (),
                       cortext::TransactionAlreadyFinishedError);
    REQUIRE_THROWS_AS (transaction->Begin (),
                       cortext::TransactionAlreadyFinishedError);
  }

  SECTION ("No active transaction error")
  {
    REQUIRE_THROWS_AS (store->Commit (), cortext::NoActiveTransactionError);

    REQUIRE_THROWS_AS (store->Rollback (), cortext::NoActiveTransactionError);
  }

  SECTION ("Constraint violation")
  {
    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY)", {});
    store->Execute ("INSERT INTO test (id) VALUES (1)", {});

    REQUIRE_THROWS_AS (store->Execute ("INSERT INTO test (id) VALUES (1)", {}),
                       cortext::StoreError);
  }

  SECTION ("Nested transaction handles must finish in stack order")
  {
    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
                    {});

    auto outer = store->Begin ();
    outer->Execute ("INSERT INTO test (value) VALUES (?)", { "outer" });
    auto inner = outer->Begin ();
    inner->Execute ("INSERT INTO test (value) VALUES (?)", { "inner" });

    REQUIRE_THROWS_AS (
        outer->Execute ("INSERT INTO test (value) VALUES (?)",
                        { "still_inner_scoped" }),
        cortext::TransactionNotCurrentError);
    REQUIRE_THROWS_AS (outer->Begin (), cortext::TransactionNotCurrentError);
    REQUIRE_THROWS_AS (outer->Commit (),
                       cortext::TransactionNotCurrentError);
    REQUIRE_THROWS_AS (outer->Rollback (),
                       cortext::TransactionNotCurrentError);

    inner->Rollback ();
    outer->Execute ("INSERT INTO test (value) VALUES (?)", { "after_inner" });
    outer->Commit ();

    auto rows = store->Execute ("SELECT value FROM test ORDER BY id", {});
    REQUIRE (rows.size () == 2);
    REQUIRE (std::any_cast<std::string> (rows[0].at ("value")) == "outer");
    REQUIRE (std::any_cast<std::string> (rows[1].at ("value"))
             == "after_inner");
  }
}

TEST_CASE ("Store-level nested rollback releases savepoint", "[store]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
                  {});

  auto outer = store->Begin ();
  outer->Execute ("INSERT INTO test (value) VALUES (?)", { "outer" });
  auto inner = outer->Begin ();
  inner->Execute ("INSERT INTO test (value) VALUES (?)", { "inner" });

  store->Rollback ();

  REQUIRE_THROWS_AS (inner->Execute ("SELECT 1", {}),
                     cortext::TransactionAlreadyFinishedError);
  REQUIRE_THROWS_AS (inner->Begin (),
                     cortext::TransactionAlreadyFinishedError);
  outer->Execute ("INSERT INTO test (value) VALUES (?)", { "after_inner" });
  outer->Commit ();

  auto rows = store->Execute ("SELECT value FROM test ORDER BY rowid", {});
  REQUIRE (rows.size () == 2);
  REQUIRE (std::any_cast<std::string> (rows[0].at ("value")) == "outer");
  REQUIRE (std::any_cast<std::string> (rows[1].at ("value"))
           == "after_inner");
}

TEST_CASE ("Store cleanup", "[store]")
{
  SECTION ("Store closes properly")
  {
    TempDatabase temp_db;
    {
      auto store = cortext::SQLiteStore::Create (temp_db.path ());
      store->Execute ("CREATE TABLE test (id INTEGER)", {});
      store->Execute ("INSERT INTO test (id) VALUES (1)", {});
      // store goes out of scope and should close properly
    }

    // Verify we can reopen and see the data
    auto store2 = cortext::SQLiteStore::Create (temp_db.path ());
    auto results = store2->Execute ("SELECT COUNT(*) as count FROM test", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("count")) == 1LL);
  }
}

TEST_CASE ("Data type handling", "[store]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");

  SECTION ("Integer types")
  {
    store->Execute ("CREATE TABLE numbers (small INTEGER, big INTEGER)", {});

    // Test various integer sizes
    store->Execute ("INSERT INTO numbers (small, big) VALUES (?, ?)",
                    { 42, 9223372036854775807LL });

    auto result = store->Execute ("SELECT small, big FROM numbers", {});
    REQUIRE (std::any_cast<long long> (result[0].at ("small")) == 42LL);
    REQUIRE (std::any_cast<long long> (result[0].at ("big"))
             == 9223372036854775807LL);
  }

  SECTION ("Text handling")
  {
    store->Execute ("CREATE TABLE texts (content TEXT)", {});

    std::string test_string = "Hello, SQLite with special chars: éñü";
    store->Execute ("INSERT INTO texts (content) VALUES (?)", { test_string });

    auto result = store->Execute ("SELECT content FROM texts", {});
    REQUIRE (std::any_cast<std::string> (result[0].at ("content"))
             == test_string);
  }

  SECTION ("NULL values")
  {
    store->Execute ("CREATE TABLE nullable (value INTEGER)", {});
    store->Execute ("INSERT INTO nullable (value) VALUES (?)", { nullptr });

    auto result = store->Execute ("SELECT value FROM nullable", {});
    REQUIRE (result[0].find ("value") != result[0].end ());
    // NULL values should be stored as nullptr in the any
    REQUIRE (result[0].at ("value").type () == typeid (std::nullptr_t));
  }

  SECTION ("Unsupported bind parameter types fail loudly")
  {
    store->Execute ("CREATE TABLE strict_params (value INTEGER)", {});

    REQUIRE_THROWS_AS (
        store->Execute ("INSERT INTO strict_params (value) VALUES (?)",
                        { UnsupportedBindParam {} }),
        cortext::StoreError);

    auto count = store->Execute (
        "SELECT COUNT(*) AS cnt FROM strict_params", {});
    REQUIRE (std::any_cast<long long> (count[0].at ("cnt")) == 0LL);

    store->Execute ("INSERT INTO strict_params (value) VALUES (?)", { 7LL });
    auto rows = store->Execute ("SELECT value FROM strict_params", {});
    REQUIRE (std::any_cast<long long> (rows[0].at ("value")) == 7LL);
  }

  SECTION ("Boolean-like values")
  {
    store->Execute ("CREATE TABLE flags (flag INTEGER)", {});

    // SQLite uses 0/1 for boolean-like behavior
    store->Execute ("INSERT INTO flags (flag) VALUES (?)", { 1LL });
    store->Execute ("INSERT INTO flags (flag) VALUES (?)", { 0LL });

    auto results = store->Execute ("SELECT flag FROM flags ORDER BY flag", {});
    REQUIRE (results.size () == 2);
    REQUIRE (std::any_cast<long long> (results[0].at ("flag")) == 0LL);
    REQUIRE (std::any_cast<long long> (results[1].at ("flag")) == 1LL);
  }
}

TEST_CASE ("SQLiteStore statement cache FIFO drops failed statements",
           "[store][statement_cache]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  store->Execute ("CREATE TABLE cache_test (id INTEGER PRIMARY KEY)", {});

  const std::string insert_sql
      = "INSERT INTO cache_test (id) VALUES (?)";
  store->Execute (insert_sql, { 1LL });
  REQUIRE (cortext::internal::SQLiteStoreStatementCacheInspector::CacheContains (
      *store, insert_sql));

  REQUIRE_THROWS_AS (store->Execute (insert_sql, { 1LL }),
                     cortext::StoreError);
  REQUIRE_FALSE (
      cortext::internal::SQLiteStoreStatementCacheInspector::CacheContains (
          *store, insert_sql));
  REQUIRE_FALSE (
      cortext::internal::SQLiteStoreStatementCacheInspector::FifoContains (
          *store, insert_sql));

  for (std::size_t i = 0;
       i < cortext::internal::SQLiteStoreStatementCacheInspector::Capacity ()
               + 8;
       ++i)
    {
      store->Execute ("SELECT " + std::to_string (i) + " AS value", {});
    }

  REQUIRE (cortext::internal::SQLiteStoreStatementCacheInspector::CacheSize (
               *store)
           <= cortext::internal::SQLiteStoreStatementCacheInspector::Capacity ());
  REQUIRE (cortext::internal::SQLiteStoreStatementCacheInspector::FifoSize (
               *store)
           == cortext::internal::SQLiteStoreStatementCacheInspector::CacheSize (
               *store));
}

TEST_CASE ("Concurrent operations", "[store]")
{
  // Test that the store handles multiple operations correctly
  auto store = cortext::SQLiteStore::Create (":memory:");

  store->Execute (
      "CREATE TABLE concurrent_test (id INTEGER PRIMARY KEY, data TEXT)", {});

  // Simulate concurrent-like operations
  const int num_operations = 10;
  for (int i = 0; i < num_operations; ++i)
    {
      auto tx = store->Begin ();
      tx->Execute ("INSERT INTO concurrent_test (data) VALUES (?)",
                   { "operation_" + std::to_string (i) });
      tx->Commit ();
    }

  auto results
      = store->Execute ("SELECT COUNT(*) as count FROM concurrent_test", {});
  REQUIRE (std::any_cast<long long> (results[0].at ("count"))
           == num_operations);
}

TEST_CASE ("Large dataset handling", "[store]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");

  SECTION ("Multiple result rows")
  {
    store->Execute (
        "CREATE TABLE large_test (id INTEGER PRIMARY KEY, data TEXT)", {});

    // Insert many rows
    const int num_rows = 100;
    for (int i = 0; i < num_rows; ++i)
      {
        store->Execute ("INSERT INTO large_test (data) VALUES (?)",
                        { "row_" + std::to_string (i) });
      }

    auto results
        = store->Execute ("SELECT COUNT(*) as count FROM large_test", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("count")) == num_rows);

    // Test pagination-like query
    auto paged_results = store->Execute (
        "SELECT id, data FROM large_test WHERE id <= ? ORDER BY id", { 10LL });
    REQUIRE (paged_results.size () == 10);
    REQUIRE (std::any_cast<long long> (paged_results[0].at ("id")) == 1LL);
    REQUIRE (std::any_cast<long long> (paged_results[9].at ("id")) == 10LL);
  }
}

TEST_CASE ("Transaction lifecycle safety", "[store][safety]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)", {});

  SECTION ("Transaction destruction removes from stack")
  {
    // Create and immediately destroy a transaction
    {
      auto tx = store->Begin ();
      tx->Execute ("INSERT INTO test (value) VALUES (?)", { "temp" });
      // tx destroyed here without commit - should unregister from stack
    }

    // Store should still work after orphaned transaction destruction
    REQUIRE_NOTHROW (store->Execute ("SELECT 1", {}));

    // New transaction should work normally
    auto tx2 = store->Begin ();
    tx2->Execute ("INSERT INTO test (value) VALUES (?)", { "after" });
    tx2->Commit ();

    auto results = store->Execute ("SELECT value FROM test", {});
    // Only "after" should be present (temp was rolled back implicitly)
    REQUIRE (results.size () == 1);
    REQUIRE (std::any_cast<std::string> (results[0].at ("value")) == "after");
  }

  SECTION ("Nested transaction destroyed before parent")
  {
    auto outer = store->Begin ();
    outer->Execute ("INSERT INTO test (value) VALUES (?)", { "outer" });

    {
      auto inner = outer->Begin ();
      inner->Execute ("INSERT INTO test (value) VALUES (?)", { "inner" });
      // inner destroyed here without commit
    }

    // Outer transaction should still work
    REQUIRE_NOTHROW (
        outer->Execute ("INSERT INTO test (value) VALUES (?)", { "after_inner" }));
    outer->Commit ();

    auto results
        = store->Execute ("SELECT value FROM test ORDER BY id", {});
    // Should have outer and after_inner (inner was implicitly rolled back)
    REQUIRE (results.size () == 2);
  }

  SECTION ("Destroying a non-current nested transaction rolls back descendants")
  {
    auto outer = store->Begin ();
    outer->Execute ("INSERT INTO test (value) VALUES (?)", { "outer" });
    auto middle = outer->Begin ();
    middle->Execute ("INSERT INTO test (value) VALUES (?)", { "middle" });
    auto inner = middle->Begin ();
    inner->Execute ("INSERT INTO test (value) VALUES (?)", { "inner" });

    middle.reset ();

    REQUIRE_THROWS_AS (inner->Execute ("SELECT 1", {}),
                       cortext::TransactionAlreadyFinishedError);
    outer->Execute ("INSERT INTO test (value) VALUES (?)", { "after_middle" });
    outer->Commit ();

    auto results
        = store->Execute ("SELECT value FROM test ORDER BY id", {});
    REQUIRE (results.size () == 2);
    REQUIRE (std::any_cast<std::string> (results[0].at ("value")) == "outer");
    REQUIRE (std::any_cast<std::string> (results[1].at ("value"))
             == "after_middle");
  }

  SECTION ("Execute after transaction destruction uses direct execution")
  {
    {
      auto tx = store->Begin ();
      tx->Execute ("INSERT INTO test (value) VALUES (?)", { "in_tx" });
      tx->Commit ();
    }

    // After transaction is destroyed and committed, Execute should work directly
    auto results = store->Execute ("SELECT COUNT(*) as cnt FROM test", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("cnt")) == 1LL);
  }

  SECTION ("Multiple transaction create-destroy cycles")
  {
    for (int i = 0; i < 10; ++i)
      {
        auto tx = store->Begin ();
        tx->Execute ("INSERT INTO test (value) VALUES (?)",
                     { "cycle_" + std::to_string (i) });
        if (i % 2 == 0)
          {
            tx->Commit ();
          }
        // Odd iterations: tx destroyed without commit
      }

    // Only even iterations should be committed
    auto results = store->Execute ("SELECT COUNT(*) as cnt FROM test", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("cnt")) == 5LL);
  }
}

TEST_CASE ("WAL mode configuration", "[store][wal]")
{
  SECTION ("WAL mode is enabled for file databases")
  {
    TempDatabase temp_db;
    cortext::SQLiteConfig config;
    config.enable_wal = true;

    auto store = cortext::SQLiteStore::Create (temp_db.path (), config);
    auto result = store->Execute ("PRAGMA journal_mode", {});
    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<std::string> (result[0].at ("journal_mode"))
             == "wal");
  }

  SECTION ("WAL autocheckpoint uses SQLite default for file databases")
  {
    cortext::testing::ScopedEnvVar wal_autocheckpoint (
        "CORTEXT_SQLITE_WAL_AUTOCHECKPOINT");
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    REQUIRE (store->WalAutoCheckpointPages () == 1000);
  }

  SECTION ("WAL autocheckpoint supports explicit override")
  {
    cortext::testing::ScopedEnvVar wal_autocheckpoint (
        "CORTEXT_SQLITE_WAL_AUTOCHECKPOINT", "0");
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    REQUIRE (store->WalAutoCheckpointPages () == 0);
  }

  SECTION ("Memory databases use memory journal mode")
  {
    auto store = cortext::SQLiteStore::Create (":memory:");
    auto result = store->Execute ("PRAGMA journal_mode", {});
    REQUIRE (result.size () == 1);
    // Memory databases ignore WAL mode
    REQUIRE (std::any_cast<std::string> (result[0].at ("journal_mode"))
             == "memory");
  }

  SECTION ("Busy timeout is configured")
  {
    TempDatabase temp_db;
    cortext::SQLiteConfig config;
    config.busy_timeout_ms = 3000;

    auto store = cortext::SQLiteStore::Create (temp_db.path (), config);
    auto result = store->Execute ("PRAGMA busy_timeout", {});
    REQUIRE (result.size () == 1);
    // SQLite returns busy_timeout PRAGMA with column name "timeout"
    REQUIRE (std::any_cast<long long> (result[0].at ("timeout")) == 3000LL);
  }

  SECTION ("Foreign keys are enabled by default")
  {
    auto store = cortext::SQLiteStore::Create (":memory:");
    auto result = store->Execute ("PRAGMA foreign_keys", {});
    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<long long> (result[0].at ("foreign_keys")) == 1LL);
  }

  SECTION ("Foreign keys can be disabled")
  {
    cortext::SQLiteConfig config;
    config.enable_foreign_keys = false;

    auto store = cortext::SQLiteStore::Create (":memory:", config);
    auto result = store->Execute ("PRAGMA foreign_keys", {});
    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<long long> (result[0].at ("foreign_keys")) == 0LL);
  }

  SECTION ("Synchronous mode is configurable")
  {
    cortext::SQLiteConfig config;
    config.synchronous = 2; // FULL

    auto store = cortext::SQLiteStore::Create (":memory:", config);
    auto result = store->Execute ("PRAGMA synchronous", {});
    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<long long> (result[0].at ("synchronous")) == 2LL);
  }

  SECTION ("Synchronous mode can be overridden for diagnostics")
  {
    cortext::testing::ScopedEnvVar sync_override (
        "CORTEXT_SQLITE_SYNCHRONOUS", "off");

    cortext::SQLiteConfig config;
    config.synchronous = 2; // FULL
    auto store = cortext::SQLiteStore::Create (":memory:", config);
    auto result = store->Execute ("PRAGMA synchronous", {});

    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<long long> (result[0].at ("synchronous")) == 0LL);
  }

  SECTION ("Journal mode can be overridden for realtime diagnostics")
  {
    cortext::testing::ScopedEnvVar journal_override (
        "CORTEXT_SQLITE_JOURNAL_MODE", "memory");

    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());
    auto result = store->Execute ("PRAGMA journal_mode", {});

    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<std::string> (result[0].at ("journal_mode"))
             == "memory");
  }

  SECTION ("Locking mode can be overridden for diagnostics")
  {
    cortext::testing::ScopedEnvVar locking_override (
        "CORTEXT_SQLITE_LOCKING_MODE", "exclusive");

    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());
    auto result = store->Execute ("PRAGMA locking_mode", {});

    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<std::string> (result[0].at ("locking_mode"))
             == "exclusive");
  }

  SECTION ("Page size can be overridden before schema creation")
  {
    cortext::testing::ScopedEnvVar page_size_override (
        "CORTEXT_SQLITE_PAGE_SIZE", "8192");

    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());
    auto result = store->Execute ("PRAGMA page_size", {});

    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<long long> (result[0].at ("page_size")) == 8192LL);
  }

  SECTION ("Temp store can be overridden for diagnostics")
  {
    cortext::testing::ScopedEnvVar temp_store_override (
        "CORTEXT_SQLITE_TEMP_STORE", "memory");

    auto store = cortext::SQLiteStore::Create (":memory:");
    auto result = store->Execute ("PRAGMA temp_store", {});

    REQUIRE (result.size () == 1);
    REQUIRE (std::any_cast<long long> (result[0].at ("temp_store")) == 2LL);
  }
}

TEST_CASE ("WAL checkpoint operations", "[store][wal]")
{
  SECTION ("Passive checkpoint succeeds")
  {
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    store->Execute ("CREATE TABLE test (id INTEGER)", {});
    for (int i = 0; i < 100; ++i)
      {
        store->Execute ("INSERT INTO test (id) VALUES (?)", { i });
      }

    REQUIRE_NOTHROW (store->Checkpoint (false));
  }

  SECTION ("Full checkpoint truncates WAL file")
  {
    cortext::testing::ScopedEnvVar wal_autocheckpoint (
        "CORTEXT_SQLITE_WAL_AUTOCHECKPOINT", "0");
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY, value BLOB)",
                    {});
    const std::vector<unsigned char> payload (4096, 7);
    for (int i = 0; i < 200; ++i)
      {
        store->Execute ("INSERT INTO test (value) VALUES (?)", { payload });
      }

    REQUIRE (store->WalFileBytes () > 0);
    REQUIRE_NOTHROW (store->Checkpoint (true));
    REQUIRE (store->WalFileBytes () == 0);
  }

  SECTION ("GetWalStatus returns valid info")
  {
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    store->Execute ("CREATE TABLE test (id INTEGER)", {});
    store->Execute ("INSERT INTO test (id) VALUES (1)", {});

    auto status = store->GetWalStatus ();
    REQUIRE (status.wal_pages >= 0);
    REQUIRE (status.checkpointed >= 0);
  }

  SECTION ("Checkpoint on memory database is no-op")
  {
    auto store = cortext::SQLiteStore::Create (":memory:");
    store->Execute ("CREATE TABLE test (id INTEGER)", {});
    // Should not throw - memory DBs don't use WAL but checkpoint is harmless
    REQUIRE_NOTHROW (store->Checkpoint (false));
  }
}

TEST_CASE ("WAL file persistence", "[store][wal]")
{
  SECTION ("Data persists after close and reopen")
  {
    TempDatabase temp_db;
    const std::string &path = temp_db.path ();

    {
      auto store = cortext::SQLiteStore::Create (path);
      store->Execute ("CREATE TABLE test (value TEXT)", {});
      store->Execute ("INSERT INTO test (value) VALUES (?)", { "persisted" });
      store->Checkpoint (true);
    }

    // Reopen and verify
    auto store2 = cortext::SQLiteStore::Create (path);
    auto results = store2->Execute ("SELECT value FROM test", {});
    REQUIRE (results.size () == 1);
    REQUIRE (std::any_cast<std::string> (results[0].at ("value"))
             == "persisted");
  }

  SECTION ("Transaction isolation in WAL mode")
  {
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
                    {});
    store->Execute ("INSERT INTO test (value) VALUES (?)", { "initial" });

    // Start transaction but don't commit
    auto tx = store->Begin ();
    tx->Execute ("UPDATE test SET value = ?", { "modified" });

    // Rollback
    tx->Rollback ();

    auto results = store->Execute ("SELECT value FROM test", {});
    REQUIRE (std::any_cast<std::string> (results[0].at ("value")) == "initial");
  }
}

TEST_CASE ("Concurrent access patterns in WAL", "[store][wal]")
{
  SECTION ("Multiple sequential transactions")
  {
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    store->Execute ("CREATE TABLE counter (value INTEGER)", {});
    store->Execute ("INSERT INTO counter (value) VALUES (0)", {});

    // Simulate concurrent-like access with many transactions
    for (int i = 0; i < 50; ++i)
      {
        auto tx = store->Begin ();
        tx->Execute ("UPDATE counter SET value = value + 1", {});
        tx->Commit ();
      }

    auto result = store->Execute ("SELECT value FROM counter", {});
    REQUIRE (std::any_cast<long long> (result[0].at ("value")) == 50LL);
  }

  SECTION ("Nested transactions with savepoints in WAL mode")
  {
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    store->Execute ("CREATE TABLE log (msg TEXT)", {});

    auto outer = store->Begin ();
    outer->Execute ("INSERT INTO log (msg) VALUES (?)", { "outer" });

    auto inner = outer->Begin ();
    inner->Execute ("INSERT INTO log (msg) VALUES (?)", { "inner" });
    inner->Rollback (); // Only inner rolled back

    outer->Execute ("INSERT INTO log (msg) VALUES (?)", { "after_inner" });
    outer->Commit ();

    auto results = store->Execute ("SELECT msg FROM log ORDER BY rowid", {});
    REQUIRE (results.size () == 2);
    REQUIRE (std::any_cast<std::string> (results[0].at ("msg")) == "outer");
    REQUIRE (std::any_cast<std::string> (results[1].at ("msg"))
             == "after_inner");
  }
}

TEST_CASE ("Error recovery in WAL mode", "[store][wal]")
{
  SECTION ("Store recovers from aborted transaction")
  {
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY)", {});

    {
      auto tx = store->Begin ();
      tx->Execute ("INSERT INTO test (id) VALUES (1)", {});
      // tx destroyed without commit - should rollback
    }

    // Store should still work
    auto tx2 = store->Begin ();
    tx2->Execute ("INSERT INTO test (id) VALUES (2)", {});
    tx2->Commit ();

    auto results = store->Execute ("SELECT id FROM test", {});
    REQUIRE (results.size () == 1);
    REQUIRE (std::any_cast<long long> (results[0].at ("id")) == 2LL);
  }

  SECTION ("Constraint errors don't corrupt WAL")
  {
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    store->Execute ("CREATE TABLE test (id INTEGER PRIMARY KEY)", {});
    store->Execute ("INSERT INTO test (id) VALUES (1)", {});

    // This should fail
    REQUIRE_THROWS (store->Execute ("INSERT INTO test (id) VALUES (1)", {}));

    // Store should still work
    store->Execute ("INSERT INTO test (id) VALUES (2)", {});
    store->Checkpoint (false);

    auto results = store->Execute ("SELECT COUNT(*) as cnt FROM test", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("cnt")) == 2LL);
  }

  SECTION ("Foreign key constraint errors work correctly")
  {
    TempDatabase temp_db;
    auto store = cortext::SQLiteStore::Create (temp_db.path ());

    store->Execute ("CREATE TABLE parent (id INTEGER PRIMARY KEY)", {});
    store->Execute (
        "CREATE TABLE child (id INTEGER PRIMARY KEY, "
        "parent_id INTEGER REFERENCES parent(id))",
        {});
    store->Execute ("INSERT INTO parent (id) VALUES (1)", {});

    // This should fail - foreign key violation
    REQUIRE_THROWS (store->Execute ("INSERT INTO child (parent_id) VALUES (99)",
                                    {}));

    // This should succeed
    REQUIRE_NOTHROW (
        store->Execute ("INSERT INTO child (parent_id) VALUES (1)", {}));

    auto results = store->Execute ("SELECT COUNT(*) as cnt FROM child", {});
    REQUIRE (std::any_cast<long long> (results[0].at ("cnt")) == 1LL);
  }
}
