// tests/store.test.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <ctime>
#include <filesystem>
#include <iostream>

namespace
{

// Helper function to create a temporary database file
std::string
create_temp_db ()
{
  auto temp_dir = std::filesystem::temp_directory_path ();
  auto db_path
      = temp_dir
        / ("test_store_" + std::to_string (std::time (nullptr)) + ".db");
  return db_path.string ();
}

// Helper function to clean up temporary database file
void
cleanup_temp_db (const std::string &db_path)
{
  try
    {
      std::filesystem::remove (db_path);
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
