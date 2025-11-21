#pragma once

#include "store.hpp"

#include <any>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <sqlite3.h>

namespace cortext
{

// Forward declaration
class SQLiteStore;

/// @brief SQLite transaction with savepoint support for nested transactions.
class SQLiteTransaction : public Transaction
{
public:
  /// @brief Constructor for root transaction.
  /// @param store Pointer to the SQLite store.
  explicit SQLiteTransaction (SQLiteStore *store);

  /// @brief Constructor for nested transaction.
  /// @param store Pointer to the SQLite store.
  /// @param parent_transaction Pointer to the parent transaction.
  SQLiteTransaction (SQLiteStore *store,
                     SQLiteTransaction *parent_transaction);

  ~SQLiteTransaction () override = default;

  /// @brief Begin a nested transaction using a savepoint.
  /// @return A unique pointer to the new transaction.
  std::unique_ptr<Transaction> Begin () override;

  /// @brief Execute a query within this transaction context.
  /// @param query The SQL query to execute.
  /// @param params The parameters for the query.
  /// @return The query results.
  std::vector<std::map<std::string, std::any> >
  Execute (const std::string &query,
           const std::vector<std::any> &params = {}) override;

  /// @brief Commit this transaction.
  void Commit () override;

  /// @brief Rollback this transaction.
  void Rollback () override;

  /// @brief Check if transaction is finished.
  /// @return True if the transaction is committed or rolled back.
  bool
  IsFinished () const
  {
    return is_committed_ || is_rolled_back_;
  }

  /// @brief Get the savepoint name (empty for root transactions).
  /// @return The savepoint name.
  const std::string &
  GetSavepointName () const
  {
    return savepoint_name_;
  }

private:
  SQLiteStore *store_;
  SQLiteTransaction *parent_transaction_;
  std::string savepoint_name_;
  bool is_committed_;
  bool is_rolled_back_;

  friend class SQLiteStore;
};

/// @brief RAII wrapper for SQLite database connection.
class SQLiteConnection
{
public:
  /// @brief Construct a SQLite connection.
  /// @param database_path Path to the database file, or ":memory:"
  /// for in-memory database.
  explicit SQLiteConnection (const std::string &database_path = ":memory:");
  ~SQLiteConnection ();

  // Delete copy operations
  SQLiteConnection (const SQLiteConnection &) = delete;
  SQLiteConnection &operator= (const SQLiteConnection &) = delete;

  // Move operations
  SQLiteConnection (SQLiteConnection &&other) noexcept;
  SQLiteConnection &operator= (SQLiteConnection &&other) noexcept;

  /// @brief Get the underlying SQLite connection handle.
  /// @return Pointer to the sqlite3 connection.
  sqlite3 *
  GetConnection () const
  {
    return connection_;
  }

  /// @brief Check if the connection is valid.
  /// @return True if the connection is valid.
  bool
  IsValid () const
  {
    return connection_ != nullptr;
  }

private:
  sqlite3 *connection_;
};

/// @brief SQLite database store with transaction support.
class SQLiteStore : public Store
{
public:
  /// @brief Create a SQLiteStore with a new database connection.
  /// @param database_path Path to the database file, or ":memory:"
  /// for in-memory database.
  /// @return A unique pointer to the SQLiteStore.
  static std::unique_ptr<SQLiteStore> Create (const std::string &database_path
                                              = ":memory:");

  /// @brief Constructor taking ownership of connection.
  /// @param connection A unique pointer to the SQLite connection.
  explicit SQLiteStore (std::unique_ptr<SQLiteConnection> connection);

  ~SQLiteStore () override = default;

  /// @brief Execute a query and return results.
  /// @param query The SQL query to execute.
  /// @param params The parameters for the query.
  /// @return The query results.
  std::vector<std::map<std::string, std::any> >
  Execute (const std::string &query,
           const std::vector<std::any> &params = {}) override;

  /// @brief Begin a new transaction.
  /// @return A unique pointer to the new transaction.
  std::unique_ptr<Transaction> Begin () override;

  /// @brief Commit the current transaction.
  void Commit () override;

  /// @brief Rollback the current transaction.
  void Rollback () override;

  /// @brief Close the database connection.
  void Close () override;

private:
  std::unique_ptr<SQLiteConnection> connection_;
  bool in_transaction_;
  std::vector<SQLiteTransaction *> transaction_stack_;

  // Execute a query directly on the connection.
  std::vector<std::map<std::string, std::any> >
  ExecuteDirect (const std::string &query,
                 const std::vector<std::any> &params = {});

  // Commit a root transaction (called by transaction.commit()).
  void CommitRootTransaction (SQLiteTransaction *transaction);

  // Rollback a root transaction (called by transaction.rollback()).
  void RollbackRootTransaction (SQLiteTransaction *transaction);

  friend class SQLiteTransaction;
};

} // namespace cortext
