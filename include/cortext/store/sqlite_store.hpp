#pragma once

#include "store.hpp"

#include <any>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <sqlite3.h>

namespace cortext
{

class SQLiteStore;

namespace internal
{
class SQLiteWalCheckpointer;

class SQLiteStoreQueryInterrupter
{
public:
  static bool Interrupt (SQLiteStore &store);
};

#if defined(CORTEXT_TESTING)
class SQLiteStoreStatementCacheInspector;
#endif
}

/// @brief Configuration options for SQLite connection.
struct SQLiteConfig
{
  bool enable_wal = true;           ///< Use WAL journal mode (file DBs only)
  int busy_timeout_ms = 5000;       ///< Wait time on SQLITE_BUSY (ms)
  bool enable_foreign_keys = true;  ///< Enforce foreign key constraints
  int synchronous = 1;              ///< NORMAL (1), FULL (2), OFF (0)
  int cache_size_kb = 65536;        ///< Cache size in KB (negative for KB)
};

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

  ~SQLiteTransaction () override;

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
  /// @brief Construct a SQLite connection with configuration.
  /// @param database_path Path to the database file, or ":memory:"
  /// for in-memory database.
  /// @param config Configuration options for the connection.
  explicit SQLiteConnection (const std::string &database_path = ":memory:",
                             const SQLiteConfig &config = SQLiteConfig{});
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
///
/// SQLiteStore is a single-owner handle. Do not call methods concurrently on
/// the same instance unless the caller provides external synchronization. The
/// transaction stack and prepared-statement cache assume serialized access.
/// Multiple SQLiteStore instances may open the same database, but Cortext's
/// runtime state model assumes one writer per database.
class SQLiteStore : public Store
{
public:
  /// @brief WAL status information.
  struct WalStatus
  {
    int wal_pages;     ///< Number of pages in WAL file.
    int checkpointed;  ///< Number of pages checkpointed.
  };

  /// @brief Create a SQLiteStore with a new database connection.
  /// @param database_path Path to the database file, or ":memory:"
  /// for in-memory database.
  /// @param config Configuration options for the connection.
  /// @return A unique pointer to the SQLiteStore.
  static std::unique_ptr<SQLiteStore> Create (
      const std::string &database_path = ":memory:",
      const SQLiteConfig &config = SQLiteConfig{});

  /// @brief Constructor taking ownership of connection.
  /// @param connection A unique pointer to the SQLite connection.
  explicit SQLiteStore (std::unique_ptr<SQLiteConnection> connection);

  ~SQLiteStore () override;

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

  /// @brief Run a WAL checkpoint on the calling thread.
  /// @param full If true, use TRUNCATE mode (waits for readers, syncs the
  /// database and shrinks the WAL): a power-loss durability barrier for every
  /// committed transaction. Otherwise PASSIVE (never waits on other
  /// connections).
  void Checkpoint (bool full = false);

  /// @brief Ask for a PASSIVE checkpoint without blocking the caller.
  ///
  /// File databases in WAL mode own a background checkpointer: a separate
  /// connection on its own thread copies committed WAL frames into the
  /// database and syncs it, so neither the copy nor its fsyncs run on the
  /// thread that writes. The WAL also requests one once it holds
  /// WalAutoCheckpointPages() pages (0: only explicit requests). Other stores
  /// (memory, non-WAL, exclusive locking) run the PASSIVE checkpoint inline.
  ///
  /// Durability: a committed transaction survives a process crash as soon as
  /// it commits (it is in the WAL); with synchronous=NORMAL it survives power
  /// loss once a checkpoint (background or Checkpoint) has synced it. The
  /// database is consistent after either crash: recovery replays only whole
  /// committed transactions.
  void RequestCheckpoint ();

  /// @brief Block until every requested background checkpoint has finished.
  void WaitForRequestedCheckpoints ();

  /// @brief Sync the WAL on the calling thread (outside a transaction).
  ///
  /// A power-loss durability barrier for every transaction committed so far:
  /// one fsync of the WAL file, without copying it into the database. No-op
  /// for stores without a WAL. Throws StoreError if the sync fails.
  void SyncWal ();

  /// @brief Get WAL status information.
  /// @return WAL status with page counts.
  WalStatus GetWalStatus () const;

  /// @brief Return the WAL size in pages at which a checkpoint is requested
  /// automatically (background when available), or zero when disabled.
  int WalAutoCheckpointPages () const;

  /// @brief Return the current WAL file size in bytes, or zero when absent.
  std::uintmax_t WalFileBytes () const;

  /// @brief Path of the database file when other connections can read it
  /// concurrently (a file database in WAL mode on a thread-safe SQLite
  /// build), else empty.
  std::string ConcurrentReadPath () const;

  /// @brief Unregister a transaction from the transaction stack.
  /// Called by SQLiteTransaction destructor to prevent dangling pointers.
  /// @param tx Pointer to the transaction to unregister.
  void UnregisterTransaction (SQLiteTransaction *tx);

private:
  std::unique_ptr<SQLiteConnection> connection_;
  // Declared after connection_ so it stops (and closes its own connection)
  // before the store's connection closes.
  std::unique_ptr<internal::SQLiteWalCheckpointer> checkpointer_;
  bool in_transaction_;
  std::vector<SQLiteTransaction *> transaction_stack_;

  // Prepared-statement cache keyed by exact SQL text. Statements are reset
  // (not finalized) after each execution so repeated queries skip SQL
  // parsing. Bounded FIFO eviction: one-off statements (DDL, savepoint
  // names) age out instead of accumulating.
  static constexpr std::size_t kStatementCacheCapacity = 256;
  std::unordered_map<std::string, sqlite3_stmt *> statement_cache_;
  std::deque<std::string> statement_cache_fifo_;

  // Return a reset, rebindable statement for `query`, preparing and caching
  // it on first use. Throws StoreError if preparation fails.
  sqlite3_stmt *AcquireStatement (const std::string &query);

  // Finalize and drop a cached statement (used when a step fails so the
  // next execution re-prepares from scratch).
  void EvictStatement (const std::string &query);

  void ClearStatementCache ();

  bool IsDbInTransaction () const;

  // Execute a query directly on the connection.
  std::vector<std::map<std::string, std::any> >
  ExecuteDirect (const std::string &query,
                 const std::vector<std::any> &params = {});

  // Commit a root transaction (called by transaction.commit()).
  void CommitRootTransaction (SQLiteTransaction *transaction);

  // Rollback a root transaction (called by transaction.rollback()).
  void RollbackRootTransaction (SQLiteTransaction *transaction);

  friend class SQLiteTransaction;
  friend class internal::SQLiteStoreQueryInterrupter;
#if defined(CORTEXT_TESTING)
  friend class internal::SQLiteStoreStatementCacheInspector;
#endif
};

} // namespace cortext
