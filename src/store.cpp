#include "cortext/store/sqlite_store.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "cortext/store/extension_loader.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext
{

namespace internal
{

bool
SQLiteStoreQueryInterrupter::Interrupt (SQLiteStore &store)
{
  if (!store.connection_ || !store.connection_->IsValid ())
    {
      return false;
    }
  sqlite3_interrupt (store.connection_->GetConnection ());
  return true;
}

} // namespace internal

namespace
{

constexpr int kDefaultWalAutoCheckpointPages = 1000;

std::string_view
ParseDbOperation (const std::string &q)
{
  std::string_view s (q);
  while (!s.empty () && (s.front () == ' ' || s.front () == '\n'
                         || s.front () == '\r' || s.front () == '\t'))
    {
      s.remove_prefix (1);
    }
  const size_t n = s.size ();
  size_t i = 0;
  while (i < n)
    {
      const char c = s[i];
      if (c == ' ' || c == '\n' || c == '\r' || c == '\t')
        break;
      i++;
    }
  std::string_view tok = s.substr (0, i);
  if (tok.empty ())
    return "UNKNOWN";
  return tok;
}

int
ResolveSynchronousMode (int configured)
{
  const char *value = std::getenv ("CORTEXT_SQLITE_SYNCHRONOUS");
  if (!value || value[0] == '\0')
    {
      return configured;
    }
  std::string mode (value);
  std::transform (mode.begin (), mode.end (), mode.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  if (mode == "off" || mode == "0")
    {
      return 0;
    }
  if (mode == "normal" || mode == "1")
    {
      return 1;
    }
  if (mode == "full" || mode == "2")
    {
      return 2;
    }
  if (mode == "extra" || mode == "3")
    {
      return 3;
    }
  return configured;
}

std::optional<long long>
ParseLongLongEnv (const char *name)
{
  const char *value = std::getenv (name);
  if (!value || value[0] == '\0')
    {
      return std::nullopt;
    }
  char *end = nullptr;
  errno = 0;
  const long long parsed = std::strtoll (value, &end, 10);
  if (errno != 0 || end == value || (end && *end != '\0'))
    {
      return std::nullopt;
    }
  return parsed;
}

int
ResolveWalAutoCheckpointPages ()
{
  if (auto pages = ParseLongLongEnv ("CORTEXT_SQLITE_WAL_AUTOCHECKPOINT"))
    {
      if (*pages >= 0)
        {
          return static_cast<int> (
              std::min (*pages,
                        static_cast<long long> (
                            std::numeric_limits<int>::max ())));
        }
    }
  return kDefaultWalAutoCheckpointPages;
}

std::optional<std::string>
ResolveJournalModeOverride ()
{
  const std::string mode = [] {
    const char *value = std::getenv ("CORTEXT_SQLITE_JOURNAL_MODE");
    if (!value || value[0] == '\0')
      {
        return std::string ();
      }
    std::string out (value);
    std::transform (out.begin (), out.end (), out.begin (),
                    [] (unsigned char c) {
                      return static_cast<char> (std::tolower (c));
                    });
    return out;
  } ();

  if (mode.empty ())
    {
      return std::nullopt;
    }
  if (mode == "wal" || mode == "delete" || mode == "truncate"
      || mode == "persist" || mode == "memory" || mode == "off")
    {
      return mode;
    }
  return std::nullopt;
}

std::optional<std::string>
ResolveLockingModeOverride ()
{
  const std::string mode = [] {
    const char *value = std::getenv ("CORTEXT_SQLITE_LOCKING_MODE");
    if (!value || value[0] == '\0')
      {
        return std::string ();
      }
    std::string out (value);
    std::transform (out.begin (), out.end (), out.begin (),
                    [] (unsigned char c) {
                      return static_cast<char> (std::tolower (c));
                    });
    return out;
  } ();

  if (mode == "exclusive" || mode == "normal")
    {
      return mode;
    }
  return std::nullopt;
}

std::optional<std::string>
ResolveTempStoreOverride ()
{
  const std::string mode = [] {
    const char *value = std::getenv ("CORTEXT_SQLITE_TEMP_STORE");
    if (!value || value[0] == '\0')
      {
        return std::string ();
      }
    std::string out (value);
    std::transform (out.begin (), out.end (), out.begin (),
                    [] (unsigned char c) {
                      return static_cast<char> (std::tolower (c));
                    });
    return out;
  } ();

  if (mode == "default" || mode == "0")
    {
      return std::string ("0");
    }
  if (mode == "file" || mode == "1")
    {
      return std::string ("1");
    }
  if (mode == "memory" || mode == "2")
    {
      return std::string ("2");
    }
  return std::nullopt;
}

std::string
ResolveBeginTransactionSql ()
{
  const std::string mode = [] {
    const char *value = std::getenv ("CORTEXT_SQLITE_TRANSACTION_MODE");
    if (!value || value[0] == '\0')
      {
        return std::string ();
      }
    std::string out (value);
    std::transform (out.begin (), out.end (), out.begin (),
                    [] (unsigned char c) {
                      return static_cast<char> (std::tolower (c));
                    });
    return out;
  } ();

  if (mode == "immediate")
    {
      return "BEGIN IMMEDIATE";
    }
  if (mode == "exclusive")
    {
      return "BEGIN EXCLUSIVE";
    }
  return "BEGIN";
}

sqlite3_stmt *
PrepareStatement (sqlite3 *db, const std::string &query, int &prepare_rc)
{
  sqlite3_stmt *stmt = nullptr;
  prepare_rc = sqlite3_prepare_v2 (
      db, query.c_str (), -1, &stmt, nullptr);
  if (prepare_rc != SQLITE_OK)
    {
      throw StoreError (
          "Failed to prepare statement: "
          + std::string (sqlite3_errmsg (db)));
    }
  return stmt;
}

void
CheckBindResult (sqlite3_stmt *stmt, int param_index, int rc)
{
  if (rc == SQLITE_OK)
    {
      return;
    }
  sqlite3 *db = sqlite3_db_handle (stmt);
  throw StoreError (
      "Failed to bind SQLite parameter " + std::to_string (param_index)
      + ": " + (db ? std::string (sqlite3_errmsg (db))
                        : std::string (sqlite3_errstr (rc))));
}

void
BindParameters (sqlite3_stmt *stmt, const std::vector<std::any> &params)
{
  const int expected_count = sqlite3_bind_parameter_count (stmt);
  if (params.size () != static_cast<std::size_t> (expected_count))
    {
      throw StoreError (
          "SQLite parameter count mismatch: statement expects "
          + std::to_string (expected_count) + ", received "
          + std::to_string (params.size ()));
    }
  for (size_t i = 0; i < params.size (); ++i)
    {
      int param_index = static_cast<int> (i + 1);
      const auto &param = params[i];
      try
        {
          if (param.type () == typeid (int))
            {
              CheckBindResult (stmt, param_index,
                               sqlite3_bind_int (
                                   stmt, param_index,
                                   std::any_cast<int> (param)));
            }
          else if (param.type () == typeid (long))
            {
              CheckBindResult (
                  stmt, param_index,
                  sqlite3_bind_int64 (stmt, param_index,
                                      std::any_cast<long> (param)));
            }
          else if (param.type () == typeid (long long))
            {
              CheckBindResult (
                  stmt, param_index,
                  sqlite3_bind_int64 (stmt, param_index,
                                      std::any_cast<long long> (param)));
            }
          else if (param.type () == typeid (double))
            {
              CheckBindResult (
                  stmt, param_index,
                  sqlite3_bind_double (stmt, param_index,
                                       std::any_cast<double> (param)));
            }
          else if (param.type () == typeid (float))
            {
              CheckBindResult (
                  stmt, param_index,
                  sqlite3_bind_double (
                      stmt, param_index,
                      static_cast<double> (std::any_cast<float> (param))));
            }
          else if (param.type () == typeid (std::string))
            {
              const std::string &str = std::any_cast<std::string> (param);
              CheckBindResult (
                  stmt, param_index,
                  sqlite3_bind_text64 (
                      stmt, param_index, str.c_str (),
                      static_cast<sqlite3_uint64> (str.size ()),
                      SQLITE_TRANSIENT, SQLITE_UTF8));
            }
          else if (param.type () == typeid (const char *))
            {
              const char *str = std::any_cast<const char *> (param);
              CheckBindResult (
                  stmt, param_index,
                  sqlite3_bind_text (stmt, param_index, str, -1,
                                     SQLITE_TRANSIENT));
            }
          else if (param.type () == typeid (std::vector<char>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<char> &> (param);
              CheckBindResult (
                  stmt, param_index,
                  sqlite3_bind_blob64 (
                      stmt, param_index, vec.empty () ? nullptr : vec.data (),
                      static_cast<sqlite3_uint64> (vec.size ()),
                      SQLITE_TRANSIENT));
            }
          else if (param.type () == typeid (std::vector<unsigned char>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<unsigned char> &> (param);
              CheckBindResult (
                  stmt, param_index,
                  sqlite3_bind_blob64 (
                      stmt, param_index,
                      vec.empty () ? nullptr
                                   : reinterpret_cast<const void *> (vec.data ()),
                      static_cast<sqlite3_uint64> (vec.size ()),
                      SQLITE_TRANSIENT));
            }
          else if (param.type () == typeid (std::vector<float>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<float> &> (param);
              CheckBindResult (
                  stmt, param_index,
                  sqlite3_bind_blob64 (
                      stmt, param_index,
                      vec.empty () ? nullptr
                                   : reinterpret_cast<const void *> (vec.data ()),
                      static_cast<sqlite3_uint64> (vec.size ()) * sizeof (float),
                      SQLITE_TRANSIENT));
            }
          else if (!param.has_value ()
                   || param.type () == typeid (std::nullptr_t))
            {
              CheckBindResult (stmt, param_index,
                               sqlite3_bind_null (stmt, param_index));
            }
          else
            {
              throw StoreError (
                  "Unsupported SQLite bind parameter type at index "
                  + std::to_string (param_index) + ": " + param.type ().name ());
            }
        }
      catch (const std::bad_any_cast &)
        {
          throw StoreError (
              "SQLite bind parameter type mismatch at index "
              + std::to_string (param_index));
        }
    }
}

void
FetchResultRow (sqlite3_stmt *stmt, std::map<std::string, std::any> &row)
{
  int column_count = sqlite3_column_count (stmt);
  for (int i = 0; i < column_count; ++i)
    {
      const char *column_name = sqlite3_column_name (stmt, i);
      if (!column_name)
        continue;
      std::string col_name (column_name);
      int column_type = sqlite3_column_type (stmt, i);
      switch (column_type)
        {
        case SQLITE_INTEGER:
          row[col_name]
              = static_cast<long long> (sqlite3_column_int64 (stmt, i));
          break;
        case SQLITE_FLOAT:
          row[col_name] = sqlite3_column_double (stmt, i);
          break;
        case SQLITE_TEXT:
          {
            const unsigned char *text = sqlite3_column_text (stmt, i);
            int text_len = sqlite3_column_bytes (stmt, i);
            row[col_name] = std::string (
                reinterpret_cast<const char *> (text), text_len);
            break;
          }
        case SQLITE_BLOB:
          {
            const void *blob = sqlite3_column_blob (stmt, i);
            int blob_len = sqlite3_column_bytes (stmt, i);
            row[col_name] = std::vector<char> (
                static_cast<const char *> (blob),
                static_cast<const char *> (blob) + blob_len);
            break;
          }
        case SQLITE_NULL:
        default:
          row[col_name] = nullptr;
          break;
        }
    }
}

} // namespace

// SQLiteConnection implementation
SQLiteConnection::SQLiteConnection (const std::string &database_path,
                                    const SQLiteConfig &config)
    : connection_ (nullptr)
{
  if (sqlite3_open (database_path.c_str (), &connection_) != SQLITE_OK)
    {
      throw StoreError ("Failed to open database: "
                        + std::string (sqlite3_errmsg (connection_)));
    }

  // Set busy timeout for concurrent access handling
  sqlite3_busy_timeout (connection_, config.busy_timeout_ms);

  // Helper to execute PRAGMA statements
  auto exec_pragma = [this] (const std::string &pragma) {
    char *errmsg = nullptr;
    int rc
        = sqlite3_exec (connection_, pragma.c_str (), nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK)
      {
        std::string msg = errmsg ? errmsg : "Unknown error";
        sqlite3_free (errmsg);
        throw StoreError ("PRAGMA failed (" + pragma + "): " + msg);
      }
  };

  if (auto page_size = ParseLongLongEnv ("CORTEXT_SQLITE_PAGE_SIZE"))
    {
      if (*page_size >= 512 && *page_size <= 65536
          && (*page_size & (*page_size - 1)) == 0)
        {
          exec_pragma ("PRAGMA page_size = " + std::to_string (*page_size));
        }
    }

  if (auto temp_store = ResolveTempStoreOverride ())
    {
      exec_pragma ("PRAGMA temp_store = " + *temp_store);
    }
  else
    {
      exec_pragma ("PRAGMA temp_store = 2");
    }

  if (auto mmap_size = ParseLongLongEnv ("CORTEXT_SQLITE_MMAP_SIZE"))
    {
      if (*mmap_size >= 0)
        {
          exec_pragma ("PRAGMA mmap_size = " + std::to_string (*mmap_size));
        }
    }
  else
    {
      exec_pragma ("PRAGMA mmap_size = 268435456");
    }

  // Enable WAL mode for file databases (not :memory:)
  if (database_path != ":memory:")
    {
      const auto journal_override = ResolveJournalModeOverride ();
      if (journal_override)
        {
          exec_pragma ("PRAGMA journal_mode = " + *journal_override);
        }
      else if (config.enable_wal)
        {
          exec_pragma ("PRAGMA journal_mode = WAL");
        }
      if ((journal_override && *journal_override == "wal")
          || (!journal_override && config.enable_wal))
        {
          const int wal_autocheckpoint_pages
              = ResolveWalAutoCheckpointPages ();
          exec_pragma ("PRAGMA wal_autocheckpoint = "
                       + std::to_string (wal_autocheckpoint_pages));
          sqlite3_wal_autocheckpoint (connection_, wal_autocheckpoint_pages);
        }
    }

  // Set synchronous mode (NORMAL by default for WAL)
  exec_pragma ("PRAGMA synchronous = "
               + std::to_string (ResolveSynchronousMode (
                   config.synchronous)));

  // Enable foreign key constraints
  if (config.enable_foreign_keys)
    {
      exec_pragma ("PRAGMA foreign_keys = ON");
    }

  // Set cache size (negative value = KB)
  if (config.cache_size_kb > 0)
    {
      const auto cache_size_override
          = ParseLongLongEnv ("CORTEXT_SQLITE_CACHE_SIZE_KB");
      const long long cache_size_kb
          = cache_size_override.value_or (config.cache_size_kb);
      if (cache_size_kb > 0)
        {
          exec_pragma ("PRAGMA cache_size = -"
                       + std::to_string (cache_size_kb));
        }
    }

  if (auto locking_mode = ResolveLockingModeOverride ())
    {
      exec_pragma ("PRAGMA locking_mode = " + *locking_mode);
    }
}

SQLiteConnection::~SQLiteConnection ()
{
  if (connection_)
    {
      sqlite3_close (connection_);
    }
}

SQLiteConnection::SQLiteConnection (SQLiteConnection &&other) noexcept
    : connection_ (other.connection_)
{
  other.connection_ = nullptr;
}

SQLiteConnection &
SQLiteConnection::operator= (SQLiteConnection &&other) noexcept
{
  if (this != &other)
    {
      if (connection_)
        {
          sqlite3_close (connection_);
        }
      connection_ = other.connection_;
      other.connection_ = nullptr;
    }
  return *this;
}

// SQLiteTransaction implementation
SQLiteTransaction::SQLiteTransaction (SQLiteStore *store)
    : store_ (store), parent_transaction_ (nullptr), savepoint_name_ (),
      is_committed_ (false), is_rolled_back_ (false)
{
}

SQLiteTransaction::~SQLiteTransaction ()
{
  if (store_ && !is_committed_ && !is_rolled_back_)
    {
      try
        {
          auto it = std::find (store_->transaction_stack_.begin (),
                               store_->transaction_stack_.end (), this);
          if (savepoint_name_.empty ())
            {
              if (store_->in_transaction_)
                {
                  if (store_->IsDbInTransaction ())
                    {
                      store_->ExecuteDirect ("ROLLBACK");
                    }
                  store_->in_transaction_ = false;
                }
              if (it != store_->transaction_stack_.end ())
                {
                  for (auto mark_it = it;
                       mark_it != store_->transaction_stack_.end (); ++mark_it)
                    {
                      if (*mark_it)
                        {
                          (*mark_it)->is_rolled_back_ = true;
                        }
                    }
                  store_->transaction_stack_.erase (
                      it, store_->transaction_stack_.end ());
                }
            }
          else if (it != store_->transaction_stack_.end ())
            {
              store_->ExecuteDirect ("ROLLBACK TO SAVEPOINT "
                                     + savepoint_name_);
              store_->ExecuteDirect ("RELEASE SAVEPOINT " + savepoint_name_);
              for (auto mark_it = it;
                   mark_it != store_->transaction_stack_.end (); ++mark_it)
                {
                  if (*mark_it)
                    {
                      (*mark_it)->is_rolled_back_ = true;
                    }
                }
              store_->transaction_stack_.erase (
                  it, store_->transaction_stack_.end ());
            }
        }
      catch (...)
        {
          // Ignore errors during cleanup - we're in a destructor
        }
    }

  // Unregister from store to prevent dangling pointer in transaction_stack_
  if (store_)
    {
      store_->UnregisterTransaction (this);
    }
}

SQLiteTransaction::SQLiteTransaction (SQLiteStore *store,
                                      SQLiteTransaction *parent_transaction)
    : store_ (store), parent_transaction_ (parent_transaction),
      savepoint_name_ (
          parent_transaction != nullptr
              ? "savepoint_"
                    + std::to_string (reinterpret_cast<uintptr_t> (this))
              : ""),
      is_committed_ (false), is_rolled_back_ (false)
{
  // For nested transactions, create the savepoint immediately
  // Since the parent transaction is active, this will work
  if (parent_transaction != nullptr && !savepoint_name_.empty ())
    {
      std::string query = "SAVEPOINT " + savepoint_name_;
      store_->ExecuteDirect (query);
    }
}

std::unique_ptr<Transaction>
SQLiteTransaction::Begin ()
{
  if (is_committed_ || is_rolled_back_)
    {
      throw TransactionAlreadyFinishedError (
          "Transaction is already committed or rolled back");
    }
  if (!store_ || store_->transaction_stack_.empty ()
      || store_->transaction_stack_.back () != this)
    {
      throw TransactionNotCurrentError (
          "Transaction is not the current transaction");
    }
  auto transaction = std::make_unique<SQLiteTransaction> (store_, this);
  store_->transaction_stack_.push_back (transaction.get ());
  return transaction;
}

std::vector<std::map<std::string, std::any> >
SQLiteTransaction::Execute (const std::string &query,
                            const std::vector<std::any> &params)
{
  if (is_committed_ || is_rolled_back_)
    {
      throw TransactionAlreadyFinishedError (
          "Transaction is already committed or rolled back");
    }
  if (!store_ || store_->transaction_stack_.empty ()
      || store_->transaction_stack_.back () != this)
    {
      throw TransactionNotCurrentError (
          "Transaction is not the current transaction");
    }
  return store_->ExecuteDirect (query, params);
}

void
SQLiteTransaction::Commit ()
{
  if (is_committed_ || is_rolled_back_)
    {
      throw TransactionAlreadyFinishedError (
          "Transaction is already committed or rolled back");
    }
  if (!store_ || store_->transaction_stack_.empty ()
      || store_->transaction_stack_.back () != this)
    {
      throw TransactionNotCurrentError (
          "Transaction is not the current transaction");
    }

  if (savepoint_name_.empty ())
    {
      // This is a root transaction - commit via store
      store_->CommitRootTransaction (this);
    }
  else
    {
      // This is a nested transaction - release the savepoint
      std::string query = "RELEASE SAVEPOINT " + savepoint_name_;
      store_->ExecuteDirect (query);
      store_->transaction_stack_.pop_back ();
    }

  is_committed_ = true;
}

void
SQLiteTransaction::Rollback ()
{
  if (is_committed_ || is_rolled_back_)
    {
      throw TransactionAlreadyFinishedError (
          "Transaction is already committed or rolled back");
    }
  if (!store_ || store_->transaction_stack_.empty ()
      || store_->transaction_stack_.back () != this)
    {
      throw TransactionNotCurrentError (
          "Transaction is not the current transaction");
    }

  if (savepoint_name_.empty ())
    {
      // This is a root transaction - rollback via store
      store_->RollbackRootTransaction (this);
    }
  else
    {
      // This is a nested transaction - rollback and release the savepoint
      store_->ExecuteDirect ("ROLLBACK TO SAVEPOINT " + savepoint_name_);
      store_->ExecuteDirect ("RELEASE SAVEPOINT " + savepoint_name_);
      store_->transaction_stack_.pop_back ();
    }

  is_rolled_back_ = true;
}

// SQLiteStore implementation
std::unique_ptr<SQLiteStore>
SQLiteStore::Create (const std::string &database_path,
                     const SQLiteConfig &config)
{
  auto connection = std::make_unique<SQLiteConnection> (database_path, config);
  return std::make_unique<SQLiteStore> (std::move (connection));
}

SQLiteStore::~SQLiteStore ()
{
  ClearStatementCache ();
}

sqlite3_stmt *
SQLiteStore::AcquireStatement (const std::string &query)
{
  auto it = statement_cache_.find (query);
  if (it != statement_cache_.end ())
    {
      sqlite3_stmt *stmt = it->second;
      sqlite3_reset (stmt);
      sqlite3_clear_bindings (stmt);
      return stmt;
    }
  int prepare_rc = SQLITE_OK;
  sqlite3_stmt *stmt
      = PrepareStatement (connection_->GetConnection (), query, prepare_rc);
  if (statement_cache_.size () >= kStatementCacheCapacity)
    {
      while (statement_cache_.size () >= kStatementCacheCapacity)
        {
          if (statement_cache_fifo_.empty ())
            {
              EvictStatement (statement_cache_.begin ()->first);
            }
          else
            {
              EvictStatement (statement_cache_fifo_.front ());
            }
        }
    }
  statement_cache_.emplace (query, stmt);
  statement_cache_fifo_.push_back (query);
  return stmt;
}

void
SQLiteStore::EvictStatement (const std::string &query)
{
  auto it = statement_cache_.find (query);
  if (it != statement_cache_.end ())
    {
      sqlite3_finalize (it->second);
      statement_cache_.erase (it);
    }
  statement_cache_fifo_.erase (
      std::remove (statement_cache_fifo_.begin (), statement_cache_fifo_.end (),
                   query),
      statement_cache_fifo_.end ());
}

void
SQLiteStore::ClearStatementCache ()
{
  for (auto &entry : statement_cache_)
    {
      sqlite3_finalize (entry.second);
    }
  statement_cache_.clear ();
  statement_cache_fifo_.clear ();
}

SQLiteStore::SQLiteStore (std::unique_ptr<SQLiteConnection> connection)
    : connection_ (std::move (connection)), in_transaction_ (false)
{
  if (!connection_->IsValid ())
    {
      throw StoreError ("Invalid database connection");
    }

  // Register statically linked extensions if present, then attempt dynamic
  // fallbacks.
  RegisterBuiltInExtensions ();
  RegisterBuiltInExtensionsOnDb (connection_->GetConnection ());
#if defined(CORTEXT_ENABLE_DYNAMIC_EXTENSIONS) && !defined(__EMSCRIPTEN__)
  TryLoadDynamicExtensions (connection_->GetConnection ());
#endif
}

bool
SQLiteStore::IsDbInTransaction () const
{
  return sqlite3_get_autocommit (connection_->GetConnection ()) == 0;
}

std::vector<std::map<std::string, std::any> >
SQLiteStore::Execute (const std::string &query,
                      const std::vector<std::any> &params)
{
  // Clean up any finished transactions from the stack
  while (!transaction_stack_.empty ())
    {
      auto *current_tx = transaction_stack_.back ();
      if (current_tx && !current_tx->IsFinished ())
        {
          // Found an active transaction, use it
          return current_tx->Execute (query, params);
        }
      // Transaction is finished or null, remove it and check next
      transaction_stack_.pop_back ();
    }
  // No active transactions, execute directly
  return ExecuteDirect (query, params);
}

std::unique_ptr<Transaction>
SQLiteStore::Begin ()
{
  SQLiteTransaction *parent
      = transaction_stack_.empty () ? nullptr : transaction_stack_.back ();

  // If this is a root transaction, start the database transaction
  if (parent == nullptr && !in_transaction_)
    {
      if (!IsDbInTransaction ())
        {
          ExecuteDirect (ResolveBeginTransactionSql ());
        }
      in_transaction_ = true;
    }

  auto transaction = std::make_unique<SQLiteTransaction> (this, parent);
  SQLiteTransaction *transaction_ptr = transaction.get ();
  transaction_stack_.push_back (transaction_ptr);
  return transaction;
}

void
SQLiteStore::Commit ()
{
  if (transaction_stack_.empty ())
    {
      throw NoActiveTransactionError ("No active transaction to commit");
    }

  auto current_tx = transaction_stack_.back ();

  if (current_tx->parent_transaction_ == nullptr)
    {
      if (in_transaction_)
        {
          if (IsDbInTransaction ())
            {
              ExecuteDirect ("COMMIT");
            }
          in_transaction_ = false;
        }
    }
  else
    {
      ExecuteDirect ("RELEASE SAVEPOINT " + current_tx->savepoint_name_);
    }

  current_tx->is_committed_ = true;
  transaction_stack_.pop_back ();
}

void
SQLiteStore::Rollback ()
{
  if (transaction_stack_.empty ())
    {
      throw NoActiveTransactionError ("No active transaction to rollback");
    }

  auto current_tx = transaction_stack_.back ();

  if (current_tx->parent_transaction_ == nullptr)
    {
      if (in_transaction_)
        {
          if (IsDbInTransaction ())
            {
              ExecuteDirect ("ROLLBACK");
            }
          in_transaction_ = false;
        }
    }
  else
    {
      ExecuteDirect ("ROLLBACK TO SAVEPOINT " + current_tx->savepoint_name_);
      ExecuteDirect ("RELEASE SAVEPOINT " + current_tx->savepoint_name_);
    }

  current_tx->is_rolled_back_ = true;
  transaction_stack_.pop_back ();
}

void
SQLiteStore::CommitRootTransaction (SQLiteTransaction *transaction)
{
  while (!transaction_stack_.empty ()
         && transaction_stack_.back () != transaction
         && transaction_stack_.back ()->IsFinished ())
    {
      transaction_stack_.pop_back ();
    }
  if (transaction_stack_.empty () || transaction_stack_.back () != transaction)
    {
      throw TransactionNotCurrentError (
          "Transaction is not the current transaction");
    }

  // Only commit if this is the root transaction (no parent)
  if (transaction->parent_transaction_ == nullptr)
    {
      if (in_transaction_)
        {
          if (IsDbInTransaction ())
            {
              ExecuteDirect ("COMMIT");
            }
          in_transaction_ = false;
        }
    }

  // Remove from stack
  transaction_stack_.pop_back ();
}

void
SQLiteStore::RollbackRootTransaction (SQLiteTransaction *transaction)
{
  while (!transaction_stack_.empty ()
         && transaction_stack_.back () != transaction
         && transaction_stack_.back ()->IsFinished ())
    {
      transaction_stack_.pop_back ();
    }
  if (transaction_stack_.empty () || transaction_stack_.back () != transaction)
    {
      throw TransactionNotCurrentError (
          "Transaction is not the current transaction");
    }

  // Only rollback if this is the root transaction (no parent)
  if (transaction->parent_transaction_ == nullptr)
    {
      if (in_transaction_)
        {
          if (IsDbInTransaction ())
            {
              ExecuteDirect ("ROLLBACK");
            }
          in_transaction_ = false;
        }
    }

  // Remove from stack
  transaction_stack_.pop_back ();
}

std::vector<std::map<std::string, std::any> >
SQLiteStore::ExecuteDirect (const std::string &query,
                            const std::vector<std::any> &params)
{
  const std::string_view op = ParseDbOperation (query);
  telemetry::ScopedSpan span (
      "cortext.db.execute",
      { telemetry::Attribute::String ("db.system", "sqlite"),
        telemetry::Attribute::String ("db.operation", op) });
  sqlite3_stmt *stmt = nullptr;
  std::vector<std::map<std::string, std::any> > results;
  try
    {
      stmt = AcquireStatement (query);
    }
  catch (const StoreError &)
    {
      const int errcode = sqlite3_errcode (connection_->GetConnection ());
      const int prepare_rc = errcode;
      const int extended_errcode
          = sqlite3_extended_errcode (connection_->GetConnection ());
      const char *errstr = sqlite3_errstr (errcode);
      const char *errmsg = sqlite3_errmsg (connection_->GetConnection ());
      span.SetStatusError ("sqlite.prepare_failed");
      telemetry::LogError (
          "SQLite prepare failed",
          { telemetry::Attribute::String ("component", "store"),
            telemetry::Attribute::String ("db.system", "sqlite"),
            telemetry::Attribute::String ("db.operation", op),
            telemetry::Attribute::Int64 ("sqlite.rc",
                                         static_cast<std::int64_t> (prepare_rc)),
            telemetry::Attribute::Int64 ("sqlite.errcode",
                                         static_cast<std::int64_t> (errcode)),
            telemetry::Attribute::Int64 (
                "sqlite.extended_errcode",
                static_cast<std::int64_t> (extended_errcode)),
            telemetry::Attribute::String ("sqlite.errmsg",
                                          errmsg ? errmsg : ""),
            telemetry::Attribute::String ("sqlite.errstr",
                                          errstr ? errstr : "") });
      throw;
    }
  try
    {
      BindParameters (stmt, params);
    }
  catch (...)
    {
      sqlite3_reset (stmt);
      sqlite3_clear_bindings (stmt);
      throw;
    }
  int rc;
  try
    {
      while ((rc = sqlite3_step (stmt)) == SQLITE_ROW)
        {
          std::map<std::string, std::any> row;
          FetchResultRow (stmt, row);
          results.push_back (std::move (row));
        }
    }
  catch (...)
    {
      sqlite3_reset (stmt);
      throw;
    }
  if (rc != SQLITE_DONE && rc != SQLITE_OK)
    {
      std::string error_msg = sqlite3_errmsg (connection_->GetConnection ());
      const int errcode = sqlite3_errcode (connection_->GetConnection ());
      const int extended_errcode
          = sqlite3_extended_errcode (connection_->GetConnection ());
      const char *errstr = sqlite3_errstr (errcode);
      EvictStatement (query);
      span.SetStatusError ("sqlite.step_failed");
      telemetry::LogError (
          "SQLite step failed",
          { telemetry::Attribute::String ("component", "store"),
            telemetry::Attribute::String ("db.system", "sqlite"),
            telemetry::Attribute::String ("db.operation", op),
            telemetry::Attribute::Int64 ("sqlite.step_rc",
                                         static_cast<std::int64_t> (rc)),
            telemetry::Attribute::Int64 ("sqlite.errcode",
                                         static_cast<std::int64_t> (errcode)),
            telemetry::Attribute::Int64 (
                "sqlite.extended_errcode",
                static_cast<std::int64_t> (extended_errcode)),
            telemetry::Attribute::String ("sqlite.errmsg", error_msg),
            telemetry::Attribute::String ("sqlite.errstr",
                                          errstr ? errstr : "") });
      throw StoreError ("Query execution failed: " + error_msg);
    }
  sqlite3_reset (stmt);
  span.SetStatusOk ();
  return results;
}

void
SQLiteStore::Close ()
{
  // The connection is automatically closed by the RAII wrapper when destroyed
  // The store doesn't own the connection lifecycle in the same way as Python,
  // but the RAII wrapper ensures proper cleanup
}

void
SQLiteStore::Checkpoint (bool full)
{
  const int mode
      = full ? SQLITE_CHECKPOINT_TRUNCATE : SQLITE_CHECKPOINT_PASSIVE;
  int wal_log = 0;
  int wal_ckpt = 0;
  int rc = sqlite3_wal_checkpoint_v2 (connection_->GetConnection (), nullptr,
                                      mode, &wal_log, &wal_ckpt);

  // SQLITE_OK = success, SQLITE_BUSY = couldn't complete (acceptable for PASSIVE)
  if (rc != SQLITE_OK && rc != SQLITE_BUSY)
    {
      throw StoreError (
          "Checkpoint failed: "
          + std::string (sqlite3_errmsg (connection_->GetConnection ())));
    }
}

SQLiteStore::WalStatus
SQLiteStore::GetWalStatus () const
{
  int wal_log = 0;
  int wal_ckpt = 0;
  // Use PASSIVE mode to get status without blocking
  sqlite3_wal_checkpoint_v2 (connection_->GetConnection (), nullptr,
                             SQLITE_CHECKPOINT_PASSIVE, &wal_log, &wal_ckpt);
  return { wal_log, wal_ckpt };
}

int
SQLiteStore::WalAutoCheckpointPages () const
{
  if (!connection_ || !connection_->IsValid ())
    {
      return 0;
    }
  auto rows = const_cast<SQLiteStore *> (this)->ExecuteDirect (
      "PRAGMA wal_autocheckpoint", {});
  if (rows.empty () || rows[0].count ("wal_autocheckpoint") == 0)
    {
      return 0;
    }
  try
    {
      return static_cast<int> (
          std::any_cast<long long> (rows[0].at ("wal_autocheckpoint")));
    }
  catch (const std::bad_any_cast &)
    {
      return 0;
    }
}

std::uintmax_t
SQLiteStore::WalFileBytes () const
{
  if (!connection_ || !connection_->IsValid ())
    {
      return 0;
    }
  const char *db_filename = sqlite3_db_filename (
      connection_->GetConnection (), "main");
  if (!db_filename || std::string_view (db_filename) == ":memory:")
    {
      return 0;
    }
  std::error_code ec;
  const auto wal_path = std::filesystem::path (db_filename).concat ("-wal");
  const auto bytes = std::filesystem::file_size (wal_path, ec);
  return ec ? 0 : bytes;
}

void
SQLiteStore::UnregisterTransaction (SQLiteTransaction *tx)
{
  auto it = std::find (transaction_stack_.begin (), transaction_stack_.end (), tx);
  if (it != transaction_stack_.end ())
    {
      transaction_stack_.erase (it);
    }
}

} // namespace cortext
