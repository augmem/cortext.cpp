#include "cortext/store/sqlite_store.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string_view>

#include "cortext/store/extension_loader.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext
{

// SQLiteConnection implementation
SQLiteConnection::SQLiteConnection (const std::string &database_path)
    : connection_ (nullptr)
{
  if (sqlite3_open (database_path.c_str (), &connection_) != SQLITE_OK)
    {
      throw StoreError ("Failed to open database: "
                        + std::string (sqlite3_errmsg (connection_)));
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
  return std::make_unique<SQLiteTransaction> (store_, this);
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

  if (savepoint_name_.empty ())
    {
      // This is a root transaction - rollback via store
      store_->RollbackRootTransaction (this);
    }
  else
    {
      // This is a nested transaction - rollback to savepoint
      std::string query = "ROLLBACK TO SAVEPOINT " + savepoint_name_;
      store_->ExecuteDirect (query);
    }

  is_rolled_back_ = true;
}

// SQLiteStore implementation
std::unique_ptr<SQLiteStore>
SQLiteStore::Create (const std::string &database_path)
{
  auto connection = std::make_unique<SQLiteConnection> (database_path);
  return std::make_unique<SQLiteStore> (std::move (connection));
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

std::vector<std::map<std::string, std::any> >
SQLiteStore::Execute (const std::string &query,
                      const std::vector<std::any> &params)
{
  if (!transaction_stack_.empty ())
    {
      // Use the current (topmost) active transaction
      auto &current_tx = transaction_stack_.back ();
      if (!current_tx->IsFinished ())
        {
          return current_tx->Execute (query, params);
        }
      // Transaction is finished, execute directly
    }
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
      ExecuteDirect ("BEGIN");
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

  // Only commit the root transaction to the database
  if (current_tx->parent_transaction_ == nullptr)
    {
      if (in_transaction_)
        {
          ExecuteDirect ("COMMIT");
          in_transaction_ = false;
        }
    }

  // Remove this transaction from the stack
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

  // Only rollback the root transaction from the database
  if (current_tx->parent_transaction_ == nullptr)
    {
      if (in_transaction_)
        {
          ExecuteDirect ("ROLLBACK");
          in_transaction_ = false;
        }
    }

  // Remove this transaction from the stack
  transaction_stack_.pop_back ();
}

void
SQLiteStore::CommitRootTransaction (SQLiteTransaction *transaction)
{
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
          ExecuteDirect ("COMMIT");
          in_transaction_ = false;
        }
    }

  // Remove from stack
  transaction_stack_.pop_back ();
}

void
SQLiteStore::RollbackRootTransaction (SQLiteTransaction *transaction)
{
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
          ExecuteDirect ("ROLLBACK");
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
  auto ParseDbOperation = [] (const std::string &q) -> std::string_view {
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
  };

  const std::string_view op = ParseDbOperation (query);
  telemetry::ScopedSpan span (
      "cortext.db.execute",
      { telemetry::Attribute::String ("db.system", "sqlite"),
        telemetry::Attribute::String ("db.operation", op) });

  sqlite3_stmt *stmt = nullptr;
  std::vector<std::map<std::string, std::any> > results;

  // Prepare the statement
  const int prepare_rc = sqlite3_prepare_v2 (
      connection_->GetConnection (), query.c_str (), -1, &stmt, nullptr);
  if (prepare_rc != SQLITE_OK)
    {
      span.SetStatusError ("sqlite.prepare_failed");
      telemetry::LogError (
          "SQLite prepare failed",
          { telemetry::Attribute::String ("component", "store"),
            telemetry::Attribute::String ("db.system", "sqlite"),
            telemetry::Attribute::String ("db.operation", op) });
      throw StoreError (
          "Failed to prepare statement: "
          + std::string (sqlite3_errmsg (connection_->GetConnection ())));
    }

  // Bind parameters
  for (size_t i = 0; i < params.size (); ++i)
    {
      int param_index = static_cast<int> (i + 1);
      const auto &param = params[i];

      try
        {
          if (param.type () == typeid (int))
            {
              sqlite3_bind_int (stmt, param_index, std::any_cast<int> (param));
            }
          else if (param.type () == typeid (long))
            {
              sqlite3_bind_int64 (stmt, param_index,
                                  std::any_cast<long> (param));
            }
          else if (param.type () == typeid (long long))
            {
              sqlite3_bind_int64 (stmt, param_index,
                                  std::any_cast<long long> (param));
            }
          else if (param.type () == typeid (double))
            {
              sqlite3_bind_double (stmt, param_index,
                                   std::any_cast<double> (param));
            }
          else if (param.type () == typeid (std::string))
            {
              const std::string &str = std::any_cast<std::string> (param);
              sqlite3_bind_text (stmt, param_index, str.c_str (),
                                 static_cast<int> (str.size ()),
                                 SQLITE_TRANSIENT);
            }
          else if (param.type () == typeid (const char *))
            {
              const char *str = std::any_cast<const char *> (param);
              sqlite3_bind_text (stmt, param_index, str, -1, SQLITE_TRANSIENT);
            }
          else if (param.type () == typeid (std::vector<char>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<char> &> (param);
              sqlite3_bind_blob (
                  stmt, param_index, vec.empty () ? nullptr : vec.data (),
                  static_cast<int> (vec.size ()), SQLITE_TRANSIENT);
            }
          else if (param.type () == typeid (std::vector<unsigned char>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<unsigned char> &> (param);
              sqlite3_bind_blob (
                  stmt, param_index,
                  vec.empty () ? nullptr
                               : reinterpret_cast<const void *> (vec.data ()),
                  static_cast<int> (vec.size ()), SQLITE_TRANSIENT);
            }
          else if (param.type () == typeid (std::vector<float>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<float> &> (param);
              sqlite3_bind_blob (
                  stmt, param_index,
                  vec.empty () ? nullptr
                               : reinterpret_cast<const void *> (vec.data ()),
                  static_cast<int> (vec.size () * sizeof (float)),
                  SQLITE_TRANSIENT);
            }
          else
            {
              // For unsupported types, bind as null
              sqlite3_bind_null (stmt, param_index);
            }
        }
      catch (const std::bad_any_cast &)
        {
          sqlite3_bind_null (stmt, param_index);
        }
    }

  // Execute and collect results
  int rc;
  while ((rc = sqlite3_step (stmt)) == SQLITE_ROW)
    {
      std::map<std::string, std::any> row;
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
      results.push_back (std::move (row));
    }

  if (rc != SQLITE_DONE && rc != SQLITE_OK)
    {
      std::string error_msg = sqlite3_errmsg (connection_->GetConnection ());
      sqlite3_finalize (stmt);
      span.SetStatusError ("sqlite.step_failed");
      telemetry::LogError (
          "SQLite step failed",
          { telemetry::Attribute::String ("component", "store"),
            telemetry::Attribute::String ("db.system", "sqlite"),
            telemetry::Attribute::String ("db.operation", op) });
      throw StoreError ("Query execution failed: " + error_msg);
    }

  sqlite3_finalize (stmt);
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

} // namespace cortext
