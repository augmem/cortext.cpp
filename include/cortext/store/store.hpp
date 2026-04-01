#pragma once

#include <any>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cortext
{

/// @brief Base exception for store errors.
class StoreError : public std::runtime_error
{
public:
  /// @brief Construct a StoreError with a message.
  /// @param message The error message.
  explicit StoreError (const std::string &message)
      : std::runtime_error (message)
  {
  }
};

/// @brief Raised when trying to operate on a transaction that is already
/// committed or rolled back.
class TransactionAlreadyFinishedError : public StoreError
{
public:
  /// @brief Construct a TransactionAlreadyFinishedError with a message.
  /// @param message The error message.
  explicit TransactionAlreadyFinishedError (const std::string &message)
      : StoreError (message)
  {
  }
};

/// @brief Raised when trying to commit or rollback without an active
/// transaction.
class NoActiveTransactionError : public StoreError
{
public:
  /// @brief Construct a NoActiveTransactionError with a message.
  /// @param message The error message.
  explicit NoActiveTransactionError (const std::string &message)
      : StoreError (message)
  {
  }
};

/// @brief Raised when trying to commit or rollback a transaction that is not
/// the current transaction.
class TransactionNotCurrentError : public StoreError
{
public:
  /// @brief Construct a TransactionNotCurrentError with a message.
  /// @param message The error message.
  explicit TransactionNotCurrentError (const std::string &message)
      : StoreError (message)
  {
  }
};

/// @brief Protocol for objects that support transactional operations.
class Transaction
{
public:
  virtual ~Transaction () = default;

  /// @brief Begin a new nested transaction.
  /// @return A unique pointer to the new transaction.
  virtual std::unique_ptr<Transaction> Begin () = 0;

  /// @brief Execute a query within transaction context.
  /// @param query The SQL query to execute.
  /// @param params The parameters for the query.
  /// @return The query results.
  virtual std::vector<std::map<std::string, std::any> >
  Execute (const std::string &query, const std::vector<std::any> &params = {})
      = 0;

  /// @brief Commit the current transaction.
  virtual void Commit () = 0;

  /// @brief Rollback the current transaction.
  virtual void Rollback () = 0;

protected:
  Transaction () = default;
};

/// @brief Database store with transaction support.
class Store
{
public:
  virtual ~Store () = default;

  /// @brief Execute a query and return results.
  /// @param query The SQL query to execute.
  /// @param params The parameters for the query.
  /// @return The query results.
  virtual std::vector<std::map<std::string, std::any> >
  Execute (const std::string &query, const std::vector<std::any> &params = {})
      = 0;

  /// @brief Begin a new transaction.
  /// @return A unique pointer to the new transaction.
  virtual std::unique_ptr<Transaction> Begin () = 0;

  /// @brief Commit the current transaction.
  virtual void Commit () = 0;

  /// @brief Rollback the current transaction.
  virtual void Rollback () = 0;

  /// @brief Close the store.
  virtual void Close () = 0;

protected:
  Store () = default;
};

} // namespace cortext
