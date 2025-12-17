#pragma once

#include <memory>
#include <vector>

namespace cortext::store
{
class SchemaRegistry;
}

namespace cortext
{

class OperationContext;
class Transaction;

/// @brief Defines the interface for an operation in the signal processing
/// model.
///
/// An operation represents a single, atomic step in the process.
class IOperation
{
public:
  virtual ~IOperation () = default;

  /// @brief Executes the operation.
  /// @param context The context object for the current signal.
  /// @param tx The active database transaction for this signal.
  virtual void Execute (OperationContext &context, Transaction &tx) const = 0;

  /// @brief Registers database schema migrations required by this operation.
  /// @param registry The registry to add migrations to.
  ///
  /// Operations that require specific tables or indexes should override this
  /// and register migrations. This is called once during SignalProcessor
  /// initialization.
  virtual void
  CollectSchema (cortext::store::SchemaRegistry &registry) const
  {
    (void)registry; // Default implementation does nothing
  }
};

} // namespace cortext
