#pragma once

#include <memory>

namespace cortext
{

class OperationContext;

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
  virtual void Execute (OperationContext &context) const = 0;
};

} // namespace cortext
