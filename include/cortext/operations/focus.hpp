#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 1: Initializes the focus-related priors.
///
/// This operation should typically be run once at the beginning of a session.
class InitializeFocusPriors : public IOperation
{
public:
  /// @brief Calculates and sets the focus priors in the ProcessorContext.
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Implements Algorithm 2: Dynamic Update per Signal (Focus).
///
/// This operation updates the relevance weighting and attention width based
/// on the incoming signal's relationship to the recent context.
class UpdateFocus : public IOperation
{
public:
  /// @brief Executes the dynamic focus update logic.
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
