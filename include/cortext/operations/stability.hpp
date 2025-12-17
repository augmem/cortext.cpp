#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 5: Stability Priors.
class InitializeStabilityPriors : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Implements Algorithm 6: Stability dynamic update.
class UpdateStability : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
