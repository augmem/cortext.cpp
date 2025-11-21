#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 3: Sensitivity Priors.
class InitializeSensitivityPriors : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

/// @brief Implements Algorithm 4: Sensitivity Dynamic Update per Signal.
class UpdateSensitivity : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
