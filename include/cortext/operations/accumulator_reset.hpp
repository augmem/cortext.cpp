#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Reset accumulator state after a flush/spike decision.
class ResetAccumulatorAfterFlush : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Reset accumulator state after an allowed interrupt to avoid
///        persisting partial thoughts.
class ResetAccumulatorOnInterrupt : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
