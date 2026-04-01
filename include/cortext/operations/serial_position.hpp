#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 26: Serial Position Effects.
///
/// Computes and exposes derived parameters based on knobs (F, S)
/// for primacy/recency windows and related multipliers. No database writes.
class ApplySerialPositionEffects : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
