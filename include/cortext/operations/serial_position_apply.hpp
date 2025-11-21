#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Applies Algorithm 26 multipliers to be used by downstream updates.
///
/// Reads the serial position parameters (windows and bonuses) exposed by
/// ApplySerialPositionEffects and computes a per-signal multiplier that
/// represents the average primacy/recency boost across used memories in
/// this signal. The multiplier is stored in OperationContext and is
/// ephemeral (not persisted).
class ApplySerialPositionMultiplier : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
