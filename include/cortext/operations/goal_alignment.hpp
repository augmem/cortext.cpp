/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 33: Graph-enhanced goal alignment.
///
/// If goal nodes exist, expands them by 1 hop and computes a cosine similarity
/// between the current signal embedding and the mean embedding of the expanded
/// goal neighborhood. Result is stored as `Metric::goal_alignment` in [0,1].
///
/// Requires `OperationContext::GetStore()` to be non-null.
class ComputeGoalAlignment : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations

