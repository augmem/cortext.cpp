/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Fallback goal alignment computation using baked affect centroids.
///
/// If graph-based goal alignment did not produce `Metric::goal_alignment`, this
/// operation computes it from affect centroids and stores it in [0,1].
class ComputeGoalAlignmentFallback : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations


