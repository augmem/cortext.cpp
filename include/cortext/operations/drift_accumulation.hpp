#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 10.4a: Drift Accumulation Tracking.
///
/// Computes: drift_accum += cosine_dist(μ_acc, prev_x)
/// Runs after ComputeCoherence to leverage context state.
class UpdateDriftAccumulation : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
