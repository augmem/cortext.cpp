#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 10.4a: Drift Accumulation Tracking.
///
/// Computes: drift_accum += dist(current_embedding, last_check_embedding)
/// Runs after ComputeCoherence to leverage context state.
class UpdateDriftAccumulation : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
