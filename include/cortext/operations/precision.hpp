/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Section 5.5: Adaptive Threshold Modulation (precision).
///
/// Computes ΔThreshold_precision_t from per-signal retrieval precision derived
/// from `OperationContext::MemoryUsageEvent` and stores it on the
/// OperationContext for Algorithm 8 (`UpdateThreshold`) to consume.
class UpdatePrecisionDelta : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations

