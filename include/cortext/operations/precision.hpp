/// @file
#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Section 5.5: Adaptive Threshold Modulation (precision).
///
/// Computes ΔThreshold_precision_t from per-signal retrieval precision derived
/// from `OperationContext::MemoryUsageEvent` and stores it on the
/// OperationContext for Algorithm 8 (`UpdateThreshold`) to consume.
class UpdatePrecisionDelta
    : public Operation<Requires<tags::StructuralCoherence>, Satisfies<tags::DeltaThresholdPrecision> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations

