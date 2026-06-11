/// @file
#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 17: Stability Feedback Adjustment.
///
/// Applies per-memory contextual gain feedback to adjust the effective
/// stability of used memories and modulate the half-life target accordingly.
/// This operation updates ProcessorContext.half_life via EWMA toward a target
/// derived from BaseHalfLifePrior(T) and the aggregated stability adjustment.
class ApplyStabilityFeedback
    : public Operation<Requires<tags::MemoryUsageEvents>, Satisfies<tags::DeltaHalfLifeAdjustment> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
