/// @file
#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 15: Focus Feedback Adjustment.
///
/// Applies per-memory contextual gain feedback to adjust focus-derived
/// parameters. Positive contextual gain increases relevance weighting and
/// narrows attention width; non-positive gain widens attention width.
class ApplyFocusFeedback
    : public Operation<Requires<tags::AccumulatorWindowState, tags::MemoryUsageEvents>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
