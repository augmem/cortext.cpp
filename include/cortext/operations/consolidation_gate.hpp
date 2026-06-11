#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Gate that runs consolidation scoring and extraction jobs
///        only when EvaluateConsolidation signaled start.
///
/// This operation dynamically instantiates ScoreConsolidation and
/// EnqueueExtractionJobs during Execute().
class ConsolidationGate
    : public Operation<Requires<tags::ConsolidationShouldStart>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
