#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Gate that runs consolidation scoring and extraction jobs
///        only when EvaluateConsolidation signaled start.
///
/// This operation dynamically instantiates ScoreConsolidation and
/// EnqueueExtractionJobs during Execute().
class ConsolidationGate : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
