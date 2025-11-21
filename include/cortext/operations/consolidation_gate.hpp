#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Gate that runs consolidation scoring and extraction jobs
///        only when EvaluateConsolidation signaled start.
class ConsolidationGate : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
