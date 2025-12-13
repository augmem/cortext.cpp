#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Gate that runs consolidation scoring and extraction jobs
///        only when EvaluateConsolidation signaled start.
///
/// This operation dynamically instantiates ScoreConsolidation and
/// EnqueueExtractionJobs during Execute(). CollectSchema() forwards to
/// these operations to ensure their table migrations are applied.
class ConsolidationGate : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
  void CollectSchema (cortext::store::SchemaRegistry &registry) const override;
};

} // namespace cortext::operations
