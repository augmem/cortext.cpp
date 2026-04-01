#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Shallow consolidation: embedding-only labeling and graph priming.
///
/// Creates association centroids without summaries and attaches labels by
/// embedding similarity. This phase runs only when explicitly requested.
class ConsolidationShallow : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
