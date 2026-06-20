#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Shallow consolidation: embedding-only graph priming.
///
/// Creates association centroids and source edges for consolidation clusters.
/// If a database already contains LABEL memories, matching labels may be linked
/// by embedding similarity; this operation does not create labels or a label
/// bank.
class ConsolidationShallow
    : public Operation<Requires<tags::ConsolidationShouldStart, tags::ConsolidationClusters>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
