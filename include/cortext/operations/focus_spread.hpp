#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 11: Contextual Entropy / Focus Spread.
///
/// Computes normalized entropy of a softmax over kNN cosine similarities
/// between the current signal embedding and the recent context embeddings.
/// Result is written as metric "focus_spread" in [0,1].
class ComputeFocusSpread
    : public Operation<Requires<>, Satisfies<tags::MetricValues> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
