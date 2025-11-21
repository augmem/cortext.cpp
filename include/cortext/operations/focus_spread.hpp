#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 11: Contextual Entropy / Focus Spread.
///
/// Computes normalized entropy of a softmax over kNN cosine similarities
/// between the current signal embedding and the recent context embeddings.
/// Result is written as metric "focus_spread" in [0,1].
class ComputeFocusSpread : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
