/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 19: Embedding-Based Influence Feedback
/// (mean-aggregated).
///
/// Computes per-memory influence from contextual gain (semantic similarity),
/// similarity to the current embedding, and drift contribution from the change
/// in embedding space. Applies the mean influence to adjust derived parameters:
/// attention width, target write rate, and threshold hysteresis.
class ApplyInfluenceFeedback : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
