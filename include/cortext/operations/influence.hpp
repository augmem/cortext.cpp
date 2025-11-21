/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 19: Logprob + Embedding Influence Feedback
/// (mean-aggregated).
///
/// Computes per-memory influence from contextual gain, similarity to the
/// current generation embedding, and drift contribution from the change in
/// generation embedding. Applies the mean influence to adjust derived
/// parameters: attention width, target write rate, and threshold hysteresis.
class ApplyInfluenceFeedback : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
