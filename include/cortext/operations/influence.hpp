/// @file
#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 19: Embedding-Based Influence Feedback
/// (mean-aggregated).
///
/// Computes per-memory influence from contextual gain (semantic similarity),
/// similarity to the current embedding, and drift contribution from the change
/// in embedding space. Applies the mean influence to adjust derived parameters:
/// attention width, target write rate, and threshold hysteresis.
class ApplyInfluenceFeedback
    : public Operation<Requires<tags::MemoryUsageEvents, tags::RetrievedMemoryEmbeddings>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
