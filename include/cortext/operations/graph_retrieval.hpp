/// @file
#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 31: Graph-augmented retrieval (explicit tables).
///
/// Seeds candidates using an embedding similarity scan over `embeddings`, then
/// expands candidates via recursive traversal over `graph_edges` up to a
/// knob-derived depth, then re-ranks by cosine similarity.
///
/// Requires `OperationContext::GetStore()` to be non-null.
class GraphAugmentedRetrieveCandidates
    : public Operation<Requires<tags::Arousal, tags::EmotionIntensity, tags::MetricValues, tags::ShouldCheckRetrieval, tags::WriteExclusionTs>, Satisfies<tags::RetrievedMemoryEmbeddings> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations

