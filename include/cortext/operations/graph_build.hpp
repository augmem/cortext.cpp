/// @file
#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 30: Graph Construction from Clusters.
///
/// Builds graph edges (ASSOCIATIONS) from clustered memories.
/// This operation reads MEMORIES with cluster_id (set by consolidation
/// clustering)
/// and creates edges in ASSOCIATIONS based on embedding similarity and temporal
/// patterns.
///
/// Edge types created:
/// - 'co_occurs': Memories with high cosine similarity within a cluster
/// - 'causes': Temporally adjacent memories with significant semantic drift
/// - 'contradicts': Memories with strong negative similarity
/// - 'reinforces': Decay applied to existing reinforcement edges
///
/// Input:
/// - MEMORIES table (with cluster_id from consolidation clustering)
/// - embeddings table (for computing similarities)
///
/// Output:
/// - ASSOCIATIONS edges between memories
class BuildGraphFromConsolidation
    : public Operation<Requires<tags::ConsolidationShouldStart>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
