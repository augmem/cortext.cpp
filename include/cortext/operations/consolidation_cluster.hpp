#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Parameters for clustering operation derived from knobs.
struct ConsolidationClusterParams
{
  double merge_threshold;  ///< Similarity threshold for clustering
  int min_cluster_size;    ///< Minimum members to form valid cluster
  int max_clusters;        ///< Cap on number of clusters per cycle

  static ConsolidationClusterParams FromKnobs (double F, double S, double T);
};

/// @brief Algorithm 29b: Cluster consolidation candidates.
///
/// Groups consolidation candidates into clusters based on embedding similarity.
/// Clusters meeting minimum size requirements are passed to shallow
/// consolidation and graph construction via OperationContext.
///
/// This operation uses a greedy clustering algorithm:
/// 1. For each unassigned candidate, start a new cluster
/// 2. Add all candidates similar to the centroid (cosine > merge_threshold)
/// 3. Update centroid as running mean of cluster members
/// 4. Filter clusters by minimum size requirement
///
/// Input:
/// - context.GetConsolidationCandidates() (populated by ScoreConsolidation)
///
/// Output:
/// - context.SetConsolidationClusters(clusters) for downstream operations
class ConsolidationCluster
    : public Operation<Requires<tags::ConsolidationShouldStart, tags::ConsolidationCandidates>, Satisfies<tags::ConsolidationClusters> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
