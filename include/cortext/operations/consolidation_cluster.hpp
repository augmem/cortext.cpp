#pragma once

#include "cortext/processor/operation.hpp"

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
/// Groups marked candidates from `consolidation_candidates` into clusters
/// based on embedding similarity. Clusters meeting minimum size requirements
/// are passed to the summarization phase via OperationContext.
///
/// This operation uses a greedy clustering algorithm:
/// 1. For each unassigned candidate, start a new cluster
/// 2. Add all candidates similar to the centroid (cosine > merge_threshold)
/// 3. Update centroid as running mean of cluster members
/// 4. Filter clusters by minimum size requirement
///
/// Input:
/// - consolidation_candidates table (populated by ScoreConsolidation)
/// - vec_embeddings table (for embedding vectors)
///
/// Output:
/// - context.SetConsolidationClusters(clusters) for downstream operations
class ConsolidationCluster : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
