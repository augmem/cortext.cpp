#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Parameters for summarization operation.
struct ConsolidationSummarizeParams
{
  int min_cluster_size_for_extraction; ///< Minimum cluster size to trigger extraction

  static ConsolidationSummarizeParams FromKnobs (double F, double S, double T);
};

/// @brief Algorithm 29c: Create summaries from clusters.
///
/// For each valid cluster from ConsolidationCluster:
/// - Generates unique summary_id
/// - Finds most representative memory (highest cosine to centroid)
/// - Fetches its text from objstore as summary_text (extractive summarization)
/// - Inserts into consolidation_summaries with centroid blob
/// - Inserts source mappings into consolidation_sources
/// - Creates vec_embeddings entry for centroid
/// - Queues ExtractionRequest for clusters meeting MinClusterSizeForExtraction
///
/// Input:
/// - context.GetConsolidationClusters() (from ConsolidationCluster)
/// - memories table (for finding representative text)
/// - objstore (for fetching payload text)
///
/// Output:
/// - consolidation_summaries table populated
/// - consolidation_sources table populated
/// - vec_embeddings entry for each cluster centroid
/// - context.SetExtractionRequests(requests) for EnqueueExtractionJobs
/// - consolidation_candidates cleared for processed sources
class ConsolidationSummarize : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
