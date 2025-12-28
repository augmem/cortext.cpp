#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Parameters for summarization operation.
struct ConsolidationSummarizeParams
{
  int min_cluster_size_for_extraction; ///< Minimum cluster size to trigger extraction
  int max_source_texts;                ///< Max number of source texts per summary
  int max_total_chars;                 ///< Max total chars across source texts
  int max_text_chars;                  ///< Max chars per source text
  int max_summary_words;               ///< Max summary words (<=0 means uncapped)

  static ConsolidationSummarizeParams FromKnobs (double F, double S, double T);
};

/// @brief Algorithm 29c: Create summaries from clusters.
///
/// For each valid cluster from ConsolidationCluster:
/// - Generates unique summary_id
/// - Fetches its text from objstore as summary_text (Gemma LiteRT-LM required)
/// - Creates MEMORIES entry (kind='ASSOCIATION') with centroid embedding
/// - Creates ASSOCIATIONS edges (edge_type='derived_from') linking centroid to sources
/// - Queues ExtractionRequest for clusters meeting MinClusterSizeForExtraction
///
/// Input:
/// - context.GetConsolidationClusters() (from ConsolidationCluster)
/// - memories table (for finding representative text)
/// - objstore (for fetching payload text)
///
/// Output:
/// - MEMORIES entry for each cluster centroid (kind='ASSOCIATION')
/// - ASSOCIATIONS edges linking centroid to source memories
/// - embeddings entry for each cluster centroid
/// - context.SetExtractionRequests(requests) for EnqueueExtractionJobs
class ConsolidationSummarize : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
