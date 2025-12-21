#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace cortext::operations
{

/// @brief Label extracted from consolidated text.
struct ExtractedLabel
{
  std::string label;
  double salience;    ///< [0,1] importance score
};

/// @brief Relation extracted from consolidated text.
struct ExtractedRelation
{
  std::string subject;
  std::string predicate; ///< "works_at", "is_a", "causes", etc.
  std::string object;
  double confidence;     ///< [0,1] extraction confidence
};

/// @brief Request for external LLM extraction.
///
/// Created by ConsolidationSummarize for clusters meeting
/// MinClusterSizeForExtraction threshold.
struct ExtractionRequest
{
  std::string summary_id;
  std::string summary_text;
  std::vector<std::string> source_texts;
  int cluster_size;
  uint64_t created_at;
};

/// @brief Result from external LLM extraction.
///
/// Submitted via Cortext::SubmitExtractionResults() after
/// processing by external extraction system.
struct ExtractionResult
{
  std::string summary_id;
  std::vector<ExtractedLabel> labels;
  std::vector<ExtractedRelation> relations;
};

/// @brief Callback type for extraction requests.
///
/// Registered via Cortext::SetExtractionCallback() to receive
/// batches of extraction requests during consolidation.
using ExtractionCallback
    = std::function<void (const std::vector<ExtractionRequest> &)>;

} // namespace cortext::operations
