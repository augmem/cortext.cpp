#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
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

/// @brief Fact assertion extracted from consolidated text.
struct ExtractedFact
{
  std::string subject;
  std::string predicate;
  std::string object;
  double confidence = 0.5; ///< [0,1] extraction confidence
  std::optional<std::uint64_t> valid_start_ts;
  std::optional<std::uint64_t> valid_end_ts;
};

/// @brief Source payload available to a consolidation extraction request.
struct ExtractionSourceBlob
{
  std::vector<unsigned char> bytes;
  std::string modality;
  std::string mime;
  int sample_rate = 0;
  std::size_t num_samples = 0;
};

/// @brief Request for external LLM extraction.
///
/// Created from clustered source memories for clusters meeting
/// MinClusterSizeForExtraction threshold. `summary_id` is an opaque internal
/// consolidation artifact id for the graph node receiving extracted labels; it
/// may identify an associative cue node, not necessarily an abstractive summary.
struct ExtractionRequest
{
  std::string summary_id;
  std::string summary_text;
  std::vector<std::string> source_texts;
  std::vector<std::string> current_labels;
  std::vector<ExtractionSourceBlob> source_blobs;
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
  std::vector<ExtractedFact> facts;
};

/// @brief Callback type for extraction requests.
///
/// Registered via Cortext::SetExtractionCallback() to receive
/// batches of extraction requests during consolidation.
using ExtractionCallback
    = std::function<void (const std::vector<ExtractionRequest> &)>;

} // namespace cortext::operations
