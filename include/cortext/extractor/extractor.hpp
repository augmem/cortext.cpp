#pragma once

#include "cortext/operations/extraction.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace cortext
{

/// @brief Abstract interface for label/relation extraction.
///
/// Implementations may use different backends (e.g., Phi-4 via OGA, external
/// APIs, etc.) to extract structured information from text or audio.
class Extractor
{
public:
  struct BatchTextItem
  {
    std::string id;
    std::string text;
  };

  virtual ~Extractor () = default;

  /// @brief Extract labels and relations from text using JSON schema
  /// constraint.
  /// @param text The input text to extract from
  /// @param schema JSON schema defining the expected output structure
  /// @return ExtractionResult containing labels and relations
  virtual operations::ExtractionResult
  ExtractFromText (const std::string &text, const nlohmann::json &schema) = 0;

  /// @brief Extract labels and relations from audio using JSON schema
  /// constraint.
  /// @param pcm Raw PCM audio samples (16kHz, mono, float32)
  /// @param num_samples Number of samples in the PCM buffer
  /// @param schema JSON schema defining the expected output structure
  /// @return ExtractionResult containing labels and relations
  virtual operations::ExtractionResult
  ExtractFromAudio (const float *pcm, size_t num_samples,
                    const nlohmann::json &schema)
      = 0;

  /// @brief Extract labels and relations from multiple independent text
  /// items. Implementations with native/server batching should override this;
  /// the default preserves existing behavior by processing each item
  /// sequentially with ExtractFromText.
  /// @param items Independent text items with caller-stable ids
  /// @param schema JSON schema defining each item's expected output structure
  /// @return One ExtractionResult per input item, in input order
  virtual std::vector<operations::ExtractionResult>
  ExtractBatchFromTexts (const std::vector<BatchTextItem> &items,
                         const nlohmann::json &schema)
  {
    std::vector<operations::ExtractionResult> results;
    results.reserve (items.size ());
    for (const auto &item : items)
      {
        auto result = ExtractFromText (item.text, schema);
        result.summary_id = item.id;
        results.push_back (std::move (result));
      }
    return results;
  }

  /// @brief Check if the extractor is available (model loaded).
  /// @return true if the extractor can process requests
  virtual bool IsAvailable () const = 0;
};

} // namespace cortext
