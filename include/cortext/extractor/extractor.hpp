#pragma once

#include "cortext/operations/extraction.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace cortext
{

/// @brief Abstract interface for entity/relation extraction.
///
/// Implementations may use different backends (e.g., Phi-4 via OGA, external
/// APIs, etc.) to extract structured information from text or audio.
class Extractor
{
public:
  virtual ~Extractor () = default;

  /// @brief Extract entities and relations from text using JSON schema
  /// constraint.
  /// @param text The input text to extract from
  /// @param schema JSON schema defining the expected output structure
  /// @return ExtractionResult containing entities and relations
  virtual operations::ExtractionResult
  ExtractFromText (const std::string &text, const nlohmann::json &schema) = 0;

  /// @brief Extract entities and relations from audio using JSON schema
  /// constraint.
  /// @param pcm Raw PCM audio samples (16kHz, mono, float32)
  /// @param num_samples Number of samples in the PCM buffer
  /// @param schema JSON schema defining the expected output structure
  /// @return ExtractionResult containing entities and relations
  virtual operations::ExtractionResult
  ExtractFromAudio (const float *pcm, size_t num_samples,
                    const nlohmann::json &schema)
      = 0;

  /// @brief Check if the extractor is available (model loaded).
  /// @return true if the extractor can process requests
  virtual bool IsAvailable () const = 0;
};

} // namespace cortext
