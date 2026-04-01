#pragma once

#include "cortext/extractor/extractor.hpp"
#include <memory>
#include <string>

namespace cortext
{

/// @brief Gemma-based extractor using LiteRT-LM.
///
/// Uses Gemma model via LiteRT-LM's Session API with structured prompting
/// for label/relation extraction from text and audio.
///
/// When CORTEXT_DISABLE_LITERT is defined, all methods throw std::runtime_error.
class GemmaExtractor : public Extractor
{
public:
  /// @brief Construct a GemmaExtractor.
  /// @param model_path Path to the Gemma model directory (.litertlm format)
  explicit GemmaExtractor (const std::string &model_path);
  ~GemmaExtractor () override;

  // Non-copyable
  GemmaExtractor (const GemmaExtractor &) = delete;
  GemmaExtractor &operator= (const GemmaExtractor &) = delete;

  // Movable
  GemmaExtractor (GemmaExtractor &&) noexcept;
  GemmaExtractor &operator= (GemmaExtractor &&) noexcept;

  operations::ExtractionResult
  ExtractFromText (const std::string &text,
                   const nlohmann::json &schema) override;

  operations::ExtractionResult ExtractFromAudio (const float *pcm,
                                                 size_t num_samples,
                                                 const nlohmann::json &schema)
      override;

  bool IsAvailable () const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
