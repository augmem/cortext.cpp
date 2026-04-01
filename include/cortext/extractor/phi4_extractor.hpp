#pragma once

#include "cortext/extractor/extractor.hpp"
#include <memory>
#include <string>

namespace cortext
{

/// @brief Phi-4 multimodal extractor using onnxruntime-genai.
///
/// Uses Phi-4-multimodal model with llguidance for JSON schema constrained
/// label/relation extraction from text and audio.
///
/// When CORTEXT_DISABLE_OGA is defined, all methods throw std::runtime_error.
class Phi4Extractor : public Extractor
{
public:
  /// @brief Construct a Phi4Extractor.
  /// @param model_path Path to the Phi-4 model directory
  explicit Phi4Extractor (const std::string &model_path);
  ~Phi4Extractor () override;

  // Non-copyable
  Phi4Extractor (const Phi4Extractor &) = delete;
  Phi4Extractor &operator= (const Phi4Extractor &) = delete;

  // Movable
  Phi4Extractor (Phi4Extractor &&) noexcept;
  Phi4Extractor &operator= (Phi4Extractor &&) noexcept;

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
