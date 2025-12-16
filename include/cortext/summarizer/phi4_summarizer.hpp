#pragma once

#include "cortext/summarizer/summarizer.hpp"
#include <memory>
#include <string>

namespace cortext
{

/// @brief Phi-4 multimodal summarizer using onnxruntime-genai.
///
/// Uses Phi-4-multimodal model for abstractive summarization of text and
/// audio content.
///
/// When CORTEXT_DISABLE_OGA is defined, all methods throw std::runtime_error.
class Phi4Summarizer : public Summarizer
{
public:
  /// @brief Construct a Phi4Summarizer.
  /// @param model_path Path to the Phi-4 model directory
  explicit Phi4Summarizer (const std::string &model_path);
  ~Phi4Summarizer () override;

  // Non-copyable
  Phi4Summarizer (const Phi4Summarizer &) = delete;
  Phi4Summarizer &operator= (const Phi4Summarizer &) = delete;

  // Movable
  Phi4Summarizer (Phi4Summarizer &&) noexcept;
  Phi4Summarizer &operator= (Phi4Summarizer &&) noexcept;

  std::string SummarizeTexts (const std::vector<std::string> &texts) override;

  std::string SummarizeAudio (const float *pcm, size_t num_samples) override;

  std::string
  SummarizeAudioSegments (const std::vector<AudioSegment> &segments) override;

  bool IsAvailable () const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
