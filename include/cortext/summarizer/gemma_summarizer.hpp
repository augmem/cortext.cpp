#pragma once

#include "cortext/summarizer/summarizer.hpp"
#include <memory>
#include <string>

namespace cortext
{

/// @brief Gemma-based summarizer using LiteRT-LM.
///
/// Uses Gemma model via LiteRT-LM's Conversation API for abstractive
/// summarization of text and audio content.
///
/// When CORTEXT_DISABLE_LITERT is defined, all methods throw std::runtime_error.
class GemmaSummarizer : public Summarizer
{
public:
  /// @brief Construct a GemmaSummarizer.
  /// @param model_path Path to the Gemma model directory (.litertlm format)
  explicit GemmaSummarizer (const std::string &model_path);
  ~GemmaSummarizer () override;

  // Non-copyable
  GemmaSummarizer (const GemmaSummarizer &) = delete;
  GemmaSummarizer &operator= (const GemmaSummarizer &) = delete;

  // Movable
  GemmaSummarizer (GemmaSummarizer &&) noexcept;
  GemmaSummarizer &operator= (GemmaSummarizer &&) noexcept;

  std::string SummarizeTexts (const std::vector<std::string> &texts) override;
  std::string
  SummarizeTextsLimited (const std::vector<std::string> &texts,
                         int max_words) override;

  std::string SummarizeAudio (const float *pcm, size_t num_samples) override;

  std::string
  SummarizeAudioSegments (const std::vector<AudioSegment> &segments) override;

  bool IsAvailable () const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
