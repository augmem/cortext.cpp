#pragma once

#include <string>
#include <vector>

namespace cortext
{

/// @brief Audio segment for multi-segment summarization.
struct AudioSegment
{
  const float *pcm;  ///< Raw PCM samples (16kHz, mono, float32)
  size_t num_samples; ///< Number of samples in the buffer
};

/// @brief Abstract interface for text/audio summarization.
///
/// Implementations may use different backends (e.g., Phi-4 via OGA, external
/// APIs, etc.) to generate summaries from text or audio content.
class Summarizer
{
public:
  virtual ~Summarizer () = default;

  /// @brief Summarize multiple text fragments into a single summary.
  /// @param texts Vector of text fragments to summarize
  /// @return Abstractive summary of the combined content
  virtual std::string SummarizeTexts (const std::vector<std::string> &texts)
      = 0;

  /// @brief Summarize multiple text fragments with a word cap (if > 0).
  /// @param texts Vector of text fragments to summarize
  /// @param max_words Maximum number of words to return (<=0 means no cap)
  /// @return Abstractive summary of the combined content
  virtual std::string
  SummarizeTextsLimited (const std::vector<std::string> &texts, int max_words)
  {
    (void)max_words;
    return SummarizeTexts (texts);
  }

  /// @brief Summarize audio content into text.
  /// @param pcm Raw PCM audio samples (16kHz, mono, float32)
  /// @param num_samples Number of samples in the PCM buffer
  /// @return Summary of the audio content
  virtual std::string SummarizeAudio (const float *pcm, size_t num_samples)
      = 0;

  /// @brief Summarize multiple audio segments into a single summary.
  /// @param segments Vector of audio segments to summarize
  /// @return Combined summary of all audio segments
  virtual std::string
  SummarizeAudioSegments (const std::vector<AudioSegment> &segments) = 0;

  /// @brief Check if the summarizer is available (model loaded).
  /// @return true if the summarizer can process requests
  virtual bool IsAvailable () const = 0;
};

} // namespace cortext
