#pragma once

#include "cortext/extractor/extractor.hpp"
#include "cortext/summarizer/summarizer.hpp"

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cortext
{

class Lfm2LlamaSummarizer final : public Summarizer
{
public:
  explicit Lfm2LlamaSummarizer (const std::string &model_path);
  ~Lfm2LlamaSummarizer () override;

  Lfm2LlamaSummarizer (const Lfm2LlamaSummarizer &) = delete;
  Lfm2LlamaSummarizer &operator= (const Lfm2LlamaSummarizer &) = delete;
  Lfm2LlamaSummarizer (Lfm2LlamaSummarizer &&) noexcept;
  Lfm2LlamaSummarizer &operator= (Lfm2LlamaSummarizer &&) noexcept;

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

class Lfm2LlamaExtractor final : public Extractor
{
public:
  explicit Lfm2LlamaExtractor (const std::string &model_path);
  ~Lfm2LlamaExtractor () override;

  Lfm2LlamaExtractor (const Lfm2LlamaExtractor &) = delete;
  Lfm2LlamaExtractor &operator= (const Lfm2LlamaExtractor &) = delete;
  Lfm2LlamaExtractor (Lfm2LlamaExtractor &&) noexcept;
  Lfm2LlamaExtractor &operator= (Lfm2LlamaExtractor &&) noexcept;

  operations::ExtractionResult
  ExtractFromText (const std::string &text, const nlohmann::json &schema)
      override;
  operations::ExtractionResult
  ExtractFromAudio (const float *pcm, size_t num_samples,
                    const nlohmann::json &schema) override;
  bool IsAvailable () const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

namespace internal
{

std::string BuildLfm2ExtractionGrammar (const nlohmann::json &schema);

} // namespace internal

} // namespace cortext
