#pragma once

#include "cortext/extractor/extractor.hpp"
#include "cortext/providers/provider.hpp"
#include "cortext/summarizer/summarizer.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cortext::providers
{

/// @brief Presents any InferenceProvider as the legacy Summarizer interface.
class ProviderSummarizer : public Summarizer
{
public:
  explicit ProviderSummarizer (std::shared_ptr<InferenceProvider> provider);

  std::string SummarizeTexts (const std::vector<std::string> &texts) override;
  std::string SummarizeTextsLimited (const std::vector<std::string> &texts,
                                     int max_words) override;
  std::vector<std::string>
  SummarizeTextBatches (const std::vector<BatchTextItem> &items) override;
  std::string SummarizeAudio (const float *pcm, size_t num_samples) override;
  std::string
  SummarizeAudioSegments (const std::vector<AudioSegment> &segments) override;
  bool IsAvailable () const override;

private:
  std::shared_ptr<InferenceProvider> provider_;
};

/// @brief Presents any InferenceProvider as the legacy Extractor interface.
///
/// Providers with ServerSchema constraints receive the schema in the
/// request; the adapter still validates the returned JSON and retries up to
/// GenerateParams::max_retries before surfacing an empty result.
class ProviderExtractor : public Extractor
{
public:
  explicit ProviderExtractor (std::shared_ptr<InferenceProvider> provider);

  operations::ExtractionResult
  ExtractFromText (const std::string &text,
                   const nlohmann::json &schema) override;
  operations::ExtractionResult
  ExtractFromAudio (const float *pcm, size_t num_samples,
                    const nlohmann::json &schema) override;
  std::vector<operations::ExtractionResult>
  ExtractBatchFromTexts (const std::vector<BatchTextItem> &items,
                         const nlohmann::json &schema) override;
  bool IsAvailable () const override;

private:
  operations::ExtractionResult
  ExtractFromParts (std::vector<ContentPart> parts,
                    const nlohmann::json &schema);

  std::shared_ptr<InferenceProvider> provider_;
};

} // namespace cortext::providers
