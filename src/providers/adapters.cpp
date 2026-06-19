#include "cortext/providers/adapters.hpp"

#include "cortext/extractor/extraction_parsing.hpp"

#include <utility>

namespace cortext::providers
{

namespace
{

constexpr const char *kSummarizerSystemPrompt
    = "You are a concise memory summarizer. Summarize the provided content "
      "faithfully in third person. Output only the summary.";

constexpr const char *kExtractorSystemPrompt
    = "You are a strict information extractor. Return only JSON valid under "
      "the provided schema.";

GenerateRequest
MakeTextRequest (Role role, const char *system_prompt,
                 const std::vector<std::string> &texts)
{
  GenerateRequest request;
  request.role = role;
  request.system_prompt = system_prompt;
  for (const auto &text : texts)
    {
      ContentPart part;
      part.kind = ContentPart::Kind::Text;
      part.text = text;
      request.parts.push_back (std::move (part));
    }
  return request;
}

} // namespace

ProviderSummarizer::ProviderSummarizer (
    std::shared_ptr<InferenceProvider> provider)
    : provider_ (std::move (provider))
{
}

std::string
ProviderSummarizer::SummarizeTexts (const std::vector<std::string> &texts)
{
  return SummarizeTextsLimited (texts, 0);
}

std::string
ProviderSummarizer::SummarizeTextsLimited (
    const std::vector<std::string> &texts, int max_words)
{
  auto request = MakeTextRequest (Role::Summarizer, kSummarizerSystemPrompt,
                                  texts);
  if (max_words > 0)
    {
      request.system_prompt += " Use at most "
                               + std::to_string (max_words) + " words.";
    }
  return provider_->Generate (request).text;
}

std::vector<std::string>
ProviderSummarizer::SummarizeTextBatches (
    const std::vector<BatchTextItem> &items)
{
  std::vector<std::string> summaries;
  summaries.resize (items.size ());
  if (items.empty ())
    {
      return summaries;
    }

  std::vector<GenerateRequest> requests;
  requests.reserve (items.size ());
  for (const auto &item : items)
    {
      auto request = MakeTextRequest (Role::Summarizer,
                                      kSummarizerSystemPrompt, item.texts);
      if (item.max_words > 0)
        {
          request.system_prompt += " Use at most "
                                   + std::to_string (item.max_words)
                                   + " words.";
        }
      requests.push_back (std::move (request));
    }

  const auto responses = provider_->GenerateBatch (requests);
  for (std::size_t i = 0; i < items.size () && i < responses.size (); ++i)
    {
      summaries[i] = responses[i].text;
    }
  return summaries;
}

std::string
ProviderSummarizer::SummarizeAudio (const float *pcm, size_t num_samples)
{
  AudioSegment segment{ pcm, num_samples };
  return SummarizeAudioSegments ({ segment });
}

std::string
ProviderSummarizer::SummarizeAudioSegments (
    const std::vector<AudioSegment> &segments)
{
  GenerateRequest request;
  request.role = Role::Summarizer;
  request.system_prompt = kSummarizerSystemPrompt;
  for (const auto &segment : segments)
    {
      ContentPart part;
      part.kind = ContentPart::Kind::AudioPcm16k;
      part.pcm.assign (segment.pcm, segment.pcm + segment.num_samples);
      request.parts.push_back (std::move (part));
    }
  return provider_->Generate (request).text;
}

bool
ProviderSummarizer::IsAvailable () const
{
  return provider_ != nullptr && provider_->Health ();
}

ProviderExtractor::ProviderExtractor (
    std::shared_ptr<InferenceProvider> provider)
    : provider_ (std::move (provider))
{
}

operations::ExtractionResult
ProviderExtractor::ExtractFromParts (std::vector<ContentPart> parts,
                                     const nlohmann::json &schema)
{
  GenerateRequest request;
  request.role = Role::Extractor;
  request.system_prompt = kExtractorSystemPrompt;
  request.parts = std::move (parts);
  request.schema = schema;

  // Constraint-capable providers enforce the schema during decoding; the
  // shared parser below additionally tolerates loose output, so a provider
  // falling back to free-form generation still degrades gracefully.
  const auto response = provider_->Generate (request);
  return ParseExtractionResponse (response.text);
}

operations::ExtractionResult
ProviderExtractor::ExtractFromText (const std::string &text,
                                    const nlohmann::json &schema)
{
  ContentPart part;
  part.kind = ContentPart::Kind::Text;
  part.text = text;
  return ExtractFromParts ({ std::move (part) }, schema);
}

operations::ExtractionResult
ProviderExtractor::ExtractFromAudio (const float *pcm, size_t num_samples,
                                     const nlohmann::json &schema)
{
  ContentPart part;
  part.kind = ContentPart::Kind::AudioPcm16k;
  part.pcm.assign (pcm, pcm + num_samples);
  return ExtractFromParts ({ std::move (part) }, schema);
}

std::vector<operations::ExtractionResult>
ProviderExtractor::ExtractBatchFromTexts (
    const std::vector<BatchTextItem> &items, const nlohmann::json &schema)
{
  std::vector<operations::ExtractionResult> results;
  results.resize (items.size ());
  if (items.empty ())
    {
      return results;
    }
  std::vector<GenerateRequest> requests;
  requests.reserve (items.size ());
  for (std::size_t i = 0; i < items.size (); ++i)
    {
      auto request = MakeTextRequest (Role::Extractor, kExtractorSystemPrompt,
                                      { items[i].text });
      request.schema = schema;
      requests.push_back (std::move (request));
    }

  const auto responses = provider_->GenerateBatch (requests);
  for (std::size_t i = 0; i < items.size () && i < responses.size (); ++i)
    {
      results[i] = ParseExtractionResponse (responses[i].text);
      results[i].summary_id = items[i].id;
    }
  return results;
}

bool
ProviderExtractor::IsAvailable () const
{
  return provider_ != nullptr && provider_->Health ();
}

} // namespace cortext::providers
