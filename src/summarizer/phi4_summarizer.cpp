#include "cortext/summarizer/phi4_summarizer.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

#if !defined(CORTEXT_DISABLE_OGA)
#include <ort_genai.h>
#endif

namespace cortext
{

#if !defined(CORTEXT_DISABLE_OGA)

namespace
{

std::string
TrimToWordLimit (const std::string &text, int max_words)
{
  if (max_words <= 0)
    {
      return text;
    }
  std::ostringstream out;
  int count = 0;
  bool in_word = false;
  for (size_t i = 0; i < text.size (); ++i)
    {
      const char c = text[i];
      const bool is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
      if (!is_space && !in_word)
        {
          in_word = true;
          count++;
          if (count > max_words)
            {
              break;
            }
        }
      if (count <= max_words)
        {
          out << c;
        }
      if (is_space)
        {
          in_word = false;
        }
    }
  return out.str ();
}

std::string
BuildSummaryPrompt (const std::vector<std::string> &texts)
{
  std::ostringstream combined;
  combined << "<|user|>Summarize the following texts into a concise summary:\n\n";
  for (size_t i = 0; i < texts.size (); ++i)
    {
      combined << "Text " << (i + 1) << ":\n" << texts[i] << "\n\n";
    }
  combined << "<|end|><|assistant|>";
  return combined.str ();
}

bool
TryCreateModelWithProvider (
    const std::string &model_path, const char *provider,
    std::unique_ptr<OgaModel> &model, std::unique_ptr<OgaTokenizer> &tokenizer,
    std::unique_ptr<OgaMultiModalProcessor> &processor,
    std::string *error_out)
{
  try
    {
      auto config = OgaConfig::Create (model_path.c_str ());
      config->ClearProviders ();
      if (provider != nullptr && provider[0] != '\0')
        {
          config->AppendProvider (provider);
          if (std::string (provider) == "cuda")
            {
              config->SetProviderOption (provider, "enable_cuda_graph", "0");
            }
        }
      model = OgaModel::Create (*config);
      tokenizer = OgaTokenizer::Create (*model);
      processor = OgaMultiModalProcessor::Create (*model);
      return true;
    }
  catch (const std::exception &e)
    {
      if (error_out != nullptr)
        {
          *error_out = e.what ();
        }
      return false;
    }
}

} // namespace

struct Phi4Summarizer::Impl
{
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaTokenizer> tokenizer;
  std::unique_ptr<OgaMultiModalProcessor> processor;
  bool available = false;

  explicit Impl (const std::string &model_path)
  {
    try
      {
        std::string error;
        const char *kGpuProviders[] = { "cuda", "dml", "rocm", "webgpu",
                                        "openvino", "qnn", "nvtensorrtrtx" };
        for (const char *provider : kGpuProviders)
          {
            if (TryCreateModelWithProvider (model_path, provider, model,
                                            tokenizer, processor, &error))
              {
                available = true;
                return;
              }
          }
        error.clear ();
        if (TryCreateModelWithProvider (model_path, nullptr, model, tokenizer,
                                        processor, &error))
          {
            available = true;
            return;
          }
        available = false;
      }
    catch (const std::exception &)
      {
        available = false;
      }
  }

  std::string
  Generate (const std::string &prompt, int max_length = 256)
  {
    if (!available)
      {
        throw std::runtime_error ("Phi4Summarizer: Model not available");
      }

    auto params = OgaGeneratorParams::Create (*model);
    params->SetSearchOption ("max_length", max_length);
    params->SetSearchOption ("temperature", 0.3);

    // Tokenize prompt
    auto sequences = OgaSequences::Create ();
    tokenizer->Encode (prompt.c_str (), *sequences);

    // Generate
    auto generator = OgaGenerator::Create (*model, *params);
    generator->AppendTokenSequences (*sequences);
    while (!generator->IsDone ())
      {
        generator->GenerateNextToken ();
      }

    // Decode output
    const int32_t *output_data = generator->GetSequenceData (0);
    size_t output_count = generator->GetSequenceCount (0);
    auto decoded = tokenizer->Decode (output_data, output_count);
    return std::string (decoded);
  }

  std::string
  GenerateFromAudio (const float *pcm, size_t num_samples,
                     const std::string &instruction, int max_length = 256)
  {
    if (!available)
      {
        throw std::runtime_error ("Phi4Summarizer: Model not available");
      }

    auto params = OgaGeneratorParams::Create (*model);
    params->SetSearchOption ("max_length", max_length);
    params->SetSearchOption ("temperature", 0.3);

    // Create audio input using multimodal processor
    const void *audio_data[] = { pcm };
    size_t audio_sizes[] = { num_samples * sizeof (float) };
    auto audios = OgaAudios::Load (audio_data, audio_sizes, 1);

    // Build prompt with audio
    std::string prompt
        = "<|audio|>" + instruction + "<|end|><|assistant|>";
    auto inputs = processor->ProcessAudios (prompt.c_str (), audios.get ());

    // Generate
    auto generator = OgaGenerator::Create (*model, *params);
    generator->SetInputs (*inputs);
    while (!generator->IsDone ())
      {
        generator->GenerateNextToken ();
      }

    // Decode output
    const int32_t *output_data = generator->GetSequenceData (0);
    size_t output_count = generator->GetSequenceCount (0);
    auto decoded = tokenizer->Decode (output_data, output_count);
    return std::string (decoded);
  }
};

Phi4Summarizer::Phi4Summarizer (const std::string &model_path)
    : impl_ (std::make_unique<Impl> (model_path))
{
}

Phi4Summarizer::~Phi4Summarizer () = default;

Phi4Summarizer::Phi4Summarizer (Phi4Summarizer &&) noexcept = default;
Phi4Summarizer &
Phi4Summarizer::operator= (Phi4Summarizer &&) noexcept = default;

std::string
Phi4Summarizer::SummarizeTexts (const std::vector<std::string> &texts)
{
  return SummarizeTextsLimited (texts, 0);
}

std::string
Phi4Summarizer::SummarizeTextsLimited (const std::vector<std::string> &texts,
                                       int max_words)
{
  if (texts.empty ())
    {
      return {};
    }
  const std::string prompt = BuildSummaryPrompt (texts);
  int max_length = 256;
  if (max_words > 0)
    {
      max_length = std::clamp (max_words * 6, 64, 512);
    }
  const std::string generated = impl_->Generate (prompt, max_length);
  return TrimToWordLimit (generated, max_words);
}

std::string
Phi4Summarizer::SummarizeAudio (const float *pcm, size_t num_samples)
{
  return impl_->GenerateFromAudio (pcm, num_samples,
                                   "Summarize this audio content.", 256);
}

std::string
Phi4Summarizer::SummarizeAudioSegments (const std::vector<AudioSegment> &segments)
{
  if (segments.empty ())
    {
      return {};
    }

  // For multiple segments, summarize each individually then combine
  // (OGA doesn't support multiple audio inputs in one prompt currently)
  std::vector<std::string> segment_summaries;
  segment_summaries.reserve (segments.size ());

  for (const auto &segment : segments)
    {
      std::string summary = impl_->GenerateFromAudio (
          segment.pcm, segment.num_samples,
          "Briefly describe what is said in this audio.", 100);
      segment_summaries.push_back (std::move (summary));
    }

  // Combine segment summaries into final summary
  return SummarizeTexts (segment_summaries);
}

bool
Phi4Summarizer::IsAvailable () const
{
  return impl_ && impl_->available;
}

#else // CORTEXT_DISABLE_OGA

// Stub implementation when OGA is disabled
struct Phi4Summarizer::Impl
{
};

Phi4Summarizer::Phi4Summarizer (const std::string & /*model_path*/)
    : impl_ (std::make_unique<Impl> ())
{
}

Phi4Summarizer::~Phi4Summarizer () = default;

Phi4Summarizer::Phi4Summarizer (Phi4Summarizer &&) noexcept = default;
Phi4Summarizer &
Phi4Summarizer::operator= (Phi4Summarizer &&) noexcept = default;

std::string
Phi4Summarizer::SummarizeTexts (const std::vector<std::string> & /*texts*/)
{
  throw std::runtime_error (
      "Phi4Summarizer: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA");
}

std::string
Phi4Summarizer::SummarizeTextsLimited (const std::vector<std::string> & /*texts*/,
                                       int /*max_words*/)
{
  throw std::runtime_error (
      "Phi4Summarizer: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA");
}

std::string
Phi4Summarizer::SummarizeAudio (const float * /*pcm*/, size_t /*num_samples*/)
{
  throw std::runtime_error (
      "Phi4Summarizer: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA");
}

std::string
Phi4Summarizer::SummarizeAudioSegments (
    const std::vector<AudioSegment> & /*segments*/)
{
  throw std::runtime_error (
      "Phi4Summarizer: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA");
}

bool
Phi4Summarizer::IsAvailable () const
{
  return false;
}

#endif // CORTEXT_DISABLE_OGA

} // namespace cortext
