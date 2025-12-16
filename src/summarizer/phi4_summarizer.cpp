#include "cortext/summarizer/phi4_summarizer.hpp"

#include <sstream>
#include <stdexcept>

#if !defined(CORTEXT_DISABLE_OGA)
#include <ort_genai.h>
#endif

namespace cortext
{

#if !defined(CORTEXT_DISABLE_OGA)

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
        model = OgaModel::Create (model_path.c_str ());
        tokenizer = OgaTokenizer::Create (*model);
        processor = OgaMultiModalProcessor::Create (*model);
        available = true;
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
  if (texts.empty ())
    {
      return {};
    }

  // Combine texts with separators
  std::ostringstream combined;
  combined << "<|user|>Summarize the following texts into a concise "
              "summary:\n\n";
  for (size_t i = 0; i < texts.size (); ++i)
    {
      combined << "Text " << (i + 1) << ":\n" << texts[i] << "\n\n";
    }
  combined << "<|end|><|assistant|>";

  return impl_->Generate (combined.str (), 256);
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
