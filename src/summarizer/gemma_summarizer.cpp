#include "cortext/summarizer/gemma_summarizer.hpp"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>

#if !defined(CORTEXT_DISABLE_LITERT)
#include "cortext/audio/gemma_audio.hpp"
#include "cortext/core/thread_config.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-compare"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "litert/c/litert_tensor_buffer.h"
#pragma clang diagnostic pop
#endif

namespace cortext
{

#if !defined(CORTEXT_DISABLE_LITERT)

namespace
{

int
CountWords (const std::string &text)
{
  int count = 0;
  bool in_word = false;
  for (char c : text)
    {
      const bool is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
      if (!is_space && !in_word)
        {
          in_word = true;
          count++;
        }
      if (is_space)
        {
          in_word = false;
        }
    }
  return count;
}

std::string
BuildSummaryPrompt (const std::vector<std::string> &texts)
{
  std::ostringstream combined;
  combined
      << "You are writing a durable memory note from conversation excerpts.\n"
      << "Write a concise factual summary in 1-3 sentences.\n"
      << "Return only the summary text.\n"
      << "Treat lines labeled 'User:' as a human user and lines labeled "
         "'Assistant:' as the assistant.\n"
      << "Summarize the underlying facts and topics, not the mechanics of the "
         "conversation.\n"
      << "Prioritize durable facts about people, events, names, preferences, "
         "plans, and outcomes over banter, greetings, or rhetorical "
         "questions.\n"
      << "Include all major durable facts that fit, especially named people, "
         "projects, technologies, and goals.\n"
      << "If the excerpts contain multiple topics, list them as separate facts "
         "instead of implying they caused each other.\n"
      << "If both user and assistant excerpts restate the same fact, prefer "
         "the underlying fact itself instead of narrating who said it.\n"
      << "Do not repeat the same fact from different perspectives.\n"
      << "State facts directly when possible. Prefer direct factual sentences "
         "over wording like 'The user...' or 'The assistant...'.\n"
      << "Do not use speaker-role subjects or second-person phrasing in the "
         "summary. Avoid 'the user', 'the assistant', 'you', and 'your' when "
         "a concrete named subject or neutral phrasing is available.\n"
      << "Do not write phrases like 'the user said', 'the assistant asked', "
         "'in a conversation', 'they discussed', or 'this occurred after' "
         "unless that wording is necessary for clarity.\n"
      << "Do not infer causality, chronology, identity, or shared beliefs "
         "beyond the text.\n"
      << "Avoid speculation, role confusion, and meta commentary.\n"
      << "Examples:\n"
      << "Bad: The user is building Cortext and the assistant says it is a "
         "memory system.\n"
      << "Good: Cortext is a C plus plus memory system for AI assistants.\n"
      << "Bad: Excerpt 2 indicates the user is focusing on consolidation "
         "quality.\n"
      << "Good: Consolidation quality is the next focus for the Cortext chat "
         "application.\n"
      << "Bad: In a conversation, Emily put the tooth under her pillow.\n"
      << "Good: Emily put her lost tooth under her pillow for the tooth "
         "fairy after celebrating with vanilla ice cream.\n\n"
      << "Conversation excerpts:\n\n";
  for (size_t i = 0; i < texts.size (); ++i)
    {
      combined << "Excerpt " << (i + 1) << ":\n" << texts[i] << "\n\n";
    }
  return combined.str ();
}

std::string
TrimAsciiWhitespace (std::string value)
{
  auto is_space = [] (unsigned char c) { return std::isspace (c) != 0; };
  value.erase (
      value.begin (),
      std::find_if (value.begin (), value.end (),
                    [&] (unsigned char c) { return !is_space (c); }));
  value.erase (
      std::find_if (value.rbegin (), value.rend (),
                    [&] (unsigned char c) { return !is_space (c); })
          .base (),
      value.end ());
  return value;
}

std::string
SanitizeSummaryText (std::string summary)
{
  if (summary.empty ())
    {
      return summary;
    }

  summary = std::regex_replace (
      summary, std::regex (R"(Excerpt\s+\d+\s+(mentions|indicates)\s+)",
                           std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(,\s*which is a user\b)", std::regex_constants::icase), "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(\bthe user is focusing on\b)",
                  std::regex_constants::icase),
      "focuses on");
  summary = std::regex_replace (
      summary, std::regex (R"(\bthe user is\b)", std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary, std::regex (R"(\bthe user\b)", std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(\bthe assistant\b)", std::regex_constants::icase), "");
  summary = std::regex_replace (
      summary,
      std::regex (R"([ \t]+)", std::regex_constants::icase), " ");
  summary = std::regex_replace (
      summary, std::regex (R"(\s+\.)", std::regex_constants::icase), ".");
  summary = std::regex_replace (
      summary, std::regex (R"(\n{3,})", std::regex_constants::icase), "\n\n");
  return TrimAsciiWhitespace (summary);
}

bool
TryCreateEngine (const std::string &model_path, litert::lm::Backend backend,
                 std::unique_ptr<litert::lm::Engine> &engine_out)
{
  auto model_assets = litert::lm::ModelAssets::Create (model_path);
  if (!model_assets.ok ())
    {
      return false;
    }

  auto settings = litert::lm::EngineSettings::CreateDefault (
      *std::move (model_assets), backend);
  if (!settings.ok ())
    {
      return false;
    }

  if (backend == litert::lm::Backend::CPU)
    {
      auto &executor_settings = settings->GetMutableMainExecutorSettings ();
      auto cpu_config_result
          = executor_settings.MutableBackendConfig<litert::lm::CpuConfig> ();
      if (cpu_config_result.ok ())
        {
          litert::lm::CpuConfig cpu_config = *cpu_config_result;
          cpu_config.number_of_threads = core::GetInferThreadCount ();
          executor_settings.SetBackendConfig (cpu_config);
        }
    }

  auto engine_result
      = litert::lm::EngineFactory::CreateDefault (*std::move (settings));
  if (!engine_result.ok ())
    {
      return false;
    }

  engine_out = std::move (*engine_result);
  return true;
}

litert::TensorBuffer
BuildAudioTensor (const GemmaAudioFeatures &features)
{
  if (features.num_frames <= 0 || features.mel_bins <= 0)
    {
      throw std::runtime_error ("GemmaSummarizer: Invalid audio feature shape");
    }

  const size_t element_count = features.mel_features.size ();
  const size_t byte_count = element_count * sizeof (float);
  void *host_buffer = nullptr;
  if (posix_memalign (&host_buffer, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                      byte_count)
          != 0
      || host_buffer == nullptr)
    {
      throw std::runtime_error (
          "GemmaSummarizer: Audio buffer allocation failed");
    }
  std::memcpy (host_buffer, features.mel_features.data (), byte_count);

  LiteRtLayout layout{};
  layout.rank = 3;
  layout.has_strides = false;
  layout.dimensions[0] = 1;
  layout.dimensions[1] = static_cast<int32_t> (features.num_frames);
  layout.dimensions[2] = static_cast<int32_t> (features.mel_bins);

  LiteRtRankedTensorType tensor_type{};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout = layout;

  LiteRtTensorBuffer buffer = nullptr;
  LiteRtStatus status = LiteRtCreateTensorBufferFromHostMemory (
      &tensor_type, host_buffer, byte_count, std::free, &buffer);
  if (status != kLiteRtStatusOk || buffer == nullptr)
    {
      std::free (host_buffer);
      throw std::runtime_error (
          "GemmaSummarizer: Audio tensor creation failed");
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  return litert::TensorBuffer::WrapCObject (buffer, litert::OwnHandle::kYes);
#pragma clang diagnostic pop
}

} // namespace

struct GemmaSummarizer::Impl
{
  std::unique_ptr<litert::lm::Engine> engine;
  bool available = false;

  ~Impl ()
  {
    // LiteRT-LM currently segfaults while tearing down Gemma engines on this
    // stack. The summarizer is constructed once per Cortext instance, so
    // releasing here avoids a shutdown crash with bounded process-lifetime
    // memory retention.
    (void)engine.release ();
  }

  explicit Impl (const std::string &model_path)
  {
    try
      {
        if (TryCreateEngine (model_path, litert::lm::Backend::GPU, engine)
            || TryCreateEngine (model_path, litert::lm::Backend::CPU, engine))
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
  Generate (const std::string &prompt, int max_words)
  {
    if (!available || !engine)
      {
        throw std::runtime_error ("GemmaSummarizer: Model not available");
      }

    litert::lm::SessionConfig session_config
        = litert::lm::SessionConfig::CreateDefault ();
#if defined(__APPLE__)
    session_config.SetSamplerBackend (litert::lm::Backend::CPU);
#endif
    auto session_result = engine->CreateSession (session_config);
    if (!session_result.ok ())
      {
        throw std::runtime_error (
            "GemmaSummarizer: Failed to create session");
      }

    std::vector<litert::lm::InputData> inputs;
    inputs.emplace_back (litert::lm::InputText (prompt));

    auto prefill_status = (*session_result)->RunPrefill (inputs);
    if (!prefill_status.ok ())
      {
        throw std::runtime_error ("GemmaSummarizer: Prefill failed");
      }

    auto decode_config = litert::lm::DecodeConfig::CreateDefault ();

    if (max_words <= 0)
      {
        auto response = (*session_result)->RunDecode (decode_config);
        if (!response.ok ())
          {
            throw std::runtime_error ("GemmaSummarizer: Decode failed");
          }

        const auto &texts = response->GetTexts ();
        if (texts.empty ())
          {
            return {};
          }
        return texts.front ();
      }

    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    bool requested_cancel = false;
    bool has_error = false;
    std::string error_message;
    std::string output;

    auto callback =
        [&] (absl::StatusOr<litert::lm::Responses> responses) mutable {
          std::unique_lock<std::mutex> lock (mu);
          if (!responses.ok ())
            {
              has_error = true;
              error_message = responses.status ().ToString ();
              done = true;
              cv.notify_all ();
              return;
            }
          const auto &task_state = responses->GetTaskState ();
          if (litert::lm::IsTaskEndState (task_state))
            {
              done = true;
              cv.notify_all ();
              return;
            }
          if (!responses->GetTexts ().empty ())
            {
              output += responses->GetTexts ()[0];
              if (!requested_cancel && CountWords (output) >= max_words)
                {
                  requested_cancel = true;
                  (*session_result)->CancelProcess ();
                }
            }
        };

    auto async_status = (*session_result)->RunDecodeAsync (callback,
                                                           decode_config);
    if (!async_status.ok ())
      {
        throw std::runtime_error (
            "GemmaSummarizer: Failed to start async decode");
      }

    {
      std::unique_lock<std::mutex> lock (mu);
      cv.wait (lock, [&] { return done; });
    }

    auto wait_status = (*session_result)->WaitUntilDone ();
    if (!wait_status.ok () && !requested_cancel)
      {
        throw std::runtime_error (
            "GemmaSummarizer: WaitUntilDone failed");
      }

    if (has_error && !requested_cancel)
      {
        throw std::runtime_error (
            "GemmaSummarizer: Decode failed: " + error_message);
      }

    return output;
  }

  std::string
  GenerateFromAudio (const float *pcm, size_t num_samples,
                     const std::string &instruction)
  {
    if (!available || !engine)
      {
        throw std::runtime_error ("GemmaSummarizer: Model not available");
      }

    // Preprocess audio to mel-spectrogram
    auto features = ExtractGemmaAudioFeatures (pcm, num_samples);

    // Create session config with audio modality enabled
    litert::lm::SessionConfig session_config
        = litert::lm::SessionConfig::CreateDefault ();
    session_config.SetAudioModalityEnabled (true);
#if defined(__APPLE__)
    session_config.SetSamplerBackend (litert::lm::Backend::CPU);
#endif

    auto session_result = engine->CreateSession (session_config);
    if (!session_result.ok ())
      {
        throw std::runtime_error (
            "GemmaSummarizer: Failed to create audio session");
      }

    litert::TensorBuffer mel_tensor = BuildAudioTensor (features);
    std::vector<litert::lm::InputData> inputs;
    inputs.emplace_back (litert::lm::InputText (instruction
                                                + "\n<start_of_audio>"));
    inputs.emplace_back (litert::lm::InputAudio (std::move (mel_tensor)));
    inputs.emplace_back (litert::lm::InputAudioEnd ());

    auto prefill_status = (*session_result)->RunPrefill (inputs);
    if (!prefill_status.ok ())
      {
        throw std::runtime_error (
            "GemmaSummarizer: Audio prefill failed");
      }

    auto decode_config = litert::lm::DecodeConfig::CreateDefault ();
    auto response = (*session_result)->RunDecode (decode_config);
    if (!response.ok ())
      {
        throw std::runtime_error ("GemmaSummarizer: Audio decode failed");
      }

    const auto &texts = response->GetTexts ();
    return texts.empty () ? std::string{} : texts.front ();
  }
};

GemmaSummarizer::GemmaSummarizer (const std::string &model_path)
    : impl_ (std::make_unique<Impl> (model_path))
{
}

GemmaSummarizer::~GemmaSummarizer () = default;

GemmaSummarizer::GemmaSummarizer (GemmaSummarizer &&) noexcept = default;
GemmaSummarizer &
GemmaSummarizer::operator= (GemmaSummarizer &&) noexcept = default;

std::string
GemmaSummarizer::SummarizeTexts (const std::vector<std::string> &texts)
{
  return SummarizeTextsLimited (texts, 0);
}

std::string
GemmaSummarizer::SummarizeTextsLimited (const std::vector<std::string> &texts,
                                        int max_words)
{
  if (texts.empty ())
    {
      return {};
    }
  const std::string prompt = BuildSummaryPrompt (texts);
  return SanitizeSummaryText (impl_->Generate (prompt, max_words));
}

std::string
GemmaSummarizer::SummarizeAudio (const float *pcm, size_t num_samples)
{
  return impl_->GenerateFromAudio (pcm, num_samples,
                                   "Summarize this audio content.");
}

std::string
GemmaSummarizer::SummarizeAudioSegments (
    const std::vector<AudioSegment> &segments)
{
  if (segments.empty ())
    {
      return {};
    }

  // For multiple segments, summarize each individually then combine
  std::vector<std::string> segment_summaries;
  segment_summaries.reserve (segments.size ());

  for (const auto &segment : segments)
    {
      std::string summary = impl_->GenerateFromAudio (
          segment.pcm, segment.num_samples,
          "Briefly describe what is said in this audio.");
      segment_summaries.push_back (std::move (summary));
    }

  // Combine segment summaries into final summary
  return SummarizeTexts (segment_summaries);
}

bool
GemmaSummarizer::IsAvailable () const
{
  return impl_ && impl_->available;
}

#else // CORTEXT_DISABLE_LITERT

// Stub implementation when LiteRT-LM is disabled
struct GemmaSummarizer::Impl
{
};

GemmaSummarizer::GemmaSummarizer (const std::string & /*model_path*/)
    : impl_ (std::make_unique<Impl> ())
{
}

GemmaSummarizer::~GemmaSummarizer () = default;

GemmaSummarizer::GemmaSummarizer (GemmaSummarizer &&) noexcept = default;
GemmaSummarizer &
GemmaSummarizer::operator= (GemmaSummarizer &&) noexcept = default;

std::string
GemmaSummarizer::SummarizeTexts (const std::vector<std::string> & /*texts*/)
{
  throw std::runtime_error (
      "GemmaSummarizer: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

std::string
GemmaSummarizer::SummarizeTextsLimited (
    const std::vector<std::string> & /*texts*/, int /*max_words*/)
{
  throw std::runtime_error (
      "GemmaSummarizer: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

std::string
GemmaSummarizer::SummarizeAudio (const float * /*pcm*/, size_t /*num_samples*/)
{
  throw std::runtime_error (
      "GemmaSummarizer: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

std::string
GemmaSummarizer::SummarizeAudioSegments (
    const std::vector<AudioSegment> & /*segments*/)
{
  throw std::runtime_error (
      "GemmaSummarizer: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

bool
GemmaSummarizer::IsAvailable () const
{
  return false;
}

#endif // CORTEXT_DISABLE_LITERT

} // namespace cortext
