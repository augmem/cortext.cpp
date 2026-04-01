#include "cortext/summarizer/gemma_summarizer.hpp"

#include <condition_variable>
#include <mutex>
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
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_settings.h"
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
      << "Avoid speculation, role confusion, and meta commentary.\n\n"
      << "Conversation excerpts:\n\n";
  for (size_t i = 0; i < texts.size (); ++i)
    {
      combined << "Excerpt " << (i + 1) << ":\n" << texts[i] << "\n\n";
    }
  return combined.str ();
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
      = litert::lm::Engine::CreateEngine (*std::move (settings));
  if (!engine_result.ok ())
    {
      return false;
    }

  engine_out = std::move (*engine_result);
  return true;
}

} // namespace

struct GemmaSummarizer::Impl
{
  std::unique_ptr<litert::lm::Engine> engine;
  bool available = false;

  static std::string
  ExtractContentText (const nlohmann::json &response_message)
  {
    if (!response_message.contains ("content"))
      {
        return {};
      }
    const auto &content = response_message["content"];
    if (content.is_string ())
      {
        return content.get<std::string> ();
      }
    if (content.is_array ())
      {
        std::ostringstream out;
        for (const auto &item : content)
          {
            if (item.is_string ())
              {
                out << item.get<std::string> ();
              }
            else if (item.is_object ())
              {
                if (item.contains ("text") && item["text"].is_string ())
                  {
                    out << item["text"].get<std::string> ();
                  }
                else if (item.contains ("content")
                         && item["content"].is_string ())
                  {
                    out << item["content"].get<std::string> ();
                  }
              }
          }
        return out.str ();
      }
    return {};
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

    auto config_result = litert::lm::ConversationConfig::CreateFromSessionConfig (
        *engine, session_config);

    if (!config_result.ok ())
      {
        throw std::runtime_error (
            "GemmaSummarizer: Failed to create conversation config");
      }

    // Create conversation
    auto conversation_result
        = litert::lm::Conversation::Create (*engine, *config_result);
    if (!conversation_result.ok ())
      {
        throw std::runtime_error (
            "GemmaSummarizer: Failed to create conversation");
      }
    auto &conversation = *conversation_result;

    // Package mel features as binary data for audio input
    std::string mel_data (
        reinterpret_cast<const char *> (features.mel_features.data ()),
        features.mel_features.size () * sizeof (float));

    // Build message with audio reference
    litert::lm::JsonMessage message = {
      { "role", "user" },
      { "content",
        nlohmann::ordered_json::array (
            { { { "type", "audio" }, { "data", mel_data } },
              { { "type", "text" }, { "text", instruction } } }) }
    };

    auto response = conversation->SendMessage (litert::lm::Message{ message });

    if (!response.ok ())
      {
        throw std::runtime_error ("GemmaSummarizer: Audio inference failed");
      }

    // Extract text content from response
    try
      {
        auto &response_message = std::get<litert::lm::JsonMessage> (*response);
        return ExtractContentText (response_message);
      }
    catch (const std::exception &)
      {
        // Return empty string on parse error
      }

    return {};
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
  return impl_->Generate (prompt, max_words);
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
