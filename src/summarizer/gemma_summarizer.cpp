#include "cortext/summarizer/gemma_summarizer.hpp"

#include <sstream>
#include <stdexcept>

#if !defined(CORTEXT_DISABLE_LITERT)
#include "cortext/audio/gemma_audio.hpp"
#include "cortext/core/thread_config.hpp"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#endif

namespace cortext
{

#if !defined(CORTEXT_DISABLE_LITERT)

struct GemmaSummarizer::Impl
{
  std::unique_ptr<litert::lm::Engine> engine;
  bool available = false;

  explicit Impl (const std::string &model_path)
  {
    try
      {
        // Load model assets
        auto model_assets = litert::lm::ModelAssets::Create (model_path);
        if (!model_assets.ok ())
          {
            available = false;
            return;
          }

        // Create engine settings with CPU backend
        auto settings = litert::lm::EngineSettings::CreateDefault (
            *std::move (model_assets), litert::lm::Backend::CPU);
        if (!settings.ok ())
          {
            available = false;
            return;
          }

        // Configure thread count from environment variable
        auto &executor_settings = settings->GetMutableMainExecutorSettings ();
        auto cpu_config_result
            = executor_settings
                  .MutableBackendConfig<litert::lm::CpuConfig> ();
        if (cpu_config_result.ok ())
          {
            litert::lm::CpuConfig cpu_config = *cpu_config_result;
            cpu_config.number_of_threads = core::GetInferThreadCount ();
            executor_settings.SetBackendConfig (cpu_config);
          }

        // Create engine
        auto engine_result
            = litert::lm::Engine::CreateEngine (*std::move (settings));
        if (!engine_result.ok ())
          {
            available = false;
            return;
          }

        engine = std::move (*engine_result);
        available = true;
      }
    catch (const std::exception &)
      {
        available = false;
      }
  }

  std::string
  Generate (const std::string &prompt)
  {
    if (!available || !engine)
      {
        throw std::runtime_error ("GemmaSummarizer: Model not available");
      }

    // Create default conversation config (no constrained decoding for
    // summarization)
    auto config_result
        = litert::lm::ConversationConfig::CreateDefault (*engine);

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

    // Send message
    litert::lm::JsonMessage message
        = { { "role", "user" }, { "content", prompt } };
    auto response = conversation->SendMessage (litert::lm::Message{ message });

    if (!response.ok ())
      {
        throw std::runtime_error ("GemmaSummarizer: Inference failed");
      }

    // Extract text content from response
    try
      {
        auto &response_message = std::get<litert::lm::JsonMessage> (*response);
        if (response_message.contains ("content"))
          {
            return response_message["content"].get<std::string> ();
          }
      }
    catch (const std::exception &)
      {
        // Return empty string on parse error
      }

    return {};
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
        if (response_message.contains ("content"))
          {
            return response_message["content"].get<std::string> ();
          }
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
  if (texts.empty ())
    {
      return {};
    }

  // Combine texts with separators
  std::ostringstream combined;
  combined << "Summarize the following texts into a concise summary:\n\n";
  for (size_t i = 0; i < texts.size (); ++i)
    {
      combined << "Text " << (i + 1) << ":\n" << texts[i] << "\n\n";
    }

  return impl_->Generate (combined.str ());
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
