#include "cortext/extractor/gemma_extractor.hpp"

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

struct GemmaExtractor::Impl
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

  operations::ExtractionResult
  ExtractWithConversation (const std::string &prompt,
                           const nlohmann::json &schema)
  {
    if (!available || !engine)
      {
        throw std::runtime_error ("GemmaExtractor: Model not available");
      }

    // Build extraction tool definition based on schema
    nlohmann::ordered_json entity_props;
    entity_props["name"] = { { "type", "string" } };
    entity_props["type"] = { { "type", "string" } };
    entity_props["salience"] = { { "type", "number" } };

    nlohmann::ordered_json relation_props;
    relation_props["subject"] = { { "type", "string" } };
    relation_props["predicate"] = { { "type", "string" } };
    relation_props["object"] = { { "type", "string" } };
    relation_props["confidence"] = { { "type", "number" } };

    nlohmann::ordered_json parameters;
    parameters["type"] = "object";
    parameters["properties"]["entities"]["type"] = "array";
    parameters["properties"]["entities"]["items"]["type"] = "object";
    parameters["properties"]["entities"]["items"]["properties"] = entity_props;
    parameters["properties"]["relations"]["type"] = "array";
    parameters["properties"]["relations"]["items"]["type"] = "object";
    parameters["properties"]["relations"]["items"]["properties"] = relation_props;

    nlohmann::ordered_json extraction_tool;
    extraction_tool["type"] = "function";
    extraction_tool["function"]["name"] = "extract_entities_and_relations";
    extraction_tool["function"]["description"]
        = "Extract entities and relations from the given text";
    extraction_tool["function"]["parameters"] = parameters;

    // Create preface with tool definition
    litert::lm::JsonPreface preface;
    preface.tools = nlohmann::ordered_json::array ({ extraction_tool });

    // Create conversation config with constrained decoding enabled
    auto config_result = litert::lm::ConversationConfig::CreateDefault (
        *engine,
        litert::lm::Preface{ preface },
        std::nullopt, // default prompt template
        std::nullopt, // default processor config
        true          // enable_constrained_decoding
    );

    if (!config_result.ok ())
      {
        throw std::runtime_error (
            "GemmaExtractor: Failed to create conversation config");
      }

    // Create conversation
    auto conversation_result
        = litert::lm::Conversation::Create (*engine, *config_result);
    if (!conversation_result.ok ())
      {
        throw std::runtime_error (
            "GemmaExtractor: Failed to create conversation");
      }
    auto &conversation = *conversation_result;

    // Build the extraction prompt
    std::string extraction_prompt
        = "Extract all entities and relations from the following text. Use "
          "the extract_entities_and_relations function to return the "
          "results.\n\nText: "
        + prompt;

    // Send message
    litert::lm::JsonMessage message
        = { { "role", "user" }, { "content", extraction_prompt } };
    auto response = conversation->SendMessage (litert::lm::Message{ message });

    if (!response.ok ())
      {
        throw std::runtime_error ("GemmaExtractor: Inference failed");
      }

    // Parse the response (tool call format)
    operations::ExtractionResult result;
    try
      {
        auto &response_message = std::get<litert::lm::JsonMessage> (*response);

        // Check for tool calls in the response
        if (response_message.contains ("tool_calls"))
          {
            for (const auto &tool_call : response_message["tool_calls"])
              {
                if (tool_call.contains ("function")
                    && tool_call["function"].contains ("arguments"))
                  {
                    auto args_str
                        = tool_call["function"]["arguments"].get<std::string> ();
                    auto args = nlohmann::json::parse (args_str);

                    // Extract entities
                    if (args.contains ("entities"))
                      {
                        for (const auto &entity : args["entities"])
                          {
                            operations::ExtractedEntity e;
                            e.name = entity.value ("name", "");
                            e.type = entity.value ("type", "");
                            e.salience = entity.value ("salience", 0.5);
                            result.entities.push_back (std::move (e));
                          }
                      }

                    // Extract relations
                    if (args.contains ("relations"))
                      {
                        for (const auto &relation : args["relations"])
                          {
                            operations::ExtractedRelation r;
                            r.subject = relation.value ("subject", "");
                            r.predicate = relation.value ("predicate", "");
                            r.object = relation.value ("object", "");
                            r.confidence = relation.value ("confidence", 0.5);
                            result.relations.push_back (std::move (r));
                          }
                      }
                  }
              }
          }
        // Also check for direct content (in case model returns JSON directly)
        else if (response_message.contains ("content"))
          {
            auto content = response_message["content"].get<std::string> ();
            auto json_output = nlohmann::json::parse (content);

            if (json_output.contains ("entities"))
              {
                for (const auto &entity : json_output["entities"])
                  {
                    operations::ExtractedEntity e;
                    e.name = entity.value ("name", "");
                    e.type = entity.value ("type", "");
                    e.salience = entity.value ("salience", 0.5);
                    result.entities.push_back (std::move (e));
                  }
              }

            if (json_output.contains ("relations"))
              {
                for (const auto &relation : json_output["relations"])
                  {
                    operations::ExtractedRelation r;
                    r.subject = relation.value ("subject", "");
                    r.predicate = relation.value ("predicate", "");
                    r.object = relation.value ("object", "");
                    r.confidence = relation.value ("confidence", 0.5);
                    result.relations.push_back (std::move (r));
                  }
              }
          }
      }
    catch (const nlohmann::json::exception &)
      {
        // Failed to parse JSON, return empty result
      }

    return result;
  }

  operations::ExtractionResult
  ExtractFromAudioImpl (const float *pcm, size_t num_samples,
                        const nlohmann::json &schema)
  {
    if (!available || !engine)
      {
        throw std::runtime_error ("GemmaExtractor: Model not available");
      }

    // Preprocess audio to mel-spectrogram
    auto features = ExtractGemmaAudioFeatures (pcm, num_samples);

    // Build extraction tool definition
    nlohmann::ordered_json entity_props;
    entity_props["name"] = { { "type", "string" } };
    entity_props["type"] = { { "type", "string" } };
    entity_props["salience"] = { { "type", "number" } };

    nlohmann::ordered_json relation_props;
    relation_props["subject"] = { { "type", "string" } };
    relation_props["predicate"] = { { "type", "string" } };
    relation_props["object"] = { { "type", "string" } };
    relation_props["confidence"] = { { "type", "number" } };

    nlohmann::ordered_json parameters;
    parameters["type"] = "object";
    parameters["properties"]["entities"]["type"] = "array";
    parameters["properties"]["entities"]["items"]["type"] = "object";
    parameters["properties"]["entities"]["items"]["properties"] = entity_props;
    parameters["properties"]["relations"]["type"] = "array";
    parameters["properties"]["relations"]["items"]["type"] = "object";
    parameters["properties"]["relations"]["items"]["properties"] = relation_props;

    nlohmann::ordered_json extraction_tool;
    extraction_tool["type"] = "function";
    extraction_tool["function"]["name"] = "extract_entities_and_relations";
    extraction_tool["function"]["description"]
        = "Extract entities and relations from the given audio";
    extraction_tool["function"]["parameters"] = parameters;

    // Create preface with tool definition
    litert::lm::JsonPreface preface;
    preface.tools = nlohmann::ordered_json::array ({ extraction_tool });

    // Create conversation config with audio modality enabled
    litert::lm::SessionConfig session_config
        = litert::lm::SessionConfig::CreateDefault ();
    session_config.SetAudioModalityEnabled (true);

    auto config_result = litert::lm::ConversationConfig::CreateFromSessionConfig (
        *engine, session_config,
        litert::lm::Preface{ preface },
        std::nullopt, // default processor config
        true          // enable_constrained_decoding
    );

    if (!config_result.ok ())
      {
        throw std::runtime_error (
            "GemmaExtractor: Failed to create conversation config");
      }

    // Create conversation
    auto conversation_result
        = litert::lm::Conversation::Create (*engine, *config_result);
    if (!conversation_result.ok ())
      {
        throw std::runtime_error (
            "GemmaExtractor: Failed to create conversation");
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
              { { "type", "text" },
                { "text",
                  "Extract all entities and relations from this audio. Use "
                  "the extract_entities_and_relations function to return the "
                  "results." } } }) }
    };

    auto response = conversation->SendMessage (litert::lm::Message{ message });

    if (!response.ok ())
      {
        throw std::runtime_error ("GemmaExtractor: Audio inference failed");
      }

    // Parse the response (same as text extraction)
    operations::ExtractionResult result;
    try
      {
        auto &response_message = std::get<litert::lm::JsonMessage> (*response);

        if (response_message.contains ("tool_calls"))
          {
            for (const auto &tool_call : response_message["tool_calls"])
              {
                if (tool_call.contains ("function")
                    && tool_call["function"].contains ("arguments"))
                  {
                    auto args_str
                        = tool_call["function"]["arguments"].get<std::string> ();
                    auto args = nlohmann::json::parse (args_str);

                    if (args.contains ("entities"))
                      {
                        for (const auto &entity : args["entities"])
                          {
                            operations::ExtractedEntity e;
                            e.name = entity.value ("name", "");
                            e.type = entity.value ("type", "");
                            e.salience = entity.value ("salience", 0.5);
                            result.entities.push_back (std::move (e));
                          }
                      }

                    if (args.contains ("relations"))
                      {
                        for (const auto &relation : args["relations"])
                          {
                            operations::ExtractedRelation r;
                            r.subject = relation.value ("subject", "");
                            r.predicate = relation.value ("predicate", "");
                            r.object = relation.value ("object", "");
                            r.confidence = relation.value ("confidence", 0.5);
                            result.relations.push_back (std::move (r));
                          }
                      }
                  }
              }
          }
      }
    catch (const nlohmann::json::exception &)
      {
        // Failed to parse JSON, return empty result
      }

    return result;
  }
};

GemmaExtractor::GemmaExtractor (const std::string &model_path)
    : impl_ (std::make_unique<Impl> (model_path))
{
}

GemmaExtractor::~GemmaExtractor () = default;

GemmaExtractor::GemmaExtractor (GemmaExtractor &&) noexcept = default;
GemmaExtractor &
GemmaExtractor::operator= (GemmaExtractor &&) noexcept = default;

operations::ExtractionResult
GemmaExtractor::ExtractFromText (const std::string &text,
                                 const nlohmann::json &schema)
{
  return impl_->ExtractWithConversation (text, schema);
}

operations::ExtractionResult
GemmaExtractor::ExtractFromAudio (const float *pcm, size_t num_samples,
                                  const nlohmann::json &schema)
{
  return impl_->ExtractFromAudioImpl (pcm, num_samples, schema);
}

bool
GemmaExtractor::IsAvailable () const
{
  return impl_ && impl_->available;
}

#else // CORTEXT_DISABLE_LITERT

// Stub implementation when LiteRT-LM is disabled
struct GemmaExtractor::Impl
{
};

GemmaExtractor::GemmaExtractor (const std::string & /*model_path*/)
    : impl_ (std::make_unique<Impl> ())
{
}

GemmaExtractor::~GemmaExtractor () = default;

GemmaExtractor::GemmaExtractor (GemmaExtractor &&) noexcept = default;
GemmaExtractor &
GemmaExtractor::operator= (GemmaExtractor &&) noexcept = default;

operations::ExtractionResult
GemmaExtractor::ExtractFromText (const std::string & /*text*/,
                                 const nlohmann::json & /*schema*/)
{
  throw std::runtime_error (
      "GemmaExtractor: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

operations::ExtractionResult
GemmaExtractor::ExtractFromAudio (const float * /*pcm*/, size_t /*num_samples*/,
                                  const nlohmann::json & /*schema*/)
{
  throw std::runtime_error (
      "GemmaExtractor: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

bool
GemmaExtractor::IsAvailable () const
{
  return false;
}

#endif // CORTEXT_DISABLE_LITERT

} // namespace cortext
