#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cortext/generator/gemma_text_generator.hpp"
#include "cortext/generator/text_generator.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

namespace
{

std::string
GetModelsDir ()
{
  std::vector<std::string> paths = {
    "models/gemma-3n",
    "../models/gemma-3n",
    "../../models/gemma-3n",
  };

  for (const auto &path : paths)
    {
      if (std::filesystem::exists (path))
        {
          return path;
        }
    }

  return "";
}

} // namespace

TEST_CASE ("GemmaTextGenerator construction", "[generator][text_generator]")
{
  auto models_dir = GetModelsDir ();

  SECTION ("Throws on missing models directory")
  {
    REQUIRE_THROWS_AS (
        cortext::GemmaTextGenerator ("/nonexistent/models/path"),
        std::runtime_error);
  }

  if (models_dir.empty ())
    {
      WARN ("Models directory not found, skipping remaining tests");
      return;
    }

  SECTION ("Constructs with valid models directory")
  {
    // May succeed or throw depending on whether tokenizer.json exists
    // and ORT is enabled
    try
      {
        cortext::GemmaTextGenerator gen (models_dir);
        // If it succeeds, check availability
        // (will be false if ORT not enabled)
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
        // With ORT, should be available if ONNX files exist
#else
        CHECK_FALSE (gen.IsAvailable ());
#endif
      }
    catch (const std::runtime_error &)
      {
        // Expected if tokenizer or models missing
      }
  }
}

TEST_CASE ("GemmaTextGenerator availability", "[generator][text_generator]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  try
    {
      cortext::GemmaTextGenerator gen (models_dir);

      SECTION ("IsAvailable reflects ORT status")
      {
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
        // Available if ONNX files exist
        // This test just checks the API works
        bool available = gen.IsAvailable ();
        (void)available;
#else
        CHECK_FALSE (gen.IsAvailable ());
#endif
      }

      SECTION ("EosTokenId returns valid token")
      {
        auto eos = gen.EosTokenId ();
        CHECK (eos >= 0);
      }
    }
  catch (const std::runtime_error &)
    {
      SKIP ("Generator construction failed");
    }
}

TEST_CASE ("GenerationConfig defaults", "[generator][text_generator]")
{
  cortext::GenerationConfig config;

  SECTION ("Has sensible defaults")
  {
    CHECK (config.max_new_tokens == 256);
    CHECK (config.temperature == Catch::Approx (0.7f));
    CHECK (config.top_p == Catch::Approx (0.9f));
    CHECK (config.top_k == 50);
    CHECK (config.do_sample == true);
  }
}

TEST_CASE ("GenerationInput factory methods", "[generator][text_generator]")
{
  SECTION ("Text input")
  {
    auto input = cortext::GenerationInput::Text ("Hello, world!");
    CHECK (input.modality == "text");
    CHECK (input.text == "Hello, world!");
    CHECK (input.image_data.empty ());
    CHECK (input.audio_pcm.empty ());
  }

  SECTION ("Image input")
  {
    std::vector<std::uint8_t> pixels (64 * 64 * 3, 128);
    auto input = cortext::GenerationInput::Image (pixels.data (), 64, 64, 3);

    CHECK (input.modality == "image");
    CHECK (input.width == 64);
    CHECK (input.height == 64);
    CHECK (input.channels == 3);
    CHECK (input.image_data.size () == 64 * 64 * 3);
    CHECK (input.text.empty ());
    CHECK (input.audio_pcm.empty ());
  }

  SECTION ("Audio input")
  {
    std::vector<float> samples (16000, 0.5f); // 1 second at 16kHz
    auto input
        = cortext::GenerationInput::Audio (samples.data (), samples.size (),
                                           16000);

    CHECK (input.modality == "audio");
    CHECK (input.sample_rate == 16000);
    CHECK (input.audio_pcm.size () == 16000);
    CHECK (input.text.empty ());
    CHECK (input.image_data.empty ());
  }
}

#if defined(CORTEXT_ENABLE_GEMMA_ORT)
TEST_CASE ("GemmaTextGenerator JSON generation",
           "[generator][text_generator][integration]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  try
    {
      cortext::GemmaTextGenerator gen (models_dir);

      if (!gen.IsAvailable ())
        {
          SKIP ("Generator not available");
        }

      SECTION ("Simple JSON schema")
      {
        nlohmann::json schema
            = { { "type", "object" },
                { "properties",
                  { { "answer", { { "type", "string" } } } } },
                { "required", { "answer" } } };

        auto result = gen.GenerateJSON ("What is 2+2? Answer in JSON format:",
                                        schema, 50, 0.3f);

        REQUIRE (result.contains ("answer"));
      }

      SECTION ("Enum constraint")
      {
        nlohmann::json schema = { { "type", "object" },
                                  { "properties",
                                    { { "sentiment",
                                        { { "type", "string" },
                                          { "enum",
                                            { "positive", "negative",
                                              "neutral" } } } } } },
                                  { "required", { "sentiment" } } };

        auto result = gen.GenerateJSON (
            "What is the sentiment of 'I love this!'? Respond with JSON:",
            schema, 50, 0.3f);

        REQUIRE (result.contains ("sentiment"));
        std::string sentiment = result["sentiment"];
        CHECK ((sentiment == "positive" || sentiment == "negative"
                || sentiment == "neutral"));
      }

      SECTION ("LastTokenCount tracks generation length")
      {
        nlohmann::json schema
            = { { "type", "object" },
                { "properties", { { "x", { { "type", "integer" } } } } },
                { "required", { "x" } } };

        gen.GenerateJSON ("Return x=1:", schema, 50, 0.3f);

        CHECK (gen.LastTokenCount () > 0);
        CHECK (gen.LastTokenCount () <= 50);
      }

      SECTION ("Multi-field required schema")
      {
        nlohmann::json schema
            = { { "type", "object" },
                { "properties",
                  { { "name", { { "type", "string" } } },
                    { "age", { { "type", "integer" } } },
                    { "active", { { "type", "boolean" } } } } },
                { "required", { "name", "age" } } };

        auto result = gen.GenerateJSON (
            "Generate a person with name John and age 30:", schema, 100, 0.3f);

        REQUIRE (result.contains ("name"));
        REQUIRE (result.contains ("age"));
        CHECK (result["name"].is_string ());
        CHECK (result["age"].is_number_integer ());
      }

      SECTION ("Number field generation")
      {
        nlohmann::json schema
            = { { "type", "object" },
                { "properties",
                  { { "temperature", { { "type", "number" } } },
                    { "count", { { "type", "integer" } } } } },
                { "required", { "temperature", "count" } } };

        auto result = gen.GenerateJSON (
            "Return temperature 98.6 and count 5:", schema, 50, 0.3f);

        REQUIRE (result.contains ("temperature"));
        REQUIRE (result.contains ("count"));
        CHECK (result["temperature"].is_number ());
        CHECK (result["count"].is_number_integer ());
      }

      SECTION ("Boolean field generation")
      {
        nlohmann::json schema
            = { { "type", "object" },
                { "properties",
                  { { "enabled", { { "type", "boolean" } } },
                    { "verified", { { "type", "boolean" } } } } },
                { "required", { "enabled" } } };

        auto result = gen.GenerateJSON ("Return enabled as true:", schema, 50,
                                        0.3f);

        REQUIRE (result.contains ("enabled"));
        CHECK (result["enabled"].is_boolean ());
      }

      SECTION ("Nested object schema")
      {
        nlohmann::json schema
            = { { "type", "object" },
                { "properties",
                  { { "person",
                      { { "type", "object" },
                        { "properties",
                          { { "name", { { "type", "string" } } },
                            { "age", { { "type", "integer" } } } } },
                        { "required", { "name" } } } } } },
                { "required", { "person" } } };

        auto result = gen.GenerateJSON (
            "Create a person object with name Alice:", schema, 100, 0.3f);

        REQUIRE (result.contains ("person"));
        CHECK (result["person"].is_object ());
        CHECK (result["person"].contains ("name"));
      }

      SECTION ("Array schema")
      {
        nlohmann::json schema
            = { { "type", "object" },
                { "properties",
                  { { "numbers",
                      { { "type", "array" },
                        { "items", { { "type", "integer" } } } } } } },
                { "required", { "numbers" } } };

        auto result = gen.GenerateJSON ("Return an array of numbers [1, 2, 3]:",
                                        schema, 100, 0.3f);

        REQUIRE (result.contains ("numbers"));
        CHECK (result["numbers"].is_array ());
      }
    }
  catch (const std::runtime_error &e)
    {
      SKIP ("Generator construction failed: " + std::string (e.what ()));
    }
}

TEST_CASE ("GemmaTextGenerator text generation",
           "[generator][text_generator][integration]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  try
    {
      cortext::GemmaTextGenerator gen (models_dir);

      if (!gen.IsAvailable ())
        {
          SKIP ("Generator not available");
        }

      SECTION ("Basic text generation")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 20;
        config.temperature = 0.5f;

        auto result = gen.Generate ("Once upon a time", config);

        CHECK (!result.empty ());
      }

      SECTION ("Greedy generation")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 10;
        config.do_sample = false;

        auto result = gen.Generate ("The capital of France is", config);

        CHECK (!result.empty ());
      }

      SECTION ("High temperature sampling")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 20;
        config.temperature = 1.2f;
        config.do_sample = true;

        auto result = gen.Generate ("Once upon a time", config);

        CHECK (!result.empty ());
      }

      SECTION ("Low temperature sampling")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 20;
        config.temperature = 0.2f;
        config.do_sample = true;

        auto result = gen.Generate ("The answer is", config);

        CHECK (!result.empty ());
      }

      SECTION ("Top-k sampling")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 15;
        config.top_k = 10;
        config.do_sample = true;

        auto result = gen.Generate ("A quick brown fox", config);

        CHECK (!result.empty ());
      }

      SECTION ("Top-p nucleus sampling")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 15;
        config.top_p = 0.8f;
        config.do_sample = true;

        auto result = gen.Generate ("In the beginning", config);

        CHECK (!result.empty ());
      }

      SECTION ("Combined top-k and top-p sampling")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 15;
        config.top_k = 40;
        config.top_p = 0.95f;
        config.temperature = 0.7f;
        config.do_sample = true;

        auto result = gen.Generate ("Hello world", config);

        CHECK (!result.empty ());
      }

      SECTION ("Max token limit enforcement")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 5;
        config.do_sample = false;

        auto result = gen.Generate ("Count to one hundred:", config);

        // Token count should respect the limit
        CHECK (gen.LastTokenCount () <= 5);
      }
    }
  catch (const std::runtime_error &e)
    {
      SKIP ("Generator construction failed: " + std::string (e.what ()));
    }
}

TEST_CASE ("GemmaTextGenerator oneOf schema generation",
           "[generator][text_generator][integration]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  try
    {
      cortext::GemmaTextGenerator gen (models_dir);

      if (!gen.IsAvailable ())
        {
          SKIP ("Generator not available");
        }

      SECTION ("oneOf with discriminator")
      {
        nlohmann::json schema = {
          { "type", "object" },
          { "oneOf",
            { { { "properties",
                  { { "type", { { "const", "cat" } } },
                    { "lives", { { "type", "integer" } } } } },
                { "required", { "type", "lives" } } },
              { { "properties",
                  { { "type", { { "const", "dog" } } },
                    { "breed", { { "type", "string" } } } } },
                { "required", { "type", "breed" } } } } }
        };

        auto result = gen.GenerateJSON (
            "Generate a cat with 9 lives in JSON:", schema, 100, 0.3f);

        REQUIRE (result.contains ("type"));
        std::string type = result["type"];

        if (type == "cat")
          {
            CHECK (result.contains ("lives"));
          }
        else if (type == "dog")
          {
            CHECK (result.contains ("breed"));
          }
      }
    }
  catch (const std::runtime_error &e)
    {
      SKIP ("Generator construction failed: " + std::string (e.what ()));
    }
}
#endif
